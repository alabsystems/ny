// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! G2 for the margin-row root frame (#soundness-spine M2d).
//!
//! The root frame publishes three kinds of enclosure over the declared input
//! box, and an `unsat` rests on all three:
//!
//! * every trunk ReLU pre-activation box `LayerGates::{l, u}`;
//! * the head pre-activation box (the root evaluation's y-box);
//! * the certified root bound `dj[k]` of every margin `y_t − y_{adv[k]}`, and
//!   the verdict's own claim that every margin is strictly positive.
//!
//! [`RootFrame`] samples them through `TwinNet::forward_points`, the lane's
//! own f64 point evaluator over its OWN IR (the stored, possibly BN-folded
//! weights), and hands each comparison to the generic
//! [`crate::guards::containment`] audit, whose band absorbs the fold error the
//! certified side charges and the point side does not.
//!
//! Independence, stated precisely: the point side shares the TwinNet
//! extraction with the certified side, so a defect in the extraction itself
//! (op semantics wrong in the same way twice) is invisible here — that is G1's
//! job (ORT on the original ONNX). What it does catch is any certified
//! TRANSFORMER, tableau, backward-pass or composition defect that publishes an
//! enclosure the network's own behaviour leaves.

use std::collections::BTreeMap;

use ndarray::Array2;

use super::bounds::YBox;
use super::net::{TwinNet, TwinOp};
use super::root::RootGates;
use crate::guards::containment::{CertifiedFrame, ContainmentAudit};

/// Conservative multiply-accumulate rate of the point evaluator, for
/// admission only (a slower host is covered by the caller's budget fraction).
const POINT_MACS_PER_SEC: f64 = 5e8;

/// At most this many neurons per trunk layer are compared (strided), which
/// bounds the collected pre-activation rows on wide layers.
const MAX_NEURONS_PER_LAYER: usize = 4096;

/// The margin-row root frame as a [`CertifiedFrame`].
pub(crate) struct RootFrame<'a> {
    net: &'a TwinNet,
    root: &'a RootGates,
    t: usize,
    adv: &'a [usize],
    dj: &'a [f64],
    ybox: YBox,
}

impl<'a> RootFrame<'a> {
    /// `dj` must be aligned with `adv` (the root evaluation's own vector) and
    /// `ybox` must be the root evaluation's head box.
    pub(crate) fn new(
        net: &'a TwinNet,
        root: &'a RootGates,
        t: usize,
        adv: &'a [usize],
        dj: &'a [f64],
        ybox: YBox,
    ) -> Self {
        Self {
            net,
            root,
            t,
            adv,
            dj,
            ybox,
        }
    }

    fn macs(&self) -> f64 {
        let mut total: usize = 0;
        for op in &self.net.ops {
            let m = match op {
                TwinOp::Conv(c) => {
                    let (co, ci, kh, kw) = c.kernel;
                    co.saturating_mul(ci)
                        .saturating_mul(kh)
                        .saturating_mul(kw)
                        .saturating_mul(c.oshape.1)
                        .saturating_mul(c.oshape.2)
                }
                TwinOp::Gemm { shape, .. } => shape.0.saturating_mul(shape.1),
                _ => 0,
            };
            total = total.saturating_add(m);
        }
        #[allow(clippy::cast_precision_loss)]
        let total = total as f64;
        total
    }

    fn selection(&self) -> BTreeMap<usize, Vec<usize>> {
        let mut sel = BTreeMap::new();
        for layer in &self.root.layers {
            let stride = (layer.n / MAX_NEURONS_PER_LAYER).max(1);
            sel.insert(layer.op, (0..layer.n).step_by(stride).collect());
        }
        sel
    }
}

impl CertifiedFrame for RootFrame<'_> {
    fn input_box(&self) -> (&[f64], &[f64]) {
        (&self.root.lo, &self.root.hi)
    }

    fn point_cost_secs(&self) -> f64 {
        self.macs() / POINT_MACS_PER_SEC
    }

    fn point_bytes(&self) -> u64 {
        // `forward_points` keeps every op's output for the residual adds.
        let tensors: usize = self.net.tsize.iter().copied().sum();
        let rows: usize = self
            .root
            .layers
            .iter()
            .map(|l| l.n.min(MAX_NEURONS_PER_LAYER))
            .sum();
        u64::try_from(tensors.saturating_add(rows).saturating_mul(8)).unwrap_or(u64::MAX)
    }

    fn audit(&self, points: &Array2<f64>, audit: &mut ContainmentAudit) -> Result<(), String> {
        if self.dj.len() != self.adv.len() {
            return Err("root bounds are not aligned with the adversarial classes".into());
        }
        let sel = self.selection();
        let (y, pre) = self
            .net
            .forward_points(points, &sel)
            .map_err(|e| e.to_string())?;
        for (li, layer) in self.root.layers.iter().enumerate() {
            let (Some(idx), Some(vals)) = (sel.get(&layer.op), pre.get(&layer.op)) else {
                return Err(format!("trunk layer {li}: no pre-activations collected"));
            };
            if vals.nrows() != idx.len() || layer.l.len() != layer.n || layer.u.len() != layer.n {
                return Err(format!("trunk layer {li}: shape mismatch"));
            }
            for (r, &i) in idx.iter().enumerate() {
                for s in 0..vals.ncols() {
                    audit.check_box("trunk_pre", i, s, layer.l[i], layer.u[i], vals[[r, s]]);
                }
            }
        }
        if y.nrows() != self.ybox.ly.len() || y.nrows() != self.ybox.uy.len() {
            return Err("head box does not match the head width".into());
        }
        for i in 0..y.nrows() {
            for s in 0..y.ncols() {
                audit.check_box("head_y", i, s, self.ybox.ly[i], self.ybox.uy[i], y[[i, s]]);
            }
        }
        let (w2, b2, (n_out, n_y)) = self.net.gemm2();
        if n_y != y.nrows() || self.t >= n_out || self.adv.iter().any(|&j| j >= n_out) {
            return Err("final Gemm does not match the head or the classes".into());
        }
        for s in 0..y.ncols() {
            let scores: Vec<f64> = (0..n_out)
                .map(|o| {
                    let mut acc = b2[o];
                    for k in 0..n_y {
                        acc += w2[o * n_y + k] * y[[k, s]].max(0.0);
                    }
                    acc
                })
                .collect();
            let scale = scores.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (k, &j) in self.adv.iter().enumerate() {
                let margin = scores[self.t] - scores[j];
                audit.check_lower("root_dj", j, s, self.dj[k], margin);
                audit.check_positive("margin", j, s, margin, scale);
            }
        }
        Ok(())
    }
}
