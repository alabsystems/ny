// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The CLASS-INDEPENDENT frame of the frozen dual state: the certified
//! pre-activation box of every ReLU (trunk and head), the committed lower
//! slope and the chord of every neuron, the generator column layout, and a
//! certified MAGNITUDE bound `mag[t] >= |v_t(x)|` for every tensor `t` of the
//! twin net over the whole input box.
//!
//! Nothing here depends on the adversarial class: the unstable sets (hence the
//! generator columns) are fixed by the boxes, and only the per-class slopes
//! `lambda` (chosen by [`super::build`]) differ between classes.
//!
//! # What the magnitudes are for
//!
//! The reverse builder charges every rounding and model error of a backward
//! step as `(error of a coefficient) x (bound on the value it multiplies)`,
//! so it needs a certified bound on `|v_t(x)|` for every tensor it walks
//! through. ReLU inputs take their certified box (`max(|l|, |u|)`), ReLU
//! outputs `max(u, 0)`, and everything else a forward interval-magnitude pass
//! `|W| mag + |b| + fold errors` rounded up. Every one of these is an upper
//! bound on `|v_t(x)|` for every `x` in the declared box, which is all the
//! charges need.

use std::time::Instant;

use crate::zono_core::kernels::check_deadline;

mod magnitude;
use magnitude::{conv_magnitude, MagnitudeWork};

use crate::zono_core::rounding::{gamma, next_up, TINY, UNIT};
use crate::zono_core::ZonoRefusal;

use super::super::bounds::{head_gates_with, YBox};
use super::super::net::{TwinNet, TwinOp};
use super::super::root::RootGates;
use super::super::rounding::RoundMode;

/// Refuse any tensor, generator count or row count at or above this. Every
/// charge in this module is a float sum of at most this many non-negative
/// terms, whose relative under-estimate `gamma_n` stays far below the
/// [`SAFE`] factor that absorbs it.
pub const MAX_DIM: usize = 100_000_000;

/// Blanket factor applied to every accumulated charge before it is published.
/// A float sum of `n < MAX_DIM` non-negative terms under-estimates the exact
/// sum by at most a RELATIVE `gamma_{n+1} < 1.2e-8` (products included), and
/// the published remainder is itself a float sum of such non-negative
/// charges, so its total relative under-estimate is at most
/// `gamma_{n+1} + gamma_{k}` (`k` = number of charges, `< MAX_DIM`) plus a
/// few `u` — the errors ADD, they do not compound per op. `1 + 1e-6`
/// dominates that with well over an order of magnitude to spare.
pub const SAFE: f64 = 1.0 + 1e-6;

/// Where a relaxed ReLU sits in the twin net.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReluSite {
    /// Trunk ReLU layer `m` (`TwinNet::trunk_relus[m]`).
    Trunk(usize),
    /// The head ReLU on the first head Gemm's output `y`.
    Head,
}

/// Caller-supplied relaxation data of one ReLU layer: its CERTIFIED
/// pre-activation box and the two slopes the state may choose between.
#[derive(Debug, Clone)]
pub struct LayerInput {
    /// Certified lower pre-activation bound, one per neuron.
    pub l: Vec<f64>,
    /// Certified upper pre-activation bound.
    pub u: Vec<f64>,
    /// Committed lower slope in `[0, 1]` (the root alpha).
    pub alpha: Vec<f64>,
    /// Chord slope `u / (u - l)` in `[0, 1]`.
    pub s: Vec<f64>,
}

/// Everything the frame is built from, decoupled from [`RootGates`] so tests
/// and later stages can supply any certified boxes.
#[derive(Debug, Clone)]
pub struct FrameInput {
    /// Declared input box, lower ends.
    pub lo: Vec<f64>,
    /// Declared input box, upper ends.
    pub hi: Vec<f64>,
    /// One entry per trunk ReLU layer, in `TwinNet::trunk_relus` order.
    pub trunk: Vec<LayerInput>,
    /// The head ReLU (box = the certified y-box).
    pub head: LayerInput,
}

impl FrameInput {
    /// The margin-row lane's frozen root artifacts: the trunk boxes after
    /// backward-interm / interm-alpha, the COMMITTED trunk alpha, the y-box of
    /// the root evaluation, and the head gates on that y-box with the root's
    /// committed head alpha (`bounds::head_gates_with`, exactly what the lane's
    /// own root evaluation uses).
    ///
    /// The trunk boxes are taken as enclosures of EVERY member of the
    /// certified fold ball, not only of the stored network. The root tableau
    /// now charges the relative weight / scale fold error sign-safely in the
    /// bias column (`root.rs` #fold-ball). Before that repair a
    /// coefficient-column charge let the `conv_bn` fixture's fold-ball members
    /// escape the trunk-0 box by ~1e-12, which the exact-rational containment
    /// gate (`frozen_dual_state_contains_exact_rational_samples`) caught.
    ///
    /// The tableau's own ROUNDING is charged the same sign-safe way
    /// (`root.rs` #root-rounding). Before that repair the rounding widened the
    /// coefficient columns, and an exact-rational counterexample left a trunk
    /// box 2.9e-15 short of the stored network on a cancelling, stable-active,
    /// near-bias-free stack (`root_box_encloses_the_exact_minimal_rounding_leak`).
    #[must_use]
    pub fn from_root(root: &RootGates, ybox: &YBox) -> Self {
        let trunk = root
            .layers
            .iter()
            .map(|g| LayerInput {
                l: g.l.clone(),
                u: g.u.clone(),
                alpha: g.alpha.clone(),
                s: g.s.clone(),
            })
            .collect();
        let hg = head_gates_with(ybox, RoundMode::Outward, root.head_alpha.as_deref());
        let head = LayerInput {
            l: ybox.ly.clone(),
            u: ybox.uy.clone(),
            alpha: hg.alpha,
            s: hg.s,
        };
        Self {
            lo: root.lo.clone(),
            hi: root.hi.clone(),
            trunk,
            head,
        }
    }
}

/// One ReLU layer of the frame.
#[derive(Debug, Clone)]
pub struct RelaxLayer {
    /// Where it sits.
    pub site: ReluSite,
    /// Its op index in `TwinNet::ops`.
    pub op: usize,
    /// Tensor id of its pre-activation (the op's input).
    pub input: usize,
    /// Certified pre-activation box, lower.
    pub l: Vec<f64>,
    /// Certified pre-activation box, upper.
    pub u: Vec<f64>,
    /// Committed lower slope.
    pub alpha: Vec<f64>,
    /// Chord slope.
    pub s: Vec<f64>,
    /// Unstable neurons (`l < 0 < u`), ascending.
    pub unst: Vec<usize>,
    /// Neuron -> position in `unst`, `usize::MAX` for a stable neuron.
    pub pos: Vec<usize>,
    /// Generator column of `unst[0]`'s relaxation symbol; `unst[p]` owns
    /// column `col0 + p`.
    pub col0: usize,
}

/// The class-independent frame (see the module docs).
#[derive(Debug, Clone)]
pub struct FrozenDualFrame {
    /// Flat input size.
    pub n_in: usize,
    /// Declared input box, lower.
    pub lo: Vec<f64>,
    /// Declared input box, upper.
    pub hi: Vec<f64>,
    /// Input centre used by the input symbols.
    pub mid: Vec<f64>,
    /// Input radius: `mid - rad <= lo` and `mid + rad >= hi` EXACTLY, `0` on a
    /// degenerate dimension (`lo == hi == mid`).
    pub rad: Vec<f64>,
    /// Input dimensions that own a symbol (`rad > 0`); generator column `i`
    /// is input dimension `in_cols[i]`.
    pub in_cols: Vec<usize>,
    /// ReLU layers in execution order (every trunk layer, then the head).
    pub layers: Vec<RelaxLayer>,
    /// Number of input symbols (`in_cols.len()`).
    pub n_in_gens: usize,
    /// Number of relaxation symbols (all unstable neurons, all layers).
    pub n_unstable: usize,
    /// Total generator count.
    pub n_gens: usize,
    /// True class.
    pub t: usize,
    /// Output classes.
    pub n_out: usize,
    /// Tensor id of the logits.
    pub out_tensor: usize,
    /// Certified magnitude bound per tensor id (`0` = input).
    pub(crate) mag: Vec<Vec<f64>>,
    /// Per op: the certified `|W| mag_in` per OUTPUT element (conv, gemm) or
    /// `|scale| * mag_in` (channel affine); empty for other ops.
    pub(crate) wmag: Vec<Vec<f64>>,
    /// Per op: index into `layers` for a ReLU op, `usize::MAX` otherwise.
    pub(crate) layer_of_op: Vec<usize>,
    /// Largest tensor size (admission).
    pub(crate) max_width: usize,
    /// Completed magnitude work, diagnostic only.
    pub(crate) magnitude_work: MagnitudeWork,
}

/// `next_up(x * (1 + 2 rel + 4u) + TINY)`: an upper bound on the exact value of
/// a non-negative quantity whose float evaluation `x` under-estimates it by at
/// most a relative `rel` (`rel <= 1/4`), with the multiply's own rounding and
/// gradual underflow covered.
#[inline]
pub(crate) fn cert_up(x: f64, rel: f64) -> f64 {
    next_up(x * (1.0 + 2.0 * rel + 4.0 * UNIT) + TINY)
}

fn finite_all(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

impl FrozenDualFrame {
    /// Validate and build the frame for `net`, true class `t`.
    ///
    /// Refuses (fail closed) on any shape mismatch, NaN/Inf, a crossed box
    /// (`l > u`), a slope outside `[0, 1]`, a ReLU op the input does not
    /// describe, or a dimension at or above [`MAX_DIM`].
    pub fn new(net: &TwinNet, input: FrameInput, t: usize) -> Result<Self, ZonoRefusal> {
        Self::new_until(net, input, t, None)
    }

    fn new_until(
        net: &TwinNet,
        input: FrameInput,
        t: usize,
        deadline: Option<Instant>,
    ) -> Result<Self, ZonoRefusal> {
        check_deadline(deadline)?;
        let n_in = net.n_in;
        if t >= net.n_out {
            return Err(ZonoRefusal::Shape(format!(
                "frozen-dual frame: true class {t} out of range {}",
                net.n_out
            )));
        }
        if net.tsize.iter().any(|&w| w >= MAX_DIM) {
            return Err(ZonoRefusal::Shape(
                "frozen-dual frame: a tensor is at or above MAX_DIM".to_string(),
            ));
        }
        if input.lo.len() != n_in || input.hi.len() != n_in {
            return Err(ZonoRefusal::Shape(format!(
                "frozen-dual frame: input box {} / {} for n_in = {n_in}",
                input.lo.len(),
                input.hi.len()
            )));
        }
        let mut mid = Vec::with_capacity(n_in);
        let mut rad = Vec::with_capacity(n_in);
        let mut in_cols = Vec::new();
        for (i, (&lo, &hi)) in input.lo.iter().zip(&input.hi).enumerate() {
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(ZonoRefusal::NonFinite("frozen-dual input box"));
            }
            if lo == hi {
                // x_i == lo exactly on the whole box: no symbol, exact centre.
                mid.push(lo);
                rad.push(0.0);
                continue;
            }
            // Halving is exact outside the subnormal range and the sum rounds
            // once; the centre is clamped into [lo, hi] so both differences
            // below are non-negative.
            let m = (0.5 * lo + 0.5 * hi).clamp(lo, hi);
            // fl(hi - m) is within half an ulp of the exact difference, so
            // `next_up` of the larger one dominates both EXACT differences:
            // mid - rad <= lo and mid + rad >= hi hold in real arithmetic.
            let r = next_up((hi - m).max(m - lo));
            if !r.is_finite() {
                return Err(ZonoRefusal::NonFinite("frozen-dual input radius"));
            }
            mid.push(m);
            rad.push(r);
            in_cols.push(i);
        }

        // ---- ReLU layers, in op order ------------------------------------
        let head_op = net.i_gemm1 + 1;
        if input.trunk.len() != net.trunk_relus.len() {
            return Err(ZonoRefusal::Shape(format!(
                "frozen-dual frame: {} trunk layers supplied for {} trunk relus",
                input.trunk.len(),
                net.trunk_relus.len()
            )));
        }
        let mut layer_of_op = vec![usize::MAX; net.ops.len()];
        let mut layers: Vec<RelaxLayer> = Vec::with_capacity(net.trunk_relus.len() + 1);
        let mut col = in_cols.len();
        let mut described: Vec<(usize, ReluSite, LayerInput)> = input
            .trunk
            .into_iter()
            .enumerate()
            .map(|(m, li)| (net.trunk_relus[m], ReluSite::Trunk(m), li))
            .collect();
        described.push((head_op, ReluSite::Head, input.head));
        described.sort_by_key(|(op, _, _)| *op);
        for (op, site, li) in described {
            check_deadline(deadline)?;
            let Some(TwinOp::Relu { input: pre, .. }) = net.ops.get(op) else {
                return Err(ZonoRefusal::Internal(format!(
                    "frozen-dual frame: op {op} ({site:?}) is not a relu"
                )));
            };
            let n = net.tsize[*pre];
            if li.l.len() != n || li.u.len() != n || li.alpha.len() != n || li.s.len() != n {
                return Err(ZonoRefusal::Shape(format!(
                    "frozen-dual frame: {site:?} carries {}/{}/{}/{} entries for width {n}",
                    li.l.len(),
                    li.u.len(),
                    li.alpha.len(),
                    li.s.len()
                )));
            }
            if !finite_all(&li.l) || !finite_all(&li.u) {
                return Err(ZonoRefusal::NonFinite("frozen-dual relu box"));
            }
            if li.l.iter().zip(&li.u).any(|(l, u)| l > u) {
                return Err(ZonoRefusal::Internal(format!(
                    "frozen-dual frame: crossed pre-activation box at {site:?}"
                )));
            }
            let slope_ok = |v: &f64| v.is_finite() && (0.0..=1.0).contains(v);
            let mut unst = Vec::new();
            let mut pos = vec![usize::MAX; n];
            for j in 0..n {
                if li.l[j] < 0.0 && li.u[j] > 0.0 {
                    if !slope_ok(&li.alpha[j]) || !slope_ok(&li.s[j]) {
                        return Err(ZonoRefusal::Internal(format!(
                            "frozen-dual frame: slope outside [0, 1] at {site:?} neuron {j}"
                        )));
                    }
                    pos[j] = unst.len();
                    unst.push(j);
                }
            }
            layer_of_op[op] = layers.len();
            let col0 = col;
            col = col
                .checked_add(unst.len())
                .ok_or_else(|| ZonoRefusal::Shape("generator count overflows".to_string()))?;
            layers.push(RelaxLayer {
                site,
                op,
                input: *pre,
                l: li.l,
                u: li.u,
                alpha: li.alpha,
                s: li.s,
                unst,
                pos,
                col0,
            });
        }
        // Every relu of the net must be described (a relu without a box would
        // have no certified relaxation at all).
        for (k, op) in net.ops.iter().enumerate() {
            if matches!(op, TwinOp::Relu { .. }) && layer_of_op[k] == usize::MAX {
                return Err(ZonoRefusal::UnsupportedGraph(format!(
                    "relu op {k} has no certified box in the frame"
                )));
            }
        }
        let n_gens = col;
        if n_gens >= MAX_DIM {
            return Err(ZonoRefusal::Shape(format!(
                "frozen-dual frame: {n_gens} generators at or above MAX_DIM"
            )));
        }
        let n_in_gens = in_cols.len();
        let n_unstable = n_gens - n_in_gens;

        // ---- certified magnitudes ---------------------------------------
        let mut magnitude_work = MagnitudeWork::default();
        let n_t = net.ops.len() + 1;
        let mut mag: Vec<Vec<f64>> = vec![Vec::new(); n_t];
        let mut wmag: Vec<Vec<f64>> = vec![Vec::new(); net.ops.len()];
        mag[0] = input
            .lo
            .iter()
            .zip(&input.hi)
            .map(|(l, h)| l.abs().max(h.abs()))
            .collect();
        // A tensor that is some relu's input takes min(IBP, box magnitude).
        let mut box_mag: Vec<Option<Vec<f64>>> = vec![None; n_t];
        for layer in &layers {
            let bm: Vec<f64> = layer
                .l
                .iter()
                .zip(&layer.u)
                .map(|(l, u)| l.abs().max(u.abs()))
                .collect();
            box_mag[layer.input] = Some(match box_mag[layer.input].take() {
                Some(prev) => prev.iter().zip(&bm).map(|(a, b)| a.min(*b)).collect(),
                None => bm,
            });
        }
        for (k, op) in net.ops.iter().enumerate() {
            check_deadline(deadline)?;
            let out = k + 1;
            let mut m_out: Vec<f64> = match op {
                TwinOp::Conv(cv) => {
                    let m_in = &mag[cv.input];
                    let wm = conv_magnitude(cv, m_in, deadline, &mut magnitude_work)?;
                    let p = cv.oshape.1 * cv.oshape.2;
                    let rel = cv.weight_rel_err;
                    let mo = wm
                        .iter()
                        .enumerate()
                        .map(|(j, &w)| {
                            let ch = j / p.max(1);
                            cert_up(
                                w * (1.0 + rel) + cv.bias[ch].abs() + cv.bias_err[ch],
                                4.0 * UNIT,
                            )
                        })
                        .collect();
                    wmag[k] = wm;
                    mo
                }
                TwinOp::ChannelAffine {
                    input,
                    scale,
                    shift,
                    scale_rel_err,
                    shift_err,
                } => {
                    let m_in = &mag[*input];
                    let sm: Vec<f64> = scale
                        .iter()
                        .zip(m_in)
                        .map(|(s, m)| cert_up(s.abs() * m, UNIT))
                        .collect();
                    let mo = sm
                        .iter()
                        .enumerate()
                        .map(|(j, &v)| {
                            cert_up(
                                v * (1.0 + scale_rel_err) + shift[j].abs() + shift_err[j],
                                4.0 * UNIT,
                            )
                        })
                        .collect();
                    wmag[k] = sm;
                    mo
                }
                TwinOp::Relu { .. } => {
                    let layer = &layers[layer_of_op[k]];
                    layer.u.iter().map(|u| u.max(0.0)).collect()
                }
                TwinOp::Add { lhs, rhs } => mag[*lhs]
                    .iter()
                    .zip(&mag[*rhs])
                    .map(|(a, b)| cert_up(a + b, UNIT))
                    .collect(),
                TwinOp::Flatten { input } => mag[*input].clone(),
                TwinOp::Gemm {
                    input,
                    weight,
                    bias,
                    bias_err,
                    weight_rel_err,
                    shape,
                } => {
                    let (no, ni) = *shape;
                    let m_in = &mag[*input];
                    let g = gamma(ni + 2);
                    let wm: Vec<f64> = (0..no)
                        .map(|o| {
                            let row = &weight[o * ni..(o + 1) * ni];
                            let acc: f64 = row.iter().zip(m_in).map(|(w, m)| w.abs() * m).sum();
                            cert_up(acc, g)
                        })
                        .collect();
                    let mo = wm
                        .iter()
                        .enumerate()
                        .map(|(o, &w)| {
                            cert_up(
                                w * (1.0 + weight_rel_err) + bias[o].abs() + bias_err[o],
                                4.0 * UNIT,
                            )
                        })
                        .collect();
                    wmag[k] = wm;
                    mo
                }
            };
            if let Some(bm) = box_mag[out].as_ref() {
                for (m, b) in m_out.iter_mut().zip(bm) {
                    *m = m.min(*b);
                }
            }
            if m_out.len() != net.tsize[out] || !finite_all(&m_out) || !finite_all(&wmag[k]) {
                return Err(ZonoRefusal::NonFinite("frozen-dual magnitude"));
            }
            mag[out] = m_out;
        }
        check_deadline(deadline)?;
        let max_width = net.tsize.iter().copied().max().unwrap_or(0);
        Ok(Self {
            n_in,
            lo: input.lo,
            hi: input.hi,
            mid,
            rad,
            in_cols,
            layers,
            n_in_gens,
            n_unstable,
            n_gens,
            t,
            n_out: net.n_out,
            out_tensor: net.ops.len(),
            mag,
            wmag,
            layer_of_op,
            max_width,
            magnitude_work,
        })
    }

    /// [`Self::new`] from the lane's frozen root and the root y-box.
    ///
    /// Refuses (fail closed) a root whose boxes are not WHOLE-BOX certified
    /// enclosures, because the frame's claim is about the whole declared box:
    ///
    /// * a root not in [`RoundMode::Outward`] (a `Parity` root's boxes are
    ///   round-to-nearest, not enclosures);
    /// * a root whose gate layers do not line up with `net.trunk_relus` (the
    ///   frame would pair a box with the wrong tensor);
    /// * a root carrying the `RootGates::build_retaining` split-BAKE signature
    ///   (an epoch child): a baked neuron keeps its box (`l < 0 < u`) but
    ///   leaves its layer's `unst`, and every box downstream of it holds only
    ///   on the split subdomain.
    ///
    /// The last check is defence in depth, not completeness: a parent-frame
    /// refresh bakes by clamping the box itself (the neuron then looks
    /// stable). The PRIMARY guard is the caller's: the tier runs only on the
    /// outer frame (`EpochFrameRoute::Outer`), whose root is built with no
    /// splits.
    pub fn from_root(
        net: &TwinNet,
        root: &RootGates,
        ybox: &YBox,
        t: usize,
    ) -> Result<Self, ZonoRefusal> {
        Self::from_root_until(net, root, ybox, t, None)
    }

    /// Production frame construction is charged to the existing tier deadline.
    /// No partially completed magnitude enclosure is returned.
    pub(crate) fn from_root_until(
        net: &TwinNet,
        root: &RootGates,
        ybox: &YBox,
        t: usize,
        deadline: Option<Instant>,
    ) -> Result<Self, ZonoRefusal> {
        check_deadline(deadline)?;
        if !root.mode.outward() {
            return Err(ZonoRefusal::Internal(
                "frozen-dual frame: the root is not in the certified-outward mode".to_string(),
            ));
        }
        if root.layers.len() != net.trunk_relus.len() {
            return Err(ZonoRefusal::Shape(format!(
                "frozen-dual frame: {} root gate layers for {} trunk relus",
                root.layers.len(),
                net.trunk_relus.len()
            )));
        }
        for (m, (g, &op)) in root.layers.iter().zip(&net.trunk_relus).enumerate() {
            if g.op != op {
                return Err(ZonoRefusal::Internal(format!(
                    "frozen-dual frame: root gate layer {m} is relu op {} but trunk relu {m} \
                     is op {op}",
                    g.op
                )));
            }
            let mut listed = vec![false; g.l.len()];
            for &j in &g.unst {
                if let Some(slot) = listed.get_mut(j) {
                    *slot = true;
                }
            }
            let baked =
                g.l.iter()
                    .zip(&g.u)
                    .zip(&listed)
                    .any(|((l, u), &in_unst)| *l < 0.0 && *u > 0.0 && !in_unst);
            if baked {
                return Err(ZonoRefusal::Internal(format!(
                    "frozen-dual frame: root gate layer {m} carries a split-baked neuron \
                     (a subdomain root, not the whole box)"
                )));
            }
        }
        Self::new_until(net, FrameInput::from_root(root, ybox), t, deadline)
    }

    /// Certified magnitude bound of tensor `t` (see the module docs).
    #[must_use]
    pub fn magnitude(&self, tensor: usize) -> &[f64] {
        &self.mag[tensor]
    }
}
