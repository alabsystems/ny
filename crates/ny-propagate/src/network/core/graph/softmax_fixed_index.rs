// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #softmax-fixed-index — the FIXED-INDEX softmax decomposition
//! (`solver.alpha-crown.softmax: complex_fixed_index`, vit_2023).
//!
//! The constant-shift rewrite in the parent module computes
//! `exp(x - c) / Σ exp(x - c)` with a CONSTANT `c`, so every Exp input keeps
//! the full width of its logit. alpha-beta-CROWN's vit recipe
//! (`bound_opts {'softmax': 'complex'}` + `fixed_reducemax_index: True`)
//! instead shifts by the logit at a FROZEN index `j*`:
//!
//! ```text
//!   x ─┬─ MulConstant(onehot) ─ ReduceSum(axis,keep) ─ Linear(1→n) ─┐   (x_{j*}, broadcast)
//!      └──────────────────────────────────────────────────────────── Sub ─ MulConstant(mask)
//!                                                                          │ d_i = x_i - x_{j*}, d_{j*} ≡ 0
//!      Exp(d) ─┬──────────────────────────────────────────── MulBinary ─ Clip(0,1) = softmax(x)
//!              └─ ReduceSum(axis,keep) ─ Reciprocal ─────────────┘
//! ```
//!
//! i.e. `softmax_i(x) = exp(d_i) / Σ_k exp(d_k)` with `d_k = x_k - x_{j*}`,
//! which holds EXACTLY for any fixed `j*` (the shift cancels between numerator
//! and denominator — the same identity the constant shift uses, with a shift
//! that is now a linear function of `x`).
//!
//! # Why the shift is worth anything
//!
//! `d_i` is a RELATIVE logit. Its CROWN bound cancels the component the row
//! shares (on vit, the query-row term of `Q·Kᵀ`), where `x_i - c` keeps the
//! full logit width. The `j*` entry is exactly zero, so its Exp term is exactly
//! 1 and the denominator is bounded below by 1 — at the IBP level too, because
//! the `mask` node zeroes that coordinate exactly (`0·x = 0` is special-cased
//! in [`MulConstantLayer`], even for an infinite input).
//!
//! # The denominator (the unsound variant this construction excludes)
//!
//! The sum keeps ALL `n` Exp terms, `j*` included, and adds NO constant. The
//! floor `Σ ≥ 1` is therefore a CONSEQUENCE of the `j*` term being present,
//! never an extra `+1`. Keeping every term AND adding 1 would inflate every
//! denominator by exactly 1 and understate every softmax output — a false-proof
//! generator. The test `softmax_fixed_index_decomposition_exact_pointwise`
//! evaluates the REWRITTEN graph's forward and would catch that double count
//! (its negative control asserts the double-counted value falls outside the
//! point enclosure on every sample), and
//! `softmax_fixed_index_denominator_floor_requires_jstar_term` pins that the
//! Reciprocal reads the bare ReduceSum.
//!
//! # j*
//!
//! `j*` is the argmax of the softmax input at the CENTRE of the input box, per
//! softmax ROW (a head × query-token row on vit), frozen as a constant one-hot
//! selector at rewrite time. Soundness does not depend on `j*` at all — any
//! fixed index gives an exact identity, which is why no argmax-stability
//! assumption is needed and why auto_LiRPA's generic fixed-index ReduceMax
//! backward is NOT copied: the selector here is an exact linear map
//! (one-hot `MulConstant` + `ReduceSum`), whose backward is exact. `j*` only
//! decides how tight the relative logits are.
//!
//! # Exp tangent
//!
//! abc disables Exp from optimization (`disable_optimization: ['Exp']`). ny's
//! DAG alpha loop has no Exp alpha at all (alpha exists for ReLU, Sigmoid/Tanh,
//! Sqrt, Reciprocal, MulBinary and bilinear nodes), so the Exp relaxation is
//! the frozen midpoint tangent / chord of `layers/activations/exp.rs`; the
//! Reciprocal and MulBinary alphas stay optimizable.
//!
//! # Guard policy (differs from the constant shift)
//!
//! The constant-shift rewrite skips a node whose PLAIN-IBP shifted upper bound
//! exceeds [`SOFTMAX_COMPLEX_SHIFT_GUARD`]. That guard was written when
//! `ExpLayer::propagate_ibp` hard-errored above 88; it now saturates soundly,
//! and the Exp CROWN domain guard degrades to a per-node IBP fallback. Applied
//! here, the guard would read the plain-IBP width of `x_i - x_{j*}` — exactly
//! the width CROWN intermediates exist to remove — and would skip every node on
//! the rows the rewrite is for (pgd: IBP logit boxes ~1e5 wide). So a node is
//! decomposed regardless of that width and only COUNTED in `wide_ibp`. The
//! final `Clip(0, 1)` keeps every softmax enclosure inside `[0, 1]` even when
//! the Exp terms saturate.

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info};

use super::{SoftmaxComplexReport, SOFTMAX_COMPLEX_SHIFT_GUARD};
use crate::layers::{
    ClipLayer, ExpLayer, Layer, LinearLayer, MulBinaryLayer, MulConstantLayer, ReciprocalLayer,
    ReduceSumLayer, SubLayer,
};
use crate::network::core::graph::{GraphNetwork, GraphNode, NETWORK_INPUT};

/// Helper-node suffixes, in insertion (topological) order. The softmax node
/// itself keeps its name and becomes the closing `Clip(0, 1)`.
pub(crate) const FIXED_INDEX_SUFFIXES: [&str; 9] = [
    "sm_fx_sel_mask",
    "sm_fx_sel",
    "sm_fx_bcast",
    "sm_fx_diff",
    "sm_fx_shift",
    "sm_fx_exp",
    "sm_fx_sum",
    "sm_fx_recip",
    "sm_fx_mul",
];

/// Suffix of the masked relative-logit node `d = mask ⊙ (x - x_{j*})` — the
/// Exp input whose CROWN completion the engagement census counts.
pub(crate) const FIXED_INDEX_SHIFT_SUFFIX: &str = "sm_fx_shift";

/// Per-node rewrite plan, computed before any graph mutation.
struct FixedIndexPlan {
    node_name: String,
    input_name: String,
    /// Resolved softmax axis; always the last axis of `full_shape`.
    axis: usize,
    /// Softmax input (= output) shape in the propagation convention.
    full_shape: Vec<usize>,
    /// 1 at the frozen index `j*` of every row, 0 elsewhere.
    onehot: ArrayD<f32>,
    /// `1 - onehot`: zeroes the `j*` coordinate of `x - x_{j*}` exactly.
    mask: ArrayD<f32>,
    /// The plain-IBP `max_i (u_i - l_{j*})` exceeds the constant-shift guard
    /// (informational; see the module docs).
    wide_ibp: bool,
}

/// The frozen index of every row: the FIRST argmax of each consecutive
/// `n`-element chunk of `values` (logical row-major order, softmax axis last).
///
/// Returns `None` on a non-finite value or a length that is not a multiple of
/// `n`.
pub(crate) fn fixed_index_rows(values: &[f64], n: usize) -> Option<Vec<usize>> {
    if n == 0 || !values.len().is_multiple_of(n) || values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(
        values
            .chunks_exact(n)
            .map(|row| {
                let mut best = 0usize;
                for (k, &v) in row.iter().enumerate().skip(1) {
                    if v > row[best] {
                        best = k;
                    }
                }
                best
            })
            .collect(),
    )
}

/// The centre of an input box as a degenerate (point) box.
fn box_centre(input: &BoundedTensor) -> std::result::Result<BoundedTensor, String> {
    let centre = ndarray::Zip::from(input.lower())
        .and(input.upper())
        .map_collect(|&l, &u| f64::midpoint(f64::from(l), f64::from(u)) as f32);
    if centre.iter().any(|v| !v.is_finite()) {
        return Err("non-finite input-box centre".to_string());
    }
    BoundedTensor::new(centre.clone(), centre).map_err(|e| format!("centre box: {e}"))
}

impl GraphNetwork {
    /// Rewrite every eligible `Softmax` node into the fixed-index primitive
    /// subgraph (see module docs).
    ///
    /// `input` is the verification instance's input box. One plain IBP pass
    /// over the box and one point-collapsing forward at its centre supply the
    /// finiteness check and the frozen index `j*` per softmax row. A node is
    /// SKIPPED (kept on
    /// the direct-LSE path, counted in `report.skipped`) when its bounds are
    /// unavailable or non-finite, when its axis is not the last axis, or on a
    /// helper-name collision. A failed bounds pass skips every node, so
    /// enabling the rewrite can never introduce a new failure mode.
    ///
    /// Prints one unconditional `[softmax-complex] stage=rewrite …` line.
    pub fn decompose_softmax_complex_fixed_index(
        &mut self,
        input: &BoundedTensor,
    ) -> Result<SoftmaxComplexReport> {
        let report = self.decompose_softmax_complex_fixed_index_inner(input)?;
        eprintln!(
            "[softmax-complex] stage=rewrite decomposed={}/{} shift=fixed-index guard_skips={} \
             wide_ibp={}",
            report.decomposed.len(),
            report.decomposed.len() + report.skipped.len(),
            report.skipped.len(),
            report.wide_ibp.len(),
        );
        for (name, reason) in &report.skipped {
            eprintln!("[softmax-complex] stage=rewrite skipped={name} reason={reason}");
        }
        Ok(report)
    }

    fn decompose_softmax_complex_fixed_index_inner(
        &mut self,
        input: &BoundedTensor,
    ) -> Result<SoftmaxComplexReport> {
        let mut report = SoftmaxComplexReport::default();

        let candidates: Vec<String> = self
            .node_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(name.as_str())
                    .is_some_and(|node| matches!(node.layer, Layer::Softmax(_)))
            })
            .cloned()
            .collect();
        if candidates.is_empty() {
            return Ok(report);
        }

        let skip_all = |report: &mut SoftmaxComplexReport, candidates: Vec<String>, why: String| {
            for name in candidates {
                report.skipped.push((name, why.clone()));
            }
        };

        let node_bounds = match self.collect_node_bounds_with_engine_and_deadline(input, None, None)
        {
            Ok(bounds) => bounds,
            Err(err) => {
                debug!("softmax-fixed-index: IBP bounds pass failed ({err}); keeping every node");
                skip_all(
                    &mut report,
                    candidates,
                    format!("IBP bounds unavailable: {err}"),
                );
                return Ok(report);
            }
        };
        let centre_input = match box_centre(input) {
            Ok(centre) => centre,
            Err(why) => {
                skip_all(&mut report, candidates, why);
                return Ok(report);
            }
        };
        // Point-collapsing forward (every node collapsed to its interval centre
        // before it is cached), so the centre stays a point through all three
        // attention blocks. It chooses `j*` only; nothing here is published as
        // a bound, and soundness does not depend on which `j*` it picks.
        let centre_bounds = match self.collect_node_activations_pointwise(&centre_input, None) {
            Ok(bounds) => bounds,
            Err(err) => {
                debug!("softmax-fixed-index: centre-point pass failed ({err}); keeping every node");
                skip_all(
                    &mut report,
                    candidates,
                    format!("centre-point bounds unavailable: {err}"),
                );
                return Ok(report);
            }
        };

        let mut plans: Vec<FixedIndexPlan> = Vec::new();
        for name in candidates {
            match self.plan_fixed_index_node(
                &name,
                input,
                &node_bounds,
                &centre_input,
                &centre_bounds,
            ) {
                Ok(plan) => plans.push(plan),
                Err(reason) => {
                    debug!(
                        "softmax-fixed-index: keeping '{name}' on the direct-LSE path: {reason}"
                    );
                    report.skipped.push((name, reason));
                }
            }
        }

        for plan in plans {
            self.apply_fixed_index_plan(&plan)?;
            info!(
                "softmax-fixed-index: decomposed '{}' (axis {}, shape {:?}, wide_ibp={})",
                plan.node_name, plan.axis, plan.full_shape, plan.wide_ibp
            );
            if plan.wide_ibp {
                report.wide_ibp.push(plan.node_name.clone());
            }
            report.decomposed.push(plan.node_name);
        }
        Ok(report)
    }

    /// Validate one Softmax node and compute its fixed-index plan.
    ///
    /// `Err(reason)` is a human-readable skip reason, NOT a hard error.
    fn plan_fixed_index_node(
        &self,
        name: &str,
        input: &BoundedTensor,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        centre_input: &BoundedTensor,
        centre_bounds: &std::collections::HashMap<String, BoundedTensor>,
    ) -> std::result::Result<FixedIndexPlan, String> {
        let node = self
            .nodes
            .get(name)
            .ok_or_else(|| "node disappeared during planning".to_string())?;
        let Layer::Softmax(softmax) = &node.layer else {
            return Err("not a Softmax node".to_string());
        };
        let input_name = node
            .require_unary_input()
            .map_err(|e| format!("malformed inputs: {e}"))?
            .to_string();

        let (in_bounds, centre) = if input_name == NETWORK_INPUT {
            (input, centre_input)
        } else {
            (
                node_bounds
                    .get(&input_name)
                    .ok_or_else(|| format!("no IBP bounds for input '{input_name}'"))?,
                centre_bounds
                    .get(&input_name)
                    .ok_or_else(|| format!("no centre-point value for input '{input_name}'"))?,
            )
        };

        let shape = in_bounds.shape().to_vec();
        let rank = shape.len();
        if rank == 0 {
            return Err("scalar softmax input".to_string());
        }
        let axis = if softmax.axis < 0 {
            softmax.axis + rank as i32
        } else {
            softmax.axis
        };
        if axis < 0 || axis as usize >= rank {
            return Err(format!(
                "axis {} out of range for rank {}",
                softmax.axis, rank
            ));
        }
        let axis = axis as usize;
        if axis != rank - 1 {
            return Err(format!(
                "fixed-index shift needs the softmax axis to be the last axis (axis {axis} of \
                 rank {rank})"
            ));
        }
        if centre.shape() != shape.as_slice() {
            return Err(format!(
                "centre-point shape {:?} differs from box shape {shape:?}",
                centre.shape()
            ));
        }
        let n = shape[axis];
        if n == 0 {
            return Err("empty softmax axis".to_string());
        }

        let lower: Vec<f32> = in_bounds.lower().iter().copied().collect();
        let upper: Vec<f32> = in_bounds.upper().iter().copied().collect();
        if lower.iter().chain(upper.iter()).any(|v| !v.is_finite()) {
            return Err("non-finite softmax input bounds".to_string());
        }
        let centre_values: Vec<f64> = centre
            .lower()
            .iter()
            .zip(centre.upper().iter())
            .map(|(&l, &u)| f64::midpoint(f64::from(l), f64::from(u)))
            .collect();
        let jstar = fixed_index_rows(&centre_values, n)
            .ok_or_else(|| "non-finite centre-point softmax input".to_string())?;

        let total = lower.len();
        let mut onehot = vec![0.0f32; total];
        let mut max_shifted_upper = f64::NEG_INFINITY;
        for (row, &j) in jstar.iter().enumerate() {
            let base = row * n;
            onehot[base + j] = 1.0;
            let l_j = f64::from(lower[base + j]);
            for k in (0..n).filter(|&k| k != j) {
                max_shifted_upper = max_shifted_upper.max(f64::from(upper[base + k]) - l_j);
            }
        }
        let mask: Vec<f32> = onehot.iter().map(|&v| 1.0 - v).collect();
        let onehot = ArrayD::from_shape_vec(IxDyn(&shape), onehot)
            .map_err(|e| format!("one-hot selector shape: {e}"))?;
        let mask =
            ArrayD::from_shape_vec(IxDyn(&shape), mask).map_err(|e| format!("mask shape: {e}"))?;

        for suffix in FIXED_INDEX_SUFFIXES {
            let candidate = format!("{name}/{suffix}");
            if self.nodes.contains_key(candidate.as_str()) {
                return Err(format!("helper node name '{candidate}' already exists"));
            }
        }

        Ok(FixedIndexPlan {
            node_name: name.to_string(),
            input_name,
            axis,
            full_shape: shape,
            onehot,
            mask,
            wide_ibp: max_shifted_upper > SOFTMAX_COMPLEX_SHIFT_GUARD,
        })
    }

    /// Insert the nine helper nodes before the softmax's position and turn the
    /// softmax node itself into the closing `Clip(0, 1)` (keeping its name, so
    /// downstream references and the graph output stay valid).
    fn apply_fixed_index_plan(&mut self, plan: &FixedIndexPlan) -> Result<()> {
        let [sel_mask, sel, bcast, diff, shift, exp, sum, recip, mul] =
            FIXED_INDEX_SUFFIXES.map(|suffix| format!("{}/{suffix}", plan.node_name));

        let position = self
            .node_order
            .iter()
            .position(|n| n == &plan.node_name)
            .ok_or_else(|| {
                NyError::InternalError(format!(
                    "softmax-fixed-index: node '{}' missing from node_order",
                    plan.node_name
                ))
            })?;

        let axis = plan.axis as i64;
        let n = plan.full_shape[plan.axis];
        let mut reduced_shape = plan.full_shape.clone();
        reduced_shape[plan.axis] = 1;

        // x_{j*} per row: an exact linear selector (one-hot scale + sum).
        let select = MulConstantLayer::try_new(plan.onehot.clone())?;
        // Broadcast [.., 1] -> [.., n] along the (last) softmax axis: a Linear
        // with an all-ones [n x 1] weight and no bias, i.e. y_k = 1 * x_0.
        let broadcast = LinearLayer::new(Array2::<f32>::ones((n, 1)), None)?;
        // Zero the j* coordinate of x - x_{j*} exactly (it is 0 pointwise).
        let zero_jstar = MulConstantLayer::try_new(plan.mask.clone())?;
        let unit_clip = ClipLayer::try_new(0.0, 1.0)?;

        let new_nodes = [
            GraphNode::new(
                &sel_mask,
                Layer::MulConstant(select),
                vec![plan.input_name.clone()],
            ),
            GraphNode::new(
                &sel,
                Layer::ReduceSum(ReduceSumLayer::new(vec![axis], true)),
                vec![sel_mask.clone()],
            ),
            GraphNode::new(&bcast, Layer::Linear(broadcast), vec![sel.clone()]),
            GraphNode::binary(
                &diff,
                Layer::Sub(SubLayer),
                plan.input_name.clone(),
                bcast.clone(),
            ),
            GraphNode::new(&shift, Layer::MulConstant(zero_jstar), vec![diff.clone()]),
            GraphNode::new(&exp, Layer::Exp(ExpLayer::new()), vec![shift.clone()]),
            GraphNode::new(
                &sum,
                Layer::ReduceSum(ReduceSumLayer::new(vec![axis], true)),
                vec![exp.clone()],
            ),
            GraphNode::new(
                &recip,
                Layer::Reciprocal(ReciprocalLayer::new()),
                vec![sum.clone()],
            ),
            GraphNode::binary(
                &mul,
                Layer::MulBinary(MulBinaryLayer),
                exp.clone(),
                recip.clone(),
            ),
        ];
        for (offset, node) in new_nodes.into_iter().enumerate() {
            let node_name = node.name.clone();
            if self.nodes.insert(node_name.clone(), node).is_some() {
                return Err(NyError::InternalError(format!(
                    "softmax-fixed-index: helper node '{node_name}' already existed at apply time"
                )));
            }
            self.node_order.insert(position + offset, node_name);
        }

        let softmax_node = self.nodes.get_mut(&plan.node_name).ok_or_else(|| {
            NyError::InternalError(format!(
                "softmax-fixed-index: node '{}' missing at apply time",
                plan.node_name
            ))
        })?;
        softmax_node.layer = Layer::Clip(unit_clip);
        softmax_node.inputs = vec![mul.clone()];

        // Declared-shape metadata for the taint-degrade path.
        for name in [&sel_mask, &bcast, &diff, &shift, &exp, &mul] {
            self.declared_shapes
                .insert(name.clone(), plan.full_shape.clone());
        }
        for name in [&sel, &sum, &recip] {
            self.declared_shapes
                .insert(name.clone(), reduced_shape.clone());
        }

        // Same cache contract as the constant-shift rewrite: bound caches keyed
        // without graph-structure identity must be dropped, and the exec-order
        // / dispatch caches rebuilt under a fresh semantic identity.
        self.invalidate_forward_linear_cache();
        self.invalidate_exec_order_cache();
        Ok(())
    }

    /// Structural census of the fixed-index rewrite: the `x - x_{j*}` Exp-input
    /// nodes it planted (execution order of insertion) and how many `Softmax`
    /// nodes remain on the direct-LSE path. Read by the `[softmax-complex]`
    /// engagement lines; `(vec![], _)` means the rewrite is absent.
    pub(crate) fn softmax_fixed_index_census(&self) -> (Vec<String>, usize) {
        let suffix = format!("/{FIXED_INDEX_SHIFT_SUFFIX}");
        let mut shift_nodes = Vec::new();
        let mut remaining_softmax = 0usize;
        for name in &self.node_order {
            match self.nodes.get(name.as_str()).map(|node| &node.layer) {
                Some(Layer::MulConstant(_)) if name.ends_with(&suffix) => {
                    shift_nodes.push(name.clone());
                }
                Some(Layer::Softmax(_)) => remaining_softmax += 1,
                _ => {}
            }
        }
        (shift_nodes, remaining_softmax)
    }
}

#[cfg(test)]
#[path = "softmax_fixed_index_tests.rs"]
mod tests;
