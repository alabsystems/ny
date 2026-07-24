// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared per-node backward dispatch helpers for graph CROWN.
//!
//! Both `propagation.rs` and `spec_propagation.rs` run the same per-node
//! backward loop body: deadline budgeting, Dense→Patches re-entry, and
//! layer-specific dispatch. This module extracts the duplicated core so
//! each coordinator only contains its site-specific accumulation logic.
//!
//! Part of #3935 / design: `designs/2026-03-16-graph-crown-backward-loop-dedup.md`

use crate::bounds::patches::CrownBounds;
use crate::bounds::LinearBounds;
use crate::layers::{BoundPropagation, Layer};
use crate::network::backward_dispatch::{
    dispatch_backward_layer, BackwardDispatchResult, DispatchContext,
};
use crate::network::core::graph::{graph_crown_dispatch_fallback_reason, GraphNode};
use crate::network::core::GraphNetwork;
use crate::types::CrownIbpFallbackReason;
use crate::MulBinaryRelaxationMode;

use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::debug;

/// Compute a per-node deadline from the remaining global budget (#3795).
///
/// Budget policy: `per_node = max(remaining / nodes_left, remaining * fraction)`.
/// Returns `None` only when the global deadline has already expired.
/// Sub-floor shares keep the global deadline (#3881).
///
/// Shared between `propagation.rs` and `spec_propagation.rs`.
pub(super) fn compute_node_deadline(
    deadline: Option<Instant>,
    node_index: usize,
    total_backward_nodes: usize,
    max_budget_fraction: f64,
    min_node_budget_secs: f64,
) -> Option<Instant> {
    deadline.and_then(|d| {
        let now = Instant::now();
        if now >= d {
            return None;
        }
        let remaining = d.duration_since(now);
        let remaining_secs = remaining.as_secs_f64();
        let remaining_count = total_backward_nodes.saturating_sub(node_index).max(1);
        let equal_share = remaining_secs / remaining_count as f64;
        let fraction_share = remaining_secs * max_budget_fraction;
        let per_node_secs = equal_share.max(fraction_share);
        if per_node_secs < min_node_budget_secs {
            return Some(d); // keep global deadline (#3881)
        }
        Some(now + Duration::from_secs_f64(per_node_secs))
    })
}

/// Try to convert Dense bounds to Patches at a unary Conv2d boundary (#3813).
///
/// When the classifier-head logits reach Conv2d nodes through Dense rows,
/// this re-enters Patches mode so the CNN trunk backward runs on the
/// efficient patches implementation. Gated by `use_patches_mode`
/// (abcrown.py:228-231).
///
/// Shared between `propagation.rs` and `spec_propagation.rs`.
pub(super) fn try_patches_reentry(
    node_cb: &mut CrownBounds,
    node: &GraphNode,
    node_bounds: &HashMap<String, BoundedTensor>,
    node_name: &str,
    use_patches_mode: bool,
    label: &str,
) {
    crate::network::core::try_dense_spatial_patches_reentry(
        node_cb,
        node,
        node_name,
        node_bounds,
        use_patches_mode,
        label,
    );
}

/// Result from dispatching a ReLU backward step.
///
/// The graph-CROWN coordinators share the same ReLU dispatch path, but each
/// caller owns its fallback policy (full IBP fallback vs. per-node
/// concretization), so the helper only reports success vs. fallback.
pub(super) enum NodeDispatchResult {
    /// ReLU succeeded — caller accumulates the new Dense bounds to the first input.
    SingleDense(Box<LinearBounds>),
    /// Layer not supported — caller should fall back to IBP.
    IbpFallback(CrownIbpFallbackReason),
}

/// Result from the shared non-ReLU graph dispatch core (#3936).
pub(super) enum SharedDispatchResult {
    Dispatch(Box<BackwardDispatchResult>),
    IbpFallback(CrownIbpFallbackReason),
}

/// Convert a node's current linear form into constant bias bounds.
pub(crate) fn concretized_node_bias(
    node_lb: &LinearBounds,
    node_output_bounds: &BoundedTensor,
) -> ConcretizedBias {
    let concretized = node_lb.concretize_sound(node_output_bounds).flatten();
    ConcretizedBias {
        lower: Box::new(Array1::from_vec(
            concretized.lower().iter().copied().collect(),
        )),
        upper: Box::new(Array1::from_vec(
            concretized.upper().iter().copied().collect(),
        )),
    }
}

/// Concretized bias bounds from a node's linear form.
pub(crate) struct ConcretizedBias {
    pub lower: Box<Array1<f32>>,
    pub upper: Box<Array1<f32>>,
}

/// Shared Div backward result used by graph CROWN and spec-CROWN.
pub(crate) enum DivBackwardResult {
    PropagateNumerator(Box<LinearBounds>),
    ConcretizeCurrentNode(ConcretizedBias),
}

/// Mirror the graph-alpha reciprocal-scaling Div helper for graph CROWN.
pub(crate) fn backward_div_to_numerator(
    node_lb: &LinearBounds,
    input_a_bounds: &BoundedTensor,
    input_b_bounds: &BoundedTensor,
    node_output_bounds: &BoundedTensor,
) -> Result<DivBackwardResult> {
    let b_lower_flat = input_b_bounds
        .lower()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div denominator lower not contiguous".into()))?;
    let b_upper_flat = input_b_bounds
        .upper()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div denominator upper not contiguous".into()))?;

    // Sound only when the denominator interval is sign-definite (0 ∉ [ly, uy]),
    // i.e. every element is strictly positive OR every element is strictly
    // negative. A mixed-sign or zero-touching denominator makes 1/y unbounded,
    // so we keep the sound concretization fallback. Reciprocal-scaling below is
    // identical for both signs: r = 1/y ∈ [1/uy, 1/ly] (recip is monotone
    // increasing on each side of zero), r_mid carries the sign of 1/y and
    // r_delta ≥ 0 is the half-width error radius — sign-independent (#Div-neg).
    let all_pos = b_lower_flat.iter().all(|&v| v > 0.0);
    let all_neg = b_upper_flat.iter().all(|&v| v < 0.0);
    if !(all_pos || all_neg) {
        return Ok(DivBackwardResult::ConcretizeCurrentNode(
            concretized_node_bias(node_lb, node_output_bounds),
        ));
    }

    let num_lower_flat = input_a_bounds
        .lower()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div numerator lower not contiguous".into()))?;
    let num_upper_flat = input_a_bounds
        .upper()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div numerator upper not contiguous".into()))?;

    let n = node_lb.num_inputs();
    if num_lower_flat.len() != n {
        return Ok(DivBackwardResult::ConcretizeCurrentNode(
            concretized_node_bias(node_lb, node_output_bounds),
        ));
    }

    let recip_lower: Vec<f64> = b_upper_flat.iter().map(|&v| 1.0 / (v as f64)).collect();
    let recip_upper: Vec<f64> = b_lower_flat.iter().map(|&v| 1.0 / (v as f64)).collect();
    let num_abs_max: Vec<f64> = num_lower_flat
        .iter()
        .zip(num_upper_flat.iter())
        .map(|(&lo, &up)| (lo.abs() as f64).max(up.abs() as f64))
        .collect();
    let r_mid: Vec<f64> = recip_lower
        .iter()
        .zip(recip_upper.iter())
        // Bit-identical to `(rl + ru) / 2.0` here: |1/(f32 as f64)| is either 0,
        // in [2.9e-39, 7.2e44], or ±inf/NaN — never in midpoint's rescale ranges.
        .map(|(&rl, &ru)| f64::midpoint(rl, ru))
        .collect();
    let r_delta: Vec<f64> = recip_lower
        .iter()
        .zip(recip_upper.iter())
        .map(|(&rl, &ru)| (ru - rl) / 2.0)
        .collect();

    let b_shape_raw = input_b_bounds.shape();
    let out_shape: Vec<usize> = node_output_bounds.shape().to_vec();
    let ndim = out_shape.len();
    let mut b_shape_aligned = vec![1usize; ndim];
    for (i, &s) in b_shape_raw.iter().rev().enumerate() {
        if i < ndim {
            b_shape_aligned[ndim - 1 - i] = s;
        }
    }
    let b_len = b_lower_flat.len();
    let mut groups: Vec<Vec<usize>> = vec![vec![]; b_len];
    for out_flat in 0..n {
        let mut remaining = out_flat;
        let mut b_flat = 0;
        let mut b_stride = 1;
        for d in (0..ndim).rev() {
            let out_idx_d = remaining % out_shape[d];
            remaining /= out_shape[d];
            let b_idx_d = if b_shape_aligned[d] == 1 {
                0
            } else {
                out_idx_d
            };
            b_flat += b_idx_d * b_stride;
            b_stride *= b_shape_aligned[d];
        }
        if b_flat >= b_len {
            return Ok(DivBackwardResult::ConcretizeCurrentNode(
                concretized_node_bias(node_lb, node_output_bounds),
            ));
        }
        groups[b_flat].push(out_flat);
    }

    let mut new_lower_a = node_lb.lower_a().to_owned();
    let mut new_upper_a = node_lb.upper_a().to_owned();
    let mut new_lower_b = node_lb.lower_b().to_owned();
    let mut new_upper_b = node_lb.upper_b().to_owned();

    for spec_idx in 0..node_lb.num_outputs() {
        for g in 0..b_len {
            let mut lower_abs_sum = 0.0_f64;
            let mut upper_abs_sum = 0.0_f64;

            for &elem in &groups[g] {
                let lo = new_lower_a[[spec_idx, elem]] as f64;
                let up = new_upper_a[[spec_idx, elem]] as f64;

                // r_mid is sign-definite (matches the denominator sign) but may
                // be negative; only require it be finite and nonzero.
                debug_assert!(r_mid[g].is_finite() && r_mid[g] != 0.0);
                new_lower_a[[spec_idx, elem]] = next_down_f32((lo * r_mid[g]) as f32);
                new_upper_a[[spec_idx, elem]] = next_up_f32((up * r_mid[g]) as f32);

                lower_abs_sum += lo.abs() * num_abs_max[elem];
                upper_abs_sum += up.abs() * num_abs_max[elem];
            }

            new_lower_b[spec_idx] -= next_up_f32((r_delta[g] * lower_abs_sum) as f32);
            new_upper_b[spec_idx] += next_up_f32((r_delta[g] * upper_abs_sum) as f32);
        }
    }

    // Migrated from from_parts_unchecked: reciprocal-scaling arithmetic can
    // produce NaN (e.g., Inf * 0.0) or Inf (near-zero denominator overflow).
    // NaN firewall falls back to conservative bounds. See #3438.
    Ok(DivBackwardResult::PropagateNumerator(Box::new(
        LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)?,
    )))
}

/// Shared MulBinary backward result for graph CROWN and spec-CROWN.
pub(super) enum MulBinaryDispatchResult {
    BinaryDense {
        bounds_a: Box<LinearBounds>,
        bounds_b: Box<LinearBounds>,
        bias_lower: Box<Array1<f32>>,
        bias_upper: Box<Array1<f32>>,
    },
    SoftmaxNonFinite,
    RecoverableError(NyError),
}

/// Context for shared MulBinary backward dispatch.
pub(super) struct MulBinaryDispatchCtx<'a> {
    pub node: &'a GraphNode,
    pub node_name: &'a str,
    pub node_lb: &'a LinearBounds,
    pub input_a_bounds: &'a BoundedTensor,
    pub input_b_bounds: &'a BoundedTensor,
    pub mul_binary_relaxation: MulBinaryRelaxationMode,
    pub mul_binary_alpha: Option<&'a Array2<f32>>,
    pub softmax_decomposition: bool,
    pub label: &'a str,
}

/// Dispatch MulBinary backward and normalize the split-path result.
pub(super) fn dispatch_mul_binary_backward(
    ctx: &MulBinaryDispatchCtx<'_>,
) -> Result<MulBinaryDispatchResult> {
    let Layer::MulBinary(mul) = &ctx.node.layer else {
        return Err(NyError::InvalidSpec(format!(
            "{} expected MulBinary at node '{}'",
            ctx.label, ctx.node_name,
        )));
    };

    match if ctx.mul_binary_alpha.is_some() {
        mul.propagate_linear_binary_with_alpha(
            ctx.node_lb,
            ctx.input_a_bounds,
            ctx.input_b_bounds,
            ctx.mul_binary_alpha,
        )
    } else {
        mul.propagate_linear_binary(
            ctx.node_lb,
            ctx.input_a_bounds,
            ctx.input_b_bounds,
            ctx.mul_binary_relaxation,
        )
    } {
        Ok((mut lb_a, mut lb_b)) => {
            if ctx.softmax_decomposition {
                let has_bad = lb_a
                    .lower_a()
                    .iter()
                    .chain(lb_a.upper_a().iter())
                    .chain(lb_a.lower_b().iter())
                    .chain(lb_a.upper_b().iter())
                    .chain(lb_b.lower_a().iter())
                    .chain(lb_b.upper_a().iter())
                    .chain(lb_b.lower_b().iter())
                    .chain(lb_b.upper_b().iter())
                    .any(|&v| !v.is_finite());
                if has_bad {
                    debug!(
                        "{}: MulBinary '{}' softmax {:?} produced inf/NaN",
                        ctx.label, ctx.node_name, ctx.mul_binary_relaxation,
                    );
                    return Ok(MulBinaryDispatchResult::SoftmaxNonFinite);
                }
            }

            let bias_lower = Box::new(lb_a.lower_b() + lb_b.lower_b());
            let bias_upper = Box::new(lb_a.upper_b() + lb_b.upper_b());
            lb_a.lower_b_mut().fill(0.0);
            lb_a.upper_b_mut().fill(0.0);
            lb_b.lower_b_mut().fill(0.0);
            lb_b.upper_b_mut().fill(0.0);
            GraphNetwork::verify_split_path_bias_zero(&lb_a, "MulBinary lhs split path")?;
            GraphNetwork::verify_split_path_bias_zero(&lb_b, "MulBinary rhs split path")?;

            Ok(MulBinaryDispatchResult::BinaryDense {
                bounds_a: Box::new(lb_a),
                bounds_b: Box::new(lb_b),
                bias_lower,
                bias_upper,
            })
        }
        Err(
            err @ NyError::UnsupportedOp(_)
            | err @ NyError::UnsupportedConfiguration(_)
            | err @ NyError::NumericalInstability(_)
            | err @ NyError::ShapeMismatch { .. }
            | err @ NyError::DeadlineExceeded(_),
        ) => Ok(MulBinaryDispatchResult::RecoverableError(err)),
        Err(err @ NyError::SoundnessRefusal(_) | err @ NyError::InternalError(_)) => Err(err),
        Err(err) => Err(NyError::InvalidSpec(format!(
            "{} failed at node '{}' (MulBinary): {}",
            ctx.label, ctx.node_name, err,
        ))),
    }
}

/// Check Linear layer dimension compatibility before dispatch (#2817).
///
/// Returns `true` when a Linear node must fall back to IBP with `ShapeMismatch`.
pub(super) fn linear_dimension_mismatch(node: &GraphNode, node_lb: &LinearBounds) -> bool {
    if let Layer::Linear(l) = &node.layer {
        let expected_inputs = l.out_features();
        let got_inputs = node_lb.num_inputs();
        // Guard: zero out_features is always a mismatch (and keeps the
        // multiple-of check on a nonzero divisor). (#2817)
        expected_inputs == 0
            || (got_inputs != expected_inputs && !got_inputs.is_multiple_of(expected_inputs))
    } else {
        false
    }
}

/// Dispatch ReLU backward using reused alpha when available, else the
/// heuristic `propagate_crown_backward` path.
pub(super) fn dispatch_relu_backward(
    cut_fold_scope: crate::beta_crown::bab_cuts::CutFoldScope,
    node: &GraphNode,
    node_lb: &LinearBounds,
    pre_activation: &BoundedTensor,
    node_name: &str,
    label: &str,
    alpha_lower: Option<&Array1<f32>>,
    alpha_upper: Option<&Array1<f32>>,
) -> Result<NodeDispatchResult> {
    // Verify unary input exists (ReLU is always single-input).
    let _first_input = node.require_unary_input()?;
    // === Certified Cut-CROWN C2 dark gate (docs/CERTIFIED_CUT_CROWN_DESIGN.md §C2) ===
    // When NY_CUT_FOLD=1 and a cut set is registered for this ReLU node OF
    // THIS GRAPH (`cut_fold_scope` — never a same-named node of another
    // graph), fold the λ-scaled cut weights onto the LOWER-side
    // POST-activation coefficients (BEFORE relaxation selection — λ·cc
    // multiplies relu(ẑ), the node output) and the −Σλ·B constant onto the
    // lower bias. Sound for any λ ≥ 0 with a valid cut bound B
    // (`cuts_fold_lower_bound`); upper side untouched.
    // Default OFF: env unset / empty registry ⇒ `None` ⇒ byte-identical path.
    let cut_folded =
        crate::beta_crown::bab_cuts::fold_lower_side(cut_fold_scope, node_name, node_lb);
    let node_lb: &LinearBounds = cut_folded.as_ref().unwrap_or(node_lb);
    let result = match (&node.layer, alpha_lower) {
        (Layer::ReLU(relu), Some(alpha_lower)) => relu
            .propagate_linear_with_alpha(node_lb, pre_activation, alpha_lower, alpha_upper)
            .map(|(new_lb, _grad_lower, _grad_upper)| new_lb),
        _ => node
            .layer
            .propagate_crown_backward(node_lb, Some(pre_activation)),
    };
    match result {
        Ok(mut new_lb) => {
            // Eager per-row discharge of the carried coefficient error over the
            // (CROWN-tightened) pre-activation cut (#cgan-conv-err-compose, see
            // LinearBounds::fold_coeff_err_over_box_eager for the enclosure and
            // tightness argument). Rows with a non-finite penalty keep carrying.
            new_lb.fold_coeff_err_over_box_eager(pre_activation);
            Ok(NodeDispatchResult::SingleDense(Box::new(new_lb)))
        }
        Err(err) => match graph_crown_dispatch_fallback_reason(&err) {
            Some(reason) => {
                debug!("{label}: ReLU '{node_name}' dispatch fallback ({reason:?})");
                Ok(NodeDispatchResult::IbpFallback(reason))
            }
            None if matches!(
                err,
                NyError::SoundnessRefusal(_) | NyError::InternalError(_)
            ) =>
            {
                Err(NyError::InternalError(format!(
                    "{label}: ReLU '{node_name}' hard error"
                )))
            }
            None => Err(NyError::InvalidSpec(format!(
                "{label} failed at node '{node_name}' (ReLU): {err}",
            ))),
        },
    }
}

/// Context for shared dispatch core — avoids clippy::too_many_arguments.
pub(super) struct SharedDispatchCtx<'a> {
    pub node: &'a GraphNode,
    pub node_name: &'a str,
    pub node_lb: &'a LinearBounds,
    pub pre_activation: &'a BoundedTensor,
    pub network_input: &'a BoundedTensor,
    pub node_bounds: &'a HashMap<String, BoundedTensor>,
    pub engine: Option<&'a dyn GemmEngine>,
    pub node_deadline: Option<Instant>,
    pub mul_binary_relaxation: MulBinaryRelaxationMode,
    pub label: &'a str,
}

/// Dispatch a node through the shared backward dispatch core (#1949 Step B).
///
/// Handles: Linear, Transpose, Conv{1d,2d,Transpose{1d,2d}}, Add, Sub, Concat,
/// MatMul, BilinearCrown, SkipMerge, OpaqueSkip, and all unary layers.
/// ReLU returns Unsupported from the shared dispatch (handled site-specifically above).
///
/// Shared between `propagation.rs` and `spec_propagation.rs`.
pub(super) fn dispatch_shared_core(ctx: &SharedDispatchCtx<'_>) -> Result<SharedDispatchResult> {
    let SharedDispatchCtx {
        node,
        node_name,
        node_lb,
        pre_activation,
        network_input,
        node_bounds,
        engine,
        node_deadline,
        mul_binary_relaxation,
        label,
    } = ctx;
    let dispatch_ctx = DispatchContext {
        node_name,
        layer: &node.layer,
        inputs: &node.inputs,
        pre_activation,
        network_input,
        node_bounds: (*node_bounds).into(),
        engine: *engine,
        deadline: *node_deadline,
        bilinear_alphas: None,
        mul_binary_relaxation: *mul_binary_relaxation,
        mul_binary_alphas: None,
        norm_inv_rms_override: None,
    };

    match dispatch_backward_layer(&dispatch_ctx, node_lb) {
        Ok(result) => match result {
            BackwardDispatchResult::Unsupported(reason) => {
                debug!(
                    "{}: {} ({}) not supported ({})",
                    label,
                    node_name,
                    node.layer.layer_type(),
                    reason,
                );
                Ok(SharedDispatchResult::IbpFallback(
                    CrownIbpFallbackReason::CrownPropagationError,
                ))
            }
            other => Ok(SharedDispatchResult::Dispatch(Box::new(other))),
        },
        Err(err) => {
            if let Some(reason) = graph_crown_dispatch_fallback_reason(&err) {
                debug!(
                    "{}: {} ({}) error ({}), fallback {:?}",
                    label,
                    node_name,
                    node.layer.layer_type(),
                    err,
                    reason,
                );
                return Ok(SharedDispatchResult::IbpFallback(reason));
            }
            Err(err)
        }
    }
}
