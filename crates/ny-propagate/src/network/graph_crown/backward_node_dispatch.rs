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
use crate::bounds::{certified_affine_sum_f32, LinearBounds, OutwardDirection};
use crate::layers::{BoundPropagation, Layer};
use crate::network::backward_dispatch::{
    dispatch_backward_layer, dispatch_backward_layer_finite_boundary, BackwardDispatchResult,
    DispatchContext,
};
use crate::network::core::graph::{graph_crown_dispatch_fallback_reason, GraphNode};
use crate::network::core::GraphNetwork;
use crate::types::CrownIbpFallbackReason;
use crate::MulBinaryRelaxationMode;

use ndarray::{Array1, Array2, ArrayD, Ix1};
use ny_core::{
    dd::{next_down_f64, next_up_f64, two_prod, two_sum},
    GemmEngine, NyError, Result,
};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;
use std::collections::HashMap;
use std::mem::size_of;
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
    deadline: Option<Instant>,
) {
    crate::network::core::try_dense_spatial_patches_reentry_with_deadline(
        node_cb,
        node,
        node_name,
        node_bounds,
        use_patches_mode,
        label,
        deadline,
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

pub(crate) fn concretized_node_bias_with_deadline(
    node_lb: &LinearBounds,
    node_output_bounds: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<ConcretizedBias> {
    let concretized = node_lb.concretize_sound_with_deadline(node_output_bounds, deadline)?;
    check_node_deadline(deadline, "before concretized-bias publication")?;
    let (lower, upper) = concretized.into_parts();
    let lower = lower.into_dimensionality::<Ix1>().map_err(|error| {
        NyError::InternalError(format!(
            "graph CROWN concretized lower bias was not one-dimensional: {error}"
        ))
    })?;
    let upper = upper.into_dimensionality::<Ix1>().map_err(|error| {
        NyError::InternalError(format!(
            "graph CROWN concretized upper bias was not one-dimensional: {error}"
        ))
    })?;
    check_node_deadline(deadline, "after concretized-bias publication")?;
    Ok(ConcretizedBias {
        lower: Box::new(lower),
        upper: Box::new(upper),
    })
}

#[inline]
fn check_node_deadline(deadline: Option<Instant>, phase: &'static str) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(NyError::DeadlineExceeded(format!(
            "graph CROWN node dispatch: deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

fn flat_values_with_deadline<'a>(
    values: &'a ArrayD<f32>,
    deadline: Option<Instant>,
    retained_base_bytes: usize,
    phase: &'static str,
) -> Result<Cow<'a, [f32]>> {
    if let Some(slice) = values.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }
    let source_bytes = values.len().saturating_mul(size_of::<f32>());
    let required_bytes = retained_base_bytes
        .saturating_add(source_bytes)
        .saturating_add(source_bytes);
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: "graph CROWN non-contiguous node-bound flatten",
        });
    }
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(values.len())
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: "graph CROWN non-contiguous node-bound flatten",
        })?;
    let actual_required_bytes = retained_base_bytes
        .saturating_add(source_bytes)
        .saturating_add(copied.capacity().saturating_mul(size_of::<f32>()));
    if actual_required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: actual_required_bytes,
            budget_bytes,
            site: "graph CROWN non-contiguous node-bound flatten",
        });
    }
    for (index, value) in values.iter().copied().enumerate() {
        if index & 1023 == 0 {
            check_node_deadline(deadline, phase)?;
        }
        copied.push(value);
    }
    check_node_deadline(deadline, phase)?;
    Ok(Cow::Owned(copied))
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

#[inline]
fn nonnegative_add_up(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 {
        return f64::INFINITY;
    }
    let (sum, residual) = two_sum(a, b);
    if !sum.is_finite() {
        f64::INFINITY
    } else if residual > 0.0 {
        next_up_f64(sum)
    } else {
        sum
    }
}

#[inline]
fn nonnegative_mul_up(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 {
        return f64::INFINITY;
    }
    let (product, residual) = two_prod(a, b);
    if !product.is_finite() {
        f64::INFINITY
    } else if residual > 0.0 {
        next_up_f64(product)
    } else {
        product
    }
}

#[inline]
fn scaled_coefficient_with_gap(a: f32, midpoint: f64) -> (f32, f64) {
    if !a.is_finite() || !midpoint.is_finite() {
        return (f32::NAN, f64::INFINITY);
    }
    let (product, product_residual) = two_prod(f64::from(a), midpoint);
    let stored = product as f32;
    if !stored.is_finite() {
        return (stored, f64::INFINITY);
    }
    let (cast_gap_hi, cast_gap_lo) = two_sum(f64::from(stored), -product);
    let cast_gap = nonnegative_add_up(cast_gap_hi.abs(), cast_gap_lo.abs());
    (stored, nonnegative_add_up(cast_gap, product_residual.abs()))
}

/// Mirror the graph-alpha reciprocal-scaling Div helper for graph CROWN.
pub(crate) fn backward_div_to_numerator(
    node_lb: &LinearBounds,
    input_a_bounds: &BoundedTensor,
    input_b_bounds: &BoundedTensor,
    node_output_bounds: &BoundedTensor,
) -> Result<DivBackwardResult> {
    backward_div_to_numerator_with_deadline(
        node_lb,
        input_a_bounds,
        input_b_bounds,
        node_output_bounds,
        None,
    )
}

pub(crate) fn backward_div_to_numerator_with_deadline(
    node_lb: &LinearBounds,
    input_a_bounds: &BoundedTensor,
    input_b_bounds: &BoundedTensor,
    node_output_bounds: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<DivBackwardResult> {
    // The reciprocal transform below has several legacy nested-Vec/grouping
    // kernels that are not cooperatively pollable. Under finite authority use
    // the established sound per-node concretization fallback before any clone,
    // coefficient-error fold, broadcast scratch, or reciprocal allocation.
    if deadline.is_some() {
        return concretized_node_bias_with_deadline(node_lb, node_output_bounds, deadline)
            .map(DivBackwardResult::ConcretizeCurrentNode);
    }

    // The reciprocal transform does not propagate coefficient-error matrices.
    // Discharge them over the Div output box first; this converts the carried
    // uncertainty into an ordinary outward bias penalty that the transform keeps.
    check_node_deadline(deadline, "before Div coefficient-error staging")?;
    let mut discharged_node_lb = node_lb.clone();
    check_node_deadline(deadline, "after Div coefficient-error staging")?;
    if discharged_node_lb.has_coeff_err() {
        let output_lower = flat_values_with_deadline(
            node_output_bounds.lower(),
            deadline,
            node_lb
                .memory_bytes()
                .saturating_add(node_output_bounds.len().saturating_mul(size_of::<f32>())),
            "while flattening Div output lower bounds",
        )?;
        let lower_clone_bytes = matches!(&output_lower, Cow::Owned(_))
            .then_some(output_lower.len().saturating_mul(size_of::<f32>()))
            .unwrap_or(0);
        let output_upper = flat_values_with_deadline(
            node_output_bounds.upper(),
            deadline,
            node_lb
                .memory_bytes()
                .saturating_add(node_output_bounds.len().saturating_mul(size_of::<f32>()))
                .saturating_add(lower_clone_bytes),
            "while flattening Div output upper bounds",
        )?;
        discharged_node_lb.fold_coeff_err_into_bias(&output_lower, &output_upper);
    }
    let node_lb = &discharged_node_lb;
    let concretize_current = || {
        concretized_node_bias_with_deadline(node_lb, node_output_bounds, deadline)
            .map(DivBackwardResult::ConcretizeCurrentNode)
    };

    let b_lower_flat = input_b_bounds
        .lower()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div denominator lower not contiguous".into()))?;
    let b_upper_flat = input_b_bounds
        .upper()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div denominator upper not contiguous".into()))?;

    let denominator_valid = b_lower_flat
        .iter()
        .zip(b_upper_flat)
        .all(|(&lower, &upper)| lower.is_finite() && upper.is_finite() && lower <= upper);
    if !denominator_valid {
        return concretize_current();
    }

    // Sound only when every denominator element is sign-definite
    // (0 ∉ [ly, uy]). Different broadcast elements may lie on different sides
    // of zero; the reciprocal center/radius construction is element-local and
    // remains valid for that useful mixed-sign-tensor case. An individual
    // zero-touching interval still requires concretization.
    let every_element_sign_definite = b_lower_flat
        .iter()
        .zip(b_upper_flat)
        .all(|(&lower, &upper)| lower > 0.0 || upper < 0.0);
    if !every_element_sign_definite {
        return concretize_current();
    }

    let num_lower_flat = input_a_bounds
        .lower()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div numerator lower not contiguous".into()))?;
    let num_upper_flat = input_a_bounds
        .upper()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div numerator upper not contiguous".into()))?;
    let numerator_valid = num_lower_flat
        .iter()
        .zip(num_upper_flat)
        .all(|(&lower, &upper)| lower.is_finite() && upper.is_finite() && lower <= upper);
    if !numerator_valid {
        return concretize_current();
    }

    let n = node_lb.num_inputs();
    if n == 0 || num_lower_flat.len() != n {
        return concretize_current();
    }

    // Validate the exact ONNX-style denominator broadcast before mapping flat
    // output columns into denominator groups.  Ignoring an extra leading axis or
    // an oversized broadcast dimension would otherwise leave some denominator
    // elements unused and publish a relaxation for different semantics.
    let b_shape_raw = input_b_bounds.shape();
    let out_shape: Vec<usize> = node_output_bounds.shape().to_vec();
    let Some(out_len) = out_shape
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
    else {
        return concretize_current();
    };
    let ndim = out_shape.len();
    if out_len != n || b_shape_raw.len() > ndim {
        return concretize_current();
    }
    let mut b_shape_aligned = vec![1usize; ndim];
    for (i, &size) in b_shape_raw.iter().rev().enumerate() {
        b_shape_aligned[ndim - 1 - i] = size;
    }
    if b_shape_aligned
        .iter()
        .zip(&out_shape)
        .any(|(&denominator, &output)| denominator != 1 && denominator != output)
    {
        return concretize_current();
    }

    let recip_lower: Vec<f64> = b_upper_flat
        .iter()
        .map(|&v| next_down_f64(1.0 / f64::from(v)))
        .collect();
    let recip_upper: Vec<f64> = b_lower_flat
        .iter()
        .map(|&v| next_up_f64(1.0 / f64::from(v)))
        .collect();
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
        .zip(r_mid.iter())
        .map(|((&rl, &ru), &mid)| next_up_f64(mid - rl).max(next_up_f64(ru - mid)))
        .collect();

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
            return concretize_current();
        }
        groups[b_flat].push(out_flat);
    }

    let mut new_lower_a = node_lb.lower_a().to_owned();
    let mut new_upper_a = node_lb.upper_a().to_owned();
    let mut new_lower_b = node_lb.lower_b().to_owned();
    let mut new_upper_b = node_lb.upper_b().to_owned();

    for spec_idx in 0..node_lb.num_outputs() {
        for g in 0..b_len {
            let mut lower_center_gap = 0.0;
            let mut upper_center_gap = 0.0;
            for &elem in &groups[g] {
                let lo = new_lower_a[[spec_idx, elem]];
                let up = new_upper_a[[spec_idx, elem]];

                // r_mid is sign-definite (matches the denominator sign) but may
                // be negative; only require it be finite and nonzero.
                debug_assert!(r_mid[g].is_finite() && r_mid[g] != 0.0);
                let (scaled_lo, gap_lo) = scaled_coefficient_with_gap(lo, r_mid[g]);
                let (scaled_up, gap_up) = scaled_coefficient_with_gap(up, r_mid[g]);
                new_lower_a[[spec_idx, elem]] = scaled_lo;
                new_upper_a[[spec_idx, elem]] = scaled_up;
                lower_center_gap = nonnegative_add_up(
                    lower_center_gap,
                    nonnegative_mul_up(gap_lo, num_abs_max[elem]),
                );
                upper_center_gap = nonnegative_add_up(
                    upper_center_gap,
                    nonnegative_mul_up(gap_up, num_abs_max[elem]),
                );
            }

            let lower_abs_sum = certified_affine_sum_f32(
                0.0,
                groups[g].iter().map(|&elem| {
                    (
                        node_lb.lower_a()[[spec_idx, elem]].abs(),
                        num_abs_max[elem] as f32,
                    )
                }),
                OutwardDirection::Upper,
            );
            let upper_abs_sum = certified_affine_sum_f32(
                0.0,
                groups[g].iter().map(|&elem| {
                    (
                        node_lb.upper_a()[[spec_idx, elem]].abs(),
                        num_abs_max[elem] as f32,
                    )
                }),
                OutwardDirection::Upper,
            );
            let lower_penalty = nonnegative_add_up(
                nonnegative_mul_up(r_delta[g], lower_abs_sum),
                lower_center_gap,
            );
            let upper_penalty = nonnegative_add_up(
                nonnegative_mul_up(r_delta[g], upper_abs_sum),
                upper_center_gap,
            );
            new_lower_b[spec_idx] = next_down_f32(next_down_f64(
                f64::from(new_lower_b[spec_idx]) - lower_penalty,
            ) as f32);
            new_upper_b[spec_idx] =
                next_up_f32(next_up_f64(f64::from(new_upper_b[spec_idx]) + upper_penalty) as f32);
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
    // The legacy Cut-CROWN C2 fold seam used to sit here. It was deleted: its
    // proof authority was hard-false, so it could only ever return the incoming
    // `node_lb` untouched, and its arithmetic was experiment-grade (plain f32,
    // no directed rounding, no widened coefficient error).
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
    /// The Dense carrier was transactionally materialized from Patches while
    /// this node's finite authority was live. Ordinary deadline-bearing Dense
    /// carriers leave this false and retain the historical dispatch policy.
    pub finite_structured_boundary: bool,
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
        finite_structured_boundary,
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

    let dispatch_result = if *finite_structured_boundary {
        dispatch_backward_layer_finite_boundary(&dispatch_ctx, node_lb)
    } else {
        dispatch_backward_layer(&dispatch_ctx, node_lb)
    };

    match dispatch_result {
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
