// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared graph-wide backward helpers for CROWN bound accumulation.
//!
//! These helpers are used across graph-CROWN, graph-alpha, and beta-CROWN graph
//! constraint coordinators. Hoisted from `graph_crown/utils.rs` (#3936) to make
//! the cross-engine dependency explicit and eliminate module-boundary leaks.

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::bounds::LinearBounds;
use crate::layers::Layer;
use crate::network::backward_dispatch::BackwardDispatchResult;

use super::merge_accumulator::CrownMergeAccumulator;
use super::{GraphNetwork, GraphNode, NETWORK_INPUT};

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use tracing::debug;

/// Return the fixed boolean select mask for a Where condition iff that condition
/// is bound-independent (a constant), i.e. `lower == upper` at every position.
///
/// `mask[i] == true` means the condition selects the true branch at flat
/// position `i` (ONNX treats any non-zero condition value as true; here the
/// bounds are integer-coded 0/1 so we threshold at 0.5). Returns `None` when the
/// condition is data-dependent (any position has a non-degenerate interval),
/// signalling the caller to keep the sound IBP/concretize fallback.
///
/// Shared by the graph-CROWN and graph-α backward Where arms (#Where-const-cond).
pub(crate) fn where_constant_mask(cond: &BoundedTensor) -> Option<Vec<bool>> {
    let lower = cond.lower();
    let upper = cond.upper();
    let mut mask = Vec::with_capacity(lower.len());
    for (&lo, &hi) in lower.iter().zip(upper.iter()) {
        // Data-dependent condition: the interval straddles the 0.5 decision
        // boundary or is otherwise non-degenerate. Bail to the loose fallback.
        if lo != hi {
            return None;
        }
        if !lo.is_finite() {
            return None;
        }
        mask.push(lo >= 0.5);
    }
    Some(mask)
}

/// Build the exact backward `LinearBounds` for one branch of a constant-condition
/// Where by zeroing the A-matrix columns that belong to the other branch.
///
/// Column `i` of the incoming `node_lb` corresponds to flat output position `i`
/// of the Where. For the true branch (`keep_true == true`) we keep column `i`
/// only where `mask[i]` is true; for the false branch we keep it where `mask[i]`
/// is false.
///
/// The two branch contributions are accumulated separately by the caller, so the
/// incoming bias must NOT be applied twice. We carry it on the true branch only
/// and zero it on the false branch.
pub(crate) fn mask_linear_bounds_columns(
    node_lb: &LinearBounds,
    mask: &[bool],
    keep_true: bool,
) -> LinearBounds {
    let mut lower_a = node_lb.lower_a().clone();
    let mut upper_a = node_lb.upper_a().clone();
    for (col, &m) in mask.iter().enumerate() {
        if m != keep_true {
            // This output position belongs to the other branch — zero the column
            // so this branch contributes nothing through it.
            for row in 0..lower_a.nrows() {
                lower_a[[row, col]] = 0.0;
                upper_a[[row, col]] = 0.0;
            }
        }
    }
    // Bias is a per-output constant independent of the split; apply it on exactly
    // one of the two branch paths to avoid double-counting.
    let (lower_b, upper_b) = if keep_true {
        (node_lb.lower_b().clone(), node_lb.upper_b().clone())
    } else {
        (
            Array1::zeros(node_lb.lower_b().len()),
            Array1::zeros(node_lb.upper_b().len()),
        )
    };
    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
        .expect("masking preserves A/bias shapes; new_or_conservative cannot fail on shape")
}

/// Minimum dense row count for Dense->Patches re-entry
/// (#cgan-alpha-on-tight-refs). The patches-mode ReLU backward selects the
/// lower/upper envelope PER TAP; when conv receptive fields overlap
/// (stride < kernel), taps of one input neuron with mixed signs each pay
/// their own relaxation intercept, while the dense backward sign-selects on
/// the SUMMED coefficient — strictly tighter. Measured on cGAN_imgSz32_nCh_1
/// prop_1: the 1-row objective backward concretizes to -7.7e7 through the
/// patches segment vs -96.9 in matrix mode (~8e5x looser), which froze the
/// alpha warmup and the per-domain BaB rebound at the root bound. Thin seeds
/// are also CHEAPER dense (rows x in_dim matrices), so route them to matrix
/// mode; patches remains for many-row seeds where dense materialization is
/// the memory wall. `NY_PATCHES_REENTRY_MIN_ROWS` overrides (1 restores the
/// pre-fix always-re-enter behavior).
fn patches_reentry_min_rows() -> usize {
    static MIN_ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN_ROWS.get_or_init(|| {
        std::env::var("NY_PATCHES_REENTRY_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(5)
    })
}

pub(crate) fn try_dense_spatial_patches_reentry(
    node_cb: &mut CrownBounds,
    node: &GraphNode,
    node_name: &str,
    current_bounds: &HashMap<String, BoundedTensor>,
    use_patches_mode: bool,
    label: &str,
) -> bool {
    if !use_patches_mode
        || !matches!(node_cb, CrownBounds::Dense(_))
        || node.inputs.len() != 1
        || !matches!(&node.layer, Layer::Conv2d(_))
    {
        return matches!(node_cb, CrownBounds::Patches(_));
    }
    if let CrownBounds::Dense(lb) = &*node_cb {
        let rows = lb.num_outputs();
        if rows < patches_reentry_min_rows() {
            debug!(
                "{}: Dense->Patches re-entry skipped at {} ({} rows < min {}): \
                 matrix mode is tighter for thin seeds through overlapping \
                 receptive fields (#cgan-alpha-on-tight-refs)",
                label,
                node_name,
                rows,
                patches_reentry_min_rows()
            );
            return false;
        }
    }

    let Some(current_bounds) = current_bounds.get(node_name) else {
        return false;
    };

    let current_shape = current_bounds.shape();
    if current_shape.len() != 3 {
        return false;
    }

    let spatial = (current_shape[0], current_shape[1], current_shape[2]);
    let spatial_dim = spatial.0 * spatial.1 * spatial.2;
    if let CrownBounds::Dense(lb) = node_cb {
        if lb.num_inputs() == spatial_dim {
            match PatchesLinearBounds::from_dense_spatial_rows(lb, spatial) {
                Ok(pb) => {
                    debug!(
                        "{}: Dense->Patches re-entry at {} with {} rows over {:?}",
                        label, node_name, pb.row_count, spatial
                    );
                    *node_cb = CrownBounds::Patches(Box::new(pb));
                }
                Err(err) => {
                    debug!(
                        "{}: Dense->Patches re-entry skipped at {}: {}",
                        label, node_name, err
                    );
                }
            }
        }
    }

    matches!(node_cb, CrownBounds::Patches(_))
}

// Justification: this helper needs the graph/node context, pass-through bounds,
// accumulator frontier, dimensions, and label to deduplicate the shell without
// taking over caller-owned fallback policy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_dense_backward_dispatch_result(
    graph: &GraphNetwork,
    node: &GraphNode,
    first_input: &str,
    pass_through_bounds: &LinearBounds,
    result: BackwardDispatchResult,
    node_crown_bounds: &mut CrownMergeAccumulator,
    output_dim: usize,
    input_dim: usize,
    input_accumulated: &mut bool,
    context_prefix: &str,
) -> Result<()> {
    match result {
        BackwardDispatchResult::Single(new_lb) => graph.accumulate_dense_bounds_to_input(
            first_input,
            *new_lb,
            node_crown_bounds,
            output_dim,
            input_dim,
            input_accumulated,
        ),
        BackwardDispatchResult::Binary {
            bounds_a,
            bounds_b,
            bias_lower,
            bias_upper,
        } => {
            let (input_a_name, input_b_name) = node.require_binary_inputs()?;
            GraphNetwork::accumulate_bias_to_network_input_crown(
                &bias_lower,
                &bias_upper,
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
            );
            GraphNetwork::verify_split_path_bias_zero(
                &bounds_a,
                &format!("{context_prefix} binary lhs split path"),
            )?;
            GraphNetwork::verify_split_path_bias_zero(
                &bounds_b,
                &format!("{context_prefix} binary rhs split path"),
            )?;
            graph.accumulate_dense_bounds_to_input(
                input_a_name,
                *bounds_a,
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
            )?;
            graph.accumulate_dense_bounds_to_input(
                input_b_name,
                *bounds_b,
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
            )
        }
        BackwardDispatchResult::Nary {
            bounds,
            bias_lower,
            bias_upper,
        } => {
            GraphNetwork::accumulate_bias_to_network_input_crown(
                &bias_lower,
                &bias_upper,
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
            );
            for (graph_idx, lb) in bounds.into_iter().flatten().enumerate() {
                GraphNetwork::verify_split_path_bias_zero(
                    &lb,
                    &format!("{context_prefix} n-ary split path"),
                )?;
                if let Some(inp_name) = node.inputs.get(graph_idx) {
                    graph.accumulate_dense_bounds_to_input(
                        inp_name,
                        lb,
                        node_crown_bounds,
                        output_dim,
                        input_dim,
                        input_accumulated,
                    )?;
                }
            }
            Ok(())
        }
        BackwardDispatchResult::PassThrough => graph.accumulate_dense_bounds_to_input(
            first_input,
            pass_through_bounds.clone(),
            node_crown_bounds,
            output_dim,
            input_dim,
            input_accumulated,
        ),
        BackwardDispatchResult::Unsupported(reason) => Err(NyError::UnsupportedOp(reason)),
    }
}

/// Try patches-native residual passthrough for Add/Sub nodes (#4382).
///
/// Returns `Ok(true)` if this node was handled in patches form (caller should
/// skip the Dense fallback). Returns `Ok(false)` if not applicable.
///
/// Only handles same-shape `Add`/`Sub` (no broadcast). The original bias is
/// routed via `accumulate_bias_to_network_input_crown`; per-input carriers
/// have zero bias to avoid double-counting.
///
/// Reference: alpha-beta-CROWN `operators/add_sub.py:37-47`
// Justification: matches the signature pattern of apply_dense_backward_dispatch_result
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_apply_patches_residual_passthrough(
    graph: &GraphNetwork,
    node: &GraphNode,
    node_cb: &CrownBounds,
    node_bounds: &HashMap<String, BoundedTensor>,
    node_crown_bounds: &mut CrownMergeAccumulator,
    output_dim: usize,
    input_dim: usize,
    input_accumulated: &mut bool,
    context_prefix: &str,
) -> Result<bool> {
    let pb = match node_cb {
        CrownBounds::Patches(pb) => pb,
        CrownBounds::Dense(_) => return Ok(false),
    };

    let is_add = matches!(&node.layer, Layer::Add(_));
    let is_sub = matches!(&node.layer, Layer::Sub(_));
    if !is_add && !is_sub {
        return Ok(false);
    }

    let (input_a_name, input_b_name) = node.require_binary_inputs()?;

    // Reject broadcasted Add/Sub — only same-shape residual fan-in
    if !residual_shapes_match(node, node_bounds) {
        debug!("{context_prefix}: patches residual passthrough skipped (shape mismatch)");
        return Ok(false);
    }

    // Route original bias through the network input accumulator
    GraphNetwork::accumulate_bias_to_network_input_crown(
        &pb.lower_b,
        &pb.upper_b,
        node_crown_bounds,
        output_dim,
        input_dim,
        input_accumulated,
    );

    // Build zero-bias carriers for each input
    let left_carrier = CrownBounds::Patches(Box::new(pb.clone_with_zero_bias()));
    let right_carrier = if is_add {
        CrownBounds::Patches(Box::new(pb.clone_with_zero_bias()))
    } else {
        CrownBounds::Patches(Box::new(pb.negated_swapped_zero_bias()))
    };

    graph.accumulate_crown_bounds_to_input(
        input_a_name,
        left_carrier,
        node_crown_bounds,
        output_dim,
        input_dim,
        input_accumulated,
    )?;
    graph.accumulate_crown_bounds_to_input(
        input_b_name,
        right_carrier,
        node_crown_bounds,
        output_dim,
        input_dim,
        input_accumulated,
    )?;

    Ok(true)
}

/// Check that both inputs and the node output have the same spatial shape.
fn residual_shapes_match(node: &GraphNode, node_bounds: &HashMap<String, BoundedTensor>) -> bool {
    if node.inputs.len() != 2 {
        return false;
    }
    let a_shape = node_bounds.get(&node.inputs[0]).map(|b| b.shape().to_vec());
    let b_shape = node_bounds.get(&node.inputs[1]).map(|b| b.shape().to_vec());
    match (a_shape, b_shape) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

impl GraphNetwork {
    /// Convenience wrapper: accumulate Dense LinearBounds into a CrownBounds map.
    ///
    /// Wraps the LinearBounds as CrownBounds::Dense and delegates to
    /// [`accumulate_crown_bounds_to_input`]. Used by the Dense dispatch paths
    /// (ReLU, MulBinary, Where, shared dispatch) in the graph engine.
    pub(crate) fn accumulate_dense_bounds_to_input(
        &self,
        input_name: &str,
        new_bounds: LinearBounds,
        node_crown_bounds: &mut CrownMergeAccumulator,
        output_dim: usize,
        input_dim: usize,
        input_accumulated: &mut bool,
    ) -> Result<()> {
        self.accumulate_crown_bounds_to_input(
            input_name,
            CrownBounds::Dense(new_bounds),
            node_crown_bounds,
            output_dim,
            input_dim,
            input_accumulated,
        )
    }

    /// Runtime invariant check for #2617/#2530 split-path bounds (#2656).
    ///
    /// `BackwardDispatchResult::Binary`/`Nary` must carry bias only in the
    /// separate channel; each per-input `LinearBounds` path must have zero bias.
    /// Violation would silently double-count bias, producing unsound (too-tight) bounds.
    ///
    /// Originally `debug_assert!`, upgraded to runtime check (#2656): the cost of
    /// iterating the bias vector is negligible vs. the CROWN backward matrix
    /// multiplications, and this invariant guards against a class of severe
    /// soundness bugs (#2520, #2527, #2529, #2530).
    ///
    /// Converted from `assert!` to `Result` (#2907) to eliminate a production
    /// panic cliff — callers can now propagate the error cleanly instead of
    /// crashing the entire verification run.
    #[inline]
    pub(crate) fn verify_split_path_bias_zero(bounds: &LinearBounds, context: &str) -> Result<()> {
        // Tolerance for floating-point rounding artifacts (#2700).
        // Split-path bias must be exactly zero by construction, but accumulated
        // float ops can produce negligible artifacts (e.g., 1e-38).
        const TOLERANCE: f32 = 1e-30;

        for (label, bias) in [("lower_b", bounds.lower_b()), ("upper_b", bounds.upper_b())] {
            // Check NaN explicitly first — IEEE 754 NaN == 0.0 returns false,
            // which would produce the misleading "non-zero" message (#2700).
            if bias.iter().any(|v| v.is_nan()) {
                return Err(NyError::InvalidSpec(format!(
                    "{context} produced NaN in {label} split-path bounds \
                     (NaN corruption in dispatch layer)"
                )));
            }
            let max_abs = bias.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            if max_abs >= TOLERANCE {
                return Err(NyError::InvalidSpec(format!(
                    "{context} produced non-zero {label} in split-path bounds \
                     (max |v| = {max_abs:.2e})"
                )));
            }
        }
        Ok(())
    }

    /// CrownBounds-aware bias accumulation for DAG-CROWN (Phase 1b, #2613).
    ///
    /// Same logic as [`accumulate_bias_to_network_input`] but operates on a
    /// `CrownBounds` map. When inserting a new NETWORK_INPUT entry, wraps as
    /// `CrownBounds::Dense`. When updating existing, ensures Dense first.
    pub(crate) fn accumulate_bias_to_network_input_crown(
        bias_lower: &Array1<f32>,
        bias_upper: &Array1<f32>,
        node_crown_bounds: &mut CrownMergeAccumulator,
        output_dim: usize,
        input_dim: usize,
        input_accumulated: &mut bool,
    ) {
        // Migrated from from_parts_unchecked: bias vectors transit from backward
        // pass and could carry NaN. Coefficients are hardcoded zeros (always safe).
        // NaN firewall falls back to conservative bounds. See #3438.
        if *input_accumulated {
            let dense_bias = LinearBounds::new_or_conservative(
                Array2::zeros((output_dim, input_dim)),
                bias_lower.clone(),
                Array2::zeros((output_dim, input_dim)),
                bias_upper.clone(),
            )
            .expect("invariant: hardcoded zero-matrix shapes always match bias dimensions");
            if let Err(error) = node_crown_bounds.merge_dense(NETWORK_INPUT, dense_bias) {
                tracing::warn!(
                    "accumulate_bias_to_network_input_crown: merge_dense failed: {error}"
                );
            }
        } else {
            let lb = LinearBounds::new_or_conservative(
                Array2::zeros((output_dim, input_dim)),
                bias_lower.clone(),
                Array2::zeros((output_dim, input_dim)),
                bias_upper.clone(),
            )
            .expect("invariant: hardcoded zero-matrix shapes always match bias dimensions");
            node_crown_bounds.insert(NETWORK_INPUT.to_string(), CrownBounds::Dense(lb));
            *input_accumulated = true;
        }
    }
}
