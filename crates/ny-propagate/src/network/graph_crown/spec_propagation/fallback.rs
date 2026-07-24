// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP fallback, truncation finalization, and output assembly for spec-guided
//! CROWN backward propagation.
//!
//! Every "stop here and return sound bounds" path lives in this module instead
//! of mixing with the main coordinator loop. Split from `core.rs` as part of
//! #3960.

use crate::batched_domain::CachedLinearBounds;
use crate::bounds::LinearBounds;
use crate::network::core::{GraphNetwork, NETWORK_INPUT};
use crate::network::tighten_crown_output;
use crate::network::CrownMergeAccumulator;
use crate::types::{BoundsProvenance, CrownBackwardResult, CrownIbpFallbackReason};

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

/// Common payload type for spec-guided CROWN backward functions.
pub(super) type SpecCrownPayload = (
    CrownBackwardResult,
    Option<LinearBounds>,
    Option<CachedLinearBounds>,
);

/// Common return type for spec-guided CROWN backward functions.
pub(super) type SpecCrownResult = Result<SpecCrownPayload>;

/// Fast-path return for empty graphs.
///
/// When the graph has no nodes, spec-guided CROWN reduces to applying the
/// spec matrix directly to the input bounds.
pub(super) fn empty_graph_fast_path(
    graph: &GraphNetwork,
    spec_matrix: &ndarray::Array2<f32>,
    input: &BoundedTensor,
) -> Result<Option<SpecCrownPayload>> {
    if !graph.nodes.is_empty() {
        return Ok(None);
    }
    let linear_bounds = LinearBounds::from_spec_matrix(spec_matrix.clone())?;
    let crown_output = linear_bounds.concretize_checked(input)?;
    Ok(Some((
        CrownBackwardResult {
            bounds: crown_output.reshape(&[spec_matrix.nrows()])?,
            provenance: BoundsProvenance::Crown,
        },
        Some(linear_bounds),
        None,
    )))
}

/// Truncation early return: finalize when `crown_backward_layers` limit is
/// reached (#3218).
///
/// Concretizes the current CROWN frontier to the network input and intersects
/// with IBP bounds for tightening.
// Justification: truncation finalization threads graph state, accumulated CROWN
// bounds, and IBP reference bounds through the same tightening path as the
// non-truncated finalization.
#[allow(clippy::too_many_arguments)]
pub(super) fn truncation_early_return(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &ndarray::Array2<f32>,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    output_node_name: &str,
    node_crown_bounds: &mut CrownMergeAccumulator,
    num_specs: usize,
    input_dim: usize,
    input_accumulated: &mut bool,
) -> SpecCrownResult {
    let final_linear_bounds = graph.concretize_crown_frontier_to_network_input(
        node_crown_bounds,
        node_bounds,
        num_specs,
        input_dim,
        input_accumulated,
    )?;
    let crown_output = final_linear_bounds
        .concretize_checked(input)?
        .reshape(&[num_specs])?;
    let ibp_spec_bounds = graph.propagate_crown_with_specs_fallback_ibp(
        input,
        spec_matrix,
        node_bounds,
        output_node_name,
    )?;
    let tightened = tighten_crown_output(crown_output, &ibp_spec_bounds, "Spec-guided CROWN")?;
    Ok((
        CrownBackwardResult {
            bounds: tightened,
            provenance: BoundsProvenance::Crown,
        },
        Some(final_linear_bounds),
        None,
    ))
}

/// Finalize the backward output after the main loop completes.
///
/// Extracts the final `NETWORK_INPUT` bounds, concretizes, checks for non-finite
/// results (falling back to IBP if degraded), intersects with IBP bounds for
/// tightening, and packages the cached linear bounds.
// Justification: finalize threads the accumulated loop state through IBP
// tightening, non-finite guard, and cache packaging — each of which needs
// the full graph/input/spec/node_bounds context.
#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_backward_output(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &ndarray::Array2<f32>,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    output_node_name: &str,
    mut node_crown_bounds: CrownMergeAccumulator,
    mut captured_linear_bounds: Option<std::collections::HashMap<String, LinearBounds>>,
    cache_capture_valid: bool,
    num_specs: usize,
) -> SpecCrownResult {
    let final_bounds = node_crown_bounds
        .take(NETWORK_INPUT)?
        .ok_or_else(|| NyError::InvalidSpec("No path to network input found".to_string()))?;
    let final_linear_bounds = final_bounds.into_dense()?;
    if let Some(ref mut linear_bounds_map) = captured_linear_bounds {
        linear_bounds_map.insert(NETWORK_INPUT.to_string(), final_linear_bounds.clone());
    }

    let crown_output = final_linear_bounds.concretize_checked(input)?;

    // Tightening heuristic: concretize_sound (via concretize_checked) repairs NaN/inversions
    // to [-inf, +inf] which is sound but maximally loose. When degraded, IBP typically
    // produces tighter results. Every other CROWN path checks this and falls back;
    // spec-guided CROWN was missing this guard. See sequential CROWN crown.rs:270-278.
    let reshaped = crown_output.reshape(&[num_specs])?;
    if reshaped
        .lower()
        .iter()
        .chain(reshaped.upper().iter())
        .any(|&v| !v.is_finite())
    {
        debug!("Spec-guided CROWN: falling back to IBP — CROWN output contains non-finite bounds");
        return fallback_to_ibp_with_reason(
            graph,
            input,
            spec_matrix,
            node_bounds,
            output_node_name,
            CrownIbpFallbackReason::CrownPropagationError,
        );
    }

    // Intersect with IBP-applied-spec forward bounds (#3037, same class as #2990).
    // CROWN backward can be strictly looser than IBP for certain weight/input
    // configurations (e.g., negative weight amplifying ReLU lower relaxation error).
    // Reference: alpha-beta-CROWN bound_general.py:1452-1453 does
    // torch.max(crown_lower, ibp_lower), torch.min(crown_upper, ibp_upper).
    // Shared tighten_crown_output handles NaN-in-forward-bounds and shape mismatch (#3043).
    let ibp_spec_bounds = graph.propagate_crown_with_specs_fallback_ibp(
        input,
        spec_matrix,
        node_bounds,
        output_node_name,
    )?;
    let tightened = tighten_crown_output(reshaped, &ibp_spec_bounds, "Spec-guided CROWN")?;
    let cached_linear_bounds = if cache_capture_valid {
        captured_linear_bounds.and_then(|map| {
            if map.is_empty() {
                None
            } else {
                Some(CachedLinearBounds::from_linear_bounds_map(map))
            }
        })
    } else {
        None
    };

    Ok((
        CrownBackwardResult {
            bounds: tightened,
            provenance: BoundsProvenance::Crown,
        },
        Some(final_linear_bounds),
        cached_linear_bounds,
    ))
}

/// IBP fallback with structured reason for provenance tracking (#3520 Packet C).
///
/// Replaces the old `fallback_to_ibp` by attaching a `CrownIbpFallbackReason`
/// to the returned `CrownBackwardResult` so callers can distinguish deadline
/// expiration from shape mismatches from generic propagation errors.
pub(super) fn fallback_to_ibp_with_reason(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &ndarray::Array2<f32>,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    output_node_name: &str,
    reason: CrownIbpFallbackReason,
) -> SpecCrownResult {
    let bounds = graph.propagate_crown_with_specs_fallback_ibp(
        input,
        spec_matrix,
        node_bounds,
        output_node_name,
    )?;
    Ok((
        CrownBackwardResult {
            bounds,
            provenance: BoundsProvenance::ForwardFallback(reason),
        },
        None,
        None,
    ))
}
