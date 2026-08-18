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
use crate::bounds::patches::{
    CrownBounds, PatchesMaterializationDeadline, PatchesMaterializationPurpose,
};
use crate::bounds::LinearBounds;
use crate::network::core::{GraphNetwork, NETWORK_INPUT};
use crate::network::tighten_crown_output_with_deadline;
use crate::network::CrownMergeAccumulator;
use crate::types::{BoundsProvenance, CrownBackwardResult, CrownIbpFallbackReason};

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;
use tracing::debug;

const SPEC_SEED_SITE: &str = "spec-guided CROWN finite specification seed";
const SPEC_CACHE_SITE: &str = "spec-guided CROWN finite cache publication";

#[inline]
fn check_spec_deadline(deadline: Option<Instant>, phase: &'static str) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(NyError::DeadlineExceeded(format!(
            "spec-guided CROWN: deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

fn has_non_finite_with_deadline(bounds: &BoundedTensor, deadline: Option<Instant>) -> Result<bool> {
    if deadline.is_none() {
        return Ok(bounds
            .lower()
            .iter()
            .chain(bounds.upper().iter())
            .any(|&value| !value.is_finite()));
    }
    for endpoints in [bounds.lower(), bounds.upper()] {
        for (index, &value) in endpoints.iter().enumerate() {
            if index.is_multiple_of(4_096) {
                check_spec_deadline(deadline, "while scanning final endpoints")?;
            }
            if !value.is_finite() {
                return Ok(true);
            }
        }
    }
    check_spec_deadline(deadline, "after scanning final endpoints")?;
    Ok(false)
}

fn captured_linear_map_memory_bytes(
    map: &HashMap<String, LinearBounds>,
    deadline: Option<Instant>,
    phase: &'static str,
) -> Result<usize> {
    check_spec_deadline(deadline, phase)?;
    let entry_bytes = size_of::<(String, LinearBounds)>().saturating_add(size_of::<usize>());
    let mut bytes = map.capacity().saturating_mul(entry_bytes);
    for (index, (name, bounds)) in map.iter().enumerate() {
        bytes = bytes
            .saturating_add(name.capacity())
            .saturating_add(bounds.memory_bytes());
        if index.is_multiple_of(4_096) {
            check_spec_deadline(deadline, phase)?;
        }
    }
    check_spec_deadline(deadline, phase)?;
    Ok(bytes)
}

fn try_clone_cache_key(
    source: &str,
    deadline: Option<Instant>,
    required_bytes: usize,
    budget_bytes: usize,
) -> Result<String> {
    check_spec_deadline(deadline, "before cache-key allocation")?;
    let mut key = String::new();
    key.try_reserve_exact(source.len())
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: SPEC_CACHE_SITE,
        })?;
    key.push_str(source);
    check_spec_deadline(deadline, "after cache-key copy")?;
    Ok(key)
}

fn capture_final_linear_bounds(
    map: &mut HashMap<String, LinearBounds>,
    node_name: &str,
    bounds: &LinearBounds,
    input_box: &BoundedTensor,
    deadline: Option<Instant>,
    retained_extra_bytes: usize,
) -> Result<()> {
    if deadline.is_none() {
        map.insert(node_name.to_string(), bounds.clone());
        return Ok(());
    }
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    let mut retained_bytes =
        captured_linear_map_memory_bytes(map, deadline, "while admitting final input capture")?
            .saturating_add(retained_extra_bytes);
    let entry_bytes = size_of::<(String, LinearBounds)>()
        .saturating_add(size_of::<usize>())
        .saturating_add(node_name.len());
    if retained_bytes.saturating_add(entry_bytes) > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: retained_bytes.saturating_add(entry_bytes),
            budget_bytes,
            site: SPEC_CACHE_SITE,
        });
    }
    map.try_reserve(1).map_err(|_| NyError::CpuMemoryExceeded {
        required_bytes: retained_bytes.saturating_add(entry_bytes),
        budget_bytes,
        site: SPEC_CACHE_SITE,
    })?;
    retained_bytes =
        captured_linear_map_memory_bytes(map, deadline, "after final input cache-map reservation")?
            .saturating_add(retained_extra_bytes);
    let captured_name = try_clone_cache_key(
        node_name,
        deadline,
        retained_bytes.saturating_add(entry_bytes),
        budget_bytes,
    )?;
    retained_bytes = retained_bytes.saturating_add(captured_name.capacity());
    let mut captured = bounds.try_clone_with_deadline(deadline, retained_bytes)?;
    if captured.has_coeff_err() {
        captured.fold_coeff_err_over_box_eager_with_deadline(input_box, deadline)?;
        if captured.has_coeff_err() {
            return Err(NyError::UnsupportedConfiguration(format!(
                "{SPEC_CACHE_SITE}: CachedLinearBounds cannot preserve non-finite coefficient-error state at '{node_name}'"
            )));
        }
    }
    check_spec_deadline(deadline, "before final input capture publication")?;
    map.insert(captured_name, captured);
    if let Err(error) = check_spec_deadline(deadline, "after final input capture publication") {
        map.remove(node_name);
        return Err(error);
    }
    Ok(())
}

/// Consume a fully staged node map into the four-map warm-start cache under
/// the same absolute authority. All buckets and duplicate keys are admitted
/// before any entry is published, and the returned object appears only after
/// the final deadline checkpoint.
fn cached_linear_bounds_from_map_with_deadline(
    map: HashMap<String, LinearBounds>,
    deadline: Option<Instant>,
) -> Result<Option<CachedLinearBounds>> {
    if map.is_empty() {
        return Ok(None);
    }
    let Some(_) = deadline else {
        return Ok(Some(CachedLinearBounds::from_linear_bounds_map(map)));
    };

    let source_bytes = captured_linear_map_memory_bytes(
        &map,
        deadline,
        "while admitting finite cache publication",
    )?;
    let entries = map.len();
    let key_bytes = map
        .keys()
        .fold(0usize, |sum, name| sum.saturating_add(name.len()));
    let nominal_bucket_bytes = entries.saturating_mul(
        size_of::<(String, ndarray::Array2<f32>)>()
            .saturating_mul(2)
            .saturating_add(size_of::<(String, ndarray::Array1<f32>)>().saturating_mul(2))
            .saturating_add(size_of::<usize>().saturating_mul(4)),
    );
    let nominal_required_bytes = source_bytes
        .saturating_add(nominal_bucket_bytes)
        .saturating_add(key_bytes.saturating_mul(3));
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    if nominal_required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: nominal_required_bytes,
            budget_bytes,
            site: SPEC_CACHE_SITE,
        });
    }

    let mut lower_a = HashMap::new();
    let mut upper_a = HashMap::new();
    let mut lower_b = HashMap::new();
    let mut upper_b = HashMap::new();
    for target in [
        &mut lower_a as &mut HashMap<String, ndarray::Array2<f32>>,
        &mut upper_a,
    ] {
        target
            .try_reserve(entries)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: nominal_required_bytes,
                budget_bytes,
                site: SPEC_CACHE_SITE,
            })?;
    }
    for target in [
        &mut lower_b as &mut HashMap<String, ndarray::Array1<f32>>,
        &mut upper_b,
    ] {
        target
            .try_reserve(entries)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: nominal_required_bytes,
                budget_bytes,
                site: SPEC_CACHE_SITE,
            })?;
    }
    check_spec_deadline(deadline, "after finite cache-map reservations")?;

    let actual_bucket_bytes = lower_a
        .capacity()
        .saturating_add(upper_a.capacity())
        .saturating_mul(
            size_of::<(String, ndarray::Array2<f32>)>().saturating_add(size_of::<usize>()),
        )
        .saturating_add(
            lower_b
                .capacity()
                .saturating_add(upper_b.capacity())
                .saturating_mul(
                    size_of::<(String, ndarray::Array1<f32>)>().saturating_add(size_of::<usize>()),
                ),
        );
    let actual_required_bytes = source_bytes
        .saturating_add(actual_bucket_bytes)
        .saturating_add(key_bytes.saturating_mul(3));
    if actual_required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: actual_required_bytes,
            budget_bytes,
            site: SPEC_CACHE_SITE,
        });
    }

    for (index, (node_name, bounds)) in map.into_iter().enumerate() {
        check_spec_deadline(deadline, "before finite cache entry conversion")?;
        if bounds.has_coeff_err() {
            return Err(NyError::UnsupportedConfiguration(format!(
                "{SPEC_CACHE_SITE}: CachedLinearBounds cannot preserve coefficient-error state at '{node_name}'"
            )));
        }
        let lower_a_name =
            try_clone_cache_key(&node_name, deadline, actual_required_bytes, budget_bytes)?;
        let upper_a_name =
            try_clone_cache_key(&node_name, deadline, actual_required_bytes, budget_bytes)?;
        let lower_b_name =
            try_clone_cache_key(&node_name, deadline, actual_required_bytes, budget_bytes)?;
        let (la, lbias, ua, ubias) = bounds.into_parts();
        lower_a.insert(lower_a_name, la);
        upper_a.insert(upper_a_name, ua);
        lower_b.insert(lower_b_name, lbias);
        upper_b.insert(node_name, ubias);
        if index.is_multiple_of(4_096) {
            check_spec_deadline(deadline, "during finite cache entry publication")?;
        }
    }
    check_spec_deadline(deadline, "before finite cache publication")?;
    Ok(Some(CachedLinearBounds {
        lower_a,
        upper_a,
        lower_b,
        upper_b,
    }))
}

/// Construct the Dense C-matrix seed without hiding an unbounded clone or
/// zero-fill behind a finite request. The source matrix remains resident while
/// both coefficient matrices and biases are staged, and every allocation is
/// reconciled against the same request budget before its buffer is touched.
pub(super) fn spec_seed_with_deadline(
    spec_matrix: &ndarray::Array2<f32>,
    deadline: Option<Instant>,
) -> Result<LinearBounds> {
    let Some(deadline) = deadline else {
        return LinearBounds::from_spec_matrix(spec_matrix.clone());
    };
    let mut authority = PatchesMaterializationDeadline::new(Some(deadline));
    authority.checkpoint("before finite spec seed admission")?;

    let coefficients = spec_matrix.len();
    let biases = spec_matrix.nrows();
    let nominal_elements = coefficients
        .saturating_add(coefficients.saturating_mul(2))
        .saturating_add(biases.saturating_mul(2));
    let nominal_required_bytes = nominal_elements.saturating_mul(size_of::<f32>());
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    if nominal_required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: nominal_required_bytes,
            budget_bytes,
            site: SPEC_SEED_SITE,
        });
    }

    let mut capacity_overage_bytes = 0usize;
    let mut reserve = |elements: usize| -> Result<Vec<f32>> {
        authority.checkpoint("before finite spec seed allocation")?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: nominal_required_bytes.saturating_add(capacity_overage_bytes),
                budget_bytes,
                site: SPEC_SEED_SITE,
            })?;
        capacity_overage_bytes = capacity_overage_bytes.saturating_add(
            values
                .capacity()
                .saturating_sub(elements)
                .saturating_mul(size_of::<f32>()),
        );
        let actual_required_bytes = nominal_required_bytes.saturating_add(capacity_overage_bytes);
        if actual_required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: actual_required_bytes,
                budget_bytes,
                site: SPEC_SEED_SITE,
            });
        }
        authority.checkpoint("after finite spec seed allocation")?;
        Ok(values)
    };

    let mut lower_a_values = reserve(coefficients)?;
    let mut upper_a_values = reserve(coefficients)?;
    let mut lower_b_values = reserve(biases)?;
    let mut upper_b_values = reserve(biases)?;
    for &coefficient in spec_matrix {
        if !coefficient.is_finite() {
            return Err(NyError::NumericalInstability(
                "LinearBounds::from_spec_matrix: specification matrix contains NaN or Inf".into(),
            ));
        }
        lower_a_values.push(coefficient);
        upper_a_values.push(coefficient);
        authority.work(3, "while copying finite spec seed coefficients")?;
    }
    for _ in 0..biases {
        lower_b_values.push(0.0);
        upper_b_values.push(0.0);
        authority.work(2, "while filling finite spec seed biases")?;
    }
    authority.checkpoint("before finite spec seed publication")?;

    let shape = spec_matrix.raw_dim();
    let lower_a = ndarray::Array2::from_shape_vec(shape, lower_a_values).map_err(|error| {
        NyError::InternalError(format!(
            "{SPEC_SEED_SITE}: lower coefficient shape: {error}"
        ))
    })?;
    let upper_a = ndarray::Array2::from_shape_vec(spec_matrix.raw_dim(), upper_a_values).map_err(
        |error| {
            NyError::InternalError(format!(
                "{SPEC_SEED_SITE}: upper coefficient shape: {error}"
            ))
        },
    )?;
    let lower_b = ndarray::Array1::from_vec(lower_b_values);
    let upper_b = ndarray::Array1::from_vec(upper_b_values);
    authority.checkpoint("after finite spec seed publication")?;
    LinearBounds::from_prevalidated_parts(lower_a, lower_b, upper_a, upper_b)
}

/// Common payload type for spec-guided CROWN backward functions.
pub(super) type SpecCrownPayload = (
    CrownBackwardResult,
    Option<LinearBounds>,
    Option<CachedLinearBounds>,
);

/// Common return type for spec-guided CROWN backward functions.
pub(super) type SpecCrownResult = Result<SpecCrownPayload>;

/// Transactionally prepare the final spec-guided carrier for concretization.
///
/// A structured CPU-memory or deadline refusal selects the caller's established
/// sound IBP fallback. Every semantic error remains an error.
fn prepare_spec_final_dense(
    bounds: &mut CrownBounds,
    deadline: Option<Instant>,
    retained_base_bytes: usize,
) -> Result<Option<CrownIbpFallbackReason>> {
    let materialized = match bounds {
        CrownBounds::Dense(_) => return Ok(None),
        CrownBounds::Patches(patches) => patches.to_dense_with_deadline_and_resident_for_purpose(
            deadline,
            retained_base_bytes,
            PatchesMaterializationPurpose::NetworkInputTerminal,
        ),
    };
    match materialized {
        Ok(dense) => {
            check_spec_deadline(deadline, "before final Dense carrier publication")?;
            *bounds = CrownBounds::Dense(dense);
            Ok(None)
        }
        Err(NyError::CpuMemoryExceeded { .. }) => {
            Ok(Some(CrownIbpFallbackReason::MemoryBudgetExceeded))
        }
        Err(NyError::DeadlineExceeded(_)) => {
            Ok(Some(CrownIbpFallbackReason::PerNodeDeadlineExceeded))
        }
        Err(error) => Err(error),
    }
}

/// Fast-path return for empty graphs.
///
/// When the graph has no nodes, spec-guided CROWN reduces to applying the
/// spec matrix directly to the input bounds.
pub(super) fn empty_graph_fast_path(
    graph: &GraphNetwork,
    spec_matrix: &ndarray::Array2<f32>,
    input: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<Option<SpecCrownPayload>> {
    if !graph.nodes.is_empty() {
        return Ok(None);
    }
    let linear_bounds = match spec_seed_with_deadline(spec_matrix, deadline) {
        Ok(bounds) => bounds,
        Err(NyError::CpuMemoryExceeded { .. }) => {
            let bounds = GraphNetwork::apply_spec_matrix_to_bounds_fallback_with_deadline(
                spec_matrix,
                input,
                deadline,
            )?;
            return Ok(Some((
                CrownBackwardResult {
                    bounds,
                    provenance: BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::MemoryBudgetExceeded,
                    ),
                },
                None,
                None,
            )));
        }
        Err(error) => return Err(error),
    };
    let (crown_output, provenance, output_linear) =
        match linear_bounds.concretize_sound_with_deadline(input, deadline) {
            Ok(bounds) => (bounds, BoundsProvenance::Crown, Some(linear_bounds)),
            Err(NyError::CpuMemoryExceeded { .. }) => (
                GraphNetwork::apply_spec_matrix_to_bounds_fallback_with_deadline(
                    spec_matrix,
                    input,
                    deadline,
                )?,
                BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::MemoryBudgetExceeded),
                None,
            ),
            Err(NyError::DeadlineExceeded(_)) => (
                GraphNetwork::apply_spec_matrix_to_bounds_fallback_with_deadline(
                    spec_matrix,
                    input,
                    deadline,
                )?,
                BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::PerNodeDeadlineExceeded),
                None,
            ),
            Err(error) => return Err(error),
        };
    debug_assert_eq!(crown_output.shape(), [spec_matrix.nrows()]);
    Ok(Some((
        CrownBackwardResult {
            bounds: crown_output,
            provenance,
        },
        output_linear,
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
    node_bounds: &HashMap<String, BoundedTensor>,
    output_node_name: &str,
    node_crown_bounds: &mut CrownMergeAccumulator,
    num_specs: usize,
    input_dim: usize,
    input_accumulated: &mut bool,
    deadline: Option<Instant>,
) -> SpecCrownResult {
    let final_linear_bounds = match graph.concretize_crown_frontier_to_network_input_with_deadline(
        node_crown_bounds,
        node_bounds,
        num_specs,
        input_dim,
        input_accumulated,
        deadline,
    ) {
        Ok(bounds) => bounds,
        Err(NyError::CpuMemoryExceeded { .. }) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::MemoryBudgetExceeded,
                deadline,
            );
        }
        Err(NyError::DeadlineExceeded(_)) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                deadline,
            );
        }
        Err(error) => return Err(error),
    };
    let crown_output = match final_linear_bounds.concretize_sound_with_deadline(input, deadline) {
        Ok(bounds) => bounds,
        Err(NyError::CpuMemoryExceeded { .. }) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::MemoryBudgetExceeded,
                deadline,
            );
        }
        Err(NyError::DeadlineExceeded(_)) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                deadline,
            );
        }
        Err(error) => return Err(error),
    };
    debug_assert_eq!(crown_output.shape(), [num_specs]);
    let ibp_spec_bounds = graph.propagate_crown_with_specs_fallback_ibp_with_deadline(
        input,
        spec_matrix,
        node_bounds,
        output_node_name,
        deadline,
    )?;
    let tightened = match tighten_crown_output_with_deadline(
        crown_output,
        &ibp_spec_bounds,
        "Spec-guided CROWN",
        deadline,
    ) {
        Ok(bounds) => bounds,
        Err(NyError::CpuMemoryExceeded { .. }) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::MemoryBudgetExceeded,
                deadline,
            );
        }
        Err(NyError::DeadlineExceeded(_)) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                deadline,
            );
        }
        Err(error) => return Err(error),
    };
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return fallback_to_ibp_with_reason(
            graph,
            input,
            spec_matrix,
            node_bounds,
            output_node_name,
            CrownIbpFallbackReason::PerNodeDeadlineExceeded,
            deadline,
        );
    }
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
    node_bounds: &HashMap<String, BoundedTensor>,
    output_node_name: &str,
    mut node_crown_bounds: CrownMergeAccumulator,
    mut captured_linear_bounds: Option<HashMap<String, LinearBounds>>,
    mut cache_capture_valid: bool,
    num_specs: usize,
    deadline: Option<Instant>,
) -> SpecCrownResult {
    let retained_capture_bytes = if deadline.is_some() {
        match captured_linear_bounds.as_ref() {
            Some(map) => captured_linear_map_memory_bytes(
                map,
                deadline,
                "while scanning retained captures before finalization",
            )?,
            None => 0,
        }
    } else {
        0
    };
    let mut final_bounds = match node_crown_bounds.take_with_deadline_and_resident(
        NETWORK_INPUT,
        deadline,
        retained_capture_bytes,
    ) {
        Ok(Some(bounds)) => bounds,
        Ok(None) => {
            return Err(NyError::InvalidSpec(
                "No path to network input found".to_string(),
            ));
        }
        Err(NyError::CpuMemoryExceeded { .. }) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::MemoryBudgetExceeded,
                deadline,
            );
        }
        Err(NyError::DeadlineExceeded(_)) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                deadline,
            );
        }
        Err(error) => return Err(error),
    };
    if let Some(reason) =
        prepare_spec_final_dense(&mut final_bounds, deadline, retained_capture_bytes)?
    {
        debug!(reason = ?reason, "Spec-guided CROWN: final carrier materialization hit its resource authority; falling back to IBP");
        return fallback_to_ibp_with_reason(
            graph,
            input,
            spec_matrix,
            node_bounds,
            output_node_name,
            reason,
            deadline,
        );
    }
    let CrownBounds::Dense(final_linear_bounds) = final_bounds else {
        unreachable!("successful spec-final preparation must publish Dense")
    };
    let crown_output = match final_linear_bounds.concretize_sound_with_deadline_and_resident(
        input,
        deadline,
        retained_capture_bytes,
    ) {
        Ok(bounds) => bounds,
        Err(NyError::CpuMemoryExceeded { .. }) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::MemoryBudgetExceeded,
                deadline,
            );
        }
        Err(NyError::DeadlineExceeded(_)) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                deadline,
            );
        }
        Err(error) => return Err(error),
    };

    // Tightening heuristic: concretize_sound (via concretize_checked) repairs NaN/inversions
    // to [-inf, +inf] which is sound but maximally loose. When degraded, IBP typically
    // produces tighter results. Every other CROWN path checks this and falls back;
    // spec-guided CROWN was missing this guard. See sequential CROWN crown.rs:270-278.
    debug_assert_eq!(crown_output.shape(), [num_specs]);
    if has_non_finite_with_deadline(&crown_output, deadline)? {
        debug!("Spec-guided CROWN: falling back to IBP — CROWN output contains non-finite bounds");
        return fallback_to_ibp_with_reason(
            graph,
            input,
            spec_matrix,
            node_bounds,
            output_node_name,
            CrownIbpFallbackReason::CrownPropagationError,
            deadline,
        );
    }

    // Intersect with IBP-applied-spec forward bounds (#3037, same class as #2990).
    // CROWN backward can be strictly looser than IBP for certain weight/input
    // configurations (e.g., negative weight amplifying ReLU lower relaxation error).
    // Reference: alpha-beta-CROWN bound_general.py:1452-1453 does
    // torch.max(crown_lower, ibp_lower), torch.min(crown_upper, ibp_upper).
    // Shared tighten_crown_output handles NaN-in-forward-bounds and shape mismatch (#3043).
    let ibp_spec_bounds = graph.propagate_crown_with_specs_fallback_ibp_with_deadline(
        input,
        spec_matrix,
        node_bounds,
        output_node_name,
        deadline,
    )?;
    let tightened = match tighten_crown_output_with_deadline(
        crown_output,
        &ibp_spec_bounds,
        "Spec-guided CROWN",
        deadline,
    ) {
        Ok(bounds) => bounds,
        Err(NyError::CpuMemoryExceeded { .. }) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::MemoryBudgetExceeded,
                deadline,
            );
        }
        Err(NyError::DeadlineExceeded(_)) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                deadline,
            );
        }
        Err(error) => return Err(error),
    };
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return fallback_to_ibp_with_reason(
            graph,
            input,
            spec_matrix,
            node_bounds,
            output_node_name,
            CrownIbpFallbackReason::PerNodeDeadlineExceeded,
            deadline,
        );
    }
    if let Some(ref mut linear_bounds_map) = captured_linear_bounds {
        let retained_concrete_bytes = tightened
            .len()
            .saturating_add(ibp_spec_bounds.len())
            .saturating_mul(2)
            .saturating_mul(size_of::<f32>());
        match capture_final_linear_bounds(
            linear_bounds_map,
            NETWORK_INPUT,
            &final_linear_bounds,
            input,
            deadline,
            retained_concrete_bytes,
        ) {
            Ok(()) => {}
            Err(NyError::CpuMemoryExceeded { .. } | NyError::UnsupportedConfiguration(_)) => {
                // Cache capture is optional proof-performance state. A typed
                // representability or memory refusal discards it atomically;
                // the already-certified bounds remain authoritative.
                captured_linear_bounds = None;
                cache_capture_valid = false;
            }
            Err(NyError::DeadlineExceeded(_)) => {
                return fallback_to_ibp_with_reason(
                    graph,
                    input,
                    spec_matrix,
                    node_bounds,
                    output_node_name,
                    CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                    deadline,
                );
            }
            Err(error) => return Err(error),
        }
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return fallback_to_ibp_with_reason(
            graph,
            input,
            spec_matrix,
            node_bounds,
            output_node_name,
            CrownIbpFallbackReason::PerNodeDeadlineExceeded,
            deadline,
        );
    }
    let cached_linear_bounds = if cache_capture_valid {
        match captured_linear_bounds {
            Some(map) => match cached_linear_bounds_from_map_with_deadline(map, deadline) {
                Ok(cache) => cache,
                Err(NyError::CpuMemoryExceeded { .. } | NyError::UnsupportedConfiguration(_)) => {
                    None
                }
                Err(NyError::DeadlineExceeded(_)) => {
                    return fallback_to_ibp_with_reason(
                        graph,
                        input,
                        spec_matrix,
                        node_bounds,
                        output_node_name,
                        CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                        deadline,
                    );
                }
                Err(error) => return Err(error),
            },
            None => None,
        }
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
    node_bounds: &HashMap<String, BoundedTensor>,
    output_node_name: &str,
    reason: CrownIbpFallbackReason,
    deadline: Option<Instant>,
) -> SpecCrownResult {
    let bounds = graph.propagate_crown_with_specs_fallback_ibp_with_deadline(
        input,
        spec_matrix,
        node_bounds,
        output_node_name,
        deadline,
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

#[cfg(test)]
mod finite_cache_tests {
    use super::*;
    use ndarray::arr2;
    use std::time::Duration;

    fn exact_capture_map() -> HashMap<String, LinearBounds> {
        let mut map = HashMap::new();
        map.insert(
            "relu".to_string(),
            LinearBounds::from_spec_matrix(arr2(&[[1.25_f32, -0.5], [0.0, 2.0]])).unwrap(),
        );
        map
    }

    #[test]
    fn finite_cache_publication_preserves_all_four_exact_maps() {
        let cache = cached_linear_bounds_from_map_with_deadline(
            exact_capture_map(),
            Some(Instant::now() + Duration::from_secs(1)),
        )
        .unwrap()
        .expect("a live finite authority must publish its complete cache");

        assert_eq!(cache.lower_a["relu"], arr2(&[[1.25, -0.5], [0.0, 2.0]]));
        assert_eq!(cache.upper_a["relu"], cache.lower_a["relu"]);
        assert_eq!(cache.lower_b["relu"], ndarray::arr1(&[0.0, 0.0]));
        assert_eq!(cache.upper_b["relu"], cache.lower_b["relu"]);
    }

    #[test]
    fn expired_cache_publication_returns_no_partial_object() {
        let error = cached_linear_bounds_from_map_with_deadline(
            exact_capture_map(),
            Some(
                Instant::now()
                    .checked_sub(Duration::from_nanos(1))
                    .expect("one nanosecond fits before now"),
            ),
        )
        .expect_err("an expired authority must refuse before cache publication");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
    }
}
