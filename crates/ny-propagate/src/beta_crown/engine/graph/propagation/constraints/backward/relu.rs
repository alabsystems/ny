// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ReLU operator override and helpers for constrained backward CROWN.
//!
//! Contains:
//! - `capture_constrained_relu_intermediate` — stores pre-ReLU A-matrices for gradient computation
//! - `apply_constrained_relu_beta_contribution` — injects β-CROWN split contributions
//! - `has_constrained_relu_beta_entry` — O(1) β-state existence check
//! - `try_process_constrained_relu_patches` — Patches-mode ReLU backward
//! - `process_relu_override` — top-level ReLU dispatch for constrained backward
//!
//! Part of #4293 (directory-module split from former backward.rs monolith).

use std::mem::{replace, size_of};

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, trace};

use crate::beta_crown::state::GraphBetaState;
use crate::bounds::patches::{CrownBounds, PatchesMaterializationPurpose};
use crate::bounds::GraphAlphaCrownIntermediate;
use crate::network::{crown_backward_step_patches, CrownStepResult};
use crate::{Layer, LinearBounds};

use super::super::super::super::super::BetaCrownVerifier;
use super::super::lookups::ConstraintLookups;
use super::super::patches::patches_dense_fallback_details;
use super::dispatch::ConstrainedNodeContext;
use super::{BackwardParams, ConstrainedBackwardSetup};

/// Default-dark owned alpha-ReLU transform. Exact `1` is the only enabling
/// spelling; absent and every non-exact value retain the historical
/// gradient-producing Patches path byte-for-byte.
fn constrained_patches_alpha_relu_in_place_enabled() -> bool {
    matches!(
        std::env::var("NY_CONSTRAINED_PATCHES_ALPHA_RELU_INPLACE").as_deref(),
        Ok("1")
    )
}

/// Default-dark selected-column Patches capture for beta-only analytical
/// gradients. Exact `1` is the only enabling spelling.
fn constrained_patches_beta_sparse_capture_enabled() -> bool {
    matches!(
        std::env::var("NY_CONSTRAINED_PATCHES_BETA_SPARSE_CAPTURE").as_deref(),
        Ok("1")
    )
}

fn flatten_constrained_preactivation_with_deadline(
    pre_activation: &BoundedTensor,
    retained_bytes: usize,
    deadline: Option<std::time::Instant>,
) -> Result<(Array1<f32>, Array1<f32>)> {
    let Some(_) = deadline else {
        let flat = pre_activation.flatten();
        let lower = flat
            .lower()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|error| {
                NyError::InternalError(format!(
                    "pre-ReLU lower not convertible to Ix1 (shape {:?}): {error}",
                    flat.lower().shape()
                ))
            })?;
        let upper = flat
            .upper()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|error| {
                NyError::InternalError(format!(
                    "pre-ReLU upper not convertible to Ix1 (shape {:?}): {error}",
                    flat.upper().shape()
                ))
            })?;
        return Ok((lower, upper));
    };
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "before constrained pre-ReLU bound capture allocation",
    )?;
    let elements = pre_activation.lower().len();
    if pre_activation.upper().len() != elements {
        return Err(NyError::shape_mismatch(
            pre_activation.lower().shape().to_vec(),
            pre_activation.upper().shape().to_vec(),
        ));
    }
    let endpoint_bytes = elements.saturating_mul(size_of::<f32>());
    let required_bytes = retained_bytes.saturating_add(endpoint_bytes.saturating_mul(4));
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: "constrained pre-ReLU finite capture",
        });
    }
    let allocation_error = |required_bytes| NyError::CpuMemoryExceeded {
        required_bytes,
        budget_bytes,
        site: "constrained pre-ReLU finite capture",
    };
    let mut lower_values = Vec::new();
    lower_values
        .try_reserve_exact(elements)
        .map_err(|_| allocation_error(required_bytes))?;
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "after constrained pre-ReLU lower allocation",
    )?;
    let lower_overage = lower_values
        .capacity()
        .saturating_sub(elements)
        .saturating_mul(size_of::<f32>());
    let mut upper_values = Vec::new();
    upper_values
        .try_reserve_exact(elements)
        .map_err(|_| allocation_error(required_bytes.saturating_add(lower_overage)))?;
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "after constrained pre-ReLU upper allocation",
    )?;
    let upper_overage = upper_values
        .capacity()
        .saturating_sub(elements)
        .saturating_mul(size_of::<f32>());
    let reconciled_bytes = required_bytes
        .saturating_add(lower_overage)
        .saturating_add(upper_overage);
    if reconciled_bytes > budget_bytes {
        return Err(allocation_error(reconciled_bytes));
    }
    for (index, (&lower, &upper)) in pre_activation
        .lower()
        .iter()
        .zip(pre_activation.upper().iter())
        .enumerate()
    {
        lower_values.push(lower);
        upper_values.push(upper);
        if index % 4096 == 4095 {
            super::super::ensure_constrained_propagation_deadline(
                deadline,
                "during constrained pre-ReLU bound capture copy",
            )?;
        }
    }
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "after constrained pre-ReLU bound capture copy",
    )?;
    Ok((
        Array1::from_vec(lower_values),
        Array1::from_vec(upper_values),
    ))
}

fn constrained_relu_intermediate_capture_bounds(
    node_cb: &CrownBounds,
    deadline: Option<std::time::Instant>,
    retained_bytes: usize,
) -> Result<Array2<f32>> {
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "before constrained ReLU intermediate materialization",
    )?;
    let captured = match node_cb {
        CrownBounds::Dense(lb) => lb.try_clone_lower_a_with_deadline(deadline, retained_bytes),
        // Borrow the structured carrier so its patch/error/anchor allocations
        // are not deep-cloned infallibly before to_dense's memory admission.
        CrownBounds::Patches(lb) => {
            let dense = lb.to_dense_with_deadline_and_resident_for_purpose(
                deadline,
                retained_bytes,
                PatchesMaterializationPurpose::Other,
            )?;
            let (lower_a, _, _, _) = dense.into_parts();
            Ok(lower_a)
        }
    }?;
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "before constrained ReLU intermediate materialization publication",
    )?;
    Ok(captured)
}

/// Resolve the exact constrained ReLU neuron set shared by history lookups and
/// beta state. A mismatch refuses sparse capture so no beta consumer can request
/// a column that was omitted.
fn matching_beta_capture_columns(
    node_name: &str,
    mode_lookups: Option<&ConstraintLookups>,
    beta_state: Option<&GraphBetaState>,
) -> Option<Vec<usize>> {
    let lookup_columns = mode_lookups?.by_relu.get(node_name)?;
    let beta_state = beta_state?;

    let mut from_lookups: Vec<usize> = lookup_columns.keys().copied().collect();
    from_lookups.sort_unstable();
    let mut from_beta: Vec<usize> = beta_state
        .entries_for_node(node_name)
        .map(|entry| entry.neuron_idx())
        .collect();
    from_beta.sort_unstable();
    from_beta.dedup();

    (!from_lookups.is_empty() && from_lookups == from_beta).then_some(from_lookups)
}

#[inline]
fn constrained_patches_alpha_relu_in_place_scope_is_bound_only(
    is_standard: bool,
    intermediate_present: bool,
) -> bool {
    is_standard && !intermediate_present
}

#[inline]
fn is_materialized_contiguous_explicit_row_alpha_candidate(
    bounds: &crate::bounds::patches::PatchesLinearBounds,
) -> bool {
    if bounds.lower_a.identity
        || bounds.upper_a.identity
        || bounds.lower_a.unstable_idx.is_some()
        || bounds.upper_a.unstable_idx.is_some()
    {
        return false;
    }
    let (Some(lower), Some(upper)) = (
        bounds.lower_a.patches.as_ref(),
        bounds.upper_a.patches.as_ref(),
    ) else {
        return false;
    };
    lower.ndim() == 7
        && upper.shape() == lower.shape()
        && lower.as_slice().is_some()
        && upper.as_slice().is_some()
}

pub(super) fn capture_constrained_relu_intermediate(
    current: &ConstrainedNodeContext<'_>,
    pre_activation: &BoundedTensor,
    node_cb: &CrownBounds,
    mode_lookups: Option<&ConstraintLookups>,
    beta_state: Option<&GraphBetaState>,
    joint_alpha_beta: bool,
    deadline: Option<std::time::Instant>,
    intermediate: &mut Option<GraphAlphaCrownIntermediate>,
) -> Result<()> {
    let should_capture_intermediate =
        mode_lookups.is_some_and(|lookups| lookups.by_relu.contains_key(current.node_name));
    if !should_capture_intermediate {
        return Ok(());
    }
    if intermediate.is_none() {
        return Ok(());
    }
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "before constrained ReLU intermediate capture",
    )?;
    let retained_intermediate_bytes = intermediate
        .as_ref()
        .map_or(0, GraphAlphaCrownIntermediate::logical_memory_bytes);
    let retained_pre_activation_bytes = pre_activation
        .len()
        .saturating_mul(2)
        .saturating_mul(size_of::<f32>());

    let mut sparse_capture: Option<(Vec<usize>, Array2<f32>)> = None;
    // Selected-column extraction is an opaque optimization without cooperative
    // polling. Finite requests use the complete deadline-aware materialization
    // path below; semantics are unchanged because the sparse carrier is only a
    // capture optimization.
    if deadline.is_none() && !joint_alpha_beta && constrained_patches_beta_sparse_capture_enabled()
    {
        if let (Some(columns), CrownBounds::Patches(bounds)) = (
            matching_beta_capture_columns(current.node_name, mode_lookups, beta_state),
            node_cb,
        ) {
            if let Some((neuron_indices, values)) = bounds.try_lower_a_beta_columns(&columns) {
                super::super::ensure_constrained_propagation_deadline(
                    deadline,
                    "after constrained ReLU sparse intermediate capture",
                )?;
                if values.ncols() == neuron_indices.len()
                    && neuron_indices.windows(2).all(|pair| pair[0] < pair[1])
                {
                    sparse_capture = Some((neuron_indices, values));
                }
            }
        }
    }

    let dense_capture = if sparse_capture.is_none() {
        Some(constrained_relu_intermediate_capture_bounds(
            node_cb,
            deadline,
            retained_intermediate_bytes.saturating_add(retained_pre_activation_bytes),
        )?)
    } else {
        None
    };

    let staged_sparse_capture_bytes = sparse_capture.as_ref().map_or(0, |(indices, values)| {
        indices
            .len()
            .saturating_mul(size_of::<usize>())
            .saturating_add(values.len().saturating_mul(size_of::<f32>()))
    });
    let staged_dense_capture_bytes = dense_capture
        .as_ref()
        .map_or(0, |values| values.len().saturating_mul(size_of::<f32>()));
    let staged_capture_bytes = retained_intermediate_bytes
        .saturating_add(node_cb.memory_bytes())
        .saturating_add(staged_sparse_capture_bytes)
        .saturating_add(staged_dense_capture_bytes);
    let (lower, upper) = flatten_constrained_preactivation_with_deadline(
        pre_activation,
        staged_capture_bytes,
        deadline,
    )?;
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "before constrained ReLU intermediate capture publication",
    )?;

    let intermediate = intermediate
        .as_mut()
        .expect("intermediate presence checked before transactional capture");
    if let Some((neuron_indices, values)) = sparse_capture {
        let inserted = intermediate.insert_beta_sparse_a(
            current.node_name.to_string(),
            neuron_indices,
            values,
        );
        debug_assert!(inserted, "staged sparse capture was prevalidated");
    } else if let Some(lower_a) = dense_capture {
        intermediate
            .a_at_relu
            .insert(current.node_name.to_string(), lower_a);
    }
    intermediate
        .pre_relu_bounds
        .insert(current.node_name.to_string(), (lower, upper));
    Ok(())
}

pub(super) fn apply_constrained_relu_beta_contribution(
    node_name: &str,
    beta_state: Option<&GraphBetaState>,
    lb: &mut LinearBounds,
    deadline: Option<std::time::Instant>,
) -> Result<()> {
    let Some(beta_state) = beta_state else {
        return Ok(());
    };

    if deadline.is_some() {
        let num_inputs = lb.num_inputs();
        let splits = beta_state.entries_for_node(node_name).filter_map(|entry| {
            let neuron_idx = entry.neuron_idx;
            if neuron_idx >= num_inputs {
                return None;
            }
            let signed_beta = entry.signed_value();
            if !signed_beta.is_finite() {
                tracing::warn!(
                    node_name,
                    neuron_idx,
                    signed_beta,
                    "Skipping non-finite beta contribution in constrained graph backward"
                );
                return None;
            }
            (signed_beta.abs() >= 1e-10).then_some((neuron_idx, signed_beta))
        });
        lb.apply_beta_splits_to_columns_with_deadline(splits, deadline)?;
        super::super::ensure_constrained_propagation_deadline(
            deadline,
            "after constrained ReLU beta contribution",
        )?;
        return Ok(());
    }

    // Part of #2936: use indexed per-node iteration instead of O(B) full scan.
    for entry in beta_state.entries_for_node(node_name) {
        super::super::ensure_constrained_propagation_deadline(
            deadline,
            "during constrained ReLU beta contribution",
        )?;
        let neuron_idx = entry.neuron_idx;
        if neuron_idx >= lb.num_inputs() {
            continue;
        }

        let signed_beta = entry.signed_value();
        if !signed_beta.is_finite() {
            tracing::warn!(
                node_name,
                neuron_idx,
                signed_beta,
                "Skipping non-finite beta contribution in constrained graph backward"
            );
            continue;
        }
        if signed_beta.abs() < 1e-10 {
            continue;
        }

        // #vnncomp-aw-soundness: the β split mutates every coefficient in this
        // neuron's column with a single f32 op (a - β / a + β). When `lb`
        // already carries certified coefficient error, that op's rounding gap
        // must be folded into the err, otherwise the certificate under-counts
        // |stored_f32 - true_coeff| → false-proof risk (mirrors conv becc501).
        lb.apply_beta_split_to_column(neuron_idx, signed_beta);
        trace!(
            "Added β contribution for {}[{}]: signed_beta={}",
            node_name,
            neuron_idx,
            signed_beta
        );
    }
    super::super::ensure_constrained_propagation_deadline(
        deadline,
        "after constrained ReLU beta contribution",
    )?;
    Ok(())
}

pub(super) fn has_constrained_relu_beta_entry(
    node_name: &str,
    beta_state: Option<&GraphBetaState>,
) -> bool {
    // Part of #2936: O(1) via index instead of O(B) linear scan.
    beta_state.is_some_and(|beta_state| beta_state.has_node_entries(node_name))
}

pub(super) fn try_process_constrained_relu_patches(
    params: &BackwardParams<'_>,
    current: &ConstrainedNodeContext<'_>,
    pre_activation: &BoundedTensor,
    node_cb: &mut CrownBounds,
    intermediate: &mut Option<GraphAlphaCrownIntermediate>,
    is_standard: bool,
) -> Result<bool> {
    if !matches!(node_cb, CrownBounds::Patches(_)) {
        return Ok(false);
    }

    let Layer::ReLU(relu) = &current.node.layer else {
        unreachable!("Constrained ReLU patches helper requires a ReLU node");
    };
    super::super::ensure_constrained_propagation_deadline(
        params.deadline,
        "before constrained Patches ReLU",
    )?;

    let mut relu_applied_via_patches = false;
    if let Some(alpha_state) = params.context.alpha_state {
        if !alpha_state.is_empty() {
            if params.deadline.is_some() {
                super::super::ensure_constrained_propagation_deadline(
                    params.deadline,
                    "before constrained finite Patches alpha-ReLU refusal",
                )?;
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Constrained CROWN: cooperative finite Patches alpha-ReLU is unavailable at '{}'",
                    current.node_name
                )));
            }
            let alphas = alpha_state.build_alpha_array(current.node_name, pre_activation);
            super::super::ensure_constrained_propagation_deadline(
                params.deadline,
                "after constrained Patches ReLU alpha preparation",
            )?;

            // The standard constrained pass has no gradient consumer. Under the
            // exact-dark gate, narrowly admit only an owned, materialized,
            // contiguous 7D carrier with a finite absolute deadline. Preparation
            // validates every fallible invariant while `node_cb` is still intact.
            // Once consumed, an error must unwind the constrained pass: the
            // placeholder and any partially transformed carrier are never
            // exposed to the historical Dense/Patches fallback.
            if constrained_patches_alpha_relu_in_place_enabled()
                && constrained_patches_alpha_relu_in_place_scope_is_bound_only(
                    is_standard,
                    intermediate.is_some(),
                )
            {
                if let Some(deadline) = params.deadline {
                    let prepared = if let CrownBounds::Patches(patches_bounds) = node_cb {
                        if is_materialized_contiguous_explicit_row_alpha_candidate(patches_bounds) {
                            match relu.prepare_patches_with_alpha_in_place(
                                patches_bounds,
                                pre_activation,
                                &alphas,
                                deadline,
                            ) {
                                Ok(prepared) => Some(prepared),
                                Err(error) if error.is_deadline_exceeded() => return Err(error),
                                Err(error) => {
                                    debug!(
                                        "Constrained CROWN: owned alpha-ReLU preparation refused \
                                         node {}: {}; retaining historical Patches alpha path",
                                        current.node_name, error
                                    );
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(prepared) = prepared {
                        let owned =
                            match replace(node_cb, CrownBounds::Dense(LinearBounds::identity(0))) {
                                CrownBounds::Patches(bounds) => bounds,
                                CrownBounds::Dense(_) => unreachable!(
                                    "owned alpha-ReLU preparation requires a Patches carrier"
                                ),
                            };
                        let transformed = relu.propagate_prepared_patches_with_alpha_in_place(
                            owned,
                            prepared,
                            pre_activation,
                        )?;
                        super::super::ensure_constrained_propagation_deadline(
                            params.deadline,
                            "after constrained owned Patches alpha-ReLU",
                        )?;
                        *node_cb = transformed;
                        return Ok(true);
                    }
                }
            }

            if let CrownBounds::Patches(patches_bounds) = node_cb {
                let propagated =
                    relu.propagate_patches_with_alpha(patches_bounds, pre_activation, &alphas);
                super::super::ensure_constrained_propagation_deadline(
                    params.deadline,
                    "after constrained Patches alpha-ReLU",
                )?;
                match propagated {
                    Ok((new_cb, grad)) => {
                        if let Some(intermediate) = intermediate.as_mut() {
                            intermediate
                                .alpha_gradients
                                .insert(current.node_name.to_string(), grad);
                            intermediate
                                .alpha_gradients_upper
                                .entry(current.node_name.to_string())
                                .or_insert_with(|| Array1::zeros(alphas.len()));
                        }
                        *node_cb = new_cb;
                        relu_applied_via_patches = true;
                    }
                    Err(error) => {
                        debug!(
                            "Constrained CROWN: Patches alpha-ReLU failed at {}: {}, falling back to Dense alpha",
                            current.node_name, error
                        );
                    }
                }
            }
        }
    }

    if relu_applied_via_patches || !matches!(node_cb, CrownBounds::Patches(_)) {
        return Ok(relu_applied_via_patches);
    }

    match crown_backward_step_patches(
        &current.node.layer,
        node_cb,
        pre_activation,
        params.context.engine,
        0,
        "Constrained CROWN",
        params.deadline,
    ) {
        Ok(CrownStepResult::Continue) => Ok(true),
        Ok(CrownStepResult::IbpFallback(fallback)) => {
            if fallback.reason == crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Constrained CROWN: patches ReLU at '{}' exceeded memory budget: {}",
                    current.node_name, fallback.details
                )));
            }
            debug!(
                "Constrained CROWN: ReLU Patches dispatch requested Dense fallback at {}: {}",
                current.node_name, fallback.details
            );
            if let Some(details) = patches_dense_fallback_details(
                node_cb,
                "constraints::backward_crown_constrained::relu",
            )? {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Constrained CROWN: {details}"
                )));
            }
            node_cb.ensure_dense_with_deadline_for_purpose(
                params.deadline,
                PatchesMaterializationPurpose::Other,
            )?;
            Ok(false)
        }
        Err(error) => {
            let error: NyError = error;
            if error.is_deadline_exceeded() {
                return Err(error);
            }
            debug!(
                "Constrained CROWN: ReLU Patches dispatch failed at {}: {}, falling back to Dense",
                current.node_name, error
            );
            if let Some(details) = patches_dense_fallback_details(
                node_cb,
                "constraints::backward_crown_constrained::relu",
            )? {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Constrained CROWN: {details}"
                )));
            }
            node_cb.ensure_dense_with_deadline_for_purpose(
                params.deadline,
                PatchesMaterializationPurpose::Other,
            )?;
            Ok(false)
        }
    }
}

impl BetaCrownVerifier {
    pub(super) fn process_relu_override(
        &self,
        params: &BackwardParams<'_>,
        current: &ConstrainedNodeContext<'_>,
        pre_activation: &BoundedTensor,
        mut node_cb: CrownBounds,
        setup: &mut ConstrainedBackwardSetup<'_, '_>,
        is_standard: bool,
    ) -> Result<()> {
        capture_constrained_relu_intermediate(
            current,
            pre_activation,
            &node_cb,
            setup.mode_lookups,
            params.beta_state,
            params
                .context
                .alpha_state
                .is_some_and(|alpha_state| !alpha_state.is_empty()),
            params.deadline,
            &mut setup.state.intermediate,
        )?;

        if is_standard && params.deadline.is_none() && tracing::enabled!(tracing::Level::DEBUG) {
            let n_constraints = params.context.history.constraints.len();
            let lower = pre_activation.lower();
            let upper = pre_activation.upper();
            let n_unstable = lower
                .iter()
                .zip(upper.iter())
                .filter(|(&l, &u)| l < 0.0 && u > 0.0)
                .count();
            if n_unstable > 0 && n_constraints >= 12 {
                let unstable_details: Vec<String> = lower
                    .iter()
                    .zip(upper.iter())
                    .enumerate()
                    .filter(|(_, (&l, &u))| l < 0.0 && u > 0.0)
                    .map(|(idx, (&l, &u))| format!("[{}]({:.4},{:.4})", idx, l, u))
                    .collect();
                debug!(
                    "[#1817] ReLU backward {} (constraints={}): {}/{} neurons UNSTABLE: {}",
                    current.node_name,
                    n_constraints,
                    n_unstable,
                    lower.len(),
                    unstable_details.join(", ")
                );
            }
        }

        if let CrownBounds::Dense(node_lb) = &node_cb {
            if is_standard
                && params.deadline.is_none()
                && tracing::enabled!(tracing::Level::DEBUG)
                && params.context.history.constraints.len() >= 12
            {
                let in_gap = (node_lb.upper_a() - node_lb.lower_a()).mapv(f32::abs).sum();
                let in_bgap = (node_lb.upper_b() - node_lb.lower_b()).mapv(f32::abs).sum();
                if in_gap > 1e-6 || in_bgap > 1e-6 {
                    debug!(
                        "[#1817 bwd] {} input A-gap={:.6} b-gap={:.6}",
                        current.node_name, in_gap, in_bgap
                    );
                }
            }
        }

        if try_process_constrained_relu_patches(
            params,
            current,
            pre_activation,
            &mut node_cb,
            &mut setup.state.intermediate,
            is_standard,
        )? {
            // Only convert Patches→Dense when beta entries exist for this node.
            // The original code called ensure_dense() inside the beta loop after
            // filtering by node_name. Unconditional ensure_dense() would change
            // the accumulation code path (Dense vs Patches) when beta_state is
            // None or has no matching entries for the current ReLU.
            if has_constrained_relu_beta_entry(current.node_name, params.beta_state) {
                let new_lb = node_cb.ensure_dense_with_deadline_for_purpose(
                    params.deadline,
                    PatchesMaterializationPurpose::Other,
                )?;
                apply_constrained_relu_beta_contribution(
                    current.node_name,
                    params.beta_state,
                    new_lb,
                    params.deadline,
                )?;
            }
            super::super::ensure_constrained_propagation_deadline(
                params.deadline,
                "before constrained Patches ReLU publication",
            )?;
            params
                .graph
                .accumulate_crown_bounds_to_input_with_deadline(
                    current.first_input,
                    node_cb,
                    &mut setup.state.node_crown_bounds,
                    setup.output_dim,
                    setup.input_dim,
                    &mut setup.state.input_accumulated,
                    params.deadline,
                )?;
            return Ok(());
        }

        // Dense constrained ReLU is not part of the finite Anchored-Patches
        // authority opened by the cGAN route: finite Dense-to-Patches re-entry
        // is deliberately closed before this point. Keep the established
        // constrained-CROWN implementation available for ordinary beta-CROWN
        // callers, bracketing its legacy kernel with the verifier deadline.
        // A fully internally-polled Dense ReLU remains separate roadmap work.
        super::super::ensure_constrained_propagation_deadline(
            params.deadline,
            "before constrained dense ReLU",
        )?;
        let node_lb = node_cb.into_dense_with_deadline_for_purpose(
            params.deadline,
            PatchesMaterializationPurpose::Other,
        )?;
        if is_standard
            && params.deadline.is_none()
            && tracing::enabled!(tracing::Level::DEBUG)
            && params.context.history.constraints.len() >= 12
        {
            let in_gap = (node_lb.upper_a() - node_lb.lower_a()).mapv(f32::abs).sum();
            let in_bgap = (node_lb.upper_b() - node_lb.lower_b()).mapv(f32::abs).sum();
            if in_gap > 1e-6 || in_bgap > 1e-6 {
                debug!(
                    "[#1817 bwd] {} input A-gap={:.6} b-gap={:.6}",
                    current.node_name, in_gap, in_bgap
                );
            }
        }

        let Layer::ReLU(relu) = &current.node.layer else {
            unreachable!("Constrained ReLU helper requires a ReLU node");
        };
        let mut new_lb = if let Some(alpha_state) = params.context.alpha_state {
            if !alpha_state.is_empty() {
                let alphas = alpha_state.build_alpha_array(current.node_name, pre_activation);
                let alphas_upper =
                    alpha_state.build_alpha_upper_array(current.node_name, pre_activation);
                super::super::ensure_constrained_propagation_deadline(
                    params.deadline,
                    "after constrained dense ReLU alpha preparation",
                )?;
                let propagated = relu.propagate_linear_with_alpha(
                    &node_lb,
                    pre_activation,
                    &alphas,
                    Some(&alphas_upper),
                );
                super::super::ensure_constrained_propagation_deadline(
                    params.deadline,
                    "after constrained dense alpha-ReLU",
                )?;
                let (lb_result, grad, grad_upper) = propagated.map_err(|error| {
                    NyError::InvalidSpec(format!(
                        "Constrained CROWN (alpha) failed at node '{}' (ReLU): {}",
                        current.node_name, error
                    ))
                })?;
                if let Some(intermediate) = setup.state.intermediate.as_mut() {
                    intermediate
                        .alpha_gradients
                        .insert(current.node_name.to_string(), grad);
                    intermediate
                        .alpha_gradients_upper
                        .insert(current.node_name.to_string(), grad_upper);
                }
                lb_result
            } else {
                let propagated = relu.propagate_linear_with_bounds(&node_lb, pre_activation);
                super::super::ensure_constrained_propagation_deadline(
                    params.deadline,
                    "after constrained dense ReLU",
                )?;
                propagated.map_err(|error| {
                    NyError::InvalidSpec(format!(
                        "Constrained CROWN failed at node '{}' (ReLU): {}",
                        current.node_name, error
                    ))
                })?
            }
        } else {
            let propagated = relu.propagate_linear_with_bounds(&node_lb, pre_activation);
            super::super::ensure_constrained_propagation_deadline(
                params.deadline,
                "after constrained dense ReLU",
            )?;
            propagated.map_err(|error| {
                NyError::InvalidSpec(format!(
                    "Constrained CROWN failed at node '{}' (ReLU): {}",
                    current.node_name, error
                ))
            })?
        };

        if is_standard
            && params.deadline.is_none()
            && tracing::enabled!(tracing::Level::DEBUG)
            && params.context.history.constraints.len() >= 12
        {
            let out_gap = (new_lb.upper_a() - new_lb.lower_a()).mapv(f32::abs).sum();
            let out_bgap = (new_lb.upper_b() - new_lb.lower_b()).mapv(f32::abs).sum();
            debug!(
                "[#1817 bwd] {} output A-gap={:.6} b-gap={:.6}",
                current.node_name, out_gap, out_bgap
            );
        }

        apply_constrained_relu_beta_contribution(
            current.node_name,
            params.beta_state,
            &mut new_lb,
            params.deadline,
        )?;
        super::super::ensure_constrained_propagation_deadline(
            params.deadline,
            "before constrained dense ReLU publication",
        )?;
        params
            .graph
            .accumulate_dense_bounds_to_input_with_deadline(
                current.first_input,
                new_lb,
                &mut setup.state.node_crown_bounds,
                setup.output_dim,
                setup.input_dim,
                &mut setup.state.input_accumulated,
                params.deadline,
            )?;
        Ok(())
    }
}

#[cfg(test)]
mod constrained_patches_relu_gate_tests {
    use super::*;
    use crate::beta_crown::state::GraphBetaEntry;
    use crate::bounds::patches::{
        patches_to_dense_call_sites, reset_patches_to_dense_call_count, PatchGeometry, PatchesData,
        PatchesLinearBounds,
    };
    use crate::layers::ReLULayer;
    use crate::network::GraphNode;
    use ndarray::{array, Array, Array1, ArrayD, Dimension, IxDyn};
    use std::collections::HashMap;

    fn explicit_bounds(lower_shape: &[usize], upper_shape: &[usize]) -> PatchesLinearBounds {
        let side = |shape: &[usize]| PatchesData {
            coeff_err: Some(Array1::from_vec(vec![1.0e-4])),
            patches: Some(ArrayD::zeros(IxDyn(shape))),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, 1),
            unstable_idx: None,
        };
        PatchesLinearBounds {
            row_count: 1,
            lower_a: side(lower_shape),
            lower_b: Array1::zeros(1),
            upper_a: side(upper_shape),
            upper_b: Array1::zeros(1),
        }
    }

    fn capture_bounds() -> PatchesLinearBounds {
        let shape = IxDyn(&[2, 1, 1, 2, 1, 1, 2]);
        let side = |values: Vec<f32>, coeff_err: Vec<f32>| PatchesData {
            coeff_err: Some(Array1::from_vec(coeff_err)),
            patches: Some(
                ArrayD::from_shape_vec(shape.clone(), values)
                    .expect("test Patches values match the explicit-row shape"),
            ),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (1, 1, 2),
            input_shape: (1, 1, 3),
            unstable_idx: None,
        };
        PatchesLinearBounds {
            row_count: 2,
            lower_a: side(
                vec![0.25, -0.5, 1.0, 2.0, -3.0, 4.0, 0.125, -0.25],
                vec![1.0e-4, 2.0e-4],
            ),
            lower_b: array![-0.0, 1.25],
            upper_a: side(
                vec![-0.75, 0.5, -1.5, 3.0, 2.25, -4.5, 0.375, 0.625],
                vec![3.0e-4, 4.0e-4],
            ),
            upper_b: array![0.0, -2.5],
        }
    }

    fn assert_array_bits_eq<D: Dimension>(actual: &Array<f32, D>, expected: &Array<f32, D>) {
        assert_eq!(actual.shape(), expected.shape());
        assert!(
            actual
                .iter()
                .zip(expected.iter())
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "array values differ bitwise: actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_linear_bounds_bits_eq(actual: &LinearBounds, expected: &LinearBounds) {
        assert_array_bits_eq(actual.lower_a(), expected.lower_a());
        assert_array_bits_eq(actual.lower_b(), expected.lower_b());
        assert_array_bits_eq(actual.upper_a(), expected.upper_a());
        assert_array_bits_eq(actual.upper_b(), expected.upper_b());
        match (actual.lower_a_err(), expected.lower_a_err()) {
            (Some(actual), Some(expected)) => assert_array_bits_eq(actual, expected),
            (None, None) => {}
            (actual, expected) => {
                panic!("lower coefficient-error presence differs: {actual:?} != {expected:?}")
            }
        }
        match (actual.upper_a_err(), expected.upper_a_err()) {
            (Some(actual), Some(expected)) => assert_array_bits_eq(actual, expected),
            (None, None) => {}
            (actual, expected) => {
                panic!("upper coefficient-error presence differs: {actual:?} != {expected:?}")
            }
        }
    }

    fn assert_patches_bounds_bits_eq(actual: &PatchesLinearBounds, expected: &PatchesLinearBounds) {
        fn assert_side(actual: &PatchesData, expected: &PatchesData) {
            assert_eq!(actual.geometry, expected.geometry);
            assert_eq!(actual.identity, expected.identity);
            assert_eq!(actual.output_shape, expected.output_shape);
            assert_eq!(actual.input_shape, expected.input_shape);
            assert_eq!(actual.unstable_idx, expected.unstable_idx);
            match (&actual.patches, &expected.patches) {
                (Some(actual), Some(expected)) => assert_array_bits_eq(actual, expected),
                (None, None) => {}
                _ => panic!("patch materialization presence differs"),
            }
            match (&actual.coeff_err, &expected.coeff_err) {
                (Some(actual), Some(expected)) => assert_array_bits_eq(actual, expected),
                (None, None) => {}
                _ => panic!("coefficient-error presence differs"),
            }
        }

        assert_eq!(actual.row_count, expected.row_count);
        assert_side(&actual.lower_a, &expected.lower_a);
        assert_array_bits_eq(&actual.lower_b, &expected.lower_b);
        assert_side(&actual.upper_a, &expected.upper_a);
        assert_array_bits_eq(&actual.upper_b, &expected.upper_b);
    }

    fn single_to_dense_call_site(context: &str) -> String {
        let sites = patches_to_dense_call_sites();
        assert_eq!(
            sites.len(),
            1,
            "{context} should perform exactly one Patches-to-Dense conversion, got {sites:?}"
        );
        sites
            .into_iter()
            .next()
            .expect("one conversion site was asserted")
    }

    fn helper_capture_route_call_site(node_cb: &CrownBounds) -> String {
        reset_patches_to_dense_call_count();
        let _captured = constrained_relu_intermediate_capture_bounds(node_cb, None, 0)
            .expect("helper route should materialize the test Patches bounds");
        single_to_dense_call_site("direct helper route")
    }

    fn production_capture_route_call_site(
        current: &ConstrainedNodeContext<'_>,
        pre_activation: &BoundedTensor,
        node_cb: &CrownBounds,
        lookups: &ConstraintLookups,
    ) -> String {
        reset_patches_to_dense_call_count();
        let mut intermediate = Some(GraphAlphaCrownIntermediate::new());
        capture_constrained_relu_intermediate(
            current,
            pre_activation,
            node_cb,
            Some(lookups),
            None,
            false,
            None,
            &mut intermediate,
        )
        .expect("production capture route should succeed");
        assert!(
            intermediate
                .as_ref()
                .is_some_and(|stored| stored.a_at_relu("relu_capture").is_some()),
            "production route should capture the selected ReLU"
        );
        single_to_dense_call_site("production capture route")
    }

    #[test]
    fn owned_alpha_relu_gate_is_exact_and_default_dark() {
        crate::tests::with_env_edits(|env| {
            env.remove("NY_CONSTRAINED_PATCHES_ALPHA_RELU_INPLACE");
            assert!(!constrained_patches_alpha_relu_in_place_enabled());
            for value in ["", "0", "true", "01", " 1", "1 "] {
                env.set("NY_CONSTRAINED_PATCHES_ALPHA_RELU_INPLACE", value);
                assert!(!constrained_patches_alpha_relu_in_place_enabled());
            }
            env.set("NY_CONSTRAINED_PATCHES_ALPHA_RELU_INPLACE", "1");
            assert!(constrained_patches_alpha_relu_in_place_enabled());
        });
    }

    #[test]
    fn beta_sparse_capture_gate_is_exact_and_default_dark() {
        crate::tests::with_env_edits(|env| {
            env.remove("NY_CONSTRAINED_PATCHES_BETA_SPARSE_CAPTURE");
            assert!(!constrained_patches_beta_sparse_capture_enabled());
            for value in ["", "0", "true", "01", " 1", "1 "] {
                env.set("NY_CONSTRAINED_PATCHES_BETA_SPARSE_CAPTURE", value);
                assert!(!constrained_patches_beta_sparse_capture_enabled());
            }
            env.set("NY_CONSTRAINED_PATCHES_BETA_SPARSE_CAPTURE", "1");
            assert!(constrained_patches_beta_sparse_capture_enabled());
        });
    }

    #[test]
    fn beta_sparse_capture_requires_exact_lookup_and_beta_column_sets() {
        let lookups = ConstraintLookups {
            by_relu: HashMap::from([(
                "relu_capture".to_string(),
                HashMap::from([(0usize, true), (2usize, false)]),
            )]),
            pre: HashMap::new(),
            pre_genbab: HashMap::new(),
        };
        let matching = GraphBetaState::from_entries(vec![
            GraphBetaEntry::new("relu_capture".to_string(), 2, 0.0, 0.0, -1.0)
                .expect("valid beta entry"),
            GraphBetaEntry::new("relu_capture".to_string(), 0, 0.0, 0.0, 1.0)
                .expect("valid beta entry"),
            GraphBetaEntry::new("relu_capture".to_string(), 2, 0.0, 0.0, 1.0)
                .expect("valid duplicate-neuron beta entry"),
        ]);
        assert_eq!(
            matching_beta_capture_columns("relu_capture", Some(&lookups), Some(&matching)),
            Some(vec![0, 2])
        );

        let missing = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu_capture".to_string(),
            0,
            0.0,
            0.0,
            1.0,
        )
        .expect("valid beta entry")]);
        assert!(
            matching_beta_capture_columns("relu_capture", Some(&lookups), Some(&missing)).is_none(),
            "lookup/beta mismatch must refuse partial capture"
        );
        assert!(
            matching_beta_capture_columns("relu_capture", Some(&lookups), None).is_none(),
            "capture without a beta consumer must remain full dense"
        );
    }

    #[test]
    fn borrowed_patches_capture_is_bit_exact_with_direct_materialization() {
        let _env = crate::tests::lock_env_shared();
        let patches = capture_bounds();
        let expected = patches
            .to_dense()
            .expect("direct Patches materialization should succeed");
        let node_cb = CrownBounds::Patches(Box::new(patches));

        let borrowed = constrained_relu_intermediate_capture_bounds(&node_cb, None, 0)
            .expect("borrowed capture should materialize");

        assert_array_bits_eq(&borrowed, expected.lower_a());
    }

    #[test]
    fn constrained_relu_capture_budget_refusal_preserves_anchored_source_and_storage() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_DENSE_BUDGET_MB", "0");

            let mut patches = capture_bounds();
            let geometry =
                PatchGeometry::anchored(vec![0], vec![0, 1]).expect("fixture axes are non-empty");
            patches.lower_a.geometry = geometry.clone();
            patches.upper_a.geometry = geometry;
            let expected = patches.clone();
            let node_cb = CrownBounds::Patches(Box::new(patches));
            let node = GraphNode::new(
                "relu_capture",
                Layer::ReLU(ReLULayer::new()),
                vec!["pre_activation".to_string()],
            );
            let current = ConstrainedNodeContext {
                node_name: "relu_capture",
                node: &node,
                first_input: "pre_activation",
            };
            let pre_activation = BoundedTensor::new(
                array![-1.25, -0.0, 2.5].into_dyn(),
                array![0.5, 1.75, 3.0].into_dyn(),
            )
            .expect("test pre-activation bounds are valid");
            let lookups = ConstraintLookups {
                by_relu: HashMap::from([("relu_capture".to_string(), HashMap::new())]),
                pre: HashMap::new(),
                pre_genbab: HashMap::new(),
            };
            let mut intermediate = Some(GraphAlphaCrownIntermediate::new());

            let error = capture_constrained_relu_intermediate(
                &current,
                &pre_activation,
                &node_cb,
                Some(&lookups),
                None,
                false,
                None,
                &mut intermediate,
            )
            .expect_err("zero budget must refuse constrained ReLU capture");
            assert!(
                matches!(error, NyError::CpuMemoryExceeded { .. }),
                "expected typed memory refusal, got {error:?}"
            );
            match &node_cb {
                CrownBounds::Patches(actual) => assert_patches_bounds_bits_eq(actual, &expected),
                CrownBounds::Dense(_) => panic!("borrowed capture replaced source Patches"),
            }
            let intermediate = intermediate.expect("capture storage remains present");
            assert!(intermediate.a_at_relu.is_empty());
            assert!(intermediate.pre_relu_bounds.is_empty());
        });
    }

    #[test]
    fn constrained_relu_capture_deadline_refusal_is_atomic() {
        let patches = capture_bounds();
        let expected = patches.clone();
        let node_cb = CrownBounds::Patches(Box::new(patches));
        let node = GraphNode::new(
            "relu_capture",
            Layer::ReLU(ReLULayer::new()),
            vec!["pre_activation".to_string()],
        );
        let current = ConstrainedNodeContext {
            node_name: "relu_capture",
            node: &node,
            first_input: "pre_activation",
        };
        let pre_activation = BoundedTensor::new(
            array![-1.25, -0.0, 2.5].into_dyn(),
            array![0.5, 1.75, 3.0].into_dyn(),
        )
        .expect("test pre-activation bounds are valid");
        let lookups = ConstraintLookups {
            by_relu: HashMap::from([("relu_capture".to_string(), HashMap::new())]),
            pre: HashMap::new(),
            pre_genbab: HashMap::new(),
        };
        let mut intermediate = Some(GraphAlphaCrownIntermediate::new());
        intermediate
            .as_mut()
            .expect("intermediate fixture")
            .a_at_relu
            .insert("sentinel".to_string(), array![[1.0]]);
        let expired = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .expect("one-second deadline subtraction");

        let error = capture_constrained_relu_intermediate(
            &current,
            &pre_activation,
            &node_cb,
            Some(&lookups),
            None,
            false,
            Some(expired),
            &mut intermediate,
        )
        .expect_err("expired capture must be terminal");

        assert!(error.is_deadline_exceeded());
        match &node_cb {
            CrownBounds::Patches(actual) => assert_patches_bounds_bits_eq(actual, &expected),
            CrownBounds::Dense(_) => panic!("deadline refusal replaced source Patches"),
        }
        let intermediate = intermediate.expect("intermediate storage remains present");
        assert_eq!(intermediate.a_at_relu.len(), 1);
        assert!(intermediate.a_at_relu.contains_key("sentinel"));
        assert!(intermediate.pre_relu_bounds.is_empty());
    }

    #[test]
    fn expired_beta_application_does_not_mutate_dense_transaction() {
        let mut bounds = LinearBounds::identity(3);
        let expected = bounds.clone();
        let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu_capture".to_string(),
            1,
            0.5,
            0.0,
            1.0,
        )
        .expect("valid beta entry")]);
        let expired = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .expect("one-second deadline subtraction");

        let error = apply_constrained_relu_beta_contribution(
            "relu_capture",
            Some(&beta),
            &mut bounds,
            Some(expired),
        )
        .expect_err("expired beta transaction must be terminal");

        assert!(error.is_deadline_exceeded());
        assert_linear_bounds_bits_eq(&bounds, &expected);
    }

    #[test]
    fn production_capture_uses_the_borrowed_helper_route_unconditionally() {
        let _env = crate::tests::lock_env_shared();
        let node_cb = CrownBounds::Patches(Box::new(capture_bounds()));
        let borrowed_site = helper_capture_route_call_site(&node_cb);
        // `file!()` yields the host separator — `constraints\backward\relu.rs` on
        // Windows — so matching a hardcoded POSIX path made this assertion
        // unsatisfiable off Unix, not merely brittle. Normalize before comparing.
        assert!(
            borrowed_site
                .replace('\\', "/")
                .contains("constraints/backward/relu.rs:"),
            "route fingerprint should identify this helper: {borrowed_site:?}"
        );

        let node = GraphNode::new(
            "relu_capture",
            Layer::ReLU(ReLULayer::new()),
            vec!["pre_activation".to_string()],
        );
        let current = ConstrainedNodeContext {
            node_name: "relu_capture",
            node: &node,
            first_input: "pre_activation",
        };
        let pre_activation = BoundedTensor::new(
            array![-1.25, -0.0, 2.5].into_dyn(),
            array![0.5, 1.75, 3.0].into_dyn(),
        )
        .expect("test pre-activation bounds are valid");
        let lookups = ConstraintLookups {
            by_relu: HashMap::from([("relu_capture".to_string(), HashMap::new())]),
            pre: HashMap::new(),
            pre_genbab: HashMap::new(),
        };

        let production_site =
            production_capture_route_call_site(&current, &pre_activation, &node_cb, &lookups);
        assert_eq!(production_site, borrowed_site);
    }

    #[test]
    fn borrowed_capture_does_not_change_dense_clone_semantics() {
        let mut dense = LinearBounds::new(
            array![[0.25, -0.0, 1.5], [-2.0, 3.25, 4.5]],
            array![-0.0, 0.75],
            array![[-0.5, 1.0, 2.5], [3.0, -4.25, 5.5]],
            array![0.0, -1.25],
        )
        .expect("test Dense bounds are valid");
        dense.set_coeff_err(
            array![[1.0e-4, 2.0e-4, 3.0e-4], [4.0e-4, 5.0e-4, 6.0e-4]],
            array![[7.0e-4, 8.0e-4, 9.0e-4], [1.0e-3, 1.1e-3, 1.2e-3]],
        );
        let node_cb = CrownBounds::Dense(dense.clone());

        let captured = constrained_relu_intermediate_capture_bounds(&node_cb, None, 0)
            .expect("Dense capture should clone");

        assert_array_bits_eq(&captured, dense.lower_a());
    }

    #[test]
    fn borrowed_capture_records_expected_intermediate_without_replacing_patches() {
        let patches = capture_bounds();
        let expected = patches
            .to_dense()
            .expect("expected capture should materialize");
        let node_cb = CrownBounds::Patches(Box::new(patches));
        let (lower_ptr, upper_ptr) = match &node_cb {
            CrownBounds::Patches(bounds) => (
                bounds
                    .lower_a
                    .patches
                    .as_ref()
                    .expect("lower Patches tensor")
                    .as_ptr(),
                bounds
                    .upper_a
                    .patches
                    .as_ref()
                    .expect("upper Patches tensor")
                    .as_ptr(),
            ),
            CrownBounds::Dense(_) => unreachable!("test starts with Patches"),
        };

        let node = GraphNode::new(
            "relu_capture",
            Layer::ReLU(ReLULayer::new()),
            vec!["pre_activation".to_string()],
        );
        let current = ConstrainedNodeContext {
            node_name: "relu_capture",
            node: &node,
            first_input: "pre_activation",
        };
        let pre_activation = BoundedTensor::new(
            array![-1.25, -0.0, 2.5].into_dyn(),
            array![0.5, 1.75, 3.0].into_dyn(),
        )
        .expect("test pre-activation bounds are valid");
        let lookups = ConstraintLookups {
            by_relu: HashMap::from([("relu_capture".to_string(), HashMap::new())]),
            pre: HashMap::new(),
            pre_genbab: HashMap::new(),
        };
        let mut intermediate = Some(GraphAlphaCrownIntermediate::new());

        capture_constrained_relu_intermediate(
            &current,
            &pre_activation,
            &node_cb,
            Some(&lookups),
            None,
            false,
            None,
            &mut intermediate,
        )
        .expect("borrowed Patches capture should succeed");

        let intermediate = intermediate.expect("capture storage remains present");
        assert_array_bits_eq(
            intermediate
                .a_at_relu("relu_capture")
                .expect("ReLU A matrix captured"),
            expected.lower_a(),
        );
        let (captured_lower, captured_upper) = intermediate
            .pre_relu_bounds("relu_capture")
            .expect("pre-ReLU bounds captured");
        assert_array_bits_eq(captured_lower, &array![-1.25, -0.0, 2.5]);
        assert_array_bits_eq(captured_upper, &array![0.5, 1.75, 3.0]);

        match &node_cb {
            CrownBounds::Patches(bounds) => {
                assert_eq!(
                    bounds
                        .lower_a
                        .patches
                        .as_ref()
                        .expect("lower Patches tensor retained")
                        .as_ptr(),
                    lower_ptr
                );
                assert_eq!(
                    bounds
                        .upper_a
                        .patches
                        .as_ref()
                        .expect("upper Patches tensor retained")
                        .as_ptr(),
                    upper_ptr
                );
            }
            CrownBounds::Dense(_) => panic!("borrowed capture replaced the Patches carrier"),
        }
    }

    #[test]
    fn beta_only_sparse_capture_avoids_dense_but_joint_alpha_beta_stays_full() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_CONSTRAINED_PATCHES_BETA_SPARSE_CAPTURE", "1");

            let patches = capture_bounds();
            let expected = patches
                .to_dense()
                .expect("expected capture should materialize");
            let node_cb = CrownBounds::Patches(Box::new(patches));
            let node = GraphNode::new(
                "relu_capture",
                Layer::ReLU(ReLULayer::new()),
                vec!["pre_activation".to_string()],
            );
            let current = ConstrainedNodeContext {
                node_name: "relu_capture",
                node: &node,
                first_input: "pre_activation",
            };
            let pre_activation = BoundedTensor::new(
                array![-1.25, -0.0, 2.5].into_dyn(),
                array![0.5, 1.75, 3.0].into_dyn(),
            )
            .expect("test pre-activation bounds are valid");
            let lookups = ConstraintLookups {
                by_relu: HashMap::from([(
                    "relu_capture".to_string(),
                    HashMap::from([(0usize, true)]),
                )]),
                pre: HashMap::new(),
                pre_genbab: HashMap::new(),
            };
            let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
                "relu_capture".to_string(),
                0,
                0.0,
                0.0,
                1.0,
            )
            .expect("valid beta entry")]);

            reset_patches_to_dense_call_count();
            let mut beta_only = Some(GraphAlphaCrownIntermediate::new());
            capture_constrained_relu_intermediate(
                &current,
                &pre_activation,
                &node_cb,
                Some(&lookups),
                Some(&beta),
                false,
                None,
                &mut beta_only,
            )
            .expect("beta-only sparse capture should succeed");
            assert!(
                patches_to_dense_call_sites().is_empty(),
                "selected-column capture must not invoke full Patches-to-Dense"
            );
            let beta_only = beta_only.expect("capture storage remains present");
            assert!(beta_only.has_beta_sparse_a("relu_capture"));
            assert!(beta_only.a_at_relu("relu_capture").is_none());
            let selected = beta_only
                .beta_a_column("relu_capture", 0)
                .expect("selected beta column captured");
            for (row, &actual) in selected.iter().enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.lower_a()[[row, 0]].to_bits(),
                    "selected beta column must match historical dense capture"
                );
            }

            reset_patches_to_dense_call_count();
            let mut joint = Some(GraphAlphaCrownIntermediate::new());
            capture_constrained_relu_intermediate(
                &current,
                &pre_activation,
                &node_cb,
                Some(&lookups),
                Some(&beta),
                true,
                None,
                &mut joint,
            )
            .expect("joint alpha-beta capture should retain full dense path");
            assert_eq!(
                patches_to_dense_call_sites().len(),
                1,
                "joint alpha-beta must materialize the historical full A matrix"
            );
            let joint = joint.expect("capture storage remains present");
            assert!(joint.a_at_relu("relu_capture").is_some());
            assert!(!joint.has_beta_sparse_a("relu_capture"));
        });
    }

    #[test]
    fn owned_alpha_relu_scope_requires_standard_without_intermediate() {
        assert!(constrained_patches_alpha_relu_in_place_scope_is_bound_only(
            true, false
        ));
        assert!(!constrained_patches_alpha_relu_in_place_scope_is_bound_only(false, false));
        assert!(!constrained_patches_alpha_relu_in_place_scope_is_bound_only(true, true));
        assert!(!constrained_patches_alpha_relu_in_place_scope_is_bound_only(false, true));
    }

    #[test]
    fn owned_alpha_relu_candidate_requires_matching_contiguous_explicit_rows() {
        let exact = explicit_bounds(&[1, 1, 1, 1, 1, 1, 1], &[1, 1, 1, 1, 1, 1, 1]);
        assert!(is_materialized_contiguous_explicit_row_alpha_candidate(
            &exact
        ));

        let mixed = explicit_bounds(&[1, 1, 1, 1, 1, 1, 1], &[1, 1, 1, 1, 1, 1]);
        assert!(!is_materialized_contiguous_explicit_row_alpha_candidate(
            &mixed
        ));

        let mut identity = exact;
        identity.lower_a.identity = true;
        identity.lower_a.patches = None;
        assert!(!is_materialized_contiguous_explicit_row_alpha_candidate(
            &identity
        ));
    }
}
