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

use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, trace};

use crate::beta_crown::state::GraphBetaState;
use crate::bounds::patches::CrownBounds;
use crate::bounds::GraphAlphaCrownIntermediate;
use crate::network::{crown_backward_step_patches, CrownStepResult};
use crate::{Layer, LinearBounds};

use super::super::super::super::super::BetaCrownVerifier;
use super::super::lookups::ConstraintLookups;
use super::super::patches::patches_dense_fallback_details;
use super::dispatch::ConstrainedNodeContext;
use super::{BackwardParams, ConstrainedBackwardSetup};

pub(super) fn capture_constrained_relu_intermediate(
    current: &ConstrainedNodeContext<'_>,
    pre_activation: &BoundedTensor,
    node_cb: &CrownBounds,
    mode_lookups: Option<&ConstraintLookups>,
    intermediate: &mut Option<GraphAlphaCrownIntermediate>,
) -> Result<()> {
    let should_capture_intermediate =
        mode_lookups.is_some_and(|lookups| lookups.by_relu.contains_key(current.node_name));
    if !should_capture_intermediate {
        return Ok(());
    }

    let stored_lb = match node_cb {
        CrownBounds::Dense(lb) => lb.clone(),
        CrownBounds::Patches(_) => node_cb.clone().into_dense()?,
    };
    let Some(intermediate) = intermediate.as_mut() else {
        return Ok(());
    };
    intermediate
        .a_at_relu
        .insert(current.node_name.to_string(), stored_lb.lower_a().clone());

    // Store pre-ReLU bounds as Ix1 for gradient computation.
    // BoundedTensor::flatten() always produces 1D arrays
    // (Array1::from_vec -> into_dyn), so into_dimensionality
    // cannot fail. We propagate the error instead of silently
    // falling back to zeros, which was unsound (#1926).
    let flat = pre_activation.flatten();
    let lower = flat
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|error| {
            NyError::InternalError(format!(
                "pre-ReLU lower at '{}' not convertible to Ix1 (shape {:?}): {}",
                current.node_name,
                flat.lower().shape(),
                error
            ))
        })?;
    let upper = flat
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|error| {
            NyError::InternalError(format!(
                "pre-ReLU upper at '{}' not convertible to Ix1 (shape {:?}): {}",
                current.node_name,
                flat.upper().shape(),
                error
            ))
        })?;
    intermediate
        .pre_relu_bounds
        .insert(current.node_name.to_string(), (lower, upper));
    Ok(())
}

pub(super) fn apply_constrained_relu_beta_contribution(
    node_name: &str,
    beta_state: Option<&GraphBetaState>,
    lb: &mut LinearBounds,
) {
    let Some(beta_state) = beta_state else {
        return;
    };

    // Part of #2936: use indexed per-node iteration instead of O(B) full scan.
    for entry in beta_state.entries_for_node(node_name) {
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
) -> Result<bool> {
    if !matches!(node_cb, CrownBounds::Patches(_)) {
        return Ok(false);
    }

    let Layer::ReLU(relu) = &current.node.layer else {
        unreachable!("Constrained ReLU patches helper requires a ReLU node");
    };

    let mut relu_applied_via_patches = false;
    if let Some(alpha_state) = params.context.alpha_state {
        if !alpha_state.is_empty() {
            let alphas = alpha_state.build_alpha_array(current.node_name, pre_activation);
            if let CrownBounds::Patches(patches_bounds) = node_cb {
                match relu.propagate_patches_with_alpha(patches_bounds, pre_activation, &alphas) {
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
            node_cb.ensure_dense()?;
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
            node_cb.ensure_dense()?;
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
            &mut setup.state.intermediate,
        )?;

        if is_standard && tracing::enabled!(tracing::Level::DEBUG) {
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
        )? {
            // Only convert Patches→Dense when beta entries exist for this node.
            // The original code called ensure_dense() inside the beta loop after
            // filtering by node_name. Unconditional ensure_dense() would change
            // the accumulation code path (Dense vs Patches) when beta_state is
            // None or has no matching entries for the current ReLU.
            if has_constrained_relu_beta_entry(current.node_name, params.beta_state) {
                let new_lb = node_cb.ensure_dense()?;
                apply_constrained_relu_beta_contribution(
                    current.node_name,
                    params.beta_state,
                    new_lb,
                );
            }
            params.graph.accumulate_crown_bounds_to_input(
                current.first_input,
                node_cb,
                &mut setup.state.node_crown_bounds,
                setup.output_dim,
                setup.input_dim,
                &mut setup.state.input_accumulated,
            )?;
            return Ok(());
        }

        let node_lb = node_cb.into_dense()?;
        if is_standard
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
                let (lb_result, grad, grad_upper) = relu
                    .propagate_linear_with_alpha(
                        &node_lb,
                        pre_activation,
                        &alphas,
                        Some(&alphas_upper),
                    )
                    .map_err(|error| {
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
                relu.propagate_linear_with_bounds(&node_lb, pre_activation)
                    .map_err(|error| {
                        NyError::InvalidSpec(format!(
                            "Constrained CROWN failed at node '{}' (ReLU): {}",
                            current.node_name, error
                        ))
                    })?
            }
        } else {
            relu.propagate_linear_with_bounds(&node_lb, pre_activation)
                .map_err(|error| {
                    NyError::InvalidSpec(format!(
                        "Constrained CROWN failed at node '{}' (ReLU): {}",
                        current.node_name, error
                    ))
                })?
        };

        if is_standard
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

        apply_constrained_relu_beta_contribution(current.node_name, params.beta_state, &mut new_lb);
        params.graph.accumulate_dense_bounds_to_input(
            current.first_input,
            new_lb,
            &mut setup.state.node_crown_bounds,
            setup.output_dim,
            setup.input_dim,
            &mut setup.state.input_accumulated,
        )?;
        Ok(())
    }
}
