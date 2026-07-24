// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Child domain construction and split-history constraint replay.

use std::sync::Arc;

use ny_core::{nan_propagating_max, nan_propagating_min, GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, trace, warn};

use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::branching::{NeuronConstraint, SplitHistory};
use crate::beta_crown::domain::BabDomain;
use crate::beta_crown::state::{BetaState, DomainAlphaState};
use crate::Network;

use super::super::tensor_ext::{has_inverted_output_bounds, BoundedTensorExt};
use super::super::BetaCrownVerifier;
use super::clip::has_infeasible_layer_bounds;

impl BetaCrownVerifier {
    /// Create a child domain with an additional constraint.
    // Justification: Child domain creation needs network, input, parent domain, split
    // location (layer_idx, neuron_idx, is_active), and engine for bound recomputation.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine) fn create_child_domain(
        &self,
        network: &Network,
        input: &BoundedTensor,
        parent: &BabDomain,
        layer_idx: usize,
        neuron_idx: usize,
        is_active: bool,
        score: f32,
        _threshold: f32,
        cut_pool: &mut CutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Option<BabDomain>> {
        let constraint = NeuronConstraint {
            layer_idx,
            neuron_idx,
            is_active,
            score,
        };
        let new_history = parent.history.with_constraint(constraint);

        // Tighten bounds based on new constraint
        // Cloning Vec<Arc<_>> is cheap - only Arc pointers are cloned, not the underlying data
        let mut new_layer_bounds = parent.layer_bounds.clone();

        // Apply constraint to pre-activation bounds
        if layer_idx > 0 && layer_idx <= new_layer_bounds.len() {
            // Read from the Arc-wrapped bounds (no mut needed, we'll replace the entry)
            let pre_bounds = &new_layer_bounds[layer_idx - 1];
            let lower = pre_bounds.lower().clone();
            let upper = pre_bounds.upper().clone();

            // Flatten to access individual neurons
            let shape = lower.shape().to_vec();
            let lower_len = lower.len();
            let upper_len = upper.len();
            let mut lower_flat = lower
                .into_shape_clone(ndarray::IxDyn(&[lower_len]))
                .map_err(|err| {
                    NyError::InternalError(format!(
                        "create_child_domain: flatten lower bounds failed for layer {}: {}",
                        layer_idx, err
                    ))
                })?;
            let mut upper_flat = upper
                .into_shape_clone(ndarray::IxDyn(&[upper_len]))
                .map_err(|err| {
                    NyError::InternalError(format!(
                        "create_child_domain: flatten upper bounds failed for layer {}: {}",
                        layer_idx, err
                    ))
                })?;

            if is_active {
                // Neuron is active: x >= 0, so lower bound becomes max(lower, 0)
                // NaN-safe: propagate NaN instead of silently clamping to 0.0 (#2643)
                lower_flat[[neuron_idx]] = nan_propagating_max(lower_flat[[neuron_idx]], 0.0);
            } else {
                // Neuron is inactive: x <= 0, so upper bound becomes min(upper, 0)
                upper_flat[[neuron_idx]] = nan_propagating_min(upper_flat[[neuron_idx]], 0.0);
            }

            // Check if constraint makes domain infeasible
            if lower_flat[[neuron_idx]] > upper_flat[[neuron_idx]] {
                trace!("Child domain infeasible: constraint makes l > u");
                return Ok(None);
            }

            // Reshape back
            let lower_new = lower_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|err| {
                    NyError::InternalError(format!(
                        "create_child_domain: reshape lower bounds failed for layer {} to {:?}: {}",
                        layer_idx, shape, err
                    ))
                })?;
            let upper_new = upper_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|err| {
                    NyError::InternalError(format!(
                        "create_child_domain: reshape upper bounds failed for layer {} to {:?}: {}",
                        layer_idx, shape, err
                    ))
                })?;

            // Wrap new BoundedTensor in Arc - only the modified layer gets a new allocation
            new_layer_bounds[layer_idx - 1] = Arc::new(BoundedTensor::new(lower_new, upper_new)?);
        }

        // Initialize beta state from history
        let mut beta_state = BetaState::from_history(&new_history)?;

        // Initialize domain-specific alpha state for joint optimization
        let mut domain_alpha_state = if self.config.use_alpha_crown {
            DomainAlphaState::from_layer_bounds_and_constraints(
                network,
                &new_layer_bounds,
                &new_history,
            )
        } else {
            DomainAlphaState::empty()
        };

        // Optimize α, β, and λ (cut) parameters jointly for tighter bounds
        // Use intermediate bound transfer if parent has intermediate bounds:
        // - The backward pass from output to layer_idx+1 is unchanged
        // - Only recompute from layer_idx to input (saves num_layers - layer_idx - 1 layers)
        let (output_bounds, intermediate_bounds) =
            if !parent.intermediate_bounds.is_empty() && layer_idx < network.layers.len() - 1 {
                // Use optimized path: start from split layer using parent's intermediate bounds
                self.optimize_joint_bounds_from_layer(
                    network,
                    input,
                    &new_history,
                    &new_layer_bounds,
                    &mut beta_state,
                    &mut domain_alpha_state,
                    cut_pool,
                    layer_idx,
                    &parent.intermediate_bounds,
                    engine,
                )?
            } else {
                // Fall back to full computation (root domain or split at last layer)
                self.optimize_joint_bounds(
                    network,
                    input,
                    &new_history,
                    &new_layer_bounds,
                    &mut beta_state,
                    &mut domain_alpha_state,
                    cut_pool,
                    engine,
                )?
            };

        // Tighten intermediate bounds using split constraints (clip_interm_domain)
        let new_layer_bounds = if self.config.enable_clip_interm_domain
            && new_history.depth() > 0
            && !intermediate_bounds.is_empty()
        {
            match self.apply_clip_interm_domain(
                &new_history,
                new_layer_bounds,
                &intermediate_bounds,
                input,
                parent.input_bounds.as_deref(),
            ) {
                Ok(bounds) => {
                    if self.config.clip_interm_prune && has_infeasible_layer_bounds(&bounds) {
                        debug!("clip_interm_domain pruned domain with infeasible bounds");
                        return Ok(None);
                    }
                    bounds
                }
                Err(err) if self.config.clip_interm_prune => {
                    debug!("clip_interm_domain pruned infeasible domain: {}", err);
                    return Ok(None);
                }
                Err(err) => return Err(err),
            }
        } else {
            new_layer_bounds
        };

        let new_lower = output_bounds.lower_scalar();
        let new_upper = output_bounds.upper_scalar();

        // #2950: Prune domains with inverted output bounds (lower > upper for any element).
        // BaB splits can make the constraint set infeasible — these domains can never
        // produce counterexamples so exploring them wastes verification budget.
        // Reference: alpha-beta-CROWN branching_domains.py:370-381 (check_infeasible_bounds).
        if has_inverted_output_bounds(&output_bounds) {
            debug!(
                "create_child_domain: pruned infeasible domain — inverted output bounds (depth={})",
                new_history.depth()
            );
            return Ok(None);
        }

        // Guard: non-finite bounds from lower_scalar/upper_scalar indicate NaN/Inf
        // propagation through bounds. These produce zombie domains that consume BaB
        // budget without contributing to verification. Prune at construction to match
        // GPU BaB is_finite() guard in batched_gpu.rs. (#2986)
        if !new_lower.is_finite() || !new_upper.is_finite() {
            warn!(
                "create_child_domain: pruned zombie domain — non-finite bounds \
                 (lb={}, ub={}, depth={})",
                new_lower,
                new_upper,
                new_history.depth()
            );
            return Ok(None);
        }

        // Use validated child constructor to enforce NaN rejection (#3125).
        // Priority defaults to lower_bound; BaB loop overwrites via
        // set_priority()/violation_priority() before queue insertion (#2682).
        Ok(Some(BabDomain::child(
            new_history,
            new_lower,
            new_upper,
            new_layer_bounds,
            parent.alpha_state.clone(),
            domain_alpha_state,
            beta_state,
            parent.input_bounds.clone(),
            parent.input_split_count,
            intermediate_bounds,
        )?))
    }

    pub(in crate::beta_crown::engine) fn apply_history_constraints(
        &self,
        base_layer_bounds: &[Arc<BoundedTensor>],
        history: &SplitHistory,
    ) -> Result<Option<Vec<Arc<BoundedTensor>>>> {
        let mut layer_bounds = base_layer_bounds.to_vec();

        for constraint in &history.constraints {
            if constraint.layer_idx == 0 || constraint.layer_idx > layer_bounds.len() {
                warn!(
                    "apply_history_constraints: skipping constraint with layer_idx {} \
                     (valid range 1..={}) — may indicate branching decision bug",
                    constraint.layer_idx,
                    layer_bounds.len()
                );
                continue;
            }

            let pre_bounds = &layer_bounds[constraint.layer_idx - 1];
            let lower = pre_bounds.lower().clone();
            let upper = pre_bounds.upper().clone();
            let shape = lower.shape().to_vec();
            let lower_len = lower.len();
            let upper_len = upper.len();
            if constraint.neuron_idx >= lower_len || constraint.neuron_idx >= upper_len {
                warn!(
                    "apply_history_constraints: skipping constraint with neuron_idx {} \
                     at layer {} (lower_len={}, upper_len={}) — may indicate branching decision bug",
                    constraint.neuron_idx, constraint.layer_idx, lower_len, upper_len
                );
                continue;
            }

            let mut lower_flat = lower
                .into_shape_clone(ndarray::IxDyn(&[lower_len]))
                .map_err(|err| {
                    NyError::InternalError(format!(
                        "apply_history_constraints: flatten lower bounds failed for layer {}: {}",
                        constraint.layer_idx, err
                    ))
                })?;
            let mut upper_flat = upper
                .into_shape_clone(ndarray::IxDyn(&[upper_len]))
                .map_err(|err| {
                    NyError::InternalError(format!(
                        "apply_history_constraints: flatten upper bounds failed for layer {}: {}",
                        constraint.layer_idx, err
                    ))
                })?;

            if constraint.is_active {
                // NaN-safe: propagate NaN instead of silently clamping to 0.0 (#2643)
                lower_flat[[constraint.neuron_idx]] =
                    nan_propagating_max(lower_flat[[constraint.neuron_idx]], 0.0);
            } else {
                upper_flat[[constraint.neuron_idx]] =
                    nan_propagating_min(upper_flat[[constraint.neuron_idx]], 0.0);
            }

            if lower_flat[[constraint.neuron_idx]] > upper_flat[[constraint.neuron_idx]] {
                return Ok(None);
            }

            let lower_new = lower_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|err| {
                    NyError::InternalError(format!(
                        "apply_history_constraints: reshape lower bounds failed for layer {} to {:?}: {}",
                        constraint.layer_idx, shape, err
                    ))
                })?;
            let upper_new = upper_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|err| {
                    NyError::InternalError(format!(
                        "apply_history_constraints: reshape upper bounds failed for layer {} to {:?}: {}",
                        constraint.layer_idx, shape, err
                    ))
                })?;
            layer_bounds[constraint.layer_idx - 1] =
                Arc::new(BoundedTensor::new(lower_new, upper_new)?);
        }

        Ok(Some(layer_bounds))
    }
}
