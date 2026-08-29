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

/// #bab-warm-start diagnostics: how often the monotone floor actually fires
/// (child bound regressed below its parent) vs children created. Read by the
/// `NY_ACASXU_PROF` tick in `engine/core.rs`.
pub(in crate::beta_crown) static MONOTONE_CLAMP_FIRED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(in crate::beta_crown) static CHILDREN_CREATED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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

        // WARM-START from the parent's optimized state (#bab-warm-start, the
        // verified=0 root-cause fix): inherited constraints keep the parent's
        // β; inherited unstable neurons keep the parent's α (+ optimizer
        // moments). Cold-starting both at every split kept every domain's
        // bound at root quality regardless of depth (measured: zero closures
        // at depth 22-31 on `mnist_fc 256x4`). Soundness: initialization
        // affects only the optimizer's starting point — any β ≥ 0 / α ∈ [0,1]
        // yields a valid bound.
        let mut beta_state = BetaState::from_history_warm(&new_history, &parent.beta_state)?;

        let mut domain_alpha_state = if self.config.use_alpha_crown {
            let mut state = DomainAlphaState::from_layer_bounds_and_constraints(
                network,
                &new_layer_bounds,
                &new_history,
            );
            state.warm_start_from(&parent.domain_alpha_state);
            state
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
        // CAPABILITY GATE (see `clip::sequential_clip_interm_domain_supported`).
        // A preset may REQUEST clip_interm_domain on the sequential engine, where
        // the feature is quarantined; the contract is that the request is SKIPPED,
        // so BaB branches exactly as if the flag were off. Skipping an optional
        // bound-tightening is sound (bounds remain valid, just less tight) —
        // erroring out every child is what silently converted provable instances
        // into `unknown`, because this gate fires for every child at depth > 0,
        // i.e. every child, so the search collapsed at the root.
        //
        // DEFENCE IN DEPTH: `apply_clip_interm_domain` independently returns the
        // caller's bounds unchanged when the capability is unsupported, so dropping
        // this conjunct costs throughput but cannot re-break the search. Both halves
        // are covered by `engine::tests::clip_interm_domain_gate`. When the
        // provenance-preserving conversion lands, that predicate flips to true and
        // this arm starts applying the tightening for real.
        // #clip-interm-notice: the skip notice lives inside
        // `apply_clip_interm_domain`, but this gate means that function is never
        // entered in production — so a preset requesting the feature got NOTHING
        // and was told nothing. `configs/vnncomp25/relusplitter.yaml:85` sets it.
        // Fire the (once-per-process) notice here, where the request is actually
        // declined. Logging only: no bound, verdict or control flow changes.
        if self.config.enable_clip_interm_domain
            && !super::clip::sequential_clip_interm_domain_supported()
        {
            super::clip::warn_sequential_clip_interm_skipped_once();
        }
        let new_layer_bounds = if self.config.enable_clip_interm_domain
            && super::clip::sequential_clip_interm_domain_supported()
            && new_history.depth() > 0
            && !intermediate_bounds.is_empty()
        {
            let bounds = self.apply_clip_interm_domain(
                network,
                &new_history,
                new_layer_bounds,
                &intermediate_bounds,
                input,
                parent.input_bounds.as_deref(),
            )?;
            if self.config.clip_interm_prune && has_infeasible_layer_bounds(&bounds) {
                debug!("clip_interm_domain pruned domain with infeasible bounds");
                return Ok(None);
            }
            bounds
        } else {
            new_layer_bounds
        };

        let mut new_lower = output_bounds.lower_scalar();
        let new_upper = output_bounds.upper_scalar();
        // MONOTONE LOWER-BOUND INHERITANCE (#bab-warm-start companion). The
        // child's feasible region is a subset of the parent's, so any sound
        // parent lower bound is a sound lower bound for the child — the
        // subset-containment argument already shipped for the graph lanes at
        // `graph/multi_objective/shared.rs` (`inherit_parent_lower_only`).
        // Without the floor, children whose per-child optimizer is weaker
        // than the engine that produced the parent bound REGRESS below it and
        // every split's gain first repays that gap. Lower (proof) side only;
        // skipped when it would invert against the child's upper bound so
        // violation detection is untouched.
        // #bab-warm-start trace (NY_BAB_CHILD_TRACE=1): sampled raw child
        // bounds BEFORE the monotone floor, against depth and the parent's
        // bound — distinguishes "splits improve the joint bound from below
        // the root baseline" (strategy converging, floor hides it) from
        // "splits are inert in the bound computation" (wiring bug).
        {
            static TRACE_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let on = *TRACE_ON.get_or_init(|| {
                ny_levers::read_raw(&ny_levers::decls::bab_diagnostics::BAB_CHILD_TRACE).as_deref()
                    == Some("1")
            });
            if on {
                let n = CHILDREN_CREATED.load(std::sync::atomic::Ordering::Relaxed);
                if n.is_multiple_of(97) {
                    eprintln!(
                        "[child-trace] depth={} preclamp_lower={:.6} parent_lower={:.6} upper={:.6}",
                        new_history.depth(),
                        new_lower,
                        parent.lower_bound,
                        new_upper,
                    );
                }
            }
        }
        if parent.lower_bound.is_finite() {
            let clamped = new_lower.max(parent.lower_bound);
            if clamped <= new_upper {
                if clamped > new_lower {
                    MONOTONE_CLAMP_FIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                new_lower = clamped;
            }
        }
        CHILDREN_CREATED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

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
