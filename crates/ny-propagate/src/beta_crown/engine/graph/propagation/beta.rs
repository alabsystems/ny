// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! β-parameter (Lagrangian multiplier) propagation for graph CROWN.
//!
//! Extends constraint-aware CROWN with Lagrangian β contributions that encode BaB
//! split decisions. For each neuron constraint, β multipliers tighten the relaxation:
//! - Active constraint (x ≥ 0): adds β × x_lower to the bound
//! - Inactive constraint (x ≤ 0): subtracts β × x_upper from the bound
//!
//! Also provides SPSA-based β optimization for DAG networks where analytical
//! gradients through skip connections are complex.

use std::sync::Arc;

use ny_core::{nan_propagating_max, GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info, trace, warn};

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::domain::{GraphBabDomain, GraphCrownContext};
use crate::beta_crown::engine::graph::{DomainCrownResult, DomainCrownResultWithIntermediates};
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::GraphNetwork;

use super::super::super::tensor_ext::BoundedTensorExt;
use super::super::super::BetaCrownVerifier;

impl BetaCrownVerifier {
    /// Propagate CROWN bounds with Lagrangian β contribution.
    ///
    /// This extends `propagate_crown_with_graph_constraints` to include β parameters
    /// as Lagrangian multipliers for split constraints. The β contribution tightens
    /// the lower bound based on constraint satisfaction.
    ///
    /// For each constraint (node_name, neuron_idx, is_active):
    /// - Active constraint (x >= 0): contribution = β * x_lower
    /// - Inactive constraint (x <= 0): contribution = -β * x_upper
    ///
    /// This is a Lagrangian relaxation: we add β * (constraint_slack) to the lower bound.
    pub(crate) fn propagate_crown_with_graph_beta(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        objective: Option<&[f32]>,
    ) -> Result<DomainCrownResult> {
        // Compute bounds with β integrated into the backward pass.
        // The β contribution is now added to linear coefficients during the ReLU
        // backward pass, rather than as a scalar offset to final bounds.
        // This is the correct Lagrangian β-CROWN formulation.
        self.propagate_crown_with_graph_constraints(
            graph,
            input,
            context,
            Some(beta_state),
            objective,
        )
    }

    // Justification: graph beta propagation combines graph/input/context with
    // optional objective, warm-start cache, and cache-capture controls.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_with_graph_beta_and_cache(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        objective: Option<&[f32]>,
        seed_cache: Option<&CachedLinearBounds>,
        capture_linear_bounds: bool,
    ) -> Result<(
        BoundedTensor,
        std::collections::HashMap<String, Arc<BoundedTensor>>,
        Option<CachedLinearBounds>,
    )> {
        self.propagate_crown_with_graph_constraints_with_cache(
            graph,
            input,
            context,
            Some(beta_state),
            objective,
            seed_cache,
            capture_linear_bounds,
        )
    }

    /// Batched spec-guided CROWN backward with β constraints (#4306).
    ///
    /// Like `propagate_crown_with_graph_beta_and_cache` but seeds the backward
    /// pass with a multi-row spec matrix instead of a single objective row.
    /// This eliminates the per-objective loop in multi-objective BaB — a single
    /// call replaces N separate CROWN backward passes.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_with_graph_beta_and_spec_matrix(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        spec_matrix: &ndarray::Array2<f32>,
        seed_cache: Option<&CachedLinearBounds>,
        capture_linear_bounds: bool,
    ) -> Result<(
        BoundedTensor,
        std::collections::HashMap<String, Arc<BoundedTensor>>,
        Option<CachedLinearBounds>,
    )> {
        self.propagate_crown_with_graph_constraints_with_spec_matrix(
            graph,
            input,
            context,
            Some(beta_state),
            spec_matrix,
            seed_cache,
            capture_linear_bounds,
        )
    }

    /// Batched spec-guided CROWN backward with intermediate capture for beta gradients.
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_with_graph_beta_and_spec_matrix_storing_intermediates(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        spec_matrix: &ndarray::Array2<f32>,
    ) -> Result<DomainCrownResultWithIntermediates> {
        self.propagate_crown_with_graph_constraints_storing_intermediates_with_spec_matrix(
            graph,
            input,
            context,
            Some(beta_state),
            spec_matrix,
        )
    }

    /// Optimize β parameters using SPSA (Simultaneous Perturbation Stochastic Approximation).
    ///
    /// This is the graph network equivalent of the sequential β optimization loop.
    /// Since analytical gradients are complex for DAGs with skip connections, we use
    /// SPSA to estimate gradients efficiently.
    ///
    /// Returns the optimized bounds and updated beta_state.
    pub(crate) fn optimize_graph_beta_spsa(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        objective: &[f32],
    ) -> Result<(
        f32,
        f32,
        std::collections::HashMap<String, Arc<BoundedTensor>>,
    )> {
        use rand::RngExt;

        // Skip if no beta parameters or iterations disabled
        if beta_state.is_empty() || self.config.beta_iterations == 0 {
            let (output, node_bounds) = self.propagate_crown_with_graph_beta(
                graph,
                input,
                context,
                beta_state,
                Some(objective),
            )?;
            let lb = output.lower_scalar();
            let ub = output.upper_scalar();
            return Ok((lb, ub, node_bounds));
        }

        let mut rng = crate::random::rng();
        let eps = 1e-3f32; // Perturbation magnitude
        let num_samples = 1; // SPSA samples per iteration (1 is often sufficient)

        let mut best_lower = f32::NEG_INFINITY;
        let mut best_upper = f32::INFINITY;
        let mut best_node_bounds = std::collections::HashMap::new();

        // Store beta values for each iteration (updated after Adam step)
        let mut current_betas: Vec<f32> = beta_state.entries.iter().map(|e| e.value).collect();

        for iter in 0..self.config.beta_iterations {
            // Deadline check (#3109): bail early if verification timeout budget
            // is exhausted. Return current best bounds instead of running all iterations.
            if self.config.alpha_config.past_deadline() {
                info!(
                    "Graph β-SPSA: deadline exceeded at iteration {}/{}, returning best bounds",
                    iter, self.config.beta_iterations
                );
                break;
            }

            // Reset gradients
            beta_state.zero_grad();

            // Compute bounds with current β
            let (output, node_bounds) = self.propagate_crown_with_graph_beta(
                graph,
                input,
                context,
                beta_state,
                Some(objective),
            )?;
            let lb = output.lower_scalar();
            let ub = output.upper_scalar();

            // Track best bounds (highest finite lower bound).
            // Non-finite values (NaN/+Inf/-Inf) must not become best_lower (#2695).
            if lb.is_finite() && lb > best_lower {
                best_lower = lb;
                best_upper = ub;
                best_node_bounds = node_bounds.clone();
            } else if !lb.is_finite() {
                trace!(
                    "Graph β-SPSA: skipping non-finite lower bound at iteration {}: {}",
                    iter,
                    lb
                );
            }

            // Early exit if already verified (lower bound > 0 for minimization problems)
            // This check is domain-specific; caller will check against threshold

            // SPSA gradient estimation
            for _sample in 0..num_samples {
                // Generate Bernoulli perturbation (+1 or -1) for each β
                let perturbations: Vec<f32> = (0..beta_state.entries.len())
                    .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
                    .collect();

                // Create perturbed states
                // +ε perturbation
                for (i, entry) in beta_state.entries.iter_mut().enumerate() {
                    entry.set_value((current_betas[i] + eps * perturbations[i]).max(0.0));
                }
                let (output_plus, _) = self.propagate_crown_with_graph_beta(
                    graph,
                    input,
                    context,
                    beta_state,
                    Some(objective),
                )?;
                let lb_plus = output_plus.lower_scalar();

                // -ε perturbation
                for (i, entry) in beta_state.entries.iter_mut().enumerate() {
                    entry.set_value((current_betas[i] - eps * perturbations[i]).max(0.0));
                }
                let (output_minus, _) = self.propagate_crown_with_graph_beta(
                    graph,
                    input,
                    context,
                    beta_state,
                    Some(objective),
                )?;
                let lb_minus = output_minus.lower_scalar();

                // Restore current values
                for (i, entry) in beta_state.entries.iter_mut().enumerate() {
                    entry.set_value(current_betas[i]);
                }

                // SPSA gradient estimate: g_i = (f+ - f-) / (2 * eps * Δ_i)
                let diff = lb_plus - lb_minus;
                for (i, entry) in beta_state.entries.iter_mut().enumerate() {
                    entry.grad += diff / (2.0 * eps * perturbations[i]) / (num_samples as f32);
                }
            }

            // Adam gradient step
            let t = iter + 1;
            let max_grad = beta_state.gradient_step_adam(&self.config.adaptive_config, t);

            // Update current_betas for next iteration (after Adam step updated values)
            for (i, entry) in beta_state.entries.iter().enumerate() {
                current_betas[i] = entry.value();
            }

            // Check convergence (NaN fails < comparison, so loop continues —
            // gradient_step_adam resets NaN params to 0 for recovery).
            if max_grad < self.config.beta_tolerance {
                trace!(
                    "Graph β-SPSA converged at iteration {} (max_grad={:.6})",
                    iter,
                    max_grad
                );
                break;
            }
        }

        // Compute final bounds with optimized β
        let (output, node_bounds) = self.propagate_crown_with_graph_beta(
            graph,
            input,
            context,
            beta_state,
            Some(objective),
        )?;
        let lb = output.lower_scalar();
        let ub = output.upper_scalar();

        // Return the better of final vs best-seen bounds.
        // Reject non-finite final lb to prevent +Inf from winning (#2695).
        if lb.is_finite() && lb >= best_lower {
            Ok((lb, ub, node_bounds))
        } else {
            Ok((best_lower, best_upper, best_node_bounds))
        }
    }
    /// Optimize β and α parameters using analytical gradients for DAG networks.
    ///
    /// This is the efficient alternative to SPSA. Instead of finite-difference gradient
    /// estimation (which requires 3 propagation passes per iteration), this computes
    /// analytical gradients from the A matrices stored during a single backward pass.
    ///
    /// For each constraint at (node_name, neuron_idx, sign), the β gradient is:
    ///   ∂lb/∂β = -sign * sensitivity
    ///
    /// where sensitivity is derived from the A matrix at that ReLU layer.
    ///
    /// When `alpha_state` is non-empty, α gradients are captured from the ReLU backward
    /// pass (`propagate_linear_with_alpha` returns ∂lb/∂α per neuron) and jointly
    /// optimized alongside β using Adam.
    ///
    /// Returns the optimized bounds and updated beta_state (alpha_state is updated in place).
    /// Issue: #1841
    pub(crate) fn optimize_graph_beta_analytical(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        alpha_state: &mut GraphDomainAlphaState,
        objective: &[f32],
    ) -> Result<(
        f32,
        f32,
        std::collections::HashMap<String, Arc<BoundedTensor>>,
    )> {
        let has_alpha = !alpha_state.is_empty();

        // Skip if no beta parameters or iterations disabled
        if beta_state.is_empty() || self.config.beta_iterations == 0 {
            // Still use alpha for a single pass if available
            let ctx = if has_alpha {
                GraphCrownContext {
                    alpha_state: Some(alpha_state),
                    ..*context
                }
            } else {
                GraphCrownContext {
                    alpha_state: None,
                    ..*context
                }
            };
            let (output, node_bounds) = self.propagate_crown_with_graph_beta(
                graph,
                input,
                &ctx,
                beta_state,
                Some(objective),
            )?;
            let lb = output.lower_scalar();
            let ub = output.upper_scalar();
            return Ok((lb, ub, node_bounds));
        }

        let mut best_lower = f32::NEG_INFINITY;
        let mut best_upper = f32::INFINITY;
        let mut best_node_bounds = std::collections::HashMap::new();

        for iter in 0..self.config.beta_iterations {
            // Deadline check (#3109): bail early if verification timeout budget
            // is exhausted. Return current best bounds instead of running all iterations.
            if self.config.alpha_config.past_deadline() {
                info!(
                    "Graph αβ-analytical: deadline exceeded at iteration {}/{}, returning best bounds",
                    iter, self.config.beta_iterations
                );
                break;
            }

            // Reset gradients
            beta_state.zero_grad();
            if has_alpha {
                alpha_state.zero_grad();
            }

            // Build context with alpha for this iteration
            let ctx = if has_alpha {
                GraphCrownContext {
                    alpha_state: Some(alpha_state),
                    ..*context
                }
            } else {
                GraphCrownContext {
                    alpha_state: None,
                    ..*context
                }
            };

            // Compute bounds with current β (and α) AND get intermediate A matrices.
            // When alpha is present, the backward pass captures lower + upper
            // α gradients in the intermediate via propagate_linear_with_alpha.
            let (output, node_bounds, intermediate) = self
                .propagate_crown_with_graph_beta_and_intermediates(
                    graph,
                    input,
                    &ctx,
                    beta_state,
                    Some(objective),
                )?;

            let lb = output.lower_scalar();
            let ub = output.upper_scalar();

            // Track best bounds (highest finite lower bound).
            // Non-finite values (NaN/+Inf/-Inf) must not become best_lower (#2695).
            if lb.is_finite() && lb > best_lower {
                best_lower = lb;
                best_upper = ub;
                best_node_bounds = node_bounds.clone();
            } else if !lb.is_finite() {
                trace!(
                    "Graph αβ-analytical: skipping non-finite lower bound at iteration {}: {}",
                    iter,
                    lb
                );
            }

            // Compute analytical β gradients from A matrices
            beta_state.compute_analytical_gradients(&intermediate);

            // Accumulate α gradients from the backward pass.
            if has_alpha {
                for (node_name, grad_array) in &intermediate.alpha_gradients {
                    for (neuron_idx, &grad_val) in grad_array.iter().enumerate() {
                        if grad_val != 0.0 {
                            alpha_state.accumulate_grad(node_name, neuron_idx, grad_val);
                        }
                    }
                }
                for (node_name, grad_array) in &intermediate.alpha_gradients_upper {
                    for (neuron_idx, &grad_val) in grad_array.iter().enumerate() {
                        if grad_val != 0.0 {
                            alpha_state.accumulate_grad_upper(node_name, neuron_idx, grad_val);
                        }
                    }
                }
            }

            // Adam gradient step for β
            let t = iter + 1;
            let max_beta_grad = beta_state.gradient_step_adam(&self.config.adaptive_config, t);

            // Adam gradient step for α (joint optimization)
            let max_alpha_grad = if has_alpha {
                alpha_state.gradient_step_adam(&self.config.adaptive_config, t)
            } else {
                0.0
            };

            let max_grad = nan_propagating_max(max_beta_grad, max_alpha_grad); // #2939

            // Check convergence (NaN fails < comparison, so loop continues).
            if max_grad < self.config.beta_tolerance {
                trace!(
                    "Graph αβ-analytical converged at iteration {} (max_grad={:.6}, β={:.6}, α={:.6})",
                    iter,
                    max_grad,
                    max_beta_grad,
                    max_alpha_grad,
                );
                break;
            }
        }

        // Compute final bounds with optimized β and α
        let ctx = if has_alpha {
            GraphCrownContext {
                alpha_state: Some(alpha_state),
                ..*context
            }
        } else {
            GraphCrownContext {
                alpha_state: None,
                ..*context
            }
        };
        let (output, node_bounds) =
            self.propagate_crown_with_graph_beta(graph, input, &ctx, beta_state, Some(objective))?;
        let lb = output.lower_scalar();
        let ub = output.upper_scalar();

        // Return the better of final vs best-seen bounds.
        // Reject non-finite final lb to prevent +Inf from winning (#2695).
        if lb.is_finite() && lb >= best_lower {
            Ok((lb, ub, node_bounds))
        } else {
            Ok((best_lower, best_upper, best_node_bounds))
        }
    }

    /// Optimize β/α on a fully-decided (no-unstable-neuron) leaf and return the
    /// tightest *verification-direction* bound seen across the optimization.
    ///
    /// `optimize_graph_beta_analytical` maximizes the **lower** bound and returns
    /// the upper bound paired with the best lower bound. That is wrong for a leaf
    /// verified by its **upper** bound: as β grows to enforce the split
    /// constraints, the upper bound of the objective can fall below the threshold
    /// while the lower bound moves the other way, so the optimizer's
    /// best-lower-paired upper bound is *looser* than the upper bound actually
    /// achieved mid-optimization (#1896). This routine runs the same analytical
    /// β/α gradient loop but keeps the tightest bound in the direction that
    /// decides verification:
    /// - `verify_upper_bound == true`: keep the minimum sound upper bound.
    /// - otherwise: keep the maximum sound lower bound.
    ///
    /// # Soundness
    /// Every per-iteration `(lb, ub)` is an independently sound CROWN-β(-α) bound
    /// for this subproblem. Returning the tightest one in a single direction is
    /// sound — the tighter side of a valid interval is still valid. Non-finite
    /// iterates are ignored so NaN/Inf can never win. `iterations` and the Adam
    /// gradient steps only *move* β/α; they never relax the reported bound below
    /// what a sound CROWN pass produced.
    ///
    /// Returns `(best_lower, best_upper)` where the non-decision direction carries
    /// the value paired with the chosen best-direction iterate (a sound bound).
    /// Issue: #1896
    pub(crate) fn optimize_graph_leaf_beta_directional(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &mut GraphBetaState,
        alpha_state: &mut GraphDomainAlphaState,
        objective: &[f32],
        iterations: usize,
        verify_upper_bound: bool,
    ) -> Result<(f32, f32)> {
        let has_alpha = !alpha_state.is_empty();

        // Single-pass baseline (also the answer when there is nothing to optimize).
        let single_pass = |beta: &GraphBetaState,
                           alpha: &GraphDomainAlphaState|
         -> Result<(f32, f32)> {
            let ctx = if has_alpha {
                GraphCrownContext {
                    alpha_state: Some(alpha),
                    ..*context
                }
            } else {
                GraphCrownContext {
                    alpha_state: None,
                    ..*context
                }
            };
            let (output, _nb) =
                self.propagate_crown_with_graph_beta(graph, input, &ctx, beta, Some(objective))?;
            Ok((output.lower_scalar(), output.upper_scalar()))
        };

        let (base_l, base_u) = single_pass(beta_state, alpha_state)?;
        let mut best_lower = base_l;
        let mut best_upper = base_u;

        if beta_state.is_empty() || iterations == 0 {
            return Ok((best_lower, best_upper));
        }

        // Track the best bound in the verification direction. The paired
        // off-direction value is carried from the same (sound) iterate.
        let mut best_decision = if verify_upper_bound { base_u } else { base_l };

        for iter in 0..iterations {
            if self.config.alpha_config.past_deadline() {
                break;
            }

            beta_state.zero_grad();
            if has_alpha {
                alpha_state.zero_grad();
            }

            let ctx = if has_alpha {
                GraphCrownContext {
                    alpha_state: Some(alpha_state),
                    ..*context
                }
            } else {
                GraphCrownContext {
                    alpha_state: None,
                    ..*context
                }
            };

            let (output, _node_bounds, intermediate) = self
                .propagate_crown_with_graph_beta_and_intermediates(
                    graph,
                    input,
                    &ctx,
                    beta_state,
                    Some(objective),
                )?;

            let lb = output.lower_scalar();
            let ub = output.upper_scalar();

            // Keep the tightest sound bound in the verification direction.
            if verify_upper_bound {
                if ub.is_finite() && ub < best_decision {
                    best_decision = ub;
                    best_upper = ub;
                    if lb.is_finite() {
                        best_lower = lb;
                    }
                }
            } else if lb.is_finite() && lb > best_decision {
                best_decision = lb;
                best_lower = lb;
                if ub.is_finite() {
                    best_upper = ub;
                }
            }

            // Analytical β/α gradient step (maximizes the lower bound; for the
            // active/inactive split constraints this drives β up, which also
            // pulls the upper bound toward the constrained sub-region).
            beta_state.compute_analytical_gradients(&intermediate);
            if has_alpha {
                for (node_name, grad_array) in &intermediate.alpha_gradients {
                    for (neuron_idx, &grad_val) in grad_array.iter().enumerate() {
                        if grad_val != 0.0 {
                            alpha_state.accumulate_grad(node_name, neuron_idx, grad_val);
                        }
                    }
                }
                for (node_name, grad_array) in &intermediate.alpha_gradients_upper {
                    for (neuron_idx, &grad_val) in grad_array.iter().enumerate() {
                        if grad_val != 0.0 {
                            alpha_state.accumulate_grad_upper(node_name, neuron_idx, grad_val);
                        }
                    }
                }
            }

            let t = iter + 1;
            let max_beta_grad = beta_state.gradient_step_adam(&self.config.adaptive_config, t);
            let max_alpha_grad = if has_alpha {
                alpha_state.gradient_step_adam(&self.config.adaptive_config, t)
            } else {
                0.0
            };
            let max_grad = nan_propagating_max(max_beta_grad, max_alpha_grad);
            if max_grad < self.config.beta_tolerance {
                break;
            }
        }

        Ok((best_lower, best_upper))
    }

    /// Propagate CROWN bounds with β and return intermediate A matrices for gradient computation.
    ///
    /// This is like `propagate_crown_with_graph_beta` but also captures the A matrices
    /// at constrained ReLU nodes during the backward pass. These A matrices are used
    /// to compute analytical β gradients.
    pub(crate) fn propagate_crown_with_graph_beta_and_intermediates(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        context: &GraphCrownContext<'_>,
        beta_state: &GraphBetaState,
        objective: Option<&[f32]>,
    ) -> Result<DomainCrownResultWithIntermediates> {
        self.propagate_crown_with_graph_constraints_storing_intermediates(
            graph,
            input,
            context,
            Some(beta_state),
            objective,
        )
    }

    /// Evaluate bounds for a graph BaB child domain using the unified optimization policy.
    ///
    /// This is the shared helper for child-domain tightening used by both
    /// `verify_graph_relu_split` and `process_graph_domain_parallel` (GPU BaB
    /// CPU fallback). It ensures consistent optimization semantics across all
    /// graph BaB paths.
    ///
    /// Policy:
    /// - If `beta_iterations > 0` and `child.depth <= beta_max_depth`: run beta
    ///   optimization (analytical or SPSA depending on config).
    /// - Otherwise: single CROWN pass with inherited beta state.
    ///
    /// Updates `child.{lower_bound, upper_bound, priority, node_bounds}` in place.
    /// Returns `Ok(true)` if bounds were computed, `Err` if propagation failed.
    /// Callers MUST treat `Err` as an unresolved domain — the sub-region of
    /// input space covered by `child` was not explored (#1861).
    ///
    /// Design: designs/2026-02-09-gpu-bab-beta-optimization-parity.md
    /// Issue: #1817
    pub(crate) fn evaluate_graph_child_bounds(
        &self,
        graph: &GraphNetwork,
        child: &mut GraphBabDomain,
        parent_node_bounds: &std::collections::HashMap<String, Arc<BoundedTensor>>,
        objective: &[f32],
        cut_pool: Option<&GraphCutPool>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<bool> {
        // Lazy alpha initialization: if alpha_state is empty (e.g., domain came
        // from GPU BaB's branch_relu_from_picked or graph_domain_from_picked which
        // don't have access to the graph for initialization), initialize it now
        // from the child's node bounds and history.  Issue: #1841
        if child.alpha_state.is_empty() {
            child.alpha_state = GraphDomainAlphaState::from_graph_bounds(
                graph,
                &child.node_bounds,
                &child.history,
                child.input_bounds.as_ref(),
            );
        }

        // Base context without alpha — analytical path manages alpha internally
        // to avoid borrow conflicts (it needs &mut alpha_state alongside &context).
        // #cone-delta: the child inherited `node_bounds` verbatim from the
        // parent whose map is passed as base, so its delta describes exactly
        // that map (dark, NY_CONE_REFRESH-gated).
        let context =
            GraphCrownContext::new(&child.history, cut_pool, Some(parent_node_bounds), engine)
                .with_delta_seeds(&child.delta_pre_nodes);

        let should_optimize =
            self.config.beta_iterations > 0 && child.depth <= self.config.beta_max_depth;

        debug!(
            depth = child.depth,
            should_optimize,
            alpha_neurons = child.alpha_state.len(),
            method = if should_optimize && self.config.use_analytical_beta_gradients {
                "analytical+alpha"
            } else if should_optimize {
                "spsa"
            } else {
                "inherited"
            },
            "[#1817/#1841] child optimization policy"
        );

        let beta_result = if should_optimize && self.config.use_analytical_beta_gradients {
            // Analytical path: optimize_graph_beta_analytical handles alpha
            // internally — it constructs alpha-enriched contexts for each
            // iteration and accepts &mut alpha_state for gradient updates.
            self.optimize_graph_beta_analytical(
                graph,
                child.input_bounds.as_ref(),
                &context,
                &mut child.beta_state,
                &mut child.alpha_state,
                objective,
            )
        } else if should_optimize {
            // SPSA path: use alpha in context (read-only, no optimization)
            let ctx_with_alpha = if !child.alpha_state.is_empty() {
                GraphCrownContext {
                    alpha_state: Some(&child.alpha_state),
                    ..context
                }
            } else {
                context
            };
            self.optimize_graph_beta_spsa(
                graph,
                child.input_bounds.as_ref(),
                &ctx_with_alpha,
                &mut child.beta_state,
                objective,
            )
        } else if let Some((l, u, cache)) =
            self.try_gpu_beta_perdomain_bound(graph, child, parent_node_bounds, objective, engine)
        {
            // #unsat-keystone step 4: GPU beta-capable resnet per-domain bound (gated,
            // sound, CPU fallback) — replaces the ~60 s/domain CPU dense backward.
            Ok((l, u, cache))
        } else {
            // No optimization — single pass with inherited β and α
            let ctx_with_alpha = if !child.alpha_state.is_empty() {
                GraphCrownContext {
                    alpha_state: Some(&child.alpha_state),
                    ..context
                }
            } else {
                context
            };
            self.propagate_crown_with_graph_beta(
                graph,
                child.input_bounds.as_ref(),
                &ctx_with_alpha,
                &child.beta_state,
                Some(objective),
            )
            .map(|(out, cache)| (out.lower_scalar(), out.upper_scalar(), cache))
        };

        match beta_result {
            Ok((l, u, node_cache)) => {
                // Guard: non-finite bounds from lower_scalar/upper_scalar indicate
                // NaN/Inf propagation through CROWN backward. Reject early to prevent
                // zombie domains from entering the BaB queue. Matches GPU BaB guard
                // in batched_gpu.rs and sequential BaB guard in core.rs. (#2986)
                if !l.is_finite() || !u.is_finite() {
                    warn!(
                        "evaluate_graph_child_bounds: rejected — non-finite bounds \
                         (lb={}, ub={}, depth={})",
                        l, u, child.depth
                    );
                    return Ok(false);
                }

                // Some propagation paths return a partial node cache. Preserve
                // parent bounds for missing nodes so unstable-neuron discovery
                // does not silently lose branch candidates.
                // #cone-delta increment 2: both maps are Arc-shared — the
                // merge moves Arcs, no tensor copies.
                let mut merged_bounds = parent_node_bounds.clone();
                for (name, bounds) in node_cache {
                    merged_bounds.insert(name, bounds);
                }
                child.node_bounds = merged_bounds;
                // #cone-delta: `node_bounds` is now the post-bounding fixpoint
                // for the child's full history — the delta restarts empty.
                child.delta_pre_nodes.clear();
                child.lower_bound = l;
                child.upper_bound = u;
                child.priority = self.config.domain_priority(l, u)?;
                Ok(true)
            }
            Err(e) => {
                // #1861: Propagate the error instead of swallowing it as Ok(false).
                // Callers must track this as an unresolved domain so the verifier
                // never claims Verified when part of the input space is unexplored.
                debug!("[#1817/#1861] child propagation failed: {e}");
                Err(e)
            }
        }
    }

    /// GPU beta-capable per-domain bound (#unsat-keystone step 4). The BaB per-domain
    /// bound is the cifar100/tinyimagenet UNSAT wall (~60 s/domain on the CPU dense
    /// backward); this routes it through the sound GPU-resident resnet backward with the
    /// β-CROWN split-constraint dual folded in (~180 ms). Returns `Some((lower, upper,
    /// node_bounds))` when the gated GPU path applies, else `None` → the caller falls back
    /// to the proven CPU β-CROWN path.
    ///
    /// SOUND + GATED: the GPU resnet backward is a sound enclosure (directed/over-bounded
    /// f32 error), and a β-CROWN bound is a valid Lagrangian dual for ANY β≥0, so the
    /// bound is sound regardless of the β values; default ON (opt out
    /// `NY_RESNET_BETA_GPU=0`) and any miss/Err falls back to CPU — the 0-wrong moat is
    /// preserved.
    fn try_gpu_beta_perdomain_bound(
        &self,
        graph: &GraphNetwork,
        child: &GraphBabDomain,
        parent_node_bounds: &std::collections::HashMap<String, Arc<BoundedTensor>>,
        objective: &[f32],
        engine: Option<&dyn GemmEngine>,
    ) -> Option<(
        f32,
        f32,
        std::collections::HashMap<String, Arc<BoundedTensor>>,
    )> {
        if !crate::network::resnet_beta_gpu_enabled() {
            return None;
        }
        let probe = std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1");
        let gpu = engine
            .and_then(|e| e.as_gpu_crown_backward())
            .filter(|g| g.provides_sound_gpu_crown())?;
        // ReLU-only splits: the additive ±β term implements the ReLU (split_point=0)
        // dual. GenBaB / non-zero split points need different semantics → CPU.
        if !child.history.genbab_constraints.is_empty() {
            return None;
        }
        // Conv resnet only — the GPU resnet decomposition path. FC/other → CPU.
        if !graph
            .nodes
            .values()
            .any(|n| matches!(n.layer, crate::layers::Layer::Conv2d(_)))
        {
            return None;
        }
        let od = objective.len();
        if od == 0 || od > 512 {
            return None;
        }
        // Constrained forward bounds (the domain's ReLU splits as tightened pre-acts).
        // #cone-delta: the child's delta describes exactly `parent_node_bounds`
        // (inherited verbatim at split time); dark, NY_CONE_REFRESH-gated.
        let (bounds_cache, constrained_input) = self
            .compute_constrained_forward_bounds(
                graph,
                child.input_bounds.as_ref(),
                &child.history,
                Some(parent_node_bounds),
                Some(&child.delta_pre_nodes),
            )
            .ok()?;
        // Decompose the output suffix with alpha=None: default ReLU slopes derived from
        // the CONSTRAINED bounds already reflect the splits; the β dual enforces them too.
        let (segments, relu_names, frontier_abs, node_abs) =
            crate::network::extract_gpu_resnet_segments_with_relu_names(
                graph,
                &constrained_input,
                &graph.output_node,
                &bounds_cache,
                &bounds_cache,
                None,
            )?;
        // Per-ReLU signed beta (β·sign) in FOLD order; 0 for non-split neurons.
        let mut beta_signed: Vec<Vec<f32>> = Vec::with_capacity(relu_names.len());
        for name in &relu_names {
            let nn = bounds_cache.get(name)?.lower().len();
            let mut bs = vec![0.0f32; nn];
            for entry in child.beta_state.entries_for_node(name) {
                if entry.split_point().abs() < 1e-6 {
                    let idx = entry.neuron_idx();
                    if idx < nn {
                        bs[idx] = entry.signed_value();
                    }
                }
            }
            beta_signed.push(bs);
        }
        // Objective seed (1 × output_dim), zero bias.
        let seed = ny_core::GpuCrownSeed {
            lower_a: objective.to_vec().into(),
            upper_a: objective.to_vec().into(),
            lower_b: vec![0.0f32].into(),
            upper_b: vec![0.0f32].into(),
            num_specs: 1,
            current_dim: od,
        };
        let in_lo: Vec<f32> = constrained_input.lower().iter().copied().collect();
        let in_hi: Vec<f32> = constrained_input.upper().iter().copied().collect();
        let result = gpu
            .crown_backward_gpu_resnet_sound_beta(
                &segments,
                &seed,
                &in_lo,
                &in_hi,
                &beta_signed,
                &frontier_abs,
                &node_abs,
            )
            .ok()?;
        let l = *result.lower_bounds.first()?;
        let u = *result.upper_bounds.first()?;
        if !l.is_finite() || !u.is_finite() {
            return None;
        }
        if probe {
            eprintln!(
                "[beta-gpu] SUCCESS relus={} num_specs(obj)=1 od={od} l={l:.4} u={u:.4}",
                relu_names.len()
            );
        }
        Some((l, u, bounds_cache))
    }
}
