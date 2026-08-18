// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Joint α/β/λ optimization loops.

use std::sync::Arc;

use ny_core::{nan_propagating_max, GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::{info, trace};

use super::super::tensor_ext::{has_inverted_output_bounds, BoundedTensorExt};
use super::super::BetaCrownVerifier;
use super::intermediate_merge::accumulate_tightest_intermediate_bounds;
use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::branching::SplitHistory;
use crate::beta_crown::domain::IntermediateLinearBounds;
use crate::beta_crown::state::{BetaState, DomainAlphaState};
use crate::Network;

/// Track the tightest finite output bounds seen so far across optimization iterations.
///
/// Returns the current iteration's scalar lower bound so callers can keep their
/// existing convergence/logging behavior while unit tests exercise the "best, not
/// last" selection logic directly.
pub(crate) fn update_best_output_bounds(
    best_lower: &mut f32,
    best_bounds: &mut Option<BoundedTensor>,
    bounds: &BoundedTensor,
) -> f32 {
    let current_lower = bounds.lower_scalar();
    if current_lower.is_finite() && current_lower > *best_lower {
        *best_lower = current_lower;
        *best_bounds = Some(bounds.clone());
    }
    current_lower
}

/// Update the "no improvement" patience counter and report whether to stop.
///
/// Mirrors alpha-beta-CROWN's `patience = 0 if need_update else patience + 1`
/// followed by `if patience > early_stop_patience: break`
/// (`optimized_bounds.py:781,838`).
pub(crate) fn patience_exhausted_after_iteration(
    previous_best_lower: f32,
    best_lower: f32,
    no_improve_iters: &mut usize,
    early_stop_patience: usize,
) -> bool {
    if best_lower > previous_best_lower {
        *no_improve_iters = 0;
    } else {
        *no_improve_iters = no_improve_iters.saturating_add(1);
    }
    *no_improve_iters > early_stop_patience
}

impl BetaCrownVerifier {
    /// Jointly optimize α, β, and λ (cut) parameters to tighten output bounds.
    ///
    /// This performs alternating optimization:
    /// 1. Compute bounds with current α, β, and λ values
    /// 2. Compute gradients of lower bound w.r.t. α, β, and λ
    /// 3. Update α via gradient ascent: α = clamp(α + lr_α * grad_α, 0, 1)
    /// 4. Update β via projected gradient ascent: β = max(0, β + lr_β * grad_β)
    /// 5. Update λ via projected gradient ascent: λ = max(0, λ + lr_λ * grad_λ)
    /// 6. Repeat until convergence or max iterations
    ///
    /// Returns:
    /// - Output bounds (concrete BoundedTensor)
    /// - Best intermediate linear bounds accumulated across finite iterations
    ///   (for transfer to children)
    // Justification: Joint optimization needs network, input, split history, layer bounds,
    // mutable alpha/beta/cut state, and engine — each from a different BaB tree source.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn optimize_joint_bounds(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &mut BetaState,
        alpha_state: &mut DomainAlphaState,
        cut_pool: &mut CutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(BoundedTensor, IntermediateLinearBounds)> {
        // Check if any optimization is needed
        let has_beta = !beta_state.is_empty();
        let has_alpha = !alpha_state.is_empty() && self.config.use_alpha_crown;
        let has_cuts = !cut_pool.is_empty() && self.config.enable_cuts;

        // Skip optimization if nothing to optimize.
        // Sound concretization is unconditional since #2239.
        if (!has_beta && !has_alpha && !has_cuts) || self.config.beta_iterations == 0 {
            return self.compute_bounds_capturing_intermediate(
                network,
                input,
                history,
                layer_bounds,
                beta_state,
                alpha_state,
                cut_pool,
                engine,
            );
        }

        let mut best_bounds: Option<BoundedTensor> = None;
        let mut best_intermediate: Option<IntermediateLinearBounds> = None;
        let mut best_lower = f32::NEG_INFINITY;
        let mut no_improve_iters = 0usize;
        let momentum = if self.config.alpha_momentum {
            self.config.alpha_config.momentum
        } else {
            0.0
        };

        for iter in 0..self.config.beta_iterations {
            // Deadline check (#3109): bail early if verification timeout budget
            // is exhausted. Return current best bounds instead of running all iterations.
            if self.past_effective_graph_bab_deadline() {
                info!(
                    "Joint α-β-λ optimization: deadline exceeded at iteration {}/{}, returning best bounds",
                    iter, self.config.beta_iterations
                );
                break;
            }

            // Reset gradients for α, β, and λ (cuts)
            beta_state.zero_grad();
            alpha_state.zero_grad();
            if has_cuts {
                cut_pool.zero_grad();
            }

            // Compute bounds with current α, β, and λ.
            // Intermediate bounds are accumulated across finite iterations so
            // child domains keep row-wise tightening even if the scalar
            // objective peaks on a different step.
            let (bounds, intermediate) = self.compute_bounds_capturing_intermediate(
                network,
                input,
                history,
                layer_bounds,
                beta_state,
                alpha_state,
                cut_pool,
                engine,
            )?;

            // #2950: Check for infeasible output bounds (lower > upper).
            // BaB splits can make a domain infeasible — continuing to optimize
            // is wasted work. Mark with (+inf, -inf) sentinel and return early.
            // Reference: alpha-beta-CROWN optimized_bounds.py:626-644.
            if has_inverted_output_bounds(&bounds) {
                trace!(
                    "Joint optimization detected infeasible bounds at iter {}, early exit",
                    iter
                );
                let mut infeasible = bounds;
                infeasible.mark_infeasible_all();
                return Ok((infeasible, intermediate));
            }

            let previous_best_lower = best_lower;
            let current_lower =
                update_best_output_bounds(&mut best_lower, &mut best_bounds, &bounds);
            let objective_lower_idx = bounds.argmin_lower_flat_idx();
            accumulate_tightest_intermediate_bounds(
                &mut best_intermediate,
                &intermediate,
                current_lower,
                layer_bounds,
            )?;

            // Intermediate bounds are accumulated independently above so we do
            // not discard layer-wise tightening from non-best output iterations.
            if !current_lower.is_finite() {
                trace!(
                    "Skipping non-finite lower bound during joint optimization iter {}: {}",
                    iter,
                    current_lower
                );
            }

            // Compute gradients; skip when all bounds are NaN/+Inf (#2561).
            if let Some(obj_idx) = objective_lower_idx {
                self.compute_joint_gradients(
                    network,
                    input,
                    history,
                    layer_bounds,
                    beta_state,
                    alpha_state,
                    cut_pool,
                    obj_idx,
                )?;
            }

            // Perform gradient ascent steps for α, β, and λ
            // Use Adam optimizer if enabled, otherwise use standard gradient ascent
            let (max_beta_grad, max_alpha_grad, max_cut_grad) = if self.config.use_adaptive {
                // Adam optimizer: adaptive learning rates per parameter
                let t = iter + 1; // Time step (1-indexed for bias correction)
                let beta_grad = if has_beta {
                    beta_state.gradient_step_adam(&self.config.adaptive_config, t)
                } else {
                    0.0
                };
                let alpha_grad = if has_alpha {
                    alpha_state.gradient_step_adam(&self.config.adaptive_config, t)
                } else {
                    0.0
                };
                // GCP-CROWN: optimize cut lambdas
                let cut_grad = if has_cuts {
                    let mut max_grad = 0.0f32;
                    for cut in cut_pool.cuts_mut() {
                        cut.gradient_step_adam(&self.config.adaptive_config, t);
                        // NaN-aware: f32::max silently drops NaN (#2939).
                        max_grad = nan_propagating_max(max_grad, cut.lambda_grad().abs());
                    }
                    max_grad
                } else {
                    0.0
                };
                (beta_grad, alpha_grad, cut_grad)
            } else {
                // Standard gradient ascent with fixed learning rates
                let beta_grad = if has_beta {
                    beta_state.gradient_step(self.config.beta_lr)
                } else {
                    0.0
                };
                let alpha_grad = if has_alpha {
                    alpha_state.gradient_step(self.config.alpha_lr, momentum)
                } else {
                    0.0
                };
                // GCP-CROWN: optimize cut lambdas with fixed learning rate
                let cut_grad = if has_cuts {
                    let lr = self
                        .config
                        .adaptive_config
                        .lr_lambda
                        .unwrap_or(self.config.beta_lr);
                    let mut max_grad = 0.0f32;
                    for cut in cut_pool.cuts_mut() {
                        // NaN-aware: f32::max silently drops NaN (#2939).
                        let grad_abs = cut.gradient_step_sgd(lr);
                        max_grad = nan_propagating_max(max_grad, grad_abs);
                    }
                    max_grad
                } else {
                    0.0
                };
                (beta_grad, alpha_grad, cut_grad)
            };

            // NaN-aware aggregation: f32::max silently drops NaN, which would hide
            // NaN gradients from the convergence check and allow premature termination.
            // After NaN recovery (in gradient_step), the loop should continue to let
            // the reset parameters re-optimize. (#2939)
            let max_grad = nan_propagating_max(
                nan_propagating_max(max_beta_grad, max_alpha_grad),
                max_cut_grad,
            );

            // Check convergence
            // NaN max_grad will fail the < comparison (NaN < x is always false),
            // so the loop correctly continues when NaN gradients were present.
            if max_grad < self.config.beta_tolerance {
                trace!(
                    "Joint α-β-λ optimization converged at iteration {} (max_grad={:.6}, adaptive={})",
                    iter, max_grad, self.config.use_adaptive
                );
                break;
            }

            if patience_exhausted_after_iteration(
                previous_best_lower,
                best_lower,
                &mut no_improve_iters,
                self.config.early_stop_patience,
            ) {
                trace!(
                    "Joint α-β-λ optimization early stop at iteration {} after {} stalled iterations",
                    iter,
                    self.config.early_stop_patience
                );
                break;
            }

            trace!(
                "Joint opt iter {}: lb={:.4}, max_α_grad={:.6}, max_β_grad={:.6}, max_λ_grad={:.6}, adaptive={}",
                iter,
                current_lower,
                max_alpha_grad,
                max_beta_grad,
                max_cut_grad,
                self.config.use_adaptive
            );
        }

        // Return best bounds and intermediate bounds found
        let bounds = best_bounds.ok_or_else(|| {
            ny_core::NyError::NumericalInstability(
                "No valid bounds computed during joint optimization".into(),
            )
        })?;
        let intermediate = best_intermediate.unwrap_or_else(IntermediateLinearBounds::empty);
        Ok((bounds, intermediate))
    }

    /// Jointly optimize α, β, and λ (cut) parameters starting from a given layer.
    ///
    /// This is the optimized version that uses intermediate bound transfer:
    /// instead of running a full backward pass from the output layer, it starts
    /// from `start_layer` using the parent domain's intermediate bounds.
    ///
    /// When splitting at layer L, the backward pass from output to layer L+1 is
    /// unchanged, so we can reuse the parent's intermediate bounds for those layers.
    ///
    /// # Arguments
    /// - `start_layer`: The layer where the split occurred (new constraint was added)
    /// - `parent_intermediate`: Parent domain's intermediate bounds
    ///
    /// # Returns
    /// Same as `optimize_joint_bounds`: (output_bounds, best_intermediate_bounds)
    // Justification: Same as optimize_joint_bounds plus start_layer and parent intermediate
    // bounds for incremental re-computation from a BaB split point.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn optimize_joint_bounds_from_layer(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &mut BetaState,
        alpha_state: &mut DomainAlphaState,
        cut_pool: &mut CutPool,
        start_layer: usize,
        parent_intermediate: &IntermediateLinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(BoundedTensor, IntermediateLinearBounds)> {
        // Check if any optimization is needed
        let has_beta = !beta_state.is_empty();
        let has_alpha = !alpha_state.is_empty() && self.config.use_alpha_crown;
        let has_cuts = !cut_pool.is_empty() && self.config.enable_cuts;

        // Skip optimization if nothing to optimize.
        // Sound concretization is unconditional since #2239.
        if (!has_beta && !has_alpha && !has_cuts) || self.config.beta_iterations == 0 {
            return self.compute_bounds_from_layer(
                network,
                input,
                history,
                layer_bounds,
                beta_state,
                alpha_state,
                cut_pool,
                start_layer,
                parent_intermediate,
                engine,
            );
        }

        let mut best_bounds: Option<BoundedTensor> = None;
        let mut best_intermediate: Option<IntermediateLinearBounds> = None;
        let mut best_lower = f32::NEG_INFINITY;
        let mut no_improve_iters = 0usize;
        let momentum = if self.config.alpha_momentum {
            self.config.alpha_config.momentum
        } else {
            0.0
        };

        for iter in 0..self.config.beta_iterations {
            // Deadline check (#3109): bail early if verification timeout budget
            // is exhausted. Return current best bounds instead of running all iterations.
            if self.past_effective_graph_bab_deadline() {
                info!(
                    "Joint α-β-λ optimization (from layer {}): deadline exceeded at iteration {}/{}, returning best bounds",
                    start_layer, iter, self.config.beta_iterations
                );
                break;
            }

            // Reset gradients for α, β, and λ (cuts)
            beta_state.zero_grad();
            alpha_state.zero_grad();
            if has_cuts {
                cut_pool.zero_grad();
            }

            // Compute bounds starting from the split layer (intermediate bound transfer)
            let (bounds, intermediate) = self.compute_bounds_from_layer(
                network,
                input,
                history,
                layer_bounds,
                beta_state,
                alpha_state,
                cut_pool,
                start_layer,
                parent_intermediate,
                engine,
            )?;

            // #2950: Check for infeasible output bounds (lower > upper).
            // Same early exit as optimize_joint_bounds — see that function for rationale.
            if has_inverted_output_bounds(&bounds) {
                trace!(
                    "Joint optimization (from layer {}) detected infeasible bounds at iter {}, early exit",
                    start_layer, iter
                );
                let mut infeasible = bounds;
                infeasible.mark_infeasible_all();
                return Ok((infeasible, intermediate));
            }

            let previous_best_lower = best_lower;
            let current_lower =
                update_best_output_bounds(&mut best_lower, &mut best_bounds, &bounds);
            let objective_lower_idx = bounds.argmin_lower_flat_idx();
            accumulate_tightest_intermediate_bounds(
                &mut best_intermediate,
                &intermediate,
                current_lower,
                layer_bounds,
            )?;

            // Intermediate bounds are accumulated independently above so we do
            // not discard layer-wise tightening from non-best output iterations.
            if !current_lower.is_finite() {
                trace!(
                    "Skipping non-finite lower bound during joint optimization (from layer {}) iter {}: {}",
                    start_layer, iter, current_lower
                );
            }

            // Compute gradients; skip when all bounds are NaN/+Inf (#2561).
            if let Some(obj_idx) = objective_lower_idx {
                self.compute_joint_gradients(
                    network,
                    input,
                    history,
                    layer_bounds,
                    beta_state,
                    alpha_state,
                    cut_pool,
                    obj_idx,
                )?;
            }

            // Perform gradient ascent steps for α, β, and λ
            // Use Adam optimizer if enabled, otherwise use standard gradient ascent
            let (max_beta_grad, max_alpha_grad, max_cut_grad) = if self.config.use_adaptive {
                // Adam optimizer: adaptive learning rates per parameter
                let t = iter + 1; // Time step (1-indexed for bias correction)
                let beta_grad = if has_beta {
                    beta_state.gradient_step_adam(&self.config.adaptive_config, t)
                } else {
                    0.0
                };
                let alpha_grad = if has_alpha {
                    alpha_state.gradient_step_adam(&self.config.adaptive_config, t)
                } else {
                    0.0
                };
                // GCP-CROWN: optimize cut lambdas
                let cut_grad = if has_cuts {
                    let mut max_grad = 0.0f32;
                    for cut in cut_pool.cuts_mut() {
                        cut.gradient_step_adam(&self.config.adaptive_config, t);
                        // NaN-aware: f32::max silently drops NaN (#2939).
                        max_grad = nan_propagating_max(max_grad, cut.lambda_grad().abs());
                    }
                    max_grad
                } else {
                    0.0
                };
                (beta_grad, alpha_grad, cut_grad)
            } else {
                // Standard gradient ascent with fixed learning rates
                let beta_grad = if has_beta {
                    beta_state.gradient_step(self.config.beta_lr)
                } else {
                    0.0
                };
                let alpha_grad = if has_alpha {
                    alpha_state.gradient_step(self.config.alpha_lr, momentum)
                } else {
                    0.0
                };
                // GCP-CROWN: optimize cut lambdas with fixed learning rate
                let cut_grad = if has_cuts {
                    let lr = self
                        .config
                        .adaptive_config
                        .lr_lambda
                        .unwrap_or(self.config.beta_lr);
                    let mut max_grad = 0.0f32;
                    for cut in cut_pool.cuts_mut() {
                        // NaN-aware: f32::max silently drops NaN (#2939).
                        let grad_abs = cut.gradient_step_sgd(lr);
                        max_grad = nan_propagating_max(max_grad, grad_abs);
                    }
                    max_grad
                } else {
                    0.0
                };
                (beta_grad, alpha_grad, cut_grad)
            };

            // NaN-aware aggregation: f32::max silently drops NaN (#2939).
            let max_grad = nan_propagating_max(
                nan_propagating_max(max_beta_grad, max_alpha_grad),
                max_cut_grad,
            );

            // Check convergence (NaN fails the < comparison, loop continues).
            if max_grad < self.config.beta_tolerance {
                trace!(
                    "Joint α-β-λ optimization (from layer {}) converged at iteration {} (max_grad={:.6})",
                    start_layer, iter, max_grad
                );
                break;
            }

            if patience_exhausted_after_iteration(
                previous_best_lower,
                best_lower,
                &mut no_improve_iters,
                self.config.early_stop_patience,
            ) {
                trace!(
                    "Joint α-β-λ optimization (from layer {}) early stop at iteration {} after {} stalled iterations",
                    start_layer,
                    iter,
                    self.config.early_stop_patience
                );
                break;
            }

            trace!(
                "Joint opt iter {} (from layer {}): lb={:.4}, max_α_grad={:.6}, max_β_grad={:.6}, max_λ_grad={:.6}",
                iter,
                start_layer,
                current_lower,
                max_alpha_grad,
                max_beta_grad,
                max_cut_grad,
            );
        }

        // Return best bounds and intermediate bounds found
        let bounds = best_bounds.ok_or_else(|| {
            ny_core::NyError::NumericalInstability(
                "No valid bounds computed during joint optimization (from layer)".into(),
            )
        })?;
        let intermediate = best_intermediate.unwrap_or_else(IntermediateLinearBounds::empty);
        Ok((bounds, intermediate))
    }
}
