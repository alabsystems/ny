// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Iterative α-CROWN optimization loop with gradient descent.
//!
//! Contains the main optimization loop that iteratively refines alpha parameters
//! via gradient descent, plus the reference bounds refresh helper.

use crate::bounds::{AlphaState, GradientMethod, LinearBounds, Optimizer};
use crate::network::alpha_crown_loop::finite_lower_sum;
use crate::network::core::GraphNetwork;
use crate::network::graph_alpha::alpha_projection::graph_alpha_state_from_sequential;
use crate::network::graph_alpha::invprop_backward::take_best_bounds;
use crate::network::graph_alpha::propagate_helpers::{
    bounds_infeasible, clamp_inverted_best_bounds, update_elementwise_best_bounds,
};
use crate::network::graph_alpha::reference_bounds::GraphAlphaReferenceBounds;
use crate::network::graph_alpha::sequential_gradients;

use ndarray::ArrayD;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::{
    SequentialAlphaOptimizationContext, SequentialBackwardPassContext,
    SequentialBackwardPassRequest,
};

#[cfg(test)]
use super::{SEQUENTIAL_REFERENCE_REFRESH_ATTEMPTS, SEQUENTIAL_REFERENCE_TIGHTENED_TARGETS_TOTAL};

impl GraphNetwork {
    pub(super) fn optimize_sequential_alpha_crown_with_reference_bounds(
        &self,
        context: SequentialAlphaOptimizationContext<'_>,
    ) -> Result<BoundedTensor> {
        let SequentialAlphaOptimizationContext {
            input,
            config,
            engine,
            reference_bounds,
            alpha_state,
            exec_order,
            output_dim,
            relu_name_to_idx,
            invprop_enabled,
            carry_forward_reference_bounds,
        } = context;
        let label = "GraphNetwork α-CROWN";
        #[cfg(test)]
        let mut reference_refresh_attempts = 0usize;
        #[cfg(test)]
        let mut reference_tightened_targets_total = 0usize;
        #[cfg(test)]
        {
            SEQUENTIAL_REFERENCE_REFRESH_ATTEMPTS.with(|slot| slot.set(0));
            SEQUENTIAL_REFERENCE_TIGHTENED_TARGETS_TOTAL.with(|slot| slot.set(0));
        }
        let crown_bounds = self
            .propagate_crown_with_engine_and_deadline(input, engine, config.deadline)
            .map(|result| result.bounds)?;
        let mut best_lower: ArrayD<f32> = crown_bounds.lower().clone();
        let mut best_upper: ArrayD<f32> = crown_bounds.upper().clone();
        let mut best_lower_sum = finite_lower_sum(crown_bounds.lower());
        let mut prev_best_lower_sum = best_lower_sum;
        let mut no_improve_iters = 0usize;
        let mut lr = config.learning_rate;
        let mut infeasible_bounds: Option<BoundedTensor> = None;
        let mut total_gradient_skips = 0usize;

        for iter in 0..config.iterations {
            if config.past_deadline() {
                info!(
                    "{label}: deadline exceeded at iteration {}/{}, returning best bounds",
                    iter, config.iterations
                );
                break;
            }

            let (numerical_gradients, numerical_gradients_upper, lower_sum, refresh_candidate) = {
                let node_bounds = reference_bounds.current();
                let mut bounds_without_oc: Option<LinearBounds> = None;
                let backward_pass = SequentialBackwardPassRequest {
                    context: SequentialBackwardPassContext {
                        input,
                        node_bounds,
                        exec_order,
                        output_dim,
                        relu_name_to_idx,
                        alpha_state,
                        engine,
                    },
                    invprop_config: if invprop_enabled {
                        Some(&config.invprop)
                    } else {
                        None
                    },
                    output_constraints: config.output_constraints.as_ref(),
                    collect_gradients: true,
                    bounds_without_oc: Some(&mut bounds_without_oc),
                };
                let (linear_bounds, gradients, gradients_upper_opt) =
                    match self.sequential_backward_pass(backward_pass) {
                        Ok(result) => result,
                        Err(
                            NyError::UnsupportedOp(_)
                            | NyError::UnsupportedConfiguration(_)
                            | NyError::DeadlineExceeded(_),
                        ) => {
                            return self
                                .propagate_crown_with_engine_and_deadline(
                                    input,
                                    engine,
                                    config.deadline,
                                )
                                .map(|result| result.bounds);
                        }
                        Err(error) => return Err(error),
                    };

                let gradients = gradients.ok_or_else(|| {
                    NyError::InternalError(
                        "Sequential α-CROWN backward pass returned no gradients".to_string(),
                    )
                })?;
                let gradients_upper = gradients_upper_opt.ok_or_else(|| {
                    NyError::InternalError(
                        "Sequential α-CROWN backward pass returned no upper gradients".to_string(),
                    )
                })?;

                let mut concrete_bounds = linear_bounds.concretize_sound(input);
                if let Some(bounds_no_oc) = bounds_without_oc {
                    let no_oc_bounds = bounds_no_oc.concretize_sound(input);
                    concrete_bounds = take_best_bounds(&concrete_bounds, &no_oc_bounds);
                }

                if let Some(ref mut state) = alpha_state.invprop_state {
                    if bounds_infeasible(&concrete_bounds) {
                        state.mark_infeasible(0)?;
                        state.apply_infeasible_mask(&mut concrete_bounds);
                        infeasible_bounds = Some(concrete_bounds);
                        break;
                    }
                }

                // Skip during warmup window to avoid locking in noisy early-iteration bounds.
                // Matches α,β-CROWN's start_save_best (optimized_bounds.py:785-797).
                let is_last_iter = iter == config.iterations - 1;
                if config.should_save_best(iter, is_last_iter) {
                    update_elementwise_best_bounds(
                        &mut best_lower,
                        &mut best_upper,
                        &concrete_bounds,
                        iter,
                    )?;
                }

                let lower_sum = finite_lower_sum(concrete_bounds.lower());
                if concrete_bounds.lower().iter().any(|value| value.is_nan())
                    || concrete_bounds.upper().iter().any(|value| value.is_nan())
                {
                    warn!(
                        "{label}: NaN in bounds at iteration {iter}, aborting optimization (#2597)"
                    );
                    break;
                }

                let improved_output = lower_sum > best_lower_sum;
                if improved_output {
                    best_lower_sum = lower_sum;
                }

                let best_improvement = best_lower_sum - prev_best_lower_sum;
                if best_improvement < config.tolerance {
                    no_improve_iters += 1;
                } else {
                    no_improve_iters = 0;
                }
                if iter > 0 && no_improve_iters >= config.early_stop_patience {
                    if !config.should_save_best(iter, false) {
                        update_elementwise_best_bounds(
                            &mut best_lower,
                            &mut best_upper,
                            &concrete_bounds,
                            iter,
                        )?;
                    }
                    debug!(
                        "{label}: Converged at iteration {} (best improvement < {} for {} iters)",
                        iter, config.tolerance, no_improve_iters
                    );
                    break;
                }

                // Seed the first post-update iteration even when the output-only
                // score is still flat. On sequential graphs, tighter intermediates
                // can be a prerequisite for the final bound to improve at all.
                let should_refresh_reference_bounds = carry_forward_reference_bounds
                    && iter >= 1
                    && !reference_bounds.targets().is_empty()
                    && (improved_output || iter == 1);
                let refresh_candidate = if should_refresh_reference_bounds {
                    Some(self.collect_sequential_reference_refresh_candidate(
                        input,
                        reference_bounds,
                        alpha_state,
                        relu_name_to_idx,
                        engine,
                        config.deadline,
                    )?)
                } else {
                    None
                };

                let numerical_gradients = sequential_gradients::compute_sequential_gradients(
                    self,
                    config,
                    alpha_state,
                    input,
                    node_bounds,
                    exec_order,
                    output_dim,
                    relu_name_to_idx,
                    engine,
                    &gradients,
                    iter,
                )?;
                let numerical_gradients_upper = match config.gradient_method {
                    GradientMethod::Analytic => gradients_upper,
                    _ => numerical_gradients.clone(),
                };

                (
                    numerical_gradients,
                    numerical_gradients_upper,
                    lower_sum,
                    refresh_candidate,
                )
            };

            if iter == 0 {
                for (relu_idx, grad) in numerical_gradients.iter().enumerate() {
                    let grad_norm = grad.iter().map(|value| value * value).sum::<f32>().sqrt();
                    debug!(
                        "{label} iter 0: ReLU layer {} gradient L2 norm={:.6}",
                        relu_idx, grad_norm
                    );
                }
            }

            let adam_params = config.adam_params(lr, iter + 1);
            for (relu_idx, grad) in numerical_gradients.iter().enumerate() {
                if grad.iter().any(|value| !value.is_finite()) {
                    warn!(
                        "{label} iter {}: skipping ReLU {} gradient update — non-finite values detected (#2835)",
                        iter, relu_idx
                    );
                    total_gradient_skips += 1;
                    continue;
                }

                let neg_grad = grad.mapv(|value| -value);
                match config.optimizer {
                    Optimizer::Adam => {
                        alpha_state.update_adam(relu_idx, &neg_grad, &adam_params);
                    }
                    Optimizer::Sgd => {
                        let momentum = if config.use_momentum {
                            config.momentum
                        } else {
                            0.0
                        };
                        alpha_state.update(relu_idx, &neg_grad, lr, momentum);
                    }
                }

                if let Some(grad_upper) = numerical_gradients_upper.get(relu_idx) {
                    if grad_upper.iter().any(|value| !value.is_finite()) {
                        continue;
                    }

                    let neg_grad_upper = grad_upper.mapv(|value| -value);
                    match config.optimizer {
                        Optimizer::Adam => {
                            alpha_state.update_adam_upper(relu_idx, &neg_grad_upper, &adam_params);
                        }
                        Optimizer::Sgd => {
                            let momentum = if config.use_momentum {
                                config.momentum
                            } else {
                                0.0
                            };
                            alpha_state.update_upper(relu_idx, &neg_grad_upper, lr, momentum);
                        }
                    }
                }
            }

            if invprop_enabled {
                alpha_state.clip_gammas();
            }

            if let Some(candidate) = refresh_candidate {
                #[cfg(test)]
                {
                    reference_refresh_attempts += 1;
                }
                let tightened_targets = reference_bounds.merge_candidate(&candidate)?;
                #[cfg(test)]
                {
                    reference_tightened_targets_total += tightened_targets;
                }
                reference_bounds.promote_best_to_current()?;
                debug!(
                    "GraphNetwork α-CROWN iter {}: refreshed {} activation-input reference targets",
                    iter, tightened_targets
                );
            }

            lr *= config.lr_decay;

            if iter % 5 == 0 {
                debug!(
                    "{label} iter {}: lower_sum = {:.6}, lr = {:.6}",
                    iter, lower_sum, lr,
                );
            }

            prev_best_lower_sum = best_lower_sum;
        }

        if total_gradient_skips > 0 {
            warn!(
                "{label}: skipped {total_gradient_skips}/{} gradient updates (non-finite)",
                config.iterations * alpha_state.alphas.len()
            );
        }

        if let Some(bounds) = infeasible_bounds {
            return Ok(bounds);
        }

        let has_nan = best_lower.iter().any(|value| value.is_nan())
            || best_upper.iter().any(|value| value.is_nan());

        if !has_nan {
            clamp_inverted_best_bounds(&mut best_lower, &mut best_upper, label);
            let bounds =
                BoundedTensor::new_allow_infinite(best_lower, best_upper).map_err(|error| {
                    NyError::InternalError(format!(
                        "{label} best bounds invalid after inversion widening: {error}"
                    ))
                })?;
            #[cfg(test)]
            {
                SEQUENTIAL_REFERENCE_REFRESH_ATTEMPTS
                    .with(|slot| slot.set(reference_refresh_attempts));
                SEQUENTIAL_REFERENCE_TIGHTENED_TARGETS_TOTAL
                    .with(|slot| slot.set(reference_tightened_targets_total));
            }
            Ok(bounds)
        } else {
            warn!("{label}: NaN in best bounds, falling back to plain CROWN (#2909)");
            self.propagate_crown_with_engine_and_deadline(input, engine, config.deadline)
                .map(|result| result.bounds)
        }
    }

    fn collect_sequential_reference_refresh_candidate(
        &self,
        input: &BoundedTensor,
        reference_bounds: &GraphAlphaReferenceBounds,
        alpha_state: &AlphaState,
        relu_name_to_idx: &HashMap<String, usize>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let graph_alpha_state = graph_alpha_state_from_sequential(alpha_state, relu_name_to_idx)
            .ok_or_else(|| {
                NyError::InternalError(
                    "Sequential α-CROWN could not project alpha state into graph state".to_string(),
                )
            })?;

        self.collect_selected_crown_bounds_with_alpha(
            input,
            reference_bounds.targets(),
            reference_bounds.current(),
            &graph_alpha_state,
            engine,
            deadline,
        )
    }
}
