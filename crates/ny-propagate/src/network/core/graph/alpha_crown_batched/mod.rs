// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched alpha-CROWN optimization for BilinearCrown (attention Q@K^T).
//!
//! SPSA-based alpha optimization for McCormick face selection in bilinear nodes.
//! Reference: auto_LiRPA/operators/bivariate.py:128-135, 39-75

mod spsa;

use std::collections::HashMap;

use ndarray::Array4;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info, warn};

use crate::bounds::AlphaCrownConfig;
use crate::layers::Layer;
use crate::network::alpha_crown_loop::finite_lower_sum;

use super::GraphNetwork;

/// Result of batched alpha-CROWN optimization.
pub(crate) struct BatchedAlphaCrownResult {
    /// Optimized bounds (element-wise best across iterations).
    pub(crate) bounds: BoundedTensor,
}

impl GraphNetwork {
    /// Run batched alpha-CROWN optimization for BilinearCrown nodes.
    ///
    /// Optimizes McCormick face selection alphas [4, m, n, k] for each BilinearCrown
    /// node via SPSA gradient estimation and Adam updates.
    ///
    /// Falls back to plain batched CROWN if no BilinearCrown nodes exist.
    ///
    /// The `engine` parameter threads GPU GEMM acceleration into every batched
    /// CROWN evaluation inside the optimizer and SPSA gradient loop (#3588).
    pub(crate) fn propagate_alpha_crown_batched(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let result = self.alpha_crown_batched_optimize(input, config, engine)?;
        Ok(result.bounds)
    }

    /// Core optimization loop for batched alpha-CROWN.
    fn alpha_crown_batched_optimize(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BatchedAlphaCrownResult> {
        // Step 1: Collect BilinearCrown nodes and initialize alphas.
        let bilinear_nodes = self.collect_bilinear_nodes(input, engine)?;
        if bilinear_nodes.is_empty() {
            let bounds = self.propagate_crown_batched_with_engine(input, engine)?;
            return Ok(BatchedAlphaCrownResult { bounds });
        }

        debug!(
            "Batched α-CROWN: optimizing {} BilinearCrown nodes with {} iterations",
            bilinear_nodes.len(),
            config.iterations
        );

        // Initialize alpha state: [4, m, n, k] per BilinearCrown node, all ones.
        let mut bilinear_alphas: HashMap<String, Array4<f32>> = bilinear_nodes
            .iter()
            .map(|(name, (m, n, k))| {
                debug!(
                    "  BilinearCrown '{}': alpha shape [4, {}, {}, {}]",
                    name, m, n, k
                );
                (name.clone(), Array4::ones((4, *m, *n, *k)))
            })
            .collect();

        // Initialize Adam state per node.
        let mut adam_m: HashMap<String, Array4<f32>> = bilinear_nodes
            .iter()
            .map(|(name, (m, n, k))| (name.clone(), Array4::zeros((4, *m, *n, *k))))
            .collect();
        let mut adam_v: HashMap<String, Array4<f32>> = bilinear_nodes
            .iter()
            .map(|(name, (m, n, k))| (name.clone(), Array4::zeros((4, *m, *n, *k))))
            .collect();

        // Step 2: Get baseline CROWN bounds (no alpha optimization).
        let crown_bounds = self.propagate_crown_batched_with_engine(input, engine)?;
        let mut best_lower = crown_bounds.lower().clone();
        let mut best_upper = crown_bounds.upper().clone();
        let mut best_lower_sum = finite_lower_sum(crown_bounds.lower());
        let mut prev_best_lower_sum = best_lower_sum;
        let mut no_improve_iters = 0usize;
        let mut lr = config.learning_rate;

        // Step 3: Optimization loop.
        let mut iter_count = 0usize;
        for iter in 0..config.iterations {
            if config.past_deadline() {
                info!(
                    "Batched α-CROWN: deadline exceeded at iteration {}/{}",
                    iter, config.iterations
                );
                break;
            }
            iter_count = iter + 1;

            // Run batched CROWN backward with current bilinear alphas (#3588: engine-aware).
            let bounds = self.propagate_crown_batched_with_bilinear_alphas_and_engine(
                input,
                &bilinear_alphas,
                engine,
            )?;

            // NaN check: abort if bounds contain NaN.
            if bounds.lower().iter().any(|v| v.is_nan())
                || bounds.upper().iter().any(|v| v.is_nan())
            {
                warn!(
                    "Batched α-CROWN: NaN in bounds at iteration {}, aborting",
                    iter
                );
                break;
            }

            // Update element-wise best bounds.
            ndarray::Zip::from(&mut best_lower)
                .and(bounds.lower())
                .for_each(|best, &curr| {
                    if curr > *best {
                        *best = curr;
                    }
                });
            ndarray::Zip::from(&mut best_upper)
                .and(bounds.upper())
                .for_each(|best, &curr| {
                    if curr < *best {
                        *best = curr;
                    }
                });

            let lower_sum = finite_lower_sum(bounds.lower());
            if lower_sum > best_lower_sum {
                best_lower_sum = lower_sum;
            }

            // Early stopping.
            let improvement = best_lower_sum - prev_best_lower_sum;
            if improvement < config.tolerance {
                no_improve_iters += 1;
            } else {
                no_improve_iters = 0;
            }
            if iter > 0 && no_improve_iters >= config.early_stop_patience {
                debug!(
                    "Batched α-CROWN: converged at iteration {} (no improvement for {} iters)",
                    iter, no_improve_iters
                );
                break;
            }

            // Step 4: SPSA gradient estimation for bilinear alphas (#3588: engine-aware).
            let gradients =
                self.spsa_bilinear_gradients(input, &mut bilinear_alphas, config, engine)?;

            // Step 5: Adam update for bilinear alphas.
            let adam_params = config.adam_params(lr, iter + 1);
            for (name, grad) in &gradients {
                if grad.iter().any(|v| !v.is_finite()) {
                    warn!(
                        "Batched α-CROWN: non-finite gradient for '{}' at iter {}, skipping",
                        name, iter
                    );
                    continue;
                }
                // Gradient ascent to maximize lower bound: negate gradient.
                let neg_grad = grad.mapv(|v| -v);
                self.update_bilinear_adam_inplace(
                    &mut bilinear_alphas,
                    &mut adam_m,
                    &mut adam_v,
                    name,
                    &neg_grad,
                    &adam_params,
                );
            }

            lr *= config.lr_decay;

            if iter % 5 == 0 {
                debug!(
                    "Batched α-CROWN iter {}: lower_sum={:.6}, best={:.6}, lr={:.6}",
                    iter, lower_sum, best_lower_sum, lr
                );
            }

            prev_best_lower_sum = best_lower_sum;
        }

        // Check for NaN in best bounds — fall back to plain CROWN.
        let has_nan =
            best_lower.iter().any(|v| v.is_nan()) || best_upper.iter().any(|v| v.is_nan());
        if has_nan {
            warn!("Batched α-CROWN: NaN in best bounds, falling back to plain CROWN");
            let bounds = self.propagate_crown_batched_with_engine(input, engine)?;
            return Ok(BatchedAlphaCrownResult { bounds });
        }

        let bounds = BoundedTensor::new_allow_infinite(best_lower, best_upper).map_err(|e| {
            ny_core::NyError::InternalError(format!("Batched α-CROWN: best bounds invalid: {e}"))
        })?;

        info!(
            "Batched α-CROWN: completed {} iterations, final lower_sum={:.6}",
            iter_count, best_lower_sum
        );

        Ok(BatchedAlphaCrownResult { bounds })
    }

    /// Collect BilinearCrown nodes and their alpha dimensions.
    ///
    /// Returns a map from node name to (m, n, k) dimensions.
    fn collect_bilinear_nodes(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, (usize, usize, usize)>> {
        let node_bounds = self.collect_node_bounds_with_engine(input, engine)?;

        let mut bilinear_nodes = HashMap::new();
        for (name, node) in &self.nodes {
            if let Layer::BilinearCrown(bilinear) = &node.layer {
                let (input_a_name, input_b_name) = node.require_binary_inputs()?;
                let input_a_bounds = node_bounds.get(input_a_name).ok_or_else(|| {
                    ny_core::NyError::InvalidSpec(format!(
                        "BilinearCrown '{}': input_a '{}' bounds not found",
                        name, input_a_name
                    ))
                })?;
                let input_b_bounds = node_bounds.get(input_b_name).ok_or_else(|| {
                    ny_core::NyError::InvalidSpec(format!(
                        "BilinearCrown '{}': input_b '{}' bounds not found",
                        name, input_b_name
                    ))
                })?;
                let (m, n, k) =
                    bilinear.alpha_shape(input_a_bounds.shape(), input_b_bounds.shape())?;
                bilinear_nodes.insert(name.clone(), (m, n, k));
            }
        }
        Ok(bilinear_nodes)
    }
}

#[cfg(test)]
mod measurement;
#[cfg(test)]
mod measurement_phase3;
#[cfg(test)]
mod tests;
