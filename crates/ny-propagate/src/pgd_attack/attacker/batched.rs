// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-batched PGD restart loop: all restarts advance in lockstep.

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tracing::{debug, info, warn};

use crate::Network;

use super::restart;
use super::PgdAttacker;
use crate::pgd_attack::result::PgdResult;

impl PgdAttacker<'_> {
    /// Batched PGD attack: all restarts run in lockstep with GPU-batched IBP.
    ///
    /// Instead of N independent restarts (each with separate IBP evaluations),
    /// this batches all N restart points into a single GPU dispatch per step:
    /// - SPSA step: one IBP forward pass with `[2N, ...input_shape]` input
    /// - Final evaluation: one IBP forward pass with `[N, ...input_shape]` input
    ///
    /// Total GPU dispatches: `num_steps + 1` (vs `N * (2 * num_steps + 1)` unbatched).
    /// For soundnessbench (N=250, S=1000): 1001 vs 500,250 dispatches.
    ///
    /// Reference: alpha-beta-CROWN attack_pgd.py:267 batches restarts via
    /// `model(inputs.view(-1, *X_shape[2:]))`.
    pub(super) fn attack_batched(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        let n = self.config.num_restarts;
        let input_shape = input_bounds.shape();
        const MAX_CONSECUTIVE_NAN_SKIPS: u32 = 5;

        // Initialize batched restart points: [N, ...input_shape]
        let mut batch_shape = vec![n];
        batch_shape.extend_from_slice(input_shape);
        let mut x_batch = ArrayD::zeros(IxDyn(&batch_shape));

        // Per-restart RNGs (same seeding as sequential for comparable randomness)
        let mut rngs: Vec<StdRng> = (0..n)
            .map(|restart| StdRng::seed_from_u64(self.config.seed.wrapping_add(restart as u64)))
            .collect();

        for (i, rng) in rngs.iter_mut().enumerate() {
            let point = self.initialize_restart(network, input_bounds, rng)?;
            x_batch.index_axis_mut(Axis(0), i).assign(&point);
        }

        let mut nan_skips = vec![0u32; n];
        let mut aborted = vec![false; n];
        let mut total_evals = 0;

        // Create per-restart step states (#4277: AdamClipping optimizer).
        let mut step_states: Vec<_> = (0..n)
            .map(|_| self.config.create_step_state(input_bounds))
            .collect();

        // Main PGD loop: all restarts advance in lockstep
        for _step in 0..self.config.num_steps {
            if self.config.past_deadline() {
                info!("PGD batched: deadline exceeded, stopping at step {}", _step);
                break;
            }

            let (gradient_batch, step_evals) = self.estimate_gradient_spsa_batch_with_bounds(
                network,
                &x_batch,
                input_bounds,
                output_idx,
                &mut rngs,
            )?;
            total_evals += step_evals;

            // Per-restart gradient computation and position update
            for i in 0..n {
                if aborted[i] {
                    continue;
                }

                let gradient = gradient_batch.index_axis(Axis(0), i).to_owned();

                // NaN guard (#2721/#2968): skip step if gradient is NaN
                if gradient.iter().any(|v| v.is_nan()) {
                    nan_skips[i] += 1;
                    if nan_skips[i] >= MAX_CONSECUTIVE_NAN_SKIPS {
                        warn!(
                            "PGD batched: restart {} aborted after {} consecutive NaN gradients",
                            i, MAX_CONSECUTIVE_NAN_SKIPS
                        );
                        aborted[i] = true;
                    }
                    continue;
                }
                nan_skips[i] = 0;

                // Apply the step using the configured optimizer (#4277).
                let current_x = x_batch.index_axis(Axis(0), i).to_owned();
                let projected =
                    step_states[i].step(&gradient, &current_x, input_bounds, verify_upper_bound);

                // Restart-when-stuck (#4278): resample this restart if projected == previous.
                if self.config.restart_when_stuck
                    && restart::projected_step_is_stuck(&current_x, &projected)
                {
                    let fresh = restart::resample_uniform_point(self, input_bounds, &mut rngs[i]);
                    step_states[i].reset();
                    x_batch.index_axis_mut(Axis(0), i).assign(&fresh);
                } else {
                    x_batch.index_axis_mut(Axis(0), i).assign(&projected);
                }
            }
        }

        // Check if all restarts aborted
        let aborted_count = aborted.iter().filter(|&&a| a).count();
        if aborted_count == n {
            return Err(NyError::InternalError(format!(
                "PGD batched attack: all {} restarts aborted due to NaN gradients. \
                 Cannot determine counterexample status.",
                n,
            )));
        }

        // Final batched evaluation: [N, ...input_shape] -> [N, ...output_shape]
        let final_outputs = self.evaluate_batch(network, &x_batch)?;
        total_evals += n;

        // Find best result across all non-aborted restarts
        let mut best_value = if verify_upper_bound {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut best_idx = 0;
        let mut found_violation = false;

        for (i, &is_aborted) in aborted.iter().enumerate() {
            if is_aborted {
                continue;
            }

            let value = final_outputs
                .index_axis(Axis(0), i)
                .iter()
                .nth(output_idx)
                .copied()
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "output_idx {} out of range for output with {} elements",
                        output_idx,
                        final_outputs.index_axis(Axis(0), i).len()
                    ))
                })?;

            let is_violation = if verify_upper_bound {
                value >= threshold
            } else {
                value <= threshold
            };

            if is_violation && !found_violation {
                found_violation = true;
                best_value = value;
                best_idx = i;
            } else if !found_violation {
                let is_better = if verify_upper_bound {
                    value > best_value
                } else {
                    value < best_value
                };
                if is_better {
                    best_value = value;
                    best_idx = i;
                }
            }
        }

        if found_violation {
            debug!(
                "PGD batched: counterexample at restart {}: output[{}] = {} {} threshold {}",
                best_idx,
                output_idx,
                best_value,
                if verify_upper_bound { ">=" } else { "<=" },
                threshold
            );
        } else {
            debug!(
                "PGD batched: {} restarts without counterexample. Best: {} vs threshold {}",
                n, best_value, threshold
            );
        }

        Ok(PgdResult {
            found_counterexample: found_violation,
            counterexample: Some(x_batch.index_axis(Axis(0), best_idx).to_owned()),
            output: Some(final_outputs.index_axis(Axis(0), best_idx).to_owned()),
            best_output_value: best_value,
            restarts_completed: n,
            failed_restarts: aborted_count,
            total_evaluations: total_evals,
        })
    }
}
