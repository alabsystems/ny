// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential and parallel PGD restart scheduling.

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tracing::{debug, info, trace, warn};

use crate::Network;

use super::eval::output_value;
use super::restart;
use super::PgdAttacker;
use crate::pgd_attack::result::{PgdResult, RestartResult};

impl PgdAttacker<'_> {
    /// Run a single PGD restart (internal helper).
    pub(super) fn run_single_restart(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        threshold: f32,
        verify_upper_bound: bool,
        seed: u64,
    ) -> Result<RestartResult> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = self.initialize_restart(network, input_bounds, &mut rng)?;
        let mut evals = 0;
        let mut consecutive_nan_skips: u32 = 0;
        // Maximum consecutive NaN gradient steps before aborting a restart.
        // If the network consistently produces NaN gradients from a given
        // starting point, continuing wastes the entire step budget. (#2968)
        const MAX_CONSECUTIVE_NAN_SKIPS: u32 = 5;

        // Create step state for this restart (#4277: AdamClipping optimizer).
        let mut step_state = self.config.create_step_state(input_bounds);

        // Run gradient descent steps
        for _step in 0..self.config.num_steps {
            let (gradient, step_evals) = self.estimate_gradient_spsa_with_bounds(
                network,
                &x,
                input_bounds,
                output_idx,
                &mut rng,
            )?;
            evals += step_evals;

            // NaN guard (#2721): If the network forward pass produces NaN
            // (e.g., Exp overflow, unstable LayerNorm), the SPSA gradient
            // estimate becomes NaN. Applying a NaN gradient corrupts x,
            // which then panics in BoundedTensor::concrete on the next
            // evaluate() call. Skip the step to break the NaN chain.
            if gradient.iter().any(|v| v.is_nan()) {
                consecutive_nan_skips += 1;
                if consecutive_nan_skips >= MAX_CONSECUTIVE_NAN_SKIPS {
                    warn!(
                        "PGD: {} consecutive NaN gradients, aborting restart",
                        consecutive_nan_skips
                    );
                    break;
                }
                trace!("PGD: NaN gradient detected, skipping step");
                continue;
            }
            consecutive_nan_skips = 0;

            // Apply the step using the configured optimizer (#4277).
            // Reference: alpha-beta-CROWN attack_pgd.py:342-350
            let projected = step_state.step(&gradient, &x, input_bounds, verify_upper_bound);

            // Restart-when-stuck (#4278): if the projected step is a no-op,
            // resample a fresh point instead of wasting remaining steps.
            if self.config.restart_when_stuck && restart::projected_step_is_stuck(&x, &projected) {
                x = restart::resample_uniform_point(self, input_bounds, &mut rng);
                step_state.reset();
                trace!("PGD: projected step is stuck, resampled restart point");
                continue;
            }
            x = projected;
        }

        // Evaluate final point
        let output = self.evaluate(network, &x)?;
        evals += 1;
        let value = output_value(&output, output_idx)?;

        let is_violation = if verify_upper_bound {
            value >= threshold
        } else {
            value <= threshold
        };

        Ok(RestartResult {
            input: x,
            output,
            value,
            is_violation,
            evaluations: evals,
        })
    }

    /// Sequential PGD attack (original implementation).
    pub(super) fn attack_sequential(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_value = if verify_upper_bound {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut total_evaluations = 0;
        let mut failed_restarts = 0usize;

        for restart in 0..self.config.num_restarts {
            // Deadline check (#3109): bail early if verification timeout budget
            // is exhausted. Return best counterexample found so far.
            if self.config.past_deadline() {
                info!(
                    "PGD attack: deadline exceeded at restart {}/{}, returning best result",
                    restart, self.config.num_restarts
                );
                break;
            }

            let seed = self.config.seed.wrapping_add(restart as u64);
            let result = match self.run_single_restart(
                network,
                input_bounds,
                output_idx,
                threshold,
                verify_upper_bound,
                seed,
            ) {
                Ok(r) => r,
                Err(e) => {
                    failed_restarts += 1;
                    debug!("PGD restart {} failed: {}", restart, e);
                    continue;
                }
            };
            total_evaluations += result.evaluations;

            let is_better = if verify_upper_bound {
                result.value > best_value
            } else {
                result.value < best_value
            };

            if is_better {
                best_value = result.value;
                best_counterexample = Some(result.input.clone());
                best_output = Some(result.output.clone());
            }

            if result.is_violation {
                debug!(
                    "PGD found counterexample at restart {}: output[{}] = {} {} threshold {}",
                    restart,
                    output_idx,
                    result.value,
                    if verify_upper_bound { ">=" } else { "<=" },
                    threshold
                );
                return Ok(PgdResult {
                    found_counterexample: true,
                    counterexample: Some(result.input),
                    output: Some(result.output),
                    best_output_value: result.value,
                    restarts_completed: restart + 1,
                    failed_restarts,
                    total_evaluations,
                });
            }

            trace!(
                "PGD restart {} complete: best output[{}] = {}, target {} threshold {}",
                restart,
                output_idx,
                best_value,
                if verify_upper_bound { ">=" } else { "<=" },
                threshold
            );
        }

        // All restarts failed — surface as error so callers don't interpret
        // empty results as "no counterexample found" (#3096).
        if best_counterexample.is_none() && failed_restarts > 0 {
            return Err(NyError::InternalError(format!(
                "PGD attack: all {} restarts failed ({} errors). \
                 Cannot determine counterexample status.",
                self.config.num_restarts, failed_restarts,
            )));
        }

        debug!(
            "PGD completed {} restarts without finding counterexample. Best: {} vs threshold {}",
            self.config.num_restarts, best_value, threshold
        );

        Ok(PgdResult {
            found_counterexample: false,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_value,
            restarts_completed: self.config.num_restarts,
            failed_restarts,
            total_evaluations,
        })
    }

    /// Parallel PGD attack using Rayon.
    pub(super) fn attack_parallel(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        let found = AtomicBool::new(false);
        let error_count = AtomicUsize::new(0);

        // Run restarts in parallel, with early termination when counterexample found
        let results: Vec<_> = (0..self.config.num_restarts)
            .into_par_iter()
            .filter_map(|restart| {
                // Skip if another thread found a counterexample
                if found.load(Ordering::Relaxed) {
                    return None;
                }

                // Deadline check (#3109): skip remaining restarts if deadline exceeded.
                if self.config.past_deadline() {
                    return None;
                }

                let seed = self.config.seed.wrapping_add(restart as u64);
                match self.run_single_restart(
                    network, input_bounds, output_idx, threshold, verify_upper_bound, seed,
                ) {
                    Ok(result) => {
                        if result.is_violation {
                            found.store(true, Ordering::Relaxed);
                            debug!(
                                "PGD found counterexample at restart {}: output[{}] = {} {} threshold {}",
                                restart, output_idx, result.value,
                                if verify_upper_bound { ">=" } else { "<=" },
                                threshold
                            );
                        }
                        Some((restart, result))
                    }
                    Err(e) => {
                        error_count.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!("PGD restart {} failed: {}", restart, e);
                        None
                    }
                }
            })
            .collect();

        // If all restarts errored and none succeeded, surface the error instead
        // of returning "no counterexample found" which callers may interpret as
        // verification passing. (#2981 Slice 1: PGD false-verified fix)
        let num_errors = error_count.load(Ordering::Relaxed);
        if results.is_empty() && num_errors > 0 {
            return Err(NyError::InternalError(format!(
                "PGD attack: all {} restarts failed ({} errors). \
                 Cannot determine counterexample status.",
                self.config.num_restarts, num_errors,
            )));
        }

        // Find the best result (counterexample if any, otherwise best candidate)
        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_value = if verify_upper_bound {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut total_evaluations = 0;
        let mut found_violation = false;
        let mut earliest_violation_restart = usize::MAX;

        for (restart, result) in results {
            total_evaluations += result.evaluations;

            if result.is_violation && restart < earliest_violation_restart {
                earliest_violation_restart = restart;
                found_violation = true;
                best_value = result.value;
                best_counterexample = Some(result.input);
                best_output = Some(result.output);
            } else if !found_violation {
                let is_better = if verify_upper_bound {
                    result.value > best_value
                } else {
                    result.value < best_value
                };
                if is_better {
                    best_value = result.value;
                    best_counterexample = Some(result.input);
                    best_output = Some(result.output);
                }
            }
        }

        if !found_violation {
            debug!(
                "PGD completed {} restarts without finding counterexample. Best: {} vs threshold {}",
                self.config.num_restarts, best_value, threshold
            );
        }

        Ok(PgdResult {
            found_counterexample: found_violation,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_value,
            restarts_completed: if found_violation {
                earliest_violation_restart + 1
            } else {
                self.config.num_restarts
            },
            failed_restarts: num_errors,
            total_evaluations,
        })
    }
}
