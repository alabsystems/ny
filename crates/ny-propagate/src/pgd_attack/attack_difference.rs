// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Difference-constraint PGD attack: finds counterexamples for `Y[i] - Y[j]` violations.

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tracing::{debug, trace, warn};

use crate::Network;

use super::attacker::{output_value, PgdAttacker};
use super::result::{PgdResult, RestartResult};

impl PgdAttacker<'_> {
    /// Run a single PGD restart for difference constraint (internal helper).
    // Justification: PGD restart needs network, input bounds, both output indices,
    // constraint direction, step size, iteration count, and RNG — all independent attack params.
    fn run_single_restart_difference(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx_i: usize,
        output_idx_j: usize,
        threshold: f32,
        verify_upper_bound: bool,
        seed: u64,
    ) -> Result<RestartResult> {
        let mut rng = self.seeded_rng(seed);
        let mut x = self.initialize_restart(network, input_bounds, &mut rng)?;
        let mut evals = 0;
        let mut consecutive_nan_skips: u32 = 0;
        // Maximum consecutive NaN gradient steps before aborting a restart.
        // Same threshold as standard PGD attacker (#2968).
        const MAX_CONSECUTIVE_NAN_SKIPS: u32 = 5;

        // Create step state for this restart (#4277: AdamClipping optimizer).
        let mut step_state = self.config().create_step_state(input_bounds);

        for _step in 0..self.config().num_steps {
            let (grad_i, evals_i) = self.estimate_gradient_spsa_with_bounds(
                network,
                &x,
                input_bounds,
                output_idx_i,
                &mut rng,
            )?;
            let (grad_j, evals_j) = self.estimate_gradient_spsa_with_bounds(
                network,
                &x,
                input_bounds,
                output_idx_j,
                &mut rng,
            )?;
            evals += evals_i + evals_j;

            let gradient_diff = &grad_i - &grad_j;

            // NaN guard (#2745): If either SPSA gradient estimate produces NaN
            // (e.g., Exp overflow, unstable LayerNorm), the difference gradient
            // becomes NaN. Applying it corrupts x, which project() then silently
            // resets to the lower corner, discarding all PGD search progress.
            // Same pattern as attacker.rs (#2721, #2968).
            if gradient_diff.iter().any(|v| v.is_nan()) {
                consecutive_nan_skips += 1;
                if consecutive_nan_skips >= MAX_CONSECUTIVE_NAN_SKIPS {
                    warn!(
                        "Difference PGD: {} consecutive NaN gradients, aborting restart",
                        consecutive_nan_skips
                    );
                    break;
                }
                trace!("Difference PGD: NaN gradient detected, skipping step");
                continue;
            }
            consecutive_nan_skips = 0;

            // Apply the step using the configured optimizer (#4277).
            let projected = step_state.step(&gradient_diff, &x, input_bounds, verify_upper_bound);
            x = projected;
        }

        let output = self.evaluate(network, &x)?;
        evals += 1;

        let val_i = output_value(&output, output_idx_i)?;
        let val_j = output_value(&output, output_idx_j)?;
        let diff = val_i - val_j;

        let is_violation = if verify_upper_bound {
            diff >= threshold
        } else {
            diff <= threshold
        };

        Ok(RestartResult {
            input: x,
            output,
            value: diff,
            is_violation,
            evaluations: evals,
        })
    }

    /// Attack to find counterexample for a difference constraint: `output[i] - output[j]` violates threshold.
    ///
    /// This is common in ACAS-Xu where we verify Y_i <= Y_j (difference <= 0).
    pub fn attack_difference(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx_i: usize,
        output_idx_j: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        if self.config().parallel && self.config().num_restarts >= 10 {
            self.attack_difference_parallel(
                network,
                input_bounds,
                output_idx_i,
                output_idx_j,
                threshold,
                verify_upper_bound,
            )
        } else {
            self.attack_difference_sequential(
                network,
                input_bounds,
                output_idx_i,
                output_idx_j,
                threshold,
                verify_upper_bound,
            )
        }
    }

    /// Sequential attack for difference constraint.
    fn attack_difference_sequential(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx_i: usize,
        output_idx_j: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_diff = if verify_upper_bound {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut total_evaluations = 0;
        let mut failed_restarts = 0usize;

        for restart in 0..self.config().num_restarts {
            // Deadline check (#3109)
            if self.config().past_deadline() {
                break;
            }

            let seed = self.config().seed.wrapping_add(restart as u64);
            let result = match self.run_single_restart_difference(
                network,
                input_bounds,
                output_idx_i,
                output_idx_j,
                threshold,
                verify_upper_bound,
                seed,
            ) {
                Ok(r) => r,
                Err(e) => {
                    failed_restarts += 1;
                    debug!("PGD difference restart {} failed: {}", restart, e);
                    continue;
                }
            };
            total_evaluations += result.evaluations;

            let is_better = if verify_upper_bound {
                result.value > best_diff
            } else {
                result.value < best_diff
            };

            if is_better {
                best_diff = result.value;
                best_counterexample = Some(result.input.clone());
                best_output = Some(result.output.clone());
            }

            if result.is_violation {
                debug!(
                    "PGD found counterexample at restart {}: Y[{}] - Y[{}] = {} {} threshold {}",
                    restart,
                    output_idx_i,
                    output_idx_j,
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
        }

        // All restarts failed — surface as error (#3096).
        if best_counterexample.is_none() && failed_restarts > 0 {
            return Err(NyError::InternalError(format!(
                "PGD difference attack: all {} restarts failed ({} errors). \
                 Cannot determine counterexample status for Y[{}] - Y[{}].",
                self.config().num_restarts,
                failed_restarts,
                output_idx_i,
                output_idx_j,
            )));
        }

        Ok(PgdResult {
            found_counterexample: false,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_diff,
            restarts_completed: self.config().num_restarts,
            failed_restarts,
            total_evaluations,
        })
    }

    /// Parallel attack for difference constraint using Rayon.
    fn attack_difference_parallel(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx_i: usize,
        output_idx_j: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<PgdResult> {
        let found = AtomicBool::new(false);
        let failed_restarts = AtomicUsize::new(0);

        let results: Vec<_> = (0..self.config().num_restarts)
            .into_par_iter()
            .filter_map(|restart| {
                if found.load(Ordering::Relaxed) {
                    return None;
                }
                // Deadline check (#3109)
                if self.config().past_deadline() {
                    return None;
                }

                let seed = self.config().seed.wrapping_add(restart as u64);
                match self.run_single_restart_difference(
                    network, input_bounds, output_idx_i, output_idx_j, threshold, verify_upper_bound, seed,
                ) {
                    Ok(result) => {
                        if result.is_violation {
                            found.store(true, Ordering::Relaxed);
                            debug!(
                                "PGD found counterexample at restart {}: Y[{}] - Y[{}] = {} {} threshold {}",
                                restart, output_idx_i, output_idx_j, result.value,
                                if verify_upper_bound { ">=" } else { "<=" },
                                threshold
                            );
                        }
                        Some((restart, result))
                    }
                    Err(e) => {
                        failed_restarts.fetch_add(1, Ordering::Relaxed);
                        debug!("PGD difference restart {} failed: {}", restart, e);
                        None
                    }
                }
            })
            .collect();

        let num_failed = failed_restarts.load(Ordering::Relaxed);
        if results.is_empty() && num_failed > 0 {
            return Err(NyError::InternalError(format!(
                "PGD difference attack: all {} restarts failed ({} errors). \
                 Cannot determine counterexample status for Y[{}] - Y[{}].",
                self.config().num_restarts,
                num_failed,
                output_idx_i,
                output_idx_j,
            )));
        }

        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_diff = if verify_upper_bound {
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
                best_diff = result.value;
                best_counterexample = Some(result.input);
                best_output = Some(result.output);
            } else if !found_violation {
                let is_better = if verify_upper_bound {
                    result.value > best_diff
                } else {
                    result.value < best_diff
                };
                if is_better {
                    best_diff = result.value;
                    best_counterexample = Some(result.input);
                    best_output = Some(result.output);
                }
            }
        }

        Ok(PgdResult {
            found_counterexample: found_violation,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_diff,
            restarts_completed: if found_violation {
                earliest_violation_restart + 1
            } else {
                self.config().num_restarts
            },
            failed_restarts: num_failed,
            total_evaluations,
        })
    }
}
