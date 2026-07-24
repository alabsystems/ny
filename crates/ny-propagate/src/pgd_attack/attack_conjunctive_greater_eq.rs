// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conjunctive GreaterEq PGD attack: finds counterexamples for `Y_target >= Y_j` constraints.

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
    /// Run a single PGD restart for conjunctive GreaterEq constraints (internal helper).
    ///
    /// For constraints Y_target >= Y_j for each j in comparison_indices,
    /// finds input minimizing max(Y_j - Y_target for all j).
    /// A counterexample is found when max <= 0 (all constraints satisfied).
    fn run_single_restart_conjunctive_greater_eq(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
        seed: u64,
    ) -> Result<RestartResult> {
        let mut rng = self.seeded_rng(seed);
        let mut x = self.initialize_restart(network, input_bounds, &mut rng)?;
        let mut evals = 0;
        let mut consecutive_nan_skips: u32 = 0;
        const MAX_CONSECUTIVE_NAN_SKIPS: u32 = 5;

        // Create step state for this restart (#4277: AdamClipping optimizer).
        // Gradient descent (sign=-1): minimize max(Y_j - Y_target).
        let mut step_state = self.config().create_step_state(input_bounds);

        for _step in 0..self.config().num_steps {
            let output = self.evaluate(network, &x)?;
            evals += 1;

            let target_val = output_value(&output, target_idx)?;

            // Find the constraint with max(Y_j - Y_target) - this is the "most violated"
            let mut max_diff = f32::NEG_INFINITY;
            let mut worst_j = comparison_indices[0];
            for &j in comparison_indices {
                let j_val = output_value(&output, j)?;
                let diff = j_val - target_val; // Flipped vs less_eq
                if diff > max_diff {
                    max_diff = diff;
                    worst_j = j;
                }
            }

            // Gradient descent on Y_worst_j - Y_target
            let (grad_target, evals_t) = self.estimate_gradient_spsa_with_bounds(
                network,
                &x,
                input_bounds,
                target_idx,
                &mut rng,
            )?;
            let (grad_j, evals_j) = self.estimate_gradient_spsa_with_bounds(
                network,
                &x,
                input_bounds,
                worst_j,
                &mut rng,
            )?;
            evals += evals_t + evals_j;

            // Gradient of (Y_j - Y_target) = grad_j - grad_target
            // To minimize, take negative gradient step
            let gradient_diff = &grad_j - &grad_target;

            // NaN guard (#2745): same pattern as conjunctive_less_eq.
            if gradient_diff.iter().any(|v| v.is_nan()) {
                consecutive_nan_skips += 1;
                if consecutive_nan_skips >= MAX_CONSECUTIVE_NAN_SKIPS {
                    warn!(
                        "Conjunctive GE PGD: {} consecutive NaN gradients, aborting restart",
                        consecutive_nan_skips
                    );
                    break;
                }
                trace!("Conjunctive GE PGD: NaN gradient detected, skipping step");
                continue;
            }
            consecutive_nan_skips = 0;

            // Apply the step using the configured optimizer (#4277).
            // Descent on the constraint: sign=-1.
            let projected = step_state.step(&gradient_diff, &x, input_bounds, false);
            x = projected;
        }

        let output = self.evaluate(network, &x)?;
        evals += 1;

        let target_val = output_value(&output, target_idx)?;
        let mut max_diff = f32::NEG_INFINITY;
        for &j in comparison_indices {
            let j_val = output_value(&output, j)?;
            let diff = j_val - target_val;
            if diff > max_diff {
                max_diff = diff;
            }
        }

        let is_violation = max_diff <= 0.0;

        Ok(RestartResult {
            input: x,
            output,
            value: max_diff,
            is_violation,
            evaluations: evals,
        })
    }

    /// Attack to find counterexample for conjunctive GreaterEq constraints.
    ///
    /// For constraints of the form: Y_target >= Y_j for each j in comparison_indices
    /// (common in ACAS-Xu prop_2 where Y_0 must be maximal).
    ///
    /// Returns counterexample if found (input where ALL constraints are satisfied),
    /// which proves the VNNLIB property is violated (unsafe condition can occur).
    pub fn attack_conjunctive_greater_eq(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
    ) -> Result<PgdResult> {
        if comparison_indices.is_empty() {
            return Err(NyError::InvalidSpec(
                "conjunctive PGD requires at least one comparison index".to_string(),
            ));
        }
        if self.config().parallel && self.config().num_restarts >= 10 {
            self.attack_conjunctive_greater_eq_parallel(
                network,
                input_bounds,
                target_idx,
                comparison_indices,
            )
        } else {
            self.attack_conjunctive_greater_eq_sequential(
                network,
                input_bounds,
                target_idx,
                comparison_indices,
            )
        }
    }

    fn attack_conjunctive_greater_eq_sequential(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
    ) -> Result<PgdResult> {
        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_max_diff = f32::INFINITY;
        let mut total_evaluations = 0;
        let mut failed_restarts = 0usize;

        for restart in 0..self.config().num_restarts {
            if self.config().past_deadline() {
                break;
            }

            let seed = self.config().seed.wrapping_add(restart as u64);
            let result = match self.run_single_restart_conjunctive_greater_eq(
                network,
                input_bounds,
                target_idx,
                comparison_indices,
                seed,
            ) {
                Ok(r) => r,
                Err(e) => {
                    failed_restarts += 1;
                    debug!("Conjunctive GE PGD restart {} failed: {}", restart, e);
                    continue;
                }
            };
            total_evaluations += result.evaluations;

            if result.value < best_max_diff {
                best_max_diff = result.value;
                best_counterexample = Some(result.input.clone());
                best_output = Some(result.output.clone());
            }

            if result.is_violation {
                debug!(
                    "Conjunctive GE PGD found counterexample at restart {}: max(Y_j - Y_{}) = {} <= 0",
                    restart, target_idx, result.value
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

        if best_counterexample.is_none() && failed_restarts > 0 {
            return Err(NyError::InternalError(format!(
                "Conjunctive GE PGD attack: all {} restarts failed ({} errors). \
                 Cannot determine counterexample status for Y_{} >= Y_j.",
                self.config().num_restarts,
                failed_restarts,
                target_idx,
            )));
        }

        Ok(PgdResult {
            found_counterexample: false,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_max_diff,
            restarts_completed: self.config().num_restarts,
            failed_restarts,
            total_evaluations,
        })
    }

    fn attack_conjunctive_greater_eq_parallel(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
    ) -> Result<PgdResult> {
        let found = AtomicBool::new(false);
        let failed_restarts = AtomicUsize::new(0);

        let results: Vec<_> = (0..self.config().num_restarts)
            .into_par_iter()
            .filter_map(|restart| {
                if found.load(Ordering::Relaxed) {
                    return None;
                }
                if self.config().past_deadline() {
                    return None;
                }

                let seed = self.config().seed.wrapping_add(restart as u64);
                match self.run_single_restart_conjunctive_greater_eq(
                    network, input_bounds, target_idx, comparison_indices, seed,
                ) {
                    Ok(result) => {
                        if result.is_violation {
                            found.store(true, Ordering::Relaxed);
                            debug!(
                                "Conjunctive GE PGD found counterexample at restart {}: max(Y_j - Y_{}) = {} <= 0",
                                restart, target_idx, result.value
                            );
                        }
                        Some((restart, result))
                    }
                    Err(e) => {
                        failed_restarts.fetch_add(1, Ordering::Relaxed);
                        debug!("Conjunctive GE PGD restart {} failed: {}", restart, e);
                        None
                    }
                }
            })
            .collect();

        let num_failed = failed_restarts.load(Ordering::Relaxed);
        if results.is_empty() && num_failed > 0 {
            return Err(NyError::InternalError(format!(
                "Conjunctive GE PGD attack: all {} restarts failed ({} errors). \
                 Cannot determine counterexample status for Y_{} >= Y_j.",
                self.config().num_restarts,
                num_failed,
                target_idx,
            )));
        }

        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_max_diff = f32::INFINITY;
        let mut total_evaluations = 0;
        let mut found_violation = false;
        let mut earliest_violation_restart = usize::MAX;

        for (restart, result) in results {
            total_evaluations += result.evaluations;

            if result.is_violation && restart < earliest_violation_restart {
                earliest_violation_restart = restart;
                found_violation = true;
                best_max_diff = result.value;
                best_counterexample = Some(result.input);
                best_output = Some(result.output);
            } else if !found_violation && result.value < best_max_diff {
                best_max_diff = result.value;
                best_counterexample = Some(result.input);
                best_output = Some(result.output);
            }
        }

        Ok(PgdResult {
            found_counterexample: found_violation,
            counterexample: best_counterexample,
            output: best_output,
            best_output_value: best_max_diff,
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
