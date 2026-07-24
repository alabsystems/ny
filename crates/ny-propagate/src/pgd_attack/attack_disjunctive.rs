// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unified disjunctive PGD attack parameterized by objective direction.
//!
//! Both `attack_disjunctive_greater_eq` (Y_j >= Y_target) and
//! `attack_disjunctive_less_eq` (Y_target >= Y_j) share identical logic
//! with only the sign of the objective/gradient flip differing.

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tracing::{debug, trace, warn};

use crate::Network;

use super::attacker::restart;
use super::attacker::{output_value, PgdAttacker};
use super::gama::{gama_guidance, gama_lambda_at, gama_lin_steps, gama_softmax};
use super::result::{PgdResult, RestartResult};

/// Direction of the disjunctive constraint objective.
#[derive(Debug, Clone, Copy)]
pub(crate) enum DisjunctiveDirection {
    /// Y_j >= Y_target: maximize Y_j - Y_target
    GreaterEq,
    /// Y_target >= Y_j: maximize Y_target - Y_j
    LessEq,
}

impl DisjunctiveDirection {
    fn objective_diff(self, target_val: f32, j_val: f32) -> f32 {
        match self {
            Self::GreaterEq => j_val - target_val,
            Self::LessEq => target_val - j_val,
        }
    }

    fn gradient_diff(self, grad_target: &ArrayD<f32>, grad_j: &ArrayD<f32>) -> ArrayD<f32> {
        match self {
            Self::GreaterEq => grad_j - grad_target,
            Self::LessEq => grad_target - grad_j,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::GreaterEq => "GE",
            Self::LessEq => "LE",
        }
    }

    /// Format string fragment for error messages: "Y_j >= Y_{target}" or "Y_{target} >= Y_j".
    fn constraint_desc(self, target_idx: usize) -> String {
        match self {
            Self::GreaterEq => format!("Y_j >= Y_{target_idx}"),
            Self::LessEq => format!("Y_{target_idx} >= Y_j"),
        }
    }

    /// Format string fragment for counterexample log: "max(Y_j - Y_N)" or "max(Y_N - Y_j)".
    fn diff_desc(self, target_idx: usize) -> String {
        match self {
            Self::GreaterEq => format!("max(Y_j - Y_{target_idx})"),
            Self::LessEq => format!("max(Y_{target_idx} - Y_j)"),
        }
    }
}

/// Minimum accepted violation margin for a disjunctive PGD witness, scaled to
/// cross-implementation forward noise (ny's exact f32 forward vs the ONNX
/// Runtime oracle deviate by ~1e-5-relative from accumulation order alone;
/// measured ~3e-5 absolute on ±14 logits, cora_2024 cifar10-point). Floored at
/// 1e-4. A boundary witness below this margin cannot survive the trusted-ORT
/// sat gate, so accepting it only burns the run on a false counterexample.
/// Mirrors ny-cli's disjunctive_pgd::noise_scaled_margin.
fn violation_margin_tol(output: &ArrayD<f32>) -> f32 {
    let max_abs = output.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    (1e-4_f32).max(1e-5 * max_abs)
}

impl PgdAttacker<'_> {
    /// Shared single-restart implementation for disjunctive PGD.
    pub(super) fn run_single_restart_disjunctive(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
        seed: u64,
        direction: DisjunctiveDirection,
    ) -> Result<RestartResult> {
        let mut rng = self.seeded_rng(seed);
        let mut x = self.initialize_restart(network, input_bounds, &mut rng)?;
        let mut evals = 0;
        let mut consecutive_nan_skips: u32 = 0;
        const MAX_CONSECUTIVE_NAN_SKIPS: u32 = 5;

        // Create step state for this restart (#4277: AdamClipping optimizer).
        // Gradient ascent (sign=+1): maximize disjunctive objective.
        let mut step_state = self.config().create_step_state(input_bounds);

        // #1449 GAMA: reference softmax `P` captured at this restart's first
        // forward; λ anneals linearly from λ₀ to 0 over the early steps. The
        // violation ACCEPT check below stays on the raw margin — GAMA only
        // changes the ascent direction, never what counts as a witness.
        let gama_lambda0 = self.config().gama_lambda;
        let gama_lin = gama_lin_steps(self.config().num_steps);
        let mut gama_p_ref: Option<Vec<f32>> = None;

        for step in 0..self.config().num_steps {
            // #cora-attack-deadline: the restart loops only polled the deadline
            // BETWEEN restarts, so one long SPSA restart (3072-dim inputs, 2
            // forwards per SPSA sample) blew through the capped attack phase —
            // measured ~12s of a 25s cora budget against a 3.75s phase cap.
            // Poll per STEP too. Schedule-only and attack-only: stopping the
            // ascent early can only return a weaker (or no) candidate, never a
            // wrong verdict.
            if self.config().past_deadline() {
                break;
            }
            let output = self.evaluate(network, &x)?;
            evals += 1;
            if gama_lambda0.is_some() && gama_p_ref.is_none() {
                gama_p_ref = Some(gama_softmax(&output));
            }

            let target_val = output_value(&output, target_idx)?;
            let mut max_diff = f32::NEG_INFINITY;
            let mut best_j = comparison_indices[0];
            for &j in comparison_indices {
                let j_val = output_value(&output, j)?;
                let diff = direction.objective_diff(target_val, j_val);
                if diff > max_diff {
                    max_diff = diff;
                    best_j = j;
                }
            }

            // Accept only past the cross-implementation noise margin: at exactly
            // >= 0.0 the restart stops on a boundary-hugging witness that the
            // trusted-ORT gate reads as SAFE (ny<->ORT f32 forwards deviate by
            // ~1e-5-relative from accumulation order alone), burning the run on
            // a false counterexample. Below the margin, keep stepping — the
            // ascent objective is exactly this diff, so a genuinely violated
            // property keeps improving past the tolerance.
            if max_diff >= violation_margin_tol(&output) {
                return Ok(RestartResult {
                    input: x,
                    output,
                    value: max_diff,
                    is_violation: true,
                    evaluations: evals,
                });
            }

            // #1449 GAMA branch: one SPSA of the scalar GAMA objective
            // (2 forwards) replaces the two per-output SPSA calls (4
            // forwards) — cheaper per step AND guidance-aware.
            let gradient_diff =
                if let (Some(lambda0), Some(p_ref)) = (gama_lambda0, gama_p_ref.as_ref()) {
                    let lambda = gama_lambda_at(lambda0, step, gama_lin);
                    let (g, evals_g) = self.estimate_gradient_spsa_gama(
                        network,
                        &x,
                        input_bounds,
                        target_idx,
                        best_j,
                        direction,
                        p_ref,
                        lambda,
                        &mut rng,
                    )?;
                    evals += evals_g;
                    g
                } else {
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
                        best_j,
                        &mut rng,
                    )?;
                    evals += evals_t + evals_j;
                    direction.gradient_diff(&grad_target, &grad_j)
                };

            if gradient_diff.iter().any(|v| v.is_nan()) {
                consecutive_nan_skips += 1;
                if consecutive_nan_skips >= MAX_CONSECUTIVE_NAN_SKIPS {
                    warn!(
                        "Disjunctive {} PGD: {} consecutive NaN gradients, aborting restart",
                        direction.label(),
                        consecutive_nan_skips
                    );
                    break;
                }
                trace!(
                    "Disjunctive {} PGD: NaN gradient detected, skipping step",
                    direction.label()
                );
                continue;
            }
            consecutive_nan_skips = 0;

            // Apply the step using the configured optimizer (#4277).
            // Ascent: sign=+1.
            let projected = step_state.step(&gradient_diff, &x, input_bounds, true);

            // Restart-when-stuck (#4278): resample a fresh point when the
            // optimizer+projection update is a no-op.
            if self.config().restart_when_stuck && restart::projected_step_is_stuck(&x, &projected)
            {
                x = restart::resample_uniform_point(self, input_bounds, &mut rng);
                step_state.reset();
                trace!(
                    "Disjunctive {} PGD: projected step is stuck, resampled restart point",
                    direction.label()
                );
                continue;
            }
            x = projected;
        }

        let output = self.evaluate(network, &x)?;
        evals += 1;

        let target_val = output_value(&output, target_idx)?;
        let mut max_diff = f32::NEG_INFINITY;
        for &j in comparison_indices {
            let j_val = output_value(&output, j)?;
            let diff = direction.objective_diff(target_val, j_val);
            if diff > max_diff {
                max_diff = diff;
            }
        }

        let is_violation = max_diff >= violation_margin_tol(&output);
        Ok(RestartResult {
            input: x,
            output,
            value: max_diff,
            is_violation,
            evaluations: evals,
        })
    }

    /// Scalar GAMA objective for a disjunctive `(target, j)` pair (#1449):
    /// direction-signed softmax margin + `λ·Σ_c (P_c − q_c)²`. The softmax
    /// margin is order-preserving in the raw logits, so it shares the raw
    /// margin's zero crossing; the accept gate stays on the raw margin.
    fn gama_disjunctive_objective(
        output: &ArrayD<f32>,
        target_idx: usize,
        j_idx: usize,
        direction: DisjunctiveDirection,
        p_ref: &[f32],
        lambda: f32,
    ) -> Result<f32> {
        let q = gama_softmax(output);
        let get = |i: usize| -> Result<f32> {
            q.get(i).copied().ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "GAMA objective index {i} out of bounds for {} outputs",
                    q.len()
                ))
            })
        };
        let margin = direction.objective_diff(get(target_idx)?, get(j_idx)?);
        Ok(margin + lambda * gama_guidance(&q, p_ref))
    }

    /// SPSA estimate of the GAMA objective gradient (#1449): one Rademacher
    /// perturbation, two box-projected forwards. Mirrors the graph lane's
    /// `spsa_gama_gradient`; returns a zero gradient on a NaN objective so the
    /// NaN-skip / restart-when-stuck logic in the caller handles it.
    #[allow(clippy::too_many_arguments)]
    fn estimate_gradient_spsa_gama(
        &self,
        network: &Network,
        x: &ArrayD<f32>,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        j_idx: usize,
        direction: DisjunctiveDirection,
        p_ref: &[f32],
        lambda: f32,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<(ArrayD<f32>, usize)> {
        use rand::RngExt;

        let delta = self
            .config()
            .suggested_spsa_delta(input_bounds)
            .max(self.config().spsa_delta);
        let perturbation: ArrayD<f32> =
            ArrayD::from_shape_fn(
                x.raw_dim(),
                |_| {
                    if rng.random::<bool>() {
                        1.0
                    } else {
                        -1.0
                    }
                },
            );
        let mut x_plus = x + &perturbation * delta;
        let mut x_minus = x - &perturbation * delta;
        crate::pgd_attack::optimizer::project_to_bounds_in_place(
            &mut x_plus,
            input_bounds.lower(),
            input_bounds.upper(),
        );
        crate::pgd_attack::optimizer::project_to_bounds_in_place(
            &mut x_minus,
            input_bounds.lower(),
            input_bounds.upper(),
        );
        let out_plus = self.evaluate(network, &x_plus)?;
        let out_minus = self.evaluate(network, &x_minus)?;
        let obj_plus = Self::gama_disjunctive_objective(
            &out_plus, target_idx, j_idx, direction, p_ref, lambda,
        )?;
        let obj_minus = Self::gama_disjunctive_objective(
            &out_minus, target_idx, j_idx, direction, p_ref, lambda,
        )?;
        if obj_plus.is_nan() || obj_minus.is_nan() {
            return Ok((ArrayD::zeros(x.raw_dim()), 2));
        }
        Ok((&perturbation * ((obj_plus - obj_minus) / (2.0 * delta)), 2))
    }

    /// Shared public entry point for disjunctive PGD attack.
    pub(super) fn attack_disjunctive(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
        direction: DisjunctiveDirection,
    ) -> Result<PgdResult> {
        if comparison_indices.is_empty() {
            return Err(NyError::InvalidSpec(
                "disjunctive PGD requires at least one comparison index".to_string(),
            ));
        }
        if self.config().parallel && self.config().num_restarts >= 10 {
            self.attack_disjunctive_parallel(
                network,
                input_bounds,
                target_idx,
                comparison_indices,
                direction,
            )
        } else {
            self.attack_disjunctive_sequential(
                network,
                input_bounds,
                target_idx,
                comparison_indices,
                direction,
            )
        }
    }

    fn attack_disjunctive_sequential(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
        direction: DisjunctiveDirection,
    ) -> Result<PgdResult> {
        let label = direction.label();
        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_max_diff = f32::NEG_INFINITY;
        let mut total_evaluations = 0;
        let mut failed_restarts = 0usize;

        for restart in 0..self.config().num_restarts {
            if self.config().past_deadline() {
                break;
            }

            let seed = self.config().seed.wrapping_add(restart as u64);
            let result = match self.run_single_restart_disjunctive(
                network,
                input_bounds,
                target_idx,
                comparison_indices,
                seed,
                direction,
            ) {
                Ok(r) => r,
                Err(e) => {
                    failed_restarts += 1;
                    debug!(
                        "Disjunctive {} PGD restart {} failed: {}",
                        label, restart, e
                    );
                    continue;
                }
            };
            total_evaluations += result.evaluations;

            if result.value > best_max_diff {
                best_max_diff = result.value;
                best_counterexample = Some(result.input.clone());
                best_output = Some(result.output.clone());
            }

            if result.is_violation {
                debug!(
                    "Disjunctive {} PGD found counterexample at restart {}: {} = {} >= 0",
                    label,
                    restart,
                    direction.diff_desc(target_idx),
                    result.value
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
                "Disjunctive {} PGD attack: all {} restarts failed ({} errors). \
                 Cannot determine counterexample status for {}.",
                label,
                self.config().num_restarts,
                failed_restarts,
                direction.constraint_desc(target_idx),
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

    pub(super) fn attack_disjunctive_parallel(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
        direction: DisjunctiveDirection,
    ) -> Result<PgdResult> {
        let label = direction.label();
        let found = AtomicBool::new(false);
        let failed_restarts = AtomicUsize::new(0);

        let results: Vec<_> = (0..self.config().num_restarts)
            .into_par_iter()
            .filter_map(|restart| {
                if found.load(Ordering::Relaxed) || self.config().past_deadline() {
                    return None;
                }

                let seed = self.config().seed.wrapping_add(restart as u64);
                match self.run_single_restart_disjunctive(
                    network,
                    input_bounds,
                    target_idx,
                    comparison_indices,
                    seed,
                    direction,
                ) {
                    Ok(result) => {
                        if result.is_violation {
                            found.store(true, Ordering::Relaxed);
                            debug!(
                                "Disjunctive {} PGD found counterexample at restart {}: {} = {} >= 0",
                                label,
                                restart,
                                direction.diff_desc(target_idx),
                                result.value
                            );
                        }
                        Some((restart, result))
                    }
                    Err(e) => {
                        failed_restarts.fetch_add(1, Ordering::Relaxed);
                        debug!("Disjunctive {} PGD restart {} failed: {}", label, restart, e);
                        None
                    }
                }
            })
            .collect();

        let num_failed = failed_restarts.load(Ordering::Relaxed);
        if results.is_empty() && num_failed > 0 {
            return Err(NyError::InternalError(format!(
                "Disjunctive {} PGD attack: all {} restarts failed ({} errors). \
                 Cannot determine counterexample status for {}.",
                label,
                self.config().num_restarts,
                num_failed,
                direction.constraint_desc(target_idx),
            )));
        }

        let mut best_counterexample: Option<ArrayD<f32>> = None;
        let mut best_output: Option<ArrayD<f32>> = None;
        let mut best_max_diff = f32::NEG_INFINITY;
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
            } else if !found_violation && result.value > best_max_diff {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Layer, LinearLayer};
    use crate::pgd_attack::config::PgdConfig;
    use ndarray::{arr1, arr2};

    fn two_output_network() -> Network {
        // y0 = x0 + 2*x1, y1 = 3*x0 + 4*x1 over x ∈ [0,1]²:
        // the disjunct Y_1 >= Y_0 (margin 2*x0 + 2*x1) is easily violated.
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
        ));
        network
    }

    fn unit_box() -> BoundedTensor {
        BoundedTensor::new(
            arr1(&[0.0_f32, 0.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap()
    }

    /// #1449: the GAMA-guided disjunctive attack still lands a raw-margin
    /// witness (the accept gate is unchanged; only the ascent objective is).
    #[ntest::timeout(10000)]
    #[test]
    fn test_disjunctive_gama_enabled_finds_violation() {
        let config = PgdConfig {
            gama_lambda: Some(crate::pgd_attack::config::GAMA_LAMBDA_DEFAULT),
            ..PgdConfig::fast()
        };
        let attacker = PgdAttacker::new(config);
        let result = attacker
            .attack_disjunctive_greater_eq(&two_output_network(), &unit_box(), 0, &[1])
            .expect("attack should run");
        assert!(
            result.found_counterexample,
            "GAMA-guided attack must find the easy Y_1 >= Y_0 violation"
        );
    }

    /// #1449: the GAMA objective matches margin + λ·guidance and errors loudly
    /// on an out-of-bounds output index instead of mis-indexing.
    #[test]
    fn test_gama_disjunctive_objective_value_and_bounds() {
        let output = arr1(&[1.0_f32, 3.0]).into_dyn();
        let q = gama_softmax(&output);
        let p_ref = vec![0.5_f32, 0.5];
        let lambda = 2.0_f32;
        let obj = PgdAttacker::gama_disjunctive_objective(
            &output,
            0,
            1,
            DisjunctiveDirection::GreaterEq,
            &p_ref,
            lambda,
        )
        .expect("in-bounds objective");
        let expected = (q[1] - q[0]) + lambda * gama_guidance(&q, &p_ref);
        assert!(
            (obj - expected).abs() < 1e-6,
            "obj={obj} expected={expected}"
        );

        let oob = PgdAttacker::gama_disjunctive_objective(
            &output,
            5,
            1,
            DisjunctiveDirection::GreaterEq,
            &p_ref,
            lambda,
        );
        assert!(oob.is_err(), "out-of-bounds index must be a typed error");
    }
}
