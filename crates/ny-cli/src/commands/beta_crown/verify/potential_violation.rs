// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Post-BaB confirmation for `PotentialViolation` results.
//!
//! When BaB identifies a subdomain where bounds suggest a property violation but
//! does not confirm with a concrete point, this module runs a bounded sampling
//! attack to either confirm the violation (upgrade to `Violated`) or downgrade
//! to `Unknown`.
//!
//! Reference: alpha-beta-CROWN `_format_result_act_bab(...)` calls
//! `check_and_save_cex(...)` on `unsafe_bab` in `complete_verifier_func.py:232-244`.
//! The reference also uses a fallback confirmation budget of 5 restarts / 5 steps
//! (`arguments.py:1029-1044`).
//!
//! Part of #3678.

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_onnx::vnnlib::VnnLibSpec;
use ny_propagate::{BabVerificationStatus, BetaCrownConfig, BetaCrownResult};
use ny_tensor::BoundedTensor;
use std::time::Instant;

use super::BetaCrownModel;

/// Default confirmation budget (mirrors alpha-beta-CROWN fallback).
const CONFIRM_RESTARTS: usize = 5;
const CONFIRM_STEPS: usize = 5;

/// Evaluate a model at a concrete input point via the exact concrete forward.
/// Completes bd68815 — IBP `.lower()` of a point box is not the network value
/// on widening models and fabricated false counterexamples (see
/// disjunctive_pgd::evaluate_model).
fn evaluate_model(model_net: &BetaCrownModel, point: &ArrayD<f32>) -> Result<ArrayD<f32>> {
    let input_bounds = BoundedTensor::concrete(point.clone())?;
    let output = match model_net {
        BetaCrownModel::Sequential(network) => {
            network.propagate_concrete_point(&input_bounds, None)?
        }
        BetaCrownModel::Graph(graph) => {
            graph.propagate_concrete_point(&input_bounds, None, None)?
        }
    };
    Ok(output.center())
}

/// Simple xorshift64 RNG (avoids `rand` dependency).
struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 & 0xFFFF_FFFF) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    fn next_bool(&mut self) -> bool {
        self.next_u32() & 1 == 0
    }
}

/// Confirm or downgrade a `PotentialViolation` result for property-backed runs.
///
/// Returns the result unchanged if it is not `PotentialViolation` or if no
/// VnnLib spec is available. Otherwise runs a bounded sampling+SPSA attack
/// to find a concrete counterexample.
///
/// On confirmation: rewrites to `Violated` with the concrete counterexample.
/// On failure: rewrites to `Unknown` so MIP fallback can still attempt verification.
///
/// Preserves all BaB metadata (domains, timing, bounds) from the original result.
// Justification: Confirmation helper bridges BaB result, model, input bounds, property spec,
// config, and deadline — all independently sourced from the caller context.
#[allow(clippy::too_many_arguments)]
pub(in crate::commands::beta_crown) fn confirm_potential_violation(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    vnnlib: Option<&VnnLibSpec>,
    result: BetaCrownResult,
    config: &BetaCrownConfig,
    deadline: Option<Instant>,
    json: bool,
) -> Result<BetaCrownResult> {
    if !matches!(result.result, BabVerificationStatus::PotentialViolation) {
        return Ok(result);
    }

    let Some(vnnlib) = vnnlib else {
        // Propertyless mode — no constraints to check against.
        return Ok(result);
    };

    // Deadline already exhausted — downgrade immediately.
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return Ok(BetaCrownResult {
            result: BabVerificationStatus::Unknown {
                reason: "Potential violation could not be confirmed before timeout".to_string(),
            },
            ..result
        });
    }

    let constraints = &vnnlib.output_constraints;
    if constraints.is_empty() {
        return Ok(result);
    }

    // Use configured PGD budget when attack is enabled, otherwise use the
    // alpha-beta-CROWN fallback confirmation budget (5 restarts / 5 steps).
    let (num_restarts, num_steps) = if config.enable_pgd_attack {
        (config.pgd_restarts, config.pgd_steps)
    } else {
        (CONFIRM_RESTARTS, CONFIRM_STEPS)
    };

    if !json {
        println!(
            "\n  Confirming potential violation ({} restarts, {} steps)...",
            num_restarts, num_steps
        );
    }

    // Run bounded sampling + SPSA to find a concrete counterexample.
    match try_confirm_attack(
        model_net,
        input,
        constraints,
        num_restarts,
        num_steps,
        deadline,
    )? {
        Some((counterexample, output)) => {
            if !json {
                println!("  Potential violation CONFIRMED with concrete counterexample.");
            }
            Ok(BetaCrownResult {
                result: BabVerificationStatus::Violated {
                    counterexample: counterexample.iter().copied().collect(),
                    output: output.iter().copied().collect(),
                },
                ..result
            })
        }
        None => {
            if !json {
                println!("  Potential violation could not be confirmed — downgrading to unknown.");
            }
            Ok(BetaCrownResult {
                result: BabVerificationStatus::Unknown {
                    reason: format!(
                        "BaB found potential violation but {} restarts x {} steps of \
                         sampling+SPSA could not confirm with a concrete counterexample",
                        num_restarts, num_steps
                    ),
                },
                ..result
            })
        }
    }
}

/// Run sampling + SPSA attack to confirm a potential violation.
///
/// Returns `Some((counterexample, output))` if a concrete counterexample is found
/// that satisfies ALL output constraints in the VnnLib spec.
fn try_confirm_attack(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    constraints: &[ny_onnx::vnnlib::OutputConstraint],
    num_restarts: usize,
    num_steps: usize,
    deadline: Option<Instant>,
) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
    let step_size = 0.01_f32;
    let spsa_delta = 0.001_f32;
    let n = input.lower().len();
    let lo = input.lower();
    let hi = input.upper();

    let lo_owned: Vec<f32> = lo.iter().copied().collect();
    let hi_owned: Vec<f32> = hi.iter().copied().collect();

    let mut rng = SimpleRng::new(7919); // Different seed from upfront attacks

    for restart in 0..num_restarts {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            tracing::info!(
                "Confirmation PGD: deadline exceeded at restart {}/{}",
                restart,
                num_restarts
            );
            break;
        }

        // Sample a random point in the input bounds.
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let l = lo_owned[i];
                let h = hi_owned[i];
                l + rng.next_f32() * (h - l)
            })
            .collect();
        let mut x = ArrayD::from_shape_vec(IxDyn(lo.shape()), vals)?;

        // Evaluate initial random point.
        let mut output = evaluate_model(model_net, &x)?;
        if super::check_unsafe_counterexample(&output, constraints) {
            return Ok(Some((x, output)));
        }

        // SPSA gradient steps toward constraint satisfaction.
        for _step in 0..num_steps {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }

            // Find the constraint with smallest satisfaction margin (most-violated).
            let mut min_margin = f32::INFINITY;
            let mut worst_is_relational = false;
            let mut worst_idx_a: usize = 0;
            let mut worst_idx_b: usize = 0;
            let mut worst_const: f32 = 0.0;
            let mut worst_negate = false;

            for constraint in constraints.iter() {
                let (margin, is_rel, a, b, c, neg) = match constraint {
                    ny_onnx::vnnlib::OutputConstraint::GreaterEq(i, j)
                    | ny_onnx::vnnlib::OutputConstraint::GreaterThan(i, j) => {
                        let yi = output.iter().nth(*i).copied().unwrap_or(0.0);
                        let yj = output.iter().nth(*j).copied().unwrap_or(0.0);
                        (yi - yj, true, *i, *j, 0.0, false)
                    }
                    ny_onnx::vnnlib::OutputConstraint::LessEq(i, j)
                    | ny_onnx::vnnlib::OutputConstraint::LessThan(i, j) => {
                        let yi = output.iter().nth(*i).copied().unwrap_or(0.0);
                        let yj = output.iter().nth(*j).copied().unwrap_or(0.0);
                        (yj - yi, true, *j, *i, 0.0, false)
                    }
                    ny_onnx::vnnlib::OutputConstraint::GreaterEqConst(i, c_val)
                    | ny_onnx::vnnlib::OutputConstraint::GreaterThanConst(i, c_val) => {
                        let y = output.iter().nth(*i).copied().unwrap_or(0.0);
                        (y - *c_val as f32, false, *i, 0, *c_val as f32, false)
                    }
                    ny_onnx::vnnlib::OutputConstraint::LessEqConst(i, c_val)
                    | ny_onnx::vnnlib::OutputConstraint::LessThanConst(i, c_val) => {
                        let y = output.iter().nth(*i).copied().unwrap_or(0.0);
                        (*c_val as f32 - y, false, *i, 0, *c_val as f32, true)
                    }
                    _ => continue, // skip unknown constraint variants
                };
                if margin < min_margin {
                    min_margin = margin;
                    worst_is_relational = is_rel;
                    worst_idx_a = a;
                    worst_idx_b = b;
                    worst_const = c;
                    worst_negate = neg;
                }
            }

            // SPSA gradient estimation.
            let pert_vals: Vec<f32> = (0..n)
                .map(|_| if rng.next_bool() { 1.0_f32 } else { -1.0_f32 })
                .collect();
            let perturbation = ArrayD::from_shape_vec(IxDyn(x.shape()), pert_vals)?;

            let x_plus = &x + &perturbation * spsa_delta;
            let x_minus = &x - &perturbation * spsa_delta;
            let out_plus = evaluate_model(model_net, &x_plus)?;
            let out_minus = evaluate_model(model_net, &x_minus)?;

            // Compute satisfaction margin for the worst constraint.
            let margin_plus = if worst_is_relational {
                let a = out_plus.iter().nth(worst_idx_a).copied().unwrap_or(0.0);
                let b = out_plus.iter().nth(worst_idx_b).copied().unwrap_or(0.0);
                a - b
            } else if worst_negate {
                worst_const - out_plus.iter().nth(worst_idx_a).copied().unwrap_or(0.0)
            } else {
                out_plus.iter().nth(worst_idx_a).copied().unwrap_or(0.0) - worst_const
            };
            let margin_minus = if worst_is_relational {
                let a = out_minus.iter().nth(worst_idx_a).copied().unwrap_or(0.0);
                let b = out_minus.iter().nth(worst_idx_b).copied().unwrap_or(0.0);
                a - b
            } else if worst_negate {
                worst_const - out_minus.iter().nth(worst_idx_a).copied().unwrap_or(0.0)
            } else {
                out_minus.iter().nth(worst_idx_a).copied().unwrap_or(0.0) - worst_const
            };

            if margin_plus.is_nan() || margin_minus.is_nan() {
                continue;
            }

            let grad = &perturbation * ((margin_plus - margin_minus) / (2.0 * spsa_delta));
            x = &x + &grad * step_size;

            // Project back into input bounds.
            for (xi, (l, h)) in x.iter_mut().zip(lo.iter().zip(hi.iter())) {
                if xi.is_nan() {
                    *xi = *l;
                } else {
                    *xi = xi.clamp(*l, *h);
                }
            }

            output = evaluate_model(model_net, &x)?;
            if super::check_unsafe_counterexample(&output, constraints) {
                return Ok(Some((x, output)));
            }
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_propagate::BabVerificationStatus;
    use std::time::Duration;

    /// Helper: create a placeholder BetaCrownModel for tests that exit before
    /// any model evaluation (deadline-expired and non-PotentialViolation paths).
    fn placeholder_model() -> BetaCrownModel {
        use ny_propagate::Network;
        let net = Network::new();
        BetaCrownModel::Sequential(Box::new(net))
    }

    /// When the deadline is already expired, PotentialViolation downgrades to Unknown.
    #[test]
    fn expired_deadline_downgrades_to_unknown() {
        let result = BetaCrownResult {
            result: BabVerificationStatus::PotentialViolation,
            domains_explored: 30,
            time_elapsed: Duration::from_millis(270),
            max_depth_reached: 5,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 0,
        };

        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let config = BetaCrownConfig::default();

        let mut vnnlib = VnnLibSpec::default();
        vnnlib.output_constraints = vec![ny_onnx::vnnlib::OutputConstraint::GreaterEqConst(0, 0.0)];

        let confirmed = confirm_potential_violation(
            &placeholder_model(),
            &BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1]))).unwrap(),
            Some(&vnnlib),
            result,
            &config,
            Some(expired),
            false,
        )
        .unwrap();

        assert!(
            matches!(confirmed.result, BabVerificationStatus::Unknown { .. }),
            "Expected Unknown but got {:?}",
            confirmed.result
        );
        // Metadata preserved.
        assert_eq!(confirmed.domains_explored, 30);
        assert_eq!(confirmed.max_depth_reached, 5);
    }

    /// Non-PotentialViolation results pass through unchanged.
    #[test]
    fn non_potential_violation_passes_through() {
        let result = BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored: 100,
            time_elapsed: Duration::from_secs(5),
            max_depth_reached: 10,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 50,
        };

        let config = BetaCrownConfig::default();
        let confirmed = confirm_potential_violation(
            &placeholder_model(),
            &BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1]))).unwrap(),
            None,
            result,
            &config,
            None,
            false,
        )
        .unwrap();

        assert_eq!(confirmed.result, BabVerificationStatus::Verified);
        assert_eq!(confirmed.domains_explored, 100);
    }

    /// No-VnnLib PotentialViolation passes through unchanged.
    #[test]
    fn no_vnnlib_passes_through() {
        let result = BetaCrownResult {
            result: BabVerificationStatus::PotentialViolation,
            domains_explored: 15,
            time_elapsed: Duration::from_millis(100),
            max_depth_reached: 3,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 0,
        };

        let config = BetaCrownConfig::default();
        let confirmed = confirm_potential_violation(
            &placeholder_model(),
            &BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1]))).unwrap(),
            None, // No VnnLib spec → propertyless mode
            result,
            &config,
            None,
            false,
        )
        .unwrap();

        assert!(
            matches!(confirmed.result, BabVerificationStatus::PotentialViolation),
            "Expected PotentialViolation (pass-through) but got {:?}",
            confirmed.result
        );
    }
}
