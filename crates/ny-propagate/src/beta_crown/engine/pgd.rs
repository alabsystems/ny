// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PGD counterexample search helpers.

use std::time::Instant;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info, warn};

use crate::beta_crown::result::{
    record_global_timeout_deadline_source, BabVerificationStatus, BetaCrownResult,
};
use crate::network::ibp::{propagate_concrete_point_until, LeadingAxisMode};
use crate::pgd_attack::PgdAttacker;
use crate::Network;

use super::BetaCrownVerifier;

impl BetaCrownVerifier {
    /// Try to find a counterexample using PGD attack with an explicit deadline.
    ///
    /// Called when verification is inconclusive to try to find a concrete
    /// input that violates the property. If found, **re-validates the
    /// counterexample with an independent forward pass** before promoting
    /// to `Violated` status.  This matches alpha-beta-CROWN's
    /// `default_adv_example_finalizer` + `test_conditions` pattern (#2679).
    ///
    /// If re-validation fails (the independent evaluation does not confirm
    /// violation), the result is left as `Unknown` and a warning is logged.
    ///
    /// The `deadline` parameter controls when PGD gives up. When called from
    /// the BaB loop with a post-BaB PGD reservation (#2206), this deadline
    /// extends past the BaB timeout so PGD has guaranteed time.
    pub(super) fn try_pgd_attack_with_deadline(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
        original_result: BetaCrownResult,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        // #phase-cost: see `ny_core::phase_cost` (observability only).
        let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::Attack);
        if !self.config.enable_pgd_attack {
            return Ok(original_result);
        }

        let original_was_timeout = matches!(original_result.result, BabVerificationStatus::Timeout);

        info!(
            "Running PGD attack with {} restarts, {} steps",
            self.config.pgd_restarts, self.config.pgd_steps
        );

        let pgd_config = self.config.pgd_attack_config(
            self.config.pgd_restarts,
            self.config.pgd_steps,
            deadline,
        );
        let attacker = PgdAttacker::new_with_optional_engine(pgd_config, self.engine());

        // Run PGD attack
        // For verify_upper_bound=true: looking for output >= threshold
        // For verify_upper_bound=false: looking for output <= threshold
        let attack_result = match attacker.attack(
            network,
            input,
            0, // Output index (scalar output assumed)
            threshold,
            self.config.verify_upper_bound,
        ) {
            Ok(result) => result,
            Err(error) if PgdAttacker::has_deadline_source(&error) && attacker.past_deadline() => {
                if original_was_timeout {
                    record_global_timeout_deadline_source(deadline);
                }
                // Expired optional falsification has no authority to replace
                // the incoming certified/undecided result with an error.
                return Ok(original_result);
            }
            Err(error) => return Err(error),
        };
        let terminal_deadline_observed = attacker.deadline_exhausted();

        if attack_result.found_counterexample {
            info!(
                "PGD found candidate counterexample: output = {} {} threshold = {}",
                attack_result.best_output_value,
                self.config.violation_direction_str(),
                threshold
            );

            let cx_input = attack_result.counterexample.ok_or_else(|| {
                NyError::InternalError(
                    "PGD reported found_counterexample=true but counterexample was missing"
                        .to_string(),
                )
            })?;

            // Re-validate with a fresh CPU concrete forward pass that bypasses
            // the attacker's engine/cached-plan path. This keeps the
            // confirmation step independent from the evaluator used during PGD.
            // Reference: alpha-beta-CROWN default_adv_example_finalizer + test_conditions.
            let reeval_output = match independent_concrete_forward(network, &cx_input, deadline) {
                Ok(output) => output,
                Err(error)
                    if PgdAttacker::has_deadline_source(&error) && attacker.past_deadline() =>
                {
                    if original_was_timeout {
                        record_global_timeout_deadline_source(deadline);
                    }
                    return Ok(original_result);
                }
                Err(error) => return Err(error),
            };
            let reeval_value = reeval_output.iter().next().copied().ok_or_else(|| {
                NyError::InternalError(
                    "Re-validation forward pass produced empty output".to_string(),
                )
            })?;

            let still_violated = if self.config.verify_upper_bound {
                reeval_value >= threshold
            } else {
                reeval_value <= threshold
            };

            if !still_violated {
                warn!(
                    "PGD counterexample failed re-validation: \
                     PGD reported output={}, independent eval output={}, threshold={}, \
                     mode=verify_upper_bound={}. Downgrading to Unknown.",
                    attack_result.best_output_value,
                    reeval_value,
                    threshold,
                    self.config.verify_upper_bound
                );
                if original_was_timeout && terminal_deadline_observed {
                    record_global_timeout_deadline_source(deadline);
                }
                return Ok(original_result);
            }

            // Also verify input is within bounds (#2679 acceptance criterion).
            if !input_within_bounds(&cx_input, input) {
                warn!("PGD counterexample is outside input bounds. Downgrading to Unknown.");
                if original_was_timeout && terminal_deadline_observed {
                    record_global_timeout_deadline_source(deadline);
                }
                return Ok(original_result);
            }

            info!(
                "PGD counterexample confirmed by independent re-evaluation: output = {} {} threshold = {}",
                reeval_value,
                self.config.violation_direction_str(),
                threshold
            );

            // Use the re-validated output, not PGD's internal output.
            let counterexample = cx_input.iter().copied().collect();
            let output = reeval_output.iter().copied().collect();

            // Containment and publication allocation do not create a fresh
            // attack grant. A late witness retains the incoming result.
            if attacker.past_deadline() {
                if original_was_timeout {
                    record_global_timeout_deadline_source(deadline);
                }
                return Ok(original_result);
            }

            return Ok(BetaCrownResult {
                result: BabVerificationStatus::Violated {
                    counterexample,
                    output,
                },
                domains_explored: original_result.domains_explored,
                time_elapsed: original_result.time_elapsed,
                max_depth_reached: original_result.max_depth_reached,
                output_bounds: original_result.output_bounds,
                cuts_generated: original_result.cuts_generated,
                domains_verified: original_result.domains_verified,
            });
        }

        debug!(
            "PGD attack completed {} restarts, no counterexample found. Best: {} vs threshold {}",
            attack_result.restarts_completed, attack_result.best_output_value, threshold
        );

        if original_was_timeout && terminal_deadline_observed {
            record_global_timeout_deadline_source(deadline);
        }

        Ok(original_result)
    }
}

fn independent_concrete_forward(
    network: &Network,
    candidate: &ndarray::ArrayD<f32>,
    deadline: Option<Instant>,
) -> Result<ndarray::ArrayD<f32>> {
    let deadline = match (deadline, ny_core::instance_budget::deadline()) {
        (Some(phase), Some(instance)) => Some(phase.min(instance)),
        (phase, instance) => phase.or(instance),
    };
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return Err(NyError::DeadlineExceeded(
            "PGD independent concrete confirmation".into(),
        ));
    }
    let concrete = BoundedTensor::concrete(candidate.clone())?;
    // TRUE concrete (point) forward, collapsing to the interval center after each
    // layer. `propagate_ibp` propagates a BOX; for a point input the per-layer
    // soundness widening (esp. BatchNorm) — amplified by a deep conv stack — makes
    // its `.lower()` deviate far from the true output, so the historic
    // `propagate_ibp(...).lower()` here re-confirmed false counterexamples that ORT
    // then rejected (cgan_2023 unknown-downgrade). #cgan-eval. Engine-free CPU path
    // keeps this re-validation independent of the attacker's evaluator.
    let output =
        propagate_concrete_point_until(network, &concrete, None, LeadingAxisMode::Plain, deadline)?;
    output.center_with_poll(|| {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            Err(NyError::DeadlineExceeded(
                "PGD independent confirmation output".into(),
            ))
        } else {
            Ok(())
        }
    })
}

/// Check that a candidate counterexample input is within the given input bounds.
///
/// Uses element-wise comparison with a small tolerance (1e-6) to account for
/// floating-point projection imprecision. Matches alpha-beta-CROWN's
/// `test_conditions` input-bounds check.
fn input_within_bounds(candidate: &ndarray::ArrayD<f32>, bounds: &BoundedTensor) -> bool {
    const TOLERANCE: f32 = 1e-6;
    let lower = bounds.lower();
    let upper = bounds.upper();

    // Shape must match.
    if candidate.shape() != lower.shape() {
        return false;
    }

    // Use .iter() (layout-agnostic) instead of .as_slice() which returns
    // None for non-contiguous arrays, causing a false `true` (#2941).
    for ((&x, &lo), &hi) in candidate.iter().zip(lower.iter()).zip(upper.iter()) {
        // NaN fails all comparisons, so `x < lo - tol || x > hi + tol` would
        // miss NaN inputs.  Reject NaN/Inf explicitly.
        if !x.is_finite() || x < lo - TOLERANCE || x > hi + TOLERANCE {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Layer, LinearLayer};
    use ndarray::{arr1, arr2, ArrayD};
    use std::time::Duration;

    fn phase_test_verifier() -> BetaCrownVerifier {
        let mut verifier = BetaCrownVerifier::default();
        verifier.config.enable_pgd_attack = true;
        verifier.config.pgd_restarts = 2;
        verifier.config.pgd_steps = 0;
        verifier.config.verify_upper_bound = true;
        verifier
    }

    fn phase_test_network() -> Network {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[1.0]]), None).unwrap(),
        ));
        network
    }

    fn incoming_timeout() -> BetaCrownResult {
        BetaCrownResult {
            result: BabVerificationStatus::Timeout,
            domains_explored: 7,
            domains_verified: 3,
            cuts_generated: 2,
            max_depth_reached: 4,
            time_elapsed: Duration::from_millis(5),
            output_bounds: Some(
                BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[3.0_f32]).into_dyn())
                    .unwrap(),
            ),
        }
    }

    #[test]
    fn expired_optional_pgd_preserves_the_incoming_timeout_and_complete_bounds() {
        let input =
            BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn()).unwrap();
        let result = phase_test_verifier()
            .try_pgd_attack_with_deadline(
                &phase_test_network(),
                &input,
                0.0,
                incoming_timeout(),
                Some(
                    Instant::now()
                        .checked_sub(Duration::from_millis(1))
                        .unwrap(),
                ),
            )
            .unwrap();
        assert!(matches!(result.result, BabVerificationStatus::Timeout));
        assert_eq!(result.domains_explored, 7);
        assert_eq!(result.domains_verified, 3);
        assert_eq!(result.cuts_generated, 2);
        assert_eq!(result.max_depth_reached, 4);
        assert_eq!(result.time_elapsed, Duration::from_millis(5));
        let bounds = result.output_bounds.unwrap();
        assert_eq!(bounds.lower()[[0]], -1.0);
        assert_eq!(bounds.upper()[[0]], 3.0);
    }

    #[test]
    fn live_optional_pgd_witness_still_crosses_independent_forward_and_containment() {
        let input =
            BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn()).unwrap();
        let result = phase_test_verifier()
            .try_pgd_attack_with_deadline(
                &phase_test_network(),
                &input,
                0.0,
                incoming_timeout(),
                Some(Instant::now() + Duration::from_secs(30)),
            )
            .unwrap();
        let BabVerificationStatus::Violated {
            counterexample,
            output,
        } = result.result
        else {
            panic!("live known witness was not independently accepted");
        };
        assert_eq!(counterexample.len(), 1);
        assert!((1.0..=2.0).contains(&counterexample[0]));
        assert_eq!(output, counterexample);
        assert_eq!(result.domains_explored, 7);
        assert_eq!(result.domains_verified, 3);
        assert_eq!(result.max_depth_reached, 4);
    }

    #[test]
    fn live_optional_pgd_does_not_reclassify_a_genuine_shape_failure_as_expiry() {
        let wrong_input = BoundedTensor::new(
            arr1(&[1.0_f32, 1.0]).into_dyn(),
            arr1(&[2.0_f32, 2.0]).into_dyn(),
        )
        .unwrap();
        let error = phase_test_verifier()
            .try_pgd_attack_with_deadline(
                &phase_test_network(),
                &wrong_input,
                0.0,
                incoming_timeout(),
                Some(Instant::now() + Duration::from_secs(30)),
            )
            .unwrap_err();
        assert!(!error.is_deadline_exceeded(), "{error:?}");
        assert!(error.to_string().contains("Shape mismatch"), "{error:?}");
    }

    /// Candidate within bounds → true.
    #[ntest::timeout(5000)]
    #[test]
    fn test_input_within_bounds_inside() {
        let bounds = BoundedTensor::new(
            arr1(&[0.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap();
        let candidate = arr1(&[0.5_f32, 0.0]).into_dyn();
        assert!(input_within_bounds(&candidate, &bounds));
    }

    /// Candidate exactly at bounds → true.
    #[ntest::timeout(5000)]
    #[test]
    fn test_input_within_bounds_at_boundary() {
        let bounds = BoundedTensor::new(
            arr1(&[0.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap();
        let candidate = arr1(&[0.0_f32, 1.0]).into_dyn();
        assert!(input_within_bounds(&candidate, &bounds));
    }

    /// Candidate outside bounds → false.
    #[ntest::timeout(5000)]
    #[test]
    fn test_input_within_bounds_outside() {
        let bounds = BoundedTensor::new(
            arr1(&[0.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap();
        let candidate = arr1(&[1.5_f32, 0.0]).into_dyn();
        assert!(!input_within_bounds(&candidate, &bounds));
    }

    /// Candidate within tolerance of bounds → true.
    #[ntest::timeout(5000)]
    #[test]
    fn test_input_within_bounds_within_tolerance() {
        let bounds =
            BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        // Slightly outside bounds but within 1e-6 tolerance.
        let candidate = arr1(&[1.0_f32 + 5e-7]).into_dyn();
        assert!(input_within_bounds(&candidate, &bounds));
    }

    /// Candidate well outside tolerance → false.
    #[ntest::timeout(5000)]
    #[test]
    fn test_input_within_bounds_outside_tolerance() {
        let bounds =
            BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        // Outside bounds by more than 1e-6 tolerance.
        let candidate = arr1(&[1.0_f32 + 2e-6]).into_dyn();
        assert!(!input_within_bounds(&candidate, &bounds));
    }

    /// NaN input → false (NaN is not within any bounds).
    #[ntest::timeout(5000)]
    #[test]
    fn test_input_within_bounds_nan_rejected() {
        let bounds =
            BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        let candidate = arr1(&[f32::NAN]).into_dyn();
        assert!(!input_within_bounds(&candidate, &bounds));
    }

    /// Infinity input → false.
    #[ntest::timeout(5000)]
    #[test]
    fn test_input_within_bounds_inf_rejected() {
        let bounds =
            BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        let candidate = arr1(&[f32::INFINITY]).into_dyn();
        assert!(!input_within_bounds(&candidate, &bounds));
    }

    /// Shape mismatch → false.
    #[ntest::timeout(5000)]
    #[test]
    fn test_input_within_bounds_shape_mismatch() {
        let bounds = BoundedTensor::new(
            arr1(&[0.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap();
        let candidate: ArrayD<f32> = arr1(&[0.5_f32]).into_dyn();
        assert!(!input_within_bounds(&candidate, &bounds));
    }
}
