// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PGD counterexample search helpers.

use std::time::Instant;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info, warn};

use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
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
        if !self.config.enable_pgd_attack {
            return Ok(original_result);
        }

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
        let attack_result = attacker.attack(
            network,
            input,
            0, // Output index (scalar output assumed)
            threshold,
            self.config.verify_upper_bound,
        )?;

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
            let reeval_output = independent_concrete_forward(network, &cx_input)?;
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
                return Ok(original_result);
            }

            // Also verify input is within bounds (#2679 acceptance criterion).
            if !input_within_bounds(&cx_input, input) {
                warn!("PGD counterexample is outside input bounds. Downgrading to Unknown.");
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

        Ok(original_result)
    }
}

fn independent_concrete_forward(
    network: &Network,
    candidate: &ndarray::ArrayD<f32>,
) -> Result<ndarray::ArrayD<f32>> {
    let concrete = BoundedTensor::concrete(candidate.clone())?;
    // TRUE concrete (point) forward, collapsing to the interval center after each
    // layer. `propagate_ibp` propagates a BOX; for a point input the per-layer
    // soundness widening (esp. BatchNorm) — amplified by a deep conv stack — makes
    // its `.lower()` deviate far from the true output, so the historic
    // `propagate_ibp(...).lower()` here re-confirmed false counterexamples that ORT
    // then rejected (cgan_2023 unknown-downgrade). #cgan-eval. Engine-free CPU path
    // keeps this re-validation independent of the attacker's evaluator.
    let output = network.propagate_concrete_point(&concrete, None)?;
    Ok(output.center())
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
    use ndarray::{arr1, ArrayD};

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
