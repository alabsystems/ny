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
use ny_propagate::{BabVerificationStatus, BetaCrownConfig, BetaCrownResult, ViolationWitness};
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
    let BabVerificationStatus::PotentialViolation { witness: carried } = &result.result else {
        return Ok(result);
    };
    // Detach the carried witness so `..result` can move the rest below.
    let carried = carried.clone();

    let Some(vnnlib) = vnnlib else {
        // Propertyless mode — no constraints to check against.
        return Ok(result);
    };

    // #advcheck-witness: BaB's adv_check probe already ran a TRUE concrete
    // forward on a point of the sub-box it was searching and found a genuine
    // violation — then dropped the point, leaving the confirmer below to
    // re-SEARCH the whole ROOT box for it. That search routinely missed, and a
    // validated counterexample downgraded to Unknown.
    //
    // VERIFY the carried point instead of re-searching for it. This is not a
    // shortcut past any gate:
    //   * the point is re-evaluated here through the same exact concrete
    //     forward (`evaluate_model`) the re-search uses;
    //   * acceptance is the same `check_unsafe_counterexample` over the same
    //     full VNN-LIB constraint set the re-search uses;
    //   * the resulting `Violated` is rendered as a witness and still has to
    //     pass the unchanged trusted ONNX-Runtime gate
    //     (`gate_sat_with_trusted_oracle` in commands/vnncomp.rs) before any
    //     `sat` is scored.
    // Anything short of a clean pass falls through to the pre-existing code
    // below completely unchanged, so a bad witness can never cost a verdict.
    //
    // ORDERING: the deadline bail-out stays FIRST, exactly as before. Checking
    // the carried witness after an exhausted deadline would be dead work —
    // `gate_result_at_deadline` (dispatch.rs) rewrites ANY result, `Violated`
    // included, to `Timeout` once the deadline has passed — and shipping a path
    // whose stated benefit never reaches production is the very defect class
    // this change exists to remove.

    // Deadline already exhausted — downgrade immediately.
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return Ok(BetaCrownResult {
            result: BabVerificationStatus::Unknown {
                reason: "Potential violation could not be confirmed before timeout".to_string(),
            },
            ..result
        });
    }

    if let Some(carried) = carried.as_deref() {
        match verify_carried_witness(model_net, input, carried, &vnnlib.output_constraints) {
            Ok(Some((point, output))) => {
                if !json {
                    println!(
                        "  Potential violation CONFIRMED from the witness BaB already held \
                         (verified in place, no re-search)."
                    );
                }
                return Ok(BetaCrownResult {
                    result: BabVerificationStatus::Violated {
                        counterexample: point.iter().copied().collect(),
                        output: output.iter().copied().collect(),
                    },
                    ..result
                });
            }
            Ok(None) => tracing::info!(
                "Carried BaB witness did not re-verify against the full property; \
                 falling back to the unchanged confirmation search"
            ),
            Err(err) => tracing::info!(
                "Carried BaB witness could not be evaluated ({err}); falling back to \
                 the unchanged confirmation search"
            ),
        }
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

/// Verify a carried BaB witness in place (#advcheck-witness).
///
/// Returns `Some((point, output))` only when the point is inside the DECLARED
/// input box AND a fresh exact concrete forward at that point satisfies every
/// VNN-LIB output constraint — the identical acceptance test
/// [`try_confirm_attack`] applies to the points it samples. Every other
/// outcome (shape mismatch, out-of-box coordinate, non-violating re-forward,
/// evaluation error) returns `None`/`Err` so the caller falls through to the
/// pre-existing search with no change in behaviour.
///
/// The witness's own recorded output is deliberately NOT trusted: it is only
/// logged against the fresh forward, because a divergence there is exactly the
/// class of bug (`cgan_2023`) the downstream ORT gate exists to catch.
fn verify_carried_witness(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    witness: &ViolationWitness,
    constraints: &[ny_onnx::vnnlib::OutputConstraint],
) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
    if constraints.is_empty() {
        return Ok(None);
    }
    if witness.input_shape.as_slice() != input.lower().shape()
        || witness.input.len() != input.lower().len()
    {
        tracing::info!(
            witness_shape = ?witness.input_shape,
            declared_shape = ?input.lower().shape(),
            "Carried BaB witness shape does not match the declared input; ignoring it"
        );
        return Ok(None);
    }

    let point = ArrayD::from_shape_vec(IxDyn(&witness.input_shape), witness.input.clone())?;

    // Box membership. The sub-box adv_check searched is contained in the root
    // box by construction, but the organizer checks membership against the
    // DECLARED bounds, so re-check rather than assume. Reject (never clamp) —
    // a clamped point is a different point and would need its own forward.
    //
    // The BOUNDS must be finite too, not just the coordinate. A degenerate
    // declared box (an endpoint that is ±inf or NaN — e.g. from an
    // OpaqueSkip-tainted or malformed spec) would otherwise vacuously admit a
    // carried point that the pre-existing sampler could never have reached,
    // making the accepted-point set LARGER than today's instead of a strict
    // subset of it. `x < lo || x > hi` is already false against a NaN bound,
    // so the comparison alone does not catch it.
    let outside = point
        .iter()
        .zip(input.lower().iter().zip(input.upper().iter()))
        .any(|(x, (lo, hi))| {
            !x.is_finite() || !lo.is_finite() || !hi.is_finite() || x < lo || x > hi
        });
    if outside {
        tracing::info!("Carried BaB witness lies outside the declared input box; ignoring it");
        return Ok(None);
    }

    // Re-evaluate. The carried output is a hint, never the decision.
    let output = evaluate_model(model_net, &point)?;
    if witness.output.len() == output.len() {
        let drift = witness
            .output
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        if drift > 0.0 {
            tracing::debug!(
                drift,
                "Carried BaB witness output vs. fresh confirmation forward"
            );
        }
    }
    if !super::check_unsafe_counterexample(&output, constraints) {
        return Ok(None);
    }
    Ok(Some((point, output)))
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
    use ny_propagate::{BabVerificationStatus, ViolationWitness};
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
            result: BabVerificationStatus::potential_violation(),
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

    // -------------------------------------------------------------------
    // #advcheck-witness: carrying the point BaB already validated.
    // -------------------------------------------------------------------

    /// y = x on one scalar input, as a sequential model.
    fn identity_model() -> BetaCrownModel {
        use ny_propagate::{layers::LinearLayer, Layer, Network};
        let identity = LinearLayer::new(ndarray::arr2(&[[1.0_f32]]), None).expect("identity");
        let mut net = Network::new();
        net.add_layer(Layer::Linear(identity));
        BetaCrownModel::Sequential(Box::new(net))
    }

    fn box_1d(lo: f32, hi: f32) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![lo]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![hi]).unwrap(),
        )
        .unwrap()
    }

    /// Unsafe iff `y_0 <= -0.5`.
    fn unsafe_below_half() -> VnnLibSpec {
        VnnLibSpec {
            output_constraints: vec![ny_onnx::vnnlib::OutputConstraint::LessEqConst(0, -0.5)],
            ..VnnLibSpec::default()
        }
    }

    fn potential_violation(witness: Option<ViolationWitness>) -> BetaCrownResult {
        BetaCrownResult {
            result: match witness {
                Some(w) => BabVerificationStatus::potential_violation_with(w),
                None => BabVerificationStatus::potential_violation(),
            },
            domains_explored: 41,
            time_elapsed: Duration::from_millis(123),
            max_depth_reached: 7,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 0,
        }
    }

    fn witness_at(x: f32) -> ViolationWitness {
        ViolationWitness {
            input: vec![x],
            input_shape: vec![1],
            output: vec![x],
        }
    }

    /// THE FIX. A carried point that genuinely violates is confirmed IN PLACE:
    /// the emitted counterexample is that exact point, not something the
    /// confirmer re-found by attacking the root box.
    #[test]
    fn carried_witness_is_verified_in_place_advcheck_witness() {
        let confirmed = confirm_potential_violation(
            &identity_model(),
            &box_1d(-1.0, 1.0),
            Some(&unsafe_below_half()),
            potential_violation(Some(witness_at(-0.875))),
            &BetaCrownConfig::default(),
            None,
            true,
        )
        .unwrap();

        match confirmed.result {
            BabVerificationStatus::Violated {
                counterexample,
                output,
            } => {
                assert_eq!(
                    counterexample,
                    vec![-0.875_f32],
                    "the confirmer must emit the CARRIED point, not a re-searched one"
                );
                assert_eq!(output, vec![-0.875_f32]);
            }
            other => unreachable!("expected Violated from the carried witness, got {other:?}"),
        }
        // BaB metadata is preserved exactly as on the re-search path.
        assert_eq!(confirmed.domains_explored, 41);
        assert_eq!(confirmed.max_depth_reached, 7);
    }

    /// A carried point that does NOT satisfy the property must change nothing.
    /// Asserted the only way that means anything: the outcome is bit-identical
    /// to the payloadless run (same fixed-seed re-search, same verdict).
    #[test]
    fn non_violating_carried_witness_behaves_exactly_as_today_advcheck_witness() {
        let run = |witness: Option<ViolationWitness>| {
            confirm_potential_violation(
                &identity_model(),
                &box_1d(-1.0, 1.0),
                Some(&unsafe_below_half()),
                potential_violation(witness),
                &BetaCrownConfig::default(),
                None,
                true,
            )
            .unwrap()
            .result
        };

        let today = run(None);
        // +0.9 is inside the box but SAFE (0.9 > -0.5).
        assert_eq!(
            run(Some(witness_at(0.9))),
            today,
            "a carried point that fails the property must leave the verdict untouched"
        );
    }

    /// A carried point outside the DECLARED input box is ignored (never
    /// clamped into one), so the run is again identical to today.
    #[test]
    fn out_of_box_carried_witness_is_ignored_advcheck_witness() {
        let run = |witness: Option<ViolationWitness>| {
            confirm_potential_violation(
                &identity_model(),
                &box_1d(-1.0, 1.0),
                Some(&unsafe_below_half()),
                potential_violation(witness),
                &BetaCrownConfig::default(),
                None,
                true,
            )
            .unwrap()
            .result
        };

        let today = run(None);
        // -9.0 WOULD satisfy `y <= -0.5`, but it is not in [-1, 1]: the
        // organizer checks membership on the declared bounds, so it is not a
        // counterexample and must not be emitted as one.
        assert_eq!(run(Some(witness_at(-9.0))), today);
        // Wrong shape is likewise ignored rather than reinterpreted.
        assert_eq!(
            run(Some(ViolationWitness {
                input: vec![-0.9, -0.9],
                input_shape: vec![2],
                output: vec![-0.9],
            })),
            today
        );
        // A non-finite coordinate can never be a witness.
        assert_eq!(run(Some(witness_at(f32::NAN))), today);
    }

    /// An exhausted deadline downgrades to Unknown WHETHER OR NOT a witness is
    /// carried — the deadline bail-out precedes the witness check, unchanged
    /// from before this feature.
    ///
    /// This pins the ordering deliberately. An earlier draft ran the carried
    /// witness first, so an expired deadline could still return `Violated`; but
    /// `gate_result_at_deadline` rewrites that to `Timeout` one frame up, so the
    /// behaviour never reached production and the code merely implied it did.
    #[test]
    fn an_expired_deadline_downgrades_with_or_without_a_carried_witness() {
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let run = |witness: Option<ViolationWitness>| {
            confirm_potential_violation(
                &identity_model(),
                &box_1d(-1.0, 1.0),
                Some(&unsafe_below_half()),
                potential_violation(witness),
                &BetaCrownConfig::default(),
                Some(expired),
                true,
            )
            .unwrap()
            .result
        };

        // -0.75 is a genuine violating point; past the deadline it still must
        // not become a verdict here.
        assert_eq!(
            run(Some(witness_at(-0.75))),
            run(None),
            "past the deadline a carried witness must downgrade exactly like no witness"
        );
        assert!(
            matches!(
                run(Some(witness_at(-0.75))),
                BabVerificationStatus::Unknown { .. }
            ),
            "the expired-deadline path downgrades to Unknown"
        );
    }

    /// No-VnnLib PotentialViolation passes through unchanged.
    #[test]
    fn no_vnnlib_passes_through() {
        let result = BetaCrownResult {
            result: BabVerificationStatus::potential_violation(),
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
            matches!(
                confirmed.result,
                BabVerificationStatus::PotentialViolation { .. }
            ),
            "Expected PotentialViolation (pass-through) but got {:?}",
            confirmed.result
        );
    }
}
