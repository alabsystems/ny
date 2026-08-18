// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared constraint evaluation helpers for VNN-LIB verification.
//!
//! Provides evaluation functions that consume the planning types from
//! `constraint_plan` and produce verification results. Used by both
//! `beta_crown verify` and `bench_acasxu` to eliminate duplicated
//! evaluation logic.
//!
//! Part of #1881: CLI verification semantics unification.

use anyhow::{Context, Result};
use ndarray::Array2;
use ny_propagate::layers::LinearLayer;
use ny_propagate::{BabVerificationStatus, Layer as PropLayer, Network};

use super::constraint_plan::ConstantConstraintParams;

/// Compute the effective threshold for constant-constraint verification.
///
/// Both modes verify the original scalar output directly:
///
/// - unsafe `Y >= c`: upper-bound mode proves `upper(Y) < c`;
/// - unsafe `Y <= c`: lower-bound mode proves `lower(Y) > c`.
///
/// The planning layer already rounds `c` toward the stricter proof endpoint.
/// Negating either the threshold or the objective in lower-bound mode would
/// change the obligation and can turn an actually unsafe `Y <= c` region into
/// a false safety proof.
pub(crate) fn compute_effective_threshold(params: &ConstantConstraintParams) -> f32 {
    params.threshold
}

/// Build a one-hot objective vector for constant-constraint verification.
///
/// Returns a vector of length `num_outputs` with `1.0` at the target output
/// index in both modes. The verifier configuration selects either
/// `upper(Y) < c` or `lower(Y) > c`.
/// Used by graph/GPU BaB paths that take a flat objective vector.
pub(crate) fn build_constant_objective(
    params: &ConstantConstraintParams,
    num_outputs: usize,
) -> Vec<f32> {
    let mut objective = vec![0.0f32; num_outputs];
    if params.output_idx < objective.len() {
        objective[params.output_idx] = 1.0;
    }
    objective
}

/// Build a specification layer for constant-constraint sequential verification.
///
/// Returns a coefficient vector selecting the target output directly in both
/// modes. Direction belongs exclusively to `BetaCrownConfig::verify_upper_bound`.
///
/// This is used by sequential (non-graph) paths that append a LinearLayer to
/// the network. Graph paths use [`build_constant_objective`] instead.
pub(crate) fn build_constant_spec_coeffs(
    params: &ConstantConstraintParams,
    num_outputs: usize,
) -> Vec<f32> {
    let mut coeffs = vec![0.0f32; num_outputs];
    if params.output_idx < coeffs.len() {
        coeffs[params.output_idx] = 1.0;
    }
    coeffs
}

/// Augment a sequential network with a specification layer.
///
/// Clones the network and appends a `LinearLayer` that computes the linear
/// combination `c @ Y` where `c` is the specification coefficient vector.
/// This transforms the network output from `Y` to a scalar `c @ Y` so that
/// BaB verification can check `c @ Y > threshold`.
///
/// Used for both constant constraints (via [`build_constant_spec_coeffs`])
/// and relational constraints (via `RelationalObjective::spec_coeffs`).
pub(crate) fn augment_network_with_spec(
    network: &Network,
    spec_coeffs: Vec<f32>,
) -> Result<Network> {
    let num_outputs = spec_coeffs.len();
    let spec_weight = Array2::from_shape_vec((1, num_outputs), spec_coeffs)
        .context("Failed to create spec weight for augmented network")?;
    let spec_layer = LinearLayer::new(spec_weight, None)?;
    let mut augmented = network.clone();
    augmented.add_layer(PropLayer::Linear(spec_layer));
    Ok(augmented)
}

/// Aggregate per-constraint results using conjunctive semantics.
///
/// For a conjunctive unsafe region (AND of constraints), the property is SAFE
/// if ANY single constraint is provably violated. This function implements
/// the early-exit aggregation logic:
///
/// - `Verified` → return Verified immediately (constraint violated = safe)
/// - `Timeout` → return Timeout immediately (budget exhausted)
/// - `Violated`/`PotentialViolation`/`Unknown` → continue to next constraint
/// - All exhausted → return Unknown
///
/// Returns `(status, total_domains_explored, total_domains_verified)`.
pub(crate) fn aggregate_conjunctive(
    results: &[(BabVerificationStatus, usize, usize)],
) -> (BabVerificationStatus, usize, usize) {
    let mut total_domains = 0usize;
    let mut total_verified = 0usize;

    for (status, domains, verified) in results {
        total_domains += domains;
        total_verified += verified;

        match status {
            BabVerificationStatus::Verified => {
                return (
                    BabVerificationStatus::Verified,
                    total_domains,
                    total_verified,
                );
            }
            BabVerificationStatus::Timeout => {
                return (
                    BabVerificationStatus::Timeout,
                    total_domains,
                    total_verified,
                );
            }
            BabVerificationStatus::Violated { .. }
            | BabVerificationStatus::PotentialViolation { .. }
            | BabVerificationStatus::Unknown { .. } => {
                // Constraint may hold; continue checking others
            }
        }
    }

    (
        BabVerificationStatus::Unknown {
            reason: "All relational constraints may hold".to_string(),
        },
        total_domains,
        total_verified,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_propagate::BabVerificationStatus;

    #[test]
    fn test_compute_effective_threshold_upper() {
        let params = ConstantConstraintParams {
            threshold: 3.99,
            verify_upper: true,
            output_idx: 0,
        };
        assert!((compute_effective_threshold(&params) - 3.99).abs() < 1e-6);
    }

    #[test]
    fn test_compute_effective_threshold_lower() {
        let params = ConstantConstraintParams {
            threshold: 3.99,
            verify_upper: false,
            output_idx: 0,
        };
        assert!((compute_effective_threshold(&params) - 3.99).abs() < 1e-6);
    }

    #[test]
    fn test_build_constant_objective() {
        let params = ConstantConstraintParams {
            threshold: 1.0,
            verify_upper: true,
            output_idx: 2,
        };
        let obj = build_constant_objective(&params, 5);
        assert_eq!(obj, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_build_constant_objective_lower_bound() {
        // Graph paths always select the original output; the mode checks lower > c.
        let params = ConstantConstraintParams {
            threshold: 1.0,
            verify_upper: false,
            output_idx: 1,
        };
        let obj = build_constant_objective(&params, 3);
        assert_eq!(obj, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_build_constant_spec_coeffs_upper() {
        let params = ConstantConstraintParams {
            threshold: 1.0,
            verify_upper: true,
            output_idx: 2,
        };
        let coeffs = build_constant_spec_coeffs(&params, 5);
        assert_eq!(coeffs, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_build_constant_spec_coeffs_lower() {
        let params = ConstantConstraintParams {
            threshold: 1.0,
            verify_upper: false,
            output_idx: 2,
        };
        let coeffs = build_constant_spec_coeffs(&params, 5);
        assert_eq!(coeffs, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_build_constant_objective_out_of_range() {
        let params = ConstantConstraintParams {
            threshold: 1.0,
            verify_upper: true,
            output_idx: 10, // out of range
        };
        let obj = build_constant_objective(&params, 5);
        assert_eq!(obj, vec![0.0; 5]);
    }

    #[test]
    fn test_augment_network_with_spec() {
        // Create a minimal network with one linear layer
        let weight = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let layer = LinearLayer::new(weight, None).unwrap();
        let mut network = Network::new();
        network.add_layer(PropLayer::Linear(layer));

        // Augment with spec coefficients [1.0, -1.0, 0.0] (compute Y_0 - Y_1)
        let augmented = augment_network_with_spec(&network, vec![1.0, -1.0, 0.0]).unwrap();
        assert_eq!(augmented.layers().len(), 2);
    }

    #[test]
    fn test_aggregate_conjunctive_verified_first() {
        let results = vec![
            (BabVerificationStatus::Verified, 100, 50),
            (
                BabVerificationStatus::Unknown {
                    reason: "x".to_string(),
                },
                200,
                0,
            ),
        ];
        let (status, domains, verified) = aggregate_conjunctive(&results);
        assert!(matches!(status, BabVerificationStatus::Verified));
        assert_eq!(domains, 100);
        assert_eq!(verified, 50);
    }

    #[test]
    fn test_aggregate_conjunctive_timeout() {
        let results = vec![
            (
                BabVerificationStatus::Unknown {
                    reason: "x".to_string(),
                },
                100,
                0,
            ),
            (BabVerificationStatus::Timeout, 50, 0),
        ];
        let (status, domains, _) = aggregate_conjunctive(&results);
        assert!(matches!(status, BabVerificationStatus::Timeout));
        assert_eq!(domains, 150);
    }

    #[test]
    fn test_aggregate_conjunctive_all_unknown() {
        let results = vec![
            (
                BabVerificationStatus::Unknown {
                    reason: "a".to_string(),
                },
                100,
                0,
            ),
            (BabVerificationStatus::potential_violation(), 50, 0),
        ];
        let (status, domains, _) = aggregate_conjunctive(&results);
        assert!(matches!(status, BabVerificationStatus::Unknown { .. }));
        assert_eq!(domains, 150);
    }

    #[test]
    fn test_aggregate_conjunctive_empty() {
        let results: Vec<(BabVerificationStatus, usize, usize)> = vec![];
        let (status, domains, _) = aggregate_conjunctive(&results);
        assert!(matches!(status, BabVerificationStatus::Unknown { .. }));
        assert_eq!(domains, 0);
    }

    #[test]
    fn test_aggregate_conjunctive_verified_after_unknown() {
        let results = vec![
            (BabVerificationStatus::potential_violation(), 100, 0),
            (
                BabVerificationStatus::Unknown {
                    reason: "x".to_string(),
                },
                50,
                0,
            ),
            (BabVerificationStatus::Verified, 200, 100),
        ];
        let (status, domains, verified) = aggregate_conjunctive(&results);
        assert!(matches!(status, BabVerificationStatus::Verified));
        assert_eq!(domains, 350);
        assert_eq!(verified, 100);
    }

    // ======================================================================
    // Semantic parity regression tests (#1881 Step 5)
    //
    // These tests verify that the shared evaluation helpers produce identical
    // results to the inline logic they replaced in bench_acasxu.rs and verify.rs.
    // ======================================================================

    /// #1881: Aggregation parity — Violated status continues (doesn't short-circuit).
    ///
    /// The old bench_acasxu inline code matched Violated to "continue checking"
    /// rather than returning immediately. Verify shared aggregate_conjunctive
    /// preserves this semantics: Violated should NOT cause early exit.
    #[test]
    fn test_aggregate_parity_violated_continues_1881() {
        let results = vec![
            (
                BabVerificationStatus::Violated {
                    counterexample: vec![0.1, 0.2],
                    output: vec![0.5],
                },
                50,
                25,
            ),
            (BabVerificationStatus::Verified, 100, 50),
        ];
        let (status, domains, verified) = aggregate_conjunctive(&results);
        // Violated at index 0 means constraint CAN hold, so we continue.
        // Verified at index 1 means a constraint was provably violated = safe.
        assert!(
            matches!(status, BabVerificationStatus::Verified),
            "Should reach Verified at index 1 after Violated at index 0"
        );
        assert_eq!(domains, 150, "Should accumulate domains from both");
        assert_eq!(verified, 75, "Should accumulate verified from both");
    }

    /// #1881: Aggregation parity — only Verified and Timeout cause early exit.
    ///
    /// The old inline code in bench_acasxu had:
    /// - Verified → return immediately
    /// - Timeout → return immediately
    /// - Violated/PotentialViolation/Unknown → continue
    ///
    /// Verify that ALL non-terminating statuses continue to the end.
    #[test]
    fn test_aggregate_parity_all_non_terminating_continue_1881() {
        let results = vec![
            (BabVerificationStatus::potential_violation(), 10, 0),
            (
                BabVerificationStatus::Violated {
                    counterexample: vec![],
                    output: vec![],
                },
                20,
                0,
            ),
            (
                BabVerificationStatus::Unknown {
                    reason: "solver gave up".to_string(),
                },
                30,
                0,
            ),
        ];
        let (status, domains, verified) = aggregate_conjunctive(&results);
        assert!(
            matches!(status, BabVerificationStatus::Unknown { .. }),
            "All non-terminating → final Unknown"
        );
        assert_eq!(domains, 60, "Should accumulate all three");
        assert_eq!(verified, 0);
    }

    /// #1881: spec coefficients parity — sequential constant upper-bound.
    ///
    /// Previously `verify_constant` in bench_acasxu built coefficients inline:
    /// `coeffs[output_idx] = if verify_upper { 1.0 } else { -1.0 }`
    /// Verify `build_constant_spec_coeffs` matches this exactly.
    #[test]
    fn test_spec_coeffs_parity_constant_upper_1881() {
        let params = ConstantConstraintParams {
            threshold: 3.99,
            verify_upper: true,
            output_idx: 0,
        };
        let coeffs = build_constant_spec_coeffs(&params, 5);
        // Old inline: coeffs[0] = 1.0 (verify_upper is true)
        assert_eq!(coeffs[0], 1.0);
        assert_eq!(coeffs[1], 0.0);
        assert_eq!(coeffs[2], 0.0);
        assert_eq!(coeffs[3], 0.0);
        assert_eq!(coeffs[4], 0.0);
    }

    /// The lower-bound lane must still select `+Y`: its mode proves
    /// `lower(Y) > c`. The historical `-Y > -c`/lower-mode combination proved
    /// the opposite inequality and could falsely verify an unsafe region.
    #[test]
    fn test_spec_coeffs_parity_constant_lower_1881() {
        let params = ConstantConstraintParams {
            threshold: 3.99,
            verify_upper: false,
            output_idx: 0,
        };
        let coeffs = build_constant_spec_coeffs(&params, 5);
        assert_eq!(coeffs[0], 1.0);
        let eff = compute_effective_threshold(&params);
        assert!((eff - 3.99).abs() < 1e-6);
    }

    /// #1881: graph objective parity — constant constraint always uses +1.0.
    ///
    /// Both graph modes select `+Y`; the verifier mode, not threshold/objective
    /// negation, determines which complement is proved.
    #[test]
    fn test_graph_objective_parity_constant_1881() {
        // Upper bound: objective should be +1.0
        let upper = ConstantConstraintParams {
            threshold: 3.99,
            verify_upper: true,
            output_idx: 2,
        };
        let obj_upper = build_constant_objective(&upper, 5);
        assert_eq!(obj_upper[2], 1.0);

        // Lower bound: objective and threshold stay in the original Y space.
        let lower = ConstantConstraintParams {
            threshold: 3.99,
            verify_upper: false,
            output_idx: 2,
        };
        let obj_lower = build_constant_objective(&lower, 5);
        assert_eq!(
            obj_lower[2], 1.0,
            "Graph objective must select the original output in both modes"
        );
        assert_eq!(compute_effective_threshold(&lower), 3.99);
    }

    /// #1881: augmented network parity — spec layer has correct shape.
    ///
    /// The old inline code created:
    /// `Array2::from_shape_vec((1, num_outputs), coeffs)`
    /// Verify `augment_network_with_spec` produces a network with the same
    /// layer count and shape characteristics.
    #[test]
    fn test_augmented_network_parity_shape_1881() {
        // 2 inputs -> 3 outputs linear layer
        let weight = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let layer = LinearLayer::new(weight, None).unwrap();
        let mut network = Network::new();
        network.add_layer(PropLayer::Linear(layer));

        // Augment with relational coefficients [1.0, -1.0, 0.0] (Y_0 - Y_1)
        let augmented = augment_network_with_spec(&network, vec![1.0, -1.0, 0.0]).unwrap();

        // Old code: network.clone() + add_layer(PropLayer::Linear(spec_layer))
        // Should have original + 1 spec layer
        assert_eq!(augmented.layers().len(), 2);

        // Original network should be unchanged (verify clone semantics)
        assert_eq!(network.layers().len(), 1);
    }

    /// #1881: Aggregation preserves exact domain/verified counts on early exit.
    ///
    /// When Verified is hit at constraint i, the returned domain count must
    /// include domains from constraints 0..=i (inclusive), not just constraint i.
    /// This matches the old inline accumulation loop behavior.
    #[test]
    fn test_aggregate_parity_domain_accumulation_on_early_exit_1881() {
        let results = vec![
            (
                BabVerificationStatus::Unknown {
                    reason: "a".to_string(),
                },
                100,
                10,
            ),
            (
                BabVerificationStatus::Unknown {
                    reason: "b".to_string(),
                },
                200,
                20,
            ),
            (BabVerificationStatus::Verified, 300, 30),
            // This should never be reached due to early exit
            (
                BabVerificationStatus::Unknown {
                    reason: "c".to_string(),
                },
                400,
                40,
            ),
        ];
        let (status, domains, verified) = aggregate_conjunctive(&results);
        assert!(matches!(status, BabVerificationStatus::Verified));
        // 100 + 200 + 300 = 600 (constraint 3 at index 3 not reached)
        assert_eq!(domains, 600);
        // 10 + 20 + 30 = 60
        assert_eq!(verified, 60);
    }
}
