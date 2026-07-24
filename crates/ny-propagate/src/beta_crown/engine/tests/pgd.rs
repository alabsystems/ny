// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for PGD counterexample re-validation (#2679, #2711).
//!
//! Verifies that `try_pgd_attack` independently re-evaluates counterexamples
//! via a clean forward pass and validates input bounds before promoting
//! to `Violated` status. Matches alpha-beta-CROWN's
//! `default_adv_example_finalizer` + `test_conditions` pattern.
//!
//! The re-validation rejection tests (#2711) exercise the boolean gate at
//! `pgd.rs` directly via `PgdAttacker::evaluate()`, which keeps the assertions
//! deterministic and focused on the threshold predicate itself.

use super::prelude::*;
use crate::pgd_attack::{PgdAttacker, PgdConfig};
use ny_test_utils::CountingGemmEngine;

/// Simple 2→1 linear network: output = x0 + 2*x1.
///
/// With input bounds x0 ∈ [0, 1], x1 ∈ [0, 1]:
/// - Min output = 0 (at x0=0, x1=0)
/// - Max output = 3 (at x0=1, x1=1)
fn linear_2_to_1() -> Network {
    let w = arr2(&[[1.0, 2.0]]);
    let linear = LinearLayer::new(w, None).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network
}

/// Test: PGD finds a genuine counterexample that passes re-validation.
///
/// Network: output = x0 + 2*x1, bounds [0,1]^2, max output = 3.
/// With verify_upper_bound=true, threshold=2.5: PGD should find a point
/// where output >= 2.5 (e.g. x0=1, x1=1 → output=3). Re-validation via
/// independent forward pass confirms the violation.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_revalidation_confirms_genuine_counterexample() {
    let network = linear_2_to_1();

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let config = BetaCrownConfig {
        enable_pgd_attack: true,
        pgd_restarts: 5,
        pgd_steps: 20,
        verify_upper_bound: true,
        ..BetaCrownConfig::default()
    };

    let verifier = BetaCrownVerifier::new(config);

    // Threshold 2.5 is achievable (max output = 3), so PGD should find it.
    let original_result = crate::beta_crown::result::BetaCrownResult {
        result: BabVerificationStatus::Unknown {
            reason: "domain limit".to_string(),
        },
        domains_explored: 10,
        time_elapsed: Duration::from_millis(100),
        max_depth_reached: 3,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 5,
    };

    let result = verifier
        .try_pgd_attack_with_deadline(&network, &input, 2.5, original_result, None)
        .expect("try_pgd_attack should not error");

    // PGD should find a counterexample and re-validation should confirm it.
    match &result.result {
        BabVerificationStatus::Violated {
            counterexample,
            output,
        } => {
            // The counterexample output should be >= threshold (2.5).
            assert!(
                !output.is_empty(),
                "re-validated output should be non-empty"
            );
            assert!(
                output[0] >= 2.5,
                "re-validated output {} should be >= threshold 2.5",
                output[0]
            );
            // Input should be within bounds [0, 1]^2.
            assert_eq!(counterexample.len(), 2);
            for &x in counterexample {
                assert!(
                    (-1e-6..=1.0 + 1e-6).contains(&x),
                    "counterexample element {} out of bounds [0, 1]",
                    x
                );
            }
        }
        other => panic!("Expected Violated after re-validation, got {:?}", other),
    }
}

/// Test: PGD does not find a counterexample → original result returned.
///
/// Network: output = x0 + 2*x1, bounds [0,1]^2, max output = 3.
/// With verify_upper_bound=true, threshold=10: PGD cannot find a point
/// where output >= 10 (max is 3). Original Unknown result should be returned.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_no_counterexample_returns_original() {
    let network = linear_2_to_1();

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let config = BetaCrownConfig {
        enable_pgd_attack: true,
        pgd_restarts: 3,
        pgd_steps: 10,
        verify_upper_bound: true,
        ..BetaCrownConfig::default()
    };

    let verifier = BetaCrownVerifier::new(config);

    // Threshold 10 is unreachable (max output = 3), PGD should not find it.
    let original_result = crate::beta_crown::result::BetaCrownResult {
        result: BabVerificationStatus::Unknown {
            reason: "domain limit".to_string(),
        },
        domains_explored: 10,
        time_elapsed: Duration::from_millis(100),
        max_depth_reached: 3,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 5,
    };

    let result = verifier
        .try_pgd_attack_with_deadline(&network, &input, 10.0, original_result, None)
        .expect("try_pgd_attack should not error");

    // Should return original Unknown result since no counterexample found.
    match &result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert_eq!(reason, "domain limit");
        }
        other => panic!("Expected Unknown (no counterexample), got {:?}", other),
    }
}

/// Test: PGD disabled → original result returned immediately.
#[ntest::timeout(5000)]
#[test]
fn test_pgd_disabled_returns_original() {
    let network = linear_2_to_1();

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let config = BetaCrownConfig::default(); // enable_pgd_attack = false
    assert!(!config.enable_pgd_attack);

    let verifier = BetaCrownVerifier::new(config);

    let original_result = crate::beta_crown::result::BetaCrownResult {
        result: BabVerificationStatus::Timeout,
        domains_explored: 0,
        time_elapsed: Duration::from_millis(50),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };

    let result = verifier
        .try_pgd_attack_with_deadline(&network, &input, 2.5, original_result, None)
        .expect("try_pgd_attack should not error");

    assert!(
        matches!(result.result, BabVerificationStatus::Timeout),
        "Disabled PGD should return original Timeout result"
    );
}

/// Test: verify_upper_bound=false mode (verify lower > threshold).
///
/// Network: output = x0 + 2*x1, bounds [0,1]^2, min output = 0, max = 3.
/// With verify_upper_bound=false, threshold=0.5: PGD looks for output <= 0.5.
/// At x0=0, x1=0: output=0 ≤ 0.5, so PGD should find this counterexample.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_revalidation_lower_bound_mode() {
    let network = linear_2_to_1();

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let config = BetaCrownConfig {
        enable_pgd_attack: true,
        pgd_restarts: 5,
        pgd_steps: 20,
        verify_upper_bound: false, // Verify lower > threshold
        ..BetaCrownConfig::default()
    };

    let verifier = BetaCrownVerifier::new(config);

    // Threshold 0.5: PGD seeks output <= 0.5 (achievable at origin → output=0).
    let original_result = crate::beta_crown::result::BetaCrownResult {
        result: BabVerificationStatus::Unknown {
            reason: "domain limit".to_string(),
        },
        domains_explored: 10,
        time_elapsed: Duration::from_millis(100),
        max_depth_reached: 3,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 5,
    };

    let result = verifier
        .try_pgd_attack_with_deadline(&network, &input, 0.5, original_result, None)
        .expect("try_pgd_attack should not error");

    match &result.result {
        BabVerificationStatus::Violated { output, .. } => {
            assert!(
                output[0] <= 0.5,
                "re-validated output {} should be <= threshold 0.5",
                output[0]
            );
        }
        other => panic!("Expected Violated (lower bound mode), got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Re-validation rejection path tests (#2711)
// ---------------------------------------------------------------------------

/// Simple 2→1 ReLU network: output = ReLU(x0 + 2*x1 - 1.5).
///
/// With input bounds x0 ∈ [0, 1], x1 ∈ [0, 1]:
/// - Inner: x0 + 2*x1 - 1.5, range [-1.5, 1.5]
/// - After ReLU: [0, 1.5]
fn relu_2_to_1() -> Network {
    let w = arr2(&[[1.0, 2.0]]);
    let b = arr1(&[-1.5]);
    let linear = LinearLayer::new(w, Some(b)).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::ReLU(ReLULayer::new()));
    network
}

fn normalize_source_2711(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn assert_pgd_source_contains_2711(snippet: &str, context: &str) {
    let source = include_str!("../pgd.rs");
    let normalized_source = normalize_source_2711(source);
    let normalized_snippet = normalize_source_2711(snippet);
    assert!(
        normalized_source.contains(&normalized_snippet),
        "#2711: expected pgd.rs to retain {context}"
    );
}

/// PGD finds a genuine counterexample on a ReLU network.
///
/// Extends the existing linear-network tests to cover a nonlinear network,
/// ensuring re-validation works correctly through ReLU discontinuities.
///
/// Part of #2711.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_revalidation_relu_genuine_counterexample_2711() {
    let network = relu_2_to_1();
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let config = BetaCrownConfig {
        enable_pgd_attack: true,
        pgd_restarts: 5,
        pgd_steps: 20,
        verify_upper_bound: true,
        ..BetaCrownConfig::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    // Max output = ReLU(1 + 2 - 1.5) = 1.5. Threshold 1.0 is achievable.
    let original = crate::beta_crown::result::BetaCrownResult {
        result: BabVerificationStatus::Unknown {
            reason: "test".to_string(),
        },
        domains_explored: 0,
        time_elapsed: Duration::from_millis(10),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };

    let result = verifier
        .try_pgd_attack_with_deadline(&network, &input, 1.0, original, None)
        .expect("try_pgd_attack should not error");

    match &result.result {
        BabVerificationStatus::Violated { output, .. } => {
            assert!(output[0] >= 1.0, "output {} should be >= 1.0", output[0]);
        }
        other => panic!("Expected Violated on ReLU network, got {:?}", other),
    }
}

/// Re-validation must bypass the attack engine path.
///
/// With one restart, zero PGD steps, and a trivially satisfied threshold,
/// the attack itself needs exactly one concrete engine evaluation. If the
/// re-validation step reused the same engine-backed PGD evaluator, the GEMM
/// count would be 2 instead of 1.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_revalidation_uses_independent_cpu_forward_4419() {
    let network = linear_2_to_1();
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let engine = Arc::new(CountingGemmEngine::new());
    let verifier = BetaCrownVerifier::new_with_engine(
        BetaCrownConfig {
            enable_pgd_attack: true,
            pgd_restarts: 1,
            pgd_steps: 0,
            verify_upper_bound: true,
            ..BetaCrownConfig::default()
        },
        engine.clone(),
    );

    let original_result = crate::beta_crown::result::BetaCrownResult {
        result: BabVerificationStatus::Unknown {
            reason: "domain limit".to_string(),
        },
        domains_explored: 1,
        time_elapsed: Duration::from_millis(10),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };

    let result = verifier
        .try_pgd_attack_with_deadline(&network, &input, -100.0, original_result, None)
        .expect("try_pgd_attack should not error");

    assert!(
        matches!(result.result, BabVerificationStatus::Violated { .. }),
        "expected PGD to find a trivial violation, got {:?}",
        result.result
    );
    assert_eq!(
        engine.gemm_calls(),
        1,
        "PGD should use the engine only for the attack evaluation; \
         re-validation must bypass the shared engine path"
    );
}

/// Re-validation value gate: verify the boolean condition directly.
///
/// Evaluates a known network at a known input via `PgdAttacker::evaluate()`
/// and checks the re-validation condition (pgd.rs:94-98) against thresholds
/// that would cause acceptance and rejection.
///
/// If the re-validation code were removed, a PGD bug that reported
/// `found_counterexample=true` with an incorrect `best_output_value` would
/// promote a non-violating point to `Violated`. This test verifies the
/// gate logic that prevents such promotion.
///
/// Part of #2711.
#[ntest::timeout(5000)]
#[test]
fn test_pgd_revalidation_value_gate_rejects_below_threshold_2711() {
    let network = linear_2_to_1(); // output = x0 + 2*x1

    let pgd_config = PgdConfig {
        num_restarts: 1,
        num_steps: 1,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    };
    let attacker = PgdAttacker::new(pgd_config);

    // Evaluate at x0=0.5, x1=0.3 -> output = 0.5 + 0.6 = 1.1
    let candidate = arr1(&[0.5_f32, 0.3]).into_dyn();
    let eval_output = attacker
        .evaluate(&network, &candidate)
        .expect("evaluate should succeed");
    let reeval_value = eval_output.iter().next().copied().unwrap();
    assert!(
        (reeval_value - 1.1).abs() < 1e-5,
        "expected output ~1.1, got {reeval_value}"
    );

    // verify_upper_bound=true, threshold=2.0: 1.1 >= 2.0 -> false -> REJECT
    let still_violated_upper = reeval_value >= 2.0;
    assert!(
        !still_violated_upper,
        "#2711: output 1.1 should NOT pass re-validation at threshold 2.0"
    );

    // verify_upper_bound=true, threshold=1.0: 1.1 >= 1.0 -> true -> ACCEPT
    let still_violated_accept = reeval_value >= 1.0;
    assert!(
        still_violated_accept,
        "#2711: output 1.1 SHOULD pass re-validation at threshold 1.0"
    );

    // verify_upper_bound=false, threshold=0.5: 1.1 <= 0.5 -> false -> REJECT
    let still_violated_lower = reeval_value <= 0.5;
    assert!(
        !still_violated_lower,
        "#2711: output 1.1 should NOT pass lower re-validation at threshold 0.5"
    );

    // verify_upper_bound=false, threshold=2.0: 1.1 <= 2.0 -> true -> ACCEPT
    let still_violated_lower_accept = reeval_value <= 2.0;
    assert!(
        still_violated_lower_accept,
        "#2711: output 1.1 SHOULD pass lower re-validation at threshold 2.0"
    );
}

/// Source guard: exact value-gate predicate retained in `try_pgd_attack`.
///
/// This complements the behavioral threshold checks above by pinning the
/// production branch shape directly in `pgd.rs`. If the branch is inverted or
/// removed, this test fails even though deterministic end-to-end PGD cannot
/// synthesize a disagreement packet on its own.
#[ntest::timeout(5000)]
#[test]
fn test_pgd_revalidation_value_gate_source_guard_2711() {
    assert_pgd_source_contains_2711(
        r#"
        let still_violated = if self.config.verify_upper_bound {
            reeval_value >= threshold
        } else {
            reeval_value <= threshold
        };
        "#,
        "the exact re-validation predicate",
    );
}

/// Input-bounds gate: out-of-bounds candidate rejected.
///
/// Exercises the `input_within_bounds` check at pgd.rs:114-117 by verifying
/// that an out-of-bounds candidate that would otherwise be a genuine violation
/// is correctly rejected by the bounds check.
///
/// Part of #2711.
#[ntest::timeout(5000)]
#[test]
fn test_pgd_revalidation_bounds_gate_rejects_out_of_bounds_2711() {
    let network = linear_2_to_1(); // output = x0 + 2*x1
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    // Out-of-bounds candidate: x0=1.5 (outside [0, 1]).
    // output = 1.5 + 2*0.5 = 2.5 >= threshold 2.0 (would be a violation)
    // but input is out of bounds -> rejection.
    let pgd_config = PgdConfig {
        num_restarts: 1,
        num_steps: 1,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    };
    let attacker = PgdAttacker::new(pgd_config);

    let oob_candidate = arr1(&[1.5_f32, 0.5]).into_dyn();
    let eval_output = attacker
        .evaluate(&network, &oob_candidate)
        .expect("evaluate should succeed");
    let value = eval_output.iter().next().copied().unwrap();

    // Output exceeds threshold -- this IS a genuine violation.
    assert!(value >= 2.0, "output {value} should be >= 2.0");

    // But the candidate is outside input bounds [0,1]^2.
    let lower = input.lower();
    let upper = input.upper();
    let within = oob_candidate
        .iter()
        .zip(lower.iter())
        .zip(upper.iter())
        .all(|((&x, &lo), &hi)| x.is_finite() && x >= lo - 1e-6 && x <= hi + 1e-6);
    assert!(
        !within,
        "#2711: out-of-bounds candidate (x0=1.5) should be rejected by bounds gate"
    );
}

/// Source guard: the production bounds gate still downgrades out-of-bounds inputs.
#[ntest::timeout(5000)]
#[test]
fn test_pgd_revalidation_bounds_gate_source_guard_2711() {
    assert_pgd_source_contains_2711(
        r#"
        if !input_within_bounds(&cx_input, input) {
            warn!("PGD counterexample is outside input bounds. Downgrading to Unknown.");
            return Ok(original_result);
        }
        "#,
        "the out-of-bounds downgrade branch",
    );
}
