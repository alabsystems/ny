// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{
    check_unsafe_counterexample, constraint_is_safety_proof, disjunctive_failure_to_final_status,
    finalize_relational_status, verify_relational_constraints, AggregationMode, BetaCrownModel,
};
use ndarray::{arr1, arr2, Array1, Array2};
use ny_onnx::vnnlib::{parse_vnnlib, VnnLibSpec};
use ny_propagate::{
    beta_crown::{BetaCrownConfig, BranchingHeuristic},
    layers::LinearLayer,
    BabVerificationStatus, BetaCrownResult, BetaCrownVerifier, GraphNetwork, GraphNode, Layer,
    Network,
};
use ny_tensor::BoundedTensor;
use std::time::Duration;

#[test]
fn beta_crown_multi_clause_disjunction_aggregates_unknown() {
    let vnnlib = parse_vnnlib(
        r#"
(vnnlib-version <2.0>)
(declare-network disjunction
    (declare-input X Real [1])
    (declare-output Y Real [2])
)
(assert (>= X[0] 0.0))
(assert (<= X[0] 0.0))
(assert (or
    (and (<= Y[1] Y[0]) (>= Y[0] Y[1]))
    (and (<= Y[0] Y[1]) (>= Y[1] Y[0]))
))
"#,
    )
    .unwrap();

    assert!(vnnlib.has_multi_constraint_disjunction());
    assert_eq!(vnnlib.output_constraint_clauses.len(), 2);
    assert!(vnnlib
        .output_constraint_clauses
        .iter()
        .all(|clause| clause.len() == 2));

    let weight = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
    let bias = Array1::from_vec(vec![0.0, 1.0]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[0.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 1,
        max_depth: 1,
        timeout: Duration::from_secs(1),
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config.clone());

    let result = verify_relational_constraints(
        &BetaCrownModel::Sequential(Box::new(network)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        false, // use_relu_split
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        1,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .unwrap();

    assert!(matches!(
        result.result,
        BabVerificationStatus::Unknown { .. }
    ));
}

#[test]
fn relational_counterexample_is_not_safety_proof_1890() {
    assert!(constraint_is_safety_proof(&BabVerificationStatus::Verified));
    assert!(!constraint_is_safety_proof(
        &BabVerificationStatus::Violated {
            counterexample: vec![0.1],
            output: vec![0.2],
        }
    ));
}

#[test]
fn disjunctive_counterexample_preserves_unsafe_verdict_1890() {
    let status = disjunctive_failure_to_final_status(
        &BabVerificationStatus::Violated {
            counterexample: vec![0.3, 0.4],
            output: vec![1.2],
        },
        "Y_0 <= Y_1",
    );

    match status {
        BabVerificationStatus::Violated {
            counterexample,
            output,
        } => {
            assert_eq!(counterexample, vec![0.3, 0.4]);
            assert_eq!(output, vec![1.2]);
        }
        other => unreachable!("expected Violated verdict, got {:?}", other),
    }
}

#[test]
fn finalize_relational_status_requires_verified_constraints_1890() {
    let status = finalize_relational_status(AggregationMode::Conjunctive, 0, 2, 2);
    assert!(
        matches!(status, BabVerificationStatus::Unknown { .. }),
        "conjunctive result with only non-verified constraints must not be SAFE"
    );
}

#[test]
fn finalize_relational_status_disjunction_requires_all_1887() {
    let status = finalize_relational_status(AggregationMode::Disjunctive, 1, 2, 2);
    assert!(
        matches!(status, BabVerificationStatus::Unknown { .. }),
        "disjunctive result is SAFE only when every constraint is proved violated"
    );
}

/// Helper: build a simple linear network (1 input → N outputs) with given bias.
/// Weight is all zeros so output = bias regardless of input.
fn make_constant_output_network(bias: Vec<f32>) -> Network {
    let num_outputs = bias.len();
    let weight = Array2::from_shape_vec((num_outputs, 1), vec![0.0f32; num_outputs]).unwrap();
    let bias_arr = Array1::from_vec(bias);
    let linear = LinearLayer::new(weight, Some(bias_arr)).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network
}

/// Helper: create a point input at x=0.
fn make_point_input() -> BoundedTensor {
    BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).unwrap()
}

/// Helper: create an interval input in [-1, 1].
fn make_interval_input() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).unwrap()
}

/// Helper: build a 1-input -> 2-output linear network with identical rows.
fn make_equal_output_network() -> Network {
    let weight = Array2::from_shape_vec((2, 1), vec![1.0f32, 1.0f32]).unwrap();
    let linear = LinearLayer::new(weight, None).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network
}

fn make_two_output_spec(assertion: &str) -> VnnLibSpec {
    parse_vnnlib(&format!(
        "\
(declare-const X_0 Real)\n\
(declare-const Y_0 Real)\n\
(declare-const Y_1 Real)\n\
(assert (>= X_0 -1.0))\n\
(assert (<= X_0 1.0))\n\
(assert {assertion})\n"
    ))
    .unwrap()
}

fn make_dummy_reduced_result() -> BetaCrownResult {
    BetaCrownResult {
        result: BabVerificationStatus::Unknown {
            reason: "dummy reduced result".to_string(),
        },
        domains_explored: 0,
        domains_verified: 0,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: Duration::from_secs(0),
        output_bounds: None,
    }
}

/// Helper: run verify_relational_constraints with default config.
fn run_verify(network: Network, vnnlib: &VnnLibSpec) -> BetaCrownResult {
    let config = BetaCrownConfig {
        max_domains: 1,
        max_depth: 1,
        timeout: Duration::from_secs(5),
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config.clone());
    let input = make_point_input();

    verify_relational_constraints(
        &BetaCrownModel::Sequential(Box::new(network)),
        &input,
        vnnlib,
        &config,
        &verifier,
        false, // use_relu_split
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        5,     // timeout
        None,  // gemm_engine
        true,  // json (suppress output)
    )
    .unwrap()
}

/// #1889 regression: Multi-constant conjunctive spec must check ALL constants.
///
/// Network: outputs Y_0=5.0, Y_1=0.0 (constant, zero-weight).
/// Spec: Y_0 >= 3.99 AND Y_1 >= 3.99 (conjunctive unsafe region).
///
/// Y_0=5.0 >= 3.99 holds, so first constraint is NOT violated.
/// Y_1=0.0 < 3.99, so second constraint IS violated → SAFE (conjunction).
///
/// Old bug: extract_constant_params returned only Y_0, which couldn't be
/// proved violated → UNKNOWN. The fix routes multi-constant specs through
/// the per-constraint loop, which finds Y_1 < 3.99 → Verified.
#[test]
fn multi_constant_conjunctive_checks_all_constants_1889() {
    let network = make_constant_output_network(vec![5.0, 0.0]);
    let vnnlib = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 0.0))
(assert (>= Y_0 3.99))
(assert (>= Y_1 3.99))
"#,
    )
    .unwrap();

    assert_eq!(
        vnnlib.output_constraints.len(),
        2,
        "spec should have 2 constant constraints"
    );

    let result = run_verify(network, &vnnlib);
    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "multi-constant conjunctive spec should be Verified (Y_1=0 < 3.99), got {:?}",
        result.result
    );
}

/// #1889 regression: Multi-constant disjunctive spec must verify ALL constants.
///
/// Network: outputs Y_0=0.0, Y_1=0.0 (constant, zero-weight).
/// Spec: Y_0 >= 100.0 OR Y_1 >= 100.0 (disjunctive unsafe region).
///
/// Both Y_0=0.0 < 100 and Y_1=0.0 < 100 are violated.
/// Disjunction requires ALL constraints proved violated → Verified.
///
/// Old bug: only first constant checked → 1/2 proved → UNKNOWN.
#[test]
fn multi_constant_disjunctive_checks_all_constants_1889() {
    let network = make_constant_output_network(vec![0.0, 0.0]);
    // For disjunctive semantics, we need the `or` construct in VNN-LIB.
    // Single-clause approach: flat constraints with is_disjunction flag.
    // The parser sets is_disjunction when `(assert (or ...))` is used.
    let vnnlib = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 0.0))
(assert (or (>= Y_0 100.0) (>= Y_1 100.0)))
"#,
    )
    .unwrap();

    // Disjunction creates clauses
    assert!(
        vnnlib.is_disjunction || vnnlib.output_constraint_clauses.len() >= 2,
        "spec should be disjunctive or have multiple clauses"
    );

    let result = run_verify(network, &vnnlib);
    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "multi-constant disjunctive spec should be Verified (both Y < 100), got {:?}",
        result.result
    );
}

/// #1888 regression: Mixed relational+constant conjunctive spec must not drop constants.
///
/// Network: outputs Y_0=2.0, Y_1=1.0, Y_2=0.0 (constant, zero-weight).
/// Spec: Y_1 <= Y_0 AND Y_2 >= 3.99 (mixed relational+constant, conjunctive).
///
/// Y_1=1.0 <= Y_0=2.0 holds (relational NOT violated).
/// Y_2=0.0 < 3.99 (constant IS violated) → SAFE for conjunction.
///
/// Old bug: constant Y_2 >= 3.99 dropped when relational present → only
/// checks Y_1 <= Y_0 (not violated) → UNKNOWN. Fix: per-constraint loop
/// includes both → finds constant violated → Verified.
#[test]
fn mixed_relational_constant_conjunctive_checks_all_1888() {
    let network = make_constant_output_network(vec![2.0, 1.0, 0.0]);
    let vnnlib = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 0.0))
(assert (<= Y_1 Y_0))
(assert (>= Y_2 3.99))
"#,
    )
    .unwrap();

    assert_eq!(
        vnnlib.output_constraints.len(),
        2,
        "spec should have 2 constraints (1 relational + 1 constant)"
    );

    let result = run_verify(network, &vnnlib);
    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "mixed conjunctive spec should be Verified (Y_2=0 < 3.99), got {:?}",
        result.result
    );
}

/// #1888 regression: Mixed relational+constant disjunctive spec preserves all constraints.
///
/// Network: outputs Y_0=0.0, Y_1=1.0, Y_2=0.0 (constant).
/// Spec: (Y_0 <= Y_1) OR (Y_2 >= 100.0) (disjunctive unsafe).
///
/// Y_0=0.0 <= Y_1=1.0 holds → relational NOT violated (clause 1 unsafe).
/// Y_2=0.0 < 100 → constant IS violated.
/// Disjunction: SAFE requires ALL clauses violated.
/// Clause 1 (relational) cannot be proved violated → NOT safe → Unknown.
///
/// This tests that constants are NOT dropped: if the constant were dropped,
/// only the relational would be checked, and we'd get Unknown for a
/// different reason (only 1 clause checked, but it still fails).
/// The key test: the result should NOT be Verified since the relational holds.
#[test]
fn mixed_relational_constant_disjunctive_not_safe_1888() {
    let network = make_constant_output_network(vec![0.0, 1.0, 0.0]);
    let vnnlib = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 0.0))
(assert (or (<= Y_0 Y_1) (>= Y_2 100.0)))
"#,
    )
    .unwrap();

    let result = run_verify(network, &vnnlib);
    // Y_0 <= Y_1 holds (0 <= 1), so relational clause is not violated.
    // Disjunction requires ALL violated → should be Unknown, not Verified.
    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "disjunctive mixed spec should NOT be Verified when relational holds, got {:?}",
        result.result
    );
}

/// #3209: check_unsafe_counterexample validates all relational constraints jointly.
#[test]
fn check_unsafe_counterexample_all_hold_3209() {
    use ndarray::ArrayD;
    use ny_onnx::vnnlib::OutputConstraint;

    // Y_0=3.0, Y_1=1.0, Y_2=2.0
    // Constraints: Y_1 <= Y_0 AND Y_2 <= Y_0 (all hold)
    let output: ArrayD<f32> = arr1(&[3.0, 1.0, 2.0]).into_dyn();
    let constraints = vec![
        OutputConstraint::LessEq(1, 0), // Y_1 <= Y_0: 1.0 <= 3.0 ✓
        OutputConstraint::LessEq(2, 0), // Y_2 <= Y_0: 2.0 <= 3.0 ✓
    ];
    assert!(
        check_unsafe_counterexample(&output, &constraints),
        "all constraints hold → should be a valid counterexample"
    );
}

/// #3209: check_unsafe_counterexample rejects when any constraint fails.
#[test]
fn check_unsafe_counterexample_one_fails_3209() {
    use ndarray::ArrayD;
    use ny_onnx::vnnlib::OutputConstraint;

    // Y_0=1.0, Y_1=2.0, Y_2=0.5
    // Constraints: Y_1 <= Y_0 AND Y_2 <= Y_0
    // Y_1=2.0 > Y_0=1.0 → constraint 1 fails
    let output: ArrayD<f32> = arr1(&[1.0, 2.0, 0.5]).into_dyn();
    let constraints = vec![
        OutputConstraint::LessEq(1, 0), // Y_1 <= Y_0: 2.0 <= 1.0 ✗
        OutputConstraint::LessEq(2, 0), // Y_2 <= Y_0: 0.5 <= 1.0 ✓
    ];
    assert!(
        !check_unsafe_counterexample(&output, &constraints),
        "first constraint fails → not a valid joint counterexample"
    );
}

/// #4375: out-of-range output indices must fail closed instead of silently
/// comparing against 0.0.
#[test]
fn check_unsafe_counterexample_oob_index_rejected_4375() {
    use ndarray::ArrayD;
    use ny_onnx::vnnlib::OutputConstraint;

    let output: ArrayD<f32> = arr1(&[1.0]).into_dyn();
    let constraints = vec![OutputConstraint::GreaterEqConst(1, 0.0)];

    assert!(
        !check_unsafe_counterexample(&output, &constraints),
        "out-of-range output coordinates must not confirm an unsafe counterexample"
    );
}

#[test]
fn conjunctive_pgd_upfront_strict_equal_outputs_rejected_3779() {
    let network = make_equal_output_network();
    let input = make_interval_input();
    let vnnlib = make_two_output_spec("(< Y_0 Y_1)");

    let result = super::pgd::try_conjunctive_pgd_attack_upfront(
        &network,
        &input,
        &vnnlib,
        4,
        4,
        Default::default(),
        20,
        None,
        None,
        true,
    )
    .unwrap();

    assert!(
        result.is_none(),
        "strict equality network should not produce an upfront PGD counterexample"
    );
}

#[test]
fn conjunctive_pgd_follow_up_strict_equal_outputs_rejected_3779() {
    let network = make_equal_output_network();
    let input = make_interval_input();
    let vnnlib = make_two_output_spec("(< Y_0 Y_1)");
    let reduced_result = make_dummy_reduced_result();

    let result = super::pgd::try_conjunctive_pgd_attack(
        &network,
        &input,
        &vnnlib,
        &reduced_result,
        4,
        4,
        Default::default(),
        20,
        None,
        None,
        true,
    )
    .unwrap();

    assert!(
        result.is_none(),
        "strict equality network should not produce a reduced-path PGD counterexample"
    );
}

#[test]
fn conjunctive_pgd_upfront_nonstrict_equal_outputs_accepted_3779() {
    let network = make_equal_output_network();
    let input = make_interval_input();
    let vnnlib = make_two_output_spec("(<= Y_0 Y_1)");

    let result = super::pgd::try_conjunctive_pgd_attack_upfront(
        &network,
        &input,
        &vnnlib,
        4,
        4,
        Default::default(),
        20,
        None,
        None,
        true,
    )
    .unwrap();

    assert!(
        result.is_some(),
        "non-strict equality network should still produce a valid PGD counterexample"
    );
}

/// #3209: Cross-validation detects joint counterexample from per-constraint BaB.
///
/// Network: Y_0=5, Y_1=3, Y_2=4 (constant output, zero-weight).
/// Property: Y_1 <= Y_0 AND Y_2 <= Y_0 (conjunctive unsafe).
///
/// Y_1=3 <= Y_0=5 ✓ and Y_2=4 <= Y_0=5 ✓ → both hold → property VIOLATED.
/// Per-constraint BaB for constraint 1 finds a counterexample (any input works
/// since the network is constant). Cross-validation of that counterexample
/// against ALL constraints should detect this is a joint counterexample.
#[test]
fn conjunctive_cross_validation_finds_joint_counterexample_3209() {
    let network = make_constant_output_network(vec![5.0, 3.0, 4.0]);
    let vnnlib = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 0.0))
(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_0))
"#,
    )
    .unwrap();

    assert_eq!(vnnlib.output_constraints.len(), 2);

    // Run with PGD attack enabled so cross-validation closure is used
    let config = BetaCrownConfig {
        max_domains: 10,
        max_depth: 2,
        timeout: Duration::from_secs(5),
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_pgd_attack: true,
        pgd_restarts: 10,
        pgd_steps: 10,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config.clone());
    let input = make_point_input();

    let result = verify_relational_constraints(
        &BetaCrownModel::Sequential(Box::new(network)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        false, // use_relu_split
        false, // gpu_bab
        true,  // pgd_attack — enables cross-validation
        10,    // pgd_restarts
        10,    // pgd_steps
        5,     // timeout
        None,  // gemm_engine
        true,  // json (suppress output)
    )
    .unwrap();

    // The property SHOULD be detected as Violated because Y_1=3 <= Y_0=5
    // and Y_2=4 <= Y_0=5 (both constraints hold for the constant output).
    // Either the upfront conjunctive PGD or the cross-validation should catch this.
    assert!(
        matches!(result.result, BabVerificationStatus::Violated { .. }),
        "conjunctive property should be VIOLATED (all constraints hold), got {:?}",
        result.result
    );
}

/// #3209 audit: Directly exercise iterate_constraints cross-validation with eval_original.
///
/// Uses mock verify_fn (returns Violated) + mock eval_original (all constraints hold)
/// to verify cross-validation detects joint counterexamples. The end-to-end test above
/// catches counterexamples via upfront PGD before reaching this path.
#[test]
fn iterate_constraints_cross_validates_counterexample_3209() {
    use super::constraint_iter::{iterate_constraints, ConstraintIterConfig};
    use ndarray::arr1;
    use ny_onnx::vnnlib::OutputConstraint;

    let constraints = vec![
        OutputConstraint::LessEq(1, 0), // Y_1 <= Y_0
        OutputConstraint::LessEq(2, 0), // Y_2 <= Y_0
    ];
    let iter_config = ConstraintIterConfig {
        aggregation: AggregationMode::Conjunctive,
        overall_timeout: Duration::from_secs(5),
        per_constraint_timeout: Duration::from_secs(2),
        min_timeout_ms: 10,
        total_constraint_count: 2,
        num_outputs: 3,
        base_config: BetaCrownConfig {
            max_domains: 1,
            max_depth: 1,
            timeout: Duration::from_secs(2),
            ..Default::default()
        },
        parent_verifier: None,
        engine: None,
        json: true,
    };

    let mut call_count = 0usize;
    let result = iterate_constraints(
        &constraints,
        &iter_config,
        |_dispatch| {
            call_count += 1;
            Ok(BetaCrownResult {
                result: BabVerificationStatus::Violated {
                    counterexample: vec![0.5],
                    output: vec![1.0],
                },
                domains_explored: 10,
                domains_verified: 0,
                cuts_generated: 0,
                max_depth_reached: 1,
                time_elapsed: Duration::from_millis(100),
                output_bounds: None,
            })
        },
        // eval_original: Y_0=5, Y_1=3, Y_2=4 (both constraints hold)
        Some(
            &|_cx_input: &[f32]| -> anyhow::Result<ndarray::ArrayD<f32>> {
                Ok(arr1(&[5.0f32, 3.0, 4.0]).into_dyn())
            },
        ),
    )
    .unwrap();

    // Cross-validation: Y_1=3<=Y_0=5 AND Y_2=4<=Y_0=5 → all hold → VIOLATED
    assert!(matches!(
        result.result,
        BabVerificationStatus::Violated { .. }
    ));
    if let BabVerificationStatus::Violated { output, .. } = &result.result {
        assert_eq!(
            output,
            &[5.0, 3.0, 4.0],
            "output should come from eval_original"
        );
    }
    assert_eq!(
        call_count, 1,
        "cross-validation should short-circuit after first constraint"
    );
}

/// #3309: PGD attack must run for single-clause disjunctions.
///
/// soundnessbench properties are parsed as `is_disjunction=true` with
/// `output_constraint_clauses.len()==1`. Before this fix, PGD was skipped
/// because `!is_disjunction` was false, even though single-clause
/// disjunctions are semantically conjunctive (all constraints in the single
/// clause must hold simultaneously).
///
/// Network: outputs Y_0=1.0 (constant). Property: Y_0 >= 0.5 (unsafe).
/// Y_0=1.0 >= 0.5 → property violated. PGD should find this immediately.
#[test]
fn single_clause_disjunction_runs_pgd_3309() {
    use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};

    // Construct a single-clause disjunction spec manually.
    // This simulates soundnessbench VNN-LIB: `(assert (or (>= Y_0 0.5)))`
    // which is is_disjunction=true but only has 1 clause.
    let mut vnnlib = VnnLibSpec::new();
    vnnlib.num_inputs = 1;
    vnnlib.num_outputs = 1;
    vnnlib.input_bounds = vec![(0.0, 0.0)];
    vnnlib.is_disjunction = true;
    vnnlib.output_constraints = vec![OutputConstraint::GreaterEqConst(0, 0.5)];
    vnnlib.output_constraint_clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.5)]];
    vnnlib.per_clause_input_bounds = vec![Default::default()];

    // Verify the pre-condition: this IS a disjunction with single clause.
    assert!(vnnlib.is_disjunction);
    assert_eq!(vnnlib.output_constraint_clauses.len(), 1);
    // has_multi_constraint_disjunction() returns false for single clause.
    assert!(!vnnlib.has_multi_constraint_disjunction());

    // Network: constant output Y_0=1.0 (zero-weight, bias=1.0).
    let network = make_constant_output_network(vec![1.0]);
    let input = make_point_input();

    let config = BetaCrownConfig {
        max_domains: 1,
        max_depth: 1,
        timeout: Duration::from_secs(5),
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_pgd_attack: true,
        pgd_restarts: 100,
        pgd_steps: 10,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config.clone());

    let result = verify_relational_constraints(
        &BetaCrownModel::Sequential(Box::new(network)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        false, // use_relu_split
        false, // gpu_bab
        true,  // pgd_attack — MUST run despite is_disjunction=true
        100,   // pgd_restarts
        10,    // pgd_steps
        5,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .unwrap();

    // Y_0=1.0 >= 0.5, so the unsafe property holds → Violated.
    // Before fix: PGD skipped → falls into per-constraint verification.
    // After fix: PGD finds counterexample immediately.
    assert!(
        matches!(result.result, BabVerificationStatus::Violated { .. }),
        "single-clause disjunction should be VIOLATED (PGD finds Y_0=1.0 >= 0.5), got {:?}",
        result.result
    );
    // Prove PGD specifically found the violation (not downstream BaB).
    // PGD returns domains_explored=0; BaB would return domains_explored>0.
    assert_eq!(
        result.domains_explored, 0,
        "domains_explored=0 proves PGD found the violation, not BaB"
    );
}

/// Build anti-correlated ReLU network for conjunctive BaB testing (#3334).
///
/// Network: 1 input → 2 hidden (ReLU) → 2 outputs.
///   Y_0 = relu(x - 0.4), Y_1 = relu(0.6 - x). Input: x ∈ [0, 1].
///
/// Biases [-0.4, 0.6] ensure both neurons have pre-activation range [-0.4, 0.6],
/// giving CROWN lower relaxation slope alpha = 1.0 (since u=0.6 > -l=0.4).
/// This is needed for β-CROWN analytical gradients to be non-zero.
///
/// The narrow both-active region [0.4, 0.6] has max(Y_i) = 0.2 < 0.3, so each
/// objective IS individually verifiable on that sub-domain — which is exactly
/// what conjunctive BaB needs (verify ANY objective per sub-domain).
///
/// Property: Y_0 >= 0.3 AND Y_1 >= 0.3 (UNSAT — Y_0 ≥ 0.3 requires x ≥ 0.7,
///   Y_1 ≥ 0.3 requires x ≤ 0.3, so no x satisfies both).
fn build_conjunctive_bab_test_3334() -> (Network, VnnLibSpec, BoundedTensor, BetaCrownConfig) {
    use ny_propagate::layers::ReLULayer;

    let w1 = Array2::from_shape_vec((2, 1), vec![1.0f32, -1.0]).unwrap();
    let b1 = Array1::from_vec(vec![-0.4f32, 0.6]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();

    let w2 = Array2::from_shape_vec((2, 2), vec![1.0f32, 0.0, 0.0, 1.0]).unwrap();
    let b2 = Array1::from_vec(vec![0.0f32, 0.0]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer::new()));
    network.add_layer(Layer::Linear(linear2));

    let vnnlib = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= Y_0 0.3))
(assert (>= Y_1 0.3))
"#,
    )
    .unwrap();

    let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).unwrap();

    // Enable β-CROWN so the Lagrangian penalty for constrained ReLU neurons
    // tightens bounds at child domains. Without β, CROWN bounds over the full
    // input [0,1] are too loose to verify either objective.
    let config = BetaCrownConfig {
        max_domains: 100,
        max_depth: 10,
        timeout: Duration::from_secs(10),
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        beta_iterations: 20,
        ..Default::default()
    };
    (network, vnnlib, input, config)
}

fn build_graph_input_split_joint_clause_test_523(
) -> (GraphNetwork, VnnLibSpec, BoundedTensor, BetaCrownConfig) {
    let linear = LinearLayer::new(
        arr2(&[[2.0_f32, 1.0_f32], [-4.0_f32, 5.0_f32]]),
        Some(arr1(&[-0.1_f32, 2.01_f32])),
    )
    .expect("two-output linear layer should build");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");

    let vnnlib = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 0.0))
(assert (<= Y_1 0.0))
"#,
    )
    .expect("joint-clause graph spec should parse");

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("finite bounds");

    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        input_split_sb_margin_weight: 1.0,
        max_domains: 8,
        max_depth: 1,
        timeout: Duration::from_secs(1),
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        ..Default::default()
    };

    (graph, vnnlib, input, config)
}

/// #3334: Joint conjunctive BaB verifies a property that per-constraint BaB cannot.
///
/// Per-constraint BaB cannot verify this because neither constraint is universally
/// false: Y_0 >= 0.3 IS satisfiable (x=0.9) and Y_1 >= 0.3 IS satisfiable (x=0.1).
///
/// Joint conjunctive BaB verifies by splitting ReLU neurons until each sub-domain
/// has at least one objective individually verifiable:
///   - Neuron 0 inactive (x < 0.4): Y_0 = 0 < 0.3 → obj 0 verified
///   - Neuron 1 inactive (x > 0.6): Y_1 = 0 < 0.3 → obj 1 verified
///   - Both active (x ∈ [0.4, 0.6]): max(Y_i) = 0.2 < 0.3 → both objs verified
///
/// Reference: designs/2026-03-05-joint-conjunctive-bab.md Phase 2.
#[test]
fn conjunctive_bab_verifies_joint_property_that_per_constraint_cannot_3334() {
    let (network, vnnlib, input, config) = build_conjunctive_bab_test_3334();
    assert!(!vnnlib.is_disjunction, "spec should be conjunctive (AND)");
    assert_eq!(vnnlib.output_constraints.len(), 2);

    let verifier = BetaCrownVerifier::new(config.clone());
    let result = verify_relational_constraints(
        &BetaCrownModel::Sequential(Box::new(network)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        true,  // use_relu_split — triggers multi-objective conjunctive BaB
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        10,    // timeout
        None,  // gemm_engine
        true,  // json (suppress output)
    )
    .unwrap();

    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "joint conjunctive BaB should verify (conjunction impossible), got {:?}, verified={}, explored={}",
        result.result, result.domains_verified, result.domains_explored
    );
    assert!(
        result.domains_verified >= 2,
        "expected >= 2 verified domains (1 per child after ReLU split), got {}",
        result.domains_verified
    );
}

/// #523 regression: graph input splitting must reach joint multi-objective clause verification.
///
/// This graph/property pair is jointly impossible but each individual constraint remains
/// satisfiable somewhere in the input box, so per-constraint input splitting returns
/// Unknown. The CLI layer must route the conjunctive graph path to
/// `verify_graph_input_split_multi_objective_conjunctive` so the root split proves one
/// different constraint on each child.
#[test]
fn graph_input_split_conjunctive_uses_joint_multi_objective_path_523() {
    let (graph, vnnlib, input, config) = build_graph_input_split_joint_clause_test_523();
    assert_eq!(
        vnnlib.output_constraints.len(),
        2,
        "expected a 2-constraint conjunction"
    );

    let verifier = BetaCrownVerifier::new(config.clone());
    let result = verify_relational_constraints(
        &BetaCrownModel::Graph(Box::new(graph)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        false, // use_relu_split=false → input splitting
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        1,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .expect("graph input-split conjunctive verification should complete");

    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "joint graph input-split path should verify this impossible conjunction, got {:?}",
        result.result
    );
    assert_eq!(
        result.domains_explored, 1,
        "joint input-split verification should resolve both children from the root split"
    );
    assert_eq!(
        result.domains_verified, 2,
        "expected one verified child per objective after the routed joint split"
    );
}
