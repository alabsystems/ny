// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for spec-guided CROWN precheck (#3384).

use super::attack_budget::upfront_pgd_deadline;
use super::disjunctive_pgd::{
    classify_disjunctive_attack, try_disjunctive_sampling_attack, DisjunctiveAttackKind,
};
use super::disjunctive_precheck::{
    build_spec_row, clause_provably_unsat, crown_precheck_clauses, is_clause_unsatisfiable,
    SpecRowKind,
};
use super::pgd_sampling::spsa_step_deadline_exceeded;
use super::try_pgd_before_mip;
use super::BetaCrownModel;
use ndarray::{arr1, arr2, Array1, Array2};
use ny_onnx::vnnlib::parse_vnnlib;
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{layers::LinearLayer, BabVerificationStatus, Layer, Network};
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;
use std::time::{Duration, Instant};

/// #3384: SpecRowKind::DiffUpperNeg correctly identifies UNSAT when upper < 0.
#[test]
fn spec_row_diff_upper_neg_unsat_when_upper_negative() {
    let kind = SpecRowKind::DiffUpperNeg;
    // upper < 0 → UNSAT
    assert!(kind.is_unsatisfiable(-2.0, -0.1));
    // upper == 0 → NOT UNSAT (boundary case, could still be SAT)
    assert!(!kind.is_unsatisfiable(-1.0, 0.0));
    // upper > 0 → NOT UNSAT
    assert!(!kind.is_unsatisfiable(-1.0, 1.0));
    // both positive → NOT UNSAT
    assert!(!kind.is_unsatisfiable(0.5, 1.0));
    // Malformed/non-finite bounds are numerical failures, never proofs.
    assert!(!kind.is_unsatisfiable(f32::NEG_INFINITY, f32::NEG_INFINITY));
    assert!(!kind.is_unsatisfiable(0.0, -0.1));
    assert!(!kind.is_unsatisfiable(f32::NAN, -0.1));
}

#[test]
fn interval_clause_prechecks_reject_nonfinite_or_inverted_boxes() {
    let clause = [OutputConstraint::LessEqConst(0, 0.0)];
    let infinite = BoundedTensor::new_allow_infinite(
        arr1(&[f32::INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .unwrap();
    assert!(!clause_provably_unsat(&infinite, &clause));

    for (lower, upper) in [
        (
            arr1(&[f32::INFINITY]).into_dyn(),
            arr1(&[f32::INFINITY]).into_dyn(),
        ),
        (arr1(&[2.0]).into_dyn(), arr1(&[1.0]).into_dyn()),
        (arr1(&[1.0, 2.0]).into_dyn(), arr1(&[2.0]).into_dyn()),
    ] {
        assert!(!is_clause_unsatisfiable(&clause, &lower, &upper));
    }
}

#[test]
fn interval_clause_prechecks_use_exact_f64_constants() {
    let lower = arr1(&[1.0_f32]).into_dyn();
    let upper = arr1(&[1.0_f32]).into_dyn();
    let just_below_one = f64::from_bits(1.0_f64.to_bits() - 1);
    let just_above_one = f64::from_bits(1.0_f64.to_bits() + 1);

    assert!(is_clause_unsatisfiable(
        &[OutputConstraint::LessEqConst(0, just_below_one)],
        &lower,
        &upper
    ));
    assert!(is_clause_unsatisfiable(
        &[OutputConstraint::GreaterEqConst(0, just_above_one)],
        &lower,
        &upper
    ));
    assert!(!is_clause_unsatisfiable(
        &[OutputConstraint::LessEqConst(0, f64::NAN)],
        &lower,
        &upper
    ));
}

/// #3384: build_spec_row for LessEq encodes Y_j - Y_i and returns DiffUpperNeg.
#[test]
fn build_spec_row_lesseq_encodes_yj_minus_yi() {
    let clause = vec![OutputConstraint::LessEq(0, 1)]; // Y_0 <= Y_1
    let num_outputs = 2;
    let mut row_data = vec![0.0f32; num_outputs];
    let row = ndarray::ArrayViewMut1::from(&mut row_data);
    let kind = build_spec_row(&clause, num_outputs, row).unwrap();
    // Row should encode Y_1 - Y_0: row[1]=1, row[0]=-1
    assert_eq!(row_data[0], -1.0);
    assert_eq!(row_data[1], 1.0);
    assert!(matches!(kind, SpecRowKind::DiffUpperNeg));
}

/// #3384: build_spec_row for GreaterEq encodes Y_i - Y_j and returns DiffUpperNeg.
#[test]
fn build_spec_row_greatereq_encodes_yi_minus_yj() {
    let clause = vec![OutputConstraint::GreaterEq(0, 1)]; // Y_0 >= Y_1
    let num_outputs = 2;
    let mut row_data = vec![0.0f32; num_outputs];
    let row = ndarray::ArrayViewMut1::from(&mut row_data);
    let kind = build_spec_row(&clause, num_outputs, row).unwrap();
    // Row should encode Y_0 - Y_1: row[0]=1, row[1]=-1
    assert_eq!(row_data[0], 1.0);
    assert_eq!(row_data[1], -1.0);
    assert!(matches!(kind, SpecRowKind::DiffUpperNeg));
}

/// Helper: build a constant-output network (1 input -> N outputs, weight=0).
fn make_const_network(bias: Vec<f32>) -> Network {
    let n = bias.len();
    let w = Array2::from_shape_vec((n, 1), vec![0.0f32; n]).unwrap();
    let b = Array1::from_vec(bias);
    let linear = LinearLayer::new(w, Some(b)).unwrap();
    let mut net = Network::new();
    net.add_layer(Layer::Linear(linear));
    net
}

fn make_equal_output_network() -> Network {
    let weight = Array2::from_shape_vec((2, 1), vec![1.0f32, 1.0f32]).unwrap();
    let linear = LinearLayer::new(weight, None).unwrap();
    let mut net = Network::new();
    net.add_layer(Layer::Linear(linear));
    net
}

fn make_flat_then_activate_network() -> Network {
    let mut net = Network::new();
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, 0.0_f32]]), Some(arr1(&[-0.95_f32]))).unwrap(),
    ));
    net.add_layer(Layer::ReLU(Default::default()));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[100.0_f32]]), None).unwrap(),
    ));
    net
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} length mismatch: actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (idx, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual_value - expected_value).abs() < 1e-6,
            "{label}[{idx}] mismatch: actual={actual_value} expected={expected_value}"
        );
    }
}

#[test]
fn pgd_before_mip_strict_equal_outputs_not_short_circuited_3779() {
    let net = make_equal_output_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).unwrap();
    let vnnlib = parse_vnnlib(
        "\
(declare-const X_0 Real)\n\
(declare-const Y_0 Real)\n\
(declare-const Y_1 Real)\n\
(assert (>= X_0 -1.0))\n\
(assert (<= X_0 1.0))\n\
(assert (< Y_0 Y_1))\n",
    )
    .unwrap();

    let precheck = try_pgd_before_mip(
        &net,
        &input,
        &vnnlib,
        4,
        4,
        Default::default(),
        20,
        None,
        false,
        None,
        true,
    )
    .unwrap();
    assert!(
        precheck.confirmed_counterexample.is_none(),
        "strict equality network should not short-circuit MIP through PGD"
    );
}

#[test]
fn pgd_before_mip_multi_clause_mixed_input_spec_skipped_3779() {
    let net = make_equal_output_network();
    let input = BoundedTensor::new(arr1(&[0.5f32]).into_dyn(), arr1(&[0.5f32]).into_dyn()).unwrap();
    let vnnlib = parse_vnnlib(
        "\
(declare-const X_0 Real)\n\
(declare-const Y_0 Real)\n\
(declare-const Y_1 Real)\n\
(assert (>= X_0 0.5))\n\
(assert (<= X_0 0.5))\n\
(assert (or\n\
  (and (<= X_0 0.25) (>= Y_0 Y_1))\n\
  (and (>= X_0 0.75) (<= Y_0 Y_1))\n\
))\n",
    )
    .unwrap();

    let precheck = try_pgd_before_mip(
        &net,
        &input,
        &vnnlib,
        4,
        4,
        Default::default(),
        20,
        None,
        false,
        None,
        true,
    )
    .unwrap();
    assert!(
        precheck.confirmed_counterexample.is_none(),
        "multi-clause specs with clause-specific input bounds must not short-circuit MIP through conjunctive PGD"
    );
    assert!(
        precheck.warm_start_candidate.is_none(),
        "multi-clause specs should not produce a warm-start candidate"
    );
}

/// #3865: PGD precheck keeps warm-start candidate when no counterexample exists.
///
/// Network: Y_0 = x + 2, Y_1 = x - 2 (gap of 4).
/// Unsafe spec: Y_0 <= Y_1, i.e. x+2 <= x-2 → 4 <= 0 — impossible.
/// PGD cannot find a counterexample, but the classified PgdAttacker still runs
/// and produces a best-candidate input. The precheck should preserve that
/// candidate for MIP warm-starting.
#[test]
fn pgd_before_mip_preserves_warm_start_candidate_3865() {
    // Y_0 = x + 2, Y_1 = x - 2. Y_0 - Y_1 = 4 > 0 always, so Y_0 <= Y_1
    // can never hold. PGD will fail to confirm a counterexample.
    let weight = Array2::from_shape_vec((2, 1), vec![1.0f32, 1.0f32]).unwrap();
    let bias = Array1::from_vec(vec![2.0, -2.0]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();
    let mut net = Network::new();
    net.add_layer(Layer::Linear(linear));

    let input =
        BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).unwrap();
    let vnnlib = parse_vnnlib(
        "\
(declare-const X_0 Real)\n\
(declare-const Y_0 Real)\n\
(declare-const Y_1 Real)\n\
(assert (>= X_0 -1.0))\n\
(assert (<= X_0 1.0))\n\
(assert (<= Y_0 Y_1))\n",
    )
    .unwrap();

    let precheck = try_pgd_before_mip(
        &net,
        &input,
        &vnnlib,
        10,
        10,
        Default::default(),
        20,
        None,
        false,
        None,
        true,
    )
    .unwrap();
    assert!(
        precheck.confirmed_counterexample.is_none(),
        "Y_0 = x+2 > Y_1 = x-2 always; PGD should not find a counterexample"
    );
    // The classified LessEq(0,1) attack should still run and populate a candidate.
    assert!(
        precheck.warm_start_candidate.is_some(),
        "PGD should preserve the best candidate input even without a counterexample"
    );
}

#[test]
fn upfront_pgd_deadline_reserves_twenty_percent_of_total_timeout_3781() {
    let start = Instant::now();
    let deadline = upfront_pgd_deadline(start, 150).expect("finite timeout should cap PGD");

    assert_eq!(deadline.duration_since(start), Duration::from_secs(30));
}

#[test]
fn upfront_pgd_deadline_unbounded_timeout_remains_open_3781() {
    assert!(upfront_pgd_deadline(Instant::now(), 0).is_none());
}

#[test]
fn spsa_deadline_check_fires_every_step_2206() {
    let expired = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );
    let future = Some(Instant::now() + Duration::from_secs(1));

    // After #2206 Packet D: deadline is checked every step, not every 50th.
    // Graph SPSA steps are expensive (seconds per step on large models), so
    // coarse deadline polling caused PGD to overshoot its 20% budget by 100+s.
    assert!(spsa_step_deadline_exceeded(0, expired));
    assert!(spsa_step_deadline_exceeded(48, expired));
    assert!(spsa_step_deadline_exceeded(49, expired));
    assert!(spsa_step_deadline_exceeded(50, expired));
    assert!(!spsa_step_deadline_exceeded(49, future));
    assert!(!spsa_step_deadline_exceeded(0, None));
}

/// #3384 regression: LessEq constraint that is always UNSAT -> precheck returns true.
///
/// Network: Y_0=5.0, Y_1=0.0 (constant). Clause: Y_0 <= Y_1.
/// Y_0=5 > Y_1=0 always -> Y_0 <= Y_1 can never hold -> UNSAT.
#[test]
fn spec_guided_precheck_lesseq_always_unsat_returns_true_3384() {
    let net = make_const_network(vec![5.0, 0.0]);
    let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).unwrap();
    let clauses = vec![vec![OutputConstraint::LessEq(0, 1)]]; // Y_0 <= Y_1

    let result = crown_precheck_clauses(
        &BetaCrownModel::Sequential(Box::new(net)),
        &input,
        &clauses,
        &[],
        None,
        None,
    );
    assert_eq!(result.len(), 1);
    assert!(
        result[0],
        "LessEq(0,1) with Y_0=5 > Y_1=0 should be UNSAT (precheck true)"
    );
}

/// #3384 regression: LessEq constraint that is always SAT -> precheck returns false.
///
/// Network: Y_0=0.0, Y_1=5.0 (constant). Clause: Y_0 <= Y_1.
/// Y_0=0 < Y_1=5 always -> Y_0 <= Y_1 always holds -> SAT -> precheck false.
///
/// Before #3384 fix, the inverted check (lower > 0 on Y_1-Y_0) would
/// incorrectly return true (declaring SAT as UNSAT -> false safety verdict).
#[test]
fn spec_guided_precheck_lesseq_always_sat_returns_false_3384() {
    let net = make_const_network(vec![0.0, 5.0]);
    let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).unwrap();
    let clauses = vec![vec![OutputConstraint::LessEq(0, 1)]]; // Y_0 <= Y_1

    let result = crown_precheck_clauses(
        &BetaCrownModel::Sequential(Box::new(net)),
        &input,
        &clauses,
        &[],
        None,
        None,
    );
    assert_eq!(result.len(), 1);
    assert!(
        !result[0],
        "LessEq(0,1) with Y_0=0 < Y_1=5 should be SAT (precheck false)"
    );
}

/// #3384: GreaterEq constraint that is always UNSAT -> precheck returns true.
///
/// Network: Y_0=0.0, Y_1=5.0 (constant). Clause: Y_0 >= Y_1.
/// Y_0=0 < Y_1=5 -> Y_0 >= Y_1 can never hold -> UNSAT.
#[test]
fn spec_guided_precheck_greatereq_unsat_returns_true_3384() {
    let net = make_const_network(vec![0.0, 5.0]);
    let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).unwrap();
    let clauses = vec![vec![OutputConstraint::GreaterEq(0, 1)]]; // Y_0 >= Y_1

    let result = crown_precheck_clauses(
        &BetaCrownModel::Sequential(Box::new(net)),
        &input,
        &clauses,
        &[],
        None,
        None,
    );
    assert_eq!(result.len(), 1);
    assert!(
        result[0],
        "GreaterEq(0,1) with Y_0=0 < Y_1=5 should be UNSAT (precheck true)"
    );
}

/// #3397 regression: disjunctive CROWN precheck must honor an expired deadline.
///
/// Network:
/// - hidden = [ReLU(x), ReLU(-x)]
/// - y0 = 1.1 - hidden[0] - hidden[1]
/// - y1 = 0
///
/// For x in [-1, 1], the clause y0 <= y1 is always UNSAT because
/// y1 - y0 = |x| - 1.1 <= -0.1. Spec-guided CROWN proves that by coupling the
/// two ReLUs, but plain IBP on the augmented network is too loose:
/// hidden[i] in [0, 1] independently gives upper(|x| - 1.1) <= 0.9.
///
/// With an already-expired deadline, the precheck must skip the expensive CROWN
/// work and fall back to the cheap bound path, which leaves the clause
/// unverified instead of spending unbounded time in the precheck.
#[test]
fn crown_precheck_expired_deadline_disables_tight_relational_proof_3397() {
    let first_weight = Array2::from_shape_vec((2, 1), vec![1.0, -1.0]).unwrap();
    let first = LinearLayer::new(first_weight, None).unwrap();

    let second_weight = Array2::from_shape_vec((2, 2), vec![-1.0, -1.0, 0.0, 0.0]).unwrap();
    let second_bias = Array1::from_vec(vec![1.1, 0.0]);
    let second = LinearLayer::new(second_weight, Some(second_bias)).unwrap();

    let mut net = Network::new();
    net.add_layer(Layer::Linear(first));
    net.add_layer(Layer::ReLU(Default::default()));
    net.add_layer(Layer::Linear(second));

    let input =
        BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).unwrap();
    let clauses = vec![vec![OutputConstraint::LessEq(0, 1)]];

    let without_deadline = crown_precheck_clauses(
        &BetaCrownModel::Sequential(Box::new(net.clone())),
        &input,
        &clauses,
        &[],
        None,
        None,
    );
    assert_eq!(without_deadline, vec![true]);

    let expired_deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );
    let with_expired_deadline = crown_precheck_clauses(
        &BetaCrownModel::Sequential(Box::new(net)),
        &input,
        &clauses,
        &[],
        None,
        expired_deadline,
    );
    assert_eq!(
        with_expired_deadline,
        vec![false],
        "expired deadline should skip tight CROWN precheck work and leave the clause for the BaB path"
    );
}

/// #3218 sanity check: without a deadline, the disjunctive sampler should report
/// a trivially satisfied clause immediately.
#[test]
fn disjunctive_sampling_attack_finds_trivial_counterexample_without_deadline_3218() {
    let net = make_const_network(vec![1.0, 0.0]);
    let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).unwrap();
    let clauses = vec![vec![OutputConstraint::GreaterEq(0, 1)]];

    let result = try_disjunctive_sampling_attack(
        &BetaCrownModel::Sequential(Box::new(net)),
        &input,
        &clauses,
        1,
        1,
        Default::default(),
        20,
        None,
        true,
        None,
        false,
    )
    .expect("disjunctive sampling attack should not error")
    .expect("disjunctive sampling attack should find the trivially SAT clause");

    match result.result {
        BabVerificationStatus::Violated {
            counterexample,
            output,
        } => {
            assert_close(&counterexample, &[0.0], "counterexample");
            assert_close(&output, &[1.0, 0.0], "output");
        }
        other => panic!("expected violated result, got {other:?}"),
    }
}

/// #3218 regression: an already-expired deadline must skip restart 0 instead of
/// evaluating even a trivially SAT clause.
#[test]
fn disjunctive_sampling_attack_expired_deadline_skips_restart_zero_3218() {
    let net = make_const_network(vec![1.0, 0.0]);
    let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).unwrap();
    let clauses = vec![vec![OutputConstraint::GreaterEq(0, 1)]];
    let expired_deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );

    let result = try_disjunctive_sampling_attack(
        &BetaCrownModel::Sequential(Box::new(net)),
        &input,
        &clauses,
        1,
        1,
        Default::default(),
        20,
        None,
        true,
        expired_deadline,
        false,
    )
    .expect("disjunctive sampling attack should not error");

    assert!(
        result.is_none(),
        "expired deadline should skip restart 0 before the random sample is evaluated"
    );
}

#[test]
fn disjunctive_sampling_attack_threads_gemm_engine_for_sequential_pgd_3954() {
    let net = make_const_network(vec![1.0, 0.0]);
    let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).unwrap();
    let clauses = vec![vec![OutputConstraint::GreaterEq(0, 1)]];
    let engine = CountingGemmEngine::new();

    let result = try_disjunctive_sampling_attack(
        &BetaCrownModel::Sequential(Box::new(net)),
        &input,
        &clauses,
        1,
        1,
        Default::default(),
        20,
        Some(&engine),
        true,
        None,
        false,
    )
    .expect("disjunctive sampling attack should not error")
    .expect("disjunctive sampling attack should find the trivially SAT clause");

    assert!(
        matches!(result.result, BabVerificationStatus::Violated { .. }),
        "sequential disjunctive PGD should still report the trivial counterexample"
    );
    assert!(
        engine.gemm_calls() > 0,
        "#3954 regression: disjunctive sequential PGD should thread GemmEngine into IBP evaluation"
    );
}

/// #4375: generic disjunctive PGD must fail closed on out-of-range output indices.
#[test]
fn disjunctive_sampling_attack_generic_oob_const_index_rejected_4375() {
    let net = make_const_network(vec![1.0]);
    let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).unwrap();
    let clauses = vec![vec![OutputConstraint::GreaterEqConst(1, 0.0)]];

    let result = try_disjunctive_sampling_attack(
        &BetaCrownModel::Sequential(Box::new(net)),
        &input,
        &clauses,
        1,
        0,
        Default::default(),
        20,
        None,
        true,
        None,
        false,
    )
    .expect("disjunctive sampling attack should not error on an OOB clause");

    assert!(
        result.is_none(),
        "OOB output coordinates must not produce a disjunctive counterexample"
    );
}

/// #4375: the generic disjunctive SAT path must re-confirm witnesses with the
/// epsilon guard before returning `Violated`.
#[test]
fn disjunctive_sampling_attack_generic_borderline_margin_rejected_4375() {
    let net = make_const_network(vec![9.0e-6]);
    let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).unwrap();
    let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.0)]];

    let result = try_disjunctive_sampling_attack(
        &BetaCrownModel::Sequential(Box::new(net)),
        &input,
        &clauses,
        1,
        0,
        Default::default(),
        20,
        None,
        true,
        None,
        false,
    )
    .expect("disjunctive sampling attack should not error on a borderline clause");

    assert!(
        result.is_none(),
        "margin below the 1e-5 guard must be rejected after witness re-confirmation"
    );
}

/// #4375: the sequential disjunctive PGD fast path must reject borderline
/// witnesses using the same epsilon guard.
#[test]
fn disjunctive_sampling_attack_sequential_borderline_margin_rejected_4375() {
    let net = make_const_network(vec![9.0e-6, 0.0]);
    let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[0.0f32]).into_dyn()).unwrap();
    let clauses = vec![vec![OutputConstraint::GreaterEq(0, 1)]];

    let result = try_disjunctive_sampling_attack(
        &BetaCrownModel::Sequential(Box::new(net)),
        &input,
        &clauses,
        1,
        1,
        Default::default(),
        20,
        None,
        true,
        None,
        false,
    )
    .expect("sequential disjunctive PGD should not error on a borderline clause");

    assert!(
        result.is_none(),
        "sequential disjunctive PGD must reject margins below the 1e-5 guard"
    );
}

#[test]
fn disjunctive_sampling_attack_restart_when_stuck_recovers_generic_spsa_4278() {
    let net = make_flat_then_activate_network();
    let input = BoundedTensor::new(
        arr1(&[0.0f32, 0.0f32]).into_dyn(),
        arr1(&[1.0f32, 1.0f32]).into_dyn(),
    )
    .unwrap();
    let clauses = vec![vec![OutputConstraint::GreaterEqConst(0, 0.5)]];

    let without_restart = try_disjunctive_sampling_attack(
        &BetaCrownModel::Sequential(Box::new(net.clone())),
        &input,
        &clauses,
        1,
        3,
        Default::default(),
        20,
        None,
        true,
        None,
        false,
    )
    .expect("disjunctive SPSA without restart_when_stuck should not error");
    assert!(
        without_restart.is_none(),
        "without restart_when_stuck, the generic disjunctive SPSA path should stay pinned in the flat dead region"
    );

    let with_restart = try_disjunctive_sampling_attack(
        &BetaCrownModel::Sequential(Box::new(net)),
        &input,
        &clauses,
        1,
        3,
        Default::default(),
        20,
        None,
        true,
        None,
        true,
    )
    .expect("disjunctive SPSA with restart_when_stuck should not error")
    .expect("restart_when_stuck should recover a generic disjunctive SPSA witness");

    match with_restart.result {
        BabVerificationStatus::Violated {
            counterexample,
            output,
        } => {
            assert!(
                counterexample[0] > 0.99,
                "restart_when_stuck should escape the flat dead region and reach the active half-space, got x={counterexample:?}"
            );
            assert!(
                output[0] >= 0.5,
                "restart_when_stuck should return a confirmed witness above the clause threshold, got output={output:?}"
            );
        }
        other => {
            panic!("expected violated result after restart_when_stuck recovery, got {other:?}")
        }
    }
}

#[test]
fn classify_disjunctive_attack_detects_shared_rhs_greater_eq_traffic_signs_3218() {
    let clauses = vec![
        vec![OutputConstraint::GreaterEq(0, 28)],
        vec![OutputConstraint::GreaterEq(1, 28)],
        vec![OutputConstraint::GreaterEq(42, 28)],
    ];

    let classification =
        classify_disjunctive_attack(&clauses).expect("traffic_signs-style clauses should classify");

    match classification {
        DisjunctiveAttackKind::AnyComparisonGeTarget {
            target,
            comparisons,
        } => {
            assert_eq!(target, 28);
            assert_eq!(comparisons, vec![0, 1, 42]);
        }
        other => panic!("unexpected classification: {other:?}"),
    }
}

#[test]
fn classify_disjunctive_attack_rejects_multi_constraint_clauses_3218() {
    let clauses = vec![
        vec![
            OutputConstraint::GreaterEq(0, 28),
            OutputConstraint::GreaterEq(1, 28),
        ],
        vec![OutputConstraint::GreaterEq(2, 28)],
    ];

    assert!(
        classify_disjunctive_attack(&clauses).is_none(),
        "multi-constraint clauses must stay on the generic disjunctive path"
    );
}
