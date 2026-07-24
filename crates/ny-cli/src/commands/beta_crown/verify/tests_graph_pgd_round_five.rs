// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PGD gradient ascent direction regression test (#4285).
//!
//! Verifies that PGD gradient steps push toward constraint *satisfaction*
//! in VNN-LIB semantics. VNN-LIB constraints describe the UNSAFE region —
//! satisfying the constraints means the point IS a counterexample to safety.
//! Therefore gradient ASCENT on satisfaction margin is correct.

use super::*;
use ny_onnx::vnnlib::OutputConstraint;

/// Regression test for #4285: PGD gradient steps converge toward counterexamples.
///
/// Network: y = 5*x + 2.5
/// Input: [-1, 1], output range: [-2.5, 7.5]
/// Spec: (assert (<= Y_0 0.0)) → UNSAFE when y <= 0
///
/// The counterexample region (y <= 0) is x in [-1, -0.5] (25% of domain).
/// Satisfaction margin = 0 - y = -5x - 2.5. Gradient ≈ -5.
/// Gradient ascent: x += step_size * (-5) → x decreases → y decreases → toward violation.
///
/// With 3 restarts and 100 steps, PGD should converge into the unsafe region
/// via gradient ascent on the satisfaction margin.
#[test]
fn test_pgd_gradient_ascent_finds_counterexample_4285() {
    // y = 5*x + 2.5: steep positive slope means the gradient is large.
    let graph = make_single_linear_graph(5.0, 2.5);
    let input = make_interval_input(-1.0, 1.0);
    // Spec: Y_0 <= 0 (VNN-LIB unsafe region). Counterexample when y <= 0 (x <= -0.5).
    let spec = make_upper_bound_spec(-1.0, 1.0, 0.0);

    // 3 restarts, 100 steps: gradient ascent on margin should reliably find x <= -0.5.
    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        3,
        100,
        Default::default(),
        20,
        None,
        None,
        true,
        false,
    )
    .expect("graph PGD should not error")
    .expect("PGD should find a counterexample in the unsafe region (y <= 0)");

    let (counterexample, output) = result;
    let y = output.iter().next().copied().unwrap_or(f32::NAN);

    // The counterexample output must satisfy the unsafe constraint (y <= 0).
    assert!(
        y <= 0.0,
        "#4285 regression: PGD should find an unsafe point (y <= 0), got y = {y}"
    );

    // With gradient ascent on margin, PGD should converge to x near -1.0
    // (the boundary of the domain where y is most negative).
    let x_val = counterexample[[0]];
    assert!(
        x_val <= -0.5,
        "#4285 regression: gradient ascent should push x toward -1.0, got x = {x_val}"
    );
}

/// #4419: Verify that `independent_graph_forward` produces the same output as
/// `evaluate_graph` with engine=None, confirming the re-validation path is
/// functionally equivalent to the attack evaluator on CPU.
#[test]
fn test_independent_graph_forward_matches_evaluate_graph_4419() {
    let graph = make_single_linear_graph(3.0, -1.0);
    let point = arr1(&[0.5_f32]).into_dyn();

    let engine_output = evaluate_graph(&graph, &point, None)
        .expect("evaluate_graph with engine=None should succeed");
    let independent_output = independent_graph_forward(&graph, &point)
        .expect("independent_graph_forward should succeed");

    assert_eq!(
        engine_output, independent_output,
        "#4419: independent_graph_forward must produce identical output to evaluate_graph(engine=None)"
    );
}

/// #4419: Verify that `independent_graph_forward` produces approximately the
/// same output as `evaluate_graph` with a GemmEngine. Small floating-point
/// differences are expected because the engine and fallback CPU paths use
/// different GEMM implementations. This is exactly the kind of drift that
/// independent re-validation is designed to catch for larger models.
#[test]
fn test_independent_graph_forward_close_to_engine_evaluation_4419() {
    let graph = make_single_linear_graph(2.0, 0.5);
    let point = arr1(&[-0.75_f32]).into_dyn();
    let engine = NaiveCpuGemmEngine;

    let with_engine = evaluate_graph(&graph, &point, Some(&engine))
        .expect("evaluate_graph with engine should succeed");
    let independent = independent_graph_forward(&graph, &point)
        .expect("independent_graph_forward should succeed");

    // Outputs should be approximately equal (within f32 precision), but
    // may differ slightly due to different GEMM code paths.
    for (&a, &b) in with_engine.iter().zip(independent.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "#4419: engine and independent outputs should be close, got {a} vs {b}"
        );
    }
}

/// #4419: End-to-end test that a graph PGD counterexample returned by
/// `try_graph_pgd_upfront` has been independently re-validated.
///
/// The returned output should match what `independent_graph_forward` produces,
/// NOT the raw PGD evaluator output. This catches the case where PGD's
/// engine-based evaluation diverges from the CPU-only re-validation.
#[test]
fn test_graph_pgd_returned_output_matches_independent_revalidation_4419() {
    // y = 2*x: unsafe region is y <= 0 (x <= 0). 50% of [-1, 1] is unsafe.
    let graph = make_single_linear_graph(2.0, 0.0);
    let input = make_interval_input(-1.0, 1.0);
    let spec = make_upper_bound_spec(-1.0, 1.0, 0.0);

    let (counterexample, returned_output) = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        5,
        20,
        Default::default(),
        0,
        None,
        None,
        true,
        false,
    )
    .expect("graph PGD should not error")
    .expect("PGD should find a counterexample (50% of domain is unsafe)");

    // The returned output must match an independent forward pass.
    let revalidated = independent_graph_forward(&graph, &counterexample)
        .expect("independent re-validation should succeed");

    assert_eq!(
        returned_output, revalidated,
        "#4419 regression: graph PGD must return the independently re-validated output, \
         not the attack evaluator output"
    );

    // Sanity: the counterexample must actually satisfy the constraint.
    assert!(
        super::super::check_unsafe_counterexample(
            &revalidated,
            &[OutputConstraint::LessEqConst(0, 0.0)]
        ),
        "#4419: re-validated output must satisfy the unsafe constraint"
    );
}
