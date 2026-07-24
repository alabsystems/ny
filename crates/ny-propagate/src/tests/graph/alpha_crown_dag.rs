// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Targeted unit tests for DAG alpha-CROWN propagation (`propagate_dag.rs`).
//!
//! The existing `alpha_crown.rs` tests exercise DAG alpha-CROWN at integration
//! level. This module adds focused tests for:
//! - Diamond-shaped DAGs where a node fans out to multiple consumers
//! - Single-iteration vs multi-iteration alpha optimization convergence
//! - `restore_dag_alpha_snapshot` behavior after gradient perturbation errors
//! - Non-finite pre-activation bound handling
//!
//! Part of #2837.

use crate::bounds::{AlphaCrownConfig, AlphaSpecEarlyExit, GradientMethod};
use crate::*;
use ndarray::{arr1, arr2, Array1, Array2};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a diamond-shaped DAG:
///
/// ```text
/// Input
///   |
/// Linear1(2→2)
///   |
/// ReLU1
///  / \
/// L2a  L2b       (two separate Linear layers consuming relu1)
///  \  /
///  Add
///   |
/// ReLU2
/// ```
///
/// This is a true diamond: relu1 fans out to two consumers (l2a, l2b) which
/// merge at Add. This exercises DAG backward accumulation at merge nodes.
fn build_diamond_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    // Layer 1: Input -> Linear1 -> ReLU1
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let b1 = arr1(&[0.1_f32, -0.1]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    // Branch A: relu1 -> Linear2a
    let w2a = arr2(&[[0.8_f32, -0.3], [-0.2, 0.6]]);
    graph.add_node(GraphNode::new(
        "linear2a",
        Layer::Linear(LinearLayer::new(w2a, None).unwrap()),
        vec!["relu1".to_string()],
    ));

    // Branch B: relu1 -> Linear2b
    let w2b = arr2(&[[-0.4_f32, 0.7], [0.5, -0.1]]);
    graph.add_node(GraphNode::new(
        "linear2b",
        Layer::Linear(LinearLayer::new(w2b, None).unwrap()),
        vec!["relu1".to_string()],
    ));

    // Merge: Add(linear2a, linear2b)
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["linear2a".to_string(), "linear2b".to_string()],
    ));

    // Output: ReLU2
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));

    graph.set_output("relu2");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

/// Manually compute forward pass through the diamond DAG for a concrete input.
fn diamond_forward(x: &[f32; 2]) -> [f32; 2] {
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let b1 = arr1(&[0.1_f32, -0.1]);
    let w2a = arr2(&[[0.8_f32, -0.3], [-0.2, 0.6]]);
    let w2b = arr2(&[[-0.4_f32, 0.7], [0.5, -0.1]]);

    let x_arr = arr1(&[x[0], x[1]]);

    // Linear1 + bias
    let h1 = w1.dot(&x_arr) + &b1;
    // ReLU1
    let h1_relu = h1.mapv(|v| v.max(0.0));
    // Branch A
    let ha = w2a.dot(&h1_relu);
    // Branch B
    let hb = w2b.dot(&h1_relu);
    // Add
    let add_out = &ha + &hb;
    // ReLU2
    let out = add_out.mapv(|v| v.max(0.0));

    [out[0], out[1]]
}

/// Build a non-sequential DAG whose only optimizable activations are Sigmoid/Tanh.
///
/// ```text
///        /-> Linear_hidden -> Sigmoid -\
/// Input                                Add -> Tanh -> Linear_out
///        \-> Linear_skip -------------/
/// ```
///
/// This exercises the `#3619` path directly:
/// - no ReLU alpha state is available
/// - the graph is non-sequential, so `propagate_dag.rs` is used
/// - both Sigmoid and Tanh tangent-point bundles must be optimized to beat the
///   fixed-table CROWN baseline.
fn build_sigmoid_tanh_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let hidden_w = arr2(&[[1.6_f32, -0.9], [-1.1, 1.4], [0.7, 1.2]]);
    let hidden_b = arr1(&[0.15_f32, -0.2, 0.05]);
    graph.add_node(GraphNode::from_input(
        "linear_hidden",
        Layer::Linear(LinearLayer::new(hidden_w, Some(hidden_b)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "sigmoid_hidden",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["linear_hidden".to_string()],
    ));

    let skip_w = arr2(&[[0.4_f32, -0.3], [0.2, 0.5], [-0.6, 0.1]]);
    let skip_b = arr1(&[0.0_f32, 0.1, -0.05]);
    graph.add_node(GraphNode::from_input(
        "linear_skip",
        Layer::Linear(LinearLayer::new(skip_w, Some(skip_b)).unwrap()),
    ));

    graph.add_node(GraphNode::new(
        "merge",
        Layer::Add(AddLayer),
        vec!["sigmoid_hidden".to_string(), "linear_skip".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "tanh_hidden",
        Layer::Tanh(TanhLayer::new()),
        vec!["merge".to_string()],
    ));

    let out_w = arr2(&[[1.3_f32, -0.8, 0.6], [-0.7, 1.1, -0.9]]);
    let out_b = arr1(&[0.0_f32, 0.05]);
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(LinearLayer::new(out_w, Some(out_b)).unwrap()),
        vec!["tanh_hidden".to_string()],
    ));
    graph.set_output("linear_out");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

fn sigmoid_tanh_dag_forward(x: &[f32; 2]) -> [f32; 2] {
    let hidden_w = arr2(&[[1.6_f32, -0.9], [-1.1, 1.4], [0.7, 1.2]]);
    let hidden_b = arr1(&[0.15_f32, -0.2, 0.05]);
    let skip_w = arr2(&[[0.4_f32, -0.3], [0.2, 0.5], [-0.6, 0.1]]);
    let skip_b = arr1(&[0.0_f32, 0.1, -0.05]);
    let out_w = arr2(&[[1.3_f32, -0.8, 0.6], [-0.7, 1.1, -0.9]]);
    let out_b = arr1(&[0.0_f32, 0.05]);

    let x_arr = arr1(&[x[0], x[1]]);
    let hidden_logits = hidden_w.dot(&x_arr) + &hidden_b;
    let hidden = hidden_logits.mapv(|v| 1.0 / (1.0 + (-v).exp()));
    let skip = skip_w.dot(&x_arr) + &skip_b;
    let merged = &hidden + &skip;
    let tanh_hidden = merged.mapv(f32::tanh);
    let out = out_w.dot(&tanh_hidden) + &out_b;
    [out[0], out[1]]
}

/// Build a config with limited iterations for testing convergence.
fn dag_test_config(iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    }
}

// ---------------------------------------------------------------------------
// Test 1: Diamond DAG soundness — bounds contain sampled true outputs
// ---------------------------------------------------------------------------

/// Diamond DAG alpha-CROWN: bounds must contain all sampled true network outputs.
///
/// This exercises the DAG backward bound accumulation at merge nodes, where
/// relu1 fans out to two linear layers that merge via Add.
///
/// Part of #2837.
#[ntest::timeout(10000)]
#[test]
fn test_dag_diamond_alpha_crown_soundness() {
    let (graph, input) = build_diamond_dag();

    let config = dag_test_config(10);
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();

    // Sample 11x11 = 121 points in the input box
    for i in 0..=10 {
        for j in 0..=10 {
            let t0 = i as f32 / 10.0;
            let t1 = j as f32 / 10.0;
            let x0 = -0.5 + t0;
            let x1 = -0.5 + t1;
            let output = diamond_forward(&[x0, x1]);

            for (dim, &out_val) in output.iter().enumerate() {
                assert!(
                    out_val >= alpha_bounds.lower()[[dim]] - 1e-5,
                    "Diamond DAG soundness: output[{dim}]={} < alpha lower {} at ({x0}, {x1})",
                    out_val,
                    alpha_bounds.lower()[[dim]],
                );
                assert!(
                    out_val <= alpha_bounds.upper()[[dim]] + 1e-5,
                    "Diamond DAG soundness: output[{dim}]={} > alpha upper {} at ({x0}, {x1})",
                    out_val,
                    alpha_bounds.upper()[[dim]],
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: Diamond DAG CROWN/IBP comparison — alpha-CROWN at least as tight
// ---------------------------------------------------------------------------

/// Diamond DAG: alpha-CROWN bounds should be at least as tight as IBP bounds.
///
/// The optimization loop maximizes the lower bound sum and minimizes the upper
/// bound sum. Even with 1 iteration, alpha-CROWN should not produce wider bounds
/// than IBP (within tolerance).
///
/// Part of #2837.
#[ntest::timeout(10000)]
#[test]
fn test_dag_diamond_alpha_crown_at_least_ibp() {
    let (graph, input) = build_diamond_dag();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let config = dag_test_config(1);
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();

    for dim in 0..2 {
        assert!(
            alpha_bounds.lower()[[dim]] >= ibp_bounds.lower()[[dim]] - 1e-4,
            "dim {dim}: alpha lower {} < IBP lower {} (wider than IBP!)",
            alpha_bounds.lower()[[dim]],
            ibp_bounds.lower()[[dim]],
        );
        assert!(
            alpha_bounds.upper()[[dim]] <= ibp_bounds.upper()[[dim]] + 1e-4,
            "dim {dim}: alpha upper {} > IBP upper {} (wider than IBP!)",
            alpha_bounds.upper()[[dim]],
            ibp_bounds.upper()[[dim]],
        );
    }
}

/// Regression for #3619 Packet A: DAG alpha-CROWN must optimize Sigmoid/Tanh
/// tangent points even when no ReLU alpha state exists.
#[ntest::timeout(10000)]
#[test]
fn test_dag_sigmoid_tanh_alpha_crown_beats_fixed_crown_3619() {
    let (graph, input) = build_sigmoid_tanh_dag();

    let fixed_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();
    let config = AlphaCrownConfig {
        iterations: 18,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();

    for i in 0..=10 {
        for j in 0..=10 {
            let x0 = -1.0 + 2.0 * (i as f32) / 10.0;
            let x1 = -1.0 + 2.0 * (j as f32) / 10.0;
            let out = sigmoid_tanh_dag_forward(&[x0, x1]);
            for (dim, &out_val) in out.iter().enumerate() {
                assert!(
                    out_val >= alpha_bounds.lower()[[dim]] - 1e-5,
                    "Sigmoid/Tanh DAG soundness: output[{dim}]={out_val} < lower {} at ({x0}, {x1})",
                    alpha_bounds.lower()[[dim]],
                );
                assert!(
                    out_val <= alpha_bounds.upper()[[dim]] + 1e-5,
                    "Sigmoid/Tanh DAG soundness: output[{dim}]={out_val} > upper {} at ({x0}, {x1})",
                    alpha_bounds.upper()[[dim]],
                );
            }
        }
    }

    let fixed_width = total_width(&fixed_bounds);
    let alpha_width = total_width(&alpha_bounds);

    assert!(
        alpha_width + 1e-4 < fixed_width,
        "Sigmoid/Tanh DAG alpha-CROWN should beat fixed-table CROWN. fixed={fixed_width:.6}, alpha={alpha_width:.6}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Multi-iteration convergence — more iterations give tighter bounds
// ---------------------------------------------------------------------------

/// Alpha-CROWN with more iterations should produce bounds at least as tight as
/// fewer iterations. The optimization loop accumulates element-wise best bounds
/// across iterations, so bound quality is monotonically non-decreasing.
///
/// Part of #2837.
#[ntest::timeout(10000)]
#[test]
fn test_dag_alpha_crown_convergence_monotonic() {
    let (graph, input) = build_diamond_dag();

    // Run with 1 iteration
    let config_1 = dag_test_config(1);
    let bounds_1 = graph
        .propagate_alpha_crown_with_config(&input, &config_1)
        .unwrap();

    // Run with 10 iterations
    let config_10 = dag_test_config(10);
    let bounds_10 = graph
        .propagate_alpha_crown_with_config(&input, &config_10)
        .unwrap();

    // More iterations should produce tighter bounds (higher lower, lower upper).
    // We check the sum (total width) rather than element-wise, since per-element
    // improvements are stochastic. The element-wise best tracking in the
    // optimization loop guarantees sum-monotonicity.
    let width_1: f32 = bounds_1
        .upper()
        .iter()
        .zip(bounds_1.lower().iter())
        .map(|(u, l)| u - l)
        .sum();
    let width_10: f32 = bounds_10
        .upper()
        .iter()
        .zip(bounds_10.lower().iter())
        .map(|(u, l)| u - l)
        .sum();

    assert!(
        width_10 <= width_1 + 1e-4,
        "10-iteration width ({width_10:.6}) should be <= 1-iteration width ({width_1:.6})"
    );
}

// ---------------------------------------------------------------------------
// Test 4: DAG alpha-CROWN with no ReLU falls back to CROWN
// ---------------------------------------------------------------------------

/// A DAG with no ReLU nodes should produce bounds identical to CROWN,
/// since alpha-CROWN has no alpha parameters to optimize.
///
/// Part of #2837.
#[ntest::timeout(10000)]
#[test]
fn test_dag_no_relu_falls_back_to_crown() {
    let mut graph = GraphNetwork::new();

    // Diamond without ReLU: Linear1 -> branch(L2a, L2b) -> Add -> Output
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, None).unwrap()),
    ));

    let w2a = arr2(&[[0.8_f32, -0.3]]);
    graph.add_node(GraphNode::new(
        "linear2a",
        Layer::Linear(LinearLayer::new(w2a, None).unwrap()),
        vec!["linear1".to_string()],
    ));

    let w2b = arr2(&[[-0.4_f32, 0.7]]);
    graph.add_node(GraphNode::new(
        "linear2b",
        Layer::Linear(LinearLayer::new(w2b, None).unwrap()),
        vec!["linear1".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["linear2a".to_string(), "linear2b".to_string()],
    ));

    graph.set_output("add");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let config = dag_test_config(5);
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();
    let crown_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();

    // Without ReLU, alpha-CROWN should match CROWN exactly.
    for dim in 0..alpha_bounds.lower().len() {
        let diff_lower = (alpha_bounds.lower().iter().nth(dim).unwrap()
            - crown_bounds.lower().iter().nth(dim).unwrap())
        .abs();
        let diff_upper = (alpha_bounds.upper().iter().nth(dim).unwrap()
            - crown_bounds.upper().iter().nth(dim).unwrap())
        .abs();

        assert!(
            diff_lower < 1e-4,
            "dim {dim}: alpha lower differs from CROWN lower by {diff_lower}"
        );
        assert!(
            diff_upper < 1e-4,
            "dim {dim}: alpha upper differs from CROWN upper by {diff_upper}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: restore_dag_alpha_snapshot — alpha state consistent after perturbation
// ---------------------------------------------------------------------------

/// Verify that `collect_alpha_crown_bounds_dag` returns a valid alpha state
/// after optimization, and that running it twice with the same input produces
/// consistent bounds (indicating alpha snapshot restore works correctly during
/// SPSA/finite-difference gradient estimation).
///
/// This is an indirect test of `restore_dag_alpha_snapshot`: if the restore
/// mechanism were broken, gradient estimation would corrupt the alpha state,
/// causing subsequent iterations to diverge and produce different final bounds
/// across runs (due to corrupted optimizer state).
///
/// Part of #2837.
#[ntest::timeout(10000)]
#[test]
fn test_dag_alpha_state_consistent_across_runs() {
    let (graph, input) = build_diamond_dag();

    let config = AlphaCrownConfig {
        iterations: 5,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };

    // Run twice — both should produce identical bounds since the graph and
    // input are deterministic. The default gradient method is AnalyticChain
    // which is deterministic.
    let (bounds_1, alpha_1) = graph
        .collect_alpha_crown_bounds_dag(&input, &config)
        .unwrap();
    let (bounds_2, alpha_2) = graph
        .collect_alpha_crown_bounds_dag(&input, &config)
        .unwrap();

    let out1 = bounds_1.get(graph.output_name()).unwrap();
    let out2 = bounds_2.get(graph.output_name()).unwrap();

    for dim in 0..out1.lower().len() {
        let diff_l =
            (out1.lower().iter().nth(dim).unwrap() - out2.lower().iter().nth(dim).unwrap()).abs();
        let diff_u =
            (out1.upper().iter().nth(dim).unwrap() - out2.upper().iter().nth(dim).unwrap()).abs();
        assert!(
            diff_l < 1e-5,
            "dim {dim}: lower bounds differ across runs by {diff_l}"
        );
        assert!(
            diff_u < 1e-5,
            "dim {dim}: upper bounds differ across runs by {diff_u}"
        );
    }

    // Alpha states should have the same number of entries.
    assert_eq!(
        alpha_1.alphas.len(),
        alpha_2.alphas.len(),
        "alpha state entry count differs across runs"
    );
}

// ---------------------------------------------------------------------------
// Test 6: SPSA gradient method produces sound bounds on DAG
// ---------------------------------------------------------------------------

/// Verify that SPSA gradient estimation on a DAG produces sound bounds.
/// SPSA perturbs all alpha values simultaneously and uses the restore_snapshot
/// mechanism to undo perturbations. If restore is broken, bounds may be unsound.
///
/// Part of #2837.
#[ntest::timeout(60000)]
#[test]
fn test_dag_spsa_gradient_soundness() {
    use crate::bounds::GradientMethod;

    let (graph, input) = build_diamond_dag();

    let config = AlphaCrownConfig {
        iterations: 5,
        gradient_method: GradientMethod::Spsa,
        spsa_samples: 2,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };

    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();

    // Sample 11x11 grid and verify soundness
    for i in 0..=10 {
        for j in 0..=10 {
            let t0 = i as f32 / 10.0;
            let t1 = j as f32 / 10.0;
            let x0 = -0.5 + t0;
            let x1 = -0.5 + t1;
            let output = diamond_forward(&[x0, x1]);

            for (dim, &out_val) in output.iter().enumerate() {
                assert!(
                    out_val >= alpha_bounds.lower()[[dim]] - 1e-5,
                    "SPSA DAG soundness: output[{dim}]={} < lower {} at ({x0}, {x1})",
                    out_val,
                    alpha_bounds.lower()[[dim]],
                );
                assert!(
                    out_val <= alpha_bounds.upper()[[dim]] + 1e-5,
                    "SPSA DAG soundness: output[{dim}]={} > upper {} at ({x0}, {x1})",
                    out_val,
                    alpha_bounds.upper()[[dim]],
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 7: Non-finite pre-activation produces error for Reciprocal in DAG
// (exercises error handling in DAG context, not just ReciprocalLayer)
// ---------------------------------------------------------------------------

/// A DAG where CROWN backward encounters a Reciprocal layer with zero-crossing
/// pre-activation bounds should return an error, not produce corrupted bounds.
///
/// This tests the DAG code path's error propagation from layer-level CROWN
/// backward through the DAG backward pass machinery.
///
/// Part of #2837.
#[ntest::timeout(10000)]
#[test]
fn test_dag_crown_error_on_zero_crossing_reciprocal() {
    let mut graph = GraphNetwork::new();

    // Linear that can produce zero-crossing output for our input
    let w1 = arr2(&[[1.0_f32, -1.0]]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, None).unwrap()),
    ));

    // Reciprocal layer — will fail if pre-activation crosses zero
    graph.add_node(GraphNode::new(
        "recip",
        Layer::Reciprocal(ReciprocalLayer::new()),
        vec!["linear1".to_string()],
    ));

    graph.set_output("recip");

    // Input box [-1, 1] × [-1, 1] means Linear1 output ∈ [-2, 2] which crosses zero
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    // Reciprocal on zero-crossing interval produces [-inf, +inf], but
    // new_repaired (replacing repair_non_finite, #3030, #3423) converts these to conservative
    // finite fallback bounds. IBP now succeeds with wide bounds.
    let ibp_result = graph.propagate_ibp(&input);
    assert!(
        ibp_result.is_ok(),
        "IBP should succeed with new_repaired fallback bounds, got: {:?}",
        ibp_result.as_ref().err()
    );

    // CROWN should also return an error because Reciprocal CROWN backward
    // rejects zero-crossing pre-activation bounds.
    let crown_result = graph.propagate_crown_fixed_slope(&input);
    assert!(
        crown_result.is_err(),
        "CROWN should error on zero-crossing Reciprocal pre-activation bounds, got: {:?}",
        crown_result
    );
}

// ---------------------------------------------------------------------------
// Test 8: INVPROP ny clipping — gammas stay non-negative after DAG optimization
// ---------------------------------------------------------------------------

/// Regression test for #2970: DAG alpha-CROWN with INVPROP must clip gammas
/// to non-negative after each optimizer step. Without clipping, negative gammas
/// invert constraint contributions, producing unsound (too-tight) bounds.
///
/// Verifies output bounds are sound (contain sampled true outputs) when INVPROP
/// is active. Uses `propagate_alpha_crown_with_config` which dispatches to
/// `propagate_dag_alpha_crown_with_config_and_engine` for the diamond DAG.
///
/// Part of #2970.
#[ntest::timeout(10000)]
#[test]
fn test_dag_invprop_ny_clipping_2970() {
    use crate::invprop::{InvpropConfig, OutputConstraints};

    let (graph, input) = build_diamond_dag();

    // Output constraint: y[0] >= 0.1 (a single linear constraint Ay >= b).
    // A = [[1, 0]], b = [0.1]. This gives the optimizer a reason to push
    // gammas nonzero (tighten bounds using the output constraint).
    let constraints = OutputConstraints::new(arr2(&[[1.0, 0.0]]), arr1(&[0.1]), true).unwrap();

    let config = AlphaCrownConfig {
        iterations: 20,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        invprop: InvpropConfig {
            enabled: true,
            apply_output_constraints_to: vec!["all".to_string()],
            ..InvpropConfig::default()
        },
        output_constraints: Some(constraints),
        ..AlphaCrownConfig::default()
    };

    // Use propagate_alpha_crown_with_config which dispatches to
    // propagate_dag_alpha_crown_with_config_and_engine for non-sequential
    // graphs (the diamond DAG). This exercises the exact code path where
    // clip_gammas was added.
    let output_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();

    // Soundness: bounds must contain all sampled true outputs.
    // If clip_gammas is missing, negative gammas invert constraint
    // contributions, producing bounds tighter than correct (unsound).
    for i in 0..=10 {
        for j in 0..=10 {
            let x0 = -0.5 + i as f32 / 10.0;
            let x1 = -0.5 + j as f32 / 10.0;
            let output = diamond_forward(&[x0, x1]);

            for (dim, &val) in output.iter().enumerate() {
                assert!(
                    val >= output_bounds.lower()[[dim]] - 1e-5,
                    "INVPROP soundness: y[{dim}]={val} < lower {} at ({x0},{x1})",
                    output_bounds.lower()[[dim]],
                );
                assert!(
                    val <= output_bounds.upper()[[dim]] + 1e-5,
                    "INVPROP soundness: y[{dim}]={val} > upper {} at ({x0},{x1})",
                    output_bounds.upper()[[dim]],
                );
            }
        }
    }
}

/// MOAT test: INVPROP gamma ascent (optimize ON) with a NON-EMPTY violation region
/// must never return an unsound (too-tight) BOX bound. The assume-violation dual is
/// valid only over the violation region; the loop must not report it as a box bound
/// when the region is non-empty. Checks the returned bounds contain every sampled
/// true output over the box grid.
#[ntest::timeout(20000)]
#[test]
fn test_dag_invprop_optimize_box_soundness() {
    use crate::invprop::{InvpropConfig, OutputConstraints};

    let (graph, input) = build_diamond_dag();

    // Violation region {y0 <= 0} — non-empty for the diamond over the sampled box,
    // so the augmented (violation-only) bound differs from the true box range.
    let constraints = OutputConstraints::new(arr2(&[[1.0, 0.0]]), arr1(&[0.0]), true).unwrap();

    let config = AlphaCrownConfig {
        iterations: 20,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        invprop: InvpropConfig {
            enabled: true,
            optimize_gammas: true,
            gamma_lr: 0.5,
            ..InvpropConfig::default()
        },
        output_constraints: Some(constraints),
        ..AlphaCrownConfig::default()
    };

    let output_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();

    // If the loop returned the (violation-only) augmented bound as a box bound, some
    // true box output outside the violation region would fall outside these bounds.
    for i in 0..=10 {
        for j in 0..=10 {
            let x0 = -0.5 + i as f32 / 10.0;
            let x1 = -0.5 + j as f32 / 10.0;
            let output = diamond_forward(&[x0, x1]);
            for (dim, &val) in output.iter().enumerate() {
                assert!(
                    val >= output_bounds.lower()[[dim]] - 1e-4
                        && val <= output_bounds.upper()[[dim]] + 1e-4,
                    "INVPROP optimize box-soundness: y[{dim}]={val} outside [{}, {}] at ({x0},{x1})",
                    output_bounds.lower()[[dim]],
                    output_bounds.upper()[[dim]],
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 9: Regression #3298 — alpha-CROWN default config improves over CROWN
// on a deep ResNet-like DAG with gradient attenuation
// ---------------------------------------------------------------------------

/// Weight matrices for the deep residual DAG (shared between builder and forward pass).
struct ResidualWeights {
    w0: Array2<f32>,
    b0: Array1<f32>,
    w1: Array2<f32>,
    w_skip: Array2<f32>,
    w2: Array2<f32>,
    w_out: Array2<f32>,
}

fn residual_weights() -> ResidualWeights {
    ResidualWeights {
        w0: arr2(&[
            [0.6, -0.3, 0.4],
            [-0.2, 0.5, 0.3],
            [0.1, -0.4, 0.7],
            [0.3, 0.2, -0.5],
        ]),
        b0: arr1(&[0.1_f32, -0.05, 0.05, -0.1]),
        w1: arr2(&[
            [0.4, -0.2, 0.3, -0.1],
            [-0.1, 0.5, -0.2, 0.3],
            [0.2, -0.3, 0.6, -0.2],
            [-0.3, 0.1, -0.1, 0.4],
        ]),
        w_skip: arr2(&[
            [0.9, 0.05, -0.05, 0.0],
            [0.0, 0.85, 0.1, -0.05],
            [-0.05, 0.0, 0.9, 0.05],
            [0.05, -0.05, 0.0, 0.85],
        ]),
        w2: arr2(&[
            [0.5, -0.2, 0.1, 0.3],
            [-0.1, 0.4, -0.3, 0.2],
            [0.3, -0.1, 0.5, -0.2],
            [-0.2, 0.3, -0.1, 0.4],
        ]),
        w_out: arr2(&[[0.4_f32, -0.3, 0.2, 0.1], [-0.2, 0.5, -0.1, 0.3]]),
    }
}

/// Add a named linear → relu pair to the graph.
fn add_linear_relu(
    graph: &mut GraphNetwork,
    linear_name: &str,
    relu_name: &str,
    w: Array2<f32>,
    bias: Option<Array1<f32>>,
    input: &str,
    is_input: bool,
) {
    let layer = Layer::Linear(LinearLayer::new(w, bias).unwrap());
    if is_input {
        graph.add_node(GraphNode::from_input(linear_name, layer));
    } else {
        graph.add_node(GraphNode::new(linear_name, layer, vec![input.to_string()]));
    }
    graph.add_node(GraphNode::new(
        relu_name,
        Layer::ReLU(ReLULayer),
        vec![linear_name.to_string()],
    ));
}

/// Build a deep ResNet-like DAG with 4 ReLU layers and a residual connection.
///
/// Structure: Input[3] → L0→R0 → (L1→R1 + Lskip) → Add → R2 → L2→R3 → Lout[2]
///
/// The residual skip from R0 creates gradient attenuation through the main path,
/// mimicking ResNet-2b where the old pilot check would exit prematurely (#3298).
fn build_deep_residual_dag() -> (GraphNetwork, BoundedTensor) {
    let w = residual_weights();
    let mut graph = GraphNetwork::new();

    add_linear_relu(&mut graph, "linear0", "relu0", w.w0, Some(w.b0), "", true);
    add_linear_relu(&mut graph, "linear1", "relu1", w.w1, None, "relu0", false);

    // Skip path: near-identity from relu0
    let skip = Layer::Linear(LinearLayer::new(w.w_skip, None).unwrap());
    graph.add_node(GraphNode::new(
        "linear_skip",
        skip,
        vec!["relu0".to_string()],
    ));

    // Merge + ReLU2
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["relu1".to_string(), "linear_skip".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));

    add_linear_relu(&mut graph, "linear2", "relu3", w.w2, None, "relu2", false);

    // Output
    let out = Layer::Linear(LinearLayer::new(w.w_out, None).unwrap());
    graph.add_node(GraphNode::new("linear_out", out, vec!["relu3".to_string()]));
    graph.set_output("linear_out");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();
    (graph, input)
}

/// Compute forward pass through the deep residual DAG for soundness checks.
fn deep_residual_forward(x: &[f32; 3]) -> [f32; 2] {
    let w = residual_weights();
    let x_arr = arr1(&[x[0], x[1], x[2]]);

    let h0 = (w.w0.dot(&x_arr) + &w.b0).mapv(|v: f32| v.max(0.0));
    let h1 = w.w1.dot(&h0).mapv(|v: f32| v.max(0.0));
    let skip = w.w_skip.dot(&h0);
    let h2 = (&h1 + &skip).mapv(|v: f32| v.max(0.0));
    let h3 = w.w2.dot(&h2).mapv(|v: f32| v.max(0.0));
    let out = w.w_out.dot(&h3);
    [out[0], out[1]]
}

/// Total bound width (sum of upper - lower across all elements).
fn total_width(bt: &BoundedTensor) -> f32 {
    bt.upper()
        .iter()
        .zip(bt.lower().iter())
        .map(|(u, l)| u - l)
        .sum()
}

/// Verify soundness of bounds on the deep residual DAG by sampling 5^3 = 125 grid points.
fn assert_residual_dag_soundness(bounds: &BoundedTensor, label: &str) {
    for i in 0..=4 {
        for j in 0..=4 {
            for k in 0..=4 {
                let x0 = -0.5 + i as f32 / 4.0;
                let x1 = -0.5 + j as f32 / 4.0;
                let x2 = -0.5 + k as f32 / 4.0;
                let output = deep_residual_forward(&[x0, x1, x2]);

                for (dim, &out_val) in output.iter().enumerate() {
                    assert!(
                        out_val >= bounds.lower()[[dim]] - 1e-5,
                        "{label}: output[{dim}]={out_val} < lower {} at ({x0},{x1},{x2})",
                        bounds.lower()[[dim]],
                    );
                    assert!(
                        out_val <= bounds.upper()[[dim]] + 1e-5,
                        "{label}: output[{dim}]={out_val} > upper {} at ({x0},{x1},{x2})",
                        bounds.upper()[[dim]],
                    );
                }
            }
        }
    }
}

/// Regression test for #3298: alpha-CROWN with default config (pilot disabled,
/// patience=10) must produce strictly tighter bounds than plain CROWN on a deep
/// ResNet-like DAG with residual connections.
///
/// Before #3298, the adaptive_skip_pilot check exited after 1 iteration when
/// improvement < 1e-3, which happened on deep DAGs due to gradient attenuation.
/// The fix: `adaptive_skip_pilot: false`, `early_stop_patience: 10` (was 3).
///
/// Part of #3298.
#[ntest::timeout(60000)]
#[test]
fn test_dag_alpha_crown_default_config_improves_over_crown_3298() {
    let (graph, input) = build_deep_residual_dag();

    // CROWN baseline — verify soundness first.
    let crown_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();
    assert_residual_dag_soundness(&crown_bounds, "CROWN");
    let crown_width = total_width(&crown_bounds);

    let public_crown_bounds = graph.propagate_crown(&input).unwrap();
    assert_residual_dag_soundness(&public_crown_bounds, "public-CROWN");
    let public_crown_width = total_width(&public_crown_bounds);

    // Alpha-CROWN with DEFAULT config (production config changed in #3298).
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &AlphaCrownConfig::default())
        .unwrap();
    assert_residual_dag_soundness(&alpha_bounds, "alpha-CROWN");
    let alpha_width = total_width(&alpha_bounds);

    let improvement_pct = 100.0 * (1.0 - alpha_width / crown_width);
    let public_improvement_pct = 100.0 * (1.0 - public_crown_width / crown_width);
    eprintln!(
        "#3298 deep residual DAG: CROWN={crown_width:.6}, alpha={alpha_width:.6}, \
         improvement={improvement_pct:.2}%"
    );
    eprintln!(
        "#3619 deep residual DAG public CROWN={public_crown_width:.6}, \
         fixed-slope={crown_width:.6}, improvement={public_improvement_pct:.2}%"
    );

    // Alpha-CROWN must not be wider than CROWN.
    assert!(
        alpha_width <= crown_width + 1e-4,
        "alpha-CROWN ({alpha_width:.6}) wider than CROWN ({crown_width:.6})"
    );
    // Alpha-CROWN must produce STRICTLY tighter bounds (>1% improvement).
    assert!(
        improvement_pct > 1.0,
        "alpha-CROWN should be >1% tighter than CROWN on deep residual DAG. \
         CROWN={crown_width:.6}, alpha={alpha_width:.6}, improvement={improvement_pct:.2}%. \
         Regression: #3298 fix (pilot disable + patience=10) may be broken."
    );
    assert!(
        public_crown_width <= crown_width + 1e-4,
        "public graph CROWN ({public_crown_width:.6}) wider than fixed-slope \
         CROWN ({crown_width:.6})"
    );
    assert!(
        public_improvement_pct > 1.0,
        "public graph CROWN should be >1% tighter than fixed-slope CROWN on \
         the deep residual DAG after #3619. fixed-slope={crown_width:.6}, \
         public={public_crown_width:.6}, improvement={public_improvement_pct:.2}%"
    );

    // Negative control: the OLD config (pilot enabled) should NOT improve,
    // proving the fix is what enables the improvement.
    let old_config = AlphaCrownConfig {
        adaptive_skip_pilot: true,
        early_stop_patience: 3,
        ..AlphaCrownConfig::default()
    };
    let old_bounds = graph
        .propagate_alpha_crown_with_config(&input, &old_config)
        .unwrap();
    assert_residual_dag_soundness(&old_bounds, "alpha-old-config");
    let old_width = total_width(&old_bounds);
    let old_improvement = 100.0 * (1.0 - old_width / crown_width);
    eprintln!("#3298 old config (pilot=true, patience=3): improvement={old_improvement:.2}%");
    // Old config improvement should be negligible (< new config improvement).
    assert!(
        old_improvement < improvement_pct,
        "old config ({old_improvement:.2}%) should improve less than new ({improvement_pct:.2}%)"
    );
}

// ---------------------------------------------------------------------------
// Warmup early-exit (#warmup-early-exit)
// ---------------------------------------------------------------------------

/// Pure-function check of the spec projection + verified predicate used by the
/// in-loop early-exit. These mirror `objective_bounds` and
/// `domain_is_verified_for_mode` exactly, so the loop early-exit sees the same
/// decision the post-warmup BaB code uses.
#[test]
fn test_alpha_spec_early_exit_projection_and_verified() {
    // objective = [1, -1] over output interval [l, u] = [(0, 2), (1, 3)]:
    //   lower = 1*0 + (-1)*3 = -3 ; upper = 1*2 + (-1)*1 = 1.
    let spec = AlphaSpecEarlyExit {
        objective: vec![1.0, -1.0],
        threshold: -5.0,
        verify_upper_bound: false,
    };
    let (lo, hi) = spec.project_bounds(&[0.0, 1.0], &[2.0, 3.0]).unwrap();
    assert!((lo - (-3.0)).abs() < 1e-6, "projected lower {lo}");
    assert!((hi - 1.0).abs() < 1e-6, "projected upper {hi}");
    // lower (-3) > threshold (-5) → verified in lower-bound mode.
    assert!(spec.is_verified(lo, hi));
    // A threshold above the lower bound is NOT verified.
    let spec_unverified = AlphaSpecEarlyExit {
        threshold: -1.0,
        ..spec.clone()
    };
    assert!(!spec_unverified.is_verified(lo, hi));
    // Non-finite inputs never verify (#2993 parity).
    assert!(!spec.is_verified(f32::NAN, hi));
    assert!(!spec.is_verified(lo, f32::INFINITY));
    // Length mismatch → None (loop then skips the check that iteration).
    assert!(spec.project_bounds(&[0.0], &[2.0]).is_none());
}

/// Change A: a small ReLU DAG whose property is already provable at the first
/// warmup iteration must early-exit BEFORE `config.iterations` and still return a
/// sound, threshold-clearing (VERIFIED) bound. We prove the early break fired by
/// asserting the 50-iteration spec-exit run returns exactly the 1-iteration bounds
/// (it stopped after iteration 0). SOUNDNESS: the returned bound is the same valid
/// over-approximation either way — early-exit changes nothing about the math.
#[ntest::timeout(10000)]
#[test]
fn test_dag_warmup_spec_early_exit_returns_verified_bound() {
    let (graph, input) = build_diamond_dag();
    let output_node = graph.output_name().to_string();
    let objective = [1.0_f32, 0.0]; // first output dim, lower-bound mode

    // Baseline: a single warmup iteration (no spec). start_save_best=0.0 so every
    // iteration is saved, making the elementwise best fully deterministic.
    let base_config = AlphaCrownConfig {
        iterations: 1,
        start_save_best: 0.0,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };
    let (base_map, _) = graph
        .collect_alpha_crown_bounds_dag(&input, &base_config)
        .unwrap();
    let base_out = base_map.get(&output_node).unwrap();
    let base_lower = base_out.lower().as_slice().unwrap().to_vec();
    let base_upper = base_out.upper().as_slice().unwrap().to_vec();

    // Projected first-output lower at iteration 0, then a threshold safely below it
    // so the property is provable at iter 0 (lower-bound mode: lower > threshold).
    let proj_lower = objective[0] * base_lower[0] + objective[1] * base_lower[1];
    let threshold = proj_lower - 1.0;

    // 50-iteration warmup carrying the spec early-exit. If the break fires at iter 0
    // (as it must, since the iter-0 bound already clears the threshold) the returned
    // output bounds equal the 1-iteration baseline above.
    let early_config = AlphaCrownConfig {
        iterations: 50,
        start_save_best: 0.0,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        spec_early_exit: Some(AlphaSpecEarlyExit {
            objective: objective.to_vec(),
            threshold,
            verify_upper_bound: false,
        }),
        ..AlphaCrownConfig::default()
    };
    let (early_map, _) = graph
        .collect_alpha_crown_bounds_dag(&input, &early_config)
        .unwrap();
    let early_out = early_map.get(&output_node).unwrap();
    let early_lower = early_out.lower().as_slice().unwrap();
    let early_upper = early_out.upper().as_slice().unwrap();

    // Early-exit fired at iter 0: identical to the 1-iteration baseline.
    for d in 0..base_lower.len() {
        assert!(
            (early_lower[d] - base_lower[d]).abs() < 1e-6,
            "dim {d}: spec-exit lower {} != 1-iter baseline {} (early-exit did not stop at iter 0)",
            early_lower[d],
            base_lower[d],
        );
        assert!(
            (early_upper[d] - base_upper[d]).abs() < 1e-6,
            "dim {d}: spec-exit upper {} != 1-iter baseline {}",
            early_upper[d],
            base_upper[d],
        );
    }

    // The returned (early-exit) bound is a VALID, threshold-clearing proof:
    // projected lower > threshold ⇒ property verified.
    let exit_proj_lower = objective[0] * early_lower[0] + objective[1] * early_lower[1];
    assert!(
        exit_proj_lower > threshold,
        "spec-exit bound must clear the threshold: projected lower {exit_proj_lower} > {threshold}"
    );

    // Soundness: the bound still contains all sampled true outputs.
    for i in 0..=8 {
        for j in 0..=8 {
            let x0 = -0.5 + i as f32 / 8.0;
            let x1 = -0.5 + j as f32 / 8.0;
            let out = diamond_forward(&[x0, x1]);
            for (dim, &v) in out.iter().enumerate() {
                assert!(
                    v >= early_lower[dim] - 1e-5 && v <= early_upper[dim] + 1e-5,
                    "spec-exit soundness: output[{dim}]={v} outside [{}, {}]",
                    early_lower[dim],
                    early_upper[dim],
                );
            }
        }
    }
}

/// Change B: the SPSA warmup loop must plateau-exit (the `no_improve` patience
/// break fires) and return the SAME bound as running those iterations without the
/// plateau break. With a huge tolerance every iteration counts as non-improving, so
/// `no_improve_iters` reaches `early_stop_patience=2` at iteration 2; the run stops
/// there. A patience-disabled run capped at exactly 3 iterations computes iterations
/// 0..2 as well. The seeded test RNG makes SPSA deterministic, so both return the
/// identical iter-2 bound — proving plateau-exit returns the best-seen valid bound.
#[ntest::timeout(10000)]
#[test]
fn test_spsa_warmup_plateau_exit_matches_capped_run() {
    let (graph, input) = build_diamond_dag();
    let output_node = graph.output_name().to_string();

    // SPSA gradient method routes through the SPSA loop in bounds/alpha.rs.
    // start_save_best=0.0 saves every iteration so the elementwise best is identical
    // between the two runs at matching iterations.
    let plateau_config = AlphaCrownConfig {
        iterations: 50,
        gradient_method: GradientMethod::Spsa,
        early_stop_patience: 2,
        tolerance: 1e9, // every iteration is "no improvement" → plateau by iter 2
        start_save_best: 0.0,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };
    let (plateau_map, _) = graph
        .collect_alpha_crown_bounds_dag(&input, &plateau_config)
        .unwrap();
    let plateau_out = plateau_map.get(&output_node).unwrap();

    // Reference: patience disabled, capped at exactly 3 iterations (runs iters 0..2,
    // then the last-iteration break). Same iterations actually executed as the
    // plateau run, so the best-seen bound must match.
    let capped_config = AlphaCrownConfig {
        iterations: 3,
        gradient_method: GradientMethod::Spsa,
        early_stop_patience: 999, // never plateau-break
        tolerance: 1e9,
        start_save_best: 0.0,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };
    let (capped_map, _) = graph
        .collect_alpha_crown_bounds_dag(&input, &capped_config)
        .unwrap();
    let capped_out = capped_map.get(&output_node).unwrap();

    let pl = plateau_out.lower().as_slice().unwrap();
    let pu = plateau_out.upper().as_slice().unwrap();
    let cl = capped_out.lower().as_slice().unwrap();
    let cu = capped_out.upper().as_slice().unwrap();
    assert_eq!(pl.len(), cl.len());
    for d in 0..pl.len() {
        assert!(
            (pl[d] - cl[d]).abs() < 1e-5,
            "dim {d}: plateau-exit lower {} != capped-run lower {}",
            pl[d],
            cl[d],
        );
        assert!(
            (pu[d] - cu[d]).abs() < 1e-5,
            "dim {d}: plateau-exit upper {} != capped-run upper {}",
            pu[d],
            cu[d],
        );
    }

    // Confirm NO behavior change when patience is not reached: a run with the same
    // setup but a large patience and the same iteration cap produces identical bounds
    // (the plateau branch never fires). This is the capped_config above, already
    // asserted equal — i.e. the plateau machinery is inert until patience is hit.
}
