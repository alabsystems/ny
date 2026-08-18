// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Spec-guided CROWN tests (#593): classification specifications via C @ output.

use crate::*;
use ndarray::{arr1, arr2, Array1};
use proptest::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_crown_classification_at_least_as_tight() {
    // Test that spec-guided CROWN produces bounds at least as tight as post-hoc
    // interval arithmetic for classification specifications like Y_0 - Y_i.
    //
    // The key insight: spec-guided CROWN computes bounds on C @ output directly,
    // preserving correlations between outputs. Post-hoc interval arithmetic computes
    // bounds on each output independently via CROWN, then combines them, losing correlations.
    //
    // For a classification property Y_0 > Y_1, the spec matrix C = [[1, -1]].
    // Spec-guided CROWN: bound(Y_0 - Y_1) directly through backward pass
    // Post-hoc: compute CROWN bounds on Y_0 and Y_1, then [l_0 - u_1, u_0 - l_1]

    let mut graph = GraphNetwork::new();

    // Network: input -> linear -> relu -> linear
    // Use weights that create correlated outputs (both grow with same input direction)
    let w1 = arr2(&[[1.0_f32, 0.5], [0.5, 1.0]]);
    let b1 = arr1(&[0.0_f32, 0.0]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    // Second linear layer with weights that preserve correlation
    let w2 = arr2(&[[1.0_f32, -0.5], [-0.5, 1.0]]);
    let b2 = arr1(&[0.1_f32, -0.1]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));

    graph.set_output("linear2");

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    // Classification spec: Y_0 - Y_1 > 0
    // C = [[1, -1]]
    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);

    // Method 1: Spec-guided CROWN (computes bounds on C @ output directly)
    let spec_guided_bounds = graph
        .propagate_crown_with_specs_and_engine(&input, &spec_matrix, None)
        .unwrap();

    // Method 2: Post-hoc interval arithmetic (what happens when spec-guided fails)
    // Compute output bounds independently via CROWN, then apply spec matrix
    let output_bounds = graph.propagate_crown(&input).unwrap();
    let (posthoc_lower, posthoc_upper) = {
        let l0 = output_bounds.lower()[[0]];
        let u0 = output_bounds.upper()[[0]];
        let l1 = output_bounds.lower()[[1]];
        let u1 = output_bounds.upper()[[1]];
        // Y_0 - Y_1 via interval arithmetic on CROWN outputs: [l_0 - u_1, u_0 - l_1]
        (l0 - u1, u0 - l1)
    };

    let spec_guided_lower = spec_guided_bounds.lower()[[0]];
    let spec_guided_upper = spec_guided_bounds.upper()[[0]];

    println!(
        "Spec-guided CROWN: [{:.4}, {:.4}]",
        spec_guided_lower, spec_guided_upper
    );
    println!(
        "Post-hoc interval arithmetic: [{:.4}, {:.4}]",
        posthoc_lower, posthoc_upper
    );

    // Spec-guided bounds should be at least as tight as post-hoc interval arithmetic
    // (spec-guided preserves correlations, so should never be looser)
    assert!(
        spec_guided_lower >= posthoc_lower - 1e-5,
        "Spec-guided lower {} should be >= post-hoc lower {}",
        spec_guided_lower,
        posthoc_lower
    );
    assert!(
        spec_guided_upper <= posthoc_upper + 1e-5,
        "Spec-guided upper {} should be <= post-hoc upper {}",
        spec_guided_upper,
        posthoc_upper
    );

    // Verify soundness by sampling
    let linear1_layer = match &graph.nodes.get("linear1").unwrap().layer {
        Layer::Linear(l) => l,
        _ => panic!("Expected Linear"),
    };
    let linear2_layer = match &graph.nodes.get("linear2").unwrap().layer {
        Layer::Linear(l) => l,
        _ => panic!("Expected Linear"),
    };

    for i in 0..100 {
        let t0 = (i * 17 % 100) as f32 / 100.0;
        let t1 = (i * 31 % 100) as f32 / 100.0;
        let x = arr1(&[t0, t1]);

        // Forward pass
        let h1: Array1<f32> = linear1_layer.weight().dot(&x) + linear1_layer.bias().unwrap();
        let h2 = h1.mapv(|v| v.max(0.0)); // ReLU
        let y: Array1<f32> = linear2_layer.weight().dot(&h2) + linear2_layer.bias().unwrap();

        // Compute Y_0 - Y_1
        let spec_value = y[0] - y[1];

        // Check soundness
        assert!(
            spec_value >= spec_guided_lower - 1e-4,
            "Sample {}: Y_0 - Y_1 = {} < spec-guided lower {}",
            i,
            spec_value,
            spec_guided_lower
        );
        assert!(
            spec_value <= spec_guided_upper + 1e-4,
            "Sample {}: Y_0 - Y_1 = {} > spec-guided upper {}",
            i,
            spec_value,
            spec_guided_upper
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_crown_multiclass_classification() {
    // Test spec-guided CROWN for multi-class classification with 3 outputs.
    // Property: Y_0 is the maximum output (Y_0 > Y_1 AND Y_0 > Y_2)
    // Spec matrix: [[1, -1, 0], [1, 0, -1]] (two comparisons)

    let mut graph = GraphNetwork::new();

    // 2-input, 3-output network with ReLU
    let w1 = arr2(&[[1.0_f32, 0.5], [0.5, 1.0], [-0.3, 0.7]]);
    let b1 = arr1(&[0.1_f32, -0.1, 0.0]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[1.0_f32, -0.5, 0.2], [-0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]);
    let b2 = arr1(&[0.2_f32, -0.2, 0.0]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));

    graph.set_output("linear2");

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    // Multi-class classification spec: Y_0 - Y_1 and Y_0 - Y_2
    let spec_matrix = arr2(&[[1.0_f32, -1.0, 0.0], [1.0, 0.0, -1.0]]);

    let spec_bounds = graph
        .propagate_crown_with_specs_and_engine(&input, &spec_matrix, None)
        .unwrap();

    assert_eq!(spec_bounds.shape(), &[2], "Should have 2 spec bounds");

    // Compare against post-hoc interval arithmetic
    let output_bounds = graph.propagate_crown(&input).unwrap();
    let l0 = output_bounds.lower()[[0]];
    let u0 = output_bounds.upper()[[0]];
    let l1 = output_bounds.lower()[[1]];
    let u1 = output_bounds.upper()[[1]];
    let l2 = output_bounds.lower()[[2]];
    let u2 = output_bounds.upper()[[2]];
    // Y_0 - Y_1: [l0 - u1, u0 - l1], Y_0 - Y_2: [l0 - u2, u0 - l2]
    let posthoc_bounds = [(l0 - u1, u0 - l1), (l0 - u2, u0 - l2)];

    println!(
        "Spec-guided: [{:.4}, {:.4}], [{:.4}, {:.4}]",
        spec_bounds.lower()[[0]],
        spec_bounds.upper()[[0]],
        spec_bounds.lower()[[1]],
        spec_bounds.upper()[[1]]
    );
    println!(
        "Post-hoc:    [{:.4}, {:.4}], [{:.4}, {:.4}]",
        posthoc_bounds[0].0, posthoc_bounds[0].1, posthoc_bounds[1].0, posthoc_bounds[1].1
    );

    // Spec-guided should be at least as tight as post-hoc
    for (i, (posthoc_l, posthoc_u)) in posthoc_bounds.iter().enumerate() {
        assert!(
            spec_bounds.lower()[[i]] >= posthoc_l - 1e-5,
            "Spec {} lower {} should be >= post-hoc {}",
            i,
            spec_bounds.lower()[[i]],
            posthoc_l
        );
        assert!(
            spec_bounds.upper()[[i]] <= posthoc_u + 1e-5,
            "Spec {} upper {} should be <= post-hoc {}",
            i,
            spec_bounds.upper()[[i]],
            posthoc_u
        );
    }

    // Verify soundness by sampling
    let linear1_layer = match &graph.nodes.get("linear1").unwrap().layer {
        Layer::Linear(l) => l,
        _ => panic!("Expected Linear"),
    };
    let linear2_layer = match &graph.nodes.get("linear2").unwrap().layer {
        Layer::Linear(l) => l,
        _ => panic!("Expected Linear"),
    };

    for i in 0..100 {
        let t0 = (i * 17 % 100) as f32 / 100.0;
        let t1 = (i * 31 % 100) as f32 / 100.0;
        let x = arr1(&[t0, t1]);

        let h1: Array1<f32> = linear1_layer.weight().dot(&x) + linear1_layer.bias().unwrap();
        let h2 = h1.mapv(|v| v.max(0.0));
        let y: Array1<f32> = linear2_layer.weight().dot(&h2) + linear2_layer.bias().unwrap();

        // Check both specs
        let spec_0 = y[0] - y[1]; // Y_0 - Y_1
        let spec_1 = y[0] - y[2]; // Y_0 - Y_2

        assert!(
            spec_0 >= spec_bounds.lower()[[0]] - 1e-4,
            "Sample {}: Y_0 - Y_1 = {} < lower {}",
            i,
            spec_0,
            spec_bounds.lower()[[0]]
        );
        assert!(
            spec_0 <= spec_bounds.upper()[[0]] + 1e-4,
            "Sample {}: Y_0 - Y_1 = {} > upper {}",
            i,
            spec_0,
            spec_bounds.upper()[[0]]
        );

        assert!(
            spec_1 >= spec_bounds.lower()[[1]] - 1e-4,
            "Sample {}: Y_0 - Y_2 = {} < lower {}",
            i,
            spec_1,
            spec_bounds.lower()[[1]]
        );
        assert!(
            spec_1 <= spec_bounds.upper()[[1]] + 1e-4,
            "Sample {}: Y_0 - Y_2 = {} > upper {}",
            i,
            spec_1,
            spec_bounds.upper()[[1]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_crown_three_input_concat_matches_scalar_crown_3870() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a",
        Layer::Linear(
            LinearLayer::new(arr2(&[[0.5_f32]]), Some(arr1(&[0.1_f32]))).expect("valid a"),
        ),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        Layer::Linear(
            LinearLayer::new(arr2(&[[-1.2_f32]]), Some(arr1(&[0.4_f32]))).expect("valid b"),
        ),
    ));
    graph.add_node(GraphNode::from_input(
        "c",
        Layer::Linear(
            LinearLayer::new(arr2(&[[0.8_f32]]), Some(arr1(&[-0.2_f32]))).expect("valid c"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, -0.5_f32, 0.75_f32]]),
                Some(arr1(&[0.25_f32])),
            )
            .expect("valid out"),
        ),
        vec!["concat".to_string()],
    ));
    graph.set_output("out");

    let input =
        BoundedTensor::new(arr1(&[-0.3_f32]).into_dyn(), arr1(&[0.7_f32]).into_dyn()).unwrap();
    let spec_matrix = arr2(&[[1.0_f32]]);

    let full_output_bounds = graph.propagate_crown(&input).unwrap();
    let spec_guided_bounds = graph
        .propagate_crown_with_specs_and_engine(&input, &spec_matrix, None)
        .expect("spec-guided CROWN should accept a valid three-input Concat");

    assert!(
        (spec_guided_bounds.lower()[[0]] - full_output_bounds.lower()[[0]]).abs() <= 1e-5,
        "spec-guided lower bound should match scalar CROWN after three-input Concat: spec={} crown={}",
        spec_guided_bounds.lower()[[0]],
        full_output_bounds.lower()[[0]],
    );
    assert!(
        (spec_guided_bounds.upper()[[0]] - full_output_bounds.upper()[[0]]).abs() <= 1e-5,
        "spec-guided upper bound should match scalar CROWN after three-input Concat: spec={} crown={}",
        spec_guided_bounds.upper()[[0]],
        full_output_bounds.upper()[[0]],
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_crown_with_node_bounds_matches_with_linear_for_silu_graph() {
    // Regression for #1942: both APIs should produce sound bounds for SiLU.
    // The with_node_bounds path may be more conservative (IBP intermediate
    // bounds) vs with_linear (tighter bounds during backward pass).
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, 0.6], [-0.4, 1.1]]);
    let b1 = arr1(&[0.05_f32, -0.08]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "silu",
        Layer::SiLU(SiLULayer::new()),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[1.2_f32, -0.7], [-0.3, 0.9]]);
    let b2 = arr1(&[0.02_f32, -0.03]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["silu".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        arr1(&[-0.8_f32, -0.5]).into_dyn(),
        arr1(&[0.9_f32, 1.1]).into_dyn(),
    )
    .unwrap();
    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);

    // Provide the same CROWN-IBP tightened intermediate bounds that with_linear
    // computes internally via collect_intermediate_bounds (which selects CROWN-IBP
    // for small graphs). Using plain IBP bounds here would produce legitimately
    // different (wider) results because looser intermediate bounds yield looser
    // SiLU relaxation. See: network/graph_crown/spec_propagation/setup.rs:22.
    let crown_ibp = graph
        .collect_crown_ibp_bounds_dag_with_status_and_deadline(&input, None, None)
        .unwrap();
    let with_node_bounds = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(
            &input,
            &spec_matrix,
            None,
            &crown_ibp.bounds,
        )
        .unwrap();
    let (with_linear, linear_opt) = graph
        .propagate_crown_with_specs_and_engine_with_linear(&input, &spec_matrix, None)
        .unwrap();

    assert!(
        linear_opt.is_some(),
        "SiLU path should keep linear coefficients (no IBP fallback)"
    );
    assert_eq!(with_node_bounds.shape(), with_linear.shape());
    // Both paths now use identical CROWN-IBP intermediate bounds, so only
    // floating-point operation ordering can cause divergence: 1e-6 tolerance.
    assert!(
        (with_node_bounds.lower()[[0]] - with_linear.lower()[[0]]).abs() <= 1e-6,
        "lower divergence: with_node_bounds={}, with_linear={}",
        with_node_bounds.lower()[[0]],
        with_linear.lower()[[0]]
    );
    assert!(
        (with_node_bounds.upper()[[0]] - with_linear.upper()[[0]]).abs() <= 1e-6,
        "upper divergence: with_node_bounds={}, with_linear={}",
        with_node_bounds.upper()[[0]],
        with_linear.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_crown_empty_graph_applies_spec_matrix() {
    // Regression for #1942: empty graph should still evaluate the spec matrix
    // over the input bounds, not silently return input bounds unchanged.
    let graph = GraphNetwork::new();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.0]).into_dyn(),
        arr1(&[2.0_f32, 3.0]).into_dyn(),
    )
    .unwrap();
    let spec_matrix = arr2(&[[1.0_f32, -1.0], [-0.5_f32, 2.0]]);

    let spec_bounds = graph
        .propagate_crown_with_specs_and_engine(&input, &spec_matrix, None)
        .unwrap();
    assert_eq!(spec_bounds.shape(), &[2]);
    assert!((spec_bounds.lower()[[0]] - (-4.0)).abs() <= 1e-6);
    assert!((spec_bounds.upper()[[0]] - 2.0).abs() <= 1e-6);
    assert!((spec_bounds.lower()[[1]] - (-1.0)).abs() <= 1e-6);
    assert!((spec_bounds.upper()[[1]] - 6.5).abs() <= 1e-6);

    let spec_bounds_with_node_bounds = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(
            &input,
            &spec_matrix,
            None,
            &std::collections::HashMap::new(),
        )
        .unwrap();
    assert_eq!(spec_bounds_with_node_bounds.shape(), &[2]);
    assert!((spec_bounds_with_node_bounds.lower()[[0]] - (-4.0)).abs() <= 1e-6);
    assert!((spec_bounds_with_node_bounds.upper()[[0]] - 2.0).abs() <= 1e-6);
    assert!((spec_bounds_with_node_bounds.lower()[[1]] - (-1.0)).abs() <= 1e-6);
    assert!((spec_bounds_with_node_bounds.upper()[[1]] - 6.5).abs() <= 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_crown_degraded_non_finite_falls_back_to_ibp() {
    // Regression for #2566: if spec-guided CROWN concretization degrades to
    // non-finite bounds, the path must fallback to IBP-derived spec bounds.
    let mut graph = GraphNetwork::new();

    // Single linear layer with huge finite weight. Backward coefficients remain
    // finite (~1e38), but concretization against wide finite input overflows to Inf.
    let linear = LinearLayer::new(arr2(&[[1.0e38_f32]]), None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");

    let input = BoundedTensor::new(
        arr1(&[-1.0e10_f32]).into_dyn(),
        arr1(&[1.0e10_f32]).into_dyn(),
    )
    .unwrap();
    let spec_matrix = arr2(&[[1.0_f32]]);

    let (spec_bounds, linear_opt) = graph
        .propagate_crown_with_specs_and_engine_with_linear(&input, &spec_matrix, None)
        .unwrap();

    // fallback_to_ibp returns None for linear coefficients.
    assert!(
        linear_opt.is_none(),
        "degraded spec-guided bounds should fallback to IBP and drop linear coefficients"
    );
    // The IBP fallback itself overflows: 1e38 * ±1e10 = ±1e48 is far outside
    // the f32 range, so the true output really does exceed every finite f32
    // in both directions. new_repaired(Conservative) preserves those ±inf
    // endpoints — a finite repair (the old ±FALLBACK_BOUND clamp) would claim
    // a bound the propagation never established. NaN must still be repaired.
    assert!(
        spec_bounds
            .lower()
            .iter()
            .chain(spec_bounds.upper().iter())
            .all(|v| !v.is_nan()),
        "fallback bounds must not contain NaN: lower={:?} upper={:?}",
        spec_bounds.lower(),
        spec_bounds.upper()
    );
    assert_eq!(
        spec_bounds.lower()[[0]],
        f32::NEG_INFINITY,
        "expected fallback lower -inf (1e38 * -1e10 underflows f32), got {}",
        spec_bounds.lower()[[0]]
    );
    assert_eq!(
        spec_bounds.upper()[[0]],
        f32::INFINITY,
        "expected fallback upper +inf (1e38 * 1e10 overflows f32), got {}",
        spec_bounds.upper()[[0]]
    );
}

// ============================================================
// SPEC-GUIDED CROWN MUST BE TIGHTER THAN IBP (#3037, same class as #2990)
// ============================================================
//
// The spec-guided CROWN path was the last remaining CROWN variant missing
// the IBP intersection. Without it, CROWN's linear relaxation for nonlinear
// layers (ReLU, Sigmoid, etc.) can produce "slack" that makes the output
// bound strictly wider than IBP. The fix intersects (elementwise max for
// lower, min for upper) with IBP forward bounds applied through the spec
// matrix.
//
// Reference: alpha-beta-CROWN bound_general.py:1452-1453 does
// torch.max(crown_lower, ibp_lower), torch.min(crown_upper, ibp_upper).

/// Assert spec-guided CROWN bounds are tighter than IBP and sound via sampling.
fn assert_spec_guided_tighter_than_ibp(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &ndarray::Array2<f32>,
    corner_points: &[ndarray::ArrayD<f32>],
) {
    let ibp_output = graph.propagate_ibp(input).unwrap();
    let spec_guided = graph
        .propagate_crown_with_specs_and_engine(input, spec_matrix, None)
        .unwrap();

    let tol = 1e-5;
    for i in 0..spec_guided.len() {
        assert!(
            spec_guided.upper()[[i]] <= ibp_output.upper()[[i]] + tol,
            "Spec-guided CROWN upper[{i}]={} exceeds IBP upper[{i}]={} (#3037)",
            spec_guided.upper()[[i]],
            ibp_output.upper()[[i]],
        );
        assert!(
            spec_guided.lower()[[i]] >= ibp_output.lower()[[i]] - tol,
            "Spec-guided CROWN lower[{i}]={} below IBP lower[{i}]={} (#3037)",
            spec_guided.lower()[[i]],
            ibp_output.lower()[[i]],
        );
    }

    // Soundness: concrete evaluations at corner points.
    for point in corner_points {
        let concrete = BoundedTensor::concrete(point.clone()).unwrap();
        let out = graph.propagate_ibp(&concrete).unwrap();
        let y = out.lower()[[0]];
        assert!(
            y >= spec_guided.lower()[[0]] - tol,
            "Soundness: f({:?}) = {} < spec lower {}",
            point,
            y,
            spec_guided.lower()[[0]],
        );
        assert!(
            y <= spec_guided.upper()[[0]] + tol,
            "Soundness: f({:?}) = {} > spec upper {}",
            point,
            y,
            spec_guided.upper()[[0]],
        );
    }
}

/// Regression test for #3037: spec-guided CROWN must not be looser than IBP.
///
/// Uses the exact #2990 minimized weights (Linear(2->2) -> ReLU -> Linear(2->1))
/// through the spec-guided GraphNetwork path with identity spec matrix.
/// Before the fix, CROWN upper was ~0.34 while IBP upper was exactly 0.0.
#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_crown_tighter_than_ibp_regression_3037() {
    let graph = build_graph_linear_relu_linear(
        arr2(&[[0.71184605_f32, -0.8403961], [0.0, 0.0]]),
        arr1(&[0.0_f32, 0.0]),
        arr2(&[[-0.8830249_f32, 0.0]]),
        arr1(&[0.0_f32]),
    );
    let input = BoundedTensor::new(
        arr1(&[-0.77716047_f32, 0.0]).into_dyn(),
        arr1(&[0.0_f32, 0.4639826]).into_dyn(),
    )
    .unwrap();
    let spec_matrix = arr2(&[[1.0_f32]]);
    let corners = vec![
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[-0.77716047_f32, 0.0]).into_dyn(),
        arr1(&[0.0_f32, 0.4639826]).into_dyn(),
        arr1(&[-0.77716047_f32, 0.4639826]).into_dyn(),
    ];
    assert_spec_guided_tighter_than_ibp(&graph, &input, &spec_matrix, &corners);
}

/// Same regression as above, but with a non-trivial spec matrix (Y_0 - Y_1).
/// Verifies spec-guided CROWN is tighter than spec-applied IBP for the #2990
/// weights on a 2-output variant.
#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_crown_tighter_than_ibp_with_spec_matrix_3037() {
    // 2-input -> 2-output network with a negative weight row to trigger relaxation slack.
    let graph = build_graph_linear_relu_linear(
        arr2(&[[0.71184605_f32, -0.8403961], [0.0, 0.0]]),
        arr1(&[0.0_f32, 0.0]),
        arr2(&[[-0.8830249_f32, 0.0], [0.5, -0.3]]),
        arr1(&[0.0_f32, 0.1]),
    );
    let input = BoundedTensor::new(
        arr1(&[-0.77716047_f32, 0.0]).into_dyn(),
        arr1(&[0.0_f32, 0.4639826]).into_dyn(),
    )
    .unwrap();

    // Classification spec: Y_0 - Y_1
    let spec_matrix = arr2(&[[1.0_f32, -1.0]]);

    let ibp_output = graph.propagate_ibp(&input).unwrap();
    let spec_guided = graph
        .propagate_crown_with_specs_and_engine(&input, &spec_matrix, None)
        .unwrap();

    // Spec-applied IBP via interval arithmetic:
    // C=[[1,-1]] → lower = ibp_l[0] - ibp_u[1], upper = ibp_u[0] - ibp_l[1]
    let ibp_spec_lower = ibp_output.lower()[[0]] - ibp_output.upper()[[1]];
    let ibp_spec_upper = ibp_output.upper()[[0]] - ibp_output.lower()[[1]];

    let tol = 1e-5;
    assert!(
        spec_guided.upper()[[0]] <= ibp_spec_upper + tol,
        "Spec-guided CROWN upper {} exceeds IBP-spec upper {} (#3037)",
        spec_guided.upper()[[0]],
        ibp_spec_upper,
    );
    assert!(
        spec_guided.lower()[[0]] >= ibp_spec_lower - tol,
        "Spec-guided CROWN lower {} below IBP-spec lower {} (#3037)",
        spec_guided.lower()[[0]],
        ibp_spec_lower,
    );
}

// ============================================================
// PROPTEST: Spec-guided CROWN tighter than IBP (randomized)
// ============================================================

/// Helper: build a Linear -> ReLU -> Linear GraphNetwork.
fn build_graph_linear_relu_linear(
    w1: ndarray::Array2<f32>,
    b1: Array1<f32>,
    w2: ndarray::Array2<f32>,
    b2: Array1<f32>,
) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

fn valid_interval(range: f32) -> impl Strategy<Value = (f32, f32)> {
    (-range..=range)
        .prop_flat_map(move |a| (-range..=range).prop_map(move |b| (a.min(b), a.max(b))))
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Spec-guided CROWN must produce bounds at least as tight as IBP for
    /// Linear(2->2) -> ReLU -> Linear(2->1) with identity spec matrix.
    ///
    /// This is the graph-network, spec-guided variant of the sequential
    /// `crown_tighter_than_ibp_linear_relu_linear` proptest in network.rs.
    /// Catches #3037: spec-guided CROWN missing IBP intersection.
    #[ntest::timeout(10000)]
    #[test]
    fn spec_guided_crown_tighter_than_ibp(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = Array1::from_vec(vec![b2]);

        let graph = build_graph_linear_relu_linear(w1, b1, w2, b2_arr);

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        ).unwrap();

        // Identity spec: bounds on the single output directly.
        let spec_matrix = arr2(&[[1.0_f32]]);

        let ibp_output = graph.propagate_ibp(&input).unwrap();
        let spec_guided_output = graph
            .propagate_crown_with_specs_and_engine(&input, &spec_matrix, None)
            .unwrap();

        let tol = 1e-5;
        prop_assert!(
            spec_guided_output.lower()[[0]] >= ibp_output.lower()[[0]] - tol,
            "Spec-guided CROWN lower ({}) is looser than IBP ({}) — #3037 regression",
            spec_guided_output.lower()[[0]], ibp_output.lower()[[0]]
        );
        prop_assert!(
            spec_guided_output.upper()[[0]] <= ibp_output.upper()[[0]] + tol,
            "Spec-guided CROWN upper ({}) is looser than IBP ({}) — #3037 regression",
            spec_guided_output.upper()[[0]], ibp_output.upper()[[0]]
        );
    }
}
