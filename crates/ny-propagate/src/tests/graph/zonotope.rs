// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork zonotope propagation tests.
use ny_test_utils::assert_bounded_tensor_close;

use crate::*;
use ndarray::{arr1, arr2};

/// Verify that every X @ X^T output falls within the given bounds.
fn assert_matmul_self_transpose_soundness(
    bounds: &BoundedTensor,
    samples: &[ndarray::Array2<f32>],
) {
    for (idx, x) in samples.iter().enumerate() {
        let output = x.dot(&x.t());
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                assert!(
                    output[[i, j]] >= bounds.lower()[[i, j]] - 1e-6,
                    "Soundness: X@X^T[{i},{j}]={} < lower {} (sample {idx})",
                    output[[i, j]],
                    bounds.lower()[[i, j]]
                );
                assert!(
                    output[[i, j]] <= bounds.upper()[[i, j]] + 1e-6,
                    "Soundness: X@X^T[{i},{j}]={} > upper {} (sample {idx})",
                    output[[i, j]],
                    bounds.upper()[[i, j]]
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_matmul_soundness() {
    // Zonotope propagation on Q@K^T with shared input (Q=K=X) should produce
    // sound bounds: every concrete X@X^T must fall within the output interval.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::binary(
        "qk_matmul",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "_input",
        "_input",
    ));
    graph.set_output("qk_matmul");

    let input = BoundedTensor::new(
        arr2(&[[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]]).into_dyn(),
        arr2(&[[1.1, 1.1, 1.1], [1.1, 1.1, 1.1]]).into_dyn(),
    )
    .unwrap();

    let zonotope_output = graph.propagate_zonotope(&input, 0.1).unwrap();
    let ibp_output = graph.propagate_ibp(&input).unwrap();
    assert_eq!(zonotope_output.shape(), &[2, 2]);

    // Interval validity
    for (&l, &u) in zonotope_output
        .lower()
        .iter()
        .zip(zonotope_output.upper().iter())
    {
        assert!(l <= u + 1e-6, "Invalid interval: {} > {}", l, u);
    }

    // Concrete point soundness: 6 samples spanning corners and mixed values
    assert_matmul_self_transpose_soundness(
        &zonotope_output,
        &[
            arr2(&[[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]]),
            arr2(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            arr2(&[[1.1, 1.1, 1.1], [1.1, 1.1, 1.1]]),
            arr2(&[[0.9, 1.1, 0.9], [1.1, 0.9, 1.1]]),
            arr2(&[[0.9, 0.9, 0.9], [1.1, 1.1, 1.1]]),
            arr2(&[[1.1, 0.9, 1.0], [0.9, 1.1, 1.0]]),
        ],
    );

    // Zonotope should not be dramatically wider than IBP
    assert!(
        zonotope_output.max_width() <= ibp_output.max_width() * 2.0,
        "Zonotope width {} >> IBP width {}",
        zonotope_output.max_width(),
        ibp_output.max_width()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_matmul_3d_smoke() {
    // Smoke test: zonotope propagation should support batched sequence tensors
    // with shape (batch, seq, dim) for Q@K^T patterns.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::binary(
        "qk_matmul",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "_input",
        "_input",
    ));
    graph.set_output("qk_matmul");

    let batch = 2_usize;
    let seq = 3_usize;
    let dim = 4_usize;
    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(vec![batch, seq, dim], -1.0_f32),
        ndarray::ArrayD::from_elem(vec![batch, seq, dim], 1.0_f32),
    )
    .unwrap();

    let out = graph.propagate_zonotope(&input, 0.1).unwrap();
    assert_eq!(out.shape(), &[batch, seq, seq]);
    for (l, u) in out.lower().iter().zip(out.upper().iter()) {
        assert!(l.is_finite() && u.is_finite(), "Non-finite bounds");
        assert!(*l <= *u + 1e-6, "Invalid interval: {} > {}", l, u);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_add_operation() {
    // Test zonotope propagation with Add operation
    let mut graph = GraphNetwork::new();

    // Create: input1 + input2 (where both are the same input)
    graph.add_node(GraphNode::binary(
        "sum",
        Layer::Add(AddLayer),
        "_input",
        "_input",
    ));
    graph.set_output("sum");

    // 2D input
    let input = BoundedTensor::new(
        arr2(&[[0.9, 1.9]]).into_dyn(),
        arr2(&[[1.1, 2.1]]).into_dyn(),
    )
    .unwrap();

    let result = graph.propagate_zonotope(&input, 0.1).unwrap();

    // x + x = 2x, so bounds should be approximately doubled
    // Center is [1, 2], so result should be [2, 4] with doubled width
    assert_eq!(result.shape(), &[1, 2]);

    // Check center is approximately [2, 4]
    let center_0 = f32::midpoint(result.lower()[[0, 0]], result.upper()[[0, 0]]);
    let center_1 = f32::midpoint(result.lower()[[0, 1]], result.upper()[[0, 1]]);
    assert!(
        (center_0 - 2.0).abs() < 0.1,
        "Center[0] should be ~2, got {}",
        center_0
    );
    assert!(
        (center_1 - 4.0).abs() < 0.1,
        "Center[1] should be ~4, got {}",
        center_1
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_fallback_to_ibp() {
    // Test that zonotope propagation falls back to IBP for unsupported operations
    let mut graph = GraphNetwork::new();

    // Create: input -> ReLU (not supported by zonotope)
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr2(&[[-0.5, 0.5]]).into_dyn(),
        arr2(&[[0.5, 1.5]]).into_dyn(),
    )
    .unwrap();

    // Should not error - should fall back to IBP
    let result = graph.propagate_zonotope(&input, 0.1).unwrap();

    // ReLU output shape should match input
    assert_eq!(result.shape(), &[1, 2]);

    // ReLU bounds should be valid
    // Element [0,0] crosses zero: [max(0, -0.5), max(0, 0.5)] = [0, 0.5]
    // Element [0,1] is positive: [max(0, 0.5), max(0, 1.5)] = [0.5, 1.5]
    assert!(result.lower()[[0, 0]] >= -0.01, "ReLU lower should be >= 0");
    assert!(
        result.lower()[[0, 1]] >= 0.49,
        "ReLU lower[1] should be >= 0.5"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_non_2d_fallback() {
    // Test that <2D input falls back to IBP
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    // 1D input (not 2D)
    let input =
        BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn()).unwrap();

    // Should fall back to IBP for non-2D input
    let result = graph.propagate_zonotope(&input, 0.1).unwrap();

    // Should produce valid output
    assert_eq!(result.shape(), &[2]);
}

/// #3548: non-finite graph input should be rejected as a fallback-class error
/// before zonotope construction so callers can degrade to IBP.
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_non_finite_input_returns_unsupported_configuration() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new_unchecked(
        arr2(&[[0.0_f32, -1.0]]).into_dyn(),
        arr2(&[[f32::INFINITY, 1.0]]).into_dyn(),
    )
    .unwrap();

    let err = graph.propagate_zonotope(&input, 0.0).unwrap_err();
    match err {
        NyError::UnsupportedConfiguration(msg) => {
            assert!(
                msg.contains("finite input bounds"),
                "expected non-finite zonotope guard diagnostic, got: {msg}"
            );
        }
        other => panic!("expected UnsupportedConfiguration, got: {other:?}"),
    }
}

/// Regression test for #2470: GELU zonotope must use GELU math, not SiLU.
///
/// Verifies that zonotope propagation through a GELU layer produces bounds
/// that contain sampled GELU outputs. Before the fix, this used `silu_affine()`
/// which evaluates SiLU(x) = x·σ(x) instead of GELU(x) = x·Φ(x/√2).
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_gelu_soundness_2470() {
    use crate::layers::softmax::{GELULayer, GeluApproximation};

    // Test both erf and tanh approximations
    for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "gelu",
            Layer::GELU(GELULayer::new(approx)),
        ));
        graph.set_output("gelu");

        // Use a 2D input to exercise the zonotope path (not IBP fallback).
        // Range [-2, 2] covers the region where GELU and SiLU differ most (~x ≈ -1.5).
        let input = BoundedTensor::new(
            arr2(&[[-2.0, -1.0, 0.0, 1.0]]).into_dyn(),
            arr2(&[[-1.0, 0.0, 1.0, 2.0]]).into_dyn(),
        )
        .unwrap();

        let result = graph.propagate_zonotope(&input, 0.0).unwrap();
        assert_eq!(result.shape(), &[1, 4]);

        // Soundness check: sample concrete points and verify they're within bounds.
        // GELU values at key points:
        // GELU_erf(-2) ≈ -0.0455, GELU_erf(-1) ≈ -0.1587, GELU_erf(0) = 0,
        // GELU_erf(1) ≈ 0.8413, GELU_erf(2) ≈ 1.9545
        let gelu_fn: fn(f32) -> f32 = match approx {
            GeluApproximation::Erf => |x| {
                let inv_sqrt2: f32 = 1.0 / 2.0_f32.sqrt();
                0.5 * x * (1.0 + libm::erff(x * inv_sqrt2))
            },
            GeluApproximation::Tanh => |x| {
                let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
            },
        };

        // Sample 11 points per element and verify containment
        for elem in 0..4 {
            let lo = input.lower()[[0, elem]];
            let hi = input.upper()[[0, elem]];
            let bound_lo = result.lower()[[0, elem]];
            let bound_hi = result.upper()[[0, elem]];

            for i in 0..=10 {
                let t = i as f32 / 10.0;
                let x = lo + (hi - lo) * t;
                let y = gelu_fn(x);

                assert!(
                    y >= bound_lo - 1e-6,
                    "{approx:?} GELU({x}) = {y} < lower bound {bound_lo} for element {elem}",
                );
                assert!(
                    y <= bound_hi + 1e-6,
                    "{approx:?} GELU({x}) = {y} > upper bound {bound_hi} for element {elem}",
                );
            }
        }

        // Also verify the zonotope bounds stay comparable to IBP.
        // Width factor per approximation:
        // - Erf: the Taylor remainder uses the exact interval max of
        //   |GELU''(x)| = |φ(x)·(2-x²)| (endpoints + interior extrema), so
        //   the zonotope stays within 2x of IBP on these intervals.
        // - Tanh: tanh-GELU'' has no closed-form extrema, so the remainder
        //   uses the proven global bound |GELU''| <= 0.8. For element 0
        //   ([-2,-1], r=0.5) that adds 2·(0.8·r²/2) = 0.2 to an affine width
        //   of |f'(-1.5)|·1 ≈ 0.128 vs IBP width ≈ 0.113 → ratio ≈ 2.9.
        let width_factor = match approx {
            GeluApproximation::Erf => 2.0,
            GeluApproximation::Tanh => 3.0,
        };
        let ibp_result = graph.propagate_ibp(&input).unwrap();
        for elem in 0..4 {
            let z_width = result.upper()[[0, elem]] - result.lower()[[0, elem]];
            let ibp_width = ibp_result.upper()[[0, elem]] - ibp_result.lower()[[0, elem]];
            assert!(
                z_width <= ibp_width * width_factor,
                "{approx:?} zonotope width {z_width} is much larger than IBP width {ibp_width} for element {elem}",
            );
        }
    }
}

/// #2991: Empty-input AddConstant is now caught at construction time by
/// GraphNode::try_new() arity validation (#2481, #2686).
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_node_missing_unary_input_returns_invalid_spec() {
    let err = GraphNode::try_new(
        "broken_add_const",
        Layer::AddConstant(AddConstantLayer::new(arr1(&[1.0_f32]).into_dyn())),
        vec![],
    )
    .expect_err("empty-input AddConstant should return InvalidSpec at construction");

    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {}",
        err
    );
    let msg = err.to_string();
    assert!(
        msg.contains("broken_add_const") && msg.contains("1 input"),
        "expected missing-input diagnostic with node name, got: {}",
        msg
    );
}

/// #2991: Empty-input ReLU is now caught at construction time.
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_fallback_missing_unary_input_returns_invalid_spec() {
    let err = GraphNode::try_new("relu", Layer::ReLU(ReLULayer), vec![])
        .expect_err("empty-input ReLU should return InvalidSpec at construction");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {}",
        err
    );
    let msg = err.to_string();
    assert!(
        msg.contains("relu") && msg.contains("1 input"),
        "expected missing-input diagnostic with node name, got: {}",
        msg
    );
}

fn identity_linear(dim: usize) -> LinearLayer {
    LinearLayer::new(ndarray::Array2::<f32>::eye(dim), None).unwrap()
}

fn build_attention_scores_graph(dim: usize) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::Linear(identity_linear(dim)),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::Linear(identity_linear(dim)),
    ));
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));
    graph.set_output("scores");
    graph
}

fn build_attention_context_graph(dim: usize) -> GraphNetwork {
    let mut graph = build_attention_scores_graph(dim);
    graph.add_node(GraphNode::from_input(
        "v",
        Layer::Linear(identity_linear(dim)),
    ));
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(0)),
        vec!["scores".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "out",
        Layer::MatMul(MatMulLayer::new(false, None)),
        "probs",
        "v",
    ));
    graph.set_output("out");
    graph
}

/// #3457/#3464: unsupported zonotope softmax configs must downgrade to
/// `UnsupportedConfiguration` so graph-level propagation falls back to IBP
/// instead of surfacing a hard `InvalidSpec`.
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_softmax_unsupported_axis_falls_back_to_ibp() {
    let dim = 3_usize;

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "scores",
        Layer::Linear(LinearLayer::new(ndarray::Array2::<f32>::eye(dim), None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(0)),
        vec!["scores".to_string()],
    ));
    graph.set_output("probs");

    let input = BoundedTensor::new(
        arr2(&[[0.0_f32, 0.5, -0.5], [1.0, -1.0, 0.25]]).into_dyn(),
        arr2(&[[0.2_f32, 0.7, -0.3], [1.2, -0.8, 0.45]]).into_dyn(),
    )
    .unwrap();

    let zonotope_bounds = graph.propagate_zonotope(&input, 0.1).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    assert_eq!(zonotope_bounds.shape(), &[2, dim]);
    assert_bounded_tensor_close(&zonotope_bounds, &ibp_bounds, 1e-6, "softmax fallback");
}

/// #318, #3464: attention-context `MatMul(transpose_b=false)` should stay on a
/// sound zonotope path even when Softmax and V no longer share symbols.
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_context_matmul_stays_tighter_than_ibp() {
    let seq = 3_usize;
    let dim = 2_usize;
    let epsilon = 0.05_f32;

    let graph = build_attention_context_graph(dim);

    let input = BoundedTensor::new(
        arr2(&[[0.2_f32, -0.4], [0.7, 0.1], [-0.3, 0.5]]).into_dyn(),
        arr2(&[[0.3_f32, -0.3], [0.8, 0.2], [-0.2, 0.6]]).into_dyn(),
    )
    .unwrap();

    let zonotope_bounds = graph.propagate_zonotope(&input, epsilon).unwrap();

    let scores_graph = build_attention_scores_graph(dim);
    let scores_bounds = scores_graph.propagate_zonotope(&input, epsilon).unwrap();
    let probs_bounds = SoftmaxLayer::new(0).propagate_ibp(&scores_bounds).unwrap();
    let value_bounds = BoundedTensor::new(input.lower().clone(), input.upper().clone()).unwrap();

    let expected_bounds = MatMulLayer::new(false, None)
        .propagate_ibp_binary(&probs_bounds, &value_bounds)
        .unwrap();

    assert_eq!(zonotope_bounds.shape(), &[seq, dim]);
    assert_eq!(zonotope_bounds.shape(), expected_bounds.shape());
    assert!(
        zonotope_bounds
            .lower()
            .iter()
            .zip(zonotope_bounds.upper().iter())
            .all(|(&lo, &hi)| lo <= hi + 1e-6),
        "context bounds must remain ordered"
    );
    assert!(
        zonotope_bounds.max_width() <= expected_bounds.max_width(),
        "context zonotope width {} should be no wider than IBP fallback {}",
        zonotope_bounds.max_width(),
        expected_bounds.max_width()
    );
}
