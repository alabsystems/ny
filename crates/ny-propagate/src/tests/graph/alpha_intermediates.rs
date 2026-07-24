// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for `dag_alpha_backward_pass_with_intermediates` and
//! `compute_graph_chain_rule_gradients` — exercised via the public
//! `propagate_alpha_crown_with_config` API with `GradientMethod::AnalyticChain`.
//!
//! Covers #1959 acceptance criteria:
//! - Direct tests exist for `dag_alpha_backward_pass_with_intermediates`
//! - Direct tests exist for `compute_graph_chain_rule_gradients`
//!
//! Reference: designs/2026-02-10-network-beta-crown-coverage-wave-plan.md Step 1

use crate::*;
use ndarray::{arr1, arr2};
use ny_test_utils::{assert_bounded_tensor_close, assert_bounds_do_not_loosen};

/// Build a small config that forces AnalyticChain gradient method.
fn analytic_chain_config(iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        gradient_method: GradientMethod::AnalyticChain,
        learning_rate: 0.15,
        lr_decay: 0.98,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        ..AlphaCrownConfig::default()
    }
}

fn total_width(bounds: &BoundedTensor) -> f32 {
    bounds
        .upper()
        .iter()
        .zip(bounds.lower().iter())
        .map(|(&upper, &lower)| upper - lower)
        .sum()
}

/// Compute manual forward pass through a Linear->ReLU->Linear->ReLU network.
fn manual_forward_2layer(
    w1: &[[f32; 2]; 3],
    b1: &[f32; 3],
    w2: &[[f32; 3]; 2],
    x: &[f32; 2],
) -> [f32; 2] {
    let z1 = [
        (w1[0][0] * x[0] + w1[0][1] * x[1] + b1[0]).max(0.0),
        (w1[1][0] * x[0] + w1[1][1] * x[1] + b1[1]).max(0.0),
        (w1[2][0] * x[0] + w1[2][1] * x[1] + b1[2]).max(0.0),
    ];
    [
        (w2[0][0] * z1[0] + w2[0][1] * z1[1] + w2[0][2] * z1[2]).max(0.0),
        (w2[1][0] * z1[0] + w2[1][1] * z1[1] + w2[1][2] * z1[2]).max(0.0),
    ]
}

/// Weights returned from `build_2layer_relu_graph`.
type TwoLayerWeights = (GraphNetwork, [[f32; 2]; 3], [f32; 3], [[f32; 3]; 2]);

/// Build a standard 2-hidden-layer ReLU graph: Input(2) -> Linear(3) -> ReLU -> Linear(2) -> ReLU
fn build_2layer_relu_graph() -> TwoLayerWeights {
    let w1_vals = [[1.0f32, -0.5], [0.5, 1.0], [-1.0, 0.3]];
    let b1_vals = [0.0f32, 0.1, -0.1];
    let w2_vals = [[0.5f32, -0.3, 0.8], [0.2, 0.6, -0.4]];

    let mut graph = GraphNetwork::new();

    let w1 = arr2(&w1_vals);
    let linear1 = LinearLayer::new(w1, Some(arr1(&b1_vals))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&w2_vals);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.set_output("relu2");

    (graph, w1_vals, b1_vals, w2_vals)
}

/// Build a DAG with skip connection (residual block):
///   Input -> Linear1 -> ReLU -> Linear2 -+-> Add -> ReLU -> Output
///   Input -> SkipLinear (identity) -------+
fn build_skip_connection_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.5_f32, -0.3], [0.2, 0.6]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));

    // Skip connection: identity projection
    let w_skip = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let linear_skip = LinearLayer::new(w_skip, None).unwrap();
    graph.add_node(GraphNode::from_input(
        "skip_linear",
        Layer::Linear(linear_skip),
    ));

    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["linear2".to_string(), "skip_linear".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));

    graph.set_output("relu2");
    graph
}

fn build_3layer_relu_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    // Keep the first two layers aligned with the proven refresh fixture used by
    // graph_alpha/reference_bounds tests, then extend with a third activation.
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1, -0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));

    let w3 = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let linear3 = LinearLayer::new(w3, Some(arr1(&[-0.1_f32, -0.1]))).unwrap();
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(linear3),
        vec!["relu2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["linear3".to_string()],
    ));
    graph.set_output("relu3");

    graph
}

// ---------------------------------------------------------------------------
// Tests for dag_alpha_backward_pass_with_intermediates (via public API)
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_sequential_graph_soundness() {
    // Exercises dag_alpha_backward_pass_with_intermediates on a sequential graph.
    // The function is called internally when GradientMethod::AnalyticChain is active.
    let (graph, w1, b1, w2) = build_2layer_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let config = analytic_chain_config(20);
    let bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();

    // Verify soundness: sample 100 inputs, check all outputs within bounds
    for i in 0..100 {
        let t1 = (i * 7 % 100) as f32 / 100.0;
        let t2 = (i * 11 % 100) as f32 / 100.0;
        let x = [-0.5 + t1, -0.5 + t2];
        let y = manual_forward_2layer(&w1, &b1, &w2, &x);

        for (j, &yj) in y.iter().enumerate() {
            assert!(
                yj >= bounds.lower()[[j]] - 1e-5 && yj <= bounds.upper()[[j]] + 1e-5,
                "AnalyticChain output {} unsound: {} not in [{}, {}] for input {:?}",
                j,
                yj,
                bounds.lower()[[j]],
                bounds.upper()[[j]],
                x
            );
        }
    }

    // Bounds must be finite
    assert!(bounds.lower().iter().all(|v| v.is_finite()));
    assert!(bounds.upper().iter().all(|v| v.is_finite()));
}

#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_tighter_than_crown() {
    // AnalyticChain should produce bounds at least as tight as legacy
    // fixed-slope CROWN, since it optimizes alpha parameters via chain-rule
    // gradients.
    let (graph, _w1, _b1, _w2) = build_2layer_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let config = analytic_chain_config(30);
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();
    let fixed_slope_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();

    for i in 0..2 {
        let alpha_width = alpha_bounds.upper()[[i]] - alpha_bounds.lower()[[i]];
        let crown_width = fixed_slope_bounds.upper()[[i]] - fixed_slope_bounds.lower()[[i]];
        assert!(
            alpha_width <= crown_width + 1e-4,
            "AnalyticChain width {} should be <= fixed-slope CROWN width {} at output {}",
            alpha_width,
            crown_width,
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_dag_skip_connection_soundness() {
    // Exercises dag_alpha_backward_pass_with_intermediates on a DAG topology
    // with skip connections. The DAG backward pass must correctly accumulate
    // bounds through multiple paths.
    let graph = build_skip_connection_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let config = analytic_chain_config(20);
    let bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();

    // Verify soundness by sampling
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let b1 = arr1(&[0.0_f32, 0.1]);
    let w2 = arr2(&[[0.5_f32, -0.3], [0.2, 0.6]]);
    let w_skip = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);

    // Cover the full 2D perturbation box (not just a 1D slice) so skip-path
    // interactions are exercised across both input dimensions.
    for i in 0..=10 {
        let t1 = i as f32 / 10.0;
        for k in 0..=10 {
            let t2 = k as f32 / 10.0;
            let sample = arr1(&[-0.5 + t1, -0.5 + t2]);

            // Forward: linear1 -> relu -> linear2 + skip -> relu
            let h1 = w1.dot(&sample) + &b1;
            let h1_relu = h1.mapv(|v| v.max(0.0));
            let h2 = w2.dot(&h1_relu);
            let skip_out = w_skip.dot(&sample);
            let add_out = &h2 + &skip_out;
            let output = add_out.mapv(|v| v.max(0.0));

            for j in 0..2 {
                assert!(
                    output[[j]] >= bounds.lower()[[j]] - 1e-4
                        && output[[j]] <= bounds.upper()[[j]] + 1e-4,
                    "DAG AnalyticChain unsound at dim {}: {} not in [{}, {}] for sample {:?}",
                    j,
                    output[[j]],
                    bounds.lower()[[j]],
                    bounds.upper()[[j]],
                    sample
                );
            }
        }
    }

    // Bounds must be finite
    assert!(bounds.lower().iter().all(|v| v.is_finite()));
    assert!(bounds.upper().iter().all(|v| v.is_finite()));
}

#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_dag_tighter_than_crown() {
    // AnalyticChain with DAG should be at least as tight as legacy fixed-slope CROWN.
    let graph = build_skip_connection_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let config = analytic_chain_config(30);
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();
    let fixed_slope_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();

    for i in 0..2 {
        let alpha_width = alpha_bounds.upper()[[i]] - alpha_bounds.lower()[[i]];
        let crown_width = fixed_slope_bounds.upper()[[i]] - fixed_slope_bounds.lower()[[i]];
        assert!(
            alpha_width <= crown_width + 1e-4,
            "DAG AnalyticChain width {} should be <= fixed-slope CROWN width {} at output {}",
            alpha_width,
            crown_width,
            i
        );
    }
}

// ---------------------------------------------------------------------------
// Tests for compute_graph_chain_rule_gradients (via observable behavior)
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_produces_tighter_bounds_than_zero_iterations() {
    // If AnalyticChain chain-rule gradients are computed and applied correctly,
    // then running with iterations > 0 should produce tighter bounds than
    // iterations = 0 (which is just CROWN with default alpha).
    let (graph, _w1, _b1, _w2) = build_2layer_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let config_0iter = analytic_chain_config(0);
    let config_30iter = analytic_chain_config(30);

    let bounds_0 = graph
        .propagate_alpha_crown_with_config(&input, &config_0iter)
        .unwrap();
    let bounds_30 = graph
        .propagate_alpha_crown_with_config(&input, &config_30iter)
        .unwrap();

    // Per-output checks avoid cancelation where one output improves but another degrades.
    let mut strict_improvement = false;
    for i in 0..2 {
        let width_0 = bounds_0.upper()[[i]] - bounds_0.lower()[[i]];
        let width_30 = bounds_30.upper()[[i]] - bounds_30.lower()[[i]];
        assert!(
            width_30 <= width_0 + 1e-5,
            "AnalyticChain with 30 iters widened output {}: width_30={} > width_0={}",
            i,
            width_30,
            width_0
        );
        if width_30 + 1e-6 < width_0 {
            strict_improvement = true;
        }
    }
    assert!(
        strict_improvement,
        "Expected at least one output width to improve with 30 AnalyticChain iterations"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_matches_or_beats_spsa() {
    // AnalyticChain uses exact chain-rule gradients, while SPSA uses stochastic
    // approximation. With enough iterations, AnalyticChain should match or beat SPSA.
    let (graph, _w1, _b1, _w2) = build_2layer_relu_graph();

    let input = BoundedTensor::new(
        arr1(&[-0.3_f32, -0.3]).into_dyn(),
        arr1(&[0.3_f32, 0.3]).into_dyn(),
    )
    .unwrap();

    let spsa_config = AlphaCrownConfig {
        iterations: 30,
        gradient_method: GradientMethod::Spsa,
        learning_rate: 0.15,
        lr_decay: 0.98,
        spsa_samples: 4,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        ..AlphaCrownConfig::default()
    };

    let chain_config = analytic_chain_config(30);

    let spsa_bounds = graph
        .propagate_alpha_crown_with_config(&input, &spsa_config)
        .unwrap();
    let chain_bounds = graph
        .propagate_alpha_crown_with_config(&input, &chain_config)
        .unwrap();

    let spsa_width: f32 = spsa_bounds
        .upper()
        .iter()
        .zip(spsa_bounds.lower().iter())
        .map(|(u, l)| u - l)
        .sum();
    let chain_width: f32 = chain_bounds
        .upper()
        .iter()
        .zip(chain_bounds.lower().iter())
        .map(|(u, l)| u - l)
        .sum();

    // Both should be sound — chain should be comparable or better.
    // Allow some tolerance since SPSA is stochastic and might get lucky.
    assert!(
        chain_width <= spsa_width + 0.1,
        "AnalyticChain width {} significantly worse than SPSA width {} \
         — chain-rule gradients may be incorrect",
        chain_width,
        spsa_width
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_no_relu_equals_crown() {
    // Without ReLU nodes, there are no alpha parameters to optimize.
    // AnalyticChain should produce identical bounds to CROWN.
    let mut graph = GraphNetwork::new();

    let w = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let config = analytic_chain_config(20);
    let chain_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();
    let fixed_slope_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();

    for i in 0..2 {
        assert!(
            (chain_bounds.lower()[[i]] - fixed_slope_bounds.lower()[[i]]).abs() < 1e-5,
            "No-ReLU AnalyticChain lower {} != fixed-slope CROWN lower {} at dim {}",
            chain_bounds.lower()[[i]],
            fixed_slope_bounds.lower()[[i]],
            i
        );
        assert!(
            (chain_bounds.upper()[[i]] - fixed_slope_bounds.upper()[[i]]).abs() < 1e-5,
            "No-ReLU AnalyticChain upper {} != fixed-slope CROWN upper {} at dim {}",
            chain_bounds.upper()[[i]],
            fixed_slope_bounds.upper()[[i]],
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_wider_perturbation_wider_bounds() {
    // Monotonicity property: wider input perturbation → wider output bounds.
    // This tests that the intermediate computation preserves monotonicity.
    let (graph, _w1, _b1, _w2) = build_2layer_relu_graph();

    let config = analytic_chain_config(15);

    let tight_input = BoundedTensor::new(
        arr1(&[-0.1_f32, -0.1]).into_dyn(),
        arr1(&[0.1_f32, 0.1]).into_dyn(),
    )
    .unwrap();

    let wide_input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let tight_bounds = graph
        .propagate_alpha_crown_with_config(&tight_input, &config)
        .unwrap();
    let wide_bounds = graph
        .propagate_alpha_crown_with_config(&wide_input, &config)
        .unwrap();

    let tight_width: f32 = tight_bounds
        .upper()
        .iter()
        .zip(tight_bounds.lower().iter())
        .map(|(u, l)| u - l)
        .sum();
    let wide_width: f32 = wide_bounds
        .upper()
        .iter()
        .zip(wide_bounds.lower().iter())
        .map(|(u, l)| u - l)
        .sum();

    assert!(
        wide_width >= tight_width - 1e-5,
        "Wider input should give wider bounds: tight_width={} > wide_width={}",
        tight_width,
        wide_width
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_single_relu_layer() {
    // Minimal network: Linear -> ReLU (single ReLU layer).
    // Tests that intermediates are correctly stored for a single ReLU.
    let mut graph = GraphNetwork::new();

    let w = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-0.3, 0.8]]);
    let linear = LinearLayer::new(w, Some(arr1(&[0.1, -0.2, 0.0]))).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let config = analytic_chain_config(20);
    let bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();
    let fixed_slope_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();

    // Soundness: sample inputs
    let w_vals = [[1.0f32, -0.5], [0.5, 1.0], [-0.3, 0.8]];
    let b_vals = [0.1f32, -0.2, 0.0];
    for i in 0..50 {
        let t1 = (i * 7 % 50) as f32 / 50.0;
        let t2 = (i * 13 % 50) as f32 / 50.0;
        let x = [-0.5 + t1, -0.5 + t2];
        for j in 0..3 {
            let y = (w_vals[j][0] * x[0] + w_vals[j][1] * x[1] + b_vals[j]).max(0.0);
            assert!(
                y >= bounds.lower()[[j]] - 1e-5 && y <= bounds.upper()[[j]] + 1e-5,
                "Single ReLU AnalyticChain unsound at dim {}: {} not in [{}, {}]",
                j,
                y,
                bounds.lower()[[j]],
                bounds.upper()[[j]]
            );
        }
    }

    // Tightness: at least as tight as fixed-slope CROWN
    for i in 0..3 {
        let alpha_width = bounds.upper()[[i]] - bounds.lower()[[i]];
        let crown_width = fixed_slope_bounds.upper()[[i]] - fixed_slope_bounds.lower()[[i]];
        assert!(
            alpha_width <= crown_width + 1e-4,
            "Single ReLU AnalyticChain width {} > fixed-slope CROWN width {} at dim {}",
            alpha_width,
            crown_width,
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_3layer_deep_network() {
    // Deeper network: Linear->ReLU->Linear->ReLU->Linear->ReLU
    // Tests that chain-rule gradient computation correctly chains through
    // multiple ReLU layers, which requires correct intermediate A-matrix storage.
    let mut graph = GraphNetwork::new();

    // Layer 1: 2 -> 3
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-0.3, 0.7]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1, -0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    // Layer 2: 3 -> 3
    let w2 = arr2(&[[0.5_f32, -0.3, 0.2], [0.1, 0.6, -0.4], [-0.2, 0.3, 0.5]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));

    // Layer 3: 3 -> 2
    let w3 = arr2(&[[0.4_f32, -0.2, 0.6], [0.3, 0.5, -0.1]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(linear3),
        vec!["relu2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["linear3".to_string()],
    ));
    graph.set_output("relu3");

    let input = BoundedTensor::new(
        arr1(&[-0.3_f32, -0.3]).into_dyn(),
        arr1(&[0.3_f32, 0.3]).into_dyn(),
    )
    .unwrap();

    let config = analytic_chain_config(30);
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();
    let fixed_slope_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();

    // Bounds must be finite
    assert!(
        alpha_bounds.lower().iter().all(|v| v.is_finite()),
        "3-layer AnalyticChain produced non-finite lower bounds"
    );
    assert!(
        alpha_bounds.upper().iter().all(|v| v.is_finite()),
        "3-layer AnalyticChain produced non-finite upper bounds"
    );

    // Lower must be <= upper
    for i in 0..2 {
        assert!(
            alpha_bounds.lower()[[i]] <= alpha_bounds.upper()[[i]] + 1e-5,
            "3-layer AnalyticChain: lower {} > upper {} at dim {}",
            alpha_bounds.lower()[[i]],
            alpha_bounds.upper()[[i]],
            i
        );
    }

    // At least as tight as fixed-slope CROWN
    for i in 0..2 {
        let alpha_width = alpha_bounds.upper()[[i]] - alpha_bounds.lower()[[i]];
        let crown_width = fixed_slope_bounds.upper()[[i]] - fixed_slope_bounds.lower()[[i]];
        assert!(
            alpha_width <= crown_width + 1e-4,
            "3-layer AnalyticChain width {} > fixed-slope CROWN width {} at dim {}",
            alpha_width,
            crown_width,
            i
        );
    }

    // Soundness sampling (manual forward pass for 3-layer network)
    let w1v = [[1.0f32, -0.5], [0.5, 1.0], [-0.3, 0.7]];
    let b1v = [0.0f32, 0.1, -0.1];
    let w2v = [[0.5f32, -0.3, 0.2], [0.1, 0.6, -0.4], [-0.2, 0.3, 0.5]];
    let w3v = [[0.4f32, -0.2, 0.6], [0.3, 0.5, -0.1]];
    for i in 0..50 {
        let t1 = (i * 7 % 50) as f32 / 50.0;
        let t2 = (i * 13 % 50) as f32 / 50.0;
        let x = [-0.3 + 0.6 * t1, -0.3 + 0.6 * t2];

        // Layer 1
        let mut z1 = [0.0f32; 3];
        for j in 0..3 {
            z1[j] = (w1v[j][0] * x[0] + w1v[j][1] * x[1] + b1v[j]).max(0.0);
        }
        // Layer 2
        let mut z2 = [0.0f32; 3];
        for j in 0..3 {
            z2[j] = (w2v[j][0] * z1[0] + w2v[j][1] * z1[1] + w2v[j][2] * z1[2]).max(0.0);
        }
        // Layer 3
        let mut z3 = [0.0f32; 2];
        for j in 0..2 {
            z3[j] = (w3v[j][0] * z2[0] + w3v[j][1] * z2[1] + w3v[j][2] * z2[2]).max(0.0);
        }

        for (j, &z3j) in z3.iter().enumerate() {
            assert!(
                z3j >= alpha_bounds.lower()[[j]] - 1e-4 && z3j <= alpha_bounds.upper()[[j]] + 1e-4,
                "3-layer unsound at dim {}: {} not in [{}, {}]",
                j,
                z3j,
                alpha_bounds.lower()[[j]],
                alpha_bounds.upper()[[j]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_sequential_reference_carry_forward_tightens_targets_3677() {
    let graph = build_3layer_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();
    let config = analytic_chain_config(30);

    let (frozen_bounds, frozen_reference_bounds, frozen_targets, _, _) = graph
        .propagate_alpha_crown_with_reference_mode_for_test(&input, &config, false)
        .expect("frozen-reference baseline should succeed");
    let (
        carried_bounds,
        carried_reference_bounds,
        carried_targets,
        refresh_attempts,
        tightened_targets_total,
    ) = graph
        .propagate_alpha_crown_with_reference_mode_for_test(&input, &config, true)
        .expect("carry-forward reference mode should succeed");

    assert_eq!(
        frozen_targets, carried_targets,
        "#3677 target list should stay stable across refresh modes"
    );
    assert_eq!(
        carried_targets,
        vec![
            "linear1".to_string(),
            "linear2".to_string(),
            "linear3".to_string()
        ],
        "#3677 deep sequential graph should refresh all three activation inputs"
    );

    let mut strict_target_improvement = false;
    let mut target_width_summaries = Vec::new();
    for target in &carried_targets {
        let frozen = frozen_reference_bounds
            .get(target)
            .unwrap_or_else(|| panic!("frozen reference bounds missing target {target}"));
        let carried = carried_reference_bounds
            .get(target)
            .unwrap_or_else(|| panic!("carry-forward reference bounds missing target {target}"));
        assert_bounds_do_not_loosen(carried, frozen, 1e-6, &format!("#3677 target {target}"));
        let frozen_width = total_width(frozen);
        let carried_width = total_width(carried);
        target_width_summaries.push(format!(
            "{target}: frozen_width={frozen_width:.6}, carried_width={carried_width:.6}"
        ));
        strict_target_improvement |= carried_width + 1e-7 < frozen_width;
    }

    assert!(
        strict_target_improvement,
        "#3677 carry-forward should strictly tighten at least one activation-input target (refresh_attempts={refresh_attempts}, tightened_targets_total={tightened_targets_total}; {})",
        target_width_summaries.join("; ")
    );
    assert_bounds_do_not_loosen(
        &carried_bounds,
        &frozen_bounds,
        1e-6,
        "#3677 sequential output",
    );
}

/// Build a 4-layer ReLU graph for deep-sequential tests.
/// Input(2) -> Linear(3) -> ReLU -> Linear(3) -> ReLU -> Linear(3) -> ReLU -> Linear(2) -> ReLU
fn build_4layer_relu_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    // Layer 1: 2 -> 3
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1, -0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    // Layer 2: 3 -> 3
    let w2 = arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4], [-0.3, 0.7, 0.1]]);
    let linear2 = LinearLayer::new(w2, Some(arr1(&[0.0, -0.05, 0.05]))).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));

    // Layer 3: 3 -> 3
    let w3 = arr2(&[[0.4_f32, 0.3, -0.5], [-0.2, 0.8, 0.1], [0.6, -0.1, 0.3]]);
    let linear3 = LinearLayer::new(w3, Some(arr1(&[-0.1, 0.0, 0.1]))).unwrap();
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(linear3),
        vec!["relu2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["linear3".to_string()],
    ));

    // Layer 4: 3 -> 2
    let w4 = arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4]]);
    let linear4 = LinearLayer::new(w4, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear4",
        Layer::Linear(linear4),
        vec!["relu3".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu4",
        Layer::ReLU(ReLULayer),
        vec!["linear4".to_string()],
    ));
    graph.set_output("relu4");

    graph
}

/// #3628: the deep-sequential auto-override should activate at the 3-activation
/// threshold and match the explicit CROWN-IBP path on the primary API output.
/// The collector API returns looser per-node bounds for the auto path because
/// fix_interm_bounds=true skips the extra collect_crown_bounds_with_alpha pass;
/// we verify soundness and monotonicity instead of exact match.
#[ntest::timeout(60000)]
#[test]
fn test_deep_sequential_override_matches_explicit_crown_ibp_at_threshold_3628() {
    let graph = build_3layer_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let auto_config = analytic_chain_config(30);
    let mut explicit_cfg = auto_config.clone();
    explicit_cfg.fix_interm_bounds = false;

    // Primary API output bounds must match between auto-override and explicit.
    let auto_bounds = graph
        .propagate_alpha_crown_with_config(&input, &auto_config)
        .expect("auto-override should succeed");
    let explicit_bounds = graph
        .propagate_alpha_crown_with_config(&input, &explicit_cfg)
        .expect("explicit CROWN-IBP should succeed");
    assert_bounded_tensor_close(&auto_bounds, &explicit_bounds, 1e-4, "#3628 output");

    // Collector: auto may be looser than explicit (no CROWN-with-alpha pass).
    let (auto_nb, _) = graph
        .collect_alpha_crown_bounds_dag(&input, &auto_config)
        .unwrap();
    let (expl_nb, _) = graph
        .collect_alpha_crown_bounds_dag(&input, &explicit_cfg)
        .unwrap();
    let auto_out = auto_nb.get("relu3").expect("auto collector missing relu3");
    let expl_out = expl_nb
        .get("relu3")
        .expect("explicit collector missing relu3");

    // Auto collector bounds: finite, lower <= upper.
    assert!(
        auto_out.lower().iter().all(|v| v.is_finite()),
        "#3628 non-finite lower"
    );
    assert!(
        auto_out.upper().iter().all(|v| v.is_finite()),
        "#3628 non-finite upper"
    );
    for i in 0..auto_out.len() {
        assert!(
            auto_out.lower()[[i]] <= auto_out.upper()[[i]] + 1e-6,
            "#3628 lower>upper"
        );
    }
    // Explicit (with extra tightening) must not be looser than auto.
    assert_bounds_do_not_loosen(expl_out, auto_out, 1e-4, "#3628 explicit vs auto collector");
}

/// #3628: Deep sequential alpha-CROWN with fix_interm_bounds=true and 4 ReLU layers.
///
/// Without the deep-sequential auto-override, IBP intermediates cause CROWN
/// relaxation error to compound through depth, collapsing output bounds to
/// IBP quality. The auto-override upgrades to CROWN-IBP intermediates when
/// the activation count meets the threshold, producing strictly tighter bounds.
#[ntest::timeout(60000)]
#[test]
fn test_deep_sequential_alpha_crown_does_not_collapse_to_ibp_3628() {
    let graph = build_4layer_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    // Run IBP as the baseline (forward interval arithmetic)
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let ibp_width = total_width(&ibp_bounds);

    // Run alpha-CROWN with fix_interm_bounds=true (default). The deep-sequential
    // auto-override (#3628) should fire because 4 activations >= threshold 3,
    // upgrading to CROWN-IBP intermediates internally.
    let config = AlphaCrownConfig {
        iterations: 30,
        gradient_method: GradientMethod::AnalyticChain,
        learning_rate: 0.15,
        lr_decay: 0.98,
        fix_interm_bounds: true,
        ..AlphaCrownConfig::default()
    };
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();
    let alpha_width = total_width(&alpha_bounds);

    // Alpha-CROWN must be strictly tighter than IBP in total width.
    // If it collapsed to IBP quality, these widths would be equal (within epsilon).
    // Note: individual elements may be wider in alpha-CROWN than IBP because
    // CROWN's linear relaxation differs from IBP's interval arithmetic.
    // The total width metric captures the overall quality improvement.
    assert!(
        alpha_width + 1e-6 < ibp_width,
        "#3628 deep sequential alpha-CROWN collapsed to IBP: \
         alpha_width={alpha_width:.6}, ibp_width={ibp_width:.6}"
    );

    // Verify finite bounds
    assert!(alpha_bounds.lower().iter().all(|v| v.is_finite()));
    assert!(alpha_bounds.upper().iter().all(|v| v.is_finite()));
}
