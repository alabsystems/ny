// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Property-based soundness tests for dual-alpha at the graph/DAG level.
//!
//! These tests verify that dual alpha (separate alpha parameters for lower
//! and upper bound paths) produces sound bounds when exercised through the
//! full `GraphNetwork` alpha-CROWN pipeline — both sequential and DAG
//! topologies.
//!
//! The layer-level dual alpha proptests (`crown_piecewise_dual_alpha.rs`)
//! verify the single-ReLU backward pass. These tests verify the full
//! graph-level integration: alpha initialization, backward pass threading,
//! gradient computation, and optimizer update across multiple layers.
//!
//! Part of #3393.

use crate::bounds::AlphaCrownConfig;
use crate::tests::proptest_soundness::{sample_points, valid_interval, FP_TOLERANCE};
use crate::*;
use ndarray::{arr1, Array1, Array2};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Manually compute forward pass through a Linear->ReLU->Linear->ReLU network.
fn sequential_forward(
    w1: &Array2<f32>,
    b1: &Array1<f32>,
    w2: &Array2<f32>,
    b2: &Array1<f32>,
    x: &Array1<f32>,
) -> Array1<f32> {
    let h1 = w1.dot(x) + b1;
    let h1_relu = h1.mapv(|v| v.max(0.0));
    let h2 = w2.dot(&h1_relu) + b2;
    h2.mapv(|v| v.max(0.0))
}

/// Manually compute forward pass through a diamond DAG:
///
/// ```text
/// Input -> Linear1 -> ReLU1 -> Linear2a \
///                            -> Linear2b -> Add -> ReLU2
/// ```
fn diamond_forward(
    w1: &Array2<f32>,
    b1: &Array1<f32>,
    w2a: &Array2<f32>,
    w2b: &Array2<f32>,
    x: &Array1<f32>,
) -> Array1<f32> {
    let h1 = w1.dot(x) + b1;
    let h1_relu = h1.mapv(|v| v.max(0.0));
    let ha = w2a.dot(&h1_relu);
    let hb = w2b.dot(&h1_relu);
    let add_out = &ha + &hb;
    add_out.mapv(|v| v.max(0.0))
}

/// Build an `AlphaCrownConfig` for proptest: enough iterations that dual alpha
/// has the opportunity to diverge from single alpha, but not so many that
/// the test is slow.
fn proptest_alpha_config() -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations: 20,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    }
}

// ---------------------------------------------------------------------------
// Deterministic test: Dual-alpha divergence and tightness (#3393 AC item 4)
// ---------------------------------------------------------------------------

/// Build a Linear(2→4)→ReLU→Linear(4→2) graph for alpha-CROWN testing.
fn graph_2_relu_2output() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let w1 = ndarray::arr2(&[[1.5_f32, -0.8], [-0.3, 1.2], [0.7, 0.5], [-1.0, 0.4]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.1, -0.2, 0.0, 0.3]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    let w2 = ndarray::arr2(&[[0.5_f32, -0.3, 0.8, -0.2], [0.2, 0.6, -0.4, 0.9]]);
    let linear2 = LinearLayer::new(w2, Some(arr1(&[-0.1, 0.05]))).unwrap();
    graph.add_node(GraphNode::new(
        "output",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("output");
    let lower =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-1.0_f32, -1.0]).unwrap();
    let upper = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0_f32, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    (graph, input)
}

/// Sum of (upper - lower) across all output dimensions.
fn total_width(bounds: &BoundedTensor) -> f32 {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(l, u)| u - l)
        .sum()
}

/// Test #3393 AC item 4: Dual-alpha optimization produces tighter intermediate
/// bounds than baseline. Both lower and upper alphas are populated and
/// updated after optimization. With SPSA gradients, both paths receive the
/// same gradient so they converge to the same values (divergence requires
/// analytic per-path gradients via `propagate_dag`, not this CROWN-IBP path).
///
/// Reference: alpha-beta-CROWN uses alpha[0] for lower bound path and alpha[1]
/// for upper bound path (auto_LiRPA/operators/relu.py:648-652).
#[ntest::timeout(10000)]
#[test]
fn test_dual_alpha_tighter_intermediate_bounds_3393() {
    let (graph, input) = graph_2_relu_2output();
    let config = AlphaCrownConfig {
        iterations: 20,
        spsa_samples: 2,
        sparse_ratio: 1.0,
        fix_interm_bounds: true,
        adaptive_skip: false,
        ..AlphaCrownConfig::default()
    };

    let (node_bounds, alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &config)
        .expect("collect_alpha_crown_bounds_dag should succeed");

    // Verify both lower and upper alphas are populated and equal (SPSA uses
    // the same gradient for both paths, so they converge).  Divergence is
    // only expected with analytic per-path gradients (#3782 Slice 1).
    for (node_name, lower_alpha) in &alpha_state.alphas {
        let upper_alpha = alpha_state
            .alphas_upper
            .get(node_name)
            .expect("upper alpha should exist for every lower alpha node");
        for (i, (&lo, &up)) in lower_alpha.iter().zip(upper_alpha.iter()).enumerate() {
            assert!(
                (lo - up).abs() < 1e-5,
                "SPSA lower/upper alphas should converge: node={node_name} \
                 neuron={i} lower={lo:.6} upper={up:.6}",
            );
        }
    }

    // Intermediate bounds at least as tight as IBP.
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    for (name, alpha_bt) in &node_bounds {
        if let Some(ibp_bt) = ibp_bounds.get(name) {
            assert!(
                total_width(alpha_bt) <= total_width(ibp_bt) + 1e-4,
                "α-CROWN intermediate '{}' should be <= IBP",
                name
            );
        }
    }

    // Output bounds at least as tight as legacy fixed-slope CROWN.
    let fixed_slope_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();
    let output_name = graph.output_name();
    let alpha_output = node_bounds.get(output_name).expect("output missing");
    assert!(
        total_width(alpha_output) <= total_width(&fixed_slope_bounds) + 1e-4,
        "α-CROWN output width should be <= fixed-slope CROWN width"
    );
}

// ---------------------------------------------------------------------------
// Proptest 1: Sequential graph alpha-CROWN soundness with dual alpha
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Graph-level alpha-CROWN on sequential networks: bounds must contain
    /// all sampled true outputs.
    ///
    /// This exercises dual alpha through the full GraphNetwork pipeline:
    /// alpha state initialization with both lower/upper arrays, backward
    /// pass grad_upper threading, and dual optimizer updates.
    ///
    /// Network: Linear(2->3) -> ReLU -> Linear(3->2) -> ReLU
    ///
    /// Part of #3393.
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_graph_alpha_crown_sequential_dual_alpha_soundness(
        w1_vec in prop::collection::vec(-2.0_f32..2.0, 6),
        b1_vec in prop::collection::vec(-1.0_f32..1.0, 3),
        w2_vec in prop::collection::vec(-2.0_f32..2.0, 6),
        b2_vec in prop::collection::vec(-1.0_f32..1.0, 2),
        (l1, u1) in valid_interval(2.0),
        (l2, u2) in valid_interval(2.0),
    ) {
        let w1 = Array2::from_shape_vec((3, 2), w1_vec).unwrap();
        let b1 = Array1::from_vec(b1_vec);
        let w2 = Array2::from_shape_vec((2, 3), w2_vec).unwrap();
        let b2 = Array1::from_vec(b2_vec);

        // Build GraphNetwork
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
            vec!["relu1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu2",
            Layer::ReLU(ReLULayer),
            vec!["linear2".to_string()],
        ));
        graph.set_output("relu2");

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        ).unwrap();

        let config = proptest_alpha_config();
        let alpha_bounds = graph
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap();

        // Bounds must be finite and non-inverted.
        for dim in 0..2 {
            prop_assert!(
                alpha_bounds.lower()[[dim]].is_finite(),
                "Alpha-CROWN lower[{dim}] is non-finite: {}",
                alpha_bounds.lower()[[dim]]
            );
            prop_assert!(
                alpha_bounds.upper()[[dim]].is_finite(),
                "Alpha-CROWN upper[{dim}] is non-finite: {}",
                alpha_bounds.upper()[[dim]]
            );
            prop_assert!(
                alpha_bounds.lower()[[dim]] <= alpha_bounds.upper()[[dim]] + 1e-5,
                "Alpha-CROWN bounds inverted at dim {dim}: lower={} > upper={}",
                alpha_bounds.lower()[[dim]],
                alpha_bounds.upper()[[dim]]
            );
        }

        // Soundness: sample points within input bounds.
        for x1 in sample_points(l1, u1, 5) {
            for x2 in sample_points(l2, u2, 5) {
                let x = arr1(&[x1, x2]);
                let y = sequential_forward(&w1, &b1, &w2, &b2, &x);

                for dim in 0..2 {
                    prop_assert!(
                        alpha_bounds.lower()[[dim]] - FP_TOLERANCE <= y[dim],
                        "Soundness: output[{dim}]={} < lower {} at ({x1}, {x2})",
                        y[dim], alpha_bounds.lower()[[dim]]
                    );
                    prop_assert!(
                        y[dim] <= alpha_bounds.upper()[[dim]] + FP_TOLERANCE,
                        "Soundness: output[{dim}]={} > upper {} at ({x1}, {x2})",
                        y[dim], alpha_bounds.upper()[[dim]]
                    );
                }
            }
        }

        // Alpha-CROWN should be at least as tight as legacy fixed-slope CROWN.
        let fixed_slope_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();
        for dim in 0..2 {
            prop_assert!(
                alpha_bounds.lower()[[dim]] >= fixed_slope_bounds.lower()[[dim]] - 1e-4,
                "Alpha-CROWN lower[{dim}] {} looser than fixed-slope CROWN lower {}",
                alpha_bounds.lower()[[dim]], fixed_slope_bounds.lower()[[dim]]
            );
            prop_assert!(
                alpha_bounds.upper()[[dim]] <= fixed_slope_bounds.upper()[[dim]] + 1e-4,
                "Alpha-CROWN upper[{dim}] {} looser than fixed-slope CROWN upper {}",
                alpha_bounds.upper()[[dim]], fixed_slope_bounds.upper()[[dim]]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Proptest 2: Diamond DAG alpha-CROWN soundness with dual alpha
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(100) })]

    /// DAG alpha-CROWN on diamond topology: bounds must contain all sampled
    /// true outputs.
    ///
    /// Diamond: Input -> Linear1 -> ReLU1 -> { Linear2a, Linear2b } -> Add -> ReLU2
    ///
    /// This exercises DAG backward pass accumulation at merge nodes (Add) with
    /// dual alpha threading through both branches. The merge node means
    /// relu1's gradient accumulates contributions from both paths, and the
    /// upper gradient must independently accumulate both contributions.
    ///
    /// Part of #3393.
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_graph_alpha_crown_diamond_dag_dual_alpha_soundness(
        w1_vec in prop::collection::vec(-2.0_f32..2.0, 4),
        b1_vec in prop::collection::vec(-1.0_f32..1.0, 2),
        w2a_vec in prop::collection::vec(-2.0_f32..2.0, 4),
        w2b_vec in prop::collection::vec(-2.0_f32..2.0, 4),
        (l1, u1) in valid_interval(2.0),
        (l2, u2) in valid_interval(2.0),
    ) {
        let w1 = Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = Array1::from_vec(b1_vec);
        let w2a = Array2::from_shape_vec((2, 2), w2a_vec).unwrap();
        let w2b = Array2::from_shape_vec((2, 2), w2b_vec).unwrap();

        // Build diamond DAG
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2a",
            Layer::Linear(LinearLayer::new(w2a.clone(), None).unwrap()),
            vec!["relu1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2b",
            Layer::Linear(LinearLayer::new(w2b.clone(), None).unwrap()),
            vec!["relu1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["linear2a".to_string(), "linear2b".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu2",
            Layer::ReLU(ReLULayer),
            vec!["add".to_string()],
        ));
        graph.set_output("relu2");

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        ).unwrap();

        let config = proptest_alpha_config();
        let alpha_bounds = graph
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap();

        // Bounds must be finite and non-inverted.
        for dim in 0..2 {
            prop_assert!(
                alpha_bounds.lower()[[dim]].is_finite(),
                "DAG alpha-CROWN lower[{dim}] is non-finite: {}",
                alpha_bounds.lower()[[dim]]
            );
            prop_assert!(
                alpha_bounds.upper()[[dim]].is_finite(),
                "DAG alpha-CROWN upper[{dim}] is non-finite: {}",
                alpha_bounds.upper()[[dim]]
            );
            prop_assert!(
                alpha_bounds.lower()[[dim]] <= alpha_bounds.upper()[[dim]] + 1e-5,
                "DAG alpha-CROWN bounds inverted at dim {dim}: lower={} > upper={}",
                alpha_bounds.lower()[[dim]],
                alpha_bounds.upper()[[dim]]
            );
        }

        // Soundness: sample points within input bounds.
        for x1 in sample_points(l1, u1, 7) {
            for x2 in sample_points(l2, u2, 7) {
                let x = arr1(&[x1, x2]);
                let y = diamond_forward(&w1, &b1, &w2a, &w2b, &x);

                for dim in 0..2 {
                    prop_assert!(
                        alpha_bounds.lower()[[dim]] - FP_TOLERANCE <= y[dim],
                        "DAG soundness: output[{dim}]={} < lower {} at ({x1}, {x2})",
                        y[dim], alpha_bounds.lower()[[dim]]
                    );
                    prop_assert!(
                        y[dim] <= alpha_bounds.upper()[[dim]] + FP_TOLERANCE,
                        "DAG soundness: output[{dim}]={} > upper {} at ({x1}, {x2})",
                        y[dim], alpha_bounds.upper()[[dim]]
                    );
                }
            }
        }

        // Alpha-CROWN should be at least as tight as IBP.
        let ibp_bounds = graph.propagate_ibp(&input).unwrap();
        for dim in 0..2 {
            prop_assert!(
                alpha_bounds.lower()[[dim]] >= ibp_bounds.lower()[[dim]] - 1e-4,
                "DAG alpha-CROWN lower[{dim}] {} looser than IBP lower {}",
                alpha_bounds.lower()[[dim]], ibp_bounds.lower()[[dim]]
            );
            prop_assert!(
                alpha_bounds.upper()[[dim]] <= ibp_bounds.upper()[[dim]] + 1e-4,
                "DAG alpha-CROWN upper[{dim}] {} looser than IBP upper {}",
                alpha_bounds.upper()[[dim]], ibp_bounds.upper()[[dim]]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Regression: DAG CROWN-IBP SPSA must update upper-path alphas (#3782 Slice 1)
// ---------------------------------------------------------------------------

/// Before the fix in #3782, `collect_alpha_crown_bounds_dag_with_engine()`
/// only updated lower-path alphas during the SPSA loop — upper-path alphas
/// stayed at their initialisation values. This test verifies that at least
/// one unstable neuron's upper alpha diverges from its initial value after
/// multiple SPSA iterations, proving the upper update path is exercised.
#[ntest::timeout(10000)]
#[test]
fn test_dag_spsa_updates_upper_alphas_3782() {
    let (graph, input) = graph_2_relu_2output();

    // Capture initial alpha values (before optimisation).
    let init_config = AlphaCrownConfig {
        iterations: 0,
        ..AlphaCrownConfig::default()
    };
    let (_, init_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &init_config)
        .expect("DAG bounds (0 iterations) should succeed");

    // Run 20 SPSA iterations to let alphas move.
    let config = AlphaCrownConfig {
        iterations: 20,
        spsa_samples: 2,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        ..AlphaCrownConfig::default()
    };
    let (_, opt_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &config)
        .expect("DAG bounds (20 iterations) should succeed");

    // Upper alphas must be populated for every ReLU node.
    assert!(
        !opt_state.alphas_upper.is_empty(),
        "alphas_upper must be non-empty after DAG alpha-CROWN"
    );
    for name in opt_state.alphas.keys() {
        assert!(
            opt_state.alphas_upper.contains_key(name),
            "alphas_upper missing key '{}' present in lower alphas",
            name
        );
    }

    // At least one unstable neuron's upper alpha must have moved from its
    // initialisation value.  Before #3782 this assertion would fail because
    // the SPSA loop never called update_*_upper().
    let mut any_upper_moved = false;
    for (name, opt_upper) in &opt_state.alphas_upper {
        if let Some(init_upper) = init_state.alphas_upper.get(name) {
            for (i, (&opt_val, &init_val)) in opt_upper.iter().zip(init_upper.iter()).enumerate() {
                if (opt_val - init_val).abs() > 1e-8 {
                    any_upper_moved = true;
                    println!(
                        "Upper alpha moved: node={}, neuron={}, init={:.6}, opt={:.6}",
                        name, i, init_val, opt_val
                    );
                }
            }
        }
    }
    assert!(
        any_upper_moved,
        "At least one upper alpha must move after SPSA optimisation (#3782)"
    );

    // Sanity: lower alphas must also have moved (pre-existing behaviour).
    let mut any_lower_moved = false;
    for (name, opt_lower) in &opt_state.alphas {
        if let Some(init_lower) = init_state.alphas.get(name) {
            if opt_lower
                .iter()
                .zip(init_lower.iter())
                .any(|(o, i)| (o - i).abs() > 1e-8)
            {
                any_lower_moved = true;
            }
        }
    }
    assert!(
        any_lower_moved,
        "Lower alphas must also move after SPSA optimisation (baseline check)"
    );
}
