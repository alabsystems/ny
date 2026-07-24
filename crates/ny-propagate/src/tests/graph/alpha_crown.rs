// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork alpha-CROWN propagation tests.
use crate::layers::GatherLayer;
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_alpha_crown_soundness() {
    // Test that GraphNetwork α-CROWN produces sound bounds
    let mut graph = GraphNetwork::new();

    // Create: Linear -> ReLU -> Linear -> ReLU
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
    graph.set_output("relu2");

    // Input with perturbation
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let alpha_bounds = graph.propagate_alpha_crown(&input).unwrap();
    let crown_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();
    let _ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify soundness: sample random inputs
    for i in 0..100 {
        let t1 = (i * 7 % 100) as f32 / 100.0;
        let t2 = (i * 11 % 100) as f32 / 100.0;
        let x1 = -0.5 + t1;
        let x2 = -0.5 + t2;

        // Forward pass through network
        let z1 = [
            (1.0 * x1 - 0.5 * x2 + 0.0).max(0.0),
            (0.5 * x1 + 1.0 * x2 + 0.1).max(0.0),
            (-x1 + 0.3 * x2 - 0.1).max(0.0),
        ];
        let z2 = [
            (0.5 * z1[0] - 0.3 * z1[1] + 0.8 * z1[2]).max(0.0),
            (0.2 * z1[0] + 0.6 * z1[1] - 0.4 * z1[2]).max(0.0),
        ];

        // Check α-CROWN bounds contain the output
        for (j, &z2_val) in z2.iter().enumerate() {
            assert!(
                z2_val >= alpha_bounds.lower()[[j]] - 1e-5
                    && z2_val <= alpha_bounds.upper()[[j]] + 1e-5,
                "Output {} outside α-CROWN bounds: {} not in [{}, {}]",
                j,
                z2_val,
                alpha_bounds.lower()[[j]],
                alpha_bounds.upper()[[j]]
            );
        }
    }

    // α-CROWN should be at least as tight as CROWN
    for i in 0..2 {
        let alpha_width = alpha_bounds.upper()[[i]] - alpha_bounds.lower()[[i]];
        let crown_width = crown_bounds.upper()[[i]] - crown_bounds.lower()[[i]];
        assert!(
            alpha_width <= crown_width + 1e-4,
            "α-CROWN width {} should be <= CROWN width {} at output {}",
            alpha_width,
            crown_width,
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_alpha_crown_with_gelu() {
    // Test α-CROWN with GELU in the network
    use ndarray::arr2;

    let mut graph = GraphNetwork::new();

    // Create: Linear -> GELU -> Linear -> ReLU
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.5_f32, -0.3], [0.2, 0.6]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.set_output("relu");

    // Input with perturbation
    let input = BoundedTensor::new(
        arr1(&[-0.3_f32, -0.3]).into_dyn(),
        arr1(&[0.3_f32, 0.3]).into_dyn(),
    )
    .unwrap();

    let alpha_bounds = graph.propagate_alpha_crown(&input).unwrap();
    let crown_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();
    let _ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify soundness: sample random inputs
    for i in 0..100 {
        let t1 = (i * 7 % 100) as f32 / 100.0;
        let t2 = (i * 11 % 100) as f32 / 100.0;
        let x1 = -0.3 + 0.6 * t1;
        let x2 = -0.3 + 0.6 * t2;

        // Forward pass through network
        let z1 = [
            gelu_eval(1.0 * x1 - 0.5 * x2 + 0.0, GeluApproximation::Erf),
            gelu_eval(0.5 * x1 + 1.0 * x2 + 0.1, GeluApproximation::Erf),
        ];
        let z2 = [
            (0.5 * z1[0] - 0.3 * z1[1]).max(0.0),
            (0.2 * z1[0] + 0.6 * z1[1]).max(0.0),
        ];

        // Check bounds contain the output
        for (j, &z2_val) in z2.iter().enumerate() {
            assert!(
                z2_val >= alpha_bounds.lower()[[j]] - 1e-5
                    && z2_val <= alpha_bounds.upper()[[j]] + 1e-5,
                "Output {} outside α-CROWN+GELU bounds: {} not in [{}, {}]",
                j,
                z2_val,
                alpha_bounds.lower()[[j]],
                alpha_bounds.upper()[[j]]
            );
        }
    }

    // α-CROWN should be at least as tight as CROWN
    for i in 0..2 {
        let alpha_width = alpha_bounds.upper()[[i]] - alpha_bounds.lower()[[i]];
        let crown_width = crown_bounds.upper()[[i]] - crown_bounds.lower()[[i]];
        // α-CROWN with GELU may not be tighter than pure CROWN since α optimization is only for ReLU
        // but it should still be sound
        assert!(
            alpha_width <= crown_width + 1e-2, // Allow small tolerance for f32 rounding
            "α-CROWN+GELU width {} significantly worse than CROWN width {} at output {}",
            alpha_width,
            crown_width,
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_alpha_crown_no_relu() {
    // Test α-CROWN on network without ReLU (should fall back to CROWN)
    use ndarray::arr2;

    let mut graph = GraphNetwork::new();

    // Create: Linear only
    let w = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let alpha_bounds = graph.propagate_alpha_crown(&input).unwrap();
    let crown_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();

    // Should be identical since no ReLU to optimize
    for i in 0..2 {
        assert!(
            (alpha_bounds.lower()[[i]] - crown_bounds.lower()[[i]]).abs() < 1e-5,
            "α-CROWN lower {} != CROWN lower {} at {}",
            alpha_bounds.lower()[[i]],
            crown_bounds.lower()[[i]],
            i
        );
        assert!(
            (alpha_bounds.upper()[[i]] - crown_bounds.upper()[[i]]).abs() < 1e-5,
            "α-CROWN upper {} != CROWN upper {} at {}",
            alpha_bounds.upper()[[i]],
            crown_bounds.upper()[[i]],
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_dag_alpha_crown_with_skip_connection() {
    // Test DAG α-CROWN on a ResNet-like structure with skip connections (Add)
    use ndarray::arr2;

    let mut graph = GraphNetwork::new();

    // Create a residual block:
    //   Input -> Linear1 -> ReLU -> Linear2 --> Add --> Output
    //          \                              /
    //           \---------(skip)-------------/

    // Main path: Linear1
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    // Main path: ReLU
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    // Main path: Linear2
    let w2 = arr2(&[[0.5_f32, -0.3], [0.2, 0.6]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));

    // Skip connection: project input to match dimensions (identity-like)
    let w_skip = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let linear_skip = LinearLayer::new(w_skip, None).unwrap();
    graph.add_node(GraphNode::from_input(
        "skip_linear",
        Layer::Linear(linear_skip),
    ));

    // Add operation: combines main path with skip connection
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["linear2".to_string(), "skip_linear".to_string()],
    ));

    // Final ReLU after add
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));

    graph.set_output("relu2");

    // Input with perturbation
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    // Run α-CROWN (should use DAG α-CROWN internally for non-sequential graph)
    let alpha_bounds = graph.propagate_alpha_crown(&input).unwrap();
    let crown_bounds = graph.propagate_crown_fixed_slope(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    println!(
        "DAG α-CROWN with skip connection: α-CROWN lower={:?}, CROWN lower={:?}, IBP lower={:?}",
        alpha_bounds.lower().as_slice().unwrap(),
        crown_bounds.lower().as_slice().unwrap(),
        ibp_bounds.lower().as_slice().unwrap()
    );
    println!(
        "DAG α-CROWN with skip connection: α-CROWN upper={:?}, CROWN upper={:?}, IBP upper={:?}",
        alpha_bounds.upper().as_slice().unwrap(),
        crown_bounds.upper().as_slice().unwrap(),
        ibp_bounds.upper().as_slice().unwrap()
    );

    // Note: For complex DAGs with ReLU, CROWN may give looser bounds than IBP in some cases.
    // This is because CROWN's linear relaxation can over-approximate more than IBP's interval
    // propagation for certain network structures. The key property is that all methods give
    // sound bounds (i.e., they contain the true output).

    // Verify soundness: all bounds contain the true output
    // Since the final layer is ReLU, the true output is always in [0, max_output]
    // We verify this by sampling the network

    // Get weight matrices for manual forward pass
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let b1 = arr1(&[0.0_f32, 0.1]);
    let w2 = arr2(&[[0.5_f32, -0.3], [0.2, 0.6]]);
    let w_skip = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);

    // Sample various inputs and verify bounds are sound
    for i in 0..20 {
        let t = (i as f32) / 20.0;
        let sample_input = arr1(&[-0.5 + t, -0.5 + 0.5 * t]);

        // Compute network output: linear1 -> relu1 -> linear2 + skip -> relu2
        let h1 = w1.dot(&sample_input) + &b1;
        let h1_relu = h1.mapv(|v| v.max(0.0));
        let h2 = w2.dot(&h1_relu);
        let skip_out = w_skip.dot(&sample_input);
        let add_out = &h2 + &skip_out;
        let output = add_out.mapv(|v| v.max(0.0)); // Final ReLU

        // Verify bounds contain the output
        for j in 0..2 {
            assert!(
                output[[j]] >= alpha_bounds.lower()[[j]] - 1e-3,
                "α-CROWN lower {} > actual output {} at dim {} (unsound!)",
                alpha_bounds.lower()[[j]],
                output[[j]],
                j
            );
            assert!(
                output[[j]] <= alpha_bounds.upper()[[j]] + 1e-3,
                "α-CROWN upper {} < actual output {} at dim {} (unsound!)",
                alpha_bounds.upper()[[j]],
                output[[j]],
                j
            );
            assert!(
                output[[j]] >= crown_bounds.lower()[[j]] - 1e-3,
                "CROWN lower {} > actual output {} at dim {} (unsound!)",
                crown_bounds.lower()[[j]],
                output[[j]],
                j
            );
            assert!(
                output[[j]] <= crown_bounds.upper()[[j]] + 1e-3,
                "CROWN upper {} < actual output {} at dim {} (unsound!)",
                crown_bounds.upper()[[j]],
                output[[j]],
                j
            );
            assert!(
                output[[j]] >= ibp_bounds.lower()[[j]] - 1e-3,
                "IBP lower {} > actual output {} at dim {} (unsound!)",
                ibp_bounds.lower()[[j]],
                output[[j]],
                j
            );
            assert!(
                output[[j]] <= ibp_bounds.upper()[[j]] + 1e-3,
                "IBP upper {} < actual output {} at dim {} (unsound!)",
                ibp_bounds.upper()[[j]],
                output[[j]],
                j
            );
        }
    }

    // α-CROWN should be at least as tight as CROWN (α-optimization can only improve)
    for i in 0..2 {
        assert!(
            alpha_bounds.lower()[[i]] >= crown_bounds.lower()[[i]] - 1e-4,
            "α-CROWN lower {} < CROWN lower {} at {} (α-CROWN should be at least as tight)",
            alpha_bounds.lower()[[i]],
            crown_bounds.lower()[[i]],
            i
        );
        assert!(
            alpha_bounds.upper()[[i]] <= crown_bounds.upper()[[i]] + 1e-4,
            "α-CROWN upper {} > CROWN upper {} at {} (α-CROWN should be at least as tight)",
            alpha_bounds.upper()[[i]],
            crown_bounds.upper()[[i]],
            i
        );
    }

    // Compute bound widths
    let alpha_width: f32 = alpha_bounds
        .upper()
        .iter()
        .zip(alpha_bounds.lower().iter())
        .map(|(u, l)| u - l)
        .sum::<f32>()
        / 2.0;
    let crown_width: f32 = crown_bounds
        .upper()
        .iter()
        .zip(crown_bounds.lower().iter())
        .map(|(u, l)| u - l)
        .sum::<f32>()
        / 2.0;
    let ibp_width: f32 = ibp_bounds
        .upper()
        .iter()
        .zip(ibp_bounds.lower().iter())
        .map(|(u, l)| u - l)
        .sum::<f32>()
        / 2.0;

    println!(
        "Average widths: α-CROWN={:.4}, CROWN={:.4}, IBP={:.4}",
        alpha_width, crown_width, ibp_width
    );
    println!(
        "Tightening: α-CROWN vs IBP={:.2}x, CROWN vs IBP={:.2}x",
        ibp_width / alpha_width.max(1e-6),
        ibp_width / crown_width.max(1e-6)
    );
}

fn tiny_alpha_config() -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations: 2,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        ..AlphaCrownConfig::default()
    }
}

fn collect_dag_alpha_output_bounds(graph: &GraphNetwork, input: &BoundedTensor) -> BoundedTensor {
    let (node_bounds, _alpha_state) = graph
        .collect_alpha_crown_bounds_dag(input, &tiny_alpha_config())
        .unwrap();
    node_bounds
        .get(graph.output_name())
        .cloned()
        .expect("output node missing from alpha-CROWN bounds")
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_dag_alpha_crown_add_constant_backward_semantics() {
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "add_const",
        Layer::AddConstant(AddConstantLayer::new(arr1(&[2.0_f32]).into_dyn())),
        vec!["relu".to_string()],
    ));

    let skip_zero = LinearLayer::new(arr2(&[[0.0_f32]]), Some(arr1(&[0.0_f32]))).unwrap();
    graph.add_node(GraphNode::from_input("skip_zero", Layer::Linear(skip_zero)));

    graph.add_node(GraphNode::new(
        "out",
        Layer::Add(AddLayer),
        vec!["add_const".to_string(), "skip_zero".to_string()],
    ));
    graph.set_output("out");

    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let alpha_bounds = collect_dag_alpha_output_bounds(&graph, &input);

    assert!(
        alpha_bounds.lower()[[0]] > 1.5,
        "AddConstant should shift alpha-CROWN lower bound; got {}",
        alpha_bounds.lower()[[0]]
    );

    for i in 0..41 {
        let x = -1.0 + 2.0 * (i as f32) / 40.0;
        let y = x.max(0.0) + 2.0;
        assert!(
            y >= alpha_bounds.lower()[[0]] - 1e-4 && y <= alpha_bounds.upper()[[0]] + 1e-4,
            "sample output {} not in alpha bounds [{}, {}]",
            y,
            alpha_bounds.lower()[[0]],
            alpha_bounds.upper()[[0]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_dag_alpha_crown_sigmoid_backward_semantics() {
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["relu".to_string()],
    ));

    let skip_zero = LinearLayer::new(arr2(&[[0.0_f32]]), Some(arr1(&[0.0_f32]))).unwrap();
    graph.add_node(GraphNode::from_input("skip_zero", Layer::Linear(skip_zero)));

    graph.add_node(GraphNode::new(
        "out",
        Layer::Add(AddLayer),
        vec!["sigmoid".to_string(), "skip_zero".to_string()],
    ));
    graph.set_output("out");

    let input =
        BoundedTensor::new(arr1(&[-2.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn()).unwrap();
    let alpha_bounds = collect_dag_alpha_output_bounds(&graph, &input);

    assert!(
        alpha_bounds.upper()[[0]] <= 1.05,
        "Sigmoid output should stay near [0,1]; got upper={}",
        alpha_bounds.upper()[[0]]
    );

    for i in 0..41 {
        let x = -2.0 + 4.0 * (i as f32) / 40.0;
        let relu_x = x.max(0.0);
        let y = 1.0 / (1.0 + (-relu_x).exp());
        assert!(
            y >= alpha_bounds.lower()[[0]] - 1e-4 && y <= alpha_bounds.upper()[[0]] + 1e-4,
            "sample output {} not in alpha bounds [{}, {}]",
            y,
            alpha_bounds.lower()[[0]],
            alpha_bounds.upper()[[0]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_dag_alpha_crown_slice_backward_semantics() {
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "slice",
        Layer::Slice(SliceLayer::new(0, 0, 1)),
        vec!["relu".to_string()],
    ));

    let skip_zero = LinearLayer::new(arr2(&[[0.0_f32, 0.0_f32]]), Some(arr1(&[0.0_f32]))).unwrap();
    graph.add_node(GraphNode::from_input("skip_zero", Layer::Linear(skip_zero)));

    graph.add_node(GraphNode::new(
        "out",
        Layer::Add(AddLayer),
        vec!["slice".to_string(), "skip_zero".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -2.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 2.0_f32]).into_dyn(),
    )
    .unwrap();
    let alpha_bounds = collect_dag_alpha_output_bounds(&graph, &input);

    assert_eq!(alpha_bounds.shape(), &[1]);
    assert!(
        alpha_bounds.upper()[[0]] <= 1.1,
        "Sliced output tracks first ReLU component; got upper={}",
        alpha_bounds.upper()[[0]]
    );

    for i in 0..21 {
        for j in 0..21 {
            let x0 = -1.0 + 2.0 * (i as f32) / 20.0;
            let _x1 = -2.0 + 4.0 * (j as f32) / 20.0;
            let y = x0.max(0.0);
            assert!(
                y >= alpha_bounds.lower()[[0]] - 1e-4 && y <= alpha_bounds.upper()[[0]] + 1e-4,
                "sample output {} not in alpha bounds [{}, {}]",
                y,
                alpha_bounds.lower()[[0]],
                alpha_bounds.upper()[[0]]
            );
        }
    }
}

/// Regression test #2099: alpha-CROWN with empty-input ReLU returns InvalidSpec,
/// not an index-out-of-bounds panic.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_empty_inputs_relu_returns_invalid_spec_2099() {
    // Updated for #2991: GraphNode::new() now asserts arity at construction time
    // (#2481, #2686). Use try_new() to verify construction-time validation.
    let err = GraphNode::try_new("relu", Layer::ReLU(ReLULayer), vec![])
        .expect_err("empty-input ReLU should return InvalidSpec at construction");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("relu") && msg.contains("1 input"),
        "expected node name and arity diagnostic, got: {msg}"
    );
}

/// Regression test #2113: alpha-CROWN best-bounds merge returns InternalError
/// on element-count mismatch instead of panicking via assert_eq!.
///
/// The shape-mismatch path is an InternalError guard that fires when CROWN
/// iteration produces bounds with a different element count than the initial
/// CROWN bounds. In normal operation this doesn't trigger, so this test
/// verifies that a working graph completes without panic (the old assert_eq!
/// would have panicked on any future mismatch regression).
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_best_bounds_merge_no_panic_2113() {
    // Build a sequential graph: Linear -> ReLU -> Linear -> ReLU
    // This exercises the best-bounds merge loop in propagate_sequential.rs
    // where assert_eq! was replaced with Result-based error handling (#2113).
    let mut graph = GraphNetwork::new();

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
    graph.set_output("relu2");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    // With the old assert_eq!, any shape mismatch would panic.
    // With the new Result-based check, mismatches return Err(InternalError).
    // A well-formed graph should succeed without panic or error.
    let result = graph.propagate_alpha_crown(&input);
    assert!(
        result.is_ok(),
        "alpha-CROWN should succeed on well-formed graph, got: {:?}",
        result.err()
    );
}

/// Build a sequential graph with a Gather layer that triggers UnsupportedOp.
fn graph_with_unsupported_gather() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let w1 = arr2(&[[1.0f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    let b1 = arr1(&[1.0f32, 2.0, 3.0]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2]).unwrap();
    graph.add_node(GraphNode::new(
        "gather1",
        Layer::Gather(GatherLayer::new(0, Some(indices), vec![])),
        vec!["relu1".to_string()],
    ));
    graph.set_output("gather1");
    let input = BoundedTensor::new(
        arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
        arr1(&[4.0f32, 5.0, 6.0]).into_dyn(),
    )
    .unwrap();
    (graph, input)
}

/// Assert bounds contain expected corners and are at least as tight as IBP.
fn assert_graph_gather_sound(bounds: &BoundedTensor, ibp: &BoundedTensor) {
    // Corners: x=[1,2,3]→[2,12], x=[4,5,6]→[5,21]
    let flat = bounds.flatten();
    assert!(flat.lower()[[0]] <= 2.0, "lower[0]={}", flat.lower()[[0]]);
    assert!(flat.upper()[[0]] >= 5.0, "upper[0]={}", flat.upper()[[0]]);
    assert!(flat.lower()[[1]] <= 12.0, "lower[1]={}", flat.lower()[[1]]);
    assert!(flat.upper()[[1]] >= 21.0, "upper[1]={}", flat.upper()[[1]]);
    let ibp_flat = ibp.flatten();
    for i in 0..2 {
        assert!(
            flat.lower()[[i]] >= ibp_flat.lower()[[i]] - 1e-5,
            "lower[{i}]: alpha={} < ibp={}",
            flat.lower()[[i]],
            ibp_flat.lower()[[i]]
        );
        assert!(
            flat.upper()[[i]] <= ibp_flat.upper()[[i]] + 1e-5,
            "upper[{i}]: alpha={} > ibp={}",
            flat.upper()[[i]],
            ibp_flat.upper()[[i]]
        );
    }
}

/// Regression test for GraphNetwork sequential alpha-CROWN UnsupportedOp fallback
/// at `propagate_sequential.rs:350`. Gather triggers UnsupportedOp; alpha-CROWN
/// must fall back to CROWN and produce sound bounds at least as tight as IBP.
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_sequential_alpha_crown_unsupported_op_fallback() {
    let (graph, input) = graph_with_unsupported_gather();
    let alpha_bounds = graph
        .propagate_alpha_crown(&input)
        .expect("Graph sequential alpha-CROWN should fall back to CROWN, not error");
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    assert_graph_gather_sound(&alpha_bounds, &ibp_bounds);
}

/// Build a Linear(2→4)→ReLU→Linear(4→2) graph for alpha-CROWN testing.
fn graph_2_relu_2output() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let w1 = arr2(&[[1.5_f32, -0.8], [-0.3, 1.2], [0.7, 0.5], [-1.0, 0.4]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.1, -0.2, 0.0, 0.3]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    let w2 = arr2(&[[0.5_f32, -0.3, 0.8, -0.2], [0.2, 0.6, -0.4, 0.9]]);
    let linear2 = LinearLayer::new(w2, Some(arr1(&[-0.1, 0.05]))).unwrap();
    graph.add_node(GraphNode::new(
        "output",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("output");
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    (graph, input)
}

/// Assert bounds are valid (finite, non-inverted) and at least as tight as IBP.
fn assert_output_at_least_ibp(
    output_bounds: &BoundedTensor,
    ibp_output: &BoundedTensor,
    expected_shape: &[usize],
) {
    assert_eq!(output_bounds.shape(), expected_shape);
    for (i, (l, u)) in output_bounds
        .lower()
        .iter()
        .zip(output_bounds.upper().iter())
        .enumerate()
    {
        assert!(l.is_finite() && u.is_finite(), "output[{i}]: non-finite");
        assert!(l <= u, "output[{i}]: inverted lower={l} > upper={u}");
    }
    for (i, ((al, au), (il, iu))) in output_bounds
        .lower()
        .iter()
        .zip(output_bounds.upper().iter())
        .zip(ibp_output.lower().iter().zip(ibp_output.upper().iter()))
        .enumerate()
    {
        assert!(*al >= il - 1e-4, "output[{i}]: alpha lower {al} < IBP {il}");
        assert!(*au <= iu + 1e-4, "output[{i}]: alpha upper {au} > IBP {iu}");
    }
}

/// Regression test for #2251: collect_alpha_crown_bounds_dag element-wise best
/// output bound tracking. Before the fix, the optimization loop tracked only a
/// scalar sum comparison which could discard per-dimension tightness.
#[ntest::timeout(10000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_elementwise_best_2251() {
    use crate::bounds::AlphaCrownConfig;

    let (graph, input) = graph_2_relu_2output();
    let config = AlphaCrownConfig {
        iterations: 10,
        spsa_samples: 2,
        sparse_ratio: 1.0,
        fix_interm_bounds: true,
        adaptive_skip: false,
        ..AlphaCrownConfig::default()
    };

    let (node_bounds, _alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &config)
        .expect("collect_alpha_crown_bounds_dag should succeed");

    let output_bounds = node_bounds
        .get("output")
        .expect("output node missing from alpha-CROWN bounds map");
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let ibp_output = ibp_bounds.get("output").expect("IBP output missing");
    assert_output_at_least_ibp(output_bounds, ibp_output, &[2]);
}

/// Build a 2-layer sequential ReLU network for testing.
/// Layout: Linear(2→3) → ReLU → Linear(3→2) → ReLU
fn build_sequential_relu_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(arr1(&[0.0, 0.1, -0.1]))).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    let w2 = arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4]]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, None).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.set_output("relu2");
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();
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

/// Check that CROWN-IBP intermediate bounds are tighter than plain IBP
/// for at least one node.
fn assert_crown_ibp_tighter_than_ibp(graph: &GraphNetwork, input: &BoundedTensor) {
    let ibp = graph.collect_node_bounds(input).unwrap();
    let cibp = graph.collect_crown_ibp_bounds_dag(input).unwrap();
    let any_tighter = ibp.iter().any(|(name, ib)| {
        cibp.get(name)
            .is_some_and(|cb| total_width(cb) < total_width(ib) - 1e-6)
    });
    assert!(
        any_tighter,
        "CROWN-IBP should be tighter than IBP for at least one node"
    );
}

/// Verify the sequential_relu_graph's sampled outputs fall within bounds.
fn assert_sequential_relu_soundness(bounds: &BoundedTensor) {
    for i in 0..100 {
        let (t1, t2) = ((i * 7 % 100) as f32 / 100.0, (i * 11 % 100) as f32 / 100.0);
        let (x1, x2) = (-0.5 + t1, -0.5 + t2);
        let z1 = [
            (x1 - 0.5 * x2).max(0.0),
            (0.5 * x1 + x2 + 0.1).max(0.0),
            (-x1 + 0.3 * x2 - 0.1).max(0.0),
        ];
        let out = [
            (0.5 * z1[0] - 0.3 * z1[1] + 0.8 * z1[2]).max(0.0),
            (0.2 * z1[0] + 0.6 * z1[1] - 0.4 * z1[2]).max(0.0),
        ];
        for (j, &v) in out.iter().enumerate() {
            assert!(
                v >= bounds.lower()[[j]] - 1e-5 && v <= bounds.upper()[[j]] + 1e-5,
                "Output {j} = {v} outside [{}, {}]",
                bounds.lower()[[j]],
                bounds.upper()[[j]]
            );
        }
    }
}

/// Regression test for #2477: sequential alpha-CROWN uses CROWN-IBP.
///
/// Strengthened for #2512: computes CROWN output bounds using IBP-only vs CROWN-IBP
/// intermediates, then verifies α-CROWN is at least as tight as CROWN-with-CROWN-IBP.
/// If propagate_sequential.rs:105 were reverted to `collect_node_bounds`, α-CROWN
/// would match the looser IBP baseline and this assertion would fail.
#[ntest::timeout(10000)]
#[test]
fn test_sequential_alpha_crown_uses_crown_ibp_2477() {
    let (graph, input) = build_sequential_relu_graph();
    assert_crown_ibp_tighter_than_ibp(&graph, &input);

    // Compare CROWN output using IBP vs CROWN-IBP intermediate bounds
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let crown_ibp_bounds = graph.collect_crown_ibp_bounds_dag(&input).unwrap();
    let spec = ndarray::Array2::<f32>::eye(2);
    let crown_ibp_only = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(&input, &spec, None, &ibp_bounds)
        .unwrap();
    let crown_cibp = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(
            &input,
            &spec,
            None,
            &crown_ibp_bounds,
        )
        .unwrap();
    let w_ibp = total_width(&crown_ibp_only);
    let w_cibp = total_width(&crown_cibp);
    assert!(
        w_cibp < w_ibp - 1e-6,
        "CROWN+CROWN-IBP width ({w_cibp:.6}) should be < CROWN+IBP width ({w_ibp:.6})"
    );

    // α-CROWN (which internally uses CROWN-IBP) must be at least as tight
    let seq_bounds = graph.propagate_alpha_crown(&input).unwrap();
    let w_alpha = total_width(&seq_bounds);
    assert!(
        w_alpha <= w_cibp + 1e-4,
        "α-CROWN width ({w_alpha:.6}) should be <= CROWN+CROWN-IBP ({w_cibp:.6})"
    );

    assert_sequential_relu_soundness(&seq_bounds);
}
