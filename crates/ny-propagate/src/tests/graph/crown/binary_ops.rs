// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward propagation tests for binary operations (Add, MatMul).

use crate::*;
use ndarray::{arr1, arr2, Array2};

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_with_add() {
    // Test CROWN with Add binary operation in DAG structure
    let mut graph = GraphNetwork::new();

    // Build a DAG: input -> proj_a AND input -> proj_b -> add
    // With identity projections, output should be 3x input (proj_a + 2*proj_b = x + 2x = 3x)
    let weight_a = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let proj_a = LinearLayer::new(weight_a, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_a", Layer::Linear(proj_a)));

    let weight_b = arr2(&[[2.0_f32, 0.0], [0.0, 2.0]]);
    let proj_b = LinearLayer::new(weight_b, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_b", Layer::Linear(proj_b)));

    graph.add_node(GraphNode::binary(
        "add",
        Layer::Add(AddLayer),
        "proj_a",
        "proj_b",
    ));
    graph.set_output("add");

    // Test with concrete input
    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // For linear networks, CROWN and IBP should give same results
    for i in 0..2 {
        assert!(
            (crown_bounds.lower()[[i]] - ibp_bounds.lower()[[i]]).abs() < 1e-4,
            "CROWN lower[{}]={} != IBP lower[{}]={}",
            i,
            crown_bounds.lower()[[i]],
            i,
            ibp_bounds.lower()[[i]]
        );
        assert!(
            (crown_bounds.upper()[[i]] - ibp_bounds.upper()[[i]]).abs() < 1e-4,
            "CROWN upper[{}]={} != IBP upper[{}]={}",
            i,
            crown_bounds.upper()[[i]],
            i,
            ibp_bounds.upper()[[i]]
        );
    }

    // Expected: [1, 2] + 2*[1, 2] = [3, 6]
    assert!((crown_bounds.lower()[[0]] - 3.0).abs() < 1e-4);
    assert!((crown_bounds.lower()[[1]] - 6.0).abs() < 1e-4);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_with_add_interval() {
    // Test CROWN with Add on interval inputs
    let mut graph = GraphNetwork::new();

    let weight_a = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let proj_a = LinearLayer::new(weight_a, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_a", Layer::Linear(proj_a)));

    let weight_b = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let proj_b = LinearLayer::new(weight_b, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_b", Layer::Linear(proj_b)));

    graph.add_node(GraphNode::binary(
        "add",
        Layer::Add(AddLayer),
        "proj_a",
        "proj_b",
    ));
    graph.set_output("add");

    // Input with interval: [0, 1] for both dimensions
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // For linear DAG with Add: output = proj_a + proj_b = x + x = 2x
    // So bounds should be [0, 2] for both dimensions
    assert!((crown_bounds.lower()[[0]] - 0.0).abs() < 1e-4);
    assert!((crown_bounds.upper()[[0]] - 2.0).abs() < 1e-4);

    // CROWN should match IBP for this simple linear case
    for i in 0..2 {
        assert!((crown_bounds.lower()[[i]] - ibp_bounds.lower()[[i]]).abs() < 1e-4);
        assert!((crown_bounds.upper()[[i]] - ibp_bounds.upper()[[i]]).abs() < 1e-4);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_matmul_crown_backward() {
    // Test MatMul CROWN backward propagation directly
    let matmul = MatMulLayer::new(false, None);

    // A: 2x2, B: 2x2 -> C: 2x2
    let a_lower = arr2(&[[0.0_f32, 0.0], [0.0, 0.0]]);
    let a_upper = arr2(&[[1.0_f32, 1.0], [1.0, 1.0]]);
    let input_a = BoundedTensor::new(a_lower.into_dyn(), a_upper.into_dyn()).unwrap();

    let b_lower = arr2(&[[0.5_f32, 0.5], [0.5, 0.5]]);
    let b_upper = arr2(&[[1.0_f32, 1.0], [1.0, 1.0]]);
    let input_b = BoundedTensor::new(b_lower.into_dyn(), b_upper.into_dyn()).unwrap();

    // Create identity linear bounds for C (4 outputs = 2x2 flattened)
    let bounds = LinearBounds::identity(4);

    // Propagate backward through MatMul
    let (bounds_a, bounds_b) = matmul
        .propagate_linear_binary(&bounds, &input_a, &input_b)
        .unwrap();

    // Verify shapes are correct
    assert_eq!(bounds_a.num_outputs(), 4);
    assert_eq!(bounds_a.num_inputs(), 4); // A is 2x2 = 4 elements
    assert_eq!(bounds_b.num_outputs(), 4);
    assert_eq!(bounds_b.num_inputs(), 4); // B is 2x2 = 4 elements
}

#[ntest::timeout(10000)]
#[test]
fn test_matmul_crown_with_transpose() {
    // Test MatMul CROWN backward with transpose_b = true
    let matmul = MatMulLayer::new(true, Some(0.5)); // transpose and scale

    // Q: 2x3, K: 2x3 (transposed to get 3x2) -> C: 2x2
    let q_lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let q_upper = arr2(&[[1.0_f32, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    let input_q = BoundedTensor::new(q_lower.into_dyn(), q_upper.into_dyn()).unwrap();

    let k_lower = arr2(&[[0.5_f32, 0.5, 0.5], [0.5, 0.5, 0.5]]);
    let k_upper = arr2(&[[1.0_f32, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    let input_k = BoundedTensor::new(k_lower.into_dyn(), k_upper.into_dyn()).unwrap();

    // Create identity linear bounds for C (4 outputs = 2x2)
    let bounds = LinearBounds::identity(4);

    // Propagate backward
    let (bounds_q, bounds_k) = matmul
        .propagate_linear_binary(&bounds, &input_q, &input_k)
        .unwrap();

    // Verify shapes
    assert_eq!(bounds_q.num_outputs(), 4);
    assert_eq!(bounds_q.num_inputs(), 6); // Q is 2x3 = 6 elements
    assert_eq!(bounds_k.num_outputs(), 4);
    assert_eq!(bounds_k.num_inputs(), 6); // K is 2x3 = 6 elements
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_matmul_soundness() {
    // Soundness: GraphNetwork DAG-CROWN with MatMul should contain sampled concrete outputs.
    let mut graph = GraphNetwork::new();

    // Use GELU to produce potentially negative bounds (exercises McCormick sign handling).
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));

    // Scores = Q @ K^T / sqrt(d)
    let head_dim = 3_usize;
    let matmul = MatMulLayer::new(true, Some(1.0 / (head_dim as f32).sqrt()));
    graph.add_node(GraphNode::binary("scores", Layer::MatMul(matmul), "q", "k"));
    graph.set_output("scores");

    let input = BoundedTensor::new(
        Array2::from_elem((2, head_dim), -1.0_f32).into_dyn(),
        Array2::from_elem((2, head_dim), 1.0_f32).into_dyn(),
    )
    .unwrap();

    let bounds = graph.propagate_crown(&input).unwrap();
    let lower = bounds
        .lower()
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let upper = bounds
        .upper()
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    for sample_idx in 0..25_usize {
        let mut x = Array2::<f32>::zeros((2, head_dim));
        for i in 0..2_usize {
            for j in 0..head_dim {
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ ((i * 31 + j) as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x[[i, j]] = -1.0 + 2.0 * t;
            }
        }

        let q = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let k = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let score = q.dot(&k.t()) * (1.0 / (head_dim as f32).sqrt());

        for i in 0..2_usize {
            for j in 0..2_usize {
                let v = score[[i, j]];
                assert!(
                    v >= lower[[i, j]] - 1e-4,
                    "MatMul CROWN lower violation at ({},{}) sample {}: {} < {}",
                    i,
                    j,
                    sample_idx,
                    v,
                    lower[[i, j]]
                );
                assert!(
                    v <= upper[[i, j]] + 1e-4,
                    "MatMul CROWN upper violation at ({},{}) sample {}: {} > {}",
                    i,
                    j,
                    sample_idx,
                    v,
                    upper[[i, j]]
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_attention_full_soundness() {
    // Full attention: (Q @ K^T / sqrt(d)) -> softmax -> @ V
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "v",
        Layer::GELU(GELULayer::default()),
    ));

    let head_dim = 3_usize;
    let scores = MatMulLayer::new(true, Some(1.0 / (head_dim as f32).sqrt()));
    graph.add_node(GraphNode::binary("scores", Layer::MatMul(scores), "q", "k"));

    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(softmax),
        vec!["scores".to_string()],
    ));

    let out = MatMulLayer::new(false, None);
    graph.add_node(GraphNode::binary("out", Layer::MatMul(out), "probs", "v"));
    graph.set_output("out");

    let input = BoundedTensor::new(
        Array2::from_elem((2, head_dim), -1.0_f32).into_dyn(),
        Array2::from_elem((2, head_dim), 1.0_f32).into_dyn(),
    )
    .unwrap();

    let bounds = graph.propagate_crown(&input).unwrap();
    let lower = bounds
        .lower()
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let upper = bounds
        .upper()
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    let sm = SoftmaxLayer::new(-1);

    for sample_idx in 0..25_usize {
        let mut x = Array2::<f32>::zeros((2, head_dim));
        for i in 0..2_usize {
            for j in 0..head_dim {
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ ((i * 31 + j) as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x[[i, j]] = -1.0 + 2.0 * t;
            }
        }

        let q = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let k = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let v = x.mapv(|val| gelu_eval(val, GeluApproximation::Erf));

        let score = q.dot(&k.t()) * (1.0 / (head_dim as f32).sqrt());

        let mut probs = Array2::<f32>::zeros((2, 2));
        for i in 0..2_usize {
            probs.row_mut(i).assign(&sm.eval(&score.row(i).to_owned()));
        }

        let out = probs.dot(&v);

        for i in 0..2_usize {
            for j in 0..head_dim {
                let val = out[[i, j]];
                assert!(
                    val >= lower[[i, j]] - 1e-4,
                    "Attention CROWN lower violation at ({},{}) sample {}: {} < {}",
                    i,
                    j,
                    sample_idx,
                    val,
                    lower[[i, j]]
                );
                assert!(
                    val <= upper[[i, j]] + 1e-4,
                    "Attention CROWN upper violation at ({},{}) sample {}: {} > {}",
                    i,
                    j,
                    sample_idx,
                    val,
                    upper[[i, j]]
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_add_crown_propagation() {
    // Test Add CROWN backward propagation
    let add = AddLayer;

    // Create identity linear bounds (4 outputs, 4 inputs)
    let bounds = LinearBounds::identity(4);

    // Propagate backward through Add
    let (bounds_a, bounds_b) = add.propagate_linear_binary(&bounds).unwrap();

    // Add passes bounds unchanged to both inputs
    assert_eq!(bounds_a.num_outputs(), bounds.num_outputs());
    assert_eq!(bounds_a.num_inputs(), bounds.num_inputs());
    assert_eq!(bounds_b.num_outputs(), bounds.num_outputs());
    assert_eq!(bounds_b.num_inputs(), bounds.num_inputs());

    // Verify coefficient matrices are the same as input
    for i in 0..4 {
        for j in 0..4 {
            assert!((bounds_a.lower_a[[i, j]] - bounds.lower_a[[i, j]]).abs() < 1e-6);
            assert!((bounds_b.lower_a[[i, j]] - bounds.lower_a[[i, j]]).abs() < 1e-6);
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_add_bias_not_duplicated() {
    // ReLU creates non-zero intercept terms for crossing intervals; Add must not double-count them.
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input("a", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("b", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::binary("add", Layer::Add(AddLayer), "a", "b"));
    graph.set_output("add");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let bounds = graph.propagate_crown(&input).unwrap();

    let test_points = vec![
        arr1(&[-1.0_f32, -1.0]),
        arr1(&[-0.5_f32, 0.5]),
        arr1(&[0.0_f32, 0.0]),
        arr1(&[0.5_f32, -0.5]),
        arr1(&[1.0_f32, 1.0]),
    ];

    for p in test_points {
        let relu = p.mapv(|v| v.max(0.0));
        let out = &relu + &relu;

        for i in 0..2_usize {
            assert!(
                out[[i]] >= bounds.lower()[[i]] - 1e-5,
                "Add CROWN lower violation: point {:?} out[{}]={} < {}",
                p,
                i,
                out[[i]],
                bounds.lower()[[i]]
            );
            assert!(
                out[[i]] <= bounds.upper()[[i]] + 1e-5,
                "Add CROWN upper violation: point {:?} out[{}]={} > {}",
                p,
                i,
                out[[i]],
                bounds.upper()[[i]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_tighter_than_ibp() {
    // Create a network where CROWN should produce tighter bounds than IBP
    let mut graph = GraphNetwork::new();

    // Use weights that cause IBP to over-approximate
    let weight = arr2(&[[1.0_f32, -1.0], [1.0, 1.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    // Input perturbation
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Calculate widths
    let crown_width: f32 = (0..2)
        .map(|i| crown_bounds.upper()[[i]] - crown_bounds.lower()[[i]])
        .sum();
    let ibp_width: f32 = (0..2)
        .map(|i| ibp_bounds.upper()[[i]] - ibp_bounds.lower()[[i]])
        .sum();

    // CROWN should be tighter (smaller total width)
    assert!(
        crown_width <= ibp_width + 1e-5,
        "CROWN width {} should be <= IBP width {}",
        crown_width,
        ibp_width
    );
}
