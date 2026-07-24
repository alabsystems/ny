// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::beta_crown::branching::LayerRef;
use crate::layers::{BilinearCrownLayer, Layer, MulBinaryLayer};
use crate::network::{GraphNetwork, GraphNode};

use super::super::{BranchingDecision, NonlinearBranching, NonlinearBranchingConfig};

fn make_non_contiguous_bounds_2x2(values: [[f32; 2]; 2]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(
        IxDyn(&[2, 2]),
        vec![values[0][0], values[1][0], values[0][1], values[1][1]],
    )
    .expect("shape should be valid")
    .view()
    .reversed_axes()
    .to_owned()
}

#[ntest::timeout(5000)]
#[test]
fn test_bilinear_crown_is_splittable() {
    let branching = NonlinearBranching::default();
    let bilinear = Layer::BilinearCrown(BilinearCrownLayer::new(true, None));
    assert!(
        branching.is_splittable(&bilinear),
        "BilinearCrown must be splittable for BaB domain refinement at attention nodes"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_score_bilinear_input_neurons() {
    let branching = NonlinearBranching::default();
    let q_bounds = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.0, -0.5]).into_dyn(),
        arr1(&[1.0_f32, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();

    let partner_avg = 1.5_f32;
    let decisions = branching
        .score_bilinear_input_neurons("qk_matmul", 0, &q_bounds, partner_avg)
        .unwrap();

    assert_eq!(decisions.len(), 3);
    for decision in &decisions {
        assert_eq!(decision.layer, LayerRef::Name("qk_matmul".to_string()));
        assert_eq!(decision.input_index, Some(0));
        assert_eq!(decision.points.len(), 1);
    }

    let scores: Vec<f32> = decisions.iter().map(|decision| decision.score).collect();
    assert!((scores[0] - 2.0 * partner_avg).abs() < 1e-5);
    assert!((scores[1] - 0.5 * partner_avg).abs() < 1e-5);

    let decisions_neutral = branching
        .score_bilinear_input_neurons("qk_matmul", 0, &q_bounds, 1.0)
        .unwrap();
    let scores_neutral: Vec<f32> = decisions_neutral
        .iter()
        .map(|decision| decision.score)
        .collect();
    assert!((scores_neutral[0] - 2.0).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_score_bilinear_input_neurons_non_contiguous_4250() {
    let branching = NonlinearBranching::default();
    let lower_contiguous =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0_f32, 0.0, -0.5, 0.25]).unwrap();
    let upper_contiguous =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f32, 0.5, 0.5, 0.75]).unwrap();
    let lower_non_contiguous = make_non_contiguous_bounds_2x2([[-1.0_f32, 0.0], [-0.5, 0.25]]);
    let upper_non_contiguous = make_non_contiguous_bounds_2x2([[1.0_f32, 0.5], [0.5, 0.75]]);
    assert!(
        lower_non_contiguous.as_slice().is_none(),
        "test setup: lower bounds should be non-contiguous"
    );
    assert!(
        upper_non_contiguous.as_slice().is_none(),
        "test setup: upper bounds should be non-contiguous"
    );

    let contiguous = BoundedTensor::new(lower_contiguous, upper_contiguous).unwrap();
    let non_contiguous = BoundedTensor::new(lower_non_contiguous, upper_non_contiguous).unwrap();

    let expected = branching
        .score_bilinear_input_neurons("qk_matmul", 0, &contiguous, 1.5)
        .unwrap();
    let actual = branching
        .score_bilinear_input_neurons("qk_matmul", 0, &non_contiguous, 1.5)
        .unwrap();

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_eq!(actual.layer, expected.layer);
        assert_eq!(actual.neuron_idx, expected.neuron_idx);
        assert_eq!(actual.points, expected.points);
        assert!((actual.score - expected.score).abs() < 1e-6);
        assert_eq!(actual.original_bounds, expected.original_bounds);
        assert_eq!(actual.input_index, expected.input_index);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_bilinear_decision_to_splits_carries_input_index() {
    let decision = BranchingDecision {
        layer: LayerRef::Name("qk_matmul".to_string()),
        neuron_idx: 7,
        points: vec![0.25],
        score: 1.5,
        original_bounds: (-0.5, 1.0),
        input_index: Some(1),
        norm_inv_rms: None,
    };

    let splits = decision.to_splits().expect("valid decision");
    assert_eq!(splits.len(), 2);
    for split in &splits {
        assert_eq!(split.input_index(), Some(1));
    }
    assert!(splits[0].lower_bound().is_none());
    assert_eq!(splits[0].upper_bound(), Some(0.25));
    assert_eq!(splits[1].lower_bound(), Some(0.25));
    assert!(splits[1].upper_bound().is_none());
}

#[ntest::timeout(5000)]
#[test]
fn test_get_decisions_bilinear_crown_both_inputs() {
    let mut graph = GraphNetwork::new();
    let w = ndarray::arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let lin_q = crate::layers::LinearLayer::new(w.clone(), Some(arr1(&[0.0, 0.0]))).unwrap();
    let lin_k = crate::layers::LinearLayer::new(w, Some(arr1(&[0.0, 0.0]))).unwrap();
    graph.add_node(GraphNode::from_input("linear_q", Layer::Linear(lin_q)));
    graph.add_node(GraphNode::from_input("linear_k", Layer::Linear(lin_k)));
    graph.add_node(GraphNode::new(
        "qk_matmul",
        Layer::BilinearCrown(BilinearCrownLayer::new(true, None)),
        vec!["linear_q".to_string(), "linear_k".to_string()],
    ));
    graph.set_output("qk_matmul");

    let mut node_bounds = std::collections::HashMap::new();
    let bt =
        |l: &[f32], u: &[f32]| BoundedTensor::new(arr1(l).into_dyn(), arr1(u).into_dyn()).unwrap();
    node_bounds.insert("linear_q".to_string(), bt(&[-1.0, 0.0], &[1.0, 1.0]));
    node_bounds.insert("linear_k".to_string(), bt(&[-1.5, 0.25], &[1.5, 0.75]));

    let branching = NonlinearBranching::new(NonlinearBranchingConfig {
        num_candidates: 10,
        ..Default::default()
    });
    let decisions = branching
        .decisions(&graph, &node_bounds, &["qk_matmul".to_string()])
        .unwrap();

    assert_eq!(decisions.len(), 4);
    let q_n = decisions
        .iter()
        .filter(|decision| decision.input_index == Some(0))
        .count();
    let k_n = decisions
        .iter()
        .filter(|decision| decision.input_index == Some(1))
        .count();
    assert_eq!(q_n, 2);
    assert_eq!(k_n, 2);
    assert_eq!(decisions[0].input_index, Some(1));
    assert!((decisions[0].score - 4.5).abs() < 1e-5);
}

#[ntest::timeout(5000)]
#[test]
fn test_relu_only_skips_bilinear_crown() {
    let mut graph = GraphNetwork::new();
    let w = ndarray::arr2(&[[1.0_f32]]);
    let linear = crate::layers::LinearLayer::new(w, Some(arr1(&[0.0]))).unwrap();
    graph.add_node(GraphNode::from_input(
        "linear_q",
        Layer::Linear(linear.clone()),
    ));
    graph.add_node(GraphNode::from_input("linear_k", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "qk",
        Layer::BilinearCrown(BilinearCrownLayer::new(true, None)),
        vec!["linear_q".to_string(), "linear_k".to_string()],
    ));
    graph.set_output("qk");

    let mut node_bounds = std::collections::HashMap::new();
    node_bounds.insert(
        "linear_q".to_string(),
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap(),
    );
    node_bounds.insert(
        "linear_k".to_string(),
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap(),
    );

    let branching = NonlinearBranching::new(NonlinearBranchingConfig {
        relu_only: true,
        num_candidates: 10,
        ..Default::default()
    });
    let decisions = branching
        .decisions(&graph, &node_bounds, &["qk".to_string()])
        .unwrap();
    assert!(decisions.is_empty());
}

// ----- MulBinary (element-wise x·y) branching tests -----

#[ntest::timeout(5000)]
#[test]
fn test_mul_binary_is_splittable() {
    let branching = NonlinearBranching::default();
    let mul = Layer::MulBinary(MulBinaryLayer);
    assert!(
        branching.is_splittable(&mul),
        "MulBinary must be splittable so BaB can reduce the McCormick envelope gap \
         at element-wise x·y nodes (e.g. ml4acopf power flow)"
    );
}

/// A MulBinary node emits split candidates for BOTH inputs (x and y), each
/// tagged with its `input_index`, scored by the McCormick-product proxy
/// width(input) * partner_avg_width. The wider input/axis scores higher.
#[ntest::timeout(5000)]
#[test]
fn test_get_decisions_mul_binary_both_inputs() {
    let mut graph = GraphNetwork::new();
    let w = ndarray::arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let lin_x = crate::layers::LinearLayer::new(w.clone(), Some(arr1(&[0.0, 0.0]))).unwrap();
    let lin_y = crate::layers::LinearLayer::new(w, Some(arr1(&[0.0, 0.0]))).unwrap();
    graph.add_node(GraphNode::from_input("linear_x", Layer::Linear(lin_x)));
    graph.add_node(GraphNode::from_input("linear_y", Layer::Linear(lin_y)));
    graph.add_node(GraphNode::new(
        "xy_mul",
        Layer::MulBinary(MulBinaryLayer),
        vec!["linear_x".to_string(), "linear_y".to_string()],
    ));
    graph.set_output("xy_mul");

    let mut node_bounds = std::collections::HashMap::new();
    let bt =
        |l: &[f32], u: &[f32]| BoundedTensor::new(arr1(l).into_dyn(), arr1(u).into_dyn()).unwrap();
    // x: widths [2.0, 1.0], avg = 1.5 ; y: widths [3.0, 0.5], avg = 1.75
    node_bounds.insert("linear_x".to_string(), bt(&[-1.0, 0.0], &[1.0, 1.0]));
    node_bounds.insert("linear_y".to_string(), bt(&[-1.5, 0.25], &[1.5, 0.75]));

    let branching = NonlinearBranching::new(NonlinearBranchingConfig {
        num_candidates: 10,
        ..Default::default()
    });
    let decisions = branching
        .decisions(&graph, &node_bounds, &["xy_mul".to_string()])
        .unwrap();

    // Two elements per input, both inputs => 4 candidates.
    assert_eq!(decisions.len(), 4);
    let x_n = decisions
        .iter()
        .filter(|d| d.input_index == Some(0))
        .count();
    let y_n = decisions
        .iter()
        .filter(|d| d.input_index == Some(1))
        .count();
    assert_eq!(x_n, 2);
    assert_eq!(y_n, 2);
    // Every decision references the Mul node by name (so with_general_split
    // tightens the correct input via input_index).
    for d in &decisions {
        assert_eq!(d.layer, LayerRef::Name("xy_mul".to_string()));
        assert_eq!(d.points.len(), 1);
    }
    // Top candidate: x element 0 has width 2.0, partner(y) avg 1.75 => 3.5,
    // which dominates y element 0 (width 3.0, partner(x) avg 1.5 => 4.5).
    // y element 0 should win (4.5 > 3.5).
    assert_eq!(decisions[0].input_index, Some(1));
    assert!((decisions[0].score - 4.5).abs() < 1e-5);
}

/// `relu_only` suppresses MulBinary just like BilinearCrown.
#[ntest::timeout(5000)]
#[test]
fn test_relu_only_skips_mul_binary() {
    let mut graph = GraphNetwork::new();
    let w = ndarray::arr2(&[[1.0_f32]]);
    let linear = crate::layers::LinearLayer::new(w, Some(arr1(&[0.0]))).unwrap();
    graph.add_node(GraphNode::from_input(
        "linear_x",
        Layer::Linear(linear.clone()),
    ));
    graph.add_node(GraphNode::from_input("linear_y", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "xy",
        Layer::MulBinary(MulBinaryLayer),
        vec!["linear_x".to_string(), "linear_y".to_string()],
    ));
    graph.set_output("xy");

    let mut node_bounds = std::collections::HashMap::new();
    node_bounds.insert(
        "linear_x".to_string(),
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap(),
    );
    node_bounds.insert(
        "linear_y".to_string(),
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap(),
    );

    let branching = NonlinearBranching::new(NonlinearBranchingConfig {
        relu_only: true,
        num_candidates: 10,
        ..Default::default()
    });
    let decisions = branching
        .decisions(&graph, &node_bounds, &["xy".to_string()])
        .unwrap();
    assert!(decisions.is_empty());
}

/// A MulBinary decision splits the chosen input axis at its midpoint and the
/// resulting NeuronSplit carries the input_index, so `with_general_split`
/// tightens the correct operand interval.
#[ntest::timeout(5000)]
#[test]
fn test_mul_binary_decision_to_splits_carries_input_index() {
    let decision = BranchingDecision {
        layer: LayerRef::Name("xy_mul".to_string()),
        neuron_idx: 3,
        points: vec![0.0],
        score: 2.0,
        original_bounds: (-1.0, 1.0),
        input_index: Some(0),
        norm_inv_rms: None,
    };
    let splits = decision.to_splits().expect("valid decision");
    assert_eq!(splits.len(), 2);
    for split in &splits {
        assert_eq!(split.input_index(), Some(0));
    }
    assert!(splits[0].lower_bound().is_none());
    assert_eq!(splits[0].upper_bound(), Some(0.0));
    assert_eq!(splits[1].lower_bound(), Some(0.0));
    assert!(splits[1].upper_bound().is_none());
}
