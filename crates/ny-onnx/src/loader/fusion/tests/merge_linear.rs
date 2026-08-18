// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{make_node, make_weight};
use crate::loader::fusion::try_fuse_merge_linear;
use crate::model::WeightStore;
use approx::assert_relative_eq;
use ndarray::{arr1, arr2};
use std::collections::{HashMap, HashSet};

fn build_consumers(nodes: &[crate::onnx_proto::NodeProto]) -> HashMap<&str, Vec<usize>> {
    let mut consumers = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for input in &node.input {
            consumers
                .entry(input.as_str())
                .or_insert_with(Vec::new)
                .push(idx);
        }
    }
    consumers
}

#[test]
fn test_merge_linear_fuses_affine_chain_with_biases() {
    let mm1 = make_node("MatMul", &["x", "w1"], &["mm1_out"]);
    let add1 = make_node("Add", &["mm1_out", "b1"], &["affine1_out"]);
    let mm2 = make_node("MatMul", &["affine1_out", "w2"], &["mm2_out"]);
    let add2 = make_node("Add", &["mm2_out", "b2"], &["out"]);
    let nodes = vec![mm1, add1, mm2, add2];
    let consumers = build_consumers(&nodes);

    let mut weights = WeightStore::new();
    weights.insert(
        "w1".to_string(),
        make_weight(&[2, 3], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
    );
    weights.insert("b1".to_string(), arr1(&[0.5, -1.0, 2.0]).into_dyn());
    weights.insert("w2".to_string(), make_weight(&[3, 1], &[2.0, -1.0, 0.5]));
    weights.insert("b2".to_string(), arr1(&[-0.25]).into_dyn());

    let (spec, consumed) =
        try_fuse_merge_linear(&nodes, 0, &consumers, &mut weights, &HashSet::new())
            .expect("affine chain should fuse");

    assert_eq!(consumed.len(), 4);
    assert!(consumed.contains(&0));
    assert!(consumed.contains(&1));
    assert!(consumed.contains(&2));
    assert!(consumed.contains(&3));
    assert_eq!(spec.layer_type, ny_core::LayerType::Linear);
    assert_eq!(spec.outputs, vec!["out".to_string()]);
    assert_eq!(spec.attributes["transB"], crate::AttributeValue::Int(1));

    let fused_weight = weights
        .get(spec.inputs[1].as_str())
        .expect("fused weight should be stored");
    let fused_bias = weights
        .get(spec.inputs[2].as_str())
        .expect("fused bias should be stored");

    let w1 = arr2(&[[1.0_f32, 5.0_f32], [4.0_f32, 3.0_f32], [2.0_f32, 6.0_f32]]);
    let b1 = arr1(&[0.5_f32, -1.0_f32, 2.0_f32]);
    let w2 = arr2(&[[2.0_f32, -1.0_f32, 0.5_f32]]);
    let b2 = arr1(&[-0.25_f32]);
    let expected_weight = w2.dot(&w1);
    let expected_bias = w2.dot(&b1) + &b2;

    assert_relative_eq!(
        fused_weight[[0, 0]],
        expected_weight[[0, 0]],
        epsilon = 1e-6
    );
    assert_relative_eq!(
        fused_weight[[0, 1]],
        expected_weight[[0, 1]],
        epsilon = 1e-6
    );
    assert_relative_eq!(fused_bias[[0]], expected_bias[[0]], epsilon = 1e-6);

    assert!(
        weights.get("w1").is_none(),
        "original first weight should be removed"
    );
    assert!(
        weights.get("w2").is_none(),
        "original second weight should be removed"
    );
    assert!(
        weights.get("b1").is_none(),
        "original first bias should be removed"
    );
    assert!(
        weights.get("b2").is_none(),
        "original second bias should be removed"
    );
}

#[test]
fn test_merge_linear_does_not_cross_relu() {
    let mm1 = make_node("MatMul", &["x", "w1"], &["mm1_out"]);
    let relu = make_node("Relu", &["mm1_out"], &["relu_out"]);
    let mm2 = make_node("MatMul", &["relu_out", "w2"], &["out"]);
    let nodes = vec![mm1, relu, mm2];
    let consumers = build_consumers(&nodes);

    let mut weights = WeightStore::new();
    weights.insert(
        "w1".to_string(),
        make_weight(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
    );
    weights.insert("w2".to_string(), make_weight(&[2, 1], &[1.0, 1.0]));

    assert!(
        try_fuse_merge_linear(&nodes, 0, &consumers, &mut weights, &HashSet::new(),).is_none(),
        "ReLU should terminate the affine chain"
    );
}

#[test]
fn test_merge_linear_does_not_cross_multi_consumer_tensor() {
    let mm1 = make_node("MatMul", &["x", "w1"], &["mm1_out"]);
    let mm2 = make_node("MatMul", &["mm1_out", "w2"], &["out_a"]);
    let relu = make_node("Relu", &["mm1_out"], &["out_b"]);
    let nodes = vec![mm1, mm2, relu];
    let consumers = build_consumers(&nodes);

    let mut weights = WeightStore::new();
    weights.insert(
        "w1".to_string(),
        make_weight(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
    );
    weights.insert("w2".to_string(), make_weight(&[2, 1], &[1.0, 1.0]));

    assert!(
        try_fuse_merge_linear(&nodes, 0, &consumers, &mut weights, &HashSet::new(),).is_none(),
        "fan-out should block merge_linear fusion"
    );
}

#[test]
fn test_merge_linear_rejects_non_bias_broadcast_add() {
    let mm1 = make_node("MatMul", &["x", "w1"], &["mm1_out"]);
    let add = make_node("Add", &["mm1_out", "bcast"], &["out"]);
    let nodes = vec![mm1, add];
    let consumers = build_consumers(&nodes);

    let mut weights = WeightStore::new();
    weights.insert(
        "w1".to_string(),
        make_weight(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
    );
    weights.insert(
        "bcast".to_string(),
        make_weight(&[2, 2], &[1.0, 1.0, 1.0, 1.0]),
    );

    assert!(
        try_fuse_merge_linear(&nodes, 0, &consumers, &mut weights, &HashSet::new(),).is_none(),
        "broadcast Add should not be mis-fused as a bias"
    );
}

#[test]
fn test_merge_linear_preserves_intermediate_and_initializer_graph_outputs() {
    let mm1 = make_node("MatMul", &["x", "w1"], &["mm1_out"]);
    let mm2 = make_node("MatMul", &["mm1_out", "w2"], &["out"]);
    let nodes = vec![mm1, mm2];
    let consumers = build_consumers(&nodes);

    for exposed in ["mm1_out", "w1"] {
        let mut weights = WeightStore::new();
        weights.insert(
            "w1".to_string(),
            make_weight(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
        );
        weights.insert("w2".to_string(), make_weight(&[2, 1], &[1.0, 1.0]));
        assert!(
            try_fuse_merge_linear(
                &nodes,
                0,
                &consumers,
                &mut weights,
                &HashSet::from([exposed.to_string()]),
            )
            .is_none(),
            "authored graph output {exposed} must be preserved"
        );
    }
}

#[test]
fn test_merge_linear_declines_generated_weight_name_collision() {
    let mm1 = make_node("MatMul", &["x", "w1"], &["mm1_out"]);
    let mm2 = make_node("MatMul", &["mm1_out", "w2"], &["out"]);
    let nodes = vec![mm1, mm2];
    let consumers = build_consumers(&nodes);
    let mut weights = WeightStore::new();
    weights.insert(
        "w1".to_string(),
        make_weight(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
    );
    weights.insert("w2".to_string(), make_weight(&[2, 1], &[1.0, 1.0]));
    weights.insert(
        "mm1_out__merge_linear_weight".to_string(),
        make_weight(&[1], &[42.0]),
    );

    assert!(try_fuse_merge_linear(&nodes, 0, &consumers, &mut weights, &HashSet::new(),).is_none());
    assert_eq!(
        weights
            .get("mm1_out__merge_linear_weight")
            .expect("colliding tensor must remain")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![42.0]
    );
}
