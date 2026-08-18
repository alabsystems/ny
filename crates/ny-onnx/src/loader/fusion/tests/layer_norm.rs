// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{make_axes_attr, make_const_scalar, make_int_attr, make_node};
use crate::loader::fusion::try_fuse_layer_norm;
use crate::model::WeightStore;
use ndarray::arr1;
use ny_core::LayerType;
use ny_propagate::layers::NORMALIZATION_MIN_EPS;
use std::collections::{HashMap, HashSet};

fn pow_div_layer_norm_nodes() -> Vec<crate::onnx_proto::NodeProto> {
    let mut mean1 = make_node("ReduceMean", &["x"], &["mean1"]);
    mean1.attribute.push(make_axes_attr(&[-1]));
    let sub = make_node("Sub", &["x", "mean1"], &["centered"]);
    let two = make_const_scalar("two", 2.0);
    let pow = make_node("Pow", &["centered", "two"], &["squared"]);
    let mut mean2 = make_node("ReduceMean", &["squared"], &["mean2"]);
    mean2.attribute.push(make_axes_attr(&[-1]));
    let eps = make_const_scalar("eps", 1e-5);
    let add_eps = make_node("Add", &["mean2", "eps"], &["var_eps"]);
    let sqrt = make_node("Sqrt", &["var_eps"], &["std"]);
    let div = make_node("Div", &["centered", "std"], &["norm"]);
    let mul_gamma = make_node("Mul", &["norm", "ny"], &["scaled"]);
    let add_beta = make_node("Add", &["scaled", "beta"], &["out"]);
    vec![
        mean1, sub, two, pow, mean2, eps, add_eps, sqrt, div, mul_gamma, add_beta,
    ]
}

fn can_fuse_layer_norm(nodes: &[crate::onnx_proto::NodeProto]) -> bool {
    can_fuse_layer_norm_with_outputs(nodes, &[])
}

fn can_fuse_layer_norm_with_outputs(
    nodes: &[crate::onnx_proto::NodeProto],
    graph_outputs: &[&str],
) -> bool {
    let mut producer_by_output = HashMap::new();
    let mut consumers_by_input: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for output in &node.output {
            producer_by_output.insert(output.as_str(), idx);
        }
        for input in &node.input {
            consumers_by_input
                .entry(input.as_str())
                .or_default()
                .push(idx);
        }
    }

    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&[1.0, 1.0]).into_dyn());
    weights.insert("beta".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    let graph_output_names = graph_outputs
        .iter()
        .map(|output| (*output).to_string())
        .collect::<HashSet<_>>();

    try_fuse_layer_norm(
        nodes,
        0,
        &producer_by_output,
        &consumers_by_input,
        &weights,
        &graph_output_names,
    )
    .is_some()
}

#[test]
fn test_try_fuse_layer_norm_mul_reciprocal_path() {
    let mut mean1 = make_node("ReduceMean", &["x"], &["mean1"]);
    mean1.attribute.push(make_axes_attr(&[-1]));

    let sub = make_node("Sub", &["x", "mean1"], &["centered"]);
    let square = make_node("Mul", &["centered", "centered"], &["squared"]);

    let mut mean2 = make_node("ReduceMean", &["squared"], &["mean2"]);
    mean2.attribute.push(make_axes_attr(&[-1]));

    let eps = make_const_scalar("eps", 1e-4);
    let add_eps = make_node("Add", &["mean2", "eps"], &["var_eps"]);
    let sqrt = make_node("Sqrt", &["var_eps"], &["std"]);
    let inv = make_node("Reciprocal", &["std"], &["inv_std"]);
    let mul_norm = make_node("Mul", &["centered", "inv_std"], &["norm"]);
    let mul_gamma = make_node("Mul", &["norm", "ny"], &["scaled"]);
    let add_beta = make_node("Add", &["scaled", "beta"], &["out"]);

    let nodes = vec![
        mean1, sub, square, mean2, eps, add_eps, sqrt, inv, mul_norm, mul_gamma, add_beta,
    ];

    let mut producer_by_output = HashMap::new();
    let mut consumers_by_input: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for output in &node.output {
            producer_by_output.insert(output.as_str(), idx);
        }
        for input in &node.input {
            consumers_by_input
                .entry(input.as_str())
                .or_default()
                .push(idx);
        }
    }

    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&[1.0, 1.0]).into_dyn());
    weights.insert("beta".to_string(), arr1(&[0.0, 0.0]).into_dyn());

    let (start_idx, spec, fused_nodes) = try_fuse_layer_norm(
        &nodes,
        0,
        &producer_by_output,
        &consumers_by_input,
        &weights,
        &HashSet::new(),
    )
    .expect("Expected LayerNorm fusion for mul+reciprocal path");

    assert_eq!(start_idx, 0);
    assert_eq!(spec.layer_type, LayerType::LayerNorm);
    assert_eq!(spec.inputs[0], "x");
    assert!(spec.inputs.iter().any(|s| s == "ny"));
    assert!(spec.inputs.iter().any(|s| s == "beta"));

    let expected_indices = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10];
    assert_eq!(fused_nodes.len(), expected_indices.len());
    for idx in expected_indices {
        assert!(fused_nodes.contains(&idx), "Missing fused index {}", idx);
    }
}

#[test]
fn test_try_fuse_layer_norm_pow_div_path() {
    let mut mean1 = make_node("ReduceMean", &["x"], &["mean1"]);
    mean1.attribute.push(make_axes_attr(&[-1]));

    let sub = make_node("Sub", &["x", "mean1"], &["centered"]);
    let two = make_const_scalar("two", 2.0);
    let pow = make_node("Pow", &["centered", "two"], &["squared"]);

    let mut mean2 = make_node("ReduceMean", &["squared"], &["mean2"]);
    mean2.attribute.push(make_axes_attr(&[-1]));

    let eps = make_const_scalar("eps", 1e-5);
    let add_eps = make_node("Add", &["mean2", "eps"], &["var_eps"]);
    let sqrt = make_node("Sqrt", &["var_eps"], &["std"]);
    let div = make_node("Div", &["centered", "std"], &["norm"]);
    let mul_gamma = make_node("Mul", &["norm", "ny"], &["scaled"]);
    let add_beta = make_node("Add", &["scaled", "beta"], &["out"]);

    let nodes = vec![
        mean1, sub, two, pow, mean2, eps, add_eps, sqrt, div, mul_gamma, add_beta,
    ];

    let mut producer_by_output = HashMap::new();
    let mut consumers_by_input: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for output in &node.output {
            producer_by_output.insert(output.as_str(), idx);
        }
        for input in &node.input {
            consumers_by_input
                .entry(input.as_str())
                .or_default()
                .push(idx);
        }
    }

    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&[1.0, 1.0]).into_dyn());
    weights.insert("beta".to_string(), arr1(&[0.0, 0.0]).into_dyn());

    let (start_idx, spec, fused_nodes) = try_fuse_layer_norm(
        &nodes,
        0,
        &producer_by_output,
        &consumers_by_input,
        &weights,
        &HashSet::new(),
    )
    .expect("Expected LayerNorm fusion for pow+div path");

    assert_eq!(start_idx, 0);
    assert_eq!(spec.layer_type, LayerType::LayerNorm);
    assert_eq!(spec.inputs[0], "x");
    assert!(spec.inputs.iter().any(|s| s == "ny"));
    assert!(spec.inputs.iter().any(|s| s == "beta"));

    let expected_indices = [0, 1, 3, 4, 6, 7, 8, 9, 10];
    assert_eq!(fused_nodes.len(), expected_indices.len());
    for idx in expected_indices {
        assert!(fused_nodes.contains(&idx), "Missing fused index {}", idx);
    }
}

#[test]
fn test_try_fuse_layer_norm_with_casted_ny_beta() {
    let mut mean1 = make_node("ReduceMean", &["x"], &["mean1"]);
    mean1.attribute.push(make_axes_attr(&[-1]));

    let sub = make_node("Sub", &["x", "mean1"], &["centered"]);
    let square = make_node("Mul", &["centered", "centered"], &["squared"]);

    let mut mean2 = make_node("ReduceMean", &["squared"], &["mean2"]);
    mean2.attribute.push(make_axes_attr(&[-1]));

    let eps = make_const_scalar("eps", 1e-4);
    let add_eps = make_node("Add", &["mean2", "eps"], &["var_eps"]);
    let sqrt = make_node("Sqrt", &["var_eps"], &["std"]);
    let inv = make_node("Reciprocal", &["std"], &["inv_std"]);
    let mul_norm = make_node("Mul", &["centered", "inv_std"], &["norm"]);

    let ny_cast = make_node("Cast", &["ny"], &["ny_cast"]);
    let beta_id = make_node("Identity", &["beta"], &["beta_id"]);
    let mul_gamma = make_node("Mul", &["norm", "ny_cast"], &["scaled"]);
    let add_beta = make_node("Add", &["scaled", "beta_id"], &["out"]);

    let nodes = vec![
        mean1, sub, square, mean2, eps, add_eps, sqrt, inv, mul_norm, ny_cast, beta_id, mul_gamma,
        add_beta,
    ];

    let mut producer_by_output = HashMap::new();
    let mut consumers_by_input: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for output in &node.output {
            producer_by_output.insert(output.as_str(), idx);
        }
        for input in &node.input {
            consumers_by_input
                .entry(input.as_str())
                .or_default()
                .push(idx);
        }
    }

    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&[1.0, 1.0]).into_dyn());
    weights.insert("beta".to_string(), arr1(&[0.0, 0.0]).into_dyn());

    let (start_idx, spec, fused_nodes) = try_fuse_layer_norm(
        &nodes,
        0,
        &producer_by_output,
        &consumers_by_input,
        &weights,
        &HashSet::new(),
    )
    .expect("Expected LayerNorm fusion with casted ny/beta");

    assert_eq!(start_idx, 0);
    assert_eq!(spec.layer_type, LayerType::LayerNorm);
    assert_eq!(spec.inputs[0], "x");
    assert!(spec.inputs.iter().any(|s| s == "ny"));
    assert!(spec.inputs.iter().any(|s| s == "beta"));

    let expected_indices = [0, 1, 2, 3, 5, 6, 7, 8, 11, 12];
    assert_eq!(fused_nodes.len(), expected_indices.len());
    for idx in expected_indices {
        assert!(fused_nodes.contains(&idx), "Missing fused index {}", idx);
    }
}

#[test]
fn layer_norm_fusion_rejects_nonterminal_or_mismatched_axes() {
    for (axes1, axes2, label) in [
        (vec![1], vec![1], "matching nonterminal axes"),
        (vec![-1], vec![1], "mismatched axes"),
    ] {
        let mut nodes = pow_div_layer_norm_nodes();
        nodes[0].attribute = vec![make_axes_attr(&axes1)];
        nodes[4].attribute = vec![make_axes_attr(&axes2)];
        assert!(!can_fuse_layer_norm(&nodes), "must reject {label}");
    }
}

#[test]
fn layer_norm_fusion_requires_both_reduce_means_to_keep_dims() {
    for mean_idx in [0, 4] {
        let mut nodes = pow_div_layer_norm_nodes();
        nodes[mean_idx].attribute.push(make_int_attr("keepdims", 0));
        assert!(
            !can_fuse_layer_norm(&nodes),
            "must reject keepdims=0 on ReduceMean node {mean_idx}"
        );
    }

    let mut nodes = pow_div_layer_norm_nodes();
    nodes[0].attribute.push(make_int_attr("keepdims", 1));
    nodes[4].attribute.push(make_int_attr("keepdims", 1));
    assert!(can_fuse_layer_norm(&nodes));
}

#[test]
fn layer_norm_fusion_rejects_reversed_noncommutative_operands() {
    for (node_idx, label) in [(1, "Sub"), (3, "Pow"), (8, "Div")] {
        let mut nodes = pow_div_layer_norm_nodes();
        nodes[node_idx].input.swap(0, 1);
        assert!(
            !can_fuse_layer_norm(&nodes),
            "must reject reversed {label} operands"
        );
    }
}

#[test]
fn layer_norm_fusion_preserves_externally_consumed_intermediates() {
    for intermediate in [
        "mean1", "centered", "squared", "mean2", "var_eps", "std", "norm", "scaled",
    ] {
        let mut nodes = pow_div_layer_norm_nodes();
        nodes.push(make_node("Identity", &[intermediate], &["aux"]));
        assert!(
            !can_fuse_layer_norm(&nodes),
            "must not delete externally consumed intermediate {intermediate}"
        );
    }
}

#[test]
fn layer_norm_fusion_preserves_graph_output_intermediates() {
    for intermediate in [
        "mean1", "centered", "squared", "mean2", "var_eps", "std", "norm", "scaled",
    ] {
        let nodes = pow_div_layer_norm_nodes();
        assert!(
            !can_fuse_layer_norm_with_outputs(&nodes, &[intermediate]),
            "must not delete graph output intermediate {intermediate}"
        );
    }
}

#[test]
fn layer_norm_fusion_allows_shared_constant_producers() {
    let mut nodes = pow_div_layer_norm_nodes();
    nodes.push(make_node("Identity", &["two"], &["shared_two"]));
    nodes.push(make_node("Identity", &["eps"], &["shared_eps"]));
    assert!(can_fuse_layer_norm(&nodes));
}

#[test]
fn layer_norm_fusion_never_changes_tiny_authored_epsilon() {
    for epsilon in [
        0.0,
        f32::from_bits(NORMALIZATION_MIN_EPS.to_bits() - 1),
        f32::NAN,
        f32::INFINITY,
    ] {
        let mut nodes = pow_div_layer_norm_nodes();
        nodes[5].attribute[0]
            .t
            .as_mut()
            .expect("epsilon Constant tensor")
            .float_data = vec![epsilon];
        assert!(
            !can_fuse_layer_norm(&nodes),
            "must preserve primitive arithmetic for epsilon {epsilon:?}"
        );
    }

    let mut nodes = pow_div_layer_norm_nodes();
    nodes[5].attribute[0]
        .t
        .as_mut()
        .expect("epsilon Constant tensor")
        .float_data = vec![NORMALIZATION_MIN_EPS];
    assert!(can_fuse_layer_norm(&nodes));
}
