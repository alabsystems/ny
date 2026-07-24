// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{make_float_attr, make_int_attr, make_node, make_weight};
use crate::loader::fusion::{
    fold_batch_norm_into_conv_linear, fold_batch_norm_into_conv_linear_with_context,
    gemm_has_exact_default_affine,
};
use crate::model::WeightStore;
use approx::assert_relative_eq;
use ndarray::arr1;
use std::collections::{HashMap, HashSet};

fn direct_gemm_bn_guard_fixture() -> (Vec<crate::onnx_proto::NodeProto>, WeightStore) {
    let mut gemm = make_node("Gemm", &["x", "gemm_w", "gemm_b"], &["gemm_y"]);
    gemm.attribute.push(make_int_attr("transB", 1));
    let mut bn = make_node(
        "BatchNormalization",
        &["gemm_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let nodes = vec![gemm, bn];
    let mut weights = WeightStore::new();
    weights.insert("gemm_w".to_string(), make_weight(&[2, 1], &[2.0, -3.0]));
    weights.insert("gemm_b".to_string(), arr1(&[0.25, -0.5]).into_dyn());
    weights.insert("bn_scale".to_string(), arr1(&[2.0, 0.5]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, -1.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[1.0, 1.0]).into_dyn());
    (nodes, weights)
}

fn gemm_reshape_bn_guard_fixture() -> (Vec<crate::onnx_proto::NodeProto>, WeightStore) {
    let mut gemm = make_node("Gemm", &["x", "gemm_w", "gemm_b"], &["gemm_y"]);
    gemm.attribute.push(make_int_attr("transB", 1));
    let reshape = make_node("Reshape", &["gemm_y", "target_shape"], &["reshape_y"]);
    let mut bn = make_node(
        "BatchNormalization",
        &["reshape_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let nodes = vec![gemm, reshape, bn];
    let mut weights = WeightStore::new();
    weights.insert(
        "gemm_w".to_string(),
        make_weight(&[4, 1], &[1.0, 2.0, 3.0, 4.0]),
    );
    weights.insert(
        "gemm_b".to_string(),
        arr1(&[0.25, -0.5, 0.75, -1.0]).into_dyn(),
    );
    weights.insert_integers("target_shape".to_string(), arr1(&[-1_i64, 2, 2]).into_dyn());
    weights.insert("bn_scale".to_string(), arr1(&[2.0, 0.5]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, -1.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[1.0, 1.0]).into_dyn());
    (nodes, weights)
}

fn fold_with_graph_outputs(
    nodes: &mut [crate::onnx_proto::NodeProto],
    weights: &mut WeightStore,
    graph_outputs: &[&str],
) -> HashSet<usize> {
    let graph_output_names = graph_outputs
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    fold_batch_norm_into_conv_linear_with_context(
        nodes,
        weights,
        &HashMap::new(),
        &graph_output_names,
    )
}

#[test]
fn test_fold_batch_norm_into_conv_adds_bias_and_rewires_output() {
    let conv = make_node("Conv", &["x", "conv_w"], &["conv_y"]);
    let mut bn = make_node(
        "BatchNormalization",
        &["conv_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let relu = make_node("Relu", &["bn_y"], &["relu_y"]);
    let mut nodes = vec![conv, bn, relu];

    let mut weights = WeightStore::new();
    weights.insert(
        "conv_w".to_string(),
        make_weight(&[2, 1, 1, 1], &[2.0, -3.0]),
    );
    weights.insert("bn_scale".to_string(), arr1(&[4.0, -2.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, 0.5]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.5, -1.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[4.0, 9.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    assert_eq!(consumed.len(), 1);
    assert!(consumed.contains(&1));

    assert_eq!(nodes[0].output[0], "bn_y");
    assert_eq!(nodes[1].output.len(), 0);

    let conv_bias_name = nodes[0]
        .input
        .get(2)
        .expect("fusion should add synthetic conv bias")
        .clone();
    let fused_weight = weights.get("conv_w").expect("fused conv weight");
    assert_relative_eq!(fused_weight[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    assert_relative_eq!(fused_weight[[1, 0, 0, 0]], 2.0, epsilon = 1e-6);

    let fused_bias = weights
        .get(conv_bias_name.as_str())
        .expect("fused synthetic conv bias");
    assert_relative_eq!(fused_bias[[0]], 0.0, epsilon = 1e-6);
    assert_relative_eq!(fused_bias[[1]], -1.0 / 6.0, epsilon = 1e-6);
}

#[test]
fn test_fold_batch_norm_into_gemm_updates_existing_bias() {
    let mut gemm = make_node("Gemm", &["x", "gemm_w", "gemm_b"], &["gemm_y"]);
    gemm.attribute.push(make_int_attr("transB", 1));
    let mut bn = make_node(
        "BatchNormalization",
        &["gemm_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let identity = make_node("Identity", &["bn_y"], &["out"]);
    let mut nodes = vec![gemm, bn, identity];

    let mut weights = WeightStore::new();
    weights.insert(
        "gemm_w".to_string(),
        make_weight(&[2, 3], &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),
    );
    weights.insert("gemm_b".to_string(), arr1(&[0.2, -0.4]).into_dyn());
    weights.insert("bn_scale".to_string(), arr1(&[3.0, -2.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[0.5, 0.25]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[1.0, -2.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[9.0, 4.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    assert_eq!(consumed.len(), 1);
    assert!(consumed.contains(&1));
    assert_eq!(nodes[0].output[0], "bn_y");
    assert_eq!(nodes[1].output.len(), 0);
    assert_eq!(nodes[0].input[2], "gemm_b");

    let fused_weight = weights.get("gemm_w").expect("fused gemm weight");
    assert_relative_eq!(fused_weight[[0, 0]], 1.0, epsilon = 1e-6);
    assert_relative_eq!(fused_weight[[0, 1]], 2.0, epsilon = 1e-6);
    assert_relative_eq!(fused_weight[[0, 2]], 3.0, epsilon = 1e-6);
    assert_relative_eq!(fused_weight[[1, 0]], 1.0, epsilon = 1e-6);
    assert_relative_eq!(fused_weight[[1, 1]], 2.0, epsilon = 1e-6);
    assert_relative_eq!(fused_weight[[1, 2]], 3.0, epsilon = 1e-6);

    let fused_bias = weights.get("gemm_b").expect("fused gemm bias");
    assert_relative_eq!(fused_bias[[0]], -0.3, epsilon = 1e-6);
    assert_relative_eq!(fused_bias[[1]], -1.35, epsilon = 1e-6);
}

#[test]
fn gemm_bn_folds_require_exact_default_alpha_and_beta() {
    let base = make_node("Gemm", &["x", "w", "c"], &["y"]);
    assert!(gemm_has_exact_default_affine(&base));
    let next_up = f32::from_bits(1.0_f32.to_bits() + 1);
    let next_down = f32::from_bits(1.0_f32.to_bits() - 1);
    for attribute in ["alpha", "beta"] {
        for value in [next_up, next_down, f32::NAN] {
            let mut gemm = base.clone();
            gemm.attribute.push(make_float_attr(attribute, value));
            assert!(
                !gemm_has_exact_default_affine(&gemm),
                "{attribute}={value:?} must fail closed"
            );
        }
        let mut gemm = base.clone();
        gemm.attribute.push(make_float_attr(attribute, 1.0));
        assert!(gemm_has_exact_default_affine(&gemm));
    }

    // Direct Gemm -> BN calls the shared guard before any mutation.
    let (mut nodes, mut weights) = direct_gemm_bn_guard_fixture();
    nodes[0].attribute.push(make_float_attr("alpha", next_up));
    let nodes_before = nodes.clone();
    let weight_before = weights.get("gemm_w").expect("weight").clone();
    let bias_before = weights.get("gemm_b").expect("bias").clone();
    assert!(fold_batch_norm_into_conv_linear(&mut nodes, &mut weights).is_empty());
    assert_eq!(nodes, nodes_before);
    assert_eq!(weights.get("gemm_w"), Some(&weight_before));
    assert_eq!(weights.get("gemm_b"), Some(&bias_before));

    // Gemm -> Reshape -> BN calls the same guard before any mutation.
    let (mut nodes, mut weights) = gemm_reshape_bn_guard_fixture();
    nodes[0].attribute.push(make_float_attr("beta", next_down));
    let nodes_before = nodes.clone();
    let weight_before = weights.get("gemm_w").expect("weight").clone();
    let bias_before = weights.get("gemm_b").expect("bias").clone();
    assert!(fold_batch_norm_into_conv_linear(&mut nodes, &mut weights).is_empty());
    assert_eq!(nodes, nodes_before);
    assert_eq!(weights.get("gemm_w"), Some(&weight_before));
    assert_eq!(weights.get("gemm_b"), Some(&bias_before));
}

#[test]
fn direct_and_across_reshape_bn_folds_preserve_observable_values() {
    for exposed in ["gemm_y", "gemm_w", "gemm_b"] {
        let (mut nodes, mut weights) = direct_gemm_bn_guard_fixture();
        let nodes_before = nodes.clone();
        let weight_before = weights.get("gemm_w").expect("weight").clone();
        let bias_before = weights.get("gemm_b").expect("bias").clone();
        assert!(fold_with_graph_outputs(&mut nodes, &mut weights, &["bn_y", exposed],).is_empty());
        assert_eq!(nodes, nodes_before);
        assert_eq!(weights.get("gemm_w"), Some(&weight_before));
        assert_eq!(weights.get("gemm_b"), Some(&bias_before));
    }

    for exposed in ["gemm_y", "reshape_y", "gemm_w", "gemm_b"] {
        let (mut nodes, mut weights) = gemm_reshape_bn_guard_fixture();
        let nodes_before = nodes.clone();
        let weight_before = weights.get("gemm_w").expect("weight").clone();
        let bias_before = weights.get("gemm_b").expect("bias").clone();
        assert!(fold_with_graph_outputs(&mut nodes, &mut weights, &["bn_y", exposed],).is_empty());
        assert_eq!(nodes, nodes_before);
        assert_eq!(weights.get("gemm_w"), Some(&weight_before));
        assert_eq!(weights.get("gemm_b"), Some(&bias_before));
    }
}

/// Square weight (2x2) with transB=1: BN scales along axis 0 (rows).
/// Without #2309 fix, this accidentally passed because axis 0 was the default.
#[test]
fn test_fold_batch_norm_into_gemm_square_weight_trans_b() {
    let mut gemm = make_node("Gemm", &["x", "gemm_w", "gemm_b"], &["gemm_y"]);
    gemm.attribute.push(make_int_attr("transB", 1));
    let mut bn = make_node(
        "BatchNormalization",
        &["gemm_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let mut nodes = vec![gemm, bn];

    let mut weights = WeightStore::new();
    weights.insert(
        "gemm_w".to_string(),
        make_weight(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
    );
    weights.insert("gemm_b".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_scale".to_string(), arr1(&[2.0, 0.5]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[1.0, 1.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    assert_eq!(consumed.len(), 1);

    let fused = weights.get("gemm_w").expect("fused gemm weight");
    // transB=1: weight (out, in), scale axis 0. Row 0 x 2.0, Row 1 x 0.5.
    assert_relative_eq!(fused[[0, 0]], 2.0, epsilon = 1e-6);
    assert_relative_eq!(fused[[0, 1]], 4.0, epsilon = 1e-6);
    assert_relative_eq!(fused[[1, 0]], 1.5, epsilon = 1e-6);
    assert_relative_eq!(fused[[1, 1]], 2.0, epsilon = 1e-6);
}

/// Square weight (2x2) with transB=0: BN scales along axis 1 (columns).
/// This is the case that #2309 fixed -- without the fix, axis 0 was used
/// regardless of transB, producing wrong fused weights for square matrices.
#[test]
fn test_fold_batch_norm_into_gemm_square_weight_no_trans_b() {
    let gemm = make_node("Gemm", &["x", "gemm_w", "gemm_b"], &["gemm_y"]);
    // No transB attribute -> default transB=0 -> weight is (in, out).
    let mut bn = make_node(
        "BatchNormalization",
        &["gemm_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let mut nodes = vec![gemm, bn];

    let mut weights = WeightStore::new();
    weights.insert(
        "gemm_w".to_string(),
        make_weight(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
    );
    weights.insert("gemm_b".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_scale".to_string(), arr1(&[2.0, 0.5]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[1.0, 1.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    assert_eq!(consumed.len(), 1);

    let fused = weights.get("gemm_w").expect("fused gemm weight");
    // transB=0: weight (in, out), scale axis 1. Col 0 x 2.0, Col 1 x 0.5.
    assert_relative_eq!(fused[[0, 0]], 2.0, epsilon = 1e-6);
    assert_relative_eq!(fused[[0, 1]], 1.0, epsilon = 1e-6);
    assert_relative_eq!(fused[[1, 0]], 6.0, epsilon = 1e-6);
    assert_relative_eq!(fused[[1, 1]], 2.0, epsilon = 1e-6);
}

/// Gemm with transA=1: fusion must be skipped because the BN fold equations
/// assume standard (non-transposed) input layout. When transA=1, the matmul
/// semantics change and the BN node must be retained for correct inference.
/// Regression test for #2320.
#[test]
fn test_fold_batch_norm_skips_gemm_trans_a() {
    let mut gemm = make_node("Gemm", &["x", "gemm_w", "gemm_b"], &["gemm_y"]);
    gemm.attribute.push(make_int_attr("transA", 1));
    gemm.attribute.push(make_int_attr("transB", 1));
    let mut bn = make_node(
        "BatchNormalization",
        &["gemm_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let mut nodes = vec![gemm, bn];

    let mut weights = WeightStore::new();
    weights.insert(
        "gemm_w".to_string(),
        make_weight(&[2, 3], &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),
    );
    weights.insert("gemm_b".to_string(), arr1(&[0.2, -0.4]).into_dyn());
    weights.insert("bn_scale".to_string(), arr1(&[3.0, -2.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[0.5, 0.25]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[1.0, -2.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[9.0, 4.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    // transA=1 -> fusion skipped -> BN node NOT consumed.
    assert!(
        consumed.is_empty(),
        "Expected fusion to be skipped for transA=1"
    );
    // BN node outputs should remain intact (not cleared).
    assert!(
        !nodes[1].output.is_empty(),
        "BN node should retain its outputs"
    );
    assert_eq!(nodes[1].output[0], "bn_y");
    // Original Gemm weight should be unchanged.
    let weight = weights.get("gemm_w").expect("original weight");
    assert_relative_eq!(weight[[0, 0]], 1.0, epsilon = 1e-6);
}

/// Gemm with alpha=2.0: fusion must be skipped because the BN fold equations
/// assume alpha=1.0 (standard Y = A * B + C). When alpha!=1.0, the matmul
/// output is scaled before BN sees it, producing incorrect fused weights.
/// Regression test for #2319.
#[test]
fn test_fold_batch_norm_skips_gemm_non_default_alpha() {
    let mut gemm = make_node("Gemm", &["x", "gemm_w", "gemm_b"], &["gemm_y"]);
    gemm.attribute.push(make_int_attr("transB", 1));
    gemm.attribute.push(make_float_attr("alpha", 2.0));
    let mut bn = make_node(
        "BatchNormalization",
        &["gemm_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let mut nodes = vec![gemm, bn];

    let mut weights = WeightStore::new();
    weights.insert(
        "gemm_w".to_string(),
        make_weight(&[2, 3], &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),
    );
    weights.insert("gemm_b".to_string(), arr1(&[0.2, -0.4]).into_dyn());
    weights.insert("bn_scale".to_string(), arr1(&[3.0, -2.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[0.5, 0.25]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[1.0, -2.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[9.0, 4.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    // alpha=2.0 -> fusion skipped -> BN node NOT consumed.
    assert!(
        consumed.is_empty(),
        "Expected fusion to be skipped for alpha=2.0"
    );
    assert!(
        !nodes[1].output.is_empty(),
        "BN node should retain its outputs"
    );
}

/// Gemm with beta=0.5: fusion must be skipped because the BN fold equations
/// assume beta=1.0. When beta!=1.0, the bias term C is pre-scaled, so
/// fuse_bias would produce incorrect results.
/// Regression test for #2319.
#[test]
fn test_fold_batch_norm_skips_gemm_non_default_beta() {
    let mut gemm = make_node("Gemm", &["x", "gemm_w", "gemm_b"], &["gemm_y"]);
    gemm.attribute.push(make_int_attr("transB", 1));
    gemm.attribute.push(make_float_attr("beta", 0.5));
    let mut bn = make_node(
        "BatchNormalization",
        &["gemm_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let mut nodes = vec![gemm, bn];

    let mut weights = WeightStore::new();
    weights.insert(
        "gemm_w".to_string(),
        make_weight(&[2, 3], &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),
    );
    weights.insert("gemm_b".to_string(), arr1(&[0.2, -0.4]).into_dyn());
    weights.insert("bn_scale".to_string(), arr1(&[3.0, -2.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[0.5, 0.25]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[1.0, -2.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[9.0, 4.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    // beta=0.5 -> fusion skipped -> BN node NOT consumed.
    assert!(
        consumed.is_empty(),
        "Expected fusion to be skipped for beta=0.5"
    );
    assert!(
        !nodes[1].output.is_empty(),
        "BN node should retain its outputs"
    );
}

/// Two Conv nodes share one weight initializer and only one is followed by
/// BatchNormalization: fusion must be skipped, because scaling the shared
/// name-keyed WeightStore entry in place would silently corrupt the branch
/// that has no BN.
#[test]
fn test_fold_batch_norm_skips_shared_conv_weight() {
    let conv_a = make_node("Conv", &["x", "shared_w"], &["a_y"]);
    let mut bn = make_node(
        "BatchNormalization",
        &["a_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let conv_b = make_node("Conv", &["x2", "shared_w"], &["b_y"]);
    let mut nodes = vec![conv_a, bn, conv_b];

    let mut weights = WeightStore::new();
    weights.insert(
        "shared_w".to_string(),
        make_weight(&[2, 1, 1, 1], &[2.0, -3.0]),
    );
    weights.insert("bn_scale".to_string(), arr1(&[4.0, -2.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, 0.5]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.5, -1.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[4.0, 9.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    assert!(
        consumed.is_empty(),
        "fusion must be skipped when the weight initializer is shared"
    );
    assert!(
        !nodes[1].output.is_empty(),
        "BN node should retain its outputs"
    );
    let weight = weights.get("shared_w").expect("original shared weight");
    assert_relative_eq!(weight[[0, 0, 0, 0]], 2.0, epsilon = 1e-6);
    assert_relative_eq!(weight[[1, 0, 0, 0]], -3.0, epsilon = 1e-6);
    assert_eq!(nodes[2].input, vec!["x2", "shared_w"]);
}

/// Siamese/twin topology: both Conv branches share the weight initializer and
/// each has its own BatchNormalization. Each branch's output single-consumer
/// guard passes, so without the shared-weight guard the fold would fire twice
/// and scale the shared tensor by the BN factor squared. Both folds must skip.
#[test]
fn test_fold_batch_norm_skips_twin_bn_shared_weight() {
    let conv_a = make_node("Conv", &["x", "shared_w"], &["a_y"]);
    let mut bn_a = make_node(
        "BatchNormalization",
        &["a_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_a_y"],
    );
    bn_a.attribute.push(make_float_attr("epsilon", 0.0));
    let conv_b = make_node("Conv", &["x2", "shared_w"], &["b_y"]);
    let mut bn_b = make_node(
        "BatchNormalization",
        &["b_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_b_y"],
    );
    bn_b.attribute.push(make_float_attr("epsilon", 0.0));
    let mut nodes = vec![conv_a, bn_a, conv_b, bn_b];

    let mut weights = WeightStore::new();
    weights.insert(
        "shared_w".to_string(),
        make_weight(&[2, 1, 1, 1], &[2.0, -3.0]),
    );
    weights.insert("bn_scale".to_string(), arr1(&[4.0, -2.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, 0.5]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.5, -1.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[4.0, 9.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    assert!(
        consumed.is_empty(),
        "fusion must be skipped for both branches of a shared-weight twin"
    );
    let weight = weights.get("shared_w").expect("original shared weight");
    assert_relative_eq!(weight[[0, 0, 0, 0]], 2.0, epsilon = 1e-6);
    assert_relative_eq!(weight[[1, 0, 0, 0]], -3.0, epsilon = 1e-6);
}

/// Two Gemm nodes with distinct weights share one bias initializer and only
/// one is followed by BatchNormalization: fusion must be skipped, because
/// rewriting the shared bias entry would corrupt the other branch.
#[test]
fn test_fold_batch_norm_skips_shared_existing_bias() {
    let mut gemm_a = make_node("Gemm", &["x", "w_a", "shared_b"], &["a_y"]);
    gemm_a.attribute.push(make_int_attr("transB", 1));
    let mut bn = make_node(
        "BatchNormalization",
        &["a_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let mut gemm_b = make_node("Gemm", &["x2", "w_b", "shared_b"], &["b_y"]);
    gemm_b.attribute.push(make_int_attr("transB", 1));
    let mut nodes = vec![gemm_a, bn, gemm_b];

    let mut weights = WeightStore::new();
    weights.insert(
        "w_a".to_string(),
        make_weight(&[2, 3], &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),
    );
    weights.insert(
        "w_b".to_string(),
        make_weight(&[2, 3], &[0.5, 1.5, 2.5, -0.5, -1.5, -2.5]),
    );
    weights.insert("shared_b".to_string(), arr1(&[7.0, 9.0]).into_dyn());
    weights.insert("bn_scale".to_string(), arr1(&[3.0, -2.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[0.5, 0.25]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[1.0, -2.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[9.0, 4.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    assert!(
        consumed.is_empty(),
        "fusion must be skipped when the bias initializer is shared"
    );
    let weight = weights.get("w_a").expect("original gemm_a weight");
    assert_relative_eq!(weight[[0, 0]], 1.0, epsilon = 1e-6);
    let bias = weights.get("shared_b").expect("original shared bias");
    assert_relative_eq!(bias[[0]], 7.0, epsilon = 1e-6);
    assert_relative_eq!(bias[[1]], 9.0, epsilon = 1e-6);
}

/// BN node without explicit epsilon attribute: `batch_norm_epsilon()` should
/// fall back to `DEFAULT_BATCH_NORM_EPSILON` (1e-5). The fused weights differ
/// from epsilon=0 because denominator = sqrt(var + 1e-5) != sqrt(var).
/// Regression test for #2321.
#[test]
fn test_fold_batch_norm_default_epsilon_fallback() {
    let conv = make_node("Conv", &["x", "conv_w"], &["conv_y"]);
    // No epsilon attribute on BN node → default 1e-5.
    let bn = make_node(
        "BatchNormalization",
        &["conv_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    let mut nodes = vec![conv, bn];

    let mut weights = WeightStore::new();
    weights.insert("conv_w".to_string(), make_weight(&[1, 1, 1, 1], &[2.0]));
    weights.insert("bn_scale".to_string(), arr1(&[1.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[0.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0]).into_dyn());
    // var=1.0: denominator = sqrt(1.0 + 1e-5) ≈ 1.000005 (not exactly 1.0)
    weights.insert("bn_var".to_string(), arr1(&[1.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    assert_eq!(
        consumed.len(),
        1,
        "fusion should succeed with default epsilon"
    );

    let fused_weight = weights.get("conv_w").expect("fused conv weight");
    // scale = ny / sqrt(var + eps) = 1.0 / sqrt(1.0 + 1e-5)
    // Weight 2.0 * scale ≈ 1.99999. If epsilon=0 were used, weight would be exactly 2.0.
    let expected = 2.0 / (1.0f32 + 1e-5).sqrt();
    assert_relative_eq!(fused_weight[[0, 0, 0, 0]], expected, epsilon = 1e-6);
    // Verify it's NOT exactly 2.0 (proving epsilon was applied).
    assert!(
        (fused_weight[[0, 0, 0, 0]] - 2.0).abs() > 1e-7,
        "fused weight should differ from eps=0 result; got {}",
        fused_weight[[0, 0, 0, 0]]
    );
}

/// var=0, eps=0: denominator = sqrt(0) = 0.0. The `denominator <= 0.0` guard
/// (batch_norm_fold.rs:196) should reject this, returning None and skipping fusion.
/// Regression test for #2321.
#[test]
fn test_fold_batch_norm_var_zero_eps_zero_rejected() {
    let conv = make_node("Conv", &["x", "conv_w"], &["conv_y"]);
    let mut bn = make_node(
        "BatchNormalization",
        &["conv_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let mut nodes = vec![conv, bn];

    let mut weights = WeightStore::new();
    weights.insert("conv_w".to_string(), make_weight(&[1, 1, 1, 1], &[2.0]));
    weights.insert("bn_scale".to_string(), arr1(&[1.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[0.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0]).into_dyn());
    // var=0.0 + eps=0.0 → denominator = sqrt(0) = 0 → guard triggers
    weights.insert("bn_var".to_string(), arr1(&[0.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    assert!(
        consumed.is_empty(),
        "fusion should be skipped when var=0, eps=0 (division by zero)"
    );
    // BN node should be left intact.
    assert!(
        !nodes[1].output.is_empty(),
        "BN node should retain its outputs when fusion is rejected"
    );
}

/// Serialize env-var mutation for the NY_BN_FOLD_EXT kill-switch tests: cargo
/// runs tests in threads within one process and the env is process-global.
/// Routed through the blessed env choke point (clippy env wall).
fn with_bn_fold_ext<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    match value {
        Some(v) => ny_test_utils::env::with_serialized_env_vars(&[("NY_BN_FOLD_EXT", v)], f),
        None => ny_test_utils::env::with_serialized_env_vars_removed(&["NY_BN_FOLD_EXT"], f),
    }
}

/// ConvTranspose+BN fold (#cgan-structural-fold): the BN per-output-channel
/// affine scales kernel axis 1 ([C_in, C_out, kH, kW]), NOT axis 0 as in Conv.
/// Same BN parameters as `test_fold_batch_norm_into_conv_adds_bias_and_rewires_output`:
/// scale = ny/sqrt(var) = [2.0, -2/3], shift = [0.0, -1/6].
#[test]
fn test_fold_batch_norm_into_conv_transpose_scales_axis_1() {
    let convt = make_node("ConvTranspose", &["x", "ct_w"], &["ct_y"]);
    let mut bn = make_node(
        "BatchNormalization",
        &["ct_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let relu = make_node("Relu", &["bn_y"], &["relu_y"]);
    let mut nodes = vec![convt, bn, relu];

    let mut weights = WeightStore::new();
    // Kernel [C_in=1, C_out=2, 1, 1]: axis-1 channels hold 2.0 and -3.0.
    weights.insert("ct_w".to_string(), make_weight(&[1, 2, 1, 1], &[2.0, -3.0]));
    weights.insert("bn_scale".to_string(), arr1(&[4.0, -2.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, 0.5]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.5, -1.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[4.0, 9.0]).into_dyn());

    let consumed = with_bn_fold_ext(None, || {
        fold_batch_norm_into_conv_linear(&mut nodes, &mut weights)
    });
    assert_eq!(consumed.len(), 1);
    assert!(consumed.contains(&1));
    assert_eq!(nodes[0].output[0], "bn_y");
    assert_eq!(nodes[1].output.len(), 0);

    let fused_weight = weights.get("ct_w").expect("fused convtranspose weight");
    // W'[0, o, 0, 0] = s[o] * W: [2*2.0, -3*(-2/3)] = [4.0, 2.0]
    assert_relative_eq!(fused_weight[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    assert_relative_eq!(fused_weight[[0, 1, 0, 0]], 2.0, epsilon = 1e-6);

    let bias_name = nodes[0].input.get(2).expect("synthetic bias added").clone();
    let fused_bias = weights.get(bias_name.as_str()).expect("fused bias");
    assert_relative_eq!(fused_bias[[0]], 0.0, epsilon = 1e-6);
    assert_relative_eq!(fused_bias[[1]], -1.0 / 6.0, epsilon = 1e-6);
}

/// ConvTranspose with an existing bias: b' = s*b + t, same as the Conv fold.
#[test]
fn test_fold_batch_norm_into_conv_transpose_updates_existing_bias() {
    let convt = make_node("ConvTranspose", &["x", "ct_w", "ct_b"], &["ct_y"]);
    let mut bn = make_node(
        "BatchNormalization",
        &["ct_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let mut nodes = vec![convt, bn];

    let mut weights = WeightStore::new();
    weights.insert("ct_w".to_string(), make_weight(&[1, 2, 1, 1], &[1.0, 1.0]));
    weights.insert("ct_b".to_string(), arr1(&[3.0, -2.0]).into_dyn());
    weights.insert("bn_scale".to_string(), arr1(&[2.0, 0.5]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, -1.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[1.0, 1.0]).into_dyn());

    let consumed = with_bn_fold_ext(None, || {
        fold_batch_norm_into_conv_linear(&mut nodes, &mut weights)
    });
    assert_eq!(consumed.len(), 1);
    assert_eq!(nodes[0].input[2], "ct_b");

    let fused_bias = weights.get("ct_b").expect("fused bias");
    // b' = s*b + t: [2*3+1, 0.5*(-2)-1] = [7.0, -2.0]
    assert_relative_eq!(fused_bias[[0]], 7.0, epsilon = 1e-6);
    assert_relative_eq!(fused_bias[[1]], -2.0, epsilon = 1e-6);
}

/// Grouped ConvTranspose (group=2): output-channel index depends on the input
/// group, so the axis-1 scale is wrong — fusion must be skipped.
#[test]
fn test_fold_batch_norm_skips_grouped_conv_transpose() {
    let mut convt = make_node("ConvTranspose", &["x", "ct_w"], &["ct_y"]);
    convt.attribute.push(make_int_attr("group", 2));
    let mut bn = make_node(
        "BatchNormalization",
        &["ct_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let mut nodes = vec![convt, bn];

    let mut weights = WeightStore::new();
    // group=2: kernel [C_in=2, C_out/group=2, 1, 1] -> C_out=4 channels.
    weights.insert(
        "ct_w".to_string(),
        make_weight(&[2, 2, 1, 1], &[1.0, 2.0, 3.0, 4.0]),
    );
    weights.insert(
        "bn_scale".to_string(),
        arr1(&[1.0, 1.0, 1.0, 1.0]).into_dyn(),
    );
    weights.insert(
        "bn_bias".to_string(),
        arr1(&[0.0, 0.0, 0.0, 0.0]).into_dyn(),
    );
    weights.insert(
        "bn_mean".to_string(),
        arr1(&[0.0, 0.0, 0.0, 0.0]).into_dyn(),
    );
    weights.insert("bn_var".to_string(), arr1(&[1.0, 1.0, 1.0, 1.0]).into_dyn());

    let consumed = with_bn_fold_ext(None, || {
        fold_batch_norm_into_conv_linear(&mut nodes, &mut weights)
    });
    assert!(consumed.is_empty(), "grouped ConvTranspose must not fold");
    assert!(!nodes[1].output.is_empty());
}

/// NY_BN_FOLD_EXT=0 disables the ConvTranspose and Gemm->Reshape->BN folds but
/// leaves the landed Conv fold active.
#[test]
fn test_bn_fold_ext_kill_switch_disables_only_extended_folds() {
    with_bn_fold_ext(Some("0"), || {
        // ConvTranspose+BN: must NOT fold with the switch off.
        let convt = make_node("ConvTranspose", &["x", "ct_w"], &["ct_y"]);
        let mut bn = make_node(
            "BatchNormalization",
            &["ct_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            &["bn_y"],
        );
        bn.attribute.push(make_float_attr("epsilon", 0.0));
        let mut nodes = vec![convt, bn];
        let mut weights = WeightStore::new();
        weights.insert("ct_w".to_string(), make_weight(&[1, 1, 1, 1], &[2.0]));
        weights.insert("bn_scale".to_string(), arr1(&[1.0]).into_dyn());
        weights.insert("bn_bias".to_string(), arr1(&[0.0]).into_dyn());
        weights.insert("bn_mean".to_string(), arr1(&[0.0]).into_dyn());
        weights.insert("bn_var".to_string(), arr1(&[1.0]).into_dyn());
        let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
        assert!(consumed.is_empty(), "kill switch must disable ConvT fold");

        // Landed Conv+BN fold: must STILL fold with the switch off.
        let conv = make_node("Conv", &["x", "conv_w"], &["conv_y"]);
        let mut bn = make_node(
            "BatchNormalization",
            &["conv_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            &["bn_y"],
        );
        bn.attribute.push(make_float_attr("epsilon", 0.0));
        let mut nodes = vec![conv, bn];
        let mut weights = WeightStore::new();
        weights.insert("conv_w".to_string(), make_weight(&[1, 1, 1, 1], &[2.0]));
        weights.insert("bn_scale".to_string(), arr1(&[1.0]).into_dyn());
        weights.insert("bn_bias".to_string(), arr1(&[0.0]).into_dyn());
        weights.insert("bn_mean".to_string(), arr1(&[0.0]).into_dyn());
        weights.insert("bn_var".to_string(), arr1(&[1.0]).into_dyn());
        let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
        assert_eq!(consumed.len(), 1, "landed Conv fold must stay active");
    });
}

/// Gemm->Reshape->BN across-Reshape fold (#cgan-structural-fold): F=4 features
/// reshape to [N, C=2, 2, 1] (block=2), so BN channel c scales features
/// [2c, 2c+1]. transB=1 weight (4, 3): rows 0-1 x s[0]=2.0, rows 2-3 x s[1]=0.5;
/// bias b'[f] = b[f]*s[f/2] + t[f/2] with t = [1.0, -1.0].
#[test]
fn test_fold_gemm_reshape_bn_block_diagonal_scale() {
    let mut gemm = make_node("Gemm", &["x", "gemm_w", "gemm_b"], &["gemm_y"]);
    gemm.attribute.push(make_int_attr("transB", 1));
    let reshape = make_node("Reshape", &["gemm_y", "target_shape"], &["reshape_y"]);
    let mut bn = make_node(
        "BatchNormalization",
        &["reshape_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let relu = make_node("Relu", &["bn_y"], &["relu_y"]);
    let mut nodes = vec![gemm, reshape, bn, relu];

    let mut weights = WeightStore::new();
    weights.insert(
        "gemm_w".to_string(),
        make_weight(
            &[4, 3],
            &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        ),
    );
    weights.insert("gemm_b".to_string(), arr1(&[1.0, 2.0, 3.0, 4.0]).into_dyn());
    weights.insert_integers(
        "target_shape".to_string(),
        arr1(&[-1_i64, 2, 2, 1]).into_dyn(),
    );
    weights.insert("bn_scale".to_string(), arr1(&[2.0, 0.5]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, -1.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[1.0, 1.0]).into_dyn());

    let consumed = with_bn_fold_ext(None, || {
        fold_batch_norm_into_conv_linear(&mut nodes, &mut weights)
    });
    assert_eq!(consumed.len(), 1);
    assert!(consumed.contains(&2));

    // Rewiring: Gemm feeds the (preserved) Reshape, which adopts the BN output.
    assert_eq!(nodes[0].output[0], "gemm_y");
    assert_eq!(nodes[1].input[0], "gemm_y");
    assert_eq!(nodes[1].output[0], "bn_y");
    assert_eq!(nodes[2].output.len(), 0);

    let fused_weight = weights.get("gemm_w").expect("fused gemm weight");
    for col in 0..3 {
        assert_relative_eq!(fused_weight[[0, col]], 2.0, epsilon = 1e-6);
        assert_relative_eq!(fused_weight[[1, col]], 4.0, epsilon = 1e-6);
        assert_relative_eq!(fused_weight[[2, col]], 1.5, epsilon = 1e-6);
        assert_relative_eq!(fused_weight[[3, col]], 2.0, epsilon = 1e-6);
    }

    let fused_bias = weights.get("gemm_b").expect("fused gemm bias");
    assert_relative_eq!(fused_bias[[0]], 3.0, epsilon = 1e-6);
    assert_relative_eq!(fused_bias[[1]], 5.0, epsilon = 1e-6);
    assert_relative_eq!(fused_bias[[2]], 0.5, epsilon = 1e-6);
    assert_relative_eq!(fused_bias[[3]], 1.0, epsilon = 1e-6);
}

/// Gemm->Reshape->BN guards: non-constant shape, channel mismatch, non-positive
/// non-batch dims, multi-consumer Reshape output, non-default Gemm alpha.
#[test]
fn test_fold_gemm_reshape_bn_guards_skip() {
    let build = |shape_known: bool, target: &[i64], alpha: Option<f32>, extra_consumer: bool| {
        let mut gemm = make_node("Gemm", &["x", "gemm_w", "gemm_b"], &["gemm_y"]);
        gemm.attribute.push(make_int_attr("transB", 1));
        if let Some(alpha) = alpha {
            gemm.attribute.push(make_float_attr("alpha", alpha));
        }
        let reshape = make_node("Reshape", &["gemm_y", "target_shape"], &["reshape_y"]);
        let mut bn = make_node(
            "BatchNormalization",
            &["reshape_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            &["bn_y"],
        );
        bn.attribute.push(make_float_attr("epsilon", 0.0));
        let mut nodes = vec![gemm, reshape, bn];
        if extra_consumer {
            nodes.push(make_node("Relu", &["reshape_y"], &["other_y"]));
        }

        let mut weights = WeightStore::new();
        weights.insert(
            "gemm_w".to_string(),
            make_weight(
                &[4, 3],
                &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
            ),
        );
        weights.insert("gemm_b".to_string(), arr1(&[1.0, 2.0, 3.0, 4.0]).into_dyn());
        if shape_known {
            weights.insert_integers(
                "target_shape".to_string(),
                ndarray::Array::from_vec(target.to_vec()).into_dyn(),
            );
        }
        weights.insert("bn_scale".to_string(), arr1(&[2.0, 0.5]).into_dyn());
        weights.insert("bn_bias".to_string(), arr1(&[1.0, -1.0]).into_dyn());
        weights.insert("bn_mean".to_string(), arr1(&[0.0, 0.0]).into_dyn());
        weights.insert("bn_var".to_string(), arr1(&[1.0, 1.0]).into_dyn());
        (nodes, weights)
    };

    with_bn_fold_ext(None, || {
        // Non-constant reshape shape.
        let (mut nodes, mut weights) = build(false, &[], None, false);
        assert!(fold_batch_norm_into_conv_linear(&mut nodes, &mut weights).is_empty());

        // target[1] != C (4 != 2).
        let (mut nodes, mut weights) = build(true, &[-1, 4, 1, 1], None, false);
        assert!(fold_batch_norm_into_conv_linear(&mut nodes, &mut weights).is_empty());

        // -1 in a non-batch position.
        let (mut nodes, mut weights) = build(true, &[1, 2, -1, 1], None, false);
        assert!(fold_batch_norm_into_conv_linear(&mut nodes, &mut weights).is_empty());

        // Non-batch product != F (2*3 != 4).
        let (mut nodes, mut weights) = build(true, &[-1, 2, 3], None, false);
        assert!(fold_batch_norm_into_conv_linear(&mut nodes, &mut weights).is_empty());

        // Reshape output consumed by BN and another node.
        let (mut nodes, mut weights) = build(true, &[-1, 2, 2, 1], None, true);
        assert!(fold_batch_norm_into_conv_linear(&mut nodes, &mut weights).is_empty());

        // Gemm alpha != 1.0.
        let (mut nodes, mut weights) = build(true, &[-1, 2, 2, 1], Some(2.0), false);
        assert!(fold_batch_norm_into_conv_linear(&mut nodes, &mut weights).is_empty());
    });
}

/// var=0, eps=1e-5: denominator = sqrt(0 + 1e-5) = sqrt(1e-5) > 0.
/// Fusion should succeed because epsilon prevents the zero-denominator case.
/// Regression test for #2321.
#[test]
fn test_fold_batch_norm_var_zero_eps_positive_succeeds() {
    let conv = make_node("Conv", &["x", "conv_w"], &["conv_y"]);
    let mut bn = make_node(
        "BatchNormalization",
        &["conv_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 1e-5));
    let mut nodes = vec![conv, bn];

    let mut weights = WeightStore::new();
    weights.insert("conv_w".to_string(), make_weight(&[1, 1, 1, 1], &[2.0]));
    weights.insert("bn_scale".to_string(), arr1(&[1.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[0.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0]).into_dyn());
    // var=0.0 but eps=1e-5 → denominator = sqrt(1e-5) > 0 → valid
    weights.insert("bn_var".to_string(), arr1(&[0.0]).into_dyn());

    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    assert_eq!(
        consumed.len(),
        1,
        "fusion should succeed when var=0 but eps>0"
    );

    let fused_weight = weights.get("conv_w").expect("fused conv weight");
    // scale = ny / sqrt(var + eps) = 1.0 / sqrt(1e-5) = 1/0.003162... ≈ 316.23
    // fused_weight = 2.0 * scale ≈ 632.46
    let expected = 2.0 / (1e-5f32).sqrt();
    assert_relative_eq!(fused_weight[[0, 0, 0, 0]], expected, epsilon = 1e-1);
}
