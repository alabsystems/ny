// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! BN -> Reshape -> Gemm cGAN tail-fusion regressions.

use super::{make_float_attr, make_int_attr, make_node, make_weight};
use crate::loader::fusion::{
    fold_batch_norm_into_conv_linear, fold_batch_norm_into_conv_linear_with_context,
};
use crate::model::WeightStore;
use approx::assert_relative_eq;
use ndarray::{arr0, arr1, arr2};
use prost::Message;
use std::collections::{HashMap, HashSet};

fn with_extended_folds<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    match value {
        Some(value) => {
            ny_test_utils::env::with_serialized_env_vars(&[("NY_BN_FOLD_EXT", value)], f)
        }
        None => ny_test_utils::env::with_serialized_env_vars_removed(&["NY_BN_FOLD_EXT"], f),
    }
}

fn tail_fixture(trans_b: bool, target: &[i64]) -> (Vec<crate::onnx_proto::NodeProto>, WeightStore) {
    let mut bn = make_node(
        "BatchNormalization",
        &["x", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let reshape = make_node("Reshape", &["bn_y", "target_shape"], &["flat_y"]);
    let mut gemm = make_node("Gemm", &["flat_y", "gemm_w", "gemm_b"], &["out"]);
    if trans_b {
        gemm.attribute.push(make_int_attr("transB", 1));
    }
    let nodes = vec![bn, reshape, gemm];

    let mut weights = WeightStore::new();
    let values = if trans_b {
        // (out=2, features=4)
        vec![1.0, 2.0, 3.0, 4.0, -1.0, 1.0, 2.0, -2.0]
    } else {
        // (features=4, out=2), the transpose of the matrix above.
        vec![1.0, -1.0, 2.0, 1.0, 3.0, 2.0, 4.0, -2.0]
    };
    let shape = if trans_b { [2, 4] } else { [4, 2] };
    weights.insert("gemm_w".to_string(), make_weight(&shape, &values));
    weights.insert("gemm_b".to_string(), arr1(&[0.25, -0.5]).into_dyn());
    weights.insert_integers(
        "target_shape".to_string(),
        ndarray::Array::from_vec(target.to_vec()).into_dyn(),
    );
    weights.insert("bn_scale".to_string(), arr1(&[2.0, 0.5]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, -1.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[1.0, 1.0]).into_dyn());
    (nodes, weights)
}

fn fold_tail(
    nodes: &mut [crate::onnx_proto::NodeProto],
    weights: &mut WeightStore,
) -> HashSet<usize> {
    fold_tail_with_context(nodes, weights, &[1, 2, 2], &["out"])
}

fn fold_tail_with_context(
    nodes: &mut [crate::onnx_proto::NodeProto],
    weights: &mut WeightStore,
    source_shape: &[i64],
    graph_outputs: &[&str],
) -> HashSet<usize> {
    let tensor_shapes = HashMap::from([("x".to_string(), source_shape.to_vec())]);
    let graph_output_names = graph_outputs
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    fold_batch_norm_into_conv_linear_with_context(
        nodes,
        weights,
        &tensor_shapes,
        &graph_output_names,
    )
}

#[test]
fn folds_cgan_bn_reshape_gemm_trans_b_weight_and_bias() {
    with_extended_folds(None, || {
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        let consumed = fold_tail(&mut nodes, &mut weights);
        assert_eq!(consumed, [0].into_iter().collect());
        assert!(nodes[0].input.is_empty() && nodes[0].output.is_empty());
        assert_eq!(nodes[1].input[0], "x");
        assert_eq!(nodes[1].output[0], "flat_y");
        assert_eq!(nodes[2].input[0], "flat_y");
        assert_eq!(nodes[2].output[0], "out");

        let weight = weights.get("gemm_w").expect("fused weight");
        let expected = [[2.0, 4.0, 1.5, 2.0], [-2.0, 2.0, 1.0, -1.0]];
        for output in 0..2 {
            for feature in 0..4 {
                assert_relative_eq!(
                    weight[[output, feature]],
                    expected[output][feature],
                    epsilon = 1e-6
                );
            }
        }
        let bias = weights.get("gemm_b").expect("fused bias");
        assert_relative_eq!(bias[[0]], -3.75, epsilon = 1e-6);
        assert_relative_eq!(bias[[1]], -0.5, epsilon = 1e-6);
    });
}

#[test]
fn folds_cgan_bn_reshape_gemm_nontransposed_weight() {
    with_extended_folds(None, || {
        let (mut nodes, mut weights) = tail_fixture(false, &[1, -1]);
        let consumed = fold_tail(&mut nodes, &mut weights);
        assert!(consumed.contains(&0));
        let weight = weights.get("gemm_w").expect("fused weight");
        let expected = [[2.0, -2.0], [4.0, 2.0], [1.5, 1.0], [2.0, -1.0]];
        for feature in 0..4 {
            for output in 0..2 {
                assert_relative_eq!(
                    weight[[feature, output]],
                    expected[feature][output],
                    epsilon = 1e-6
                );
            }
        }
        let bias = weights.get("gemm_b").expect("fused bias");
        assert_relative_eq!(bias[[0]], -3.75, epsilon = 1e-6);
        assert_relative_eq!(bias[[1]], -0.5, epsilon = 1e-6);
    });
}

#[test]
fn folds_cgan_bn_reshape_gemm_without_authored_bias() {
    with_extended_folds(None, || {
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        nodes[2].input.pop();
        let consumed = fold_tail(&mut nodes, &mut weights);
        assert!(consumed.contains(&0));
        let bias_name = nodes[2].input.get(2).expect("synthetic Gemm bias");
        assert_ne!(bias_name, "gemm_b");
        let bias = weights.get(bias_name).expect("stored synthetic bias");
        assert_relative_eq!(bias[[0]], -4.0, epsilon = 1e-6);
        assert_relative_eq!(bias[[1]], 0.0, epsilon = 1e-6);
    });
}

#[test]
fn cgan_tail_accepts_only_row_invariant_gemm_c_broadcasts() {
    with_extended_folds(None, || {
        // Scalar C is row-invariant and expands to one value per output.
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        weights.insert("gemm_b".to_string(), arr0(0.25).into_dyn());
        assert!(fold_tail(&mut nodes, &mut weights).contains(&0));
        let bias = weights.get("gemm_b").expect("normalized scalar C");
        assert_eq!(bias.shape(), &[2]);
        assert_relative_eq!(bias[[0]], -3.75, epsilon = 1e-6);
        assert_relative_eq!(bias[[1]], 0.25, epsilon = 1e-6);

        // `[1,N]` is the other rank-2 form whose broadcast is independent of M.
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        weights.insert(
            "gemm_b".to_string(),
            arr2(&[[0.25_f32, -0.5_f32]]).into_dyn(),
        );
        assert!(fold_tail(&mut nodes, &mut weights).contains(&0));
        let bias = weights.get("gemm_b").expect("normalized [1,N] C");
        assert_eq!(bias.shape(), &[2]);
        assert_relative_eq!(bias[[0]], -3.75, epsilon = 1e-6);
        assert_relative_eq!(bias[[1]], -0.5, epsilon = 1e-6);

        // `[M,1]` varies by runtime row and cannot become a single Gemm bias.
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        weights.insert(
            "gemm_b".to_string(),
            arr2(&[[0.25_f32], [-0.5_f32]]).into_dyn(),
        );
        let nodes_before = nodes.clone();
        let weight_before = weights.get("gemm_w").expect("weight").clone();
        let bias_before = weights.get("gemm_b").expect("bias").clone();
        assert!(fold_tail(&mut nodes, &mut weights).is_empty());
        assert_eq!(nodes, nodes_before);
        assert_eq!(weights.get("gemm_w"), Some(&weight_before));
        assert_eq!(weights.get("gemm_b"), Some(&bias_before));
    });
}

#[test]
fn cgan_tail_requires_bit_exact_default_gemm_affine_without_mutation() {
    with_extended_folds(None, || {
        let next_up = f32::from_bits(1.0_f32.to_bits() + 1);
        let next_down = f32::from_bits(1.0_f32.to_bits() - 1);
        for attribute in ["alpha", "beta"] {
            for value in [next_up, next_down, f32::NAN] {
                let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
                nodes[2].attribute.push(make_float_attr(attribute, value));
                let nodes_before: Vec<Vec<u8>> = nodes.iter().map(Message::encode_to_vec).collect();
                let weight_before = weights.get("gemm_w").expect("weight").clone();
                let bias_before = weights.get("gemm_b").expect("bias").clone();

                assert!(fold_tail(&mut nodes, &mut weights).is_empty());
                assert_eq!(
                    nodes.iter().map(Message::encode_to_vec).collect::<Vec<_>>(),
                    nodes_before,
                    "{attribute}={value:?} must leave graph bytes unchanged"
                );
                assert_eq!(weights.get("gemm_w"), Some(&weight_before));
                assert_eq!(weights.get("gemm_b"), Some(&bias_before));
            }
        }
    });
}

#[test]
fn cgan_tail_requires_exact_channel_major_source_shape() {
    with_extended_folds(None, || {
        // Shape syntax alone looks accepted, but flattening [1,2,2] to [-1,2]
        // creates two rows that each contain one channel, not two features whose
        // columns correspond to channels.
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 2]);
        weights.insert(
            "gemm_w".to_string(),
            make_weight(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
        );
        assert!(fold_tail_with_context(&mut nodes, &mut weights, &[1, 2, 2], &[]).is_empty());

        // `[1,-1]` is safe only when the exact source batch dimension is one.
        let (mut nodes, mut weights) = tail_fixture(true, &[1, -1]);
        assert!(fold_tail_with_context(&mut nodes, &mut weights, &[2, 2, 2], &[]).is_empty());

        // Missing or symbolic source dimensions do not prove the layout.
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        assert!(fold_batch_norm_into_conv_linear_with_context(
            &mut nodes,
            &mut weights,
            &HashMap::new(),
            &HashSet::new(),
        )
        .is_empty());
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        assert!(fold_tail_with_context(&mut nodes, &mut weights, &[1, 2, -1], &[]).is_empty());
    });
}

#[test]
fn cgan_tail_requires_inference_bn_and_unobservable_intermediate_y() {
    with_extended_folds(None, || {
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        nodes[0].output.push("running_mean".to_string());
        assert!(fold_tail(&mut nodes, &mut weights).is_empty());

        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        nodes[0].attribute.push(make_int_attr("training_mode", 1));
        assert!(fold_tail(&mut nodes, &mut weights).is_empty());

        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        assert!(fold_tail_with_context(&mut nodes, &mut weights, &[1, 2, 2], &["bn_y"]).is_empty());
        assert_eq!(nodes[0].output, ["bn_y"]);
        assert_eq!(nodes[1].input[0], "bn_y");

        // Reshape output remains named after fusion but would change from the
        // flattened BN value to the flattened raw input.
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        let nodes_before = nodes.clone();
        let weight_before = weights.get("gemm_w").expect("weight").clone();
        let bias_before = weights.get("gemm_b").expect("bias").clone();
        assert!(
            fold_tail_with_context(&mut nodes, &mut weights, &[1, 2, 2], &["out", "flat_y"],)
                .is_empty()
        );
        assert_eq!(nodes, nodes_before);
        assert_eq!(weights.get("gemm_w"), Some(&weight_before));
        assert_eq!(weights.get("gemm_b"), Some(&bias_before));
    });
}

#[test]
fn cgan_tail_preserves_graph_output_exposed_gemm_initializers() {
    with_extended_folds(None, || {
        for exposed in ["gemm_w", "gemm_b"] {
            let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
            let nodes_before = nodes.clone();
            let weight_before = weights.get("gemm_w").expect("weight").clone();
            let bias_before = weights.get("gemm_b").expect("bias").clone();
            assert!(fold_tail_with_context(
                &mut nodes,
                &mut weights,
                &[1, 2, 2],
                &["out", exposed],
            )
            .is_empty());
            assert_eq!(nodes, nodes_before);
            assert_eq!(weights.get("gemm_w"), Some(&weight_before));
            assert_eq!(weights.get("gemm_b"), Some(&bias_before));
        }
    });
}

#[test]
fn cgan_tail_rejects_aliased_b_c_and_allocates_fresh_bias_name() {
    with_extended_folds(None, || {
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        nodes[2].input[2] = "gemm_w".to_string();
        let nodes_before = nodes.clone();
        let weight_before = weights.get("gemm_w").expect("weight").clone();
        assert!(fold_tail(&mut nodes, &mut weights).is_empty());
        assert_eq!(nodes, nodes_before);
        assert_eq!(weights.get("gemm_w"), Some(&weight_before));

        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        nodes[2].input.pop();
        weights.insert(
            "out__bn_fused_bias".to_string(),
            arr1(&[91.0_f32]).into_dyn(),
        );
        nodes.push(make_node(
            "Identity",
            &["unrelated"],
            &["out__bn_fused_bias__1"],
        ));
        let sentinel = weights
            .get("out__bn_fused_bias")
            .expect("collision sentinel")
            .clone();
        assert!(fold_tail(&mut nodes, &mut weights).contains(&0));
        assert_eq!(nodes[2].input[2], "out__bn_fused_bias__2");
        assert_eq!(
            weights.get("out__bn_fused_bias"),
            Some(&sentinel),
            "fresh allocation must not overwrite the collision"
        );
        assert!(weights.get("out__bn_fused_bias__2").is_some());
    });
}

#[test]
fn cgan_bn_reshape_gemm_fold_is_killable_and_fail_closed() {
    with_extended_folds(Some("0"), || {
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        assert!(fold_tail(&mut nodes, &mut weights).is_empty());
        assert!(!nodes[0].output.is_empty());
    });

    with_extended_folds(None, || {
        // Only the two official exporter shapes are accepted.
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 2, 2]);
        assert!(fold_tail(&mut nodes, &mut weights).is_empty());

        // A second BN-output consumer makes removing BN invalid.
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        nodes.push(make_node("Identity", &["bn_y"], &["other"]));
        assert!(fold_tail(&mut nodes, &mut weights).is_empty());

        // Mutating a shared Gemm initializer would alter the second consumer.
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        nodes.push(make_node("Gemm", &["z", "gemm_w"], &["other"]));
        assert!(fold_tail(&mut nodes, &mut weights).is_empty());

        // Non-default Gemm scaling is outside this exact algebra.
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        nodes[2].attribute.push(make_float_attr("alpha", 2.0));
        assert!(fold_tail(&mut nodes, &mut weights).is_empty());

        // A named but non-constant Gemm C input is dynamic, not an absent bias.
        let (mut nodes, mut weights) = tail_fixture(true, &[-1, 4]);
        weights.remove("gemm_b");
        assert!(fold_tail(&mut nodes, &mut weights).is_empty());

        // The same fail-closed rule applies to the two existing predecessor
        // folds: direct Gemm->BN and Gemm->Reshape->BN.
        let dynamic_predecessor = |across_reshape: bool| {
            let mut gemm = make_node("Gemm", &["x", "pre_w", "dynamic_c"], &["pre_y"]);
            gemm.attribute.push(make_int_attr("transB", 1));
            let mut bn = make_node(
                "BatchNormalization",
                &["pre_y", "pre_scale", "pre_beta", "pre_mean", "pre_var"],
                &["out"],
            );
            bn.attribute.push(make_float_attr("epsilon", 0.0));
            let mut weights = WeightStore::new();
            weights.insert("pre_w".to_string(), make_weight(&[2, 1], &[1.0, 2.0]));
            weights.insert("pre_scale".to_string(), arr1(&[1.0, 1.0]).into_dyn());
            weights.insert("pre_beta".to_string(), arr1(&[0.0, 0.0]).into_dyn());
            weights.insert("pre_mean".to_string(), arr1(&[0.0, 0.0]).into_dyn());
            weights.insert("pre_var".to_string(), arr1(&[1.0, 1.0]).into_dyn());
            if across_reshape {
                let reshape = make_node("Reshape", &["pre_y", "pre_shape"], &["reshape_y"]);
                bn.input[0] = "reshape_y".to_string();
                weights.insert_integers("pre_shape".to_string(), arr1(&[-1_i64, 2, 1]).into_dyn());
                (vec![gemm, reshape, bn], weights)
            } else {
                (vec![gemm, bn], weights)
            }
        };
        for across_reshape in [false, true] {
            let (mut nodes, mut weights) = dynamic_predecessor(across_reshape);
            assert!(fold_batch_norm_into_conv_linear(&mut nodes, &mut weights).is_empty());
        }
    });
}
