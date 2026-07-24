// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! QuantizeLinear constant-fold tests, in particular the opset-21
//! `output_dtype` attribute paths.

use super::common::{assert_folded_tensor, attr_int, fold, node};
use crate::onnx_proto::GraphProto;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};

fn quantize_weights() -> WeightStore {
    let mut weights = WeightStore::new();
    weights.insert(
        "w".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, -0.5, 0.5, 1.0]).unwrap(),
    );
    weights.insert(
        "s".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.1]).unwrap(),
    );
    weights
}

/// output_dtype=INT8 with no y_zero_point: negative quantized values must
/// survive with the int8 clamp range (-128, 127), not collapse to 0 under a
/// uint8 default.
#[test]
fn test_quantize_linear_output_dtype_int8_no_zero_point() {
    let graph = GraphProto {
        node: vec![node(
            "q",
            "QuantizeLinear",
            &["w", "s"],
            &["q_out"],
            vec![attr_int("output_dtype", 3)],
        )],
        ..Default::default()
    };
    let mut weights = quantize_weights();

    fold(&graph, &mut weights);

    assert_folded_tensor(&weights, "q_out", &[4], &[-10.0, -5.0, 5.0, 10.0]);
    assert_eq!(
        weights.get_integer_range("q_out"),
        Some((i8::MIN as i64, i8::MAX as i64))
    );
}

/// No output_dtype and no y_zero_point: the spec default is uint8, so
/// negative quantized values clamp to 0.
#[test]
fn test_quantize_linear_defaults_to_uint8_without_output_dtype_or_zero_point() {
    let graph = GraphProto {
        node: vec![node(
            "q",
            "QuantizeLinear",
            &["w", "s"],
            &["q_out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = quantize_weights();

    fold(&graph, &mut weights);

    assert_folded_tensor(&weights, "q_out", &[4], &[0.0, 0.0, 5.0, 10.0]);
    assert_eq!(
        weights.get_integer_range("q_out"),
        Some((0, u8::MAX as i64))
    );
}

/// An output_dtype the fold cannot model exactly (FLOAT8E4M3FN = 17, whose
/// result also depends on `saturate`) must leave the node unfolded.
#[test]
fn test_quantize_linear_unmodelled_output_dtype_not_folded() {
    let graph = GraphProto {
        node: vec![node(
            "q",
            "QuantizeLinear",
            &["w", "s"],
            &["q_out"],
            vec![attr_int("output_dtype", 17)],
        )],
        ..Default::default()
    };
    let mut weights = quantize_weights();

    fold(&graph, &mut weights);

    assert!(
        !weights.contains_key("q_out"),
        "unmodelled output_dtype must not be folded"
    );
}

/// output_dtype disagreeing with the y_zero_point type is malformed and must
/// leave the node unfolded rather than picking either range.
#[test]
fn test_quantize_linear_output_dtype_zero_point_mismatch_not_folded() {
    let graph = GraphProto {
        node: vec![node(
            "q",
            "QuantizeLinear",
            &["w", "s", "zp"],
            &["q_out"],
            vec![attr_int("output_dtype", 2)],
        )],
        ..Default::default()
    };
    let mut weights = quantize_weights();
    weights.insert(
        "zp".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
    );
    weights.insert_integers(
        "zp".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0i64]).unwrap(),
    );
    weights.insert_integer_range("zp".to_string(), i8::MIN as i64, i8::MAX as i64);

    fold(&graph, &mut weights);

    assert!(
        !weights.contains_key("q_out"),
        "output_dtype/zero_point type mismatch must not be folded"
    );
}

/// output_dtype agreeing with the y_zero_point type folds normally.
#[test]
fn test_quantize_linear_output_dtype_matches_zero_point() {
    let graph = GraphProto {
        node: vec![node(
            "q",
            "QuantizeLinear",
            &["w", "s", "zp"],
            &["q_out"],
            vec![attr_int("output_dtype", 3)],
        )],
        ..Default::default()
    };
    let mut weights = quantize_weights();
    weights.insert(
        "zp".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    );
    weights.insert_integers(
        "zp".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1i64]).unwrap(),
    );
    weights.insert_integer_range("zp".to_string(), i8::MIN as i64, i8::MAX as i64);

    fold(&graph, &mut weights);

    assert_folded_tensor(&weights, "q_out", &[4], &[-9.0, -4.0, 6.0, 11.0]);
    assert_eq!(
        weights.get_integer_range("q_out"),
        Some((i8::MIN as i64, i8::MAX as i64))
    );
}
