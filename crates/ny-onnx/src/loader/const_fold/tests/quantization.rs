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

/// Output dtypes the fold cannot load/model exactly must leave the node
/// unfolded. This includes FLOAT8 and packed nibble storage (UINT4/INT4).
#[test]
fn test_quantize_linear_unmodelled_output_dtype_not_folded() {
    for dtype in [17, 21, 22] {
        let graph = GraphProto {
            node: vec![node(
                "q",
                "QuantizeLinear",
                &["w", "s"],
                &["q_out"],
                vec![attr_int("output_dtype", dtype)],
            )],
            ..Default::default()
        };
        let mut weights = quantize_weights();

        fold(&graph, &mut weights);

        assert!(
            !weights.contains_key("q_out"),
            "unmodelled output_dtype {dtype} must not be folded"
        );
    }
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

#[test]
fn test_quantize_linear_inexact_ratio_uses_float32_division() {
    let graph = GraphProto {
        node: vec![node(
            "quantize",
            "QuantizeLinear",
            &["x", "scale"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert(
        "x".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.1]).unwrap(),
    );
    weights.insert(
        "scale".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![0.3]).unwrap(),
    );

    fold(&graph, &mut weights);
    assert_folded_tensor(&weights, "out", &[1], &[0.0]);
}

/// The exact-real quotient is slightly above 2.5, but FLOAT division rounds
/// it to exactly 2.5 before ties-to-even. The ONNX result is therefore 2.
#[test]
fn test_quantize_linear_rounding_seam_matches_float32_program() {
    let graph = GraphProto {
        node: vec![node(
            "quantize",
            "QuantizeLinear",
            &["x", "scale", ""],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert(
        "x".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.750_000_06]).unwrap(),
    );
    weights.insert(
        "scale".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![0.3]).unwrap(),
    );

    fold(&graph, &mut weights);
    assert_folded_tensor(&weights, "out", &[1], &[2.0]);
}

#[test]
fn test_quantize_linear_bare_float_zero_point_is_not_authenticated() {
    let graph = GraphProto {
        node: vec![node(
            "quantize",
            "QuantizeLinear",
            &["w", "s", "zp"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = quantize_weights();
    weights.insert(
        "zp".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![0.0]).unwrap(),
    );

    fold(&graph, &mut weights);
    assert!(
        !weights.contains_key("out"),
        "an integral-looking FLOAT is not proof of an integer zero-point dtype"
    );
}

#[test]
fn test_quantize_linear_int32_zero_point_is_not_a_legal_output_type() {
    let graph = GraphProto {
        node: vec![node(
            "quantize",
            "QuantizeLinear",
            &["w", "s", "zp"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = quantize_weights();
    weights.insert_integers(
        "zp".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![0]).unwrap(),
    );
    weights.insert_integer_range("zp".to_string(), i32::MIN as i64, i32::MAX as i64);

    fold(&graph, &mut weights);
    assert!(!weights.contains_key("out"));
}

#[test]
fn test_dequantize_linear_reproduces_float32_arithmetic() {
    let graph = GraphProto {
        node: vec![node(
            "dequantize",
            "DequantizeLinear",
            &["x", "scale"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert_integers(
        "x".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3]).unwrap(),
    );
    weights.insert_integer_range("x".to_string(), i8::MIN as i64, i8::MAX as i64);
    weights.insert(
        "scale".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![0.1]).unwrap(),
    );

    fold(&graph, &mut weights);
    assert_folded_tensor(&weights, "out", &[1], &[3.0f32 * 0.1f32]);
}

#[test]
fn test_dequantize_linear_omitted_zero_point_broadcasts_over_rankful_input() {
    for shape in [vec![2], vec![2, 2]] {
        let graph = GraphProto {
            node: vec![node(
                "dequantize",
                "DequantizeLinear",
                &["x", "scale"],
                &["out"],
                Vec::new(),
            )],
            ..Default::default()
        };
        let len = shape.iter().product();
        let mut weights = WeightStore::new();
        weights.insert_integers(
            "x".to_string(),
            ArrayD::from_shape_vec(IxDyn(&shape), (1..=i64::try_from(len).unwrap()).collect())
                .unwrap(),
        );
        weights.insert_integer_range("x".to_string(), i8::MIN as i64, i8::MAX as i64);
        weights.insert(
            "scale".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![0.25]).unwrap(),
        );

        fold(&graph, &mut weights);
        assert_folded_tensor(
            &weights,
            "out",
            &shape,
            &(1..=len)
                .map(|value| value as f32 * 0.25)
                .collect::<Vec<_>>(),
        );
    }
}

#[test]
fn test_dequantize_linear_bare_float_input_is_not_authenticated() {
    let graph = GraphProto {
        node: vec![node(
            "dequantize",
            "DequantizeLinear",
            &["x", "scale"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert(
        "x".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
    );
    weights.insert(
        "scale".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![0.1]).unwrap(),
    );

    fold(&graph, &mut weights);
    assert!(!weights.contains_key("out"));
}
