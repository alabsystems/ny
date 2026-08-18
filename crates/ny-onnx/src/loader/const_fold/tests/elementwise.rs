// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::common::{fold, node};
use crate::onnx_proto::GraphProto;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_mul_constant_fold_broadcasts_shapes() {
    // Mul with broadcasting: [2,2] * [2] → [2,2] (row-wise broadcast)
    let graph = GraphProto {
        node: vec![node("mul", "Mul", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![5.0, 6.0]).unwrap();
    weights.insert("a".to_string(), a);
    weights.insert("b".to_string(), b);

    fold(&graph, &mut weights);

    let out = weights
        .get("out")
        .expect("Mul should fold with broadcasting");
    assert_eq!(out.shape(), &[2, 2]);
    // [1*5, 2*6, 3*5, 4*6] = [5, 12, 15, 24]
    assert!((out[[0, 0]] - 5.0).abs() < 1.0e-6);
    assert!((out[[0, 1]] - 12.0).abs() < 1.0e-6);
    assert!((out[[1, 0]] - 15.0).abs() < 1.0e-6);
    assert!((out[[1, 1]] - 24.0).abs() < 1.0e-6);
}

#[test]
fn test_div_rejects_zero_divisor() {
    let graph = GraphProto {
        node: vec![node("div", "Div", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let a = ArrayD::from_shape_vec(IxDyn(&[2]), vec![4.0, 6.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    weights.insert("a".to_string(), a);
    weights.insert("b".to_string(), b);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

fn insert_integer_tensor(weights: &mut WeightStore, name: &str, shape: &[usize], values: Vec<i64>) {
    weights.insert(
        name.to_string(),
        ArrayD::from_shape_vec(
            IxDyn(shape),
            values.iter().map(|&value| value as f32).collect(),
        )
        .unwrap(),
    );
    weights.insert_integers(
        name.to_string(),
        ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
    );
}

#[test]
fn integer_div_without_dtype_range_is_not_folded() {
    let graph = GraphProto {
        node: vec![node("div", "Div", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    insert_integer_tensor(&mut weights, "a", &[3], vec![64, -17, 0]);
    insert_integer_tensor(&mut weights, "b", &[], vec![8]);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn integer_div_declines_exact_payload_on_zero_or_overflow() {
    for (numerator, denominator) in [(7_i64, 0_i64), (i64::MIN, -1_i64)] {
        let graph = GraphProto {
            node: vec![node("div", "Div", &["a", "b"], &["out"], Vec::new())],
            ..Default::default()
        };
        let mut weights = WeightStore::new();
        insert_integer_tensor(&mut weights, "a", &[], vec![numerator]);
        insert_integer_tensor(&mut weights, "b", &[], vec![denominator]);

        fold(&graph, &mut weights);

        assert!(!weights.contains_key("out"));
        assert!(
            weights.get_integers("out").is_none(),
            "invalid integer division must not publish exact integer provenance"
        );
    }
}

#[test]
fn integer_div_declines_incompatible_broadcast() {
    let graph = GraphProto {
        node: vec![node("div", "Div", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    insert_integer_tensor(&mut weights, "a", &[2], vec![8, 16]);
    insert_integer_tensor(&mut weights, "b", &[3], vec![2, 4, 8]);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
    assert!(weights.get_integers("out").is_none());
}

#[test]
fn integer_mul_without_dtype_range_is_not_folded() {
    let graph = GraphProto {
        node: vec![node("mul", "Mul", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    insert_integer_tensor(&mut weights, "a", &[3], vec![2, -3, 0]);
    insert_integer_tensor(&mut weights, "b", &[], vec![8]);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn integer_mul_declines_exact_payload_on_overflow_or_incompatible_broadcast() {
    for (lhs_shape, lhs, rhs_shape, rhs) in [
        (&[][..], vec![i64::MAX], &[][..], vec![2_i64]),
        (&[2][..], vec![2_i64, 3], &[3][..], vec![4_i64, 5, 6]),
    ] {
        let graph = GraphProto {
            node: vec![node("mul", "Mul", &["a", "b"], &["out"], Vec::new())],
            ..Default::default()
        };
        let mut weights = WeightStore::new();
        insert_integer_tensor(&mut weights, "a", lhs_shape, lhs);
        insert_integer_tensor(&mut weights, "b", rhs_shape, rhs);

        fold(&graph, &mut weights);

        assert!(!weights.contains_key("out"));
        assert!(
            weights.get_integers("out").is_none(),
            "overflow or shape mismatch must not publish exact Mul provenance"
        );
    }
}

#[test]
fn test_int64_overflow_does_not_fall_back_to_float_fold() {
    let graph = GraphProto {
        node: vec![node("add", "Add", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    for (name, value) in [("a", i64::MAX), ("b", 1_i64)] {
        weights.insert(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![value as f32]).unwrap(),
        );
        weights.insert_integers(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![value]).unwrap(),
        );
        weights.insert_integer_range(name.to_string(), i64::MIN, i64::MAX);
    }

    fold(&graph, &mut weights);

    assert!(
        !weights.contains_key("out"),
        "checked INT64 overflow must fail folding instead of publishing an f32 approximation"
    );
}

#[test]
fn test_int64_shape_sentinel_arithmetic_fails_closed() {
    let graph = GraphProto {
        node: vec![node("add", "Add", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let sentinel = ny_core::reshape_copy_axis_sentinel(3).expect("axis sentinel");
    let mut weights = WeightStore::new();
    for (name, value) in [("a", sentinel), ("b", 1_i64)] {
        weights.insert(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![value as f32]).unwrap(),
        );
        weights.insert_integers(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![value]).unwrap(),
        );
        weights.insert_integer_range(name.to_string(), i64::MIN, i64::MAX);
    }

    fold(&graph, &mut weights);

    assert!(
        !weights.contains_key("out"),
        "a private copy-axis sentinel must never be transformed as an ordinary INT64 value"
    );
}

#[test]
fn test_int64_arithmetic_cannot_synthesize_shape_sentinel() {
    let graph = GraphProto {
        node: vec![node("add", "Add", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let sentinel = ny_core::reshape_copy_axis_sentinel(0).expect("axis sentinel");
    let mut weights = WeightStore::new();
    for (name, value) in [("a", sentinel - 1), ("b", 1_i64)] {
        weights.insert(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![value as f32]).unwrap(),
        );
        weights.insert_integers(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![value]).unwrap(),
        );
        weights.insert_integer_range(name.to_string(), i64::MIN, i64::MAX);
    }

    fold(&graph, &mut weights);

    assert!(
        !weights.contains_key("out"),
        "ordinary INT64 arithmetic must not publish a value in the private sentinel range"
    );
}

#[test]
fn test_int64_rankful_singleton_broadcast_preserves_rank() {
    let graph = GraphProto {
        node: vec![node("add", "Add", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    for (name, shape, values) in [
        ("a", &[1, 1][..], vec![10_i64]),
        ("b", &[2][..], vec![1_i64, 2]),
    ] {
        weights.insert(
            name.to_string(),
            ArrayD::from_shape_vec(
                IxDyn(shape),
                values.iter().map(|&value| value as f32).collect(),
            )
            .unwrap(),
        );
        weights.insert_integers(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
        );
        weights.insert_integer_range(name.to_string(), i64::MIN, i64::MAX);
    }

    fold(&graph, &mut weights);

    let output = weights.get_integers("out").expect("exact Add should fold");
    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(output.iter().copied().collect::<Vec<_>>(), vec![11, 12]);
}

#[test]
fn test_integer_comparisons_do_not_collapse_above_f32_precision() {
    let graph = GraphProto {
        node: vec![
            node("equal", "Equal", &["a", "b"], &["equal_out"], Vec::new()),
            node("less", "Less", &["a", "b"], &["less_out"], Vec::new()),
            node(
                "where",
                "Where",
                &["equal_out", "true_value", "false_value"],
                &["where_out"],
                Vec::new(),
            ),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    for (name, value) in [("a", 16_777_216_i64), ("b", 16_777_217_i64)] {
        // Both exact integers collapse to the same f32; only the authenticated
        // INT32 payload can distinguish them.
        weights.insert(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![value as f32]).unwrap(),
        );
        weights.insert_integers(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![value]).unwrap(),
        );
        weights.insert_integer_range(name.to_string(), i32::MIN as i64, i32::MAX as i64);
    }
    weights.insert(
        "true_value".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![9.0]).unwrap(),
    );
    weights.insert(
        "false_value".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![5.0]).unwrap(),
    );

    fold(&graph, &mut weights);

    assert_eq!(
        weights
            .get("equal_out")
            .expect("Equal should fold exactly")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![0.0]
    );
    assert_eq!(
        weights
            .get("less_out")
            .expect("Less should fold exactly")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1.0]
    );
    assert_eq!(
        weights
            .get("where_out")
            .expect("comparison-driven Where should fold")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![5.0]
    );
}

#[test]
fn integer_sidecar_without_dtype_range_never_falls_back_to_float_arithmetic() {
    let graph = GraphProto {
        node: vec![node("add", "Add", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    for (name, integer, mirror) in [
        ("a", 16_777_217_i64, 16_777_216.0_f32),
        ("b", -16_777_216_i64, -16_777_216.0_f32),
    ] {
        weights.insert(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![mirror]).unwrap(),
        );
        weights.insert_integers(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![integer]).unwrap(),
        );
    }

    fold(&graph, &mut weights);
    assert!(
        !weights.contains_key("out"),
        "unknown integer provenance must not use rounded f32 mirrors"
    );
}

#[test]
fn test_typed_integer_unary_and_pow_never_fall_back_to_float() {
    let graph = GraphProto {
        node: vec![
            node("neg", "Neg", &["large"], &["neg_out"], Vec::new()),
            node("abs", "Abs", &["minimum"], &["abs_out"], Vec::new()),
            node("pow", "Pow", &["large", "power"], &["pow_out"], Vec::new()),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    for (name, value) in [
        ("large", 16_777_217_i64),
        ("minimum", i64::MIN),
        ("power", 2_i64),
    ] {
        weights.insert(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![value as f32]).unwrap(),
        );
        weights.insert_integers(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![value]).unwrap(),
        );
        weights.insert_integer_range(name.to_string(), i64::MIN, i64::MAX);
    }

    fold(&graph, &mut weights);

    assert_eq!(
        weights
            .get_integers("neg_out")
            .expect("checked INT64 Neg should fold")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![-16_777_217]
    );
    assert!(
        !weights.contains_key("abs_out"),
        "i64::MIN Abs must fail closed"
    );
    assert!(
        !weights.contains_key("pow_out"),
        "typed INT64 Pow must not use f32"
    );
}

#[test]
fn test_add_constant_fold_same_shape() {
    let graph = GraphProto {
        node: vec![node("add", "Add", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap();
    weights.insert("a".to_string(), a);
    weights.insert("b".to_string(), b);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Add output");
    assert_eq!(out.shape(), &[3]);
    let expected = [5.0, 7.0, 9.0];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1.0e-6);
    }
}

#[test]
fn test_add_constant_fold_scalar_broadcast() {
    let graph = GraphProto {
        node: vec![node("add", "Add", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let a = ArrayD::from_shape_vec(IxDyn(&[1]), vec![10.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    weights.insert("a".to_string(), a);
    weights.insert("b".to_string(), b);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Add output");
    assert_eq!(out.shape(), &[3]);
    let expected = [11.0, 12.0, 13.0];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1.0e-6);
    }
}

#[test]
fn test_float_rankful_singleton_broadcast_preserves_rank_and_order() {
    let graph = GraphProto {
        node: vec![node("sub", "Sub", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert(
        "a".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![10.0]).unwrap(),
    );
    weights.insert(
        "b".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
    );

    fold(&graph, &mut weights);

    let output = weights.get("out").expect("Sub should fold");
    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(output.iter().copied().collect::<Vec<_>>(), vec![9.0, 8.0]);
}

#[test]
fn test_pow_rankful_singleton_exponent_broadcast_preserves_rank() {
    let graph = GraphProto {
        node: vec![node("pow", "Pow", &["base", "power"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert(
        "base".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 3.0]).unwrap(),
    );
    weights.insert(
        "power".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![2.0]).unwrap(),
    );

    fold(&graph, &mut weights);

    let output = weights.get("out").expect("Pow should fold");
    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(output.iter().copied().collect::<Vec<_>>(), vec![4.0, 9.0]);
}

#[test]
fn test_pow_constant_fold_preserves_ieee_signed_zero_rules() {
    let graph = GraphProto {
        node: vec![node("pow", "Pow", &["base", "power"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert(
        "base".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.0, -0.0, -0.0, 0.0]).unwrap(),
    );
    weights.insert(
        "power".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0, 3.0, 0.5, 3.0]).unwrap(),
    );

    fold(&graph, &mut weights);

    let output = weights.get("out").expect("exact zero powers should fold");
    let bits = output
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(bits, vec![0, (-0.0_f32).to_bits(), 0, 0]);
}

#[test]
fn test_elementwise_constant_fold_rejects_extra_inputs() {
    let mut custom_add = node(
        "custom_add",
        "Add",
        &["a", "b"],
        &["custom_out"],
        Vec::new(),
    );
    custom_add.domain = "vendor.example".to_string();
    let graph = GraphProto {
        node: vec![
            node("add", "Add", &["a", "b", "c"], &["add_out"], Vec::new()),
            node("neg", "Neg", &["a", "b"], &["neg_out"], Vec::new()),
            node(
                "where",
                "Where",
                &["a", "b", "c", "d"],
                &["where_out"],
                Vec::new(),
            ),
            node("empty", "Add", &["a", "b"], &[""], Vec::new()),
            custom_add,
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    for name in ["a", "b", "c", "d"] {
        weights.insert(
            name.to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap(),
        );
    }

    fold(&graph, &mut weights);

    for output in ["add_out", "neg_out", "where_out"] {
        assert!(
            !weights.contains_key(output),
            "malformed extra-input node must not publish {output}"
        );
    }
    assert!(
        !weights.contains_key(""),
        "a sole empty output name must never become a WeightStore key"
    );
    assert!(
        !weights.contains_key("custom_out"),
        "custom-domain lookalike must not use standard ONNX folding"
    );
}

#[test]
fn test_sub_constant_fold() {
    let graph = GraphProto {
        node: vec![node("sub", "Sub", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    weights.insert("a".to_string(), a);
    weights.insert("b".to_string(), b);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Sub output");
    let expected = [9.0, 18.0, 27.0];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1.0e-6);
    }
}

#[test]
fn test_neg_constant_fold() {
    let graph = GraphProto {
        node: vec![node("neg", "Neg", &["a"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, -2.0, 3.0]).unwrap();
    weights.insert("a".to_string(), a);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Neg output");
    let expected = [-1.0, 2.0, -3.0];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1.0e-6);
    }
}

#[test]
fn test_sin_cos_constant_fold_only_exact_special_values() {
    let graph = GraphProto {
        node: vec![
            node("sin_op", "Sin", &["x"], &["sin_out"], Vec::new()),
            node("cos_op", "Cos", &["x"], &["cos_out"], Vec::new()),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let x = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    weights.insert("x".to_string(), x);

    fold(&graph, &mut weights);

    let sin_out = weights.get("sin_out").expect("missing Sin output");
    assert_eq!(sin_out.as_slice().unwrap(), &[0.0]);

    let cos_out = weights.get("cos_out").expect("missing Cos output");
    assert_eq!(cos_out.as_slice().unwrap(), &[1.0]);

    let graph = GraphProto {
        node: vec![node("sin_op", "Sin", &["x"], &["sin_out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert(
        "x".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![std::f32::consts::FRAC_PI_2]).unwrap(),
    );
    fold(&graph, &mut weights);
    assert!(
        !weights.contains_key("sin_out"),
        "a rounded transcendental result must not become an exact constant"
    );
}

#[test]
fn test_inexact_float_matmul_is_not_constant_folded() {
    let graph = GraphProto {
        node: vec![node("mm", "MatMul", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert(
        "a".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.1]).unwrap(),
    );
    weights.insert(
        "b".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.1]).unwrap(),
    );
    fold(&graph, &mut weights);
    assert!(!weights.contains_key("out"));
}

#[test]
fn test_relu_constant_fold() {
    let graph = GraphProto {
        node: vec![node("relu", "Relu", &["x"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let x = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-2.0, -0.5, 0.0, 3.0]).unwrap();
    weights.insert("x".to_string(), x);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Relu output");
    let expected = [0.0, 0.0, 0.0, 3.0];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1.0e-6);
    }
}

#[test]
fn test_matmul_constant_fold_2d() {
    let graph = GraphProto {
        node: vec![node("mm", "MatMul", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    // 2x3 @ 3x2 = 2x2
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    weights.insert("a".to_string(), a);
    weights.insert("b".to_string(), b);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing MatMul output");
    assert_eq!(out.shape(), &[2, 2]);
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert!((out[[0, 0]] - 58.0).abs() < 1.0e-4);
    assert!((out[[0, 1]] - 64.0).abs() < 1.0e-4);
    assert!((out[[1, 0]] - 139.0).abs() < 1.0e-4);
    assert!((out[[1, 1]] - 154.0).abs() < 1.0e-4);
}

#[test]
fn test_matmul_constant_fold_1d_dot_product() {
    let graph = GraphProto {
        node: vec![node("mm", "MatMul", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap();
    weights.insert("a".to_string(), a);
    weights.insert("b".to_string(), b);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing MatMul output");
    assert_eq!(out.ndim(), 0, "1D @ 1D MatMul should fold to a scalar");
    let scalar = out.iter().copied().next().expect("scalar output");
    assert!((scalar - 32.0).abs() < 1.0e-4);
}

#[test]
fn test_equal_constant_fold() {
    let graph = GraphProto {
        node: vec![node("eq", "Equal", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 5.0, 3.0, 6.0]).unwrap();
    weights.insert("a".to_string(), a);
    weights.insert("b".to_string(), b);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Equal output");
    let expected = [1.0, 0.0, 1.0, 0.0];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1.0e-6);
    }
}

#[test]
fn test_float_equal_uses_exact_onnx_semantics() {
    let graph = GraphProto {
        node: vec![node("eq", "Equal", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let adjacent_to_half = f32::from_bits(0.5_f32.to_bits() + 1);
    assert!(
        (adjacent_to_half - 0.5).abs() < f32::EPSILON,
        "regression input must expose the former epsilon comparison"
    );
    let mut weights = WeightStore::new();
    weights.insert(
        "a".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.5, f32::NAN, 0.0, -0.0, 1.0]).unwrap(),
    );
    weights.insert(
        "b".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[5]),
            vec![adjacent_to_half, f32::NAN, -0.0, 0.0, 1.0],
        )
        .unwrap(),
    );

    fold(&graph, &mut weights);

    let output = weights.get("out").expect("Equal should fold");
    assert_eq!(
        output.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 0.0, 1.0, 1.0, 1.0],
        "Equal must distinguish adjacent floats, treat NaN as unequal, and ±0 as equal"
    );
}

#[test]
fn test_greater_constant_fold() {
    let graph = GraphProto {
        node: vec![node("gt", "Greater", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0, 1.0, 5.0, 3.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 1.0, 7.0, 0.0]).unwrap();
    weights.insert("a".to_string(), a);
    weights.insert("b".to_string(), b);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Greater output");
    let expected = [1.0, 0.0, 0.0, 1.0];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1.0e-6);
    }
}

#[test]
fn test_less_or_equal_constant_fold() {
    let graph = GraphProto {
        node: vec![node("le", "LessOrEqual", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0, 1.0, 5.0, 3.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 1.0, 7.0, 0.0]).unwrap();
    weights.insert("a".to_string(), a);
    weights.insert("b".to_string(), b);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing LessOrEqual output");
    let expected = [0.0, 1.0, 1.0, 0.0];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1.0e-6);
    }
}

#[test]
fn test_where_constant_fold() {
    let graph = GraphProto {
        node: vec![node(
            "wh",
            "Where",
            &["cond", "true_val", "false_val"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let cond = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 0.0, 1.0]).unwrap();
    let true_val = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let false_val = ArrayD::from_shape_vec(IxDyn(&[3]), vec![100.0, 200.0, 300.0]).unwrap();
    weights.insert("cond".to_string(), cond);
    weights.insert("true_val".to_string(), true_val);
    weights.insert("false_val".to_string(), false_val);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Where output");
    let expected = [10.0, 200.0, 30.0];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1.0e-6);
    }
}

// ==========================================================================
// Overflow regression tests (#3280)
//
// Verify that broadcast_binop rejects output shapes exceeding the
// MAX_BROADCAST_ELEMENTS cap (which relies on checked_shape_product).
// ==========================================================================

/// Regression test (#3280): broadcast_binop returns None when output
/// shape exceeds MAX_BROADCAST_ELEMENTS (10_000_000).
///
/// Creates two small arrays whose broadcast output shape would be
/// [10001, 10001] = 100,020,001 elements, exceeding the cap.
/// Without the checked_shape_product + cap guard, this could attempt
/// a >400 MB allocation.
#[ntest::timeout(5000)]
#[test]
fn test_broadcast_binop_overflow_cap_rejected_3280() {
    use super::super::broadcast::broadcast_binop;

    let a = ArrayD::<f32>::zeros(IxDyn(&[10001, 1]));
    let b = ArrayD::<f32>::zeros(IxDyn(&[1, 10001]));
    let result = broadcast_binop(&a, &b, |x, y| x + y);
    assert!(
        result.is_none(),
        "#3280: broadcast_binop must reject output exceeding MAX_BROADCAST_ELEMENTS"
    );
}

/// Regression test (#3280): broadcast_where returns None when output
/// shape exceeds MAX_BROADCAST_ELEMENTS.
#[ntest::timeout(5000)]
#[test]
fn test_broadcast_where_overflow_cap_rejected_3280() {
    use super::super::broadcast::broadcast_where;

    let cond = ArrayD::<f32>::zeros(IxDyn(&[10001, 1]));
    let true_val = ArrayD::<f32>::zeros(IxDyn(&[1, 10001]));
    let false_val = ArrayD::<f32>::zeros(IxDyn(&[1, 1]));
    let result = broadcast_where(&cond, &true_val, &false_val);
    assert!(
        result.is_none(),
        "#3280: broadcast_where must reject output exceeding MAX_BROADCAST_ELEMENTS"
    );
}

/// Regression test (#3280): checked_shape_product rejects overflowing products.
///
/// Direct test of the guard function used by all loader overflow paths.
#[ntest::timeout(5000)]
#[test]
fn test_checked_shape_product_overflow_3280() {
    use ny_core::checked_shape_product;

    // (2^32) * (2^32) = 2^64 > usize::MAX on 64-bit
    assert_eq!(checked_shape_product(&[1usize << 32, 1usize << 32]), None);
    // usize::MAX * 2 overflows
    assert_eq!(checked_shape_product(&[usize::MAX, 2]), None);
    // Normal shapes work
    assert_eq!(checked_shape_product(&[3, 4, 5]), Some(60));
    // Empty shape = 1 element (scalar)
    assert_eq!(checked_shape_product(&[]), Some(1));
}
