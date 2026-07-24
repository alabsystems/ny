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
fn test_sin_cos_constant_fold() {
    let graph = GraphProto {
        node: vec![
            node("sin_op", "Sin", &["x"], &["sin_out"], Vec::new()),
            node("cos_op", "Cos", &["x"], &["cos_out"], Vec::new()),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let x = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, std::f32::consts::FRAC_PI_2]).unwrap();
    weights.insert("x".to_string(), x);

    fold(&graph, &mut weights);

    let sin_out = weights.get("sin_out").expect("missing Sin output");
    assert!((sin_out[[0]] - 0.0).abs() < 1.0e-6);
    assert!((sin_out[[1]] - 1.0).abs() < 1.0e-6);

    let cos_out = weights.get("cos_out").expect("missing Cos output");
    assert!((cos_out[[0]] - 1.0).abs() < 1.0e-6);
    assert!((cos_out[[1]] - 0.0).abs() < 1.0e-5);
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

    let a = ArrayD::zeros(IxDyn(&[10001, 1]));
    let b = ArrayD::zeros(IxDyn(&[1, 10001]));
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

    let cond = ArrayD::zeros(IxDyn(&[10001, 1]));
    let true_val = ArrayD::zeros(IxDyn(&[1, 10001]));
    let false_val = ArrayD::zeros(IxDyn(&[1, 1]));
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
