// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for MatMul and Add layer IBP propagation.
//!
//! This module tests bounded matrix multiplication and element-wise addition:
//! - Concrete values (tight bounds)
//! - Transpose modes
//! - Scaling (attention-style)
//! - Interval soundness with mixed signs
//! - Shape validation

use super::*;
use ndarray::{arr1, arr2, Array2};

// ============================================================
// BOUNDED MATMUL TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_bounded_matmul_concrete() {
    // Test with concrete (point) values: C = A @ B
    // A = [[1, 2], [3, 4]] (2x2), B = [[5, 6], [7, 8]] (2x2)
    // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //   = [[19, 22], [43, 50]]
    let a = BoundedTensor::new(
        arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]).into_dyn(),
        arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        arr2(&[[5.0_f32, 6.0], [7.0, 8.0]]).into_dyn(),
        arr2(&[[5.0_f32, 6.0], [7.0, 8.0]]).into_dyn(),
    )
    .unwrap();

    let matmul = MatMulLayer::new(false, None);
    let c = matmul.propagate_ibp_binary(&a, &b).unwrap();

    // Concrete result: bounds should be tight
    assert!((c.lower()[[0, 0]] - 19.0).abs() < 1e-4);
    assert!((c.upper()[[0, 0]] - 19.0).abs() < 1e-4);
    assert!((c.lower()[[0, 1]] - 22.0).abs() < 1e-4);
    assert!((c.upper()[[0, 1]] - 22.0).abs() < 1e-4);
    assert!((c.lower()[[1, 0]] - 43.0).abs() < 1e-4);
    assert!((c.upper()[[1, 0]] - 43.0).abs() < 1e-4);
    assert!((c.lower()[[1, 1]] - 50.0).abs() < 1e-4);
    assert!((c.upper()[[1, 1]] - 50.0).abs() < 1e-4);
}

#[ntest::timeout(10000)]
#[test]
fn test_bounded_matmul_transpose_b() {
    // Test A @ B^T
    // A = [[1, 2]] (1x2), B = [[3, 4]] (1x2)
    // A @ B^T = [[1*3 + 2*4]] = [[11]] (1x1)
    let a = BoundedTensor::new(
        arr2(&[[1.0_f32, 2.0]]).into_dyn(),
        arr2(&[[1.0_f32, 2.0]]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        arr2(&[[3.0_f32, 4.0]]).into_dyn(),
        arr2(&[[3.0_f32, 4.0]]).into_dyn(),
    )
    .unwrap();

    let matmul = MatMulLayer::new(true, None);
    let c = matmul.propagate_ibp_binary(&a, &b).unwrap();

    assert_eq!(c.shape(), &[1, 1]);
    assert!((c.lower()[[0, 0]] - 11.0).abs() < 1e-4);
    assert!((c.upper()[[0, 0]] - 11.0).abs() < 1e-4);
}

#[ntest::timeout(10000)]
#[test]
fn test_bounded_matmul_with_scale() {
    // Test with scaling (like attention)
    // A @ B * scale where scale = 0.5
    let a = BoundedTensor::new(
        arr2(&[[2.0_f32, 4.0]]).into_dyn(),
        arr2(&[[2.0_f32, 4.0]]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        arr2(&[[3.0_f32], [5.0]]).into_dyn(),
        arr2(&[[3.0_f32], [5.0]]).into_dyn(),
    )
    .unwrap();

    // A @ B = [[2*3 + 4*5]] = [[26]]
    // With scale 0.5: [[13]]
    let matmul = MatMulLayer::new(false, Some(0.5));
    let c = matmul.propagate_ibp_binary(&a, &b).unwrap();

    assert!((c.lower()[[0, 0]] - 13.0).abs() < 1e-4);
    assert!((c.upper()[[0, 0]] - 13.0).abs() < 1e-4);
}

#[ntest::timeout(10000)]
#[test]
fn test_bounded_matmul_interval_soundness() {
    // Test with interval inputs: verify bounds are sound
    // A in [[0, 1]] x [[0, 1]] (2x2, each element in [0, 1])
    // B in [[0, 1]] x [[0, 1]]
    let a = BoundedTensor::new(
        arr2(&[[0.0_f32, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[1.0_f32, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        arr2(&[[0.0_f32, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[1.0_f32, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let matmul = MatMulLayer::new(false, None);
    let c = matmul.propagate_ibp_binary(&a, &b).unwrap();

    // Each element of C = sum of 2 products
    // Each product: [0,1] * [0,1] = [0,1]
    // Sum of 2: [0,2]
    assert!(
        c.lower()[[0, 0]] >= -1e-5,
        "lower bound too low: {}",
        c.lower()[[0, 0]]
    );
    assert!(
        c.upper()[[0, 0]] <= 2.0 + 1e-5,
        "upper bound too high: {}",
        c.upper()[[0, 0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_bounded_matmul_mixed_signs() {
    // Test with mixed positive/negative values
    // Both A and B have elements in [-1, 1]
    let a = BoundedTensor::new(
        arr2(&[[-1.0_f32, -1.0], [-1.0, -1.0]]).into_dyn(),
        arr2(&[[1.0_f32, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        arr2(&[[-1.0_f32, -1.0], [-1.0, -1.0]]).into_dyn(),
        arr2(&[[1.0_f32, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let matmul = MatMulLayer::new(false, None);
    let c = matmul.propagate_ibp_binary(&a, &b).unwrap();

    // Each element: sum of 2 products of [-1,1]*[-1,1]
    // Each product: [-1, 1]
    // Sum of 2: [-2, 2]
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                c.lower()[[i, j]] >= -2.0 - 1e-5,
                "lower[{},{}] = {} too low",
                i,
                j,
                c.lower()[[i, j]]
            );
            assert!(
                c.upper()[[i, j]] <= 2.0 + 1e-5,
                "upper[{},{}] = {} too high",
                i,
                j,
                c.upper()[[i, j]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_bounded_matmul_economic_bounds_looser() {
    let a = BoundedTensor::new(
        arr2(&[[-1.0_f32, 0.2], [0.1, -0.7]]).into_dyn(),
        arr2(&[[0.9_f32, 1.3], [0.4, 0.8]]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        arr2(&[[-0.6_f32, 0.1], [-0.2, -0.4]]).into_dyn(),
        arr2(&[[1.1_f32, 0.9], [0.5, 0.6]]).into_dyn(),
    )
    .unwrap();

    let economic = MatMulLayer::new_with_ibp_mode(false, None, MatMulIbpMode::Economic)
        .propagate_ibp_binary(&a, &b)
        .unwrap();

    let a_lower = a
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let a_upper = a
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let b_lower = b
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let b_upper = b
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

    for mask in 0..(1u32 << 8) {
        let mut a_vals = Vec::with_capacity(4);
        let mut b_vals = Vec::with_capacity(4);
        for idx in 0..4 {
            let bit = (mask >> idx) & 1;
            let value = if bit == 0 {
                a_lower.as_slice().unwrap()[idx]
            } else {
                a_upper.as_slice().unwrap()[idx]
            };
            a_vals.push(value);
        }
        for idx in 0..4 {
            let bit = (mask >> (idx + 4)) & 1;
            let value = if bit == 0 {
                b_lower.as_slice().unwrap()[idx]
            } else {
                b_upper.as_slice().unwrap()[idx]
            };
            b_vals.push(value);
        }

        let a_point = Array2::from_shape_vec((2, 2), a_vals).unwrap();
        let b_point = Array2::from_shape_vec((2, 2), b_vals).unwrap();
        let c_point = a_point.dot(&b_point);
        for i in 0..2 {
            for j in 0..2 {
                let v = c_point[[i, j]];
                assert!(
                    v >= economic.lower()[[i, j]] - 1e-5,
                    "economic lower violated at ({},{}): {} < {}",
                    i,
                    j,
                    v,
                    economic.lower()[[i, j]]
                );
                assert!(
                    v <= economic.upper()[[i, j]] + 1e-5,
                    "economic upper violated at ({},{}): {} > {}",
                    i,
                    j,
                    v,
                    economic.upper()[[i, j]]
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_bounded_matmul_economic_falls_back_with_concrete_operand() {
    let a = BoundedTensor::new(
        arr2(&[[-0.5_f32, 0.5], [0.1, -0.1]]).into_dyn(),
        arr2(&[[0.5_f32, 1.0], [0.2, 0.3]]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]).into_dyn(),
        arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]).into_dyn(),
    )
    .unwrap();

    let standard = MatMulLayer::new(false, None)
        .propagate_ibp_binary(&a, &b)
        .unwrap();
    let economic = MatMulLayer::new_with_ibp_mode(false, None, MatMulIbpMode::Economic)
        .propagate_ibp_binary(&a, &b)
        .unwrap();

    assert_eq!(economic.lower(), standard.lower());
    assert_eq!(economic.upper(), standard.upper());
}

// ============================================================
// BOUNDED ADD TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_bounded_add_concrete() {
    // Test element-wise addition with concrete values
    let a = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0, 3.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0, 3.0]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        arr1(&[4.0_f32, 5.0, 6.0]).into_dyn(),
        arr1(&[4.0_f32, 5.0, 6.0]).into_dyn(),
    )
    .unwrap();

    let add = AddLayer;
    let c = add.propagate_ibp_binary(&a, &b).unwrap();

    assert!((c.lower()[[0]] - 5.0).abs() < 1e-5);
    assert!((c.lower()[[1]] - 7.0).abs() < 1e-5);
    assert!((c.lower()[[2]] - 9.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_bounded_add_interval() {
    // Test element-wise addition with interval bounds
    // A in [[0, 1], [0, 1]], B in [[2, 3], [2, 3]]
    // C should be in [[2, 4], [2, 4]]
    let a = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        arr1(&[2.0_f32, 2.0]).into_dyn(),
        arr1(&[3.0_f32, 3.0]).into_dyn(),
    )
    .unwrap();

    let add = AddLayer;
    let c = add.propagate_ibp_binary(&a, &b).unwrap();

    assert!((c.lower()[[0]] - 2.0).abs() < 1e-5);
    assert!((c.upper()[[0]] - 4.0).abs() < 1e-5);
    assert!((c.lower()[[1]] - 2.0).abs() < 1e-5);
    assert!((c.upper()[[1]] - 4.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_bounded_add_shape_mismatch() {
    // Test that shape mismatch returns error
    let a = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0, 3.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0, 3.0]).into_dyn(),
    )
    .unwrap();

    let add = AddLayer;
    let result = add.propagate_ibp_binary(&a, &b);

    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_is_binary() {
    assert!(Layer::MatMul(MatMulLayer::new(false, None)).is_binary());
    assert!(Layer::Add(AddLayer).is_binary());
    assert!(!Layer::ReLU(ReLULayer).is_binary());
    assert!(!Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()).is_binary());
}
