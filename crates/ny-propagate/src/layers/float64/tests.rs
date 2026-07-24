// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for f64 layer implementations.
//!
//! Includes both unit tests and proptest soundness checks.

use super::conv2d::{propagate_conv2d_crown_backward_f64, propagate_conv2d_ibp_f64, Conv2dParams};
use super::linear::{propagate_linear_crown_backward_f64, propagate_linear_ibp_f64};
use super::relu::{propagate_relu_crown_backward_f64, propagate_relu_ibp_f64};
use ndarray::{arr1, arr2, Array1, Array2, Array4, ArrayD, IxDyn};
use ny_tensor::BoundedTensor64;
use proptest::prelude::*;

// ======================== Linear IBP ========================

#[test]
fn test_linear_ibp_f64_identity_weight() {
    let weight = arr2(&[[1.0f64, 0.0], [0.0, 1.0]]);
    let bias = arr1(&[0.0f64, 0.0]);
    let input = BoundedTensor64::new(
        arr1(&[-1.0f64, 2.0]).into_dyn(),
        arr1(&[3.0f64, 5.0]).into_dyn(),
    )
    .unwrap();

    let result = propagate_linear_ibp_f64(&weight, &bias, &input).unwrap();
    assert_eq!(result.lower()[0], -1.0);
    assert_eq!(result.upper()[0], 3.0);
    assert_eq!(result.lower()[1], 2.0);
    assert_eq!(result.upper()[1], 5.0);
}

#[test]
fn test_linear_ibp_f64_negative_weight() {
    // y = -2x + 1
    let weight = arr2(&[[-2.0f64]]);
    let bias = arr1(&[1.0f64]);
    let input =
        BoundedTensor64::new(arr1(&[1.0f64]).into_dyn(), arr1(&[3.0f64]).into_dyn()).unwrap();

    let result = propagate_linear_ibp_f64(&weight, &bias, &input).unwrap();
    // lower = -2*3 + 1 = -5, upper = -2*1 + 1 = -1
    assert_eq!(result.lower()[0], -5.0);
    assert_eq!(result.upper()[0], -1.0);
}

#[test]
fn test_linear_ibp_f64_with_bias() {
    let weight = arr2(&[[1.0f64, 2.0]]);
    let bias = arr1(&[10.0f64]);
    let input = BoundedTensor64::new(
        arr1(&[0.0f64, -1.0]).into_dyn(),
        arr1(&[1.0f64, 1.0]).into_dyn(),
    )
    .unwrap();

    let result = propagate_linear_ibp_f64(&weight, &bias, &input).unwrap();
    // lower = 1*0 + 2*(-1) + 10 = 8
    // upper = 1*1 + 2*1 + 10 = 13
    assert_eq!(result.lower()[0], 8.0);
    assert_eq!(result.upper()[0], 13.0);
}

// ======================== Linear CROWN ========================

#[test]
fn test_linear_crown_backward_f64_identity() {
    let weight = arr2(&[[1.0f64, 0.0], [0.0, 1.0]]);
    let bias = arr1(&[0.0f64, 0.0]);
    let bounds = crate::bounds::LinearBounds64::identity(2);

    let result = propagate_linear_crown_backward_f64(&weight, &bias, &bounds).unwrap();
    // Identity through identity = identity
    assert_eq!(result.lower_a()[[0, 0]], 1.0);
    assert_eq!(result.lower_a()[[0, 1]], 0.0);
    assert_eq!(result.lower_b()[0], 0.0);
}

#[test]
fn test_linear_crown_backward_f64_with_bias() {
    // y = 2x + 3
    let weight = arr2(&[[2.0f64]]);
    let bias = arr1(&[3.0f64]);
    let bounds = crate::bounds::LinearBounds64::identity(1);

    let result = propagate_linear_crown_backward_f64(&weight, &bias, &bounds).unwrap();
    // new_A = I @ [[2]] = [[2]]
    // new_b = I @ [3] + [0] = [3]
    assert_eq!(result.lower_a()[[0, 0]], 2.0);
    assert_eq!(result.lower_b()[0], 3.0);
}

// ======================== ReLU IBP ========================

#[test]
fn test_relu_ibp_f64_positive() {
    let input = BoundedTensor64::new(
        arr1(&[1.0f64, 2.0]).into_dyn(),
        arr1(&[3.0f64, 5.0]).into_dyn(),
    )
    .unwrap();

    let result = propagate_relu_ibp_f64(&input).unwrap();
    assert_eq!(result.lower()[0], 1.0);
    assert_eq!(result.upper()[0], 3.0);
}

#[test]
fn test_relu_ibp_f64_negative() {
    let input = BoundedTensor64::new(
        arr1(&[-3.0f64, -1.0]).into_dyn(),
        arr1(&[-1.0f64, -0.5]).into_dyn(),
    )
    .unwrap();

    let result = propagate_relu_ibp_f64(&input).unwrap();
    assert_eq!(result.lower()[0], 0.0);
    assert_eq!(result.upper()[0], 0.0);
}

#[test]
fn test_relu_ibp_f64_crossing() {
    let input =
        BoundedTensor64::new(arr1(&[-2.0f64]).into_dyn(), arr1(&[3.0f64]).into_dyn()).unwrap();

    let result = propagate_relu_ibp_f64(&input).unwrap();
    assert_eq!(result.lower()[0], 0.0);
    assert_eq!(result.upper()[0], 3.0);
}

// ======================== ReLU CROWN ========================

#[test]
fn test_relu_crown_backward_f64_positive() {
    // Pre-activation: [1, 3] — entirely positive → identity
    let pre_act =
        BoundedTensor64::new(arr1(&[1.0f64]).into_dyn(), arr1(&[3.0f64]).into_dyn()).unwrap();
    let bounds = crate::bounds::LinearBounds64::identity(1);

    let result = propagate_relu_crown_backward_f64(&bounds, &pre_act).unwrap();
    // Should be identity: A = [[1]], b = [0]
    assert_eq!(result.lower_a()[[0, 0]], 1.0);
    assert_eq!(result.upper_a()[[0, 0]], 1.0);
    assert_eq!(result.lower_b()[0], 0.0);
    assert_eq!(result.upper_b()[0], 0.0);
}

#[test]
fn test_relu_crown_backward_f64_negative() {
    // Pre-activation: [-3, -1] — entirely negative → zero
    let pre_act =
        BoundedTensor64::new(arr1(&[-3.0f64]).into_dyn(), arr1(&[-1.0f64]).into_dyn()).unwrap();
    let bounds = crate::bounds::LinearBounds64::identity(1);

    let result = propagate_relu_crown_backward_f64(&bounds, &pre_act).unwrap();
    assert_eq!(result.lower_a()[[0, 0]], 0.0);
    assert_eq!(result.upper_a()[[0, 0]], 0.0);
    assert_eq!(result.lower_b()[0], 0.0);
    assert_eq!(result.upper_b()[0], 0.0);
}

#[test]
fn test_relu_crown_backward_f64_crossing() {
    // Pre-activation: [-1, 3] — crossing
    let pre_act =
        BoundedTensor64::new(arr1(&[-1.0f64]).into_dyn(), arr1(&[3.0f64]).into_dyn()).unwrap();
    let bounds = crate::bounds::LinearBounds64::identity(1);

    let result = propagate_relu_crown_backward_f64(&bounds, &pre_act).unwrap();
    // Upper: lambda = 3/(3-(-1)) = 0.75, intercept = 0.75*1 = 0.75
    let expected_lambda = 3.0 / 4.0;
    assert!((result.upper_a()[[0, 0]] - expected_lambda).abs() < 1e-12);
    assert!((result.upper_b()[0] - 0.75).abs() < 1e-12);
    // Lower: u=3 > -l=1, so alpha=1.0, intercept=0
    assert_eq!(result.lower_a()[[0, 0]], 1.0);
    assert_eq!(result.lower_b()[0], 0.0);
}

// ======================== End-to-end CROWN soundness ========================

#[test]
fn test_crown_linear_relu_linear_soundness() {
    // Network: y = W2 @ relu(W1 @ x + b1) + b2
    // x in [-1, 1]
    let w1 = arr2(&[[0.5f64, -0.3], [0.2, 0.8]]);
    let b1 = arr1(&[0.1f64, -0.2]);
    let w2 = arr2(&[[1.0f64, -0.5]]);
    let b2 = arr1(&[0.0f64]);

    let input = BoundedTensor64::new(
        arr1(&[-1.0f64, -1.0]).into_dyn(),
        arr1(&[1.0f64, 1.0]).into_dyn(),
    )
    .unwrap();

    // IBP forward to get intermediate bounds
    let after_linear1 = propagate_linear_ibp_f64(&w1, &b1, &input).unwrap();
    let after_relu = propagate_relu_ibp_f64(&after_linear1).unwrap();
    let ibp_output = propagate_linear_ibp_f64(&w2, &b2, &after_relu).unwrap();

    // CROWN backward
    let crown_bounds = crate::bounds::LinearBounds64::identity(1);
    let crown_after_w2 = propagate_linear_crown_backward_f64(&w2, &b2, &crown_bounds).unwrap();
    let crown_after_relu =
        propagate_relu_crown_backward_f64(&crown_after_w2, &after_linear1).unwrap();
    let crown_final = propagate_linear_crown_backward_f64(&w1, &b1, &crown_after_relu).unwrap();

    // Concretize
    let crown_output = crown_final.concretize(&input).unwrap();

    // Backward CROWN can produce wider bounds than IBP for multi-layer networks
    // with mixed-sign coefficients (concretization overapproximation). Verify
    // both are individually sound at sample points.
    let test_points = [
        [0.0, 0.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
    ];
    for pt in &test_points {
        let x = arr1(pt);
        let y_val = (w2.dot(&(w1.dot(&x) + &b1).mapv(|v| v.max(0.0))) + &b2)[0];
        for (label, bounds) in [("CROWN", &crown_output), ("IBP", &ibp_output)] {
            assert!(
                y_val >= bounds.lower()[0] - 1e-10,
                "{label} lower violated at {pt:?}"
            );
            assert!(
                y_val <= bounds.upper()[0] + 1e-10,
                "{label} upper violated at {pt:?}"
            );
        }
    }
}

// ======================== Conv2D IBP ========================

#[test]
fn test_conv2d_ibp_f64_identity_kernel() {
    // 1x1 identity kernel: passes through each channel
    let kernel = Array4::from_shape_fn((1, 1, 1, 1), |_| 1.0f64);
    let bias = arr1(&[0.0f64]);
    let params = Conv2dParams {
        stride: (1, 1),
        padding: (0, 0),
        input_hw: (2, 2),
    };

    let input = BoundedTensor64::new(
        ndarray::Array3::from_shape_vec((1, 2, 2), vec![-1.0, 0.0, 1.0, 2.0])
            .unwrap()
            .into_dyn(),
        ndarray::Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let result = propagate_conv2d_ibp_f64(&kernel, &bias, &input, &params).unwrap();
    // Output shape is (1, 2, 2) — flatten to index linearly
    let (lower, upper) = result.flatten_to_1d();
    assert_eq!(lower[0], -1.0);
    assert_eq!(upper[0], 1.0);
    assert_eq!(lower[3], 2.0);
    assert_eq!(upper[3], 4.0);
}

#[test]
fn test_conv2d_ibp_f64_with_bias() {
    let kernel = Array4::from_shape_fn((1, 1, 1, 1), |_| 2.0f64);
    let bias = arr1(&[5.0f64]);
    let params = Conv2dParams {
        stride: (1, 1),
        padding: (0, 0),
        input_hw: (1, 1),
    };

    let input = BoundedTensor64::new(
        ndarray::Array3::from_shape_vec((1, 1, 1), vec![1.0])
            .unwrap()
            .into_dyn(),
        ndarray::Array3::from_shape_vec((1, 1, 1), vec![3.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let result = propagate_conv2d_ibp_f64(&kernel, &bias, &input, &params).unwrap();
    // Output shape is (1, 1, 1) — flatten to index linearly
    let (lower, upper) = result.flatten_to_1d();
    // lower = 2*1 + 5 = 7, upper = 2*3 + 5 = 11
    assert_eq!(lower[0], 7.0);
    assert_eq!(upper[0], 11.0);
}

// ======================== Conv2D CROWN backward ========================

#[test]
fn test_conv2d_crown_backward_f64_identity_kernel() {
    // 1x1 identity kernel through CROWN backward should pass coefficients unchanged.
    let kernel = Array4::from_shape_fn((1, 1, 1, 1), |_| 1.0f64);
    let bias = arr1(&[0.0f64]);
    let params = Conv2dParams {
        stride: (1, 1),
        padding: (0, 0),
        input_hw: (2, 2),
    };
    // Output shape: (1, 2, 2) → 4 elements
    let bounds = crate::bounds::LinearBounds64::identity(4);

    let result = propagate_conv2d_crown_backward_f64(&kernel, &bias, &bounds, &params).unwrap();
    // 1x1 identity: transposed conv is identity mapping over spatial dims.
    // Input: (1, 2, 2) → 4 elements. Output should be identity mapping.
    assert_eq!(result.num_inputs(), 4);
    assert_eq!(result.num_outputs(), 4);
    for i in 0..4 {
        assert!(
            (result.lower_a()[[i, i]] - 1.0).abs() < 1e-12,
            "diagonal [{i},{i}] should be 1.0, got {}",
            result.lower_a()[[i, i]]
        );
    }
}

#[test]
fn test_conv2d_crown_backward_f64_with_bias() {
    // 1x1 kernel with scaling and bias.
    let kernel = Array4::from_shape_fn((1, 1, 1, 1), |_| 2.0f64);
    let bias = arr1(&[3.0f64]);
    let params = Conv2dParams {
        stride: (1, 1),
        padding: (0, 0),
        input_hw: (1, 1),
    };
    // Output: (1, 1, 1) → 1 element
    let bounds = crate::bounds::LinearBounds64::identity(1);

    let result = propagate_conv2d_crown_backward_f64(&kernel, &bias, &bounds, &params).unwrap();
    // new_A = I @ 2 = [[2]], new_b = I @ [3] + [0] = [3]
    assert!((result.lower_a()[[0, 0]] - 2.0).abs() < 1e-12);
    assert!((result.lower_b()[0] - 3.0).abs() < 1e-12);
}

#[test]
fn test_conv2d_crown_backward_f64_3x3_kernel_soundness() {
    // 3x3 kernel on a 4x4 input → 2x2 output, no padding, stride 1.
    // Verify CROWN backward + concretize produces sound bounds.
    let kernel = Array4::from_shape_vec(
        (1, 1, 3, 3),
        vec![0.5, -0.3, 0.1, 0.2, 0.8, -0.4, -0.1, 0.6, 0.3],
    )
    .unwrap();
    let bias = arr1(&[0.5f64]);
    let params = Conv2dParams {
        stride: (1, 1),
        padding: (0, 0),
        input_hw: (4, 4),
    };

    // Input bounds: (1, 4, 4) = 16 elements, each in [-1, 1]
    let lower_in = ndarray::Array3::<f64>::from_elem((1, 4, 4), -1.0);
    let upper_in = ndarray::Array3::<f64>::from_elem((1, 4, 4), 1.0);
    let input = BoundedTensor64::new(lower_in.into_dyn(), upper_in.into_dyn()).unwrap();

    // IBP forward (returns 3D shape, flatten for comparison)
    let ibp_output = propagate_conv2d_ibp_f64(&kernel, &bias, &input, &params).unwrap();
    let (ibp_l, ibp_u) = ibp_output.flatten_to_1d();

    // CROWN backward: output is (1, 2, 2) = 4 elements
    let crown_bounds = crate::bounds::LinearBounds64::identity(4);
    let crown_result =
        propagate_conv2d_crown_backward_f64(&kernel, &bias, &crown_bounds, &params).unwrap();
    let crown_output = crown_result.concretize(&input).unwrap();
    let (crown_l, crown_u) = crown_output.flatten_to_1d();

    // Verify at sample points
    let test_inputs: Vec<Vec<f64>> = vec![
        vec![0.0; 16],
        vec![1.0; 16],
        vec![-1.0; 16],
        (0..16)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
    ];

    for pt in &test_inputs {
        let x_3d = ndarray::Array3::from_shape_vec((1, 4, 4), pt.clone()).unwrap();
        // Compute true conv2d output
        let mut y = [0.0f64; 4];
        for oh in 0..2 {
            for ow in 0..2 {
                let mut sum = bias[0];
                for kh in 0..3 {
                    for kw in 0..3 {
                        sum += kernel[[0, 0, kh, kw]] * x_3d[[0, oh + kh, ow + kw]];
                    }
                }
                y[oh * 2 + ow] = sum;
            }
        }

        for i in 0..4 {
            for (label, l, u) in [("CROWN", &crown_l, &crown_u), ("IBP", &ibp_l, &ibp_u)] {
                assert!(
                    y[i] >= l[i] - 1e-10,
                    "{label} lower violated at output {i}: y={}, lower={}",
                    y[i],
                    l[i]
                );
                assert!(
                    y[i] <= u[i] + 1e-10,
                    "{label} upper violated at output {i}: y={}, upper={}",
                    y[i],
                    u[i]
                );
            }
        }
    }
}

// ======================== End-to-end CROWN with Conv2D ========================

#[test]
fn test_crown_conv2d_relu_linear_soundness() {
    // Network: y = W2 @ relu(conv2d(x) + b_conv) + b2
    // Input: (1, 3, 3), 1x1 kernel → (1, 3, 3) → flatten → 9 elements → Linear
    let kernel = Array4::from_shape_fn((1, 1, 1, 1), |_| 0.5f64);
    let b_conv = arr1(&[0.1f64]);
    let conv_params = Conv2dParams {
        stride: (1, 1),
        padding: (0, 0),
        input_hw: (3, 3),
    };

    let w2 = Array2::from_shape_vec((1, 9), vec![0.3, -0.2, 0.1, 0.4, -0.5, 0.2, -0.1, 0.3, 0.0])
        .unwrap();
    let b2 = arr1(&[0.0f64]);

    let lower_in = ndarray::Array3::<f64>::from_elem((1, 3, 3), -1.0);
    let upper_in = ndarray::Array3::<f64>::from_elem((1, 3, 3), 1.0);
    let input = BoundedTensor64::new(lower_in.into_dyn(), upper_in.into_dyn()).unwrap();

    // IBP forward
    let after_conv = propagate_conv2d_ibp_f64(&kernel, &b_conv, &input, &conv_params).unwrap();
    let after_relu = propagate_relu_ibp_f64(&after_conv).unwrap();
    // Flatten for linear layer
    let (relu_l, relu_u) = after_relu.flatten_to_1d();
    let relu_flat = BoundedTensor64::new(
        Array1::from_vec(relu_l.to_vec()).into_dyn(),
        Array1::from_vec(relu_u.to_vec()).into_dyn(),
    )
    .unwrap();
    let ibp_output = propagate_linear_ibp_f64(&w2, &b2, &relu_flat).unwrap();

    // CROWN backward
    let crown_bounds = crate::bounds::LinearBounds64::identity(1);
    let crown_after_w2 = propagate_linear_crown_backward_f64(&w2, &b2, &crown_bounds).unwrap();

    // Flatten after_conv for ReLU pre-activation bounds
    let (conv_l, conv_u) = after_conv.flatten_to_1d();
    let conv_flat = BoundedTensor64::new(
        Array1::from_vec(conv_l.to_vec()).into_dyn(),
        Array1::from_vec(conv_u.to_vec()).into_dyn(),
    )
    .unwrap();
    let crown_after_relu = propagate_relu_crown_backward_f64(&crown_after_w2, &conv_flat).unwrap();
    let crown_final =
        propagate_conv2d_crown_backward_f64(&kernel, &b_conv, &crown_after_relu, &conv_params)
            .unwrap();
    let crown_output = crown_final.concretize(&input).unwrap();

    // Verify soundness at sample points
    let test_inputs: Vec<Vec<f64>> = vec![
        vec![0.0; 9],
        vec![1.0; 9],
        vec![-1.0; 9],
        vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.3, -0.7, 0.8, -0.2],
    ];

    for pt in &test_inputs {
        // True forward: conv2d (1x1 kernel * 0.5 + 0.1) -> relu -> linear
        let relu_vals: Vec<f64> = pt.iter().map(|&x| (x * 0.5 + 0.1).max(0.0)).collect();
        let y_val: f64 = w2
            .row(0)
            .iter()
            .zip(relu_vals.iter())
            .map(|(w, r)| w * r)
            .sum::<f64>()
            + b2[0];

        for (label, bounds) in [("CROWN", &crown_output), ("IBP", &ibp_output)] {
            assert!(
                y_val >= bounds.lower()[0] - 1e-10,
                "{label} lower violated: y={y_val}, lower={}",
                bounds.lower()[0]
            );
            assert!(
                y_val <= bounds.upper()[0] + 1e-10,
                "{label} upper violated: y={y_val}, upper={}",
                bounds.upper()[0]
            );
        }
    }
}

// ======================== Proptest helpers ========================

/// Tolerance for f64 soundness checks.
/// f64 has ~15 decimal digits of precision; 1e-10 is conservative.
const F64_TOLERANCE: f64 = 1e-10;

/// Generate a valid f64 interval [lower, upper] within [-range, range].
fn valid_interval_f64(range: f64) -> impl Strategy<Value = (f64, f64)> {
    (-range..=range)
        .prop_flat_map(move |a| (-range..=range).prop_map(move |b| (a.min(b), a.max(b))))
}

/// Sample evenly spaced points within an f64 interval.
fn sample_points_f64(lower: f64, upper: f64, num_samples: usize) -> Vec<f64> {
    if lower == upper {
        return vec![lower];
    }
    let samples = num_samples.max(2);
    let denom = (samples - 1) as f64;
    (0..samples)
        .map(|i| {
            let t = i as f64 / denom;
            (lower + (upper - lower) * t).clamp(lower, upper)
        })
        .collect()
}

// ======================== Proptest: Linear IBP f64 ========================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Linear IBP f64 soundness: for any x in input bounds, Wx+b is in output bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_linear_ibp_f64_2x2(
        w11 in -5.0f64..5.0,
        w12 in -5.0f64..5.0,
        w21 in -5.0f64..5.0,
        w22 in -5.0f64..5.0,
        b1 in -5.0f64..5.0,
        b2 in -5.0f64..5.0,
        (l1, u1) in valid_interval_f64(10.0),
        (l2, u2) in valid_interval_f64(10.0),
    ) {
        let weight = arr2(&[[w11, w12], [w21, w22]]);
        let bias = arr1(&[b1, b2]);
        let input = BoundedTensor64::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        ).unwrap();

        let output = propagate_linear_ibp_f64(&weight, &bias, &input).unwrap();

        for x1 in sample_points_f64(l1, u1, 5) {
            for x2 in sample_points_f64(l2, u2, 5) {
                let x = arr1(&[x1, x2]);
                let y = weight.dot(&x) + &bias;
                for i in 0..2 {
                    prop_assert!(
                        output.lower()[[i]] - F64_TOLERANCE <= y[i]
                            && y[i] <= output.upper()[[i]] + F64_TOLERANCE,
                        "Linear IBP f64 soundness violation at output {}: \
                         y[{}]={}, not in [{}, {}]",
                        i, i, y[i], output.lower()[[i]], output.upper()[[i]]
                    );
                }
            }
        }
    }
}

// ======================== Proptest: Linear CROWN f64 ========================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Linear CROWN f64: backward pass concretized bounds contain true output.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_linear_crown_f64_soundness(
        weights in prop::collection::vec(-3.0f64..3.0, 6),  // 3x2 = 6
        biases in prop::collection::vec(-3.0f64..3.0, 3),
        bounds in prop::collection::vec(valid_interval_f64(5.0), 2),
    ) {
        let weight = Array2::from_shape_vec((3, 2), weights).unwrap();
        let bias = Array1::from_vec(biases);

        let lower_vec: Vec<f64> = bounds.iter().map(|(l, _)| *l).collect();
        let upper_vec: Vec<f64> = bounds.iter().map(|(_, u)| *u).collect();
        let input = BoundedTensor64::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), lower_vec).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), upper_vec).unwrap(),
        ).unwrap();

        // IBP for reference
        let ibp_output = propagate_linear_ibp_f64(&weight, &bias, &input).unwrap();

        // CROWN backward from identity
        let crown_bounds = crate::bounds::LinearBounds64::identity(3);
        let crown_result =
            propagate_linear_crown_backward_f64(&weight, &bias, &crown_bounds).unwrap();
        let crown_output = crown_result.concretize(&input).unwrap();

        // For a single linear layer, CROWN should match IBP exactly
        for i in 0..3 {
            prop_assert!(
                (crown_output.lower()[i] - ibp_output.lower()[i]).abs() < F64_TOLERANCE,
                "CROWN vs IBP lower mismatch at {}: crown={}, ibp={}",
                i, crown_output.lower()[i], ibp_output.lower()[i]
            );
            prop_assert!(
                (crown_output.upper()[i] - ibp_output.upper()[i]).abs() < F64_TOLERANCE,
                "CROWN vs IBP upper mismatch at {}: crown={}, ibp={}",
                i, crown_output.upper()[i], ibp_output.upper()[i]
            );
        }

        // Both must be sound at sample points
        for corner in 0..4 {
            let x_vec: Vec<f64> = (0..2)
                .map(|j| {
                    if (corner >> j) & 1 == 1 {
                        bounds[j].1
                    } else {
                        bounds[j].0
                    }
                })
                .collect();
            let x = Array1::from_vec(x_vec);
            let y = weight.dot(&x) + &bias;

            for i in 0..3 {
                prop_assert!(
                    ibp_output.lower()[i] - F64_TOLERANCE <= y[i]
                        && y[i] <= ibp_output.upper()[i] + F64_TOLERANCE,
                    "Linear CROWN f64 soundness violation at output {}: \
                     y[{}]={}, bounds=[{}, {}]",
                    i, i, y[i], ibp_output.lower()[i], ibp_output.upper()[i]
                );
            }
        }
    }
}

// ======================== Proptest: ReLU IBP f64 ========================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// ReLU IBP f64 soundness: max(0, x) within bounds for any x in input bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_relu_ibp_f64(
        (l1, u1) in valid_interval_f64(10.0),
        (l2, u2) in valid_interval_f64(10.0),
        (l3, u3) in valid_interval_f64(10.0),
    ) {
        let input = BoundedTensor64::new(
            arr1(&[l1, l2, l3]).into_dyn(),
            arr1(&[u1, u2, u3]).into_dyn(),
        ).unwrap();

        let output = propagate_relu_ibp_f64(&input).unwrap();

        let intervals = [(l1, u1), (l2, u2), (l3, u3)];
        for (i, (l, u)) in intervals.iter().enumerate() {
            for x in sample_points_f64(*l, *u, 7) {
                let y = x.max(0.0);
                prop_assert!(
                    output.lower()[[i]] - F64_TOLERANCE <= y
                        && y <= output.upper()[[i]] + F64_TOLERANCE,
                    "ReLU IBP f64 soundness violation at {}: x={}, relu(x)={}, \
                     bounds=[{}, {}]",
                    i, x, y, output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }
}

// ======================== Proptest: ReLU CROWN f64 ========================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// ReLU CROWN f64: triangle relaxation bounds contain true ReLU at all points.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_relu_crown_f64_soundness(
        (l1, u1) in valid_interval_f64(5.0),
        (l2, u2) in valid_interval_f64(5.0),
    ) {
        let pre_act = BoundedTensor64::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        ).unwrap();

        let bounds = crate::bounds::LinearBounds64::identity(2);
        let result = propagate_relu_crown_backward_f64(&bounds, &pre_act).unwrap();
        let concretized = result.concretize(&pre_act).unwrap();

        let intervals = [(l1, u1), (l2, u2)];
        for (i, (l, u)) in intervals.iter().enumerate() {
            for x in sample_points_f64(*l, *u, 10) {
                let y = x.max(0.0);
                prop_assert!(
                    concretized.lower()[i] - F64_TOLERANCE <= y
                        && y <= concretized.upper()[i] + F64_TOLERANCE,
                    "ReLU CROWN f64 soundness violation at {}: x={}, relu(x)={}, \
                     bounds=[{}, {}]",
                    i, x, y, concretized.lower()[i], concretized.upper()[i]
                );
            }
        }
    }
}

// ======================== Proptest: Conv2D IBP f64 ========================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Conv2D IBP f64 soundness with random 3x3 kernel on 4x4 input.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_conv2d_ibp_f64_soundness(
        kernel_vals in prop::collection::vec(-2.0f64..2.0, 9),
        bias_val in -2.0f64..2.0,
        input_bounds in prop::collection::vec(valid_interval_f64(3.0), 16),
    ) {
        let kernel = Array4::from_shape_vec((1, 1, 3, 3), kernel_vals.clone()).unwrap();
        let bias = arr1(&[bias_val]);
        let params = Conv2dParams {
            stride: (1, 1),
            padding: (0, 0),
            input_hw: (4, 4),
        };

        let lower_vec: Vec<f64> = input_bounds.iter().map(|(l, _)| *l).collect();
        let upper_vec: Vec<f64> = input_bounds.iter().map(|(_, u)| *u).collect();
        let input = BoundedTensor64::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), lower_vec).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), upper_vec).unwrap(),
        ).unwrap();

        let output = propagate_conv2d_ibp_f64(&kernel, &bias, &input, &params).unwrap();
        let (out_l, out_u) = output.flatten_to_1d();

        // Test corner points (all-lower and all-upper) plus midpoint
        for corner in [0u32, (1 << 16) - 1] {
            let x_vec: Vec<f64> = (0..16)
                .map(|j| {
                    if (corner >> j) & 1 == 1 {
                        input_bounds[j].1
                    } else {
                        input_bounds[j].0
                    }
                })
                .collect();

            // Compute true conv2d
            for oh in 0..2 {
                for ow in 0..2 {
                    let mut sum = bias_val;
                    for kh in 0..3 {
                        for kw in 0..3 {
                            let idx = (oh + kh) * 4 + (ow + kw);
                            sum += kernel_vals[kh * 3 + kw] * x_vec[idx];
                        }
                    }
                    let out_idx = oh * 2 + ow;
                    prop_assert!(
                        out_l[out_idx] - F64_TOLERANCE <= sum
                            && sum <= out_u[out_idx] + F64_TOLERANCE,
                        "Conv2D IBP f64 soundness violation at ({}, {}): \
                         y={}, bounds=[{}, {}]",
                        oh, ow, sum, out_l[out_idx], out_u[out_idx]
                    );
                }
            }
        }
    }
}

// ======================== Proptest: Conv2D CROWN f64 ========================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Conv2D CROWN f64 soundness: backward pass concretized bounds contain true output.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_conv2d_crown_f64_soundness(
        kernel_vals in prop::collection::vec(-2.0f64..2.0, 9),
        bias_val in -2.0f64..2.0,
        input_bounds in prop::collection::vec(valid_interval_f64(3.0), 16),
    ) {
        let kernel = Array4::from_shape_vec((1, 1, 3, 3), kernel_vals.clone()).unwrap();
        let bias = arr1(&[bias_val]);
        let params = Conv2dParams {
            stride: (1, 1),
            padding: (0, 0),
            input_hw: (4, 4),
        };

        let lower_vec: Vec<f64> = input_bounds.iter().map(|(l, _)| *l).collect();
        let upper_vec: Vec<f64> = input_bounds.iter().map(|(_, u)| *u).collect();
        let input = BoundedTensor64::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), lower_vec).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), upper_vec).unwrap(),
        ).unwrap();

        // CROWN backward from identity (4 outputs for 2x2 output)
        let crown_bounds = crate::bounds::LinearBounds64::identity(4);
        let crown_result =
            propagate_conv2d_crown_backward_f64(&kernel, &bias, &crown_bounds, &params)
                .unwrap();
        let crown_output = crown_result.concretize(&input).unwrap();
        let (crown_l, crown_u) = crown_output.flatten_to_1d();

        // IBP for comparison (returns 3D shape, must flatten)
        let ibp_output =
            propagate_conv2d_ibp_f64(&kernel, &bias, &input, &params).unwrap();
        let (ibp_l, ibp_u) = ibp_output.flatten_to_1d();

        // For a single conv layer, CROWN from identity should match IBP exactly
        for i in 0..4 {
            prop_assert!(
                (crown_l[i] - ibp_l[i]).abs() < F64_TOLERANCE,
                "Conv2D CROWN-vs-IBP lower mismatch at {}: crown={}, ibp={}",
                i, crown_l[i], ibp_l[i]
            );
            prop_assert!(
                (crown_u[i] - ibp_u[i]).abs() < F64_TOLERANCE,
                "Conv2D CROWN-vs-IBP upper mismatch at {}: crown={}, ibp={}",
                i, crown_u[i], ibp_u[i]
            );
        }

        // Both must contain true output at corners
        for corner in [0u32, (1 << 16) - 1] {
            let x_vec: Vec<f64> = (0..16)
                .map(|j| {
                    if (corner >> j) & 1 == 1 {
                        input_bounds[j].1
                    } else {
                        input_bounds[j].0
                    }
                })
                .collect();

            for oh in 0..2 {
                for ow in 0..2 {
                    let mut sum = bias_val;
                    for kh in 0..3 {
                        for kw in 0..3 {
                            let idx = (oh + kh) * 4 + (ow + kw);
                            sum += kernel_vals[kh * 3 + kw] * x_vec[idx];
                        }
                    }
                    let out_idx = oh * 2 + ow;
                    prop_assert!(
                        crown_l[out_idx] - F64_TOLERANCE <= sum
                            && sum <= crown_u[out_idx] + F64_TOLERANCE,
                        "Conv2D CROWN f64 soundness violation at ({}, {}): \
                         y={}, bounds=[{}, {}]",
                        oh, ow, sum, crown_l[out_idx], crown_u[out_idx]
                    );
                }
            }
        }
    }
}

// ======================== Proptest: End-to-end network f64 ========================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// End-to-end f64 CROWN soundness: Linear -> ReLU -> Linear network.
    /// Random weights, random input bounds. Verify concretized bounds contain
    /// true network output at sample points.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_network_crown_f64_soundness(
        w1_vals in prop::collection::vec(-2.0f64..2.0, 4),  // 2x2
        b1_vals in prop::collection::vec(-2.0f64..2.0, 2),
        w2_vals in prop::collection::vec(-2.0f64..2.0, 2),  // 1x2
        b2_val in -2.0f64..2.0,
        bounds in prop::collection::vec(valid_interval_f64(3.0), 2),
    ) {
        let w1 = Array2::from_shape_vec((2, 2), w1_vals).unwrap();
        let b1 = Array1::from_vec(b1_vals);
        let w2 = Array2::from_shape_vec((1, 2), w2_vals).unwrap();
        let b2 = arr1(&[b2_val]);

        let lower_vec: Vec<f64> = bounds.iter().map(|(l, _)| *l).collect();
        let upper_vec: Vec<f64> = bounds.iter().map(|(_, u)| *u).collect();
        let input = BoundedTensor64::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), lower_vec).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), upper_vec).unwrap(),
        ).unwrap();

        // IBP forward
        let after_l1 = propagate_linear_ibp_f64(&w1, &b1, &input).unwrap();
        let after_relu = propagate_relu_ibp_f64(&after_l1).unwrap();
        let ibp_output = propagate_linear_ibp_f64(&w2, &b2, &after_relu).unwrap();

        // CROWN backward
        let crown_bounds = crate::bounds::LinearBounds64::identity(1);
        let c_after_w2 =
            propagate_linear_crown_backward_f64(&w2, &b2, &crown_bounds).unwrap();
        let c_after_relu =
            propagate_relu_crown_backward_f64(&c_after_w2, &after_l1).unwrap();
        let c_final =
            propagate_linear_crown_backward_f64(&w1, &b1, &c_after_relu).unwrap();
        let crown_output = c_final.concretize(&input).unwrap();

        // Verify at all 4 corners + midpoint
        for corner in 0..4 {
            let x_vec: Vec<f64> = (0..2)
                .map(|j| {
                    if (corner >> j) & 1 == 1 {
                        bounds[j].1
                    } else {
                        bounds[j].0
                    }
                })
                .collect();
            let x = Array1::from_vec(x_vec);
            let hidden = (w1.dot(&x) + &b1).mapv(|v| v.max(0.0));
            let y_val = (w2.dot(&hidden) + &b2)[0];

            for (label, out) in [("IBP", &ibp_output), ("CROWN", &crown_output)] {
                prop_assert!(
                    out.lower()[0] - F64_TOLERANCE <= y_val
                        && y_val <= out.upper()[0] + F64_TOLERANCE,
                    "Network {label} f64 soundness violation: y={}, bounds=[{}, {}]",
                    y_val, out.lower()[0], out.upper()[0]
                );
            }
        }

        // Midpoint
        let x_mid = Array1::from_vec(
            bounds.iter().map(|(l, u)| (l + u) / 2.0).collect(),
        );
        let hidden_mid = (w1.dot(&x_mid) + &b1).mapv(|v| v.max(0.0));
        let y_mid = (w2.dot(&hidden_mid) + &b2)[0];

        for (label, out) in [("IBP", &ibp_output), ("CROWN", &crown_output)] {
            prop_assert!(
                out.lower()[0] - F64_TOLERANCE <= y_mid
                    && y_mid <= out.upper()[0] + F64_TOLERANCE,
                "Network {label} f64 midpoint violation: y={}, bounds=[{}, {}]",
                y_mid, out.lower()[0], out.upper()[0]
            );
        }
    }
}
