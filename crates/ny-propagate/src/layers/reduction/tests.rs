// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// Licensed under the Apache License, Version 2.0

use super::*;
use ndarray::{Array1, Array2, ArrayD, IxDyn};

fn assert_close(actual: f32, expected: f32, tol: f32, label: impl std::fmt::Display) {
    assert!(
        (actual - expected).abs() < tol,
        "{label}: expected {expected}, got {actual}"
    );
}

// ========== ReduceMeanLayer ==========

#[test]
fn test_reduce_mean_new() {
    let layer = ReduceMeanLayer::new(vec![1, 2], false);
    assert_eq!(layer.axes, vec![1, 2]);
    assert!(
        !layer.keepdims,
        "ReduceMeanLayer::new should preserve keepdims=false"
    );
}

#[test]
fn test_reduce_mean_last_axis() {
    let layer = ReduceMeanLayer::last_axis();
    assert_eq!(layer.axes, vec![-1]);
    assert!(
        layer.keepdims,
        "ReduceMeanLayer::last_axis should preserve dimensions"
    );
}

#[test]
fn test_reduce_mean_resolve_axes_negative() {
    let layer = ReduceMeanLayer::new(vec![-1], true);
    let resolved = layer.resolve_axes(3).unwrap();
    assert_eq!(resolved, vec![2]);
}

#[test]
fn test_reduce_mean_resolve_axes_empty_reduces_all() {
    let layer = ReduceMeanLayer::new(vec![], true);
    let resolved = layer.resolve_axes(3).unwrap();
    assert_eq!(resolved, vec![0, 1, 2]);
}

#[test]
fn test_reduce_mean_ibp_1d_keepdims() {
    // Input: bounds [1,3], [3,5], [5,7]
    // Mean over axis 0 keepdims=true: mean_l = mean([1,3,5])=3, mean_u = mean([3,5,7])=5
    let layer = ReduceMeanLayer::new(vec![0], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 3.0, 5.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 5.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.shape(), &[1]);
    assert_close(out.lower()[[0]], 3.0, 1e-5, "mean ibp keepdims lower[0]");
    assert_close(out.upper()[[0]], 5.0, 1e-5, "mean ibp keepdims upper[0]");
}

#[test]
fn test_reduce_mean_ibp_1d_no_keepdims() {
    let layer = ReduceMeanLayer::new(vec![0], false);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 3.0, 5.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 5.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    // Scalar output after reducing all elements
    assert_eq!(out.shape(), &[] as &[usize]);
    assert_close(out.lower()[[]], 3.0, 1e-5, "mean ibp scalar lower");
    assert_close(out.upper()[[]], 5.0, 1e-5, "mean ibp scalar upper");
}

#[test]
fn test_reduce_mean_ibp_2d_last_axis_keepdims() {
    // Shape [2, 3], reduce axis -1 (axis 1), keepdims
    // Row 0: lower=[1,2,3], upper=[4,5,6] -> mean_l=2, mean_u=5
    // Row 1: lower=[10,20,30], upper=[40,50,60] -> mean_l=20, mean_u=50
    let layer = ReduceMeanLayer::last_axis();

    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![4.0, 5.0, 6.0, 40.0, 50.0, 60.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    assert_close(out.lower()[[0, 0]], 2.0, 1e-5, "mean ibp 2d lower[0,0]");
    assert_close(out.upper()[[0, 0]], 5.0, 1e-5, "mean ibp 2d upper[0,0]");
    assert_close(out.lower()[[1, 0]], 20.0, 1e-5, "mean ibp 2d lower[1,0]");
    assert_close(out.upper()[[1, 0]], 50.0, 1e-5, "mean ibp 2d upper[1,0]");
}

#[test]
fn test_reduce_mean_ibp_soundness_random_corners() {
    // Verify that for any corner point of the input box, the concrete
    // mean lies within the IBP bounds.
    let layer = ReduceMeanLayer::new(vec![-1], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-1.0, 0.0, 2.0, -3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 2.0, 5.0, 0.0]).unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let out = layer.propagate_ibp(&input).unwrap();

    // Test all 16 corners of a 4D box
    for mask in 0..16u32 {
        let mut vals = [0.0f32; 4];
        for i in 0..4 {
            vals[i] = if mask & (1 << i) != 0 {
                upper[[0, i]]
            } else {
                lower[[0, i]]
            };
        }
        let mean = vals.iter().sum::<f32>() / 4.0;
        assert!(
            out.lower()[[0, 0]] <= mean + 1e-5,
            "Lower bound {} > mean {} at mask {}",
            out.lower()[[0, 0]],
            mean,
            mask
        );
        assert!(
            out.upper()[[0, 0]] >= mean - 1e-5,
            "Upper bound {} < mean {} at mask {}",
            out.upper()[[0, 0]],
            mean,
            mask
        );
    }
}

#[test]
fn test_reduce_mean_ibp_point_input() {
    // When lower == upper, output bounds should equal the concrete mean
    let layer = ReduceMeanLayer::new(vec![0], true);

    let vals = ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0, 4.0, 6.0, 8.0]).unwrap();
    let input = BoundedTensor::new(vals.clone(), vals).unwrap();
    let out = layer.propagate_ibp(&input).unwrap();

    assert_close(out.lower()[[0]], 5.0, 1e-5, "mean point lower[0]");
    assert_close(out.upper()[[0]], 5.0, 1e-5, "mean point upper[0]");
}

#[test]
fn test_reduce_mean_propagate_linear_returns_unsupported() {
    let layer = ReduceMeanLayer::new(vec![-1], true);
    let bounds = LinearBounds::new(
        Array2::eye(3),
        Array1::zeros(3),
        Array2::eye(3),
        Array1::zeros(3),
    )
    .unwrap();
    let result = layer.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "ReduceMeanLayer::propagate_linear should reject unsupported CROWN propagation",
    );
}

#[test]
fn test_reduce_mean_requires_pre_activation_bounds() {
    let layer = ReduceMeanLayer::new(vec![-1], true);
    assert!(
        layer.requires_pre_activation_bounds(),
        "ReduceMeanLayer should require pre-activation bounds",
    );
}

#[test]
fn test_reduce_mean_crown_backward_last_axis() {
    // Input shape [1, 4], reduce axis -1 keepdims -> output [1, 1]
    // CROWN backward should expand coefficients by 1/4 per input element
    let layer = ReduceMeanLayer::new(vec![-1], true);

    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0, 0.0, 0.0, 0.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Output has 1 element -> bounds shape is (1, 1)
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
        Array1::from_vec(vec![0.5]),
        Array2::from_shape_vec((1, 1), vec![3.0]).unwrap(),
        Array1::from_vec(vec![1.0]),
    )
    .unwrap();

    let result = layer
        .propagate_crown_backward(&bounds, Some(&pre_act))
        .unwrap();

    // Result should have shape (1, 4) with coefficients = original * (1/4)
    assert_eq!(result.lower_a.shape(), &[1, 4]);
    assert_eq!(result.upper_a.shape(), &[1, 4]);
    for j in 0..4 {
        assert_close(
            result.lower_a[[0, j]],
            0.5,
            1e-5,
            format!("mean crown lower_a[0,{j}]"),
        );
        assert_close(
            result.upper_a[[0, j]],
            0.75,
            1e-5,
            format!("mean crown upper_a[0,{j}]"),
        );
    }
    // Bias unchanged
    assert_close(result.lower_b[0], 0.5, 1e-5, "mean crown lower_b[0]");
    assert_close(result.upper_b[0], 1.0, 1e-5, "mean crown upper_b[0]");
}

#[test]
fn test_reduce_mean_crown_backward_no_pre_activation_errors() {
    let layer = ReduceMeanLayer::new(vec![-1], true);
    let bounds = LinearBounds::new(
        Array2::eye(1),
        Array1::zeros(1),
        Array2::eye(1),
        Array1::zeros(1),
    )
    .unwrap();
    let result = layer.propagate_crown_backward(&bounds, None);
    assert!(
        result.is_err(),
        "ReduceMeanLayer::propagate_crown_backward should reject missing pre-activation bounds",
    );
}

// ========== ReduceSumLayer ==========

#[test]
fn test_reduce_sum_new() {
    let layer = ReduceSumLayer::new(vec![0, 2], true);
    assert_eq!(layer.axes, vec![0, 2]);
    assert!(
        layer.keepdims,
        "ReduceSumLayer::new should preserve keepdims=true"
    );
}

#[test]
fn test_reduce_sum_last_axis() {
    let layer = ReduceSumLayer::last_axis();
    assert_eq!(layer.axes, vec![-1]);
    assert!(
        layer.keepdims,
        "ReduceSumLayer::last_axis should preserve dimensions"
    );
}

#[test]
fn test_reduce_sum_ibp_1d_keepdims() {
    // Input bounds: [1,3], [5,7], [2,4] -> sum_l=8, sum_u=14
    let layer = ReduceSumLayer::new(vec![0], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 5.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 7.0, 4.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.shape(), &[1]);
    assert_close(out.lower()[[0]], 8.0, 1e-5, "sum ibp keepdims lower[0]");
    assert_close(out.upper()[[0]], 14.0, 1e-5, "sum ibp keepdims upper[0]");
}

#[test]
fn test_reduce_sum_ibp_1d_no_keepdims() {
    let layer = ReduceSumLayer::new(vec![0], false);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 5.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 7.0, 4.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.shape(), &[] as &[usize]);
    assert_close(out.lower()[[]], 8.0, 1e-5, "sum ibp scalar lower");
    assert_close(out.upper()[[]], 14.0, 1e-5, "sum ibp scalar upper");
}

#[test]
fn test_reduce_sum_ibp_2d_last_axis_keepdims() {
    // Shape [2, 3], reduce axis -1 (axis 1), keepdims
    // Row 0: lower=[1,2,3], upper=[4,5,6] -> sum_l=6, sum_u=15
    // Row 1: lower=[10,20,30], upper=[40,50,60] -> sum_l=60, sum_u=150
    let layer = ReduceSumLayer::last_axis();

    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![4.0, 5.0, 6.0, 40.0, 50.0, 60.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    // The sound IBP directed-rounds OUTWARD, so bounds enclose the true sum within ~1 ULP
    // (which is ~1.5e-5 at magnitude 150 — exceeds a 1e-5 abs tolerance). Assert enclosure
    // + tightness relative to the value's ULP.
    let tol = |v: f32| v.abs() * 2e-7 + 1e-6;
    assert!(out.lower()[[0, 0]] <= 6.0 && (out.lower()[[0, 0]] - 6.0).abs() < tol(6.0));
    assert!(out.upper()[[0, 0]] >= 15.0 && (out.upper()[[0, 0]] - 15.0).abs() < tol(15.0));
    assert!(out.lower()[[1, 0]] <= 60.0 && (out.lower()[[1, 0]] - 60.0).abs() < tol(60.0));
    assert!(out.upper()[[1, 0]] >= 150.0 && (out.upper()[[1, 0]] - 150.0).abs() < tol(150.0));
}

#[test]
fn test_reduce_sum_ibp_soundness_corners() {
    let layer = ReduceSumLayer::new(vec![-1], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![-2.0, 1.0, -1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![3.0, 4.0, 2.0]).unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let out = layer.propagate_ibp(&input).unwrap();

    // Test all 8 corners
    for mask in 0..8u32 {
        let mut sum = 0.0f32;
        for i in 0..3 {
            sum += if mask & (1 << i) != 0 {
                upper[[0, i]]
            } else {
                lower[[0, i]]
            };
        }
        assert!(
            out.lower()[[0, 0]] <= sum + 1e-5,
            "Lower {} > sum {} at mask {}",
            out.lower()[[0, 0]],
            sum,
            mask
        );
        assert!(
            out.upper()[[0, 0]] >= sum - 1e-5,
            "Upper {} < sum {} at mask {}",
            out.upper()[[0, 0]],
            sum,
            mask
        );
    }
}

#[test]
fn test_reduce_sum_propagate_linear_returns_unsupported() {
    let layer = ReduceSumLayer::new(vec![-1], true);
    let bounds = LinearBounds::new(
        Array2::eye(3),
        Array1::zeros(3),
        Array2::eye(3),
        Array1::zeros(3),
    )
    .unwrap();
    let result = layer.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "ReduceSumLayer::propagate_linear should reject unsupported CROWN propagation",
    );
}

#[test]
fn test_reduce_sum_requires_pre_activation_bounds() {
    let layer = ReduceSumLayer::new(vec![-1], true);
    assert!(
        layer.requires_pre_activation_bounds(),
        "ReduceSumLayer should require pre-activation bounds",
    );
}

#[test]
fn test_reduce_sum_crown_backward_last_axis() {
    // Input shape [1, 3], reduce axis -1 keepdims -> output [1, 1]
    // CROWN backward: coefficients copied without scaling (sum, not mean)
    let layer = ReduceSumLayer::new(vec![-1], true);

    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![0.0, 0.0, 0.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 1.0, 1.0]).unwrap();
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
        Array1::from_vec(vec![0.5]),
        Array2::from_shape_vec((1, 1), vec![3.0]).unwrap(),
        Array1::from_vec(vec![1.0]),
    )
    .unwrap();

    let result = layer
        .propagate_crown_backward(&bounds, Some(&pre_act))
        .unwrap();

    // Result should have shape (1, 3) with coefficients copied directly (no 1/n)
    assert_eq!(result.lower_a.shape(), &[1, 3]);
    assert_eq!(result.upper_a.shape(), &[1, 3]);
    for j in 0..3 {
        assert_close(
            result.lower_a[[0, j]],
            2.0,
            1e-5,
            format!("sum crown lower_a[0,{j}]"),
        );
        assert_close(
            result.upper_a[[0, j]],
            3.0,
            1e-5,
            format!("sum crown upper_a[0,{j}]"),
        );
    }
    // Bias unchanged
    assert_close(result.lower_b[0], 0.5, 1e-5, "sum crown lower_b[0]");
    assert_close(result.upper_b[0], 1.0, 1e-5, "sum crown upper_b[0]");
}

#[test]
fn test_reduce_sum_crown_backward_no_pre_activation_errors() {
    let layer = ReduceSumLayer::new(vec![-1], true);
    let bounds = LinearBounds::new(
        Array2::eye(1),
        Array1::zeros(1),
        Array2::eye(1),
        Array1::zeros(1),
    )
    .unwrap();
    let result = layer.propagate_crown_backward(&bounds, None);
    assert!(
        result.is_err(),
        "ReduceSumLayer::propagate_crown_backward should reject missing pre-activation bounds",
    );
}

#[test]
fn test_reduce_mean_vs_sum_consistency() {
    // Mean = Sum / n, so IBP results should relate by factor of n
    let mean_layer = ReduceMeanLayer::new(vec![-1], true);
    let sum_layer = ReduceSumLayer::new(vec![-1], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let mean_out = mean_layer.propagate_ibp(&input).unwrap();
    let sum_out = sum_layer.propagate_ibp(&input).unwrap();

    let n = 4.0f32;
    assert_close(
        mean_out.lower()[[0, 0]] * n,
        sum_out.lower()[[0, 0]],
        1e-4,
        "mean lower * n should match sum lower",
    );
    assert_close(
        mean_out.upper()[[0, 0]] * n,
        sum_out.upper()[[0, 0]],
        1e-4,
        "mean upper * n should match sum upper",
    );
}

// ========== ReduceMaxLayer ==========

#[test]
fn test_reduce_max_new() {
    let layer = ReduceMaxLayer::new(vec![-1], true);
    assert_eq!(layer.axes, vec![-1]);
    assert!(
        layer.keepdims,
        "ReduceMaxLayer::new should preserve keepdims=true"
    );
    assert!(
        layer.fixed_max_index,
        "ReduceMaxLayer::new should default to fixed_max_index=true",
    );
}

#[test]
fn test_reduce_max_ibp_1d_keepdims() {
    // Input bounds: [1,3], [5,7], [2,4]
    // max(lower) = max(1,5,2) = 5, max(upper) = max(3,7,4) = 7
    let layer = ReduceMaxLayer::new(vec![0], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 5.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 7.0, 4.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.shape(), &[1]);
    assert_close(out.lower()[[0]], 5.0, 1e-5, "max ibp lower[0]");
    assert_close(out.upper()[[0]], 7.0, 1e-5, "max ibp upper[0]");
}

#[test]
fn test_reduce_max_ibp_2d_last_axis_keepdims() {
    // Shape [2, 3], reduce last axis keepdims
    // Row 0: lower=[1,2,3], upper=[4,5,6] -> max_l=3, max_u=6
    // Row 1: lower=[10,20,30], upper=[40,50,60] -> max_l=30, max_u=60
    let layer = ReduceMaxLayer::last_axis();

    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![4.0, 5.0, 6.0, 40.0, 50.0, 60.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    assert_close(out.lower()[[0, 0]], 3.0, 1e-5, "max ibp 2d lower[0,0]");
    assert_close(out.upper()[[0, 0]], 6.0, 1e-5, "max ibp 2d upper[0,0]");
    assert_close(out.lower()[[1, 0]], 30.0, 1e-5, "max ibp 2d lower[1,0]");
    assert_close(out.upper()[[1, 0]], 60.0, 1e-5, "max ibp 2d upper[1,0]");
}

#[test]
fn test_reduce_max_ibp_soundness_corners() {
    // Verify that for any corner point of the input box, the concrete
    // max lies within the IBP bounds.
    let layer = ReduceMaxLayer::new(vec![-1], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-1.0, 0.0, 2.0, -3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 2.0, 5.0, 0.0]).unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let out = layer.propagate_ibp(&input).unwrap();

    // Test all 16 corners of a 4D box
    for mask in 0..16u32 {
        let mut vals = [0.0f32; 4];
        for i in 0..4 {
            vals[i] = if mask & (1 << i) != 0 {
                upper[[0, i]]
            } else {
                lower[[0, i]]
            };
        }
        let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            out.lower()[[0, 0]] <= max_val + 1e-5,
            "Lower bound {} > max {} at mask {}",
            out.lower()[[0, 0]],
            max_val,
            mask
        );
        assert!(
            out.upper()[[0, 0]] >= max_val - 1e-5,
            "Upper bound {} < max {} at mask {}",
            out.upper()[[0, 0]],
            max_val,
            mask
        );
    }
}

#[test]
fn test_reduce_max_crown_backward_scatter() {
    // Input shape [1, 4], reduce last axis keepdims -> output [1, 1]
    // Center = (lower+upper)/2 = [0.5, 1.0, 3.5, -1.5]
    // argmax at center = index 2 (value 3.5)
    // CROWN backward should scatter coefficient to index 2 only
    let layer = ReduceMaxLayer::new(vec![-1], true);

    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-1.0, 0.0, 2.0, -3.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![2.0, 2.0, 5.0, 0.0]).unwrap();
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
        Array1::from_vec(vec![0.5]),
        Array2::from_shape_vec((1, 1), vec![3.0]).unwrap(),
        Array1::from_vec(vec![1.0]),
    )
    .unwrap();

    let result = layer
        .propagate_crown_backward(&bounds, Some(&pre_act))
        .unwrap();

    // Result should have shape (1, 4) with coefficient only at index 2
    assert_eq!(result.lower_a.shape(), &[1, 4]);
    assert_eq!(result.upper_a.shape(), &[1, 4]);
    for j in 0..4 {
        if j == 2 {
            assert!(
                (result.lower_a[[0, j]] - 2.0).abs() < 1e-5,
                "lower_a[0,2] = {} expected 2.0",
                result.lower_a[[0, j]]
            );
            assert!(
                (result.upper_a[[0, j]] - 3.0).abs() < 1e-5,
                "upper_a[0,2] = {} expected 3.0",
                result.upper_a[[0, j]]
            );
        } else {
            assert!(
                result.lower_a[[0, j]].abs() < 1e-5,
                "lower_a[0,{}] = {} expected 0.0",
                j,
                result.lower_a[[0, j]]
            );
            assert!(
                result.upper_a[[0, j]].abs() < 1e-5,
                "upper_a[0,{}] = {} expected 0.0",
                j,
                result.upper_a[[0, j]]
            );
        }
    }
    // Bias unchanged
    assert!(
        (result.lower_b[0] - 0.5).abs() < 1e-5,
        "lower_b should be 0.5, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.upper_b[0] - 1.0).abs() < 1e-5,
        "upper_b should be 1.0, got {}",
        result.upper_b[0]
    );
}

/// SOUNDNESS regression: when the argmax is NOT stable over the input box (no
/// definite winner), ReduceMax CROWN backward must NOT scatter a single fixed
/// index (that underestimates the max and is unsound). It must fall back to the
/// sound IBP interval folded into the bias (zero A-row). Previously the code
/// scattered the center-argmax unconditionally, yielding a falsely-tight bound.
#[test]
fn test_reduce_max_crown_backward_unstable_argmax_constant_fold() {
    // Input [1,2], reduce last axis keepdims -> output [1,1].
    // Box: x0,x1 in [0,2]. No definite winner (each lower 0 < other upper 2).
    let layer = ReduceMaxLayer::new(vec![-1], true);
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![2.0, 2.0]).unwrap();
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Incoming linear bound rows: lower = 2*y + 0.5, upper = 3*y + 1.0 over y=max(x).
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
        Array1::from_vec(vec![0.5]),
        Array2::from_shape_vec((1, 1), vec![3.0]).unwrap(),
        Array1::from_vec(vec![1.0]),
    )
    .unwrap();

    let result = layer
        .propagate_crown_backward(&bounds, Some(&pre_act))
        .unwrap();

    // No gradient flows to any input (constant fold): both A-rows are zero.
    for j in 0..2 {
        assert!(
            result.lower_a[[0, j]].abs() < 1e-6,
            "unstable argmax must zero lower_a[0,{j}], got {}",
            result.lower_a[[0, j]]
        );
        assert!(
            result.upper_a[[0, j]].abs() < 1e-6,
            "unstable argmax must zero upper_a[0,{j}], got {}",
            result.upper_a[[0, j]]
        );
    }
    // y = max(x) in [0,2]. Lower row 2*y+0.5: min over y => 2*0+0.5 = 0.5.
    assert!(
        (result.lower_b[0] - 0.5).abs() < 1e-5,
        "lower_b should fold to 0.5, got {}",
        result.lower_b[0]
    );
    // Upper row 3*y+1: max over y => 3*2+1 = 7.0. (The old fixed-index code gave
    // 3*x_argmax+1 which at x=[0,2] is 1.0 < true 7.0 -> UNSOUND.) Must be >= 7.0.
    assert!(
        result.upper_b[0] >= 7.0 - 1e-5,
        "upper_b must soundly cover the moving max (>=7.0), got {}",
        result.upper_b[0]
    );

    // Adversarial soundness check at the box corner where the OLD code failed:
    // the upper bound 3*y+upper_b evaluated as a function of inputs must dominate
    // the true 3*max(x)+1 everywhere in the box. With A=0, upper = upper_b alone,
    // and true 3*max+1 maxes at 7.0, so upper_b >= 7.0 guarantees soundness.
    assert!(result.upper_b[0] >= 7.0 - 1e-5);
}

#[test]
fn test_reduce_max_requires_pre_activation_bounds() {
    let layer = ReduceMaxLayer::new(vec![-1], true);
    assert!(
        layer.requires_pre_activation_bounds(),
        "ReduceMaxLayer should require pre-activation bounds",
    );
}

// ========== ReduceMinLayer ==========

#[test]
fn test_reduce_min_ibp_1d_keepdims() {
    // Input bounds: [1,3], [5,7], [2,4]
    // min(lower) = min(1,5,2) = 1, min(upper) = min(3,7,4) = 3
    let layer = ReduceMinLayer::new(vec![0], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 5.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 7.0, 4.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.shape(), &[1]);
    assert!(
        (out.lower()[[0]] - 1.0).abs() < 1e-5,
        "reduce_min lower should be 1.0, got {}",
        out.lower()[[0]]
    );
    assert!(
        (out.upper()[[0]] - 3.0).abs() < 1e-5,
        "reduce_min upper should be 3.0, got {}",
        out.upper()[[0]]
    );
}

#[test]
fn test_reduce_min_crown_backward_scatter() {
    // Input shape [1, 4], reduce last axis keepdims -> output [1, 1]
    // Center = [0.5, 1.0, 3.5, -1.5]
    // argmin at center = index 3 (value -1.5)
    let layer = ReduceMinLayer::new(vec![-1], true);

    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-1.0, 0.0, 2.0, -3.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![2.0, 2.0, 5.0, 0.0]).unwrap();
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
        Array1::from_vec(vec![0.5]),
        Array2::from_shape_vec((1, 1), vec![3.0]).unwrap(),
        Array1::from_vec(vec![1.0]),
    )
    .unwrap();

    let result = layer
        .propagate_crown_backward(&bounds, Some(&pre_act))
        .unwrap();

    // This box is a MOVING-argmin case: idx3.upper(0) is NOT <= idx0.lower(-1), so
    // no group member provably dominates. The SOUND backward must NOT scatter the
    // center-argmin (that underestimates the min envelope and is unsound); it must
    // constant-fold the IBP interval [min lowers, min uppers] = [-3, 0] into the
    // bias with a zero A-row. (Previously the code scattered idx3 unconditionally.)
    assert_eq!(result.lower_a.shape(), &[1, 4]);
    for j in 0..4 {
        assert!(
            result.lower_a[[0, j]].abs() < 1e-6,
            "unstable argmin must zero lower_a[0,{j}], got {}",
            result.lower_a[[0, j]]
        );
        assert!(
            result.upper_a[[0, j]].abs() < 1e-6,
            "unstable argmin must zero upper_a[0,{j}], got {}",
            result.upper_a[[0, j]]
        );
    }
    // y = min(x) in [-3, 0]. Lower row 2*y+0.5: min over y => 2*(-3)+0.5 = -5.5.
    assert!(
        (result.lower_b[0] - (-5.5)).abs() < 1e-4,
        "lower_b should fold to -5.5, got {}",
        result.lower_b[0]
    );
    // Upper row 3*y+1: max over y => 3*0+1 = 1.0.
    assert!(
        (result.upper_b[0] - 1.0).abs() < 1e-4,
        "upper_b should fold to 1.0, got {}",
        result.upper_b[0]
    );
}

// ========== NaN propagation regression tests (#3318) ==========
//
// These tests verify the defense-in-depth chain for NaN inputs:
// 1. nan_propagating_max/min propagates NaN through the fold (instead of absorbing)
// 2. BoundedTensor::new_repaired(Conservative) catches the NaN and widens to ±inf (#3423)
//
// Before the fix, f32::max/min silently absorbed NaN, producing tight-but-wrong
// bounds that passed BoundedTensor::new without any repair firing.

#[test]
fn test_reduce_max_ibp_nan_in_lower_widens_to_infinity() {
    // NaN in lower[0] must propagate through max fold, then get widened
    // to -inf. Old code: f32::max absorbed NaN → lower=3.0 (wrong).
    let layer = ReduceMaxLayer::new(vec![0], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 2.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let result = layer.propagate_ibp(&input).expect(
        "NaN lower should propagate, hit new_repaired, and produce valid conservative bounds",
    );
    // NaN propagated → new_repaired(Conservative) → -inf
    assert_eq!(
        result.lower()[[0]],
        f32::NEG_INFINITY,
        "lower must be widened to -inf, not 3.0"
    );
    assert_eq!(
        result.upper()[[0]],
        7.0,
        "upper is NaN-free and should be unchanged"
    );
}

#[test]
fn test_reduce_max_ibp_nan_in_upper_widens_to_infinity() {
    let layer = ReduceMaxLayer::new(vec![0], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![5.0, f32::NAN, 7.0]).unwrap();
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let result = layer.propagate_ibp(&input).expect(
        "NaN upper should propagate, hit new_repaired, and produce valid conservative bounds",
    );
    assert_eq!(
        result.lower()[[0]],
        3.0,
        "lower is NaN-free and should be unchanged"
    );
    // NaN propagated → new_repaired(Conservative) → +inf
    assert_eq!(
        result.upper()[[0]],
        f32::INFINITY,
        "upper must be widened to +inf, not 7.0"
    );
}

#[test]
fn test_reduce_min_ibp_nan_in_lower_widens_to_infinity() {
    // Same pattern for ReduceMin: f32::min also silently absorbs NaN.
    let layer = ReduceMinLayer::new(vec![0], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, f32::NAN, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let result = layer.propagate_ibp(&input).expect(
        "NaN lower should propagate through min fold and get repaired to conservative bounds",
    );
    // NaN propagated → new_repaired(Conservative) → -inf
    assert_eq!(
        result.lower()[[0]],
        f32::NEG_INFINITY,
        "lower must be widened to -inf, not 1.0"
    );
    assert_eq!(
        result.upper()[[0]],
        5.0,
        "upper is NaN-free and should be unchanged"
    );
}

#[test]
fn test_reduce_min_ibp_nan_in_upper_widens_to_infinity() {
    let layer = ReduceMinLayer::new(vec![0], true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let result = layer.propagate_ibp(&input).expect(
        "NaN upper should propagate through min fold and get repaired to conservative bounds",
    );
    assert_eq!(
        result.lower()[[0]],
        1.0,
        "lower is NaN-free and should be unchanged"
    );
    // NaN propagated → new_repaired(Conservative) → +inf
    assert_eq!(
        result.upper()[[0]],
        f32::INFINITY,
        "upper must be widened to +inf, not 6.0"
    );
}

#[test]
fn test_reduce_max_ibp_2d_nan_in_reduced_axis_widens_to_infinity() {
    // NaN at position [0, 1] in lower bounds; reduce last axis (1) → NaN propagates
    // only in row 0, row 1 is NaN-free
    let layer = ReduceMaxLayer::last_axis();

    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let result = layer
        .propagate_ibp(&input)
        .expect("NaN in 2D lower should propagate and get repaired per-row");
    // Row 0: NaN propagated → -inf; Row 1: max(4,5,6) = 6.0 (no NaN)
    assert_eq!(
        result.lower()[[0, 0]],
        f32::NEG_INFINITY,
        "row 0 lower must be widened to -inf"
    );
    assert_eq!(
        result.lower()[[1, 0]],
        6.0,
        "row 1 lower is NaN-free, should be max(4,5,6)=6"
    );
    assert_eq!(
        result.upper()[[0, 0]],
        6.0,
        "row 0 upper is NaN-free, should be max(4,5,6)=6"
    );
    assert_eq!(
        result.upper()[[1, 0]],
        9.0,
        "row 1 upper is NaN-free, should be max(7,8,9)=9"
    );
}

// ========== Batched CROWN for ReduceMax/ReduceMin ==========

#[test]
fn test_reduce_max_batched_crown_last_axis_1d() {
    // Input shape [4], reduce last axis keepdims -> output [1]
    // Center = [0.25, 0.25, 0.75, 0.25], argmax = 2
    // Only index 2 gets the coefficient
    let layer = ReduceMaxLayer::last_axis();

    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 0.0, 0.5, 0.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.5, 0.5, 1.0, 0.5]).unwrap();
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = BatchedLinearBounds::identity(&[1]).unwrap();
    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();

    assert_eq!(result.lower_a().shape(), &[1, 4]);
    for j in 0..4 {
        if j == 2 {
            assert!(
                (result.lower_a()[[0, j]] - 1.0).abs() < 1e-6,
                "lower_a[0,2] = {} expected 1.0",
                result.lower_a()[[0, j]]
            );
        } else {
            assert!(
                result.lower_a()[[0, j]].abs() < 1e-6,
                "lower_a[0,{}] = {} expected 0.0",
                j,
                result.lower_a()[[0, j]]
            );
        }
    }
    assert_eq!(result.input_shape(), &[4]);
}

#[test]
fn test_reduce_max_batched_crown_2d() {
    // Input shape [2, 4], reduce last axis keepdims -> output [2, 1]
    // Row 0 center: [0.5, 1.0, 3.5, -1.5] -> argmax = 2
    // Row 1 center: [10.0, 50.0, 20.0, 30.0] -> argmax = 1
    let layer = ReduceMaxLayer::last_axis();

    let pre_lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![-1.0, 0.0, 2.0, -3.0, 0.0, 40.0, 10.0, 20.0],
    )
    .unwrap();
    let pre_upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![2.0, 2.0, 5.0, 0.0, 20.0, 60.0, 30.0, 40.0],
    )
    .unwrap();
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = BatchedLinearBounds::identity(&[2, 1]).unwrap();
    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();

    assert_eq!(result.lower_a().shape(), &[2, 1, 4]);
    assert_eq!(result.input_shape(), &[2, 4]);

    // Row 0: only index 2 gets coefficient
    for j in 0..4 {
        let expected = if j == 2 { 1.0 } else { 0.0 };
        assert!(
            (result.lower_a()[[0, 0, j]] - expected).abs() < 1e-6,
            "lower_a[0,0,{}] = {} expected {}",
            j,
            result.lower_a()[[0, 0, j]],
            expected
        );
    }
    // Row 1: only index 1 gets coefficient
    for j in 0..4 {
        let expected = if j == 1 { 1.0 } else { 0.0 };
        assert!(
            (result.lower_a()[[1, 0, j]] - expected).abs() < 1e-6,
            "lower_a[1,0,{}] = {} expected {}",
            j,
            result.lower_a()[[1, 0, j]],
            expected
        );
    }
}

#[test]
fn test_reduce_min_batched_crown_2d() {
    // Input shape [2, 3], reduce last axis keepdims
    // Row 0 center: [5.0, 1.0, 3.0] -> argmin = 1
    // Row 1 center: [10.0, 20.0, 5.0] -> argmin = 2
    let layer = ReduceMinLayer::last_axis();

    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![4.0, 0.0, 2.0, 8.0, 18.0, 3.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![6.0, 2.0, 4.0, 12.0, 22.0, 7.0]).unwrap();
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = BatchedLinearBounds::identity(&[2, 1]).unwrap();
    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();

    assert_eq!(result.lower_a().shape(), &[2, 1, 3]);
    // Row 0: argmin at index 1
    for j in 0..3 {
        let expected = if j == 1 { 1.0 } else { 0.0 };
        assert!(
            (result.lower_a()[[0, 0, j]] - expected).abs() < 1e-6,
            "lower_a[0,0,{}] = {} expected {}",
            j,
            result.lower_a()[[0, 0, j]],
            expected
        );
    }
    // Row 1: argmin at index 2
    for j in 0..3 {
        let expected = if j == 2 { 1.0 } else { 0.0 };
        assert!(
            (result.lower_a()[[1, 0, j]] - expected).abs() < 1e-6,
            "lower_a[1,0,{}] = {} expected {}",
            j,
            result.lower_a()[[1, 0, j]],
            expected
        );
    }
}

#[test]
fn test_reduce_max_batched_crown_non_last_axis_rejected() {
    let layer = ReduceMaxLayer::new(vec![0], true);

    let pre_lower = ArrayD::zeros(IxDyn(&[3, 4]));
    let pre_upper = ArrayD::from_elem(IxDyn(&[3, 4]), 1.0);
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = BatchedLinearBounds::identity(&[1, 4]).unwrap();
    let result = layer.propagate_linear_batched(&bounds, &pre_act);
    assert!(
        result.is_err(),
        "ReduceMax batched CROWN should reject non-last-axis reductions",
    );
}

#[test]
fn test_reduce_max_batched_crown_keepdims_false_rejected() {
    let layer = ReduceMaxLayer::new(vec![-1], false);

    let pre_lower = ArrayD::zeros(IxDyn(&[4]));
    let pre_upper = ArrayD::from_elem(IxDyn(&[4]), 1.0);
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = BatchedLinearBounds::identity(&[1]).unwrap();
    let result = layer.propagate_linear_batched(&bounds, &pre_act);
    assert!(
        result.is_err(),
        "ReduceMax batched CROWN should reject keepdims=false",
    );
}

// ========== Out-of-range axis validation (regression for #1951) ==========

#[test]
fn test_reduce_mean_negative_axis_out_of_range() {
    // axis=-5 for a 3D tensor: resolves to 3 + (-5) = -2, which is invalid
    let layer = ReduceMeanLayer::new(vec![-5], true);
    let result = layer.resolve_axes(3);
    assert!(result.is_err(), "Expected error for axis=-5 with ndim=3");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("axis -5") && msg.contains("out of range"),
        "Error message should mention the axis: {}",
        msg
    );
}

#[test]
fn test_reduce_mean_positive_axis_out_of_range() {
    // axis=3 for a 3D tensor: only valid axes are 0, 1, 2
    let layer = ReduceMeanLayer::new(vec![3], true);
    let result = layer.resolve_axes(3);
    assert!(result.is_err(), "Expected error for axis=3 with ndim=3");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("out of range"),
        "Error message should mention out of range: {}",
        msg
    );
}

#[test]
fn test_reduce_mean_ibp_negative_axis_out_of_range() {
    // Verify the error propagates through IBP
    let layer = ReduceMeanLayer::new(vec![-4], true);
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0; 6]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let result = layer.propagate_ibp(&input);
    assert!(result.is_err(), "IBP should fail for axis=-4 with ndim=2");
}

#[test]
fn test_reduce_sum_negative_axis_out_of_range() {
    let layer = ReduceSumLayer::new(vec![-5], true);
    let result = layer.resolve_axes(3);
    assert!(result.is_err(), "Expected error for axis=-5 with ndim=3");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("axis -5") && msg.contains("out of range"),
        "Error message should mention the axis: {}",
        msg
    );
}

#[test]
fn test_reduce_sum_positive_axis_out_of_range() {
    let layer = ReduceSumLayer::new(vec![3], true);
    let result = layer.resolve_axes(3);
    assert!(result.is_err(), "Expected error for axis=3 with ndim=3");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("out of range"),
        "Error message should mention out of range: {}",
        msg
    );
}

#[test]
fn test_reduce_sum_ibp_negative_axis_out_of_range() {
    // Verify the error propagates through IBP
    let layer = ReduceSumLayer::new(vec![-4], true);
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0; 6]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let result = layer.propagate_ibp(&input);
    assert!(result.is_err(), "IBP should fail for axis=-4 with ndim=2");
}

#[test]
fn test_reduce_mean_boundary_negative_axis() {
    // axis=-3 for ndim=3 should resolve to axis 0 (boundary case)
    let layer = ReduceMeanLayer::new(vec![-3], true);
    let resolved = layer.resolve_axes(3).unwrap();
    assert_eq!(resolved, vec![0]);
}

#[test]
fn test_reduce_sum_boundary_negative_axis() {
    // axis=-3 for ndim=3 should resolve to axis 0 (boundary case)
    let layer = ReduceSumLayer::new(vec![-3], true);
    let resolved = layer.resolve_axes(3).unwrap();
    assert_eq!(resolved, vec![0]);
}

// ========== Batched CROWN backward tests (#3221) ==========

use crate::BatchedLinearBounds;

#[test]
fn test_reduce_mean_batched_crown_last_axis_1d() {
    // Input shape [4], reduce axis -1 keepdims -> output [1]
    // Batched CROWN: expand in_dim from 1 to 4, scale by 1/4
    let layer = ReduceMeanLayer::last_axis();

    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0; 4]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).unwrap();
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity bounds at output: A=[1x1], b=[0]
    // input_shape=[1], output_shape=[1]
    let bounds = BatchedLinearBounds::identity(&[1]).unwrap();

    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();

    // Result: A should be [1, 4] with all elements = 1/4
    assert_eq!(result.lower_a().shape(), &[1, 4]);
    assert_eq!(result.upper_a().shape(), &[1, 4]);
    let scale = 0.25; // 1/4
    for j in 0..4 {
        assert_close(
            result.lower_a()[[0, j]],
            scale,
            1e-6,
            format!("mean batched lower_a[0,{j}]"),
        );
        assert_close(
            result.upper_a()[[0, j]],
            scale,
            1e-6,
            format!("mean batched upper_a[0,{j}]"),
        );
    }
    // Bias unchanged (zeros)
    assert!(
        result.lower_b().iter().all(|&v| v == 0.0),
        "batched lower_b should be all zeros, got {:?}",
        result.lower_b()
    );
    assert!(
        result.upper_b().iter().all(|&v| v == 0.0),
        "batched upper_b should be all zeros, got {:?}",
        result.upper_b()
    );
    // Input shape updated
    assert_eq!(result.input_shape(), &[4]);
}

#[test]
fn test_reduce_mean_batched_crown_2d() {
    // Input shape [2, 4], reduce last axis keepdims -> output [2, 1]
    // Batched CROWN: A goes from [2, out_dim, 1] to [2, out_dim, 4]
    let layer = ReduceMeanLayer::last_axis();

    let pre_lower = ArrayD::zeros(IxDyn(&[2, 4]));
    let pre_upper = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0);
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity at [2, 1]: A shape [2, 1, 1], bias [2, 1]
    let bounds = BatchedLinearBounds::identity(&[2, 1]).unwrap();

    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();

    assert_eq!(result.lower_a().shape(), &[2, 1, 4]);
    assert_eq!(result.input_shape(), &[2, 4]);

    let scale = 0.25; // 1/4
    for b in 0..2 {
        for j in 0..4 {
            assert_close(
                result.lower_a()[[b, 0, j]],
                scale,
                1e-6,
                format!("mean batched lower_a[{b},0,{j}]"),
            );
        }
    }
}

#[test]
fn test_reduce_sum_batched_crown_last_axis() {
    // Input shape [3], reduce axis -1 keepdims -> output [1]
    // Sum: scale = 1.0 (no division)
    let layer = ReduceSumLayer::last_axis();

    let pre_lower = ArrayD::zeros(IxDyn(&[3]));
    let pre_upper = ArrayD::from_elem(IxDyn(&[3]), 1.0);
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = BatchedLinearBounds::identity(&[1]).unwrap();

    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();

    assert_eq!(result.lower_a().shape(), &[1, 3]);
    // Sum: all coefficients = 1.0 (identity * 1.0)
    for j in 0..3 {
        assert_close(
            result.lower_a()[[0, j]],
            1.0,
            1e-6,
            format!("sum batched lower_a[0,{j}]"),
        );
    }
    assert_eq!(result.input_shape(), &[3]);
}

#[test]
fn test_reduce_mean_batched_crown_soundness_corners() {
    // Verify batched CROWN bounds contain all corner evaluations.
    // Input: [1, 4] with bounds [-5, 5] per element.
    // ReduceMean(axis=-1, keepdims=true) → [1, 1]
    let layer = ReduceMeanLayer::last_axis();

    let pre_lower = ArrayD::from_elem(IxDyn(&[1, 4]), -5.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1, 4]), 5.0f32);
    let pre_act = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let bounds = BatchedLinearBounds::identity(&[1, 1]).unwrap();
    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();

    // Concretize to get concrete bounds
    let input_bounds = BoundedTensor::new(pre_lower, pre_upper).unwrap();
    let concrete = result.concretize(&input_bounds).unwrap();

    // Check all 16 corners
    for mask in 0..16u32 {
        let mut mean = 0.0f32;
        for i in 0..4 {
            let val = if mask & (1 << i) != 0 { 5.0 } else { -5.0 };
            mean += val;
        }
        mean /= 4.0;

        assert!(
            concrete.lower()[[0, 0]] <= mean + 1e-4,
            "Lower {} > corner mean {} at mask {}",
            concrete.lower()[[0, 0]],
            mean,
            mask
        );
        assert!(
            concrete.upper()[[0, 0]] >= mean - 1e-4,
            "Upper {} < corner mean {} at mask {}",
            concrete.upper()[[0, 0]],
            mean,
            mask
        );
    }

    // Bounds should be tight: for [-5,5]^4, mean range is [-5, 5]
    assert!(
        (concrete.lower()[[0, 0]] - (-5.0)).abs() < 1e-3,
        "Lower bound should be -5.0, got {}",
        concrete.lower()[[0, 0]]
    );
    assert!(
        (concrete.upper()[[0, 0]] - 5.0).abs() < 1e-3,
        "Upper bound should be 5.0, got {}",
        concrete.upper()[[0, 0]]
    );
}

#[test]
fn test_reduce_mean_batched_crown_with_nonidentity_coefficients() {
    // Test with non-identity incoming coefficients (simulates downstream layers).
    // Downstream: y = 2*z + 0.5 (z is the ReduceMean output)
    let layer = ReduceMeanLayer::last_axis();

    let pre_lower = ArrayD::zeros(IxDyn(&[4]));
    let pre_upper = ArrayD::from_elem(IxDyn(&[4]), 4.0);
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // bounds: A=[[2.0]], b=[0.5] for both lower and upper
    let la = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![2.0]).unwrap();
    let lb = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap();
    let ua = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![2.0]).unwrap();
    let ub = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap();
    let bounds = BatchedLinearBounds::new(la, lb, ua, ub, vec![1], vec![1]).unwrap();

    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();

    // Each coefficient should be 2.0 * (1/4) = 0.5
    assert_eq!(result.lower_a().shape(), &[1, 4]);
    for j in 0..4 {
        assert_close(
            result.lower_a()[[0, j]],
            0.5,
            1e-6,
            format!("mean nonidentity lower_a[0,{j}]"),
        );
    }
    // Bias unchanged at 0.5
    assert_close(
        result.lower_b()[[0]],
        0.5,
        1e-6,
        "mean nonidentity lower_b[0]",
    );
}

#[test]
fn test_reduce_mean_batched_crown_non_last_axis_rejected() {
    // Non-last-axis reduction should return UnsupportedOp
    let layer = ReduceMeanLayer::new(vec![0], true); // axis 0, not last

    let pre_lower = ArrayD::zeros(IxDyn(&[3, 4]));
    let pre_upper = ArrayD::from_elem(IxDyn(&[3, 4]), 1.0);
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = BatchedLinearBounds::identity(&[1, 4]).unwrap();
    let result = layer.propagate_linear_batched(&bounds, &pre_act);

    assert!(
        result.is_err(),
        "ReduceMean batched CROWN should reject non-last-axis reductions",
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("last-axis"),
        "Error should mention last-axis restriction: {msg}"
    );
}

#[test]
fn test_reduce_mean_batched_crown_keepdims_false_rejected() {
    // keepdims=false should return UnsupportedOp
    let layer = ReduceMeanLayer::new(vec![-1], false);

    let pre_lower = ArrayD::zeros(IxDyn(&[4]));
    let pre_upper = ArrayD::from_elem(IxDyn(&[4]), 1.0);
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = BatchedLinearBounds::identity(&[1]).unwrap();
    let result = layer.propagate_linear_batched(&bounds, &pre_act);

    assert!(
        result.is_err(),
        "ReduceMean batched CROWN should reject keepdims=false",
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("keepdims"),
        "Error should mention keepdims restriction: {msg}"
    );
}

#[test]
fn test_reduce_mean_vs_sum_batched_consistency() {
    // Batched CROWN: ReduceMean coefficients should be ReduceSum / n
    let mean_layer = ReduceMeanLayer::last_axis();
    let sum_layer = ReduceSumLayer::last_axis();

    let pre_lower = ArrayD::zeros(IxDyn(&[6]));
    let pre_upper = ArrayD::from_elem(IxDyn(&[6]), 1.0);
    let pre_act = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let bounds = BatchedLinearBounds::identity(&[1]).unwrap();
    let mean_result = mean_layer
        .propagate_linear_batched(&bounds, &pre_act)
        .unwrap();
    let sum_result = sum_layer
        .propagate_linear_batched(&bounds, &pre_act)
        .unwrap();

    let n = 6.0f32;
    for j in 0..6 {
        assert!(
            (mean_result.lower_a()[[0, j]] * n - sum_result.lower_a()[[0, j]]).abs() < 1e-5,
            "mean_a * n ({}) != sum_a ({}) at j={}",
            mean_result.lower_a()[[0, j]] * n,
            sum_result.lower_a()[[0, j]],
            j
        );
    }
}

// ========== Directed cast sign witnesses ==========

/// A sum of provably non-negative terms (e.g. a sum of squares) must keep an
/// exactly-zero lower bound: the outward f64->f32 directed cast alone would
/// step it one denormal below zero, which spuriously enters sqrt-negative-
/// domain handling and fails `>= 0` output specs downstream.
#[test]
fn test_reduce_sum_nonnegative_terms_keep_zero_lower_bound() {
    let layer = ReduceSumLayer::new(vec![0], false);
    let lower = ArrayD::zeros(IxDyn(&[4]));
    let upper = ArrayD::from_elem(IxDyn(&[4]), 2.0);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = layer.propagate_ibp(&input).unwrap();
    assert_eq!(
        output.lower()[[]],
        0.0,
        "sum of non-negative lower terms must not be stepped below zero"
    );
    assert!(output.upper()[[]] >= 8.0, "upper must enclose 4 * 2.0");
}

/// Symmetric witness on the upper side: a sum of provably non-positive terms
/// must keep an exactly-zero upper bound.
#[test]
fn test_reduce_sum_nonpositive_terms_keep_zero_upper_bound() {
    let layer = ReduceSumLayer::new(vec![0], false);
    let lower = ArrayD::from_elem(IxDyn(&[4]), -2.0);
    let upper = ArrayD::zeros(IxDyn(&[4]));
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = layer.propagate_ibp(&input).unwrap();
    assert_eq!(
        output.upper()[[]],
        0.0,
        "sum of non-positive upper terms must not be stepped above zero"
    );
    assert!(output.lower()[[]] <= -8.0, "lower must enclose 4 * -2.0");
}

/// Mean over non-negative terms (variance = mean of squared deviations) keeps
/// its zero lower bound; mixed-sign terms still get the plain outward step.
#[test]
fn test_reduce_mean_sign_witness_zero_lower_and_mixed_sign_step() {
    let layer = ReduceMeanLayer::new(vec![0], false);

    // All lower terms >= 0 -> clamped at zero.
    let input = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[3])),
        ArrayD::from_elem(IxDyn(&[3]), 1.0),
    )
    .unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    assert_eq!(
        output.lower()[[]],
        0.0,
        "mean of non-negative lower terms must not be stepped below zero"
    );

    // Mixed-sign lower terms -> no witness; outward step must still apply
    // (enclosure of the true mean 1/3 past the f32 grid).
    let mixed = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0f32, 1.0, 1.0]).unwrap(),
        ArrayD::from_elem(IxDyn(&[3]), 2.0),
    )
    .unwrap();
    let mixed_out = layer.propagate_ibp(&mixed).unwrap();
    assert!(
        mixed_out.lower()[[]] <= 1.0 / 3.0,
        "mixed-sign mean lower bound must enclose the true mean"
    );
}
