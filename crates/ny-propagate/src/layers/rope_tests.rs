// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the RoPE (Rotary Position Embedding) layer.

use super::rope::RopeLayer;
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;
use ndarray::{array, Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

/// Test RoPE construction with valid frequencies.
#[test]
fn test_rope_construction() {
    let layer = RopeLayer::new(vec![1.0, 0.0], vec![0.0, 1.0]).unwrap();
    assert_eq!(layer.num_pairs(), 2);
    assert_eq!(layer.head_dim(), 4);
}

/// Test RoPE construction rejects mismatched lengths.
#[test]
fn test_rope_construction_mismatch() {
    let err = RopeLayer::new(vec![1.0], vec![0.0, 1.0]).unwrap_err();
    assert!(err.to_string().contains("length"));
}

/// Test RoPE construction rejects empty frequencies.
#[test]
fn test_rope_construction_empty() {
    let err = RopeLayer::new(vec![], vec![]).unwrap_err();
    assert!(err.to_string().contains("at least one"));
}

/// Test RoPE construction rejects non-finite frequencies.
#[test]
fn test_rope_construction_nonfinite() {
    let err = RopeLayer::new(vec![f32::NAN], vec![0.0]).unwrap_err();
    assert!(err.to_string().contains("non-finite"));
}

/// Test RoPE construction rejects finite but non-rotational frequency pairs.
#[test]
fn test_rope_construction_rejects_non_unit_rotation() {
    let err = RopeLayer::new(vec![1.0], vec![1.0]).unwrap_err();
    assert!(
        err.to_string().contains("unit-rotation invariant") && err.to_string().contains("norm_sq"),
        "expected unit-rotation error, got: {err}"
    );
}

/// Test RoPE construction accepts valid rotations quantized through bf16 storage.
#[test]
fn test_rope_construction_accepts_bf16_quantized_rotation() {
    // cos(1.0) and sin(1.0) rounded independently to bf16, then decoded to f32.
    let layer = RopeLayer::new(vec![0.539_062_5], vec![0.839_843_75])
        .expect("bf16-rounded RoPE tables should remain valid");
    assert_eq!(layer.num_pairs(), 1);
    assert_eq!(layer.head_dim(), 2);
}

/// Test RoPE from_position creates valid layer.
#[test]
fn test_rope_from_position() {
    let layer = RopeLayer::from_position(0, 4, 10000.0).unwrap();
    assert_eq!(layer.head_dim(), 4);
    // Position 0 → all angles are 0 → cos=1, sin=0 → identity
    assert!((layer.cos_freqs[0] - 1.0).abs() < 1e-6);
    assert!(layer.sin_freqs[0].abs() < 1e-6);
}

/// Test RoPE from_position rejects invalid base values.
#[test]
fn test_rope_from_position_bad_base() {
    let err = RopeLayer::from_position(1, 4, 0.0).unwrap_err();
    assert!(
        err.to_string().contains("base"),
        "expected base error, got: {err}"
    );
    let err = RopeLayer::from_position(1, 4, -1.0).unwrap_err();
    assert!(
        err.to_string().contains("base"),
        "expected base error, got: {err}"
    );
    let err = RopeLayer::from_position(1, 4, f32::NAN).unwrap_err();
    assert!(
        err.to_string().contains("base"),
        "expected base error, got: {err}"
    );
}

/// Test RoPE from_position rejects odd head_dim.
#[test]
fn test_rope_from_position_odd_head_dim() {
    let err = RopeLayer::from_position(0, 3, 10000.0).unwrap_err();
    assert!(
        err.to_string().contains("even"),
        "expected even error, got: {err}"
    );
    let err = RopeLayer::from_position(0, 0, 10000.0).unwrap_err();
    assert!(
        err.to_string().contains("positive"),
        "expected positive error, got: {err}"
    );
}

/// Test IBP with identity rotation (position=0, all angles=0).
/// cos(0) = 1, sin(0) = 0, so y = x (identity).
#[test]
fn test_ibp_identity_rotation() {
    let layer = RopeLayer::new(vec![1.0, 1.0], vec![0.0, 0.0]).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, -2.0, -3.0, -4.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    )
    .unwrap();

    let output = layer.propagate_ibp(&input).unwrap();
    // Identity rotation: output bounds should equal input bounds.
    for (o, i) in output.lower().iter().zip(input.lower().iter()) {
        assert!((o - i).abs() < 1e-6, "lower: {o} != {i}");
    }
    for (o, i) in output.upper().iter().zip(input.upper().iter()) {
        assert!((o - i).abs() < 1e-6, "upper: {o} != {i}");
    }
}

/// Test IBP with 90-degree rotation: cos(π/2) ≈ 0, sin(π/2) ≈ 1.
/// y[0] = 0*x[0] - 1*x[1] = -x[1]
/// y[1] = 1*x[0] + 0*x[1] = x[0]
#[test]
fn test_ibp_90_degree_rotation() {
    let angle = std::f32::consts::FRAC_PI_2;
    let layer = RopeLayer::new(vec![angle.cos()], vec![angle.sin()]).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 5.0]).unwrap(),
    )
    .unwrap();

    let output = layer.propagate_ibp(&input).unwrap();
    // y[0] = -x[1], so y[0] ∈ [-5, -2]
    assert!((output.lower()[[0]] - (-5.0)).abs() < 1e-5);
    assert!((output.upper()[[0]] - (-2.0)).abs() < 1e-5);
    // y[1] = x[0], so y[1] ∈ [1, 3]
    assert!((output.lower()[[1]] - 1.0).abs() < 1e-5);
    assert!((output.upper()[[1]] - 3.0).abs() < 1e-5);
}

/// Test IBP with a 45-degree rotation to verify soundness.
/// Verify that the true rotation output at corners is within the computed bounds.
#[test]
fn test_ibp_45_degree_soundness() {
    let angle = std::f32::consts::FRAC_PI_4;
    let c = angle.cos();
    let s = angle.sin();
    let layer = RopeLayer::new(vec![c], vec![s]).unwrap();

    let x0_lo = -1.0f32;
    let x0_hi = 2.0f32;
    let x1_lo = -0.5f32;
    let x1_hi = 1.5f32;

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![x0_lo, x1_lo]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![x0_hi, x1_hi]).unwrap(),
    )
    .unwrap();

    let output = layer.propagate_ibp(&input).unwrap();

    // Check all 4 corners of the input box
    for &x0 in &[x0_lo, x0_hi] {
        for &x1 in &[x1_lo, x1_hi] {
            let y0 = c * x0 - s * x1;
            let y1 = s * x0 + c * x1;
            assert!(
                y0 >= output.lower()[[0]] - 1e-5,
                "y0={y0} < lower={}, x0={x0}, x1={x1}",
                output.lower()[[0]]
            );
            assert!(
                y0 <= output.upper()[[0]] + 1e-5,
                "y0={y0} > upper={}, x0={x0}, x1={x1}",
                output.upper()[[0]]
            );
            assert!(
                y1 >= output.lower()[[1]] - 1e-5,
                "y1={y1} < lower={}, x0={x0}, x1={x1}",
                output.lower()[[1]]
            );
            assert!(
                y1 <= output.upper()[[1]] + 1e-5,
                "y1={y1} > upper={}, x0={x0}, x1={x1}",
                output.upper()[[1]]
            );
        }
    }
}

/// Test IBP with batched input (multiple vectors).
/// Identity rotation (cos=1, sin=0) on shape [2, 2]: output should equal input.
#[test]
fn test_ibp_batched_input() {
    let layer = RopeLayer::new(vec![1.0], vec![0.0]).unwrap();
    // Shape [2, 2]: two vectors of head_dim=2
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, -2.0, -3.0, -4.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    )
    .unwrap();

    let output = layer.propagate_ibp(&input).unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    // Identity rotation: verify bounds match per-vector (catches indexing bugs in batched loop)
    for (o, i) in output.lower().iter().zip(input.lower().iter()) {
        assert!((o - i).abs() < 1e-6, "batched lower: {o} != {i}");
    }
    for (o, i) in output.upper().iter().zip(input.upper().iter()) {
        assert!((o - i).abs() < 1e-6, "batched upper: {o} != {i}");
    }
}

/// Test CROWN backward with identity rotation.
#[test]
fn test_crown_identity_rotation() {
    let layer = RopeLayer::new(vec![1.0], vec![0.0]).unwrap();
    let bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear(&bounds).unwrap();

    // Identity rotation → bounds should be unchanged.
    for j in 0..2 {
        for i in 0..2 {
            assert!(
                (result.lower_a()[[j, i]] - bounds.lower_a()[[j, i]]).abs() < 1e-6,
                "lower_a[{j},{i}] mismatch"
            );
            assert!(
                (result.upper_a()[[j, i]] - bounds.upper_a()[[j, i]]).abs() < 1e-6,
                "upper_a[{j},{i}] mismatch"
            );
        }
    }
}

/// Test CROWN backward with 90-degree rotation.
/// R = [0 -1; 1 0], so A @ R swaps and negates columns.
#[test]
fn test_crown_90_degree_rotation() {
    let angle = std::f32::consts::FRAC_PI_2;
    let layer = RopeLayer::new(vec![angle.cos()], vec![angle.sin()]).unwrap();

    // A = I_2
    let bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear(&bounds).unwrap();

    // A @ R = I @ R = R
    // R = [cos(π/2) -sin(π/2); sin(π/2) cos(π/2)] ≈ [0 -1; 1 0]
    // So new_A[0, 0] = cos(π/2) ≈ 0, new_A[0, 1] = -sin(π/2) ≈ -1
    //    new_A[1, 0] = sin(π/2) ≈ 1, new_A[1, 1] = cos(π/2) ≈ 0

    // Wait, the CROWN backward computes: new_A = A @ R
    // Where R has the property that y = R @ x (forward).
    // For backward: we substitute y = R @ x into A @ y + b:
    //   A @ (R @ x) + b = (A @ R) @ x + b
    // So new_A = A @ R.
    //
    // With A = I: new_A = R
    // new_A[:, 0] = A[:, 0] * cos + A[:, 1] * sin = [cos, sin]
    // new_A[:, 1] = A[:, 0] * (-sin) + A[:, 1] * cos = [-sin, cos]

    let c = angle.cos(); // ≈ 0
    let s = angle.sin(); // ≈ 1

    assert!((result.lower_a()[[0, 0]] - c).abs() < 1e-5);
    assert!((result.lower_a()[[0, 1]] - (-s)).abs() < 1e-5);
    assert!((result.lower_a()[[1, 0]] - s).abs() < 1e-5);
    assert!((result.lower_a()[[1, 1]] - c).abs() < 1e-5);
}

/// Test CROWN backward preserves orthogonality.
/// Rotation matrices are orthogonal, so A @ R should preserve the column norms.
#[test]
fn test_crown_preserves_column_norms() {
    let angle = 0.7f32; // arbitrary angle
    let layer = RopeLayer::from_angles(&[angle]).unwrap();

    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let bounds =
        LinearBounds::new(a.clone(), Array1::zeros(2), a.clone(), Array1::zeros(2)).unwrap();

    let result = layer.propagate_linear(&bounds).unwrap();

    // Check that Frobenius norm is preserved (A @ R has same norm as A for orthogonal R)
    let orig_norm: f32 = a.iter().map(|x| x * x).sum();
    let new_norm: f32 = result.lower_a().iter().map(|x| x * x).sum();
    assert!(
        (orig_norm - new_norm).abs() < 1e-4,
        "Frobenius norm changed: {orig_norm} -> {new_norm}"
    );
}

/// Test that CROWN backward and IBP produce consistent results.
/// For linear layers, CROWN with identity spec should match IBP.
#[test]
fn test_crown_ibp_consistency() {
    let angle = 1.2f32;
    let layer = RopeLayer::from_angles(&[angle]).unwrap();

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap(),
    )
    .unwrap();

    // IBP bounds
    let ibp_output = layer.propagate_ibp(&input).unwrap();

    // CROWN with identity spec, then concretize
    let bounds = LinearBounds::identity(2);
    let crown_bounds = layer.propagate_linear(&bounds).unwrap();
    let crown_output = crown_bounds.concretize(&input);

    // For a linear layer, CROWN with identity spec should give exact same bounds as IBP.
    // Tolerance is ~1 ULP: IBP applies directed rounding (next_down/next_up) while
    // concretize uses plain f64→f32 cast, so IBP bounds are at most 1 ULP wider.
    for i in 0..2 {
        assert!(
            (ibp_output.lower()[[i]] - crown_output.lower()[[i]]).abs() < 1e-6,
            "lower[{i}]: IBP={} CROWN={} (diff={})",
            ibp_output.lower()[[i]],
            crown_output.lower()[[i]],
            (ibp_output.lower()[[i]] - crown_output.lower()[[i]]).abs()
        );
        assert!(
            (ibp_output.upper()[[i]] - crown_output.upper()[[i]]).abs() < 1e-6,
            "upper[{i}]: IBP={} CROWN={} (diff={})",
            ibp_output.upper()[[i]],
            crown_output.upper()[[i]],
            (ibp_output.upper()[[i]] - crown_output.upper()[[i]]).abs()
        );
    }
}

/// Test dimension mismatch errors.
#[test]
fn test_ibp_dimension_mismatch() {
    let layer = RopeLayer::new(vec![1.0], vec![0.0]).unwrap(); // head_dim = 2
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0; 3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0; 3]).unwrap(),
    )
    .unwrap();

    let err = layer.propagate_ibp(&input).unwrap_err();
    assert!(err.to_string().contains("Shape"));
}

/// Test CROWN dimension mismatch errors.
#[test]
fn test_crown_dimension_mismatch() {
    let layer = RopeLayer::new(vec![1.0], vec![0.0]).unwrap(); // head_dim = 2
    let bounds = LinearBounds::identity(3); // wrong dimension

    let err = layer.propagate_linear(&bounds).unwrap_err();
    assert!(err.to_string().contains("Shape"));
}
