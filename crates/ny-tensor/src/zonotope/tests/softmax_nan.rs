// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NaN safety tests for zonotope softmax (#2676 Site 1).
//! Split from softmax.rs to stay under 1000-line limit.

use super::super::*;
use ndarray::{ArrayD, IxDyn};

/// #2676 Site 1: softmax with NaN in center should produce finite output
/// (uniform distribution fallback), not propagate NaN through division.
#[test]
fn test_softmax_affine_nan_center_returns_uniform_2676() {
    // Create a zonotope with NaN in one element of the center.
    let data: Vec<f32> = vec![
        1.0,
        f32::NAN,
        2.0, // center: [1, NaN, 2]
    ];
    let coeffs = ArrayD::from_shape_vec(IxDyn(&[1, 3]), data).unwrap();
    let z = ZonotopeTensor::new(coeffs).unwrap();
    assert_eq!(z.n_error_terms, 0);

    let result = z.softmax_affine(-1).unwrap();
    let center = result.center();

    // With NaN guard: compute_softmax should return uniform [1/3, 1/3, 1/3].
    // Without guard: NaN propagates through exp/sum/division, all outputs NaN.
    for i in 0..3 {
        assert!(
            center[i].is_finite(),
            "#2676: softmax center[{}] should be finite with NaN input, got {}",
            i,
            center[i]
        );
    }

    // Uniform distribution: each element ≈ 1/3
    let expected = 1.0 / 3.0;
    for i in 0..3 {
        assert!(
            (center[i] - expected).abs() < 1e-5,
            "#2676: softmax center[{}] should be uniform 1/3 on NaN input, got {}",
            i,
            center[i]
        );
    }
}

/// #2676 Site 1: causal softmax with NaN in center should produce finite output.
#[test]
fn test_softmax_affine_causal_nan_center_returns_uniform_2676() {
    // Create a 2x2 zonotope with NaN in one entry.
    let data: Vec<f32> = vec![
        f32::NAN,
        1.0, // row 0: [NaN, 1.0]
        2.0,
        3.0, // row 1: [2.0, 3.0]
    ];
    let coeffs = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), data).unwrap();
    let z = ZonotopeTensor::new(coeffs).unwrap();

    let result = z.softmax_affine_causal(-1).unwrap();
    let center = result.center();

    // Row 0 (query 0): sees only col 0. softmax([NaN]) should be [1.0].
    // NaN guard: uniform [1/1] = [1.0].
    assert!(
        center[[0, 0]].is_finite(),
        "#2676: causal softmax[0,0] should be finite, got {}",
        center[[0, 0]]
    );

    // Row 1 (query 1): sees cols 0..=1 = [2.0, 3.0]. Normal softmax.
    assert!(
        center[[1, 0]].is_finite() && center[[1, 1]].is_finite(),
        "#2676: causal softmax row 1 should be finite"
    );
}

/// #2676 Site 1: softmax 2D path with NaN in one row should handle gracefully.
#[test]
fn test_softmax_affine_2d_nan_center_returns_uniform_2676() {
    // Create a 2x3 zonotope where row 1 has NaN.
    let data: Vec<f32> = vec![
        1.0,
        2.0,
        3.0, // row 0: normal
        f32::NAN,
        1.0,
        2.0, // row 1: NaN in first element
    ];
    let coeffs = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), data).unwrap();
    let z = ZonotopeTensor::new(coeffs).unwrap();

    let result = z.softmax_affine(-1).unwrap();
    let center = result.center();

    // Row 0 should be normal softmax.
    let e1 = 1.0_f32.exp();
    let e2 = 2.0_f32.exp();
    let e3 = 3.0_f32.exp();
    let sum = e1 + e2 + e3;
    assert!(
        (center[[0, 0]] - e1 / sum).abs() < 1e-5,
        "#2676: row 0 should be normal softmax"
    );

    // Row 1 should be uniform [1/3, 1/3, 1/3] due to NaN guard.
    let expected = 1.0 / 3.0;
    for d in 0..3 {
        assert!(
            center[[1, d]].is_finite(),
            "#2676: softmax 2D row 1 col {} should be finite with NaN input, got {}",
            d,
            center[[1, d]]
        );
        assert!(
            (center[[1, d]] - expected).abs() < 1e-5,
            "#2676: softmax 2D row 1 col {} should be uniform 1/3 on NaN input, got {}",
            d,
            center[[1, d]]
        );
    }
}
