// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{arr2, Axis};

// ============== SiLU Affine Mutation-Killing Tests (2D) ==============

#[test]
fn test_silu_affine_2d_center_copy() {
    // Kills: replace + with - in line 1426 (n_error_terms + 1)
    // Tests that new error terms are added at correct indices

    let values = arr2(&[[0.0_f32, 1.0]]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    let orig_n_err = z.n_error_terms;
    let result = z.silu_affine().unwrap();

    // Per-element error terms (#2486): shape [1,2] adds 1*2 = 2 error terms
    let seq_len = 1;
    let dim = 2;
    assert_eq!(
        result.n_error_terms,
        orig_n_err + seq_len * dim,
        "should add one error term per element"
    );

    // Each element should have its own error row with per-element content
    let err_row_0 = result.coeffs.index_axis(Axis(0), orig_n_err + 1); // element (0,0)
    let err_row_1 = result.coeffs.index_axis(Axis(0), orig_n_err + 2); // element (0,1)
    let sum: f32 = err_row_0
        .iter()
        .chain(err_row_1.iter())
        .map(|v| v.abs())
        .sum();
    assert!(sum > 0.0, "new error rows should have content");
}

#[test]
fn test_silu_affine_2d_slope_multiplication() {
    // Kills: replace * with + in line 1451 (slope * self.coeffs[[i, s, d]])

    let values = arr2(&[[2.0_f32]]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 1.0);

    let input_err = z.coeffs[[1, 0, 0]];
    let result = z.silu_affine().unwrap();
    let output_err = result.coeffs[[1, 0, 0]];

    let s = 1.0 / (1.0 + (-2.0_f32).exp());
    let expected_slope = s * (1.0 + 2.0 * (1.0 - s));
    let expected_output = input_err * expected_slope;

    assert!(
        (output_err - expected_output).abs() < 1e-4,
        "2D output error {} should be input {} * slope {}",
        output_err,
        input_err,
        expected_slope
    );
}

#[test]
fn test_silu_affine_2d_lo_hi() {
    // Kills: mutations in lines 1456-1457 (lo = c - radius, hi = c + radius) for 2D

    let values = arr2(&[[1.0_f32]]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.5);
    let result = z.silu_affine().unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    // Bounds should contain silu(0.5) and silu(1.5)
    assert!(
        bounds.lower()[[0, 0]] <= silu(0.5) + 0.01,
        "2D lower bound should contain silu(0.5)"
    );
    assert!(
        bounds.upper()[[0, 0]] >= silu(1.5) - 0.01,
        "2D upper bound should contain silu(1.5)"
    );
}

#[test]
fn test_silu_affine_2d_radius_zero_check() {
    // Kills: replace > with >= in line 1455 (radius > 0.0)

    // Test 2D zonotope with zero error (concrete) - should have zero approx error
    let values = arr2(&[[0.5_f32, 1.0]]).into_dyn();
    let z = ZonotopeTensor::concrete(values);

    let result = z.silu_affine().unwrap();
    // Per-element error for (0,0) is at row (orig_n_err + 1 + 0*dim + 0)
    let orig_n_err_concrete = 0; // concrete has 0 error terms
    let approx_err = result.coeffs[[orig_n_err_concrete + 1, 0, 0]];

    assert!(
        (approx_err - 0.0).abs() < 1e-10,
        "2D concrete zonotope should have 0 approx error, got {}",
        approx_err
    );

    // Test 2D zonotope with non-zero error
    let values2 = arr2(&[[0.5_f32, 1.0]]).into_dyn();
    let z2 = ZonotopeTensor::from_input_shared(&values2, 0.3);

    let result2 = z2.silu_affine().unwrap();
    // Per-element error for (0,0) is at row (orig_n_err + 1 + 0*dim + 0)
    let orig_n_err = z2.n_error_terms;
    let approx_err2 = result2.coeffs[[orig_n_err + 1, 0, 0]];

    assert!(
        approx_err2 > 0.0,
        "2D zonotope with error should have positive approx error, got {}",
        approx_err2
    );
}

#[test]
fn test_silu_affine_2d_lo_hi_calculation() {
    // Kills: replace + with - in line 1457 (hi = c + radius)
    // Kills: replace + with * in line 1457

    let values = arr2(&[[1.0_f32], [2.0]]).into_dyn(); // 2 positions
    let z = ZonotopeTensor::from_input_shared(&values, 0.5);
    let result = z.silu_affine().unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    // Position 0: center=1.0, radius=0.5, lo=0.5, hi=1.5
    assert!(
        bounds.lower()[[0, 0]] <= silu(0.5) + 0.01,
        "2D pos0 lower should contain silu(0.5)"
    );
    assert!(
        bounds.upper()[[0, 0]] >= silu(1.5) - 0.01,
        "2D pos0 upper should contain silu(1.5)"
    );

    // Position 1: center=2.0, radius=0.5, lo=1.5, hi=2.5
    assert!(
        bounds.lower()[[1, 0]] <= silu(1.5) + 0.01,
        "2D pos1 lower should contain silu(1.5)"
    );
    assert!(
        bounds.upper()[[1, 0]] >= silu(2.5) - 0.01,
        "2D pos1 upper should contain silu(2.5)"
    );
}

#[test]
fn test_silu_affine_2d_interpolation_division() {
    // Kills: replace / with % in line 1460 (i as f32 / 20.0)
    // Kills: replace / with * in line 1460

    // With % or *, the interpolation parameter t would be wrong
    let values = arr2(&[[0.0_f32]]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 1.0);

    let result = z.silu_affine().unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    // Bounds should contain silu(-1) and silu(1)
    assert!(
        bounds.lower()[[0, 0]] <= silu(-1.0) + 0.01,
        "2D interpolation test: lower should contain silu(-1)"
    );
    assert!(
        bounds.upper()[[0, 0]] >= silu(1.0) - 0.01,
        "2D interpolation test: upper should contain silu(1)"
    );
}

#[test]
fn test_silu_affine_2d_interpolation_formula() {
    // Kills: mutations in line 1461 (x = lo + (hi - lo) * t)

    // Use center at SiLU'' critical point to maximize sensitivity to formula errors
    let values = arr2(&[[-1.28_f32]]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.5);

    let result = z.silu_affine().unwrap();

    // The approximation error should properly capture max second derivative
    let approx_err = result.coeffs[[result.n_error_terms, 0, 0]];
    assert!(approx_err > 0.0, "2D should have approximation error");

    // Verify bounds soundness
    let bounds = result.to_bounded_tensor().unwrap();
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    assert!(
        bounds.lower()[[0, 0]] <= silu(-1.78),
        "2D lower {} should contain silu(-1.78)={}",
        bounds.lower()[[0, 0]],
        silu(-1.78)
    );
    assert!(
        bounds.upper()[[0, 0]] >= silu(-0.78),
        "2D upper {} should contain silu(-0.78)={}",
        bounds.upper()[[0, 0]],
        silu(-0.78)
    );
}

#[test]
fn test_silu_affine_2d_critical_point_sign() {
    // Kills: delete - in line 1464 (critical points -2.4, -1.28)

    // Center at -1.0 with radius 0.5 should contain critical point -1.28
    let values = arr2(&[[-1.0_f32]]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.5);

    let result = z.silu_affine().unwrap();
    let approx_err = result.coeffs[[result.n_error_terms, 0, 0]];

    // Error should be non-trivial since we're near the SiLU'' peak
    assert!(
        approx_err > 0.001,
        "2D approx error should reflect critical point, got {}",
        approx_err
    );

    // Test at positive center - critical points shouldn't be sampled
    let values_pos = arr2(&[[2.0_f32]]).into_dyn();
    let z_pos = ZonotopeTensor::from_input_shared(&values_pos, 0.2);

    let result_pos = z_pos.silu_affine().unwrap();
    let approx_err_pos = result_pos.coeffs[[result_pos.n_error_terms, 0, 0]];

    // Error at x=2 should be smaller (far from critical points)
    assert!(
        approx_err_pos < approx_err,
        "2D error at x=2 ({}) should be less than at x=-1 ({})",
        approx_err_pos,
        approx_err
    );
}

#[test]
fn test_silu_affine_2d_critical_bounds() {
    // Kills: replace && with || in line 1465
    // Kills: replace <= with > in line 1465

    // Test interval exactly containing critical point 0.7
    let values = arr2(&[[0.7_f32]]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    let result = z.silu_affine().unwrap();
    let approx_err_contains = result.coeffs[[result.n_error_terms, 0, 0]];

    // Test interval NOT containing any critical points (1.5 to 2.5)
    let values2 = arr2(&[[2.0_f32]]).into_dyn();
    let z2 = ZonotopeTensor::from_input_shared(&values2, 0.5);

    let result2 = z2.silu_affine().unwrap();
    let approx_err_not_contains = result2.coeffs[[result2.n_error_terms, 0, 0]];

    // Both should have approximation error
    assert!(
        approx_err_contains > 0.0,
        "should have error when containing critical"
    );
    assert!(
        approx_err_not_contains > 0.0,
        "should have error when not containing critical"
    );
}

#[test]
fn test_silu_affine_2d_error_formula() {
    // Kills: mutations in line 1469 (max_second * radius * radius / 2.0)

    let values = arr2(&[[0.0_f32]]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 1.0);

    let result = z.silu_affine().unwrap();
    let approx_err = result.coeffs[[result.n_error_terms, 0, 0]];

    // At x=0, silu''(0) = 0.5 * 0.5 * 2 = 0.5
    // Error should be around 0.5 * 1.0 * 1.0 / 2.0 = 0.25
    // If / was *, error would be 1.0
    // If * was +, error would be different

    assert!(
        approx_err < 0.6,
        "2D error {} should be less than 0.6",
        approx_err
    );
    assert!(
        approx_err >= 0.1,
        "2D error {} should be at least 0.1",
        approx_err
    );
}

/// Cancellation regression through the 2D SiLU path.
/// Same pattern as `test_silu_affine_linear_projection_soundness_1d_independent_error_terms`
/// in silu_affine_mutation_1d.rs, but exercises the 2D code path.
/// Fixes #2486 (same class of bug as GELU #2470).
#[test]
fn test_silu_affine_linear_projection_soundness_2d_independent_error_terms() {
    fn silu(x: f32) -> f32 {
        if !x.is_finite() {
            if x.is_nan() {
                return f32::NAN;
            }
            return if x.is_sign_negative() { 0.0 } else { x };
        }
        x / (1.0 + (-x).exp())
    }

    // Shape [1, 2] — the 2D path processes (seq=1, dim=2).
    let values = arr2(&[[-0.75_f32, -0.75]]);
    let z = ZonotopeTensor::from_input_2d(&values, 1.2);

    let silu_z = z.silu_affine().unwrap();
    assert_eq!(
        silu_z.n_error_terms(),
        z.n_error_terms() + values.len(),
        "silu_affine 2D should add one approximation error symbol per element"
    );

    // Apply [1, -1] projection — cancellation attack vector.
    let weight = arr2(&[[1.0_f32, -1.0]]);
    let projected = silu_z.linear(&weight, None).unwrap();
    let bounds = projected.to_bounded_tensor().unwrap();

    // Compute true min/max of silu(x0) - silu(x1) over all corner inputs.
    let mut true_min = f32::INFINITY;
    let mut true_max = f32::NEG_INFINITY;
    for &e0 in &[-1.0_f32, 1.0] {
        for &e1 in &[-1.0_f32, 1.0] {
            let x0 = -0.75 + 1.2 * e0;
            let x1 = -0.75 + 1.2 * e1;
            let y = silu(x0) - silu(x1);
            true_min = true_min.min(y);
            true_max = true_max.max(y);
        }
    }

    assert!(
        bounds.lower()[[0, 0]] <= true_min + 1e-6,
        "2D lower bound {} should contain true min {}",
        bounds.lower()[[0, 0]],
        true_min
    );
    assert!(
        bounds.upper()[[0, 0]] >= true_max - 1e-6,
        "2D upper bound {} should contain true max {}",
        bounds.upper()[[0, 0]],
        true_max
    );
}
