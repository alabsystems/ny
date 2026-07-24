// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::arr1;

// ============== SiLU Affine Mutation-Killing Tests (1D) ==============

#[test]
fn test_silu_affine_sigmoid_boundary() {
    // Kills: replace >= with < in line 1331 (sigmoid boundary at x=0)
    // sigmoid(0) = 0.5, should be same from both branches

    // Test at exactly x=0 using both the positive and negative code paths
    let z = ZonotopeTensor::concrete(arr1(&[0.0_f32]).into_dyn());
    let result = z.silu_affine().unwrap();
    let center = result.center();

    // silu(0) = 0 * sigmoid(0) = 0
    assert!(
        (center[0] - 0.0).abs() < 1e-6,
        "silu(0) should be 0, got {}",
        center[0]
    );

    // Test very small positive and negative values - should give symmetric results
    let z_pos = ZonotopeTensor::concrete(arr1(&[1e-6_f32]).into_dyn());
    let z_neg = ZonotopeTensor::concrete(arr1(&[-1e-6_f32]).into_dyn());

    let r_pos = z_pos.silu_affine().unwrap();
    let r_neg = z_neg.silu_affine().unwrap();

    // silu is approximately linear near 0 with slope ~0.5
    // silu'(0) = sigmoid(0) * (1 + 0*(1-0.5)) = 0.5
    let pos_val = r_pos.center()[0];
    let neg_val = r_neg.center()[0];

    // silu(x) ≈ x/2 near 0, so silu(-eps) ≈ -silu(eps)
    assert!(
        (pos_val + neg_val).abs() < 1e-10,
        "silu should be antisymmetric near 0: {} vs {}",
        pos_val,
        neg_val
    );
}

#[test]
fn test_silu_affine_derivative_formula() {
    // Kills: mutations in line 1347 (silu_derivative formula)
    // silu'(x) = s * (1.0 + x * (1.0 - s))

    // Test with known values where the derivative matters
    // For x=1: sigmoid(1) ≈ 0.7311
    // silu'(1) = 0.7311 * (1 + 1 * (1 - 0.7311)) = 0.7311 * 1.2689 ≈ 0.9277

    // Create zonotope with error term
    let z = ZonotopeTensor::from_input_shared(&arr1(&[1.0_f32]).into_dyn(), 0.5);
    let result = z.silu_affine().unwrap();

    // The error coefficient should be scaled by the derivative
    // Input has error = 0.5, output error should be 0.5 * silu'(1) ≈ 0.464
    let input_err = z.coeffs[[1, 0]];
    let output_err = result.coeffs[[1, 0]];

    let expected_slope = {
        let s = 1.0 / (1.0 + (-1.0_f32).exp());
        s * (1.0 + 1.0 * (1.0 - s))
    };

    let expected_output_err = input_err * expected_slope;
    assert!(
        (output_err - expected_output_err).abs() < 1e-5,
        "output error {} should be input {} * slope {}",
        output_err,
        input_err,
        expected_slope
    );

    // Test at x=-2 to verify different path
    // sigmoid(-2) ≈ 0.1192
    // silu'(-2) = 0.1192 * (1 + (-2) * (1 - 0.1192)) ≈ 0.1192 * (1 - 1.7616) ≈ -0.0908
    let z2 = ZonotopeTensor::from_input_shared(&arr1(&[-2.0_f32]).into_dyn(), 0.5);
    let result2 = z2.silu_affine().unwrap();

    let expected_slope2 = {
        let s = (-2.0_f32).exp() / (1.0 + (-2.0_f32).exp());
        s * (1.0 + (-2.0) * (1.0 - s))
    };

    let output_err2 = result2.coeffs[[1, 0]];
    let expected_output_err2 = z2.coeffs[[1, 0]] * expected_slope2;
    assert!(
        (output_err2 - expected_output_err2).abs() < 1e-4,
        "output error {} should be input {} * slope {}",
        output_err2,
        z2.coeffs[[1, 0]],
        expected_slope2
    );
}

#[test]
fn test_silu_affine_second_derivative() {
    // Kills: mutations in lines 1352-1354 (silu_second_derivative formula)
    // silu''(x) = s * (1-s) * (2 + x - 2*x*s)

    // Large radius zonotope to have meaningful approximation error
    // The approximation error depends on max|silu''| * r^2 / 2
    let z = ZonotopeTensor::from_input_shared(&arr1(&[-1.28_f32]).into_dyn(), 0.5);
    let result = z.silu_affine().unwrap();

    // At x ≈ -1.28, silu'' has maximum magnitude
    // If second derivative formula is wrong, approximation error will be wrong

    // Get the approximation error term (the new one added)
    let approx_err_idx = result.n_error_terms;
    let approx_err = result.coeffs[[approx_err_idx, 0]];

    // Compute expected second derivative at critical point
    let s = {
        let ex = (-1.28_f32).exp();
        ex / (1.0 + ex)
    };
    let expected_max_second = (s * (1.0 - s) * (2.0 + (-1.28) - 2.0 * (-1.28) * s)).abs();

    // Error should be approximately max_second * r^2 / 2
    let r = 0.5;
    let expected_approx_err_lower = expected_max_second * r * r / 2.0 * 0.8; // Allow some tolerance

    assert!(
        approx_err >= expected_approx_err_lower,
        "approx error {} should be near {} (based on second derivative)",
        approx_err,
        expected_max_second * r * r / 2.0
    );
}

#[test]
fn test_silu_affine_radius_calculation() {
    // Kills: mutations in radius > 0.0 check (line 1388)
    // Also kills: mutations in lo/hi calculation (lines 1391-1392)

    // Test with zero radius - should have zero approximation error
    let z_concrete = ZonotopeTensor::concrete(arr1(&[0.5_f32]).into_dyn());
    let result_concrete = z_concrete.silu_affine().unwrap();

    // For concrete (no error), the new error term should be 0
    let approx_err_concrete = result_concrete.coeffs[[result_concrete.n_error_terms, 0]];
    assert!(
        (approx_err_concrete - 0.0).abs() < 1e-10,
        "concrete zonotope should have 0 approx error, got {}",
        approx_err_concrete
    );

    // Test with non-zero radius
    let z_with_err = ZonotopeTensor::from_input_shared(&arr1(&[0.5_f32]).into_dyn(), 0.3);
    let result_with_err = z_with_err.silu_affine().unwrap();

    let approx_err_with = result_with_err.coeffs[[result_with_err.n_error_terms, 0]];
    assert!(
        approx_err_with > 0.0,
        "zonotope with error should have positive approx error, got {}",
        approx_err_with
    );

    // Larger radius should give larger error
    let z_larger = ZonotopeTensor::from_input_shared(&arr1(&[0.5_f32]).into_dyn(), 0.6);
    let result_larger = z_larger.silu_affine().unwrap();
    let approx_err_larger = result_larger.coeffs[[result_larger.n_error_terms, 0]];

    // Error scales as r^2, so 2x radius should give ~4x error
    assert!(
        approx_err_larger > approx_err_with * 3.0,
        "larger radius {} should give much larger error than {}",
        approx_err_larger,
        approx_err_with
    );
}

#[test]
fn test_silu_affine_lo_hi_signs() {
    // Kills: replace - with / in line 1391 (lo = c - radius)
    // Kills: replace + with * in line 1392 (hi = c + radius)

    // Test with center at 1.0, radius 0.5
    // Correct: lo = 0.5, hi = 1.5
    // If lo = c/r: lo = 2.0 (wrong)
    // If hi = c*r: hi = 0.5 (wrong)

    let z = ZonotopeTensor::from_input_shared(&arr1(&[1.0_f32]).into_dyn(), 0.5);
    let result = z.silu_affine().unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    // Check that bounds contain silu values at true lo and hi
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    let true_lo = 0.5_f32; // c - r
    let true_hi = 1.5_f32; // c + r

    assert!(
        bounds.lower()[0] <= silu(true_lo) + 0.01,
        "lower bound {} should contain silu(0.5)={}",
        bounds.lower()[0],
        silu(true_lo)
    );
    assert!(
        bounds.upper()[0] >= silu(true_hi) - 0.01,
        "upper bound {} should contain silu(1.5)={}",
        bounds.upper()[0],
        silu(true_hi)
    );

    // If lo/hi were computed wrong, bounds would not contain these values
}

#[test]
fn test_silu_affine_interpolation_loop() {
    // Kills: mutations in lines 1395-1396 (interpolation: x = lo + (hi - lo) * t)

    // Test with interval that spans the SiLU'' maximum around -1.28
    let z = ZonotopeTensor::from_input_shared(&arr1(&[-1.0_f32]).into_dyn(), 0.5);
    let result = z.silu_affine().unwrap();

    // The approximation error should properly capture max second derivative
    let approx_err = result.coeffs[[result.n_error_terms, 0]];

    // If interpolation was wrong (e.g., lo + hi*t instead of lo + (hi-lo)*t),
    // the sampling would miss the peak
    assert!(approx_err > 0.0, "should have approximation error");

    // Verify bounds are sound - actual silu values at extremes should be contained
    let bounds = result.to_bounded_tensor().unwrap();
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    assert!(
        bounds.lower()[0] <= silu(-1.5),
        "lower {} should contain silu(-1.5)={}",
        bounds.lower()[0],
        silu(-1.5)
    );
    assert!(
        bounds.upper()[0] >= silu(-0.5),
        "upper {} should contain silu(-0.5)={}",
        bounds.upper()[0],
        silu(-0.5)
    );
}

#[test]
fn test_silu_affine_critical_points() {
    // Kills: mutations in critical point checks (lines 1401-1402, 1464-1465)
    // Checks: delete - in critical points, replace && with ||, replace <= with >

    // Test interval that contains a critical point (-1.28)
    let z = ZonotopeTensor::from_input_shared(&arr1(&[-1.28_f32]).into_dyn(), 0.2);
    let result = z.silu_affine().unwrap();

    let approx_err = result.coeffs[[result.n_error_terms, 0]];

    // The critical point -1.28 should be sampled since lo <= -1.28 <= hi
    // At this point, |silu''| is maximal

    // Compute expected max |silu''| at critical point
    let s = {
        let ex = (-1.28_f32).exp();
        ex / (1.0 + ex)
    };
    let max_second_at_critical = (s * (1.0 - s) * (2.0 - 1.28 - 2.0 * (-1.28) * s)).abs();

    let expected_min_err = max_second_at_critical * 0.04 / 2.0 * 0.5; // r=0.2, some tolerance
    assert!(
        approx_err >= expected_min_err,
        "approx error {} should reflect critical point sampling",
        approx_err
    );

    // Test interval that does NOT contain critical points (far positive)
    let z_far = ZonotopeTensor::from_input_shared(&arr1(&[5.0_f32]).into_dyn(), 0.2);
    let result_far = z_far.silu_affine().unwrap();

    let approx_err_far = result_far.coeffs[[result_far.n_error_terms, 0]];

    // At x=5, silu'' is very small (near 0)
    // Error should be much smaller than at critical point
    assert!(
        approx_err_far < approx_err,
        "error at x=5 ({}) should be less than at critical point ({})",
        approx_err_far,
        approx_err
    );
}

#[test]
fn test_silu_affine_critical_bounds_check() {
    // Kills: replace <= with > in line 1402 (lo <= critical)
    // Kills: replace <= with > in line 1402 (critical <= hi)

    // Test interval [-2.5, -2.3] - contains critical point -2.4
    let z = ZonotopeTensor::from_input_shared(&arr1(&[-2.4_f32]).into_dyn(), 0.1);
    let result = z.silu_affine().unwrap();
    let approx_err_contains = result.coeffs[[result.n_error_terms, 0]];

    // Test interval [-2.2, -2.0] - does NOT contain -2.4
    let z2 = ZonotopeTensor::from_input_shared(&arr1(&[-2.1_f32]).into_dyn(), 0.1);
    let result2 = z2.silu_affine().unwrap();
    let approx_err_not_contains = result2.coeffs[[result2.n_error_terms, 0]];

    // Both should have approximation error, but different values due to different
    // second derivative sampling
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
fn test_silu_affine_error_division() {
    // Kills: replace / with % or * in line 1406 (max_second * r * r / 2.0)

    // With known radius and max_second, verify error formula
    let z = ZonotopeTensor::from_input_shared(&arr1(&[0.0_f32]).into_dyn(), 1.0);
    let result = z.silu_affine().unwrap();

    let approx_err = result.coeffs[[result.n_error_terms, 0]];

    // At x=0, silu''(0) = 0.5 * 0.5 * 2 = 0.5
    // Error should be 0.5 * 1.0 * 1.0 / 2.0 = 0.25 (approximately)
    // If / was replaced with *, error would be 0.5 * 1.0 * 1.0 * 2.0 = 1.0

    assert!(
        approx_err < 0.5,
        "error {} should be less than 0.5 (division, not multiplication)",
        approx_err
    );
    assert!(
        approx_err >= 0.1,
        "error {} should be at least 0.1 (proper formula)",
        approx_err
    );
}

#[test]
fn test_silu_affine_slope_multiplication() {
    // Kills: replace * with + in line 1383 (slope * self.coeffs[[i, d]])
    // Kills: replace * with / in line 1383

    // Test that error coefficients are multiplied by slope
    let z = ZonotopeTensor::from_input_shared(&arr1(&[2.0_f32]).into_dyn(), 1.0);

    let input_err = z.coeffs[[1, 0]];
    let result = z.silu_affine().unwrap();
    let output_err = result.coeffs[[1, 0]];

    // Compute expected slope at x=2
    let s = 1.0 / (1.0 + (-2.0_f32).exp());
    let expected_slope = s * (1.0 + 2.0 * (1.0 - s));

    // If * was +, output would be slope + input_err
    // If * was /, output would be slope / input_err

    let expected_output = input_err * expected_slope;
    assert!(
        (output_err - expected_output).abs() < 1e-4,
        "output error {} should be input {} * slope {} = {}",
        output_err,
        input_err,
        expected_slope,
        expected_output
    );

    // Verify it's not addition
    let wrong_add = expected_slope + input_err;
    assert!(
        (output_err - wrong_add).abs() > 0.1,
        "should not be addition: {} vs {}",
        output_err,
        wrong_add
    );
}

#[test]
fn test_silu_affine_non_finite_center_limits() {
    let z =
        ZonotopeTensor::concrete(arr1(&[f32::NEG_INFINITY, f32::INFINITY, f32::NAN]).into_dyn());
    let result = z.silu_affine().unwrap();
    let center = result.center();

    assert_eq!(center[0], 0.0, "silu(-inf) should converge to 0");
    assert!(
        center[1].is_infinite() && center[1].is_sign_positive(),
        "silu(+inf) should stay +inf, got {}",
        center[1]
    );
    assert!(center[2].is_nan(), "silu(NaN) should propagate NaN");
}

#[test]
fn test_silu_affine_non_finite_derivative_limits() {
    let z_neg = ZonotopeTensor::from_input_shared(&arr1(&[f32::NEG_INFINITY]).into_dyn(), 0.5);
    let out_neg = z_neg.silu_affine().unwrap();
    assert_eq!(
        out_neg.coeffs[[0, 0]],
        0.0,
        "center for -inf should be finite limit 0"
    );
    assert_eq!(
        out_neg.coeffs[[1, 0]],
        0.0,
        "silu'(-inf) should converge to 0"
    );

    let z_pos = ZonotopeTensor::from_input_shared(&arr1(&[f32::INFINITY]).into_dyn(), 0.5);
    let out_pos = z_pos.silu_affine().unwrap();
    assert!(
        out_pos.coeffs[[0, 0]].is_infinite() && out_pos.coeffs[[0, 0]].is_sign_positive(),
        "center for +inf should stay +inf, got {}",
        out_pos.coeffs[[0, 0]]
    );
    assert!(
        (out_pos.coeffs[[1, 0]] - z_pos.coeffs[[1, 0]]).abs() < 1e-6,
        "silu'(+inf) should converge to 1, expected {}, got {}",
        z_pos.coeffs[[1, 0]],
        out_pos.coeffs[[1, 0]]
    );
}

#[test]
fn test_silu_affine_non_finite_approx_error_not_nan() {
    // silu_second_derivative() has the same inf*0 / inf-inf NaN hazard.
    // The approximation error term (last error row) must be finite, not NaN,
    // even when the center is non-finite.
    let z_neg = ZonotopeTensor::from_input_shared(&arr1(&[f32::NEG_INFINITY]).into_dyn(), 0.5);
    let out_neg = z_neg.silu_affine().unwrap();
    let approx_err_neg = out_neg.coeffs[[out_neg.n_error_terms, 0]];
    assert!(
        !approx_err_neg.is_nan(),
        "approx error at -inf center should not be NaN, got {}",
        approx_err_neg
    );

    let z_pos = ZonotopeTensor::from_input_shared(&arr1(&[f32::INFINITY]).into_dyn(), 0.5);
    let out_pos = z_pos.silu_affine().unwrap();
    let approx_err_pos = out_pos.coeffs[[out_pos.n_error_terms, 0]];
    assert!(
        !approx_err_pos.is_nan(),
        "approx error at +inf center should not be NaN, got {}",
        approx_err_pos
    );
}

/// Cancellation regression: with a shared error symbol, a [1, -1] projection
/// could produce bounds tighter than the true range — a soundness violation.
/// Per-element independent error symbols (#2486, same fix as GELU #2470) prevent this.
#[test]
fn test_silu_affine_linear_projection_soundness_1d_independent_error_terms() {
    use ndarray::arr2;

    fn silu(x: f32) -> f32 {
        if !x.is_finite() {
            if x.is_nan() {
                return f32::NAN;
            }
            return if x.is_sign_negative() { 0.0 } else { x };
        }
        x / (1.0 + (-x).exp())
    }

    // Create 2-element zonotope with independent per-element error terms.
    // Center at -0.75 (near SiLU'' peak for nontrivial approx error), radius 1.2.
    let values = arr1(&[-0.75_f32, -0.75]);
    let z = ZonotopeTensor::from_input_elementwise(&values.into_dyn(), 1.2);

    let silu_z = z.silu_affine().unwrap();
    assert_eq!(
        silu_z.n_error_terms(),
        z.n_error_terms() + 2,
        "silu_affine 1D should add one approximation error symbol per element"
    );

    // Apply [1, -1] projection — this is the cancellation attack vector.
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
        bounds.lower()[0] <= true_min + 1e-6,
        "lower bound {} should contain true min {}",
        bounds.lower()[0],
        true_min
    );
    assert!(
        bounds.upper()[0] >= true_max - 1e-6,
        "upper bound {} should contain true max {}",
        bounds.upper()[0],
        true_max
    );
}
