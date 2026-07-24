// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for `ZonotopeTensor::gelu_affine()`.
//! Regression tests for #2470: GELU zonotope must use GELU math, not SiLU.

use super::super::*;
use ndarray::{arr1, arr2};
use proptest::prelude::*;

fn erff_approx(x: f32) -> f32 {
    // Abramowitz & Stegun eq. 7.1.26, max error ~1.5e-7.
    let sign = x.signum();
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-a * a).exp())
}

fn gelu_erf(x: f32) -> f32 {
    let inv_sqrt2: f32 = 1.0 / 2.0_f32.sqrt();
    0.5 * x * (1.0 + erff_approx(x * inv_sqrt2))
}

fn gelu_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn test_gelu_affine_concrete_erf() {
    // GELU on concrete (no error) zonotope should produce GELU values, not SiLU.
    let values = arr1(&[-1.5_f32, -1.0, 0.0, 1.0, 2.0]);
    let z = ZonotopeTensor::concrete(values.clone().into_dyn());

    let result = z.gelu_affine(false).unwrap();
    let center = result.center();

    for d in 0..5 {
        let expected = gelu_erf(values[d]);
        assert!(
            (center[d] - expected).abs() < 1e-5,
            "GELU_erf({}) = {}, got {}",
            values[d],
            expected,
            center[d]
        );
    }
}

#[test]
fn test_gelu_affine_concrete_tanh() {
    let values = arr1(&[-1.5_f32, -1.0, 0.0, 1.0, 2.0]);
    let z = ZonotopeTensor::concrete(values.clone().into_dyn());

    let result = z.gelu_affine(true).unwrap();
    let center = result.center();

    for d in 0..5 {
        let expected = gelu_tanh(values[d]);
        assert!(
            (center[d] - expected).abs() < 1e-5,
            "GELU_tanh({}) = {}, got {}",
            values[d],
            expected,
            center[d]
        );
    }
}

/// Verify GELU center differs from SiLU center at x = -1.5 (max divergence).
#[test]
fn test_gelu_affine_differs_from_silu_2470() {
    let values = arr1(&[-1.5_f32]);
    let z = ZonotopeTensor::concrete(values.into_dyn());

    let gelu_result = z.gelu_affine(false).unwrap();
    let silu_result = z.silu_affine().unwrap();

    let gelu_center = gelu_result.center()[0];
    let silu_center = silu_result.center()[0];

    let gelu_expected = gelu_erf(-1.5);
    let silu_expected = silu(-1.5);

    // GELU(-1.5) ≈ -0.1005, SiLU(-1.5) ≈ -0.2672. Difference ≈ 0.17.
    assert!(
        (gelu_expected - silu_expected).abs() > 0.1,
        "GELU and SiLU should differ significantly at x=-1.5: GELU={gelu_expected}, SiLU={silu_expected}"
    );
    assert!(
        (gelu_center - gelu_expected).abs() < 1e-5,
        "gelu_affine center should match GELU, not SiLU: got {gelu_center}, expected {gelu_expected}"
    );
    assert!(
        (silu_center - silu_expected).abs() < 1e-5,
        "silu_affine center should match SiLU: got {silu_center}, expected {silu_expected}"
    );
}

#[test]
fn test_gelu_affine_with_error_erf_soundness() {
    // Verify bounds contain sampled GELU outputs.
    let values = arr1(&[-1.5_f32, 0.0, 1.0]);
    let z = ZonotopeTensor::from_input_shared(&values.clone().into_dyn(), 0.5);

    let result = z.gelu_affine(false).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    // Sample 21 points per element and verify containment.
    for d in 0..3 {
        let c = values[d];
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let x = (c - 0.5) + t * 1.0; // from c-0.5 to c+0.5
            let y = gelu_erf(x);

            assert!(
                y >= bounds.lower()[d] - 1e-5,
                "GELU_erf({x}) = {y} < lower bound {} for element {d}",
                bounds.lower()[d]
            );
            assert!(
                y <= bounds.upper()[d] + 1e-5,
                "GELU_erf({x}) = {y} > upper bound {} for element {d}",
                bounds.upper()[d]
            );
        }
    }
}

#[test]
fn test_gelu_affine_with_error_tanh_soundness() {
    let values = arr1(&[-1.5_f32, 0.0, 1.0]);
    let z = ZonotopeTensor::from_input_shared(&values.clone().into_dyn(), 0.5);

    let result = z.gelu_affine(true).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    for d in 0..3 {
        let c = values[d];
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let x = (c - 0.5) + t * 1.0;
            let y = gelu_tanh(x);

            assert!(
                y >= bounds.lower()[d] - 1e-5,
                "GELU_tanh({x}) = {y} < lower bound {} for element {d}",
                bounds.lower()[d]
            );
            assert!(
                y <= bounds.upper()[d] + 1e-5,
                "GELU_tanh({x}) = {y} > upper bound {} for element {d}",
                bounds.upper()[d]
            );
        }
    }
}

#[test]
fn test_gelu_affine_2d() {
    let values = arr2(&[[0.0_f32, 1.0], [-1.5, 2.0]]);
    let z = ZonotopeTensor::concrete(values.clone().into_dyn());

    let result = z.gelu_affine(false).unwrap();
    assert_eq!(result.element_shape, vec![2, 2]);

    let center = result.center();
    for s in 0..2 {
        for d in 0..2 {
            let expected = gelu_erf(values[[s, d]]);
            assert!(
                (center[[s, d]] - expected).abs() < 1e-5,
                "GELU_erf({}) = {}, got {}",
                values[[s, d]],
                expected,
                center[[s, d]]
            );
        }
    }
}

#[test]
fn test_gelu_affine_nd_recursive() {
    // Test 3D input — should work via reshape->1D->reshape back.
    let values = ndarray::Array3::<f32>::from_elem((1, 2, 2), 1.0).into_dyn();
    let z = ZonotopeTensor::concrete(values);

    let result = z.gelu_affine(false).unwrap();
    assert_eq!(result.element_shape, vec![1, 2, 2]);

    let expected = gelu_erf(1.0);
    let center = result.center();
    for val in center.iter() {
        assert!(
            (*val - expected).abs() < 1e-5,
            "3D gelu should compute correctly: expected {expected}, got {val}"
        );
    }
}

/// Regression for follow-up audit after #2470:
/// GELU approximation errors must be independent per element so they cannot
/// cancel under downstream linear combinations (for example [1, -1] projection).
#[test]
fn test_gelu_affine_linear_projection_soundness_1d_independent_error_terms() {
    let values = arr1(&[-0.75_f32, -0.75]);
    let z = ZonotopeTensor::from_input_elementwise(&values.into_dyn(), 1.2);

    let gelu_z = z.gelu_affine(false).unwrap();
    assert_eq!(
        gelu_z.n_error_terms(),
        z.n_error_terms() + 2,
        "gelu_affine 1D should add one approximation error symbol per element"
    );

    let weight = arr2(&[[1.0_f32, -1.0]]);
    let projected = gelu_z.linear(&weight, None).unwrap();
    let bounds = projected.to_bounded_tensor().unwrap();

    let mut true_min = f32::INFINITY;
    let mut true_max = f32::NEG_INFINITY;
    for &e0 in &[-1.0_f32, 1.0] {
        for &e1 in &[-1.0_f32, 1.0] {
            let x0 = -0.75 + 1.2 * e0;
            let x1 = -0.75 + 1.2 * e1;
            let y = gelu_erf(x0) - gelu_erf(x1);
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

/// Same cancellation regression as above, but through the 2D GELU path.
#[test]
fn test_gelu_affine_linear_projection_soundness_2d_independent_error_terms() {
    let values = arr2(&[[-0.75_f32, -0.75]]);
    let z = ZonotopeTensor::from_input_2d(&values, 1.2);

    let gelu_z = z.gelu_affine(false).unwrap();
    assert_eq!(
        gelu_z.n_error_terms(),
        z.n_error_terms() + values.len(),
        "gelu_affine 2D should add one approximation error symbol per element"
    );

    let weight = arr2(&[[1.0_f32, -1.0]]);
    let projected = gelu_z.linear(&weight, None).unwrap();
    let bounds = projected.to_bounded_tensor().unwrap();

    let mut true_min = f32::INFINITY;
    let mut true_max = f32::NEG_INFINITY;
    for &e0 in &[-1.0_f32, 1.0] {
        for &e1 in &[-1.0_f32, 1.0] {
            let x0 = -0.75 + 1.2 * e0;
            let x1 = -0.75 + 1.2 * e1;
            let y = gelu_erf(x0) - gelu_erf(x1);
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

// #2850: Proptest that zonotope GELU (erf) error bounds contain all sampled outputs.
proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(256) })]
    #[test]
    fn proptest_gelu_erf_affine_bounds_contain_sampled_outputs(
        center in -10.0f32..10.0,
        radius in 0.01f32..5.0,
    ) {
        let values = arr1(&[center]);
        let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), radius);
        let result = z.gelu_affine(false).unwrap();
        let bounds = result.to_bounded_tensor().unwrap();

        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let x = (center - radius) + t * 2.0 * radius;
            let y = gelu_erf(x);

            prop_assert!(
                y >= bounds.lower()[0] - 1e-5,
                "GELU_erf({x}) = {y} < lower {} for c={center}, r={radius}",
                bounds.lower()[0]
            );
            prop_assert!(
                y <= bounds.upper()[0] + 1e-5,
                "GELU_erf({x}) = {y} > upper {} for c={center}, r={radius}",
                bounds.upper()[0]
            );
        }
    }

    #[test]
    fn proptest_gelu_tanh_affine_bounds_contain_sampled_outputs(
        center in -10.0f32..10.0,
        radius in 0.01f32..5.0,
    ) {
        let values = arr1(&[center]);
        let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), radius);
        let result = z.gelu_affine(true).unwrap();
        let bounds = result.to_bounded_tensor().unwrap();

        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let x = (center - radius) + t * 2.0 * radius;
            let y = gelu_tanh(x);

            prop_assert!(
                y >= bounds.lower()[0] - 1e-5,
                "GELU_tanh({x}) = {y} < lower {} for c={center}, r={radius}",
                bounds.lower()[0]
            );
            prop_assert!(
                y <= bounds.upper()[0] + 1e-5,
                "GELU_tanh({x}) = {y} > upper {} for c={center}, r={radius}",
                bounds.upper()[0]
            );
        }
    }
}

// ==================== NaN/Inf safety tests (#2676 Site 2) ====================

/// #2676 Site 2: GELU with very large center should produce finite bounds,
/// not NaN from 0·Inf indeterminate form in the second derivative.
#[test]
fn test_gelu_affine_large_center_no_nan_2676() {
    // Large center: GELU''(1000) would produce 0·(-Inf)=NaN without the guard.
    // The error bound must remain finite.
    let values = arr1(&[1000.0_f32]);
    let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 0.1);

    let result_erf = z.gelu_affine(false).unwrap();
    let bounds_erf = result_erf.to_bounded_tensor().unwrap();
    assert!(
        bounds_erf.lower()[0].is_finite() && bounds_erf.upper()[0].is_finite(),
        "#2676: GELU erf bounds should be finite for large center, got [{}, {}]",
        bounds_erf.lower()[0],
        bounds_erf.upper()[0]
    );

    let result_tanh = z.gelu_affine(true).unwrap();
    let bounds_tanh = result_tanh.to_bounded_tensor().unwrap();
    assert!(
        bounds_tanh.lower()[0].is_finite() && bounds_tanh.upper()[0].is_finite(),
        "#2676: GELU tanh bounds should be finite for large center, got [{}, {}]",
        bounds_tanh.lower()[0],
        bounds_tanh.upper()[0]
    );
}

/// #2676 Site 2: GELU with Inf center should not produce NaN in error bound.
#[test]
fn test_gelu_affine_inf_center_no_nan_2676() {
    let values = arr1(&[f32::INFINITY]);
    let z = ZonotopeTensor::concrete(values.into_dyn());

    // Even with Inf center, GELU should not crash. Center output is GELU(Inf) = Inf.
    let result = z.gelu_affine(false).unwrap();
    let center = result.center();
    // GELU(+Inf) = +Inf (guarded in gelu_erf)
    assert!(
        center[0] == f32::INFINITY,
        "#2676: GELU(+Inf) should be +Inf, got {}",
        center[0]
    );
}

/// #2676 Site 2: GELU with negative Inf center should return 0.
#[test]
fn test_gelu_affine_neg_inf_center_2676() {
    let values = arr1(&[f32::NEG_INFINITY]);
    let z = ZonotopeTensor::concrete(values.into_dyn());

    let result = z.gelu_affine(false).unwrap();
    let center = result.center();
    // GELU(-Inf) = 0 (guarded in gelu_erf)
    assert!(
        center[0] == 0.0,
        "#2676: GELU(-Inf) should be 0, got {}",
        center[0]
    );
}

/// #2676: GELU derivative at ±Inf must not produce NaN.
/// When error terms exist, the derivative is used as the slope to transform them.
/// gelu_erf_derivative(+Inf) = 1 (not NaN from Inf·0), gelu_erf_derivative(-Inf) = 0.
#[test]
fn test_gelu_affine_inf_center_with_error_erf_2676() {
    // Create zonotope with Inf center and an error term
    let values = arr1(&[f32::INFINITY]);
    let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 1.0);

    let result = z.gelu_affine(false).unwrap();
    // Error coefficient should be slope * original_coeff = 1.0 * 1.0 = 1.0 (not NaN)
    let bounds = result.to_bounded_tensor().unwrap();
    assert!(
        !bounds.lower()[0].is_nan() && !bounds.upper()[0].is_nan(),
        "#2676: GELU erf bounds with Inf center+error should not be NaN, got [{}, {}]",
        bounds.lower()[0],
        bounds.upper()[0]
    );
}

/// #2676: Same test for tanh variant.
#[test]
fn test_gelu_affine_inf_center_with_error_tanh_2676() {
    let values = arr1(&[f32::INFINITY]);
    let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 1.0);

    let result = z.gelu_affine(true).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();
    assert!(
        !bounds.lower()[0].is_nan() && !bounds.upper()[0].is_nan(),
        "#2676: GELU tanh bounds with Inf center+error should not be NaN, got [{}, {}]",
        bounds.lower()[0],
        bounds.upper()[0]
    );
}

/// #2676: GELU derivative at -Inf with error terms.
/// Slope = 0, so error terms should be zeroed out.
#[test]
fn test_gelu_affine_neg_inf_center_with_error_2676() {
    let values = arr1(&[f32::NEG_INFINITY]);
    let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 1.0);

    let result_erf = z.gelu_affine(false).unwrap();
    let bounds_erf = result_erf.to_bounded_tensor().unwrap();
    assert!(
        !bounds_erf.lower()[0].is_nan() && !bounds_erf.upper()[0].is_nan(),
        "#2676: GELU erf bounds with -Inf center+error should not be NaN, got [{}, {}]",
        bounds_erf.lower()[0],
        bounds_erf.upper()[0]
    );

    let result_tanh = z.gelu_affine(true).unwrap();
    let bounds_tanh = result_tanh.to_bounded_tensor().unwrap();
    assert!(
        !bounds_tanh.lower()[0].is_nan() && !bounds_tanh.upper()[0].is_nan(),
        "#2676: GELU tanh bounds with -Inf center+error should not be NaN, got [{}, {}]",
        bounds_tanh.lower()[0],
        bounds_tanh.upper()[0]
    );
}
