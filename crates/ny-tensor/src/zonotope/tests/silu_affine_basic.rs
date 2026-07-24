// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{arr1, arr2};
use proptest::prelude::*;

#[test]
fn test_silu_affine_concrete() {
    // Test SiLU on concrete (no error) zonotope
    // SiLU(x) = x * sigmoid(x)
    let values = arr1(&[-1.0_f32, 0.0, 1.0, 2.0]);
    let z = ZonotopeTensor::concrete(values.clone().into_dyn());

    let result = z.silu_affine().unwrap();
    let center = result.center();

    // Expected SiLU values
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    for d in 0..4 {
        let expected = silu(values[d]);
        assert!(
            (center[d] - expected).abs() < 1e-5,
            "SiLU({}) = {}, got {}",
            values[d],
            expected,
            center[d]
        );
    }
}

#[test]
fn test_silu_affine_with_error() {
    // Test SiLU preserves error structure with approximation
    let values = arr1(&[0.0_f32, 1.0]);
    let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 0.1);

    let result = z.silu_affine().unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    // Verify bounds are sound: check that actual SiLU values at bounds are contained
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    // For input [0±0.1, 1±0.1]
    assert!(
        bounds.lower()[0] <= silu(-0.1),
        "lower[0] {} should be <= silu(-0.1) = {}",
        bounds.lower()[0],
        silu(-0.1)
    );
    assert!(
        bounds.upper()[0] >= silu(0.1),
        "upper[0] {} should be >= silu(0.1) = {}",
        bounds.upper()[0],
        silu(0.1)
    );
    assert!(
        bounds.lower()[1] <= silu(0.9),
        "lower[1] {} should be <= silu(0.9) = {}",
        bounds.lower()[1],
        silu(0.9)
    );
    assert!(
        bounds.upper()[1] >= silu(1.1),
        "upper[1] {} should be >= silu(1.1) = {}",
        bounds.upper()[1],
        silu(1.1)
    );
}

#[test]
fn test_silu_affine_2d() {
    // Test 2D SiLU (needed for transformer FFN)
    let values = arr2(&[[0.0_f32, 1.0], [-1.0, 2.0]]);
    let z = ZonotopeTensor::concrete(values.clone().into_dyn());

    let result = z.silu_affine().unwrap();
    assert_eq!(result.element_shape, vec![2, 2]);

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    let center = result.center();
    for s in 0..2 {
        for d in 0..2 {
            let expected = silu(values[[s, d]]);
            assert!(
                (center[[s, d]] - expected).abs() < 1e-5,
                "SiLU({}) = {}, got {}",
                values[[s, d]],
                expected,
                center[[s, d]]
            );
        }
    }
}

#[test]
fn test_silu_affine_nd_recursive() {
    // Kills: delete match arm 2 (3D+ case)

    // Test 3D input - should work via reshape->1D->reshape back
    let values = ndarray::Array3::<f32>::from_elem((1, 2, 2), 1.0).into_dyn();
    let z = ZonotopeTensor::concrete(values);

    let result = z.silu_affine().unwrap();

    assert_eq!(
        result.element_shape,
        vec![1, 2, 2],
        "should preserve 3D shape"
    );

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
    let expected = silu(1.0);

    let center = result.center();
    for val in center.iter() {
        assert!(
            (*val - expected).abs() < 1e-5,
            "3D silu should compute correctly"
        );
    }
}

#[test]
fn test_silu_affine_2d_multiple_positions() {
    // Test with multiple sequence positions to ensure 2D loop is correct

    // Create 3x2 input (3 positions, 2 features)
    let values = arr2(&[
        [-1.28_f32, 0.0], // Position 0: critical region, center
        [1.0, 2.0],       // Position 1: positive values
        [-2.0, -3.0],     // Position 2: negative values
    ])
    .into_dyn();

    let z = ZonotopeTensor::from_input_shared(&values, 0.3);
    let result = z.silu_affine().unwrap();

    assert_eq!(result.element_shape, vec![3, 2]);

    let bounds = result.to_bounded_tensor().unwrap();

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    // Check bounds at each position
    for s in 0..3 {
        for d in 0..2 {
            let c = values[[s, d]];
            let lo = c - 0.3;
            let hi = c + 0.3;

            assert!(
                bounds.lower()[[s, d]] <= silu(lo) + 0.05,
                "2D multi-pos lower[{},{}] should contain silu({})",
                s,
                d,
                lo
            );
            assert!(
                bounds.upper()[[s, d]] >= silu(hi) - 0.05,
                "2D multi-pos upper[{},{}] should contain silu({})",
                s,
                d,
                hi
            );
        }
    }
}

/// SiLU''(x) evaluated in f64 with the cancellation-free sigmoid(-x) form.
/// Reference for curvature assertions below.
fn silu_second_derivative_f64(x: f64) -> f64 {
    let s = 1.0 / (1.0 + (-x).exp());
    let s_neg = 1.0 / (1.0 + x.exp());
    s * s_neg * (2.0 + x - 2.0 * x * s)
}

/// A naive f32 evaluation of SiLU'' via `1.0 - sigmoid(x)` cancels to exactly
/// zero once sigmoid(x) rounds to 1 (x ≳ 17), which would silently drop the
/// Taylor-remainder error symbol in the saturated positive tail. The true
/// curvature there is small but nonzero (|SiLU''(x)| ≈ (x - 2)·e^-x), so the
/// error symbol must stay strictly positive.
#[test]
fn test_silu_affine_saturated_tail_error_symbol_positive() {
    for &c in &[17.0_f32, 20.0, 25.0, 40.0, 60.0] {
        let z = ZonotopeTensor::from_input_shared(&arr1(&[c]).into_dyn(), 1.0);
        let result = z.silu_affine().unwrap();
        let approx_err = result.coeffs[[result.n_error_terms, 0]];
        assert!(
            approx_err > 0.0,
            "approx error at center {c} must be positive, got {approx_err}"
        );
    }

    // Quantitative check at c = 20, r = 1: max |SiLU''| over [19, 21] is
    // attained at x = 19 (|SiLU''| decays monotonically past the lobe peak),
    // so the error symbol must be at least |SiLU''(19)| * r^2 / 2.
    let z = ZonotopeTensor::from_input_shared(&arr1(&[20.0_f32]).into_dyn(), 1.0);
    let result = z.silu_affine().unwrap();
    let approx_err = result.coeffs[[result.n_error_terms, 0]];
    let required = (silu_second_derivative_f64(19.0).abs() / 2.0) as f32;
    assert!(
        approx_err >= required * 0.99,
        "approx error {approx_err} must cover |SiLU''(19)|/2 = {required}"
    );
}

/// Same saturated-tail regression through the 2D code path.
#[test]
fn test_silu_affine_2d_saturated_tail_error_symbol_positive() {
    let values = arr2(&[[20.0_f32, 30.0]]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 1.0);
    let orig_n_err = z.n_error_terms;

    let result = z.silu_affine().unwrap();
    for d in 0..2 {
        // Per-element error row for element (0, d).
        let approx_err = result.coeffs[[orig_n_err + 1 + d, 0, d]];
        assert!(
            approx_err > 0.0,
            "2D approx error for element {d} must be positive, got {approx_err}"
        );
    }
}

/// The intervals [±3.4 - 0.5, ±3.4 + 0.5] contain the negative-lobe curvature
/// peaks of SiLU (x ≈ ±3.436, |SiLU''| ≈ 0.03691) strictly in their interior,
/// where neither endpoint attains the maximum. The error symbol must cover the
/// true interval maximum of |SiLU''|, not just the endpoint values.
#[test]
fn test_silu_affine_error_covers_lobe_curvature_peak() {
    let r = 0.5_f32;
    for &c in &[3.4_f32, -3.4] {
        let z = ZonotopeTensor::from_input_shared(&arr1(&[c]).into_dyn(), r);
        let result = z.silu_affine().unwrap();
        let approx_err = result.coeffs[[result.n_error_terms, 0]];

        // True max |SiLU''| over [c-r, c+r], densely sampled in f64.
        let lo = (c - r) as f64;
        let hi = (c + r) as f64;
        let mut true_max = 0.0_f64;
        for i in 0..=4000 {
            let x = lo + (i as f64 / 4000.0) * (hi - lo);
            true_max = true_max.max(silu_second_derivative_f64(x).abs());
        }
        let required = (true_max * f64::from(r) * f64::from(r) / 2.0) as f32;
        assert!(
            approx_err >= required,
            "approx error {approx_err} at center {c} must be >= true Taylor bound {required}"
        );
    }
}

// #2850: Proptest that zonotope SiLU error bounds are sound for random intervals.
// For each random center and radius, the zonotope bounds must contain all sampled
// SiLU outputs within the interval [center-radius, center+radius].
proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(256) })]
    #[test]
    fn proptest_silu_affine_bounds_contain_sampled_outputs(
        center in -10.0f32..10.0,
        radius in 0.01f32..5.0,
    ) {
        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        let values = arr1(&[center]);
        let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), radius);
        let result = z.silu_affine().unwrap();
        let bounds = result.to_bounded_tensor().unwrap();

        // Sample 101 points and verify containment.
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let x = (center - radius) + t * 2.0 * radius;
            let y = silu(x);

            prop_assert!(
                y >= bounds.lower()[0] - 1e-5,
                "SiLU({x}) = {y} < lower bound {} for center={center}, radius={radius}",
                bounds.lower()[0]
            );
            prop_assert!(
                y <= bounds.upper()[0] + 1e-5,
                "SiLU({x}) = {y} > upper bound {} for center={center}, radius={radius}",
                bounds.upper()[0]
            );
        }
    }

    // Containment sweep over wide centers: reaches the saturated tails
    // (|x| > 17, where a naive f32 SiLU'' evaluation cancels to zero) and the
    // curvature lobes (|x| ≈ 3.4, where the peak lies strictly inside the
    // interval). Reference values are computed in f64.
    #[test]
    fn proptest_silu_affine_bounds_contain_sampled_outputs_wide(
        center in -40.0f32..40.0,
        radius in 0.01f32..8.0,
    ) {
        fn silu_f64(x: f64) -> f64 {
            x / (1.0 + (-x).exp())
        }

        let values = arr1(&[center]);
        let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), radius);
        let result = z.silu_affine().unwrap();
        let bounds = result.to_bounded_tensor().unwrap();

        for i in 0..=200 {
            let t = i as f32 / 200.0;
            let x = (center - radius) + t * 2.0 * radius;
            let y = silu_f64(f64::from(x)) as f32;
            // Slack scales with |y|: the bounds are computed in f32, so a few
            // ulps of the output magnitude are expected from rounding alone.
            let tol = 1e-5 * (1.0 + y.abs());

            prop_assert!(
                y >= bounds.lower()[0] - tol,
                "SiLU({x}) = {y} < lower bound {} for center={center}, radius={radius}",
                bounds.lower()[0]
            );
            prop_assert!(
                y <= bounds.upper()[0] + tol,
                "SiLU({x}) = {y} > upper bound {} for center={center}, radius={radius}",
                bounds.upper()[0]
            );
        }
    }
}
