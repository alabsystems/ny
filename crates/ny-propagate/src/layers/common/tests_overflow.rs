// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #3009: Non-finite coeff × slope overflow fallback tests for activation CROWN backward.

use super::*;

use crate::layers::activations::LinearRelaxation;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{array, Array1, ArrayD, IxDyn};
use ny_core::Result;
use proptest::prelude::*;

#[test]
fn test_elementwise_large_coeff_slope_overflow_3009() -> Result<()> {
    // #3009: When coefficient × slope overflows f32 to Inf, the affected row
    // should fall back to zero A-coefficients and ±Inf bias (sound but loose),
    // rather than returning NumericalInstability error.
    //
    // Setup: 2 outputs, 2 neurons.
    // Row 0: coeff = 1e20, slope = 1e20 → product = Inf (overflow)
    // Row 1: coeff = 1.0, slope = 1e20 → product = 1e20 (finite, preserved)
    fn large_slope_relax(_l: f32, _u: f32) -> LinearRelaxation {
        LinearRelaxation::new(1e20, 0.5, 1e20, 0.5)
    }

    let bounds = LinearBounds::new(
        array![[1e20_f32, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
        array![[1e20_f32, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
    )
    .unwrap();
    let pre = BoundedTensor::new(
        array![-1.0_f32, -1.0].into_dyn(),
        array![1.0_f32, 1.0].into_dyn(),
    )?;

    let result = crown_elementwise_backward(&bounds, &pre, large_slope_relax)?;

    // Row 0: overflow detected → zeroed A, ±Inf bias
    assert_eq!(
        result.lower_a[[0, 0]],
        0.0,
        "overflow row lower_a must be zeroed"
    );
    assert_eq!(
        result.lower_a[[0, 1]],
        0.0,
        "overflow row lower_a must be zeroed"
    );
    assert_eq!(
        result.lower_b[0],
        f32::NEG_INFINITY,
        "overflow row lower_b must be -Inf"
    );
    assert_eq!(
        result.upper_a[[0, 0]],
        0.0,
        "overflow row upper_a must be zeroed"
    );
    assert_eq!(
        result.upper_a[[0, 1]],
        0.0,
        "overflow row upper_a must be zeroed"
    );
    assert_eq!(
        result.upper_b[0],
        f32::INFINITY,
        "overflow row upper_b must be +Inf"
    );

    // Row 1: no overflow → preserved
    assert!(
        (result.lower_a[[1, 1]] - 1e20).abs() < 1e15,
        "finite row preserved"
    );
    assert!(result.lower_b[1].is_finite(), "finite row bias is finite");
    assert!(result.upper_b[1].is_finite(), "finite row bias is finite");
    Ok(())
}

#[test]
fn test_elementwise_large_coeff_slope_overflow_soundness_3009() -> Result<()> {
    // #3009 soundness: After overflow fallback, the bounds must be sound.
    // For the overflowed row, bounds are [-Inf, +Inf] which trivially contain
    // any true output. For the non-overflowed row, bounds must contain the
    // true activation output for sampled inputs.
    //
    // Use large slopes (simulating Exp relaxation at large pre-activation values)
    // with large coefficients (accumulated through deep linear backward passes).
    fn large_slope_relax(_l: f32, _u: f32) -> LinearRelaxation {
        // Simulates Exp relaxation slope ~3000, intercept ~-20000
        LinearRelaxation::new(3000.0, -20000.0, 3000.0, -20000.0)
    }

    // Row 0: coeffs = 1e36, slope = 3000 → product = 3e39 > f32::MAX (3.4e38) → Inf
    // Row 1: coeffs = 1.0, slope = 3000 → product = 3000 (finite)
    let bounds = LinearBounds::new(
        array![[1e36_f32, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
        array![[1e36_f32, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
    )
    .unwrap();
    let pre = BoundedTensor::new(
        array![-1.0_f32, -1.0].into_dyn(),
        array![1.0_f32, 1.0].into_dyn(),
    )?;

    let result = crown_elementwise_backward(&bounds, &pre, large_slope_relax)?;

    // Row 0 soundness: overflowed → [-Inf, +Inf] trivially sound
    assert_eq!(result.lower_b[0], f32::NEG_INFINITY);
    assert_eq!(result.upper_b[0], f32::INFINITY);
    // lower <= upper is the soundness invariant
    assert!(result.lower_b[0] <= result.upper_b[0]);

    // Row 1 soundness: finite, bounds should be valid
    assert!(result.lower_b[1].is_finite());
    assert!(result.upper_b[1].is_finite());
    assert!(result.lower_b[1] <= result.upper_b[1]);
    Ok(())
}

#[test]
fn test_batched_elementwise_large_coeff_slope_overflow_3009() -> Result<()> {
    // #3009: Same overflow test for the batched path.
    fn large_slope_relax(_l: f32, _u: f32, _i: usize) -> LinearRelaxation {
        LinearRelaxation::new(1e20, 0.5, 1e20, 0.5)
    }

    let lower_a = array![[[1e20_f32, 0.0], [0.0, 1.0]]].into_dyn();
    let upper_a = lower_a.clone();
    let lower_b = ArrayD::zeros(IxDyn(&[1, 2]));
    let upper_b = ArrayD::zeros(IxDyn(&[1, 2]));
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        vec![2],
        vec![2],
    );
    let pre = BoundedTensor::new(
        array![[-1.0_f32, -1.0]].into_dyn(),
        array![[1.0_f32, 1.0]].into_dyn(),
    )?;

    let result = crown_elementwise_backward_batched_indexed(&bounds, &pre, large_slope_relax)?;

    // Row 0 (batch=0, out=0): overflow → zeroed A, ±Inf bias
    assert_eq!(result.lower_b[IxDyn(&[0, 0])], f32::NEG_INFINITY);
    assert_eq!(result.upper_b[IxDyn(&[0, 0])], f32::INFINITY);

    // Row 1 (batch=0, out=1): no overflow → finite
    assert!(result.lower_b[IxDyn(&[0, 1])].is_finite());
    assert!(result.upper_b[IxDyn(&[0, 1])].is_finite());
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// #3009: CROWN backward through activation with large coefficients returns
    /// sound conservative bounds, not NumericalInstability error.
    ///
    /// Property: For any combination of coefficient magnitude and slope magnitude
    /// (including those that overflow f32), crown_elementwise_backward must:
    /// 1. Return Ok (not error)
    /// 2. Produce lower_b <= upper_b for every output (soundness invariant)
    /// 3. Not contain NaN in any output
    #[test]
    fn proptest_crown_backward_large_coeff_soundness_3009(
        // Coefficient exponent: 1e0 to 1e38 (covers near-overflow range)
        coeff_exp in 0.0f32..38.0,
        // Slope magnitude: 1.0 to 1e20
        slope_exp in 0.0f32..20.0,
        // Intercept magnitude
        intercept in -1e4f32..1e4,
        // Pre-activation bound half-width
        pre_half in 0.01f32..10.0,
    ) {
        let coeff = 10.0_f32.powf(coeff_exp);
        let slope = 10.0_f32.powf(slope_exp);

        // Skip degenerate cases where coeff itself is non-finite
        prop_assume!(coeff.is_finite());
        prop_assume!(slope.is_finite());

        let relax_slope = slope;
        let relax_intercept = intercept;
        let relax_fn = move |_l: f32, _u: f32| {
            LinearRelaxation::new(relax_slope, relax_intercept, relax_slope, relax_intercept)
        };

        // 2x2 system: row 0 has the large coefficient, row 1 has coeff=1.0
        let bounds = LinearBounds::new(
            array![[coeff, 0.0], [0.0, 1.0]],
            Array1::zeros(2),
            array![[coeff, 0.0], [0.0, 1.0]],
            Array1::zeros(2),
        ).unwrap();
        let pre = BoundedTensor::new(
            array![-pre_half, -pre_half].into_dyn(),
            array![pre_half, pre_half].into_dyn(),
        ).map_err(|e| TestCaseError::fail(format!("BoundedTensor: {e}")))?;

        let result = crown_elementwise_backward(&bounds, &pre, relax_fn)
            .map_err(|e| TestCaseError::fail(format!("crown_elementwise_backward: {e}")))?;

        // Soundness: lower_b <= upper_b for every output
        for j in 0..2 {
            prop_assert!(
                result.lower_b[j] <= result.upper_b[j],
                "row {j}: lower_b ({}) > upper_b ({})",
                result.lower_b[j], result.upper_b[j]
            );
        }

        // No NaN anywhere
        for j in 0..2 {
            for i in 0..2 {
                prop_assert!(
                    !result.lower_a[[j, i]].is_nan(),
                    "lower_a[{j},{i}] is NaN"
                );
                prop_assert!(
                    !result.upper_a[[j, i]].is_nan(),
                    "upper_a[{j},{i}] is NaN"
                );
            }
            prop_assert!(!result.lower_b[j].is_nan(), "lower_b[{j}] is NaN");
            prop_assert!(!result.upper_b[j].is_nan(), "upper_b[{j}] is NaN");
        }

        // If overflow occurred (coeff * slope > f32::MAX), row 0 must have ±Inf bias
        let product = coeff * slope;
        if !product.is_finite() {
            prop_assert_eq!(result.lower_a[[0, 0]], 0.0, "overflow: lower_a not zeroed");
            prop_assert_eq!(result.lower_a[[0, 1]], 0.0, "overflow: lower_a not zeroed");
            prop_assert_eq!(result.lower_b[0], f32::NEG_INFINITY, "overflow: lower_b not -Inf");
            prop_assert_eq!(result.upper_a[[0, 0]], 0.0, "overflow: upper_a not zeroed");
            prop_assert_eq!(result.upper_a[[0, 1]], 0.0, "overflow: upper_a not zeroed");
            prop_assert_eq!(result.upper_b[0], f32::INFINITY, "overflow: upper_b not +Inf");
        }

        // Row 1 (coeff=1.0) should always be finite
        prop_assert!(result.lower_b[1].is_finite(), "row 1 lower_b not finite");
        prop_assert!(result.upper_b[1].is_finite(), "row 1 upper_b not finite");
    }

    /// #3009 strengthened: CROWN backward with negative coefficients and asymmetric
    /// relaxation slopes.
    ///
    /// The original proptest only uses positive coefficients and symmetric slopes.
    /// Negative coefficients trigger a different branch in crown_elementwise_backward
    /// (la < 0.0 multiplies by upper_slopes instead of lower_slopes). This test
    /// covers:
    /// 1. Negative coefficients (the slope-swap branch)
    /// 2. Asymmetric lower/upper relaxation slopes (different overflow thresholds)
    ///
    /// Note: lower_a and upper_a MUST have the same sign for sound bounds. In valid
    /// CROWN backward, activation relaxation slopes are non-negative, so the sign of
    /// the A coefficient is preserved through backward propagation. Opposite-sign
    /// lower_a/upper_a produces mathematically valid but vacuously unsound bounds
    /// (lower > upper), which is expected behavior, not a bug.
    #[test]
    fn proptest_crown_backward_negative_coeff_asymmetric_3009(
        // Coefficient exponent: 1e0 to ~f32::MAX
        coeff_exp in 0.0f32..38.5,
        // Separate slope exponents for lower and upper relaxation
        lower_slope_exp in 0.0f32..20.0,
        upper_slope_exp in 0.0f32..20.0,
        // Intercept magnitude
        intercept in -1e4f32..1e4,
        // Pre-activation bound half-width
        pre_half in 0.01f32..10.0,
        // Whether to negate BOTH coefficients (exercises la < 0, ua < 0 branch)
        negate_both in proptest::bool::ANY,
    ) {
        let coeff = 10.0_f32.powf(coeff_exp);
        let lower_slope = 10.0_f32.powf(lower_slope_exp);
        let upper_slope = 10.0_f32.powf(upper_slope_exp);

        // Skip degenerate cases where inputs themselves are non-finite
        prop_assume!(coeff.is_finite());
        prop_assume!(lower_slope.is_finite());
        prop_assume!(upper_slope.is_finite());

        // Both lower_a and upper_a have the same sign (valid CROWN invariant).
        // When negate_both=true, both are negative → exercises the la<0/ua<0 branches.
        let signed_coeff = if negate_both { -coeff } else { coeff };

        let relax_lower_slope = lower_slope;
        let relax_upper_slope = upper_slope;
        let relax_intercept = intercept;
        let relax_fn = move |_l: f32, _u: f32| {
            LinearRelaxation::new(
                relax_lower_slope,
                relax_intercept,
                relax_upper_slope,
                relax_intercept,
            )
        };

        // 2x2 system: row 0 has the test coefficient (possibly negative),
        // row 1 has coeff=1.0 as control
        let bounds = LinearBounds::new(
            array![[signed_coeff, 0.0], [0.0, 1.0]],
            Array1::zeros(2),
            array![[signed_coeff, 0.0], [0.0, 1.0]],
            Array1::zeros(2),
        ).unwrap();
        let pre = BoundedTensor::new(
            array![-pre_half, -pre_half].into_dyn(),
            array![pre_half, pre_half].into_dyn(),
        )
        .map_err(|e| TestCaseError::fail(format!("BoundedTensor: {e}")))?;

        let result = crown_elementwise_backward(&bounds, &pre, relax_fn)
            .map_err(|e| TestCaseError::fail(format!("crown_elementwise_backward: {e}")))?;

        // Soundness invariant: lower_b <= upper_b for every output row
        for j in 0..2 {
            prop_assert!(
                result.lower_b[j] <= result.upper_b[j],
                "row {j}: lower_b ({}) > upper_b ({}), coeff={}, negate={}",
                result.lower_b[j],
                result.upper_b[j],
                signed_coeff,
                negate_both
            );
        }

        // No NaN in any output
        for j in 0..2 {
            for i in 0..2 {
                prop_assert!(
                    !result.lower_a[[j, i]].is_nan(),
                    "lower_a[{j},{i}] is NaN"
                );
                prop_assert!(
                    !result.upper_a[[j, i]].is_nan(),
                    "upper_a[{j},{i}] is NaN"
                );
            }
            prop_assert!(!result.lower_b[j].is_nan(), "lower_b[{j}] is NaN");
            prop_assert!(!result.upper_b[j].is_nan(), "upper_b[{j}] is NaN");
        }

        // For negative coefficients: overflow occurs when |coeff| * upper_slope > f32::MAX
        // (slope swap: negative coeff uses the other direction's slope)
        let lower_product = if negate_both {
            coeff * upper_slope // la < 0 → multiplied by upper_slopes
        } else {
            coeff * lower_slope // la > 0 → multiplied by lower_slopes
        };
        if !lower_product.is_finite() {
            prop_assert_eq!(
                result.lower_a[[0, 0]], 0.0,
                "lower overflow: lower_a not zeroed"
            );
            prop_assert_eq!(
                result.lower_b[0],
                f32::NEG_INFINITY,
                "lower overflow: lower_b not -Inf"
            );
        }

        let upper_product = if negate_both {
            coeff * lower_slope // ua < 0 → multiplied by lower_slopes
        } else {
            coeff * upper_slope // ua > 0 → multiplied by upper_slopes
        };
        if !upper_product.is_finite() {
            prop_assert_eq!(
                result.upper_a[[0, 0]], 0.0,
                "upper overflow: upper_a not zeroed"
            );
            prop_assert_eq!(
                result.upper_b[0], f32::INFINITY,
                "upper overflow: upper_b not +Inf"
            );
        }

        // Control row (coeff=1.0) should always be finite
        prop_assert!(result.lower_b[1].is_finite(), "row 1 lower_b not finite");
        prop_assert!(result.upper_b[1].is_finite(), "row 1 upper_b not finite");
    }

    /// #2786: Verify directed rounding on slope products in CROWN backward.
    ///
    /// Property: For any finite coefficient and slope, the backward pass must
    /// produce slope products rounded in the sound direction:
    /// - lower_a values are <= the f64 reference product (toward -inf)
    /// - upper_a values are >= the f64 reference product (toward +inf)
    ///
    /// This ensures that 1-ULP f32 multiplication rounding errors cannot make
    /// bounds unsoundly tight.
    #[test]
    fn proptest_crown_backward_slope_directed_rounding_2786(
        coeff in -1e6f32..1e6,
        lower_slope in 0.0f32..100.0,
        upper_slope in 0.0f32..100.0,
        pre_half in 0.01f32..5.0,
    ) {
        use ny_tensor::{next_down_f32, next_up_f32};

        // Skip zero coefficient (no multiplication happens in that branch)
        prop_assume!(coeff != 0.0 && coeff.is_finite());
        prop_assume!(lower_slope.is_finite());
        prop_assume!(upper_slope.is_finite());

        let relax_ls = lower_slope;
        let relax_us = upper_slope;
        let relax_fn = move |_l: f32, _u: f32| {
            LinearRelaxation::new(relax_ls, 0.0, relax_us, 0.0)
        };

        let bounds = LinearBounds::new(
            array![[coeff]],
            Array1::zeros(1),
            array![[coeff]],
            Array1::zeros(1),
        ).unwrap();
        let pre = BoundedTensor::new(
            array![-pre_half].into_dyn(),
            array![pre_half].into_dyn(),
        ).map_err(|e| TestCaseError::fail(format!("{e}")))?;

        let result = crown_elementwise_backward(&bounds, &pre, relax_fn)
            .map_err(|e| TestCaseError::fail(format!("{e}")))?;

        // Determine which slope was used for each bound direction
        // (positive coeff uses lower_slope for lower, negative uses upper_slope for lower)
        let (lower_used_slope, upper_used_slope) = if coeff > 0.0 {
            (lower_slope, upper_slope)
        } else {
            (upper_slope, lower_slope)
        };

        // Skip if the f32 product overflows (handled by separate overflow tests)
        let f32_lower_product = coeff * lower_used_slope;
        let f32_upper_product = coeff * upper_used_slope;
        if !f32_lower_product.is_finite() || !f32_upper_product.is_finite() {
            return Ok(());
        }

        // f64 reference products (higher precision than the f32 computation)
        let f64_lower_product = coeff as f64 * lower_used_slope as f64;
        let f64_upper_product = coeff as f64 * upper_used_slope as f64;

        // lower_a must be <= f64 reference (directed rounding toward -inf)
        // next_down_f32(f32_product) is guaranteed <= true mathematical product
        // because f32_product is within 0.5 ULP of truth, and next_down moves 1 ULP down.
        prop_assert!(
            (result.lower_a[[0, 0]] as f64) <= f64_lower_product,
            "UNSOUND: lower_a ({}) > f64 product ({}) for coeff={}, slope={}",
            result.lower_a[[0, 0]], f64_lower_product, coeff, lower_used_slope,
        );

        // upper_a must be >= f64 reference (directed rounding toward +inf)
        prop_assert!(
            (result.upper_a[[0, 0]] as f64) >= f64_upper_product,
            "UNSOUND: upper_a ({}) < f64 product ({}) for coeff={}, slope={}",
            result.upper_a[[0, 0]], f64_upper_product, coeff, upper_used_slope,
        );

        // Verify the values match expected next_down/next_up exactly
        let expected_lower = next_down_f32(f32_lower_product);
        let expected_upper = next_up_f32(f32_upper_product);
        prop_assert_eq!(
            result.lower_a[[0, 0]], expected_lower,
            "lower_a should be next_down_f32(coeff * slope)"
        );
        prop_assert_eq!(
            result.upper_a[[0, 0]], expected_upper,
            "upper_a should be next_up_f32(coeff * slope)"
        );
    }
}
