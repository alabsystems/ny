// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::activations::LinearRelaxation;
use crate::*;
use ndarray::{ArrayD, IxDyn};
use proptest::prelude::*;

// ==================== PowConstant tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_pow_square_positive() {
    // Test x^2 for positive bounds
    let lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let pow = PowConstantLayer::square();
    let output = pow.propagate_ibp(&input).unwrap();

    // [2, 4]^2 = [4, 16]
    // Production code uses directed rounding: next_down for lower, next_up for upper.
    // This shifts bounds by 1 ULP (~1.9e-6 at magnitude 16), so check soundness
    // (containment) and tightness separately.
    for i in 0..3 {
        assert!(
            output.lower()[[i]] <= 4.0,
            "lower bound must be <= exact 2^2=4, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.lower()[[i]] - 4.0).abs() < 1e-5,
            "lower bound should be tight (within 1e-5 of 4.0), got {}",
            output.lower()[[i]]
        );
        assert!(
            output.upper()[[i]] >= 16.0,
            "upper bound must be >= exact 4^2=16, got {}",
            output.upper()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 16.0).abs() < 1e-5,
            "upper bound should be tight (within 1e-5 of 16.0), got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_pow_square_negative() {
    // Test x^2 for negative bounds
    let lower = ArrayD::from_elem(IxDyn(&[2]), -4.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), -2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let pow = PowConstantLayer::square();
    let output = pow.propagate_ibp(&input).unwrap();

    // [-4, -2]^2 = [4, 16] (monotonically decreasing for negative x)
    // Directed rounding: next_down for lower, next_up for upper (~1 ULP shift).
    for i in 0..2 {
        assert!(
            output.lower()[[i]] <= 4.0,
            "lower bound must be <= exact (-2)^2=4, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.lower()[[i]] - 4.0).abs() < 1e-5,
            "lower bound should be tight (within 1e-5 of 4.0), got {}",
            output.lower()[[i]]
        );
        assert!(
            output.upper()[[i]] >= 16.0,
            "upper bound must be >= exact (-4)^2=16, got {}",
            output.upper()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 16.0).abs() < 1e-5,
            "upper bound should be tight (within 1e-5 of 16.0), got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_pow_square_straddles_zero() {
    // Test x^2 when bounds straddle zero
    let lower = ArrayD::from_elem(IxDyn(&[2]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let pow = PowConstantLayer::square();
    let output = pow.propagate_ibp(&input).unwrap();

    // [-3, 2]^2 = [0, 9] (minimum at 0, max is max(9, 4) = 9)
    for i in 0..2 {
        assert!(
            output.lower()[[i]].abs() < 1e-6,
            "Min of x^2 for [-3, 2] should be 0, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 9.0).abs() < 1e-6,
            "Max of x^2 for [-3, 2] should be 9, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_pow_square_crown_positive_region_coefficients() {
    let pre_lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[3]), 4.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let pow = PowConstantLayer::square();

    let result = pow
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // For [2,4], both upper (chord) and lower (tangent at midpoint) have slope 6.
    // Intercepts include closed-form safety margins (see pow2_linear_relaxation):
    // margin = 4·ε·max(l², u²) = 4·2⁻²³·16 ≈ 7.6e-6. Lower shifted down, upper shifted up.
    for i in 0..3 {
        assert!((result.lower_a[[i, i]] - 6.0).abs() < 1e-6);
        assert!((result.upper_a[[i, i]] - 6.0).abs() < 1e-6);
        // Lower intercept: -m² - margin ≈ -9 - 7.6e-6
        assert!(
            result.lower_b[i] < -9.0,
            "lower intercept should be below -9.0"
        );
        assert!(
            (result.lower_b[i] + 9.0).abs() < 1e-4,
            "lower intercept should be close to -9.0"
        );
        // Upper intercept: -l*u + margin ≈ -8 + 7.6e-6
        assert!(
            result.upper_b[i] > -8.0,
            "upper intercept should be above -8.0"
        );
        assert!(
            (result.upper_b[i] + 8.0).abs() < 1e-4,
            "upper intercept should be close to -8.0"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_pow_square_crown_crosses_zero_lower_is_zero() {
    let pre_lower = ArrayD::from_elem(IxDyn(&[2]), -3.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[2]), 2.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(2);
    let pow = PowConstantLayer::square();

    let result = pow
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Crossing 0: use y >= 0 lower bound.
    for i in 0..2 {
        assert!(result.lower_a[[i, i]].abs() < 1e-6);
        assert!(result.lower_b[i].abs() < 1e-6);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_pow_square_crown_soundness() {
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, 0.5, -1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 2.0, 4.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let pow = PowConstantLayer::square();

    let result = pow
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    let test_points: [Vec<f32>; 4] = [
        vec![-2.0, 0.5, -1.0], // lower
        vec![3.0, 2.0, 4.0],   // upper
        vec![0.0, 1.0, 0.0],   // middle
        vec![-1.0, 1.5, 2.0],  // random
    ];

    for point in &test_points {
        let pow_output: Vec<f32> = point.iter().map(|x| x * x).collect();

        for (j, &pow_val) in pow_output.iter().enumerate() {
            let lb_val: f32 = (0..3)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..3)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-4;
            assert!(
                lb_val <= pow_val + tol,
                "Lower bound violated at point {:?}: lb {} > x^2 {}",
                point,
                lb_val,
                pow_val
            );
            assert!(
                ub_val + tol >= pow_val,
                "Upper bound violated at point {:?}: ub {} < x^2 {}",
                point,
                ub_val,
                pow_val
            );
        }
    }
}

/// Regression test for #1780: x^2 CROWN relaxation must remain sound on tiny intervals.
#[ntest::timeout(10000)]
#[test]
fn test_pow_square_crown_subnormal_and_near_degenerate_soundness_1780() {
    let cases = [
        // Counterexample region from proof/audit notes (non-crossing negative interval).
        (-1.53e-5f32, -3.9e-10f32),
        // Tiny non-point interval that previously risked being treated as degenerate.
        (1.0e-7f32, 1.0e-7f32 + 5.0e-13f32),
    ];

    for (l, u) in cases {
        assert!(
            l < u,
            "test interval must be non-degenerate: [{l:e}, {u:e}]"
        );
        let LinearRelaxation {
            lower_slope,
            lower_intercept,
            upper_slope,
            upper_intercept,
        } = pow2_linear_relaxation(l, u);

        // Validate endpoints and interior samples.
        for i in 0..=256 {
            let t = i as f32 / 256.0;
            let x = l + (u - l) * t;
            let fx = x * x;
            let lb = lower_slope * x + lower_intercept;
            let ub = upper_slope * x + upper_intercept;

            assert!(
                lb <= fx,
                "Pow2 lower relaxation unsound at x={x:e} in [{l:e}, {u:e}]: lb={lb:e}, x2={fx:e}, slope={lower_slope:e}, intercept={lower_intercept:e}"
            );
            assert!(
                ub >= fx,
                "Pow2 upper relaxation unsound at x={x:e} in [{l:e}, {u:e}]: ub={ub:e}, x2={fx:e}, slope={upper_slope:e}, intercept={upper_intercept:e}"
            );
        }
    }
}

/// Regression test for #1795: Kani found Pow2 lower bound unsound at subnormal scale.
///
/// Concrete counterexample from Kani: l ≈ -3.74e-23, u = -MIN_POSITIVE.
/// The tangent intercept rounds to -0.0, slope*x rounds up to MIN_POSITIVE_SUBNORMAL,
/// but true x² underflows to 0.0 → lower bound exceeds x².
/// Fix: detect subnormal-scale inputs and return constant y = 0 lower bound.
#[ntest::timeout(10000)]
#[test]
fn test_pow_square_crown_subnormal_lower_bound_regression_1795() {
    // Exact Kani counterexample values (little-endian f32 bit patterns)
    let l: f32 = f32::from_bits(0x9A35_04F3); // ≈ -3.743392e-23
    let u: f32 = f32::from_bits(0x8080_0000); // = -f32::MIN_POSITIVE ≈ -1.175494e-38
                                              // Verify the test values match the issue description
    assert!(l < u, "l={l:e} should be < u={u:e}");
    assert!(l < 0.0 && u < 0.0, "both should be negative");

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = pow2_linear_relaxation(l, u);

    // The fix: subnormal inputs should get constant y=0 lower bound
    assert_eq!(
        lower_slope, 0.0,
        "subnormal-scale pow2 lower slope should be 0"
    );
    assert_eq!(
        lower_intercept, 0.0,
        "subnormal-scale pow2 lower intercept should be 0"
    );

    // Verify soundness at the counterexample point x ≈ -1.87e-23
    let x: f32 = f32::from_bits(0x99B5_0880); // ≈ -1.871839e-23
    let true_sq = x * x; // underflows to 0.0 in f32
    let lb = lower_slope * x + lower_intercept;
    assert!(
        lb <= true_sq,
        "lower bound {lb:e} must not exceed true x²={true_sq:e} at x={x:e}"
    );

    // Upper bound should still be valid
    let ub = upper_slope * x + upper_intercept;
    assert!(
        ub >= true_sq,
        "upper bound {ub:e} must not be below true x²={true_sq:e} at x={x:e}"
    );
}

/// Regression test for issue #1644: fractional exponent with negative inputs must error.
#[ntest::timeout(10000)]
#[test]
fn test_pow_fractional_rejects_negative_input_1644() {
    // x^0.3 with x ∈ [-1, 4] — undefined for x < 0
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 4.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(0.3);
    let err = pow
        .propagate_ibp(&input)
        .expect_err("Fractional exponent with negative input should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("non-integer exponent") || msg.contains("non-negative"),
        "error should mention non-integer exponent requirement: {msg}"
    );
}

/// Regression test for issue #1644: non-integer exponent > 1 with negative inputs must error.
#[ntest::timeout(10000)]
#[test]
fn test_pow_non_integer_gt1_rejects_negative_input_1644() {
    // x^1.5 with x ∈ [-2, 3] — undefined for x < 0
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -2.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 3.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(1.5);
    let err = pow
        .propagate_ibp(&input)
        .expect_err("Non-integer exponent with negative input should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("non-integer exponent") || msg.contains("non-negative"),
        "error should mention requirement: {msg}"
    );
}

/// Regression test for issue #1653: x^4 with positive inputs.
#[ntest::timeout(10000)]
#[test]
fn test_pow_fourth_positive_1653() {
    // x^4 with x ∈ [2, 3] → [16, 81]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), 2.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 3.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(4.0);
    let output = pow.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - 16.0).abs() < 1e-3,
            "2^4 should be 16, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 81.0).abs() < 1e-3,
            "3^4 should be 81, got {}",
            output.upper()[[i]]
        );
    }
}

/// Regression test for issue #1653: x^4 with negative inputs.
#[ntest::timeout(10000)]
#[test]
fn test_pow_fourth_negative_1653() {
    // x^4 with x ∈ [-4, -2] → [16, 256] (monotonically decreasing for negative x)
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -4.0f32),
        ArrayD::from_elem(IxDyn(&[2]), -2.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(4.0);
    let output = pow.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - 16.0).abs() < 1e-3,
            "(-2)^4 should be 16, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 256.0).abs() < 1e-3,
            "(-4)^4 should be 256, got {}",
            output.upper()[[i]]
        );
    }
}

/// Regression test for issue #1653: x^4 straddling zero.
#[ntest::timeout(10000)]
#[test]
fn test_pow_fourth_straddles_zero_1653() {
    // x^4 with x ∈ [-3, 2] → [0, 81] (minimum at 0, max(81, 16) = 81)
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -3.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 2.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(4.0);
    let output = pow.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            output.lower()[[i]].abs() < 1e-6,
            "Min of x^4 for [-3, 2] should be 0, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 81.0).abs() < 1e-3,
            "Max of x^4 for [-3, 2] should be 81, got {}",
            output.upper()[[i]]
        );
    }
}

/// Regression test for issue #1653: x^6 straddling zero.
#[ntest::timeout(10000)]
#[test]
fn test_pow_sixth_straddles_zero_1653() {
    // x^6 with x ∈ [-2, 3] → [0, 729] (minimum at 0, max(64, 729) = 729)
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -2.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 3.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(6.0);
    let output = pow.propagate_ibp(&input).unwrap();

    assert!(
        output.lower()[[0]].abs() < 1e-6,
        "Min of x^6 for [-2, 3] should be 0, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 729.0).abs() < 1e-4,
        "Max of x^6 for [-2, 3] should be 729, got {}",
        output.upper()[[0]]
    );
}

/// Odd integer exponent (x^3) should preserve monotonicity, not use even path.
#[ntest::timeout(10000)]
#[test]
fn test_pow_cube_negative_monotonic() {
    // x^3 with x ∈ [-3, -1] → [-27, -1] (monotonically increasing)
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -3.0f32),
        ArrayD::from_elem(IxDyn(&[1]), -1.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(3.0);
    let output = pow.propagate_ibp(&input).unwrap();

    assert!(
        (output.lower()[[0]] - (-27.0)).abs() < 1e-3,
        "(-3)^3 should be -27, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - (-1.0)).abs() < 1e-3,
        "(-1)^3 should be -1, got {}",
        output.upper()[[0]]
    );
}

/// Fractional exponent with all-positive inputs should succeed.
#[ntest::timeout(10000)]
#[test]
fn test_pow_fractional_positive_input_succeeds() {
    // x^0.3 with x ∈ [1, 4] — well-defined
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 4.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(0.3);
    let output = pow.propagate_ibp(&input).unwrap();

    // 1^0.3 = 1, 4^0.3 ≈ 1.516
    assert!(
        (output.lower()[[0]] - 1.0).abs() < 1e-5,
        "1^0.3 should be 1, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 4.0_f32.powf(0.3)).abs() < 1e-5,
        "4^0.3 should be ~1.516, got {}",
        output.upper()[[0]]
    );
}

// ==================== Negative exponent tests (issue #1657) ====================

/// Regression test for #1657: p=-1, all-positive interval.
/// x^{-1} is monotonically decreasing for x > 0: [2, 4]^{-1} = [0.25, 0.5]
#[ntest::timeout(10000)]
#[test]
fn test_pow_neg1_positive_1657() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), 2.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 4.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(-1.0);
    let output = pow.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - 0.25).abs() < 1e-6,
            "4^(-1) should be 0.25, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 0.5).abs() < 1e-6,
            "2^(-1) should be 0.5, got {}",
            output.upper()[[i]]
        );
    }
}

/// Regression test for #1657: p=-1, all-negative interval.
/// x^{-1} is monotonically decreasing for x < 0: [-4, -2]^{-1} = [-0.5, -0.25]
#[ntest::timeout(10000)]
#[test]
fn test_pow_neg1_negative_1657() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -4.0f32),
        ArrayD::from_elem(IxDyn(&[1]), -2.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(-1.0);
    let output = pow.propagate_ibp(&input).unwrap();

    assert!(
        (output.lower()[[0]] - (-0.5)).abs() < 1e-6,
        "(-2)^(-1) should be -0.5, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - (-0.25)).abs() < 1e-6,
        "(-4)^(-1) should be -0.25, got {}",
        output.upper()[[0]]
    );
}

/// Regression test for #1657: p=-1 with zero-crossing must error.
#[ntest::timeout(10000)]
#[test]
fn test_pow_neg1_zero_crossing_rejects_1657() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(-1.0);
    let err = pow
        .propagate_ibp(&input)
        .expect_err("p=-1 with zero-crossing should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("excluding zero") || msg.contains("negative exponent"),
        "error should mention zero exclusion: {msg}"
    );
}

/// Regression test for #1657: p=-2, all-positive interval.
/// x^{-2} for x > 0 is monotonically decreasing: [2, 4]^{-2} = [1/16, 1/4]
#[ntest::timeout(10000)]
#[test]
fn test_pow_neg2_positive_1657() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 4.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(-2.0);
    let output = pow.propagate_ibp(&input).unwrap();

    // [2,4]^{-2} = [1/16, 1/4] = [0.0625, 0.25]
    assert!(
        (output.lower()[[0]] - 0.0625).abs() < 1e-6,
        "4^(-2) should be 0.0625, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 0.25).abs() < 1e-6,
        "2^(-2) should be 0.25, got {}",
        output.upper()[[0]]
    );
}

/// Regression test for #1657: p=-2, all-negative interval.
/// x^{-2} for x < 0 is monotonically increasing: [-4, -2]^{-2} = [1/16, 1/4]
#[ntest::timeout(10000)]
#[test]
fn test_pow_neg2_negative_1657() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -4.0f32),
        ArrayD::from_elem(IxDyn(&[1]), -2.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(-2.0);
    let output = pow.propagate_ibp(&input).unwrap();

    // [-4,-2]^{-2} = [1/16, 1/4] = [0.0625, 0.25]
    assert!(
        (output.lower()[[0]] - 0.0625).abs() < 1e-6,
        "(-4)^(-2) should be 0.0625, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 0.25).abs() < 1e-6,
        "(-2)^(-2) should be 0.25, got {}",
        output.upper()[[0]]
    );
}

/// Regression test for #1657: p=-2 with zero-crossing must error.
#[ntest::timeout(10000)]
#[test]
fn test_pow_neg2_zero_crossing_rejects_1657() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -3.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(-2.0);
    let err = pow
        .propagate_ibp(&input)
        .expect_err("p=-2 with zero-crossing should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("excluding zero") || msg.contains("negative exponent"),
        "error should mention zero exclusion: {msg}"
    );
}

/// Regression test for #1657: p=-0.5 with zero input must error.
#[ntest::timeout(10000)]
#[test]
fn test_pow_neg_half_zero_rejects_1657() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 4.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(-0.5);
    let err = pow
        .propagate_ibp(&input)
        .expect_err("p=-0.5 with zero input should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("excluding zero") || msg.contains("negative exponent"),
        "error should mention zero exclusion: {msg}"
    );
}

/// Regression test for #1657: p=-0.5 with all-positive input succeeds.
/// x^{-0.5} = 1/sqrt(x), monotonically decreasing for x > 0.
#[ntest::timeout(10000)]
#[test]
fn test_pow_neg_half_positive_1657() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 4.0f32),
    )
    .unwrap();

    let pow = PowConstantLayer::new(-0.5);
    let output = pow.propagate_ibp(&input).unwrap();

    // [1,4]^{-0.5} = [4^{-0.5}, 1^{-0.5}] = [0.5, 1.0] (decreasing)
    assert!(
        (output.lower()[[0]] - 0.5).abs() < 1e-5,
        "4^(-0.5) should be 0.5, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 1.0).abs() < 1e-5,
        "1^(-0.5) should be 1.0, got {}",
        output.upper()[[0]]
    );
}

// ==================== Per-element zero-crossing tests (issue #1699) ====================

/// Regression test for #1699: multi-element tensor with mixed-sign elements
/// that individually don't cross zero should NOT be rejected.
/// Element 0: [2, 4] (positive), Element 1: [-4, -2] (negative).
/// The old global check would see min_lower=-4 and max_upper=4, falsely rejecting.
#[ntest::timeout(10000)]
#[test]
fn test_pow_neg1_multi_element_mixed_sign_no_crossing_1699() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0f32, -4.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![4.0f32, -2.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let pow = PowConstantLayer::new(-1.0);
    let output = pow
        .propagate_ibp(&input)
        .expect("Mixed-sign elements without zero-crossing should succeed");

    // Element 0: [2,4]^{-1} = [0.25, 0.5]
    assert!(
        (output.lower()[[0]] - 0.25).abs() < 1e-6,
        "4^(-1) should be 0.25, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 0.5).abs() < 1e-6,
        "2^(-1) should be 0.5, got {}",
        output.upper()[[0]]
    );
    // Element 1: [-4,-2]^{-1} = [-0.5, -0.25]
    assert!(
        (output.lower()[[1]] - (-0.5)).abs() < 1e-6,
        "(-2)^(-1) should be -0.5, got {}",
        output.lower()[[1]]
    );
    assert!(
        (output.upper()[[1]] - (-0.25)).abs() < 1e-6,
        "(-4)^(-1) should be -0.25, got {}",
        output.upper()[[1]]
    );
}

/// Regression test for #1699: per-element check still rejects individual elements crossing zero.
#[ntest::timeout(10000)]
#[test]
fn test_pow_neg1_multi_element_one_crosses_zero_rejects_1699() {
    // Element 0: [2, 4] (positive, fine), Element 1: [-1, 1] (crosses zero!)
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0f32, -1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![4.0f32, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let pow = PowConstantLayer::new(-1.0);
    let err = pow
        .propagate_ibp(&input)
        .expect_err("One element crossing zero should still be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("element 1"),
        "error should identify the offending element: {msg}"
    );
}

/// Regression test for #1699: p=-2 with 3-element mixed sign, no individual zero-crossing.
#[ntest::timeout(10000)]
#[test]
fn test_pow_neg2_multi_element_mixed_sign_1699() {
    // Elements: [1,3], [-5,-2], [0.5,1.0] — none cross zero individually
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0f32, -5.0, 0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0f32, -2.0, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let pow = PowConstantLayer::new(-2.0);
    let output = pow
        .propagate_ibp(&input)
        .expect("No element crosses zero, should succeed");

    // Element 0: [1,3]^{-2} = [1/9, 1] (decreasing for positive x)
    assert!(
        (output.lower()[[0]] - 1.0 / 9.0).abs() < 1e-5,
        "3^(-2) should be 1/9, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 1.0).abs() < 1e-5,
        "1^(-2) should be 1, got {}",
        output.upper()[[0]]
    );
    // Element 1: [-5,-2]^{-2} = [1/25, 1/4] (increasing for negative x)
    assert!(
        (output.lower()[[1]] - 1.0 / 25.0).abs() < 1e-5,
        "(-5)^(-2) should be 1/25, got {}",
        output.lower()[[1]]
    );
    assert!(
        (output.upper()[[1]] - 0.25).abs() < 1e-5,
        "(-2)^(-2) should be 0.25, got {}",
        output.upper()[[1]]
    );
    // Element 2: [0.5,1.0]^{-2} = [1, 4] (decreasing for positive x)
    assert!(
        (output.lower()[[2]] - 1.0).abs() < 1e-5,
        "1.0^(-2) should be 1.0, got {}",
        output.lower()[[2]]
    );
    assert!(
        (output.upper()[[2]] - 4.0).abs() < 1e-5,
        "0.5^(-2) should be 4.0, got {}",
        output.upper()[[2]]
    );
}

/// Regression test for #2911: extreme exponent overflows i32 cast in parity check.
/// Before fix: (1e30_f32.round() as i32) saturates to i32::MAX (odd), misclassifying
/// an even exponent as odd. After fix: returns InvalidSpec error.
#[ntest::timeout(10000)]
#[test]
fn test_pow_extreme_exponent_returns_error_2911() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -2.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
    )
    .unwrap();

    // 1e30 is an integer-like float (rounds to itself) but exceeds i32 range
    let pow = PowConstantLayer::new(1e30);
    let err = pow
        .propagate_ibp(&input)
        .expect_err("Extreme exponent should error, not silently misclassify parity");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "Expected InvalidSpec, got: {err:?}"
    );

    // Inf exponent should be rejected at construction.
    let err_inf =
        PowConstantLayer::try_new(f32::INFINITY).expect_err("Infinite exponent should be rejected");
    assert!(
        matches!(&err_inf, NyError::InvalidSpec(_)),
        "Expected InvalidSpec, got: {err_inf:?}"
    );

    // NaN exponent should also be rejected at construction.
    let err_nan = PowConstantLayer::try_new(f32::NAN).expect_err("NaN exponent should be rejected");
    assert!(
        matches!(&err_nan, NyError::InvalidSpec(_)),
        "Expected InvalidSpec, got: {err_nan:?}"
    );
}

// Proptest for #1699: random multi-element tensors with negative exponents.
// Each element either positive-only or negative-only (no zero crossing).
// Verifies IBP bounds contain the true output at sampled points.
proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    #[test]
    fn proptest_pow_neg_exponent_multi_element_soundness_1699(
        // Generate 2-8 elements
        n in 2usize..=8,
        seed in any::<u64>(),
        // Negative integer exponents: -1, -2, -3
        exp_idx in 0usize..3,
    ) {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let exponents = [-1.0f32, -2.0, -3.0];
        let p = exponents[exp_idx];

        let mut rng = SmallRng::seed_from_u64(seed);
        let mut lower_vec = Vec::with_capacity(n);
        let mut upper_vec = Vec::with_capacity(n);

        for _ in 0..n {
            // Randomly decide positive or negative interval (avoiding zero)
            let positive: bool = rng.random_bool(0.5);
            if positive {
                let l: f32 = rng.random_range(0.1..10.0);
                let u: f32 = rng.random_range(l..l + 10.0);
                lower_vec.push(l);
                upper_vec.push(u);
            } else {
                let u: f32 = rng.random_range(-10.0..-0.1);
                let l: f32 = rng.random_range(u - 10.0..u);
                lower_vec.push(l);
                upper_vec.push(u);
            }
        }

        let lower = ArrayD::from_shape_vec(IxDyn(&[n]), lower_vec.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[n]), upper_vec.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let pow = PowConstantLayer::new(p);
        let output = pow.propagate_ibp(&input).unwrap();

        // Verify soundness: true output at endpoints must be within [lower, upper]
        for i in 0..n {
            let true_at_lower = lower_vec[i].powf(p);
            let true_at_upper = upper_vec[i].powf(p);

            // Check both endpoints are contained in the output bounds
            let tol = 1e-4;
            prop_assert!(
                output.lower()[[i]] <= true_at_lower + tol,
                "Element {}: lower bound {} > f({})^{} = {}",
                i, output.lower()[[i]], lower_vec[i], p, true_at_lower
            );
            prop_assert!(
                output.lower()[[i]] <= true_at_upper + tol,
                "Element {}: lower bound {} > f({})^{} = {}",
                i, output.lower()[[i]], upper_vec[i], p, true_at_upper
            );
            prop_assert!(
                output.upper()[[i]] + tol >= true_at_lower,
                "Element {}: upper bound {} < f({})^{} = {}",
                i, output.upper()[[i]], lower_vec[i], p, true_at_lower
            );
            prop_assert!(
                output.upper()[[i]] + tol >= true_at_upper,
                "Element {}: upper bound {} < f({})^{} = {}",
                i, output.upper()[[i]], upper_vec[i], p, true_at_upper
            );
        }
    }
}
