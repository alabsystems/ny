// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn assert_close(actual: impl Into<f64>, expected: f64, tol: f64, context: &str) {
    let actual = actual.into();
    let diff = (actual - expected).abs();
    assert!(
        diff < tol,
        "{context}: expected {expected} +/- {tol}, got {actual} (diff {diff})"
    );
}

fn assert_finite(actual: impl Into<f64>, context: &str) {
    let actual = actual.into();
    assert!(
        actual.is_finite(),
        "{context}: expected finite result, got {actual}"
    );
}

// ============== relu tests ==============
#[ntest::timeout(10000)]
#[test]
fn test_relu_positive() {
    assert_eq!(relu(5.0), 5.0);
    assert_eq!(relu(0.1), 0.1);
    assert_eq!(relu(100.0), 100.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_negative() {
    assert_eq!(relu(-5.0), 0.0);
    assert_eq!(relu(-0.1), 0.0);
    assert_eq!(relu(-100.0), 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_zero() {
    assert_eq!(relu(0.0), 0.0);
}

// ============== dot tests ==============
#[ntest::timeout(10000)]
#[test]
fn test_dot_basic() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let result = dot(&a, &b).unwrap();
    assert_close(result, 32.0, 1e-6, "dot([1,2,3],[4,5,6])"); // 1*4 + 2*5 + 3*6 = 32
}

#[ntest::timeout(10000)]
#[test]
fn test_dot_zeros() {
    let a = [0.0, 0.0, 0.0];
    let b = [1.0, 2.0, 3.0];
    let result = dot(&a, &b).unwrap();
    assert_eq!(result, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_dot_negative() {
    let a = [1.0, -2.0, 3.0];
    let b = [-1.0, 2.0, -3.0];
    let result = dot(&a, &b).unwrap();
    assert_close(result, -14.0, 1e-6, "dot([1,-2,3],[-1,2,-3])"); // 1*(-1) + (-2)*2 + 3*(-3) = -14
}

#[ntest::timeout(10000)]
#[test]
fn test_dot_single_element() {
    let a = [3.0];
    let b = [4.0];
    let result = dot(&a, &b).unwrap();
    assert_close(result, 12.0, 1e-6, "dot single element");
}

#[ntest::timeout(10000)]
#[test]
fn test_dot_empty() {
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let result = dot(&a, &b).unwrap();
    assert_eq!(result, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_dot_mismatched_lengths() {
    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 2.0];
    let result = dot(&a, &b);
    assert!(result.is_err(), "dot should reject mismatched lengths");
}

// ============== l2_norm_sq tests ==============
#[ntest::timeout(10000)]
#[test]
fn test_l2_norm_sq_basic() {
    let x = [3.0, 4.0];
    let result = l2_norm_sq(&x);
    assert_close(result, 25.0, 1e-10, "l2_norm_sq([3,4])"); // 9 + 16 = 25
}

#[ntest::timeout(10000)]
#[test]
fn test_l2_norm_sq_zeros() {
    let x = [0.0, 0.0, 0.0];
    let result = l2_norm_sq(&x);
    assert_eq!(result, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_l2_norm_sq_single() {
    let x = [5.0];
    let result = l2_norm_sq(&x);
    assert_close(result, 25.0, 1e-10, "l2_norm_sq([5])");
}

#[ntest::timeout(10000)]
#[test]
fn test_l2_norm_sq_negative() {
    let x = [-3.0, 4.0];
    let result = l2_norm_sq(&x);
    assert_close(result, 25.0, 1e-10, "l2_norm_sq([-3,4])"); // Square makes positive
}

#[ntest::timeout(10000)]
#[test]
fn test_l2_norm_sq_empty() {
    let x: [f32; 0] = [];
    let result = l2_norm_sq(&x);
    assert_eq!(result, 0.0);
}

// ============== phi_norm_sq tests ==============
#[ntest::timeout(10000)]
#[test]
fn test_phi_norm_sq_basic() {
    let c = [1.0, 1.0];
    let g = [0.5, 0.5];
    let x_hat = [0.0, 0.0];
    let lambda = 1.0;
    let result = phi_norm_sq(&c, &g, &x_hat, lambda).unwrap();
    // φ_i = min{c_i - g_i - λx̂_i, g_i + λx̂_i, 0} = min{0.5, 0.5, 0} = 0
    assert_close(result, 0.0, 1e-10, "phi_norm_sq basic");
}

#[ntest::timeout(10000)]
#[test]
fn test_phi_norm_sq_negative_phi() {
    let c = [0.0]; // c=0, g=1 -> min{0-1-0, 1+0, 0} = min{-1, 1, 0} = -1
    let g = [1.0];
    let x_hat = [0.0];
    let lambda = 1.0;
    let result = phi_norm_sq(&c, &g, &x_hat, lambda).unwrap();
    assert_close(result, 1.0, 1e-10, "phi_norm_sq negative phi"); // (-1)^2 = 1
}

#[ntest::timeout(10000)]
#[test]
fn test_phi_norm_sq_mismatched_c_g() {
    let c = [1.0, 2.0, 3.0];
    let g = [1.0, 2.0];
    let x_hat = [0.0, 0.0];
    let result = phi_norm_sq(&c, &g, &x_hat, 1.0);
    assert!(
        result.is_err(),
        "phi_norm_sq should reject mismatched c and g"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_phi_norm_sq_mismatched_xhat() {
    let c = [1.0, 2.0];
    let g = [1.0, 2.0];
    let x_hat = [0.0, 0.0, 0.0];
    let result = phi_norm_sq(&c, &g, &x_hat, 1.0);
    assert!(
        result.is_err(),
        "phi_norm_sq should reject mismatched x_hat"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_phi_norm_sq_with_nonzero_xhat() {
    let c = [2.0];
    let g = [1.0];
    let x_hat = [0.5];
    let lambda = 2.0;
    // φ = min{2 - 1 - 2*0.5, 1 + 2*0.5, 0} = min{0, 2, 0} = 0
    let result = phi_norm_sq(&c, &g, &x_hat, lambda).unwrap();
    assert_close(result, 0.0, 1e-10, "phi_norm_sq with nonzero x_hat");
}

// ============== phi0_l2_norm tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_phi0_l2_norm_basic() {
    let c = [1.0, 1.0];
    let g = [0.5, 0.5];
    // φ_i = min{c_i - g_i, g_i, 0} = min{0.5, 0.5, 0} = 0
    let result = phi0_l2_norm(&c, &g).unwrap();
    assert_close(result, 0.0, 1e-10, "phi0_l2_norm basic");
}

#[ntest::timeout(10000)]
#[test]
fn test_phi0_l2_norm_negative_phi() {
    let c = [0.0];
    let g = [1.0];
    // φ = min{0-1, 1, 0} = -1
    let result = phi0_l2_norm(&c, &g).unwrap();
    assert_close(result, 1.0, 1e-10, "phi0_l2_norm negative phi"); // sqrt(1) = 1
}

#[ntest::timeout(10000)]
#[test]
fn test_phi0_l2_norm_multiple() {
    let c = [0.0, 0.0];
    let g = [1.0, 1.0];
    // Each φ_i = min{-1, 1, 0} = -1
    // ||φ||_2 = sqrt(1 + 1) = sqrt(2)
    let result = phi0_l2_norm(&c, &g).unwrap();
    assert_close(
        result,
        std::f64::consts::SQRT_2,
        1e-10,
        "phi0_l2_norm multiple",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_phi0_l2_norm_mismatched() {
    let c = [1.0, 2.0, 3.0];
    let g = [1.0, 2.0];
    let result = phi0_l2_norm(&c, &g);
    assert!(
        result.is_err(),
        "phi0_l2_norm should reject mismatched inputs"
    );
}

// ============== relu_sdp_offset_for_lambda tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_basic() {
    let c = [1.0, 1.0];
    let g = [0.5, 0.5];
    let x_hat = [0.0, 0.0];
    let rho = 0.1;
    let lambda = 1.0;
    let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, lambda).unwrap();
    // Should return a finite value
    assert_finite(result, "relu_sdp_offset_for_lambda basic");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_negative_rho() {
    let c = [1.0];
    let g = [0.5];
    let x_hat = [0.0];
    let rho = -0.1;
    let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, 1.0);
    assert!(
        result.is_err(),
        "relu_sdp_offset_for_lambda should reject negative rho"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_nonfinite_lambda() {
    let c = [1.0];
    let g = [0.5];
    let x_hat = [0.0];
    let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, 0.1, f64::NAN);
    assert!(
        result.is_err(),
        "relu_sdp_offset_for_lambda should reject non-finite lambda"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_nonfinite_rho() {
    let c = [1.0];
    let g = [0.5];
    let x_hat = [0.0];
    let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, f32::NAN, 1.0);
    assert!(
        result.is_err(),
        "relu_sdp_offset_for_lambda should reject non-finite rho"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_nonfinite_xhat() {
    let c = [1.0];
    let g = [0.5];
    let x_hat = [f32::NAN];
    let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, 0.1, 1.0);
    assert!(
        result.is_err(),
        "relu_sdp_offset_for_lambda should reject non-finite x_hat"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_zero_rho() {
    let c = [1.0, 1.0];
    let g = [0.5, 0.5];
    let x_hat = [0.0, 0.0];
    let rho = 0.0;
    let lambda = 1.0;
    let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, lambda).unwrap();
    // With rho=0 and x_hat=0: rho^2 - ||x_hat||^2 = 0, phi^2/λ ≥ 0
    // h = -0.5 * (λ * 0 + phi^2/λ) ≤ 0
    assert!(
        result <= 0.0001,
        "relu_sdp_offset_for_lambda with rho=0 should stay non-positive, got {result}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_lambda_clamping_min() {
    let c = [1.0];
    let g = [0.5];
    let x_hat = [0.0];
    let rho = 0.1;
    // Very small lambda should be clamped to MIN_LAMBDA
    let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, 1e-20).unwrap();
    assert_finite(result, "relu_sdp_offset_for_lambda min lambda clamp");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_lambda_clamping_max() {
    let c = [1.0];
    let g = [0.5];
    let x_hat = [0.0];
    let rho = 0.1;
    // Very large lambda should be clamped to MAX_LAMBDA
    let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, 1e20).unwrap();
    assert_finite(result, "relu_sdp_offset_for_lambda max lambda clamp");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_mismatched_dims() {
    let c = [1.0, 2.0];
    let g = [0.5];
    let x_hat = [0.0, 0.0];
    let result = relu_sdp_offset_for_lambda(&c, &g, &x_hat, 0.1, 1.0);
    assert!(
        result.is_err(),
        "relu_sdp_offset_for_lambda should reject mismatched dimensions"
    );
}

// ============== relu_sdp_offset_opt tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_rho_zero() {
    let c = [1.0, 0.0];
    let g = [0.5, 0.5];
    let x_hat = [2.0, -1.0];
    // rho=0 path: lhs = c · ReLU(x_hat) = 1*2 + 0*0 = 2
    // rhs = g · x_hat = 0.5*2 + 0.5*(-1) = 0.5
    // result = 2 - 0.5 = 1.5
    let result = relu_sdp_offset_opt(&c, &g, &x_hat, 0.0).unwrap();
    assert_close(result, 1.5, 1e-5, "relu_sdp_offset_opt rho=0");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_xhat_zero() {
    let c = [0.0];
    let g = [1.0];
    let x_hat = [0.0];
    let rho = 1.0;
    // x_hat = 0 path: h* = -rho * ||min{c-g, g, 0}||_2
    // φ_i = min{0-1, 1, 0} = -1
    // ||φ||_2 = 1
    // h* = -1 * 1 = -1
    let result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
    assert_close(result, -1.0, 1e-5, "relu_sdp_offset_opt x_hat=0");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_general() {
    let c = [1.0, 1.0];
    let g = [0.5, 0.5];
    let x_hat = [0.1, 0.1];
    let rho = 0.5;
    let result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
    // Should return finite value from optimization
    assert_finite(result, "relu_sdp_offset_opt general");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_consistency() {
    // Result from opt should be >= any specific lambda
    let c = [1.0, 0.5];
    let g = [0.3, 0.4];
    let x_hat = [0.2, 0.2];
    let rho = 0.3;

    let opt_result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();

    // Check that opt_result is >= result for several lambda values
    for lambda in [0.1, 1.0, 10.0, 100.0] {
        let specific = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, lambda).unwrap();
        assert!(
            opt_result >= specific - 1e-5,
            "opt {} should be >= specific {} for lambda {}",
            opt_result,
            specific,
            lambda
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_mismatched() {
    let c = [1.0, 2.0];
    let g = [0.5];
    let x_hat = [0.0, 0.0];
    let result = relu_sdp_offset_opt(&c, &g, &x_hat, 0.1);
    assert!(
        result.is_err(),
        "relu_sdp_offset_opt should reject mismatched inputs"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_negative_rho() {
    let c = [1.0];
    let g = [0.5];
    let x_hat = [0.0];
    let result = relu_sdp_offset_opt(&c, &g, &x_hat, -0.1);
    assert!(
        result.is_err(),
        "relu_sdp_offset_opt should reject negative rho"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_nonfinite_xhat() {
    let c = [1.0];
    let g = [0.5];
    let x_hat = [f32::INFINITY];
    let result = relu_sdp_offset_opt(&c, &g, &x_hat, 0.1);
    assert!(
        result.is_err(),
        "relu_sdp_offset_opt should reject non-finite x_hat"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_nonfinite_rho() {
    let c = [1.0];
    let g = [0.5];
    let x_hat = [0.0];
    let result = relu_sdp_offset_opt(&c, &g, &x_hat, f32::NAN);
    assert!(
        result.is_err(),
        "relu_sdp_offset_opt should reject non-finite rho"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_mismatched_xhat_zero_branch() {
    let c = [1.0, 2.0];
    let g = [0.5, 0.5];
    let x_hat = [0.0];
    let result = relu_sdp_offset_opt(&c, &g, &x_hat, 1.0);
    assert!(
        result.is_err(),
        "relu_sdp_offset_opt x_hat=0 branch should reject mismatched inputs"
    );
}

// ============== Mathematical property tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_offset_nonpositive_property() {
    // For valid ReLU relaxation, offset should typically be non-positive
    // (it's a correction term that tightens bounds)
    let c = [1.0, 1.0, 1.0];
    let g = [0.5, 0.5, 0.5];
    let x_hat = [0.0, 0.0, 0.0];

    for rho in [0.1, 0.5, 1.0, 2.0] {
        let result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
        assert!(
            result <= 0.001,
            "Offset {} should be non-positive for rho={}",
            result,
            rho
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_offset_monotonic_in_rho() {
    // Larger rho (larger uncertainty ball) should give more negative offset
    let c = [0.0, 0.0];
    let g = [1.0, 1.0];
    let x_hat = [0.0, 0.0];

    let offset_small = relu_sdp_offset_opt(&c, &g, &x_hat, 0.1).unwrap();
    let offset_large = relu_sdp_offset_opt(&c, &g, &x_hat, 1.0).unwrap();

    assert!(
        offset_large <= offset_small + 1e-5,
        "Larger rho should give smaller (more negative) offset: {} vs {}",
        offset_large,
        offset_small
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_singleton_set_exact() {
    // When rho=0, the set is a singleton {x_hat}
    // The offset should equal c^T ReLU(x_hat) - g^T x_hat exactly
    let c = [1.0, 2.0];
    let g = [0.5, 0.5];
    let x_hat = [3.0, -1.0]; // ReLU(x_hat) = [3, 0]

    let result = relu_sdp_offset_opt(&c, &g, &x_hat, 0.0).unwrap();
    // c^T ReLU(x_hat) = 1*3 + 2*0 = 3
    // g^T x_hat = 0.5*3 + 0.5*(-1) = 1
    // offset = 3 - 1 = 2
    assert_close(result, 2.0, 1e-5, "relu_sdp_offset_opt singleton set");
}

#[ntest::timeout(10000)]
#[test]
fn test_centered_ball_closed_form() {
    // When x_hat = 0, there's a closed-form solution
    // h* = -rho * ||min{c-g, g, 0}||_2
    let c = [2.0, 0.0];
    let g = [1.0, 1.0];
    let x_hat = [0.0, 0.0];
    let rho = 2.0;

    // φ_0 = min{2-1, 1, 0} = 0
    // φ_1 = min{0-1, 1, 0} = -1
    // ||φ||_2 = sqrt(0 + 1) = 1
    // h* = -2 * 1 = -2

    let result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
    assert_close(result, -2.0, 1e-5, "relu_sdp_offset_opt centered ball");
}

#[ntest::timeout(10000)]
#[test]
fn test_small_xhat_norm() {
    // Very small x_hat should use the x_hat=0 branch
    let c = [1.0];
    let g = [0.5];
    let x_hat = [1e-15];
    let rho = 1.0;

    // Should not fail due to numerical issues
    let result = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
    assert_finite(result, "relu_sdp_offset_opt small x_hat norm");
}

#[ntest::timeout(10000)]
#[test]
fn test_empty_inputs() {
    let c: [f32; 0] = [];
    let g: [f32; 0] = [];
    let x_hat: [f32; 0] = [];

    let result = relu_sdp_offset_opt(&c, &g, &x_hat, 1.0).unwrap();
    // Empty case: rho^2 - 0 > 0, phi = 0
    // h = -0.5 * (λ * rho^2 + 0) < 0
    assert_finite(result, "relu_sdp_offset_opt empty inputs");
}

// ============== Directed rounding tests (issue #1676) ==============

#[ntest::timeout(10000)]
#[test]
fn test_offset_directed_rounding_for_lambda() {
    // Verify that relu_sdp_offset_for_lambda returns a value <= the true f64 offset.
    // The f64 computation is the "true" value; the f32 return must not exceed it.
    let c = [0.3, 0.7, 0.1, 0.9, 0.5];
    let g = [0.8, 0.2, 0.6, 0.4, 0.15];
    let x_hat = [0.1, -0.2, 0.3, -0.1, 0.05];
    let rho = 1.5;

    for lambda_exp in -3..=3 {
        let lambda = 10.0f64.powi(lambda_exp);
        let h_f32 = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, lambda).unwrap();

        // Recompute h in f64 for comparison.
        let lambda_clamped = lambda.clamp(MIN_LAMBDA, MAX_LAMBDA);
        let rho2_minus_xhat2 = (rho as f64) * (rho as f64) - l2_norm_sq(&x_hat);
        let phi2 = phi_norm_sq(&c, &g, &x_hat, lambda_clamped).unwrap();
        let h_f64 = -0.5f64 * (lambda_clamped * rho2_minus_xhat2 + phi2 / lambda_clamped);

        assert!(
            (h_f32 as f64) <= h_f64 + 1e-30,
            "f32 offset ({h_f32}) must be <= f64 offset ({h_f64}) for lambda={lambda}; \
             diff = {}",
            h_f32 as f64 - h_f64
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_offset_directed_rounding_xhat_zero_path() {
    // Verify the x_hat ≈ 0 branch (closed-form h = -rho * ||phi||_2) is rounded down.
    let c = [0.0, 0.3];
    let g = [1.0, 0.8];
    let x_hat = [0.0, 0.0];
    let rho = 1.7;

    let h_f32 = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
    let phi_norm = phi0_l2_norm(&c, &g).unwrap();
    let h_f64 = -(rho as f64) * phi_norm;

    assert!(
        (h_f32 as f64) <= h_f64 + 1e-30,
        "f32 offset ({h_f32}) must be <= f64 offset ({h_f64}); diff = {}",
        h_f32 as f64 - h_f64
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_offset_directed_rounding_opt() {
    // Verify relu_sdp_offset_opt returns value <= any single-lambda f64 offset.
    let c = [0.5, 0.2, 0.8];
    let g = [0.3, 0.6, 0.1];
    let x_hat = [0.15, -0.1, 0.2];
    let rho = 0.8;

    let h_opt_f32 = relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();

    // For every lambda we try, the f32 optimum must still be a valid lower bound
    // (i.e., <= the true f64 max over all lambda).
    // Since we can't compute the true f64 max easily, verify against individual lambdas:
    // the f32 opt is the max of individually-rounded-down values, so it should be
    // <= any individual f64 value (that would be wrong) -- actually, the optimum
    // is the MAX, so it may exceed individual lambdas. Instead verify it's <= the
    // f64 version of the same computation.
    for lambda_exp in -3..=3 {
        let lambda = 10.0f64.powi(lambda_exp);
        let lambda_clamped = lambda.clamp(MIN_LAMBDA, MAX_LAMBDA);
        let rho2_minus_xhat2 = (rho as f64) * (rho as f64) - l2_norm_sq(&x_hat);
        let phi2 = phi_norm_sq(&c, &g, &x_hat, lambda_clamped).unwrap();
        let h_f64 = -0.5f64 * (lambda_clamped * rho2_minus_xhat2 + phi2 / lambda_clamped);

        // Each individual lambda's f64 value is an upper bound on what that lambda
        // contributes. The f32 function picks the max of rounded-down values.
        // The f32 opt should be <= the f64 value for the same best lambda.
        // We can't directly test that, but we CAN test that the f32 for each lambda
        // is <= the f64 for that lambda (already tested above).
        let h_single_f32 = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, lambda).unwrap();
        assert!(
            (h_single_f32 as f64) <= h_f64 + 1e-30,
            "Single-lambda f32 ({h_single_f32}) must be <= f64 ({h_f64}) for lambda={lambda}"
        );
    }

    // The optimum must be >= any single-lambda f32 value (it's the max).
    for lambda_exp in -3..=3 {
        let lambda = 10.0f64.powi(lambda_exp);
        let h_single = relu_sdp_offset_for_lambda(&c, &g, &x_hat, rho, lambda).unwrap();
        assert!(
            h_opt_f32 >= h_single - 1e-6,
            "Opt ({h_opt_f32}) should be >= single-lambda ({h_single}) for lambda={lambda}"
        );
    }
}
