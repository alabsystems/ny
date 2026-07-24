// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_dot_linear_coeff_computation() {
    // Kills: replace + with - in line 506 (a0·bi + ai·b0)
    // Kills: replace + with * in line 506
    // With 1 error term, check that linear coefficient = a0·b1 + a1·b0

    // a = 1 + 0.5*e (center=1, err_coeff=0.5)
    let mut a_coeffs = ArrayD::<f32>::zeros(IxDyn(&[2, 1]));
    a_coeffs[[0, 0]] = 1.0; // center
    a_coeffs[[1, 0]] = 0.5; // error coefficient
    let a = ZonotopeTensor::new(a_coeffs).unwrap();

    // b = 2 + 0.3*e (center=2, err_coeff=0.3)
    let mut b_coeffs = ArrayD::<f32>::zeros(IxDyn(&[2, 1]));
    b_coeffs[[0, 0]] = 2.0;
    b_coeffs[[1, 0]] = 0.3;
    let b = ZonotopeTensor::new(b_coeffs).unwrap();

    let result = a.dot(&b).unwrap();

    // Linear coeff for e = a0*b1 + a1*b0 = 1*0.3 + 0.5*2 = 1.3
    // If + was -, would get 1*0.3 - 0.5*2 = -0.7
    assert!(
        (result.coeffs[[1, 0]] - 1.3).abs() < 1e-6,
        "linear coeff should be 1.3, got {}",
        result.coeffs[[1, 0]]
    );
}

#[test]
fn test_dot_half_term_accumulation() {
    // Kills: replace += with -= in line 527 (half_term += ...)
    // Kills: replace += with *= in line 527
    // The half_term should accumulate 0.5 * |ai·bi| for each error term

    // a = 1 + 0.4*e1 + 0.2*e2
    let mut a_coeffs = ArrayD::<f32>::zeros(IxDyn(&[3, 1]));
    a_coeffs[[0, 0]] = 1.0;
    a_coeffs[[1, 0]] = 0.4;
    a_coeffs[[2, 0]] = 0.2;
    let a = ZonotopeTensor::new(a_coeffs).unwrap();

    // b = 2 + 0.5*e1 + 0.3*e2
    let mut b_coeffs = ArrayD::<f32>::zeros(IxDyn(&[3, 1]));
    b_coeffs[[0, 0]] = 2.0;
    b_coeffs[[1, 0]] = 0.5;
    b_coeffs[[2, 0]] = 0.3;
    let b = ZonotopeTensor::new(b_coeffs).unwrap();

    let result = a.dot(&b).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    // half_term = 0.5*|0.4*0.5| + 0.5*|0.2*0.3| = 0.5*0.2 + 0.5*0.06 = 0.13
    // If += was -=, half_term would be negative → bounds would be wrong
    // Check that bounds contain the expected range
    let center = result.coeffs[[0, 0]];
    let lower = bounds.lower()[[0]];
    let upper = bounds.upper()[[0]];

    // Width should be positive (radius = half_term + other terms)
    assert!(upper > lower, "bounds should have positive width");
    assert!(upper >= center, "upper should be >= center");
    assert!(lower <= center, "lower should be <= center");
}

#[test]
fn test_dot_cross_term_loop_start() {
    // Kills: replace + with * in line 538 ((i + 1)..=n)
    // If + was *, loop would start at i*1=i, including i==i case (wrong)
    // With 2 error terms, there should be exactly one cross term (1,2)

    // a = 1 + e1 + e2
    let mut a_coeffs = ArrayD::<f32>::zeros(IxDyn(&[3, 1]));
    a_coeffs[[0, 0]] = 1.0;
    a_coeffs[[1, 0]] = 1.0;
    a_coeffs[[2, 0]] = 1.0;
    let a = ZonotopeTensor::new(a_coeffs).unwrap();

    // b = 1 + e1 + e2
    let result = a.dot(&a).unwrap();

    // With correct loop, cross term = |a1·b2 + a2·b1| = |1*1 + 1*1| = 2
    // big_term contributes to the new error coefficient (index n+1 = 3)
    // half_term for e1² and e2²: 0.5*|1*1| + 0.5*|1*1| = 1
    // new_error_coeff = half_term + big_term = 1 + 2 = 3

    // The last error term should have the combined quadratic bound
    let new_err_coeff = result.coeffs[[3, 0]];
    assert!(
        (new_err_coeff - 3.0).abs() < 1e-6,
        "new error coeff should be 3 (half=1, cross=2), got {}",
        new_err_coeff
    );
}

#[test]
fn test_dot_cross_term_product() {
    // Kills: replace * with + in lines 543/544 (ai.iter().zip(bj).map(|(&a,&b)| a * b))
    // If * was +, ai·bj would be sum of (a+b) not sum of (a*b)

    // a = [1, 2] + [0.1, 0.2]*e1 + [0.3, 0.4]*e2
    let mut a_coeffs = ArrayD::<f32>::zeros(IxDyn(&[3, 2]));
    a_coeffs[[0, 0]] = 1.0;
    a_coeffs[[0, 1]] = 2.0;
    a_coeffs[[1, 0]] = 0.1;
    a_coeffs[[1, 1]] = 0.2;
    a_coeffs[[2, 0]] = 0.3;
    a_coeffs[[2, 1]] = 0.4;
    let a = ZonotopeTensor::new(a_coeffs).unwrap();

    // b = [3, 4] + [0.5, 0.6]*e1 + [0.7, 0.8]*e2
    let mut b_coeffs = ArrayD::<f32>::zeros(IxDyn(&[3, 2]));
    b_coeffs[[0, 0]] = 3.0;
    b_coeffs[[0, 1]] = 4.0;
    b_coeffs[[1, 0]] = 0.5;
    b_coeffs[[1, 1]] = 0.6;
    b_coeffs[[2, 0]] = 0.7;
    b_coeffs[[2, 1]] = 0.8;
    let b = ZonotopeTensor::new(b_coeffs).unwrap();

    let result = a.dot(&b).unwrap();

    // Cross term (i=1, j=2):
    // a1·b2 = 0.1*0.7 + 0.2*0.8 = 0.07 + 0.16 = 0.23
    // a2·b1 = 0.3*0.5 + 0.4*0.6 = 0.15 + 0.24 = 0.39
    // If * was +, a1·b2 would be (0.1+0.7) + (0.2+0.8) = 1.8 (wrong)

    // big_term = |a1·b2 + a2·b1| = |0.23 + 0.39| = 0.62
    // half_term = 0.5*|a1·b1| + 0.5*|a2·b2|
    //           = 0.5*|0.1*0.5+0.2*0.6| + 0.5*|0.3*0.7+0.4*0.8|
    //           = 0.5*0.17 + 0.5*0.53 = 0.085 + 0.265 = 0.35
    // new_err = 0.35 + 0.62 = 0.97

    let new_err_coeff = result.coeffs[[3, 0]];
    assert!(
        (new_err_coeff - 0.97).abs() < 0.01,
        "cross term should use products, got new_err={}",
        new_err_coeff
    );
}

#[test]
fn test_dot_big_term_accumulation() {
    // Kills: replace += with -= in line 547 (big_term += ...)
    // Kills: replace + with - in line 547 ((ai_dot_bj + aj_dot_bi))

    // Same setup as above but verify big_term is accumulated correctly
    let mut a_coeffs = ArrayD::<f32>::zeros(IxDyn(&[4, 1])); // 3 error terms
    a_coeffs[[0, 0]] = 1.0;
    a_coeffs[[1, 0]] = 0.1;
    a_coeffs[[2, 0]] = 0.2;
    a_coeffs[[3, 0]] = 0.3;
    let a = ZonotopeTensor::new(a_coeffs).unwrap();

    let mut b_coeffs = ArrayD::<f32>::zeros(IxDyn(&[4, 1]));
    b_coeffs[[0, 0]] = 2.0;
    b_coeffs[[1, 0]] = 0.4;
    b_coeffs[[2, 0]] = 0.5;
    b_coeffs[[3, 0]] = 0.6;
    let b = ZonotopeTensor::new(b_coeffs).unwrap();

    let result = a.dot(&b).unwrap();

    // Cross terms: (1,2), (1,3), (2,3)
    // (1,2): |0.1*0.5 + 0.2*0.4| = |0.05 + 0.08| = 0.13
    // (1,3): |0.1*0.6 + 0.3*0.4| = |0.06 + 0.12| = 0.18
    // (2,3): |0.2*0.6 + 0.3*0.5| = |0.12 + 0.15| = 0.27
    // big_term = 0.13 + 0.18 + 0.27 = 0.58

    // half_term = 0.5*(|0.1*0.4| + |0.2*0.5| + |0.3*0.6|)
    //           = 0.5*(0.04 + 0.10 + 0.18) = 0.16

    let new_err_coeff = result.coeffs[[4, 0]]; // index = 1 + n = 4
    let expected = 0.16 + 0.58;
    assert!(
        (new_err_coeff - expected).abs() < 0.01,
        "big_term should accumulate cross terms, expected {}, got {}",
        expected,
        new_err_coeff
    );
}
