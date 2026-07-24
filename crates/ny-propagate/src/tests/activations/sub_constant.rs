// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== SubConstant tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_sub_constant_ibp_normal() {
    // Test y = x - constant
    let lower = ArrayD::from_elem(IxDyn(&[3]), 5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sub = SubConstantLayer::scalar(3.0);
    let output = sub.propagate_ibp(&input).unwrap();

    // [5, 10] - 3 = [2, 7]
    for i in 0..3 {
        assert!(
            (output.lower()[[i]] - 2.0).abs() < 1e-6,
            "5-3 should be 2, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 7.0).abs() < 1e-6,
            "10-3 should be 7, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sub_constant_ibp_reverse() {
    // Test y = constant - x (bounds swap)
    let lower = ArrayD::from_elem(IxDyn(&[2]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 8.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let constant = ArrayD::from_elem(IxDyn(&[2]), 10.0f32);
    let sub = SubConstantLayer::new_reverse(constant);
    let output = sub.propagate_ibp(&input).unwrap();

    // 10 - [2, 8] = [2, 8]
    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - 2.0).abs() < 1e-6,
            "10-8 should be 2, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 8.0).abs() < 1e-6,
            "10-2 should be 8, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_div_linear_propagation() {
    // Test that DivConstant linear propagation is consistent
    let lower = ArrayD::from_elem(IxDyn(&[4]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 8.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let div = DivConstantLayer::scalar(2.0);
    let ibp_result = div.propagate_ibp(&input).unwrap();

    // Create identity linear bounds and propagate
    let linear_bounds = LinearBounds::identity(4);
    let linear_result = div.propagate_linear(&linear_bounds).unwrap().into_owned();
    let concretized = linear_result.concretize(&input);

    // Results should match
    for i in 0..4 {
        assert!(
            (ibp_result.lower()[[i]] - concretized.lower()[[i]]).abs() < 1e-5,
            "Linear lower doesn't match IBP: {} vs {}",
            ibp_result.lower()[[i]],
            concretized.lower()[[i]]
        );
        assert!(
            (ibp_result.upper()[[i]] - concretized.upper()[[i]]).abs() < 1e-5,
            "Linear upper doesn't match IBP: {} vs {}",
            ibp_result.upper()[[i]],
            concretized.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sub_linear_propagation() {
    // Test that SubConstant linear propagation is consistent
    let lower = ArrayD::from_elem(IxDyn(&[4]), 3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 9.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sub = SubConstantLayer::scalar(2.0);
    let ibp_result = sub.propagate_ibp(&input).unwrap();

    // Create identity linear bounds and propagate
    let linear_bounds = LinearBounds::identity(4);
    let linear_result = sub.propagate_linear(&linear_bounds).unwrap().into_owned();
    let concretized = linear_result.concretize(&input);

    // Results should match
    for i in 0..4 {
        assert!(
            (ibp_result.lower()[[i]] - concretized.lower()[[i]]).abs() < 1e-5,
            "Linear lower doesn't match IBP: {} vs {}",
            ibp_result.lower()[[i]],
            concretized.lower()[[i]]
        );
        assert!(
            (ibp_result.upper()[[i]] - concretized.upper()[[i]]).abs() < 1e-5,
            "Linear upper doesn't match IBP: {} vs {}",
            ibp_result.upper()[[i]],
            concretized.upper()[[i]]
        );
    }
}
