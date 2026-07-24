// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== NonZero tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_nonzero_ibp_all_nonzero() {
    // All elements are definitely non-zero (positive interval)
    let lower = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let nonzero = NonZeroLayer;
    let output = nonzero.propagate_ibp(&input).unwrap();

    // Output shape: [rank(input), max_nonzero] = [2, 6]
    assert_eq!(output.shape(), &[2, 6]);

    // All lower bounds should be 0 (min index)
    for val in output.lower().iter() {
        assert_eq!(*val, 0.0);
    }

    // Upper bounds for dim 0 should be 1 (shape[0]-1 = 2-1)
    // Upper bounds for dim 1 should be 2 (shape[1]-1 = 3-1)
    for col in 0..6 {
        assert_eq!(output.upper()[[0, col]], 1.0);
        assert_eq!(output.upper()[[1, col]], 2.0);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_nonzero_ibp_all_zeros() {
    // All elements are exactly zero
    let lower = ArrayD::from_elem(IxDyn(&[3, 4]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3, 4]), 0.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let nonzero = NonZeroLayer;
    let output = nonzero.propagate_ibp(&input).unwrap();

    // Output shape: [rank(input), 0] = [2, 0] (no nonzero elements)
    assert_eq!(output.shape(), &[2, 0]);
}

#[ntest::timeout(10000)]
#[test]
fn test_nonzero_ibp_mixed() {
    // Some elements could be nonzero, some definitely zero
    let mut lower = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);
    let mut upper = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);

    // Element 0: [0, 0] - definitely zero
    // Element 1: [1, 2] - definitely non-zero
    lower[[1]] = 1.0;
    upper[[1]] = 2.0;
    // Element 2: [-1, 1] - could be zero or nonzero
    lower[[2]] = -1.0;
    upper[[2]] = 1.0;
    // Element 3: [0, 0] - definitely zero

    let input = BoundedTensor::new(lower, upper).unwrap();

    let nonzero = NonZeroLayer;
    let output = nonzero.propagate_ibp(&input).unwrap();

    // Elements 1 and 2 could be non-zero, so max_nonzero = 2
    // Output shape: [1, 2] (1D input)
    assert_eq!(output.shape(), &[1, 2]);

    // Lower bounds: all 0
    for val in output.lower().iter() {
        assert_eq!(*val, 0.0);
    }

    // Upper bounds: 3 (input shape - 1 = 4 - 1 = 3)
    for val in output.upper().iter() {
        assert_eq!(*val, 3.0);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_nonzero_linear_not_supported() {
    // Linear bounds should fail (data-dependent output shape)
    let bounds = LinearBounds::identity(4);
    let nonzero = NonZeroLayer;
    assert!(nonzero.propagate_linear(&bounds).is_err());
}
