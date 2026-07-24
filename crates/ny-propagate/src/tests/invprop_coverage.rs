// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for InvpropState methods that lacked test coverage:
//! - apply_infeasible_mask multi-batch path
//! - apply_infeasible_mask no-op when no elements are infeasible
//! - apply_infeasible_mask batch dimension mismatch
//! - is_infeasible out-of-bounds returns false

use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::invprop::{InvpropState, OutputConstraints};

#[ntest::timeout(10000)]
#[test]
fn test_apply_infeasible_mask_multi_batch() {
    let constraints = OutputConstraints::new(arr2(&[[1.0]]), arr1(&[0.0]), true).unwrap();
    let mut state = InvpropState::new(constraints, 3);
    state
        .mark_infeasible(1)
        .expect("invariant: batch_idx=1 within batch_size=3"); // Only middle batch element infeasible

    // Create bounds with batch dimension = 3
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut bounds = BoundedTensor::new(lower, upper).unwrap();

    state.apply_infeasible_mask(&mut bounds);

    // Batch 0 and 2 should be unchanged
    assert_eq!(bounds.lower()[[0, 0]], 0.0);
    assert_eq!(bounds.lower()[[0, 1]], 1.0);
    assert_eq!(bounds.lower()[[2, 0]], 4.0);
    assert_eq!(bounds.lower()[[2, 1]], 5.0);

    // Batch 1 should be infeasible: lower=+inf, upper=-inf
    assert!(bounds.lower()[[1, 0]].is_infinite() && bounds.lower()[[1, 0]].is_sign_positive());
    assert!(bounds.upper()[[1, 0]].is_infinite() && bounds.upper()[[1, 0]].is_sign_negative());
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_infeasible_mask_no_infeasible_is_noop() {
    let constraints = OutputConstraints::new(arr2(&[[1.0]]), arr1(&[0.0]), true).unwrap();
    let state = InvpropState::new(constraints, 2);
    // No elements marked infeasible

    let lower = arr1(&[0.0, 1.0]).into_dyn();
    let upper = arr1(&[2.0, 3.0]).into_dyn();
    let mut bounds = BoundedTensor::new(lower, upper).unwrap();

    state.apply_infeasible_mask(&mut bounds);

    // Should be unchanged
    assert_eq!(bounds.lower()[[0]], 0.0);
    assert_eq!(bounds.lower()[[1]], 1.0);
    assert_eq!(bounds.upper()[[0]], 2.0);
    assert_eq!(bounds.upper()[[1]], 3.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_infeasible_mask_batch_mismatch_marks_all_infeasible() {
    let constraints = OutputConstraints::new(arr2(&[[1.0]]), arr1(&[0.0]), true).unwrap();
    let mut state = InvpropState::new(constraints, 2);
    state
        .mark_infeasible(0)
        .expect("invariant: batch_idx=0 within batch_size=2");

    // Bounds have shape [3, 2] — batch dim 3 != mask len 2
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut bounds = BoundedTensor::new(lower, upper).unwrap();

    state.apply_infeasible_mask(&mut bounds);

    // When batch dim doesn't match mask len, ALL elements become infeasible
    assert!(bounds
        .lower()
        .iter()
        .all(|v| v.is_infinite() && v.is_sign_positive()));
    assert!(bounds
        .upper()
        .iter()
        .all(|v| v.is_infinite() && v.is_sign_negative()));
}

#[ntest::timeout(10000)]
#[test]
fn test_is_infeasible_out_of_bounds_returns_false() {
    let constraints = OutputConstraints::new(arr2(&[[1.0]]), arr1(&[0.0]), true).unwrap();
    let state = InvpropState::new(constraints, 2);

    // Out-of-bounds index should return false (not panic)
    assert!(!state.is_infeasible(99));
}
