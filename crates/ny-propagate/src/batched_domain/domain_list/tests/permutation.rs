// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the `apply_permutation` in-place reordering utility.

use super::super::apply_permutation;

#[test]
fn test_apply_permutation_identity() {
    let mut data = vec![10, 20, 30, 40];
    let mut perm = vec![0, 1, 2, 3];
    apply_permutation(&mut data, &mut perm).unwrap();
    assert_eq!(data, vec![10, 20, 30, 40]);
}

#[test]
fn test_apply_permutation_reverse() {
    let mut data = vec![10, 20, 30, 40];
    let mut perm = vec![3, 2, 1, 0];
    apply_permutation(&mut data, &mut perm).unwrap();
    assert_eq!(data, vec![40, 30, 20, 10]);
}

#[test]
fn test_apply_permutation_single_cycle() {
    let mut data = vec![10, 20, 30, 40];
    // Rotate: position 0 gets data[1], pos 1 gets data[2], etc.
    let mut perm = vec![1, 2, 3, 0];
    apply_permutation(&mut data, &mut perm).unwrap();
    assert_eq!(data, vec![20, 30, 40, 10]);
}

#[test]
fn test_apply_permutation_two_swaps() {
    let mut data = vec![10, 20, 30, 40];
    // Swap pairs: (0,1) and (2,3)
    let mut perm = vec![1, 0, 3, 2];
    apply_permutation(&mut data, &mut perm).unwrap();
    assert_eq!(data, vec![20, 10, 40, 30]);
}

#[test]
fn test_apply_permutation_single_element() {
    let mut data = vec![42];
    let mut perm = vec![0];
    apply_permutation(&mut data, &mut perm).unwrap();
    assert_eq!(data, vec![42]);
}

/// Regression test for #2998 Slice A: wrong-length permutation returns
/// `NyError::InternalError` instead of panicking.
#[test]
fn test_apply_permutation_wrong_length_returns_error_2998() {
    let mut data = vec![10, 20, 30];
    let mut perm = vec![0, 1]; // length 2 != data length 3
    let result = apply_permutation(&mut data, &mut perm);
    assert!(result.is_err());
    let err = result.unwrap_err();
    match &err {
        ny_core::NyError::InternalError(msg) => {
            assert!(
                msg.contains("permutation length"),
                "expected permutation length mismatch message, got: {msg}"
            );
        }
        other => panic!("expected InternalError, got: {other:?}"),
    }
}
