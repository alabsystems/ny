// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::metadata::merge_tensor_shape;

#[test]
fn test_merge_tensor_shape_fills_empty_existing() {
    let mut existing = Vec::new();
    merge_tensor_shape("t", &mut existing, &[1, 2, 3]);
    assert_eq!(existing, vec![1, 2, 3]);
}

#[test]
fn test_merge_tensor_shape_ignores_length_mismatch() {
    let mut existing = vec![1, -1];
    merge_tensor_shape("t", &mut existing, &[1, 2, 3]);
    assert_eq!(existing, vec![1, -1]);
}

#[test]
fn test_merge_tensor_shape_preserves_known_dims() {
    let mut existing = vec![1, -1, 0];
    merge_tensor_shape("t", &mut existing, &[2, 4, 5]);
    assert_eq!(existing, vec![1, 4, 5]);
}

#[test]
fn test_merge_tensor_shape_keeps_existing_positive_dims() {
    let mut existing = vec![2, 3];
    merge_tensor_shape("t", &mut existing, &[5, 6]);
    assert_eq!(existing, vec![2, 3]);
}

#[test]
fn test_merge_tensor_shape_updates_all_unknown_dims() {
    let mut existing = vec![-1, -1];
    merge_tensor_shape("t", &mut existing, &[4, 5]);
    assert_eq!(existing, vec![4, 5]);
}

#[test]
fn test_merge_tensor_shape_skips_non_positive_inferred_dims() {
    let mut existing = vec![-1, 2, -1];
    merge_tensor_shape("t", &mut existing, &[-1, 0, -1]);
    assert_eq!(existing, vec![-1, 2, -1]);
}

#[test]
fn test_merge_tensor_shape_ignores_length_mismatch_with_unknowns() {
    let mut existing = vec![-1, -1];
    merge_tensor_shape("t", &mut existing, &[3]);
    assert_eq!(existing, vec![-1, -1]);
}

#[test]
fn test_merge_tensor_shape_ignores_empty_inferred() {
    let mut existing = vec![1, -1, 3];
    merge_tensor_shape("t", &mut existing, &[]);
    assert_eq!(existing, vec![1, -1, 3]);
}

#[test]
fn test_merge_tensor_shape_allows_empty_scalar_shape() {
    let mut existing = Vec::new();
    merge_tensor_shape("t", &mut existing, &[]);
    assert!(existing.is_empty());
}

#[test]
fn test_merge_tensor_shape_accepts_unknown_dims_for_empty_existing() {
    let mut existing = Vec::new();
    merge_tensor_shape("t", &mut existing, &[-1, 0]);
    assert_eq!(existing, vec![-1, 0]);
}

#[test]
fn test_merge_tensor_shape_noops_on_all_unknown_dims() {
    let mut existing = vec![-1, -1, -1];
    merge_tensor_shape("t", &mut existing, &[-1, 0, -1]);
    assert_eq!(existing, vec![-1, -1, -1]);
}

#[test]
fn test_merge_tensor_shape_logs_conflict_but_keeps_existing() {
    let mut existing = vec![1, 30];
    merge_tensor_shape("hidden_output", &mut existing, &[1, 98]);
    assert_eq!(existing, vec![1, 30]);
}
