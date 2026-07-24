// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for #2998 Slice A: batched-domain panic-to-Result conversions.
//!
//! Verifies that `extract_updates_with_layer_bounds`, `apply_permutation`, and
//! checked bound lookups return structured `NyError` instead of panicking.

use super::super::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::NyError;
use std::collections::HashMap;

/// Helper: build a 2-domain batch with one ReLU layer.
fn two_domain_batch() -> BatchedDomains {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);
    let layer_bounds: HashMap<String, (ArrayD<f32>, ArrayD<f32>)> = [(
        "relu0".to_string(),
        (
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        ),
    )]
    .into_iter()
    .collect();
    let il = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.0]).unwrap();
    let iu = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.0]).unwrap();
    builder.add_domain(&layer_bounds, il.clone(), iu.clone(), -1.0, 1.0, 0, vec![]);
    builder.add_domain(&layer_bounds, il, iu, -0.5, 0.5, 0, vec![]);
    builder.build().unwrap()
}

/// Mismatched lower-bounds length returns `NyError::InternalError`.
#[ntest::timeout(5000)]
#[test]
fn test_extract_updates_mismatched_bounds_returns_error_2998() {
    let batched = two_domain_batch();
    // Wrong length: 1 lower bound for batch_size 2
    let result = batched.extract_updates_with_layer_bounds(&[-0.5], &[0.8, 0.9], None, None);
    assert!(result.is_err());
    match result.unwrap_err() {
        NyError::InternalError(msg) => {
            assert!(
                msg.contains("bounds len"),
                "expected bounds len mismatch, got: {msg}"
            );
        }
        other => panic!("expected InternalError, got: {other:?}"),
    }

    // Wrong length: 3 upper bounds for batch_size 2
    let result =
        batched.extract_updates_with_layer_bounds(&[-0.5, -0.3], &[0.8, 0.9, 1.0], None, None);
    assert!(result.is_err());
    match result.unwrap_err() {
        NyError::InternalError(msg) => {
            assert!(
                msg.contains("bounds len"),
                "expected bounds len mismatch, got: {msg}"
            );
        }
        other => panic!("expected InternalError, got: {other:?}"),
    }
}
