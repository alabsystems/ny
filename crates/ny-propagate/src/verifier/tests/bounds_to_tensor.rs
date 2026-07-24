// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// Licensed under the Apache License, Version 2.0

use super::super::*;
use ny_core::{Bound, NyError};

#[ntest::timeout(5000)]
#[test]
fn test_spec_to_tensor_simple_1d() {
    let bounds = vec![Bound::new(-1.0, 1.0), Bound::new(0.0, 2.0)];

    let result = Verifier::bounds_to_tensor(&bounds, None).unwrap();

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.lower()[[0]], -1.0);
    assert_eq!(result.lower()[[1]], 0.0);
    assert_eq!(result.upper()[[0]], 1.0);
    assert_eq!(result.upper()[[1]], 2.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_spec_to_tensor_with_shape_2d() {
    let bounds = vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-2.0, 2.0),
        Bound::new(-3.0, 3.0),
        Bound::new(-4.0, 4.0),
    ];

    let result = Verifier::bounds_to_tensor(&bounds, Some(&[2, 2])).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
}

#[ntest::timeout(5000)]
#[test]
fn test_spec_to_tensor_with_shape_3d() {
    let bounds: Vec<Bound> = (0..24)
        .map(|i| Bound::new(i as f32, i as f32 + 1.0))
        .collect();

    let result = Verifier::bounds_to_tensor(&bounds, Some(&[2, 3, 4])).unwrap();

    assert_eq!(result.shape(), &[2, 3, 4]);
}

#[ntest::timeout(5000)]
#[test]
fn test_spec_to_tensor_shape_mismatch_error() {
    let bounds = vec![Bound::new(-1.0, 1.0), Bound::new(0.0, 2.0)];

    // 3x3=9 elements but only 2 bounds provided
    let result = Verifier::bounds_to_tensor(&bounds, Some(&[3, 3]));

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, NyError::InvalidSpec(_)));
    let msg = format!("{}", err);
    assert!(msg.contains("9 elements"));
    assert!(msg.contains("2"));
}

#[ntest::timeout(5000)]
#[test]
fn test_spec_to_tensor_shape_overflow_error_2602() {
    let bounds = vec![Bound::new(-1.0, 1.0)];

    let result = Verifier::bounds_to_tensor(&bounds, Some(&[usize::MAX, 2]));

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, NyError::InvalidSpec(_)));
    let msg = format!("{err}");
    assert!(msg.contains("overflows usize"), "unexpected error: {msg}");
}

#[ntest::timeout(5000)]
#[test]
fn test_spec_to_tensor_empty_bounds() {
    let bounds: Vec<Bound> = vec![];

    let result = Verifier::bounds_to_tensor(&bounds, None).unwrap();

    assert_eq!(result.shape(), &[0]);
}

#[ntest::timeout(5000)]
#[test]
fn test_spec_to_tensor_single_element() {
    let bounds = vec![Bound::new(0.5, 1.5)];

    let result = Verifier::bounds_to_tensor(&bounds, None).unwrap();

    assert_eq!(result.shape(), &[1]);
    assert_eq!(result.lower()[[0]], 0.5);
    assert_eq!(result.upper()[[0]], 1.5);
}

#[ntest::timeout(5000)]
#[test]
fn test_spec_to_tensor_large_values() {
    // Test with large but finite values instead of infinities
    // (BoundedTensor::new rejects infinities)
    let large = 1e30f32;
    let bounds = vec![
        Bound::new(-large, large),
        Bound::new(0.0, large),
        Bound::new(-large, 0.0),
    ];

    let result = Verifier::bounds_to_tensor(&bounds, None).unwrap();

    assert_eq!(result.lower()[[0]], -large);
    assert_eq!(result.upper()[[0]], large);
    assert_eq!(result.lower()[[1]], 0.0);
    assert_eq!(result.upper()[[1]], large);
    assert_eq!(result.lower()[[2]], -large);
    assert_eq!(result.upper()[[2]], 0.0);
}
