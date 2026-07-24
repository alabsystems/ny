// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::BatchedScalarBounds;

#[test]
fn test_batched_scalar_bounds_field_roundtrip_4404() {
    let bounds = BatchedScalarBounds {
        lower_bounds: vec![-1.0_f32, 0.25],
        upper_bounds: vec![1.5_f32, 2.0],
        linear_bounds: vec![None, None],
    };

    assert_eq!(bounds.lower_bounds, vec![-1.0_f32, 0.25]);
    assert_eq!(bounds.upper_bounds, vec![1.5_f32, 2.0]);
    assert!(
        bounds.linear_bounds.iter().all(Option::is_none),
        "test carrier should preserve per-domain optional linear bounds"
    );
}
