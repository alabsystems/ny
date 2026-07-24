// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::arr1;
use ny_tensor::BoundedTensor;
use std::sync::Arc;

#[test]
fn test_set_node_bounds_override_rejects_nonfinite_bounds_4143() {
    let mut metadata = DomainMetadata::root(-1.0, 1.0).expect("finite root bounds");
    let invalid_override = Arc::new(HashMap::from([(
        "relu".to_string(),
        BoundedTensor::new_allow_infinite(
            arr1(&[0.0_f32]).into_dyn(),
            arr1(&[f32::INFINITY]).into_dyn(),
        )
        .expect("infinite bounds allowed for construction"),
    )]));

    let err = metadata
        .set_node_bounds_override(Some(invalid_override))
        .expect_err("non-finite deferred node bounds must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
    assert!(
        metadata.node_bounds_override().is_none(),
        "failed setter must not mutate metadata"
    );
}
