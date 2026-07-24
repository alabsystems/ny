// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for bounds module, split by component.

mod alpha_state;
mod alpha_state_api;
mod batched_linear;
mod batched_linear_validation;
mod graph_alpha_state;
mod helpers;
mod interval_arithmetic;
mod linear;
mod linear_validation;
mod linear_validation_fastpath;
mod panic_cliff;

use crate::BoundedTensor;
use ndarray::ArrayD;

/// Create bounds without validation (for testing edge cases like infinities).
pub(crate) fn unchecked_bounds(lower: ArrayD<f32>, upper: ArrayD<f32>) -> BoundedTensor {
    BoundedTensor::new_unchecked(lower, upper).expect("bounds shape mismatch")
}

/// Create validated bounds (for normal test cases).
pub(crate) fn checked_bounds(lower: ArrayD<f32>, upper: ArrayD<f32>) -> BoundedTensor {
    BoundedTensor::new(lower, upper).expect("bounds should be valid")
}
