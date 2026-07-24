// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for `bounds` module, split by API family.
//!
//! - `safe_arithmetic`: safe_add/mul/array_add edge cases (NaN, Inf, broadcasting)
//! - `batched_matvec`: batched matrix-vector multiplication
//! - `alpha_state`: AlphaState construction, update, unstable counting
//! - `graph_alpha_state`: GraphAlphaState construction, update, node tracking
//! - `linear_bounds`: LinearBounds concretize, BatchedLinearBounds identity/compose

use super::*;

use crate::bounds::{
    batched_matvec, safe_add_for_bounds, safe_add_for_bounds_with_polarity,
    safe_add_lower_for_bounds, safe_add_upper_for_bounds, safe_array_add, safe_array_add_checked,
    safe_mul_for_bounds, AlphaState, BatchedLinearBounds, GraphAlphaState, LinearBounds,
};

use crate::bounds::interval_mul_for_bounds;

// Helper for tests that need to bypass debug checks (inf/NaN coverage).
fn unchecked_bounds(lower: ndarray::ArrayD<f32>, upper: ndarray::ArrayD<f32>) -> BoundedTensor {
    BoundedTensor::new_unchecked(lower, upper).expect("bounds shape mismatch")
}

mod alpha_state;
mod batched_matvec;
mod graph_alpha_state;
mod linear_bounds;
mod safe_arithmetic;
