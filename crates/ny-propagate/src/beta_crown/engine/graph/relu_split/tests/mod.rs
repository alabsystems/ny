// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for ReLU-splitting branch-and-bound.

mod aggregate;
mod bab_loop;
mod branch_selection;
mod metrics_emission;

use ny_tensor::BoundedTensor;

use crate::beta_crown::domain::GraphBabDomain;

pub(super) fn test_input() -> BoundedTensor {
    BoundedTensor::new(
        ndarray::arr1(&[-1.0f32, -1.0]).into_dyn(),
        ndarray::arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .expect("invariant: symmetric bounds are valid")
}

pub(super) fn test_domain(lower: f32, upper: f32) -> GraphBabDomain {
    let input = test_input();
    GraphBabDomain::root(
        std::collections::HashMap::new(),
        lower,
        upper,
        &input,
        false,
    )
    .expect("invariant: finite test bounds are valid")
}
