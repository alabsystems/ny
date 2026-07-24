// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{array, Array2};

#[ntest::timeout(10000)]
#[test]
fn test_spectral_norm_handles_dominant_direction_missing_from_seed() {
    // Regression for #2389:
    // Old fixed-seed power iteration started with v[0] = 0. For a diagonal matrix where
    // the dominant singular direction is axis 0, the estimate could never reach sigma_max.
    //
    // diag([10, 9, 1]) has exact spectral norm 10.
    let weight = Array2::from_diag(&array![10.0_f32, 9.0, 1.0]);
    let layer = LinearLayer::new(weight, None).expect("valid");
    let spectral = layer.spectral_norm();

    assert!(
        spectral >= 10.0,
        "spectral norm upper bound must not underestimate true sigma_max=10, got {spectral}"
    );
    assert!(
        spectral < 10.01,
        "spectral norm upper bound should stay tight for diagonal matrices, got {spectral}"
    );
}
