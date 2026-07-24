// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `crown_elementwise_backward` (Dense variant) in
//! `layers/common/crown_dense.rs`.
//!
//! This file uses the **same relaxation function and inputs** as
//! `batched_backward.rs` and `patches_backward.rs` to enable three-way
//! cross-variant consistency verification: Dense, Batched, and Patches
//! backward paths must produce identical coefficient and bias values for
//! identical mathematical inputs.
//!
//! Part of #3463.

use crate::layers::activations::LinearRelaxation;
use crate::layers::common::crown_elementwise_backward;
use crate::LinearBounds;
use ndarray::{array, Array1, Array2};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

/// Synthetic relaxation shared with `batched_backward.rs` and `patches_backward.rs`.
///
/// lower_slope = |l| + 1, lower_intercept = |l| + 0.25
/// upper_slope = u + 0.5, upper_intercept = u + 1.25
fn test_relaxation(lower: f32, upper: f32) -> LinearRelaxation {
    LinearRelaxation::new(
        lower.abs() + 1.0,
        lower.abs() + 0.25,
        upper + 0.5,
        upper + 1.25,
    )
}

/// Mixed-sign coefficients — expected values match
/// `test_crown_elementwise_backward_batched_mixed_sign_coefficients` and
/// `test_crown_elementwise_backward_patches_dense_mixed_sign_coefficients`.
///
/// Verifies compose_lower/compose_upper coefficient selection:
///   positive coeff → lower/upper relaxation (preserves direction)
///   negative coeff → upper/lower relaxation (flips direction)
/// Plus directed rounding on both coefficients and f64→f32 bias conversion.
///
/// Hand-computed derivation (shared across all three variants):
///   Neuron 0: l=-1, u=2 → relax=(ls=2.0, li=1.25, us=2.5, ui=3.25)
///   Neuron 1: l=-2, u=4 → relax=(ls=3.0, li=2.25, us=4.5, ui=5.25)
///
///   Lower: la[0]=1.5 > 0  → lower relax n0: coeff=1.5×2.0=3.0
///          la[1]=-2.0 < 0 → upper relax n1: coeff=-2.0×4.5=-9.0
///          bias = 0.25 + 1.5×1.25 + (-2.0)×5.25 = -8.375
///   Upper: ua[0]=-0.75 < 0 → lower relax n0: coeff=-0.75×2.0=-1.5
///          ua[1]=3.0 > 0   → upper relax n1: coeff=3.0×4.5=13.5
///          bias = -0.5 + (-0.75)×1.25 + 3.0×5.25 = 14.3125
#[test]
fn test_crown_elementwise_backward_dense_mixed_sign_cross_variant() {
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 2), vec![1.5_f32, -2.0]).unwrap(),
        Array1::from_vec(vec![0.25_f32]),
        Array2::from_shape_vec((1, 2), vec![-0.75_f32, 3.0]).unwrap(),
        Array1::from_vec(vec![-0.5_f32]),
    )
    .unwrap();

    let pre_activation = BoundedTensor::new(
        array![-1.0_f32, -2.0].into_dyn(),
        array![2.0_f32, 4.0].into_dyn(),
    )
    .unwrap();

    let result = crown_elementwise_backward(&bounds, &pre_activation, test_relaxation).unwrap();

    // These 6 values are identical across Dense, Batched, and Patches variants.
    assert_eq!(result.lower_a[[0, 0]], next_down_f32(3.0));
    assert_eq!(result.lower_a[[0, 1]], next_down_f32(-9.0));
    assert_eq!(result.lower_b[0], next_down_f32(-8.375));

    assert_eq!(result.upper_a[[0, 0]], next_up_f32(-1.5));
    assert_eq!(result.upper_a[[0, 1]], next_up_f32(13.5));
    assert_eq!(result.upper_b[0], next_up_f32(14.3125));
}

/// Multi-output variant — 2 output neurons, 3 input neurons, mixed signs.
///
/// Tests the `for j in 0..out_dim` outer loop with distinct coefficient
/// patterns per output row. This catches row-indexing bugs that the
/// single-output cross-variant test cannot.
///
/// All bias values are dyadic rationals (exactly representable in f32/f64)
/// to avoid representability-induced ULP shifts in the f64→f32 bias path.
///
/// Hand-computed:
///   Neuron 0: l=-1, u=2 → relax=(ls=2.0, li=1.25, us=2.5, ui=3.25)
///   Neuron 1: l= 0, u=3 → relax=(ls=1.0, li=0.25, us=3.5, ui=4.25)
///   Neuron 2: l=-3, u=1 → relax=(ls=4.0, li=3.25, us=1.5, ui=2.25)
///
///   Row 0 lower: la=[2.0, -1.0, 0.5]
///     n0: 2.0>0 → ls=2.0 → 4.0, intercept=2.0×1.25=2.5
///     n1: -1.0<0 → us=3.5 → -3.5, intercept=-1.0×4.25=-4.25
///     n2: 0.5>0 → ls=4.0 → 2.0, intercept=0.5×3.25=1.625
///     bias = 0.25 + 2.5 - 4.25 + 1.625 = 0.125
///   Row 0 upper: ua=[-0.5, 1.5, -2.0]
///     n0: -0.5<0 → ls=2.0 → -1.0, intercept=-0.5×1.25=-0.625
///     n1: 1.5>0 → us=3.5 → 5.25, intercept=1.5×4.25=6.375
///     n2: -2.0<0 → ls=4.0 → -8.0, intercept=-2.0×3.25=-6.5
///     bias = -0.5 + (-0.625) + 6.375 + (-6.5) = -1.25
///   Row 1 lower: la=[-3.0, 0.0, 1.0]
///     n0: -3.0<0 → us=2.5 → -7.5, intercept=-3.0×3.25=-9.75
///     n1: 0.0 → ZERO → 0.0, intercept=0.0
///     n2: 1.0>0 → ls=4.0 → 4.0, intercept=1.0×3.25=3.25
///     bias = 0.375 + (-9.75) + 0.0 + 3.25 = -6.125
///   Row 1 upper: ua=[0.0, -0.5, 3.0]
///     n0: 0.0 → ZERO → 0.0, intercept=0.0
///     n1: -0.5<0 → ls=1.0 → -0.5, intercept=-0.5×0.25=-0.125
///     n2: 3.0>0 → us=1.5 → 4.5, intercept=3.0×2.25=6.75
///     bias = 0.125 + 0.0 + (-0.125) + 6.75 = 6.75
#[test]
fn test_crown_elementwise_backward_dense_multi_output() {
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((2, 3), vec![2.0_f32, -1.0, 0.5, -3.0, 0.0, 1.0]).unwrap(),
        Array1::from_vec(vec![0.25_f32, 0.375]),
        Array2::from_shape_vec((2, 3), vec![-0.5_f32, 1.5, -2.0, 0.0, -0.5, 3.0]).unwrap(),
        Array1::from_vec(vec![-0.5_f32, 0.125]),
    )
    .unwrap();

    let pre_activation = BoundedTensor::new(
        array![-1.0_f32, 0.0, -3.0].into_dyn(),
        array![2.0_f32, 3.0, 1.0].into_dyn(),
    )
    .unwrap();

    let result = crown_elementwise_backward(&bounds, &pre_activation, test_relaxation).unwrap();

    // Row 0
    assert_eq!(result.lower_a[[0, 0]], next_down_f32(4.0));
    assert_eq!(result.lower_a[[0, 1]], next_down_f32(-3.5));
    assert_eq!(result.lower_a[[0, 2]], next_down_f32(2.0));
    assert_eq!(result.lower_b[0], next_down_f32(0.125));

    assert_eq!(result.upper_a[[0, 0]], next_up_f32(-1.0));
    assert_eq!(result.upper_a[[0, 1]], next_up_f32(5.25));
    assert_eq!(result.upper_a[[0, 2]], next_up_f32(-8.0));
    assert_eq!(result.upper_b[0], next_up_f32(-1.25));

    // Row 1
    assert_eq!(result.lower_a[[1, 0]], next_down_f32(-7.5));
    assert_eq!(result.lower_a[[1, 1]], 0.0); // zero coeff → exactly 0.0
    assert_eq!(result.lower_a[[1, 2]], next_down_f32(4.0));
    assert_eq!(result.lower_b[1], next_down_f32(-6.125));

    assert_eq!(result.upper_a[[1, 0]], 0.0); // zero coeff → exactly 0.0
    assert_eq!(result.upper_a[[1, 1]], next_up_f32(-0.5));
    assert_eq!(result.upper_a[[1, 2]], next_up_f32(4.5));
    assert_eq!(result.upper_b[1], next_up_f32(6.75));
}
