// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `crown_elementwise_backward_batched` and
//! `crown_elementwise_backward_batched_indexed` in `layers/common/crown_batched.rs`.
//!
//! These tests cover the shared batched CROWN backward helper used by 18+
//! elementwise activations. Coverage targets: coefficient × relaxation composition,
//! directed rounding, f64 bias accumulation, non-finite row fallback (#3009),
//! zero-coefficient handling (#1736), and neuron index forwarding.
//!
//! Part of #3463.

use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{
    crown_elementwise_backward_batched, crown_elementwise_backward_batched_indexed,
};
use crate::BatchedLinearBounds;
use ndarray::{ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

/// Synthetic relaxation matching `patches_backward.rs` for cross-variant comparison.
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

/// Single batch, mixed-sign coefficients — mirrors `patches_backward` test for
/// cross-variant consistency.
///
/// Verifies compose_lower/compose_upper coefficient selection:
///   positive coeff → lower/upper relaxation (preserves direction)
///   negative coeff → upper/lower relaxation (flips direction)
/// Plus directed rounding on both coefficients and f64→f32 bias conversion.
///
/// Hand-computed expected values match
/// `test_crown_elementwise_backward_patches_dense_mixed_sign_coefficients`.
#[test]
fn test_crown_elementwise_backward_batched_mixed_sign_coefficients() {
    // Shape: out_dim=1, in_dim=2 (no batch dims)
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.5_f32, -2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.25_f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-0.75_f32, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.5_f32]).unwrap(),
        vec![2],
        vec![1],
    )
    .unwrap();

    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0_f32, 4.0]).unwrap(),
    )
    .unwrap();

    let result =
        crown_elementwise_backward_batched(&bounds, &pre_activation, test_relaxation).unwrap();

    // Neuron 0: l=-1, u=2 → relax=(ls=2.0, li=1.25, us=2.5, ui=3.25)
    // Neuron 1: l=-2, u=4 → relax=(ls=3.0, li=2.25, us=4.5, ui=5.25)
    //
    // Lower: la[0]=1.5 > 0 → lower relax n0: coeff=next_down(1.5×2.0)
    //        la[1]=-2.0 < 0 → upper relax n1: coeff=next_down(-2.0×4.5)
    //        bias = 0.25 + 1.5×1.25 + (-2.0)×5.25 = 0.25 + 1.875 - 10.5 = -8.375
    assert_eq!(result.lower_a()[[0, 0]], next_down_f32(3.0));
    assert_eq!(result.lower_a()[[0, 1]], next_down_f32(-9.0));
    assert_eq!(result.lower_b()[[0]], next_down_f32(-8.375));

    // Upper: ua[0]=-0.75 < 0 → lower relax n0: coeff=next_up(-0.75×2.0)
    //        ua[1]=3.0 > 0  → upper relax n1: coeff=next_up(3.0×4.5)
    //        bias = -0.5 + (-0.75)×1.25 + 3.0×5.25 = -0.5 - 0.9375 + 15.75 = 14.3125
    assert_eq!(result.upper_a()[[0, 0]], next_up_f32(-1.5));
    assert_eq!(result.upper_a()[[0, 1]], next_up_f32(13.5));
    assert_eq!(result.upper_b()[[0]], next_up_f32(14.3125));
}

/// Two batches — verifies batch iteration and per-batch relaxation.
///
/// Each batch has different pre-activation bounds, producing different relaxations
/// for the same neuron index. Verifies that batch 0 and batch 1 are composed
/// independently without cross-contamination.
#[test]
fn test_crown_elementwise_backward_batched_multi_batch() {
    // Shape: batch=2, out_dim=1, in_dim=2
    let bounds = BatchedLinearBounds::new(
        // lower_a [2,1,2]: batch 0 = [[1.0, -1.0]], batch 1 = [[-2.0, 0.5]]
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), vec![1.0, -1.0, -2.0, 0.5]).unwrap(),
        // lower_b [2,1]: batch 0 = [0.0], batch 1 = [0.125]
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![0.0, 0.125]).unwrap(),
        // upper_a [2,1,2]: batch 0 = [[0.5, 2.0]], batch 1 = [[-1.0, -0.5]]
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), vec![0.5, 2.0, -1.0, -0.5]).unwrap(),
        // upper_b [2,1]: batch 0 = [0.0], batch 1 = [-0.125]
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![0.0, -0.125]).unwrap(),
        vec![2, 2],
        vec![2, 1],
    )
    .unwrap();

    // pre_activation [2,2]: batch 0 = [-1, -2], batch 1 = [-3, 0]
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, -2.0, -3.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 4.0, 1.0, 5.0]).unwrap(),
    )
    .unwrap();

    let result =
        crown_elementwise_backward_batched(&bounds, &pre_activation, test_relaxation).unwrap();

    // --- Batch 0 ---
    // n0: l=-1, u=2 → relax=(ls=2.0, li=1.25, us=2.5, ui=3.25)
    // n1: l=-2, u=4 → relax=(ls=3.0, li=2.25, us=4.5, ui=5.25)
    //
    // Lower: la=[1.0, -1.0]
    //   n0: 1.0>0 → lower: coeff=next_down(1.0×2.0), intercept=1.0×1.25=1.25
    //   n1: -1.0<0 → upper: coeff=next_down(-1.0×4.5), intercept=-1.0×5.25=-5.25
    //   bias = 0.0 + 1.25 - 5.25 = -4.0
    assert_eq!(result.lower_a()[[0, 0, 0]], next_down_f32(2.0));
    assert_eq!(result.lower_a()[[0, 0, 1]], next_down_f32(-4.5));
    assert_eq!(result.lower_b()[[0, 0]], next_down_f32(-4.0));

    // Upper: ua=[0.5, 2.0]
    //   n0: 0.5>0 → upper: coeff=next_up(0.5×2.5), intercept=0.5×3.25=1.625
    //   n1: 2.0>0 → upper: coeff=next_up(2.0×4.5), intercept=2.0×5.25=10.5
    //   bias = 0.0 + 1.625 + 10.5 = 12.125
    assert_eq!(result.upper_a()[[0, 0, 0]], next_up_f32(1.25));
    assert_eq!(result.upper_a()[[0, 0, 1]], next_up_f32(9.0));
    assert_eq!(result.upper_b()[[0, 0]], next_up_f32(12.125));

    // --- Batch 1 ---
    // n0: l=-3, u=1 → relax=(ls=4.0, li=3.25, us=1.5, ui=2.25)
    // n1: l=0,  u=5 → relax=(ls=1.0, li=0.25, us=5.5, ui=6.25)
    //
    // Lower: la=[-2.0, 0.5]
    //   n0: -2.0<0 → upper: coeff=next_down(-2.0×1.5), intercept=-2.0×2.25=-4.5
    //   n1: 0.5>0  → lower: coeff=next_down(0.5×1.0),  intercept=0.5×0.25=0.125
    //   bias = 0.125 + (-4.5) + 0.125 = -4.25
    assert_eq!(result.lower_a()[[1, 0, 0]], next_down_f32(-3.0));
    assert_eq!(result.lower_a()[[1, 0, 1]], next_down_f32(0.5));
    assert_eq!(result.lower_b()[[1, 0]], next_down_f32(-4.25));

    // Upper: ua=[-1.0, -0.5]
    //   n0: -1.0<0 → lower: coeff=next_up(-1.0×4.0), intercept=-1.0×3.25=-3.25
    //   n1: -0.5<0 → lower: coeff=next_up(-0.5×1.0), intercept=-0.5×0.25=-0.125
    //   bias = -0.125 + (-3.25) + (-0.125) = -3.5
    assert_eq!(result.upper_a()[[1, 0, 0]], next_up_f32(-4.0));
    assert_eq!(result.upper_a()[[1, 0, 1]], next_up_f32(-0.5));
    assert_eq!(result.upper_b()[[1, 0]], next_up_f32(-3.5));
}

/// Non-finite coefficient overflow → row fallback (#3009).
///
/// When coeff × slope overflows to ±Inf, the entire output row is zeroed
/// and the bias is set to ±Inf (maximally conservative but sound).
/// Even valid coefficients in the same row are zeroed to prevent mixed
/// finite/infinite bound evaluation.
#[test]
fn test_crown_elementwise_backward_batched_nonfinite_row_fallback() {
    // Shape: out_dim=1, in_dim=2
    // One coefficient overflows per bound direction; the other is normal.
    let bounds = BatchedLinearBounds::new(
        // lower: [MAX, 0.5] — n0 will overflow
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![f32::MAX, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![10.0_f32]).unwrap(),
        // upper: [0.5, MAX] — n1 will overflow
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.5_f32, f32::MAX]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-5.0_f32]).unwrap(),
        vec![2],
        vec![1],
    )
    .unwrap();

    // Both neurons: l=-1, u=2 → relax=(ls=2.0, ..., us=2.5, ...)
    // MAX × 2.0 = Inf → nonfinite flag set
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0_f32, 2.0]).unwrap(),
    )
    .unwrap();

    let result =
        crown_elementwise_backward_batched(&bounds, &pre_activation, test_relaxation).unwrap();

    // Lower row: n0 overflow → entire row zeroed, bias → NEG_INFINITY
    assert_eq!(result.lower_a()[[0, 0]], 0.0);
    assert_eq!(result.lower_a()[[0, 1]], 0.0);
    assert_eq!(result.lower_b()[[0]], f32::NEG_INFINITY);

    // Upper row: n1 overflow → entire row zeroed, bias → INFINITY
    assert_eq!(result.upper_a()[[0, 0]], 0.0);
    assert_eq!(result.upper_a()[[0, 1]], 0.0);
    assert_eq!(result.upper_b()[[0]], f32::INFINITY);
}

/// Zero coefficients produce no contribution and no NaN (#1736).
///
/// When a coefficient is exactly 0.0, compose_lower/compose_upper returns
/// ComposeResult::ZERO without touching the relaxation slopes (which could be
/// ±Inf). Verifies the A-matrix entry stays 0.0 and no bias is contributed.
#[test]
fn test_crown_elementwise_backward_batched_zero_coeff_no_contribution() {
    // Shape: out_dim=1, in_dim=2
    // Neuron 0 has zero coefficients, neuron 1 has normal coefficients.
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0_f32, 1.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0_f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0_f32, -0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0_f32]).unwrap(),
        vec![2],
        vec![1],
    )
    .unwrap();

    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0_f32, 2.0]).unwrap(),
    )
    .unwrap();

    let result =
        crown_elementwise_backward_batched(&bounds, &pre_activation, test_relaxation).unwrap();

    // Both neurons: l=-1, u=2 → relax=(ls=2.0, li=1.25, us=2.5, ui=3.25)
    //
    // Lower: la[0]=0.0 → ZERO (coeff=0, intercept=0)
    //        la[1]=1.5>0 → lower: coeff=next_down(1.5×2.0), intercept=1.5×1.25=1.875
    //        bias = 0.0 + 0.0 + 1.875 = 1.875
    assert_eq!(result.lower_a()[[0, 0]], 0.0);
    assert_eq!(result.lower_a()[[0, 1]], next_down_f32(3.0));
    assert_eq!(result.lower_b()[[0]], next_down_f32(1.875));

    // Upper: ua[0]=0.0 → ZERO
    //        ua[1]=-0.5<0 → lower: coeff=next_up(-0.5×2.0), intercept=-0.5×1.25=-0.625
    //        bias = 0.0 + 0.0 + (-0.625) = -0.625
    assert_eq!(result.upper_a()[[0, 0]], 0.0);
    assert_eq!(result.upper_a()[[0, 1]], next_up_f32(-1.0));
    assert_eq!(result.upper_b()[[0]], next_up_f32(-0.625));
}

/// Indexed variant forwards neuron index to relaxation function.
///
/// Uses an index-dependent relaxation (scaled by `i+1`) to verify that
/// neurons with identical pre-activation bounds produce different coefficients
/// when the relaxation depends on the neuron index.
#[test]
fn test_crown_elementwise_backward_batched_indexed_neuron_index() {
    // Shape: out_dim=1, in_dim=2. Uniform coefficients to isolate index effect.
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0_f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0_f32]).unwrap(),
        vec![2],
        vec![1],
    )
    .unwrap();

    // Same bounds for both neurons — only the index differs.
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0_f32, 2.0]).unwrap(),
    )
    .unwrap();

    // Index-scaled relaxation: scale = i + 1
    // n0 (i=0, scale=1): relax=(ls=2.0,  li=1.25, us=2.5, ui=3.25)
    // n1 (i=1, scale=2): relax=(ls=4.0,  li=2.50, us=5.0, ui=6.50)
    let result = crown_elementwise_backward_batched_indexed(&bounds, &pre_activation, |l, u, i| {
        let scale = (i + 1) as f32;
        LinearRelaxation::new(
            scale * (l.abs() + 1.0),
            scale * (l.abs() + 0.25),
            scale * (u + 0.5),
            scale * (u + 1.25),
        )
    })
    .unwrap();

    // Lower: all coeff=1.0>0 → lower relaxation
    //   n0: coeff=next_down(1.0×2.0), intercept=1.0×1.25=1.25
    //   n1: coeff=next_down(1.0×4.0), intercept=1.0×2.50=2.50
    //   bias = 0.0 + 1.25 + 2.50 = 3.75
    assert_eq!(result.lower_a()[[0, 0]], next_down_f32(2.0));
    assert_eq!(result.lower_a()[[0, 1]], next_down_f32(4.0));
    // Key assertion: n0 and n1 coefficients DIFFER (2.0 vs 4.0)
    // despite identical pre-activation bounds, proving index was forwarded.
    assert_ne!(result.lower_a()[[0, 0]], result.lower_a()[[0, 1]]);
    assert_eq!(result.lower_b()[[0]], next_down_f32(3.75));

    // Upper: all coeff=1.0>0 → upper relaxation
    //   n0: coeff=next_up(1.0×2.5), intercept=1.0×3.25=3.25
    //   n1: coeff=next_up(1.0×5.0), intercept=1.0×6.50=6.50
    //   bias = 0.0 + 3.25 + 6.50 = 9.75
    assert_eq!(result.upper_a()[[0, 0]], next_up_f32(2.5));
    assert_eq!(result.upper_a()[[0, 1]], next_up_f32(5.0));
    assert_ne!(result.upper_a()[[0, 0]], result.upper_a()[[0, 1]]);
    assert_eq!(result.upper_b()[[0]], next_up_f32(9.75));
}

/// A first batched activation must create a coefficient-error carrier for its
/// own f32 coefficient product; this cannot depend on an incoming carrier.
#[test]
fn test_crown_elementwise_backward_batched_carries_fresh_product_gap() {
    let lower_coeff = 1.3_f32;
    let upper_coeff = -1.7_f32;
    let lower_slope = 0.7_f32;
    let upper_slope = 0.9_f32;
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![lower_coeff]).unwrap(),
        ArrayD::zeros(IxDyn(&[1])),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![upper_coeff]).unwrap(),
        ArrayD::zeros(IxDyn(&[1])),
        vec![1],
        vec![1],
    )
    .unwrap();
    assert!(!bounds.has_coeff_err());
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
    )
    .unwrap();

    let result = crown_elementwise_backward_batched(&bounds, &pre_activation, |_l, _u| {
        LinearRelaxation::new(lower_slope, 0.0, upper_slope, 0.0)
    })
    .unwrap();

    assert!(result.has_coeff_err());
    let lower_gap = (f64::from(lower_coeff) * f64::from(lower_slope)
        - f64::from(result.lower_a()[[0, 0]]))
    .abs();
    let upper_gap = (f64::from(upper_coeff) * f64::from(lower_slope)
        - f64::from(result.upper_a()[[0, 0]]))
    .abs();
    assert!(
        f64::from(result.lower_a_err.as_ref().unwrap()[[0, 0]]) >= lower_gap,
        "fresh lower product gap was not enclosed"
    );
    assert!(
        f64::from(result.upper_a_err.as_ref().unwrap()[[0, 0]]) >= upper_gap,
        "fresh upper product gap was not enclosed"
    );
}
