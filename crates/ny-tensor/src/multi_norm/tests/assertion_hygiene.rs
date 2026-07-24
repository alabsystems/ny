// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_scale_zero_with_infinite_inputs_returns_finite_zero() {
    let lw = Array4::from_shape_vec((1, 1, 1, 1), vec![f32::INFINITY])
        .expect("invariant: shape matches element count");
    let uw = Array4::from_shape_vec((1, 1, 1, 1), vec![f32::NEG_INFINITY])
        .expect("invariant: shape matches element count");
    let lb = Array3::from_shape_vec((1, 1, 1), vec![f32::INFINITY])
        .expect("invariant: shape matches element count");
    let ub = Array3::from_shape_vec((1, 1, 1), vec![f32::NEG_INFINITY])
        .expect("invariant: shape matches element count");
    let bounds = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub)
        .expect("invariant: valid MultiNormBounds construction");

    let scaled = bounds.scale(0.0);

    assert_eq!(scaled.lw[[0, 0, 0, 0]], 0.0);
    assert_eq!(scaled.uw[[0, 0, 0, 0]], 0.0);
    assert_eq!(scaled.lb[[0, 0, 0]], 0.0);
    assert_eq!(scaled.ub[[0, 0, 0]], 0.0);

    assert!(
        scaled.lw[[0, 0, 0, 0]].is_finite(),
        "zero scale should zero finite lower weights"
    );
    assert!(
        scaled.uw[[0, 0, 0, 0]].is_finite(),
        "zero scale should zero finite upper weights"
    );
    assert!(
        scaled.lb[[0, 0, 0]].is_finite(),
        "zero scale should zero finite lower bias"
    );
    assert!(
        scaled.ub[[0, 0, 0]].is_finite(),
        "zero scale should zero finite upper bias"
    );
}

#[test]
fn test_mul_elementwise_contains_uncertain_product() {
    let lw = Array4::from_shape_vec((1, 1, 1, 1), vec![1.0]).unwrap();
    let uw = lw.clone();
    let lb = Array3::zeros((1, 1, 1));
    let ub = lb.clone();
    let a = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub).unwrap();
    let out = a.mul_elementwise(&a).unwrap();
    let concretized = out.concretize().unwrap();
    assert!(
        concretized.lower()[[0, 0, 0]] <= -1.0,
        "uncertain product lower bound {} should contain -1.0",
        concretized.lower()[[0, 0, 0]]
    );
    assert!(
        concretized.upper()[[0, 0, 0]] >= 1.0,
        "uncertain product upper bound {} should contain 1.0",
        concretized.upper()[[0, 0, 0]]
    );
}
