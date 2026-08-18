// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for relaxed_clip.rs static helper methods:
//! `concretize_dm_lb`, `concretize_dm_lb_from_dyn`, and `any_verified`.
//!
//! These are pure functions that compute interval-arithmetic lower bounds
//! and verification decisions. They are soundness-critical: incorrect
//! concretization can claim verification when the property doesn't hold.
//!
//! Math reference: `crates/ny-propagate/src/relaxed_clip.rs:concretize_bounds`
//!   result[b,s] = A·x_hat + sign·|A|·eps + bias
//!   where sign = -1 (lower) or +1 (upper), x_hat = (x_l+x_u)/2, eps = (x_u-x_l)/2

use ndarray::{arr2, Array3, ArrayD, IxDyn};

use crate::beta_crown::engine::BetaCrownVerifier;

// ── any_verified ──────────────────────────────────────────────────

#[test]
fn test_any_verified_true_when_exceeds_threshold() {
    // dm_lb has one entry above the threshold -> should be verified
    let dm_lb = arr2(&[[0.5, 1.2]]);
    let thresholds = arr2(&[[0.6, 1.0]]);
    assert!(BetaCrownVerifier::any_verified(&dm_lb, &thresholds));
}

#[test]
fn test_any_verified_false_when_all_at_or_below() {
    let dm_lb = arr2(&[[0.5, 1.0]]);
    let thresholds = arr2(&[[0.6, 1.0]]);
    // 0.5 < 0.6, 1.0 == 1.0 (not strictly greater) -> false
    assert!(!BetaCrownVerifier::any_verified(&dm_lb, &thresholds));
}

#[test]
fn test_any_verified_false_shape_mismatch() {
    let dm_lb = arr2(&[[0.5]]);
    let thresholds = arr2(&[[0.1, 0.2]]);
    // Mismatched shapes -> returns false defensively
    assert!(!BetaCrownVerifier::any_verified(&dm_lb, &thresholds));
}

#[test]
fn test_any_verified_rejects_nonfinite_authority() {
    assert!(!BetaCrownVerifier::any_verified(
        &arr2(&[[f32::INFINITY]]),
        &arr2(&[[0.0]]),
    ));
    assert!(!BetaCrownVerifier::any_verified(
        &arr2(&[[1.0]]),
        &arr2(&[[f32::NEG_INFINITY]]),
    ));
}

#[test]
fn test_any_verified_multi_batch_finds_any() {
    // Batch 0: both below; Batch 1: one exceeds -> true
    let dm_lb = arr2(&[[0.1, 0.2], [0.8, 0.3]]);
    let thresholds = arr2(&[[0.5, 0.5], [0.5, 0.5]]);
    assert!(BetaCrownVerifier::any_verified(&dm_lb, &thresholds));
}

#[test]
fn test_any_verified_multi_batch_none_exceed() {
    let dm_lb = arr2(&[[0.1, 0.2], [0.3, 0.4]]);
    let thresholds = arr2(&[[0.5, 0.5], [0.5, 0.5]]);
    assert!(!BetaCrownVerifier::any_verified(&dm_lb, &thresholds));
}

// ── concretize_dm_lb ──────────────────────────────────────────────

#[test]
fn test_concretize_dm_lb_point_domain() {
    // When x_l == x_u, eps = 0, result should be A*x + bias exactly
    // (modulo directed rounding).
    //
    // x_l = x_u = [2.0], so x_hat = 2.0, eps = 0.0
    // A = [[3.0]] (batch=1, n_spec=1, x_dim=1), bias = [1.0]
    // Expected: 3.0 * 2.0 + (-1)*|3.0|*0.0 + 1.0 = 7.0
    // is_lower=true so directed rounding rounds down; for exact value, result <= 7.0
    let x_l = arr2(&[[2.0]]);
    let x_u = arr2(&[[2.0]]);
    let l_a = Array3::from_shape_vec((1, 1, 1), vec![3.0]).unwrap();
    let lbias = arr2(&[[1.0]]);

    let result = BetaCrownVerifier::concretize_dm_lb(&x_l, &x_u, &l_a, &lbias, true);
    assert_eq!(result.shape(), &[1, 1]);
    // Directed rounding: result should be <= 7.0 for lower bound
    assert!(
        result[[0, 0]] <= 7.0,
        "lower bound {} should be <= 7.0",
        result[[0, 0]]
    );
    // But very close to 7.0
    assert!(
        (result[[0, 0]] - 7.0).abs() < 1e-5,
        "lower bound {} should be near 7.0",
        result[[0, 0]]
    );
}

#[test]
fn test_concretize_dm_lb_interval_domain() {
    // x_l = [1.0], x_u = [3.0] -> x_hat = 2.0, eps = 1.0
    // A = [[2.0]] (1, 1, 1), bias = [0.5]
    // Lower bound: A*x_hat - |A|*eps + bias = 2*2 - 2*1 + 0.5 = 2.5
    let x_l = arr2(&[[1.0]]);
    let x_u = arr2(&[[3.0]]);
    let l_a = Array3::from_shape_vec((1, 1, 1), vec![2.0]).unwrap();
    let lbias = arr2(&[[0.5]]);

    let result = BetaCrownVerifier::concretize_dm_lb(&x_l, &x_u, &l_a, &lbias, true);
    // With directed rounding, result <= 2.5
    assert!(
        result[[0, 0]] <= 2.5 + 1e-5,
        "lower bound should be near 2.5"
    );
    assert!(
        (result[[0, 0]] - 2.5).abs() < 1e-4,
        "lower bound {} should be near 2.5",
        result[[0, 0]]
    );
}

#[test]
fn test_concretize_dm_lb_negative_coeff() {
    // A = [[-1.0]], x_l = [0.0], x_u = [4.0]
    // x_hat = 2.0, eps = 2.0
    // Lower: (-1)*2 - |-1|*2 + 0 = -2 - 2 = -4.0
    // This is min of -x on [0, 4] = -4, correct.
    let x_l = arr2(&[[0.0]]);
    let x_u = arr2(&[[4.0]]);
    let l_a = Array3::from_shape_vec((1, 1, 1), vec![-1.0]).unwrap();
    let lbias = arr2(&[[0.0]]);

    let result = BetaCrownVerifier::concretize_dm_lb(&x_l, &x_u, &l_a, &lbias, true);
    assert!(
        (result[[0, 0]] - (-4.0)).abs() < 1e-4,
        "lower bound {} should be near -4.0",
        result[[0, 0]]
    );
}

#[test]
fn test_concretize_dm_lb_multi_spec_multi_dim() {
    // batch=1, n_spec=2, x_dim=2
    // x_l = [0, 1], x_u = [2, 3] -> x_hat = [1, 2], eps = [1, 1]
    // A = [[[1, 0], [0, 1]]], bias = [0, 0]
    // spec 0: 1*1 - 1*1 + 0*2 - 0*1 = 0 -> min(x0) on [0,2] = 0
    // spec 1: 0*1 - 0*1 + 1*2 - 1*1 = 1 -> min(x1) on [1,3] = 1
    let x_l = arr2(&[[0.0, 1.0]]);
    let x_u = arr2(&[[2.0, 3.0]]);
    let l_a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let lbias = arr2(&[[0.0, 0.0]]);

    let result = BetaCrownVerifier::concretize_dm_lb(&x_l, &x_u, &l_a, &lbias, true);
    assert_eq!(result.shape(), &[1, 2]);
    assert!(
        (result[[0, 0]] - 0.0).abs() < 1e-4,
        "spec 0 lower {} should be near 0.0",
        result[[0, 0]]
    );
    assert!(
        (result[[0, 1]] - 1.0).abs() < 1e-4,
        "spec 1 lower {} should be near 1.0",
        result[[0, 1]]
    );
}

// ── concretize_dm_lb_from_dyn matches typed variant ───────────────

#[test]
fn test_concretize_dm_lb_from_dyn_matches_typed() {
    // Same inputs in typed and dynamic form should produce identical results.
    let x_l_typed = arr2(&[[1.0, 2.0]]);
    let x_u_typed = arr2(&[[3.0, 5.0]]);
    let l_a = Array3::from_shape_vec((1, 1, 2), vec![2.0, -1.0]).unwrap();
    let lbias = arr2(&[[0.5]]);

    let result_typed =
        BetaCrownVerifier::concretize_dm_lb(&x_l_typed, &x_u_typed, &l_a, &lbias, true);

    // Convert to dyn for the from_dyn variant
    let x_l_dyn: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap();
    let x_u_dyn: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![3.0, 5.0]).unwrap();

    let result_dyn =
        BetaCrownVerifier::concretize_dm_lb_from_dyn(&x_l_dyn, &x_u_dyn, &l_a, &lbias, true);

    assert_eq!(result_typed.shape(), result_dyn.shape());
    assert_eq!(
        result_typed[[0, 0]],
        result_dyn[[0, 0]],
        "typed {} != dyn {}",
        result_typed[[0, 0]],
        result_dyn[[0, 0]]
    );
}

// ── soundness: lower bound is actually a lower bound ──────────────

#[test]
fn test_concretize_dm_lb_is_sound_lower_bound() {
    // For a linear function f(x) = A*x + bias on domain [x_l, x_u],
    // the concretized lower bound must be <= f(x) for all x in the domain.
    // Test at corners of a 2D box.
    let x_l = arr2(&[[0.0, 0.0]]);
    let x_u = arr2(&[[1.0, 1.0]]);
    let l_a = Array3::from_shape_vec((1, 1, 2), vec![3.0, -2.0]).unwrap();
    let lbias = arr2(&[[1.0]]);

    let lb = BetaCrownVerifier::concretize_dm_lb(&x_l, &x_u, &l_a, &lbias, true);

    // Evaluate f at all 4 corners of [0,1]^2
    let corners: &[(f32, f32)] = &[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
    for &(x0, x1) in corners {
        let f_val = 3.0 * x0 + (-2.0) * x1 + 1.0;
        assert!(
            lb[[0, 0]] <= f_val + 1e-6,
            "lower bound {} exceeds f({},{}) = {}",
            lb[[0, 0]],
            x0,
            x1,
            f_val
        );
    }

    // The tightest lower bound = min(f) = f(0, 1) = 3*0 - 2*1 + 1 = -1
    // Our bound should match this (interval arithmetic is exact for linear functions)
    assert!(
        (lb[[0, 0]] - (-1.0)).abs() < 1e-4,
        "lower bound {} should match exact minimum -1.0",
        lb[[0, 0]]
    );
}
