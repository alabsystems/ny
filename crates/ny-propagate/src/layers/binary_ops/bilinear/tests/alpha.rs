// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::super::*;
use super::core::{assert_bounds_at_least_as_tight, assert_bounds_no_nan_ordered};
use crate::tests::assert_batched_bounds_close;

/// Build mixed-sign downstream bounds for bilinear composition tests.
fn make_mixed_sign_downstream(z_size: usize) -> crate::BatchedLinearBounds {
    let mut ds_la = ArrayD::zeros(IxDyn(&[z_size, z_size]));
    let mut ds_ua = ArrayD::zeros(IxDyn(&[z_size, z_size]));
    for i in 0..z_size {
        for j in 0..z_size {
            ds_la[[i, j]] = if (i + j) % 3 == 0 { 0.1 } else { -0.05 };
            ds_ua[[i, j]] = if (i + j) % 3 == 0 { 0.15 } else { -0.02 };
        }
    }
    crate::BatchedLinearBounds::new(
        ds_la,
        ArrayD::from_elem(IxDyn(&[z_size]), -0.1_f32),
        ds_ua,
        ArrayD::from_elem(IxDyn(&[z_size]), 0.1_f32),
        vec![z_size],
        vec![z_size],
    )
    .unwrap()
}

/// Build Q/K bounds for bilinear composition tests.
fn make_bilinear_test_bounds() -> (BoundedTensor, BoundedTensor) {
    let (m, k, n) = (3, 2, 4);
    let q = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[m, k]), vec![-1.0, 0.0, -0.5, 0.5, 0.2, -0.3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[m, k]), vec![0.5, 1.0, 0.5, 1.5, 0.8, 0.7]).unwrap(),
    )
    .unwrap();
    let kk = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[n, k]),
            vec![-0.5, -1.0, 0.0, 0.5, -0.3, 0.2, 0.1, -0.4],
        )
        .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n, k]), vec![0.5, 0.0, 1.0, 1.5, 0.7, 1.2, 1.1, 0.6])
            .unwrap(),
    )
    .unwrap();
    (q, kk)
}

/// Alpha path produces sound, NaN-free bounds (#286 Approach A+B).
#[ntest::timeout(10000)]
#[test]
fn test_alpha_path_broadcast_soundness() {
    let (m, n, k, scale) = (3, 4, 2, 0.5_f32);
    let (q_bounds, k_bounds) = make_bilinear_test_bounds();
    let downstream = make_mixed_sign_downstream(m * n);
    let layer = BilinearCrownLayer::new(true, Some(scale));
    let alphas = layer.init_alpha(&[m, k], &[n, k]).unwrap();

    let (bq, bk) = layer
        .propagate_linear_batched_binary_with_alpha(
            &downstream,
            &q_bounds,
            &k_bounds,
            Some(&alphas),
        )
        .expect("Alpha path should succeed");

    assert_bounds_no_nan_ordered("Q_alpha", &bq);
    assert_bounds_no_nan_ordered("K_alpha", &bk);
}

/// Alpha broadcast composition is at least as tight as interval-mul (#286).
#[ntest::timeout(10000)]
#[test]
fn test_alpha_broadcast_tighter_than_interval_mul() {
    let (m, n, k, scale) = (3, 4, 2, 0.5_f32);
    let (q_bounds, k_bounds) = make_bilinear_test_bounds();
    let downstream = make_mixed_sign_downstream(m * n);
    let layer = BilinearCrownLayer::new(true, Some(scale));
    let alphas = layer.init_alpha(&[m, k], &[n, k]).unwrap();

    // Broadcast (sign-split) via the alpha production path
    let (bq_bc, bk_bc) = layer
        .propagate_linear_batched_binary_with_alpha(
            &downstream,
            &q_bounds,
            &k_bounds,
            Some(&alphas),
        )
        .unwrap();

    // Interval-mul baseline
    let relaxation = BilinearRelaxation::from_bounds_with_alpha(
        &q_bounds,
        &k_bounds,
        true,
        Some(scale),
        &ndarray::Array4::ones((2, m, n, k)),
    )
    .unwrap();
    let (bq_iv, bk_iv) = relaxation.compose_backward(&downstream).unwrap();

    assert_bounds_at_least_as_tight("Q", &bq_bc, &bq_iv, 1e-3);
    assert_bounds_at_least_as_tight("K", &bk_bc, &bk_iv, 1e-3);
}

/// Direction-dependent alpha: with uniform alphas (all ones), bidirectional
/// composition produces identical results to single-direction (#286 Phase 2).
#[ntest::timeout(10000)]
#[test]
fn test_bidirectional_matches_single_when_uniform() {
    let (m, n, k, scale) = (3, 4, 2, 0.5_f32);
    let (q_bounds, k_bounds) = make_bilinear_test_bounds();
    let downstream = make_mixed_sign_downstream(m * n);

    // Single-direction: same relaxation for both lower and upper
    let relax_single = BilinearRelaxation::from_bounds_with_alpha(
        &q_bounds,
        &k_bounds,
        true,
        Some(scale),
        &ndarray::Array4::ones((2, m, n, k)),
    )
    .unwrap();
    let (sq, sk) = relax_single
        .compose_backward_broadcast(&downstream)
        .unwrap();

    // Bidirectional with identical relaxations: should match single
    let relax_lower = BilinearRelaxation::from_bounds_with_alpha(
        &q_bounds,
        &k_bounds,
        true,
        Some(scale),
        &ndarray::Array4::ones((2, m, n, k)),
    )
    .unwrap();
    let relax_upper = BilinearRelaxation::from_bounds_with_alpha(
        &q_bounds,
        &k_bounds,
        true,
        Some(scale),
        &ndarray::Array4::ones((2, m, n, k)),
    )
    .unwrap();
    let (bq, bk) = relax_lower
        .compose_backward_broadcast_bidirectional(&relax_upper, &downstream)
        .unwrap();

    assert_batched_bounds_close(&sq, &bq, 1e-6, "Q single vs bidirectional");
    assert_batched_bounds_close(&sk, &bk, 1e-6, "K single vs bidirectional");
}

/// Direction-dependent alpha: different r_l/r_u for lower vs upper directions
/// still produces sound (NaN-free, lower <= upper) bounds (#286 Phase 2).
#[ntest::timeout(10000)]
#[test]
fn test_bidirectional_different_alphas_sound() {
    let (m, n, k, scale) = (3, 4, 2, 0.5_f32);
    let (q_bounds, k_bounds) = make_bilinear_test_bounds();
    let downstream = make_mixed_sign_downstream(m * n);
    let layer = BilinearCrownLayer::new(true, Some(scale));

    // Create alphas with different values per direction
    let mut alphas = ndarray::Array4::ones((4, m, n, k));
    // Lower direction: r_l=0.3, r_u=0.7
    alphas.slice_mut(ndarray::s![0, .., .., ..]).fill(0.3);
    alphas.slice_mut(ndarray::s![2, .., .., ..]).fill(0.7);
    // Upper direction: r_l=0.8, r_u=0.2
    alphas.slice_mut(ndarray::s![1, .., .., ..]).fill(0.8);
    alphas.slice_mut(ndarray::s![3, .., .., ..]).fill(0.2);

    let (bq, bk) = layer
        .propagate_linear_batched_binary_with_alpha(
            &downstream,
            &q_bounds,
            &k_bounds,
            Some(&alphas),
        )
        .expect("Direction-dependent alpha path should succeed");

    assert_bounds_no_nan_ordered("Q_bidir", &bq);
    assert_bounds_no_nan_ordered("K_bidir", &bk);
}

/// Direction-dependent alpha via full propagate path matches direct bidirectional
/// compose when using the same alpha slices (#286 Phase 2).
#[ntest::timeout(10000)]
#[test]
fn test_propagate_alpha_matches_direct_bidirectional() {
    let (m, n, k, scale) = (3, 4, 2, 0.5_f32);
    let (q_bounds, k_bounds) = make_bilinear_test_bounds();
    let downstream = make_mixed_sign_downstream(m * n);
    let layer = BilinearCrownLayer::new(true, Some(scale));

    // Build alphas with distinct per-direction values
    let mut alphas = ndarray::Array4::ones((4, m, n, k));
    alphas.slice_mut(ndarray::s![0, .., .., ..]).fill(0.4);
    alphas.slice_mut(ndarray::s![1, .., .., ..]).fill(0.6);
    alphas.slice_mut(ndarray::s![2, .., .., ..]).fill(0.9);
    alphas.slice_mut(ndarray::s![3, .., .., ..]).fill(0.1);

    // Via propagate_linear_batched_binary_with_alpha
    let (pq, pk) = layer
        .propagate_linear_batched_binary_with_alpha(
            &downstream,
            &q_bounds,
            &k_bounds,
            Some(&alphas),
        )
        .unwrap();

    // Manually build the two relaxations and compose directly
    let mut rl_ru_lower = ndarray::Array4::zeros((2, m, n, k));
    rl_ru_lower.slice_mut(ndarray::s![0, .., .., ..]).fill(0.4);
    rl_ru_lower.slice_mut(ndarray::s![1, .., .., ..]).fill(0.9);
    let mut rl_ru_upper = ndarray::Array4::zeros((2, m, n, k));
    rl_ru_upper.slice_mut(ndarray::s![0, .., .., ..]).fill(0.6);
    rl_ru_upper.slice_mut(ndarray::s![1, .., .., ..]).fill(0.1);

    let relax_lower = BilinearRelaxation::from_bounds_with_alpha(
        &q_bounds,
        &k_bounds,
        true,
        Some(scale),
        &rl_ru_lower,
    )
    .unwrap();
    let relax_upper = BilinearRelaxation::from_bounds_with_alpha(
        &q_bounds,
        &k_bounds,
        true,
        Some(scale),
        &rl_ru_upper,
    )
    .unwrap();
    let (dq, dk) = relax_lower
        .compose_backward_broadcast_bidirectional(&relax_upper, &downstream)
        .unwrap();

    assert_batched_bounds_close(&pq, &dq, 1e-5, "Q propagate vs direct");
    assert_batched_bounds_close(&pk, &dk, 1e-5, "K propagate vs direct");
}
