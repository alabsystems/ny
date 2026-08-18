// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for decomposed grouped centered-normalization CROWN backward propagation.
//!
//! Verifies `decomposed_grouped_centered_crown_backward` from `grouped_centered.rs`
//! produces sound linear bounds that always contain the true GroupNorm output.
//! GroupNorm normalizes each group of channels independently across all channels
//! in the group and all spatial/time positions.
//!
//! Part of #4209.

use super::grouped_centered::decomposed_grouped_centered_crown_backward;
use super::tests_support::{constant_batched_bounds, interpolate};
use ndarray::{arr1, Array1, Array2};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

/// Compute true GroupNorm output for a flat [C*T] input.
///
/// GroupNorm(x)[c, t] = ny[c] * (x[c, t] - mean_g) / sqrt(var_g + eps) + beta[c]
/// where mean_g and var_g are computed over all channels in group g and all time steps.
fn true_group_norm(
    x: &[f32],
    ny: &[f32],
    beta: &[f32],
    eps: f32,
    num_channels: usize,
    num_groups: usize,
) -> Vec<f64> {
    let total = x.len();
    let time_len = total / num_channels;
    let channels_per_group = num_channels / num_groups;
    let group_size = channels_per_group * time_len;
    let mut output = vec![0.0_f64; total];

    for g in 0..num_groups {
        let channel_start = g * channels_per_group;
        // Collect all elements in this group
        let mut group_elems = Vec::with_capacity(group_size);
        for c in channel_start..channel_start + channels_per_group {
            for t in 0..time_len {
                group_elems.push(f64::from(x[c * time_len + t]));
            }
        }

        let n = group_elems.len() as f64;
        let mean = group_elems.iter().sum::<f64>() / n;
        let var = group_elems
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f64>()
            / n;
        let std = (var + f64::from(eps)).sqrt();

        for c in channel_start..channel_start + channels_per_group {
            for t in 0..time_len {
                let idx = c * time_len + t;
                output[idx] =
                    f64::from(ny[c]) * (f64::from(x[idx]) - mean) / std + f64::from(beta[c]);
            }
        }
    }
    output
}

/// Create identity upstream bounds for flat [C*T] layout: A = eye(n), b = 0.
fn identity_upstream_flat(n: usize) -> crate::BatchedLinearBounds {
    let eye = Array2::eye(n);
    let zeros = Array1::zeros(n);
    constant_batched_bounds(eye.clone(), zeros.clone(), eye, zeros, n)
}

// --- GroupNorm with 2 groups ---

#[ntest::timeout(10000)]
#[test]
fn test_grouped_centered_identity_upstream_returns_ok() -> Result<()> {
    let num_channels = 4;
    let num_groups = 2;
    let time_len = 2;
    let n = num_channels * time_len;
    let ny = arr1(&[1.0, 1.0, 1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0, 0.0, 0.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 0.8, 1.2, 0.6, 1.1, 0.9, 1.3]).into_dyn(),
        arr1(&[1.5, 2.0, 1.8, 2.2, 1.6, 2.1, 1.9, 2.3]).into_dyn(),
    )?;
    let upstream = identity_upstream_flat(n);

    let result = decomposed_grouped_centered_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
        num_groups,
    )?;

    assert_eq!(
        result.validation.total_rows, n,
        "expected total_rows == {n}, got {}",
        result.validation.total_rows
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_grouped_centered_soundness_at_center() -> Result<()> {
    let num_channels = 4;
    let num_groups = 2;
    let time_len = 2;
    let n = num_channels * time_len;
    let ny = arr1(&[1.0, 1.0, 1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0, 0.0, 0.0]);
    let eps = 1e-5;
    let x_lower = [0.5_f32, 1.0, 0.8, 1.2, 0.6, 1.1, 0.9, 1.3];
    let x_upper = [1.5_f32, 2.0, 1.8, 2.2, 1.6, 2.1, 1.9, 2.3];
    let x_ibp = BoundedTensor::new(arr1(&x_lower).into_dyn(), arr1(&x_upper).into_dyn())?;
    let upstream = identity_upstream_flat(n);

    let result = decomposed_grouped_centered_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
        num_groups,
    )?;
    let bounds = &result.bounds;

    let x_center: Vec<f32> = x_lower
        .iter()
        .zip(x_upper.iter())
        .map(|(&l, &u)| f32::midpoint(l, u))
        .collect();
    let true_output = true_group_norm(
        &x_center,
        ny.as_slice().unwrap(),
        beta.as_slice().unwrap(),
        eps,
        num_channels,
        num_groups,
    );

    let point = BoundedTensor::new(
        Array1::from_vec(x_center.clone()).into_dyn(),
        Array1::from_vec(x_center).into_dyn(),
    )?;
    let result_ibp = bounds.concretize_sound(&point)?;

    for j in 0..n {
        let lower = f64::from(result_ibp.lower().as_slice().unwrap()[j]);
        let upper = f64::from(result_ibp.upper().as_slice().unwrap()[j]);
        assert!(
            lower <= true_output[j] + 1e-4,
            "dim {j}: lower bound {lower} > true output {} + tolerance",
            true_output[j]
        );
        assert!(
            upper >= true_output[j] - 1e-4,
            "dim {j}: upper bound {upper} < true output {} - tolerance",
            true_output[j]
        );
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_grouped_centered_soundness_at_corners() -> Result<()> {
    // 2 channels, 1 group (=LayerNorm equivalent), 2 time steps
    let num_channels = 2;
    let num_groups = 1;
    let time_len = 2;
    let n = num_channels * time_len;
    let ny = arr1(&[1.5, 0.8]);
    let beta = arr1(&[0.1, -0.2]);
    let eps = 1e-5;
    let x_lower = [0.5_f32, 1.0, 0.8, 1.2];
    let x_upper = [1.5_f32, 2.0, 1.8, 2.2];
    let x_ibp = BoundedTensor::new(arr1(&x_lower).into_dyn(), arr1(&x_upper).into_dyn())?;
    let upstream = identity_upstream_flat(n);

    let result = decomposed_grouped_centered_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
        num_groups,
    )?;
    let bounds = &result.bounds;
    let ny_slice = ny.as_slice().unwrap();
    let beta_slice = beta.as_slice().unwrap();

    // Check all 16 corners
    for &x0 in &[x_lower[0], x_upper[0]] {
        for &x1 in &[x_lower[1], x_upper[1]] {
            for &x2 in &[x_lower[2], x_upper[2]] {
                for &x3 in &[x_lower[3], x_upper[3]] {
                    let x = vec![x0, x1, x2, x3];
                    let true_output =
                        true_group_norm(&x, ny_slice, beta_slice, eps, num_channels, num_groups);
                    let point = BoundedTensor::new(
                        Array1::from_vec(x.clone()).into_dyn(),
                        Array1::from_vec(x).into_dyn(),
                    )?;
                    let result_ibp = bounds.concretize_sound(&point)?;

                    for j in 0..n {
                        let lower = f64::from(result_ibp.lower().as_slice().unwrap()[j]);
                        let upper = f64::from(result_ibp.upper().as_slice().unwrap()[j]);
                        assert!(
                            lower <= true_output[j] + 1e-3,
                            "corner ({x0},{x1},{x2},{x3}) dim {j}: lower {lower} > true {}",
                            true_output[j]
                        );
                        assert!(
                            upper >= true_output[j] - 1e-3,
                            "corner ({x0},{x1},{x2},{x3}) dim {j}: upper {upper} < true {}",
                            true_output[j]
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_grouped_centered_forward_mode_returns_ok() -> Result<()> {
    let num_channels = 4;
    let num_groups = 2;
    let time_len = 2;
    let n = num_channels * time_len;
    let ny = arr1(&[1.0, 1.0, 1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0, 0.0, 0.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 0.8, 1.2, 0.6, 1.1, 0.9, 1.3]).into_dyn(),
        arr1(&[1.5, 2.0, 1.8, 2.2, 1.6, 2.1, 1.9, 2.3]).into_dyn(),
    )?;
    let upstream = identity_upstream_flat(n);

    let result = decomposed_grouped_centered_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        true, // forward_mode
        num_channels,
        num_groups,
    )?;

    assert_eq!(
        result.validation.total_rows, n,
        "forward mode should produce same total_rows"
    );
    Ok(())
}

// --- Error path tests ---

#[ntest::timeout(10000)]
#[test]
fn test_grouped_centered_zero_channels_error() {
    let n = 4;
    let ny = arr1::<f32>(&[]);
    let beta = arr1::<f32>(&[]);
    let eps = 1e-5;
    let upstream = identity_upstream_flat(n);
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5, 2.0]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5, 3.0]).into_dyn(),
    )
    .unwrap();

    let result = decomposed_grouped_centered_crown_backward(
        &upstream, &ny, &beta, eps, &x_ibp, false, 0, // zero channels
        1,
    );
    assert!(result.is_err(), "should error on num_channels == 0");
}

#[ntest::timeout(10000)]
#[test]
fn test_grouped_centered_zero_groups_error() {
    let n = 4;
    let ny = arr1(&[1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0]);
    let eps = 1e-5;
    let upstream = identity_upstream_flat(n);
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5, 2.0]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5, 3.0]).into_dyn(),
    )
    .unwrap();

    let result = decomposed_grouped_centered_crown_backward(
        &upstream, &ny, &beta, eps, &x_ibp, false, 2, 0, // zero groups
    );
    assert!(result.is_err(), "should error on num_groups == 0");
}

#[ntest::timeout(10000)]
#[test]
fn test_grouped_centered_channels_not_divisible_by_groups_error() {
    let n = 6;
    let ny = arr1(&[1.0, 1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0, 0.0]);
    let eps = 1e-5;
    let upstream = identity_upstream_flat(n);
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5, 0.8, 1.2, 1.6]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5, 1.8, 2.2, 2.6]).into_dyn(),
    )
    .unwrap();

    let result = decomposed_grouped_centered_crown_backward(
        &upstream, &ny, &beta, eps, &x_ibp, false, 3,
        2, // 3 channels not divisible by 2 groups
    );
    assert!(
        result.is_err(),
        "should error when num_channels not divisible by num_groups"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_grouped_centered_ny_mismatch_error() {
    let n = 4;
    let ny = arr1(&[1.0]); // wrong size: 1 instead of 2
    let beta = arr1(&[0.0, 0.0]);
    let eps = 1e-5;
    let upstream = identity_upstream_flat(n);
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5, 2.0]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5, 3.0]).into_dyn(),
    )
    .unwrap();

    let result =
        decomposed_grouped_centered_crown_backward(&upstream, &ny, &beta, eps, &x_ibp, false, 2, 1);
    assert!(
        result.is_err(),
        "should error on ny size mismatch with num_channels"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_grouped_centered_fallback_count_nonnegative() -> Result<()> {
    let num_channels = 4;
    let num_groups = 2;
    let time_len = 2;
    let n = num_channels * time_len;
    let ny = arr1(&[1.0, 0.5, 2.0, 0.75]);
    let beta = arr1(&[0.0, 0.1, -0.1, 0.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.2, 0.8, 0.4, 1.0, 0.3, 0.9, 0.5, 1.1]).into_dyn(),
        arr1(&[0.6, 1.2, 0.8, 1.4, 0.7, 1.3, 0.9, 1.5]).into_dyn(),
    )?;
    let upstream = identity_upstream_flat(n);

    let result = decomposed_grouped_centered_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
        num_groups,
    )?;
    assert!(
        result.validation.fallback_rows <= result.validation.total_rows,
        "fallback_rows {} should not exceed total_rows {}",
        result.validation.fallback_rows,
        result.validation.total_rows
    );
    Ok(())
}

// --- Instance norm as special case of grouped centered ---

#[ntest::timeout(10000)]
#[test]
fn test_grouped_centered_instance_norm_equivalence() -> Result<()> {
    // When num_groups == num_channels, grouped centered should behave like instance norm
    let num_channels = 2;
    let num_groups = 2; // == num_channels
    let time_len = 3;
    let n = num_channels * time_len;
    let ny = arr1(&[1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0]);
    let eps = 1e-5;
    let x_lower = [0.5_f32, 1.0, 1.5, 0.8, 1.2, 1.6];
    let x_upper = [1.5_f32, 2.0, 2.5, 1.8, 2.2, 2.6];
    let x_ibp = BoundedTensor::new(arr1(&x_lower).into_dyn(), arr1(&x_upper).into_dyn())?;
    let upstream = identity_upstream_flat(n);

    let result = decomposed_grouped_centered_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
        num_groups,
    )?;
    let bounds = &result.bounds;

    // Should still produce sound bounds at center
    let x_center: Vec<f32> = x_lower
        .iter()
        .zip(x_upper.iter())
        .map(|(&l, &u)| f32::midpoint(l, u))
        .collect();

    // True instance norm: each channel normalized independently
    let true_output = true_group_norm(
        &x_center,
        ny.as_slice().unwrap(),
        beta.as_slice().unwrap(),
        eps,
        num_channels,
        num_groups,
    );

    let point = BoundedTensor::new(
        Array1::from_vec(x_center.clone()).into_dyn(),
        Array1::from_vec(x_center).into_dyn(),
    )?;
    let result_ibp = bounds.concretize_sound(&point)?;

    for j in 0..n {
        let lower = f64::from(result_ibp.lower().as_slice().unwrap()[j]);
        let upper = f64::from(result_ibp.upper().as_slice().unwrap()[j]);
        assert!(
            lower <= true_output[j] + 1e-4,
            "dim {j}: lower {lower} > true {} + tol",
            true_output[j]
        );
        assert!(
            upper >= true_output[j] - 1e-4,
            "dim {j}: upper {upper} < true {} - tol",
            true_output[j]
        );
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(64) })]

    #[test]
    fn proptest_grouped_centered_contains_true_output(
        // 4 channels, 2 groups, 1 time step each — minimal shape
        x0_l in 0.2f32..2.0,
        x0_w in 0.05f32..0.8,
        x1_l in 0.2f32..2.0,
        x1_w in 0.05f32..0.8,
        x2_l in 0.2f32..2.0,
        x2_w in 0.05f32..0.8,
        x3_l in 0.2f32..2.0,
        x3_w in 0.05f32..0.8,
        // Randomized ny/beta to stress-test affine interaction with CROWN relaxation
        g0 in 0.3f32..2.5,
        g1 in 0.3f32..2.5,
        g2 in 0.3f32..2.5,
        g3 in 0.3f32..2.5,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        b2 in -1.0f32..1.0,
        b3 in -1.0f32..1.0,
        t0 in 0.0f32..1.0,
        t1 in 0.0f32..1.0,
        t2 in 0.0f32..1.0,
        t3 in 0.0f32..1.0,
    ) {
        let num_channels = 4;
        let num_groups = 2;
        let time_len = 1;
        let n = num_channels * time_len;
        let ny = arr1(&[g0, g1, g2, g3]);
        let beta = arr1(&[b0, b1, b2, b3]);
        let eps = 1e-5_f32;
        let x_lower = [x0_l, x1_l, x2_l, x3_l];
        let x_upper = [x0_l + x0_w, x1_l + x1_w, x2_l + x2_w, x3_l + x3_w];
        let x_ibp = BoundedTensor::new(
            arr1(&x_lower).into_dyn(),
            arr1(&x_upper).into_dyn(),
        ).unwrap();
        let upstream = identity_upstream_flat(n);

        let result = decomposed_grouped_centered_crown_backward(
            &upstream, &ny, &beta, eps, &x_ibp, false,
            num_channels, num_groups,
        ).map_err(|error| TestCaseError::fail(
            format!("decomposed grouped centered must accept the generated finite domain: {error}")
        ))?;
        let bounds = &result.bounds;

        let x_sample = vec![
            interpolate(x_lower[0], x_upper[0], t0),
            interpolate(x_lower[1], x_upper[1], t1),
            interpolate(x_lower[2], x_upper[2], t2),
            interpolate(x_lower[3], x_upper[3], t3),
        ];
        let true_output = true_group_norm(
            &x_sample,
            ny.as_slice().unwrap(),
            beta.as_slice().unwrap(),
            eps,
            num_channels,
            num_groups,
        );

        let point = BoundedTensor::new(
            Array1::from_vec(x_sample.clone()).into_dyn(),
            Array1::from_vec(x_sample).into_dyn(),
        ).unwrap();
        let result_ibp = bounds.concretize_sound(&point).unwrap();

        for j in 0..n {
            let lower = f64::from(result_ibp.lower().as_slice().unwrap()[j]);
            let upper = f64::from(result_ibp.upper().as_slice().unwrap()[j]);
            prop_assert!(
                lower <= true_output[j] + 1e-2,
                "dim {}: lower {} > true {} + tol", j, lower, true_output[j]
            );
            prop_assert!(
                upper >= true_output[j] - 1e-2,
                "dim {}: upper {} < true {} - tol", j, upper, true_output[j]
            );
        }
    }
}
