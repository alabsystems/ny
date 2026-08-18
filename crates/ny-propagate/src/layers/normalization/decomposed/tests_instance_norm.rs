// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for decomposed InstanceNorm1d CROWN backward propagation.
//!
//! Verifies both the flat adapter (`decomposed_instance_norm_crown_backward`,
//! which delegates to grouped centered-normalization with `num_groups == num_channels`)
//! and the channel-batched adapter (`decomposed_instance_norm_crown_backward_channel_batched`,
//! which processes each batch position independently).
//!
//! Part of #4209.

use super::instance_norm::{
    decomposed_instance_norm_crown_backward,
    decomposed_instance_norm_crown_backward_channel_batched,
};
use super::tests_support::{constant_batched_bounds, interpolate};
use ndarray::{arr1, Array1, Array2};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

/// Compute true InstanceNorm output for a single channel.
///
/// InstanceNorm(x)[c, t] = ny[c] * (x[c, t] - mean_c) / sqrt(var_c + eps) + beta[c]
/// where mean_c and var_c are computed over the time dimension for channel c.
fn true_instance_norm(
    x: &[f32],
    ny: &[f32],
    beta: &[f32],
    eps: f32,
    num_channels: usize,
) -> Vec<f64> {
    let total = x.len();
    let time_len = total / num_channels;
    let mut output = vec![0.0_f64; total];
    for c in 0..num_channels {
        let start = c * time_len;
        let end = start + time_len;
        let channel_x = &x[start..end];

        let n = time_len as f64;
        let mean = channel_x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = channel_x
            .iter()
            .map(|&v| {
                let d = f64::from(v) - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = (var + f64::from(eps)).sqrt();

        for t in 0..time_len {
            output[start + t] =
                f64::from(ny[c]) * (f64::from(channel_x[t]) - mean) / std + f64::from(beta[c]);
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

// --- Flat adapter tests (delegates to grouped centered) ---

#[ntest::timeout(10000)]
#[test]
fn test_instance_norm_flat_identity_upstream_returns_ok() -> Result<()> {
    let num_channels = 2;
    let time_len = 3;
    let n = num_channels * time_len;
    let ny = arr1(&[1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5, 0.8, 1.2, 1.6]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5, 1.8, 2.2, 2.6]).into_dyn(),
    )?;
    let upstream = identity_upstream_flat(n);

    let result = decomposed_instance_norm_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
    )?;

    assert_eq!(
        result.validation.total_rows, n,
        "expected total_rows == {n}, got {}",
        result.validation.total_rows
    );
    assert!(
        result.bounds.lower_a().shape()[result.bounds.lower_a().ndim() - 1] == n,
        "A matrix last dimension should be {n}"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_instance_norm_flat_soundness_at_center() -> Result<()> {
    let num_channels = 2;
    let time_len = 3;
    let n = num_channels * time_len;
    let ny = arr1(&[1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0]);
    let eps = 1e-5;
    let x_lower = [0.5_f32, 1.0, 1.5, 0.8, 1.2, 1.6];
    let x_upper = [1.5_f32, 2.0, 2.5, 1.8, 2.2, 2.6];
    let x_ibp = BoundedTensor::new(arr1(&x_lower).into_dyn(), arr1(&x_upper).into_dyn())?;
    let upstream = identity_upstream_flat(n);

    let result = decomposed_instance_norm_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
    )?;
    let bounds = &result.bounds;

    let x_center: Vec<f32> = x_lower
        .iter()
        .zip(x_upper.iter())
        .map(|(&l, &u)| f32::midpoint(l, u))
        .collect();
    let true_output = true_instance_norm(
        &x_center,
        ny.as_slice().unwrap(),
        beta.as_slice().unwrap(),
        eps,
        num_channels,
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
fn test_instance_norm_flat_soundness_at_corners() -> Result<()> {
    let num_channels = 2;
    let time_len = 2;
    let n = num_channels * time_len;
    let ny = arr1(&[1.5, 0.8]);
    let beta = arr1(&[0.1, -0.2]);
    let eps = 1e-5;
    let x_lower = [0.5_f32, 1.0, 0.8, 1.2];
    let x_upper = [1.5_f32, 2.0, 1.8, 2.2];
    let x_ibp = BoundedTensor::new(arr1(&x_lower).into_dyn(), arr1(&x_upper).into_dyn())?;
    let upstream = identity_upstream_flat(n);

    let result = decomposed_instance_norm_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
    )?;
    let bounds = &result.bounds;
    let ny_slice = ny.as_slice().unwrap();
    let beta_slice = beta.as_slice().unwrap();

    // Check all 16 corners of 4D input box
    for &x0 in &[x_lower[0], x_upper[0]] {
        for &x1 in &[x_lower[1], x_upper[1]] {
            for &x2 in &[x_lower[2], x_upper[2]] {
                for &x3 in &[x_lower[3], x_upper[3]] {
                    let x = vec![x0, x1, x2, x3];
                    let true_output =
                        true_instance_norm(&x, ny_slice, beta_slice, eps, num_channels);
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
fn test_instance_norm_flat_forward_mode_returns_ok() -> Result<()> {
    let num_channels = 2;
    let time_len = 3;
    let n = num_channels * time_len;
    let ny = arr1(&[1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5, 0.8, 1.2, 1.6]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5, 1.8, 2.2, 2.6]).into_dyn(),
    )?;
    let upstream = identity_upstream_flat(n);

    let result = decomposed_instance_norm_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        true, // forward_mode
        num_channels,
    )?;

    assert_eq!(
        result.validation.total_rows, n,
        "forward mode should produce same total_rows"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_instance_norm_flat_ny_beta_mismatch_error() {
    let num_channels = 2;
    let time_len = 3;
    let n = num_channels * time_len;
    let ny = arr1(&[1.0]); // wrong size: 1 instead of 2
    let beta = arr1(&[0.0, 0.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5, 0.8, 1.2, 1.6]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5, 1.8, 2.2, 2.6]).into_dyn(),
    )
    .unwrap();
    let upstream = identity_upstream_flat(n);

    let result = decomposed_instance_norm_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
    );
    assert!(
        result.is_err(),
        "should error on ny/beta size mismatch with num_channels"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_instance_norm_flat_fallback_count_nonnegative() -> Result<()> {
    let num_channels = 2;
    let time_len = 3;
    let n = num_channels * time_len;
    let ny = arr1(&[1.0, 0.5]);
    let beta = arr1(&[0.0, 0.1]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.2, 0.8, 1.2, 0.4, 1.0, 1.6]).into_dyn(),
        arr1(&[0.6, 1.2, 1.6, 0.8, 1.4, 2.0]).into_dyn(),
    )?;
    let upstream = identity_upstream_flat(n);

    let result = decomposed_instance_norm_crown_backward(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
    )?;
    assert!(
        result.validation.fallback_rows <= result.validation.total_rows,
        "fallback_rows {} should not exceed total_rows {}",
        result.validation.fallback_rows,
        result.validation.total_rows
    );
    Ok(())
}

// --- Channel-batched adapter tests ---

/// Create identity upstream bounds for channel-batched mode.
///
/// A has shape [C, out_dim, T] (3D) and b has shape [C, out_dim] (2D).
/// The production code reshapes these internally to [total_batch, out_dim, T]
/// and [total_batch, out_dim] respectively. Identity means each output row
/// copies one input element.
fn identity_upstream_channel_batched(
    num_channels: usize,
    time_len: usize,
) -> crate::BatchedLinearBounds {
    use ndarray::Array3;

    // out_dim = time_len for identity mapping
    let out_dim = time_len;

    // A has shape [C, out_dim, T], identity per-channel
    let mut a = Array3::<f32>::zeros((num_channels, out_dim, time_len));
    for c in 0..num_channels {
        for t in 0..time_len {
            a[[c, t, t]] = 1.0;
        }
    }
    // b has shape [C, out_dim] to match A.shape()[..A.ndim()-1]
    let b = Array2::<f32>::zeros((num_channels, out_dim));
    crate::BatchedLinearBounds::new(
        a.clone().into_dyn(),
        b.clone().into_dyn(),
        a.into_dyn(),
        b.into_dyn(),
        vec![num_channels, time_len],
        vec![num_channels, out_dim],
    )
    .expect("channel-batched identity bounds should be valid")
}

#[ntest::timeout(10000)]
#[test]
fn test_instance_norm_channel_batched_returns_ok() -> Result<()> {
    let num_channels = 2;
    let time_len = 3;
    let ny = arr1(&[1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0]);
    let eps = 1e-5;
    // Input shape [C, T] = [2, 3]
    let x_ibp = BoundedTensor::new(
        Array2::from_shape_vec((2, 3), vec![0.5, 1.0, 1.5, 0.8, 1.2, 1.6])
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 3), vec![1.5, 2.0, 2.5, 1.8, 2.2, 2.6])
            .unwrap()
            .into_dyn(),
    )?;
    let upstream = identity_upstream_channel_batched(num_channels, time_len);

    let result = decomposed_instance_norm_crown_backward_channel_batched(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
    )?;

    let total_rows = num_channels * time_len;
    assert_eq!(
        result.validation.total_rows, total_rows,
        "expected total_rows == {total_rows}"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_instance_norm_channel_batched_soundness_at_center() -> Result<()> {
    let num_channels = 2;
    let time_len = 3;
    let ny = arr1(&[1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0]);
    let eps = 1e-5;
    let x_lower_flat = [0.5_f32, 1.0, 1.5, 0.8, 1.2, 1.6];
    let x_upper_flat = [1.5_f32, 2.0, 2.5, 1.8, 2.2, 2.6];
    let x_ibp = BoundedTensor::new(
        Array2::from_shape_vec((2, 3), x_lower_flat.to_vec())
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 3), x_upper_flat.to_vec())
            .unwrap()
            .into_dyn(),
    )?;
    let upstream = identity_upstream_channel_batched(num_channels, time_len);

    let result = decomposed_instance_norm_crown_backward_channel_batched(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
    )?;
    let bounds = &result.bounds;

    // Evaluate at center
    let x_center: Vec<f32> = x_lower_flat
        .iter()
        .zip(x_upper_flat.iter())
        .map(|(&l, &u)| f32::midpoint(l, u))
        .collect();
    let true_output = true_instance_norm(
        &x_center,
        ny.as_slice().unwrap(),
        beta.as_slice().unwrap(),
        eps,
        num_channels,
    );

    let point = BoundedTensor::new(
        Array2::from_shape_vec((2, 3), x_center.clone())
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 3), x_center).unwrap().into_dyn(),
    )?;
    let result_ibp = bounds.concretize_sound(&point)?;
    let result_flat = result_ibp.lower().as_slice().unwrap();
    let result_flat_u = result_ibp.upper().as_slice().unwrap();

    let n = num_channels * time_len;
    for j in 0..n {
        let lower = f64::from(result_flat[j]);
        let upper = f64::from(result_flat_u[j]);
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
fn test_instance_norm_channel_batched_zero_channels_error() {
    let ny = arr1::<f32>(&[]);
    let beta = arr1::<f32>(&[]);
    let eps = 1e-5;
    // Construct minimal upstream bounds with shape [1, 1, 1]
    let upstream = crate::BatchedLinearBounds::new(
        ndarray::Array3::<f32>::zeros((1, 1, 1)).into_dyn(),
        Array2::<f32>::zeros((1, 1)).into_dyn(),
        ndarray::Array3::<f32>::zeros((1, 1, 1)).into_dyn(),
        Array2::<f32>::zeros((1, 1)).into_dyn(),
        vec![1, 1],
        vec![1, 1],
    )
    .unwrap();
    let x_ibp = BoundedTensor::new(
        Array2::<f32>::zeros((1, 1)).into_dyn(),
        Array2::<f32>::zeros((1, 1)).into_dyn(),
    )
    .unwrap();

    let result = decomposed_instance_norm_crown_backward_channel_batched(
        &upstream, &ny, &beta, eps, &x_ibp, false, 0, // zero channels
    );
    assert!(result.is_err(), "should error on num_channels == 0");
}

#[ntest::timeout(10000)]
#[test]
fn test_instance_norm_channel_batched_ny_mismatch_error() {
    let num_channels = 2;
    let time_len = 3;
    let ny = arr1(&[1.0]); // wrong size: 1 instead of 2
    let beta = arr1(&[0.0, 0.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        Array2::from_shape_vec((2, 3), vec![0.5, 1.0, 1.5, 0.8, 1.2, 1.6])
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 3), vec![1.5, 2.0, 2.5, 1.8, 2.2, 2.6])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();
    let upstream = identity_upstream_channel_batched(num_channels, time_len);

    let result = decomposed_instance_norm_crown_backward_channel_batched(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
    );
    assert!(
        result.is_err(),
        "should error on ny size mismatch with num_channels"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_instance_norm_channel_batched_fallback_count_valid() -> Result<()> {
    let num_channels = 2;
    let time_len = 3;
    let ny = arr1(&[1.0, 0.5]);
    let beta = arr1(&[0.0, 0.1]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        Array2::from_shape_vec((2, 3), vec![0.2, 0.8, 1.2, 0.4, 1.0, 1.6])
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 3), vec![0.6, 1.2, 1.6, 0.8, 1.4, 2.0])
            .unwrap()
            .into_dyn(),
    )?;
    let upstream = identity_upstream_channel_batched(num_channels, time_len);

    let result = decomposed_instance_norm_crown_backward_channel_batched(
        &upstream,
        &ny,
        &beta,
        eps,
        &x_ibp,
        false,
        num_channels,
    )?;
    assert!(
        result.validation.fallback_rows <= result.validation.total_rows,
        "fallback_rows {} should not exceed total_rows {}",
        result.validation.fallback_rows,
        result.validation.total_rows
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(64) })]

    #[test]
    fn proptest_instance_norm_flat_contains_true_output(
        // 2 channels, 2 time steps each
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
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        t0 in 0.0f32..1.0,
        t1 in 0.0f32..1.0,
        t2 in 0.0f32..1.0,
        t3 in 0.0f32..1.0,
    ) {
        let num_channels = 2;
        let time_len = 2;
        let n = num_channels * time_len;
        let ny = arr1(&[g0, g1]);
        let beta = arr1(&[b0, b1]);
        let eps = 1e-5_f32;
        let x_lower = [x0_l, x1_l, x2_l, x3_l];
        let x_upper = [x0_l + x0_w, x1_l + x1_w, x2_l + x2_w, x3_l + x3_w];
        let x_ibp = BoundedTensor::new(
            arr1(&x_lower).into_dyn(),
            arr1(&x_upper).into_dyn(),
        ).unwrap();
        let upstream = identity_upstream_flat(n);

        let result = decomposed_instance_norm_crown_backward(
            &upstream, &ny, &beta, eps, &x_ibp, false, num_channels,
        ).map_err(|error| TestCaseError::fail(
            format!("decomposed instance norm must accept the generated finite domain: {error}")
        ))?;
        let bounds = &result.bounds;

        let x_sample = vec![
            interpolate(x_lower[0], x_upper[0], t0),
            interpolate(x_lower[1], x_upper[1], t1),
            interpolate(x_lower[2], x_upper[2], t2),
            interpolate(x_lower[3], x_upper[3], t3),
        ];
        let true_output = true_instance_norm(
            &x_sample,
            ny.as_slice().unwrap(),
            beta.as_slice().unwrap(),
            eps,
            num_channels,
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
