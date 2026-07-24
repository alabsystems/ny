// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for decomposed LayerNorm CROWN backward propagation.
//!
//! Verifies `decomposed_norm_crown_backward` from `layernorm.rs` produces
//! sound linear bounds that always contain the true LayerNorm output.
//!
//! Part of #4209.

use super::layernorm::decomposed_norm_crown_backward;
use super::tests_support::{constant_batched_bounds, interpolate};
use ndarray::{arr1, Array1, Array2};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

/// Compute true LayerNorm output for a single sample.
///
/// LayerNorm(x)[i] = ny[i] * (x[i] - mean(x)) / sqrt(var(x) + eps) + beta[i]
fn true_layernorm(x: &[f32], ny: &[f32], beta: &[f32], eps: f32) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var = x
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = (var + f64::from(eps)).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &v)| f64::from(ny[i]) * (f64::from(v) - mean) / std + f64::from(beta[i]))
        .collect()
}

/// Create identity upstream bounds: A = eye(n), b = 0.
/// When composed with LayerNorm, the result bounds directly bound LayerNorm(x).
fn identity_upstream(n: usize) -> crate::BatchedLinearBounds {
    let eye = Array2::eye(n);
    let zeros = Array1::zeros(n);
    constant_batched_bounds(eye.clone(), zeros.clone(), eye, zeros, n)
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_identity_upstream_backward_mode_returns_ok() -> Result<()> {
    let n = 3;
    let ny = arr1(&[1.0, 1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0, 0.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5]).into_dyn(),
    )?;
    let upstream = identity_upstream(n);

    let result = decomposed_norm_crown_backward(&upstream, &ny, &beta, eps, &x_ibp, false)?;

    assert_eq!(result.validation.total_rows, n);
    assert_eq!(
        result.bounds.lower_a().shape()[result.bounds.lower_a().ndim() - 1],
        n
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_identity_upstream_forward_mode_returns_ok() -> Result<()> {
    let n = 3;
    let ny = arr1(&[1.0, 1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0, 0.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5]).into_dyn(),
    )?;
    let upstream = identity_upstream(n);

    let result = decomposed_norm_crown_backward(&upstream, &ny, &beta, eps, &x_ibp, true)?;

    assert_eq!(result.validation.total_rows, n);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_soundness_at_interval_center() -> Result<()> {
    let n = 3;
    let ny = arr1(&[1.0, 1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0, 0.0]);
    let eps = 1e-5;
    let x_lower = [0.5_f32, 1.0, 1.5];
    let x_upper = [1.5_f32, 2.0, 2.5];
    let x_ibp = BoundedTensor::new(arr1(&x_lower).into_dyn(), arr1(&x_upper).into_dyn())?;
    let upstream = identity_upstream(n);

    let result = decomposed_norm_crown_backward(&upstream, &ny, &beta, eps, &x_ibp, false)?;
    let bounds = &result.bounds;

    // Evaluate at the center of the input interval
    let x_center: Vec<f32> = x_lower
        .iter()
        .zip(x_upper.iter())
        .map(|(&l, &u)| f32::midpoint(l, u))
        .collect();
    let true_output = true_layernorm(
        &x_center,
        ny.as_slice().unwrap(),
        beta.as_slice().unwrap(),
        eps,
    );

    // Concretize bounds at x_center
    let result_ibp = bounds.concretize_sound(&BoundedTensor::new(
        Array1::from_vec(x_center.clone()).into_dyn(),
        Array1::from_vec(x_center).into_dyn(),
    )?)?;

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
fn test_layernorm_soundness_at_interval_corners() -> Result<()> {
    let n = 2;
    let ny = arr1(&[1.5, 0.8]);
    let beta = arr1(&[0.1, -0.2]);
    let eps = 1e-5;
    let x_lower = [0.5_f32, 1.0];
    let x_upper = [1.5_f32, 2.0];
    let x_ibp = BoundedTensor::new(arr1(&x_lower).into_dyn(), arr1(&x_upper).into_dyn())?;
    let upstream = identity_upstream(n);

    let result = decomposed_norm_crown_backward(&upstream, &ny, &beta, eps, &x_ibp, false)?;
    let bounds = &result.bounds;

    // Check all 4 corners of 2D input box
    let ny_slice = ny.as_slice().unwrap();
    let beta_slice = beta.as_slice().unwrap();
    for &x0 in &[x_lower[0], x_upper[0]] {
        for &x1 in &[x_lower[1], x_upper[1]] {
            let x = vec![x0, x1];
            let true_output = true_layernorm(&x, ny_slice, beta_slice, eps);
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
                    "corner x=({x0},{x1}) dim {j}: lower {lower} > true {}",
                    true_output[j]
                );
                assert!(
                    upper >= true_output[j] - 1e-3,
                    "corner x=({x0},{x1}) dim {j}: upper {upper} < true {}",
                    true_output[j]
                );
            }
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_dimension_zero_error() {
    let ny = arr1::<f32>(&[]);
    let beta = arr1::<f32>(&[]);
    let eps = 1e-5;
    // Cannot construct valid BatchedLinearBounds with n=0,
    // so we test through the function's error path with mismatched shapes.
    let upstream = constant_batched_bounds(
        Array2::zeros((1, 1)),
        arr1(&[0.0]),
        Array2::zeros((1, 1)),
        arr1(&[0.0]),
        1,
    );
    let x_ibp =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[0.0_f32]).into_dyn()).unwrap();

    let result = decomposed_norm_crown_backward(&upstream, &ny, &beta, eps, &x_ibp, false);
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_ny_beta_mismatch_error() {
    let n = 3;
    let ny = arr1(&[1.0, 1.0]); // wrong size
    let beta = arr1(&[0.0, 0.0, 0.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5]).into_dyn(),
    )
    .unwrap();
    let upstream = identity_upstream(n);

    let result = decomposed_norm_crown_backward(&upstream, &ny, &beta, eps, &x_ibp, false);
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_fallback_count_nonnegative() -> Result<()> {
    let n = 4;
    let ny = arr1(&[1.0, 0.5, 2.0, 0.75]);
    let beta = arr1(&[0.0, 0.1, -0.1, 0.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.2, 0.8, 1.2, 1.8]).into_dyn(),
        arr1(&[0.6, 1.2, 1.6, 2.2]).into_dyn(),
    )?;
    let upstream = identity_upstream(n);

    let result = decomposed_norm_crown_backward(&upstream, &ny, &beta, eps, &x_ibp, false)?;
    assert!(result.validation.fallback_rows <= result.validation.total_rows);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(64) })]

    #[test]
    fn proptest_layernorm_decomposed_contains_true_output(
        x0_l in 0.2f32..2.0,
        x0_w in 0.05f32..0.8,
        x1_l in 0.2f32..2.0,
        x1_w in 0.05f32..0.8,
        x2_l in 0.2f32..2.0,
        x2_w in 0.05f32..0.8,
        // Randomized ny/beta to stress-test affine interaction with CROWN relaxation
        g0 in 0.3f32..2.5,
        g1 in 0.3f32..2.5,
        g2 in 0.3f32..2.5,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        b2 in -1.0f32..1.0,
        t0 in 0.0f32..1.0,
        t1 in 0.0f32..1.0,
        t2 in 0.0f32..1.0,
    ) {
        let n = 3;
        let ny = arr1(&[g0, g1, g2]);
        let beta = arr1(&[b0, b1, b2]);
        let eps = 1e-5_f32;
        let x_lower = [x0_l, x1_l, x2_l];
        let x_upper = [x0_l + x0_w, x1_l + x1_w, x2_l + x2_w];
        let x_ibp = BoundedTensor::new(
            arr1(&x_lower).into_dyn(),
            arr1(&x_upper).into_dyn(),
        ).unwrap();
        let upstream = identity_upstream(n);

        let result = decomposed_norm_crown_backward(
            &upstream, &ny, &beta, eps, &x_ibp, false
        );
        // Use prop_assume! so proptest generates replacement cases for numerically
        // ill-conditioned inputs, rather than silently counting errors as passes.
        // Without this, a regression that always returns Err passes the test vacuously.
        prop_assume!(result.is_ok(), "decomposed layernorm returned error: {:?}", result.err());
        let result = result.unwrap();
        let bounds = &result.bounds;

        let x_sample = vec![
            interpolate(x_lower[0], x_upper[0], t0),
            interpolate(x_lower[1], x_upper[1], t1),
            interpolate(x_lower[2], x_upper[2], t2),
        ];
        let true_output = true_layernorm(
            &x_sample,
            ny.as_slice().unwrap(),
            beta.as_slice().unwrap(),
            eps,
        );

        let point = BoundedTensor::new(
            Array1::from_vec(x_sample.clone()).into_dyn(),
            Array1::from_vec(x_sample).into_dyn(),
        ).unwrap();
        let result_ibp = bounds.concretize_sound(&point).unwrap();

        for j in 0..n {
            let lower = f64::from(result_ibp.lower().as_slice().unwrap()[j]);
            let upper = f64::from(result_ibp.upper().as_slice().unwrap()[j]);
            // Tolerance accounts for McCormick relaxation looseness
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
