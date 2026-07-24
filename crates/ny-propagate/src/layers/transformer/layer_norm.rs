// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{checked_dim_product, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::{nan_propagating_max, nan_propagating_min};

/// Bound propagation through LayerNorm.
///
/// LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * ny + beta
///
/// The division by variance makes this challenging because:
/// 1. Variance depends on all inputs
/// 2. Division can amplify small denominators
///
/// We use a conservative interval arithmetic approach.
fn layer_norm_bounds(
    input: &BoundedTensor,
    ny: &ArrayD<f32>,
    beta: &ArrayD<f32>,
    eps: f32,
    normalized_shape: &[usize],
) -> Result<BoundedTensor> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(NyError::InvalidSpec(format!(
            "LayerNorm epsilon must be finite and non-negative, got {eps}"
        )));
    }
    // Clamp eps to prevent division-by-zero when all inputs are identical (var=0).
    // With eps=0, std_lower=0 and division produces Inf/NaN.
    let eps = eps.max(1e-12);

    let input_shape = input.lower().shape().to_vec();
    if normalized_shape.is_empty() || normalized_shape.len() > input_shape.len() {
        return Err(NyError::InvalidSpec(
            "normalized_shape must be non-empty and no larger than input rank".to_string(),
        ));
    }
    let norm_axes_start = input_shape.len() - normalized_shape.len();
    if input_shape[norm_axes_start..] != normalized_shape[..] {
        return Err(NyError::shape_mismatch(
            input_shape[norm_axes_start..].to_vec(),
            normalized_shape.to_vec(),
        ));
    }

    let norm_size: usize = checked_dim_product(normalized_shape, "LayerNorm normalized shape")?;
    let leading_size: usize = checked_dim_product(
        &input_shape[..norm_axes_start],
        "LayerNorm leading dimensions",
    )?;
    let n = norm_size as f32;

    let broadcast_param = |param: &ArrayD<f32>| -> Result<ArrayD<f32>> {
        if param.shape() == input_shape.as_slice() {
            return Ok(param.to_owned());
        }
        let matches_norm =
            param.shape() == normalized_shape || (param.ndim() == 1 && param.len() == norm_size);
        if !matches_norm {
            return Err(NyError::shape_mismatch(
                input_shape.clone(),
                param.shape().to_vec(),
            ));
        }
        let mut data = Vec::with_capacity(leading_size * norm_size);
        for _ in 0..leading_size {
            data.extend(param.iter());
        }
        ArrayD::from_shape_vec(IxDyn(&input_shape), data)
            .map_err(|e| NyError::InvalidSpec(e.to_string()))
    };

    let ny = broadcast_param(ny)?;
    let beta = broadcast_param(beta)?;

    let lower = input
        .lower()
        .view()
        .into_shape_with_order((leading_size, norm_size))
        .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
    let upper = input
        .upper()
        .view()
        .into_shape_with_order((leading_size, norm_size))
        .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
    let ny = ny
        .view()
        .into_shape_with_order((leading_size, norm_size))
        .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
    let beta = beta
        .view()
        .into_shape_with_order((leading_size, norm_size))
        .map_err(|e| NyError::InvalidSpec(e.to_string()))?;

    let mut output_lower = Array2::<f32>::zeros((leading_size, norm_size));
    let mut output_upper = Array2::<f32>::zeros((leading_size, norm_size));

    let max_norm = if norm_size > 1 {
        ((norm_size as f32) - 1.0).sqrt()
    } else {
        0.0
    };

    for row in 0..leading_size {
        let lower_row = lower.row(row);
        let upper_row = upper.row(row);

        let mean_lower = lower_row.sum() / n;
        let mean_upper = upper_row.sum() / n;

        let mut var_upper_sum = 0.0_f32;
        for i in 0..norm_size {
            let l = lower_row[i];
            let u = upper_row[i];
            // NaN-propagating max: if l or u is NaN (from upstream), NaN must
            // surface rather than silently picking the non-NaN operand (#2635).
            let from_lower = nan_propagating_max((l - mean_upper).abs(), (l - mean_lower).abs());
            let from_upper = nan_propagating_max((u - mean_upper).abs(), (u - mean_lower).abs());
            let dev = nan_propagating_max(from_lower, from_upper);
            var_upper_sum += dev * dev;
        }
        let var_upper = var_upper_sum / n;
        let std_lower = eps.sqrt();
        let std_upper = (var_upper + eps).sqrt();

        for i in 0..norm_size {
            let l = lower_row[i];
            let u = upper_row[i];
            let num_lower = l - mean_upper;
            let num_upper = u - mean_lower;
            let candidates = [
                num_lower / std_lower,
                num_lower / std_upper,
                num_upper / std_lower,
                num_upper / std_upper,
            ];
            // NaN-propagating fold: division by std near zero can produce NaN — see #2577.
            let norm_lower = candidates
                .iter()
                .cloned()
                .fold(f32::INFINITY, nan_propagating_min);
            let norm_upper = candidates
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, nan_propagating_max);
            // LayerNorm outputs are bounded by sqrt(n-1) for n elements since sum devs = 0.
            // NaN-propagating clamp: if the fold above produced NaN (from division by
            // near-zero std), the clamp must not silently replace it with max_norm (#2635).
            let norm_lower = nan_propagating_max(norm_lower, -max_norm);
            let norm_upper = nan_propagating_min(norm_upper, max_norm);

            let g = ny[[row, i]];
            let b = beta[[row, i]];
            if g >= 0.0 {
                output_lower[[row, i]] = norm_lower * g + b;
                output_upper[[row, i]] = norm_upper * g + b;
            } else {
                output_lower[[row, i]] = norm_upper * g + b;
                output_upper[[row, i]] = norm_lower * g + b;
            }
        }
    }

    let output_lower = output_lower
        .into_shape_with_order(input_shape.clone())
        .map_err(|e| NyError::InvalidSpec(e.to_string()))?
        .into_dyn();
    let output_upper = output_upper
        .into_shape_with_order(input_shape)
        .map_err(|e| NyError::InvalidSpec(e.to_string()))?
        .into_dyn();

    BoundedTensor::new(output_lower, output_upper)
}

#[cfg(test)]
mod tests {
    use super::layer_norm_bounds;
    use ndarray::{arr1, Array1};
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;

    #[test]
    fn layer_norm_bounds_rejects_non_finite_eps() {
        let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("input shape should be valid");
        let ny = arr1(&[1.0f32]).into_dyn();
        let beta = arr1(&[0.0f32]).into_dyn();

        let err = layer_norm_bounds(&input, &ny, &beta, f32::NAN, &[1])
            .expect_err("NaN eps should be rejected");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    #[test]
    fn layer_norm_bounds_rejects_negative_eps() {
        let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("input shape should be valid");
        let ny = arr1(&[1.0f32]).into_dyn();
        let beta = arr1(&[0.0f32]).into_dyn();

        let err = layer_norm_bounds(&input, &ny, &beta, -1e-5, &[1])
            .expect_err("negative eps should be rejected");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    /// Compute true LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * ny + beta.
    fn true_layer_norm(x: &[f32], ny: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let std = (var + eps).sqrt();
        x.iter()
            .zip(ny.iter().zip(beta.iter()))
            .map(|(&xi, (&gi, &bi))| (xi - mean) / std * gi + bi)
            .collect()
    }

    /// Enumerate all 2^n corners of the interval [lower, upper] and verify that
    /// the computed IBP bounds contain the true LayerNorm at every corner.
    fn assert_layer_norm_soundness_corners(
        lower: &[f32],
        upper: &[f32],
        ny: &[f32],
        beta: &[f32],
        eps: f32,
    ) {
        let n = lower.len();
        assert_eq!(n, upper.len());
        assert_eq!(n, ny.len());
        assert_eq!(n, beta.len());

        let input = BoundedTensor::new(
            Array1::from_vec(lower.to_vec()).into_dyn(),
            Array1::from_vec(upper.to_vec()).into_dyn(),
        )
        .expect("valid input bounds");
        let ny_arr = Array1::from_vec(ny.to_vec()).into_dyn();
        let beta_arr = Array1::from_vec(beta.to_vec()).into_dyn();

        let result =
            layer_norm_bounds(&input, &ny_arr, &beta_arr, eps, &[n]).expect("should succeed");

        let out_lower = result.lower().as_slice().expect("contiguous");
        let out_upper = result.upper().as_slice().expect("contiguous");

        // Enumerate all 2^n corners
        for mask in 0..(1u32 << n) {
            let corner: Vec<f32> = (0..n)
                .map(|i| {
                    if mask & (1 << i) == 0 {
                        lower[i]
                    } else {
                        upper[i]
                    }
                })
                .collect();
            let true_out = true_layer_norm(&corner, ny, beta, eps);
            for (j, &tv) in true_out.iter().enumerate() {
                assert!(
                    out_lower[j] <= tv + 1e-5,
                    "Corner {mask:#b}: lower bound {:.6} > true output {:.6} at index {j}",
                    out_lower[j],
                    tv
                );
                assert!(
                    out_upper[j] >= tv - 1e-5,
                    "Corner {mask:#b}: upper bound {:.6} < true output {:.6} at index {j}",
                    out_upper[j],
                    tv
                );
            }
        }
    }

    /// Soundness: 3-element LayerNorm with ny=1, beta=0 (identity affine).
    /// Enumerates all 8 corners and verifies IBP bounds contain all true outputs.
    #[test]
    fn layer_norm_ibp_soundness_identity_affine_3elem() {
        assert_layer_norm_soundness_corners(
            &[1.0, 2.0, 3.0],
            &[1.5, 2.5, 3.5],
            &[1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            1e-5,
        );
    }

    /// Soundness: 4-element LayerNorm with mixed ny signs.
    /// Negative ny flips the bound direction — tests the sign-aware branch.
    /// Enumerates all 16 corners.
    #[test]
    fn layer_norm_ibp_soundness_mixed_ny_4elem() {
        assert_layer_norm_soundness_corners(
            &[0.0, -1.0, 2.0, -0.5],
            &[1.0, 0.0, 3.0, 0.5],
            &[1.0, -1.0, 0.5, -2.0],
            &[0.0, 1.0, -0.5, 0.0],
            1e-5,
        );
    }

    /// Soundness: all elements have the same interval — variance approaches zero.
    /// Tests the std_lower = sqrt(eps) clamping path.
    #[test]
    fn layer_norm_ibp_soundness_near_zero_variance() {
        // When all elements are in [1.0, 1.0], the output should be ny * 0 + beta.
        // When intervals are very tight, variance is near zero → std_lower = sqrt(eps).
        assert_layer_norm_soundness_corners(
            &[1.0, 1.0, 1.0],
            &[1.01, 1.01, 1.01],
            &[1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            1e-5,
        );
    }

    /// Soundness: wide intervals that stress the max_norm = sqrt(n-1) clamping.
    /// With n=3 elements and wide ranges, the normalized values can approach ±sqrt(2).
    #[test]
    fn layer_norm_ibp_soundness_wide_intervals() {
        assert_layer_norm_soundness_corners(
            &[-10.0, -5.0, 0.0],
            &[0.0, 5.0, 10.0],
            &[1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            1e-5,
        );
    }

    /// Soundness: single-element LayerNorm (n=1).
    /// LayerNorm of a single element is always (x - x) / std * ny + beta = beta.
    /// max_norm = 0 for n=1.
    #[test]
    fn layer_norm_ibp_soundness_single_element() {
        assert_layer_norm_soundness_corners(&[5.0], &[10.0], &[2.0], &[3.0], 1e-5);
    }

    /// Soundness: asymmetric intervals where some elements are tight and others wide.
    #[test]
    fn layer_norm_ibp_soundness_asymmetric_intervals() {
        assert_layer_norm_soundness_corners(
            &[0.0, 4.99, -100.0],
            &[0.01, 5.0, 100.0],
            &[1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            1e-5,
        );
    }
}
