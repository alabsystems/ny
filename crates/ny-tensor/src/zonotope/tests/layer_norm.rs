// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{arr0, arr1, arr2};

#[test]
fn test_layer_norm_affine_concrete() {
    // Test LayerNorm on a concrete zonotope (no error terms)
    // Input: [[1, 2, 3], [4, 5, 6]] (2 positions, 3 features)
    let values = arr2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let z = ZonotopeTensor::concrete(values.into_dyn());

    // LayerNorm parameters: ny=1, beta=0, eps=1e-5
    let ny = arr1(&[1.0_f32, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0]);
    let eps = 1e-5;

    let result = z.layer_norm_affine(&ny, &beta, eps).unwrap();

    // For concrete input, output center should match standard LayerNorm
    let center = result.center();

    // Row 0: mean=2, var=2/3, std=sqrt(2/3)≈0.8165
    // Normalized: [-1.22, 0, 1.22] (approx)
    // center[0] + center[2] should be ~0 (symmetric around zero)
    assert!((center[[0, 0]] + center[[0, 2]]).abs() < 0.01);
    assert!(center[[0, 1]].abs() < 0.01); // Mean feature should be ~0

    // Row 1: mean=5, var=2/3, std=sqrt(2/3)≈0.8165
    // Same normalization pattern
    assert!((center[[1, 0]] + center[[1, 2]]).abs() < 0.01);
    assert!(center[[1, 1]].abs() < 0.01);
}

#[test]
fn test_layer_norm_affine_with_error() {
    // Test that LayerNorm preserves zonotope correlations
    // Input: 2x3 zonotope with per-position error symbols
    let values = arr2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let z = ZonotopeTensor::from_input_2d(&values, 0.1);

    let ny = arr1(&[1.0_f32, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0]);
    let eps = 1e-5;

    let result = z.layer_norm_affine(&ny, &beta, eps).unwrap();

    // Should have original error terms + per-element approximation error terms (#2522)
    // prefix_size=2, dim=3, n_new_error_terms = 2*3 = 6
    assert_eq!(result.n_error_terms, z.n_error_terms + 6);
    assert_eq!(result.element_shape, z.element_shape);

    // Zonotope bounds must be finite
    let bounds = result.to_bounded_tensor().unwrap();
    assert!(
        bounds.lower().iter().all(|&v| v.is_finite()),
        "all lower bounds should be finite, got {:?}",
        bounds.lower()
    );
    assert!(
        bounds.upper().iter().all(|&v| v.is_finite()),
        "all upper bounds should be finite, got {:?}",
        bounds.upper()
    );

    // Soundness: zonotope bounds must contain the concrete center output.
    // For input [[1,2,3],[4,5,6]] with ny=1, beta=0:
    //   mean=2 (row0) / mean=5 (row1), var=2/3, std≈0.8165
    //   normalized ≈ [-1.2247, 0, 1.2247] for both rows.
    let center = result.center();
    for i in 0..2 {
        for j in 0..3 {
            let lo = bounds.lower()[[i, j]];
            let hi = bounds.upper()[[i, j]];
            let c = center[[i, j]];
            assert!(
                lo <= c && c <= hi,
                "Soundness: center[{i},{j}]={c} not in [{lo}, {hi}]"
            );
        }
    }

    // Tightness: bounds width should be modest (input eps=0.1, dim=3).
    // Trivially-wide bounds (width > 10) would indicate a broken implementation.
    for i in 0..2 {
        for j in 0..3 {
            let width = bounds.upper()[[i, j]] - bounds.lower()[[i, j]];
            assert!(
                width < 10.0,
                "Tightness: bounds[{i},{j}] width={width} unexpectedly wide"
            );
        }
    }
}

#[test]
fn test_layer_norm_affine_with_ny_beta() {
    // Test with non-trivial ny and beta
    let values = arr2(&[[0.0_f32, 0.0, 0.0]]);
    let z = ZonotopeTensor::concrete(values.into_dyn());

    let ny = arr1(&[2.0_f32, 1.0, 0.5]);
    let beta = arr1(&[1.0_f32, 2.0, 3.0]);
    let eps = 1e-5;

    let result = z.layer_norm_affine(&ny, &beta, eps).unwrap();
    let center = result.center();

    // For constant input [0,0,0]:
    // mean=0, var=0, std=sqrt(eps)
    // Each output_i = ny_i * 0 / std + beta_i = beta_i
    for d in 0..3 {
        assert!((center[[0, d]] - beta[d]).abs() < 0.1);
    }
}

// ==================== Mean-only LayerNorm tests ====================

#[test]
fn test_layer_norm_affine_mean_only_concrete_matches_exact_center() {
    let values = arr2(&[[1.0_f32, 3.0, 5.0], [2.0, 4.0, 8.0]]);
    let z = ZonotopeTensor::concrete(values.into_dyn());

    let ny = arr1(&[1.0_f32, 2.0, -1.0]);
    let beta = arr1(&[0.5_f32, -0.5, 1.0]);

    let result = z.layer_norm_affine_mean_only(&ny, &beta).unwrap();
    let center = result.center();

    assert_eq!(result.n_error_terms(), 0);
    assert_eq!(result.shape(), &[2, 3]);

    let expected = [
        [-1.5_f32, -0.5, -1.0],
        [-13.0 / 6.0, -11.0 / 6.0, -7.0 / 3.0],
    ];
    for row in 0..2 {
        for col in 0..3 {
            assert!(
                (center[[row, col]] - expected[row][col]).abs() < 1e-6,
                "mean-only LayerNorm center[{row},{col}] should be {}, got {}",
                expected[row][col],
                center[[row, col]]
            );
        }
    }
}

#[test]
fn test_layer_norm_affine_mean_only_cancels_shared_shift_exactly() {
    // Mean-only LayerNorm is translation-invariant along the normalized axis:
    // if every feature in a row shifts by the same epsilon, x - mean(x) is unchanged.
    let values = arr2(&[[1.0_f32, 3.0, 5.0], [2.0, 4.0, 6.0]]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.25);

    let ny = arr1(&[1.0_f32, -2.0, 0.5]);
    let beta = arr1(&[0.0_f32, 1.0, -1.0]);

    let result = z.layer_norm_affine_mean_only(&ny, &beta).unwrap();
    assert_eq!(result.n_error_terms(), z.n_error_terms());

    for row in 0..2 {
        for col in 0..3 {
            assert!(
                result.coeffs()[[1, row, col]].abs() < 1e-7,
                "shared shift should cancel exactly at [{row},{col}], got {}",
                result.coeffs()[[1, row, col]]
            );
        }
    }

    let bounds = result.to_bounded_tensor().unwrap();
    for (lo, hi) in bounds.lower().iter().zip(bounds.upper().iter()) {
        assert!(
            (hi - lo).abs() < 1e-7,
            "mean-only LayerNorm should remove uniform shared uncertainty, got width {}",
            hi - lo
        );
    }
}

#[test]
fn test_layer_norm_affine_mean_only_transforms_per_element_errors_exactly() {
    let values = arr2(&[[2.0_f32, 6.0]]);
    let z = ZonotopeTensor::from_input_2d(&values, 0.4);

    let ny = arr1(&[1.5_f32, -0.5]);
    let beta = arr1(&[0.25_f32, -1.0]);

    let result = z.layer_norm_affine_mean_only(&ny, &beta).unwrap();

    assert_eq!(result.n_error_terms(), z.n_error_terms());
    assert_eq!(result.shape(), &[1, 2]);

    let expected_center = [-2.75_f32, -2.0];
    let expected_err_1 = [0.3_f32, 0.1];
    let expected_err_2 = [-0.3_f32, -0.1];

    assert!((result.coeffs()[[0, 0, 0]] - expected_center[0]).abs() < 1e-6);
    assert!((result.coeffs()[[0, 0, 1]] - expected_center[1]).abs() < 1e-6);
    assert!((result.coeffs()[[1, 0, 0]] - expected_err_1[0]).abs() < 1e-6);
    assert!((result.coeffs()[[1, 0, 1]] - expected_err_1[1]).abs() < 1e-6);
    assert!((result.coeffs()[[2, 0, 0]] - expected_err_2[0]).abs() < 1e-6);
    assert!((result.coeffs()[[2, 0, 1]] - expected_err_2[1]).abs() < 1e-6);
}

#[test]
fn test_layer_norm_affine_mean_only_rejects_scalar_input() {
    let z = ZonotopeTensor::concrete(arr0(3.0_f32).into_dyn());
    let ny = arr1(&[1.0_f32]);
    let beta = arr1(&[0.0_f32]);

    let result = z.layer_norm_affine_mean_only(&ny, &beta);
    assert!(
        result.is_err(),
        "mean-only LayerNorm should reject scalar inputs without a normalized axis"
    );
}

#[test]
fn test_layer_norm_affine_mean_only_rejects_dim_mismatch() {
    let values = arr2(&[[1.0_f32, 2.0, 3.0]]).into_dyn();
    let z = ZonotopeTensor::concrete(values);

    let ny = arr1(&[1.0_f32, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0]);
    assert!(
        z.layer_norm_affine_mean_only(&ny, &beta).is_err(),
        "mean-only LayerNorm should reject ny dimension mismatch"
    );

    let ny = arr1(&[1.0_f32, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0]);
    assert!(
        z.layer_norm_affine_mean_only(&ny, &beta).is_err(),
        "mean-only LayerNorm should reject beta dimension mismatch"
    );
}

// Mutation-killing tests for layer_norm_affine()
// ============================================================

#[test]
fn test_layer_norm_affine_rejects_ny_dim_mismatch() {
    // Kills: replace || with && in line 1190
    let values = arr2(&[[1.0, 2.0, 3.0]]).into_dyn(); // dim=3
    let z = ZonotopeTensor::concrete(values);

    let ny = arr1(&[1.0, 1.0]); // dim=2 (wrong)
    let beta = arr1(&[0.0, 0.0, 0.0]); // dim=3 (correct)
    let result = z.layer_norm_affine(&ny, &beta, 1e-5);

    assert!(result.is_err(), "should reject ny with wrong dimension");
}

#[test]
fn test_layer_norm_affine_rejects_beta_dim_mismatch() {
    // Complements ny test - tests the beta part of the || condition
    let values = arr2(&[[1.0, 2.0, 3.0]]).into_dyn();
    let z = ZonotopeTensor::concrete(values);

    let ny = arr1(&[1.0, 1.0, 1.0]); // correct
    let beta = arr1(&[0.0, 0.0]); // wrong dim
    let result = z.layer_norm_affine(&ny, &beta, 1e-5);

    assert!(result.is_err(), "should reject beta with wrong dimension");
}

#[test]
fn test_layer_norm_affine_variance_computation() {
    // Kills: replace * with + in line 1227 (c * c for variance)
    // Kills: replace / with * in line 1227 (var / dim)
    // If * was +, variance would be sum of values not sum of squares
    // If / was *, would multiply by dim instead of divide

    // Use values where sum != sum of squares
    let values = arr2(&[[2.0, 4.0]]).into_dyn(); // mean=3, var=(1+1)/2=1
    let z = ZonotopeTensor::concrete(values);

    let ny = arr1(&[1.0, 1.0]); // identity scale
    let beta = arr1(&[0.0, 0.0]);
    let result = z.layer_norm_affine(&ny, &beta, 1e-5).unwrap();
    let center = result.center();

    // With variance=1, std≈1: normalized = (x - 3) / 1 = [-1, 1]
    // Output = ny * normalized + beta = [-1, 1]
    assert!(
        (center[[0, 0]] - (-1.0)).abs() < 0.01,
        "first element should be ~-1"
    );
    assert!(
        (center[[0, 1]] - 1.0).abs() < 0.01,
        "second element should be ~1"
    );
}

#[test]
fn test_layer_norm_affine_eps_addition() {
    // Kills: replace + with - in line 1228 (var + eps)
    // Kills: replace + with * in line 1228
    // Use zero-variance input where eps matters

    let values = arr2(&[[3.0, 3.0]]).into_dyn(); // constant row, var=0
    let z = ZonotopeTensor::concrete(values);

    let ny = arr1(&[1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0]);
    let eps = 1.0; // large eps so std = sqrt(0 + 1) = 1

    let result = z.layer_norm_affine(&ny, &beta, eps).unwrap();
    let center = result.center();

    // With var=0 and eps=1: std=1, centered=[0,0], output=beta=[0,0]
    // If + was -, sqrt would fail or give NaN (var - eps = -1)
    assert!(center[[0, 0]].is_finite(), "result should be finite");
    assert!((center[[0, 0]]).abs() < 0.01, "output should be ~0");
}

#[test]
fn test_layer_norm_affine_ny_division() {
    // Kills: replace / with * in line 1237 (g / std_safe)
    // Kills: replace / with % in line 1237

    let values = arr2(&[[0.0, 4.0]]).into_dyn(); // mean=2, var=4, std=2
    let z = ZonotopeTensor::concrete(values);

    let ny = arr1(&[2.0, 2.0]); // ny=2
    let beta = arr1(&[0.0, 0.0]);

    let result = z.layer_norm_affine(&ny, &beta, 0.0).unwrap();
    let center = result.center();

    // eff_gamma = ny/std = 2/2 = 1
    // centered = [-2, 2]
    // output = 1 * centered = [-2, 2]
    // If / was *, eff_gamma = 2*2 = 4, output = [-8, 8] (wrong)
    assert!(
        (center[[0, 0]] - (-2.0)).abs() < 0.01,
        "should be -2, not -8"
    );
    assert!((center[[0, 1]] - 2.0).abs() < 0.01, "should be 2, not 8");
}

#[test]
fn test_layer_norm_affine_error_term_scaling() {
    // Kills: replace * with + in line 1255 (eff_gamma[d] * coeffs_3d[...])
    // Error coefficients should be scaled by ny/std

    let values = arr2(&[[0.0, 4.0]]).into_dyn(); // std=2
    let z = ZonotopeTensor::from_input_shared(&values, 0.5); // 0.5 perturbation

    let ny = arr1(&[4.0, 4.0]); // ny=4, so eff_gamma = 4/2 = 2
    let beta = arr1(&[0.0, 0.0]);

    let result = z.layer_norm_affine(&ny, &beta, 0.0).unwrap();

    // Original error coeff was 0.5
    // After scaling by eff_gamma=2, should be ~1.0
    // If * was +, would be eff_gamma + coeff = 2 + 0.5 = 2.5 (wrong)
    let err_coeff_0 = result.coeffs[[1, 0, 0]];
    assert!(
        (err_coeff_0.abs() - 1.0).abs() < 0.1,
        "error coeff should be ~1.0 (0.5*2), got {}",
        err_coeff_0
    );
}

#[test]
fn test_layer_norm_affine_radius_accumulation() {
    // Kills: sign mutations in the per-feature radius accumulation
    // (sum of |coeffs| feeding the approximation-error enclosure)

    // Use 2D input to match layer_norm_affine's expectation
    let values = arr2(&[[1.0, 2.0, 3.0]]).into_dyn(); // shape [1, 3]
    let z = ZonotopeTensor::from_input_shared(&values, 0.2);

    let ny = arr1(&[1.0, 1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0, 0.0]);

    let result = z.layer_norm_affine(&ny, &beta, 1e-5).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    // The bounds should have positive width due to accumulated radius
    let width_0 = bounds.upper()[[0, 0]] - bounds.lower()[[0, 0]];
    assert!(
        width_0 > 0.0,
        "bounds should have positive width from accumulated radius"
    );

    // If += was -=, radius would be negative and bounds would be inverted
    assert!(
        bounds.upper()[[0, 0]] >= bounds.lower()[[0, 0]],
        "upper should be >= lower (not inverted)"
    );
}

#[test]
fn test_layer_norm_affine_mean_deriv_division() {
    // Kills: replace / with * in the approximation-error enclosure
    // (mean_radius = sum(radius) / n and the (x - mean)/std quotient)

    // With multiple positions, ensure approximation error is reasonable
    let values = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(); // 2 rows
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    let ny = arr1(&[1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0]);

    let result = z.layer_norm_affine(&ny, &beta, 1e-5).unwrap();

    // The new error term (for approximation) should be small and finite
    let new_err_idx = z.n_error_terms + 1;
    let approx_err = result.coeffs[[new_err_idx, 0, 0]];

    assert!(
        approx_err.is_finite(),
        "approximation error should be finite"
    );
    // If / was *, error would be huge (ny * dim * std instead of ny / (dim * std))
    assert!(
        approx_err < 10.0,
        "approximation error should be small, got {}",
        approx_err
    );
}

// ==================== Per-element error cancellation regression test (#2522) ====================

/// Regression test for #2522: layer_norm approximation errors must be independent per element
/// so they cannot cancel under downstream [1, -1] linear projection.
///
/// With shared error (the bug), layer_norm(x)[0] - layer_norm(x)[1] would have error
/// terms cancel: err - err = 0. With per-element errors, each gets its own symbol.
#[test]
fn test_layer_norm_affine_linear_projection_soundness_2522() {
    use ndarray::arr2;

    // Shape [1, 2]: 1 position, 2 features. ny=1, beta=0.
    let values = arr2(&[[2.0_f32, 4.0]]);
    let z = ZonotopeTensor::from_input_2d(&values, 0.5);

    let ny = arr1(&[1.0_f32, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0]);
    let eps = 1e-5;

    let ln_z = z.layer_norm_affine(&ny, &beta, eps).unwrap();
    // n_new_error_terms = prefix_size * dim = 1 * 2 = 2
    assert_eq!(
        ln_z.n_error_terms(),
        z.n_error_terms() + 2,
        "layer_norm_affine should add per-element error symbols (#2522)"
    );

    // Project through [1, -1] to test cancellation
    let weight = arr2(&[[1.0_f32, -1.0]]);
    let projected = ln_z.linear(&weight, None).unwrap();
    let bounds = projected.to_bounded_tensor().unwrap();

    fn layer_norm_2(x: [f32; 2], eps: f32) -> [f32; 2] {
        let mean = f32::midpoint(x[0], x[1]);
        let var = f32::midpoint((x[0] - mean).powi(2), (x[1] - mean).powi(2));
        let std = (var + eps).sqrt();
        [(x[0] - mean) / std, (x[1] - mean) / std]
    }

    let mut true_min = f32::INFINITY;
    let mut true_max = f32::NEG_INFINITY;
    // from_input_2d creates per-element error terms, so each element varies independently.
    for &e0 in &[-1.0_f32, 1.0] {
        for &e1 in &[-1.0_f32, 1.0] {
            let x = [2.0 + 0.5 * e0, 4.0 + 0.5 * e1];
            let ln = layer_norm_2(x, eps);
            let y = ln[0] - ln[1];
            true_min = true_min.min(y);
            true_max = true_max.max(y);
        }
    }

    assert!(
        bounds.lower()[[0, 0]] <= true_min + 1e-4,
        "#2522 layer_norm: lower bound {} should contain true min {} \
         (cancellation if shared error terms)",
        bounds.lower()[[0, 0]],
        true_min
    );
    assert!(
        bounds.upper()[[0, 0]] >= true_max - 1e-4,
        "#2522 layer_norm: upper bound {} should contain true max {} \
         (cancellation if shared error terms)",
        bounds.upper()[[0, 0]],
        true_max
    );
}

// ==================== Mixed-sign gamma containment ====================

/// Mixed-sign gamma must keep the approximation-error symbols large enough to
/// cover the true LayerNorm range. A signed accumulation over gamma lets
/// negative entries cancel the error term (to exactly zero when sum(gamma)=0),
/// while the off-diagonal Jacobian and variance-shift contributions scale with
/// |gamma| — so the bound must be derived per output feature from |gamma|.
#[test]
fn test_layer_norm_affine_mixed_sign_gamma_containment() {
    let values = arr2(&[[1.0_f32, 2.0, -1.0, 0.5]]);
    let radius = 0.5_f32;
    let z = ZonotopeTensor::from_input_2d(&values, radius);

    let gamma = [1.0_f32, -1.0, 1.0, -1.0];
    let ny = arr1(&gamma);
    let beta = arr1(&[0.0_f32, 0.0, 0.0, 0.0]);
    let eps = 1e-5_f32;

    let result = z.layer_norm_affine(&ny, &beta, eps).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    fn layer_norm_4(x: [f32; 4], gamma: [f32; 4], eps: f32) -> [f32; 4] {
        let mean = x.iter().sum::<f32>() / 4.0;
        let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        let std = (var + eps).sqrt();
        [
            gamma[0] * (x[0] - mean) / std,
            gamma[1] * (x[1] - mean) / std,
            gamma[2] * (x[2] - mean) / std,
            gamma[3] * (x[3] - mean) / std,
        ]
    }

    // from_input_2d gives each element its own error symbol, so every grid
    // point of the box is a reachable zonotope point.
    let offsets = [-radius, -radius / 2.0, 0.0, radius / 2.0, radius];
    for &o0 in &offsets {
        for &o1 in &offsets {
            for &o2 in &offsets {
                for &o3 in &offsets {
                    let x = [1.0 + o0, 2.0 + o1, -1.0 + o2, 0.5 + o3];
                    let y = layer_norm_4(x, gamma, eps);
                    for (d, &yd) in y.iter().enumerate() {
                        let lo = bounds.lower()[[0, d]];
                        let hi = bounds.upper()[[0, d]];
                        assert!(
                            lo - 1e-4 <= yd && yd <= hi + 1e-4,
                            "LayerNorm({x:?})[{d}] = {yd} escapes [{lo}, {hi}]"
                        );
                    }
                }
            }
        }
    }
}

// ==================== NaN safety tests (#2676 Sites 3+4) ====================

/// #2676 Site 3: LayerNorm with NaN in center should return Err, not silently
/// mask via f32::max(NaN, 1e-10) = 1e-10 and produce huge amplified coefficients.
#[test]
fn test_layer_norm_affine_nan_center_returns_error_2676() {
    // Create a zonotope with NaN in one center element.
    let data: Vec<f32> = vec![f32::NAN, 2.0, 3.0];
    let coeffs = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 3]), data).unwrap();
    let z = ZonotopeTensor::new(coeffs).unwrap();

    let ny = arr1(&[1.0_f32, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0]);

    let result = z.layer_norm_affine(&ny, &beta, 1e-5);

    // With #2676 fix: should return Err because std is NaN.
    // Without fix: f32::max(NaN, 1e-10) = 1e-10, producing silently wrong output.
    assert!(
        result.is_err(),
        "#2676: layer_norm_affine should return Err for NaN center, got Ok"
    );
}

/// #2676 Site 3: LayerNorm with all-NaN center should return Err.
#[test]
fn test_layer_norm_affine_all_nan_center_returns_error_2676() {
    let data: Vec<f32> = vec![f32::NAN, f32::NAN];
    let coeffs = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 2]), data).unwrap();
    let z = ZonotopeTensor::new(coeffs).unwrap();

    let ny = arr1(&[1.0_f32, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0]);

    let result = z.layer_norm_affine(&ny, &beta, 1e-5);
    assert!(
        result.is_err(),
        "#2676: layer_norm_affine should return Err for all-NaN center"
    );
}

/// #2676 Site 4: Verify that non-NaN inputs still work correctly after the guard.
/// The std.max(1e-10) guard for legitimate near-zero variance must still function.
#[test]
fn test_layer_norm_affine_near_zero_variance_still_works_2676() {
    // Constant input → var=0 → std=sqrt(eps) → should still work.
    let values = arr2(&[[5.0, 5.0, 5.0]]);
    let z = ZonotopeTensor::concrete(values.into_dyn());

    let ny = arr1(&[1.0_f32, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0]);

    let result = z.layer_norm_affine(&ny, &beta, 1e-5);
    assert!(
        result.is_ok(),
        "#2676: layer_norm_affine should succeed for constant input (var=0)"
    );

    let result = result.unwrap();
    let center = result.center();
    // All outputs should be beta (0) since centered = [0, 0, 0].
    for d in 0..3 {
        assert!(
            center[[0, d]].is_finite(),
            "#2676: output should be finite for constant input"
        );
        assert!(
            center[[0, d]].abs() < 0.01,
            "#2676: output should be ~0 for constant input, got {}",
            center[[0, d]]
        );
    }
}
