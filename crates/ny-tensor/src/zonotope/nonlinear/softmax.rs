// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Softmax approximations for zonotope propagation.

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::borrow::Cow;

use super::super::ZonotopeTensor;

/// Tight upper bound on max_{i,j,k} |∂²σ_i/∂x_j∂x_k| for softmax σ.
///
/// All five index-equality cases of the softmax Hessian reduce to maximizing
/// |p(1-p)(1-2p)| on p ∈ (0,1). Setting the derivative 6p²-6p+1=0 gives
/// p* = (3±√3)/6, and the maximum value is √3/18 ≈ 0.09623.
///
/// This bound is dimension-independent: the worst case involves only two
/// softmax components (the others can approach zero), so d does not matter.
///
/// Used in the Taylor remainder bound for the zonotope linear approximation:
///   |σ(x) - σ(c) - Dσ(c)(x-c)|_∞ ≤ (1/2) · H_MAX · ||x-c||₁²
///
/// CITE: Direct derivation from softmax Hessian structure. See also
/// Bonaert et al. (2021) "Fast and Precise Certification of Transformers"
/// for related zonotope softmax analysis.
const SOFTMAX_HESSIAN_MAX: f32 = 0.096_225_05; // √3/18, conservative rounding up

impl ZonotopeTensor {
    /// Softmax with linear approximation to preserve zonotope form.
    ///
    /// softmax(x)_i = exp(x_i) / sum_j exp(x_j)
    ///
    /// This approximation linearizes softmax around the center and adds an error
    /// term to bound the non-linearity, enabling zonotope propagation through
    /// attention patterns.
    ///
    /// # Mathematical Details
    ///
    /// The Jacobian of softmax at center c is:
    ///   J[i,j] = s_c[i] * (δ[i,j] - s_c[j])
    /// where s_c = softmax(c).
    ///
    /// For a zonotope z = c + Σₖ aₖeₖ:
    ///   softmax(z) ≈ s_c + J @ (z - c) = s_c + Σₖ (J @ aₖ)eₖ
    ///
    /// # Error Bound
    ///
    /// The approximation error is bounded by the Taylor remainder:
    ///   |σ(z) - linear_approx|_∞ ≤ (1/2) · H_MAX · r²
    ///
    /// where r = Σₖ ||aₖ||₁ and H_MAX = √3/18 ≈ 0.0962 is the tight bound on
    /// max_{i,j,k} |∂²σ_i/∂x_j∂x_k| over the probability simplex. All five
    /// index-equality cases reduce to maximizing |p(1-p)(1-2p)| on (0,1).
    ///
    /// # Arguments
    /// * `axis` - The axis along which to apply softmax (default: last axis)
    pub fn softmax_affine(&self, axis: i32) -> Result<Self> {
        if self.element_shape.is_empty() {
            return Err(NyError::InvalidSpec(
                "softmax_affine requires at least 1 dimension".to_string(),
            ));
        }

        let ndim = self.element_shape.len();
        let axis_usize = ny_core::resolve_axis_i32(axis, ndim, "softmax_affine")?;

        // Get the size of the softmax dimension
        let softmax_dim = self.element_shape[axis_usize];

        // Helper: stable softmax computation
        // #2676 Site 1: Guard against NaN input and non-finite sum.
        // NaN input → nan_propagating_max returns NaN → exp(NaN) = NaN → sum = NaN.
        // Fallback: uniform distribution (maximally uncertain, sound over-approximation).
        fn compute_softmax(x: &[f32]) -> Vec<f32> {
            let n = x.len();
            let max_val = x
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, ny_core::nan_propagating_max);
            if max_val.is_nan() {
                return vec![1.0 / n as f32; n];
            }
            let exp_x: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f32 = exp_x.iter().sum();
            if !sum.is_finite() || sum <= 0.0 {
                return vec![1.0 / n as f32; n];
            }
            exp_x.iter().map(|&e| e / sum).collect()
        }

        // Helper: compute Jacobian @ vector for softmax
        // J[i,j] = s[i] * (δ[i,j] - s[j])
        // (J @ v)[i] = sum_j s[i] * (δ[i,j] - s[j]) * v[j]
        //            = s[i] * v[i] - s[i] * sum_j s[j] * v[j]
        //            = s[i] * (v[i] - dot(s, v))
        fn jacobian_vector_product(s: &[f32], v: &[f32]) -> Vec<f32> {
            let dot_sv: f32 = s.iter().zip(v.iter()).map(|(&si, &vi)| si * vi).sum();
            s.iter()
                .zip(v.iter())
                .map(|(&si, &vi)| si * (vi - dot_sv))
                .collect()
        }

        // For 1D input (shape [dim])
        if ndim == 1 {
            let dim = softmax_dim;
            // Each element gets its own approximation-error symbol to prevent
            // false cancellation across coordinates after downstream linear ops.
            // Same fix as GELU (gelu.rs gelu_affine_1d, #2470) and SiLU (#2486).
            let n_new_error_terms = dim;
            let n_rows = (1usize + self.n_error_terms)
                .checked_add(n_new_error_terms)
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "softmax_affine 1D: n_rows overflows: 1 + {} + {}",
                        self.n_error_terms, n_new_error_terms
                    ))
                })?;
            let mut result_coeffs = ndarray::Array2::<f32>::zeros((n_rows, dim));

            // Get center and compute softmax
            let center: Vec<f32> = self.coeffs.index_axis(Axis(0), 0).iter().cloned().collect();
            let s_c = compute_softmax(&center);

            // Output center = softmax(center)
            for (i, &s) in s_c.iter().enumerate() {
                result_coeffs[[0, i]] = s;
            }

            // Transform each error coefficient through Jacobian
            // Accumulate total radius: worst-case perturbation from center is
            // Σ_k ||a_k||_1 (sum over all error terms), NOT max_k ||a_k||_1.
            // The zonotope z = c + Σ_k ε_k·a_k achieves max perturbation when
            // all ε_k align, giving ||z - c||_1 = Σ_k ||a_k||_1.
            // Fix for #2473: was `max_radius.max(radius_k)`, now accumulates sum.
            let mut total_radius = 0.0f32;
            for k in 1..=self.n_error_terms {
                let err_k: Vec<f32> = self.coeffs.index_axis(Axis(0), k).iter().cloned().collect();

                // Accumulate L1 radius for error bound
                let radius_k: f32 = err_k.iter().map(|x| x.abs()).sum();
                total_radius += radius_k;

                // Apply Jacobian: J @ err_k
                let transformed = jacobian_vector_product(&s_c, &err_k);
                for (i, &t) in transformed.iter().enumerate() {
                    result_coeffs[[k, i]] = t;
                }
            }

            // Add per-element approximation error terms (#2522)
            // Taylor remainder: |σ(z) - linear_approx|_∞ ≤ 0.5 * H_MAX * r²
            // where r = Σ_k ||a_k||_1 is the total worst-case perturbation radius
            // and H_MAX = √3/18 ≈ 0.0962 is the tight element-wise Hessian bound.
            // Each element gets its own independent error symbol so downstream
            // linear combinations cannot cancel errors across elements.
            let approx_error = 0.5 * SOFTMAX_HESSIAN_MAX * total_radius * total_radius;
            for d in 0..dim {
                let approx_err_row = self.n_error_terms + 1 + d;
                result_coeffs[[approx_err_row, d]] = approx_error;
            }

            let out_coeffs = result_coeffs
                .into_dyn()
                .into_shape_with_order(IxDyn(&[n_rows, dim]))
                .map_err(|_| NyError::InvalidSpec("Cannot reshape softmax output".to_string()))?;

            return Ok(Self {
                coeffs: out_coeffs,
                n_error_terms: self.n_error_terms + n_new_error_terms,
                element_shape: self.element_shape.clone(),
            });
        }

        // For N-D input with softmax on the last axis, flatten the prefix dimensions
        // into independent rows and reuse the same row-wise Jacobian transform.
        if axis_usize == ndim - 1 {
            let prefix_size =
                checked_shape_product(&self.element_shape[..ndim - 1]).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "softmax_affine: prefix shape product overflows: {:?}",
                        &self.element_shape[..ndim - 1]
                    ))
                })?;
            let dim = softmax_dim;
            // Each element gets its own approximation-error symbol (#2522).
            // Same per-element fix as GELU (#2470) and SiLU (#2486).
            let n_new_error_terms = prefix_size.checked_mul(dim).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "softmax_affine: error term count overflows: {} * {}",
                    prefix_size, dim
                ))
            })?;
            let existing_rows = self.n_error_terms.checked_add(1).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "softmax_affine: existing row count overflows: 1 + {}",
                    self.n_error_terms
                ))
            })?;
            let n_rows = existing_rows
                .checked_add(n_new_error_terms)
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "softmax_affine: total row count overflows: {} + {}",
                        existing_rows, n_new_error_terms
                    ))
                })?;
            let mut result_coeffs = ndarray::Array3::<f32>::zeros((n_rows, prefix_size, dim));

            let coeffs: Cow<'_, ArrayD<f32>> = if self.coeffs.is_standard_layout() {
                Cow::Borrowed(&self.coeffs)
            } else {
                Cow::Owned(self.coeffs.as_standard_layout().to_owned())
            };
            let coeffs = coeffs.as_ref();

            let coeffs_3d = coeffs
                .view()
                .into_shape_with_order(IxDyn(&[existing_rows, prefix_size, dim]))
                .map_err(|_| {
                    NyError::InvalidSpec("Cannot reshape softmax coeffs to 3D".to_string())
                })?
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| {
                    NyError::InvalidSpec("Cannot view softmax coeffs as 3D".to_string())
                })?;

            // Process each prefix row independently (softmax along the last axis).
            for row_idx in 0..prefix_size {
                let center: Vec<f32> = (0..dim).map(|d| coeffs_3d[[0, row_idx, d]]).collect();
                let s_c = compute_softmax(&center);

                // Output center
                for (d, &sc) in s_c.iter().enumerate() {
                    result_coeffs[[0, row_idx, d]] = sc;
                }

                // Transform error coefficients
                // Fix for #2473: accumulate sum of per-term L1 radii, not max.
                // See 1D path comment for mathematical justification.
                let mut total_radius_this_row = 0.0f32;
                for k in 1..=self.n_error_terms {
                    let err_k: Vec<f32> = (0..dim).map(|d| coeffs_3d[[k, row_idx, d]]).collect();
                    let radius_k: f32 = err_k.iter().map(|x| x.abs()).sum();
                    total_radius_this_row += radius_k;

                    let transformed = jacobian_vector_product(&s_c, &err_k);
                    for (d, &t) in transformed.iter().enumerate() {
                        result_coeffs[[k, row_idx, d]] = t;
                    }
                }

                // Per-element approximation error terms (#2522)
                // Each (row_idx, d) pair gets its own independent error symbol.
                // Taylor remainder with tight Hessian bound: 0.5 * √3/18 * r²
                let approx_error =
                    0.5 * SOFTMAX_HESSIAN_MAX * total_radius_this_row * total_radius_this_row;
                for d in 0..dim {
                    let approx_err_row = self.n_error_terms + 1 + row_idx * dim + d;
                    result_coeffs[[approx_err_row, row_idx, d]] = approx_error;
                }
            }

            let mut out_shape = vec![n_rows];
            out_shape.extend_from_slice(&self.element_shape);
            let out_coeffs = result_coeffs
                .into_dyn()
                .into_shape_with_order(IxDyn(&out_shape))
                .map_err(|_| NyError::InvalidSpec("Cannot reshape softmax output".to_string()))?;

            return Ok(Self {
                coeffs: out_coeffs,
                n_error_terms: self.n_error_terms + n_new_error_terms,
                element_shape: self.element_shape.clone(),
            });
        }

        // For higher-dimensional inputs or different axis, fall back to general case
        // This handles shapes like [batch, heads, seq_q, seq_k] for attention
        Err(NyError::InvalidSpec(format!(
            "softmax_affine for shape {:?} axis {} not yet implemented",
            self.element_shape, axis
        )))
    }

    /// Causal softmax with linear approximation to preserve zonotope form.
    ///
    /// Causal attention masks out "future" keys: for each query position `i`, only keys
    /// `j <= i` participate in the softmax. Masked positions (`j > i`) output exactly 0.
    ///
    /// This uses the same Jacobian-based linearization as `softmax_affine`, but:
    /// - Computes softmax and Jacobian only over the unmasked prefix `0..=i`
    /// - Forces masked outputs (and their coefficients) to 0
    /// - Does not add approximation error to masked outputs
    ///
    /// # Arguments
    /// * `axis` - The axis along which to apply softmax (must be the last axis)
    pub fn softmax_affine_causal(&self, axis: i32) -> Result<Self> {
        if self.element_shape.len() < 2 {
            return Err(NyError::InvalidSpec(format!(
                "softmax_affine_causal requires at least 2 dimensions, got {:?}",
                self.element_shape
            )));
        }

        let ndim = self.element_shape.len();
        let axis_usize = ny_core::resolve_axis_i32(axis, ndim, "softmax_affine_causal")?;
        if axis_usize != ndim - 1 {
            return Err(NyError::InvalidSpec(format!(
                "softmax_affine_causal only supports last-axis softmax, got axis {} for shape {:?}",
                axis, self.element_shape
            )));
        }

        let seq_q = self.element_shape[ndim - 2];
        let seq_k = self.element_shape[ndim - 1];
        if seq_q > seq_k {
            return Err(NyError::InvalidSpec(format!(
                "softmax_affine_causal requires seq_q ({}) <= seq_k ({})",
                seq_q, seq_k
            )));
        }

        let prefix_size =
            checked_shape_product(&self.element_shape[..ndim - 2]).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "softmax_affine_causal: prefix shape product overflows: {:?}",
                    &self.element_shape[..ndim - 2]
                ))
            })?;
        let n_attn_rows = prefix_size.checked_mul(seq_q).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "softmax_affine_causal: n_attn_rows overflows: {} * {}",
                prefix_size, seq_q
            ))
        })?;

        let n_terms = self.n_error_terms.checked_add(1).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "softmax_affine_causal: n_terms overflows: 1 + {}",
                self.n_error_terms
            ))
        })?;
        // center + existing errors
        // Each element gets its own approximation-error symbol (#2522).
        // Same per-element fix as GELU (#2470), SiLU (#2486), and softmax 1D/2D.
        let n_new_error_terms = n_attn_rows.checked_mul(seq_k).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "softmax_affine_causal: n_new_error_terms overflows: {} * {}",
                n_attn_rows, seq_k
            ))
        })?;
        let n_rows_out = n_terms.checked_add(n_new_error_terms).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "softmax_affine_causal: n_rows_out overflows: {} + {}",
                n_terms, n_new_error_terms
            ))
        })?;

        // #2676 Site 1: Same NaN guard as compute_softmax (see softmax_affine).
        fn compute_softmax_prefix(x: &[f32]) -> Vec<f32> {
            let n = x.len();
            let max_val = x
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, ny_core::nan_propagating_max);
            if max_val.is_nan() {
                return vec![1.0 / n as f32; n];
            }
            let exp_x: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f32 = exp_x.iter().sum();
            if !sum.is_finite() || sum <= 0.0 {
                return vec![1.0 / n as f32; n];
            }
            exp_x.iter().map(|&e| e / sum).collect()
        }

        fn jacobian_vector_product_prefix(s: &[f32], v: &[f32]) -> Vec<f32> {
            let dot_sv: f32 = s.iter().zip(v.iter()).map(|(&si, &vi)| si * vi).sum();
            s.iter()
                .zip(v.iter())
                .map(|(&si, &vi)| si * (vi - dot_sv))
                .collect()
        }

        let coeffs: Cow<'_, ArrayD<f32>> = if self.coeffs.is_standard_layout() {
            Cow::Borrowed(&self.coeffs)
        } else {
            Cow::Owned(self.coeffs.as_standard_layout().to_owned())
        };
        let coeffs = coeffs.as_ref();

        let in_coeffs_3d = coeffs
            .view()
            .into_shape_with_order(IxDyn(&[n_terms, n_attn_rows, seq_k]))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape causal softmax coeffs to 3D".to_string())
            })?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| {
                NyError::InvalidSpec("Cannot view causal softmax coeffs as 3D".to_string())
            })?;

        let mut out_coeffs_3d = ndarray::Array3::<f32>::zeros((n_rows_out, n_attn_rows, seq_k));

        for row in 0..n_attn_rows {
            let query_i = row % seq_q;
            let allowed = (query_i + 1).min(seq_k);

            let center: Vec<f32> = (0..allowed).map(|j| in_coeffs_3d[[0, row, j]]).collect();
            let s_c = compute_softmax_prefix(&center);

            for (j, &sc) in s_c.iter().enumerate() {
                out_coeffs_3d[[0, row, j]] = sc;
            }

            // Fix for #2473: accumulate sum of per-term L1 radii, not max.
            // See softmax_affine 1D path comment for mathematical justification.
            let mut total_radius_this_row = 0.0f32;
            for k in 1..=self.n_error_terms {
                let err_k: Vec<f32> = (0..allowed).map(|j| in_coeffs_3d[[k, row, j]]).collect();
                let radius_k: f32 = err_k.iter().map(|x| x.abs()).sum();
                total_radius_this_row += radius_k;

                let transformed = jacobian_vector_product_prefix(&s_c, &err_k);
                for (j, &t) in transformed.iter().enumerate() {
                    out_coeffs_3d[[k, row, j]] = t;
                }
            }

            // Per-element approximation error terms (#2522), only for unmasked entries.
            // Each (row, j) pair gets its own independent error symbol.
            // Taylor remainder with tight Hessian bound: 0.5 * √3/18 * r²
            let approx_error =
                0.5 * SOFTMAX_HESSIAN_MAX * total_radius_this_row * total_radius_this_row;
            for j in 0..allowed {
                let approx_err_row = n_terms + row * seq_k + j;
                out_coeffs_3d[[approx_err_row, row, j]] = approx_error;
            }
        }

        let mut out_shape = vec![n_rows_out];
        out_shape.extend_from_slice(&self.element_shape);
        let out_coeffs = out_coeffs_3d
            .into_dyn()
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape causal softmax output".to_string())
            })?;

        Ok(Self {
            coeffs: out_coeffs,
            n_error_terms: self.n_error_terms + n_new_error_terms,
            element_shape: self.element_shape.clone(),
        })
    }
}
