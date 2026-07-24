// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! LayerNorm approximations for zonotope propagation.

use ndarray::{Array1, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::borrow::Cow;

use super::super::ZonotopeTensor;

impl ZonotopeTensor {
    /// Apply LayerNorm using affine approximation around the center point.
    ///
    /// LayerNorm(x) = ny * (x - mean(x)) / sqrt(var(x) + eps) + beta
    ///
    /// This is approximately linear near the center, so we:
    /// 1. Compute center output: y_c = LayerNorm(center)
    /// 2. Compute effective scale: s = ny / std(center)
    /// 3. Transform error terms: error_i -> s * error_i (preserves zonotope form)
    /// 4. Add per-element error terms bounding |LayerNorm(x) - affine(x)| over
    ///    the whole input box (interval enclosure of the true output range vs
    ///    the range of the emitted affine map)
    ///
    /// # Arguments
    /// * `ny` - Scale parameter (per feature)
    /// * `beta` - Shift parameter (per feature)
    /// * `eps` - Small constant for numerical stability
    ///
    /// # Note
    /// This approximation is tighter for small perturbations. The added error
    /// terms grow with the perturbation radius because they must cover the
    /// box-wide variation of mean(x) and 1/sqrt(var(x) + eps), which the
    /// center-pinned linearization ignores.
    pub fn layer_norm_affine(
        &self,
        ny: &Array1<f32>,
        beta: &Array1<f32>,
        eps: f32,
    ) -> Result<Self> {
        if self.element_shape.is_empty() {
            return Err(NyError::InvalidSpec(
                "layer_norm_affine requires at least 1 dimension".to_string(),
            ));
        }

        let dim = *self
            .element_shape
            .last()
            .ok_or_else(|| NyError::InvalidSpec("Empty element shape".to_string()))?;
        let prefix_shape = &self.element_shape[..self.element_shape.len() - 1];
        let prefix_size = checked_shape_product(prefix_shape)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "zonotope layer_norm: prefix shape product overflows: {:?}",
                    prefix_shape
                ))
            })?
            .max(1);

        if ny.len() != dim {
            return Err(NyError::shape_mismatch(vec![dim], vec![ny.len()]));
        }
        if beta.len() != dim {
            return Err(NyError::shape_mismatch(vec![dim], vec![beta.len()]));
        }

        let coeffs: Cow<'_, ndarray::ArrayD<f32>> = if self.coeffs.is_standard_layout() {
            Cow::Borrowed(&self.coeffs)
        } else {
            Cow::Owned(self.coeffs.as_standard_layout().to_owned())
        };
        let coeffs = coeffs.as_ref();

        let coeffs_3d = coeffs
            .view()
            .into_shape_with_order(IxDyn(&[1 + self.n_error_terms, prefix_size, dim]))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape LayerNorm coeffs to 3D".to_string()))?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| NyError::InvalidSpec("Cannot view LayerNorm coeffs as 3D".to_string()))?;

        let center_2d = coeffs_3d.index_axis(Axis(0), 0);

        // Each element gets its own approximation-error symbol (#2522).
        // Same per-element fix as GELU (#2470), SiLU (#2486), and softmax.
        let n_new_error_terms = prefix_size * dim;
        let n_rows = 1 + self.n_error_terms + n_new_error_terms;
        let mut result_coeffs = ndarray::Array3::<f32>::zeros((n_rows, prefix_size, dim));

        for row_idx in 0..prefix_size {
            let row = center_2d.row(row_idx);

            // Compute mean and variance of this row
            let mean: f32 = row.iter().sum::<f32>() / dim as f32;
            let centered: Vec<f32> = row.iter().map(|&x| x - mean).collect();
            let var: f32 = centered.iter().map(|&c| c * c).sum::<f32>() / dim as f32;
            let std = (var + eps).sqrt();

            // #2676 Site 3: NaN center coefficients → NaN mean → NaN var → NaN std.
            // f32::max(NaN, 1e-10) returns 1e-10, silently masking the NaN and
            // producing extremely large but finite coefficients downstream (Site 4).
            // Propagate the error instead of masking.
            if std.is_nan() {
                return Err(NyError::InvalidSpec(
                    "layer_norm_affine: NaN detected in center (std is NaN)".to_string(),
                ));
            }

            // Guard against division by zero or very small std
            let std_safe = std.max(1e-10);

            // Compute effective scale per feature: ny / std
            // This is the approximate linear scaling factor
            let eff_gamma: Vec<f32> = ny
                .iter()
                .map(|&g| {
                    // Cap effective ny to prevent overflow in large-ny models
                    let raw = g / std_safe;
                    raw.clamp(-1e6, 1e6)
                })
                .collect();

            // Compute LayerNorm output at center
            for d in 0..dim {
                let y_c = eff_gamma[d] * centered[d] + beta[d];
                result_coeffs[[0, row_idx, d]] = y_c;
            }

            // Transform error coefficients by effective ny
            // This preserves the zonotope form: if input has coefficient a_i,
            // output has coefficient (ny/std) * a_i (approximately)
            //
            // The full Jacobian is more complex (includes mean/var derivatives),
            // but this diagonal approximation works well for small perturbations.
            for i in 1..=self.n_error_terms {
                for d in 0..dim {
                    result_coeffs[[i, row_idx, d]] = eff_gamma[d] * coeffs_3d[[i, row_idx, d]];
                }
            }

            // Bound the approximation error per output feature over the input box.
            //
            // The rows emitted above form the affine map
            //   A_d(e) = y_c_d + eff_gamma_d * sum_k a_{k,d} e_k,
            // while the true output is
            //   y_d(x) = ny_d * (x_d - mean(x)) / sqrt(var(x) + eps) + beta_d.
            // Every zonotope point lies in the box |x_d - c_d| <= radius_d, so an
            // interval enclosure Y_d of y_d over the box and the exact range L_d
            // of A_d over |e_k| <= 1 bound the pointwise gap via the difference
            // interval Y_d - L_d. This covers the mean shift, the variance shift
            // (the linearization pins std at the center), all off-diagonal
            // Jacobian terms, and the eff_gamma clamp above.
            let radius: Vec<f32> = (0..dim)
                .map(|d| {
                    (1..=self.n_error_terms)
                        .map(|i| coeffs_3d[[i, row_idx, d]].abs())
                        .sum::<f32>()
                })
                .collect();
            // |mean(x) - mean(c)| <= mean_radius over the box.
            let mean_radius: f32 = radius.iter().sum::<f32>() / dim as f32;

            // Enclosure of x_d - mean(x): centered_d ± (radius_d + mean_radius).
            let dev: Vec<(f32, f32)> = (0..dim)
                .map(|d| {
                    let slack = radius[d] + mean_radius;
                    (centered[d] - slack, centered[d] + slack)
                })
                .collect();

            // Enclosure of var(x) = sum_d (x_d - mean(x))^2 / n.
            let mut var_lo = 0.0f32;
            let mut var_hi = 0.0f32;
            for &(lo, hi) in &dev {
                let sq_lo = if lo <= 0.0 && hi >= 0.0 {
                    0.0
                } else {
                    (lo * lo).min(hi * hi)
                };
                var_lo += sq_lo / dim as f32;
                var_hi += (lo * lo).max(hi * hi) / dim as f32;
            }
            let std_lo = (var_lo + eps).sqrt();
            let std_hi = (var_hi + eps).sqrt();
            // Without a positive lower bound on std, 1/std is unbounded over the
            // box and no finite error coefficient is sound.
            if std_lo.is_nan() || std_lo <= 0.0 || !std_hi.is_finite() {
                return Err(NyError::NumericalInstability(
                    "layer_norm_affine: cannot bound 1/std over the input box".to_string(),
                ));
            }

            for d in 0..dim {
                let (dev_lo, dev_hi) = dev[d];
                // (x_d - mean)/std over the positive std interval [std_lo, std_hi].
                let q_lo = if dev_lo >= 0.0 {
                    dev_lo / std_hi
                } else {
                    dev_lo / std_lo
                };
                let q_hi = if dev_hi >= 0.0 {
                    dev_hi / std_lo
                } else {
                    dev_hi / std_hi
                };
                let (y_lo, y_hi) = if ny[d] >= 0.0 {
                    (ny[d] * q_lo + beta[d], ny[d] * q_hi + beta[d])
                } else {
                    (ny[d] * q_hi + beta[d], ny[d] * q_lo + beta[d])
                };

                // Exact range of the emitted affine part over |e_k| <= 1.
                let y_c = result_coeffs[[0, row_idx, d]];
                let lin_halfwidth = eff_gamma[d].abs() * radius[d];
                let lin_lo = y_c - lin_halfwidth;
                let lin_hi = y_c + lin_halfwidth;

                // |y_d - A_d| <= max endpoint of the difference interval Y_d - L_d.
                // Check the endpoints before taking the max: f32::max would
                // silently absorb a NaN from inf - inf into the other operand.
                let gap_hi = y_hi - lin_lo;
                let gap_lo = lin_hi - y_lo;
                if !gap_hi.is_finite() || !gap_lo.is_finite() {
                    return Err(NyError::NumericalInstability(
                        "layer_norm_affine: approximation error bound is not finite".to_string(),
                    ));
                }
                let approx_error = gap_hi.max(gap_lo).max(0.0);

                // Per-element error terms (#2522): each (row_idx, d) gets its own row.
                let approx_err_row = self.n_error_terms + 1 + row_idx * dim + d;
                result_coeffs[[approx_err_row, row_idx, d]] = approx_error;
            }
        }

        let mut out_shape = vec![n_rows];
        out_shape.extend_from_slice(prefix_shape);
        out_shape.push(dim);
        let out_coeffs = result_coeffs
            .into_dyn()
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape LayerNorm output".to_string()))?;

        Ok(Self {
            coeffs: out_coeffs,
            n_error_terms: self.n_error_terms + n_new_error_terms,
            element_shape: self.element_shape.clone(),
        })
    }

    /// Apply mean-only LayerNorm using an exact affine transform.
    ///
    /// Mean-only LayerNorm(x) = ny * (x - mean(x)) + beta
    ///
    /// This is linear in x, so we can propagate the zonotope exactly without
    /// adding a new error term.
    pub fn layer_norm_affine_mean_only(
        &self,
        ny: &Array1<f32>,
        beta: &Array1<f32>,
    ) -> Result<Self> {
        if self.element_shape.is_empty() {
            return Err(NyError::InvalidSpec(
                "layer_norm_affine_mean_only requires at least 1 dimension".to_string(),
            ));
        }

        let dim = *self
            .element_shape
            .last()
            .ok_or_else(|| NyError::InvalidSpec("Empty element shape".to_string()))?;
        let prefix_shape = &self.element_shape[..self.element_shape.len() - 1];
        let prefix_size = checked_shape_product(prefix_shape)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "zonotope layer_norm: prefix shape product overflows: {:?}",
                    prefix_shape
                ))
            })?
            .max(1);

        if ny.len() != dim {
            return Err(NyError::shape_mismatch(vec![dim], vec![ny.len()]));
        }
        if beta.len() != dim {
            return Err(NyError::shape_mismatch(vec![dim], vec![beta.len()]));
        }

        let coeffs: Cow<'_, ndarray::ArrayD<f32>> = if self.coeffs.is_standard_layout() {
            Cow::Borrowed(&self.coeffs)
        } else {
            Cow::Owned(self.coeffs.as_standard_layout().to_owned())
        };
        let coeffs = coeffs.as_ref();

        let coeffs_3d = coeffs
            .view()
            .into_shape_with_order(IxDyn(&[1 + self.n_error_terms, prefix_size, dim]))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape LayerNorm coeffs to 3D".to_string()))?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| NyError::InvalidSpec("Cannot view LayerNorm coeffs as 3D".to_string()))?;

        let center_2d = coeffs_3d.index_axis(Axis(0), 0);

        let n_rows = 1 + self.n_error_terms;
        let mut result_coeffs = ndarray::Array3::<f32>::zeros((n_rows, prefix_size, dim));
        let inv_dim = 1.0_f32 / dim as f32;

        for row_idx in 0..prefix_size {
            let row = center_2d.row(row_idx);
            let mean: f32 = row.iter().sum::<f32>() * inv_dim;

            for d in 0..dim {
                let centered = row[d] - mean;
                result_coeffs[[0, row_idx, d]] = ny[d] * centered + beta[d];
            }

            for err_idx in 1..=self.n_error_terms {
                let mut mean_err = 0.0_f32;
                for d in 0..dim {
                    mean_err += coeffs_3d[[err_idx, row_idx, d]];
                }
                mean_err *= inv_dim;

                for d in 0..dim {
                    let centered_err = coeffs_3d[[err_idx, row_idx, d]] - mean_err;
                    result_coeffs[[err_idx, row_idx, d]] = ny[d] * centered_err;
                }
            }
        }

        let mut out_shape = vec![n_rows];
        out_shape.extend_from_slice(prefix_shape);
        out_shape.push(dim);
        let out_coeffs = result_coeffs
            .into_dyn()
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape LayerNorm output".to_string()))?;

        Ok(Self {
            coeffs: out_coeffs,
            n_error_terms: self.n_error_terms,
            element_shape: self.element_shape.clone(),
        })
    }
}
