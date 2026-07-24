// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SiLU (Swish) activation approximation for zonotope propagation.

use ndarray::Axis;
use ny_core::{checked_shape_product, nan_propagating_max, NyError, Result};

use super::super::ZonotopeTensor;

impl ZonotopeTensor {
    /// SiLU (Swish) activation with linear approximation to preserve zonotope form.
    ///
    /// SiLU(x) = x * sigmoid(x) is used in SwiGLU FFN layers of modern LLMs.
    /// This approximation evaluates SiLU at the center and uses the derivative
    /// as a linear scaling factor, adding an error term for the approximation.
    ///
    /// # Key Insight
    ///
    /// Without this approximation, SiLU falls back to IBP which loses all correlations.
    /// This causes FFN bounds to explode (~36x per block from SwiGLU multiplication).
    /// With the linear approximation, correlations are preserved through SiLU,
    /// enabling tighter SwiGLU multiplication bounds.
    ///
    /// # Mathematical Details
    ///
    /// For a zonotope z = c + Σᵢ aᵢeᵢ:
    /// - output_center = silu(c)
    /// - slope = silu'(c) = sigmoid(c) * (1 + c * (1 - sigmoid(c)))
    /// - output = silu(c) + slope * (z - c) = silu(c) - slope*c + slope*z
    /// - Error bound from second derivative: |silu''(x)| * r² / 2
    ///
    /// # Soundness
    ///
    /// The linear approximation is sound because the added error symbol covers
    /// the Taylor remainder |f(x) - f(c) - f'(c)·(x-c)| ≤ max|f''| · r² / 2,
    /// where max|f''| over [c-r, c+r] is an upper bound derived from the
    /// interval endpoints and the interior extrema of |SiLU''|: the global
    /// maximum 1/2 at x = 0 and the negative-lobe peaks near x ≈ ±3.436
    /// (|SiLU''| ≈ 0.03691, covered by the constant 0.038).
    pub fn silu_affine(&self) -> Result<Self> {
        // SiLU helper functions
        fn sigmoid(x: f32) -> f32 {
            // Numerically stable sigmoid
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let ex = x.exp();
                ex / (1.0 + ex)
            }
        }

        fn silu(x: f32) -> f32 {
            // Avoid NaN from (-inf) * 0 while preserving mathematical limits:
            // lim x->-inf x*sigmoid(x) = 0 and lim x->+inf x*sigmoid(x) = +inf.
            if !x.is_finite() {
                if x.is_nan() {
                    return f32::NAN;
                }
                return if x.is_sign_negative() { 0.0 } else { x };
            }
            x * sigmoid(x)
        }

        fn silu_derivative(x: f32) -> f32 {
            // Avoid NaN from inf * 0 in the closed form.
            // lim x->-inf silu'(x) = 0, lim x->+inf silu'(x) = 1.
            if !x.is_finite() {
                if x.is_nan() {
                    return f32::NAN;
                }
                return if x.is_sign_negative() { 0.0 } else { 1.0 };
            }
            // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            let s = sigmoid(x);
            s * (1.0 + x * (1.0 - s))
        }

        fn silu_second_derivative(x: f32) -> f32 {
            // Guard for #2474 (same pattern as silu() and silu_derivative() guards).
            // Avoid NaN from inf*0 and inf-inf in the closed form.
            // lim x->-inf SiLU''(x) = 0, lim x->+inf SiLU''(x) = 0.
            if !x.is_finite() {
                if x.is_nan() {
                    return f32::NAN;
                }
                return 0.0;
            }
            // SiLU''(x) = sigmoid(x) * sigmoid(-x) * (2 + x - 2*x*sigmoid(x)),
            // using sigmoid(-x) = 1 - sigmoid(x). Evaluated in f64 and via the
            // sigmoid(-x) form: the naive `1.0 - s` cancels catastrophically
            // once sigmoid(x) rounds to 1 (x ≳ 17 in f32), collapsing the
            // curvature — and with it the Taylor-remainder error symbol — to
            // exactly 0 while the true value is still ≈ (x - 2)·e^-x ≠ 0.
            fn sigmoid_f64(x: f64) -> f64 {
                if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let ex = x.exp();
                    ex / (1.0 + ex)
                }
            }
            let xd = f64::from(x);
            let s = sigmoid_f64(xd);
            let s_neg = sigmoid_f64(-xd);
            (s * s_neg * (2.0 + xd - 2.0 * xd * s)) as f32
        }

        /// Global curvature bound: |SiLU''(x)| ≤ 1/2 for all x, attained at
        /// x = 0 where SiLU''(0) = sigmoid(0)² · 2 = 1/2 exactly.
        const SILU_CURVATURE_BOUND_GLOBAL: f32 = 0.5;

        /// Upper bound on the negative-lobe local maxima of |SiLU''|: besides
        /// x = 0, the only interior extrema of |SiLU''| sit at x ≈ ±3.436 with
        /// |SiLU''| ≈ 0.03691 < 0.038.
        const SILU_CURVATURE_LOBE_BOUND: f32 = 0.038;

        /// Bracket certainly containing the positive lobe-peak abscissa
        /// (peak at x ≈ 3.436; the negative peak mirrors it, SiLU'' is even
        /// because SiLU(x) - SiLU(-x) = x).
        const SILU_LOBE_PEAK_MIN: f32 = 3.0;
        const SILU_LOBE_PEAK_MAX: f32 = 4.0;

        /// Upper bound on max |SiLU''(x)| over [lo, hi].
        ///
        /// |SiLU''| has interior local maxima only at x = 0 (the global
        /// maximum 1/2) and at the negative-lobe peaks x ≈ ±3.436, so the
        /// maximum over any interval is attained at an interval endpoint or at
        /// one of those points. Endpoints are evaluated directly; x = 0
        /// contributes the exact 1/2; a lobe peak contributes the proven bound
        /// 0.038 whenever the interval can contain it. (A sampled maximum
        /// would only be a lower bound on max|f''| and could under-estimate
        /// the Taylor remainder.) Uses nan_propagating_max so a NaN endpoint
        /// surfaces as NaN (#2850).
        fn max_silu_second_deriv(lo: f32, hi: f32) -> f32 {
            let mut max_val = nan_propagating_max(
                silu_second_derivative(lo).abs(),
                silu_second_derivative(hi).abs(),
            );
            if lo <= 0.0 && 0.0 <= hi {
                // Interval contains the global maximum at x = 0.
                return nan_propagating_max(max_val, SILU_CURVATURE_BOUND_GLOBAL);
            }
            // The lobe peak at +p (p ∈ [SILU_LOBE_PEAK_MIN, SILU_LOBE_PEAK_MAX])
            // can lie strictly inside [lo, hi] only if lo < p < hi is possible;
            // mirrored condition for -p. Over-covering is sound: the candidate
            // only ever raises the bound.
            if (lo < SILU_LOBE_PEAK_MAX && hi > SILU_LOBE_PEAK_MIN)
                || (lo < -SILU_LOBE_PEAK_MIN && hi > -SILU_LOBE_PEAK_MAX)
            {
                max_val = nan_propagating_max(max_val, SILU_CURVATURE_LOBE_BOUND);
            }
            max_val
        }

        // Support 1D and 2D zonotopes
        match self.element_shape.len() {
            1 => {
                let dim = self.element_shape[0];
                // Each element gets its own approximation-error symbol to prevent
                // false cancellation across coordinates after downstream linear ops.
                // Same fix as GELU (gelu.rs gelu_affine_1d, #2470).
                let n_new_error_terms = dim;
                let n_rows = 1 + self.n_error_terms + n_new_error_terms;
                let mut result_coeffs = ndarray::Array2::<f32>::zeros((n_rows, dim));

                // Get center values
                let center = self.coeffs.index_axis(Axis(0), 0);

                for d in 0..dim {
                    let c = center[d];

                    // Compute radius (sum of absolute error coefficients)
                    let radius: f32 = (1..=self.n_error_terms)
                        .map(|i| self.coeffs[[i, d]].abs())
                        .sum();

                    // Output center = silu(c)
                    result_coeffs[[0, d]] = silu(c);

                    // Transform error coefficients by slope
                    let slope = silu_derivative(c);
                    for i in 1..=self.n_error_terms {
                        result_coeffs[[i, d]] = slope * self.coeffs[[i, d]];
                    }

                    // Bound approximation error using second derivative
                    // |f(x) - f(c) - f'(c)*(x-c)| <= max|f''| * r^2 / 2
                    if radius > 0.0 {
                        let max_second = max_silu_second_deriv(c - radius, c + radius);
                        let approx_error = max_second * radius * radius / 2.0;
                        // Per-element error: each element d gets its own row
                        let approx_err_row = self.n_error_terms + 1 + d;
                        result_coeffs[[approx_err_row, d]] = approx_error;
                    }
                }

                Ok(Self {
                    coeffs: result_coeffs.into_dyn(),
                    n_error_terms: self.n_error_terms + n_new_error_terms,
                    element_shape: self.element_shape.clone(),
                })
            }
            2 => {
                // 2D case: (seq, dim)
                // Each element gets its own approximation-error symbol to prevent
                // false cancellation across coordinates after downstream linear ops.
                // Same fix as GELU (gelu.rs gelu_affine_2d, #2470).
                let seq_len = self.element_shape[0];
                let dim = self.element_shape[1];
                let n_new_error_terms = seq_len * dim;
                let n_rows = 1 + self.n_error_terms + n_new_error_terms;
                let mut result_coeffs = ndarray::Array3::<f32>::zeros((n_rows, seq_len, dim));

                let center = self.coeffs.index_axis(Axis(0), 0);
                let center_2d = center
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| NyError::InvalidSpec("Cannot view center as 2D".to_string()))?;

                for s in 0..seq_len {
                    for d in 0..dim {
                        let c = center_2d[[s, d]];

                        // Compute radius
                        let radius: f32 = (1..=self.n_error_terms)
                            .map(|i| self.coeffs[[i, s, d]].abs())
                            .sum();

                        // Output center
                        result_coeffs[[0, s, d]] = silu(c);

                        // Transform error coefficients
                        let slope = silu_derivative(c);
                        for i in 1..=self.n_error_terms {
                            result_coeffs[[i, s, d]] = slope * self.coeffs[[i, s, d]];
                        }

                        // Bound approximation error
                        if radius > 0.0 {
                            let max_second = max_silu_second_deriv(c - radius, c + radius);
                            let approx_error = max_second * radius * radius / 2.0;
                            // Per-element error: each (s, d) gets its own row
                            let approx_err_row = self.n_error_terms + 1 + s * dim + d;
                            result_coeffs[[approx_err_row, s, d]] = approx_error;
                        }
                    }
                }

                Ok(Self {
                    coeffs: result_coeffs.into_dyn(),
                    n_error_terms: self.n_error_terms + n_new_error_terms,
                    element_shape: self.element_shape.clone(),
                })
            }
            _ => {
                // Generalize to N-D by flattening elements; SiLU is applied element-wise.
                let element_shape = self.element_shape.clone();
                let n_elements = checked_shape_product(&element_shape).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "silu_affine: element shape product overflows: {:?}",
                        element_shape
                    ))
                })?;
                let flat = self.reshape(&[n_elements])?;
                let out_flat = flat.silu_affine()?;
                out_flat.reshape(&element_shape)
            }
        }
    }
}
