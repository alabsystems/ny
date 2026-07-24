// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GELU affine zonotope approximation.
//!
//! Linearizes GELU around each element's center value and adds conservative
//! per-element error symbols bounded by `max|GELU''| * r² / 2`.
//! Fix for #2470: the GELU arm previously called `silu_affine()` which
//! evaluates SiLU, not GELU.

use ndarray::Axis;
use ny_core::{checked_shape_product, nan_propagating_max, NyError, Result};

use super::ZonotopeTensor;

// =============================================================================
// GELU helper functions (defined inline to avoid cross-crate dependency).
// Ref: ny-propagate/src/layers/softmax/gelu/eval.rs
// =============================================================================

/// Abramowitz & Stegun erf approximation (eq. 7.1.26), max error ~1.5e-7.
/// Used instead of libm::erff because ny-tensor does not depend on libm.
fn erff_approx(x: f32) -> f32 {
    let sign = x.signum();
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-a * a).exp())
}

fn gelu_erf(x: f32) -> f32 {
    // Guard against 0 * inf = NaN when x = ±inf.
    // GELU(-inf) = 0, GELU(+inf) = +inf. Ref: #1836.
    if !x.is_finite() {
        if x.is_nan() {
            return f32::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let inv_sqrt2: f32 = 1.0 / 2.0_f32.sqrt();
    0.5 * x * (1.0 + erff_approx(x * inv_sqrt2))
}

fn gelu_tanh(x: f32) -> f32 {
    if !x.is_finite() {
        if x.is_nan() {
            return f32::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_erf_derivative(x: f32) -> f32 {
    // GELU'(x) = Φ(x) + x·φ(x) where Φ = normal CDF, φ = normal PDF
    //
    // #2676: For ±Inf, pdf=exp(-Inf)=0 but x·pdf = ±Inf·0 = NaN.
    // Correct limits: GELU'(+∞) = 1, GELU'(-∞) = 0.
    // NaN input propagates as NaN (correct diagnostic signal).
    if x.is_infinite() {
        return if x.is_sign_positive() { 1.0 } else { 0.0 };
    }
    let inv_sqrt2: f32 = 1.0 / 2.0_f32.sqrt();
    let inv_sqrt_2pi: f32 = 1.0 / (2.0 * std::f32::consts::PI).sqrt();
    // Not a midpoint: this is the normal-CDF term Φ(x) = (1 + erf(x/√2))/2 of
    // GELU'(x). Keep the formula literal — relaxation anchors must stay bit-identical.
    #[allow(clippy::manual_midpoint)]
    let phi: f32 = 0.5 * (1.0 + erff_approx(x * inv_sqrt2));
    let pdf: f32 = inv_sqrt_2pi * (-0.5 * x * x).exp();
    phi + x * pdf
}

// Not a midpoint: `0.5 * (1 + tanh_t)` is the CDF-like term of the tanh-GELU
// derivative. Keep the formula literal — relaxation anchors must stay bit-identical.
#[allow(clippy::manual_midpoint)]
fn gelu_tanh_derivative(x: f32) -> f32 {
    // #2676: For ±Inf, sech²(g)=0 but x·sech²·dt_dx = Inf·0·Inf = NaN.
    // Correct limits: GELU'(+∞) = 1, GELU'(-∞) = 0.
    // NaN input propagates as NaN (correct diagnostic signal).
    if x.is_infinite() {
        return if x.is_sign_positive() { 1.0 } else { 0.0 };
    }
    let k: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
    let t: f32 = k * (x + 0.044715 * x * x * x);
    let tanh_t: f32 = t.tanh();
    let sech2_t: f32 = 1.0 - tanh_t * tanh_t;
    let dt_dx: f32 = k * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + tanh_t) + 0.5 * x * sech2_t * dt_dx
}

fn gelu_erf_second_derivative(x: f32) -> f32 {
    // GELU''(x) = φ(x)·(2 - x²) where φ is the standard normal PDF.
    // Closed-form from differentiating GELU'(x) = Φ(x) + x·φ(x).
    //
    // #2676 Site 2: For ±Inf, pdf=exp(-Inf)=0 but (2-x²)=-Inf,
    // giving 0·(-Inf)=NaN. Guard: GELU''(x) → 0 as |x| → ∞.
    // NaN input propagates as NaN (correct diagnostic signal).
    if x.is_infinite() {
        return 0.0;
    }
    let inv_sqrt_2pi: f32 = 1.0 / (2.0 * std::f32::consts::PI).sqrt();
    let pdf: f32 = inv_sqrt_2pi * (-0.5 * x * x).exp();
    pdf * (2.0 - x * x)
}

/// Interior extrema locations of erf-GELU''. GELU'''(x) = -x·φ(x)·(4 - x²)
/// vanishes exactly at x ∈ {-2, 0, 2}, so GELU'' is monotone between
/// consecutive candidates and max|GELU''| over any interval is attained at an
/// interval endpoint or at one of these points.
const GELU_ERF_CURVATURE_EXTREMA: &[f32] = &[-2.0, 0.0, 2.0];

/// Global curvature bound shared by both GELU variants: |GELU''(x)| ≤ 0.8 for
/// all x (erf: max|φ(x)·(2 - x²)| = 2·φ(0) ≈ 0.798, at x = 0).
/// Ref: ny-propagate test_gelu_erf_curvature_bound_global,
///      ny-propagate test_gelu_tanh_curvature_bound_global_sampled.
const GELU_CURVATURE_BOUND_GLOBAL: f32 = 0.8;

/// Upper bound on max|GELU_erf''(x)| over [lo, hi].
///
/// Evaluates |GELU''| at the interval endpoints and at the interior extrema
/// {-2, 0, 2}; GELU'' is monotone between consecutive candidates, so no other
/// point of the interval can exceed them. (A sampled maximum would only be a
/// lower bound on max|f''| and could under-estimate the Taylor remainder.)
/// Uses nan_propagating_max so a NaN endpoint surfaces as NaN (#2850).
fn gelu_erf_max_curvature(lo: f32, hi: f32) -> f32 {
    let mut max_val = nan_propagating_max(
        gelu_erf_second_derivative(lo).abs(),
        gelu_erf_second_derivative(hi).abs(),
    );
    for &x in GELU_ERF_CURVATURE_EXTREMA {
        if lo <= x && x <= hi {
            max_val = nan_propagating_max(max_val, gelu_erf_second_derivative(x).abs());
        }
    }
    max_val
}

/// Upper bound on max|GELU_tanh''(x)| over [lo, hi].
///
/// tanh-GELU'' has no closed-form extrema, so the proven global bound is used
/// for every interval. A NaN endpoint (NaN center or radius) propagates as NaN.
fn gelu_tanh_max_curvature(lo: f32, hi: f32) -> f32 {
    if lo.is_nan() || hi.is_nan() {
        return f32::NAN;
    }
    GELU_CURVATURE_BOUND_GLOBAL
}

impl ZonotopeTensor {
    /// Apply GELU using affine approximation around the center point.
    ///
    /// GELU(x) = 0.5 * x * (1 + erf(x / √2))  (erf variant)
    /// GELU(x) = 0.5 * x * (1 + tanh(√(2/π)(x + 0.044715x³)))  (tanh variant)
    ///
    /// This linearizes GELU around each element's center value and adds a
    /// conservative error term bounded by `max|GELU''| * r² / 2`.
    ///
    /// # Arguments
    /// * `use_tanh_approx` - If true, use the tanh approximation; otherwise use erf.
    ///
    /// # Soundness
    ///
    /// The linear approximation is sound because the added error symbol covers
    /// the Taylor remainder |f(x) - f(c) - f'(c)·(x-c)| ≤ max|f''| · r² / 2,
    /// where max|f''| over [c-r, c+r] is an upper bound:
    /// - erf-GELU: exact interval maximum from the endpoints and the interior
    ///   extrema {-2, 0, 2} of GELU''(x) = φ(x)·(2 - x²).
    /// - tanh-GELU: the proven global bound |GELU''(x)| ≤ 0.8.
    ///
    /// Ref: ny-propagate test_gelu_erf_curvature_bound_global,
    /// ny-propagate test_gelu_tanh_curvature_bound_global_sampled.
    pub fn gelu_affine(&self, use_tanh_approx: bool) -> Result<Self> {
        let gelu_fn: fn(f32) -> f32 = if use_tanh_approx { gelu_tanh } else { gelu_erf };
        let gelu_deriv_fn: fn(f32) -> f32 = if use_tanh_approx {
            gelu_tanh_derivative
        } else {
            gelu_erf_derivative
        };
        let max_curvature_fn: fn(f32, f32) -> f32 = if use_tanh_approx {
            gelu_tanh_max_curvature
        } else {
            gelu_erf_max_curvature
        };

        match self.element_shape.len() {
            1 => self.gelu_affine_1d(gelu_fn, gelu_deriv_fn, max_curvature_fn),
            2 => self.gelu_affine_2d(gelu_fn, gelu_deriv_fn, max_curvature_fn),
            _ => {
                let element_shape = self.element_shape.clone();
                let n_elements = checked_shape_product(&element_shape).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "gelu_affine: element shape product overflows: {:?}",
                        element_shape
                    ))
                })?;
                let flat = self.reshape(&[n_elements])?;
                let out_flat = flat.gelu_affine(use_tanh_approx)?;
                out_flat.reshape(&element_shape)
            }
        }
    }

    fn gelu_affine_1d(
        &self,
        gelu_fn: fn(f32) -> f32,
        gelu_deriv_fn: fn(f32) -> f32,
        max_curvature_fn: fn(f32, f32) -> f32,
    ) -> Result<Self> {
        let dim = self.element_shape[0];
        let n_new_error_terms = dim;
        let n_rows = 1 + self.n_error_terms + n_new_error_terms;
        let mut result_coeffs = ndarray::Array2::<f32>::zeros((n_rows, dim));

        let center = self.coeffs.index_axis(Axis(0), 0);
        for d in 0..dim {
            let c = center[d];
            let radius: f32 = (1..=self.n_error_terms)
                .map(|i| self.coeffs[[i, d]].abs())
                .sum();

            result_coeffs[[0, d]] = gelu_fn(c);
            let slope = gelu_deriv_fn(c);
            for i in 1..=self.n_error_terms {
                result_coeffs[[i, d]] = slope * self.coeffs[[i, d]];
            }

            // |f(x) - f(c) - f'(c)*(x-c)| <= max|f''| * r^2 / 2
            let approx_error = if radius > 0.0 {
                max_curvature_fn(c - radius, c + radius) * radius * radius / 2.0
            } else {
                0.0
            };
            // Each element gets its own approximation-error symbol to prevent
            // false cancellation across coordinates after downstream linear ops.
            let approx_err_row = self.n_error_terms + 1 + d;
            result_coeffs[[approx_err_row, d]] = approx_error;
        }

        Ok(Self {
            coeffs: result_coeffs.into_dyn(),
            n_error_terms: self.n_error_terms + n_new_error_terms,
            element_shape: self.element_shape.clone(),
        })
    }

    fn gelu_affine_2d(
        &self,
        gelu_fn: fn(f32) -> f32,
        gelu_deriv_fn: fn(f32) -> f32,
        max_curvature_fn: fn(f32, f32) -> f32,
    ) -> Result<Self> {
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
                let radius: f32 = (1..=self.n_error_terms)
                    .map(|i| self.coeffs[[i, s, d]].abs())
                    .sum();

                result_coeffs[[0, s, d]] = gelu_fn(c);
                let slope = gelu_deriv_fn(c);
                for i in 1..=self.n_error_terms {
                    result_coeffs[[i, s, d]] = slope * self.coeffs[[i, s, d]];
                }

                let approx_error = if radius > 0.0 {
                    max_curvature_fn(c - radius, c + radius) * radius * radius / 2.0
                } else {
                    0.0
                };
                let approx_err_row = self.n_error_terms + 1 + s * dim + d;
                result_coeffs[[approx_err_row, s, d]] = approx_error;
            }
        }

        Ok(Self {
            coeffs: result_coeffs.into_dyn(),
            n_error_terms: self.n_error_terms + n_new_error_terms,
            element_shape: self.element_shape.clone(),
        })
    }
}
