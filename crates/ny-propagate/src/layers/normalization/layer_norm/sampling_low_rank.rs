// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Low-rank sampling-based CROWN backward for LayerNorm (#1957).
//!
//! Exploits `J[i,j] = ny_i/std * [δ_ij - 1/n - z_i*z_j/n]` for O(n)
//! Jacobian-vector products instead of materializing the full n×n matrix.
//! Replaces the generic O(n²) `sampling_crown_scalar()` for LayerNorm `Sampling`.
//!
//! Preserves all dense-path error semantics: non-finite guard (#3259),
//! ny/std overflow (#2901), per-row fallback (#3128), directed rounding.
//!
//! Reference: designs/2026-03-17-issue-1957-layernorm-sampling-low-rank.md

use ndarray::{Array1, Array2};
use ny_core::{is_crown_coeff_safe_f64, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};

use super::types::LayerNormLayer;
use crate::LinearBounds;

/// Cached low-rank linearization of LayerNorm at a center point.
///
/// Stores the decomposed Jacobian structure so that `apply(v)` and
/// `backprop_row(a)` run in O(n) instead of O(n²).
struct LayerNormSamplingLinearization {
    /// Linear approximation bias: `b = y_center - J @ x_center`.
    b_approx: Array1<f32>,
    /// `ny / std` per neuron, f64 for precision.
    scale: Array1<f64>,
    /// Normalized values `z_i = (x_i - mean) / std`, f64.
    z: Array1<f64>,
    /// `1.0 / n`.
    inv_n: f64,
}

impl LayerNormSamplingLinearization {
    /// Build the linearization from layer parameters and a center point.
    ///
    /// Returns `NumericalInstability` if `ny/std` or eval output overflows
    /// (#2901), matching the dense path's Jacobian non-finite guard.
    fn new(layer: &LayerNormLayer, x_center: &Array1<f32>) -> Result<Self> {
        let n = x_center.len();
        let nf = n as f64;
        let inv_n = 1.0 / nf;

        // f64 accumulation for mean/var to avoid catastrophic cancellation (#3325)
        let mean = x_center.iter().map(|&xi| xi as f64).sum::<f64>() * inv_n;
        let var = x_center
            .iter()
            .map(|&xi| {
                let d = xi as f64 - mean;
                d * d
            })
            .sum::<f64>()
            * inv_n;
        let std = (var + layer.eps as f64).sqrt();

        let z: Array1<f64> = x_center
            .iter()
            .map(|&xi| (xi as f64 - mean) / std)
            .collect();
        let scale: Array1<f64> = layer.ny.iter().map(|&g| g as f64 / std).collect();

        // Scale overflow guard (#2901): ny/std overflow when std is tiny.
        // The dense path materializes the Jacobian in f32, where ny/std > f32::MAX
        // overflows to Inf. Our f64 scale avoids the overflow, but the sampling
        // results become unreliable at extreme scales. Match the dense behavior:
        // return NumericalInstability when scale would overflow f32.
        if scale
            .iter()
            .any(|&s| !s.is_finite() || s.abs() > f32::MAX as f64)
        {
            return Err(NyError::NumericalInstability(
                "LayerNorm CROWN: non-finite Jacobian or eval output (ny/std overflow)".to_string(),
            ));
        }

        let y_center = layer.eval(x_center)?;
        if y_center.iter().any(|v| !v.is_finite()) {
            return Err(NyError::NumericalInstability(
                "LayerNorm CROWN: non-finite Jacobian or eval output (ny/std overflow)".to_string(),
            ));
        }

        // b_approx = y_center - J @ x_center, computed in f64 then cast.
        let jx_center = Self::apply_raw(&scale, &z, inv_n, x_center);
        let b_approx: Array1<f32> = y_center
            .iter()
            .zip(jx_center.iter())
            .map(|(&y, &jx)| (y as f64 - jx) as f32)
            .collect();

        Ok(Self {
            b_approx,
            scale,
            z,
            inv_n,
        })
    }

    /// `J @ v` using the low-rank identity:
    ///   `(J @ v)_i = s_i * (v_i - mean(v) - z_i * mean(z ⊙ v))`
    ///
    /// O(n) per application. All arithmetic in f64.
    fn apply(&self, v: &Array1<f32>) -> Array1<f64> {
        Self::apply_raw(&self.scale, &self.z, self.inv_n, v)
    }

    fn apply_raw(scale: &Array1<f64>, z: &Array1<f64>, inv_n: f64, v: &Array1<f32>) -> Array1<f64> {
        let n = v.len();
        let mut sum_v = 0.0_f64;
        let mut sum_zv = 0.0_f64;
        for i in 0..n {
            let vi = v[i] as f64;
            sum_v += vi;
            sum_zv += z[i] * vi;
        }
        let mean_v = sum_v * inv_n;
        let mean_zv = sum_zv * inv_n;

        let mut result = Array1::<f64>::zeros(n);
        for i in 0..n {
            result[i] = scale[i] * (v[i] as f64 - mean_v - z[i] * mean_zv);
        }
        result
    }

    /// `a @ J` using the low-rank identity:
    ///   `r = a ⊙ s`
    ///   `(a @ J)_j = r_j - mean(r) - mean(r ⊙ z) * z_j`
    ///
    /// O(n) per row. All arithmetic in f64. `a` is a slice of f64 coefficients
    /// (one row of the incoming bounds matrix).
    fn backprop_row(&self, a: &[f64]) -> Array1<f64> {
        let n = a.len();
        let mut sum_r = 0.0_f64;
        let mut sum_rz = 0.0_f64;
        for ((&a_i, &s_i), &z_i) in a.iter().zip(self.scale.iter()).zip(self.z.iter()) {
            let r_i = a_i * s_i;
            sum_r += r_i;
            sum_rz += r_i * z_i;
        }
        let mean_r = sum_r * self.inv_n;
        let mean_rz = sum_rz * self.inv_n;

        let mut result = Array1::<f64>::zeros(n);
        for i in 0..n {
            result[i] = a[i] * self.scale[i] - mean_r - mean_rz * self.z[i];
        }
        result
    }
}

/// Sampling-based CROWN scalar backward for LayerNorm using low-rank Jacobian.
///
/// Drop-in replacement for `sampling_crown_scalar()` on the LayerNorm
/// `Sampling` path. Same algorithm shape: non-finite guard → center-point
/// linearization → sampling error estimation → backward propagation with
/// per-row conservative fallback → directed rounding. Only the dense algebra
/// is replaced with O(n) low-rank operations.
pub(crate) fn sampling_crown_scalar_low_rank(
    layer: &LayerNormLayer,
    bounds: &LinearBounds,
    pre_lower: &Array1<f32>,
    pre_upper: &Array1<f32>,
) -> Result<LinearBounds> {
    let num_neurons = pre_lower.len();
    let num_outputs = bounds.num_outputs();

    // Non-finite pre-activation guard (#3259): when any dimension has infinite
    // or NaN bounds, center-point construction fails. Return trivially sound
    // constant bounds: A = 0, bias = [-inf, +inf].
    let has_non_finite =
        pre_lower.iter().any(|&v| !v.is_finite()) || pre_upper.iter().any(|&v| !v.is_finite());
    if has_non_finite {
        return LinearBounds::new_or_conservative(
            Array2::zeros((num_outputs, num_neurons)),
            Array1::from_elem(num_outputs, f32::NEG_INFINITY),
            Array2::zeros((num_outputs, num_neurons)),
            Array1::from_elem(num_outputs, f32::INFINITY),
        );
    }

    // Center point: l + (u - l) / 2 to avoid overflow when both are large finite.
    let x_center: Array1<f32> = pre_lower
        .iter()
        .zip(pre_upper.iter())
        .map(|(&l, &u)| l + (u - l) / 2.0)
        .collect();

    // Build low-rank linearization. Returns NumericalInstability on ny/std
    // overflow (#2901), matching the dense Jacobian non-finite guard.
    let lin = LayerNormSamplingLinearization::new(layer, &x_center)?;

    // Sampling-based error estimation using low-rank apply.
    let (max_error_above, max_error_below) = estimate_sampling_error_low_rank(
        layer,
        num_neurons,
        pre_lower,
        pre_upper,
        &x_center,
        &lin,
    )?;

    // Safety margin (50% extra) and minimum floor (1e-6).
    let (max_error_above, max_error_below) = apply_safety_margin(max_error_above, max_error_below);

    // Backward propagation with low-rank backprop_row.
    backward_propagate_low_rank(
        num_outputs,
        num_neurons,
        bounds,
        &lin,
        &max_error_above,
        &max_error_below,
    )
}

/// Sampling-based error estimation: 3-level grid + axis-aligned + hash-random.
///
/// Identical sampling strategy to `crown_common::estimate_sampling_error`, but
/// uses the low-rank `lin.apply()` for the linear approximation instead of the
/// dense `jacobian.dot()`.
fn estimate_sampling_error_low_rank(
    layer: &LayerNormLayer,
    num_neurons: usize,
    pre_lower: &Array1<f32>,
    pre_upper: &Array1<f32>,
    x_center: &Array1<f32>,
    lin: &LayerNormSamplingLinearization,
) -> Result<(Array1<f32>, Array1<f32>)> {
    let max_grid_dims = 6; // 3^6 = 729 grid points max
    let num_grid = if num_neurons <= max_grid_dims {
        3_usize.pow(num_neurons as u32)
    } else {
        0
    };
    let num_axis_aligned = if num_grid > 0 { 0 } else { num_neurons * 2 };
    let num_random = 50;
    let num_samples = (num_grid + num_axis_aligned + num_random).max(50);

    let mut max_error_above = Array1::<f32>::zeros(num_neurons);
    let mut max_error_below = Array1::<f32>::zeros(num_neurons);

    let mut x_sample = x_center.clone();
    for sample_idx in 0..num_samples {
        if sample_idx < num_grid {
            // 3-level grid: {lower, center, upper}^n
            let mut grid_idx = sample_idx;
            for i in 0..num_neurons {
                let level = grid_idx % 3;
                grid_idx /= 3;
                x_sample[i] = match level {
                    0 => pre_lower[i],
                    1 => pre_lower[i] + (pre_upper[i] - pre_lower[i]) * 0.5,
                    _ => pre_upper[i],
                };
            }
        } else if sample_idx < num_grid + num_axis_aligned {
            // Axis-aligned corner sampling (fallback for large n)
            x_sample.assign(x_center);
            let offset = sample_idx - num_grid;
            let dim = offset / 2;
            if dim < num_neurons {
                x_sample[dim] = if offset % 2 == 0 {
                    pre_lower[dim]
                } else {
                    pre_upper[dim]
                };
            }
        } else {
            // Hash-based pseudo-random sampling (deterministic)
            for i in 0..num_neurons {
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x_sample[i] = pre_lower[i] + (pre_upper[i] - pre_lower[i]) * t;
            }
        }

        let y_actual = layer.eval(&x_sample)?;
        // Low-rank linear approximation: J @ x_sample + b_approx
        let jx_sample = lin.apply(&x_sample);
        let y_approx: Array1<f32> = jx_sample
            .iter()
            .zip(lin.b_approx.iter())
            .map(|(&jx, &b)| (jx + b as f64) as f32)
            .collect();

        for i in 0..num_neurons {
            let error = y_actual[i] - y_approx[i];
            if error > max_error_above[i] {
                max_error_above[i] = error;
            }
            if -error > max_error_below[i] {
                max_error_below[i] = -error;
            }
        }
    }

    Ok((max_error_above, max_error_below))
}

/// Apply safety margin (50% extra) and minimum floor (1e-6).
fn apply_safety_margin(
    mut max_error_above: Array1<f32>,
    mut max_error_below: Array1<f32>,
) -> (Array1<f32>, Array1<f32>) {
    let safety_factor = 1.5;
    let min_margin = 1e-6_f32;
    for i in 0..max_error_above.len() {
        max_error_above[i] *= safety_factor;
        max_error_below[i] *= safety_factor;
        if max_error_above[i] < min_margin {
            max_error_above[i] = min_margin;
        }
        if max_error_below[i] < min_margin {
            max_error_below[i] = min_margin;
        }
    }
    (max_error_above, max_error_below)
}

/// Backward propagation using low-rank `backprop_row` for A-coefficients
/// and per-neuron sign-gated bias accumulation.
///
/// Per-row non-finite guard (#3128, #3027) widens affected rows to
/// conservative bounds. Directed rounding on final f32 bias cast (#1992, #2164).
fn backward_propagate_low_rank(
    num_outputs: usize,
    num_neurons: usize,
    bounds: &LinearBounds,
    lin: &LayerNormSamplingLinearization,
    max_error_above: &Array1<f32>,
    max_error_below: &Array1<f32>,
) -> Result<LinearBounds> {
    let mut new_lower_a_f64 = Array2::<f64>::zeros((num_outputs, num_neurons));
    let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
    let mut new_upper_a_f64 = Array2::<f64>::zeros((num_outputs, num_neurons));
    let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

    for j in 0..num_outputs {
        // Bias accumulation: sign-dependent error adjustment per neuron.
        // Skip zero coefficients to avoid 0*inf NaN (#1739).
        for i in 0..num_neurons {
            let la = bounds.lower_a()[[j, i]];
            let ua = bounds.upper_a()[[j, i]];

            if la > 0.0 {
                new_lower_b_f64[j] +=
                    la as f64 * (lin.b_approx[i] as f64 - max_error_below[i] as f64);
            } else if la < 0.0 {
                new_lower_b_f64[j] +=
                    la as f64 * (lin.b_approx[i] as f64 + max_error_above[i] as f64);
            }

            if ua > 0.0 {
                new_upper_b_f64[j] +=
                    ua as f64 * (lin.b_approx[i] as f64 + max_error_above[i] as f64);
            } else if ua < 0.0 {
                new_upper_b_f64[j] +=
                    ua as f64 * (lin.b_approx[i] as f64 - max_error_below[i] as f64);
            }
        }

        // A-coefficient update: new_A[j, :] = bounds.A[j, :] @ J
        // Low-rank backprop_row gives O(n) instead of O(n²).
        let lower_row: Vec<f64> = (0..num_neurons)
            .map(|i| bounds.lower_a()[[j, i]] as f64)
            .collect();
        let upper_row: Vec<f64> = (0..num_neurons)
            .map(|i| bounds.upper_a()[[j, i]] as f64)
            .collect();

        let new_lower_row = lin.backprop_row(&lower_row);
        let new_upper_row = lin.backprop_row(&upper_row);

        for k in 0..num_neurons {
            new_lower_a_f64[[j, k]] = new_lower_row[k];
            new_upper_a_f64[[j, k]] = new_upper_row[k];
        }

        // Per-row unsafe coefficient guard (#3128, #3027, #3228): when Inf
        // coefficients from compose() produce non-finite accumulation, widen
        // the affected row to conservative bounds. Uses
        // is_crown_coeff_safe_f64() (finite + magnitude ≤ CROWN_COEFF_MAX).
        let lower_row_nonfinite =
            (0..num_neurons).any(|k| !is_crown_coeff_safe_f64(new_lower_a_f64[[j, k]]));
        let upper_row_nonfinite =
            (0..num_neurons).any(|k| !is_crown_coeff_safe_f64(new_upper_a_f64[[j, k]]));
        if lower_row_nonfinite {
            for k in 0..num_neurons {
                new_lower_a_f64[[j, k]] = 0.0;
            }
            new_lower_b_f64[j] = f64::NEG_INFINITY;
        }
        if upper_row_nonfinite {
            for k in 0..num_neurons {
                new_upper_a_f64[[j, k]] = 0.0;
            }
            new_upper_b_f64[j] = f64::INFINITY;
        }
    }

    // A-matrix: standard f64→f32 rounding (round-to-nearest), matching
    // alpha-beta-CROWN. Directed rounding on A is not unconditionally sound
    // because the sign of the coefficient determines which direction is
    // conservative during concretization (#2208).
    LinearBounds::new_or_conservative(
        new_lower_a_f64.mapv(|x| x as f32),
        new_lower_b_f64.mapv(|x| next_down_f32(x as f32)),
        new_upper_a_f64.mapv(|x| x as f32),
        new_upper_b_f64.mapv(|x| next_up_f32(x as f32)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn custom_layer(ny: &[f32], beta: &[f32]) -> LayerNormLayer {
        LayerNormLayer::new(
            Array1::from_vec(ny.to_vec()),
            Array1::from_vec(beta.to_vec()),
            1e-5,
        )
        .unwrap()
    }

    /// Verify low-rank `apply(v)` matches dense `jacobian.dot(&v)`.
    #[ntest::timeout(10000)]
    #[test]
    fn test_layernorm_sampling_low_rank_apply_matches_dense_1957() {
        let layer = custom_layer(&[2.0, 0.5, 1.5], &[0.1, -0.2, 0.3]);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let lin = LayerNormSamplingLinearization::new(&layer, &x).unwrap();
        let jacobian = layer.jacobian(&x).unwrap();

        // Test with several different vectors
        for v in &[
            arr1(&[1.0, 0.0, 0.0]),
            arr1(&[0.0, 1.0, 0.0]),
            arr1(&[0.0, 0.0, 1.0]),
            arr1(&[1.0, 2.0, 3.0]),
            arr1(&[-0.5, 0.3, 0.7]),
        ] {
            let low_rank_result = lin.apply(v);
            let dense_result = jacobian.dot(v);

            for i in 0..3 {
                let lr = low_rank_result[i];
                let dr = dense_result[i] as f64;
                assert!(
                    (lr - dr).abs() < 1e-6,
                    "apply mismatch at [{i}]: low_rank={lr}, dense={dr}, v={v:?}"
                );
            }
        }
    }

    /// Verify low-rank `backprop_row(a)` matches dense `a.dot(&jacobian)`.
    #[ntest::timeout(10000)]
    #[test]
    fn test_layernorm_sampling_low_rank_backprop_row_matches_dense_1957() {
        let layer = custom_layer(&[2.0, 0.5, 1.5], &[0.1, -0.2, 0.3]);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let lin = LayerNormSamplingLinearization::new(&layer, &x).unwrap();
        let jacobian = layer.jacobian(&x).unwrap();

        for a in &[
            vec![1.0_f64, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 2.0, 3.0],
            vec![-0.5, 0.3, 0.7],
        ] {
            let low_rank_result = lin.backprop_row(a);
            // Dense: a @ J
            let a_f32: Array1<f32> = a.iter().map(|&v| v as f32).collect();
            let dense_result = a_f32.dot(&jacobian);

            for j in 0..3 {
                let lr = low_rank_result[j];
                let dr = dense_result[j] as f64;
                assert!(
                    (lr - dr).abs() < 1e-6,
                    "backprop_row mismatch at [{j}]: low_rank={lr}, dense={dr}, a={a:?}"
                );
            }
        }
    }
}
