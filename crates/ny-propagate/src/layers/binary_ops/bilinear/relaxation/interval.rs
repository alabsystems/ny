// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::IxDyn;
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};

use super::BilinearRelaxation;
use crate::bounds::safe_math::interval_mul_for_bounds;

impl BilinearRelaxation {
    /// Compose per-batch McCormick coefficients with downstream CROWN bounds.
    ///
    /// Given downstream A: [batch..., out_dim, m*n] (flattened bilinear output),
    /// computes:
    ///   A_q[b, o, i*k+l] = sum_j interval_compose(A[b,o,i,j], alpha[b,i,j,l])
    ///   A_k[b, o, k_idx]  = sum_i interval_compose(A[b,o,i,j], beta[b,i,j,l])
    ///   bias[b, o]         = sum_{i,j} interval_compose(A[b,o,i,j], ny[b,i,j])
    ///
    /// Returns (bounds_for_Q, bounds_for_K) with bias split equally between paths.
    ///
    /// Interval-mul baseline using per-batch coefficients `alpha[b,...]`.
    ///
    /// Note: Production code uses `compose_backward_broadcast` (sign-split, tighter).
    /// This method is retained for test comparison (interval-mul baseline).
    pub(crate) fn compose_backward(
        &self,
        downstream: &crate::BatchedLinearBounds,
    ) -> Result<(crate::BatchedLinearBounds, crate::BatchedLinearBounds)> {
        let ds_a_shape = downstream.lower_a().shape();
        let ds_ndim = ds_a_shape.len();
        if ds_ndim < 2 {
            return Err(NyError::InvalidSpec(
                "BilinearRelaxation::compose_backward: downstream must be >= 2D".to_string(),
            ));
        }

        let ds_batch = &ds_a_shape[..ds_ndim - 2];
        let out_dim = ds_a_shape[ds_ndim - 2];
        let z_size = ds_a_shape[ds_ndim - 1];

        if z_size != self.m * self.n {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.m * self.n],
                got: vec![z_size],
            });
        }

        let ds_batch_size: usize = checked_shape_product(ds_batch)
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "BilinearRelaxation::compose_backward: batch overflow".to_string(),
                )
            })?
            .max(1);

        let relax_batch_size = self.alpha_lower.shape()[0];
        let out_batch_size = if ds_batch_size == 1 {
            relax_batch_size
        } else if relax_batch_size == 1 || ds_batch_size == relax_batch_size {
            ds_batch_size
        } else {
            return Err(NyError::ShapeMismatch {
                expected: vec![relax_batch_size],
                got: vec![ds_batch_size],
            });
        };

        let q_size = self.m * self.k;
        // K operand flat size: layout is [n, k] when transpose_b, [k, n]
        // otherwise — the element count is the same either way.
        let k_size = self.k * self.n;

        let ds_lower_a = downstream
            .lower_a()
            .view()
            .into_shape_with_order((ds_batch_size, out_dim, self.m, self.n))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearRelaxation: reshape downstream lower_a: {e}"
                ))
            })?;
        let ds_upper_a = downstream
            .upper_a()
            .view()
            .into_shape_with_order((ds_batch_size, out_dim, self.m, self.n))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearRelaxation: reshape downstream upper_a: {e}"
                ))
            })?;

        let mut q_lower_a = ndarray::Array3::<f32>::zeros((out_batch_size, out_dim, q_size));
        let mut q_upper_a = ndarray::Array3::<f32>::zeros((out_batch_size, out_dim, q_size));
        let mut k_lower_a = ndarray::Array3::<f32>::zeros((out_batch_size, out_dim, k_size));
        let mut k_upper_a = ndarray::Array3::<f32>::zeros((out_batch_size, out_dim, k_size));

        for b in 0..out_batch_size {
            let ds_b = if ds_batch_size == 1 { 0 } else { b };
            let relax_b = if relax_batch_size == 1 { 0 } else { b };
            for o in 0..out_dim {
                for i in 0..self.m {
                    for l in 0..self.k {
                        let q_idx = i * self.k + l;
                        let mut lower_sum = 0.0_f64;
                        let mut upper_sum = 0.0_f64;

                        for j in 0..self.n {
                            let a_l = ds_lower_a[[ds_b, o, i, j]];
                            let a_u = ds_upper_a[[ds_b, o, i, j]];
                            let coeff_l = self.alpha_lower[[relax_b, i, j, l]];
                            let coeff_u = self.alpha_upper[[relax_b, i, j, l]];

                            let (prod_l, prod_u) =
                                interval_mul_for_bounds(a_l, a_u, coeff_l, coeff_u);
                            lower_sum += prod_l as f64;
                            upper_sum += prod_u as f64;
                        }

                        q_lower_a[[b, o, q_idx]] = if lower_sum.is_nan() {
                            f32::NEG_INFINITY
                        } else {
                            next_down_f32(lower_sum as f32)
                        };
                        q_upper_a[[b, o, q_idx]] = if upper_sum.is_nan() {
                            f32::INFINITY
                        } else {
                            next_up_f32(upper_sum as f32)
                        };
                    }
                }

                for j in 0..self.n {
                    for l in 0..self.k {
                        let k_idx = if self.transpose_b {
                            j * self.k + l
                        } else {
                            l * self.n + j
                        };
                        let mut lower_sum = 0.0_f64;
                        let mut upper_sum = 0.0_f64;

                        for i in 0..self.m {
                            let a_l = ds_lower_a[[ds_b, o, i, j]];
                            let a_u = ds_upper_a[[ds_b, o, i, j]];
                            let coeff_l = self.beta_lower[[relax_b, i, j, l]];
                            let coeff_u = self.beta_upper[[relax_b, i, j, l]];

                            let (prod_l, prod_u) =
                                interval_mul_for_bounds(a_l, a_u, coeff_l, coeff_u);
                            lower_sum += prod_l as f64;
                            upper_sum += prod_u as f64;
                        }

                        k_lower_a[[b, o, k_idx]] = if lower_sum.is_nan() {
                            f32::NEG_INFINITY
                        } else {
                            next_down_f32(lower_sum as f32)
                        };
                        k_upper_a[[b, o, k_idx]] = if upper_sum.is_nan() {
                            f32::INFINITY
                        } else {
                            next_up_f32(upper_sum as f32)
                        };
                    }
                }
            }
        }

        let ds_lower_b = downstream.lower_b();
        let ds_upper_b = downstream.upper_b();
        let ds_b_shape = ds_lower_b.shape();
        let b_batch_size = if ds_b_shape.len() > 1 {
            ds_b_shape[..ds_b_shape.len() - 1]
                .iter()
                .product::<usize>()
                .max(1)
        } else {
            1
        };
        let b_out = *ds_b_shape.last().unwrap_or(&1);

        let ds_lb_flat = ds_lower_b
            .view()
            .into_shape_with_order((b_batch_size, b_out))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearRelaxation: reshape ds bias: {e}"))
            })?;
        let ds_ub_flat = ds_upper_b
            .view()
            .into_shape_with_order((b_batch_size, b_out))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearRelaxation: reshape ds bias: {e}"))
            })?;

        let mut total_bias_lower = ndarray::Array2::<f32>::zeros((out_batch_size, out_dim));
        let mut total_bias_upper = ndarray::Array2::<f32>::zeros((out_batch_size, out_dim));

        for b in 0..out_batch_size {
            let ds_b = if ds_batch_size == 1 { 0 } else { b };
            let relax_b = if relax_batch_size == 1 { 0 } else { b };
            let ds_bias_b = if b_batch_size == 1 { 0 } else { b };
            for o in 0..out_dim {
                let mut bias_l_sum = 0.0_f64;
                let mut bias_u_sum = 0.0_f64;

                for i in 0..self.m {
                    for j in 0..self.n {
                        let a_l = ds_lower_a[[ds_b, o, i, j]];
                        let a_u = ds_upper_a[[ds_b, o, i, j]];
                        let g_l = self.bias_lower[[relax_b, i, j]];
                        let g_u = self.bias_upper[[relax_b, i, j]];

                        let (prod_l, prod_u) = interval_mul_for_bounds(a_l, a_u, g_l, g_u);
                        bias_l_sum += prod_l as f64;
                        bias_u_sum += prod_u as f64;
                    }
                }

                bias_l_sum += ds_lb_flat[[ds_bias_b, o]] as f64;
                bias_u_sum += ds_ub_flat[[ds_bias_b, o]] as f64;

                total_bias_lower[[b, o]] = if bias_l_sum.is_nan() {
                    f32::NEG_INFINITY
                } else {
                    next_down_f32(bias_l_sum as f32)
                };
                total_bias_upper[[b, o]] = if bias_u_sum.is_nan() {
                    f32::INFINITY
                } else {
                    next_up_f32(bias_u_sum as f32)
                };
            }
        }

        let half_bias_lower = total_bias_lower.mapv(|v| next_down_f32(v * 0.5));
        let half_bias_upper = total_bias_upper.mapv(|v| next_up_f32(v * 0.5));

        let out_batch_dims: Vec<usize> = if relax_batch_size == 1 && ds_batch_size > 1 {
            ds_batch.to_vec()
        } else {
            self.batch_dims.clone()
        };
        let out_batch_dims = &out_batch_dims;
        let mut q_a_shape: Vec<usize> = out_batch_dims.clone();
        q_a_shape.push(out_dim);
        q_a_shape.push(q_size);
        let mut bias_shape: Vec<usize> = out_batch_dims.clone();
        bias_shape.push(out_dim);
        let mut k_a_shape: Vec<usize> = out_batch_dims.clone();
        k_a_shape.push(out_dim);
        k_a_shape.push(k_size);

        let q_la = q_lower_a
            .into_shape_with_order(IxDyn(&q_a_shape))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearRelaxation: reshape q_lower_a: {e}"))
            })?;
        let q_ua = q_upper_a
            .into_shape_with_order(IxDyn(&q_a_shape))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearRelaxation: reshape q_upper_a: {e}"))
            })?;
        let k_la = k_lower_a
            .into_shape_with_order(IxDyn(&k_a_shape))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearRelaxation: reshape k_lower_a: {e}"))
            })?;
        let k_ua = k_upper_a
            .into_shape_with_order(IxDyn(&k_a_shape))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearRelaxation: reshape k_upper_a: {e}"))
            })?;
        let h_bl = half_bias_lower
            .into_shape_with_order(IxDyn(&bias_shape))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearRelaxation: reshape bias_lower: {e}"))
            })?;
        let h_bu = half_bias_upper
            .into_shape_with_order(IxDyn(&bias_shape))
            .map_err(|e| {
                NyError::InternalError(format!("BilinearRelaxation: reshape bias_upper: {e}"))
            })?;

        let q_input_shape = vec![self.m, self.k];
        let k_input_shape = if self.transpose_b {
            vec![self.n, self.k]
        } else {
            vec![self.k, self.n]
        };
        let output_shape = downstream.output_shape().to_vec();

        let bounds_q = crate::BatchedLinearBounds::new_or_conservative(
            q_la,
            h_bl.clone(),
            q_ua,
            h_bu.clone(),
            q_input_shape,
            output_shape.clone(),
        )?;
        let bounds_k = crate::BatchedLinearBounds::new_or_conservative(
            k_la,
            h_bl,
            k_ua,
            h_bu,
            k_input_shape,
            output_shape,
        )?;

        Ok((bounds_q, bounds_k))
    }
}
