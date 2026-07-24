// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::IxDyn;
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};

use super::BilinearRelaxation;

impl BilinearRelaxation {
    /// Compose backward through bilinear layer using sign-split matmul.
    ///
    /// Optimized version of `compose_backward` that:
    /// 1. Uses sign-split composition (auto_LiRPA `propagate_A_xy` style) instead of
    ///    interval multiplication — tighter bounds by processing each bound direction
    ///    independently rather than treating downstream A as an interval.
    /// 2. Uses ndarray `.dot()` matrix multiplication for contraction loops instead of
    ///    element-by-element indexing — enables BLAS acceleration.
    ///
    /// Mathematical justification for sign-split vs interval-mul:
    /// In CROWN backward, `lower(y) >= A_L @ z + b_L`. For each z[j]:
    /// - If `A_L[i,j] >= 0`: use lower McCormick (`alpha_l`) for z[j] lower bound
    /// - If `A_L[i,j] < 0`: use upper McCormick (`alpha_u`) for z[j] upper bound
    ///
    /// This is sound because `A_L` is a known matrix for the lower direction, not an
    /// interval `[A_L, A_U]`. The `interval_mul` approach over-approximates by treating
    /// both A and alpha as intervals simultaneously.
    ///
    /// Reference: auto_LiRPA `operators/linear.py:490-510` (`propagate_A_xy`)
    /// Design: `designs/2026-03-04-286-attention-bilinear-alternative.md` Approach A
    pub(crate) fn compose_backward_broadcast(
        &self,
        downstream: &crate::BatchedLinearBounds,
    ) -> Result<(crate::BatchedLinearBounds, crate::BatchedLinearBounds)> {
        self.compose_backward_broadcast_inner(downstream, None)
    }

    /// Compose with direction-dependent McCormick relaxation.
    ///
    /// Uses `self` for the lower-bound direction (processing A_L) and `upper_dir`
    /// for the upper-bound direction (processing A_U). This enables 4 independent
    /// alpha parameters per McCormick element: r_l and r_u can differ between
    /// the lower-bound and upper-bound computation paths.
    ///
    /// # Why direction-dependent?
    ///
    /// In auto_LiRPA, the alpha tensor has shape [4, ...] where:
    /// - [0]: r_l for lower-bound direction (used with pos(A_L) to select lower face)
    /// - [1]: r_l for upper-bound direction (used with neg(A_U) to select lower face)
    /// - [2]: r_u for lower-bound direction (used with neg(A_L) to select upper face)
    /// - [3]: r_u for upper-bound direction (used with pos(A_U) to select upper face)
    ///
    /// Different face selections can be optimal for lower vs upper bound computation
    /// because the downstream structure that consumes these bounds is different.
    ///
    /// Reference: auto_LiRPA operators/bivariate.py:306-315 (sign-dependent dispatch)
    pub(crate) fn compose_backward_broadcast_bidirectional(
        &self,
        upper_dir: &Self,
        downstream: &crate::BatchedLinearBounds,
    ) -> Result<(crate::BatchedLinearBounds, crate::BatchedLinearBounds)> {
        self.compose_backward_broadcast_inner(downstream, Some(upper_dir))
    }

    /// Inner implementation for compose_backward_broadcast with optional
    /// direction-dependent upper relaxation.
    ///
    /// When `upper_dir` is None, `self` is used for both lower and upper directions
    /// (standard single-relaxation mode). When Some, the upper direction uses the
    /// provided relaxation's coefficients.
    fn compose_backward_broadcast_inner(
        &self,
        downstream: &crate::BatchedLinearBounds,
        upper_dir: Option<&Self>,
    ) -> Result<(crate::BatchedLinearBounds, crate::BatchedLinearBounds)> {
        let upper_relax = upper_dir.unwrap_or(self);

        let ds_a_shape = downstream.lower_a().shape();
        let ds_ndim = ds_a_shape.len();
        if ds_ndim < 2 {
            return Err(NyError::InvalidSpec(
                "BilinearRelaxation::compose_backward_broadcast: downstream must be >= 2D"
                    .to_string(),
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
                    "BilinearRelaxation::compose_backward_broadcast: batch overflow".to_string(),
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

        if let Some(ud) = upper_dir {
            let upper_batch_size = ud.alpha_lower.shape()[0];
            if upper_batch_size != relax_batch_size {
                return Err(NyError::ShapeMismatch {
                    expected: vec![relax_batch_size],
                    got: vec![upper_batch_size],
                });
            }
            if ud.m != self.m || ud.n != self.n || ud.k != self.k {
                return Err(NyError::ShapeMismatch {
                    expected: vec![self.m, self.n, self.k],
                    got: vec![ud.m, ud.n, ud.k],
                });
            }
        }

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
                    "BilinearRelaxation broadcast: reshape downstream lower_a: {e}"
                ))
            })?;
        let ds_upper_a = downstream
            .upper_a()
            .view()
            .into_shape_with_order((ds_batch_size, out_dim, self.m, self.n))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearRelaxation broadcast: reshape downstream upper_a: {e}"
                ))
            })?;

        let mut q_lower_a = ndarray::Array3::<f32>::zeros((out_batch_size, out_dim, q_size));
        let mut q_upper_a = ndarray::Array3::<f32>::zeros((out_batch_size, out_dim, q_size));
        let mut k_lower_a = ndarray::Array3::<f32>::zeros((out_batch_size, out_dim, k_size));
        let mut k_upper_a = ndarray::Array3::<f32>::zeros((out_batch_size, out_dim, k_size));

        for b in 0..out_batch_size {
            let ds_b = if ds_batch_size == 1 { 0 } else { b };
            let relax_b = if relax_batch_size == 1 { 0 } else { b };

            for i in 0..self.m {
                let a_l_slice = ds_lower_a.slice(ndarray::s![ds_b, .., i, ..]);
                let a_u_slice = ds_upper_a.slice(ndarray::s![ds_b, .., i, ..]);

                let a_l_pos = a_l_slice.mapv(|v: f32| (v.max(0.0)) as f64);
                let a_l_neg = a_l_slice.mapv(|v: f32| (v.min(0.0)) as f64);
                let a_u_pos = a_u_slice.mapv(|v: f32| (v.max(0.0)) as f64);
                let a_u_neg = a_u_slice.mapv(|v: f32| (v.min(0.0)) as f64);

                let alpha_l = self.alpha_lower.slice(ndarray::s![relax_b, i, .., ..]);
                let alpha_u = self.alpha_upper.slice(ndarray::s![relax_b, i, .., ..]);
                let alpha_l_f64 = alpha_l.mapv(|v| v as f64);
                let alpha_u_f64 = alpha_u.mapv(|v| v as f64);

                let alpha_l_upper = upper_relax
                    .alpha_lower
                    .slice(ndarray::s![relax_b, i, .., ..]);
                let alpha_u_upper = upper_relax
                    .alpha_upper
                    .slice(ndarray::s![relax_b, i, .., ..]);
                let alpha_l_upper_f64 = alpha_l_upper.mapv(|v| v as f64);
                let alpha_u_upper_f64 = alpha_u_upper.mapv(|v| v as f64);

                let q_lower_ik = a_l_pos.dot(&alpha_l_f64) + a_l_neg.dot(&alpha_u_f64);
                let q_upper_ik = a_u_pos.dot(&alpha_u_upper_f64) + a_u_neg.dot(&alpha_l_upper_f64);

                let q_offset = i * self.k;
                for o in 0..out_dim {
                    for l in 0..self.k {
                        let vl = q_lower_ik[[o, l]];
                        q_lower_a[[b, o, q_offset + l]] = if vl.is_nan() {
                            f32::NEG_INFINITY
                        } else {
                            next_down_f32(vl as f32)
                        };
                        let vu = q_upper_ik[[o, l]];
                        q_upper_a[[b, o, q_offset + l]] = if vu.is_nan() {
                            f32::INFINITY
                        } else {
                            next_up_f32(vu as f32)
                        };
                    }
                }
            }

            for j in 0..self.n {
                let a_l_slice = ds_lower_a.slice(ndarray::s![ds_b, .., .., j]);
                let a_u_slice = ds_upper_a.slice(ndarray::s![ds_b, .., .., j]);

                let a_l_pos = a_l_slice.mapv(|v: f32| (v.max(0.0)) as f64);
                let a_l_neg = a_l_slice.mapv(|v: f32| (v.min(0.0)) as f64);
                let a_u_pos = a_u_slice.mapv(|v: f32| (v.max(0.0)) as f64);
                let a_u_neg = a_u_slice.mapv(|v: f32| (v.min(0.0)) as f64);

                let beta_l = self.beta_lower.slice(ndarray::s![relax_b, .., j, ..]);
                let beta_u = self.beta_upper.slice(ndarray::s![relax_b, .., j, ..]);
                let beta_l_f64 = beta_l.mapv(|v| v as f64);
                let beta_u_f64 = beta_u.mapv(|v| v as f64);

                let beta_l_upper = upper_relax
                    .beta_lower
                    .slice(ndarray::s![relax_b, .., j, ..]);
                let beta_u_upper = upper_relax
                    .beta_upper
                    .slice(ndarray::s![relax_b, .., j, ..]);
                let beta_l_upper_f64 = beta_l_upper.mapv(|v| v as f64);
                let beta_u_upper_f64 = beta_u_upper.mapv(|v| v as f64);

                let k_lower_jk = a_l_pos.dot(&beta_l_f64) + a_l_neg.dot(&beta_u_f64);
                let k_upper_jk = a_u_pos.dot(&beta_u_upper_f64) + a_u_neg.dot(&beta_l_upper_f64);

                for o in 0..out_dim {
                    for l in 0..self.k {
                        let k_idx = if self.transpose_b {
                            j * self.k + l
                        } else {
                            l * self.n + j
                        };
                        let vl = k_lower_jk[[o, l]];
                        k_lower_a[[b, o, k_idx]] = if vl.is_nan() {
                            f32::NEG_INFINITY
                        } else {
                            next_down_f32(vl as f32)
                        };
                        let vu = k_upper_jk[[o, l]];
                        k_upper_a[[b, o, k_idx]] = if vu.is_nan() {
                            f32::INFINITY
                        } else {
                            next_up_f32(vu as f32)
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
                NyError::InternalError(format!(
                    "BilinearRelaxation broadcast: reshape ds bias: {e}"
                ))
            })?;
        let ds_ub_flat = ds_upper_b
            .view()
            .into_shape_with_order((b_batch_size, b_out))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearRelaxation broadcast: reshape ds bias: {e}"
                ))
            })?;

        let mut total_bias_lower = ndarray::Array2::<f32>::zeros((out_batch_size, out_dim));
        let mut total_bias_upper = ndarray::Array2::<f32>::zeros((out_batch_size, out_dim));

        for b in 0..out_batch_size {
            let ds_b_idx = if ds_batch_size == 1 { 0 } else { b };
            let relax_b = if relax_batch_size == 1 { 0 } else { b };
            let ds_bias_b = if b_batch_size == 1 { 0 } else { b };

            let g_l = self.bias_lower.slice(ndarray::s![relax_b, .., ..]);
            let g_u = self.bias_upper.slice(ndarray::s![relax_b, .., ..]);
            let g_l_f64 = g_l.mapv(|v| v as f64);
            let g_u_f64 = g_u.mapv(|v| v as f64);

            let g_l_upper = upper_relax.bias_lower.slice(ndarray::s![relax_b, .., ..]);
            let g_u_upper = upper_relax.bias_upper.slice(ndarray::s![relax_b, .., ..]);
            let g_l_upper_f64 = g_l_upper.mapv(|v| v as f64);
            let g_u_upper_f64 = g_u_upper.mapv(|v| v as f64);

            for o in 0..out_dim {
                let a_l_slice = ds_lower_a.slice(ndarray::s![ds_b_idx, o, .., ..]);
                let a_u_slice = ds_upper_a.slice(ndarray::s![ds_b_idx, o, .., ..]);

                let a_l_pos = a_l_slice.mapv(|v: f32| (v.max(0.0)) as f64);
                let a_l_neg = a_l_slice.mapv(|v: f32| (v.min(0.0)) as f64);
                let a_u_pos = a_u_slice.mapv(|v: f32| (v.max(0.0)) as f64);
                let a_u_neg = a_u_slice.mapv(|v: f32| (v.min(0.0)) as f64);

                let bias_l_sum = (&a_l_pos * &g_l_f64).sum()
                    + (&a_l_neg * &g_u_f64).sum()
                    + ds_lb_flat[[ds_bias_b, o]] as f64;
                let bias_u_sum = (&a_u_pos * &g_u_upper_f64).sum()
                    + (&a_u_neg * &g_l_upper_f64).sum()
                    + ds_ub_flat[[ds_bias_b, o]] as f64;

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
        let mut q_a_shape: Vec<usize> = out_batch_dims.clone();
        q_a_shape.push(out_dim);
        q_a_shape.push(q_size);
        let mut bias_shape: Vec<usize> = out_batch_dims.clone();
        bias_shape.push(out_dim);
        let mut k_a_shape: Vec<usize> = out_batch_dims;
        k_a_shape.push(out_dim);
        k_a_shape.push(k_size);

        let q_la = q_lower_a
            .into_shape_with_order(IxDyn(&q_a_shape))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearRelaxation broadcast: reshape q_lower_a: {e}"
                ))
            })?;
        let q_ua = q_upper_a
            .into_shape_with_order(IxDyn(&q_a_shape))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearRelaxation broadcast: reshape q_upper_a: {e}"
                ))
            })?;
        let k_la = k_lower_a
            .into_shape_with_order(IxDyn(&k_a_shape))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearRelaxation broadcast: reshape k_lower_a: {e}"
                ))
            })?;
        let k_ua = k_upper_a
            .into_shape_with_order(IxDyn(&k_a_shape))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearRelaxation broadcast: reshape k_upper_a: {e}"
                ))
            })?;
        let h_bl = half_bias_lower
            .into_shape_with_order(IxDyn(&bias_shape))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearRelaxation broadcast: reshape bias_lower: {e}"
                ))
            })?;
        let h_bu = half_bias_upper
            .into_shape_with_order(IxDyn(&bias_shape))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "BilinearRelaxation broadcast: reshape bias_upper: {e}"
                ))
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
