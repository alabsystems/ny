// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::super::matmul::{decode_batch_index_into_buf, parse_matmul_dims, MatMulLayer};
use super::super::validate_mccormick_inputs;
use super::{interpolated_mccormick, BilinearCrownLayer};

impl BilinearCrownLayer {
    /// CROWN backward propagation for LinearBounds (flattened graph networks).
    ///
    /// Delegates to MatMulLayer's McCormick relaxation implementation.
    /// This enables BilinearCrown to work with DAG-CROWN graph networks
    /// that use `LinearBounds` instead of `BatchedLinearBounds`.
    ///
    /// # Arguments
    /// * `bounds` - Incoming LinearBounds on the output C
    /// * `input_a_bounds` - IBP bounds on input A (Q in attention)
    /// * `input_b_bounds` - IBP bounds on input B (K in attention)
    ///
    /// # Returns
    /// Two sets of linear bounds: one for A input, one for B input.
    pub fn propagate_linear_binary(
        &self,
        bounds: &crate::LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(crate::LinearBounds, crate::LinearBounds)> {
        let matmul = MatMulLayer::new(self.transpose_b, self.scale);
        matmul.propagate_linear_binary(bounds, input_a_bounds, input_b_bounds)
    }

    /// CROWN backward propagation with alpha-parameterized McCormick interpolation.
    ///
    /// Like `propagate_linear_binary`, but uses `interpolated_mccormick` with
    /// optimizable `r_l, r_u ∈ [0,1]` per-element instead of a fixed midpoint
    /// heuristic. This enables joint ReLU + bilinear alpha optimization in
    /// the DAG α-CROWN loop (#3287).
    ///
    /// Alpha shape: `[4, m, n, k]` where m, n, k are the matmul dimensions.
    /// - `[0, i, j, l]`: r_l for lower bound computation
    /// - `[1, i, j, l]`: r_l for upper bound computation (direction-dependent)
    /// - `[2, i, j, l]`: r_u for lower bound computation
    /// - `[3, i, j, l]`: r_u for upper bound computation (direction-dependent)
    ///
    /// When `alphas` is None, delegates to the fixed `propagate_linear_binary`.
    ///
    /// # Reference
    /// auto_LiRPA operators/bivariate.py:MulHelper.interpolated_relaxation (r_l, r_u)
    /// Design: designs/2026-03-04-286-attention-bilinear-alternative.md Approach B
    pub fn propagate_linear_binary_with_alpha(
        &self,
        bounds: &crate::LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        alphas: Option<&ndarray::Array4<f32>>,
    ) -> Result<(crate::LinearBounds, crate::LinearBounds)> {
        let alphas = match alphas {
            Some(a) => a,
            None => return self.propagate_linear_binary(bounds, input_a_bounds, input_b_bounds),
        };

        let dims = parse_matmul_dims(
            self.transpose_b,
            input_a_bounds.shape(),
            input_b_bounds.shape(),
        )?;

        // Validate alpha shape matches [4, m, n, k]
        if alphas.shape() != [4, dims.m, dims.n, dims.k] {
            return Err(NyError::ShapeMismatch {
                expected: vec![4, dims.m, dims.n, dims.k],
                got: alphas.shape().to_vec(),
            });
        }

        validate_mccormick_inputs(input_a_bounds, input_b_bounds, "BilinearCrown")?;

        let batch_size = dims.batch_size()?;
        let c_size_per_batch = dims.c_size_per_batch()?;
        let a_size_per_batch = dims.a_size_per_batch()?;

        let total_c_size = batch_size.checked_mul(c_size_per_batch).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BilinearCrown: total_c_size overflow: {batch_size} * {c_size_per_batch}",
            ))
        })?;
        let total_a_size = batch_size.checked_mul(a_size_per_batch).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BilinearCrown: total_a_size overflow: {batch_size} * {a_size_per_batch}",
            ))
        })?;
        let total_b_size = batch_size
            .checked_mul(dims.b_size_per_batch)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "BilinearCrown: total_b_size overflow: {batch_size} * {}",
                    dims.b_size_per_batch,
                ))
            })?;

        if bounds.num_inputs() != total_c_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_c_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        let scale = self.scale.unwrap_or(1.0);

        let mut lower_a_a = Array2::<f32>::zeros((num_outputs, total_a_size));
        let mut lower_a_b = Array2::<f32>::zeros((num_outputs, total_b_size));
        let mut upper_a_a = Array2::<f32>::zeros((num_outputs, total_a_size));
        let mut upper_a_b = Array2::<f32>::zeros((num_outputs, total_b_size));

        let mut lower_b_total = Array1::<f64>::zeros(num_outputs);
        let mut upper_b_total = Array1::<f64>::zeros(num_outputs);

        for out_idx in 0..num_outputs {
            let mut const_lower = bounds.lower_b()[out_idx] as f64;
            let mut const_upper = bounds.upper_b()[out_idx] as f64;
            let batch_index_len = dims.batch_dims.len();
            // Stack-allocated index buffers (#2237 F4).
            assert!(
                batch_index_len + 2 <= 8,
                "BilinearCrown: batch_index_len + 2 exceeds stack buffer"
            );
            let mut a_idx = [0usize; 8];
            let mut b_idx = [0usize; 8];
            let idx_len = batch_index_len + 2;

            for batch_idx in 0..batch_size {
                decode_batch_index_into_buf(
                    batch_idx,
                    &dims.batch_dims,
                    &mut a_idx[..batch_index_len],
                )?;
                b_idx[..batch_index_len].copy_from_slice(&a_idx[..batch_index_len]);
                a_idx[batch_index_len + 1] = 0;
                b_idx[batch_index_len] = 0;
                b_idx[batch_index_len + 1] = 0;

                for i in 0..dims.m {
                    a_idx[batch_index_len] = i;
                    for j in 0..dims.n {
                        let c_flat = batch_idx * c_size_per_batch + i * dims.n + j;
                        let w_lower = bounds.lower_a()[[out_idx, c_flat]] * scale;
                        let w_upper = bounds.upper_a()[[out_idx, c_flat]] * scale;
                        if self.transpose_b {
                            b_idx[batch_index_len] = j;
                        } else {
                            b_idx[batch_index_len + 1] = j;
                        }

                        for l in 0..dims.k {
                            // Get A[batch..., i, l]
                            a_idx[batch_index_len + 1] = l;
                            let lx = input_a_bounds.lower()[&a_idx[..idx_len]];
                            let ux = input_a_bounds.upper()[&a_idx[..idx_len]];

                            let a_flat = batch_idx * a_size_per_batch + i * dims.k + l;

                            // Get B[batch..., l, j] or B[batch..., j, l] if transposed
                            let b_flat = if self.transpose_b {
                                b_idx[batch_index_len + 1] = l;
                                batch_idx * dims.b_size_per_batch + j * dims.k + l
                            } else {
                                b_idx[batch_index_len] = l;
                                batch_idx * dims.b_size_per_batch + l * dims.n + j
                            };
                            let ly = input_b_bounds.lower()[&b_idx[..idx_len]];
                            let uy = input_b_bounds.upper()[&b_idx[..idx_len]];

                            // Non-finite guard: if any bound is non-finite, use
                            // conservative trivial relaxation (same as
                            // select_mccormick_plane NaN guard).
                            // Assign -inf/+inf directly; do NOT multiply by weight,
                            // because negative weights would flip the sign (#3319).
                            if !lx.is_finite()
                                || !ux.is_finite()
                                || !ly.is_finite()
                                || !uy.is_finite()
                            {
                                if w_lower != 0.0 {
                                    const_lower = f64::NEG_INFINITY;
                                }
                                if w_upper != 0.0 {
                                    const_upper = f64::INFINITY;
                                }
                                continue;
                            }

                            // Get interpolation parameters from alpha state.
                            // [0]: r_l for lower, [1]: r_l for upper,
                            // [2]: r_u for lower, [3]: r_u for upper.
                            let r_l_lower = alphas[[0, i, j, l]].clamp(0.0, 1.0);
                            let r_u_lower = alphas[[2, i, j, l]].clamp(0.0, 1.0);
                            let r_l_upper = alphas[[1, i, j, l]].clamp(0.0, 1.0);
                            let r_u_upper = alphas[[3, i, j, l]].clamp(0.0, 1.0);

                            // Compute interpolated McCormick for lower bound direction
                            let (al_l, bl_l, gl_l, au_l, bu_l, gu_l) =
                                interpolated_mccormick(lx, ux, ly, uy, r_l_lower, r_u_lower);

                            // Compute interpolated McCormick for upper bound direction
                            let (al_u, bl_u, gl_u, au_u, bu_u, gu_u) =
                                interpolated_mccormick(lx, ux, ly, uy, r_l_upper, r_u_upper);

                            if w_lower != 0.0 {
                                // For lower bound: w >= 0 uses lower relaxation,
                                // w < 0 uses upper relaxation.
                                let (ax, ay, c) = if w_lower >= 0.0 {
                                    (al_l, bl_l, gl_l)
                                } else {
                                    (au_l, bu_l, gu_l)
                                };
                                lower_a_a[[out_idx, a_flat]] += w_lower * ax;
                                lower_a_b[[out_idx, b_flat]] += w_lower * ay;
                                const_lower += w_lower as f64 * c as f64;
                            }

                            if w_upper != 0.0 {
                                // For upper bound: w >= 0 uses upper relaxation,
                                // w < 0 uses lower relaxation.
                                let (ax, ay, c) = if w_upper >= 0.0 {
                                    (au_u, bu_u, gu_u)
                                } else {
                                    (al_u, bl_u, gl_u)
                                };
                                upper_a_a[[out_idx, a_flat]] += w_upper * ax;
                                upper_a_b[[out_idx, b_flat]] += w_upper * ay;
                                const_upper += w_upper as f64 * c as f64;
                            }
                        }
                    }
                }
            }

            lower_b_total[out_idx] = const_lower;
            upper_b_total[out_idx] = const_upper;
        }

        // Split constant terms across both inputs for GraphNetwork accumulation.
        // Halve in f64, then apply directed rounding on final f32 cast (#2164).
        let lower_b_half = lower_b_total.mapv(|v| next_down_f32((v * 0.5) as f32));
        let upper_b_half = upper_b_total.mapv(|v| next_up_f32((v * 0.5) as f32));

        let bounds_a = crate::LinearBounds::new_or_conservative(
            lower_a_a,
            lower_b_half.clone(),
            upper_a_a,
            upper_b_half.clone(),
        )?;

        let bounds_b = crate::LinearBounds::new_or_conservative(
            lower_a_b,
            lower_b_half,
            upper_a_b,
            upper_b_half,
        )?;

        Ok((bounds_a, bounds_b))
    }

    /// Compute output shape for alpha parameters.
    ///
    /// Returns (m, n, k) where alphas should have shape [4, m, n, k].
    /// Inputs must be at least 2D (matmul operands).
    pub fn alpha_shape(
        &self,
        input_a_shape: &[usize],
        input_b_shape: &[usize],
    ) -> Result<(usize, usize, usize)> {
        let a_ndim = input_a_shape.len();
        let b_ndim = input_b_shape.len();
        if a_ndim < 2 || b_ndim < 2 {
            return Err(NyError::InvalidSpec(format!(
                "BilinearCrown alpha_shape: inputs must be >= 2D, got {a_ndim}D and {b_ndim}D"
            )));
        }
        let m = input_a_shape[a_ndim - 2];
        let k = input_a_shape[a_ndim - 1];
        let n = if self.transpose_b {
            input_b_shape[b_ndim - 2]
        } else {
            input_b_shape[b_ndim - 1]
        };
        Ok((m, n, k))
    }

    /// Initialize alpha parameters with default values.
    ///
    /// Default initialization uses r=1.0 (matching auto_LiRPA's torch.ones() default),
    /// which starts optimization from the L1/U1 McCormick planes.
    ///
    /// Returns [4, m, n, k] with direction-dependent layout.
    pub fn init_alpha(
        &self,
        input_a_shape: &[usize],
        input_b_shape: &[usize],
    ) -> Result<ndarray::Array4<f32>> {
        let (m, n, k) = self.alpha_shape(input_a_shape, input_b_shape)?;
        Ok(ndarray::Array4::ones((4, m, n, k)))
    }
}
