// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward propagation for MatMul using McCormick envelopes.

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::checked_shape_product;
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::debug;

use super::super::validate_mccormick_inputs;
use super::shape::{decode_batch_index_into, parse_matmul_dims};
use super::{select_mccormick_plane, BatchedLinearBounds, BoundDir, MatMulLayer, NyError, Result};

impl MatMulLayer {
    /// Batched CROWN backward propagation for MatMul (C = A @ B or A @ B^T).
    ///
    /// Uses McCormick envelope relaxation for the bilinear terms.
    /// Supports N-D batched inputs: A has shape [..., M, K], B has shape [..., K, N] or [..., N, K].
    ///
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_batched_binary(
        &self,
        bounds: &BatchedLinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
        debug!("MatMul batched CROWN backward propagation");

        let dims = parse_matmul_dims(
            self.transpose_b,
            input_a_bounds.shape(),
            input_b_bounds.shape(),
        )?;

        validate_mccormick_inputs(input_a_bounds, input_b_bounds, "MatMul batched")?;

        let batch_size = dims.batch_size()?;
        let c_size = dims.c_size_per_batch()?;
        let a_size = dims.a_size_per_batch()?;
        let b_size = dims.b_size_per_batch;

        // Get the bounds shape
        let bounds_a_shape = bounds.lower_a.shape();
        if bounds_a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = bounds_a_shape[bounds_a_shape.len() - 2];
        let mid_dim = bounds_a_shape[bounds_a_shape.len() - 1];

        if mid_dim != c_size {
            return Err(NyError::UnsupportedOp(format!(
                "MatMul batched CROWN expects flattened output dim m*n = {} (got in_dim = {}); consider reshaping MatMul outputs or fall back to IBP",
                c_size, mid_dim
            )));
        }

        let bounds_batch_dims = &bounds_a_shape[..bounds_a_shape.len() - 2];
        let total_bounds_batch: usize = checked_shape_product(bounds_batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "MatMul batched CROWN: bounds batch dimensions {bounds_batch_dims:?} overflow usize",
                ))
            })?
            .max(1);

        // Output shapes
        let mut out_a_shape: Vec<usize> = bounds_batch_dims.to_vec();
        out_a_shape.push(out_dim);
        out_a_shape.push(a_size);

        let mut out_b_shape: Vec<usize> = bounds_batch_dims.to_vec();
        out_b_shape.push(out_dim);
        out_b_shape.push(b_size);

        let mut out_bias_shape: Vec<usize> = bounds_batch_dims.to_vec();
        out_bias_shape.push(out_dim);

        let scale = self.scale.unwrap_or(1.0);

        // Reshape bounds to [total_batch, out_dim, c_size]
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim, c_size))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim, c_size))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;
        let lower_b_2d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
        let upper_b_2d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

        // Allocate output coefficient matrices
        let mut new_lower_a_a = Array2::zeros((total_bounds_batch * out_dim, a_size));
        let mut new_upper_a_a = Array2::zeros((total_bounds_batch * out_dim, a_size));
        let mut new_lower_a_b = Array2::zeros((total_bounds_batch * out_dim, b_size));
        let mut new_upper_a_b = Array2::zeros((total_bounds_batch * out_dim, b_size));
        // Certified-error abssum accumulators (#matmul-batched-mccormick): each coeff
        // cell is summed in round-to-nearest f32 over the (batch_idx, j) pairs for the
        // A side and (batch_idx, i) for the B side — the cell index a_flat=i*k+l /
        // b_flat omits batch_idx, so batch_size*dims.n (A) / batch_size*dims.m (B)
        // terms collide into one cell. Carry the Higham f32 accumulation error
        // gamma_depth * S (S = Sum|w*a| in exact f64) so it reaches concretize.
        // #vnncomp-aw-soundness.
        let mut s_lower_a_a = Array2::<f64>::zeros((total_bounds_batch * out_dim, a_size));
        let mut s_upper_a_a = Array2::<f64>::zeros((total_bounds_batch * out_dim, a_size));
        let mut s_lower_a_b = Array2::<f64>::zeros((total_bounds_batch * out_dim, b_size));
        let mut s_upper_a_b = Array2::<f64>::zeros((total_bounds_batch * out_dim, b_size));
        let mut new_lower_b = Array2::<f64>::zeros((total_bounds_batch, out_dim));
        let mut new_upper_b = Array2::<f64>::zeros((total_bounds_batch, out_dim));
        let batch_index_len = dims.batch_dims.len();

        // Process each batch position and output dimension
        for bb in 0..total_bounds_batch {
            // Copy bias terms
            for d in 0..out_dim {
                new_lower_b[[bb, d]] = lower_b_2d[[bb, d]] as f64;
                new_upper_b[[bb, d]] = upper_b_2d[[bb, d]] as f64;
            }

            let mut batch_indices = Vec::with_capacity(batch_index_len);
            let mut a_idx = Vec::with_capacity(batch_index_len + 2);
            let mut b_idx = Vec::with_capacity(batch_index_len + 2);

            for d in 0..out_dim {
                let row_idx = bb * out_dim + d;

                for batch_idx in 0..batch_size {
                    decode_batch_index_into(batch_idx, &dims.batch_dims, &mut batch_indices)?;
                    a_idx.clear();
                    a_idx.extend_from_slice(&batch_indices);
                    a_idx.resize(batch_index_len + 2, 0);
                    b_idx.clear();
                    b_idx.extend_from_slice(&batch_indices);
                    b_idx.resize(batch_index_len + 2, 0);

                    for i in 0..dims.m {
                        a_idx[batch_index_len] = i;
                        for j in 0..dims.n {
                            let c_flat = i * dims.n + j;

                            let w_lower = lower_a_3d[[bb, d, c_flat]] * scale;
                            let w_upper = upper_a_3d[[bb, d, c_flat]] * scale;

                            if w_lower == 0.0 && w_upper == 0.0 {
                                continue;
                            }

                            if self.transpose_b {
                                b_idx[batch_index_len] = j;
                            } else {
                                b_idx[batch_index_len + 1] = j;
                            }

                            for l in 0..dims.k {
                                // Get A[batch..., i, l]
                                a_idx[batch_index_len + 1] = l;
                                let lx = input_a_bounds.lower()[a_idx.as_slice()];
                                let ux = input_a_bounds.upper()[a_idx.as_slice()];
                                // Bit-identical McCormick anchor: f32::midpoint rounds differently at overflow/subnormal edges.
                                #[allow(clippy::manual_midpoint)]
                                let x0 = (lx + ux) * 0.5;

                                let a_flat = i * dims.k + l;

                                // Get B[batch..., l, j] or B[batch..., j, l] if transposed
                                let b_flat = if self.transpose_b {
                                    b_idx[batch_index_len + 1] = l;
                                    j * dims.k + l
                                } else {
                                    b_idx[batch_index_len] = l;
                                    l * dims.n + j
                                };

                                let ly = input_b_bounds.lower()[b_idx.as_slice()];
                                let uy = input_b_bounds.upper()[b_idx.as_slice()];
                                // Bit-identical McCormick anchor: f32::midpoint rounds differently at overflow/subnormal edges.
                                #[allow(clippy::manual_midpoint)]
                                let y0 = (ly + uy) * 0.5;

                                if w_lower != 0.0 {
                                    let (ax, ay, c) = select_mccormick_plane(
                                        lx,
                                        ux,
                                        ly,
                                        uy,
                                        x0,
                                        y0,
                                        w_lower,
                                        BoundDir::Lower,
                                    );
                                    new_lower_a_a[[row_idx, a_flat]] += w_lower * ax;
                                    new_lower_a_b[[row_idx, b_flat]] += w_lower * ay;
                                    s_lower_a_a[[row_idx, a_flat]] +=
                                        (w_lower as f64).abs() * (ax as f64).abs();
                                    s_lower_a_b[[row_idx, b_flat]] +=
                                        (w_lower as f64).abs() * (ay as f64).abs();
                                    new_lower_b[[bb, d]] += w_lower as f64 * c as f64;
                                }

                                if w_upper != 0.0 {
                                    let (ax, ay, c) = select_mccormick_plane(
                                        lx,
                                        ux,
                                        ly,
                                        uy,
                                        x0,
                                        y0,
                                        w_upper,
                                        BoundDir::Upper,
                                    );
                                    new_upper_a_a[[row_idx, a_flat]] += w_upper * ax;
                                    new_upper_a_b[[row_idx, b_flat]] += w_upper * ay;
                                    s_upper_a_a[[row_idx, a_flat]] +=
                                        (w_upper as f64).abs() * (ax as f64).abs();
                                    s_upper_a_b[[row_idx, b_flat]] +=
                                        (w_upper as f64).abs() * (ay as f64).abs();
                                    new_upper_b[[bb, d]] += w_upper as f64 * c as f64;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Reshape back to output shape
        let (new_lower_a_a_vec, _) = new_lower_a_a.into_raw_vec_and_offset();
        let (new_upper_a_a_vec, _) = new_upper_a_a.into_raw_vec_and_offset();
        let (new_lower_a_b_vec, _) = new_lower_a_b.into_raw_vec_and_offset();
        let (new_upper_a_b_vec, _) = new_upper_a_b.into_raw_vec_and_offset();
        // Halve bias in f64 before directed-rounded f32 cast (#2173).
        let new_lower_b_f32 = new_lower_b.mapv(|v| next_down_f32((v * 0.5) as f32));
        let new_upper_b_f32 = new_upper_b.mapv(|v| next_up_f32((v * 0.5) as f32));
        let (new_lower_b_vec, _) = new_lower_b_f32.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b_f32.into_raw_vec_and_offset();

        let new_lower_a_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a_a".to_string()))?;
        let new_upper_a_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a_a".to_string()))?;
        let new_lower_a_b = ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_a_b_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a_b".to_string()))?;
        let new_upper_a_b = ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_a_b_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a_b".to_string()))?;
        let lower_b_half = ArrayD::from_shape_vec(IxDyn(&out_bias_shape), new_lower_b_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?;
        let upper_b_half = ArrayD::from_shape_vec(IxDyn(&out_bias_shape), new_upper_b_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?;

        // Update input shapes to flattened representations (preserve batch dims).
        let a_batch = &dims.batch_dims;
        let b_batch = &dims.batch_dims;
        let mut new_input_shape_a = a_batch.clone();
        new_input_shape_a.push(a_size);
        let mut new_input_shape_b = b_batch.clone();
        new_input_shape_b.push(b_size);

        // Certified coefficient error (#matmul-batched-mccormick). Conservative depth:
        // a cell collides over at most batch_size*dims.n (A) / batch_size*dims.m (B)
        // f32 += operations; gamma_n_f32 of that upper-bounds the true Higham factor
        // (sound whether or not batch_idx actually collides). err = gamma*S rounded UP,
        // carried via set_coeff_err so concretize penalizes outward. #vnncomp-aw-soundness.
        let depth_a = batch_size.saturating_mul(dims.n).max(1);
        let depth_b = batch_size.saturating_mul(dims.m).max(1);
        let gamma_a = crate::layers::linear::crown_single_gamma_n_f32(depth_a);
        let gamma_b = crate::layers::linear::crown_single_gamma_n_f32(depth_b);
        let mk_err = |s: Array2<f64>, gamma: f64, shape: &[usize]| -> Result<ArrayD<f32>> {
            let e = s.mapv(|sv| next_up_f32((gamma * sv) as f32));
            let (v, _) = e.into_raw_vec_and_offset();
            ArrayD::from_shape_vec(IxDyn(shape), v)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape coeff err".to_string()))
        };
        let lower_a_a_err = mk_err(s_lower_a_a, gamma_a, &out_a_shape)?;
        let upper_a_a_err = mk_err(s_upper_a_a, gamma_a, &out_a_shape)?;
        let lower_a_b_err = mk_err(s_lower_a_b, gamma_b, &out_b_shape)?;
        let upper_a_b_err = mk_err(s_upper_a_b, gamma_b, &out_b_shape)?;

        // Phase 4 audit: per-layer MatMul McCormick output — catches NaN from McCormick.
        let mut bounds_a = BatchedLinearBounds::new_or_conservative(
            new_lower_a_a,
            lower_b_half.clone(),
            new_upper_a_a,
            upper_b_half.clone(),
            new_input_shape_a,
            bounds.output_shape.clone(),
        )?;
        bounds_a.set_coeff_err(lower_a_a_err, upper_a_a_err);

        let mut bounds_b = BatchedLinearBounds::new_or_conservative(
            new_lower_a_b,
            lower_b_half,
            new_upper_a_b,
            upper_b_half,
            new_input_shape_b,
            bounds.output_shape.clone(),
        )?;
        bounds_b.set_coeff_err(lower_a_b_err, upper_a_b_err);

        Ok((bounds_a, bounds_b))
    }
}
