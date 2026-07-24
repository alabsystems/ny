// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Non-batched CROWN backward propagation for MatMul using McCormick envelopes.

use ndarray::{Array1, Array2};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::super::validate_mccormick_inputs;
use super::shape::{decode_batch_index_into, parse_matmul_dims};
use super::{select_mccormick_plane, BoundDir, LinearBounds, MatMulLayer, NyError, Result};

impl MatMulLayer {
    /// CROWN backward propagation for MatMul (C = A @ B or A @ B^T).
    ///
    /// Uses McCormick envelope relaxation for the bilinear terms.
    /// Supports batched N-D inputs: A has shape [..., M, K], B has shape [..., K, N] or [..., N, K].
    ///
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(LinearBounds, LinearBounds)> {
        let dims = parse_matmul_dims(
            self.transpose_b,
            input_a_bounds.shape(),
            input_b_bounds.shape(),
        )?;

        validate_mccormick_inputs(input_a_bounds, input_b_bounds, "MatMul")?;

        let batch_size = dims.batch_size()?;
        let c_size_per_batch = dims.c_size_per_batch()?;
        let a_size_per_batch = dims.a_size_per_batch()?;

        let total_c_size = batch_size.checked_mul(c_size_per_batch).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MatMul: total_c_size overflow: {batch_size} * {c_size_per_batch}",
            ))
        })?;
        let total_a_size = batch_size.checked_mul(a_size_per_batch).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MatMul: total_a_size overflow: {batch_size} * {a_size_per_batch}",
            ))
        })?;
        let total_b_size = batch_size
            .checked_mul(dims.b_size_per_batch)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "MatMul: total_b_size overflow: {batch_size} * {}",
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

        // Certified-error accumulators (#matmul-dense-mccormick): each coefficient
        // cell is summed in round-to-nearest f32 over the j axis (A side, depth
        // dims.n) / i axis (B side, depth dims.m), so it carries the Higham f32
        // accumulation error gamma_n * S that MUST reach concretize. S = Sum|w*a| is
        // accumulated exactly in f64 (f32*f32 is exact in f64). #vnncomp-aw-soundness.
        let mut lower_s_a = Array2::<f64>::zeros((num_outputs, total_a_size));
        let mut lower_s_b = Array2::<f64>::zeros((num_outputs, total_b_size));
        let mut upper_s_a = Array2::<f64>::zeros((num_outputs, total_a_size));
        let mut upper_s_b = Array2::<f64>::zeros((num_outputs, total_b_size));

        let mut lower_b_total = Array1::<f64>::zeros(num_outputs);
        let mut upper_b_total = Array1::<f64>::zeros(num_outputs);

        for out_idx in 0..num_outputs {
            let mut const_lower = bounds.lower_b()[out_idx] as f64;
            let mut const_upper = bounds.upper_b()[out_idx] as f64;
            let batch_index_len = dims.batch_dims.len();
            let mut batch_indices = Vec::with_capacity(batch_index_len);
            let mut a_idx = Vec::with_capacity(batch_index_len + 2);
            let mut b_idx = Vec::with_capacity(batch_index_len + 2);

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
                            let lx = input_a_bounds.lower()[a_idx.as_slice()];
                            let ux = input_a_bounds.upper()[a_idx.as_slice()];
                            // Bit-identical McCormick anchor: f32::midpoint rounds differently at overflow/subnormal edges.
                            #[allow(clippy::manual_midpoint)]
                            let x0 = (lx + ux) * 0.5;

                            let a_flat = batch_idx * a_size_per_batch + i * dims.k + l;

                            // Get B[batch..., l, j] or B[batch..., j, l] if transposed
                            let b_flat = if self.transpose_b {
                                b_idx[batch_index_len + 1] = l;
                                batch_idx * dims.b_size_per_batch + j * dims.k + l
                            } else {
                                b_idx[batch_index_len] = l;
                                batch_idx * dims.b_size_per_batch + l * dims.n + j
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
                                lower_a_a[[out_idx, a_flat]] += w_lower * ax;
                                lower_a_b[[out_idx, b_flat]] += w_lower * ay;
                                lower_s_a[[out_idx, a_flat]] +=
                                    (w_lower as f64).abs() * (ax as f64).abs();
                                lower_s_b[[out_idx, b_flat]] +=
                                    (w_lower as f64).abs() * (ay as f64).abs();
                                const_lower += w_lower as f64 * c as f64;
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
                                upper_a_a[[out_idx, a_flat]] += w_upper * ax;
                                upper_a_b[[out_idx, b_flat]] += w_upper * ay;
                                upper_s_a[[out_idx, a_flat]] +=
                                    (w_upper as f64).abs() * (ax as f64).abs();
                                upper_s_b[[out_idx, b_flat]] +=
                                    (w_upper as f64).abs() * (ay as f64).abs();
                                const_upper += w_upper as f64 * c as f64;
                            }
                        }
                    }
                }
            }

            lower_b_total[out_idx] = const_lower;
            upper_b_total[out_idx] = const_upper;
        }

        // Split constant terms across both inputs so that GraphNetwork accumulation
        // sums correctly. Halve in f64, then apply directed rounding on final f32
        // cast (#2164).
        let lower_b_half = lower_b_total.mapv(|v| next_down_f32((v * 0.5) as f32));
        let upper_b_half = upper_b_total.mapv(|v| next_up_f32((v * 0.5) as f32));

        // Certified coefficient error (#matmul-dense-mccormick): A-side cells
        // accumulate over dims.n terms in f32, B-side cells over dims.m terms.
        // |stored - exact| <= gamma_n_f32(count) * S (Higham, f32 unit roundoff),
        // rounded UP to a sound f32 and carried so concretize penalizes outward.
        let gamma_a = crate::layers::linear::crown_single_gamma_n_f32(dims.n);
        let gamma_b = crate::layers::linear::crown_single_gamma_n_f32(dims.m);
        let lower_a_a_err = lower_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let upper_a_a_err = upper_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let lower_a_b_err = lower_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));
        let upper_a_b_err = upper_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));

        let bounds_a = LinearBounds::new_or_conservative_with_err(
            lower_a_a,
            lower_b_half.clone(),
            upper_a_a,
            upper_b_half.clone(),
            lower_a_a_err,
            upper_a_a_err,
        )?;

        let bounds_b = LinearBounds::new_or_conservative_with_err(
            lower_a_b,
            lower_b_half,
            upper_a_b,
            upper_b_half,
            lower_a_b_err,
            upper_a_b_err,
        )?;

        Ok((bounds_a, bounds_b))
    }
}
