// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Decomposed LayerNorm CROWN backward propagation.
//!
//! Decomposes LayerNorm into primitives and propagates CROWN backward through
//! each step:
//!   x → mean(x) → d=x-mean → d² → var=mean(d²) → sqrt(var+eps) → 1/std → d*inv_std → γ·norm+β
//!
//! Part of #2077, #318.

use ndarray::{Array1, Array2};
use ny_core::{checked_dim_product, checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::debug;

use crate::bounds::BatchedLinearBounds;
use crate::layers::activations::ClipLayer;
use crate::layers::arithmetic::sqrt_linear_relaxation;
use crate::layers::common::BoundPropagation;
use crate::layers::misc::reciprocal::reciprocal_linear_relaxation;
use crate::layers::normalization::math_common::square_interval_bounds;
use crate::layers::LayerNormLayer;
use crate::LinearBounds;

use super::bilinear::accumulate_mccormick_bilinear_term;
use super::variance_chain::accumulate_variance_chain;
use super::{
    finalize_decomposed_norm_bounds, validate_norm_against_fused_ibp, DecomposedNormBackwardResult,
    DecomposedNormFinalizeMetadata, RowValidationCounts,
};

/// Decomposed LayerNorm CROWN backward (see module-level docs for chain).
///
/// Uses McCormick relaxation for bilinear `d * inv_std`, directed rounding for
/// all compositions, and fan-out summation at `d` for product + variance paths.
///
/// Ref: `auto_LiRPA/operators/normalization.py:303-331`. Part of #2077, #318.
pub(crate) fn decomposed_norm_crown_backward(
    a_output: &BatchedLinearBounds,
    ny: &Array1<f32>,
    beta: &Array1<f32>,
    eps: f32,
    x_ibp: &BoundedTensor,
    forward_mode: bool,
) -> Result<DecomposedNormBackwardResult> {
    let a_shape = a_output.lower_a().shape();
    let ndim = a_shape.len();
    if ndim < 2 {
        return Err(NyError::InvalidSpec(
            "decomposed_norm_crown_backward: A must have at least 2 dimensions".into(),
        ));
    }

    let n = a_shape[ndim - 1];
    let out_dim = a_shape[ndim - 2];
    let batch_dims = &a_shape[..ndim - 2];
    let total_batch = checked_shape_product(batch_dims)
        .ok_or_else(|| NyError::InvalidSpec("decomposed_norm: batch dimensions overflow".into()))?
        .max(1);
    let nf = n as f32;

    if n == 0 {
        return Err(NyError::InvalidSpec(
            "decomposed_norm_crown_backward: normalization dimension is 0".into(),
        ));
    }
    if ny.len() != n || beta.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![ny.len()],
        });
    }

    let a_l_3d = a_output
        .lower_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, n))
        .map_err(|e| NyError::InvalidSpec(format!("reshape lower_a: {}", e)))?;
    let a_u_3d = a_output
        .upper_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, n))
        .map_err(|e| NyError::InvalidSpec(format!("reshape upper_a: {}", e)))?;
    let b_l_2d = a_output
        .lower_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|e| NyError::InvalidSpec(format!("reshape lower_b: {}", e)))?;
    let b_u_2d = a_output
        .upper_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|e| NyError::InvalidSpec(format!("reshape upper_b: {}", e)))?;

    let x_l_2d = x_ibp
        .lower()
        .view()
        .into_shape_with_order((total_batch, n))
        .map_err(|e| NyError::InvalidSpec(format!("reshape x_lower: {}", e)))?;
    let x_u_2d = x_ibp
        .upper()
        .view()
        .into_shape_with_order((total_batch, n))
        .map_err(|e| NyError::InvalidSpec(format!("reshape x_upper: {}", e)))?;

    let total_rows = checked_dim_product(
        &[total_batch, out_dim],
        "decomposed_norm_crown_backward total rows",
    )?;
    let mut new_a_l = Array2::<f32>::zeros((total_rows, n));
    let mut new_a_u = Array2::<f32>::zeros((total_rows, n));
    let mut new_b_l = Array2::<f64>::zeros((total_batch, out_dim));
    let mut new_b_u = Array2::<f64>::zeros((total_batch, out_dim));

    let mut lower_nonfinite_rows = vec![false; total_rows];
    let mut upper_nonfinite_rows = vec![false; total_rows];

    for b in 0..total_batch {
        for j in 0..out_dim {
            new_b_l[[b, j]] = b_l_2d[[b, j]] as f64;
            new_b_u[[b, j]] = b_u_2d[[b, j]] as f64;
        }
    }

    for b in 0..total_batch {
        let (d_l, d_u, var_eps_l, var_eps_u, std_l, std_u, inv_std_l, inv_std_u) = if forward_mode {
            let mut center_sum = 0.0_f64;
            for i in 0..n {
                // Bit-identical to `(l + u) * 0.5`: f32-cast operands stay on
                // f64::midpoint's non-overflow `(a + b) * 0.5` path.
                center_sum += f64::midpoint(x_l_2d[[b, i]] as f64, x_u_2d[[b, i]] as f64);
            }
            let mean_c = (center_sum / nf as f64) as f32;
            let mut d_l_v = vec![0.0_f32; n];
            let mut d_u_v = vec![0.0_f32; n];
            let mut var_f64 = 0.0_f64;
            for i in 0..n {
                d_l_v[i] = x_l_2d[[b, i]] - mean_c;
                d_u_v[i] = x_u_2d[[b, i]] - mean_c;
                let center_d =
                    f64::midpoint(x_l_2d[[b, i]] as f64, x_u_2d[[b, i]] as f64) - mean_c as f64;
                var_f64 += center_d * center_d;
            }
            let var_c = (var_f64 / nf as f64) as f32;
            let ve_c = var_c + eps;
            let std_c = ve_c.sqrt();
            let inv_c = 1.0 / std_c;
            (d_l_v, d_u_v, ve_c, ve_c, std_c, std_c, inv_c, inv_c)
        } else {
            let mut mean_l_f64 = 0.0_f64;
            let mut mean_u_f64 = 0.0_f64;
            for i in 0..n {
                mean_l_f64 += x_l_2d[[b, i]] as f64;
                mean_u_f64 += x_u_2d[[b, i]] as f64;
            }
            let mean_l = next_down_f32((mean_l_f64 / nf as f64) as f32);
            let mean_u = next_up_f32((mean_u_f64 / nf as f64) as f32);
            let mut d_l_v = vec![0.0_f32; n];
            let mut d_u_v = vec![0.0_f32; n];
            let mut var_l_f64 = 0.0_f64;
            let mut var_u_f64 = 0.0_f64;
            for i in 0..n {
                d_l_v[i] = next_down_f32(x_l_2d[[b, i]] - mean_u);
                d_u_v[i] = next_up_f32(x_u_2d[[b, i]] - mean_l);
                let (sq_l, sq_u) = square_interval_bounds(d_l_v[i], d_u_v[i]);
                var_l_f64 += sq_l as f64;
                var_u_f64 += sq_u as f64;
            }
            let var_l = next_down_f32((var_l_f64 / nf as f64) as f32);
            let var_u = next_up_f32((var_u_f64 / nf as f64) as f32);
            let ve_l = next_down_f32((var_l as f64 + eps as f64) as f32);
            let ve_u = next_up_f32((var_u as f64 + eps as f64) as f32);
            let s_l = next_down_f32(((ve_l as f64).sqrt()) as f32);
            let s_u = next_up_f32(((ve_u as f64).sqrt()) as f32);
            let inv_l = next_down_f32(1.0 / s_u);
            let inv_u = next_up_f32(1.0 / s_l);
            (d_l_v, d_u_v, ve_l, ve_u, s_l, s_u, inv_l, inv_u)
        };

        let max_norm = if n > 1 {
            next_up_f32((nf - 1.0).sqrt())
        } else {
            0.0
        };
        let norm_clip = ClipLayer::new(-max_norm, max_norm);
        let mut norm_l = Vec::with_capacity(n);
        let mut norm_u = Vec::with_capacity(n);
        for i in 0..n {
            let corners = [
                d_l[i] as f64 * inv_std_l as f64,
                d_l[i] as f64 * inv_std_u as f64,
                d_u[i] as f64 * inv_std_l as f64,
                d_u[i] as f64 * inv_std_u as f64,
            ];
            let corner_min = corners.iter().copied().fold(f64::INFINITY, f64::min);
            let corner_max = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !corner_min.is_finite() || !corner_max.is_finite() {
                return Err(NyError::NumericalInstability(
                    "decomposed_norm_crown_backward: non-finite norm interval".into(),
                ));
            }
            norm_l.push(next_down_f32(corner_min as f32));
            norm_u.push(next_up_f32(corner_max as f32));
        }
        let norm_ibp = BoundedTensor::new(
            Array1::from_vec(norm_l).into_dyn(),
            Array1::from_vec(norm_u).into_dyn(),
        )?;

        let recip_relax = reciprocal_linear_relaxation(std_l, std_u);
        let sqrt_relax = sqrt_linear_relaxation(var_eps_l, var_eps_u);

        let mut a_d_total_l = vec![0.0_f64; n];
        let mut a_d_total_u = vec![0.0_f64; n];

        for j in 0..out_dim {
            let row_idx = b * out_dim + j;

            let mut a_inv_std_l_f64 = 0.0_f64;
            let mut a_inv_std_u_f64 = 0.0_f64;

            a_d_total_l.fill(0.0);
            a_d_total_u.fill(0.0);

            let mut clipped_norm_lower_a = Array2::<f32>::zeros((1, n));
            let mut clipped_norm_upper_a = Array2::<f32>::zeros((1, n));
            for i in 0..n {
                let g = ny[i];
                clipped_norm_lower_a[[0, i]] = a_l_3d[[b, j, i]] * g;
                clipped_norm_upper_a[[0, i]] = a_u_3d[[b, j, i]] * g;
                new_b_l[[b, j]] += a_l_3d[[b, j, i]] as f64 * beta[i] as f64;
                new_b_u[[b, j]] += a_u_3d[[b, j, i]] as f64 * beta[i] as f64;
            }
            let clipped_norm_bounds = LinearBounds::new(
                clipped_norm_lower_a,
                Array1::zeros(1),
                clipped_norm_upper_a,
                Array1::zeros(1),
            )?;
            let clipped_norm =
                norm_clip.propagate_linear_with_bounds(&clipped_norm_bounds, &norm_ibp)?;
            let clip_has_nonfinite = clipped_norm
                .lower_a()
                .iter()
                .chain(clipped_norm.upper_a().iter())
                .chain(clipped_norm.lower_b().iter())
                .chain(clipped_norm.upper_b().iter())
                .any(|v| !v.is_finite());
            if clip_has_nonfinite {
                lower_nonfinite_rows[row_idx] = true;
                upper_nonfinite_rows[row_idx] = true;
                continue;
            }
            new_b_l[[b, j]] += clipped_norm.lower_b()[0] as f64;
            new_b_u[[b, j]] += clipped_norm.upper_b()[0] as f64;

            for i in 0..n {
                let w_l = clipped_norm.lower_a()[[0, i]];
                let w_u = clipped_norm.upper_a()[[0, i]];
                let (lower_nonfinite, upper_nonfinite) = accumulate_mccormick_bilinear_term(
                    w_l,
                    w_u,
                    d_l[i],
                    d_u[i],
                    inv_std_l,
                    inv_std_u,
                    &mut a_d_total_l[i],
                    &mut a_d_total_u[i],
                    &mut a_inv_std_l_f64,
                    &mut a_inv_std_u_f64,
                    &mut new_b_l[[b, j]],
                    &mut new_b_u[[b, j]],
                );
                lower_nonfinite_rows[row_idx] |= lower_nonfinite;
                upper_nonfinite_rows[row_idx] |= upper_nonfinite;
            }

            let (lower_nonfinite, upper_nonfinite) = accumulate_variance_chain(
                a_inv_std_l_f64,
                a_inv_std_u_f64,
                &recip_relax,
                &sqrt_relax,
                &d_l,
                &d_u,
                n,
                eps,
                &mut a_d_total_l,
                &mut a_d_total_u,
                &mut new_b_l[[b, j]],
                &mut new_b_u[[b, j]],
            );
            lower_nonfinite_rows[row_idx] |= lower_nonfinite;
            upper_nonfinite_rows[row_idx] |= upper_nonfinite;

            let mut sum_a_l_f64 = 0.0_f64;
            let mut sum_a_u_f64 = 0.0_f64;
            for i in 0..n {
                sum_a_l_f64 += a_d_total_l[i];
                sum_a_u_f64 += a_d_total_u[i];
            }
            let mean_corr_l_f64 = sum_a_l_f64 / nf as f64;
            let mean_corr_u_f64 = sum_a_u_f64 / nf as f64;

            for i in 0..n {
                new_a_l[[row_idx, i]] = next_down_f32((a_d_total_l[i] - mean_corr_l_f64) as f32);
                new_a_u[[row_idx, i]] = next_up_f32((a_d_total_u[i] - mean_corr_u_f64) as f32);
            }
        }
    }

    let mut result = finalize_decomposed_norm_bounds(
        new_a_l,
        new_a_u,
        new_b_l,
        new_b_u,
        DecomposedNormFinalizeMetadata {
            lower_nonfinite_rows: &lower_nonfinite_rows,
            upper_nonfinite_rows: &upper_nonfinite_rows,
            total_rows,
            out_dim,
            n,
            batch_dims,
            input_shape: a_output.input_shape(),
            output_shape: a_output.output_shape(),
            label: "Decomposed normalization",
        },
    )?;

    let fused_ibp =
        LayerNormLayer::new(ny.to_owned(), beta.to_owned(), eps)?.propagate_ibp(x_ibp)?;
    let fallback_rows =
        validate_norm_against_fused_ibp(&mut result, a_output, &fused_ibp, x_ibp, total_rows, n)?;
    if fallback_rows > 0 {
        debug!(
            "Decomposed normalization: collapsed {fallback_rows}/{total_rows} rows to fused LayerNorm IBP"
        );
    }

    Ok(DecomposedNormBackwardResult {
        bounds: result,
        validation: RowValidationCounts {
            fallback_rows,
            total_rows,
        },
    })
}
