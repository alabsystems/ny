// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! MeanOnly interval bound propagation for LayerNorm.
//!
//! Handles the `LayerNormMode::MeanOnly` interval arithmetic path where
//! `y_i = ny_i * (x_i - mean(X)) + beta_i`. The Jacobian is constant
//! (independent of x), so standard interval arithmetic is used.

use ndarray::Axis;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use super::super::types::LayerNormLayer;
use super::common::{mean_axis_f64_lower, mean_axis_f64_upper, IbpShapeContext};
use super::slices;

/// MeanOnly interval propagation for LayerNorm.
///
/// Computes bounds on `y_i = ny_i * (x_i - mean(X)) + beta_i` using
/// directed-rounding interval arithmetic with f64-accumulated mean bounds.
pub(super) fn propagate_interval(
    layer: &LayerNormLayer,
    input: &BoundedTensor,
    ctx: &IbpShapeContext,
) -> Result<BoundedTensor> {
    let shape = input.shape();
    let ndim = ctx.ndim;
    let norm_size = ctx.norm_size;

    // Compute bounds on mean using f64 accumulation with directed rounding.
    // mean_lower uses next_down_f32 so result <= true mean (sound for lower).
    // mean_upper uses next_up_f32 so result >= true mean (sound for upper).
    // Part of #2423.
    let mean_lower = mean_axis_f64_lower(input.lower(), Axis(ndim - 1)).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "LayerNorm: mean_axis failed for axis {} on {}D input",
            ndim - 1,
            ndim
        ))
    })?;
    let mean_upper = mean_axis_f64_upper(input.upper(), Axis(ndim - 1)).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "LayerNorm: mean_axis failed for axis {} on {}D input",
            ndim - 1,
            ndim
        ))
    })?;

    let has_nonfinite_mean = mean_lower
        .iter()
        .chain(mean_upper.iter())
        .any(|&v| !v.is_finite());
    if has_nonfinite_mean {
        return layer.fallback_output_bounds(shape);
    }

    let mut out_lower = input.lower().clone();
    let mut out_upper = input.upper().clone();

    if ndim == 1 {
        let mean_l = slices::mean_value_at(&mean_lower, &[], "mean-only 1D: mean_lower empty")?;
        let mean_u = slices::mean_value_at(&mean_upper, &[], "mean-only 1D: mean_upper empty")?;

        for i in 0..norm_size {
            let xl = input.lower()[[i]];
            let xu = input.upper()[[i]];
            // Directed rounding on intermediate deviation. Part of #3344.
            let diff_l = next_down_f32(xl - mean_u);
            let diff_u = next_up_f32(xu - mean_l);
            let g = layer.ny[i];
            let b = layer.beta[i];

            // Directed rounding on final bounds. Part of #3344.
            if g >= 0.0 {
                out_lower[[i]] = next_down_f32(g * diff_l + b);
                out_upper[[i]] = next_up_f32(g * diff_u + b);
            } else {
                out_lower[[i]] = next_down_f32(g * diff_u + b);
                out_upper[[i]] = next_up_f32(g * diff_l + b);
            }
        }
    } else {
        let bs = slices::batch_size(shape)?;
        let prefix_len = ndim - 1;
        let mut full_idx = [0usize; 8]; // stack buffer, part of #2237

        for batch_idx in 0..bs {
            slices::decode_batch_prefix_into(shape, batch_idx, &mut full_idx[..prefix_len]);

            let mean_l = slices::mean_value_at(
                &mean_lower,
                &full_idx[..prefix_len],
                "mean-only: mean_lower empty",
            )?;
            let mean_u = slices::mean_value_at(
                &mean_upper,
                &full_idx[..prefix_len],
                "mean-only: mean_upper empty",
            )?;

            for i in 0..norm_size {
                full_idx[prefix_len] = i;
                let idx = &full_idx[..=prefix_len];

                let xl = input.lower()[idx];
                let xu = input.upper()[idx];
                // Directed rounding on intermediate deviation. Part of #3344.
                let diff_l = next_down_f32(xl - mean_u);
                let diff_u = next_up_f32(xu - mean_l);
                let g = layer.ny[i];
                let b = layer.beta[i];

                // Directed rounding on final bounds. Part of #3344.
                if g >= 0.0 {
                    out_lower[idx] = next_down_f32(g * diff_l + b);
                    out_upper[idx] = next_up_f32(g * diff_u + b);
                } else {
                    out_lower[idx] = next_down_f32(g * diff_u + b);
                    out_upper[idx] = next_up_f32(g * diff_l + b);
                }
            }
        }
    }

    // Repair non-finite outputs: g * diff can produce Inf for large
    // ny or wide input intervals. Consistent with IBP overflow
    // strategy (#3030, #3060).
    BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
}
