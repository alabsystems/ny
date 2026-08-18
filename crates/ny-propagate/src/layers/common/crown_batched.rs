// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward for element-wise activation functions.
//!
//! Operates on `ArrayD` A-matrices reshaped to 3D (batch × output × input).

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::compose;
use crate::layers::activations::LinearRelaxation;
use crate::BatchedLinearBounds;

/// Generic batched CROWN backward for element-wise activation functions.
///
/// Delegates to [`crown_elementwise_backward_batched_indexed`] with a no-op neuron index.
pub fn crown_elementwise_backward_batched<F>(
    bounds: &BatchedLinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
) -> Result<BatchedLinearBounds>
where
    F: Fn(f32, f32) -> LinearRelaxation,
{
    crown_elementwise_backward_batched_indexed(bounds, pre_activation, |l, u, _i| {
        relaxation_fn(l, u)
    })
}

/// Indexed batched CROWN backward for element-wise activations.
///
/// Like [`crown_elementwise_backward_batched`], but the relaxation function receives
/// the neuron index `i`. Supports per-neuron parameters (e.g., PReLU per-channel slopes).
///
/// Bias accumulation uses f64 to prevent catastrophic cancellation (#1745).
pub fn crown_elementwise_backward_batched_indexed<F>(
    bounds: &BatchedLinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
) -> Result<BatchedLinearBounds>
where
    F: Fn(f32, f32, usize) -> LinearRelaxation,
{
    let pre_shape = pre_activation.shape();
    let a_shape = bounds.lower_a().shape();

    if a_shape.len() < 2 {
        return Err(NyError::InvalidSpec(
            "BatchedLinearBounds must have at least 2 dimensions".to_string(),
        ));
    }

    let out_dim = a_shape[a_shape.len() - 2];
    let in_dim = a_shape[a_shape.len() - 1];
    let batch_dims = &a_shape[..a_shape.len() - 2];
    let total_batch: usize = checked_shape_product(batch_dims)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "compose_batched_bounds_3d: batch dimensions {batch_dims:?} overflow usize",
            ))
        })?
        .max(1);

    let pre_in_dim = *pre_shape.last().unwrap_or(&0);
    if pre_in_dim != in_dim {
        return Err(NyError::ShapeMismatch {
            expected: vec![in_dim],
            got: vec![pre_in_dim],
        });
    }

    // Reshape to working dimensions
    let pre_lower_flat = pre_activation
        .lower()
        .view()
        .into_shape_with_order((total_batch, in_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape pre_lower".to_string()))?;
    let pre_upper_flat = pre_activation
        .upper()
        .view()
        .into_shape_with_order((total_batch, in_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape pre_upper".to_string()))?;

    let lower_a_3d = bounds
        .lower_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, in_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
    let upper_a_3d = bounds
        .upper_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, in_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;
    let lower_b_2d = bounds
        .lower_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
    let upper_b_2d = bounds
        .upper_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

    // Output arrays — bias in f64 for numerical stability (#1745)
    let total_rows = total_batch * out_dim;
    let mut new_lower_a = Array2::<f32>::zeros((total_rows, in_dim));
    let mut new_upper_a = Array2::<f32>::zeros((total_rows, in_dim));
    let mut new_lower_b = Array2::<f64>::zeros((total_batch, out_dim));
    let mut new_upper_b = Array2::<f64>::zeros((total_batch, out_dim));

    // Incoming certified coefficient error (#vnncomp-aw-soundness). The batched
    // activation backward MUST propagate it through the relaxation exactly like
    // the scalar `crown_elementwise_backward` (crown_dense.rs). Every composed
    // coefficient carries the fresh f32 product gap, even when no incoming error
    // exists. Incoming error uses the chosen envelope while its sign is stable;
    // otherwise both envelopes cover a possible sign flip. Intercept uncertainty
    // is folded OUTWARD into the bias. Dropping any of these terms makes the
    // batched (β-CROWN/BaB) verdict optimistic relative to the real composition.
    let in_lower_err = bounds.lower_a_err.as_ref();
    let in_upper_err = bounds.upper_a_err.as_ref();
    let in_lower_err_3d = in_lower_err
        .map(|e| {
            e.view()
                .into_shape_with_order((total_batch, out_dim, in_dim))
        })
        .transpose()
        .map_err(|_| NyError::InvalidSpec("Cannot reshape incoming lower_a_err".to_string()))?;
    let in_upper_err_3d = in_upper_err
        .map(|e| {
            e.view()
                .into_shape_with_order((total_batch, out_dim, in_dim))
        })
        .transpose()
        .map_err(|_| NyError::InvalidSpec("Cannot reshape incoming upper_a_err".to_string()))?;
    let mut new_lower_a_err = Array2::<f32>::zeros((total_rows, in_dim));
    let mut new_upper_a_err = Array2::<f32>::zeros((total_rows, in_dim));
    // Per-(batch,out) certified intercept error: subtracted from lower, added to upper.
    let mut lower_b_err = Array2::<f64>::zeros((total_batch, out_dim));
    let mut upper_b_err = Array2::<f64>::zeros((total_batch, out_dim));

    // Track which output rows have non-finite coefficients after coeff × slope (#3009).
    let mut lower_nonfinite_rows = vec![false; total_rows];
    let mut upper_nonfinite_rows = vec![false; total_rows];

    for b in 0..total_batch {
        for j in 0..out_dim {
            new_lower_b[[b, j]] = lower_b_2d[[b, j]] as f64;
            new_upper_b[[b, j]] = upper_b_2d[[b, j]] as f64;
        }
    }

    for b in 0..total_batch {
        for i in 0..in_dim {
            let l = pre_lower_flat[[b, i]];
            let u = pre_upper_flat[[b, i]];
            let relax = relaxation_fn(l, u, i);
            let slope_sum = (relax.lower_slope.abs() + relax.upper_slope.abs()) as f64;
            let int_sum = (relax.lower_intercept.abs() + relax.upper_intercept.abs()) as f64;
            // Direction-selected slopes actually used by compose_lower/compose_upper
            // (lower_slope for a>0, upper_slope for a<0, in the lower direction).
            let lr_slope = |a: f32| {
                if a > 0.0 {
                    relax.lower_slope as f64
                } else if a < 0.0 {
                    relax.upper_slope as f64
                } else {
                    0.0
                }
            };
            let ur_slope = |a: f32| {
                if a > 0.0 {
                    relax.upper_slope as f64
                } else if a < 0.0 {
                    relax.lower_slope as f64
                } else {
                    0.0
                }
            };

            for j in 0..out_dim {
                let la = lower_a_3d[[b, j, i]];
                let ua = upper_a_3d[[b, j, i]];
                let row_idx = b * out_dim + j;

                let lr = compose::compose_lower(la, &relax);
                new_lower_a[[row_idx, i]] = lr.new_coeff;
                new_lower_b[[b, j]] += lr.intercept_contrib;
                lower_nonfinite_rows[row_idx] |= lr.nonfinite;

                let ur = compose::compose_upper(ua, &relax);
                new_upper_a[[row_idx, i]] = ur.new_coeff;
                new_upper_b[[b, j]] += ur.intercept_contrib;
                upper_nonfinite_rows[row_idx] |= ur.nonfinite;

                // Always carry the fresh coefficient-product rounding gap, even
                // when the incoming bound had no error carrier. Otherwise a
                // first batched activation silently treats its stored f32
                // coefficient as the exact real product. Directed coefficient
                // rounding alone is not a box-independent enclosure: moving a
                // lower coefficient downward raises the affine value at negative
                // inputs (and conversely for an upper coefficient).
                {
                    let gap = if la != 0.0 {
                        (la as f64 * lr_slope(la) - lr.new_coeff as f64).abs()
                    } else {
                        0.0
                    };
                    let ea = in_lower_err_3d
                        .as_ref()
                        .map_or(0.0, |e| e[[b, j, i]] as f64);
                    if gap != 0.0 || ea != 0.0 {
                        let (slope_cover, int_cover) = if (la as f64).abs() > ea {
                            let slope = lr_slope(la).abs();
                            let intercept = if la > 0.0 {
                                relax.lower_intercept.abs() as f64
                            } else {
                                relax.upper_intercept.abs() as f64
                            };
                            (slope, intercept)
                        } else {
                            (slope_sum, int_sum)
                        };
                        new_lower_a_err[[row_idx, i]] =
                            next_up_f32((ea * slope_cover + gap) as f32);
                        if ea != 0.0 {
                            lower_b_err[[b, j]] += ea * int_cover;
                        }
                    }
                }
                {
                    let gap = if ua != 0.0 {
                        (ua as f64 * ur_slope(ua) - ur.new_coeff as f64).abs()
                    } else {
                        0.0
                    };
                    let ea = in_upper_err_3d
                        .as_ref()
                        .map_or(0.0, |e| e[[b, j, i]] as f64);
                    if gap != 0.0 || ea != 0.0 {
                        let (slope_cover, int_cover) = if (ua as f64).abs() > ea {
                            let slope = ur_slope(ua).abs();
                            let intercept = if ua > 0.0 {
                                relax.upper_intercept.abs() as f64
                            } else {
                                relax.lower_intercept.abs() as f64
                            };
                            (slope, intercept)
                        } else {
                            (slope_sum, int_sum)
                        };
                        new_upper_a_err[[row_idx, i]] =
                            next_up_f32((ea * slope_cover + gap) as f32);
                        if ea != 0.0 {
                            upper_b_err[[b, j]] += ea * int_cover;
                        }
                    }
                }
            }
        }
    }

    // Fold the certified intercept error OUTWARD into the bias accumulators BEFORE
    // the directed cast (lower decreases, upper increases).
    for b in 0..total_batch {
        for j in 0..out_dim {
            new_lower_b[[b, j]] -= lower_b_err[[b, j]];
            new_upper_b[[b, j]] += upper_b_err[[b, j]];
        }
    }

    // #3009: Non-finite row fallback for batched activation CROWN backward.
    let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
    let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
    compose::log_nonfinite_fallback(
        "Batched activation",
        lower_affected,
        upper_affected,
        total_rows,
    );

    // Convert bias to f32 with directed rounding, then apply row fallback
    let (new_lower_b_raw, _) = new_lower_b.into_raw_vec_and_offset();
    let mut new_lower_b_f32: Vec<f32> = new_lower_b_raw
        .into_iter()
        .map(|x| next_down_f32(x as f32))
        .collect();
    let (new_upper_b_raw, _) = new_upper_b.into_raw_vec_and_offset();
    let mut new_upper_b_f32: Vec<f32> = new_upper_b_raw
        .into_iter()
        .map(|x| next_up_f32(x as f32))
        .collect();

    for row_idx in 0..total_rows {
        let b = row_idx / out_dim;
        let j = row_idx % out_dim;
        let bias_idx = b * out_dim + j;
        if lower_nonfinite_rows[row_idx] {
            for i in 0..in_dim {
                new_lower_a[[row_idx, i]] = 0.0;
                new_lower_a_err[[row_idx, i]] = 0.0;
            }
            new_lower_b_f32[bias_idx] = f32::NEG_INFINITY;
        }
        if upper_nonfinite_rows[row_idx] {
            for i in 0..in_dim {
                new_upper_a[[row_idx, i]] = 0.0;
                new_upper_a_err[[row_idx, i]] = 0.0;
            }
            new_upper_b_f32[bias_idx] = f32::INFINITY;
        }
    }

    // Reshape back to original batch structure
    let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
    let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();

    let out_a_shape: Vec<usize> = batch_dims
        .iter()
        .cloned()
        .chain([out_dim, in_dim])
        .collect();
    let out_b_shape: Vec<usize> = batch_dims.iter().cloned().chain([out_dim]).collect();

    // CROWN backward NaN firewall (#2812): shared path for ALL activation CROWN backward
    // (ReLU, Sigmoid, Exp, Log, SiLU, etc.). Falls back to conservative bounds instead of
    // aborting verification when any relaxation function produces NaN.
    let mut result = BatchedLinearBounds::new_or_conservative(
        ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
        ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_f32)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
        ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
        ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_f32)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
        bounds.input_shape.clone(),
        bounds.output_shape.clone(),
    )?;
    // Attach the propagated certified error (#vnncomp-aw-soundness). `set_coeff_err`
    // no-ops on shape mismatch (i.e. if the NaN firewall degraded to conservative),
    // which is sound: the conservative bounds already dominate any penalty.
    let le = ArrayD::from_shape_vec(
        IxDyn(&out_a_shape),
        new_lower_a_err.into_raw_vec_and_offset().0,
    )
    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a_err".to_string()))?;
    let ue = ArrayD::from_shape_vec(
        IxDyn(&out_a_shape),
        new_upper_a_err.into_raw_vec_and_offset().0,
    )
    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a_err".to_string()))?;
    result.set_coeff_err(le, ue);
    Ok(result)
}
