// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::LinearBounds;
use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

pub(super) enum DivBackwardResult {
    PropagateNumerator(Box<LinearBounds>),
    ConcretizeCurrentNode {
        lower: Box<Array1<f32>>,
        upper: Box<Array1<f32>>,
    },
}

fn concretized_node_bias(
    node_lb: &LinearBounds,
    node_output_bounds: &BoundedTensor,
) -> DivBackwardResult {
    let concretized = node_lb.concretize_sound(node_output_bounds).flatten();
    DivBackwardResult::ConcretizeCurrentNode {
        lower: Box::new(Array1::from_vec(
            concretized.lower().iter().copied().collect(),
        )),
        upper: Box::new(Array1::from_vec(
            concretized.upper().iter().copied().collect(),
        )),
    }
}

pub(super) fn backward_div_to_numerator(
    _node_name: &str,
    node_lb: &LinearBounds,
    input_a_bounds: &BoundedTensor,
    input_b_bounds: &BoundedTensor,
    node_output_bounds: &BoundedTensor,
) -> Result<DivBackwardResult> {
    let b_lower_flat = input_b_bounds
        .lower()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div denominator lower not contiguous".into()))?;
    let b_upper_flat = input_b_bounds
        .upper()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div denominator upper not contiguous".into()))?;

    // Sound only when the denominator is sign-definite (0 ∉ [ly, uy]): all
    // strictly positive OR all strictly negative. The reciprocal-scaling
    // arithmetic below is sign-independent (r_mid carries the sign of 1/y,
    // r_delta ≥ 0 is the half-width error radius). Mixed/zero-touching
    // denominators keep the sound concretization fallback (#Div-neg).
    let all_pos = b_lower_flat.iter().all(|&v| v > 0.0);
    let all_neg = b_upper_flat.iter().all(|&v| v < 0.0);
    if !(all_pos || all_neg) {
        return Ok(concretized_node_bias(node_lb, node_output_bounds));
    }

    let num_lower_flat = input_a_bounds
        .lower()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div numerator lower not contiguous".into()))?;
    let num_upper_flat = input_a_bounds
        .upper()
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Div numerator upper not contiguous".into()))?;

    let n = node_lb.num_inputs();
    if num_lower_flat.len() != n {
        return Ok(concretized_node_bias(node_lb, node_output_bounds));
    }

    let recip_lower: Vec<f64> = b_upper_flat.iter().map(|&v| 1.0 / (v as f64)).collect();
    let recip_upper: Vec<f64> = b_lower_flat.iter().map(|&v| 1.0 / (v as f64)).collect();
    let num_abs_max: Vec<f64> = num_lower_flat
        .iter()
        .zip(num_upper_flat.iter())
        .map(|(&lo, &up)| (lo.abs() as f64).max(up.abs() as f64))
        .collect();
    let r_mid: Vec<f64> = recip_lower
        .iter()
        .zip(recip_upper.iter())
        // Bit-identical to `(rl + ru) / 2.0` here: |1/(f32 as f64)| is either 0,
        // in [2.9e-39, 7.2e44], or ±inf/NaN — never in midpoint's rescale ranges.
        .map(|(&rl, &ru)| f64::midpoint(rl, ru))
        .collect();
    let r_delta: Vec<f64> = recip_lower
        .iter()
        .zip(recip_upper.iter())
        .map(|(&rl, &ru)| (ru - rl) / 2.0)
        .collect();

    let b_shape_raw = input_b_bounds.shape();
    let out_shape: Vec<usize> = node_output_bounds.shape().to_vec();
    let ndim = out_shape.len();
    let mut b_shape_aligned = vec![1usize; ndim];
    for (i, &s) in b_shape_raw.iter().rev().enumerate() {
        if i < ndim {
            b_shape_aligned[ndim - 1 - i] = s;
        }
    }
    let b_len = b_lower_flat.len();
    let mut groups: Vec<Vec<usize>> = vec![vec![]; b_len];
    for out_flat in 0..n {
        let mut remaining = out_flat;
        let mut b_flat = 0;
        let mut b_stride = 1;
        for d in (0..ndim).rev() {
            let out_idx_d = remaining % out_shape[d];
            remaining /= out_shape[d];
            let b_idx_d = if b_shape_aligned[d] == 1 {
                0
            } else {
                out_idx_d
            };
            b_flat += b_idx_d * b_stride;
            b_stride *= b_shape_aligned[d];
        }
        if b_flat >= b_len {
            return Ok(concretized_node_bias(node_lb, node_output_bounds));
        }
        groups[b_flat].push(out_flat);
    }

    let mut new_lower_a = node_lb.lower_a().to_owned();
    let mut new_upper_a = node_lb.upper_a().to_owned();
    let mut new_lower_b = node_lb.lower_b().to_owned();
    let mut new_upper_b = node_lb.upper_b().to_owned();

    for spec_idx in 0..node_lb.num_outputs() {
        for g in 0..b_len {
            let mut lower_abs_sum = 0.0_f64;
            let mut upper_abs_sum = 0.0_f64;

            for &elem in &groups[g] {
                let lo = new_lower_a[[spec_idx, elem]] as f64;
                let up = new_upper_a[[spec_idx, elem]] as f64;

                // r_mid is sign-definite (matches the denominator sign) but may
                // be negative; only require it be finite and nonzero.
                debug_assert!(r_mid[g].is_finite() && r_mid[g] != 0.0);
                new_lower_a[[spec_idx, elem]] = next_down_f32((lo * r_mid[g]) as f32);
                new_upper_a[[spec_idx, elem]] = next_up_f32((up * r_mid[g]) as f32);

                lower_abs_sum += lo.abs() * num_abs_max[elem];
                upper_abs_sum += up.abs() * num_abs_max[elem];
            }

            new_lower_b[spec_idx] -= next_up_f32((r_delta[g] * lower_abs_sum) as f32);
            new_upper_b[spec_idx] += next_up_f32((r_delta[g] * upper_abs_sum) as f32);
        }
    }

    // Migrated from from_parts_unchecked: reciprocal-scaling arithmetic can
    // produce NaN (e.g., Inf * 0.0) or Inf (near-zero denominator overflow).
    // NaN firewall falls back to conservative bounds. See #3438.
    Ok(DivBackwardResult::PropagateNumerator(Box::new(
        LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)?,
    )))
}
