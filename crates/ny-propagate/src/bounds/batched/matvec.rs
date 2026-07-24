// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use super::super::safe_math::safe_mul_for_bounds;
#[cfg(test)]
use crate::contiguous_flat_slice_mut;
#[cfg(test)]
use ndarray::{s, Array2, ArrayD, ArrayView1, ArrayView2, IxDyn};
#[cfg(test)]
use ny_core::{checked_shape_product, NyError, Result};

/// Compute matvec for a single batch using f64 accumulators and safe inf/NaN handling.
///
/// Writes `m` results into `out[offset..offset+m]`.
/// Uses `safe_mul_for_bounds` for 0*inf=0 semantics, f64 accumulation to reduce
/// rounding error (#2214), and a NaN guard that distinguishes inf-cancellation NaN
/// from input NaN (preserves propagation).
///
/// When `is_lower` is true, inf-cancellation NaN maps to `f32::NEG_INFINITY`
/// (sound lower bound). When false, maps to `f32::INFINITY` (sound upper bound).
/// See `safe_add_for_bounds_with_polarity` for the same convention.
#[cfg(test)]
pub(crate) fn matvec_safe_f64(
    a: &ArrayView2<f32>,
    x: &ArrayView1<f32>,
    out: &mut [f32],
    offset: usize,
    m: usize,
    n: usize,
    is_lower: bool,
) {
    let nan_fallback = if is_lower {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    for i in 0..m {
        let mut sum = 0.0f64;
        let mut has_nan_term = false;
        for j in 0..n {
            let term = safe_mul_for_bounds(a[[i, j]], x[[j]]);
            if term.is_nan() {
                has_nan_term = true;
            }
            sum += term as f64;
        }
        out[offset + i] = if sum.is_nan() && !has_nan_term {
            nan_fallback
        } else {
            sum as f32
        };
    }
}

/// Compute matvec for a single batch using f64 accumulators (fast path, no inf/NaN).
///
/// Writes `m` results into `out[offset..offset+m]`.
/// Uses f64 accumulation to match `batched_interval_matvec_checked` precision.
#[cfg(test)]
pub(crate) fn matvec_fast_f64(
    a: &ArrayView2<f32>,
    x: &ArrayView1<f32>,
    out: &mut [f32],
    offset: usize,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        let mut sum = 0.0f64;
        for j in 0..n {
            sum += a[[i, j]] as f64 * x[[j]] as f64;
        }
        out[offset + i] = sum as f32;
    }
}

/// Batched matrix-vector multiplication with safe handling of 0 * inf.
///
/// For A with shape [..., m, n] and x with shape [..., n],
/// computes y with shape `[..., m]` where `y[...][i] = sum_j A[...][i,j] * x[...][j]`
///
/// Uses safe_mul_for_bounds to handle cases where A contains zeros and x contains
/// infinite values, which would otherwise produce NaN in standard multiplication.
/// Uses f64 accumulators to reduce rounding error for high-dimensional inner products
/// (same rationale as `batched_interval_matvec_checked` — see #2214).
///
/// `is_lower` controls NaN polarity for inf-cancellation: when true, NaN from
/// `inf + (-inf)` maps to `f32::NEG_INFINITY` (sound lower bound); when false,
/// maps to `f32::INFINITY` (sound upper bound).
///
/// REQUIRES: `a.shape().len() >= 2` and `x.shape().len() >= 1`.
/// REQUIRES: `a.shape()[..a.ndim()-2] == x.shape()[..x.ndim()-1]` (batch dims match).
/// REQUIRES: `a.shape()[a.ndim()-1] == x.shape()[x.ndim()-1]` (inner dim match).
/// ENSURES: `result.shape() == [...a.shape()[..a.ndim()-2], a.shape()[a.ndim()-2]]`.
#[cfg(test)]
pub fn batched_matvec(a: &ArrayD<f32>, x: &ArrayD<f32>, is_lower: bool) -> Result<ArrayD<f32>> {
    let a_shape = a.shape();
    let x_shape = x.shape();

    if a_shape.len() < 2 || x_shape.is_empty() {
        return Ok(ArrayD::zeros(IxDyn(&[])));
    }

    let m = a_shape[a_shape.len() - 2];
    let n = a_shape[a_shape.len() - 1];

    let x_inner = x_shape[x_shape.len() - 1];
    if x_inner != n {
        return Err(NyError::InvalidSpec(format!(
            "batched_matvec: input dimension mismatch: a inner dim {n} != x last dim {x_inner}"
        )));
    }

    let batch_dims = &a_shape[..a_shape.len() - 2];
    let mut out_shape: Vec<usize> = batch_dims.to_vec();
    out_shape.push(m);

    let total_batch: usize = checked_shape_product(batch_dims)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "batched_matvec: batch dimensions {batch_dims:?} overflow usize",
            ))
        })?
        .max(1);

    let a_flat = a
        .view()
        .into_shape_with_order((total_batch, m, n))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "batched_matvec: failed to reshape a {:?} to ({total_batch}, {m}, {n}): {e}",
                a_shape
            ))
        })?;
    let x_flat = x
        .view()
        .into_shape_with_order((total_batch, n))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "batched_matvec: failed to reshape x {:?} to ({total_batch}, {n}): {e}",
                x_shape
            ))
        })?;

    let has_inf_or_nan = x.iter().any(|&v| v.is_infinite() || v.is_nan())
        || a.iter().any(|&v| v.is_infinite() || v.is_nan());

    let mut result = Array2::zeros((total_batch, m));
    let result_slice = contiguous_flat_slice_mut(&mut result)?;

    for b in 0..total_batch {
        let a_batch = a_flat.slice(s![b, .., ..]);
        let x_batch = x_flat.slice(s![b, ..]);
        if has_inf_or_nan {
            matvec_safe_f64(&a_batch, &x_batch, result_slice, b * m, m, n, is_lower);
        } else {
            matvec_fast_f64(&a_batch, &x_batch, result_slice, b * m, m, n);
        }
    }

    let (vec, _offset) = result.into_raw_vec_and_offset();
    ArrayD::from_shape_vec(IxDyn(&out_shape), vec).map_err(|e| {
        NyError::InvalidSpec(format!(
            "batched_matvec: failed to reshape result to {out_shape:?}: {e}"
        ))
    })
}
