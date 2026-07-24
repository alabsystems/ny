// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use super::super::safe_math::interval_mul_for_bounds;
#[cfg(test)]
use ndarray::{Array2, ArrayD, IxDyn};
#[cfg(test)]
use ny_core::{checked_shape_product, NyError, Result};
#[cfg(test)]
use ny_tensor::{next_down_f32, next_up_f32};

/// Batched interval matrix-vector multiplication.
///
/// Computes the interval product of coefficient intervals [a_lower, a_upper] with
/// input intervals [x_lower, x_upper], returning (result_lower, result_upper).
///
/// For each output element i:
///   result[i] = sum_j( [a_l[i,j], a_u[i,j]] * [x_l[j], x_u[j]] )
///
/// where interval multiplication uses all four products:
///   [a_l, a_u] * [x_l, x_u] = [min(products), max(products)]
///
/// This is necessary when coefficient bounds are themselves intervals (lower_a != upper_a),
/// which occurs after compose() operations on asymmetric bounds.
///
/// REQUIRES: `a_lower.shape() == a_upper.shape()` (matching coefficient bounds).
/// REQUIRES: `x_lower.shape() == x_upper.shape()` (matching input bounds).
/// REQUIRES: `a_shape.len() >= 2` (at least an [m, n] matrix).
/// REQUIRES: `x_shape.last() == a_shape.last()` (compatible inner dimension).
/// REQUIRES: `a_shape[..-2] == x_shape[..-1]` (matching batch dimensions).
/// ENSURES: On `Ok((lower, upper))`, output shape is `[...batch, m]`.
/// ENSURES: For all valid coefficient and input selections:
/// `result_lower[i] <= sum_j(a[i,j] * x[j]) <= result_upper[i]`.
/// ENSURES: Result is always a valid interval (`result_lower <= result_upper` element-wise).
///
/// # Errors
/// Returns `Err(...)` for invalid or mismatched shapes.
#[cfg(test)]
pub fn batched_interval_matvec(
    a_lower: &ArrayD<f32>,
    a_upper: &ArrayD<f32>,
    x_lower: &ArrayD<f32>,
    x_upper: &ArrayD<f32>,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    batched_interval_matvec_checked(a_lower, a_upper, x_lower, x_upper)
}

/// Backward-compatible alias for explicit call sites that still use the checked name.
///
/// Kept to avoid churn while callers migrate to `batched_interval_matvec(...) -> Result<_>`.
///
/// REQUIRES: `a_lower.shape() == a_upper.shape()` (otherwise returns `Err(ShapeMismatch)`).
/// REQUIRES: `x_lower.shape() == x_upper.shape()` (otherwise returns `Err(ShapeMismatch)`).
/// REQUIRES: `a_lower.ndim() >= 2` (otherwise returns `Err(InvalidSpec)`).
/// REQUIRES: `x_lower.ndim() >= 1` (otherwise returns `Err(InvalidSpec)`).
/// REQUIRES: `x_lower.shape().last() == a_lower.shape().last()` (otherwise returns `Err(ShapeMismatch)`).
/// REQUIRES: `a_lower` and `x_lower` have matching batch dimensions (otherwise returns `Err(ShapeMismatch)`).
/// ENSURES: On `Ok((lower, upper))`, `lower.shape() == upper.shape() == [...batch, m]`.
/// ENSURES: On `Ok((lower, upper))`, `lower <= upper` element-wise.
///
/// # Errors
/// - `NyError::ShapeMismatch` if coefficient or input bounds have mismatched shapes
/// - `NyError::ShapeMismatch` if inner dimensions are incompatible
/// - `NyError::ShapeMismatch` if batch dimensions don't match
/// - `NyError::InvalidSpec` if coefficient array has fewer than 2 dimensions
/// - `NyError::InvalidSpec` if input array is empty
#[cfg(test)]
pub fn batched_interval_matvec_checked(
    a_lower: &ArrayD<f32>,
    a_upper: &ArrayD<f32>,
    x_lower: &ArrayD<f32>,
    x_upper: &ArrayD<f32>,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    let a_shape = a_lower.shape();
    let x_shape = x_lower.shape();

    // Check minimum dimensions
    if a_shape.len() < 2 {
        return Err(NyError::InvalidSpec(format!(
            "batched_interval_matvec requires coefficient array with at least 2 dimensions, got {}",
            a_shape.len()
        )));
    }
    if x_shape.is_empty() {
        return Err(NyError::InvalidSpec(
            "batched_interval_matvec requires non-empty input array".to_string(),
        ));
    }

    // Verify shape consistency for interval bounds
    if a_lower.shape() != a_upper.shape() {
        return Err(NyError::shape_mismatch(
            a_lower.shape().to_vec(),
            a_upper.shape().to_vec(),
        ));
    }
    if x_lower.shape() != x_upper.shape() {
        return Err(NyError::shape_mismatch(
            x_lower.shape().to_vec(),
            x_upper.shape().to_vec(),
        ));
    }

    let m = a_shape[a_shape.len() - 2];
    let n = a_shape[a_shape.len() - 1];
    let x_n = *x_shape.last().ok_or_else(|| {
        NyError::InvalidSpec("batched_interval_matvec: x_shape is empty".to_string())
    })?;

    if x_n != n {
        return Err(NyError::shape_mismatch(vec![n], vec![x_n]));
    }

    // Validate batch dimensions match exactly
    let a_batch_dims = &a_shape[..a_shape.len() - 2];
    let x_batch_dims = &x_shape[..x_shape.len() - 1];
    if a_batch_dims != x_batch_dims {
        return Err(NyError::shape_mismatch(
            a_batch_dims.to_vec(),
            x_batch_dims.to_vec(),
        ));
    }

    // Output shape: [...batch, m]
    let batch_dims = &a_shape[..a_shape.len() - 2];
    let mut out_shape: Vec<usize> = batch_dims.to_vec();
    out_shape.push(m);

    if out_shape.contains(&0) {
        return Ok((
            ArrayD::zeros(IxDyn(&out_shape)),
            ArrayD::zeros(IxDyn(&out_shape)),
        ));
    }

    // Compute total batch size
    let total_batch = checked_shape_product(batch_dims).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "Batched interval matmul: batch dims product overflows: {:?}",
            batch_dims
        ))
    })?;
    if total_batch == 0 || m == 0 || n == 0 {
        return Ok((
            ArrayD::zeros(IxDyn(&out_shape)),
            ArrayD::zeros(IxDyn(&out_shape)),
        ));
    }

    // Reshape to [batch, m, n] and [batch, n]
    let a_l_flat = a_lower
        .view()
        .into_shape_with_order((total_batch, m, n))
        .map_err(|e| NyError::InternalError(format!("a_lower reshape failed: {e}")))?;
    let a_u_flat = a_upper
        .view()
        .into_shape_with_order((total_batch, m, n))
        .map_err(|e| NyError::InternalError(format!("a_upper reshape failed: {e}")))?;
    let x_l_flat = x_lower
        .view()
        .into_shape_with_order((total_batch, n))
        .map_err(|e| NyError::InternalError(format!("x_lower reshape failed: {e}")))?;
    let x_u_flat = x_upper
        .view()
        .into_shape_with_order((total_batch, n))
        .map_err(|e| NyError::InternalError(format!("x_upper reshape failed: {e}")))?;

    // Fast path: when all coefficients and inputs are finite, skip NaN/Inf/0*inf
    // checks in the inner loop. This eliminates per-element branch overhead and
    // enables compiler auto-vectorization. (#2220 Packet B)
    let all_finite = a_l_flat.iter().all(|v| v.is_finite())
        && a_u_flat.iter().all(|v| v.is_finite())
        && x_l_flat.iter().all(|v| v.is_finite())
        && x_u_flat.iter().all(|v| v.is_finite());

    let (result_lower, result_upper) = if all_finite {
        batched_interval_matvec_finite(&a_l_flat, &a_u_flat, &x_l_flat, &x_u_flat, m, n)
    } else {
        batched_interval_matvec_scalar(&a_l_flat, &a_u_flat, &x_l_flat, &x_u_flat, m, n)
    };

    // Reshape back
    let (vec_lower, _) = result_lower.into_raw_vec_and_offset();
    let (vec_upper, _) = result_upper.into_raw_vec_and_offset();

    let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), vec_lower)
        .map_err(|e| NyError::InternalError(format!("lower reshape back failed: {e}")))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), vec_upper)
        .map_err(|e| NyError::InternalError(format!("upper reshape back failed: {e}")))?;
    Ok((lower, upper))
}

/// Fast path for all-finite values: computes 4 products per element using f64
/// arithmetic with no NaN/Inf/0*inf branching. Compiler can auto-vectorize.
#[cfg(test)]
fn batched_interval_matvec_finite(
    a_l: &ndarray::ArrayView3<f32>,
    a_u: &ndarray::ArrayView3<f32>,
    x_l: &ndarray::ArrayView2<f32>,
    x_u: &ndarray::ArrayView2<f32>,
    m: usize,
    n: usize,
) -> (Array2<f32>, Array2<f32>) {
    let total_batch = a_l.shape()[0];
    let mut result_lower = Array2::zeros((total_batch, m));
    let mut result_upper = Array2::zeros((total_batch, m));

    for b in 0..total_batch {
        for i in 0..m {
            let mut sum_lower = 0.0f64;
            let mut sum_upper = 0.0f64;
            for j in 0..n {
                let al = a_l[[b, i, j]];
                let au = a_u[[b, i, j]];
                let xl = x_l[[b, j]];
                let xu = x_u[[b, j]];
                // Products in f32 matching scalar path, then f64 for accumulation.
                let p1 = (al * xl) as f64;
                let p2 = (al * xu) as f64;
                let p3 = (au * xl) as f64;
                let p4 = (au * xu) as f64;
                sum_lower += p1.min(p2).min(p3.min(p4));
                sum_upper += p1.max(p2).max(p3.max(p4));
            }
            result_lower[[b, i]] = next_down_f32(sum_lower as f32);
            result_upper[[b, i]] = next_up_f32(sum_upper as f32);
        }
    }
    (result_lower, result_upper)
}

/// Scalar fallback with full NaN/Inf/0*inf handling via `interval_mul_for_bounds`.
#[cfg(test)]
fn batched_interval_matvec_scalar(
    a_l: &ndarray::ArrayView3<f32>,
    a_u: &ndarray::ArrayView3<f32>,
    x_l: &ndarray::ArrayView2<f32>,
    x_u: &ndarray::ArrayView2<f32>,
    m: usize,
    n: usize,
) -> (Array2<f32>, Array2<f32>) {
    let total_batch = a_l.shape()[0];
    let mut result_lower = Array2::zeros((total_batch, m));
    let mut result_upper = Array2::zeros((total_batch, m));

    for b in 0..total_batch {
        for i in 0..m {
            let mut sum_lower = 0.0f64;
            let mut sum_upper = 0.0f64;
            for j in 0..n {
                let (prod_lower, prod_upper) = interval_mul_for_bounds(
                    a_l[[b, i, j]],
                    a_u[[b, i, j]],
                    x_l[[b, j]],
                    x_u[[b, j]],
                );
                sum_lower += prod_lower as f64;
                sum_upper += prod_upper as f64;
            }
            result_lower[[b, i]] = if sum_lower.is_nan() {
                f32::NEG_INFINITY
            } else {
                next_down_f32(sum_lower as f32)
            };
            result_upper[[b, i]] = if sum_upper.is_nan() {
                f32::INFINITY
            } else {
                next_up_f32(sum_upper as f32)
            };
        }
    }
    (result_lower, result_upper)
}
