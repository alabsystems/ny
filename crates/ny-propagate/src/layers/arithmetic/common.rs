// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for arithmetic layer implementations.

use ndarray::ArrayD;
use ny_core::{NyError, Result};

/// Extract a scalar constant from a potentially multi-element array.
///
/// Batched CROWN backward for constant-arithmetic layers requires scalar constants.
/// This helper validates and extracts the single element.
#[inline]
pub(super) fn extract_scalar_constant_for_batched(
    constant: &ArrayD<f32>,
    layer_name: &str,
) -> Result<f32> {
    if constant.len() != 1 {
        return Err(NyError::UnsupportedOp(format!(
            "{layer_name} batched CROWN only supports scalar constants"
        )));
    }

    constant.iter().copied().next().ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "{layer_name} batched CROWN: scalar constant unexpectedly empty"
        ))
    })
}

/// Epsilon guard for division-by-zero in DivConstant CROWN relaxation.
pub(super) const DIV_CONSTANT_EPS: f32 = 1e-10;

/// Compute A @ c with f64 accumulation for soundness (#3157).
///
/// f32 dot products lose precision through catastrophic cancellation on
/// high-dimensional constants. This function promotes each element to f64
/// before multiply-accumulate, matching the pattern in linear/bias.rs (#1863).
///
/// Returns `(lower_accum, upper_accum)` as f64 arrays.
pub(super) fn dot_bias_f64(
    lower_a: &ndarray::Array2<f32>,
    upper_a: &ndarray::Array2<f32>,
    c: &ndarray::Array1<f32>,
) -> (ndarray::Array1<f64>, ndarray::Array1<f64>) {
    let num_outputs = lower_a.nrows();
    let num_cols = lower_a.ncols();
    let mut lower_f64 = ndarray::Array1::<f64>::zeros(num_outputs);
    let mut upper_f64 = ndarray::Array1::<f64>::zeros(num_outputs);
    for i in 0..num_outputs {
        for j in 0..num_cols {
            lower_f64[i] += lower_a[[i, j]] as f64 * c[j] as f64;
            upper_f64[i] += upper_a[[i, j]] as f64 * c[j] as f64;
        }
    }
    (lower_f64, upper_f64)
}

/// Compute scalar * sum(A, axis=1) with f64 accumulation for soundness (#3157).
///
/// For scalar constants, A @ c_broadcast = c * sum(A, axis=1).
/// The row sum accumulation benefits from f64 precision.
pub(super) fn scalar_row_sum_f64(
    lower_a: &ndarray::Array2<f32>,
    upper_a: &ndarray::Array2<f32>,
    c_scalar: f32,
) -> (ndarray::Array1<f64>, ndarray::Array1<f64>) {
    let num_outputs = lower_a.nrows();
    let num_cols = lower_a.ncols();
    let c_f64 = c_scalar as f64;
    let mut lower_f64 = ndarray::Array1::<f64>::zeros(num_outputs);
    let mut upper_f64 = ndarray::Array1::<f64>::zeros(num_outputs);
    for i in 0..num_outputs {
        let mut lower_sum = 0.0f64;
        let mut upper_sum = 0.0f64;
        for j in 0..num_cols {
            lower_sum += lower_a[[i, j]] as f64;
            upper_sum += upper_a[[i, j]] as f64;
        }
        lower_f64[i] = c_f64 * lower_sum;
        upper_f64[i] = c_f64 * upper_sum;
    }
    (lower_f64, upper_f64)
}
