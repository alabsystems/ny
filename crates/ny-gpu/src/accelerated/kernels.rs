// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Standalone IBP computation kernels: linear and batched matmul.
//!
//! These functions are used by `AcceleratedDevice` and also by other device
//! implementations that share the same CPU kernel logic.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{
    checked_shape_product, nan_propagating_max, nan_propagating_max_zero, nan_propagating_min,
    nan_propagating_min_zero, NyError, Result,
};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;
use tracing::debug;

/// Parallel IBP for linear layers using Rayon.
///
/// This implementation splits work across batch elements and output features
/// for maximum parallelism.
pub fn linear_ibp_parallel(
    input: &BoundedTensor,
    weight: &Array2<f32>,
    bias: Option<&Array1<f32>>,
    batch_size: usize,
    in_features: usize,
    out_features: usize,
) -> Result<BoundedTensor> {
    let shape = input.shape();

    // Pre-compute positive and negative weight matrices.
    // NaN-propagating variants (#2642): Rust's f32::max(NaN, 0.0) = 0.0 (IEEE 754-2008),
    // silently dropping NaN weights. Use nan_propagating_max_zero/min_zero so NaN weights
    // poison the accumulator and trigger downstream NaN guards, matching the CPU path (#2415).
    let weight_pos: Vec<f32> = weight
        .iter()
        .map(|&w| nan_propagating_max_zero(w))
        .collect();
    let weight_neg: Vec<f32> = weight
        .iter()
        .map(|&w| nan_propagating_min_zero(w))
        .collect();

    // Get input data as contiguous slices
    let lower_data = input.lower().as_slice().ok_or_else(|| {
        NyError::InternalError("linear_ibp_parallel: input lower not contiguous".into())
    })?;
    let upper_data = input.upper().as_slice().ok_or_else(|| {
        NyError::InternalError("linear_ibp_parallel: input upper not contiguous".into())
    })?;

    // Allocate output buffers
    let output_size = batch_size * out_features;
    let mut result_lower = vec![0.0_f32; output_size];
    let mut result_upper = vec![0.0_f32; output_size];

    // Parallel computation over (batch, output) pairs
    // Use chunks to enable better cache utilization
    result_lower
        .par_chunks_mut(out_features)
        .zip(result_upper.par_chunks_mut(out_features))
        .enumerate()
        .for_each(|(batch_idx, (lower_chunk, upper_chunk))| {
            let input_offset = batch_idx * in_features;
            let xl = &lower_data[input_offset..input_offset + in_features];
            let xu = &upper_data[input_offset..input_offset + in_features];

            for o in 0..out_features {
                let weight_offset = o * in_features;
                let wp = &weight_pos[weight_offset..weight_offset + in_features];
                let wn = &weight_neg[weight_offset..weight_offset + in_features];

                // Vectorized dot products (compiler will auto-vectorize)
                let mut low = 0.0_f32;
                let mut high = 0.0_f32;

                for i in 0..in_features {
                    low += wp[i] * xl[i] + wn[i] * xu[i];
                    high += wp[i] * xu[i] + wn[i] * xl[i];
                }

                if let Some(b) = bias {
                    low += b[o];
                    high += b[o];
                }

                lower_chunk[o] = low;
                upper_chunk[o] = high;
            }
        });

    // Sanitize NaN/Inf values before creating BoundedTensor (#2642 self-audit).
    // After the nan_propagating weight split, NaN weights correctly poison the
    // accumulator. Convert NaN/Inf to conservative FALLBACK_BOUND before
    // BoundedTensor::new (which rejects NaN), matching the CPU NaN guard in
    // linear.rs:191-196 and the matmul sanitization loop below.
    use crate::FALLBACK_BOUND;
    for i in 0..output_size {
        let l = result_lower[i];
        let u = result_upper[i];
        if l.is_nan() || l.is_infinite() || u.is_nan() || u.is_infinite() {
            result_lower[i] = -FALLBACK_BOUND;
            result_upper[i] = FALLBACK_BOUND;
        }
    }

    // Reshape to output shape [..., out_features]
    let mut out_shape = shape[..shape.len() - 1].to_vec();
    out_shape.push(out_features);

    let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), result_lower)
        .map_err(|_| NyError::shape_mismatch(out_shape.clone(), vec![output_size]))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), result_upper)
        .map_err(|_| NyError::shape_mismatch(out_shape, vec![output_size]))?;

    BoundedTensor::new(lower, upper)
}

/// Parallel IBP for batched matrix multiplication.
///
/// Computes [A_l, A_u] @ [B_l, B_u] with interval arithmetic.
/// Supports N-D batched inputs for transformer attention patterns.
pub fn matmul_ibp_parallel(
    input_a: &BoundedTensor,
    input_b: &BoundedTensor,
) -> Result<BoundedTensor> {
    let shape_a = input_a.shape();
    let shape_b = input_b.shape();

    if shape_a.len() < 2 || shape_b.len() < 2 {
        return Err(NyError::shape_mismatch(
            vec![2],
            vec![shape_a.len().min(shape_b.len())],
        ));
    }

    // Get matrix dimensions
    let m = shape_a[shape_a.len() - 2]; // rows of A
    let k = shape_a[shape_a.len() - 1]; // cols of A = rows of B
    let n = shape_b[shape_b.len() - 1]; // cols of B

    // Verify inner dimensions match
    if shape_b[shape_b.len() - 2] != k {
        return Err(NyError::shape_mismatch(
            vec![k],
            vec![shape_b[shape_b.len() - 2]],
        ));
    }

    // Compute batch dimensions
    let batch_dims_a = &shape_a[..shape_a.len() - 2];
    let batch_dims_b = &shape_b[..shape_b.len() - 2];

    // Batch dims should match (simplified - full broadcasting not implemented)
    if batch_dims_a != batch_dims_b {
        return Err(NyError::shape_mismatch(
            batch_dims_a.to_vec(),
            batch_dims_b.to_vec(),
        ));
    }

    let batch_size: usize = checked_shape_product(batch_dims_a).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "AcceleratedDevice matmul_ibp: batch dims {batch_dims_a:?} overflow usize",
        ))
    })?;
    let output_size = batch_size * m * n;

    debug!(
        "AcceleratedDevice matmul_ibp: batch={}, m={}, k={}, n={}",
        batch_size, m, k, n
    );

    // Get input data
    let al = input_a.lower().as_slice().ok_or_else(|| {
        NyError::InternalError("matmul_ibp_parallel: input_a lower not contiguous".into())
    })?;
    let au = input_a.upper().as_slice().ok_or_else(|| {
        NyError::InternalError("matmul_ibp_parallel: input_a upper not contiguous".into())
    })?;
    let bl = input_b.lower().as_slice().ok_or_else(|| {
        NyError::InternalError("matmul_ibp_parallel: input_b lower not contiguous".into())
    })?;
    let bu = input_b.upper().as_slice().ok_or_else(|| {
        NyError::InternalError("matmul_ibp_parallel: input_b upper not contiguous".into())
    })?;

    // Allocate output
    let mut result_lower = vec![0.0_f32; output_size];
    let mut result_upper = vec![0.0_f32; output_size];

    let matrix_size_a = m * k;
    let matrix_size_b = k * n;
    let matrix_size_out = m * n;

    // Parallel over batch elements
    result_lower
        .par_chunks_mut(matrix_size_out)
        .zip(result_upper.par_chunks_mut(matrix_size_out))
        .enumerate()
        .for_each(|(batch_idx, (lower_chunk, upper_chunk))| {
            let a_offset = batch_idx * matrix_size_a;
            let b_offset = batch_idx * matrix_size_b;

            // For each output element C[i,j]
            for i in 0..m {
                for j in 0..n {
                    let mut low = 0.0_f32;
                    let mut high = 0.0_f32;

                    // Dot product with interval arithmetic
                    for kk in 0..k {
                        let a_l = al[a_offset + i * k + kk];
                        let a_u = au[a_offset + i * k + kk];
                        let b_l = bl[b_offset + kk * n + j];
                        let b_u = bu[b_offset + kk * n + j];

                        // Interval multiplication: [a,b] * [c,d]
                        let products = [a_l * b_l, a_l * b_u, a_u * b_l, a_u * b_u];
                        // NaN-propagating min/max: if any product is NaN (e.g., 0*Inf
                        // not caught upstream), NaN must propagate rather than be
                        // silently absorbed by IEEE 754 min/max. Issue #2577.
                        let min_prod = products
                            .iter()
                            .cloned()
                            .fold(f32::INFINITY, nan_propagating_min);
                        let max_prod = products
                            .iter()
                            .cloned()
                            .fold(f32::NEG_INFINITY, nan_propagating_max);

                        low += min_prod;
                        high += max_prod;
                    }

                    lower_chunk[i * n + j] = low;
                    upper_chunk[i * n + j] = high;
                }
            }
        });

    // Build output shape
    let mut out_shape = batch_dims_a.to_vec();
    out_shape.push(m);
    out_shape.push(n);

    // Sanitize NaN/Inf values before creating BoundedTensor
    // Interval arithmetic can overflow with very wide bounds
    use crate::FALLBACK_BOUND;
    let mut sanitized_count = 0;

    for i in 0..output_size {
        let l = result_lower[i];
        let u = result_upper[i];
        if l.is_nan() || l.is_infinite() || u.is_nan() || u.is_infinite() {
            result_lower[i] = -FALLBACK_BOUND;
            result_upper[i] = FALLBACK_BOUND;
            sanitized_count += 1;
        }
    }

    if sanitized_count > 0 {
        debug!(
            "matmul_ibp_batched_parallel: sanitized {} NaN/Inf values ({}% of output)",
            sanitized_count,
            100.0 * sanitized_count as f64 / output_size as f64
        );
    }

    let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), result_lower)
        .map_err(|_| NyError::shape_mismatch(out_shape.clone(), vec![output_size]))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), result_upper)
        .map_err(|_| NyError::shape_mismatch(out_shape, vec![output_size]))?;

    BoundedTensor::new(lower, upper)
}
