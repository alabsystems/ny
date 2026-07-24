// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched GEMM implementations for Conv1d/ConvTranspose1d CROWN backward (#3598).
//!
//! Extracted from `ops.rs` for file size limits. These functions replace the
//! per-row scalar loops with single GEMM + col2im/im2col operations, enabling
//! GPU dispatch via `GemmEngine`.

use faer::Mat;
use ndarray::{Array2, ArrayD};
use ny_core::{GemmEngine, NyError, Result};
use tracing::debug;

use crate::faer_parallelism::mat_mul;

/// Helper: convert faer column-major Mat to flat row-major Vec.
///
/// faer stores matrices in column-major order. GEMM output and col2im scatter
/// use row-major indexing. This helper converts in O(rows * cols) without
/// allocating an intermediate buffer. Identical to conv2d/ops.rs helper.
fn faer_mat_to_row_major(mat: &Mat<f32>, rows: usize, cols: usize) -> Vec<f32> {
    let mut flat = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let offset = row * cols;
        for col in 0..cols {
            flat[offset + col] = mat[(row, col)];
        }
    }
    flat
}

/// Dispatch a per-group GEMM through GemmEngine, falling back to faer CPU.
///
/// Returns flat row-major result of (m x k) @ (k x n) = (m x n).
fn dispatch_group_gemm(
    a: &Mat<f32>,
    b: &Mat<f32>,
    m: usize,
    k: usize,
    n: usize,
    engine: Option<&dyn GemmEngine>,
) -> Vec<f32> {
    if let Some(eng) = engine {
        let a_flat = faer_mat_to_row_major(a, m, k);
        let b_flat = faer_mat_to_row_major(b, k, n);
        match eng.gemm_f32(m, k, n, &a_flat, &b_flat) {
            Ok(result) => return result,
            Err(e) => {
                debug!("Conv1d groups>1 GEMM: engine failed, CPU fallback: {e}");
            }
        }
    }
    let result = mat_mul(a, b);
    faer_mat_to_row_major(&result, m, n)
}

/// Batched GEMM + col2im for Conv1d CROWN backward (transposed convolution).
///
/// Replaces the per-row `conv1d_transpose` loop with a single batched GEMM
/// that can dispatch to GPU via GemmEngine. The transform is:
///
///   A_reshaped = reshape(A, (N * out_len, out_c))
///   col_matrix = A_reshaped * W_col             // (N*out_len, in_c_per_group * k)
///   result = col2im_scatter(col_matrix)          // (N, in_c * in_len)
///
/// where W_col is the kernel reshaped to (out_c, in_c_per_group * k).
///
/// Reference: Conv2d `conv2d_transpose_batched_gemm` at conv2d/ops.rs:232.
/// Groups > 1 falls back to per-group CPU GEMM (rare path).
// Justification: Batched GEMM for CROWN backward requires conv parameters
// (stride, padding, dilation, groups), spatial dimensions, and engine context.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_transpose_batched_gemm(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    out_channels: usize,
    out_len: usize,
    in_len: usize,
    engine: Option<&dyn GemmEngine>,
) -> Result<Array2<f32>> {
    let num_objectives = a_coefficients.nrows();

    if kernel.ndim() != 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![kernel.ndim()],
        });
    }

    let out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let k = kernel.shape()[2];

    if out_c != out_channels {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_channels],
            got: vec![out_c],
        });
    }

    let expected_cols = out_c * out_len;
    if a_coefficients.ncols() != expected_cols {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_cols],
            got: vec![a_coefficients.ncols()],
        });
    }

    let in_c = in_c_per_group * groups;
    let conv_in_size = in_c * in_len;
    let out_c_per_group = out_c / groups;
    let kernel_cols = in_c_per_group * k;
    let total_spatial = num_objectives * out_len;

    if groups > 1 {
        // Per-group GEMM with engine dispatch (#3598).
        let mut result_flat = vec![0.0f32; num_objectives * conv_in_size];

        for g in 0..groups {
            let oc_start = g * out_c_per_group;
            let ic_start = g * in_c_per_group;

            // Kernel slice for this group: (out_c_per_group, in_c_per_group * k)
            let w_group = Mat::<f32>::from_fn(out_c_per_group, kernel_cols, |oc_local, col| {
                let ic = col / k;
                let ki = col % k;
                kernel[[oc_start + oc_local, ic, ki]]
            });

            // A slice for this group: reshape to (N * out_len, out_c_per_group)
            let a_group = Mat::<f32>::from_fn(total_spatial, out_c_per_group, |pos, oc_local| {
                let obj = pos / out_len;
                let spatial = pos % out_len;
                a_coefficients[[obj, (oc_start + oc_local) * out_len + spatial]]
            });

            let col_flat = dispatch_group_gemm(
                &a_group,
                &w_group,
                total_spatial,
                out_c_per_group,
                kernel_cols,
                engine,
            );

            // col2im scatter for this group
            for gl in 0..out_len {
                for ic_local in 0..in_c_per_group {
                    let ic = ic_start + ic_local;
                    for ki in 0..k {
                        let il = (gl * stride + ki * dilation) as isize - padding as isize;
                        if il >= 0 && il < in_len as isize {
                            let col_idx = ic_local * k + ki;
                            for obj in 0..num_objectives {
                                let col_row = obj * out_len + gl;
                                result_flat[obj * conv_in_size + ic * in_len + il as usize] +=
                                    col_flat[col_row * kernel_cols + col_idx];
                            }
                        }
                    }
                }
            }
        }

        return Array2::from_shape_vec((num_objectives, conv_in_size), result_flat)
            .map_err(|e| NyError::InternalError(format!("conv1d col2im reshape: {e}")));
    }

    // groups == 1 fast path: single GEMM with engine dispatch.

    // Step 1: Reshape A from (N, out_c * out_len) to (N * out_len, out_c).
    // Current layout: A[obj, oc * out_len + gl]
    // Target layout:  A_reshaped[obj * out_len + gl, oc]
    let a_reshaped = Mat::<f32>::from_fn(total_spatial, out_c, |pos, oc| {
        let obj = pos / out_len;
        let spatial = pos % out_len;
        a_coefficients[[obj, oc * out_len + spatial]]
    });

    // Step 2: Reshape kernel (out_c, in_c, k) -> (out_c, in_c * k).
    let w_col = Mat::<f32>::from_fn(out_c, kernel_cols, |oc, col| {
        let ic = col / k;
        let ki = col % k;
        kernel[[oc, ic, ki]]
    });

    // Step 3: Single GEMM — (total_spatial, out_c) x (out_c, in_c * k)
    // Dispatch to GemmEngine when available (GPU path), fall back to faer CPU.
    let col_flat: Vec<f32> = if let Some(eng) = engine {
        let mut a_flat = vec![0.0f32; total_spatial * out_c];
        for row in 0..total_spatial {
            for col in 0..out_c {
                a_flat[row * out_c + col] = a_reshaped[(row, col)];
            }
        }
        let mut w_flat = vec![0.0f32; out_c * kernel_cols];
        for row in 0..out_c {
            for col in 0..kernel_cols {
                w_flat[row * kernel_cols + col] = w_col[(row, col)];
            }
        }
        match eng.gemm_f32(total_spatial, out_c, kernel_cols, &a_flat, &w_flat) {
            Ok(result_flat) => {
                debug!(
                    "Conv1d CROWN backward: GemmEngine GEMM {}x{}x{} succeeded",
                    total_spatial, out_c, kernel_cols
                );
                result_flat
            }
            Err(e) => {
                debug!("Conv1d CROWN backward: GemmEngine failed, falling back to CPU: {e}");
                let result = mat_mul(&a_reshaped, &w_col);
                faer_mat_to_row_major(&result, total_spatial, kernel_cols)
            }
        }
    } else {
        let result = mat_mul(&a_reshaped, &w_col);
        faer_mat_to_row_major(&result, total_spatial, kernel_cols)
    };

    // Step 4: col2im scatter to (num_objectives, in_c * in_len).
    //
    // Pre-compute scatter indices: the mapping from GEMM output positions to
    // spatial input positions is deterministic. Computing it once eliminates
    // per-objective bounds checking from the inner loop.
    let mut scatter_map: Vec<(usize, usize, usize)> = Vec::with_capacity(out_len * in_c * k);
    for gl in 0..out_len {
        for ic in 0..in_c {
            for ki in 0..k {
                let il = (gl * stride + ki * dilation) as isize - padding as isize;
                if il >= 0 && il < in_len as isize {
                    let col_idx = ic * k + ki;
                    let out_idx = ic * in_len + il as usize;
                    scatter_map.push((gl, col_idx, out_idx));
                }
            }
        }
    }

    let mut result_flat = vec![0.0f32; num_objectives * conv_in_size];
    for obj in 0..num_objectives {
        let obj_col_offset = obj * out_len;
        let obj_result_offset = obj * conv_in_size;
        for &(spatial_offset, col_idx, out_idx) in &scatter_map {
            let col_row = obj_col_offset + spatial_offset;
            result_flat[obj_result_offset + out_idx] += col_flat[col_row * kernel_cols + col_idx];
        }
    }

    Array2::from_shape_vec((num_objectives, conv_in_size), result_flat)
        .map_err(|e| NyError::InternalError(format!("conv1d col2im reshape: {e}")))
}

/// Batched im2col + GEMM for Conv1d forward (ConvTranspose1d CROWN backward).
///
/// ConvTranspose1d backward through CROWN is a forward convolution.
/// Replaces the per-row `conv1d_single` loop with a single batched GEMM.
///
///   im2col_matrix = im2col(A)                     // (N * conv_out_len, in_c_per_group * k)
///   W_T = reshape(kernel, (in_c_per_group * k, out_c_conv))
///   result = im2col_matrix * W_T                  // (N * conv_out_len, out_c_conv)
///   output = reshape(result, (N, out_c_conv * conv_out_len))
///
/// The kernel is in ONNX ConvTranspose layout (in_c, out_c/groups, k) which is
/// treated as standard Conv1d layout: out_c_conv=kernel[0], in_c_per_group=kernel[1].
///
/// Reference: Conv2d pattern; forward conv as the backward of ConvTranspose.
// Justification: Batched im2col+GEMM for CROWN backward requires conv parameters
// (stride, padding, dilation, groups), spatial dimensions, and engine context.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_forward_batched_gemm(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    conv_in_channels: usize,
    conv_in_len: usize,
    engine: Option<&dyn GemmEngine>,
) -> Result<Array2<f32>> {
    let num_objectives = a_coefficients.nrows();

    if kernel.ndim() != 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![kernel.ndim()],
        });
    }

    let out_c_conv = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let k = kernel.shape()[2];

    if conv_in_channels != in_c_per_group * groups {
        return Err(NyError::ShapeMismatch {
            expected: vec![in_c_per_group * groups],
            got: vec![conv_in_channels],
        });
    }

    let expected_cols = conv_in_channels * conv_in_len;
    if a_coefficients.ncols() != expected_cols {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_cols],
            got: vec![a_coefficients.ncols()],
        });
    }

    // Compute output length: (padded - effective_k) / stride + 1
    let effective_k = dilation * (k - 1) + 1;
    let padded = conv_in_len + 2 * padding;
    if padded < effective_k {
        return Err(NyError::InvalidSpec(format!(
            "conv1d_forward_batched_gemm: effective kernel ({effective_k}) > padded input ({padded})"
        )));
    }
    if stride == 0 {
        return Err(NyError::InvalidSpec(
            "conv1d_forward_batched_gemm: stride must be >= 1".to_string(),
        ));
    }
    let conv_out_len = (padded - effective_k) / stride + 1;
    let conv_out_size = out_c_conv * conv_out_len;
    let total_spatial = num_objectives * conv_out_len;
    let col_width = in_c_per_group * k;
    let out_c_per_group = out_c_conv / groups;

    if groups > 1 {
        // Per-group im2col + GEMM on CPU.
        let mut result_flat = vec![0.0f32; num_objectives * conv_out_size];

        for g in 0..groups {
            let ic_start = g * in_c_per_group;
            let oc_start = g * out_c_per_group;

            let im2col = Mat::<f32>::from_fn(total_spatial, col_width, |pos, col| {
                let obj = pos / conv_out_len;
                let ol = pos % conv_out_len;
                let ic_local = col / k;
                let ki = col % k;
                let il = (ol * stride + ki * dilation) as isize - padding as isize;
                if il >= 0 && il < conv_in_len as isize {
                    let ic = ic_start + ic_local;
                    a_coefficients[[obj, ic * conv_in_len + il as usize]]
                } else {
                    0.0
                }
            });

            let w_t = Mat::<f32>::from_fn(col_width, out_c_per_group, |col, oc_local| {
                let ic = col / k;
                let ki = col % k;
                kernel[[oc_start + oc_local, ic, ki]]
            });

            let gemm_flat = dispatch_group_gemm(
                &im2col,
                &w_t,
                total_spatial,
                col_width,
                out_c_per_group,
                engine,
            );

            for obj in 0..num_objectives {
                for ol in 0..conv_out_len {
                    let row = obj * conv_out_len + ol;
                    for oc_local in 0..out_c_per_group {
                        let oc = oc_start + oc_local;
                        result_flat[obj * conv_out_size + oc * conv_out_len + ol] =
                            gemm_flat[row * out_c_per_group + oc_local];
                    }
                }
            }
        }

        return Array2::from_shape_vec((num_objectives, conv_out_size), result_flat)
            .map_err(|e| NyError::InternalError(format!("conv1d forward reshape: {e}")));
    }

    // groups == 1 fast path: single GEMM with engine dispatch.

    // Step 1: im2col — gather patches from A coefficients.
    // For each objective and output position, collect the input elements that
    // contribute to that position via the convolution kernel.
    let im2col = Mat::<f32>::from_fn(total_spatial, col_width, |pos, col| {
        let obj = pos / conv_out_len;
        let ol = pos % conv_out_len;
        let ic = col / k;
        let ki = col % k;
        let il = (ol * stride + ki * dilation) as isize - padding as isize;
        if il >= 0 && il < conv_in_len as isize {
            a_coefficients[[obj, ic * conv_in_len + il as usize]]
        } else {
            0.0
        }
    });

    // Step 2: Kernel transposed to (in_c * k, out_c_conv).
    let w_t = Mat::<f32>::from_fn(col_width, out_c_conv, |col, oc| {
        let ic = col / k;
        let ki = col % k;
        kernel[[oc, ic, ki]]
    });

    // Step 3: GEMM — (total_spatial, col_width) x (col_width, out_c_conv)
    let gemm_flat: Vec<f32> = if let Some(eng) = engine {
        let mut im2col_flat = vec![0.0f32; total_spatial * col_width];
        for row in 0..total_spatial {
            for col in 0..col_width {
                im2col_flat[row * col_width + col] = im2col[(row, col)];
            }
        }
        let mut w_flat = vec![0.0f32; col_width * out_c_conv];
        for row in 0..col_width {
            for col in 0..out_c_conv {
                w_flat[row * out_c_conv + col] = w_t[(row, col)];
            }
        }
        match eng.gemm_f32(total_spatial, col_width, out_c_conv, &im2col_flat, &w_flat) {
            Ok(result) => {
                debug!(
                    "ConvTranspose1d CROWN backward: GemmEngine GEMM {}x{}x{} succeeded",
                    total_spatial, col_width, out_c_conv
                );
                result
            }
            Err(e) => {
                debug!("ConvTranspose1d CROWN backward: GemmEngine failed, CPU fallback: {e}");
                let result = mat_mul(&im2col, &w_t);
                faer_mat_to_row_major(&result, total_spatial, out_c_conv)
            }
        }
    } else {
        let result = mat_mul(&im2col, &w_t);
        faer_mat_to_row_major(&result, total_spatial, out_c_conv)
    };

    // Step 4: Reshape from (N * conv_out_len, out_c_conv) row-major
    // to (N, out_c_conv * conv_out_len).
    // GEMM layout: gemm_flat[(obj * conv_out_len + ol) * out_c_conv + oc]
    // Target:      output[obj, oc * conv_out_len + ol]
    let mut output_flat = vec![0.0f32; num_objectives * conv_out_size];
    for obj in 0..num_objectives {
        for ol in 0..conv_out_len {
            let row = obj * conv_out_len + ol;
            for oc in 0..out_c_conv {
                output_flat[obj * conv_out_size + oc * conv_out_len + ol] =
                    gemm_flat[row * out_c_conv + oc];
            }
        }
    }

    Array2::from_shape_vec((num_objectives, conv_out_size), output_flat)
        .map_err(|e| NyError::InternalError(format!("conv1d forward reshape: {e}")))
}

/// f64 transpose-conv backward coefficient recomputation for Conv1d
/// (#vnncomp-aw-soundness).
///
/// Computes the SAME contraction as [`conv1d_transpose_batched_gemm`] (GEMM +
/// col2im) but accumulates in **f64**. Exact f32→f64 widening; only the f64 sum
/// rounds. The caller stores the directed f32 of this value and certifies a
/// per-coefficient `cast_err = |f64 − stored_f32|` plus `γ_n^f64·S`. CPU-only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_transpose_backward_coeff_f64(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    out_channels: usize,
    out_len: usize,
    in_len: usize,
) -> Result<Array2<f64>> {
    let num_objectives = a_coefficients.nrows();

    if kernel.ndim() != 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![kernel.ndim()],
        });
    }

    let out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let k = kernel.shape()[2];

    if out_c != out_channels {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_channels],
            got: vec![out_c],
        });
    }
    let expected_cols = out_c * out_len;
    if a_coefficients.ncols() != expected_cols {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_cols],
            got: vec![a_coefficients.ncols()],
        });
    }

    let in_c = in_c_per_group * groups;
    let conv_in_size = in_c * in_len;
    let out_c_per_group = out_c / groups;
    let kernel_cols = in_c_per_group * k;
    let total_spatial = num_objectives * out_len;

    let mut result_flat = vec![0.0f64; num_objectives * conv_in_size];
    for g in 0..groups {
        let oc_start = g * out_c_per_group;
        let ic_start = g * in_c_per_group;

        // f64 GEMM: (total_spatial, oc_per_group) x (oc_per_group, kernel_cols).
        let mut col_flat = vec![0.0f64; total_spatial * kernel_cols];
        for row in 0..total_spatial {
            let obj = row / out_len;
            let spatial = row % out_len;
            let col_base = row * kernel_cols;
            for oc_local in 0..out_c_per_group {
                let oc = oc_start + oc_local;
                let av = a_coefficients[[obj, oc * out_len + spatial]] as f64;
                if av == 0.0 {
                    continue;
                }
                for col in 0..kernel_cols {
                    let ic = col / k;
                    let ki = col % k;
                    col_flat[col_base + col] += av * (kernel[[oc, ic, ki]] as f64);
                }
            }
        }

        // col2im scatter (f64 accumulation).
        for gl in 0..out_len {
            for ic_local in 0..in_c_per_group {
                let ic = ic_start + ic_local;
                for ki in 0..k {
                    let il = (gl * stride + ki * dilation) as isize - padding as isize;
                    if il >= 0 && il < in_len as isize {
                        let col_idx = ic_local * k + ki;
                        for obj in 0..num_objectives {
                            let col_row = obj * out_len + gl;
                            result_flat[obj * conv_in_size + ic * in_len + il as usize] +=
                                col_flat[col_row * kernel_cols + col_idx];
                        }
                    }
                }
            }
        }
    }

    Array2::from_shape_vec((num_objectives, conv_in_size), result_flat)
        .map_err(|e| NyError::InternalError(format!("conv1d f64 col2im reshape: {e}")))
}

/// f64 forward-conv recomputation for ConvTranspose1d CROWN backward
/// (#vnncomp-aw-soundness).
///
/// Computes the SAME contraction as [`conv1d_forward_batched_gemm`] (im2col +
/// GEMM) but accumulates in **f64**. The caller stores the directed f32 of this
/// value and certifies `cast_err = |f64 − stored_f32|` plus `γ_n^f64·S`. CPU-only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_forward_backward_coeff_f64(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    conv_in_channels: usize,
    conv_in_len: usize,
) -> Result<Array2<f64>> {
    let num_objectives = a_coefficients.nrows();

    if kernel.ndim() != 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![kernel.ndim()],
        });
    }

    let out_c_conv = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let k = kernel.shape()[2];

    if conv_in_channels != in_c_per_group * groups {
        return Err(NyError::ShapeMismatch {
            expected: vec![in_c_per_group * groups],
            got: vec![conv_in_channels],
        });
    }
    let expected_cols = conv_in_channels * conv_in_len;
    if a_coefficients.ncols() != expected_cols {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_cols],
            got: vec![a_coefficients.ncols()],
        });
    }

    let effective_k = dilation * (k - 1) + 1;
    let padded = conv_in_len + 2 * padding;
    if padded < effective_k {
        return Err(NyError::InvalidSpec(format!(
            "conv1d_forward_backward_coeff_f64: effective kernel ({effective_k}) > padded input ({padded})"
        )));
    }
    if stride == 0 {
        return Err(NyError::InvalidSpec(
            "conv1d_forward_backward_coeff_f64: stride must be >= 1".to_string(),
        ));
    }
    let conv_out_len = (padded - effective_k) / stride + 1;
    let conv_out_size = out_c_conv * conv_out_len;
    let col_width = in_c_per_group * k;
    let out_c_per_group = out_c_conv / groups;

    let mut result_flat = vec![0.0f64; num_objectives * conv_out_size];
    for g in 0..groups {
        let ic_start = g * in_c_per_group;
        let oc_start = g * out_c_per_group;
        for obj in 0..num_objectives {
            for ol in 0..conv_out_len {
                for oc_local in 0..out_c_per_group {
                    let oc = oc_start + oc_local;
                    let mut acc = 0.0f64;
                    for col in 0..col_width {
                        let ic_local = col / k;
                        let ki = col % k;
                        let il = (ol * stride + ki * dilation) as isize - padding as isize;
                        if il >= 0 && il < conv_in_len as isize {
                            let ic = ic_start + ic_local;
                            let av = a_coefficients[[obj, ic * conv_in_len + il as usize]] as f64;
                            if av == 0.0 {
                                continue;
                            }
                            acc += av * (kernel[[oc, ic_local, ki]] as f64);
                        }
                    }
                    result_flat[obj * conv_out_size + oc * conv_out_len + ol] = acc;
                }
            }
        }
    }

    Array2::from_shape_vec((num_objectives, conv_out_size), result_flat)
        .map_err(|e| NyError::InternalError(format!("conv1d f64 forward reshape: {e}")))
}
