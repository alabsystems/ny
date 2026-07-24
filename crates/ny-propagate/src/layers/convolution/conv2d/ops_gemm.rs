// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched GEMM for Conv2d forward (ConvTranspose2d CROWN backward) (#3598).
//!
//! Replaces per-row `conv2d_single` calls with a single im2col + GEMM,
//! enabling GPU dispatch via `GemmEngine`. Mirrors the pattern in
//! `conv1d/ops_gemm.rs::conv1d_forward_batched_gemm`.

use faer::Mat;
use ndarray::{Array2, ArrayD};
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use std::{
    ffi::OsStr,
    mem::size_of,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
};
use tracing::debug;

use crate::faer_parallelism::{mat_mul, mat_mul_f64};

/// Helper: convert faer column-major Mat to flat row-major Vec.
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

/// Batched forward conv2d via im2col + GEMM for ConvTranspose2d CROWN backward.
///
/// Given CROWN A-coefficients `(N, conv_in_c * in_h * in_w)` and kernel
/// `(conv_out_c, conv_in_c, kh, kw)`, computes the forward convolution for all
/// N rows simultaneously via a single GEMM dispatched through GemmEngine.
///
/// # Algorithm
/// 1. **im2col**: Gather receptive field patches into `(N * out_h * out_w, conv_in_c * kh * kw)`
/// 2. **Kernel reshape**: `(conv_out_c, conv_in_c, kh, kw)` → W_T `(conv_in_c * kh * kw, conv_out_c)`
/// 3. **GEMM**: `im2col @ W_T` = `(N * out_h * out_w, conv_out_c)`
/// 4. **Reshape**: to `(N, conv_out_c * out_h * out_w)`
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_forward_batched_gemm(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    input_size: (usize, usize),
    engine: Option<&dyn GemmEngine>,
) -> Result<Array2<f32>> {
    let num_objectives = a_coefficients.nrows();

    if kernel.ndim() != 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }

    let conv_out_c = kernel.shape()[0]; // in_c of ConvTranspose2d
    let conv_in_c = kernel.shape()[1]; // out_c of ConvTranspose2d
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    let (in_h, in_w) = input_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    if dh == 0 || dw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: dilation must be >= 1, got ({dh},{dw})"
        )));
    }
    // Effective (dilated) kernel span: dilation*(kernel-1) + 1.
    let eff_kh = dh * (kh - 1) + 1;
    let eff_kw = dw * (kw - 1) + 1;

    let expected_cols = checked_shape_product(&[conv_in_c, in_h, in_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: input dims overflow: {conv_in_c} * {in_h} * {in_w}"
        ))
    })?;
    if a_coefficients.ncols() != expected_cols {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_cols],
            got: vec![a_coefficients.ncols()],
        });
    }

    // Output spatial dimensions of forward conv
    let pad_h = ph.checked_mul(2).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: pad_h overflows: 2 * {ph}"
        ))
    })?;
    let pad_w = pw.checked_mul(2).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: pad_w overflows: 2 * {pw}"
        ))
    })?;
    let padded_h = in_h.checked_add(pad_h).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: padded_h overflows: {in_h} + 2*{ph}"
        ))
    })?;
    let padded_w = in_w.checked_add(pad_w).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: padded_w overflows: {in_w} + 2*{pw}"
        ))
    })?;
    if padded_h < eff_kh || padded_w < eff_kw {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: effective kernel ({eff_kh},{eff_kw}) larger than \
             padded input ({padded_h},{padded_w})"
        )));
    }
    let out_h = (padded_h - eff_kh) / sh + 1;
    let out_w = (padded_w - eff_kw) / sw + 1;
    let kernel_spatial = kh.checked_mul(kw).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: kernel spatial dims overflow: {kh} * {kw}"
        ))
    })?;
    let input_spatial = in_h.checked_mul(in_w).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: input spatial dims overflow: {in_h} * {in_w}"
        ))
    })?;
    let col_width = checked_shape_product(&[conv_in_c, kh, kw]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: col_width overflows: {conv_in_c} * {kh} * {kw}"
        ))
    })?;
    let spatial_per_obj = out_h.checked_mul(out_w).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: output spatial dims overflow: {out_h} * {out_w}"
        ))
    })?;
    let total_spatial = checked_shape_product(&[num_objectives, out_h, out_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: total_spatial overflows: {num_objectives} * {out_h} * {out_w}"
        ))
    })?;
    let conv_out_size = checked_shape_product(&[conv_out_c, out_h, out_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: output dims overflow: {conv_out_c} * {out_h} * {out_w}"
        ))
    })?;

    // Step 1: im2col — gather receptive field patches from A-coefficients.
    // For each output position (oh, ow), gather conv_in_c * kh * kw values.
    let im2col_matrix = Mat::<f32>::from_fn(total_spatial, col_width, |pos, col| {
        let obj = pos / spatial_per_obj;
        let spatial = pos % spatial_per_obj;
        let oh = spatial / out_w;
        let ow = spatial % out_w;
        let ic = col / kernel_spatial;
        let rem = col % kernel_spatial;
        let ki = rem / kw;
        let kj = rem % kw;
        let ih = (oh * sh + ki * dh) as isize - ph as isize;
        let iw = (ow * sw + kj * dw) as isize - pw as isize;
        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
            a_coefficients[[obj, ic * input_spatial + ih as usize * in_w + iw as usize]]
        } else {
            0.0
        }
    });

    // Step 2: Kernel reshape — (conv_out_c, conv_in_c, kh, kw) → W_T (col_width, conv_out_c)
    let w_t = Mat::<f32>::from_fn(col_width, conv_out_c, |col, oc| {
        let ic = col / kernel_spatial;
        let rem = col % kernel_spatial;
        let ki = rem / kw;
        let kj = rem % kw;
        kernel[[oc, ic, ki, kj]]
    });

    // Step 3: GEMM — (total_spatial, col_width) @ (col_width, conv_out_c) = (total_spatial, conv_out_c)
    let gemm_flat: Vec<f32> = if let Some(eng) = engine {
        let a_flat = faer_mat_to_row_major(&im2col_matrix, total_spatial, col_width);
        let w_flat = faer_mat_to_row_major(&w_t, col_width, conv_out_c);
        match eng.gemm_f32(total_spatial, col_width, conv_out_c, &a_flat, &w_flat) {
            Ok(result) => {
                debug!(
                    "ConvTranspose2d CROWN backward: GemmEngine GEMM {}x{}x{} succeeded",
                    total_spatial, col_width, conv_out_c
                );
                result
            }
            Err(e) => {
                debug!("ConvTranspose2d CROWN backward: GemmEngine failed, CPU fallback: {e}");
                let result = mat_mul(&im2col_matrix, &w_t);
                faer_mat_to_row_major(&result, total_spatial, conv_out_c)
            }
        }
    } else {
        let result = mat_mul(&im2col_matrix, &w_t);
        faer_mat_to_row_major(&result, total_spatial, conv_out_c)
    };

    // Step 4: Reshape GEMM output to (N, conv_out_c * out_h * out_w).
    // GEMM output layout: gemm_flat[obj * out_h * out_w + oh * out_w + ow, oc]
    // Target layout: result[obj, oc * out_h * out_w + oh * out_w + ow]
    let result_len = num_objectives.checked_mul(conv_out_size).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_batched_gemm: result alloc overflow: {num_objectives} * {conv_out_size}"
        ))
    })?;
    let mut result_flat = vec![0.0f32; result_len];
    for obj in 0..num_objectives {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let gemm_row = obj * spatial_per_obj + oh * out_w + ow;
                for oc in 0..conv_out_c {
                    result_flat[obj * conv_out_size + oc * spatial_per_obj + oh * out_w + ow] =
                        gemm_flat[gemm_row * conv_out_c + oc];
                }
            }
        }
    }

    Array2::from_shape_vec((num_objectives, conv_out_size), result_flat)
        .map_err(|e| NyError::InternalError(format!("conv2d forward reshape: {e}")))
}

/// f64 forward-conv recomputation for ConvTranspose2d CROWN backward
/// (#vnncomp-aw-soundness).
///
/// Computes the SAME contraction as [`conv2d_forward_batched_gemm`] (im2col +
/// GEMM) but accumulates in **f64** (exact f32→f64 widening; only the f64 sum
/// rounds). Returns the exact real coefficient up to a sound `γ_n^f64·S` error,
/// so the caller stores the directed f32 of it and certifies a per-coefficient
/// `cast_err = |f64 − stored_f32|` plus `γ_n^f64·S`. Mirrors the Linear
/// `aw_f64_with_abssum` fix. Only exact opt-in
/// `NY_CONVTRANSPOSE_SOUND_F64_GPU=1` lets large blocks use the process-global
/// sound f64 GEMM engine (cuBLAS Dgemm); unset/malformed values and
/// unavailable/failed engines use the same faer CPU path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_forward_backward_coeff_f64(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    input_size: (usize, usize),
) -> Result<Array2<f64>> {
    conv2d_forward_backward_coeff_f64_with_deadline(
        a_coefficients,
        kernel,
        stride,
        padding,
        dilation,
        input_size,
        None,
    )
}

/// Deadline-bearing variant of [`conv2d_forward_backward_coeff_f64`]: polls the
/// deadline between objective blocks (each block is one bounded im2col+GEMM),
/// so a doomed ConvTranspose CROWN walk stops burning budget mid-recompute
/// (#wall-deadwork ConvTranspose port). `deadline: None` is byte-identical to
/// the undeadlined function.
pub(crate) fn conv2d_forward_backward_coeff_f64_with_deadline(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    input_size: (usize, usize),
    deadline: Option<std::time::Instant>,
) -> Result<Array2<f64>> {
    // This route changes only the summation implementation, not the certified
    // enclosure, but it remains dark by default until the required on-device
    // parity, contention, memory, and sealed-scorecard gates have passed.
    let sound_f64_gpu_enabled = convtranspose_sound_f64_gpu_enabled(
        std::env::var_os("NY_CONVTRANSPOSE_SOUND_F64_GPU").as_deref(),
    );
    conv2d_forward_backward_coeff_f64_with_deadline_and_engine(
        a_coefficients,
        kernel,
        stride,
        padding,
        dilation,
        input_size,
        deadline,
        super::ops_transpose_gemm::CONV_SOUND_F64_GEMM_MIN_MACS,
        |lhs, rhs, block_deadline| {
            if !sound_f64_gpu_enabled {
                return Ok(None);
            }
            crate::sound_f64_gemm::with_engine(|eng| {
                conv2d_forward_f64_block_with_engine(eng, lhs, rhs, block_deadline)
            })
            .unwrap_or(Ok(None))
        },
    )
}

/// Testable core for [`conv2d_forward_backward_coeff_f64_with_deadline`].
///
/// `try_engine` is invoked only for blocks at or above `engine_min_macs`.
/// Production supplies the lazy process-global sound-f64 engine only under the
/// exact qualification gate; tests inject explicit CPU/reordered/failing
/// engines without materializing the global `OnceLock`.
#[allow(clippy::too_many_arguments)]
fn conv2d_forward_backward_coeff_f64_with_deadline_and_engine(
    a_coefficients: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    input_size: (usize, usize),
    deadline: Option<std::time::Instant>,
    engine_min_macs: usize,
    try_engine: impl Fn(&Mat<f64>, &Mat<f64>, Option<std::time::Instant>) -> Result<Option<Vec<f64>>>,
) -> Result<Array2<f64>> {
    let num_objectives = a_coefficients.nrows();

    if kernel.ndim() != 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }

    let conv_out_c = kernel.shape()[0];
    let conv_in_c = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    let (in_h, in_w) = input_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    if sh == 0 || sw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_forward_backward_coeff_f64: stride must be >= 1, got ({sh},{sw})"
        )));
    }
    if dh == 0 || dw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_forward_backward_coeff_f64: dilation must be >= 1, got ({dh},{dw})"
        )));
    }
    if kh == 0 || kw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_forward_backward_coeff_f64: kernel spatial dimensions must be >= 1, got \
             ({kh},{kw})"
        )));
    }
    let eff_kh = kh
        .checked_sub(1)
        .and_then(|extent| dh.checked_mul(extent))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "conv2d_forward_backward_coeff_f64: effective kernel height overflow: \
                 {dh} * ({kh} - 1) + 1"
            ))
        })?;
    let eff_kw = kw
        .checked_sub(1)
        .and_then(|extent| dw.checked_mul(extent))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "conv2d_forward_backward_coeff_f64: effective kernel width overflow: \
                 {dw} * ({kw} - 1) + 1"
            ))
        })?;

    let expected_cols = checked_shape_product(&[conv_in_c, in_h, in_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_backward_coeff_f64: input dims overflow: {conv_in_c} * {in_h} * {in_w}"
        ))
    })?;
    if a_coefficients.ncols() != expected_cols {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_cols],
            got: vec![a_coefficients.ncols()],
        });
    }

    let padded_h = ph
        .checked_mul(2)
        .and_then(|pad| in_h.checked_add(pad))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "conv2d_forward_backward_coeff_f64: padded height overflow: {in_h} + 2 * {ph}"
            ))
        })?;
    let padded_w = pw
        .checked_mul(2)
        .and_then(|pad| in_w.checked_add(pad))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "conv2d_forward_backward_coeff_f64: padded width overflow: {in_w} + 2 * {pw}"
            ))
        })?;
    if padded_h < eff_kh || padded_w < eff_kw {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_forward_backward_coeff_f64: effective kernel ({eff_kh},{eff_kw}) larger than \
             padded input ({padded_h},{padded_w})"
        )));
    }
    let out_h = padded_h
        .checked_sub(eff_kh)
        .and_then(|extent| extent.checked_div(sh))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "conv2d_forward_backward_coeff_f64: output height overflow".to_string(),
            )
        })?;
    let out_w = padded_w
        .checked_sub(eff_kw)
        .and_then(|extent| extent.checked_div(sw))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "conv2d_forward_backward_coeff_f64: output width overflow".to_string(),
            )
        })?;
    let kernel_spatial = kh.checked_mul(kw).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_backward_coeff_f64: kernel area overflow: {kh} * {kw}"
        ))
    })?;
    let input_spatial = in_h.checked_mul(in_w).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_backward_coeff_f64: input area overflow: {in_h} * {in_w}"
        ))
    })?;
    let col_width = conv_in_c.checked_mul(kernel_spatial).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_backward_coeff_f64: im2col width overflow: \
             {conv_in_c} * {kernel_spatial}"
        ))
    })?;
    let spatial_per_obj = out_h.checked_mul(out_w).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_backward_coeff_f64: output area overflow: {out_h} * {out_w}"
        ))
    })?;
    let conv_out_size = checked_shape_product(&[conv_out_c, out_h, out_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_backward_coeff_f64: output dims overflow: {conv_out_c} * {out_h} * {out_w}"
        ))
    })?;
    let _kernel_matrix_bytes = col_width
        .checked_mul(conv_out_c)
        .and_then(|elements| elements.checked_mul(size_of::<f64>()))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "conv2d_forward_backward_coeff_f64: f64 kernel matrix size overflow: \
                 {col_width} * {conv_out_c}"
            ))
        })?;

    // im2col + f64 GEMM, blocked over objectives (#cgan-conv-f64-gemm).
    //
    // The previous implementation was a naive single-threaded quintuple scalar
    // loop (obj × oh × ow × oc × col_width). On cGAN-class ConvTranspose stacks
    // one intermediate-node CROWN backward hit ~10^10 scalar iterations (~22s
    // for a SINGLE node bound), starving the whole verification: intermediate
    // bounds fell back to IBP, α-CROWN never iterated, and root bounds were
    // O(10^3) loose. This replaces it with the SAME contraction as the f32 path
    // ([`conv2d_forward_batched_gemm`]) but in f64 via faer GEMM.
    //
    // SOUNDNESS: identical certificate to the scalar loop. The f32→f64 widening
    // of `a` and `kernel` is exact and `f32*f32` is exact in f64; the only
    // rounding is the f64 accumulation, and the caller's certified error
    // `cast_err + γ_n^f64·S` is summation-order independent (Higham), so
    // faer's blocked reduction order is covered — the same argument this
    // codebase already applies to routing the Conv2d twin through cuBLAS Dgemm
    // (`conv_group_col_flat_f64`). Only exact
    // `NY_CONVTRANSPOSE_SOUND_F64_GPU=1` lets blocks above the twin's crossover
    // use that same process-global sound-f64 seam; unset/malformed gates and any
    // unavailable/error/malformed engine result use the unchanged faer CPU
    // multiplication.
    //
    // Blocking caps the transient f64 im2col buffer (identity-A backwards make
    // `num_objectives` large: 4608+ rows on cGAN nodes).
    const F64_IM2COL_BLOCK_ELEMS: usize = 1 << 23; // 8M f64 = 64 MB per block
    let row_elems = spatial_per_obj.checked_mul(col_width).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_forward_backward_coeff_f64: per-objective im2col size overflow: \
             {spatial_per_obj} * {col_width}"
        ))
    })?;
    let block_objs = F64_IM2COL_BLOCK_ELEMS.checked_div(row_elems).map_or_else(
        || num_objectives.max(1),
        |quotient| quotient.clamp(1, num_objectives.max(1)),
    );

    let result_len =
        num_objectives
            .checked_mul(conv_out_size)
            .ok_or(NyError::CpuMemoryExceeded {
                required_bytes: usize::MAX,
                budget_bytes: usize::MAX,
                site: "conv2d_forward_backward_coeff_f64/result",
            })?;
    let result_bytes =
        result_len
            .checked_mul(size_of::<f64>())
            .ok_or(NyError::CpuMemoryExceeded {
                required_bytes: usize::MAX,
                budget_bytes: usize::MAX,
                site: "conv2d_forward_backward_coeff_f64/result",
            })?;
    let mut result_flat = try_zeroed_f64(result_len).ok_or(NyError::CpuMemoryExceeded {
        required_bytes: result_bytes,
        budget_bytes: usize::MAX,
        site: "conv2d_forward_backward_coeff_f64/result",
    })?;

    // Kernel reshape — (conv_out_c, conv_in_c, kh, kw) → W_T (col_width, conv_out_c).
    let w_t = Mat::<f64>::from_fn(col_width, conv_out_c, |col, oc| {
        let ic = col / kernel_spatial;
        let rem = col % kernel_spatial;
        let ki = rem / kw;
        let kj = rem % kw;
        kernel[[oc, ic, ki, kj]] as f64
    });

    let mut obj_start = 0usize;
    while obj_start < num_objectives {
        // Inter-block deadline poll: abort between bounded im2col+GEMM blocks.
        // The caller propagates the DeadlineExceeded (collector falls back to
        // sound reference bounds for the node) — never a partial result.
        if deadline.is_some_and(|dl| std::time::Instant::now() >= dl) {
            return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
        }
        let block = block_objs.min(num_objectives - obj_start);
        let block_rows = block.checked_mul(spatial_per_obj).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "conv2d_forward_backward_coeff_f64: block row count overflow: \
                 {block} * {spatial_per_obj}"
            ))
        })?;
        let _im2col_block_bytes = block_rows
            .checked_mul(col_width)
            .and_then(|elements| elements.checked_mul(size_of::<f64>()))
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "conv2d_forward_backward_coeff_f64: f64 im2col block size overflow: \
                     {block_rows} * {col_width}"
                ))
            })?;

        // im2col — gather receptive-field patches for this objective block.
        let im2col_block = Mat::<f64>::from_fn(block_rows, col_width, |pos, col| {
            let obj = obj_start
                .checked_add(pos / spatial_per_obj)
                .expect("block/objective geometry was checked before im2col");
            let spatial = pos % spatial_per_obj;
            let oh = spatial / out_w;
            let ow = spatial % out_w;
            let ic = col / kernel_spatial;
            let rem = col % kernel_spatial;
            let ki = rem / kw;
            let kj = rem % kw;
            let ih = oh
                .checked_mul(sh)
                .and_then(|base| {
                    ki.checked_mul(dh)
                        .and_then(|offset| base.checked_add(offset))
                })
                .and_then(|padded| padded.checked_sub(ph))
                .filter(|&index| index < in_h);
            let iw = ow
                .checked_mul(sw)
                .and_then(|base| {
                    kj.checked_mul(dw)
                        .and_then(|offset| base.checked_add(offset))
                })
                .and_then(|padded| padded.checked_sub(pw))
                .filter(|&index| index < in_w);
            match (ih, iw) {
                (Some(ih), Some(iw)) => {
                    let input_index = ic
                        .checked_mul(input_spatial)
                        .and_then(|base| ih.checked_mul(in_w).and_then(|row| base.checked_add(row)))
                        .and_then(|base| base.checked_add(iw))
                        .expect("validated im2col input index");
                    a_coefficients[[obj, input_index]] as f64
                }
                _ => 0.0,
            }
        });

        // Do not launch a newly-routed accelerator call after im2col itself has
        // consumed the node budget. This is an additional bounded-work poll;
        // the pre-existing inter-block poll above remains unchanged.
        if deadline.is_some_and(|dl| std::time::Instant::now() >= dl) {
            return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
        }

        // f64 GEMM: (block_rows, col_width) @ (col_width, conv_out_c).
        //
        // Thresholding avoids paying row-major staging + GPU launch costs for
        // small products. The engine helper validates output length and
        // finiteness. No engine, any engine error, or malformed output all
        // leave `engine_flat=None` and run the exact prior faer CPU path.
        let macs = block_rows
            .checked_mul(col_width)
            .and_then(|work| work.checked_mul(conv_out_c))
            .unwrap_or(usize::MAX);
        let engine_flat = if macs >= engine_min_macs {
            try_engine(&im2col_block, &w_t, deadline)?
                .filter(|v| block_rows.checked_mul(conv_out_c) == Some(v.len()))
        } else {
            None
        };
        let cpu_block = if engine_flat.is_none() {
            // An accelerator may fail only after consuming the remaining node
            // budget. Do not then start the expensive CPU retry; propagate the
            // same sound reference-bound fallback used by every other deadline
            // poll. Before expiry, fail-open behavior is unchanged.
            if deadline.is_some_and(|dl| std::time::Instant::now() >= dl) {
                return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
            }
            Some(mat_mul_f64(&im2col_block, &w_t))
        } else {
            debug!(
                "ConvTranspose2d sound f64 recompute: sound GemmEngine {}x{}x{} succeeded",
                block_rows, col_width, conv_out_c
            );
            None
        };

        // Scatter to (obj, oc * spatial + oh*out_w + ow) layout.
        for local_obj in 0..block {
            let obj = obj_start
                .checked_add(local_obj)
                .expect("block/objective geometry was checked before scatter");
            for spatial in 0..spatial_per_obj {
                let gemm_row = local_obj
                    .checked_mul(spatial_per_obj)
                    .and_then(|base| base.checked_add(spatial))
                    .expect("block/GEMM row geometry was checked before scatter");
                for oc in 0..conv_out_c {
                    let result_index = obj
                        .checked_mul(conv_out_size)
                        .and_then(|base| {
                            oc.checked_mul(spatial_per_obj)
                                .and_then(|channel| base.checked_add(channel))
                        })
                        .and_then(|base| base.checked_add(spatial))
                        .expect("result geometry was checked before scatter");
                    result_flat[result_index] = if let Some(ref flat) = engine_flat {
                        let engine_index = gemm_row
                            .checked_mul(conv_out_c)
                            .and_then(|base| base.checked_add(oc))
                            .expect("engine output geometry was checked before scatter");
                        flat[engine_index]
                    } else {
                        cpu_block
                            .as_ref()
                            .expect("CPU block exists when engine result is absent")[(gemm_row, oc)]
                    };
                }
            }
        }
        obj_start = obj_start
            .checked_add(block)
            .expect("block progression is bounded by num_objectives");
    }

    Array2::from_shape_vec((num_objectives, conv_out_size), result_flat)
        .map_err(|e| NyError::InternalError(format!("conv2d f64 forward reshape: {e}")))
}

/// Optional paired direct-layout recompute for ConvTranspose lower/upper CROWN
/// coefficients. This route is admitted only by the same exact
/// `NY_CONVTRANSPOSE_SOUND_F64_GPU=1` gate as the existing per-side CUDA seam.
/// Gate-off execution never enters this function's engine or allocation path.
///
/// Unlike the legacy route, this constructs exact-widened row-major im2col
/// blocks directly, builds the shared row-major kernel once, and submits the two
/// independent products through `GemmEngine::gemm_f64_pair_shared_rhs`. Any
/// unsupported geometry, allocation failure, engine error/panic, or malformed
/// output returns `Ok(None)` so the caller can run the unchanged legacy faer
/// recomputes. Deadline expiry remains a hard `DeadlineExceeded`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_forward_backward_coeff_f64_pair_with_deadline(
    lower: &Array2<f32>,
    upper: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    input_size: (usize, usize),
    deadline: Option<std::time::Instant>,
) -> Result<Option<(Array2<f64>, Array2<f64>)>> {
    let enabled = convtranspose_sound_f64_gpu_enabled(
        std::env::var_os("NY_CONVTRANSPOSE_SOUND_F64_GPU").as_deref(),
    );
    if !enabled {
        return Ok(None);
    }
    crate::sound_f64_gemm::with_engine(|engine| {
        conv2d_forward_backward_coeff_f64_pair_with_deadline_and_engine(
            lower,
            upper,
            kernel,
            stride,
            padding,
            dilation,
            input_size,
            deadline,
            super::ops_transpose_gemm::CONV_SOUND_F64_GEMM_MIN_MACS,
            engine,
            &CONVTRANSPOSE_SOUND_F64_GEMM_GATE,
        )
    })
    .unwrap_or(Ok(None))
}

#[derive(Clone, Copy, Debug)]
struct DirectF64PairGeometry {
    num_objectives: usize,
    conv_out_c: usize,
    kw: usize,
    in_h: usize,
    in_w: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    kernel_spatial: usize,
    input_spatial: usize,
    col_width: usize,
    spatial_per_obj: usize,
    out_w: usize,
    conv_out_size: usize,
    block_objs: usize,
    result_len: usize,
}

impl DirectF64PairGeometry {
    fn checked(
        lower: &Array2<f32>,
        upper: &Array2<f32>,
        kernel: &ArrayD<f32>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        input_size: (usize, usize),
    ) -> Option<Self> {
        if lower.raw_dim() != upper.raw_dim() || kernel.ndim() != 4 {
            return None;
        }
        let num_objectives = lower.nrows();
        let conv_out_c = kernel.shape()[0];
        let conv_in_c = kernel.shape()[1];
        let kh = kernel.shape()[2];
        let kw = kernel.shape()[3];
        let (in_h, in_w) = input_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (dh, dw) = dilation;
        if sh == 0 || sw == 0 || dh == 0 || dw == 0 || kh == 0 || kw == 0 {
            return None;
        }
        let eff_kh = kh.checked_sub(1)?.checked_mul(dh)?.checked_add(1)?;
        let eff_kw = kw.checked_sub(1)?.checked_mul(dw)?.checked_add(1)?;
        let input_spatial = in_h.checked_mul(in_w)?;
        let expected_cols = conv_in_c.checked_mul(input_spatial)?;
        if lower.ncols() != expected_cols {
            return None;
        }
        let padded_h = ph.checked_mul(2)?.checked_add(in_h)?;
        let padded_w = pw.checked_mul(2)?.checked_add(in_w)?;
        if padded_h < eff_kh || padded_w < eff_kw {
            return None;
        }
        let out_h = padded_h
            .checked_sub(eff_kh)?
            .checked_div(sh)?
            .checked_add(1)?;
        let out_w = padded_w
            .checked_sub(eff_kw)?
            .checked_div(sw)?
            .checked_add(1)?;
        let kernel_spatial = kh.checked_mul(kw)?;
        let col_width = conv_in_c.checked_mul(kernel_spatial)?;
        let spatial_per_obj = out_h.checked_mul(out_w)?;
        let conv_out_size = conv_out_c.checked_mul(spatial_per_obj)?;
        let row_elems = spatial_per_obj.checked_mul(col_width)?;
        if row_elems == 0 || col_width == 0 || conv_out_c == 0 {
            return None;
        }
        // Keep the same 64-MiB per-side f64 im2col quantum as the legacy path.
        const F64_IM2COL_BLOCK_ELEMS: usize = 1 << 23;
        let block_objs = F64_IM2COL_BLOCK_ELEMS
            .checked_div(row_elems)?
            .clamp(1, num_objectives.max(1));
        let result_len = num_objectives.checked_mul(conv_out_size)?;
        // Representability checks for all three simultaneously-live operands
        // and both retained outputs. Allocation itself remains fallible.
        col_width
            .checked_mul(conv_out_c)?
            .checked_mul(size_of::<f64>())?;
        result_len.checked_mul(size_of::<f64>())?.checked_mul(2)?;

        Some(Self {
            num_objectives,
            conv_out_c,
            kw,
            in_h,
            in_w,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            kernel_spatial,
            input_spatial,
            col_width,
            spatial_per_obj,
            out_w,
            conv_out_size,
            block_objs,
            result_len,
        })
    }
}

fn direct_f64_kernel_row_major(
    kernel: &ArrayD<f32>,
    geometry: DirectF64PairGeometry,
) -> Option<Vec<f64>> {
    let len = geometry.col_width.checked_mul(geometry.conv_out_c)?;
    let mut row_major = try_zeroed_f64(len)?;
    for col in 0..geometry.col_width {
        let ic = col / geometry.kernel_spatial;
        let rem = col % geometry.kernel_spatial;
        let ki = rem / geometry.kw;
        let kj = rem % geometry.kw;
        let base = col * geometry.conv_out_c;
        for oc in 0..geometry.conv_out_c {
            row_major[base + oc] = f64::from(kernel[[oc, ic, ki, kj]]);
        }
    }
    Some(row_major)
}

fn direct_f64_im2col_row_major(
    coefficients: &Array2<f32>,
    geometry: DirectF64PairGeometry,
    obj_start: usize,
    block: usize,
) -> Option<Vec<f64>> {
    let block_rows = block.checked_mul(geometry.spatial_per_obj)?;
    let len = block_rows.checked_mul(geometry.col_width)?;
    let mut row_major = try_zeroed_f64(len)?;
    for pos in 0..block_rows {
        let obj = obj_start.checked_add(pos / geometry.spatial_per_obj)?;
        let spatial = pos % geometry.spatial_per_obj;
        let oh = spatial / geometry.out_w;
        let ow = spatial % geometry.out_w;
        let row_base = pos.checked_mul(geometry.col_width)?;
        for col in 0..geometry.col_width {
            let ic = col / geometry.kernel_spatial;
            let rem = col % geometry.kernel_spatial;
            let ki = rem / geometry.kw;
            let kj = rem % geometry.kw;
            let ih = oh
                .checked_mul(geometry.sh)
                .and_then(|base| ki.checked_mul(geometry.dh)?.checked_add(base))
                .and_then(|padded| padded.checked_sub(geometry.ph))
                .filter(|&index| index < geometry.in_h);
            let iw = ow
                .checked_mul(geometry.sw)
                .and_then(|base| kj.checked_mul(geometry.dw)?.checked_add(base))
                .and_then(|padded| padded.checked_sub(geometry.pw))
                .filter(|&index| index < geometry.in_w);
            if let (Some(ih), Some(iw)) = (ih, iw) {
                let input_index = ic
                    .checked_mul(geometry.input_spatial)?
                    .checked_add(ih.checked_mul(geometry.in_w)?)?
                    .checked_add(iw)?;
                row_major[row_base + col] = f64::from(coefficients[[obj, input_index]]);
            }
        }
    }
    Some(row_major)
}

fn scatter_direct_f64_block(
    gemm: &[f64],
    result: &mut [f64],
    geometry: DirectF64PairGeometry,
    obj_start: usize,
    block: usize,
) -> bool {
    let Some(block_rows) = block.checked_mul(geometry.spatial_per_obj) else {
        return false;
    };
    if block_rows.checked_mul(geometry.conv_out_c) != Some(gemm.len()) {
        return false;
    }
    for local_obj in 0..block {
        let Some(obj) = obj_start.checked_add(local_obj) else {
            return false;
        };
        for spatial in 0..geometry.spatial_per_obj {
            let Some(gemm_row) = local_obj
                .checked_mul(geometry.spatial_per_obj)
                .and_then(|base| base.checked_add(spatial))
            else {
                return false;
            };
            for oc in 0..geometry.conv_out_c {
                let Some(result_index) = obj
                    .checked_mul(geometry.conv_out_size)
                    .and_then(|base| {
                        oc.checked_mul(geometry.spatial_per_obj)
                            .and_then(|channel| base.checked_add(channel))
                    })
                    .and_then(|base| base.checked_add(spatial))
                else {
                    return false;
                };
                let Some(gemm_index) = gemm_row
                    .checked_mul(geometry.conv_out_c)
                    .and_then(|base| base.checked_add(oc))
                else {
                    return false;
                };
                result[result_index] = gemm[gemm_index];
            }
        }
    }
    true
}

const CONVTRANSPOSE_F64_PAIR_REPORT_AT: [u64; 4] = [1, 64, 4_096, 262_144];
static CONVTRANSPOSE_F64_PAIR_BLOCKS: AtomicU64 = AtomicU64::new(0);
static CONVTRANSPOSE_F64_PAIR_ROWS: AtomicU64 = AtomicU64::new(0);
static CONVTRANSPOSE_F64_PAIR_PREP_US: AtomicU64 = AtomicU64::new(0);
static CONVTRANSPOSE_F64_PAIR_DISPATCH_US: AtomicU64 = AtomicU64::new(0);
static CONVTRANSPOSE_F64_PAIR_SCATTER_US: AtomicU64 = AtomicU64::new(0);

fn saturating_micros(elapsed: std::time::Duration) -> u64 {
    u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX)
}

fn record_convtranspose_f64_pair_block(
    rows: usize,
    prep: std::time::Duration,
    dispatch: std::time::Duration,
    scatter: std::time::Duration,
) {
    let blocks = CONVTRANSPOSE_F64_PAIR_BLOCKS
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1);
    let rows = u64::try_from(rows).unwrap_or(u64::MAX);
    let rows = CONVTRANSPOSE_F64_PAIR_ROWS
        .fetch_add(rows, Ordering::Relaxed)
        .wrapping_add(rows);
    let prep_us = saturating_micros(prep);
    let prep_us = CONVTRANSPOSE_F64_PAIR_PREP_US
        .fetch_add(prep_us, Ordering::Relaxed)
        .wrapping_add(prep_us);
    let dispatch_us = saturating_micros(dispatch);
    let dispatch_us = CONVTRANSPOSE_F64_PAIR_DISPATCH_US
        .fetch_add(dispatch_us, Ordering::Relaxed)
        .wrapping_add(dispatch_us);
    let scatter_us = saturating_micros(scatter);
    let scatter_us = CONVTRANSPOSE_F64_PAIR_SCATTER_US
        .fetch_add(scatter_us, Ordering::Relaxed)
        .wrapping_add(scatter_us);
    if CONVTRANSPOSE_F64_PAIR_REPORT_AT.contains(&blocks) {
        debug!(
            "ConvTranspose2d sound f64 pair aggregate: blocks={blocks} rows={rows} \
             prep_us={prep_us} dispatch_us={dispatch_us} scatter_us={scatter_us}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn conv2d_forward_backward_coeff_f64_pair_with_deadline_and_engine(
    lower: &Array2<f32>,
    upper: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    input_size: (usize, usize),
    deadline: Option<std::time::Instant>,
    engine_min_macs: usize,
    engine: &dyn GemmEngine,
    dispatch_gate: &Mutex<()>,
) -> Result<Option<(Array2<f64>, Array2<f64>)>> {
    let Some(geometry) =
        DirectF64PairGeometry::checked(lower, upper, kernel, stride, padding, dilation, input_size)
    else {
        return Ok(None);
    };
    if geometry.num_objectives == 0 {
        let lower = Array2::from_shape_vec((0, geometry.conv_out_size), Vec::new())
            .map_err(|e| NyError::InternalError(format!("conv2d f64 pair lower reshape: {e}")))?;
        let upper = Array2::from_shape_vec((0, geometry.conv_out_size), Vec::new())
            .map_err(|e| NyError::InternalError(format!("conv2d f64 pair upper reshape: {e}")))?;
        return Ok(Some((lower, upper)));
    }

    let first_block = geometry.block_objs.min(geometry.num_objectives);
    let first_rows = first_block
        .checked_mul(geometry.spatial_per_obj)
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d f64 pair: first block row overflow".to_string())
        })?;
    let first_macs = first_rows
        .checked_mul(geometry.col_width)
        .and_then(|work| work.checked_mul(geometry.conv_out_c))
        .unwrap_or(usize::MAX);
    if first_macs < engine_min_macs {
        return Ok(None);
    }

    let Some(kernel_row_major) = direct_f64_kernel_row_major(kernel, geometry) else {
        return Ok(None);
    };
    let Some(mut lower_result) = try_zeroed_f64(geometry.result_len) else {
        return Ok(None);
    };
    let Some(mut upper_result) = try_zeroed_f64(geometry.result_len) else {
        return Ok(None);
    };

    let mut obj_start = 0usize;
    while obj_start < geometry.num_objectives {
        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
        }
        let block = geometry.block_objs.min(geometry.num_objectives - obj_start);
        let block_rows = block
            .checked_mul(geometry.spatial_per_obj)
            .ok_or_else(|| NyError::InvalidSpec("conv2d f64 pair: block row overflow".into()))?;

        let prep_started = std::time::Instant::now();
        let (lower_block, upper_block) = rayon::join(
            || direct_f64_im2col_row_major(lower, geometry, obj_start, block),
            || direct_f64_im2col_row_major(upper, geometry, obj_start, block),
        );
        let (Some(lower_block), Some(upper_block)) = (lower_block, upper_block) else {
            return Ok(None);
        };
        let prep_elapsed = prep_started.elapsed();
        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
        }

        let dispatch_started = std::time::Instant::now();
        let Some([lower_gemm, upper_gemm]) = conv2d_forward_f64_pair_with_engine_and_gate(
            engine,
            block_rows,
            geometry.col_width,
            geometry.conv_out_c,
            [&lower_block, &upper_block],
            &kernel_row_major,
            deadline,
            dispatch_gate,
        )?
        else {
            return Ok(None);
        };
        let dispatch_elapsed = dispatch_started.elapsed();

        let scatter_started = std::time::Instant::now();
        let (lower_ok, upper_ok) = rayon::join(
            || scatter_direct_f64_block(&lower_gemm, &mut lower_result, geometry, obj_start, block),
            || scatter_direct_f64_block(&upper_gemm, &mut upper_result, geometry, obj_start, block),
        );
        if !lower_ok || !upper_ok {
            return Ok(None);
        }
        let scatter_elapsed = scatter_started.elapsed();
        record_convtranspose_f64_pair_block(
            block_rows,
            prep_elapsed,
            dispatch_elapsed,
            scatter_elapsed,
        );
        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
        }
        obj_start = obj_start.checked_add(block).ok_or_else(|| {
            NyError::InvalidSpec("conv2d f64 pair: block progression overflow".into())
        })?;
    }

    let lower = match Array2::from_shape_vec(
        (geometry.num_objectives, geometry.conv_out_size),
        lower_result,
    ) {
        Ok(result) => result,
        Err(_) => return Ok(None),
    };
    let upper = match Array2::from_shape_vec(
        (geometry.num_objectives, geometry.conv_out_size),
        upper_result,
    ) {
        Ok(result) => result,
        Err(_) => return Ok(None),
    };
    Ok(Some((lower, upper)))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CheckedSoundF64GemmShape {
    lhs_len: usize,
    rhs_len: usize,
    output_len: usize,
}

/// Runtime admission for the still-experimental ConvTranspose sound-f64 GPU
/// route. Exact `OsStr` comparison makes an unset, malformed, or non-Unicode
/// value fail closed; sealed measurements separately accept only typed `0/1`.
fn convtranspose_sound_f64_gpu_enabled(raw: Option<&OsStr>) -> bool {
    raw == Some(OsStr::new("1"))
}

/// Allocate initialized f64 storage without invoking the allocator's
/// infallible allocation path. Once reserve succeeds, `resize` cannot allocate
/// and cloning `f64` cannot panic.
fn try_zeroed_f64(len: usize) -> Option<Vec<f64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).ok()?;
    values.resize(len, 0.0);
    Some(values)
}

/// Drop a caught Rust panic payload without allowing a pathological
/// user-supplied panicking destructor to start a second unwind.
fn drop_caught_panic_payload(payload: Box<dyn std::any::Any + Send>) {
    if let Err(drop_payload) =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(payload)))
    {
        std::mem::forget(drop_payload);
    }
}

/// Validate every dimension, cuBLAS leading dimension, element product, and
/// byte product before staging operands or entering an opaque `GemmEngine`.
///
/// The process-global sound engine is CUDA in production. Its row-major
/// `C=A·B` call is implemented as column-major `Cᵀ=Bᵀ·Aᵀ`, whose cuBLAS
/// dimensions are `(n,m,k)` and leading dimensions `(n,k,n)`. All must be
/// positive `i32`; rejecting an unrepresentable shape here returns control to
/// the unchanged faer CPU path instead of allowing a truncating cast inside an
/// engine implementation.
fn checked_sound_f64_gemm_shape(m: usize, k: usize, n: usize) -> Option<CheckedSoundF64GemmShape> {
    let m_i32 = i32::try_from(m).ok().filter(|&dim| dim > 0)?;
    let k_i32 = i32::try_from(k).ok().filter(|&dim| dim > 0)?;
    let n_i32 = i32::try_from(n).ok().filter(|&dim| dim > 0)?;
    let cublas_dimensions = [n_i32, m_i32, k_i32];
    let leading_dimensions = [n_i32, k_i32, n_i32];
    if cublas_dimensions
        .into_iter()
        .chain(leading_dimensions)
        .any(|dim| dim <= 0)
    {
        return None;
    }

    let lhs_len = m.checked_mul(k)?;
    let rhs_len = k.checked_mul(n)?;
    let output_len = m.checked_mul(n)?;
    let lhs_bytes = lhs_len.checked_mul(size_of::<f64>())?;
    let rhs_bytes = rhs_len.checked_mul(size_of::<f64>())?;
    let output_bytes = output_len.checked_mul(size_of::<f64>())?;
    lhs_bytes
        .checked_add(rhs_bytes)?
        .checked_add(output_bytes)?;

    Some(CheckedSoundF64GemmShape {
        lhs_len,
        rhs_len,
        output_len,
    })
}

/// ConvTranspose lower/upper f64 recomputes use `rayon::join`, while the CUDA
/// engine itself owns one serialized cuBLAS stream. Serialize this call site
/// before entering that opaque engine lock, then recheck the node deadline.
///
/// Other subsystems can still contend on an engine-internal lock after this
/// gate. If that happens, a call admitted before its deadline may launch late;
/// the post-call poll below discards its result as `DeadlineExceeded`. We do not
/// claim that an opaque engine can be cancelled after admission.
static CONVTRANSPOSE_SOUND_F64_GEMM_GATE: Mutex<()> = Mutex::new(());

fn conv2d_forward_f64_pair_with_engine_and_gate(
    eng: &dyn GemmEngine,
    m: usize,
    k: usize,
    n: usize,
    lhs: [&[f64]; 2],
    rhs: &[f64],
    deadline: Option<std::time::Instant>,
    dispatch_gate: &Mutex<()>,
) -> Result<Option<[Vec<f64>; 2]>> {
    let Some(shape) = checked_sound_f64_gemm_shape(m, k, n) else {
        return Ok(None);
    };
    if lhs.iter().any(|operand| operand.len() != shape.lhs_len) || rhs.len() != shape.rhs_len {
        return Ok(None);
    }
    if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
        return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
    }
    let _dispatch_permit = match dispatch_gate.lock() {
        Ok(permit) => permit,
        Err(_) => return Ok(None),
    };
    if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
        return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
    }
    let engine_call = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        eng.gemm_f64_pair_shared_rhs(m, k, n, lhs, rhs)
    }));
    let result = match engine_call {
        Ok(Ok(output))
            if output.iter().all(|member| {
                member.len() == shape.output_len && member.iter().all(|value| value.is_finite())
            }) =>
        {
            Some(output)
        }
        Ok(_) => None,
        Err(payload) => {
            debug!(
                "ConvTranspose2d sound f64 pair: GemmEngine Rust unwind; failing open to legacy"
            );
            drop_caught_panic_payload(payload);
            None
        }
    };
    if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
        return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
    }
    Ok(result)
}

/// Convert the two faer matrices to exact row-major f64 operands and invoke an
/// explicit sound GEMM engine. Kept separate from the process-global lookup so
/// operand layout and fail-open behavior are CPU-unit-testable.
fn conv2d_forward_f64_block_with_engine(
    eng: &dyn GemmEngine,
    lhs: &Mat<f64>,
    rhs: &Mat<f64>,
    deadline: Option<std::time::Instant>,
) -> Result<Option<Vec<f64>>> {
    conv2d_forward_f64_block_with_engine_and_gate(
        eng,
        lhs,
        rhs,
        deadline,
        &CONVTRANSPOSE_SOUND_F64_GEMM_GATE,
    )
}

fn conv2d_forward_f64_block_with_engine_and_gate(
    eng: &dyn GemmEngine,
    lhs: &Mat<f64>,
    rhs: &Mat<f64>,
    deadline: Option<std::time::Instant>,
    dispatch_gate: &Mutex<()>,
) -> Result<Option<Vec<f64>>> {
    let m = lhs.nrows();
    let k = lhs.ncols();
    if rhs.nrows() != k {
        return Ok(None);
    }
    let n = rhs.ncols();
    let Some(shape) = checked_sound_f64_gemm_shape(m, k, n) else {
        return Ok(None);
    };

    let Some(mut lhs_row_major) = try_zeroed_f64(shape.lhs_len) else {
        return Ok(None);
    };
    for row in 0..m {
        for col in 0..k {
            lhs_row_major[row * k + col] = lhs[(row, col)];
        }
    }
    if deadline.is_some_and(|dl| std::time::Instant::now() >= dl) {
        return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
    }
    let Some(mut rhs_row_major) = try_zeroed_f64(shape.rhs_len) else {
        // `lhs_row_major` drops before the caller enters the unchanged faer
        // fallback, so failed accelerator staging cannot compound peak memory.
        return Ok(None);
    };
    for row in 0..k {
        for col in 0..n {
            rhs_row_major[row * n + col] = rhs[(row, col)];
        }
    }
    if deadline.is_some_and(|dl| std::time::Instant::now() >= dl) {
        return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
    }

    let _dispatch_permit = match dispatch_gate.lock() {
        Ok(permit) => permit,
        // A prior panic while this local optimization gate was held must not
        // decide a verdict. Fail open without entering the engine.
        Err(_) => return Ok(None),
    };
    if deadline.is_some_and(|dl| std::time::Instant::now() >= dl) {
        return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
    }
    // `GemmEngine::gemm_f64` is a safe-Rust seam. Catch a Rust unwind from the
    // current invocation (including an ordinary poisoned mutex in a safe
    // engine implementation) so this optimization fails open and, because the
    // unwind is consumed while `_dispatch_permit` is alive, does not poison
    // the local ConvTranspose gate. A conforming safe implementation cannot
    // retain the borrowed operand slices after returning/unwinding.
    //
    // This deliberately makes no claim about `panic=abort`, process aborts,
    // native faults, hangs, undefined behavior, or an unsafe FFI engine that
    // returns before its native work has quiesced; none is catchable here.
    let engine_call = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        eng.gemm_f64(m, k, n, &lhs_row_major, &rhs_row_major)
    }));
    let result = match engine_call {
        Ok(Ok(v)) if v.len() == shape.output_len && v.iter().all(|value| value.is_finite()) => {
            Some(v)
        }
        Ok(_) => None,
        Err(payload) => {
            debug!(
                "ConvTranspose2d sound f64 recompute: GemmEngine Rust unwind; failing open to faer"
            );
            // A user-supplied panic payload can itself have a panicking Drop.
            // Consume an ordinary payload normally, but catch that pathological
            // second unwind too. Only its replacement payload is leaked.
            drop_caught_panic_payload(payload);
            None
        }
    };
    if deadline.is_some_and(|dl| std::time::Instant::now() >= dl) {
        return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Barrier, Mutex};
    use std::time::{Duration, Instant};

    use faer::Mat;
    use ndarray::{array, Array2, ArrayD, IxDyn};
    use ny_core::{GemmEngine, NaiveCpuGemmEngine, NyError, Result};

    use super::{
        checked_sound_f64_gemm_shape,
        conv2d_forward_backward_coeff_f64_pair_with_deadline_and_engine,
        conv2d_forward_backward_coeff_f64_with_deadline_and_engine,
        conv2d_forward_f64_block_with_engine_and_gate,
        conv2d_forward_f64_pair_with_engine_and_gate, convtranspose_sound_f64_gpu_enabled,
        try_zeroed_f64, CheckedSoundF64GemmShape,
    };

    struct CountingCpuF64Engine {
        calls: AtomicUsize,
    }

    impl CountingCpuF64Engine {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl GemmEngine for CountingCpuF64Engine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }

        fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            NaiveCpuGemmEngine.gemm_f64(m, k, n, a, b)
        }
    }

    struct RecordingPairF64Engine {
        pair_calls: AtomicUsize,
        scalar_calls: AtomicUsize,
        rhs_lengths: Mutex<Vec<usize>>,
    }

    impl RecordingPairF64Engine {
        fn new() -> Self {
            Self {
                pair_calls: AtomicUsize::new(0),
                scalar_calls: AtomicUsize::new(0),
                rhs_lengths: Mutex::new(Vec::new()),
            }
        }
    }

    impl GemmEngine for RecordingPairF64Engine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }

        fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
            self.scalar_calls.fetch_add(1, Ordering::Relaxed);
            NaiveCpuGemmEngine.gemm_f64(m, k, n, a, b)
        }

        fn gemm_f64_pair_shared_rhs(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: [&[f64]; 2],
            b: &[f64],
        ) -> Result<[Vec<f64>; 2]> {
            self.pair_calls.fetch_add(1, Ordering::Relaxed);
            self.rhs_lengths
                .lock()
                .expect("recording pair lock")
                .push(b.len());
            Ok([
                NaiveCpuGemmEngine.gemm_f64(m, k, n, a[0], b)?,
                NaiveCpuGemmEngine.gemm_f64(m, k, n, a[1], b)?,
            ])
        }
    }

    struct NonFinitePairF64Engine {
        pair_calls: AtomicUsize,
    }

    impl NonFinitePairF64Engine {
        fn new() -> Self {
            Self {
                pair_calls: AtomicUsize::new(0),
            }
        }
    }

    impl GemmEngine for NonFinitePairF64Engine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            unreachable!("the direct f64 pair must not call f32 GEMM")
        }

        fn gemm_f64_pair_shared_rhs(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            _a: [&[f64]; 2],
            _b: &[f64],
        ) -> Result<[Vec<f64>; 2]> {
            self.pair_calls.fetch_add(1, Ordering::Relaxed);
            Ok([vec![f64::NAN; m * n], vec![0.0; m * n]])
        }
    }

    struct ReverseF64Engine {
        calls: AtomicUsize,
    }

    impl ReverseF64Engine {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl GemmEngine for ReverseF64Engine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }

        fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut out = vec![0.0f64; m * n];
            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0f64;
                    for inner in (0..k).rev() {
                        sum += a[row * k + inner] * b[inner * n + col];
                    }
                    out[row * n + col] = sum;
                }
            }
            Ok(out)
        }
    }

    struct FailingF64Engine {
        calls: AtomicUsize,
        delay: Duration,
    }

    impl FailingF64Engine {
        fn new(delay: Duration) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                delay,
            }
        }
    }

    impl GemmEngine for FailingF64Engine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(NyError::UnsupportedOp("injected f32 failure".to_string()))
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(self.delay);
            Err(NyError::UnsupportedOp(
                "injected sound-f64 GEMM failure".to_string(),
            ))
        }
    }

    struct MalformedF64Engine {
        calls: AtomicUsize,
    }

    impl MalformedF64Engine {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    struct NonFiniteF64Engine {
        calls: AtomicUsize,
    }

    impl NonFiniteF64Engine {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl GemmEngine for NonFiniteF64Engine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(NyError::UnsupportedOp("injected f32 failure".to_string()))
        }

        fn gemm_f64(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(vec![f64::NAN; m * n])
        }
    }

    impl GemmEngine for MalformedF64Engine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(NyError::UnsupportedOp("injected f32 failure".to_string()))
        }

        fn gemm_f64(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(vec![0.0f64; m * n + 1])
        }
    }

    struct HoldingF64Engine {
        calls: AtomicUsize,
        entered_first_call: Barrier,
        hold_first_call: Duration,
    }

    impl HoldingF64Engine {
        fn new(hold_first_call: Duration) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                entered_first_call: Barrier::new(2),
                hold_first_call,
            }
        }
    }

    impl GemmEngine for HoldingF64Engine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }

        fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            if call == 0 {
                self.entered_first_call.wait();
                std::thread::sleep(self.hold_first_call);
            }
            NaiveCpuGemmEngine.gemm_f64(m, k, n, a, b)
        }
    }

    struct PanickingF64Engine {
        calls: AtomicUsize,
    }

    impl PanickingF64Engine {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl GemmEngine for PanickingF64Engine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("injected f32 panic")
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            panic!("injected current-call f64 panic")
        }
    }

    struct PoisonedInnerF64Engine {
        calls: AtomicUsize,
        inner: Mutex<()>,
    }

    impl PoisonedInnerF64Engine {
        fn new() -> Self {
            let inner = Mutex::new(());
            let poison = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _poison_guard = inner.lock().expect("fresh injected engine mutex");
                panic!("inject prior engine-inner poison");
            }));
            assert!(poison.is_err());
            assert!(inner.is_poisoned());
            Self {
                calls: AtomicUsize::new(0),
                inner,
            }
        }
    }

    impl GemmEngine for PoisonedInnerF64Engine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("injected f32 panic")
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let _guard = self
                .inner
                .lock()
                .expect("injected engine-inner mutex is poisoned");
            unreachable!("pre-poisoned engine mutex must panic")
        }
    }

    fn identity_engine_operands() -> (Mat<f64>, Mat<f64>) {
        (
            Mat::<f64>::from_fn(2, 2, |row, col| (row * 2 + col + 1) as f64),
            Mat::<f64>::from_fn(2, 2, |row, col| if row == col { 1.0 } else { 0.0 }),
        )
    }

    fn fixture_inputs() -> (Array2<f32>, ArrayD<f32>) {
        let a = Array2::from_shape_fn((3, 2 * 4 * 5), |(row, col)| {
            (((row * 37 + col * 13) % 29) as f32 - 14.0) / 7.0
        });
        let kernel = ArrayD::from_shape_fn(IxDyn(&[3, 2, 2, 2]), |idx| {
            (((idx[0] * 17 + idx[1] * 11 + idx[2] * 5 + idx[3] * 3) % 19) as f32 - 9.0) / 8.0
        });
        (a, kernel)
    }

    fn run_fixture(
        engine_min_macs: usize,
        deadline: Option<Instant>,
        try_engine: impl Fn(&Mat<f64>, &Mat<f64>, Option<Instant>) -> Result<Option<Vec<f64>>>,
    ) -> Result<Array2<f64>> {
        let (a, kernel) = fixture_inputs();
        conv2d_forward_backward_coeff_f64_with_deadline_and_engine(
            &a,
            &kernel,
            (2, 1),
            (1, 0),
            (1, 2),
            (4, 5),
            deadline,
            engine_min_macs,
            try_engine,
        )
    }

    fn run_pair_fixture(
        lower: &Array2<f32>,
        upper: &Array2<f32>,
        engine_min_macs: usize,
        deadline: Option<Instant>,
        engine: &dyn GemmEngine,
        gate: &Mutex<()>,
    ) -> Result<Option<(Array2<f64>, Array2<f64>)>> {
        let (_, kernel) = fixture_inputs();
        conv2d_forward_backward_coeff_f64_pair_with_deadline_and_engine(
            lower,
            upper,
            &kernel,
            (2, 1),
            (1, 0),
            (1, 2),
            (4, 5),
            deadline,
            engine_min_macs,
            engine,
            gate,
        )
    }

    fn run_cancellation_fixture(
        a: &Array2<f32>,
        kernel: &ArrayD<f32>,
        try_engine: impl Fn(&Mat<f64>, &Mat<f64>, Option<Instant>) -> Result<Option<Vec<f64>>>,
    ) -> Result<Array2<f64>> {
        conv2d_forward_backward_coeff_f64_with_deadline_and_engine(
            a,
            kernel,
            (1, 1),
            (0, 0),
            (1, 1),
            (1, 1),
            None,
            0,
            try_engine,
        )
    }

    fn invalid_geometry_error(
        a: &Array2<f32>,
        kernel: &ArrayD<f32>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        input_size: (usize, usize),
    ) -> NyError {
        conv2d_forward_backward_coeff_f64_with_deadline_and_engine(
            a,
            kernel,
            stride,
            padding,
            dilation,
            input_size,
            None,
            0,
            |_, _, _| panic!("invalid geometry must be rejected before engine admission"),
        )
        .expect_err("invalid convolution geometry")
    }

    #[test]
    fn convtranspose_sound_f64_cublas_shape_guard_checks_boundaries_without_allocating() {
        assert_eq!(
            checked_sound_f64_gemm_shape(2, 3, 4),
            Some(CheckedSoundF64GemmShape {
                lhs_len: 6,
                rhs_len: 12,
                output_len: 8,
            })
        );
        assert!(checked_sound_f64_gemm_shape(0, 1, 1).is_none());
        assert!(checked_sound_f64_gemm_shape(1, 0, 1).is_none());
        assert!(checked_sound_f64_gemm_shape(1, 1, 0).is_none());

        let beyond_i32 = i32::MAX as usize + 1;
        assert!(checked_sound_f64_gemm_shape(beyond_i32, 1, 1).is_none());
        assert!(checked_sound_f64_gemm_shape(1, beyond_i32, 1).is_none());
        assert!(checked_sound_f64_gemm_shape(1, 1, beyond_i32).is_none());

        #[cfg(target_pointer_width = "64")]
        {
            let max_i32 = i32::MAX as usize;
            // Independently exercise m, ldb=k, and lda/ldc=n at their exact
            // positive-i32 boundary. This pure helper allocates nothing.
            assert!(checked_sound_f64_gemm_shape(max_i32, 1, 1).is_some());
            assert!(checked_sound_f64_gemm_shape(1, max_i32, 1).is_some());
            assert!(checked_sound_f64_gemm_shape(1, 1, max_i32).is_some());

            // Dimensions individually fit i32, but each possible pair product
            // has an f64 byte-size overflow and must fail closed to CPU.
            assert!(checked_sound_f64_gemm_shape(max_i32, max_i32, 1).is_none());
            assert!(checked_sound_f64_gemm_shape(1, max_i32, max_i32).is_none());
            assert!(checked_sound_f64_gemm_shape(max_i32, 1, max_i32).is_none());
        }
    }

    #[test]
    fn convtranspose_sound_f64_gate_is_exact_and_default_off() {
        assert!(!convtranspose_sound_f64_gpu_enabled(None));
        assert!(convtranspose_sound_f64_gpu_enabled(Some(
            std::ffi::OsStr::new("1")
        )));
        for malformed in ["", "0", "true", " 1", "1 ", "01", "１"] {
            assert!(
                !convtranspose_sound_f64_gpu_enabled(Some(std::ffi::OsStr::new(malformed))),
                "{malformed:?} must fail closed"
            );
        }

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!convtranspose_sound_f64_gpu_enabled(Some(
                std::ffi::OsStr::from_bytes(&[b'1', 0xff])
            )));
        }
    }

    #[test]
    fn convtranspose_sound_f64_staging_allocation_is_fallible_without_large_allocation() {
        assert_eq!(try_zeroed_f64(4), Some(vec![0.0; 4]));
        assert!(
            try_zeroed_f64(usize::MAX).is_none(),
            "capacity overflow must decline accelerator staging"
        );
    }

    #[test]
    fn convtranspose_sound_f64_geometry_overflow_fails_before_allocation_or_engine() {
        let a = Array2::<f32>::zeros((1, 1));
        let unit_kernel = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1, 1]));

        let zero_stride = invalid_geometry_error(&a, &unit_kernel, (0, 1), (0, 0), (1, 1), (1, 1));
        assert!(format!("{zero_stride}").contains("stride"));

        let tall_kernel = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 2, 1]));
        let dilation_overflow =
            invalid_geometry_error(&a, &tall_kernel, (1, 1), (0, 0), (usize::MAX, 1), (1, 1));
        assert!(format!("{dilation_overflow}").contains("effective kernel height overflow"));

        let padding_overflow =
            invalid_geometry_error(&a, &unit_kernel, (1, 1), (usize::MAX, 0), (1, 1), (1, 1));
        assert!(format!("{padding_overflow}").contains("padded height overflow"));

        // A zero input-channel dimension keeps ndarray storage empty while
        // allowing pure near-usize geometry checks without a huge allocation.
        let empty_a = Array2::<f32>::zeros((1, 0));
        let empty_kernel = ArrayD::<f32>::zeros(IxDyn(&[1, 0, 1, 1]));
        let input_area_overflow = invalid_geometry_error(
            &empty_a,
            &empty_kernel,
            (1, 1),
            (0, 0),
            (1, 1),
            (usize::MAX, usize::MAX),
        );
        assert!(format!("{input_area_overflow}").contains("input area overflow"));

        let max_safe_symmetric_padding = usize::MAX / 2;
        let output_area_overflow = invalid_geometry_error(
            &empty_a,
            &empty_kernel,
            (1, 1),
            (max_safe_symmetric_padding, max_safe_symmetric_padding),
            (1, 1),
            (1, 1),
        );
        assert!(format!("{output_area_overflow}").contains("output area overflow"));
    }

    #[test]
    fn convtranspose_sound_f64_engine_matches_cpu_and_threshold_bypasses_small_work() {
        let cpu = run_fixture(usize::MAX, None, |_, _, _| {
            panic!("threshold must bypass engine")
        })
        .expect("faer CPU reference");

        let engine = CountingCpuF64Engine::new();
        let gate = Mutex::new(());
        let accelerated = run_fixture(0, None, |lhs, rhs, deadline| {
            conv2d_forward_f64_block_with_engine_and_gate(&engine, lhs, rhs, deadline, &gate)
        })
        .expect("explicit sound-f64 engine");
        assert_eq!(engine.calls.load(Ordering::Relaxed), 1);
        assert_eq!(accelerated.raw_dim(), cpu.raw_dim());
        for (index, (&got, &want)) in accelerated.iter().zip(cpu.iter()).enumerate() {
            let tolerance = 2.0e-13 * (1.0 + want.abs());
            assert!(
                (got - want).abs() <= tolerance,
                "engine/CPU parity mismatch at {index}: got={got:e} want={want:e}"
            );
        }

        let thresholded_engine = CountingCpuF64Engine::new();
        let thresholded_gate = Mutex::new(());
        let thresholded = run_fixture(usize::MAX, None, |lhs, rhs, deadline| {
            conv2d_forward_f64_block_with_engine_and_gate(
                &thresholded_engine,
                lhs,
                rhs,
                deadline,
                &thresholded_gate,
            )
        })
        .expect("small product remains on CPU");
        assert_eq!(thresholded_engine.calls.load(Ordering::Relaxed), 0);
        assert_eq!(thresholded, cpu);
    }

    #[test]
    fn convtranspose_sound_f64_direct_pair_matches_legacy_and_uses_shared_rhs_seam() {
        let (lower, kernel) = fixture_inputs();
        let upper = lower.mapv(|value| value.mul_add(-0.375, 0.125));
        let legacy_lower = conv2d_forward_backward_coeff_f64_with_deadline_and_engine(
            &lower,
            &kernel,
            (2, 1),
            (1, 0),
            (1, 2),
            (4, 5),
            None,
            usize::MAX,
            |_, _, _| Ok(None),
        )
        .expect("legacy lower faer");
        let legacy_upper = conv2d_forward_backward_coeff_f64_with_deadline_and_engine(
            &upper,
            &kernel,
            (2, 1),
            (1, 0),
            (1, 2),
            (4, 5),
            None,
            usize::MAX,
            |_, _, _| Ok(None),
        )
        .expect("legacy upper faer");

        let engine = RecordingPairF64Engine::new();
        let gate = Mutex::new(());
        let (paired_lower, paired_upper) =
            run_pair_fixture(&lower, &upper, 0, None, &engine, &gate)
                .expect("direct pair route")
                .expect("pair admitted");
        assert_eq!(engine.pair_calls.load(Ordering::Relaxed), 1);
        assert_eq!(engine.scalar_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            *engine.rhs_lengths.lock().expect("recorded RHS lengths"),
            vec![2 * 2 * 2 * 3]
        );

        for (label, got, want) in [
            ("lower", &paired_lower, &legacy_lower),
            ("upper", &paired_upper, &legacy_upper),
        ] {
            assert_eq!(got.raw_dim(), want.raw_dim());
            for (index, (&actual, &expected)) in got.iter().zip(want.iter()).enumerate() {
                let tolerance = 2.0e-13 * (1.0 + expected.abs());
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{label} direct/legacy mismatch at {index}: got={actual:e} want={expected:e}"
                );
            }
        }
    }

    #[test]
    fn convtranspose_sound_f64_direct_pair_handles_nonstandard_array_layout() {
        let (standard_lower, _) = fixture_inputs();
        let standard_upper = standard_lower.mapv(|value| value * 0.25 - 0.75);
        let lower = Array2::from_shape_fn(
            (standard_lower.ncols(), standard_lower.nrows()),
            |(col, row)| standard_lower[[row, col]],
        )
        .reversed_axes();
        let upper = Array2::from_shape_fn(
            (standard_upper.ncols(), standard_upper.nrows()),
            |(col, row)| standard_upper[[row, col]],
        )
        .reversed_axes();
        assert!(!lower.is_standard_layout());
        assert!(!upper.is_standard_layout());

        let engine = RecordingPairF64Engine::new();
        let gate = Mutex::new(());
        let nonstandard = run_pair_fixture(&lower, &upper, 0, None, &engine, &gate)
            .expect("nonstandard direct pair")
            .expect("pair admitted");
        let standard = run_pair_fixture(&standard_lower, &standard_upper, 0, None, &engine, &gate)
            .expect("standard direct pair")
            .expect("pair admitted");
        assert_eq!(nonstandard, standard);
    }

    #[test]
    fn convtranspose_sound_f64_direct_pair_matches_legacy_geometry_grid() {
        let engine = RecordingPairF64Engine::new();
        let gate = Mutex::new(());
        let (in_h, in_w) = (5usize, 6usize);
        let (in_c, out_c, objectives) = (2usize, 3usize, 2usize);
        let lower = Array2::from_shape_fn((objectives, in_c * in_h * in_w), |(row, col)| {
            (((row * 31 + col * 17) % 41) as f32 - 20.0) / 9.0
        });
        let upper = lower.mapv(|value| value.mul_add(-0.625, 0.375));
        let mut cases = 0usize;

        for (kh, kw) in [(1, 1), (2, 3), (3, 2)] {
            for stride in [(1, 1), (2, 1), (2, 2)] {
                for padding in [(0, 0), (1, 0), (1, 1)] {
                    for dilation in [(1, 1), (2, 1)] {
                        let eff_h = dilation.0 * (kh - 1) + 1;
                        let eff_w = dilation.1 * (kw - 1) + 1;
                        if in_h + 2 * padding.0 < eff_h || in_w + 2 * padding.1 < eff_w {
                            continue;
                        }
                        let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, in_c, kh, kw]), |idx| {
                            (((idx[0] * 19 + idx[1] * 13 + idx[2] * 7 + idx[3] * 5) % 37) as f32
                                - 18.0)
                                / 11.0
                        });
                        let legacy_lower =
                            conv2d_forward_backward_coeff_f64_with_deadline_and_engine(
                                &lower,
                                &kernel,
                                stride,
                                padding,
                                dilation,
                                (in_h, in_w),
                                None,
                                usize::MAX,
                                |_, _, _| Ok(None),
                            )
                            .expect("geometry-grid legacy lower");
                        let legacy_upper =
                            conv2d_forward_backward_coeff_f64_with_deadline_and_engine(
                                &upper,
                                &kernel,
                                stride,
                                padding,
                                dilation,
                                (in_h, in_w),
                                None,
                                usize::MAX,
                                |_, _, _| Ok(None),
                            )
                            .expect("geometry-grid legacy upper");
                        let (paired_lower, paired_upper) =
                            conv2d_forward_backward_coeff_f64_pair_with_deadline_and_engine(
                                &lower,
                                &upper,
                                &kernel,
                                stride,
                                padding,
                                dilation,
                                (in_h, in_w),
                                None,
                                0,
                                &engine,
                                &gate,
                            )
                            .expect("geometry-grid direct pair")
                            .expect("valid geometry must admit pair");

                        for (side, got, want) in [
                            ("lower", &paired_lower, &legacy_lower),
                            ("upper", &paired_upper, &legacy_upper),
                        ] {
                            assert_eq!(got.raw_dim(), want.raw_dim());
                            for (index, (&actual, &expected)) in
                                got.iter().zip(want.iter()).enumerate()
                            {
                                let tolerance = 3.0e-13 * (1.0 + expected.abs());
                                assert!(
                                    (actual - expected).abs() <= tolerance,
                                    "{side} geometry-grid mismatch case={cases} index={index}: \
                                     got={actual:e} want={expected:e}; kh={kh} kw={kw} \
                                     stride={stride:?} padding={padding:?} dilation={dilation:?}"
                                );
                            }
                        }
                        cases += 1;
                    }
                }
            }
        }
        assert!(cases >= 40, "expected broad geometry coverage, got {cases}");
        assert_eq!(engine.pair_calls.load(Ordering::Relaxed), cases);
        assert_eq!(engine.scalar_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn convtranspose_sound_f64_direct_pair_threshold_and_nonfinite_fail_open() {
        let (lower, _) = fixture_inputs();
        let upper = lower.mapv(|value| -value);
        let recording = RecordingPairF64Engine::new();
        let gate = Mutex::new(());
        let thresholded = run_pair_fixture(&lower, &upper, usize::MAX, None, &recording, &gate)
            .expect("thresholded pair");
        assert!(thresholded.is_none());
        assert_eq!(recording.pair_calls.load(Ordering::Relaxed), 0);

        let nonfinite = NonFinitePairF64Engine::new();
        let rejected = run_pair_fixture(&lower, &upper, 0, None, &nonfinite, &gate)
            .expect("non-finite output rejection");
        assert!(rejected.is_none());
        assert_eq!(nonfinite.pair_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn convtranspose_sound_f64_pair_helper_rejects_malformed_shapes_before_engine() {
        let engine = RecordingPairF64Engine::new();
        let gate = Mutex::new(());
        let lhs = [1.0f64, 2.0];
        let rhs = [3.0f64, 4.0];
        let declined = conv2d_forward_f64_pair_with_engine_and_gate(
            &engine,
            1,
            2,
            1,
            [&lhs, &[]],
            &rhs,
            None,
            &gate,
        )
        .expect("malformed operands fail open");
        assert!(declined.is_none());
        assert_eq!(engine.pair_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn convtranspose_sound_f64_pair_engine_panic_fails_open_and_gate_remains_reusable() {
        let panicking = PanickingF64Engine::new();
        let healthy = RecordingPairF64Engine::new();
        let gate = Mutex::new(());
        let lower = [1.0f64, 2.0, 3.0, 4.0];
        let upper = [5.0f64, 6.0, 7.0, 8.0];
        let identity = [1.0f64, 0.0, 0.0, 1.0];

        let declined = conv2d_forward_f64_pair_with_engine_and_gate(
            &panicking,
            2,
            2,
            2,
            [&lower, &upper],
            &identity,
            None,
            &gate,
        )
        .expect("safe-Rust pair panic must fail open");
        assert!(declined.is_none());
        assert_eq!(panicking.calls.load(Ordering::Relaxed), 1);
        assert!(
            !gate.is_poisoned(),
            "caught pair unwind must not poison local dispatch gate"
        );

        let recovered = conv2d_forward_f64_pair_with_engine_and_gate(
            &healthy,
            2,
            2,
            2,
            [&lower, &upper],
            &identity,
            None,
            &gate,
        )
        .expect("subsequent healthy pair")
        .expect("healthy pair admitted");
        assert_eq!(recovered, [lower.to_vec(), upper.to_vec()]);
        assert_eq!(healthy.pair_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn convtranspose_sound_f64_reordered_engine_and_cpu_enclose_cancellation() {
        // Exact-real contraction: 2^53 + 1 - 2^53 = 1. Legal accumulation
        // orders may produce 0 or 1, but Higham's order-independent gamma_n*S
        // enclosure must contain 1 for both CPU and engine results.
        let big = 2.0f32.powi(53);
        let a = array![[big, 1.0, -big]];
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 3, 1, 1]), vec![1.0f32; 3]).expect("kernel");
        let cpu = run_cancellation_fixture(&a, &kernel, |_, _, _| Ok(None)).expect("faer CPU");
        let reverse = ReverseF64Engine::new();
        let gate = Mutex::new(());
        let reordered = run_cancellation_fixture(&a, &kernel, |lhs, rhs, deadline| {
            conv2d_forward_f64_block_with_engine_and_gate(&reverse, lhs, rhs, deadline, &gate)
        })
        .expect("reordered engine");
        assert_eq!(reverse.calls.load(Ordering::Relaxed), 1);

        let n = 3.0f64;
        let unit_roundoff = f64::EPSILON / 2.0;
        let gamma_n = n * unit_roundoff / (1.0 - n * unit_roundoff);
        let sum_abs = 2.0 * f64::from(big) + 1.0;
        let exact_real = 1.0f64;
        for (label, got) in [("CPU", cpu[[0, 0]]), ("engine", reordered[[0, 0]])] {
            assert!(
                (got - exact_real).abs() <= gamma_n * sum_abs,
                "{label}={got:e} lies outside certified [{:e}, {:e}]",
                exact_real - gamma_n * sum_abs,
                exact_real + gamma_n * sum_abs
            );
        }
    }

    #[test]
    fn convtranspose_sound_f64_engine_error_or_malformed_output_falls_back_bitwise() {
        let cpu = run_fixture(usize::MAX, None, |_, _, _| Ok(None)).expect("faer CPU reference");

        let failing = FailingF64Engine::new(Duration::ZERO);
        let failing_gate = Mutex::new(());
        let failed = run_fixture(0, None, |lhs, rhs, deadline| {
            conv2d_forward_f64_block_with_engine_and_gate(
                &failing,
                lhs,
                rhs,
                deadline,
                &failing_gate,
            )
        })
        .expect("engine error must fail open to CPU");
        assert_eq!(failing.calls.load(Ordering::Relaxed), 1);
        assert_eq!(failed, cpu);

        let malformed = MalformedF64Engine::new();
        let malformed_gate = Mutex::new(());
        let wrong_len = run_fixture(0, None, |lhs, rhs, deadline| {
            conv2d_forward_f64_block_with_engine_and_gate(
                &malformed,
                lhs,
                rhs,
                deadline,
                &malformed_gate,
            )
        })
        .expect("malformed engine output must fail open to CPU");
        assert_eq!(malformed.calls.load(Ordering::Relaxed), 1);
        assert_eq!(wrong_len, cpu);

        let non_finite = NonFiniteF64Engine::new();
        let non_finite_gate = Mutex::new(());
        let rejected_non_finite = run_fixture(0, None, |lhs, rhs, deadline| {
            conv2d_forward_f64_block_with_engine_and_gate(
                &non_finite,
                lhs,
                rhs,
                deadline,
                &non_finite_gate,
            )
        })
        .expect("non-finite engine output must fail open to CPU");
        assert_eq!(non_finite.calls.load(Ordering::Relaxed), 1);
        assert_eq!(rejected_non_finite, cpu);
    }

    #[test]
    fn convtranspose_sound_f64_current_engine_panic_fails_open_and_gate_remains_reusable() {
        let panicking = PanickingF64Engine::new();
        let healthy = CountingCpuF64Engine::new();
        let gate = Mutex::new(());
        let (lhs, rhs) = identity_engine_operands();

        let failed_open =
            conv2d_forward_f64_block_with_engine_and_gate(&panicking, &lhs, &rhs, None, &gate)
                .expect("safe-Rust engine panic must fail open");
        assert_eq!(failed_open, None);
        assert_eq!(panicking.calls.load(Ordering::Relaxed), 1);
        assert!(
            !gate.is_poisoned(),
            "caught engine unwind must not poison local dispatch gate"
        );

        let recovered =
            conv2d_forward_f64_block_with_engine_and_gate(&healthy, &lhs, &rhs, None, &gate)
                .expect("subsequent healthy engine call");
        assert_eq!(recovered, Some(vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(healthy.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn convtranspose_sound_f64_prepoisoned_engine_inner_fails_open_and_gate_remains_reusable() {
        let poisoned = PoisonedInnerF64Engine::new();
        let healthy = CountingCpuF64Engine::new();
        let gate = Mutex::new(());
        let (lhs, rhs) = identity_engine_operands();

        let failed_open =
            conv2d_forward_f64_block_with_engine_and_gate(&poisoned, &lhs, &rhs, None, &gate)
                .expect("pre-poisoned safe-Rust engine must fail open");
        assert_eq!(failed_open, None);
        assert_eq!(poisoned.calls.load(Ordering::Relaxed), 1);
        assert!(
            !gate.is_poisoned(),
            "caught inner-mutex panic must not poison local dispatch gate"
        );

        let recovered =
            conv2d_forward_f64_block_with_engine_and_gate(&healthy, &lhs, &rhs, None, &gate)
                .expect("subsequent healthy engine call");
        assert_eq!(recovered, Some(vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(healthy.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn convtranspose_sound_f64_prepoisoned_dispatch_gate_declines_engine() {
        let engine = CountingCpuF64Engine::new();
        let gate = Mutex::new(());
        let poison = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _poison_guard = gate.lock().expect("fresh local dispatch gate");
            panic!("inject prior local dispatch-gate poison");
        }));
        assert!(poison.is_err());
        assert!(gate.is_poisoned());
        let (lhs, rhs) = identity_engine_operands();

        let declined =
            conv2d_forward_f64_block_with_engine_and_gate(&engine, &lhs, &rhs, None, &gate)
                .expect("poisoned optimization gate must fail open");
        assert_eq!(declined, None);
        assert_eq!(
            engine.calls.load(Ordering::Relaxed),
            0,
            "poisoned dispatch gate must not enter GemmEngine"
        );
    }

    #[test]
    fn convtranspose_sound_f64_failed_engine_honors_deadline_before_cpu_retry() {
        let failing = FailingF64Engine::new(Duration::from_millis(350));
        let gate = Mutex::new(());
        let deadline = Instant::now() + Duration::from_millis(250);
        let error = run_fixture(0, Some(deadline), |lhs, rhs, deadline| {
            conv2d_forward_f64_block_with_engine_and_gate(&failing, lhs, rhs, deadline, &gate)
        })
        .expect_err("expired budget after engine failure must skip CPU retry");
        assert_eq!(failing.calls.load(Ordering::Relaxed), 1);
        assert!(
            matches!(error, NyError::DeadlineExceeded(_)),
            "expected DeadlineExceeded, got {error:?}"
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn convtranspose_sound_f64_contended_gate_rechecks_deadline_before_engine_call() {
        let engine = HoldingF64Engine::new(Duration::from_millis(200));
        let gate = Mutex::new(());
        let lhs = Mat::<f64>::from_fn(2, 2, |row, col| (row * 2 + col + 1) as f64);
        let rhs = Mat::<f64>::from_fn(2, 2, |row, col| if row == col { 1.0 } else { 0.0 });

        let (first_result, contended_result) = std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                conv2d_forward_f64_block_with_engine_and_gate(&engine, &lhs, &rhs, None, &gate)
            });
            engine.entered_first_call.wait();

            let deadline = Instant::now() + Duration::from_millis(50);
            let contended_engine = &engine;
            let contended_lhs = &lhs;
            let contended_rhs = &rhs;
            let contended_gate = &gate;
            let contended = scope.spawn(move || {
                conv2d_forward_f64_block_with_engine_and_gate(
                    contended_engine,
                    contended_lhs,
                    contended_rhs,
                    Some(deadline),
                    contended_gate,
                )
            });
            (
                first.join().expect("first helper thread"),
                contended.join().expect("contended helper thread"),
            )
        });

        assert_eq!(
            first_result.expect("first engine call"),
            Some(vec![1.0, 2.0, 3.0, 4.0])
        );
        let error = contended_result
            .expect_err("expired contender must stop after acquiring the dispatch gate");
        assert!(
            matches!(error, NyError::DeadlineExceeded(_)),
            "expected DeadlineExceeded, got {error:?}"
        );
        assert_eq!(
            engine.calls.load(Ordering::Relaxed),
            1,
            "contended expired request must not enter GemmEngine"
        );
    }
}
