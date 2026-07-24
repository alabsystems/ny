// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CPU im2col + GEMM for the Conv2d **forward IBP** path (#hot-conv-ibp).
//!
//! The previous CPU IBP path (`bound.rs`) called the naive nested-loop
//! `conv2d_single_grouped` four times (W+/W- on lower/upper). That routine
//! recomputes strided offsets and does per-output-element dynamic `ArrayD`
//! indexing in its innermost loop (`ops.rs` `output[[oc,oh,ow]] +=
//! input[[ic, ih, iw]] * kernel[[..]]`), which dominates per-domain cost on
//! conv-heavy VNN-COMP models (e.g. the TinyImageNet ResNet root domain).
//!
//! This module gathers each input bound into a contiguous im2col matrix once
//! and replaces the convolution inner products with cache-friendly GEMMs. When
//! an injected [`GemmEngine`] is present (e.g. `--backend wgpu`/Metal) the four
//! per-group matmuls are dispatched to it via [`GemmEngine::gemm_f32`]; otherwise
//! they fall back to the CPU faer `mat_mul`. It supports `groups > 1` (the
//! GPU `ops_ibp_gemm.rs` path does not), so it serves as the default IBP forward
//! for all grouped convs and the engine-routed path for grouped convs (#hot-conv-ibp).
//!
//! Interval math (round-to-nearest, identical decomposition to the reference):
//!   W+ = max(W, 0), W- = min(W, 0)
//!   lower = im2col(in_lower) @ W+^T + im2col(in_upper) @ W-^T   (+ bias)
//!   upper = im2col(in_upper) @ W+^T + im2col(in_lower) @ W-^T   (+ bias)
//!
//! The GEMM result is mathematically identical regardless of engine; only the
//! contraction order/backend differs (engines must be a sound matmul). The
//! W+/W- interval decomposition is unchanged, so the engine path produces the
//! same sound bounds as the CPU path. See `test_conv2d_ibp_forward_engine_*`.

use faer::Mat;
use ndarray::ArrayD;
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};

use crate::bounds::{nan_propagating_max_zero, nan_propagating_min_zero};
use crate::faer_parallelism::mat_mul;

/// Computed output interval of a grouped forward conv, in (out_c, out_h, out_w)
/// layout, before bias is applied.
pub(crate) struct ConvIbpForward {
    pub lower: ArrayD<f32>,
    pub upper: ArrayD<f32>,
    pub out_h: usize,
    pub out_w: usize,
}

/// Result of one per-group matmul, abstracting over the CPU faer `Mat`
/// (column-major) and the engine's row-major flat `Vec`. Both index by
/// `(row, col)` via [`GroupGemm::get`] so the scatter loop is backend-agnostic.
enum GroupGemm {
    /// CPU faer result, indexed `mat[(row, col)]`.
    Faer(Mat<f32>),
    /// Engine result: row-major flat `Vec` of length `rows * cols`.
    Flat { data: Vec<f32>, cols: usize },
}

impl GroupGemm {
    #[inline]
    fn get(&self, row: usize, col: usize) -> f32 {
        match self {
            GroupGemm::Faer(m) => m[(row, col)],
            GroupGemm::Flat { data, cols } => data[row * cols + col],
        }
    }
}

/// Build a row-major flat buffer from a faer `Mat` (`rows * cols`, written once).
fn mat_to_row_major_flat(mat: &Mat<f32>, rows: usize, cols: usize) -> Vec<f32> {
    let mut flat = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            flat.push(mat[(r, c)]);
        }
    }
    flat
}

/// Dispatch a single (m=`spatial`, k=`col_width`, n=`out_c_per_group`) matmul
/// `lhs @ rhs` through the engine when present, else CPU faer `mat_mul`.
///
/// On engine error this degrades to the CPU faer path for this matmul — a
/// sound, numerically-faithful fallback (same product, CPU contraction). The
/// engine result is the same matrix product as faer; only the backend differs,
/// so the W+/W- interval decomposition done by the caller is unaffected.
fn group_gemm(
    engine: Option<&dyn GemmEngine>,
    lhs: &Mat<f32>,
    rhs: &Mat<f32>,
    spatial: usize,
    col_width: usize,
    out_c_per_group: usize,
) -> GroupGemm {
    if let Some(eng) = engine {
        // lhs: (spatial, col_width) row-major a; rhs: (col_width, out_c_per_group) row-major b.
        let a_flat = mat_to_row_major_flat(lhs, spatial, col_width);
        let b_flat = mat_to_row_major_flat(rhs, col_width, out_c_per_group);
        match eng.gemm_f32(spatial, col_width, out_c_per_group, &a_flat, &b_flat) {
            Ok(data) => {
                return GroupGemm::Flat {
                    data,
                    cols: out_c_per_group,
                }
            }
            Err(_e) => {
                // Engine unavailable/failed for this matmul: fall back to CPU faer
                // so we never emit unsound or NaN bounds from a dropped GEMM.
            }
        }
    }
    GroupGemm::Faer(mat_mul(lhs, rhs))
}

/// Grouped Conv2d forward IBP via im2col + GEMM on a single (in_c, H, W) input.
///
/// `input_lower` / `input_upper` must both be (in_c, H, W). Returns lower/upper
/// output bounds (out_c, out_h, out_w) using the W+/W- interval decomposition.
/// Bias is NOT added here (caller adds it, matching the reference path).
///
/// When `engine` is `Some`, the four per-group matmuls are dispatched through
/// [`GemmEngine::gemm_f32`] (GPU/accelerator); on engine error the call degrades
/// to the CPU faer `mat_mul` for that group. When `engine` is `None` the CPU
/// path is used directly. Both paths apply the identical W+/W- decomposition,
/// so the result is the same sound interval regardless of backend.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_ibp_forward_grouped(
    input_lower: &ArrayD<f32>,
    input_upper: &ArrayD<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    engine: Option<&dyn GemmEngine>,
) -> Result<ConvIbpForward> {
    if input_lower.ndim() < 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![input_lower.ndim()],
        });
    }
    if kernel.ndim() < 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }

    let in_c = input_lower.shape()[0];
    let in_h = input_lower.shape()[1];
    let in_w = input_lower.shape()[2];

    let out_c = kernel.shape()[0];
    let ker_in_c_per_group = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    let expected_in_c = ker_in_c_per_group * groups;
    if in_c != expected_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_in_c],
            got: vec![in_c],
        });
    }
    if groups == 0 || !out_c.is_multiple_of(groups) {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_ibp_forward: out_channels {out_c} not divisible by groups {groups}"
        )));
    }

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    if dh == 0 || dw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_ibp_forward: dilation must be >= 1, got ({dh},{dw})"
        )));
    }
    if sh == 0 || sw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_ibp_forward: stride must be >= 1, got ({sh},{sw})"
        )));
    }

    let eff_kh = dh * (kh - 1) + 1;
    let eff_kw = dw * (kw - 1) + 1;
    let padded_h = in_h.checked_add(2 * ph).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: padded height overflow".to_string())
    })?;
    let padded_w = in_w.checked_add(2 * pw).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: padded width overflow".to_string())
    })?;
    if padded_h < eff_kh || padded_w < eff_kw {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_ibp_forward: effective kernel ({eff_kh},{eff_kw}) larger than padded input \
             ({padded_h},{padded_w})"
        )));
    }
    let out_h = (padded_h - eff_kh) / sh + 1;
    let out_w = (padded_w - eff_kw) / sw + 1;

    let out_c_per_group = out_c / groups;
    let kernel_spatial = kh * kw;
    let col_width = ker_in_c_per_group * kernel_spatial; // K dimension of GEMM
    let spatial = out_h * out_w; // M dimension (rows of im2col)

    // Output buffers in (out_c, out_h, out_w) layout, flat row-major.
    let out_size = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_ibp_forward: output dims overflow: {out_c} * {out_h} * {out_w}"
        ))
    })?;
    let mut lower_flat = vec![0.0f32; out_size];
    let mut upper_flat = vec![0.0f32; out_size];

    // Contiguous (in_c, H*W) views of each bound for fast gather.
    // `as_slice_memory_order` succeeds for the standard-layout inputs we receive.
    let lower_std = input_lower.as_standard_layout();
    let upper_std = input_upper.as_standard_layout();
    let lower_data = lower_std.as_slice().ok_or_else(|| {
        NyError::InternalError("conv2d_ibp_forward: input_lower not contiguous".to_string())
    })?;
    let upper_data = upper_std.as_slice().ok_or_else(|| {
        NyError::InternalError("conv2d_ibp_forward: input_upper not contiguous".to_string())
    })?;
    let in_spatial = in_h * in_w;

    for g in 0..groups {
        let ic_start = g * ker_in_c_per_group;
        let oc_start = g * out_c_per_group;

        // Build im2col matrices (spatial x col_width) for lower and upper.
        // Row = output position (oh,ow); col = (ic_local, ki, kj).
        // Gathering from contiguous channel slices avoids per-element ArrayD
        // index arithmetic in the original triple-nested inner loop.
        let gather = |data: &[f32], pos: usize, col: usize| -> f32 {
            let oh = pos / out_w;
            let ow = pos % out_w;
            let ic_local = col / kernel_spatial;
            let rem = col % kernel_spatial;
            let ki = rem / kw;
            let kj = rem % kw;
            let ih = (oh * sh + ki * dh) as isize - ph as isize;
            let iw = (ow * sw + kj * dw) as isize - pw as isize;
            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                let base = (ic_start + ic_local) * in_spatial;
                data[base + ih as usize * in_w + iw as usize]
            } else {
                0.0
            }
        };
        let im2col_l = Mat::<f32>::from_fn(spatial, col_width, |p, c| gather(lower_data, p, c));
        let im2col_u = Mat::<f32>::from_fn(spatial, col_width, |p, c| gather(upper_data, p, c));

        // W+^T and W-^T for this group: (col_width, out_c_per_group).
        let wpos_t = Mat::<f32>::from_fn(col_width, out_c_per_group, |col, oc_local| {
            let ic_local = col / kernel_spatial;
            let rem = col % kernel_spatial;
            let ki = rem / kw;
            let kj = rem % kw;
            nan_propagating_max_zero(kernel[[oc_start + oc_local, ic_local, ki, kj]])
        });
        let wneg_t = Mat::<f32>::from_fn(col_width, out_c_per_group, |col, oc_local| {
            let ic_local = col / kernel_spatial;
            let rem = col % kernel_spatial;
            let ki = rem / kw;
            let kj = rem % kw;
            nan_propagating_min_zero(kernel[[oc_start + oc_local, ic_local, ki, kj]])
        });

        // lower = im2col_l @ W+^T + im2col_u @ W-^T
        // upper = im2col_u @ W+^T + im2col_l @ W-^T
        //
        // Each matmul is (spatial, col_width) x (col_width, out_c_per_group).
        // The engine (when present) and the CPU faer path compute the same
        // matrix product; only the backend/contraction order differs. The
        // W+/W- decomposition and the +/- accumulation are identical in both
        // paths, so the resulting interval is unchanged (#hot-conv-ibp).
        let l_pos = group_gemm(
            engine,
            &im2col_l,
            &wpos_t,
            spatial,
            col_width,
            out_c_per_group,
        );
        let l_neg = group_gemm(
            engine,
            &im2col_u,
            &wneg_t,
            spatial,
            col_width,
            out_c_per_group,
        );
        let u_pos = group_gemm(
            engine,
            &im2col_u,
            &wpos_t,
            spatial,
            col_width,
            out_c_per_group,
        );
        let u_neg = group_gemm(
            engine,
            &im2col_l,
            &wneg_t,
            spatial,
            col_width,
            out_c_per_group,
        );

        // Scatter GEMM result (spatial x out_c_per_group) into (out_c,out_h,out_w).
        for oc_local in 0..out_c_per_group {
            let oc = oc_start + oc_local;
            let out_base = oc * spatial;
            for p in 0..spatial {
                lower_flat[out_base + p] = l_pos.get(p, oc_local) + l_neg.get(p, oc_local);
                upper_flat[out_base + p] = u_pos.get(p, oc_local) + u_neg.get(p, oc_local);
            }
        }
    }

    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_h, out_w]), lower_flat)
        .map_err(|e| NyError::InternalError(format!("conv2d_ibp_forward lower reshape: {e}")))?;
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_h, out_w]), upper_flat)
        .map_err(|e| NyError::InternalError(format!("conv2d_ibp_forward upper reshape: {e}")))?;

    Ok(ConvIbpForward {
        lower,
        upper,
        out_h,
        out_w,
    })
}
