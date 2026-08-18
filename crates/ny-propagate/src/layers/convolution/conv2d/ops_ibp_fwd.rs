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
use ndarray::{Array1, ArrayD, ArrayViewD};
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};
use std::time::Instant;

use crate::bounds::{nan_propagating_max_zero, nan_propagating_min_zero};
use crate::faer_parallelism::mat_mul;

/// Maximum multiply/add work admitted between finite-deadline polls.
const DEADLINE_CPU_POLL_OPS: usize = 4_096;

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
    if input_lower.ndim() != 3 || input_upper.ndim() != 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![input_lower.ndim().max(input_upper.ndim())],
        });
    }
    if input_lower.shape() != input_upper.shape() {
        return Err(NyError::ShapeMismatch {
            expected: input_lower.shape().to_vec(),
            got: input_upper.shape().to_vec(),
        });
    }
    if kernel.ndim() != 4 {
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

    if groups == 0 {
        return Err(NyError::InvalidSpec(
            "conv2d_ibp_forward: groups must be >= 1".to_string(),
        ));
    }
    if out_c == 0 || ker_in_c_per_group == 0 || kh == 0 || kw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_ibp_forward: kernel dimensions must be nonzero, got {:?}",
            kernel.shape()
        )));
    }
    let expected_in_c = ker_in_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: grouped input channels overflow".to_string())
    })?;
    if in_c != expected_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_in_c],
            got: vec![in_c],
        });
    }
    if !out_c.is_multiple_of(groups) {
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

    let eff_kh = kh
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dh))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d_ibp_forward: effective kernel height overflow".to_string())
        })?;
    let eff_kw = kw
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dw))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d_ibp_forward: effective kernel width overflow".to_string())
        })?;
    let padded_h = ph
        .checked_mul(2)
        .and_then(|padding| in_h.checked_add(padding))
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d_ibp_forward: padded height overflow".to_string())
        })?;
    let padded_w = pw
        .checked_mul(2)
        .and_then(|padding| in_w.checked_add(padding))
        .ok_or_else(|| {
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
    let kernel_spatial = kh.checked_mul(kw).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: kernel spatial size overflow".to_string())
    })?;
    let col_width = ker_in_c_per_group
        .checked_mul(kernel_spatial)
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d_ibp_forward: im2col width overflow".to_string())
        })?; // K dimension of GEMM
    let spatial = out_h.checked_mul(out_w).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: output spatial size overflow".to_string())
    })?; // M dimension (rows of im2col)
    checked_shape_product(&[spatial, col_width]).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: im2col size overflow".to_string())
    })?;
    checked_shape_product(&[col_width, out_c / groups]).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: kernel matrix size overflow".to_string())
    })?;

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
    let in_spatial = in_h.checked_mul(in_w).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: input spatial size overflow".to_string())
    })?;

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
            let ih = oh
                .checked_mul(sh)
                .and_then(|base| ki.checked_mul(dh)?.checked_add(base))
                .and_then(|padded| padded.checked_sub(ph))
                .filter(|&index| index < in_h);
            let iw = ow
                .checked_mul(sw)
                .and_then(|base| kj.checked_mul(dw)?.checked_add(base))
                .and_then(|padded| padded.checked_sub(pw))
                .filter(|&index| index < in_w);
            if let (Some(ih), Some(iw)) = (ih, iw) {
                let base = (ic_start + ic_local) * in_spatial;
                data[base + ih * in_w + iw]
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

/// Validated geometry shared by the finite-deadline scalar conv forwards.
///
/// Both the plain f32 deadline contraction and the certified f64
/// dual-accumulator kernel below iterate the SAME tap enumeration; validating
/// (and deriving out_h/out_w/spatial) in one place removes the risk of the two
/// arms drifting on grouped/dilated/padded/strided index math.
pub(crate) struct Conv2dDeadlineGeometry {
    pub in_h: usize,
    pub in_w: usize,
    pub out_c: usize,
    pub in_c_per_group: usize,
    pub out_c_per_group: usize,
    pub kh: usize,
    pub kw: usize,
    pub out_h: usize,
    pub out_w: usize,
    /// `out_h * out_w`.
    pub spatial: usize,
    /// `out_c * out_h * out_w`.
    pub out_size: usize,
}

/// Validate shapes/params for the deadline scalar forwards and derive the
/// output geometry. Error variants and messages are those of the historical
/// [`conv2d_ibp_forward_grouped_with_deadline`] validation block, verbatim.
fn conv2d_deadline_geometry(
    input_lower: &ArrayViewD<'_, f32>,
    input_upper: &ArrayViewD<'_, f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> Result<Conv2dDeadlineGeometry> {
    if input_lower.ndim() != 3 || input_upper.ndim() != 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![input_lower.ndim().max(input_upper.ndim())],
        });
    }
    if input_lower.shape() != input_upper.shape() {
        return Err(NyError::ShapeMismatch {
            expected: input_lower.shape().to_vec(),
            got: input_upper.shape().to_vec(),
        });
    }
    if kernel.ndim() != 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }
    if groups == 0 {
        return Err(NyError::InvalidSpec(
            "conv2d_ibp_forward: groups must be >= 1".to_string(),
        ));
    }

    let in_c = input_lower.shape()[0];
    let in_h = input_lower.shape()[1];
    let in_w = input_lower.shape()[2];
    let out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    if out_c == 0 || in_c_per_group == 0 || kh == 0 || kw == 0 {
        return Err(NyError::InvalidSpec(
            "conv2d_ibp_forward: kernel dimensions must be nonzero".to_string(),
        ));
    }
    let expected_in_c = in_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: grouped input channels overflow".to_string())
    })?;
    if in_c != expected_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_in_c],
            got: vec![in_c],
        });
    }
    if !out_c.is_multiple_of(groups) {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_ibp_forward: out_channels {out_c} not divisible by groups {groups}"
        )));
    }

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;
    if sh == 0 || sw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_ibp_forward: stride must be >= 1, got ({sh},{sw})"
        )));
    }
    if dh == 0 || dw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_ibp_forward: dilation must be >= 1, got ({dh},{dw})"
        )));
    }
    let eff_kh = kh
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dh))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d_ibp_forward: effective kernel height overflow".to_string())
        })?;
    let eff_kw = kw
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dw))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d_ibp_forward: effective kernel width overflow".to_string())
        })?;
    let padded_h = ph
        .checked_mul(2)
        .and_then(|pad| in_h.checked_add(pad))
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d_ibp_forward: padded height overflow".to_string())
        })?;
    let padded_w = pw
        .checked_mul(2)
        .and_then(|pad| in_w.checked_add(pad))
        .ok_or_else(|| {
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
    let out_size = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_ibp_forward: output dims overflow: {out_c} * {out_h} * {out_w}"
        ))
    })?;
    let spatial = out_h.checked_mul(out_w).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: output spatial size overflow".to_string())
    })?;
    Ok(Conv2dDeadlineGeometry {
        in_h,
        in_w,
        out_c,
        in_c_per_group,
        out_c_per_group: out_c / groups,
        kh,
        kw,
        out_h,
        out_w,
        spatial,
        out_size,
    })
}

/// Resolve one window tap `(oh, ow, ki, kj)` to its source input coordinate,
/// or `None` when the tap falls in the zero padding. This is the SINGLE
/// implementation of the grouped/dilated/padded/strided tap index math shared
/// by the plain f32 deadline contraction and the certified f64 kernel below.
#[inline]
#[allow(clippy::too_many_arguments)]
fn conv2d_tap_source(
    oh: usize,
    ow: usize,
    ki: usize,
    kj: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    in_h: usize,
    in_w: usize,
) -> Option<(usize, usize)> {
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;
    let ih = oh
        .checked_mul(sh)
        .and_then(|base| ki.checked_mul(dh)?.checked_add(base))
        .and_then(|padded| padded.checked_sub(ph))
        .filter(|&index| index < in_h)?;
    let iw = ow
        .checked_mul(sw)
        .and_then(|base| kj.checked_mul(dw)?.checked_add(base))
        .and_then(|padded| padded.checked_sub(pw))
        .filter(|&index| index < in_w)?;
    Some((ih, iw))
}

/// Finite-deadline grouped Conv2d interval forward.
///
/// The historical im2col path above can enter either an opaque caller engine or
/// one large faer GEMM. Neither surface has a cooperative deadline contract.
/// Deadline-scored graph work therefore uses this direct CPU contraction,
/// polling between bounded scalar-work quanta and before publishing the result.
/// The lower/upper sign decomposition is mathematically identical to
/// [`conv2d_ibp_forward_grouped`] and supports grouped/depthwise convolution.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_ibp_forward_grouped_with_deadline(
    input_lower: ArrayViewD<'_, f32>,
    input_upper: ArrayViewD<'_, f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    deadline: Instant,
) -> Result<ConvIbpForward> {
    let deadline_error = || {
        NyError::DeadlineExceeded(
            "Conv2d IBP forward: deadline exceeded during pollable CPU contraction".to_string(),
        )
    };
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }
    let geometry = conv2d_deadline_geometry(
        &input_lower,
        &input_upper,
        kernel,
        stride,
        padding,
        dilation,
        groups,
    )?;
    let Conv2dDeadlineGeometry {
        in_h,
        in_w,
        out_c,
        in_c_per_group,
        out_c_per_group,
        kh,
        kw,
        out_h,
        out_w,
        spatial,
        out_size,
    } = geometry;
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }
    let mut lower_flat = vec![0.0f32; out_size];
    let mut upper_flat = vec![0.0f32; out_size];
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }

    let mut operations = 0usize;
    for oc in 0..out_c {
        if Instant::now() >= deadline {
            return Err(deadline_error());
        }
        let group = oc / out_c_per_group;
        let ic_start = group * in_c_per_group;
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut lower_sum = 0.0f32;
                let mut upper_sum = 0.0f32;
                for ic_local in 0..in_c_per_group {
                    let ic = ic_start + ic_local;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            operations += 1;
                            if operations == DEADLINE_CPU_POLL_OPS {
                                if Instant::now() >= deadline {
                                    return Err(deadline_error());
                                }
                                operations = 0;
                            }
                            let Some((ih, iw)) = conv2d_tap_source(
                                oh, ow, ki, kj, stride, padding, dilation, in_h, in_w,
                            ) else {
                                continue;
                            };
                            let weight = kernel[[oc, ic_local, ki, kj]];
                            let input_lo = input_lower[[ic, ih, iw]];
                            let input_up = input_upper[[ic, ih, iw]];
                            if weight >= 0.0 {
                                lower_sum += input_lo * weight;
                                upper_sum += input_up * weight;
                            } else {
                                lower_sum += input_up * weight;
                                upper_sum += input_lo * weight;
                            }
                        }
                    }
                }
                let output_index = oc * spatial + oh * out_w + ow;
                lower_flat[output_index] = lower_sum;
                upper_flat[output_index] = upper_sum;
            }
        }
    }
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }
    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_h, out_w]), lower_flat)
        .map_err(|error| {
            NyError::InternalError(format!("conv2d_ibp_forward lower reshape: {error}"))
        })?;
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_h, out_w]), upper_flat)
        .map_err(|error| {
            NyError::InternalError(format!("conv2d_ibp_forward upper reshape: {error}"))
        })?;
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }
    Ok(ConvIbpForward {
        lower,
        upper,
        out_h,
        out_w,
    })
}

/// f64 unit roundoff `u64 = 2^-53` for the certified dual-accumulator kernel
/// (the historical f32 arm's `u = 2^-24` does not appear on this path).
const UNIT_ROUNDOFF_F64: f64 = 1.0 / ((1u64 << 53) as f64);

/// Multiplicative outward slack for the `err = γ·(acc_abs·abs_inflate)` value:
/// counting EVERY round-to-nearest f64 operation that could make the computed
/// `err` under-shoot its exact real-arithmetic value — the 3 multiplications
/// of the `err` expression, plus the subtract+divide inside `γ = n·u/(1−n·u)`
/// (`n·u` itself is exact: an integer < 2^53 scaled by a power of two), plus
/// the subtract+divide inside `abs_inflate = 1/(1−γ)` — gives ≤ 7 roundings,
/// so the under-shoot factor is at least `(1−u64)^7 > 1 − 8·u64`. Multiplying
/// by `1 + 8·u64 = 1 + 4·f64::EPSILON` restores `computed ≥ exact` up to an
/// `O(u64²)`-relative sliver from the slack multiplication's own rounding;
/// that sliver is absorbed many orders of magnitude over by the γ-index
/// margin (γ_{K+2} is charged where ≤ K+1 roundings occur, an absolute spare
/// of ≥ u64·A_true ≫ O(u64²)·err). A single closed-form factor was chosen over per-step
/// `next_up`-style stepping because it is auditable at a glance and the two
/// differ by well under one f64 ulp here (documented choice per
/// #cgan-conv-ibp-magnitude-floor).
const ERR_PRODUCT_SLACK_F64: f64 = 1.0 + 4.0 * f64::EPSILON;

/// Round an f64 LOWER endpoint outward (toward -inf) onto the f32 grid.
///
/// The f64→f32 `as` cast rounds to nearest (error ≤ ½ f32 ulp), and the f64
/// add/sub that produced `value` rounded to nearest (error ≤ u64·|value|,
/// which is < 2^-28 of an f32 ulp of the same magnitude for normals and
/// < 2^-149 for every f32-subnormal-range value). The single `next_down_f32`
/// step is a full f32 ulp, so it dominates both and makes the cast OUTWARD.
/// A cast that saturates to `+inf` means `value > f32::MAX`; the largest
/// finite f32 is then still a sound (smaller) lower endpoint. `-inf` and NaN
/// pass through (NaN is repaired conservatively by the caller).
#[inline]
fn certified_cast_lower_f32(value: f64) -> f32 {
    let cast = value as f32;
    if cast == f32::INFINITY {
        return f32::MAX;
    }
    next_down_f32(cast)
}

/// Round an f64 UPPER endpoint outward (toward +inf) onto the f32 grid.
/// Mirror of [`certified_cast_lower_f32`]; a cast saturating to `-inf` means
/// `value < -f32::MAX`, for which `-f32::MAX` is a sound (larger) upper
/// endpoint.
#[inline]
fn certified_cast_upper_f32(value: f64) -> f32 {
    let cast = value as f32;
    if cast == f32::NEG_INFINITY {
        return -f32::MAX;
    }
    next_up_f32(cast)
}

/// Finite-deadline CERTIFIED grouped Conv2d interval forward: f64 dual
/// accumulators with an exact per-output Higham widening
/// (#cgan-conv-ibp-magnitude-floor).
///
/// This kernel REPLACES the finite-deadline arm's previous 3-pass f32
/// structure (interval forward, then a second `|W|·max(|l|,|u|)` abs-conv
/// pass for `S`, then a `γ_{K+2}^{f32}·S_safe + 2u·|y|` widening). That arm's
/// error charge scales with activation MAGNITUDE (`S`), not box width, so the
/// widening was a floor that input splitting cannot erode. The audit contract
/// that motivated it ("no finite deadline may enter an opaque engine/faer
/// kernel without a cancellation contract", commit `6f49a660`) stays closed:
/// this kernel is scalar-CPU, engine-free, and polls at the shared
/// `DEADLINE_CPU_POLL_OPS` cadence.
///
/// # Accumulation
///
/// Per output `o`, in ONE loop over the `K = (in_c/groups)·kh·kw` window taps
/// (tap enumeration shared verbatim with the plain deadline forward via
/// [`conv2d_tap_source`] / [`conv2d_deadline_geometry`]):
///
/// ```text
/// acc_lo  += if w >= 0 { w·l_k } else { w·u_k }     (f64)
/// acc_hi  += if w >= 0 { w·u_k } else { w·l_k }     (f64)
/// acc_abs += |w| · max(|l_k|, |u_k|)                (f64, SAME loop)
/// ```
///
/// plus the bias term (`acc_lo/acc_hi += f64::from(b_o)`,
/// `acc_abs += |b_o|`). Every product of two f32 values is EXACT in f64
/// (24-bit significands ⇒ ≤ 48 significand bits < 53, and the exponent range
/// of any f32×f32 product, down to 2^-298, is far inside f64's normal range —
/// no underflow, no FTZ/DAZ exposure on this scalar CPU path). The ONLY
/// roundings are the f64 additions: ≤ K+1 per accumulator (K taps + bias).
///
/// # Error bound (as certified)
///
/// By the Higham summation bound (Accuracy and Stability of Numerical
/// Algorithms, Thm 3.1; its no-underflow hypothesis holds because every
/// partial sum of exact f32×f32 products stays in f64 normal range unless the
/// true sum itself is ~1e-308, which is 0 after any f32 cast anyway):
///
/// ```text
/// |acc_lo − Σ_exact| ≤ γ_{K+1}^{f64} · Σ_k |term_k| ≤ γ_{K+2}^{f64} · A_true
/// A_true = Σ_k |w_k|·max(|l_k|,|u_k|) + |b_o|  ≥ Σ_k |term_k|
/// γ_n^{f64} = n·u64 / (1 − n·u64),  u64 = 2^-53
/// ```
///
/// saturated to `+inf` when `n·u64 ≥ 1` (matching the f32 arm's convention;
/// the resulting non-finite `err` publishes `[-inf, +inf]`). `acc_abs` is
/// itself a round-to-nearest f64 sum of NON-NEGATIVE exact terms, so
/// `acc_abs ≥ A_true·(1 − γ_{K+2}^{f64})`, i.e.
/// `A_true ≤ acc_abs / (1 − γ_{K+2}^{f64})`. At f64 this inflation factor is
/// ~`1 + 1e-12` for realistic K, but it is written explicitly — never assumed
/// away:
///
/// ```text
/// err = γ_{K+2}^{f64} · (acc_abs · abs_inflate) · ERR_PRODUCT_SLACK_F64
/// abs_inflate = 1 / (1 − γ_{K+2}^{f64})        (+inf when γ ≥ 1)
/// ```
///
/// where [`ERR_PRODUCT_SLACK_F64`] covers the ≤ 3 f64 multiplications of the
/// `err` expression itself. Final endpoints:
///
/// ```text
/// lo_o = next_down_f32((acc_lo − err) as f32)
/// hi_o = next_up_f32  ((acc_hi + err) as f32)
/// ```
///
/// via [`certified_cast_lower_f32`] / [`certified_cast_upper_f32`], whose doc
/// comments justify that the one-ulp directed step dominates both the f64
/// add/sub rounding and the round-to-nearest f64→f32 cast, making the cast
/// OUTWARD. Errors here are widening-only by construction: every inequality
/// above over-covers.
///
/// NaN taps surface as either a NaN endpoint (f64 `max` discards NaN, so
/// `acc_abs`/`err` can stay finite while `acc_lo`/`acc_hi` go NaN — the NaN
/// endpoint is then repaired by the caller's
/// `BoundedTensor::new_repaired(.., RepairStrategy::Conservative)`, exactly as
/// on the historical arm) or, when `err` itself is non-finite, directly as
/// `[-inf, +inf]`. Both outcomes are conservative.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_ibp_forward_grouped_certified_f64_with_deadline(
    input_lower: ArrayViewD<'_, f32>,
    input_upper: ArrayViewD<'_, f32>,
    kernel: &ArrayD<f32>,
    bias: Option<&Array1<f32>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    deadline: Instant,
) -> Result<ConvIbpForward> {
    let deadline_error = || {
        NyError::DeadlineExceeded(
            "Conv2d certified IBP forward: deadline exceeded during pollable f64 contraction"
                .to_string(),
        )
    };
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }
    let geometry = conv2d_deadline_geometry(
        &input_lower,
        &input_upper,
        kernel,
        stride,
        padding,
        dilation,
        groups,
    )?;
    let Conv2dDeadlineGeometry {
        in_h,
        in_w,
        out_c,
        in_c_per_group,
        out_c_per_group,
        kh,
        kw,
        out_h,
        out_w,
        spatial,
        out_size,
    } = geometry;
    if let Some(b) = bias {
        if b.len() != out_c {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_c],
                got: vec![b.len()],
            });
        }
    }

    // γ_{K+2}^{f64} over K window taps + bias + one term of slack; saturate to
    // +inf when n·u64 ≥ 1 exactly as the f32 arm did for its γ.
    let macs = in_c_per_group.saturating_mul(kh).saturating_mul(kw);
    let n_terms = macs.saturating_add(2) as f64;
    let gamma = if n_terms * UNIT_ROUNDOFF_F64 < 1.0 {
        (n_terms * UNIT_ROUNDOFF_F64) / (1.0 - n_terms * UNIT_ROUNDOFF_F64)
    } else {
        f64::INFINITY
    };
    // acc_abs's own accumulation deficit: true abs-sum ≤ acc_abs/(1−γ).
    let abs_inflate = if gamma < 1.0 {
        1.0 / (1.0 - gamma)
    } else {
        f64::INFINITY
    };

    if Instant::now() >= deadline {
        return Err(deadline_error());
    }
    let mut lower_flat = vec![0.0f32; out_size];
    let mut upper_flat = vec![0.0f32; out_size];
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }

    let mut operations = 0usize;
    for oc in 0..out_c {
        if Instant::now() >= deadline {
            return Err(deadline_error());
        }
        let group = oc / out_c_per_group;
        let ic_start = group * in_c_per_group;
        let bias_o = bias.map(|b| f64::from(b[oc]));
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc_lo = 0.0f64;
                let mut acc_hi = 0.0f64;
                let mut acc_abs = 0.0f64;
                for ic_local in 0..in_c_per_group {
                    let ic = ic_start + ic_local;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            operations += 1;
                            if operations == DEADLINE_CPU_POLL_OPS {
                                if Instant::now() >= deadline {
                                    return Err(deadline_error());
                                }
                                operations = 0;
                            }
                            let Some((ih, iw)) = conv2d_tap_source(
                                oh, ow, ki, kj, stride, padding, dilation, in_h, in_w,
                            ) else {
                                continue;
                            };
                            let w = f64::from(kernel[[oc, ic_local, ki, kj]]);
                            let l = f64::from(input_lower[[ic, ih, iw]]);
                            let u = f64::from(input_upper[[ic, ih, iw]]);
                            // f64 products of f32 values are EXACT; only the
                            // += additions round (covered by γ_{K+2}^{f64}).
                            if w >= 0.0 {
                                acc_lo += w * l;
                                acc_hi += w * u;
                            } else {
                                acc_lo += w * u;
                                acc_hi += w * l;
                            }
                            acc_abs += w.abs() * l.abs().max(u.abs());
                        }
                    }
                }
                if let Some(b) = bias_o {
                    acc_lo += b;
                    acc_hi += b;
                    acc_abs += b.abs();
                }
                let err = gamma * (acc_abs * abs_inflate) * ERR_PRODUCT_SLACK_F64;
                let output_index = oc * spatial + oh * out_w + ow;
                if err.is_finite() {
                    lower_flat[output_index] = certified_cast_lower_f32(acc_lo - err);
                    upper_flat[output_index] = certified_cast_upper_f32(acc_hi + err);
                } else {
                    // γ saturation, NaN taps, or infinite abs-sum: publish the
                    // universal interval (widening-only; never a tight lie).
                    lower_flat[output_index] = f32::NEG_INFINITY;
                    upper_flat[output_index] = f32::INFINITY;
                }
            }
        }
    }
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }
    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_h, out_w]), lower_flat)
        .map_err(|error| {
            NyError::InternalError(format!(
                "conv2d_ibp_forward certified lower reshape: {error}"
            ))
        })?;
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_h, out_w]), upper_flat)
        .map_err(|error| {
            NyError::InternalError(format!(
                "conv2d_ibp_forward certified upper reshape: {error}"
            ))
        })?;
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }
    Ok(ConvIbpForward {
        lower,
        upper,
        out_h,
        out_w,
    })
}
