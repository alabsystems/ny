// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::ArrayD;
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use std::time::Instant;

/// Decode binary32 directly from its bits, so a subnormal source never passes
/// through a conversion instruction whose DAZ mode could erase it.
#[inline]
fn f32_to_f64_exact_bits(value: f32) -> f64 {
    const FRACTION_BITS: u32 = 52;
    const EXPONENT_BIAS: i32 = 1023;

    let bits = value.to_bits();
    let sign = u64::from(bits >> 31) << 63;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;
    match (exponent, fraction) {
        (0, 0) => f64::from_bits(sign),
        (0, _) => {
            let leading = fraction.ilog2();
            let unbiased_exponent = leading as i32 - 149;
            let exponent64 = (unbiased_exponent + EXPONENT_BIAS) as u64;
            let leading_bit = 1_u32 << leading;
            let fraction64 = u64::from(fraction - leading_bit) << (FRACTION_BITS - leading);
            f64::from_bits(sign | (exponent64 << FRACTION_BITS) | fraction64)
        }
        (0xff, 0) => f64::from_bits(sign | (0x7ff_u64 << FRACTION_BITS)),
        (0xff, _) => f64::NAN,
        _ => {
            let unbiased_exponent = exponent as i32 - 127;
            let exponent64 = (unbiased_exponent + EXPONENT_BIAS) as u64;
            let fraction64 = u64::from(fraction) << (FRACTION_BITS - 23);
            f64::from_bits(sign | (exponent64 << FRACTION_BITS) | fraction64)
        }
    }
}

/// Perform 2D convolution on a single (channels, height, width) input with groups support.
///
/// Kernel shape: `(out_c, in_c/groups, kh, kw)`.
/// With groups > 1, input channels and output channels are partitioned into
/// `groups` independent groups, each processed separately.
/// Reference: PyTorch `torch.nn.functional.conv2d`.
pub(crate) fn conv2d_single(
    input: &ArrayD<f32>,  // (in_channels, height, width)
    kernel: &ArrayD<f32>, // (out_channels, in_channels/groups, kh, kw)
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<ArrayD<f32>> {
    conv2d_single_grouped(input, kernel, stride, padding, dilation, 1)
}

/// Grouped 2D convolution on a single (channels, height, width) input.
pub(crate) fn conv2d_single_grouped(
    input: &ArrayD<f32>,  // (in_channels, height, width)
    kernel: &ArrayD<f32>, // (out_channels, in_channels/groups, kh, kw)
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> Result<ArrayD<f32>> {
    // Guard: ndim checks prevent panic on shape indexing (#2920 WP-B).
    if input.ndim() != 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![input.ndim()],
        });
    }
    if kernel.ndim() != 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }

    let in_c = input.shape()[0];
    let in_h = input.shape()[1];
    let in_w = input.shape()[2];

    let out_c = kernel.shape()[0];
    let ker_in_c_per_group = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    if groups == 0 {
        return Err(NyError::InvalidSpec(
            "conv2d_single: groups must be >= 1".to_string(),
        ));
    }
    if out_c == 0 || ker_in_c_per_group == 0 || kh == 0 || kw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_single: kernel dimensions must be nonzero, got {:?}",
            kernel.shape()
        )));
    }
    // Validate: in_c == ker_in_c_per_group * groups
    let expected_in_c = ker_in_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_single: grouped input channels overflow".to_string())
    })?;
    if in_c != expected_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_in_c],
            got: vec![in_c],
        });
    }
    if !out_c.is_multiple_of(groups) {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_single: out_channels {out_c} not divisible by groups {groups}"
        )));
    }

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    if sh == 0 || sw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_single: stride must be >= 1, got ({sh},{sw})"
        )));
    }
    if dh == 0 || dw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_single: dilation must be >= 1, got ({dh},{dw})"
        )));
    }

    // Effective (dilated) kernel span: dilation*(kernel-1) + 1.
    let eff_kh = kh
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dh))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d_single: effective kernel height overflow".to_string())
        })?;
    let eff_kw = kw
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dw))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d_single: effective kernel width overflow".to_string())
        })?;

    // Checked arithmetic: (in_h + 2*ph - eff_kh) / sh + 1
    // Guard against underflow when eff_kh > in_h + 2*ph, and div-by-zero when sh=0.
    let padded_h = ph
        .checked_mul(2)
        .and_then(|padding| in_h.checked_add(padding))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "conv2d_single: padded height overflow: in_h={in_h}, ph={ph}"
            ))
        })?;
    let padded_w = pw
        .checked_mul(2)
        .and_then(|padding| in_w.checked_add(padding))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "conv2d_single: padded width overflow: in_w={in_w}, pw={pw}"
            ))
        })?;
    if padded_h < eff_kh || padded_w < eff_kw {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_single: effective kernel ({eff_kh},{eff_kw}) larger than padded input \
             ({padded_h},{padded_w}): input=({in_h},{in_w}), padding=({ph},{pw}), \
             dilation=({dh},{dw})"
        )));
    }
    let out_h = (padded_h - eff_kh) / sh + 1;
    let out_w = (padded_w - eff_kw) / sw + 1;

    let out_c_per_group = out_c / groups;
    let mut output = ArrayD::zeros(ndarray::IxDyn(&[out_c, out_h, out_w]));

    for g in 0..groups {
        let ic_start = g * ker_in_c_per_group;
        let oc_start = g * out_c_per_group;
        for oc_local in 0..out_c_per_group {
            let oc = oc_start + oc_local;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for ic_local in 0..ker_in_c_per_group {
                        let ic = ic_start + ic_local;
                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let ih = oh
                                    .checked_mul(sh)
                                    .and_then(|base| kh_idx.checked_mul(dh)?.checked_add(base))
                                    .and_then(|padded| padded.checked_sub(ph))
                                    .filter(|&index| index < in_h);
                                let iw = ow
                                    .checked_mul(sw)
                                    .and_then(|base| kw_idx.checked_mul(dw)?.checked_add(base))
                                    .and_then(|padded| padded.checked_sub(pw))
                                    .filter(|&index| index < in_w);

                                if let (Some(ih), Some(iw)) = (ih, iw) {
                                    sum += input[[ic, ih, iw]]
                                        * kernel[[oc, ic_local, kh_idx, kw_idx]];
                                }
                            }
                        }
                    }
                    output[[oc, oh, ow]] = sum;
                }
            }
        }
    }

    Ok(output)
}

/// Perform 2D transposed convolution (deconvolution) for CROWN backward pass.
///
/// Input shape: (out_channels, out_h, out_w) - the gradient w.r.t. conv output
/// Kernel shape: (out_channels, in_channels/groups, kh, kw) - same as forward conv
/// Output shape: (in_channels, in_h, in_w) - the gradient w.r.t. conv input
///
/// This implements: conv_transpose2d(grad, weight) which is the backward pass through conv.
/// output_size specifies the expected output spatial dimensions to handle (W-F+2P)%S != 0.
#[cfg(test)]
pub(crate) fn conv2d_transpose(
    input: &ArrayD<f32>,  // (out_channels, out_h, out_w) - gradient from above
    kernel: &ArrayD<f32>, // (out_channels, in_channels/groups, kh, kw)
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_size: (usize, usize), // (in_h, in_w) - the expected input size
) -> Result<ArrayD<f32>> {
    conv2d_transpose_grouped(input, kernel, stride, padding, dilation, output_size, 1)
}

struct Conv2dTransposeGeometry {
    grad_h: usize,
    grad_w: usize,
    in_c_per_group: usize,
    kh: usize,
    kw: usize,
    out_c_per_group: usize,
    total_in_c: usize,
    output_spatial: usize,
    output_len: usize,
}

fn validate_conv2d_transpose_geometry(
    input: &ArrayD<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    dilation: (usize, usize),
    output_size: (usize, usize),
    groups: usize,
    operation: &str,
) -> Result<Conv2dTransposeGeometry> {
    if input.ndim() != 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![input.ndim()],
        });
    }
    if kernel.ndim() != 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }
    if groups == 0 {
        return Err(NyError::InvalidSpec(format!(
            "{operation}: groups must be >= 1"
        )));
    }
    if stride.0 == 0 || stride.1 == 0 {
        return Err(NyError::InvalidSpec(format!(
            "{operation}: stride must be >= 1, got {stride:?}"
        )));
    }
    if dilation.0 == 0 || dilation.1 == 0 {
        return Err(NyError::InvalidSpec(format!(
            "{operation}: dilation must be >= 1, got {dilation:?}"
        )));
    }

    let out_c = input.shape()[0];
    let grad_h = input.shape()[1];
    let grad_w = input.shape()[2];
    let kernel_out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    if kernel_out_c == 0 || in_c_per_group == 0 || kh == 0 || kw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "{operation}: kernel dimensions must be nonzero, got {:?}",
            kernel.shape()
        )));
    }
    if out_c != kernel_out_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![kernel_out_c],
            got: vec![out_c],
        });
    }
    if !out_c.is_multiple_of(groups) {
        return Err(NyError::InvalidSpec(format!(
            "{operation}: groups={groups} does not divide out_c={out_c}"
        )));
    }
    let total_in_c = in_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec(format!("{operation}: grouped input channels overflow"))
    })?;
    let output_spatial = output_size
        .0
        .checked_mul(output_size.1)
        .ok_or_else(|| NyError::InvalidSpec(format!("{operation}: output spatial overflow")))?;
    let output_len = total_in_c
        .checked_mul(output_spatial)
        .ok_or_else(|| NyError::InvalidSpec(format!("{operation}: output size overflow")))?;
    Ok(Conv2dTransposeGeometry {
        grad_h,
        grad_w,
        in_c_per_group,
        kh,
        kw,
        out_c_per_group: out_c / groups,
        total_in_c,
        output_spatial,
        output_len,
    })
}

/// Grouped 2D transposed convolution for CROWN backward pass.
pub(crate) fn conv2d_transpose_grouped(
    input: &ArrayD<f32>,  // (out_channels, out_h, out_w) - gradient from above
    kernel: &ArrayD<f32>, // (out_channels, in_channels/groups, kh, kw)
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_size: (usize, usize), // (in_h, in_w) - the expected input size
    groups: usize,
) -> Result<ArrayD<f32>> {
    // Thin allocating wrapper over `conv2d_transpose_grouped_into`: compute the
    // output shape, allocate the `(total_in_c, in_h, in_w)` tensor once, and
    // scatter directly into it. Kept so callers/tests wanting an owned `ArrayD`
    // are unchanged. The two ndim guards are replicated here (in the original
    // order) only to derive the shape / preserve exact error ordering before the
    // allocation; `_into` re-validates fully.
    let geometry = validate_conv2d_transpose_geometry(
        input,
        kernel,
        stride,
        dilation,
        output_size,
        groups,
        "conv2d_transpose",
    )?;
    let (in_h, in_w) = output_size;
    let mut output = ArrayD::<f32>::zeros(ndarray::IxDyn(&[geometry.total_in_c, in_h, in_w]));
    let dst = output
        .as_slice_mut()
        .expect("freshly allocated ArrayD is contiguous row-major");
    conv2d_transpose_grouped_into(
        dst,
        input,
        kernel,
        stride,
        padding,
        dilation,
        output_size,
        groups,
    )?;
    Ok(output)
}

/// Slice-output variant of [`conv2d_transpose_grouped`]: scatters the grouped
/// transposed-convolution result directly into `dst` — a `total_in_c*in_h*in_w`
/// row-major buffer over `(ic, ih, iw)` — instead of allocating a fresh `ArrayD`.
///
/// This eliminates the dominant per-position heap-allocation churn on the hot
/// patches-backward path (the caller passes its own already-owned output chunk).
/// `dst` is zeroed first because the scatter is `+=`, so the result is
/// byte-identical to the owned-`ArrayD` form: same operands, same fixed loop
/// order, same per-element accumulation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_grouped_into(
    dst: &mut [f32],
    input: &ArrayD<f32>,  // (out_channels, out_h, out_w) - gradient from above
    kernel: &ArrayD<f32>, // (out_channels, in_channels/groups, kh, kw)
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_size: (usize, usize), // (in_h, in_w) - the expected input size
    groups: usize,
) -> Result<()> {
    conv2d_transpose_grouped_into_with_deadline(
        dst,
        input,
        kernel,
        stride,
        padding,
        dilation,
        output_size,
        groups,
        None,
    )
}

/// f64 twin of [`conv2d_transpose_grouped_into`] for the patches compose's
/// A-POSTERIORI certified-error channel (#patches-f64-err).
///
/// Identical scatter, but every product and accumulation happens in f64 while
/// the operands stay the caller's f32 (f32->f64 widening is exact, so the only
/// rounding is the f64 accumulation, bounded by `gamma_n_f64`).
///
/// The compose's VALUE path is untouched and still f32 — this exists only so the
/// caller can measure `|f32_result - f64_result|` and charge that measured gap
/// instead of the a-priori `gamma_n_f32 * S` worst case, which measured up to
/// 26 million times wider than the IBP box it was supposed to tighten.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_grouped_into_f64(
    dst: &mut [f64],
    input: &ArrayD<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_size: (usize, usize),
    groups: usize,
) -> Result<()> {
    let geometry = validate_conv2d_transpose_geometry(
        input,
        kernel,
        stride,
        dilation,
        output_size,
        groups,
        "conv2d_transpose_grouped_into_f64",
    )?;
    let grad_h = geometry.grad_h;
    let grad_w = geometry.grad_w;
    let in_c_per_group = geometry.in_c_per_group;
    let kh = geometry.kh;
    let kw = geometry.kw;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;
    let (in_h, in_w) = output_size;
    let out_c_per_group = geometry.out_c_per_group;
    let hw = geometry.output_spatial;
    if dst.len() != geometry.output_len {
        return Err(NyError::ShapeMismatch {
            expected: vec![geometry.output_len],
            got: vec![dst.len()],
        });
    }
    dst.fill(0.0);
    for g in 0..groups {
        for oc_local in 0..out_c_per_group {
            let oc = g * out_c_per_group + oc_local;
            for grad_y in 0..grad_h {
                for grad_x in 0..grad_w {
                    let grad_val = f32_to_f64_exact_bits(input[[oc, grad_y, grad_x]]);
                    if grad_val == 0.0 {
                        continue;
                    }
                    for ic_local in 0..in_c_per_group {
                        let ic = g * in_c_per_group + ic_local;
                        for kh_idx in 0..kh {
                            let ih = grad_y
                                .checked_mul(sh)
                                .and_then(|base| kh_idx.checked_mul(dh)?.checked_add(base))
                                .and_then(|padded| padded.checked_sub(ph))
                                .filter(|&index| index < in_h);
                            let Some(ih) = ih else {
                                continue;
                            };
                            for kw_idx in 0..kw {
                                let iw = grad_x
                                    .checked_mul(sw)
                                    .and_then(|base| kw_idx.checked_mul(dw)?.checked_add(base))
                                    .and_then(|padded| padded.checked_sub(pw))
                                    .filter(|&index| index < in_w);
                                let Some(iw) = iw else {
                                    continue;
                                };
                                dst[ic * hw + ih * in_w + iw] += grad_val
                                    * f32_to_f64_exact_bits(kernel[[oc, ic_local, kh_idx, kw_idx]]);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Deadline-aware CPU scatter used by patches-native CROWN.
///
/// A finite deadline deliberately stays on this scalar CPU path instead of
/// entering an opaque caller GEMM. The innermost contraction polls after a
/// bounded number of candidate multiply-adds, and a partially filled `dst` is
/// never published by the patches caller when expiry is reported.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_grouped_into_with_deadline(
    dst: &mut [f32],
    input: &ArrayD<f32>,  // (out_channels, out_h, out_w) - gradient from above
    kernel: &ArrayD<f32>, // (out_channels, in_channels/groups, kh, kw)
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_size: (usize, usize), // (in_h, in_w) - the expected input size
    groups: usize,
    deadline: Option<Instant>,
) -> Result<()> {
    const DEADLINE_POLL_OPS: usize = 4_096;
    let deadline_exceeded = || {
        NyError::DeadlineExceeded(
            "Conv2d Patches backward: deadline exceeded during bounded CPU composition".to_string(),
        )
    };
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(deadline_exceeded());
    }

    let geometry = validate_conv2d_transpose_geometry(
        input,
        kernel,
        stride,
        dilation,
        output_size,
        groups,
        "conv2d_transpose",
    )?;
    let grad_h = geometry.grad_h;
    let grad_w = geometry.grad_w;
    let in_c_per_group = geometry.in_c_per_group;
    let kh = geometry.kh;
    let kw = geometry.kw;

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;
    let (in_h, in_w) = output_size;

    let out_c_per_group = geometry.out_c_per_group;
    let hw = geometry.output_spatial;
    if dst.len() != geometry.output_len {
        return Err(NyError::ShapeMismatch {
            expected: vec![geometry.output_len],
            got: vec![dst.len()],
        });
    }
    // The scatter accumulates (`+=`), so the destination must start at zero. This
    // is what makes reusing a caller-owned buffer byte-identical to a fresh alloc.
    dst.fill(0.0);

    // Transposed convolution with groups: scatter gradient to input positions.
    // Each group g scatters from out_c range [g*oc_per_group, (g+1)*oc_per_group)
    // to input channel range [g*ic_per_group, (g+1)*ic_per_group).
    let mut ops_since_poll = 0usize;
    for g in 0..groups {
        let oc_start = g * out_c_per_group;
        let ic_start = g * in_c_per_group;
        for oc_local in 0..out_c_per_group {
            let oc = oc_start + oc_local;
            for grad_y in 0..grad_h {
                for grad_x in 0..grad_w {
                    let grad_val = input[[oc, grad_y, grad_x]];
                    if grad_val == 0.0 {
                        continue;
                    }
                    for ic_local in 0..in_c_per_group {
                        let ic = ic_start + ic_local;
                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                if deadline.is_some() {
                                    ops_since_poll += 1;
                                    if ops_since_poll >= DEADLINE_POLL_OPS {
                                        if deadline.is_some_and(|limit| Instant::now() >= limit) {
                                            return Err(deadline_exceeded());
                                        }
                                        ops_since_poll = 0;
                                    }
                                }
                                let ih = grad_y
                                    .checked_mul(sh)
                                    .and_then(|base| kh_idx.checked_mul(dh)?.checked_add(base))
                                    .and_then(|padded| padded.checked_sub(ph))
                                    .filter(|&index| index < in_h);
                                let iw = grad_x
                                    .checked_mul(sw)
                                    .and_then(|base| kw_idx.checked_mul(dw)?.checked_add(base))
                                    .and_then(|padded| padded.checked_sub(pw))
                                    .filter(|&index| index < in_w);

                                if let (Some(ih), Some(iw)) = (ih, iw) {
                                    // Flat (ic, ih, iw) index into the row-major
                                    // (total_in_c, in_h, in_w) destination.
                                    dst[ic * hw + ih * in_w + iw] +=
                                        grad_val * kernel[[oc, ic_local, kh_idx, kw_idx]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(deadline_exceeded());
    }
    Ok(())
}

/// Materialize the grouped transposed-convolution as a dense operator matrix
/// `M` of shape `(K, N)` with `K = out_c·prev_kh·prev_kw` (a flattened input
/// patch) and `N = in_c·new_kh·new_kw` (a flattened output patch), such that for
/// any patch vector `x` (row-major over `(oc, gy, gx)`), `x · M` equals the
/// flattened result of [`conv2d_transpose_grouped`] on that patch
/// (`stride = (sh, sw)`, `padding = (0,0)`, `dilation = (1,1)`).
///
/// `M` is identical across every patch position, so building it ONCE and
/// batching all positions through a single GEMM replaces the per-position
/// scatter loop in the patches-mode Conv2d CROWN backward with one matrix
/// multiply — the step that can then run on the GPU. Group structure is encoded
/// directly (off-group entries stay zero), so the same GEMM handles `groups > 1`.
pub(crate) fn conv2d_transpose_operator_matrix(
    kernel: &ArrayD<f32>,
    sh: usize,
    sw: usize,
    prev_kh: usize,
    prev_kw: usize,
    new_kh: usize,
    new_kw: usize,
    in_c: usize,
    groups: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    if kernel.ndim() != 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }
    let out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    if sh == 0
        || sw == 0
        || prev_kh == 0
        || prev_kw == 0
        || new_kh == 0
        || new_kw == 0
        || in_c == 0
        || groups == 0
        || out_c == 0
        || in_c_per_group == 0
        || kh == 0
        || kw == 0
    {
        return Err(NyError::InvalidSpec(
            "conv2d_transpose operator dimensions, stride, channels, and groups must be nonzero"
                .into(),
        ));
    }
    let total_in_c = in_c_per_group.checked_mul(groups).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose operator grouped channels overflow".into())
    })?;
    if !out_c.is_multiple_of(groups) || total_in_c != in_c {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_transpose operator: incompatible groups={groups}, out_c={out_c}, \
             in_c_per_group={in_c_per_group}, in_c={in_c}"
        )));
    }
    let expected_new_kh = prev_kh
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(sh))
        .and_then(|extent| extent.checked_add(kh))
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d_transpose operator height geometry overflow".into())
        })?;
    let expected_new_kw = prev_kw
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(sw))
        .and_then(|extent| extent.checked_add(kw))
        .ok_or_else(|| {
            NyError::InvalidSpec("conv2d_transpose operator width geometry overflow".into())
        })?;
    if (new_kh, new_kw) != (expected_new_kh, expected_new_kw) {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_new_kh, expected_new_kw],
            got: vec![new_kh, new_kw],
        });
    }
    let out_c_per_group = out_c / groups;
    let k_dim = checked_shape_product(&[out_c, prev_kh, prev_kw]).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose operator input size overflow".into())
    })?;
    let n_dim = checked_shape_product(&[in_c, new_kh, new_kw]).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose operator output size overflow".into())
    })?;
    let matrix_len = k_dim.checked_mul(n_dim).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose operator matrix size overflow".into())
    })?;
    let mut m = vec![0.0f32; matrix_len];
    for g in 0..groups {
        for oc_local in 0..out_c_per_group {
            let oc = g * out_c_per_group + oc_local;
            for gy in 0..prev_kh {
                for gx in 0..prev_kw {
                    let k_idx = (oc * prev_kh + gy) * prev_kw + gx;
                    for ic_local in 0..in_c_per_group {
                        let ic = g * in_c_per_group + ic_local;
                        for kh_idx in 0..kh {
                            let ih = gy
                                .checked_mul(sh)
                                .and_then(|base| base.checked_add(kh_idx))
                                .ok_or_else(|| {
                                    NyError::InvalidSpec(
                                        "conv2d_transpose operator row coordinate overflow".into(),
                                    )
                                })?;
                            if ih >= new_kh {
                                continue;
                            }
                            for kw_idx in 0..kw {
                                let iw = gx
                                    .checked_mul(sw)
                                    .and_then(|base| base.checked_add(kw_idx))
                                    .ok_or_else(|| {
                                        NyError::InvalidSpec(
                                            "conv2d_transpose operator column coordinate overflow"
                                                .into(),
                                        )
                                    })?;
                                if iw >= new_kw {
                                    continue;
                                }
                                let n_idx = (ic * new_kh + ih) * new_kw + iw;
                                // For fixed (gy,gx), distinct kernel taps map to
                                // distinct (ih,iw), so assignment preserves even
                                // subnormal kernel bits without an FTZ-sensitive
                                // `0 + tap` operation.
                                m[k_idx * n_dim + n_idx] = kernel[[oc, ic_local, kh_idx, kw_idx]];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((m, k_dim, n_dim))
}

/// Batched grouped transposed convolution over many patch positions via a single
/// GEMM — the engine-routed (GPU-capable) equivalent of calling
/// [`conv2d_transpose_grouped`] once per position with `padding = (0,0)`,
/// `dilation = (1,1)`.
///
/// `patches` is row-major `(num_positions, out_c·prev_kh·prev_kw)`; the result is
/// row-major `(num_positions, in_c·new_kh·new_kw)`. Runs on whatever backend the
/// `engine` provides (GPU for a device engine). The result matches the
/// per-position scatter up to f32 GEMM reduction-order rounding, so the caller
/// must keep this on an opt-in path that preserves the conv-CROWN soundness
/// contract (the patches composition currently treats the scatter as exact).
/// Largest patches-compose GEMM (`num_positions·k_dim·n_dim` MACs) below which
/// the per-position CPU scatter wins (the GEMM is launch/transfer-bound). Same
/// crossover the linear/conv f64 seams use.
const PATCHES_COMPOSE_FAST_F32_MIN_MACS: usize = 1 << 24;

/// Patches-compose operator-matrix GEMM that routes LARGE products to the
/// process-global fast f32 accelerator (cuBLAS `Sgemm` over GB10 coherent
/// unified memory — no D2H readback, the cost that made the synchronous wgpu
/// patches seam a regression), else the passed engine, else `None` so the
/// caller keeps its per-position CPU scatter.
///
/// SOUND for the certified coefficient error: the patches-mode conv compose
/// carries a reduction-order-independent Higham term plus an absolute FTZ/DAZ
/// source/result-flushing charge, then propagates `‖k‖₁·old_err`. Thus both an
/// arbitrary GEMM summation order and flushed underflow are covered by the same
/// error channel the CPU scatter carries. The certificate is computed from the
/// incoming coefficients, never from this GEMM's output. Returns
/// `Some(Ok(..))` with the `num_positions × n_dim` row-major result on success.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_grouped_batched_fast(
    patches: &[f32],
    num_positions: usize,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    prev_spatial: (usize, usize),
    new_spatial: (usize, usize),
    in_c: usize,
    groups: usize,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Option<Result<Vec<f32>>> {
    let (prev_kh, prev_kw) = prev_spatial;
    let (new_kh, new_kw) = new_spatial;
    let (sh, sw) = stride;
    if kernel.ndim() != 4 {
        return Some(Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        }));
    }
    // MAC count from params alone (no operator-matrix build yet): the operator
    // matrix is `k_dim × n_dim`, the GEMM is `num_positions × k_dim × n_dim`.
    let out_c = kernel.shape()[0];
    let k_dim = match checked_shape_product(&[out_c, prev_kh, prev_kw]) {
        Some(value) => value,
        None => {
            return Some(Err(NyError::InvalidSpec(
                "Conv2d Patches batched input size overflow".into(),
            )));
        }
    };
    let n_dim = match checked_shape_product(&[in_c, new_kh, new_kw]) {
        Some(value) => value,
        None => {
            return Some(Err(NyError::InvalidSpec(
                "Conv2d Patches batched output size overflow".into(),
            )));
        }
    };
    if k_dim == 0 || n_dim == 0 || num_positions == 0 {
        return Some(Err(NyError::InvalidSpec(
            "Conv2d Patches batched geometry must be nonzero".into(),
        )));
    }
    let macs = match num_positions
        .checked_mul(k_dim)
        .and_then(|value| value.checked_mul(n_dim))
    {
        Some(value) => value,
        None => {
            return Some(Err(NyError::InvalidSpec(
                "Conv2d Patches batched MAC count overflow".into(),
            )));
        }
    };
    // Route large products to cuBLAS when the fast f32 accelerator is installed;
    // otherwise only bother building the operator matrix if a passed engine can
    // consume it. No accelerator worth using ⇒ None ⇒ caller does the CPU scatter.
    let global_fast_available = if deadline.is_some() {
        crate::fast_f32_gemm::is_preinitialized()
    } else {
        crate::fast_f32_gemm::is_installed()
    };
    let want_fast = macs >= PATCHES_COMPOSE_FAST_F32_MIN_MACS && global_fast_available;
    if !want_fast && engine.is_none() {
        return None;
    }
    let expected_patches = match num_positions.checked_mul(k_dim) {
        Some(value) => value,
        None => {
            return Some(Err(NyError::InvalidSpec(
                "Conv2d Patches batched input matrix size overflow".into(),
            )));
        }
    };
    if patches.len() != expected_patches {
        return Some(Err(NyError::ShapeMismatch {
            expected: vec![num_positions, k_dim],
            got: vec![patches.len()],
        }));
    }
    let (m, mk, mn) = match conv2d_transpose_operator_matrix(
        kernel, sh, sw, prev_kh, prev_kw, new_kh, new_kw, in_c, groups,
    ) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    debug_assert_eq!((mk, mn), (k_dim, n_dim));
    if want_fast {
        if let Some(res) = crate::fast_f32_gemm::with_engine_for_deadline(deadline, |e| {
            e.gemm_f32(num_positions, mk, mn, patches, &m)
        }) {
            return Some(res);
        }
    }
    engine.map(|eng| eng.gemm_f32(num_positions, mk, mn, patches, &m))
}

#[cfg(test)]
mod batched_transpose_tests {
    use super::*;
    use ndarray::IxDyn;
    use ny_core::NaiveCpuGemmEngine;

    #[test]
    fn operator_matrix_rejects_malformed_and_overflowing_geometry() {
        let rank_five = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1, 1, 1]));
        assert!(matches!(
            conv2d_transpose_operator_matrix(&rank_five, 1, 1, 1, 1, 1, 1, 1, 1),
            Err(NyError::ShapeMismatch { .. })
        ));

        let kernel = ArrayD::<f32>::ones(IxDyn(&[1, 1, 1, 1]));
        assert!(matches!(
            conv2d_transpose_operator_matrix(&kernel, 0, 1, 1, 1, 1, 1, 1, 1),
            Err(NyError::InvalidSpec(_))
        ));
        assert!(matches!(
            conv2d_transpose_operator_matrix(&kernel, 1, 1, 1, 1, 2, 1, 1, 1),
            Err(NyError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            conv2d_transpose_operator_matrix(&kernel, usize::MAX, 1, 2, 1, 1, 1, 1, 1,),
            Err(NyError::InvalidSpec(_))
        ));

        // Every individual extent is nonzero and the transposed-convolution
        // output formula is exact, but K*N cannot be represented.
        assert!(matches!(
            conv2d_transpose_operator_matrix(&kernel, 1, 1, usize::MAX, 1, usize::MAX, 1, 1, 1,),
            Err(NyError::InvalidSpec(_))
        ));

        let grouped_kernel = ArrayD::<f32>::ones(IxDyn(&[1, 2, 1, 1]));
        assert!(matches!(
            conv2d_transpose_operator_matrix(&grouped_kernel, 1, 1, 1, 1, 1, 1, 1, usize::MAX,),
            Err(NyError::InvalidSpec(_))
        ));
    }

    /// #patches-col2im throughput: operator-matrix GEMM vs GEMM+col2im at the
    /// cifar100 ResNet compose shape. RUN IN RELEASE — the dev profile leaves
    /// faer unoptimised and the comparison is meaningless there.
    ///
    /// MEASURED NEGATIVE RESULT — do not "optimise" the operator matrix away.
    ///
    /// The operator matrix `[k_dim x n_dim]` is structurally sparse: each k-row
    /// carries at most `in_c_per_group*kh*kw` nonzeros, so the GEMM performs
    /// `groups*new_kh*new_kw/(kh*kw)` times the essential MACs — 2.78x at this
    /// shape. Replacing it with the dense path's GEMM-then-col2im (which does
    /// only the essential MACs, and is index-identical — see
    /// `col2im_batched_matches_per_position_scatter`) makes the compose SLOWER:
    ///
    ///     opmat   1.06 ms   58 MMAC   ~109 GFLOP/s
    ///     col2im  3.00 ms   21 MMAC   ~ 14 GFLOP/s
    ///     MAC ratio 2.78x   TIME ratio 0.35x
    ///
    /// A big dense well-blocked GEMM sustains an order of magnitude more
    /// throughput than a col2im scatter, which is memory-bound. MAC count is the
    /// wrong cost model here: the "wasted" multiplies are free relative to the
    /// scatter traffic they avoid. The structural waste is real and irrelevant.
    #[test]
    fn col2im_vs_operator_matrix_throughput() {
        // Pin the CPU dense budget: NY_DENSE_BUDGET_MB is process-global and
        // sibling tests set it to 1 MiB, which makes the memory guards in these
        // paths refuse under a parallel run.
        crate::tests::with_crown_dense_budget_mb("2048", || {
            use std::time::Instant;

            // A representative cifar100 compose: 16->16 channels, 3x3 stride-1,
            // incoming patch 3x3 so the composed kernel is 5x5.
            let (out_c, in_c, kh, kw, prev_kh, prev_kw) =
                (16usize, 16usize, 3usize, 3usize, 3usize, 3usize);
            let (sh, sw, groups) = (1usize, 1usize, 1usize);
            let new_kh = (prev_kh - 1) * sh + kh;
            let new_kw = (prev_kw - 1) * sw + kw;
            let num_positions = 1024usize;
            let k_dim = out_c * prev_kh * prev_kw;
            let n_dim = in_c * new_kh * new_kw;

            let mut seed: u64 = 0xC0FF_EE11;
            let mut rnd = || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            };
            let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, in_c / groups, kh, kw]), |_| rnd());
            let patches: Vec<f32> = (0..num_positions * k_dim).map(|_| rnd()).collect();
            let a = ndarray::Array2::from_shape_vec((num_positions, k_dim), patches.clone())
                .expect("pmat");

            let engine = crate::faer_parallelism::FaerCpuGemmEngine;
            let reps = 5;

            // Operator matrix: [num_positions x k_dim] @ [k_dim x n_dim].
            let _ = conv2d_transpose_grouped_batched_fast(
                &patches,
                num_positions,
                &kernel,
                (sh, sw),
                (prev_kh, prev_kw),
                (new_kh, new_kw),
                in_c,
                groups,
                Some(&engine),
                None,
            );
            let t0 = Instant::now();
            for _ in 0..reps {
                let _ = conv2d_transpose_grouped_batched_fast(
                    &patches,
                    num_positions,
                    &kernel,
                    (sh, sw),
                    (prev_kh, prev_kw),
                    (new_kh, new_kw),
                    in_c,
                    groups,
                    Some(&engine),
                    None,
                )
                .expect("seam ran")
                .expect("opmat");
            }
            let opmat_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(reps);

            // col2im: GEMM over out_c_per_group then scatter.
            let _ = super::super::conv2d_transpose_batched_gemm_grouped_with_deadline(
                &a,
                &kernel,
                (sh, sw),
                (0, 0),
                (1, 1),
                (new_kh, new_kw),
                (prev_kh, prev_kw),
                out_c,
                groups,
                1,
                Some(&engine),
                None,
            );
            let t1 = Instant::now();
            for _ in 0..reps {
                let _ = super::super::conv2d_transpose_batched_gemm_grouped_with_deadline(
                    &a,
                    &kernel,
                    (sh, sw),
                    (0, 0),
                    (1, 1),
                    (new_kh, new_kw),
                    (prev_kh, prev_kw),
                    out_c,
                    groups,
                    1,
                    Some(&engine),
                    None,
                )
                .expect("col2im");
            }
            let col2im_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(reps);

            let opmat_macs = num_positions * k_dim * n_dim;
            let col2im_macs = num_positions * prev_kh * prev_kw * out_c * (in_c / groups) * kh * kw;
            eprintln!(
                "[col2im-vs-opmat] opmat {opmat_ms:.2} ms ({} MMAC)  col2im {col2im_ms:.2} ms ({} MMAC)  \
                 MAC ratio {:.2}x  time ratio {:.2}x",
                opmat_macs / 1_000_000,
                col2im_macs / 1_000_000,
                opmat_macs as f64 / col2im_macs as f64,
                opmat_ms / col2im_ms.max(1e-9),
            );
            assert!(opmat_ms > 0.0 && col2im_ms > 0.0);
        });
    }

    /// #patches-col2im: the DENSE path's GEMM-then-col2im formulation must
    /// reproduce the very same per-position scatter that the operator-matrix
    /// GEMM does, over the same configs.
    ///
    /// The operator matrix is `[k_dim x n_dim]` with `k_dim = out_c*prev_kh*prev_kw`
    /// and `n_dim = in_c*new_kh*new_kw`, but each of its k-rows carries at most
    /// `in_c_per_group*kh*kw` nonzeros — so the GEMM performs
    /// `groups*new_kh*new_kw/(kh*kw)` times the essential MACs (2.8x for a 3x3
    /// stride-1 conv, where prev_k=3 becomes new_k=5). `conv2d_transpose_batched_
    /// gemm_grouped_with_deadline` spreads the taps with a col2im scatter instead
    /// of with structural zeros in a matrix, doing only the essential MACs.
    ///
    /// This test is the correctness precondition for substituting it into the
    /// patches compose seam.
    #[test]
    fn col2im_batched_matches_per_position_scatter() {
        // (out_c, in_c_per_group, kh, kw, sh, sw, prev_kh, prev_kw, groups)
        let configs = [
            (
                4usize, 3usize, 3usize, 3usize, 1usize, 1usize, 2usize, 2usize, 1usize,
            ),
            (6, 2, 3, 3, 2, 2, 3, 2, 1),
            (4, 2, 2, 2, 1, 1, 3, 3, 2), // groups = 2
            (2, 1, 1, 1, 1, 1, 4, 4, 1),
            // The cifar100 ResNet shape: 3x3 stride-1, prev_k=3 -> new_k=5.
            (8, 8, 3, 3, 1, 1, 3, 3, 1),
        ];
        let mut seed: u64 = 0x5EED_0C21;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        for (ci, &(out_c, in_cpg, kh, kw, sh, sw, prev_kh, prev_kw, groups)) in
            configs.iter().enumerate()
        {
            let in_c = in_cpg * groups;
            let new_kh = (prev_kh - 1) * sh + kh;
            let new_kw = (prev_kw - 1) * sw + kw;
            let num_positions = 5usize;

            let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, in_cpg, kh, kw]), |_| rnd());
            let k_dim = out_c * prev_kh * prev_kw;
            let n_dim = in_c * new_kh * new_kw;
            let mut patches = vec![0.0f32; num_positions * k_dim];
            for v in patches.iter_mut() {
                *v = rnd();
            }

            // Reference: per-position CPU scatter, identical to the operator-matrix test.
            let mut reference = vec![0.0f32; num_positions * n_dim];
            for pos in 0..num_positions {
                let patch = ArrayD::from_shape_fn(IxDyn(&[out_c, prev_kh, prev_kw]), |ix| {
                    let (oc, gy, gx) = (ix[0], ix[1], ix[2]);
                    patches[pos * k_dim + (oc * prev_kh + gy) * prev_kw + gx]
                });
                let out = conv2d_transpose_grouped(
                    &patch,
                    &kernel,
                    (sh, sw),
                    (0, 0),
                    (1, 1),
                    (new_kh, new_kw),
                    groups,
                )
                .expect("scatter");
                for (j, v) in out.iter().enumerate() {
                    reference[pos * n_dim + j] = *v;
                }
            }

            // Candidate: the dense path's GEMM + col2im, fed the SAME row layout
            // the patches seam already builds (`pmat`, [num_positions x k_dim] in
            // (oc, ki, kj) order).
            let a = ndarray::Array2::from_shape_vec((num_positions, k_dim), patches.clone())
                .expect("pmat shape");
            let got = super::super::conv2d_transpose_batched_gemm_grouped_with_deadline(
                &a,
                &kernel,
                (sh, sw),
                (0, 0),
                (1, 1),
                (new_kh, new_kw),
                (prev_kh, prev_kw),
                out_c,
                groups,
                1,
                None,
                None,
            )
            .expect("col2im batched");

            assert_eq!(got.dim(), (num_positions, n_dim), "config {ci}: shape");
            for pos in 0..num_positions {
                for j in 0..n_dim {
                    let r = reference[pos * n_dim + j];
                    let b = got[[pos, j]];
                    let tol = 1e-4 * (1.0 + r.abs());
                    assert!(
                        (b - r).abs() <= tol,
                        "config {ci} pos {pos} idx {j}: col2im {b} vs scatter {r}"
                    );
                }
            }
        }
    }

    /// The batched operator-matrix GEMM must reproduce the per-position scatter
    /// of `conv2d_transpose_grouped` (up to f32 reduction-order rounding). This
    /// is the correctness contract for routing the patches-mode Conv2d CROWN
    /// backward through a single GPU-capable GEMM.
    #[test]
    fn batched_via_engine_matches_per_position_scatter() {
        let engine = NaiveCpuGemmEngine;
        // (out_c, in_c_per_group, kh, kw, sh, sw, prev_kh, prev_kw, groups)
        let configs = [
            (
                4usize, 3usize, 3usize, 3usize, 1usize, 1usize, 2usize, 2usize, 1usize,
            ),
            (6, 2, 3, 3, 2, 2, 3, 2, 1),
            (4, 2, 2, 2, 1, 1, 3, 3, 2), // groups = 2
            (2, 1, 1, 1, 1, 1, 4, 4, 1),
        ];
        let mut seed: u64 = 0xABCD_1234;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        for (ci, &(out_c, in_cpg, kh, kw, sh, sw, prev_kh, prev_kw, groups)) in
            configs.iter().enumerate()
        {
            let in_c = in_cpg * groups;
            let new_kh = (prev_kh - 1) * sh + kh;
            let new_kw = (prev_kw - 1) * sw + kw;
            let num_positions = 5usize;

            let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, in_cpg, kh, kw]), |_| rnd());
            let k_dim = out_c * prev_kh * prev_kw;
            let mut patches = vec![0.0f32; num_positions * k_dim];
            for v in patches.iter_mut() {
                *v = rnd();
            }

            // Reference: per-position CPU scatter.
            let n_dim = in_c * new_kh * new_kw;
            let mut reference = vec![0.0f32; num_positions * n_dim];
            for pos in 0..num_positions {
                let patch = ArrayD::from_shape_fn(IxDyn(&[out_c, prev_kh, prev_kw]), |ix| {
                    let (oc, gy, gx) = (ix[0], ix[1], ix[2]);
                    patches[pos * k_dim + (oc * prev_kh + gy) * prev_kw + gx]
                });
                let out = conv2d_transpose_grouped(
                    &patch,
                    &kernel,
                    (sh, sw),
                    (0, 0),
                    (1, 1),
                    (new_kh, new_kw),
                    groups,
                )
                .expect("scatter");
                for (j, v) in out.iter().enumerate() {
                    reference[pos * n_dim + j] = *v;
                }
            }

            let batched = conv2d_transpose_grouped_batched_fast(
                &patches,
                num_positions,
                &kernel,
                (sh, sw),
                (prev_kh, prev_kw),
                (new_kh, new_kw),
                in_c,
                groups,
                Some(&engine),
                None,
            )
            .expect("seam ran (passed engine present)")
            .expect("batched");
            assert_eq!(batched.len(), reference.len(), "config {ci}: length");
            for (idx, (b, r)) in batched.iter().zip(reference.iter()).enumerate() {
                let tol = 1e-4 * (1.0 + r.abs());
                assert!(
                    (b - r).abs() <= tol,
                    "config {ci} idx {idx}: batched {b} != scatter {r}"
                );
            }
        }
    }
}
