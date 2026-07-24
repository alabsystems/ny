// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::ArrayD;
use ny_core::{GemmEngine, NyError, Result};

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
    if input.ndim() < 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![input.ndim()],
        });
    }
    if kernel.ndim() < 4 {
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

    // Validate: in_c == ker_in_c_per_group * groups
    let expected_in_c = ker_in_c_per_group * groups;
    if in_c != expected_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_in_c],
            got: vec![in_c],
        });
    }

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    if dh == 0 || dw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_single: dilation must be >= 1, got ({dh},{dw})"
        )));
    }

    // Effective (dilated) kernel span: dilation*(kernel-1) + 1.
    let eff_kh = dh * (kh - 1) + 1;
    let eff_kw = dw * (kw - 1) + 1;

    // Checked arithmetic: (in_h + 2*ph - eff_kh) / sh + 1
    // Guard against underflow when eff_kh > in_h + 2*ph, and div-by-zero when sh=0.
    let padded_h = in_h.checked_add(2 * ph).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "conv2d_single: padded height overflow: in_h={in_h}, ph={ph}"
        ))
    })?;
    let padded_w = in_w.checked_add(2 * pw).ok_or_else(|| {
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
    if sh == 0 || sw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_single: stride must be >= 1, got ({sh},{sw})"
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
                                let ih = (oh * sh + kh_idx * dh) as isize - ph as isize;
                                let iw = (ow * sw + kw_idx * dw) as isize - pw as isize;

                                // SAFETY(as usize): ih/iw are isize, guard ensures >= 0 and < in_h/in_w.
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    sum += input[[ic, ih as usize, iw as usize]]
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
    if input.ndim() < 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![input.ndim()],
        });
    }
    if kernel.ndim() < 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }
    let in_c_per_group = kernel.shape()[1];
    let total_in_c = in_c_per_group * groups;
    let (in_h, in_w) = output_size;
    let mut output = ArrayD::<f32>::zeros(ndarray::IxDyn(&[total_in_c, in_h, in_w]));
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
    // Guard: ndim checks prevent panic on shape indexing (#2920 WP-B).
    if input.ndim() < 3 {
        return Err(NyError::ShapeMismatch {
            expected: vec![3],
            got: vec![input.ndim()],
        });
    }
    if kernel.ndim() < 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }

    let out_c = input.shape()[0]; // This is out_channels of the conv (in_channels for gradient)
    let grad_h = input.shape()[1];
    let grad_w = input.shape()[2];

    let ker_out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1]; // in_channels / groups
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    if out_c != ker_out_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![ker_out_c],
            got: vec![out_c],
        });
    }

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;
    let (in_h, in_w) = output_size;

    if dh == 0 || dw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_transpose: dilation must be >= 1, got ({dh},{dw})"
        )));
    }

    let total_in_c = in_c_per_group * groups;
    let out_c_per_group = out_c / groups;
    let hw = in_h * in_w;
    if dst.len() != total_in_c * hw {
        return Err(NyError::ShapeMismatch {
            expected: vec![total_in_c * hw],
            got: vec![dst.len()],
        });
    }
    // The scatter accumulates (`+=`), so the destination must start at zero. This
    // is what makes reusing a caller-owned buffer byte-identical to a fresh alloc.
    dst.fill(0.0);

    // Transposed convolution with groups: scatter gradient to input positions.
    // Each group g scatters from out_c range [g*oc_per_group, (g+1)*oc_per_group)
    // to input channel range [g*ic_per_group, (g+1)*ic_per_group).
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
                                let ih = (grad_y * sh + kh_idx * dh) as isize - ph as isize;
                                let iw = (grad_x * sw + kw_idx * dw) as isize - pw as isize;

                                // SAFETY(as usize): ih/iw are isize, guard ensures >= 0 and < in_h/in_w.
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    // Flat (ic, ih, iw) index into the row-major
                                    // (total_in_c, in_h, in_w) destination.
                                    dst[ic * hw + ih as usize * in_w + iw as usize] +=
                                        grad_val * kernel[[oc, ic_local, kh_idx, kw_idx]];
                                }
                            }
                        }
                    }
                }
            }
        }
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
fn conv2d_transpose_operator_matrix(
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
    if kernel.ndim() < 4 {
        return Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        });
    }
    let out_c = kernel.shape()[0];
    let in_c_per_group = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    if groups == 0 || !out_c.is_multiple_of(groups) || in_c_per_group * groups != in_c {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_transpose operator: incompatible groups={groups}, out_c={out_c}, \
             in_c_per_group={in_c_per_group}, in_c={in_c}"
        )));
    }
    let out_c_per_group = out_c / groups;
    let k_dim = out_c * prev_kh * prev_kw;
    let n_dim = in_c * new_kh * new_kw;
    let mut m = vec![0.0f32; k_dim * n_dim];
    for g in 0..groups {
        for oc_local in 0..out_c_per_group {
            let oc = g * out_c_per_group + oc_local;
            for gy in 0..prev_kh {
                for gx in 0..prev_kw {
                    let k_idx = (oc * prev_kh + gy) * prev_kw + gx;
                    for ic_local in 0..in_c_per_group {
                        let ic = g * in_c_per_group + ic_local;
                        for kh_idx in 0..kh {
                            let ih = gy * sh + kh_idx;
                            if ih >= new_kh {
                                continue;
                            }
                            for kw_idx in 0..kw {
                                let iw = gx * sw + kw_idx;
                                if iw >= new_kw {
                                    continue;
                                }
                                let n_idx = (ic * new_kh + ih) * new_kw + iw;
                                // Accumulate (kernel taps can alias the same (ih,iw)
                                // only across distinct (kh_idx,kw_idx), which cannot
                                // happen for fixed (gy,gx); still use += for safety).
                                m[k_idx * n_dim + n_idx] += kernel[[oc, ic_local, kh_idx, kw_idx]];
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
/// carries `coeff_err = γ_K·RowMaxAbs·‖k‖₁ + ‖k‖₁·old_err`, which over-bounds
/// `|compose − true|` for ANY summation order (Higham's bound is
/// reduction-order independent), so cuBLAS's accumulation order is covered by
/// the SAME err the CPU scatter carries — the err is computed from the incoming
/// coefficients, never from this GEMM's output. Returns `Some(Ok(..))` with the
/// `num_positions × n_dim` row-major result on success.
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
) -> Option<Result<Vec<f32>>> {
    let (prev_kh, prev_kw) = prev_spatial;
    let (new_kh, new_kw) = new_spatial;
    let (sh, sw) = stride;
    if kernel.ndim() < 4 {
        return Some(Err(NyError::ShapeMismatch {
            expected: vec![4],
            got: vec![kernel.ndim()],
        }));
    }
    // MAC count from params alone (no operator-matrix build yet): the operator
    // matrix is `k_dim × n_dim`, the GEMM is `num_positions × k_dim × n_dim`.
    let out_c = kernel.shape()[0];
    let k_dim = out_c * prev_kh * prev_kw;
    let n_dim = in_c * new_kh * new_kw;
    let macs = num_positions.saturating_mul(k_dim).saturating_mul(n_dim);
    // Route large products to cuBLAS when the fast f32 accelerator is installed;
    // otherwise only bother building the operator matrix if a passed engine can
    // consume it. No accelerator worth using ⇒ None ⇒ caller does the CPU scatter.
    let want_fast =
        macs >= PATCHES_COMPOSE_FAST_F32_MIN_MACS && crate::fast_f32_gemm::is_installed();
    if !want_fast && engine.is_none() {
        return None;
    }
    if patches.len() != num_positions * k_dim {
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
        if let Some(res) =
            crate::fast_f32_gemm::with_engine(|e| e.gemm_f32(num_positions, mk, mn, patches, &m))
        {
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
