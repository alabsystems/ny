// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ConvTranspose2d forward operation (ONNX ConvTranspose).
//!
//! Separated from `ops.rs` to keep per-file line count under 500.

use faer::Mat;
use ndarray::ArrayD;
use ny_core::{checked_shape_product, NyError, Result};

use crate::faer_parallelism::mat_mul;

/// Perform 2D transposed convolution (forward op).
///
/// Input shape: (in_channels, in_h, in_w)
/// Kernel shape: (in_channels, out_channels, kh, kw) (ONNX ConvTranspose layout)
/// Output shape: (out_channels, out_h, out_w)
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_transpose_forward(
    input: &ArrayD<f32>,  // (in_channels, in_h, in_w)
    kernel: &ArrayD<f32>, // (in_channels, out_channels, kh, kw)
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_padding: (usize, usize),
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

    let ker_in_c = kernel.shape()[0];
    let out_c = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    if ker_in_c == 0 || out_c == 0 || kh == 0 || kw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_transpose_forward: kernel dimensions must be nonzero, got {:?}",
            kernel.shape()
        )));
    }
    if in_c != ker_in_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![ker_in_c],
            got: vec![in_c],
        });
    }

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;
    let (oph, opw) = output_padding;

    if sh == 0 || sw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_transpose_forward: stride must be >= 1, got ({sh},{sw})"
        )));
    }
    if dh == 0 || dw == 0 {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_transpose_forward: dilation must be >= 1, got ({dh},{dw})"
        )));
    }
    if oph >= sh || opw >= sw {
        return Err(NyError::UnsupportedConfiguration(format!(
            "conv2d_transpose_forward: output_padding ({oph},{opw}) must be < stride \
             ({sh},{sw}) per dimension"
        )));
    }

    // Effective (dilated) kernel span: dilation*(kernel-1) + 1.
    let eff_kh = kh
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dh))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "conv2d_transpose_forward: effective kernel height overflow".to_string(),
            )
        })?;
    let eff_kw = kw
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(dw))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "conv2d_transpose_forward: effective kernel width overflow".to_string(),
            )
        })?;

    // Checked arithmetic: (in_h - 1) * sh + eff_kh - 2 * ph + output_padding_h
    // Guard against underflow when in_h=0 or 2*ph > (in_h-1)*sh + eff_kh + oph.
    let expanded_h = in_h
        .checked_sub(1)
        .and_then(|v| v.checked_mul(sh))
        .and_then(|v| v.checked_add(eff_kh))
        .and_then(|v| v.checked_add(oph))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "conv2d_transpose_forward: height overflow: in_h={in_h}, sh={sh}, kh={kh}"
            ))
        })?;
    let expanded_w = in_w
        .checked_sub(1)
        .and_then(|v| v.checked_mul(sw))
        .and_then(|v| v.checked_add(eff_kw))
        .and_then(|v| v.checked_add(opw))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "conv2d_transpose_forward: width overflow: in_w={in_w}, sw={sw}, kw={kw}"
            ))
        })?;
    let double_ph = ph.checked_mul(2).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose_forward: padding height overflow".to_string())
    })?;
    let double_pw = pw.checked_mul(2).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose_forward: padding width overflow".to_string())
    })?;
    if expanded_h < double_ph || expanded_w < double_pw {
        return Err(NyError::InvalidSpec(format!(
            "conv2d_transpose_forward: output size underflow: \
             input=({in_h},{in_w}), stride=({sh},{sw}), kernel=({kh},{kw}), \
             dilation=({dh},{dw}), padding=({ph},{pw}), output_padding=({oph},{opw})"
        )));
    }
    let out_h = expanded_h - double_ph;
    let out_w = expanded_w - double_pw;

    // GEMM (im2col/col2im) reformulation of the transposed-conv forward map
    // (#cgan-convt-fwd-hotspot). The naive body was a 6-deep scatter loop
    //   for ic { for ih { for iw { for oc { for kh { for kw {...}}}}}}
    // O(in_c·in_h·in_w·out_c·kh·kw) in scalar Rust — the #1 self-time hotspot on
    // the cGAN/ConvTranspose forward path. The identical linear map is
    //   output[oc,oh,ow] = Σ_{ic,ki,kj : oh=ih·sh+ki·dh-ph, ow=iw·sw+kj·dw-pw}
    //                          input[ic,ih,iw] · kernel[ic,oc,ki,kj]
    // whose ic-contraction is a single dense GEMM:
    //   (1) input as A = inputᵀ, shape (in_h·in_w, in_c);
    //   (2) kernel as B, shape (in_c, out_c·kh·kw), col = oc·(kh·kw)+ki·kw+kj;
    //   (3) contrib = A·B, shape (in_h·in_w, out_c·kh·kw) (the per-tap products
    //       summed over ic — the GEMM's job);
    //   (4) col2im: scatter-add each contrib[ih·in_w+iw, oc·(kh·kw)+ki·kw+kj]
    //       into output[oc,oh,ow] with the SAME geometry + bounds guard as the
    //       naive loop.
    //
    // SOUNDNESS: this only reorders the f32 accumulation vs the scatter loop (the
    // ic-sum now runs inside the GEMM, before the tap-sum). The result differs
    // from the naive loop by at most a few ULP over the K = in_c·kh·kw fan-in.
    // The caller (`propagate_ibp_sound_with_engine`) folds an OUTWARD Higham
    // margin `up(γ_{K+2}·S + 2u·|y|)` that is summation-order independent and
    // covers exactly this K-term f32 error, so the sound node bound is unchanged.
    // No margin/soundness code is touched here — only how the (already-non-sound-
    // on-its-own) point forward map is computed.
    let out_spatial = out_h.checked_mul(out_w).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose_forward: output spatial overflow".to_string())
    })?;
    let output_len = out_c.checked_mul(out_spatial).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose_forward: output size overflow".to_string())
    })?;
    let mut output_flat = vec![0.0f32; output_len];

    let in_spatial = in_h.checked_mul(in_w).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose_forward: input spatial overflow".to_string())
    })?;
    let kernel_spatial = kh.checked_mul(kw).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose_forward: kernel spatial overflow".to_string())
    })?;
    let cols = out_c.checked_mul(kernel_spatial).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose_forward: kernel matrix width overflow".to_string())
    })?;
    checked_shape_product(&[in_spatial, in_c]).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose_forward: input matrix size overflow".to_string())
    })?;
    checked_shape_product(&[in_c, cols]).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_transpose_forward: kernel matrix size overflow".to_string())
    })?;
    checked_shape_product(&[in_spatial, cols]).ok_or_else(|| {
        NyError::InvalidSpec(
            "conv2d_transpose_forward: contribution matrix size overflow".to_string(),
        )
    })?;

    // A = inputᵀ : (in_h·in_w, in_c). A[ih·in_w+iw, ic] = input[ic,ih,iw].
    // Zero-sized dims (in_spatial==0 or in_c==0) yield an empty matrix and the
    // closure is never called, so the div/mod by in_w below is guarded.
    let a = Mat::<f32>::from_fn(in_spatial, in_c, |r, ic| {
        let ih = r / in_w;
        let iw = r % in_w;
        input[[ic, ih, iw]]
    });
    // B = kernel : (in_c, out_c·kh·kw). B[ic, oc·ksp + ki·kw + kj].
    // cols==0 (kernel_spatial==0 or out_c==0) yields no columns; the closure is
    // never called, guarding the div/mod by kernel_spatial/kw.
    let b = Mat::<f32>::from_fn(in_c, cols, |ic, col| {
        let oc = col / kernel_spatial;
        let rem = col % kernel_spatial;
        let ki = rem / kw;
        let kj = rem % kw;
        kernel[[ic, oc, ki, kj]]
    });

    // contrib = A·B : (in_h·in_w, out_c·kh·kw). Row-major flatten for a
    // cache-friendly col2im scatter (faer's Mat is column-major).
    let contrib = mat_mul(&a, &b);
    let contrib_len = in_spatial.checked_mul(cols).ok_or_else(|| {
        NyError::InvalidSpec(
            "conv2d_transpose_forward: contribution matrix size overflow".to_string(),
        )
    })?;
    let mut contrib_rm = Vec::with_capacity(contrib_len);
    for r in 0..in_spatial {
        for c in 0..cols {
            contrib_rm.push(contrib[(r, c)]);
        }
    }

    // col2im: identical geometry + bounds guard as the naive scatter, but with
    // the ic-contraction already folded into `contrib`.
    for ih in 0..in_h {
        for iw in 0..in_w {
            let row_base = (ih * in_w + iw) * cols;
            for oc in 0..out_c {
                let oc_col_base = oc * kernel_spatial;
                let oc_out_base = oc * out_spatial;
                for kh_idx in 0..kh {
                    let oh = ih
                        .checked_mul(sh)
                        .and_then(|base| kh_idx.checked_mul(dh)?.checked_add(base))
                        .and_then(|padded| padded.checked_sub(ph))
                        .filter(|&index| index < out_h);
                    let Some(oh) = oh else {
                        continue;
                    };
                    let oh_out_base = oc_out_base + oh * out_w;
                    let col_row_base = row_base + oc_col_base + kh_idx * kw;
                    for kw_idx in 0..kw {
                        let ow = iw
                            .checked_mul(sw)
                            .and_then(|base| kw_idx.checked_mul(dw)?.checked_add(base))
                            .and_then(|padded| padded.checked_sub(pw))
                            .filter(|&index| index < out_w);
                        if let Some(ow) = ow {
                            output_flat[oh_out_base + ow] += contrib_rm[col_row_base + kw_idx];
                        }
                    }
                }
            }
        }
    }

    ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_h, out_w]), output_flat)
        .map_err(|e| NyError::InternalError(format!("conv2d_transpose_forward reshape: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference naive scatter-loop implementation (the pre-GEMM body), retained
    /// as the differential oracle for [`conv2d_transpose_forward`]. Byte-for-byte
    /// the same guards + scatter geometry as the original function.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_transpose_forward_naive(
        input: &ArrayD<f32>,
        kernel: &ArrayD<f32>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        output_padding: (usize, usize),
    ) -> Result<ArrayD<f32>> {
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

        let ker_in_c = kernel.shape()[0];
        let out_c = kernel.shape()[1];
        let kh = kernel.shape()[2];
        let kw = kernel.shape()[3];

        if in_c != ker_in_c {
            return Err(NyError::ShapeMismatch {
                expected: vec![ker_in_c],
                got: vec![in_c],
            });
        }

        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (dh, dw) = dilation;
        let (oph, opw) = output_padding;

        if dh == 0 || dw == 0 {
            return Err(NyError::InvalidSpec(format!(
                "conv2d_transpose_forward: dilation must be >= 1, got ({dh},{dw})"
            )));
        }

        let eff_kh = dh * (kh - 1) + 1;
        let eff_kw = dw * (kw - 1) + 1;

        let expanded_h = in_h
            .checked_sub(1)
            .and_then(|v| v.checked_mul(sh))
            .and_then(|v| v.checked_add(eff_kh))
            .and_then(|v| v.checked_add(oph))
            .ok_or_else(|| NyError::InvalidSpec("height overflow".into()))?;
        let expanded_w = in_w
            .checked_sub(1)
            .and_then(|v| v.checked_mul(sw))
            .and_then(|v| v.checked_add(eff_kw))
            .and_then(|v| v.checked_add(opw))
            .ok_or_else(|| NyError::InvalidSpec("width overflow".into()))?;
        let double_ph = 2 * ph;
        let double_pw = 2 * pw;
        if expanded_h < double_ph || expanded_w < double_pw {
            return Err(NyError::InvalidSpec("output size underflow".into()));
        }
        let out_h = expanded_h - double_ph;
        let out_w = expanded_w - double_pw;

        let mut output = ArrayD::zeros(ndarray::IxDyn(&[out_c, out_h, out_w]));

        for ic in 0..in_c {
            for ih in 0..in_h {
                for iw in 0..in_w {
                    let input_val = input[[ic, ih, iw]];
                    if input_val == 0.0 {
                        continue;
                    }
                    for oc in 0..out_c {
                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let oh = (ih * sh + kh_idx * dh) as isize - ph as isize;
                                let ow = (iw * sw + kw_idx * dw) as isize - pw as isize;
                                if oh >= 0 && oh < out_h as isize && ow >= 0 && ow < out_w as isize
                                {
                                    output[[oc, oh as usize, ow as usize]] +=
                                        input_val * kernel[[ic, oc, kh_idx, kw_idx]];
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Deterministic xorshift64 fill in `[-scale, scale)`.
    fn det_fill(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut rng = seed | 1;
        (0..len)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let u = (rng as f32) / (u64::MAX as f32);
                (u * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// Differential test: the GEMM `conv2d_transpose_forward` must match the
    /// naive scatter loop to a tight relative tolerance across a range of shapes
    /// including stride>1, padding>0, dilation>1, and output_padding>0. The two
    /// differ only by f32 accumulation order (a few ULP over the K=in_c·kh·kw
    /// fan-in), which the caller's sound Higham margin covers.
    #[test]
    fn gemm_matches_naive() {
        // (in_c, in_h, in_w, out_c, kh, kw, sh, sw, ph, pw, dh, dw, oph, opw)
        let cases: &[(
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
        )] = &[
            // baseline stride/pad/dil/op = 1/0/1/0
            (3, 5, 5, 4, 3, 3, 1, 1, 0, 0, 1, 1, 0, 0),
            // stride 2 + output_padding 1
            (2, 4, 6, 3, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1),
            // stride 2 + padding 1
            (4, 6, 5, 2, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0),
            // dilation 2
            (2, 5, 5, 3, 2, 3, 1, 1, 0, 0, 2, 2, 0, 0),
            // stride 3 + padding 1 + dilation 2 + output_padding 2 (op < stride)
            (3, 4, 4, 3, 3, 2, 3, 3, 1, 1, 2, 2, 2, 2),
            // large padding, single channel
            (1, 7, 7, 1, 4, 4, 1, 1, 2, 2, 1, 1, 0, 0),
            // asymmetric strides/pads/dilations
            (2, 6, 4, 3, 3, 2, 2, 1, 1, 0, 1, 2, 1, 0),
        ];

        for (ci, &(in_c, in_h, in_w, out_c, kh, kw, sh, sw, ph, pw, dh, dw, oph, opw)) in
            cases.iter().enumerate()
        {
            let input = ArrayD::from_shape_vec(
                ndarray::IxDyn(&[in_c, in_h, in_w]),
                det_fill(in_c * in_h * in_w, 0x1234_5678 ^ ci as u64, 1.5),
            )
            .unwrap();
            let kernel = ArrayD::from_shape_vec(
                ndarray::IxDyn(&[in_c, out_c, kh, kw]),
                det_fill(in_c * out_c * kh * kw, 0x9E37_79B9 ^ ci as u64, 2.0),
            )
            .unwrap();

            let naive = conv2d_transpose_forward_naive(
                &input,
                &kernel,
                (sh, sw),
                (ph, pw),
                (dh, dw),
                (oph, opw),
            );
            let gemm =
                conv2d_transpose_forward(&input, &kernel, (sh, sw), (ph, pw), (dh, dw), (oph, opw));

            match (naive, gemm) {
                (Ok(n), Ok(g)) => {
                    assert_eq!(n.shape(), g.shape(), "case {ci}: shape mismatch");
                    for (idx, (&nv, &gv)) in n.iter().zip(g.iter()).enumerate() {
                        let diff = (nv - gv).abs();
                        let scale = nv.abs().max(gv.abs()).max(1.0);
                        assert!(
                            diff <= 1e-5 * scale,
                            "case {ci} idx {idx}: naive={nv} gemm={gv} diff={diff}"
                        );
                    }
                }
                (Err(_), Err(_)) => { /* both reject the config identically */ }
                (n, g) => panic!("case {ci}: Ok/Err disagreement: naive={n:?} gemm={g:?}"),
            }
        }
    }
}
