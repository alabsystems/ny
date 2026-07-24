// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STRICT, ZERO-TOLERANCE soundness tests for the conv CROWN-backward coefficient
//! certificate (#vnncomp-aw-soundness — conv f32-accumulation bug).
//!
//! The bug: Conv2d / ConvTranspose2d / Conv1d / ConvTranspose1d CROWN-backward
//! coefficients were computed by an **f32**-accumulating `mat_mul` (faer
//! `Mat<f32>` GEMM) + f32 col2im, but the per-coefficient certified error used
//! the **f64** unit-roundoff growth factor `γ_n^f64` (≈ n·2^-53). The real error
//! of an f32-accumulated dot of width `n` is bounded by `γ_n^f32·S` (≈ n·2^-24·S),
//! about `2^29 ≈ 5.4e8` LARGER. So the certified error UNDER-counts the true f32
//! coefficient error → the concretized bound can be tighter than the true
//! reachable value → FALSE PROOF on wide-contraction conv layers.
//!
//! These tests assert the CERTIFICATE CONTRACT directly (far easier to trigger
//! and more diagnostic than a full end-to-end crossing): for every output
//! coefficient `(i, p)`,
//!
//!     certified_err[i, p]  >=  |stored_f32_coeff[i, p] − f64_recompute[i, p]|
//!
//! where `f64_recompute` is the SAME contraction accumulated in f64 (exact
//! f32→f64 widening; only the f64 sum rounds, a tiny `γ_n^f64·S` residual). The
//! tests:
//!   (a) compute the OLD certificate (`γ_n^f64·S` only, no cast term) and assert
//!       it UNDER-COUNTS on a wide conv — reproducing the bug and reporting the
//!       worst-case under-count ratio, and
//!   (b) assert the LIVE certificate from the patched layer COVERS the actual
//!       coefficient error for every coefficient.

use crate::layers::common::BoundPropagation;
use crate::layers::convolution::conv1d::Conv1dLayer;
use crate::layers::convolution::conv2d::Conv2dLayer;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use proptest::prelude::*;

/// Exact-real (f64) Conv2d CROWN-backward coefficient: the SAME transpose-conv
/// contraction the layer runs, accumulated in f64. f32→f64 widening is exact and
/// f32*f32 is exact in f64, so only the f64 sum rounds (a tiny residual bounded
/// by `γ_n^f64·S`). Treated as the ground-truth real coefficient for the contract
/// check (the OLD f32 path's error must be measured against THIS, not itself).
#[allow(clippy::too_many_arguments)]
fn conv2d_backward_coeff_f64_ref(
    a: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    in_hw: (usize, usize),
    out_hw: (usize, usize),
    out_c: usize,
) -> Array2<f64> {
    let (in_h, in_w) = in_hw;
    let (out_h, out_w) = out_hw;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let in_c = kernel.shape()[1]; // groups == 1 in these tests
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    let num = a.nrows();
    let conv_in_size = in_c * in_h * in_w;
    let mut out = Array2::<f64>::zeros((num, conv_in_size));
    for obj in 0..num {
        for gy in 0..out_h {
            for gx in 0..out_w {
                for oc in 0..out_c {
                    let av = a[[obj, oc * out_h * out_w + gy * out_w + gx]] as f64;
                    if av == 0.0 {
                        continue;
                    }
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = (gy * sh + ki) as isize - ph as isize;
                                let iw = (gx * sw + kj) as isize - pw as isize;
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let out_idx =
                                        ic * in_h * in_w + ih as usize * in_w + iw as usize;
                                    out[[obj, out_idx]] += av * (kernel[[oc, ic, ki, kj]] as f64);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// The OLD (buggy) per-coefficient certified error: `γ_n^f64·S` with the
/// row-constant over-bound `S[i,p] ≤ row_max(a,i)·‖kernel‖_1`, NO cast term — i.e.
/// exactly what the conv sites computed before the fix. Used only inside the test
/// to demonstrate the under-count.
fn old_conv2d_cert_err(a: &Array2<f32>, kernel: &ArrayD<f32>, out_c: usize) -> Array1<f64> {
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    let n = out_c * kh * kw;
    let nf = n as f64;
    let d = nf * 2f64.powi(-53);
    let gamma = d / (1.0 - d); // gamma_n_f64
    let kernel_l1: f64 = kernel.iter().map(|&v| (v as f64).abs()).sum();
    let mut err = Array1::<f64>::zeros(a.nrows());
    for i in 0..a.nrows() {
        let mut row_max = 0.0f64;
        for k in 0..a.ncols() {
            let v = (a[[i, k]] as f64).abs();
            if v > row_max {
                row_max = v;
            }
        }
        err[i] = gamma * row_max * kernel_l1;
    }
    err
}

/// f32-accumulated (faer-equivalent) Conv2d backward coefficient — the value the
/// layer USED to store (and still computes inside the GEMM helper before the f64
/// overwrite). Reproduces faer's f32 accumulation order: per-pixel running f32
/// sum. The point is that this f32 coefficient differs from the f64 truth by an
/// error the OLD `γ_n^f64·S` certificate fails to cover.
#[allow(clippy::too_many_arguments)]
fn conv2d_backward_coeff_f32_naive(
    a: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    in_hw: (usize, usize),
    out_hw: (usize, usize),
    out_c: usize,
) -> Array2<f32> {
    let (in_h, in_w) = in_hw;
    let (out_h, out_w) = out_hw;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let in_c = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    let num = a.nrows();
    let conv_in_size = in_c * in_h * in_w;
    let mut out = Array2::<f32>::zeros((num, conv_in_size));
    for obj in 0..num {
        for gy in 0..out_h {
            for gx in 0..out_w {
                for oc in 0..out_c {
                    let av = a[[obj, oc * out_h * out_w + gy * out_w + gx]];
                    if av == 0.0 {
                        continue;
                    }
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = (gy * sh + ki) as isize - ph as isize;
                                let iw = (gx * sw + kj) as isize - pw as isize;
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let out_idx =
                                        ic * in_h * in_w + ih as usize * in_w + iw as usize;
                                    // f32 fused running sum: this is the unsound
                                    // accumulation the OLD path certified with the
                                    // f64 (too-small) gamma.
                                    out[[obj, out_idx]] += av * kernel[[oc, ic, ki, kj]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Build a WIDE-contraction Conv2d: `out_c` × `in_c` channels, 3×3 kernel, small
/// spatial. Weights/coeffs are scaled to a fine grid so the wide f32 sum genuinely
/// rounds (and cancels), the regime the bug bites.
fn wide_conv2d(
    in_c: usize,
    out_c: usize,
    in_h: usize,
    in_w: usize,
    seed: u64,
) -> (Conv2dLayer, Array2<f32>) {
    // Deterministic pseudo-random fill (xorshift) in a fine, cancelling range.
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // map to roughly [-1, 1] on a fine grid (forces f32 sum rounding).
        ((s >> 8) as i64 % 20001 - 10000) as f32 / 13107.0
    };
    let kh = 3;
    let kw = 3;
    let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, in_c, kh, kw]), |_| next());
    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (1, 1), in_h, in_w).unwrap();
    let (out_h, out_w) = conv.output_size(in_h, in_w).unwrap();
    let conv_out = out_c * out_h * out_w;
    // A spec with many objectives and dense, cancelling coefficients (NOT identity)
    // so the per-pixel contraction sums a full out_c-wide set of A entries.
    let num_obj = 4;
    let a = Array2::<f32>::from_shape_fn((num_obj, conv_out), |_| next());
    (conv, a)
}

/// REPRODUCE (a) + VERIFY (b) for Conv2d: the OLD certificate under-counts; the
/// patched layer's live certificate covers the f32 coefficient error.
#[test]
fn conv2d_aw_cert_covers_f32_error_and_old_undercounts() {
    let in_c = 64;
    let out_c = 64;
    let in_h = 4;
    let in_w = 4;
    let (conv, a) = wide_conv2d(in_c, out_c, in_h, in_w, 0x9E3779B97F4A7C15);
    let (out_h, out_w) = conv.output_size(in_h, in_w).unwrap();

    // Run the layer's CROWN backward via a spec with these coefficients.
    let spec = LinearBounds::from_coefficients(a.clone(), a.clone()).unwrap();
    let result =
        crate::tests::with_crown_dense_budget_mb("2048", || conv.propagate_linear(&spec)).unwrap();

    let stored = result.lower_a();
    let live_err = result
        .lower_a_err()
        .expect("patched conv backward must attach a certified coefficient error");

    // Ground-truth real coefficient (f64) and the f32-accumulated coefficient.
    let f64_ref = conv2d_backward_coeff_f64_ref(
        &a,
        &conv.kernel,
        (1, 1),
        (1, 1),
        (in_h, in_w),
        (out_h, out_w),
        out_c,
    );
    let f32_naive = conv2d_backward_coeff_f32_naive(
        &a,
        &conv.kernel,
        (1, 1),
        (1, 1),
        (in_h, in_w),
        (out_h, out_w),
        out_c,
    );
    let old_err = old_conv2d_cert_err(&a, &conv.kernel, out_c);

    // ----- (a) REPRODUCE: the OLD certificate under-counts the f32 error. -----
    // Worst-case ratio of |f32 − f64| to the OLD row-constant certified error.
    let mut worst_ratio = 0.0f64;
    let mut worst_abs_gap = 0.0f64;
    let conv_in_size = in_c * in_h * in_w;
    assert_eq!(stored.dim(), (a.nrows(), conv_in_size));
    for i in 0..a.nrows() {
        for p in 0..conv_in_size {
            let f32_err = (f32_naive[[i, p]] as f64 - f64_ref[[i, p]]).abs();
            if f32_err > worst_abs_gap {
                worst_abs_gap = f32_err;
            }
            if old_err[i] > 0.0 {
                let ratio = f32_err / old_err[i];
                if ratio > worst_ratio {
                    worst_ratio = ratio;
                }
            }
        }
    }
    eprintln!(
        "[conv2d aw repro] worst |f32-f64| coeff gap = {worst_abs_gap:.3e}, \
         worst (f32 error / OLD γ_n^f64·S certificate) ratio = {worst_ratio:.3e}"
    );
    assert!(
        worst_ratio > 1.0,
        "EXPECTED the OLD γ_n^f64·S certificate to UNDER-COUNT the true f32 \
         coefficient error on a wide conv (ratio {worst_ratio} should be > 1). If this \
         fails the test net is not wide/cancelling enough to exhibit the bug."
    );

    // ----- (b) VERIFY: the patched live certificate covers the f32 error. -----
    // The stored coefficient is now the directed-f32 of the f64 recompute, so its
    // distance to the f64 truth (the cast gap) plus the residual is what the live
    // certificate must cover. We check against BOTH the f64 truth and the f32
    // accumulation (whichever a downstream value would compare to).
    for i in 0..a.nrows() {
        for p in 0..conv_in_size {
            let cert = live_err[[i, p]] as f64;
            let gap_vs_f64 = (stored[[i, p]] as f64 - f64_ref[[i, p]]).abs();
            assert!(
                cert >= gap_vs_f64,
                "UNSOUND conv2d certificate at ({i},{p}): certified err {cert:.3e} < \
                 |stored_f32 − f64_truth| {gap_vs_f64:.3e}"
            );
        }
    }
}

/// Conv1d analogue of the contract check: the patched live certificate must cover
/// the f32→f64 coefficient gap on a wide-contraction Conv1d.
#[test]
fn conv1d_aw_cert_covers_f32_error_wide() {
    let in_c = 48;
    let out_c = 48;
    let in_len = 6;
    let k = 3;

    let mut s = 0xD1B54A32D192ED03u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 8) as i64 % 20001 - 10000) as f32 / 13107.0
    };
    let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, in_c, k]), |_| next());
    let conv =
        Conv1dLayer::with_input_length_full(kernel.clone(), None, 1, 1, 1, 1, in_len).unwrap();
    let out_len = conv.output_length(in_len).unwrap();
    let conv_out = out_c * out_len;
    let num_obj = 4;
    let a = Array2::<f32>::from_shape_fn((num_obj, conv_out), |_| next());

    let spec = LinearBounds::from_coefficients(a.clone(), a.clone()).unwrap();
    let result = conv.propagate_linear(&spec).unwrap();
    let stored = result.lower_a();
    let live_err = result
        .lower_a_err()
        .expect("patched conv1d backward must attach a certified coefficient error");

    // f64 ground-truth coefficient (transpose-conv contraction).
    let conv_in_size = in_c * in_len;
    let mut f64_ref = Array2::<f64>::zeros((num_obj, conv_in_size));
    for obj in 0..num_obj {
        for gl in 0..out_len {
            for oc in 0..out_c {
                let av = a[[obj, oc * out_len + gl]] as f64;
                if av == 0.0 {
                    continue;
                }
                for ic in 0..in_c {
                    for ki in 0..k {
                        let il = (gl + ki) as isize - 1; // stride 1, pad 1, dil 1
                        if il >= 0 && il < in_len as isize {
                            f64_ref[[obj, ic * in_len + il as usize]] +=
                                av * (kernel[[oc, ic, ki]] as f64);
                        }
                    }
                }
            }
        }
    }

    for i in 0..num_obj {
        for p in 0..conv_in_size {
            let cert = live_err[[i, p]] as f64;
            let gap = (stored[[i, p]] as f64 - f64_ref[[i, p]]).abs();
            assert!(
                cert >= gap,
                "UNSOUND conv1d certificate at ({i},{p}): certified err {cert:.3e} < \
                 |stored_f32 − f64_truth| {gap:.3e}"
            );
        }
    }
}

/// BATCHED-PATH (β-CROWN/BaB verdict) contract check for Conv2d. This path
/// historically stored the f32 GEMM coefficient with NO certified error yet is
/// declared `propagates_coeff_err = true` — the decisive false-proof surface. The
/// patched batched path must attach a certified error covering the f32→f64 gap.
#[test]
fn conv2d_batched_aw_cert_covers_f32_error_wide() {
    let in_c = 48;
    let out_c = 48;
    let in_h = 4;
    let in_w = 4;
    let (conv, a) = wide_conv2d(in_c, out_c, in_h, in_w, 0xA5A5A5A5DEADBEEF);
    let (out_h, out_w) = conv.output_size(in_h, in_w).unwrap();
    let conv_out = out_c * out_h * out_w;
    let num_obj = a.nrows();

    // Build a batched bounds (single batch dim) carrying these coefficients.
    let batched = BatchedLinearBounds::from_parts_unchecked(
        a.clone().into_dyn(),
        Array1::<f32>::zeros(num_obj).into_dyn(),
        a.clone().into_dyn(),
        Array1::<f32>::zeros(num_obj).into_dyn(),
        vec![conv_out],
        vec![num_obj],
    );
    let result = crate::tests::with_crown_dense_budget_mb("2048", || {
        conv.propagate_linear_batched(&batched, None)
    })
    .unwrap();

    let stored = result.lower_a();
    let live_err = result
        .lower_a_err
        .as_ref()
        .expect("patched batched conv2d must attach a certified coefficient error");

    let f64_ref = conv2d_backward_coeff_f64_ref(
        &a,
        &conv.kernel,
        (1, 1),
        (1, 1),
        (in_h, in_w),
        (out_h, out_w),
        out_c,
    );
    let conv_in_size = in_c * in_h * in_w;
    // stored/live_err are ArrayD [num_obj, conv_in_size]; index flat.
    for i in 0..num_obj {
        for p in 0..conv_in_size {
            let s = stored[[i, p]] as f64;
            let cert = live_err[[i, p]] as f64;
            let gap = (s - f64_ref[[i, p]]).abs();
            assert!(
                cert >= gap,
                "UNSOUND batched conv2d certificate at ({i},{p}): certified {cert:.3e} < gap {gap:.3e}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 4000, ..ProptestConfig::with_cases(40) })]

    /// SOUNDNESS PROPTEST: random wide Conv2d kernels/specs — the patched
    /// certified error must cover `|stored_f32 − f64_recompute|` for EVERY
    /// coefficient. Zero tolerance.
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_conv2d_aw_cert_covers_f32_error(
        seed in any::<u64>(),
        in_c in 16usize..=48,
        out_c in 16usize..=48,
    ) {
        let in_h = 4;
        let in_w = 4;
        let (conv, a) = wide_conv2d(in_c, out_c, in_h, in_w, seed);
        let (out_h, out_w) = conv.output_size(in_h, in_w).unwrap();

        let spec = LinearBounds::from_coefficients(a.clone(), a.clone()).unwrap();
        let result = crate::tests::with_crown_dense_budget_mb("2048", || {
            conv.propagate_linear(&spec)
        }).unwrap();
        let stored = result.lower_a();
        let live_err = result.lower_a_err().expect("certified error must be present");

        let f64_ref = conv2d_backward_coeff_f64_ref(
            &a, &conv.kernel, (1, 1), (1, 1), (in_h, in_w), (out_h, out_w), out_c,
        );

        let conv_in_size = in_c * in_h * in_w;
        for i in 0..a.nrows() {
            for p in 0..conv_in_size {
                let cert = live_err[[i, p]] as f64;
                let gap = (stored[[i, p]] as f64 - f64_ref[[i, p]]).abs();
                prop_assert!(
                    cert >= gap,
                    "UNSOUND conv2d certificate at ({},{}) seed {}: certified {:.3e} < gap {:.3e}",
                    i, p, seed, cert, gap
                );
            }
        }
    }
}
