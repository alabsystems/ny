// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STRICT, ZERO-TOLERANCE soundness tests for the AveragePool CROWN-backward
//! coefficient certificate AND the AveragePool IBP window sum
//! (#vnncomp-aw-soundness — avgpool f32-accumulation bug, same class as conv
//! becc501).
//!
//! TWO bugs are covered:
//!
//! (1) CROWN: the backward coefficient `A'[i,x] = Σ_{windows w covering x}
//!     A[i,y_w]·weight_w` is accumulated in **round-to-nearest f32**
//!     (`new_*_a[[..]] += a[..]·weight`), but the per-coefficient certified error
//!     used the **f64** unit-roundoff growth factor `γ_n^f64` (≈ n·2^-53). The
//!     real error of an f32-accumulated dot of width `n` is bounded by `γ_n^f32·S`
//!     (≈ n·2^-24·S), ~2^29 LARGER. So the certified error UNDER-counts the true
//!     f32 coefficient error → the concretized bound can be tighter than the true
//!     reachable value → FALSE PROOF. The stored coefficient IS the f32 sum, so
//!     the fix swaps to `γ_n^f32` (no cast term — `coeff_f64 = None` mode).
//!
//! (2) IBP: the window sum was accumulated in **f32** (`sum += x[..]`), then only
//!     the final divide was directed-rounded. But each f32 `+=` rounds to nearest,
//!     which can pull the lower running sum UP / the upper running sum DOWN →
//!     uncertified inward rounding → the concretized IBP bound can EXCLUDE the true
//!     value → FALSE PROOF. The fix accumulates the sum in f64 (exact widening;
//!     only the f64 sum rounds, sub-ULP) then directed-rounds the single f64→f32
//!     store OUTWARD.
//!
//! The CROWN tests assert the CERTIFICATE CONTRACT directly:
//!
//!     certified_err[i, p]  >=  |stored_f32_coeff[i, p] − f64_recompute[i, p]|
//!
//! and reproduce the OLD `γ_n^f64`-only under-count. The IBP test asserts the
//! concretized bound encloses the true f32-evaluated avgpool output with ZERO
//! tolerance.

use crate::layers::common::BoundPropagation;
use crate::layers::pooling::AveragePoolLayer;
use crate::LinearBounds;
use ndarray::{Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

/// Deterministic xorshift fill mapped to a fine, cancelling grid in ~[-1, 1].
/// A fine grid forces the wide f32 sum to genuinely round (and cancel), the
/// regime the bug bites.
fn xorshift_fill(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 8) as i64 % 20001 - 10000) as f32 / 13107.0
    }
}

/// Exact-real (f64) AveragePool CROWN-backward coefficient: the SAME window
/// contraction the layer runs, accumulated in f64. f32→f64 widening is exact and
/// f32*f32 is exact in f64, so only the f64 sum rounds (a tiny `γ_n^f64·S`
/// residual). Ground-truth real coefficient for the contract check.
fn avgpool_backward_coeff_f64_ref(
    a: &Array2<f32>,
    layer: &AveragePoolLayer,
    channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Array2<f64> {
    let (kh, kw) = layer.kernel_size;
    let (sh, sw) = layer.stride;
    let (ph, pw) = layer.padding;
    let num = a.nrows();
    let in_size = channels * in_h * in_w;
    let mut out = Array2::<f64>::zeros((num, in_size));
    for i in 0..num {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let ih_start = oh * sh;
                    let iw_start = ow * sw;
                    let mut count = 0usize;
                    for kh_off in 0..kh {
                        for kw_off in 0..kw {
                            let ih = (ih_start + kh_off) as isize - ph as isize;
                            let iw = (iw_start + kw_off) as isize - pw as isize;
                            if (ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize)
                                || layer.count_include_pad
                            {
                                count += 1;
                            }
                        }
                    }
                    let divisor = if layer.count_include_pad {
                        (kh * kw) as f64
                    } else {
                        count.max(1) as f64
                    };
                    let weight = 1.0f64 / divisor;
                    let y_flat = c * out_h * out_w + oh * out_w + ow;
                    let av = a[[i, y_flat]] as f64;
                    for kh_off in 0..kh {
                        for kw_off in 0..kw {
                            let ih = (ih_start + kh_off) as isize - ph as isize;
                            let iw = (iw_start + kw_off) as isize - pw as isize;
                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                let x_flat = c * in_h * in_w + ih as usize * in_w + iw as usize;
                                out[[i, x_flat]] += av * weight;
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
/// row-constant over-bound `S[i] ≤ row_max(a,i)·n`, NO cast term — exactly what
/// the avgpool site computed before the fix (the f64 gamma over an f32-accumulated
/// coefficient). Used only inside the test to demonstrate the under-count.
fn old_avgpool_cert_err(
    a: &Array2<f32>,
    layer: &AveragePoolLayer,
    out_h: usize,
    out_w: usize,
) -> Vec<f64> {
    let (kh, kw) = layer.kernel_size;
    let (sh, sw) = layer.stride;
    let nw_h = kh.div_ceil(sh.max(1));
    let nw_w = kw.div_ceil(sw.max(1));
    let n = nw_h
        .saturating_mul(nw_w)
        .min(out_h.saturating_mul(out_w))
        .max(1);
    let nf = n as f64;
    let d = nf * 2f64.powi(-53);
    let gamma_f64 = d / (1.0 - d); // gamma_n_f64 (the OLD, too-small factor)
    let weight_l1 = n as f64;
    let mut err = vec![0.0f64; a.nrows()];
    for i in 0..a.nrows() {
        let mut row_max = 0.0f64;
        for k in 0..a.ncols() {
            let v = (a[[i, k]] as f64).abs();
            if v > row_max {
                row_max = v;
            }
        }
        err[i] = gamma_f64 * row_max * weight_l1;
    }
    err
}

/// Build a wide-overlap AveragePool spec: a large stride-1 kernel so many output
/// windows cover each input pixel (wide contraction), with dense cancelling
/// coefficients so the per-pixel f32 sum genuinely rounds.
fn wide_avgpool(
    channels: usize,
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    seed: u64,
) -> (AveragePoolLayer, Array2<f32>, usize, usize) {
    let layer = AveragePoolLayer::new((kh, kw), (1, 1), (0, 0), false);
    let (out_h, out_w) = layer.output_size(in_h, in_w).unwrap();
    let out_size = channels * out_h * out_w;
    let num_obj = 4;
    let mut next = xorshift_fill(seed);
    let a = Array2::<f32>::from_shape_fn((num_obj, out_size), |_| next());
    (layer, a, out_h, out_w)
}

/// REPRODUCE (a) + VERIFY (b) for AveragePool CROWN backward: the OLD γ_n^f64·S
/// certificate UNDER-counts the true f32 coefficient error; the patched
/// (γ_n^f32·S) certificate COVERS it for every coefficient.
#[test]
fn avgpool_aw_cert_covers_f32_error_and_old_undercounts() {
    let channels = 4;
    let in_h = 9;
    let in_w = 9;
    let kh = 7;
    let kw = 7;
    let (layer, a, out_h, out_w) = wide_avgpool(channels, in_h, in_w, kh, kw, 0x9E3779B97F4A7C15);

    // Pre-activation bounds (unused by linear avgpool except for shape).
    let in_shape = vec![channels, in_h, in_w];
    let pre = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&in_shape), -1.0f32),
        ArrayD::from_elem(IxDyn(&in_shape), 1.0f32),
    )
    .unwrap();

    let spec = LinearBounds::from_coefficients(a.clone(), a.clone()).unwrap();
    let result = layer.propagate_linear_with_bounds(&spec, &pre).unwrap();

    let stored = result.lower_a();
    let live_err = result
        .lower_a_err()
        .expect("avgpool backward must attach a certified coefficient error");

    let f64_ref = avgpool_backward_coeff_f64_ref(&a, &layer, channels, in_h, in_w, out_h, out_w);
    let old_err = old_avgpool_cert_err(&a, &layer, out_h, out_w);

    let in_size = channels * in_h * in_w;
    assert_eq!(stored.dim(), (a.nrows(), in_size));

    // ----- (a) REPRODUCE: the OLD γ_n^f64·S certificate under-counts. -----
    // The stored coefficient is the f32-accumulated sum; its distance to the f64
    // truth is the real coefficient error the OLD f64 certificate must cover.
    let mut worst_ratio = 0.0f64;
    let mut worst_abs_gap = 0.0f64;
    for i in 0..a.nrows() {
        for p in 0..in_size {
            let f32_err = (stored[[i, p]] as f64 - f64_ref[[i, p]]).abs();
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
        "[avgpool aw repro] worst |f32-f64| coeff gap = {worst_abs_gap:.3e}, \
         worst (f32 error / OLD γ_n^f64·S certificate) ratio = {worst_ratio:.3e}"
    );
    assert!(
        worst_ratio > 1.0,
        "EXPECTED the OLD γ_n^f64·S certificate to UNDER-COUNT the true f32 \
         coefficient error on a wide-overlap avgpool (ratio {worst_ratio} should be > 1). \
         If this fails the net is not wide/cancelling enough to exhibit the bug."
    );

    // ----- (b) VERIFY: the patched γ_n^f32·S certificate covers the f32 error. -----
    for i in 0..a.nrows() {
        for p in 0..in_size {
            let cert = live_err[[i, p]] as f64;
            let gap_vs_f64 = (stored[[i, p]] as f64 - f64_ref[[i, p]]).abs();
            assert!(
                cert >= gap_vs_f64,
                "UNSOUND avgpool certificate at ({i},{p}): certified err {cert:.3e} < \
                 |stored_f32 − f64_truth| {gap_vs_f64:.3e}"
            );
        }
    }
}

/// Exact-real (f64) global-average-pool of one channel over a flat slice.
fn global_avg_f64(vals: &[f64]) -> f64 {
    let n = vals.len().max(1) as f64;
    vals.iter().sum::<f64>() / n
}

/// IBP soundness: the concretized AveragePool IBP bound must ENCLOSE the true
/// avgpool of EVERY concrete input in the box — checked here at the corner that a
/// pre-fix f32 inward-rounding running sum would expose. ZERO tolerance.
///
/// The pre-fix path accumulated `sum_lower`/`sum_upper` in f32; with many terms
/// on a fine grid the running sum rounds to nearest each step, which for the lower
/// bound can round UP (so the stored lower exceeds the true min) and for the upper
/// can round DOWN (stored upper below the true max). We feed a wide global pool
/// with cancelling values and assert the output bound brackets the f64-exact pool
/// of the lower-corner and upper-corner inputs.
#[test]
fn avgpool_ibp_global_sum_encloses_true_value_zero_tol() {
    let channels = 3;
    let in_h = 13;
    let in_w = 13; // 169-wide sum per channel — wide enough for f32 to round.

    let mut next_l = xorshift_fill(0xD1B54A32D192ED03);
    let mut next_d = xorshift_fill(0x2545F4914F6CDD1D);

    let n = channels * in_h * in_w;
    let mut lowers = Vec::with_capacity(n);
    let mut uppers = Vec::with_capacity(n);
    for _ in 0..n {
        let l = next_l();
        let d = (next_d().abs()) * 0.5; // non-negative width
        lowers.push(l);
        uppers.push(l + d);
    }

    let lower_arr = ArrayD::from_shape_vec(IxDyn(&[channels, in_h, in_w]), lowers.clone()).unwrap();
    let upper_arr = ArrayD::from_shape_vec(IxDyn(&[channels, in_h, in_w]), uppers.clone()).unwrap();
    let input = BoundedTensor::new(lower_arr, upper_arr).unwrap();

    // Global average pool: kernel (0,0) sentinel.
    let layer = AveragePoolLayer::new((0, 0), (1, 1), (0, 0), false);
    let out = layer.propagate_ibp(&input).unwrap();

    for c in 0..channels {
        // True f64 pool of the lower-corner and upper-corner concrete inputs.
        let mut lo_vals = Vec::with_capacity(in_h * in_w);
        let mut hi_vals = Vec::with_capacity(in_h * in_w);
        for ih in 0..in_h {
            for iw in 0..in_w {
                let idx = c * in_h * in_w + ih * in_w + iw;
                lo_vals.push(lowers[idx] as f64);
                hi_vals.push(uppers[idx] as f64);
            }
        }
        let true_lo = global_avg_f64(&lo_vals); // pool of the all-lower corner
        let true_hi = global_avg_f64(&hi_vals); // pool of the all-upper corner

        let out_lo = out.lower()[[c, 0, 0]] as f64;
        let out_hi = out.upper()[[c, 0, 0]] as f64;

        // ZERO-tolerance enclosure: the IBP lower bound must not exceed the true
        // pooled value of any input (the lower corner is the channel-min for a
        // monotone sum), and the upper bound must not fall below it.
        assert!(
            out_lo <= true_lo,
            "UNSOUND avgpool IBP lower (ch {c}): out_lo {out_lo:.9e} > true pool of lower-corner {true_lo:.9e}"
        );
        assert!(
            out_hi >= true_hi,
            "UNSOUND avgpool IBP upper (ch {c}): out_hi {out_hi:.9e} < true pool of upper-corner {true_hi:.9e}"
        );
    }
}
