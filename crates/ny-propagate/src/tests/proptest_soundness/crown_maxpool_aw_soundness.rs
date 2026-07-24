// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STRICT, ZERO-TOLERANCE soundness tests for the two MaxPool CROWN-backward bugs
//! (#maxpool-patches-lumped-bias, #maxpool-dense-winner-coeff-accum).
//!
//! ## Bug A — patches Step-2 lumped bias (max_patches.rs)
//!
//! The patches bias loop summed EVERY tap coefficient of a downstream spec row
//! into a single `pos_sum`/`neg_sum` and multiplied by the MaxPool output bound at
//! the SPEC index `[o_c,o_h,o_w]` — but each tap maps to a DIFFERENT MaxPool output
//! element (the spec's receptive field spans several MaxPool outputs), each with its
//! own constant bound when non-linear. The lumped bias under/over-counts → the
//! concretized bound can land on the wrong side of the true reachable value = FALSE
//! PROOF. The fix is a per-tap lookup of the tap's own MaxPool output bound.
//!
//! Test A builds a MaxPool (wide intervals → non-linear windows) → Conv2d(3×3)
//! chain. The Conv's downstream patches spec makes each Conv-output receptive field
//! span a 3×3 block of MaxPool outputs with mixed-sign coefficients — exactly the
//! regime the lumped bias gets wrong. We concretize the patches bound over the input
//! box and assert it encloses the TRUE sampled value `Conv(MaxPool(x))` for many
//! concrete `x` with ZERO tolerance.
//!
//! ## Bug B — dense winner-coefficient f32 accumulation (max.rs)
//!
//! A single input column can be the definite winner (or the i* lower-witness) for
//! several overlapping output windows (stride < kernel), so the dense backward
//! `+=` an f32 coefficient into one column multiple times. The OLD path stored that
//! round-to-nearest f32 running sum with NO certified error → the concretized bound
//! could be tighter than the true value. The fix accumulates in f64 and certifies
//! `|f64−stored_f32| + γ_n^f64·S`.
//!
//! Test B builds an overlapping-window MaxPool with a wide, cancelling spec so the
//! per-column f32 sum genuinely rounds, then asserts (a) the patched layer attaches
//! a certified coefficient error (the OLD path attached none → repro), and (b) for
//! every coefficient `certified_err >= |stored_f32 − exact_f64_recompute|`.

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::layers::common::{BoundPropagation, PatchesPropagation};
use crate::layers::convolution::conv2d::Conv2dLayer;
use crate::layers::pooling::max::MaxPool2dLayer;
use crate::LinearBounds;
use ndarray::{Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

/// Deterministic xorshift fill in a fine, cancelling range (forces f32 rounding).
fn filler(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 8) as i64 % 20001 - 10000) as f32 / 13107.0
    }
}

/// Concretize per-output scalar bounds of `lb` over an input box IN f64, INCLUDING
/// the certified coefficient-error penalty (lower goes down, upper goes up). This
/// mirrors `LinearBounds::concretize_sound` so the test exercises the same contract
/// the verifier relies on.
fn concretize_sound_f64(lb: &LinearBounds, input: &BoundedTensor) -> (Vec<f64>, Vec<f64>) {
    let in_l: Vec<f64> = input.lower().iter().map(|&v| v as f64).collect();
    let in_u: Vec<f64> = input.upper().iter().map(|&v| v as f64).collect();
    let out_dim = lb.lower_b().len();
    let le = lb.lower_a_err();
    let ue = lb.upper_a_err();
    let mut lowers = Vec::with_capacity(out_dim);
    let mut uppers = Vec::with_capacity(out_dim);
    for o in 0..out_dim {
        let mut lo = lb.lower_b()[o] as f64;
        let mut hi = lb.upper_b()[o] as f64;
        for j in 0..in_l.len() {
            let la = lb.lower_a()[[o, j]] as f64;
            let ua = lb.upper_a()[[o, j]] as f64;
            lo += la.min(0.0) * in_u[j] + la.max(0.0) * in_l[j];
            hi += ua.max(0.0) * in_u[j] + ua.min(0.0) * in_l[j];
            let mag = in_l[j].abs().max(in_u[j].abs());
            if let Some(e) = le {
                lo -= e[[o, j]] as f64 * mag;
            }
            if let Some(e) = ue {
                hi += e[[o, j]] as f64 * mag;
            }
        }
        lowers.push(lo);
        uppers.push(hi);
    }
    (lowers, uppers)
}

/// True value of `Conv(MaxPool(x))` at a concrete input point `x` (3D C,H,W).
/// Evaluated exactly via point-IBP: for lower==upper==x, MaxPool and Conv IBP are
/// exact, so the returned lower (== upper) is the real function value.
fn true_conv_of_maxpool(maxpool: &MaxPool2dLayer, conv: &Conv2dLayer, x: &ArrayD<f32>) -> Vec<f64> {
    let pt = BoundedTensor::new(x.clone(), x.clone()).unwrap();
    let pooled = maxpool.propagate_ibp(&pt).unwrap();
    let out = conv.propagate_ibp(&pooled).unwrap();
    out.lower().iter().map(|&v| v as f64).collect()
}

/// REPRO + VERIFY for Bug A: the patches MaxPool bound must enclose the true
/// `Conv(MaxPool(x))` value over the input box with ZERO tolerance.
///
/// On the BUGGY (lumped-bias) code this FAILS: the lumped bias under/over-counts
/// the constant contribution of the non-linear MaxPool windows, so the concretized
/// patches bound crosses the true value. On the patched (per-tap) code it PASSES.
#[test]
fn maxpool_patches_per_tap_bias_encloses_true_value() {
    // MaxPool 2×2 stride 2 on a 12×12 input → 6×6 pooled feature map. The pooled
    // map must be larger than the conv kernel area (6×6 > 3×3) so the conv patches
    // survive instead of degrading to dense (see should_fallback_to_dense).
    let channels = 2usize;
    let in_h = 12usize;
    let in_w = 12usize;
    let maxpool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    let pool_h = 6usize;
    let pool_w = 6usize;

    // Conv2d 3×3 stride 1 pad 1 on the 6×6 pooled map → 6×6 output. Each conv
    // output's receptive field is a 3×3 block of MaxPool outputs (different
    // bounds), with mixed-sign kernel coeffs — the lumped-bias failure regime.
    let out_c = 2usize;
    let mut kf = filler(0x51ED270B_2E5A1C33);
    let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, channels, 3, 3]), |_| kf());
    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (1, 1), pool_h, pool_w).unwrap();

    // Pre-activation built so each MaxPool output is DELIBERATELY linear or
    // non-linear on a checkerboard of the pooled grid:
    //   (oh+ow) even  → DEFINITE WINNER (linear): one cell dominates the 2×2
    //                   window → lower_b_per_pos/upper_b_per_pos = 0 there.
    //   (oh+ow) odd   → NO WINNER (non-linear): all 4 cells overlap on a LARGE
    //                   positive interval → a big constant bound.
    // The Conv(3×3,pad1) downstream spec at an EVEN (linear) output position reaches
    // its ODD (non-linear, large-constant) neighbors. The OLD lumped bias reads the
    // bound at the SPEC index (== 0 for a linear output) and SKIPS the whole bias
    // (`if lb==0 && ub==0 continue`), DROPPING the large constant the neighbors
    // contribute → the concretized bound is far too tight and CROSSES the true
    // value. The per-tap fix looks up each tap's own (non-linear neighbor) bound.
    let mut lower = ArrayD::<f32>::zeros(IxDyn(&[channels, in_h, in_w]));
    let mut upper = ArrayD::<f32>::zeros(IxDyn(&[channels, in_h, in_w]));
    for c in 0..channels {
        for oh in 0..pool_h {
            for ow in 0..pool_w {
                let nonlinear = (oh + ow) % 2 == 1;
                // The 2×2 input window for this pooled output.
                let h0 = oh * 2;
                let w0 = ow * 2;
                for dh in 0..2 {
                    for dw in 0..2 {
                        let h = h0 + dh;
                        let w = w0 + dw;
                        if nonlinear {
                            // Large positive, overlapping interval → no winner;
                            // bound ≈ [49, 51] + small per-cell jitter.
                            let jit = ((c + oh * 3 + ow * 5 + dh + dw) % 7) as f32 * 0.05;
                            lower[[c, h, w]] = 49.0 + jit;
                            upper[[c, h, w]] = 51.0 + jit;
                        } else {
                            // One dominating cell (winner) + three clearly-smaller
                            // cells → definite winner (tight intervals).
                            let winner = dh == 1 && dw == 1;
                            let v = if winner { 2.0 } else { -3.0 };
                            lower[[c, h, w]] = v - 0.01;
                            upper[[c, h, w]] = v + 0.01;
                        }
                    }
                }
            }
        }
    }
    let pre_act = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // --- Patches path: identity at Conv output → Conv patches → MaxPool patches. ---
    let patches_id =
        PatchesLinearBounds::identity((out_c, pool_h, pool_w), (out_c, pool_h, pool_w));
    // Conv backward in patches mode (its pre-activation is the pooled map bounds).
    let pooled_bt = maxpool.propagate_ibp(&pre_act).unwrap();
    let after_conv = conv
        .propagate_patches_with_bounds(&patches_id, &pooled_bt)
        .unwrap();
    let conv_patches = match after_conv {
        CrownBounds::Patches(pb) => *pb,
        CrownBounds::Dense(_) => {
            panic!("conv patches degraded to dense; need patches to hit MaxPool patches bias")
        }
    };
    let after_maxpool = maxpool
        .propagate_patches_with_bounds(&conv_patches, &pre_act)
        .unwrap();
    let maxpool_dense = after_maxpool.into_dense().unwrap();

    // Confirm the non-linear bias path actually ran (else the test is vacuous):
    // some output's bias must be non-zero.
    let any_bias = maxpool_dense.lower_b().iter().any(|&v| v != 0.0)
        || maxpool_dense.upper_b().iter().any(|&v| v != 0.0);
    assert!(
        any_bias,
        "test setup did not exercise the non-linear MaxPool bias path (all biases 0)"
    );

    let (lo, hi) = concretize_sound_f64(&maxpool_dense, &pre_act);

    // Sample concrete inputs across the box and verify enclosure with ZERO tolerance.
    let mut sf = filler(0x00C0_FFEE_D00D_1234);
    let mut worst_lower_violation = 0.0f64;
    let mut worst_upper_violation = 0.0f64;
    for _ in 0..400 {
        let mut x = ArrayD::<f32>::zeros(IxDyn(&[channels, in_h, in_w]));
        for c in 0..channels {
            for h in 0..in_h {
                for w in 0..in_w {
                    // Uniform-ish sample in [lower, upper] via a 0..1 fraction.
                    let frac = ((sf() + 0.7635).rem_euclid(1.0)).clamp(0.0, 1.0);
                    let l = lower[[c, h, w]] as f64;
                    let u = upper[[c, h, w]] as f64;
                    x[[c, h, w]] = (l + (u - l) * frac as f64) as f32;
                }
            }
        }
        let truth = true_conv_of_maxpool(&maxpool, &conv, &x);
        for o in 0..truth.len() {
            // ZERO tolerance: the certified bound MUST enclose the true value.
            if truth[o] < lo[o] {
                worst_lower_violation = worst_lower_violation.max(lo[o] - truth[o]);
            }
            if truth[o] > hi[o] {
                worst_upper_violation = worst_upper_violation.max(truth[o] - hi[o]);
            }
        }
    }
    assert!(
        worst_lower_violation == 0.0 && worst_upper_violation == 0.0,
        "UNSOUND MaxPool patches bound crosses true Conv(MaxPool(x)): \
         worst lower violation {worst_lower_violation:.3e}, worst upper violation {worst_upper_violation:.3e}"
    );
}

/// Exact-real (f64) MaxPool dense backward coefficient: re-run the SAME definite-
/// winner / i* routing the layer uses, accumulating each contributed coefficient in
/// f64. The f32→f64 widening is exact, so this is the ground-truth real coefficient.
fn maxpool_dense_winner_coeff_f64(
    a: &Array2<f32>,
    pre_lower: &ArrayD<f32>,
    pre_upper: &ArrayD<f32>,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    out_h: usize,
    out_w: usize,
) -> (Array2<f64>, Array2<f64>) {
    let num = a.nrows();
    let in_size = channels * in_h * in_w;
    let mut la = Array2::<f64>::zeros((num, in_size));
    let mut ua = Array2::<f64>::zeros((num, in_size));
    for c in 0..channels {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let y_flat = c * out_h * out_w + oh * out_w + ow;
                // Collect window (x_flat, l, u).
                let mut win: Vec<(usize, f32, f32)> = Vec::new();
                for ki in 0..kh {
                    for kj in 0..kw {
                        let ih = oh * sh + ki;
                        let iw = ow * sw + kj;
                        if ih < in_h && iw < in_w {
                            let xf = c * in_h * in_w + ih * in_w + iw;
                            win.push((xf, pre_lower[[c, ih, iw]], pre_upper[[c, ih, iw]]));
                        }
                    }
                }
                if win.is_empty() {
                    continue;
                }
                let definite = win
                    .iter()
                    .find(|&&(idx, l, _)| win.iter().all(|&(o, _, u)| idx == o || l >= u));
                if let Some(&(winner, _, _)) = definite {
                    for o in 0..num {
                        la[[o, winner]] += a[[o, y_flat]] as f64;
                        ua[[o, winner]] += a[[o, y_flat]] as f64;
                    }
                } else {
                    // i* = argmax_i l_i (lower witness; routes la>0 and ua<0 rows).
                    let istar = win
                        .iter()
                        .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                        .map(|&(idx, _, _)| idx)
                        .unwrap();
                    for o in 0..num {
                        let lc = a[[o, y_flat]];
                        let uc = a[[o, y_flat]];
                        if lc > 0.0 {
                            la[[o, istar]] += lc as f64;
                        }
                        if uc < 0.0 {
                            ua[[o, istar]] += uc as f64;
                        }
                    }
                }
            }
        }
    }
    (la, ua)
}

/// REPRO + VERIFY for Bug B: overlapping-window MaxPool dense backward must attach a
/// certified coefficient error covering the f32→f64 accumulation gap.
///
/// On the BUGGY code the dense path used `new_or_conservative` (NO error matrix), so
/// `lower_a_err()` is `None` → the `.expect(...)` REPRO fires. On the patched code
/// the error is attached and covers `|stored_f32 − f64_recompute|` for every coeff.
#[test]
fn maxpool_dense_winner_coeff_cert_covers_f32_accum_error() {
    // Overlapping windows: kernel 3×3, stride 1 → each interior input is the winner
    // for up to 9 output windows, so its column receives up to 9 f32 `+=`.
    let channels = 1usize;
    let in_h = 8usize;
    let in_w = 8usize;
    let kh = 3;
    let kw = 3;
    let sh = 1;
    let sw = 1;
    let maxpool = MaxPool2dLayer::new((kh, kw), (sh, sw), (0, 0));
    let out_h = (in_h - kh) / sh + 1;
    let out_w = (in_w - kw) / sw + 1;

    // Pre-activation bounds: a near-zero, slightly-varying background with a few
    // sparse, well-separated DOMINANT spikes. Each spike dominates its 5×5
    // neighborhood, so it is the SINGLE definite winner of all (up to 9)
    // overlapping 3×3 windows that contain it — forcing up to 9 f32 `+=` into the
    // SAME input column for each spec row. Tight intervals keep the winner definite.
    let mut lower = ArrayD::<f32>::zeros(IxDyn(&[channels, in_h, in_w]));
    let mut upper = ArrayD::<f32>::zeros(IxDyn(&[channels, in_h, in_w]));
    let mut bf = filler(0x1234_5678_9ABC_DEF0);
    for h in 0..in_h {
        for w in 0..in_w {
            // Tiny, distinct background so non-spike windows still have a winner.
            let v = bf() * 0.001;
            lower[[0, h, w]] = v - 0.0001;
            upper[[0, h, w]] = v + 0.0001;
        }
    }
    // Spikes at separated interior cells (>=2 apart so neighborhoods don't collide).
    for &(sh_, sw_) in &[(3usize, 3usize), (3, 6), (6, 3), (5, 5), (4, 6)] {
        let v = 100.0_f32 + (sh_ * 7 + sw_) as f32;
        lower[[0, sh_, sw_]] = v - 0.0001;
        upper[[0, sh_, sw_]] = v + 0.0001;
    }
    let pre_act = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // WIDE, cancelling spec (many objectives, fine-grid coeffs) so the per-column
    // f32 running sum genuinely rounds.
    let out_dim = channels * out_h * out_w;
    let num_obj = 6;
    let mut af = filler(0x9E3779B9_7F4A7C15);
    let a = Array2::<f32>::from_shape_fn((num_obj, out_dim), |_| af() * 50.0);
    let spec = LinearBounds::from_coefficients(a.clone(), a.clone()).unwrap();

    let result = maxpool
        .propagate_linear_with_bounds(&spec, &pre_act)
        .unwrap();

    // (a) REPRO: the patched layer MUST attach a certified coefficient error. The
    //     OLD path returned `new_or_conservative` (no err) → this `.expect` fired.
    let live_lower_err = result
        .lower_a_err()
        .expect("patched MaxPool dense backward must attach a certified lower coeff error");
    let live_upper_err = result
        .upper_a_err()
        .expect("patched MaxPool dense backward must attach a certified upper coeff error");

    let (f64_la, f64_ua) = maxpool_dense_winner_coeff_f64(
        &a, &lower, &upper, channels, in_h, in_w, kh, kw, sh, sw, out_h, out_w,
    );

    let stored_la = result.lower_a();
    let stored_ua = result.upper_a();
    let in_size = channels * in_h * in_w;
    assert_eq!(stored_la.dim(), (num_obj, in_size));

    // Sanity: confirm a column really did receive multiple accumulations (else the
    // bug regime is not exercised) by checking some |stored − f64| gap > 0.
    let mut worst_gap = 0.0f64;
    for i in 0..num_obj {
        for p in 0..in_size {
            let gl = (stored_la[[i, p]] as f64 - f64_la[[i, p]]).abs();
            let gu = (stored_ua[[i, p]] as f64 - f64_ua[[i, p]]).abs();
            worst_gap = worst_gap.max(gl).max(gu);
            // (b) VERIFY: certified err covers the f32→f64 coefficient gap.
            assert!(
                live_lower_err[[i, p]] as f64 >= gl,
                "UNSOUND MaxPool lower cert at ({i},{p}): err {:.3e} < gap {:.3e}",
                live_lower_err[[i, p]],
                gl
            );
            assert!(
                live_upper_err[[i, p]] as f64 >= gu,
                "UNSOUND MaxPool upper cert at ({i},{p}): err {:.3e} < gap {:.3e}",
                live_upper_err[[i, p]],
                gu
            );
        }
    }
    eprintln!("[maxpool dense aw] worst |f32 − f64| winner-coeff gap = {worst_gap:.3e}");
    assert!(
        worst_gap > 0.0,
        "test did not exercise multi-accumulation (no f32/f64 coeff gap); \
         widen the spec or overlap to trigger repeated += into shared columns"
    );
}
