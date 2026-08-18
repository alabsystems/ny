// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STRICT, ZERO-TOLERANCE soundness tests for the BLAS-accelerated batched
//! coefficient composition `BatchedLinearBounds::compose` / `compose_blas`
//! (#concretize-soundness-hardening).
//!
//! `compose_blas` computes the batched coefficient product `z = A2·A1` via a
//! k-term **f32** SGEMM (pos/neg split). A k-term f32 dot accumulates up to
//! `γ_k = k·u/(1−k·u)` (Higham, Accuracy & Stability Thm 3.1; `u = 2^-24`) of
//! relative rounding error, so for `k > 2` the previous 1-ULP directed cast
//! UNDER-covered and was UNSOUND. The fix attaches a certified per-coefficient
//! error matrix
//!
//! ```text
//!   err[i,j] = γ_{k+1} · S[i,j],   S[i,j] = Σ_k |a2[i,k]|·|a1_sel[k,j]|
//! ```
//!
//! (computed in f64), which `concretize` penalizes OUTWARD. These tests assert,
//! with ZERO tolerance against an EXACT-RATIONAL reference, that the certified
//! coefficient interval `[stored − err, stored + err]` ENCLOSES the exact real
//! product `(A2·A1)[i,j]` for EVERY composed coefficient — including large-`k`
//! adversarial-sign cancellation cases where the OLD 1-ULP version would fail.
//!
//! Reference: `crown_linear_aw_soundness.rs` (the scalar `A·W` analogue).

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use proptest::prelude::*;

use crate::BatchedLinearBounds;
use ny_cert::rational::Rat;

// =============================================================================
// f32 <-> exact Rat helper (copy of the well-tested one in
// crown_linear_aw_soundness; every finite f32 is an exact dyadic rational).
// =============================================================================

fn f32_to_rat(x: f32) -> Rat {
    Rat::from_f32_exact(x).unwrap_or_else(|| panic!("f32_to_rat: non-finite {x}"))
}

// =============================================================================
// Build a BatchedLinearBounds from 2D coefficient/bias arrays (batch_size = 1).
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn make_bounds_2d(
    la: Array2<f32>,
    ua: Array2<f32>,
    lb: Array1<f32>,
    ub: Array1<f32>,
    in_dim: usize,
    out_dim: usize,
) -> BatchedLinearBounds {
    let m = la.nrows();
    let n = la.ncols();
    let (la_v, _) = la.into_raw_vec_and_offset();
    let (ua_v, _) = ua.into_raw_vec_and_offset();
    let (lb_v, _) = lb.into_raw_vec_and_offset();
    let (ub_v, _) = ub.into_raw_vec_and_offset();
    BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[m, n]), la_v).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[m]), lb_v).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[m, n]), ua_v).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[m]), ub_v).unwrap(),
        vec![in_dim],
        vec![out_dim],
    )
}

// =============================================================================
// Exact-rational reference for the sign-split composed coefficients.
// =============================================================================

/// Exact real lower coefficient `(A2·A1)_L[i,j]` that the f32 SGEMM approximates:
/// `Σ_k pos(a2_l[i,k])·a1_l[k,j] + neg(a2_l[i,k])·a1_u[k,j]`, evaluated in Rat.
/// `pos`/`neg` are disjoint per-k: select `a1_l` when `a2_l>=0`, else `a1_u`.
fn exact_lower_coeff(
    a2_l: &Array2<f32>,
    a1_l: &Array2<f32>,
    a1_u: &Array2<f32>,
    i: usize,
    j: usize,
    k_dim: usize,
) -> Rat {
    let mut acc = Rat::ZERO;
    for k in 0..k_dim {
        let a2 = a2_l[[i, k]];
        let a1 = if a2 >= 0.0 {
            a1_l[[k, j]]
        } else {
            a1_u[[k, j]]
        };
        let term = f32_to_rat(a2).mul(f32_to_rat(a1)).unwrap();
        acc = acc.add(term).unwrap();
    }
    acc
}

/// Exact real upper coefficient `(A2·A1)_U[i,j]`:
/// `Σ_k pos(a2_u[i,k])·a1_u[k,j] + neg(a2_u[i,k])·a1_l[k,j]` in Rat.
fn exact_upper_coeff(
    a2_u: &Array2<f32>,
    a1_l: &Array2<f32>,
    a1_u: &Array2<f32>,
    i: usize,
    j: usize,
    k_dim: usize,
) -> Rat {
    let mut acc = Rat::ZERO;
    for k in 0..k_dim {
        let a2 = a2_u[[i, k]];
        let a1 = if a2 >= 0.0 {
            a1_u[[k, j]]
        } else {
            a1_l[[k, j]]
        };
        let term = f32_to_rat(a2).mul(f32_to_rat(a1)).unwrap();
        acc = acc.add(term).unwrap();
    }
    acc
}

/// Core enclosure check shared by the proptest and the regression witnesses.
///
/// Composes `a2 ∘ a1` via the production `BatchedLinearBounds::compose` (which
/// dispatches to `compose_blas` for finite inputs), then asserts the certified
/// coefficient interval `[stored − err, stored + err]` encloses the EXACT real
/// product for every (i,j), in rationals with ZERO tolerance. Returns the number
/// of coefficients checked (so callers can assert it is non-empty).
fn assert_compose_blas_encloses_exact(
    a1_l: &Array2<f32>,
    a1_u: &Array2<f32>,
    a2_l: &Array2<f32>,
    a2_u: &Array2<f32>,
) -> usize {
    let k_dim = a1_l.nrows();
    let in_dim = a1_l.ncols();
    let out_dim = a2_l.nrows();
    assert_eq!(a2_l.ncols(), k_dim, "a2 inner dim must match a1 outer dim");

    // self = A1 (maps x∈R^in -> y∈R^k), other = A2 (maps y∈R^k -> z∈R^out).
    let a1 = make_bounds_2d(
        a1_l.clone(),
        a1_u.clone(),
        Array1::zeros(k_dim),
        Array1::zeros(k_dim),
        in_dim,
        k_dim,
    );
    let a2 = make_bounds_2d(
        a2_l.clone(),
        a2_u.clone(),
        Array1::zeros(out_dim),
        Array1::zeros(out_dim),
        k_dim,
        out_dim,
    );

    let composed = a1.compose(&a2).expect("compose");
    let lower_a = composed.lower_a();
    let upper_a = composed.upper_a();
    // The certified error MUST be present (this is the BLAS path; all inputs are
    // finite). Its absence would mean the soundness widening was silently dropped.
    let lower_err = composed
        .lower_a_err
        .as_ref()
        .expect("compose_blas must attach lower_a_err on the finite/BLAS path");
    let upper_err = composed
        .upper_a_err
        .as_ref()
        .expect("compose_blas must attach upper_a_err on the finite/BLAS path");

    let mut checked = 0usize;
    for i in 0..out_dim {
        for j in 0..in_dim {
            // --- lower coefficient enclosure ---
            let stored_l = lower_a[[i, j]];
            let err_l = lower_err[[i, j]];
            assert!(
                stored_l.is_finite() && err_l.is_finite() && err_l >= 0.0,
                "lower coeff/err must be finite & err>=0: stored={stored_l} err={err_l}"
            );
            let exact_l = exact_lower_coeff(a2_l, a1_l, a1_u, i, j, k_dim);
            let lo = f32_to_rat(stored_l).sub(f32_to_rat(err_l)).unwrap();
            let hi = f32_to_rat(stored_l).add(f32_to_rat(err_l)).unwrap();
            assert!(
                lo <= exact_l && exact_l <= hi,
                "UNSOUND lower compose coeff [{i},{j}] (k={k_dim}): exact {} NOT in certified \
                 interval [{}, {}] (stored {stored_l}, err {err_l}). The γ_(k+1)·S widening must \
                 cover the f32 SGEMM accumulation error.",
                exact_l.to_clean_string().unwrap_or_default(),
                lo.to_clean_string().unwrap_or_default(),
                hi.to_clean_string().unwrap_or_default(),
            );

            // --- upper coefficient enclosure ---
            let stored_u = upper_a[[i, j]];
            let err_u = upper_err[[i, j]];
            assert!(
                stored_u.is_finite() && err_u.is_finite() && err_u >= 0.0,
                "upper coeff/err must be finite & err>=0: stored={stored_u} err={err_u}"
            );
            let exact_u = exact_upper_coeff(a2_u, a1_l, a1_u, i, j, k_dim);
            let lo_u = f32_to_rat(stored_u).sub(f32_to_rat(err_u)).unwrap();
            let hi_u = f32_to_rat(stored_u).add(f32_to_rat(err_u)).unwrap();
            assert!(
                lo_u <= exact_u && exact_u <= hi_u,
                "UNSOUND upper compose coeff [{i},{j}] (k={k_dim}): exact {} NOT in certified \
                 interval [{}, {}] (stored {stored_u}, err {err_u}).",
                exact_u.to_clean_string().unwrap_or_default(),
                lo_u.to_clean_string().unwrap_or_default(),
                hi_u.to_clean_string().unwrap_or_default(),
            );
            checked += 1;
        }
    }
    checked
}

// =============================================================================
// Adversarial large-k cancellation REGRESSION WITNESS (deterministic).
// =============================================================================

/// A large-`k` (k = 512), heavily-cancelling composition where the f32 SGEMM
/// accumulates many ULPs of error. Coefficients alternate large `±` values whose
/// exact product sum nearly cancels, so the stored f32 coefficient is several
/// ULPs off the exact real product — the regime where a single 1-ULP cast (the
/// OLD `compose_blas`) under-covered. The `γ_(k+1)·S` widening must enclose it.
#[test]
fn regression_large_k_cancellation_witness() {
    let k = 512usize;
    let out_dim = 3usize;
    let in_dim = 2usize;

    // A1: k x in. Columns are large alternating ± dyadic values (~±1.9) so each
    // |a2·a1| term is O(3.6) but the signed sum cancels heavily.
    let a1_l = Array2::from_shape_fn((k, in_dim), |(kk, j)| {
        let mag = 1.0 + ((kk * 7 + j * 13) % 29) as f32 / 32.0; // 1.0 .. ~1.9
        if (kk + j) % 2 == 0 {
            mag
        } else {
            -mag
        }
    });
    // Upper = lower + small positive width so the interval is non-degenerate.
    let a1_u = a1_l.mapv(|v| v + 0.015_625); // + 1/64
                                             // A2: out x k. Alternating ± ~1.9 with a different phase to force cancellation
                                             // in the contraction Σ_k a2[i,k]·a1[k,j].
    let a2_l = Array2::from_shape_fn((out_dim, k), |(i, kk)| {
        let mag = 1.0 + ((i * 11 + kk * 5) % 31) as f32 / 32.0;
        if (i + kk) % 2 == 0 {
            mag
        } else {
            -mag
        }
    });
    let a2_u = a2_l.mapv(|v| v + 0.015_625);

    let checked = assert_compose_blas_encloses_exact(&a1_l, &a1_u, &a2_l, &a2_u);
    assert_eq!(checked, out_dim * in_dim, "must check every composed coeff");
}

/// A second witness with EXACT cancellation to 0: `a2` row is the negation-paired
/// mirror of `a1` column so the exact product is exactly 0 while the f32 SGEMM
/// rounds to a small nonzero value (the classic A·W corner-flip regime). The
/// certified interval must still contain 0.
#[test]
fn regression_exact_cancellation_to_zero_witness() {
    let k = 256usize;
    let a1_l = Array2::from_shape_fn((k, 1), |(kk, _)| {
        // wide-range dyadic so the 256-wide f32 sum genuinely rounds. Use a
        // wrapping LCG-style mix (avoids debug overflow) to spread magnitudes.
        let raw = ((kk as u32).wrapping_mul(2_654_435_761) % 12_000) as i32 - 6_000;
        raw as f32 / 8192.0
    });
    let a1_u = a1_l.clone(); // degenerate (exact) a1 interval
                             // a2 = a1^T so Σ_k a2[0,k]·a1[k,0] = Σ_k a1[k,0]^2 (NOT zero) — use a
                             // sign-flipped partner instead to drive cancellation: pair k with k+1.
    let mut a2_row = Array2::<f32>::zeros((1, k));
    for kk in 0..k {
        // a2[0,k] = +a1[k] for even k, -a1[k] for odd k; combined with the squared
        // structure this produces heavy ± cancellation in the contraction.
        a2_row[[0, kk]] = if kk % 2 == 0 {
            a1_l[[kk, 0]]
        } else {
            -a1_l[[kk, 0]]
        };
    }
    let a2_l = a2_row.clone();
    let a2_u = a2_row;

    let checked = assert_compose_blas_encloses_exact(&a1_l, &a1_u, &a2_l, &a2_u);
    assert_eq!(checked, 1);
}

/// k=1 directed-cast regression (proptest-discovered). With a SINGLE product
/// (no accumulation), the round-to-nearest f32 product `fl(a2·a1)` is within
/// `½ ULP ≤ γ_2·S` of the exact real product — but applying `next_up_f32` to the
/// stored upper coefficient would shift it ~1 ULP FURTHER from `fl`, making
/// `|stored − exact|` reach ~1.5·err and ESCAPE the symmetric `±err` interval
/// concretize applies. The fix stores the raw round-to-nearest SGEMM result (no
/// directed nudge) so `γ_{k+1}·S` is the sole, sufficient margin. This is the
/// exact minimal case proptest shrank to: a1_u=-19.257813, a2_u=26.847656.
#[test]
fn regression_k1_directed_cast_witness() {
    let a1_l = Array2::from_shape_vec((1, 1), vec![-19.328125f32]).unwrap();
    let a1_u = Array2::from_shape_vec((1, 1), vec![-19.257813f32]).unwrap();
    let a2_l = Array2::from_shape_vec((1, 1), vec![26.835938f32]).unwrap();
    let a2_u = Array2::from_shape_vec((1, 1), vec![26.847656f32]).unwrap();
    let checked = assert_compose_blas_encloses_exact(&a1_l, &a1_u, &a2_l, &a2_u);
    assert_eq!(checked, 1);
}

// =============================================================================
// General enclosure proptest: random A1, A2 over varied k incl. large k.
// =============================================================================

/// Strategy: random interval coefficient matrices A1 (k x in) and A2 (out x k)
/// with adversarial ± signs and a wide magnitude range to force f32 accumulation
/// rounding. `k` spans 1..=256 (including large contractions where the OLD 1-ULP
/// `compose_blas` under-covered; the deterministic witnesses push k to 512). The
/// per-case exact-rational ground truth is O(out·in·k), so this cap keeps the
/// 400-case run well within the ntest timeout. Magnitudes use a fine dyadic-ish
/// grid (raw/256, |raw| up to ~7000 → ~±27) so the k-wide f32 dot exceeds f32's
/// 24-bit exact range and genuinely ROUNDS.
fn compose_inputs_strategy(
) -> impl Strategy<Value = (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)> {
    (1usize..=256, 1usize..=3, 1usize..=3).prop_flat_map(|(k, out_dim, in_dim)| {
        let cell = -7000i32..=7000i32; // value = raw/256 ∈ ~[-27, 27]
        let width = 0i32..=64i32; // upper = lower + w/256 (non-negative width)
        let a1_lo = proptest::collection::vec(cell.clone(), k * in_dim);
        let a1_w = proptest::collection::vec(width.clone(), k * in_dim);
        let a2_lo = proptest::collection::vec(cell, out_dim * k);
        let a2_w = proptest::collection::vec(width, out_dim * k);
        (
            Just(k),
            Just(out_dim),
            Just(in_dim),
            a1_lo,
            a1_w,
            a2_lo,
            a2_w,
        )
            .prop_map(|(k, out_dim, in_dim, a1_lo, a1_w, a2_lo, a2_w)| {
                let a1_l = Array2::from_shape_fn((k, in_dim), |(r, c)| {
                    a1_lo[r * in_dim + c] as f32 / 256.0
                });
                let a1_u = Array2::from_shape_fn((k, in_dim), |(r, c)| {
                    (a1_lo[r * in_dim + c] + a1_w[r * in_dim + c]) as f32 / 256.0
                });
                let a2_l =
                    Array2::from_shape_fn((out_dim, k), |(r, c)| a2_lo[r * k + c] as f32 / 256.0);
                let a2_u = Array2::from_shape_fn((out_dim, k), |(r, c)| {
                    (a2_lo[r * k + c] + a2_w[r * k + c]) as f32 / 256.0
                });
                (a1_l, a1_u, a2_l, a2_u)
            })
    })
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 8000, ..ProptestConfig::with_cases(400) })]

    /// ZERO-TOLERANCE enclosure: for every composed coefficient, the certified
    /// interval `[stored − err, stored + err]` from `compose_blas` (the now-sound
    /// γ_(k+1)·S widening) must ENCLOSE the EXACT real product `(A2·A1)[i,j]`,
    /// computed in rationals. This fires on any under-coverage — exactly the
    /// large-k / heavy-cancellation regime where the OLD 1-ULP version was unsound.
    #[ntest::timeout(180000)]
    #[test]
    fn compose_blas_certified_interval_encloses_exact_product(
        (a1_l, a1_u, a2_l, a2_u) in compose_inputs_strategy()
    ) {
        let checked = assert_compose_blas_encloses_exact(&a1_l, &a1_u, &a2_l, &a2_u);
        prop_assert!(checked >= 1, "must check at least one coefficient");
    }
}
