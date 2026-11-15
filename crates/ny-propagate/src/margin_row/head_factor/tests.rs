// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #head-factor stage 1 equivalence suite.
//!
//! These tests are the whole point of stage 1. The factored head seed is only
//! useful if it is EXACTLY the incumbent, because every later stage pushes the
//! product through a trunk op and any drift introduced here would be
//! indistinguishable from a bug in that push. So the assertions are on raw bit
//! patterns (`f64::to_bits`), not on an epsilon.

use ndarray::Array2;

use super::HeadFactor;
use crate::margin_row::net::axpy;
use crate::margin_row::rounding::{certify_up, gamma_n};

/// Deterministic, reproducible pseudo-random fill — no `rand` dependency and no
/// hidden global state, so a failure is replayable from the seed alone.
fn lcg_fill(n: usize, seed: u64, zero_every: usize) -> Vec<f64> {
    let mut x = seed | 1;
    (0..n)
        .map(|i| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if zero_every != 0 && i % zero_every == 0 {
                0.0
            } else {
                // Spread across a few orders of magnitude so the certified
                // rounding terms are actually exercised.
                let m = ((x >> 11) as f64) / ((1u64 << 53) as f64);
                (m - 0.5) * 8.0
            }
        })
        .collect()
}

fn fixture(n_y: usize, n_h: usize, r: usize, zero_every: usize) -> HeadFactor {
    let w1 = lcg_fill(n_y * n_h, 0x5eed_1234, zero_every);
    let s = Array2::from_shape_vec((n_y, r), lcg_fill(n_y * r, 0xabcd_9876, 0))
        .expect("spec block shape");
    HeadFactor { w1, s, n_h }
}

/// The incumbent seed loop, transcribed verbatim from
/// `engine.rs:513-525` (the `par_chunks_mut` form; each chunk `k` is
/// independent and its inner `j` loop is sequential, so a sequential
/// `chunks_mut` accumulates in the identical order).
fn incumbent_value(f: &HeadFactor) -> Array2<f64> {
    let r = f.ncols();
    let n_y = f.n_y();
    let mut l0 = Array2::<f64>::zeros((f.n_h, r));
    let ss = f.s.as_slice().expect("standard layout");
    let dst = l0.as_slice_mut().expect("standard layout");
    dst.chunks_mut(r).enumerate().for_each(|(k, acc)| {
        for j in 0..n_y {
            let w = f.w1[j * f.n_h + k];
            if w != 0.0 {
                axpy(acc, w, &ss[j * r..(j + 1) * r]);
            }
        }
    });
    l0
}

/// The incumbent certified-error loop, transcribed from `engine.rs:536-553`.
fn incumbent_error(f: &HeadFactor, combined: &Array2<f64>) -> Array2<f64> {
    let r = f.ncols();
    let n_y = f.n_y();
    let mut e0 = Array2::<f64>::zeros((f.n_h, r));
    let cs = combined.as_slice().expect("standard layout");
    let g2 = gamma_n(n_y + 8);
    let dst = e0.as_slice_mut().expect("standard layout");
    dst.chunks_mut(r).enumerate().for_each(|(k, acc)| {
        for j in 0..n_y {
            let w = f.w1[j * f.n_h + k].abs();
            if w != 0.0 {
                axpy(acc, w, &cs[j * r..(j + 1) * r]);
            }
        }
        for v in acc.iter_mut() {
            *v = certify_up(*v, g2);
        }
    });
    e0
}

fn assert_bitwise(label: &str, got: &Array2<f64>, want: &Array2<f64>) {
    assert_eq!(got.shape(), want.shape(), "{label}: shape");
    let mut diffs = 0usize;
    let mut first: Option<(usize, f64, f64)> = None;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g.to_bits() != w.to_bits() {
            diffs += 1;
            if first.is_none() {
                first = Some((i, *g, *w));
            }
        }
    }
    assert!(
        diffs == 0,
        "{label}: {diffs} of {} elements differ in BIT PATTERN; first at flat index {:?}. \
         The factored head seed must be the incumbent EXACTLY — an epsilon-equal \
         restatement is not good enough, because later stages push this product \
         through a trunk op and any drift here would be misattributed there.",
        got.len(),
        first,
    );
}

#[test]
fn factored_value_is_bitwise_identical_to_the_incumbent_dense_seed() {
    for &(n_y, n_h, r, z) in &[(10, 64, 4, 0), (100, 512, 20, 0), (8, 33, 7, 3)] {
        let f = fixture(n_y, n_h, r, z);
        assert_bitwise(
            &format!("value n_y={n_y} n_h={n_h} r={r} zero_every={z}"),
            &f.materialize_value(),
            &incumbent_value(&f),
        );
    }
}

#[test]
fn factored_error_is_bitwise_identical_to_the_incumbent_certified_seed() {
    for &(n_y, n_h, r, z) in &[(10, 64, 4, 0), (100, 512, 20, 0), (8, 33, 7, 3)] {
        let f = fixture(n_y, n_h, r, z);
        // A stand-in for `combined_err(S, Es, gamma_n(n_y+2))`: any nonnegative
        // block exercises the same accumulation and certification path.
        let combined = Array2::from_shape_vec(
            (n_y, r),
            lcg_fill(n_y * r, 0x1111_2222, 0)
                .into_iter()
                .map(f64::abs)
                .collect(),
        )
        .expect("combined shape");
        assert_bitwise(
            &format!("error n_y={n_y} n_h={n_h} r={r} zero_every={z}"),
            &f.materialize_error(&combined),
            &incumbent_error(&f, &combined),
        );
    }
}

#[test]
fn materialized_error_is_never_below_the_raw_accumulation() {
    // certify_up must only ever widen: the published outward envelope may not
    // shrink as a side effect of the factored restatement.
    let f = fixture(16, 48, 5, 0);
    let combined = Array2::from_shape_vec(
        (16, 5),
        lcg_fill(16 * 5, 0x3333_4444, 0)
            .into_iter()
            .map(f64::abs)
            .collect(),
    )
    .expect("combined shape");
    let certified = f.materialize_error(&combined);
    let mut raw = Array2::<f64>::zeros((f.n_h, f.ncols()));
    {
        let cs = combined.as_slice().expect("standard layout");
        let r = f.ncols();
        let dst = raw.as_slice_mut().expect("standard layout");
        for (k, acc) in dst.chunks_mut(r).enumerate() {
            for j in 0..f.n_y() {
                let w = f.w1[j * f.n_h + k].abs();
                if w != 0.0 {
                    axpy(acc, w, &cs[j * r..(j + 1) * r]);
                }
            }
        }
    }
    for (c, r0) in certified.iter().zip(raw.iter()) {
        assert!(
            *c >= *r0,
            "certified error {c} is BELOW the raw accumulation {r0} — the outward \
             envelope must never tighten as a side effect of factoring"
        );
    }
}

#[test]
fn live_hidden_support_counts_only_columns_with_a_nonzero_weight() {
    let n_y = 4;
    let n_h = 6;
    // Zero out hidden columns 1 and 4 across every spec row.
    let mut w1 = vec![1.0f64; n_y * n_h];
    for j in 0..n_y {
        w1[j * n_h + 1] = 0.0;
        w1[j * n_h + 4] = 0.0;
    }
    let f = HeadFactor {
        w1,
        s: Array2::<f64>::zeros((n_y, 3)),
        n_h,
    };
    assert_eq!(
        f.live_hidden_support(),
        4,
        "two of six hidden columns are identically zero across all spec rows, so \
         only four are live"
    );
    assert_eq!(f.nrows(), n_h);
    assert_eq!(f.ncols(), 3);
}

#[test]
fn a_fully_dense_head_has_no_support_compression() {
    // The honest negative: if the head Gemm is dense (the usual classifier
    // head), the factored form compresses NOTHING at this seam, which is
    // exactly why the previous patches attempt was refuted. This test pins that
    // fact so a later stage cannot quietly assume otherwise.
    let f = fixture(100, 512, 20, 0);
    assert_eq!(
        f.live_hidden_support(),
        512,
        "a dense head Gemm touches every hidden unit; compression at the SEED is \
         1x by construction and any gain must come from pushing the product \
         through a trunk op instead"
    );
}

#[test]
fn head_factor_ships_dark() {
    assert!(
        !super::head_factor_armed(),
        "stage 1 must be unreachable: no production caller, no yaml, no env"
    );
}
