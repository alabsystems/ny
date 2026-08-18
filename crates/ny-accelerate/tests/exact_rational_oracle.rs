// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The CI half of the conformance obligation (probe spec, "CI (not runtime)").
//!
//! Two things the ~0.4 ms runtime probe deliberately cannot afford:
//!
//! 1. **An EXACT-RATIONAL envelope check.** Every other soundness test in this
//!    crate compares against a double-double reference. That reference is very
//!    good (error ~2⁻¹⁰⁴·S) but it is still floating point. Here the comparison
//!    is done in `BigRational` — infinite precision, no reference error at all —
//!    so the `γ_k·S` claim is checked against the true real number. This also
//!    validates the double-double reference itself, which is what licenses its
//!    use in the cheaper tests.
//!
//! 2. **A LARGE-N non-mixing check.** The runtime probe caps its
//!    Strassen/Winograd detector at N=128 for cost. Only CI can economically
//!    exclude a large-N-only Strassen cutoff, which is the exact shape of defect
//!    that would break the per-entry `S`-based error structure while leaving
//!    every small shape perfect.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use ny_accelerate::AccelerateGemmEngine;
use ny_core::GemmEngine;

fn engine() -> AccelerateGemmEngine {
    AccelerateGemmEngine::new_with_gates(true, false).expect("engine")
}

/// Exact rational value of an f64. Panics only on NaN/inf, which the generators
/// below never produce.
fn exact(x: f64) -> BigRational {
    BigRational::from_float(x).expect("finite f64 has an exact rational value")
}

/// `γ_k = k·2⁻⁵³ / (1 − k·2⁻⁵³)` as an exact rational.
fn gamma_exact(k: usize) -> BigRational {
    let two53 = BigInt::from(1u64) << 53u32;
    let ku = BigRational::new(BigInt::from(k), two53);
    let one = BigRational::one();
    assert!(ku < one, "gamma_k is degenerate for k={k}");
    &ku / (one - &ku)
}

#[test]
fn gamma_envelope_holds_against_an_exact_rational_oracle() {
    let eng = engine();
    // Small enough for BigRational to stay quick, adversarial enough to matter:
    // forced cancelling pairs plus a 2^±200 dynamic range, which is the case
    // shape the qualification oracle used.
    // 8x512x8 = 32768 MACs — at the engine's offload floor, and the exact case
    // shape the qualification oracle used.
    let (m, k, n) = (8usize, 512usize, 8usize);
    let mut s: u64 = 0x0A0B_0C0D_0E0F_1011;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let e = ((s >> 40) % 400) as i32 - 200;
        let mant = ((s >> 12) & ((1u64 << 52) - 1)) as f64 / (1u64 << 52) as f64;
        let sign = if (s >> 3) & 1 == 1 { -1.0 } else { 1.0 };
        sign * (1.0 + mant) * f64::powi(2.0, e)
    };
    let mut a: Vec<f64> = (0..m * k).map(|_| next()).collect();
    let b: Vec<f64> = (0..k * n).map(|_| next()).collect();
    // Force catastrophic cancellation.
    for i in 0..m {
        let mut l = 0;
        while l + 1 < k {
            a[i * k + l + 1] = -a[i * k + l];
            l += 2;
        }
    }

    let got = eng.gemm_f64(m, k, n, &a, &b).expect("gemm");
    let gamma = gamma_exact(k);
    let mut worst = BigRational::zero();
    for i in 0..m {
        for j in 0..n {
            let mut dot = BigRational::zero();
            let mut sabs = BigRational::zero();
            for l in 0..k {
                let x = exact(a[i * k + l]);
                let y = exact(b[l * n + j]);
                let p = &x * &y;
                sabs += p.abs();
                dot += p;
            }
            let err = (exact(got[i * n + j]) - &dot).abs();
            let bound = &gamma * &sabs;
            assert!(
                err <= bound,
                "EXACT-RATIONAL gamma_k*S VIOLATION at ({i},{j}): the published \
                 enclosure would not contain the true product"
            );
            if !bound.is_zero() {
                let ratio = &err / &bound;
                if ratio > worst {
                    worst = ratio;
                }
            }
        }
    }
    let worst_f = worst
        .to_string()
        .parse::<f64>()
        .unwrap_or_else(|_| ratio_to_f64(&worst));
    println!("exact-rational oracle: worst |err| / (gamma_k*S) = {worst_f:.9}");
    assert!(worst <= BigRational::one());
}

fn ratio_to_f64(r: &BigRational) -> f64 {
    let num = r.numer().to_string().parse::<f64>().unwrap_or(f64::NAN);
    let den = r.denom().to_string().parse::<f64>().unwrap_or(f64::NAN);
    num / den
}

/// The double-double reference used by the cheap tests must itself be inside a
/// far tighter envelope than the thing it is used to certify. Checked exactly.
#[test]
fn the_double_double_reference_is_far_tighter_than_the_gamma_envelope() {
    let k = 256usize;
    let mut s: u64 = 0xDEAD_BEEF_CAFE_F00D;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let e = ((s >> 40) % 200) as i32 - 100;
        let mant = ((s >> 12) & ((1u64 << 52) - 1)) as f64 / (1u64 << 52) as f64;
        let sign = if (s >> 3) & 1 == 1 { -1.0 } else { 1.0 };
        sign * (1.0 + mant) * f64::powi(2.0, e)
    };
    let a: Vec<f64> = (0..k).map(|_| next()).collect();
    let mut b: Vec<f64> = (0..k).map(|_| next()).collect();
    for l in (0..k - 1).step_by(2) {
        b[l + 1] = -b[l] * (a[l] / a[l + 1]); // near-total cancellation
    }

    // double-double reference (the one `engine_soundness.rs` uses)
    let (mut hi, mut lo, mut comp) = (0.0f64, 0.0f64, 0.0f64);
    let add = |x: f64, hi: &mut f64, lo: &mut f64, comp: &mut f64| {
        let two_sum = |a: f64, b: f64| {
            let s = a + b;
            let bb = s - a;
            (s, (a - (s - bb)) + (b - bb))
        };
        let (h, e) = two_sum(*hi, x);
        *hi = h;
        let (l, e2) = two_sum(*lo, e);
        *lo = l;
        *comp += e2;
    };
    let mut sabs_f = 0.0f64;
    for l in 0..k {
        let p = a[l] * b[l];
        let e = a[l].mul_add(b[l], -p);
        add(p, &mut hi, &mut lo, &mut comp);
        add(e, &mut hi, &mut lo, &mut comp);
        sabs_f += a[l].abs() * b[l].abs();
    }
    let dd = hi + (lo + comp);

    // exact truth
    let mut dot = BigRational::zero();
    let mut sabs = BigRational::zero();
    for l in 0..k {
        let p = exact(a[l]) * exact(b[l]);
        sabs += p.abs();
        dot += p;
    }
    let dd_err = (exact(dd) - &dot).abs();
    let gamma = gamma_exact(k);
    let envelope = &gamma * &sabs;
    println!(
        "double-double reference error / (gamma_k*S) = {:.3e}  (S = {sabs_f:e})",
        ratio_to_f64(&(&dd_err / &envelope))
    );
    // 2^-40 is a huge margin over the ~2^-51 expected ratio, and still proves
    // the reference is not doing the certifying work in the cheaper tests.
    let margin = BigRational::new(BigInt::one(), BigInt::from(1u64) << 40u32);
    assert!(
        dd_err <= envelope * margin,
        "the double-double reference is not tight enough to certify with"
    );
}

/// LARGE-N non-mixing: the runtime probe stops at N=128. A Strassen/Winograd
/// cutoff that only engages for large N would leave the per-entry `S`-based
/// error structure invalid exactly where the folds are biggest.
#[test]
fn no_block_mixing_at_large_n() {
    for &n in &[256usize, 1024, 2048] {
        let mut s: u64 = 0xB10C_5EED_0000_0001u64.wrapping_add(n as u64);
        let mut next_int = |lo: i64, hi: i64| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lo + ((s >> 33) % ((hi - lo + 1) as u64)) as i64
        };
        // 1e150-scale everywhere, except the last row of A / last col of B,
        // which carry small exact integers whose dot product must come back
        // BIT-EXACT despite being surrounded by values 10^150 times larger.
        let mut a: Vec<f64> = (0..n * n)
            .map(|_| next_int(-1000, 1000) as f64 * 1e147)
            .collect();
        let mut b: Vec<f64> = (0..n * n)
            .map(|_| next_int(-1000, 1000) as f64 * 1e147)
            .collect();
        let mut acc: i64 = 0;
        for l in 0..n {
            // |ra|,|cb| <= 1023 and n <= 2048 => |acc| <= 2048*1023^2 < 2^32:
            // exact in f64 for EVERY summation order.
            let ra = next_int(-1023, 1023);
            let cb = next_int(-1023, 1023);
            a[(n - 1) * n + l] = ra as f64;
            b[l * n + (n - 1)] = cb as f64;
            acc += ra * cb;
        }
        a[3 * n + 5] = f64::NAN;
        b[7 * n + 11] = f64::INFINITY;

        // The engine's G2/non-finite guard would (correctly) decline NaN input,
        // so this check goes through the probe's kernel hook, which is the same
        // `cblas_dgemm` call the engine makes.
        let mut c = vec![f64::NAN; n * n];
        let ran = ny_accelerate::probe::accelerate_dgemm(n, n, n, &a, &b, &mut c);
        assert!(ran, "dgemm refused N={n}");

        let mut leak = 0usize;
        for i in 0..n {
            for j in 0..n {
                if i != 3 && j != 11 && !c[i * n + j].is_finite() {
                    leak += 1;
                }
            }
        }
        assert_eq!(leak, 0, "NaN/Inf leaked outside its row/col at N={n}");
        assert_eq!(
            c[(n - 1) * n + (n - 1)].to_bits(),
            (acc as f64).to_bits(),
            "tiny exact entry was contaminated by 1e150 neighbours at N={n}"
        );
        println!("N={n}: no block mixing, tiny exact entry bit-exact");
    }
}
