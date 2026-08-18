// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness tests for [`AccelerateGemmEngine`].
//!
//! The claim under test is EXACTLY the one NY's CROWN certificate needs, and
//! nothing stronger:
//!
//! ```text
//! |Ĉ_ij − Σ_l a_il·b_lj| ≤ γ_k · Σ_l |a_il|·|b_lj|,  γ_k = k·2⁻⁵³/(1 − k·2⁻⁵³)
//! ```
//!
//! Note what is deliberately NOT asserted: bit-equality with faer, or with any
//! particular summation order. Accelerate is a different (legal) order, so the
//! last ulps differ; the certificate is order-independent, which is the whole
//! reason the substitution is admissible. The differential test below therefore
//! asserts that BOTH engines' enclosures are valid, and reports the size of the
//! disagreement rather than forbidding it.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use ny_accelerate::AccelerateGemmEngine;
use ny_core::{GemmEngine, NyError};

// ---------------------------------------------------------------------------
// Exact-ish reference: products via FMA (exact), Neumaier/double-double
// accumulation. Its own error is <= ~2^-104 * S, i.e. ~2^-51 times smaller than
// the gamma_k*S allowance being tested, so it is a legitimate stand-in for the
// exact rational value at these k. (The C prototype of this reference was
// verified bit-exact against Python `fractions.Fraction` during qualification.)
// ---------------------------------------------------------------------------

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

struct Acc3 {
    hi: f64,
    lo: f64,
    comp: f64,
}

impl Acc3 {
    fn new() -> Self {
        Self {
            hi: 0.0,
            lo: 0.0,
            comp: 0.0,
        }
    }
    fn add(&mut self, x: f64) {
        let (h, e) = two_sum(self.hi, x);
        self.hi = h;
        let (l, e2) = two_sum(self.lo, e);
        self.lo = l;
        self.comp += e2;
    }
}

/// Returns `(reference_dot, S = Σ|a||b|)`.
fn ref_dot(
    a: &[f64],
    ia: usize,
    sa: usize,
    b: &[f64],
    ib: usize,
    sb: usize,
    k: usize,
) -> (f64, f64) {
    let mut acc = Acc3::new();
    let mut s = 0.0f64;
    for l in 0..k {
        let x = a[ia + l * sa];
        let y = b[ib + l * sb];
        let p = x * y;
        let e = x.mul_add(y, -p); // exact residual of the product
        acc.add(p);
        acc.add(e);
        s += x.abs() * y.abs();
    }
    (acc.hi + (acc.lo + acc.comp), s)
}

fn gamma_f64(k: usize) -> f64 {
    let ku = (k as f64) * f64::powi(2.0, -53);
    ku / (1.0 - ku)
}

/// Adversarial stream: exponents spanning 2^±200 with random signs, so that
/// cancellation drives the accumulation error toward the Higham worst case
/// instead of averaging it away.
struct Adversarial(u64);

impl Adversarial {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let e = ((self.0 >> 40) % 400) as i32 - 200;
        let mant = ((self.0 >> 12) & ((1u64 << 52) - 1)) as f64 / (1u64 << 52) as f64;
        let sign = if (self.0 >> 3) & 1 == 1 { -1.0 } else { 1.0 };
        sign * (1.0 + mant) * f64::powi(2.0, e)
    }
}

fn engine() -> AccelerateGemmEngine {
    AccelerateGemmEngine::new_with_gates(true, true)
        .expect("Accelerate engine must construct on this host")
}

fn faer_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let am = faer::MatRef::from_row_major_slice(a, m, k);
    let bm = faer::MatRef::from_row_major_slice(b, k, n);
    let mut c = faer::Mat::<f64>::zeros(m, n);
    faer::linalg::matmul::matmul(&mut c, faer::Accum::Replace, am, bm, 1.0, faer::Par::Seq);
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = c[(i, j)];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// THE CERTIFICATE
// ---------------------------------------------------------------------------

#[test]
fn gemm_f64_stays_inside_the_gamma_envelope_under_forced_cancellation() {
    let eng = engine();
    let mut worst_ratio = 0.0f64;
    for &(m, k, n) in &[
        (32usize, 128usize, 32usize),
        (16, 512, 16),
        (8, 2048, 8),
        (64, 64, 64),
    ] {
        let mut rng = Adversarial::new(0xACCE_1E7A ^ (k as u64));
        let mut a: Vec<f64> = (0..m * k).map(|_| rng.next()).collect();
        let b: Vec<f64> = (0..k * n).map(|_| rng.next()).collect();
        // Force catastrophic cancellation: half of A's entries are the exact
        // negation of the entry that pairs with the same B row two columns over.
        for i in 0..m {
            let mut l = 0;
            while l + 1 < k {
                a[i * k + l + 1] = -a[i * k + l];
                l += 2;
            }
        }
        let got = eng.gemm_f64(m, k, n, &a, &b).expect("engine gemm_f64");
        assert_eq!(got.len(), m * n);
        let gamma = gamma_f64(k);
        for i in 0..m {
            for j in 0..n {
                let (exact, s) = ref_dot(&a, i * k, 1, &b, j, n, k);
                let err = (got[i * n + j] - exact).abs();
                let bound = gamma * s;
                assert!(
                    err <= bound,
                    "gamma_k*S VIOLATED at ({i},{j}) m={m} k={k} n={n}: err={err:e} > {bound:e}"
                );
                if bound > 0.0 {
                    worst_ratio = worst_ratio.max(err / bound);
                }
            }
        }
    }
    println!("worst |err| / (gamma_k*S) over all shapes = {worst_ratio:.6}");
    assert!(worst_ratio <= 1.0);
}

#[test]
fn gemm_f64_is_bit_exact_on_order_free_integer_products() {
    // Every product and partial sum exactly representable => every summation
    // order returns the identical f64, so bit-equality is a legitimate
    // assertion here (and ONLY here).
    let eng = engine();
    for &(m, k, n) in &[(32usize, 128usize, 32usize), (8, 4096, 8), (64, 64, 64)] {
        let mut s: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) % 2047) as i64 - 1023
        };
        let ai: Vec<i64> = (0..m * k).map(|_| next()).collect();
        let bi: Vec<i64> = (0..k * n).map(|_| next()).collect();
        let a: Vec<f64> = ai.iter().map(|&v| v as f64).collect();
        let b: Vec<f64> = bi.iter().map(|&v| v as f64).collect();
        let got = eng.gemm_f64(m, k, n, &a, &b).expect("engine gemm_f64");
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i64;
                for l in 0..k {
                    acc += ai[i * k + l] * bi[l * n + j];
                }
                assert_eq!(
                    got[i * n + j].to_bits(),
                    (acc as f64).to_bits(),
                    "exact-integer product differed at ({i},{j}) for {m}x{k}x{n}"
                );
            }
        }
    }
}

/// The CROWN seam's real domain: operands are `f32 -> f64` widenings, so the G2
/// underflow regime is unreachable by construction. Assert (a) the guard never
/// fires there, and (b) the published enclosure `[Ĉ − γS, Ĉ + γS]` contains the
/// exact product — which is the property a verdict depends on.
#[test]
fn crown_seam_domain_enclosure_contains_the_exact_product() {
    let eng = engine();
    let (m, k, p) = (64usize, 512usize, 64usize);
    let mut s: u64 = 0xC0FF_EE00_1234_5678;
    let mut next32 = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let e = ((s >> 40) % 30) as i32 - 15;
        let mant = ((s >> 12) & 0x7f_ffff) as f32 / (1u32 << 23) as f32;
        let sign = if (s >> 3) & 1 == 1 { -1.0f32 } else { 1.0 };
        sign * (1.0 + mant) * f32::powi(2.0, e)
    };
    let a32: Vec<f32> = (0..m * k).map(|_| next32()).collect();
    let w32: Vec<f32> = (0..k * p).map(|_| next32()).collect();
    // Exactly what `aw_via_engine` does: widen, then two f64 GEMMs.
    let a64: Vec<f64> = a32.iter().map(|&v| f64::from(v)).collect();
    let w64: Vec<f64> = w32.iter().map(|&v| f64::from(v)).collect();
    let absa: Vec<f64> = a64.iter().map(|v| v.abs()).collect();
    let absw: Vec<f64> = w64.iter().map(|v| v.abs()).collect();

    let aw = eng.gemm_f64(m, k, p, &a64, &w64).expect("A*W");
    let sabs = eng.gemm_f64(m, k, p, &absa, &absw).expect("|A|*|W|");

    let gamma = gamma_f64(k);
    for i in 0..m {
        for j in 0..p {
            let (exact, _) = ref_dot(&a64, i * k, 1, &w64, j, p, k);
            // S as the engine itself computed it, inflated to a proven
            // over-bound the way `aw_f64_with_abssum` does.
            let s_hat = sabs[i * p + j] * (1.0 + gamma);
            let eps = gamma * s_hat;
            let lo = aw[i * p + j] - eps;
            let hi = aw[i * p + j] + eps;
            assert!(
                lo <= exact && exact <= hi,
                "enclosure did NOT contain the exact product at ({i},{j}): \
                 [{lo:e}, {hi:e}] vs {exact:e}"
            );
        }
    }
}

/// Differential vs the incumbent faer engine. Both are legal summation orders;
/// the test proves BOTH enclosures are valid and reports the disagreement, it
/// does not demand bit-equality (which would be false, and demanding it would
/// be demanding the wrong thing).
#[test]
fn accelerate_and_faer_enclosures_are_both_valid() {
    let eng = engine();
    let (m, k, n) = (32usize, 256usize, 32usize);
    let mut rng = Adversarial::new(0xFEED_BEEF);
    let a: Vec<f64> = (0..m * k).map(|_| rng.next()).collect();
    let b: Vec<f64> = (0..k * n).map(|_| rng.next()).collect();
    let acc = eng.gemm_f64(m, k, n, &a, &b).expect("accelerate");
    let fae = faer_f64(m, k, n, &a, &b);
    let gamma = gamma_f64(k);
    let mut bitidentical = 0usize;
    let mut worst_gap_ratio = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let (exact, s) = ref_dot(&a, i * k, 1, &b, j, n, k);
            let bound = gamma * s;
            let ea = (acc[i * n + j] - exact).abs();
            let ef = (fae[i * n + j] - exact).abs();
            assert!(ea <= bound, "accelerate outside gamma_k*S at ({i},{j})");
            assert!(ef <= bound, "faer outside gamma_k*S at ({i},{j})");
            if acc[i * n + j].to_bits() == fae[i * n + j].to_bits() {
                bitidentical += 1;
            }
            if bound > 0.0 {
                worst_gap_ratio =
                    worst_gap_ratio.max((acc[i * n + j] - fae[i * n + j]).abs() / bound);
            }
        }
    }
    println!(
        "accelerate vs faer: {bitidentical}/{} entries bit-identical, worst |Δ|/(gamma_k*S) = {worst_gap_ratio:.6}",
        m * n
    );
    assert!(
        worst_gap_ratio <= 2.0,
        "the two engines disagree by more than both certificates allow"
    );
}

// ---------------------------------------------------------------------------
// FAIL-CLOSED SURFACE
// ---------------------------------------------------------------------------

#[test]
fn g2_declines_operands_that_could_produce_a_subnormal_product() {
    let eng = engine();
    let (m, k, n) = (32usize, 128usize, 32usize);
    let a = vec![1e-300f64; m * k];
    let b = vec![1e-300f64; k * n];
    assert!(
        matches!(
            eng.gemm_f64(m, k, n, &a, &b),
            Err(NyError::UnsupportedOp(_))
        ),
        "G2 must decline the one regime gamma_n*S cannot cover"
    );
    // The same shape with normal magnitudes is accepted.
    let a = vec![1.5f64; m * k];
    let b = vec![0.25f64; k * n];
    assert!(eng.gemm_f64(m, k, n, &a, &b).is_ok());
}

#[test]
fn declines_non_finite_operands_instead_of_propagating_them() {
    let eng = engine();
    let (m, k, n) = (32usize, 128usize, 32usize);
    let mut a = vec![1.0f64; m * k];
    a[7] = f64::NAN;
    let b = vec![1.0f64; k * n];
    assert!(matches!(
        eng.gemm_f64(m, k, n, &a, &b),
        Err(NyError::UnsupportedOp(_))
    ));
    a[7] = f64::INFINITY;
    assert!(matches!(
        eng.gemm_f64(m, k, n, &a, &b),
        Err(NyError::UnsupportedOp(_))
    ));
}

#[test]
fn g1_declines_bad_shapes_and_lengths_without_panicking() {
    let eng = engine();
    // length mismatch
    assert!(matches!(
        eng.gemm_f64(2, 3, 4, &[0.0; 5], &[0.0; 12]),
        Err(NyError::InvalidSpec(_))
    ));
    // usize overflow
    let huge = 1usize << (usize::BITS - 1);
    assert!(eng.gemm_f64(huge, huge, huge, &[], &[]).is_err());
    assert!(eng.gemm_f32(huge, huge, huge, &[], &[]).is_err());
    // beyond the LP64 i32 ABI: must DECLINE, never truncate. (No allocation is
    // attempted because the MAC floor / length check rejects first.)
    let beyond = usize::try_from(i32::MAX).unwrap() + 1;
    assert!(eng.gemm_f64(1, beyond, 1, &[], &[]).is_err());
}

#[test]
fn empty_contraction_returns_the_zero_matrix() {
    let eng = engine();
    assert_eq!(eng.gemm_f64(3, 0, 4, &[], &[]).unwrap(), vec![0.0f64; 12]);
    assert_eq!(eng.gemm_f32(3, 0, 4, &[], &[]).unwrap(), vec![0.0f32; 12]);
    assert!(eng.gemm_f64(0, 5, 4, &[], &[0.0; 20]).unwrap().is_empty());
}

#[test]
fn small_products_decline_so_the_caller_keeps_faer() {
    let eng = engine();
    let (m, k, n) = (4usize, 8usize, 4usize); // 128 MACs, far below the floor
    let a = vec![1.0f64; m * k];
    let b = vec![1.0f64; k * n];
    assert!(matches!(
        eng.gemm_f64(m, k, n, &a, &b),
        Err(NyError::UnsupportedOp(_))
    ));
}

#[test]
fn unarmed_engine_refuses_f64_and_keeps_f32_on_faer() {
    let eng = AccelerateGemmEngine::new_with_gates(false, true)
        .expect("f32-only gate should still construct");
    assert!(matches!(
        eng.gemm_f64(32, 128, 32, &vec![1.0; 4096], &vec![1.0; 4096]),
        Err(NyError::UnsupportedOp(_))
    ));
    assert!(AccelerateGemmEngine::new_with_gates(false, false).is_none());
}

/// KA-9 as an ENGINE-level invariant: after real engine traffic the calling
/// thread's FPCR is still clean and subnormal arithmetic is still gradual. A
/// dgemm that set `FPCR.FZ` and forgot to restore it would silently break NY's
/// error-free-transformation primitives, which require no-FTZ.
#[test]
fn engine_use_leaves_gradual_underflow_intact_on_the_calling_thread() {
    let eng = engine();
    let tiny = f64::from_bits(1); // 2^-1074
    assert_ne!(tiny * 1.0, 0.0);
    let a = vec![1.5f64; 32 * 128];
    let b = vec![0.5f64; 128 * 32];
    let _ = eng.gemm_f64(32, 128, 32, &a, &b).expect("gemm");
    let half_min_normal = f64::MIN_POSITIVE / 2.0;
    assert_ne!(
        half_min_normal, 0.0,
        "MIN_POSITIVE/2 flushed to zero: FPCR.FZ was left set"
    );
    assert_eq!(half_min_normal.to_bits(), 1u64 << 51);
    assert_ne!(tiny * 1.0, 0.0, "smallest subnormal flushed after the call");
}

/// The engine object must be safe to share across rayon workers (it is
/// installed as a process-global `Arc<dyn GemmEngine>`).
#[test]
fn engine_is_deterministic_across_concurrent_threads() {
    use std::sync::Arc;
    let eng = Arc::new(engine());
    let (m, k, n) = (32usize, 128usize, 32usize);
    let mut rng = Adversarial::new(0x5EED);
    let a: Arc<Vec<f64>> = Arc::new((0..m * k).map(|_| rng.next()).collect());
    let b: Arc<Vec<f64>> = Arc::new((0..k * n).map(|_| rng.next()).collect());
    let reference = eng.gemm_f64(m, k, n, &a, &b).expect("reference");
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let (eng, a, b) = (Arc::clone(&eng), Arc::clone(&a), Arc::clone(&b));
            std::thread::spawn(move || {
                (0..20)
                    .map(|_| eng.gemm_f64(m, k, n, &a, &b).expect("gemm"))
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    for h in handles {
        for got in h.join().expect("worker panicked") {
            assert!(
                got.iter()
                    .zip(&reference)
                    .all(|(g, r)| g.to_bits() == r.to_bits()),
                "concurrent dgemm was not bit-reproducible"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// f32 free-rider (non-verdict traffic)
// ---------------------------------------------------------------------------

#[test]
fn gemm_f32_stays_inside_the_f32_gamma_envelope() {
    const U: f64 = 5.960_464_477_539_063e-8; // 2^-24
    for f32_armed in [false, true] {
        let eng = AccelerateGemmEngine::new_with_gates(true, f32_armed).expect("engine");
        assert_eq!(eng.f32_via_accelerate(), f32_armed);
        let (m, k, n) = (32usize, 256usize, 32usize);
        let mut s: u64 = 0xF1_F32;
        let mut next = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let e = ((s >> 40) % 30) as i32 - 15;
            let mant = ((s >> 12) & 0x7f_ffff) as f32 / (1u32 << 23) as f32;
            let sign = if (s >> 3) & 1 == 1 { -1.0f32 } else { 1.0 };
            sign * (1.0 + mant) * f32::powi(2.0, e)
        };
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| next()).collect();
        let got = eng.gemm_f32(m, k, n, &a, &b).expect("gemm_f32");
        let ku = (k as f64) * U;
        let gamma = ku / (1.0 - ku);
        for i in 0..m {
            for j in 0..n {
                let mut dot = 0.0f64;
                let mut sabs = 0.0f64;
                for l in 0..k {
                    let x = f64::from(a[i * k + l]);
                    let y = f64::from(b[l * n + j]);
                    dot += x * y;
                    sabs += x.abs() * y.abs();
                }
                let err = (f64::from(got[i * n + j]) - dot).abs();
                assert!(
                    err <= gamma * sabs * (1.0 + 1e-6),
                    "gemm_f32 outside the f32 envelope at ({i},{j}), f32_armed={f32_armed}"
                );
            }
        }
    }
}

#[test]
fn telemetry_reports_what_the_seam_actually_did() {
    let before = ny_accelerate::telemetry();
    let eng = engine();
    let a = vec![1.5f64; 32 * 128];
    let b = vec![0.5f64; 128 * 32];
    let _ = eng.gemm_f64(32, 128, 32, &a, &b).expect("gemm");
    let after = ny_accelerate::telemetry();
    assert!(after.f64_calls > before.f64_calls);
    println!("{after:?}");
    println!("{}", eng.install_summary());
    let (image, symbol) = eng.symbol_provenance().expect("dladdr provenance");
    println!("G3 provenance: symbol={symbol} image={image}");
    assert!(image.contains("BLAS") || image.contains("Accelerate") || image.contains("veclib"));
}
