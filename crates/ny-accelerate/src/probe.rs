// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Runtime conformance probe for Apple Accelerate's `cblas_dgemm` / `cblas_sgemm`.
//!
//! Run ONCE per process, inside [`crate::AccelerateGemmEngine::new`]. On ANY
//! failure the constructor returns `None`, so `install_*_if_armed` never
//! publishes a factory and the incumbent faer engine keeps the seam. Fail-closed
//! by construction: a probe that cannot run is a probe that failed.
//!
//! # What the probe is testing
//!
//! NOT "the result equals some particular summation order" — that form is both
//! too strong (an FMA kernel violates it verbatim while being strictly *more*
//! accurate) and untestable against an opaque vendor kernel. The probe tests the
//! ERROR ENVELOPE the CROWN certificate actually needs,
//!
//! ```text
//! |Ĉ_ij − Σ_l a_il·b_lj| ≤ γ_k · Σ_l |a_il|·|b_lj|,  γ_k = k·2⁻⁵³/(1 − k·2⁻⁵³)
//! ```
//!
//! which holds for any conventional inner product (exact real products, each
//! rounded OR fused, combined by any binary summation tree, round-to-nearest,
//! no underflow), and the *hazards that would break it*: reduced precision,
//! flush-to-zero, denormals-are-zero, a non-RN rounding mode, saturating
//! overflow, Strassen/Winograd block mixing, `beta=0` reading `C`, and FPCR
//! corruption.
//!
//! # THE DESIGN RULE (learned by measurement — do not weaken it)
//!
//! Exact-bit assertions are made ONLY on
//!
//! * order-INDEPENDENT vectors — every product and every partial sum exactly
//!   representable, so all summation orders yield the identical f64; or
//! * `k ≤ 2`, where all orders coincide.
//!
//! MEASURED: Accelerate switches from a sequential to a multi-accumulator
//! strided reduction at exactly `k ≥ 4`. A k=512 tie vector returned `1+2⁻⁵²`
//! where the sequential order gives `1+2⁻⁵¹` — both are legal orders (error
//! 2⁻⁵³ against an allowance of 5.7e−14). A probe that bit-asserts an
//! order-DEPENDENT vector at `k ≥ 4` WILL fail on a perfectly sound library and
//! take the engine offline.

use crate::ffi;

/// A single named conformance check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Check {
    /// Stable id, e.g. `"KA-5d"`.
    pub id: &'static str,
    /// What the check proves.
    pub what: &'static str,
    /// Whether the host's BLAS satisfied it.
    pub passed: bool,
}

/// Outcome of one probe run.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProbeReport {
    /// Every check, in execution order.
    pub checks: Vec<Check>,
}

impl ProbeReport {
    /// ACCEPT iff every check passed and at least one ran.
    #[must_use]
    pub fn accepted(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(|c| c.passed)
    }

    /// Ids of the checks that failed (empty on ACCEPT).
    #[must_use]
    pub fn failures(&self) -> Vec<&'static str> {
        self.checks
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.id)
            .collect()
    }

    fn check(&mut self, id: &'static str, what: &'static str, passed: bool) {
        self.checks.push(Check { id, what, passed });
    }
}

/// Fixed LCG from the probe spec — no data tables, so the probe is a few
/// hundred bytes of code rather than a megabyte of vectors.
struct Lcg(u64);

impl Lcg {
    fn new() -> Self {
        Self(0x9E37_79B9_7F4A_7C15)
    }

    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let span = (hi - lo + 1) as u64;
        lo + ((self.0 >> 33) % span) as i64
    }
}

#[inline]
fn biteq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

#[inline]
fn biteq32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits()
}

/// `2^e` as an exact f64, for `-1074 <= e <= 1023` (0.0 below that).
///
/// Built from the bit pattern rather than `powi`: `f64::powi(2.0, -1074)`
/// evaluates `1/2^1074`, and `2^1074` overflows to infinity, so it returns 0.0
/// — which would have made the entire subnormal battery vacuously "pass".
fn exp2(e: i32) -> f64 {
    if e > 1023 {
        f64::INFINITY
    } else if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else if e >= -1074 {
        f64::from_bits(1u64 << (e + 1074))
    } else {
        0.0
    }
}

/// `x · 2^e`, exact — chunked so no intermediate over/underflows.
fn ldexp(x: f64, e: i32) -> f64 {
    let mut r = x;
    let mut e = e;
    while e > 1000 {
        r *= exp2(1000);
        e -= 1000;
    }
    while e < -1000 {
        r *= exp2(-1000);
        e += 1000;
    }
    r * exp2(e)
}

/// A row-major `C = A·B` (alpha=1, beta=0) under test. Returns `false` if the
/// implementation refuses the shape.
///
/// The probe is written against this indirection so it can be MUTATION TESTED:
/// `tests/probe_mutation.rs` feeds it deliberately defective pure-Rust kernels
/// (f32 accumulation, FTZ/DAZ confined to the blocked kernel, saturating
/// overflow, `beta=0` reading `C`, Strassen-style block mixing, round-upward)
/// and asserts every one is REFUSED, plus a clean scalar kernel that must be
/// ACCEPTED. A probe that never fires is worthless; this is how we know it does.
pub type DgemmUnderTest<'a> = &'a dyn Fn(usize, usize, usize, &[f64], &[f64], &mut [f64]) -> bool;

/// f32 twin of [`DgemmUnderTest`].
pub type SgemmUnderTest<'a> = &'a dyn Fn(usize, usize, usize, &[f32], &[f32], &mut [f32]) -> bool;

/// The host's real `cblas_dgemm`, in the exact production call form
/// (RowMajor / NoTrans / NoTrans, alpha=1, beta=0, lda=k, ldb=n, ldc=n).
///
/// Exposed so CI-only checks can exercise the kernel WITHOUT the engine's
/// caller-protection guards. In particular the large-N non-mixing test feeds a
/// deliberately NaN/Inf-poisoned operand, which `gemm_f64` correctly declines
/// (the non-finite refusal) — that refusal protects callers, but it would also
/// prevent the very check that proves poison stays confined. Returns `false`
/// without calling into C on any ABI violation.
#[must_use]
pub fn accelerate_dgemm(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> bool {
    ffi::dgemm_row_major(m, k, n, a, b, c)
}

/// Exact-integer reference product. All operands are integers small enough that
/// every product and every partial sum is exact in f64 — hence ORDER-FREE, so a
/// bit assertion is legitimate.
fn exact_int_reference(m: usize, k: usize, n: usize, ai: &[i64], bi: &[i64]) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: i64 = 0;
            for l in 0..k {
                acc += ai[i * k + l] * bi[l * n + j];
            }
            out[i * n + j] = acc as f64;
        }
    }
    out
}

/// The f64 conformance probe (KA-1 .. KA-10) against the host's Accelerate.
#[must_use]
pub fn dgemm_conformance_probe() -> ProbeReport {
    dgemm_conformance_probe_with(&|m, k, n, a, b, c| ffi::dgemm_row_major(m, k, n, a, b, c))
}

/// The f64 conformance probe (KA-1 .. KA-10) against an arbitrary kernel. ~0.4 ms.
///
/// KA-9 always reads the REAL thread FPCR (a pure-Rust mutant cannot corrupt
/// it), so on a mutant run those two checks are expected to pass; every other
/// check is fully exercised.
#[must_use]
pub fn dgemm_conformance_probe_with(g: DgemmUnderTest<'_>) -> ProbeReport {
    let mut r = ProbeReport::default();
    let dgemm = |m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]| -> bool {
        g(m, k, n, a, b, c)
    };
    let dot1 = |a: f64, b: f64| -> Option<f64> {
        let mut c = [f64::NAN];
        g(1, 1, 1, &[a], &[b], &mut c).then_some(c[0])
    };

    // ---- KA-3: rounding of the PRODUCT (1x1x1, order-free) ----
    {
        let a = 1.0 + ldexp(1.0, -52);
        let want = 1.0 + ldexp(1.0, -51);
        r.check(
            "KA-3a",
            "(1+2^-52)^2 rounds to nearest, not upward",
            dot1(a, a).is_some_and(|c| biteq(c, want)),
        );
        r.check(
            "KA-3b",
            "-(1+2^-52)^2 rounds to nearest, not downward",
            dot1(-a, a).is_some_and(|c| biteq(c, -want)),
        );
        r.check(
            "KA-3c",
            "1*(1+2^-52) keeps all 53 significand bits",
            dot1(1.0, a).is_some_and(|c| biteq(c, a)),
        );
    }

    // ---- KA-4: ties-to-even in the ADD (k=2 => every order coincides) ----
    // Neither half suffices alone: (a) is the only check that catches true
    // ties-away, (b) the only one that catches round-toward-zero.
    {
        let mut c = [f64::NAN];
        let ok_a = dgemm(1, 2, 1, &[1.0, 1.0], &[1.0, ldexp(1.0, -53)], &mut c) && biteq(c[0], 1.0);
        r.check("KA-4a", "exact tie 1+2^-53 rounds to even (1.0)", ok_a);

        let mut c = [f64::NAN];
        let ok_b = dgemm(1, 2, 1, &[1.0, 1.0], &[1.0, 3.0 * ldexp(1.0, -54)], &mut c)
            && biteq(c[0], 1.0 + ldexp(1.0, -52));
        r.check(
            "KA-4b",
            "above-tie 1+3*2^-54 rounds up, not toward zero",
            ok_b,
        );
    }

    // ---- KA-5: subnormal battery (FTZ / DAZ), five parts ----
    {
        let min_sub = ldexp(1.0, -1074);
        r.check(
            "KA-5a",
            "1*2^-1074 survives (no result flush-to-zero)",
            dot1(1.0, min_sub).is_some_and(|c| biteq(c, min_sub)),
        );
        r.check(
            "KA-5b",
            "2^-1074*2^100 = 2^-974 (no denormals-are-zero)",
            dot1(min_sub, ldexp(1.0, 100)).is_some_and(|c| biteq(c, ldexp(1.0, -974))),
        );
        r.check(
            "KA-5c",
            "2^-1000*2^-60 = 2^-1060 (gradual underflow of the product)",
            dot1(ldexp(1.0, -1000), ldexp(1.0, -60)).is_some_and(|c| biteq(c, ldexp(1.0, -1060))),
        );

        // (d) and (e) are the LOAD-BEARING ones and are absent from the
        // prior-art probe list: a 1x1x1 call may take a scalar path that a
        // vectorized FTZ never touches. Mutation testing: mutant N1 (FTZ only
        // when m,n>=16 and k>=8) is caught by KA-5d ALONE; N2 (DAZ, same
        // condition) by KA-5e ALONE.
        const N: usize = 64;
        let a = vec![ldexp(1.0, -530); N * N];
        let mut c = vec![f64::NAN; N * N];
        let want = ldexp(N as f64, -1060); // 64*2^-1060 = 2^-1054, exact
        let ok_d = dgemm(N, N, N, &a, &a, &mut c) && c.iter().all(|&v| biteq(v, want));
        r.check(
            "KA-5d",
            "64^3 all-subnormal partial sums: no FTZ in the BLOCKED kernel",
            ok_d,
        );

        let a = vec![ldexp(1.0, -1060); N * N];
        let b = vec![ldexp(1.0, 200); N * N];
        let mut c = vec![f64::NAN; N * N];
        let want = ldexp(N as f64, -860);
        let ok_e = dgemm(N, N, N, &a, &b, &mut c) && c.iter().all(|&v| biteq(v, want));
        r.check(
            "KA-5e",
            "64^3 subnormal OPERAND: no DAZ in the BLOCKED kernel",
            ok_e,
        );
    }

    // ---- KA-8: overflow -> +/-Inf, never saturation ----
    // Saturating to DBL_MAX would convert an overflow into a large finite
    // number and could silently TIGHTEN a published bound.
    {
        r.check(
            "KA-8a",
            "1e300^2 -> +Inf (not DBL_MAX)",
            dot1(1e300, 1e300).is_some_and(|c| biteq(c, f64::INFINITY)),
        );
        r.check(
            "KA-8b",
            "DBL_MAX*(1+2^-52) -> +Inf",
            dot1(f64::MAX, 1.0 + ldexp(1.0, -52)).is_some_and(|c| biteq(c, f64::INFINITY)),
        );
    }

    // ---- KA-1 (+ KA-7): exact-int 32x128x32 with a NaN-poisoned C ----
    {
        let (m, k, n) = (32usize, 128usize, 32usize);
        let mut lcg = Lcg::new();
        let ai: Vec<i64> = (0..m * k).map(|_| lcg.int(-8191, 8191)).collect();
        let bi: Vec<i64> = (0..k * n).map(|_| lcg.int(-8191, 8191)).collect();
        let a: Vec<f64> = ai.iter().map(|&v| v as f64).collect();
        let b: Vec<f64> = bi.iter().map(|&v| v as f64).collect();
        // beta=0 must not READ C (KA-7): any read of NaN poisons the output.
        let mut c = vec![f64::NAN; m * n];
        let want = exact_int_reference(m, k, n, &ai, &bi);
        let ok =
            dgemm(m, k, n, &a, &b, &mut c) && c.iter().zip(&want).all(|(&got, &w)| biteq(got, w));
        r.check(
            "KA-1/7",
            "exact-int 32x128x32 bit-exact; beta=0 does not read a NaN C",
            ok,
        );
    }

    // ---- KA-2: exact-int DEEP-k 8x4096x8 ----
    // Guards the deep-contraction kernel KA-1's k=128 never reaches (mutant N3:
    // f32 accumulation only for k >= 1024).
    {
        let (m, k, n) = (8usize, 4096usize, 8usize);
        let mut lcg = Lcg::new();
        let ai: Vec<i64> = (0..m * k).map(|_| lcg.int(-1023, 1023)).collect();
        let bi: Vec<i64> = (0..k * n).map(|_| lcg.int(-1023, 1023)).collect();
        let a: Vec<f64> = ai.iter().map(|&v| v as f64).collect();
        let b: Vec<f64> = bi.iter().map(|&v| v as f64).collect();
        let mut c = vec![f64::NAN; m * n];
        let want = exact_int_reference(m, k, n, &ai, &bi);
        let ok =
            dgemm(m, k, n, &a, &b, &mut c) && c.iter().zip(&want).all(|(&got, &w)| biteq(got, w));
        r.check("KA-2", "exact-int deep-k 8x4096x8 bit-exact", ok);
    }

    // ---- KA-10: magnitude invariance (no small/large fast path) ----
    {
        let (m, k, n) = (8usize, 256usize, 8usize);
        for &e in &[-450i32, 450i32] {
            let sc = ldexp(1.0, e);
            let mut lcg = Lcg::new();
            let ai: Vec<i64> = (0..m * k).map(|_| lcg.int(-8191, 8191)).collect();
            let bi: Vec<i64> = (0..k * n).map(|_| lcg.int(-8191, 8191)).collect();
            let a: Vec<f64> = ai.iter().map(|&v| v as f64 * sc).collect();
            let b: Vec<f64> = bi.iter().map(|&v| v as f64 * sc).collect();
            let mut c = vec![f64::NAN; m * n];
            let base = exact_int_reference(m, k, n, &ai, &bi);
            let ok = dgemm(m, k, n, &a, &b, &mut c)
                && c.iter()
                    .zip(&base)
                    .all(|(&got, &w)| biteq(got, ldexp(w, 2 * e)));
            r.check(
                if e < 0 { "KA-10a" } else { "KA-10b" },
                "exact-int product is scale-invariant (2^-450 / 2^+450)",
                ok,
            );
        }
    }

    // ---- KA-6: no cross-entry mixing (Strassen / Winograd detector) ----
    // This is the assumption gamma_n*S structurally REQUIRES: a Strassen error
    // bound is norm-based, not per-entry-S-based. Two independent refutations in
    // one call: (a) NaN/Inf poison must stay confined to its own row/col, and
    // (b) a tiny exact-integer row/col surrounded by 1e150-scale neighbours must
    // come back bit-exact.
    {
        const N: usize = 128;
        let mut lcg = Lcg::new();
        let mut a: Vec<f64> = (0..N * N)
            .map(|_| lcg.int(-1000, 1000) as f64 * 1e147)
            .collect();
        let mut b: Vec<f64> = (0..N * N)
            .map(|_| lcg.int(-1000, 1000) as f64 * 1e147)
            .collect();
        let mut acc: i64 = 0;
        for l in 0..N {
            let ra = lcg.int(-1023, 1023);
            let cb = lcg.int(-1023, 1023);
            a[(N - 1) * N + l] = ra as f64;
            b[l * N + (N - 1)] = cb as f64;
            acc += ra * cb;
        }
        a[3 * N + 5] = f64::NAN;
        b[7 * N + 11] = f64::INFINITY;
        let mut c = vec![f64::NAN; N * N];
        if dgemm(N, N, N, &a, &b, &mut c) {
            let mut leak = false;
            for i in 0..N {
                for j in 0..N {
                    if i != 3 && j != 11 && !c[i * N + j].is_finite() {
                        leak = true;
                    }
                }
            }
            r.check("KA-6a", "NaN/Inf stays confined to its own row/col", !leak);
            r.check(
                "KA-6b",
                "tiny exact entry uncontaminated by 1e150 neighbours",
                biteq(c[(N - 1) * N + (N - 1)], acc as f64),
            );
        } else {
            r.check("KA-6a", "NaN/Inf stays confined to its own row/col", false);
            r.check(
                "KA-6b",
                "tiny exact entry uncontaminated by 1e150 neighbours",
                false,
            );
        }
    }

    // ---- KA-9: FPCR is not disturbed ----
    {
        let before = ffi::read_fpcr();
        let (m, k, n) = (32usize, 128usize, 32usize);
        let a = vec![1.5f64; m * k];
        let b = vec![0.5f64; k * n];
        let mut c = vec![0.0f64; m * n];
        let ran = dgemm(m, k, n, &a, &b, &mut c);
        let after = ffi::read_fpcr();
        // KA-9a is diagnostic only (the mutation matrix showed it catches
        // nothing 9b does not); 9b is the load-bearing assertion.
        r.check(
            "KA-9a",
            "FPCR identical across the call (diagnostic)",
            before == after,
        );
        let fz = (after >> 24) & 1;
        let fz16 = (after >> 19) & 1;
        let rmode = (after >> 22) & 3;
        r.check(
            "KA-9b",
            "FPCR.FZ / FZ16 / RMode all clear after the call",
            ran && fz == 0 && fz16 == 0 && rmode == 0,
        );
    }

    r
}

// ---------------------------------------------------------------------------
// f32 free-rider probe (SA-*). Non-verdict traffic (IBP forward / PGD / BaB),
// but `GemmEngine::gemm_f32`'s contract is still "plain IEEE round-to-nearest
// f32, any summation order", and the CROWN abs-sum seam charges
// `gamma_k^f32 * S`. So the sgemm free-rider gets its own probe with the same
// hazard list at f32 width, and the same order-independence design rule.
// ---------------------------------------------------------------------------

/// `2^e` as an exact f32 (same bit-pattern construction as [`exp2`], same
/// reason: `f32::powi(2.0, -149)` returns 0.0).
fn exp2_32(e: i32) -> f32 {
    if e > 127 {
        f32::INFINITY
    } else if e >= -126 {
        f32::from_bits(((e + 127) as u32) << 23)
    } else if e >= -149 {
        f32::from_bits(1u32 << (e + 149))
    } else {
        0.0
    }
}

/// `x · 2^e`, exact in f32 — chunked so no intermediate over/underflows.
fn ldexp32(x: f32, e: i32) -> f32 {
    let mut r = x;
    let mut e = e;
    while e > 100 {
        r *= exp2_32(100);
        e -= 100;
    }
    while e < -100 {
        r *= exp2_32(-100);
        e += 100;
    }
    r * exp2_32(e)
}

/// The f32 conformance probe (SA-1 .. SA-6) against the host's Accelerate.
#[must_use]
pub fn sgemm_conformance_probe() -> ProbeReport {
    sgemm_conformance_probe_with(&|m, k, n, a, b, c| ffi::sgemm_row_major(m, k, n, a, b, c))
}

/// The f32 conformance probe (SA-1 .. SA-6) against an arbitrary kernel.
#[must_use]
pub fn sgemm_conformance_probe_with(g: SgemmUnderTest<'_>) -> ProbeReport {
    let mut r = ProbeReport::default();
    let sgemm = |m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]| -> bool {
        g(m, k, n, a, b, c)
    };
    let sdot1 = |a: f32, b: f32| -> Option<f32> {
        let mut c = [f32::NAN];
        g(1, 1, 1, &[a], &[b], &mut c).then_some(c[0])
    };

    // ---- SA-1: rounding of the product / RN not RU-RD-RZ ----
    {
        let a = 1.0f32 + ldexp32(1.0, -23);
        let want = 1.0f32 + ldexp32(1.0, -22);
        r.check(
            "SA-1a",
            "(1+2^-23)^2 rounds to nearest, not upward",
            sdot1(a, a).is_some_and(|c| biteq32(c, want)),
        );
        r.check(
            "SA-1b",
            "-(1+2^-23)^2 rounds to nearest, not downward",
            sdot1(-a, a).is_some_and(|c| biteq32(c, -want)),
        );
        r.check(
            "SA-1c",
            "1*(1+2^-23) keeps all 24 significand bits",
            sdot1(1.0, a).is_some_and(|c| biteq32(c, a)),
        );
    }

    // ---- SA-2: ties-to-even in the ADD (k=2 => order-free) ----
    {
        let mut c = [f32::NAN];
        let ok_a =
            sgemm(1, 2, 1, &[1.0, 1.0], &[1.0, ldexp32(1.0, -24)], &mut c) && biteq32(c[0], 1.0);
        r.check("SA-2a", "exact tie 1+2^-24 rounds to even (1.0)", ok_a);

        let mut c = [f32::NAN];
        let ok_b = sgemm(
            1,
            2,
            1,
            &[1.0, 1.0],
            &[1.0, 3.0 * ldexp32(1.0, -25)],
            &mut c,
        ) && biteq32(c[0], 1.0 + ldexp32(1.0, -23));
        r.check(
            "SA-2b",
            "above-tie 1+3*2^-25 rounds up, not toward zero",
            ok_b,
        );
    }

    // ---- SA-3: subnormal battery, incl. the blocked kernel ----
    {
        let min_sub = ldexp32(1.0, -149);
        r.check(
            "SA-3a",
            "1*2^-149 survives (no result flush-to-zero)",
            sdot1(1.0, min_sub).is_some_and(|c| biteq32(c, min_sub)),
        );
        r.check(
            "SA-3b",
            "2^-149*2^100 = 2^-49 (no denormals-are-zero)",
            sdot1(min_sub, ldexp32(1.0, 100)).is_some_and(|c| biteq32(c, ldexp32(1.0, -49))),
        );
        const N: usize = 64;
        // 2^-70 * 2^-70 = 2^-140 (subnormal, exact); 64 of them = 2^-134.
        let a = vec![ldexp32(1.0, -70); N * N];
        let mut c = vec![f32::NAN; N * N];
        let want = ldexp32(N as f32, -140);
        let ok_c = sgemm(N, N, N, &a, &a, &mut c) && c.iter().all(|&v| biteq32(v, want));
        r.check(
            "SA-3c",
            "64^3 all-subnormal partial sums: no FTZ in the blocked kernel",
            ok_c,
        );
        // subnormal OPERAND, normal product: catches denormals-are-zero.
        let a = vec![ldexp32(1.0, -140); N * N];
        let b = vec![ldexp32(1.0, 60); N * N];
        let mut c = vec![f32::NAN; N * N];
        let want = ldexp32(N as f32, -80);
        let ok_d = sgemm(N, N, N, &a, &b, &mut c) && c.iter().all(|&v| biteq32(v, want));
        r.check(
            "SA-3d",
            "64^3 subnormal OPERAND: no DAZ in the blocked kernel",
            ok_d,
        );
    }

    // ---- SA-4: exact-int accumulation (order-free) at seam shape ----
    // |v| <= 181 => products <= 32761 and |partial sum| <= 128*32761 =
    // 4_193_408 < 2^23, so every partial sum is exact in f32 in ANY order.
    {
        let (m, k, n) = (32usize, 128usize, 32usize);
        let mut lcg = Lcg::new();
        let ai: Vec<i64> = (0..m * k).map(|_| lcg.int(-181, 181)).collect();
        let bi: Vec<i64> = (0..k * n).map(|_| lcg.int(-181, 181)).collect();
        let a: Vec<f32> = ai.iter().map(|&v| v as f32).collect();
        let b: Vec<f32> = bi.iter().map(|&v| v as f32).collect();
        let mut c = vec![f32::NAN; m * n]; // beta=0 must not read C
        let want = exact_int_reference(m, k, n, &ai, &bi);
        let ok = sgemm(m, k, n, &a, &b, &mut c)
            && c.iter().zip(&want).all(|(&got, &w)| biteq32(got, w as f32));
        r.check(
            "SA-4",
            "exact-int 32x128x32 bit-exact; beta=0 does not read a NaN C",
            ok,
        );
    }

    // ---- SA-5: overflow -> +Inf, never saturation ----
    {
        r.check(
            "SA-5a",
            "1e38^2 -> +Inf (not FLT_MAX)",
            sdot1(1e38, 1e38).is_some_and(|c| biteq32(c, f32::INFINITY)),
        );
        r.check(
            "SA-5b",
            "FLT_MAX*(1+2^-23) -> +Inf",
            sdot1(f32::MAX, 1.0 + ldexp32(1.0, -23)).is_some_and(|c| biteq32(c, f32::INFINITY)),
        );
    }

    // ---- SA-6: no cross-entry mixing at f32 ----
    {
        const N: usize = 64;
        let mut lcg = Lcg::new();
        let mut a: Vec<f32> = (0..N * N)
            .map(|_| lcg.int(-1000, 1000) as f32 * 1e15)
            .collect();
        let mut b: Vec<f32> = (0..N * N)
            .map(|_| lcg.int(-1000, 1000) as f32 * 1e15)
            .collect();
        let mut acc: i64 = 0;
        for l in 0..N {
            let ra = lcg.int(-127, 127);
            let cb = lcg.int(-127, 127);
            a[(N - 1) * N + l] = ra as f32;
            b[l * N + (N - 1)] = cb as f32;
            acc += ra * cb;
        }
        a[3 * N + 5] = f32::NAN;
        b[7 * N + 11] = f32::INFINITY;
        let mut c = vec![f32::NAN; N * N];
        if sgemm(N, N, N, &a, &b, &mut c) {
            let mut leak = false;
            for i in 0..N {
                for j in 0..N {
                    if i != 3 && j != 11 && !c[i * N + j].is_finite() {
                        leak = true;
                    }
                }
            }
            r.check("SA-6a", "NaN/Inf stays confined to its own row/col", !leak);
            r.check(
                "SA-6b",
                "tiny exact entry uncontaminated by 1e30-scale neighbours",
                biteq32(c[(N - 1) * N + (N - 1)], acc as f32),
            );
        } else {
            r.check("SA-6a", "NaN/Inf stays confined to its own row/col", false);
            r.check(
                "SA-6b",
                "tiny exact entry uncontaminated by 1e30-scale neighbours",
                false,
            );
        }
    }

    r
}
