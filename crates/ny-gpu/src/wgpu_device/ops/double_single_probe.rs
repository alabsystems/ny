// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! MAKE-OR-BREAK probe: is "double-single" (compensated two-`f32`) sound
//! arithmetic even POSSIBLE on THIS live Metal/wgpu adapter?
//!
//! This module is a STANDALONE experiment. It is wired into NOTHING — no verdict
//! path, no `provides_sound_gpu_*` predicate, no shipped code. The whole file only
//! compiles under `cfg(all(test, feature = "gpu-tests"))` (see the gated `mod`
//! declaration in `ops/mod.rs`), so a plain build/verdict run never sees it.
//!
//! # The question
//!
//! Double-single stores an `f64`-like value as an `f32` `(hi, lo)` pair and depends
//! on TWO error-free transforms (EFTs), BOTH of which must survive the shader
//! compiler for a compensated dot product / matmul to gain any precision:
//!   * `TwoProduct` — the exact product of two `f32` as `(hi, lo)`. Two forms:
//!       - Dekker/Veltkamp SPLIT-based (splitter `2^12 + 1 = 4097`; `f32` has 24
//!         mantissa bits = 2*12). Its error term is a chain of `a*b + c` sub-
//!         expressions, so an FMA CONTRACTION silently changes `lo` and breaks it.
//!       - FMA-based `lo = fma(a, b, -hi)`, exact iff `fma` is a true single-rounding
//!         fused multiply-add (and USELESS if `fma` is emulated as `a*b + c`).
//!   * `TwoSum` — the exact sum of two `f32` as `(hi, lo)` (Knuth/Møller, pure adds).
//!     Its error term `(a - (s - b')) + (b - b')` is algebraically zero, so an
//!     aggressive fast-math REASSOCIATION (treating `+` as exact/associative) collapses
//!     `lo` to `0` and destroys the compensation.
//!
//! NY's own `f32` self-check (`ops/f32_selfcheck.rs`, probe 3) EXPLICITLY PERMITS
//! `a*b + c` contraction, so we do NOT yet know whether the Metal WGSL compiler
//! preserves either EFT. This probe measures all three primitives directly, per lane,
//! BIT-EXACTLY, on the live adapter.
//!
//! # Verdict
//!
//! Double-single is VIABLE only if SOME `TwoProduct` variant is bit-exact AND `TwoSum`
//! is bit-exact (a compensated dot needs both). A running double-single dot vs a
//! compensated `f64` Dot2 oracle corroborates: a working stack reaches ~`2^-46`
//! relative; a broken one degrades to plain-`f32` ~`2^-24`.

#![cfg(all(test, feature = "gpu-tests"))]

use super::super::WgpuDevice;
use super::ibp_forward::create_buffer;
use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_verdict_device};

/// 16-byte std140/std430-clean uniform params. Layout MUST match WGSL `struct Params`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    /// Number of product/dot lanes.
    n_prod: u32,
    /// Number of TwoSum lanes.
    n_sum: u32,
    _pad0: u32,
    _pad1: u32,
}

/// WGSL source. Every operand is read from a storage buffer at runtime, so the
/// compiler CANNOT constant-fold the arithmetic (which would defeat the probe). A
/// single-thread dispatch performs the serial accumulation in host-matching order.
///
/// Isolation of the three EFTs is deliberate:
///   * `two_prod_dekker` is the ONLY FMA-contraction-sensitive site (its `+ sa.x*sb.y`
///     / `+ sa.y*sb.x` / `+ sa.y*sb.y` additions are fuse opportunities).
///   * `two_prod_fma` is the ONLY site that depends on `fma` being a true fused op.
///   * `two_sum` is pure adds — its error term is the reassociation-sensitive site.
const DOUBLE_SINGLE_SHADER: &str = r#"
struct Params { n_prod: u32, n_sum: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>        p_in: array<vec2<f32>>;   // (a, b) product/dot lanes
@group(0) @binding(2) var<storage, read>        s_in: array<vec2<f32>>;   // (a, b) sum lanes
@group(0) @binding(3) var<storage, read_write>  prod_dekker: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write>  prod_fma:    array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write>  sum_out:     array<vec2<f32>>;
// dot_out lanes (each a double-single (hi, lo) accumulator result):
//   [0] dekker-product, PLAIN ds_add        (legacy lane)
//   [1] fma-product,    PLAIN ds_add        (legacy lane)
//   [2] fma-product,    barrier TwoSum ONLY (plain FastTwoSum)   <- isolation
//   [3] fma-product,    barrier FastTwoSum ONLY (plain TwoSum)   <- isolation
//   [4] fma-product,    FULLY barrier ds_add                     <- THE composed test
//   [5] dekker-product, FULLY barrier ds_add                     <- control
@group(0) @binding(6) var<storage, read_write>  dot_out:     array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write>  sum_bar:     array<vec2<f32>>; // fma-barrier TwoSum

// Veltkamp splitter for f32: 2^12 + 1. f32 has 24 mantissa bits = 2*12.
const SPLITTER: f32 = 4097.0;

// Knuth/Møller TwoSum: exact a + b = (s, err). Pure adds; err is reassociation-sensitive.
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    return vec2<f32>(s, err);
}

// Dekker FastTwoSum: exact when |a| >= |b|. Pure adds. Renormalizes the accumulator.
fn fast_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let err = b - (s - a);
    return vec2<f32>(s, err);
}

// fma-barrier TwoSum (upgrade probe): every subtraction of the Knuth sequence is
// routed through the fma intrinsic. The bet: the compiler will not algebraically
// simplify ACROSS intrinsic calls, so the (algebraically-zero) compensation term
// survives even where the plain-adds form is folded to 0. The counter-bet: a
// smart driver canonicalizes fma(-1, x, y) back to y - x (rounding-identical!)
// and then reassociates anyway. Which one wins is exactly what this measures.
fn two_sum_fma_barrier(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = fma(-1.0, a, s);   // s - a
    let sb = fma(-1.0, bb, s);  // s - bb
    let da = fma(-1.0, sb, a);  // a - (s - bb)
    let db = fma(-1.0, bb, b);  // b - bb
    return vec2<f32>(s, da + db);
}

// Veltkamp split: a = hi + lo, hi holds the top 12 bits, both exact.
fn veltkamp_split(a: f32) -> vec2<f32> {
    let c = SPLITTER * a;
    let big = c - a;
    let hi = c - big;
    let lo = a - hi;
    return vec2<f32>(hi, lo);
}

// Dekker split-based TwoProduct: p + err == a*b EXACTLY, iff no FMA contraction of
// the `... + sa.x*sb.y + sa.y*sb.x` additions occurs.
fn two_prod_dekker(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let sa = veltkamp_split(a);
    let sb = veltkamp_split(b);
    let err = ((sa.x * sb.x - p) + sa.x * sb.y + sa.y * sb.x) + sa.y * sb.y;
    return vec2<f32>(p, err);
}

// FMA-based TwoProduct: err = fma(a, b, -p) is exact IFF `fma` is a true single-
// rounding fused multiply-add.
fn two_prod_fma(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let err = fma(a, b, -p);
    return vec2<f32>(p, err);
}

// fma-barrier FastTwoSum: the Dekker renormalization step with its (algebraically
// zero) error term routed through the fma intrinsic, exactly as two_sum_fma_barrier
// does for Knuth. `fast_two_sum` above is PURE ADDS and was never probed per-lane —
// it is the second reassociation-sensitive site inside ds_add.
fn fast_two_sum_fma_barrier(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let t = fma(-1.0, a, s);    // s - a
    let err = fma(-1.0, t, b);  // b - (s - a)
    return vec2<f32>(s, err);
}

// Add a double-single y into a double-single x (two-sum accumulation + renormalize).
//
// NOTE (this is what the composed-dot corroboration line actually measured): BOTH
// EFTs used here are the PLAIN forms. `two_sum` is measured FAILING on this adapter
// (the compiler reassociates its compensation term to 0) and `fast_two_sum` is pure
// adds and never probed at all. So both legacy dot lanes accumulate through broken
// primitives regardless of which TwoProduct feeds them — which is why they printed
// the identical plain-f32-grade number.
fn ds_add(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let s = two_sum(x.x, y.x);
    let e = s.y + (x.y + y.y);
    return fast_two_sum(s.x, e);
}

// Isolation variant A: barrier TwoSum, PLAIN FastTwoSum.
fn ds_add_bar_sum(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let s = two_sum_fma_barrier(x.x, y.x);
    let e = s.y + (x.y + y.y);
    return fast_two_sum(s.x, e);
}

// Isolation variant B: PLAIN TwoSum, barrier FastTwoSum.
fn ds_add_bar_fast(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let s = two_sum(x.x, y.x);
    let e = s.y + (x.y + y.y);
    return fast_two_sum_fma_barrier(s.x, e);
}

// THE composed test: both EFTs in the fma-barrier form. The middle line is left
// BYTE-IDENTICAL to `ds_add` on purpose — those are genuine (non-degenerate) adds,
// so the ONLY difference between this and `ds_add` is the two EFT primitives.
fn ds_add_barrier(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let s = two_sum_fma_barrier(x.x, y.x);
    let e = s.y + (x.y + y.y);
    return fast_two_sum_fma_barrier(s.x, e);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    // Per-lane TwoProduct (both variants) + running double-single dot (both variants).
    var acc_d = vec2<f32>(0.0, 0.0);
    var acc_f = vec2<f32>(0.0, 0.0);
    var acc_bs = vec2<f32>(0.0, 0.0);  // fma-product, barrier TwoSum only
    var acc_bf = vec2<f32>(0.0, 0.0);  // fma-product, barrier FastTwoSum only
    var acc_fb = vec2<f32>(0.0, 0.0);  // fma-product, FULLY barrier
    var acc_db = vec2<f32>(0.0, 0.0);  // dekker-product, FULLY barrier
    for (var i: u32 = 0u; i < params.n_prod; i = i + 1u) {
        let ab = p_in[i];
        let pd = two_prod_dekker(ab.x, ab.y);
        let pf = two_prod_fma(ab.x, ab.y);
        prod_dekker[i] = pd;
        prod_fma[i] = pf;
        acc_d = ds_add(acc_d, pd);
        acc_f = ds_add(acc_f, pf);
        acc_bs = ds_add_bar_sum(acc_bs, pf);
        acc_bf = ds_add_bar_fast(acc_bf, pf);
        acc_fb = ds_add_barrier(acc_fb, pf);
        acc_db = ds_add_barrier(acc_db, pd);
    }
    dot_out[0] = acc_d;
    dot_out[1] = acc_f;
    dot_out[2] = acc_bs;
    dot_out[3] = acc_bf;
    dot_out[4] = acc_fb;
    dot_out[5] = acc_db;

    // Per-lane TwoSum (plain Knuth + fma-barrier variant).
    for (var j: u32 = 0u; j < params.n_sum; j = j + 1u) {
        let ab = s_in[j];
        sum_out[j] = two_sum(ab.x, ab.y);
        sum_bar[j] = two_sum_fma_barrier(ab.x, ab.y);
    }
}
"#;

// ---------------------------------------------------------------------------
// CPU ground truth (exact-in-f64 EFT reference + compensated Dot2 oracle).
// ---------------------------------------------------------------------------

/// Compensated `two_sum` in `f64` (Ogita-Rump-Oishi / Knuth), mirrors the
/// `margin_row` oracle.
#[inline]
fn two_sum_f64(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// Compensated `two_prod` in `f64` via a fused multiply-add, mirrors the `margin_row`
/// oracle. (Rust `f64::mul_add` is a hardware FMA on Apple Silicon.)
#[inline]
fn two_prod_f64(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// Ogita-Rump-Oishi Dot2 accumulator — the INDEPENDENT high-precision reference for
/// the dot product (error ~ eps^2 relative even under cancellation).
#[derive(Clone, Copy)]
struct Acc {
    s: f64,
    c: f64,
}
impl Acc {
    #[inline]
    fn new() -> Self {
        Self { s: 0.0, c: 0.0 }
    }
    #[inline]
    fn add_prod(&mut self, a: f64, b: f64) {
        let (p, ep) = two_prod_f64(a, b);
        let (s, es) = two_sum_f64(self.s, p);
        self.s = s;
        self.c += ep + es;
    }
    #[inline]
    fn value(&self) -> f64 {
        self.s + self.c
    }
}

/// The TRUE error-free `(hi, lo)` for the `f32` product `a*b`, computed exactly in
/// `f64`. A product of two 24-bit values is 48 bits — exact in `f64`'s 53-bit mantissa
/// — so `hi = fl_f32(a*b)`, `lo = fl_f32(a*b - hi)`, and `f64(hi)+f64(lo)==f64(a)*f64(b)`.
fn true_two_product(a: f32, b: f32) -> (f32, f32, f64) {
    let exact = f64::from(a) * f64::from(b); // exact
    let hi = a * b; // correctly-rounded f32 product
    let lo = (exact - f64::from(hi)) as f32; // exact rounding error, representable in f32
    (hi, lo, exact)
}

/// The TRUE error-free `(hi, lo)` for the `f32` sum `a+b`, computed exactly in `f64`.
/// Valid ONLY when the exact sum fits in `f64` (operand exponents within ~29 of each
/// other) — the sum inputs below are constructed to guarantee that.
fn true_two_sum(a: f32, b: f32) -> (f32, f32, f64) {
    let exact = f64::from(a) + f64::from(b); // exact for the constrained inputs
    let hi = a + b;
    let lo = (exact - f64::from(hi)) as f32; // exact rounding error, representable in f32
    (hi, lo, exact)
}

/// Deterministic xorshift64 PRNG (no external dep) for adversarial mantissas.
struct XorShift64(u64);
impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Adversarial `(a, b)` PRODUCT pairs: explicit hand-picked worst cases plus many
/// random full-24-bit-mantissa normals whose product generically has a nonzero low
/// part (so a fused/contracted multiply yields a DIFFERENT `lo` than the true EFT).
/// POSITIVE with a bounded dynamic range so the running dot has condition ~1 and its
/// precision is a clean corroboration of the per-lane verdict.
fn build_product_inputs() -> Vec<(f32, f32)> {
    let mut v = Vec::new();
    let two_m12 = f32::from_bits(0x3F80_0800); // 1 + 2^-12
    v.push((two_m12, two_m12)); // (1+2^-12)^2 = 1 + 2^-11 + 2^-24 -> lo = 2^-24
    let ulp = f32::from_bits(0x3400_0000); // 2^-23
    v.push((1.0 + ulp, 1.0 + ulp));
    v.push((1.0 + 3.0 * ulp, 1.0 + 5.0 * ulp));
    v.push((1.3333333, 3.0000002));
    v.push((std::f32::consts::PI, std::f32::consts::E));
    v.push((std::f32::consts::SQRT_2, 1.7320508)); // sqrt2 * sqrt3
    v.push((123.456, 789.012));
    v.push((1.9999999, 1.9999999));
    v.push((1.0 + 2047.0 * ulp, 1.0 + 2047.0 * ulp));

    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
    let make = |rng: &mut XorShift64| -> f32 {
        let r = rng.next();
        // 23 random fraction bits + exponent in [118, 132] (~2^-9 .. 2^9): products
        // span ~2^-18 .. 2^18 (< the double-single's ~46-bit reach), all positive.
        let frac = (r & 0x7F_FFFF) as u32;
        let exp = 118u32 + ((r >> 24) % 15) as u32;
        f32::from_bits((exp << 23) | frac)
    };
    for _ in 0..500 {
        v.push((make(&mut rng), make(&mut rng)));
    }
    v
}

/// Adversarial `(a, b)` SUM pairs whose exact sum has a nonzero low part but still fits
/// in `f64` (exponent offsets bounded to <= 26), so `true_two_sum` is exact. Exercises
/// the reassociation-sensitive TwoSum error term.
fn build_sum_inputs() -> Vec<(f32, f32)> {
    let mut v = Vec::new();
    let ulp = f32::from_bits(0x3400_0000); // 2^-23
    v.push((1.0, ulp)); // 1 + 2^-23 = next-up, lo captured on the boundary
    v.push((1.0, 0.5 * ulp)); // half-ulp: rounds, nonzero lo
    v.push((1.0 + ulp, -1.0)); // Sterbenz-ish cancellation
    v.push((16777216.0, 1.0)); // 2^24 + 1: 1 falls below the ulp of 2^24
    v.push((16777217.0f32, 0.5)); // rounding at the 2^24 boundary
    v.push((std::f32::consts::PI, 1.0 / 3.0));
    v.push((1234.5678, 0.001_234_5));

    let mut rng = XorShift64(0x1234_5678_9ABC_DEF0);
    for _ in 0..300 {
        let r = rng.next();
        // base in [1, 2), offset in [1, 2) scaled by 2^-d with d in [1, 26] so the
        // exact sum has up to ~50 bits (fits f64). Random sign on the offset.
        let base = f32::from_bits(0x3F80_0000 | ((r & 0x7F_FFFF) as u32));
        let d = 1 + ((r >> 24) % 26) as i32;
        let off_frac = ((r >> 30) & 0x7F_FFFF) as u32;
        let mut off = f32::from_bits(0x3F80_0000 | off_frac) * 2f32.powi(-d);
        if (r >> 55) & 1 == 1 {
            off = -off;
        }
        v.push((base, off));
    }
    v
}

/// Result of one per-lane EFT variant's evaluation.
struct EftVerdict {
    name: &'static str,
    lanes: usize,
    hi_mismatches: usize,
    lo_mismatches: usize,
    max_lo_ulp_err: f64, // max |gpu_lo - true_lo| in ULPs of true_lo (0 if exact)
    max_recon_abs: f64,  // max |f64(hi)+f64(lo) - exact value|
    max_recon_rel: f64,  // relative form
    pass: bool,
}

/// Evaluate a per-lane EFT: bit-exact `(hi, lo)` and exact reconstruction of `exact`.
fn evaluate_eft(
    name: &'static str,
    refs: &[(f32, f32, f64)], // (true_hi, true_lo, exact) per lane
    gpu_bits: &[u32],         // 2 u32 per lane (hi_bits, lo_bits)
) -> EftVerdict {
    let mut hi_mismatches = 0usize;
    let mut lo_mismatches = 0usize;
    let mut max_lo_ulp_err = 0.0f64;
    let mut max_recon_abs = 0.0f64;
    let mut max_recon_rel = 0.0f64;

    for (i, &(true_hi, true_lo, exact)) in refs.iter().enumerate() {
        let gpu_hi = f32::from_bits(gpu_bits[2 * i]);
        let gpu_lo = f32::from_bits(gpu_bits[2 * i + 1]);
        if gpu_hi.to_bits() != true_hi.to_bits() {
            hi_mismatches += 1;
        }
        if gpu_lo.to_bits() != true_lo.to_bits() {
            lo_mismatches += 1;
        }
        let lo_diff = (f64::from(gpu_lo) - f64::from(true_lo)).abs();
        if lo_diff != 0.0 {
            let ulp = if true_lo != 0.0 {
                f64::from(ulp_f32(true_lo))
            } else {
                f64::from(ulp_f32(true_hi))
            };
            max_lo_ulp_err = max_lo_ulp_err.max(lo_diff / ulp);
        }
        let recon = f64::from(gpu_hi) + f64::from(gpu_lo);
        let abs = (recon - exact).abs();
        max_recon_abs = max_recon_abs.max(abs);
        if exact != 0.0 {
            max_recon_rel = max_recon_rel.max(abs / exact.abs());
        }
    }
    let pass = hi_mismatches == 0 && lo_mismatches == 0 && max_recon_abs == 0.0;
    EftVerdict {
        name,
        lanes: refs.len(),
        hi_mismatches,
        lo_mismatches,
        max_lo_ulp_err,
        max_recon_abs,
        max_recon_rel,
        pass,
    }
}

/// ULP of an `f32` (distance to the next representable value away from zero).
fn ulp_f32(x: f32) -> f32 {
    let bits = x.abs().to_bits();
    let up = f32::from_bits(bits.wrapping_add(1));
    up - x.abs()
}

impl EftVerdict {
    fn report(&self) -> String {
        format!(
            "  [{}] {}\n    hi mismatches: {} / {} | lo mismatches: {} / {}\n    \
             max lo-term error: {:.3e} ULP(lo) | max reconstruction error: {:.3e} abs, {:.3e} rel",
            self.name,
            if self.pass { "PASS" } else { "FAIL" },
            self.hi_mismatches,
            self.lanes,
            self.lo_mismatches,
            self.lanes,
            self.max_lo_ulp_err,
            self.max_recon_abs,
            self.max_recon_rel,
        )
    }
}

/// THE probe: dispatch the shader on the live adapter and decide viability.
#[test]
fn double_single_eft_viability_on_metal() {
    let _serial = gpu_test_serial_guard();
    let device = require_verdict_device();

    let prod_pairs = build_product_inputs();
    let sum_pairs = build_sum_inputs();
    let n_prod = prod_pairs.len();
    let n_sum = sum_pairs.len();

    // Sanity-check the hand-picked adversarial case before trusting the harness:
    // (1 + 2^-12)^2 has true (hi, lo) = (1 + 2^-11, 2^-24).
    {
        let c = f32::from_bits(0x3F80_0800);
        let (hi, lo, exact) = true_two_product(c, c);
        assert_eq!(hi.to_bits(), 0x3F80_1000, "hi of (1+2^-12)^2 = 1 + 2^-11");
        assert_eq!(lo.to_bits(), 0x3380_0000, "lo of (1+2^-12)^2 = 2^-24");
        assert_eq!(
            f64::from(hi) + f64::from(lo),
            exact,
            "product EFT reconstructs exactly on CPU"
        );
        // And a sum reference: 2^24 + 1 -> hi = 2^24 (1 is below its ulp of 2), lo = 1.
        let (sh, sl, sexact) = true_two_sum(16777216.0, 1.0);
        assert_eq!(sh, 16777216.0);
        assert_eq!(sl, 1.0);
        assert_eq!(
            f64::from(sh) + f64::from(sl),
            sexact,
            "sum EFT reconstructs exactly on CPU"
        );
    }

    // CPU references.
    let prod_refs: Vec<(f32, f32, f64)> = prod_pairs
        .iter()
        .map(|&(a, b)| true_two_product(a, b))
        .collect();
    let sum_refs: Vec<(f32, f32, f64)> =
        sum_pairs.iter().map(|&(a, b)| true_two_sum(a, b)).collect();
    let mut oracle = Acc::new();
    for &(a, b) in &prod_pairs {
        oracle.add_prod(f64::from(a), f64::from(b));
    }
    let dot_ref = oracle.value();

    // Calibration: the UNCOMPENSATED plain-f32 serial fold, same order. This is the
    // "broken" level a double-single dot degrades to when its EFTs are folded away.
    let mut plain_f32_fold = 0.0f32;
    for &(a, b) in &prod_pairs {
        plain_f32_fold += a * b;
    }

    // Flatten the inputs to (a, b) pairs (WGSL vec2<f32>, tightly packed).
    let p_flat: Vec<f32> = prod_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let s_flat: Vec<f32> = sum_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();

    // ---- Build buffers + pipeline (its OWN shader; nothing shipped is touched) ----
    let params = Params {
        n_prod: n_prod as u32,
        n_sum: n_sum as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = create_buffer(
        &device.device,
        "ds_probe_params",
        size_of::<Params>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    device
        .queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

    let p_buf = create_buffer(
        &device.device,
        "ds_probe_p_in",
        (p_flat.len() * size_of::<f32>()) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    device
        .queue
        .write_buffer(&p_buf, 0, bytemuck::cast_slice(&p_flat));
    let s_buf = create_buffer(
        &device.device,
        "ds_probe_s_in",
        (s_flat.len() * size_of::<f32>()) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    device
        .queue
        .write_buffer(&s_buf, 0, bytemuck::cast_slice(&s_flat));

    let prod_bytes = (n_prod * 2 * size_of::<f32>()) as u64;
    let sum_bytes = (n_sum * 2 * size_of::<f32>()) as u64;
    // 6 double-single dot lanes (see the `dot_out` comment in the shader).
    const N_DOT_LANES: usize = 6;
    let dot_bytes = (N_DOT_LANES * 2 * size_of::<f32>()) as u64;
    let storage_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let prod_dekker_buf = create_buffer(
        &device.device,
        "ds_probe_prod_dekker",
        prod_bytes,
        storage_usage,
    );
    let prod_fma_buf = create_buffer(
        &device.device,
        "ds_probe_prod_fma",
        prod_bytes,
        storage_usage,
    );
    let sum_buf = create_buffer(&device.device, "ds_probe_sum", sum_bytes, storage_usage);
    let dot_buf = create_buffer(&device.device, "ds_probe_dot", dot_bytes, storage_usage);
    let sum_bar_buf = create_buffer(&device.device, "ds_probe_sum_bar", sum_bytes, storage_usage);

    // bindings 1.. : p_in(r) s_in(r) prod_dekker(rw) prod_fma(rw) sum(rw) dot(rw) sum_bar(rw)
    let (pipeline, layout) = device.create_simple_pipeline(
        DOUBLE_SINGLE_SHADER,
        "double_single_probe",
        &[false, false, true, true, true, true, true],
    );
    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ds_probe_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: s_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: prod_dekker_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: prod_fma_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: sum_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: dot_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: sum_bar_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ds_probe_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ds_probe_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let mk_staging = |label: &'static str, bytes: u64| {
        create_buffer(
            &device.device,
            label,
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )
    };
    let dekker_staging = mk_staging("ds_probe_dekker_staging", prod_bytes);
    let fma_staging = mk_staging("ds_probe_fma_staging", prod_bytes);
    let sum_staging = mk_staging("ds_probe_sum_staging", sum_bytes);
    let dot_staging = mk_staging("ds_probe_dot_staging", dot_bytes);
    let sum_bar_staging = mk_staging("ds_probe_sum_bar_staging", sum_bytes);
    encoder.copy_buffer_to_buffer(&prod_dekker_buf, 0, &dekker_staging, 0, prod_bytes);
    encoder.copy_buffer_to_buffer(&prod_fma_buf, 0, &fma_staging, 0, prod_bytes);
    encoder.copy_buffer_to_buffer(&sum_buf, 0, &sum_staging, 0, sum_bytes);
    encoder.copy_buffer_to_buffer(&dot_buf, 0, &dot_staging, 0, dot_bytes);
    encoder.copy_buffer_to_buffer(&sum_bar_buf, 0, &sum_bar_staging, 0, sum_bytes);
    device.queue.submit(std::iter::once(encoder.finish()));

    let dekker_bits = WgpuDevice::read_u32_buffer(&device.device, &dekker_staging, n_prod * 2)
        .expect("read Dekker product buffer");
    let fma_bits = WgpuDevice::read_u32_buffer(&device.device, &fma_staging, n_prod * 2)
        .expect("read FMA product buffer");
    let sum_bits = WgpuDevice::read_u32_buffer(&device.device, &sum_staging, n_sum * 2)
        .expect("read TwoSum buffer");
    let dot_bits = WgpuDevice::read_u32_buffer(&device.device, &dot_staging, N_DOT_LANES * 2)
        .expect("read dot buffer");

    let bar_bits = WgpuDevice::read_u32_buffer(&device.device, &sum_bar_staging, n_sum * 2)
        .expect("read fma-barrier TwoSum buffer");

    let dekker = evaluate_eft("Dekker split TwoProduct", &prod_refs, &dekker_bits);
    let fma = evaluate_eft("FMA TwoProduct lo=fma(a,b,-hi)", &prod_refs, &fma_bits);
    let two_sum_v = evaluate_eft("Knuth TwoSum", &sum_refs, &sum_bits);
    let two_sum_bar = evaluate_eft("fma-barrier TwoSum", &sum_refs, &bar_bits);

    // Reconstruct each double-single lane as f64(hi) + f64(lo).
    let ds_lane = |lane: usize| -> f64 {
        f64::from(f32::from_bits(dot_bits[2 * lane]))
            + f64::from(f32::from_bits(dot_bits[2 * lane + 1]))
    };
    let ds_dot_dekker = ds_lane(0);
    let ds_dot_fma = ds_lane(1);
    let ds_dot_bar_sum = ds_lane(2);
    let ds_dot_bar_fast = ds_lane(3);
    let ds_dot_barrier = ds_lane(4);
    let ds_dot_barrier_dekker = ds_lane(5);
    let rel = |v: f64| {
        if dot_ref != 0.0 {
            (v - dot_ref).abs() / dot_ref.abs()
        } else {
            (v - dot_ref).abs()
        }
    };

    // Viability: SOME TwoProduct variant is bit-exact AND SOME TwoSum variant
    // is bit-exact (a compensated dot needs BOTH the product and the sum EFT).
    let some_product = dekker.pass || fma.pass;
    let some_sum = two_sum_v.pass || two_sum_bar.pass;
    let viable = some_product && some_sum;
    let product_via = if dekker.pass && fma.pass {
        "Dekker AND fma"
    } else if dekker.pass {
        "Dekker split (no fma needed)"
    } else if fma.pass {
        "fma only"
    } else {
        "NEITHER"
    };

    println!(
        "\n================ DOUBLE-SINGLE EFT PROBE (live Metal/wgpu adapter) ================"
    );
    println!(
        "adapter: {} | backend: {:?}",
        device.adapter_info.name, device.adapter_info.backend
    );
    println!("product/dot lanes: {n_prod} | sum lanes: {n_sum} | Dot2 f64 oracle = {dot_ref:.17e}");
    println!("-- TwoProduct EFT (does the compiler CONTRACT a*b + c ?) --");
    println!("{}", dekker.report());
    println!("{}", fma.report());
    println!("-- TwoSum EFT (does the compiler REASSOCIATE the compensation to 0 ?) --");
    println!("{}", two_sum_v.report());
    println!("{}", two_sum_bar.report());
    println!("-- Running double-single dot vs compensated Dot2 f64 oracle --");
    println!(
        "    [PLAIN ds_add: two_sum + fast_two_sum]   dekker-product: {:.3e} | fma-product: {:.3e}",
        rel(ds_dot_dekker),
        rel(ds_dot_fma)
    );
    println!(
        "    [barrier TwoSum ONLY  ]  fma-product: {:.3e}\n    \
         [barrier FastTwoSum ONLY]  fma-product: {:.3e}",
        rel(ds_dot_bar_sum),
        rel(ds_dot_bar_fast),
    );
    println!(
        "    [FULLY barrier ds_add ]  fma-product: {:.3e} | dekker-product: {:.3e}",
        rel(ds_dot_barrier),
        rel(ds_dot_barrier_dekker),
    );
    // Does the PLAIN lane's hi word bit-equal the uncompensated host fold? If yes, the
    // legacy `ds_add` delivered LITERALLY zero compensation (both EFT terms folded away).
    let plain_hi = f32::from_bits(dot_bits[2]);
    let plain_lo = f32::from_bits(dot_bits[3]);
    println!(
        "    calibration: uncompensated host f32 serial fold rel-err: {:.3e} \
         | plain-lane hi bit-equals host fold: {} | plain-lane lo = {:.6e}",
        rel(f64::from(plain_f32_fold)),
        plain_hi.to_bits() == plain_f32_fold.to_bits(),
        plain_lo,
    );
    println!("    (working double-single ~2^-46 ≈ 1.4e-14; broken degrades to ~2^-24 ≈ 6e-8)");
    println!("--------------------------------------------------------------------------------");
    println!(
        "TwoProduct bit-exact via: {product_via} | TwoSum bit-exact: {}",
        if two_sum_v.pass {
            "YES (plain Knuth)"
        } else if two_sum_bar.pass {
            "YES (fma-barrier only)"
        } else {
            "NO (plain AND fma-barrier both destroyed)"
        }
    );
    println!(
        "VERDICT: double-single is {} on this Metal adapter.",
        if viable { "VIABLE" } else { "DEAD" }
    );
    if !viable {
        let reason = if !some_product {
            "no TwoProduct variant survives (the compiler contracts a*b + c AND fma is not usable)"
        } else if !some_sum {
            "every TwoSum variant is destroyed (fast-math reassociation zeroes the plain \
             compensation term, and the driver canonicalizes the fma-barrier form back to \
             subtractions), so a compensated dot/matmul gains no precision even though fma \
             TwoProduct is exact"
        } else {
            "unknown"
        };
        println!("REASON: {reason}.");
    }
    println!("================================================================================\n");

    // This is a MEASUREMENT, not a feature under test: the probe succeeds when it runs
    // and produces its verdict, so the suite stays green. The asserts below pin the
    // CURRENT measured behavior of THIS Metal adapter — they are REGRESSION TRIPWIRES.
    // If any ever fails, the Metal WGSL compiler's EFT behavior CHANGED and the whole
    // double-single direction must be re-evaluated (the message says exactly how).

    // (1) POSITIVE capability that IS solid on Metal: fma is a true single-rounding
    // fused multiply-add, so the fma-based TwoProduct is bit-exact. (If this trips,
    // even the product EFT died — double-single is even more dead.)
    assert!(
        fma.pass,
        "TRIPWIRE: the fma-based TwoProduct is no longer bit-exact on this Metal adapter \
         ({} lo mismatches / {} lanes, recon {:.3e}) — WGSL `fma` is no longer a true fused op.",
        fma.lo_mismatches, fma.lanes, fma.max_recon_abs
    );

    // (2) The Dekker SPLIT product is broken by a*b+c CONTRACTION (measured today)
    // — a statement about the METAL shader compiler specifically, so it is only
    // evidence when this adapter IS Metal. Tripwire (1) is a hardware property of
    // `fma` and (3) was measured "GB10 Vulkan + Metal alike", but contraction of
    // `a*b + c` is a per-compiler choice: on this Vulkan/AMD adapter the Dekker
    // split IS bit-exact, which fired this tripwire and told the reader to
    // "re-open the split-based double-single path" on the strength of a
    // measurement from the wrong compiler. Report it everywhere, assert it only
    // where its premise holds.
    if device.adapter_info.backend == wgpu::Backend::Metal {
        assert!(
            !dekker.pass,
            "TRIPWIRE (good news): the Dekker split TwoProduct is now bit-exact — the Metal \
             compiler stopped contracting a*b + c. Re-open the split-based double-single path."
        );
    } else {
        println!(
            "NOTE: Dekker split TwoProduct {} on {:?} — the (2) tripwire is Metal-specific \
             and is not asserted here.",
            if dekker.pass {
                "is BIT-EXACT"
            } else {
                "is broken by contraction"
            },
            device.adapter_info.backend
        );
    }

    // (3) The PLAIN Knuth TwoSum is broken by fast-math REASSOCIATION (measured on
    // GB10 Vulkan + Metal): the compiler folds the algebraically-zero compensation
    // term. Any EFT shader must therefore use the fma-barrier form, never the plain one.
    //
    // Gated for the same reason as (2), and the same rule applies across this probe:
    // a POSITIVE capability — (1) fma is a true fused op, (4) the fma-barrier form is
    // exact — is asserted everywhere, because the EFT channel depends on it wherever
    // it runs. A "this is BROKEN here" observation is a statement about one shader
    // COMPILER and is asserted only where it was measured. AMD's Vulkan compiler does
    // not reassociate this term, so on this adapter the plain form is exact and the
    // tripwire fired for an adapter it never described. Nothing is unsound either
    // way: the fma-barrier form stays required by construction and remains
    // sufficient on every backend.
    if device.adapter_info.backend == wgpu::Backend::Metal {
        assert!(
            !two_sum_v.pass,
            "TRIPWIRE: the PLAIN Knuth TwoSum EFT is now bit-exact on this adapter — the \
             compiler no longer reassociates the compensation term to zero. The fma-barrier \
             form is then no longer required (but remains sufficient); update the EFT design."
        );
    } else {
        println!(
            "NOTE: plain Knuth TwoSum {} on {:?} — the (3) tripwire pins GB10-Vulkan/Metal \
             compiler behavior and is not asserted here.",
            if two_sum_v.pass {
                "is BIT-EXACT"
            } else {
                "is broken by reassociation"
            },
            device.adapter_info.backend
        );
    }

    // (4) THE load-bearing POSITIVE result (measured 2026-07-23, GB10/Vulkan): the
    // fma-barrier TwoSum — every subtraction of the Knuth sequence routed through the
    // fma intrinsic — IS bit-exact: the driver does not canonicalize fma(-1, x, y)
    // back to a subtraction and then reassociate. Together with the bit-exact fma
    // TwoProduct this makes BOTH EFTs available on device, which is the foundation of
    // (a) the EFT-compensated certified-error channel
    // (docs/EFT_COMPENSATED_CERTIFIED_ERROR_DESIGN.md) and (b) full double-single
    // (df64-class) sound GPU arithmetic. If THIS trips, a driver/naga update started
    // canonicalizing through fma intrinsics — the EFT shader channel must fail closed
    // back to the Higham bound (its own on-device self-test does exactly that; this
    // tripwire is the early warning).
    assert!(
        two_sum_bar.pass,
        "TRIPWIRE (bad news): the fma-barrier TwoSum is no longer bit-exact on this \
         adapter ({} lo mismatches / {} lanes) — the driver now canonicalizes through \
         fma intrinsics. The EFT compensated channel's device self-test must be refusing; \
         verify it does, and re-pin this probe.",
        two_sum_bar.lo_mismatches, two_sum_bar.lanes
    );

    // (5) The make-or-break conclusion for THIS adapter: VIABLE (product EFT via fma,
    // sum EFT via the fma-barrier form).
    assert!(
        viable,
        "TRIPWIRE (bad news): double-single stopped being viable on this adapter \
         (product via {product_via}). Re-evaluate the whole EFT/double-single direction."
    );

    // (6) THE COMPOSED result (measured 2026-08-04, Apple M5 Max / Metal): with BOTH
    // EFTs in the fma-barrier form, the 509-term double-single dot reaches df64 class
    // (measured 3.493e-14 ≈ 2^-44.8) instead of the plain-f32 1.215e-7. This is the
    // corroboration the per-lane verdict was missing: the EFTs survive not only in
    // isolation but COMPOSED into a running accumulator, across 509 loop iterations,
    // with the compiler free to reassociate the whole chain. The threshold is set two
    // orders above the measured value and four below the broken level, so it cannot be
    // satisfied by an accidentally-uncompensated fold.
    let barrier_rel = rel(ds_dot_barrier);
    assert!(
        barrier_rel < 1e-12,
        "TRIPWIRE (bad news): the FULLY fma-barrier double-single dot no longer composes \
         on this adapter (rel-err {barrier_rel:.3e}, expected < 1e-12; plain-f32 level is \
         {:.3e}). The compiler started folding the compensation ACROSS the accumulator \
         even through fma intrinsics — double-single value arithmetic is dead here again.",
        rel(f64::from(plain_f32_fold))
    );

    // (7) DISCRIMINATION, the other half of (6): the PLAIN `ds_add` lane must stay at the
    // uncompensated level. If this ever tightens, the probe stopped discriminating and
    // (6) would pass vacuously. Measured today: the plain lane is 1.215e-7, BIT-EQUAL to
    // the host's uncompensated f32 serial fold — i.e. the legacy dot lanes deliver
    // literally ZERO compensation, which is precisely why both of them printed the same
    // number regardless of TwoProduct variant.
    let plain_rel = rel(ds_dot_fma);
    assert!(
        plain_rel > 1e-9,
        "TRIPWIRE (good news): the PLAIN-`ds_add` double-single dot now compensates \
         (rel-err {plain_rel:.3e}) — the compiler stopped folding the pure-adds EFTs. \
         Re-pin (6)'s discrimination and revisit whether the fma-barrier discipline is \
         still required."
    );

    // (8) NEITHER barrier form ALONE is sufficient — the design constraint that both the
    // Knuth TwoSum AND the Dekker FastTwoSum renormalization must be barriered. Measured:
    // barrier-TwoSum-only 1.215e-7 (no gain at all), barrier-FastTwoSum-only 9.012e-8.
    // `fast_two_sum` is pure adds and had never been probed; it is the second
    // reassociation-sensitive site, and on its own it destroys the whole compensation.
    // NOTE (checked 2026-08-04): no shipped shader contains a FastTwoSum — the production
    // EFT channel accumulates residual MAGNITUDES (`r += |ep| + |es|`) and never
    // renormalizes a double-single value, so this constraint binds the df64 VALUE
    // direction only, not the certified-error channel already in tree.
    assert!(
        rel(ds_dot_bar_sum) > 1e-9 && rel(ds_dot_bar_fast) > 1e-9,
        "TRIPWIRE (good news): a SINGLE barriered EFT now suffices for the composed dot \
         (barrier-TwoSum-only {:.3e}, barrier-FastTwoSum-only {:.3e}) — the both-must-be-\
         barriered constraint relaxed on this adapter; re-derive the df64 requirements.",
        rel(ds_dot_bar_sum),
        rel(ds_dot_bar_fast)
    );
}
