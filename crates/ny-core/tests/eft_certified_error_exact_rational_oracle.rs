// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SOUNDNESS PROOF ARTIFACT for S2 — the EFT-compensated certified-error
//! channel (`docs/CONV_CROWN_WALL_DESIGN_2026-07-27.md` §S2,
//! `docs/EFT_COMPENSATED_CERTIFIED_ERROR_DESIGN.md`).
//!
//! S2 remains a **NO-GO for arming on this host** (the a-priori Higham share of
//! the CPU CROWN charge is 0.5–1.6%, not the claimed 10⁴×, because the CPU conv
//! backward already f64-recomputes every coefficient — the 10⁴× tax lives on the
//! f32-ACCUMULATED paths, i.e. the GPU fold and the disabled CPU f32 fast path —
//! and f64 accumulation strictly dominates EFT at 1.02–1.33× wall vs 4.08×).
//! This file is the artifact that must exist *before* anyone re-arms the channel
//! on a host where it does pay: an **exact-rational differential oracle that
//! CATCHES a non-enclosing EFT error bound**. As of 2026-08-02 it also gates the
//! two production defects it found, both now fixed, and the production
//! downgrade-only combinator.
//!
//! # What is being proved
//!
//! The channel's contract is a certified **radius**, not a bound:
//!
//! ```text
//!   |Σ aᵢ·bᵢ  −  value|  ≤  err          (value, err : f32; the sum is EXACT)
//! ```
//!
//! Ordinary float assertions cannot check this, because "close in f32" is
//! precisely the property under test. Every enclosure claim here is decided in
//! exact arithmetic: each finite f32 is an integer multiple of `2^-149`, so a
//! product of two f32s is an integer multiple of `2^-298`, and the whole dot
//! product is carried as a `BigInt` count of `2^-298` units — *lossless*, then
//! surfaced as `BigRational` for reporting. [`exact_dot_is_the_naive_rational_dot`]
//! pins the fast integer kernel against the naive `BigRational` fold.
//!
//! # Why this file has teeth (and the last one did not)
//!
//! A soundness test that passes against a broken implementation is worse than
//! no test: this campaign shipped one and had to delete it. So the artifact is
//! built as a **differential** over a mutable twin:
//!
//! 1. [`ref_dot`] is a local re-implementation of the channel with an
//!    injection knob.
//! 2. [`reference_is_bit_identical_to_production`] proves
//!    `ref_dot(.., Inject::None)` is **bit-identical** to
//!    `ny_core::eft::eft_dot_f32` — value AND err — over the whole adversarial
//!    corpus and a 512-case proptest. The twin is therefore a faithful stand-in.
//! 3. [`oracle_catches_every_injected_undercharge`] mutates that twin (a
//!    one-ULP `next_down` on the certified radius; a zeroed residual channel;
//!    dropped product/sum residuals; a missing outward round; the pre-fix
//!    underflow guard) and asserts the oracle **finds a violation for each**.
//!    If any mutant survived, the oracle would be decoration.
//!
//! The one-ULP demonstration is not left to luck. [`teeth_case`] is a
//! hand-built 2050-term fold whose residuals are *all the same sign* and whose
//! exact residual sum `R = 2⁻¹³ + 2⁻²³ + 2⁻³⁶ + 2⁻⁴⁰` sits `2⁻⁴⁰` above an f32
//! grid point, so the certified radius is the f32 **immediately above** the
//! true error and a single `next_down` is already unsound. See the constant's
//! doc-comment for the full derivation.
//!
//! # Live defects this artifact caught — BOTH FIXED 2026-08-02
//!
//! * `eft_dot_f32`'s TwoProd exactness guard was `|p| < 2^-126`
//!   (`PROD_UNDERFLOW_FLOOR_F32`). The TwoProdFMA exactness theorem needs
//!   `e_a + e_b ≥ e_min + p − 1 = −103`, i.e. `|p| ≥ 2^-101` to be safe on the
//!   product alone. In the gap `[2^-126, 2^-102)` the residual `fma(a,b,−p)`
//!   is ITSELF rounded, the leak was never charged, and the certified radius
//!   did not enclose. The fix is a one-constant raise of the *guard* to
//!   `ny_core::eft::TWO_PROD_EXACT_FLOOR_F32`; the charged floor stays `2^-126`,
//!   which remains valid since `|p| < 2^-101` ⇒ `|a·b − p| ≤ 2^-126`. The old
//!   guard survives here as the `Inject::LegacyUnderflowGuard` **mutant**, and
//!   [`oracle_catches_every_injected_undercharge`] proves the oracle still
//!   catches it — that is what keeps the fix from silently regressing.
//! * the comparator arm `higham_dot_err_f32` charged `γ·Σ|aᵢbᵢ|`, a purely
//!   RELATIVE model. Under gradual underflow the rounding error is ABSOLUTE
//!   (up to `2^-150`), so the Higham radius was non-enclosing there. This
//!   matters for the combinator: `min(E_eft, E_higham)` is sound **iff both
//!   arms are sound**, and it preferentially publishes the smaller — i.e. the
//!   broken — arm. The arm now carries a `4n·2^-149` absolute floor and refuses
//!   (`+inf`) instead of under-charging a degenerate `γ`;
//!   [`higham_arm_encloses_under_underflow`] is the regression gate.
//!
//! # Downgrade-only
//!
//! The design brief's "`max(higham, eft)`" is direction-ambiguous. On radii the
//! downgrade-only rule is `min`, not `max`
//! ([`max_on_radii_is_sound_but_recovers_nothing`] shows `max` is a strict
//! no-op that throws away the entire point of S2). The combinator under test is
//! now the PRODUCTION one, `ny_core::eft::combine_downgrade_only` — the same
//! rule the GPU twin ships (`CROWN_EFT_MIN_COMBINE_SHADER`):
//! `err = min(err_higham, round_up(err_eft))`. [`combinator_is_downgrade_only`]
//! asserts both halves: never tighter than the truth, never worse than pure
//! Higham.

#![allow(clippy::float_cmp)]

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use proptest::prelude::*;

use ny_core::eft::{
    combine_downgrade_only, eft_dot_f32, eft_dot_f32_downgrade_only, eft_self_check,
    higham_dot_err_f32, two_prod_f32, two_sum_f32, EftDot, PROD_UNDERFLOW_FLOOR_F32,
    TWO_PROD_EXACT_FLOOR_F32,
};

// ---------------------------------------------------------------------------
// Exact oracle
// ---------------------------------------------------------------------------

/// Every finite f32 is an integer multiple of `2^-149` (the smallest
/// subnormal). This returns that integer count — lossless, by construction.
fn f32_units(x: f32) -> BigInt {
    assert!(x.is_finite(), "oracle is defined on finite f32 only");
    let bits = x.to_bits();
    let neg = bits >> 31 == 1;
    let raw_exp = (bits >> 23) & 0xff;
    let frac = bits & 0x007f_ffff;
    // normal:    (2^23 + frac) · 2^(raw_exp − 150)  ⇒  units = (2^23+frac)·2^(raw_exp−1)
    // subnormal:  frac · 2^-149                     ⇒  units = frac
    let mut u = if raw_exp == 0 {
        BigInt::from(frac)
    } else {
        BigInt::from(0x0080_0000u32 + frac) << (raw_exp - 1)
    };
    if neg {
        u = -u;
    }
    u
}

/// `2^-298` units: the exact scale of a product of two f32s.
const PROD_SCALE_BITS: u32 = 298;

/// Exact `Σ aᵢ·bᵢ` in units of `2^-298`. Integer arithmetic throughout, so a
/// 20 000-term fold costs microseconds instead of a `BigRational` gcd storm.
fn exact_dot_units(a: &[f32], b: &[f32]) -> BigInt {
    assert_eq!(a.len(), b.len());
    let mut acc = BigInt::zero();
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += f32_units(x) * f32_units(y);
    }
    acc
}

/// An f32 lifted into the same `2^-298` units.
fn f32_prod_units(x: f32) -> BigInt {
    f32_units(x) << 149
}

/// The exact rational value of `Σ aᵢ·bᵢ` — the reporting face of the oracle.
fn exact_dot(a: &[f32], b: &[f32]) -> BigRational {
    BigRational::new(exact_dot_units(a, b), BigInt::from(1u8) << PROD_SCALE_BITS)
}

/// Lossless rational for one finite f32.
fn exact_f32(x: f32) -> BigRational {
    BigRational::new(f32_units(x), BigInt::from(1u8) << 149u32)
}

fn next_up_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let b = x.to_bits();
    f32::from_bits(if x > 0.0 { b + 1 } else { b - 1 })
}

fn next_down_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f32::from_bits(1);
    }
    let b = x.to_bits();
    f32::from_bits(if x > 0.0 { b - 1 } else { b + 1 })
}

/// Independent f64 TwoSum used by the mutable twin's directed reduction.
/// Keep this local rather than importing the production helper: the oracle must
/// bind production behavior without making a shared implementation bug tautological.
fn ref_two_sum_f64(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let residual = (a - (sum - b_virtual)) + (b - b_virtual);
    (sum, residual)
}

/// Independent successor operation for the non-negative f64 residual sum.
fn ref_next_up_f64(x: f64) -> f64 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude >= f64::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Add one non-negative certificate term with the result directed toward +inf.
/// The sign of TwoSum's exact residual says whether round-to-nearest landed below
/// the real sum; only that case needs the successor.
fn ref_add_nonnegative_f64_up(accumulator: f64, term: f64) -> f64 {
    debug_assert!(accumulator >= 0.0);
    debug_assert!(term >= 0.0);
    let (sum, residual) = ref_two_sum_f64(accumulator, term);
    if residual > 0.0 {
        ref_next_up_f64(sum)
    } else {
        sum
    }
}

/// Publish an already upward-directed f64 certificate in f32. `outward = false`
/// is the deliberate `NoOutwardRounding` mutant; every honest path takes the
/// successor when the round-to-nearest cast landed below the f64 term.
fn ref_publish_directed_err_up_f32(term: f64, outward: bool) -> Option<f32> {
    if !term.is_finite() || term < 0.0 {
        return None;
    }
    let mut err = term as f32;
    if outward && f64::from(err) < term {
        err = f32::from_bits(err.to_bits() + 1);
    }
    err.is_finite().then_some(err)
}

// ---------------------------------------------------------------------------
// The contract under test
// ---------------------------------------------------------------------------

/// One certified dot: a value and the RADIUS around it that must contain the
/// exact rational sum.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Certified {
    value: f32,
    err: f32,
}

impl From<EftDot> for Certified {
    fn from(d: EftDot) -> Self {
        Self {
            value: d.value,
            err: d.err,
        }
    }
}

/// How badly a certified pair misses the exact value, or `None` when it
/// encloses. A `Some(_)` here is a MOAT VIOLATION: ny published a radius that
/// does not contain the truth.
fn enclosure_excess(a: &[f32], b: &[f32], c: Certified) -> Option<BigRational> {
    // A non-finite radius is a refusal, not a claim; a non-finite value can
    // never be checked. Both are fail-closed by construction upstream.
    if !c.value.is_finite() || !c.err.is_finite() {
        return None;
    }
    if c.err < 0.0 {
        return Some(BigRational::zero() - exact_f32(c.err));
    }
    let exact = exact_dot_units(a, b);
    let value = f32_prod_units(c.value);
    let err = f32_prod_units(c.err);
    let diff = (exact - value).abs();
    if diff > err {
        Some(BigRational::new(
            diff - err,
            BigInt::from(1u8) << PROD_SCALE_BITS,
        ))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// The mutable twin
// ---------------------------------------------------------------------------

/// Deliberate defects injected into [`ref_dot`]. `Inject::None` is NOT a defect
/// knob — it reproduces production byte-for-byte and is the binding that makes
/// every other mutant meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Inject {
    /// The pre-2026-08-02 defect, retained as a MUTANT: TwoProd exactness guard
    /// lowered back to `2^-126`, where the residual is itself rounded.
    LegacyUnderflowGuard,
    /// Bit-identical to `ny_core::eft::eft_dot_f32` (guard at `2^-101`).
    None,
    /// The required demonstration: shave exactly one ULP off the radius.
    OneUlpUnderCharge,
    /// The characteristic EFT failure — a reassociating/constant-folding
    /// compiler collapses TwoSum's algebraically-zero residual to `0.0`. The
    /// result is not LOOSE, it is WRONG, and `min` publishes it preferentially.
    ZeroResidual,
    /// Charge only the summation residuals (forget TwoProd).
    DropProductResiduals,
    /// Charge only the product residuals (forget TwoSum).
    DropSumResiduals,
    /// Cast the f64 residual sum to f32 with round-to-NEAREST and skip the
    /// outward `next_up` — a one-line "cleanup" that silently breaks enclosure.
    NoOutwardRounding,
}

/// A local re-implementation of `ny_core::eft::eft_dot_f32` with an injection
/// knob. `Inject::None` must stay bit-identical to production —
/// [`reference_is_bit_identical_to_production`] is the gate.
fn ref_dot(a: &[f32], b: &[f32], inject: Inject) -> Option<Certified> {
    assert_eq!(a.len(), b.len());
    let guard = match inject {
        Inject::LegacyUnderflowGuard => PROD_UNDERFLOW_FLOOR_F32,
        _ => TWO_PROD_EXACT_FLOOR_F32,
    };
    let mut acc = 0.0f32;
    let mut resid = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (p, ep) = two_prod_f32(x, y);
        if !p.is_finite() {
            return None;
        }
        if x != 0.0 && y != 0.0 && p.abs() < guard {
            if inject != Inject::DropProductResiduals {
                resid = ref_add_nonnegative_f64_up(resid, f64::from(PROD_UNDERFLOW_FLOOR_F32));
            }
        } else if inject != Inject::DropProductResiduals {
            resid = ref_add_nonnegative_f64_up(resid, f64::from(ep).abs());
        }
        let (s, es) = two_sum_f32(acc, p);
        if !s.is_finite() {
            return None;
        }
        if inject != Inject::DropSumResiduals {
            resid = ref_add_nonnegative_f64_up(resid, f64::from(es).abs());
        }
        acc = s;
    }
    if inject == Inject::ZeroResidual {
        resid = 0.0;
    }
    if !resid.is_finite() {
        return None;
    }
    let mut err = ref_publish_directed_err_up_f32(resid, inject != Inject::NoOutwardRounding)?;
    if inject == Inject::OneUlpUnderCharge {
        err = next_down_f32(err);
        if err < 0.0 {
            err = 0.0;
        }
    }
    Some(Certified { value: acc, err })
}

// ---------------------------------------------------------------------------
// Adversarial corpus
// ---------------------------------------------------------------------------

struct Case {
    name: &'static str,
    a: Vec<f32>,
    b: Vec<f32>,
}

/// xorshift64* — deterministic, so every failure in this file reproduces.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 1_000_003) as f32 / 1_000_003.0
    }
    fn sign(&mut self) -> f32 {
        if self.next() & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

/// THE one-ULP teeth case. Built so the exact residual sum sits a hair above an
/// f32 grid point and every residual carries the SAME sign, making the
/// certified radius the immediate f32 successor of the true error.
///
/// * terms 0..2048: `a = b = 1 + 2⁻¹²`. Then `p = 1 + 2⁻¹¹` exactly and
///   `e_prod = +2⁻²⁴` (this is the module's own FMA-fusedness probe). The
///   running sum `k·(1 + 2⁻¹¹) = k + k·2⁻¹¹` needs ≤ 22 significant bits for
///   `k ≤ 2048`, so every TwoSum residual is exactly `0`. After 2048 terms
///   `value = 2049` exactly and `Σ|e_prod| = 2048·2⁻²⁴ = 2⁻¹³`.
/// * term 2048: `a = 1 + 2⁻¹²`, `b = 2⁻¹²·(1 + 2⁻¹²)`. The exact product is
///   `2⁻¹²(1 + 2⁻¹¹ + 2⁻²⁴)`; the `2⁻²⁴` relative tail is exactly half an ULP
///   and ties-to-even rounds DOWN, so `p = 2⁻¹² + 2⁻²³` and `e_prod = +2⁻³⁶`.
///   `ulp(2049) = 2⁻¹²`, so the sum drops the `2⁻²³` tail: `e_sum = +2⁻²³`.
/// * term 2049: `a = b = 2⁻²⁰`. `p = 2⁻⁴⁰` exactly (`e_prod = 0`), and
///   `2⁻⁴⁰ ≪ ulp(2049 + 2⁻¹²)/2`, so the whole addend survives as
///   `e_sum = +2⁻⁴⁰`.
///
/// All residuals positive ⇒ true error `= R = 2⁻¹³ + 2⁻²³ + 2⁻³⁶ + 2⁻⁴⁰`
/// `= 2⁻¹³·(1 + 2⁻¹⁰ + 2⁻²³ + 2⁻²⁷)`. The `2⁻²⁷` relative bit is past f32's 23,
/// so `R` is NOT an f32: it lies `2⁻⁴⁰` above its f32 predecessor `L`. The
/// channel therefore publishes `err = next_up(L)`, and `next_down(err) = L < R`.
/// A single ULP of under-charge is already non-enclosing.
fn teeth_case() -> Case {
    let eps = f32::from_bits(0x3980_0000); // 2^-12
    let one_p = 1.0f32 + eps;
    let mut a = vec![one_p; 2048];
    let mut b = vec![one_p; 2048];
    a.push(one_p);
    b.push(eps * one_p);
    a.push(f32::from_bits(0x35800000)); // 2^-20
    b.push(f32::from_bits(0x35800000));
    Case {
        name: "teeth_tight_same_sign_residuals",
        a,
        b,
    }
}

fn adversarial_corpus() -> Vec<Case> {
    let mut out = Vec::new();
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);

    // 1. The CROWN regime: mixed-sign coefficients, large |A|·|W| mass, small
    //    running sum. This is where the a-posteriori channel is supposed to win.
    for &n in &[64usize, 1024, 4096] {
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for _ in 0..n {
            a.push(rng.sign() * (0.5 + 1.5 * rng.unit()));
            b.push(rng.sign() * (0.5 + 1.5 * rng.unit()));
        }
        out.push(Case {
            name: "crown_cancellation_mixed_sign",
            a,
            b,
        });
    }

    // 2. Catastrophic cancellation: exact ± pairs, so the true sum is 0 while
    //    Σ|aᵢbᵢ| is enormous. The a-priori bound is maximally wrong here.
    {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..2048 {
            let x = rng.sign() * (1.0 + 1e6 * rng.unit());
            let y = rng.sign() * (1.0 + 1e6 * rng.unit());
            a.push(x);
            b.push(y);
            a.push(x);
            b.push(-y);
        }
        out.push(Case {
            name: "catastrophic_cancellation_exact_pairs",
            a,
            b,
        });
    }

    // 3. Magnitudes spanning ~60 orders inside one fold.
    {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for k in 0..3000i32 {
            let e = (k % 61) - 30;
            a.push(rng.sign() * (1.0 + rng.unit()) * 10.0f32.powi(e));
            b.push(rng.sign() * (1.0 + rng.unit()) * 10.0f32.powi(-e / 2));
        }
        out.push(Case {
            name: "mixed_magnitudes_60_orders",
            a,
            b,
        });
    }

    // 4. Near-overflow: the fold's intermediates brush f32::MAX. The channel
    //    must either enclose or REFUSE (None) — never publish a finite lie.
    {
        let big = f32::MAX / 4.0;
        out.push(Case {
            name: "near_overflow_alternating",
            a: vec![big, big, -big, big, -big, big],
            b: vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.9],
        });
        out.push(Case {
            name: "near_overflow_products",
            a: vec![1e30, 1e30, -1e30, 3.4e38],
            b: vec![1e8, 1e8, 1e8, 1.9],
        });
    }

    // 5. Subnormal mesh: operands and products at and below 2^-126, where the
    //    relative-error model has no force at all.
    {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for k in 1..=64u32 {
            a.push(f32::from_bits(k)); // k · 2^-149
            b.push(1.0 + rng.unit());
            a.push(f32::MIN_POSITIVE * (1.0 + rng.unit()));
            b.push(f32::from_bits(0x0000_0003));
            a.push(1e-30);
            b.push(1e-20);
            a.push(-1e-30);
            b.push(1e-20);
        }
        out.push(Case {
            name: "subnormal_mesh",
            a,
            b,
        });
    }

    // 6. Deep-conv row length with realistic conv magnitudes (3×3×1024 = 9216
    //    taps is a real ResNet row; 20 001 exercises the long-n residual sum).
    for &n in &[9216usize, 20_001] {
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for _ in 0..n {
            a.push(rng.sign() * 0.3 * rng.unit());
            b.push(rng.sign() * (0.01 + 0.2 * rng.unit()));
        }
        out.push(Case {
            name: "deep_conv_row",
            a,
            b,
        });
    }

    // 7. All-positive long fold: no cancellation anywhere, so the residual sum
    //    is at its most load-bearing.
    {
        let n = 20_001;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for _ in 0..n {
            a.push(0.5 + rng.unit());
            b.push(0.5 + rng.unit());
        }
        out.push(Case {
            name: "long_all_positive",
            a,
            b,
        });
    }

    // 8. Zeros, signed zeros, and 1-ULP neighbours.
    {
        out.push(Case {
            name: "zeros_and_ulp_neighbours",
            a: vec![
                0.0,
                -0.0,
                1.0,
                next_up_f32(1.0),
                -1.0,
                f32::from_bits(1),
                0.0,
            ],
            b: vec![
                1e30,
                1e30,
                next_up_f32(1.0),
                1.0,
                next_down_f32(1.0),
                1.0,
                -0.0,
            ],
        });
    }

    out.push(teeth_case());
    out
}

/// The family that broke the pre-2026-08-02 channel, kept as its own generator
/// so the fix has a named witness. Products land in `[2^-126, 2^-101)`: NORMAL,
/// so the old `f32::MIN_POSITIVE` guard let them through, but below the
/// TwoProdFMA exactness threshold, so the residual is itself rounded. Production
/// now diverts exactly this band to the charged floor.
fn tiny_normal_product_cases() -> Vec<Case> {
    let mut out = Vec::new();

    // The minimal witness: one term. p is the smallest NORMAL f32 grid step
    // above 2^-126, and the exact product carries a 2^-172 tail that no
    // representable residual can hold.
    let a = 1.0f32 + f32::from_bits(0x3400_0000); // 1 + 2^-23
    let b = f32::MIN_POSITIVE * (1.0f32 + f32::from_bits(0x3400_0000));
    out.push(Case {
        name: "tiny_normal_product_single_term",
        a: vec![a],
        b: vec![b],
    });

    // A swept family across the whole [2^-126, 2^-102) gap.
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut av = Vec::new();
    let mut bv = Vec::new();
    for _ in 0..4096 {
        let ea = -(20 + (rng.next() % 80) as i32);
        let eb = -126 - ea + 1 + (rng.next() % 24) as i32;
        let ma = 1.0f32 + (rng.next() % (1 << 23)) as f32 / 8_388_608.0;
        let mb = 1.0f32 + (rng.next() % (1 << 23)) as f32 / 8_388_608.0;
        let x = ma * 2.0f32.powi(ea);
        let y = mb * 2.0f32.powi(eb);
        if !x.is_finite() || !y.is_finite() || x == 0.0 || y == 0.0 {
            continue;
        }
        let p = x * y;
        if !(p.abs() >= f32::MIN_POSITIVE && p.abs() < TWO_PROD_EXACT_FLOOR_F32) {
            continue;
        }
        av.push(x);
        bv.push(y);
    }
    out.push(Case {
        name: "tiny_normal_product_sweep",
        a: av,
        b: bv,
    });
    out
}

/// Run a channel over a corpus; return one line per enclosure violation.
fn scan(chan: impl Fn(&[f32], &[f32]) -> Option<Certified>, corpus: &[Case]) -> Vec<String> {
    let mut bad = Vec::new();
    for c in corpus {
        let Some(cert) = chan(&c.a, &c.b) else {
            continue; // a refusal is fail-closed, not a claim
        };
        if let Some(excess) = enclosure_excess(&c.a, &c.b, cert) {
            bad.push(format!(
                "{} (n={}): value={:e} err={:e} — exact escapes the certified \
                 radius by {:e} ({} in exact rationals)",
                c.name,
                c.a.len(),
                cert.value,
                cert.err,
                rational_to_f64(&excess),
                excess,
            ));
        }
    }
    bad
}

fn rational_to_f64(r: &BigRational) -> f64 {
    let (n, d) = (r.numer(), r.denom());
    let (nf, df) = (bigint_to_f64(n), bigint_to_f64(d));
    nf / df
}

fn bigint_to_f64(v: &BigInt) -> f64 {
    // Enough for reporting; the ASSERTIONS never route through f64.
    v.to_string().parse::<f64>().unwrap_or(f64::INFINITY)
}

// ---------------------------------------------------------------------------
// 0. The oracle itself must be exact, or every test below is decoration
// ---------------------------------------------------------------------------

#[test]
fn oracle_is_lossless_on_representative_f32() {
    for x in [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.1,
        1.0 / 3.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x007f_ffff), // largest subnormal
        f32::MAX,
        -f32::MAX,
        1e-30,
        3.402_823_5e38,
    ] {
        let r = exact_f32(x);
        // Round-trip through the units representation.
        let back = BigRational::new(f32_units(x), BigInt::from(1u8) << 149u32);
        assert_eq!(r, back, "exact_f32({x:e}) is not stable");
        if x != 0.0 {
            assert_eq!(r.is_negative(), x < 0.0, "sign lost for {x:e}");
        }
        // Doubling in the rational domain must match doubling in f32 (exact).
        if x != 0.0 && (2.0 * x).is_finite() {
            assert_eq!(
                exact_f32(2.0 * x),
                r * BigRational::from_integer(BigInt::from(2u8)),
                "scale-by-2 is not exact for {x:e}"
            );
        }
    }
}

#[test]
fn exact_dot_is_the_naive_rational_dot() {
    // The fast BigInt kernel must agree with the textbook BigRational fold.
    let mut rng = Rng(0xDEAD_BEEF_CAFE_1234);
    for _ in 0..64 {
        let n = 1 + (rng.next() % 40) as usize;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for _ in 0..n {
            a.push(rng.sign() * (rng.unit() * 1e6 + f32::MIN_POSITIVE));
            b.push(rng.sign() * (rng.unit() * 1e-6 + f32::from_bits(7)));
        }
        let mut naive = BigRational::zero();
        for (&x, &y) in a.iter().zip(b.iter()) {
            naive += exact_f32(x) * exact_f32(y);
        }
        assert_eq!(exact_dot(&a, &b), naive, "the integer kernel is not exact");
    }
}

#[test]
fn eft_preconditions_hold_on_this_target() {
    assert_eq!(
        eft_self_check(),
        Ok(()),
        "f32 error-free transformations are broken on this target; the \
         compensated certified-error channel would be silently unsound"
    );
}

/// Obligation A from the S2 review, now LANDED: `eft_self_check` carries
/// `#[inline(never)]` + `black_box` on every probe operand and is cached behind
/// `eft_available`, exactly as `dd_selfcheck::run_probes` is. Without the
/// barriers the probe operands are compile-time constants, so a constant-folding
/// pass can evaluate the probe with exact semantics while the runtime kernel is
/// reassociated — probe passes, channel broken. This test pins that the barrier
/// form gives the same answers, so a future "cleanup" that removes them is
/// visible rather than silent.
#[test]
fn self_check_probes_survive_optimization_barriers() {
    use std::hint::black_box;
    let a = black_box(1.0f32 + f32::from_bits(0x3980_0000)); // 1 + 2^-12
    let (p, e) = two_prod_f32(black_box(a), black_box(a));
    assert_eq!(
        e,
        f32::from_bits(0x3380_0000),
        "FMA fusedness probe changed under a black_box barrier"
    );
    assert_eq!(black_box(a * a) - black_box(p), 0.0);

    let tiny = black_box(f32::from_bits(0x0D80_0000)); // 2^-100
    let (s, t) = two_sum_f32(black_box(1.0f32), tiny);
    assert_eq!((s, t), (1.0, tiny), "two_sum probe changed under black_box");

    let sub = black_box(f32::from_bits(1));
    assert_eq!(
        black_box(sub + 0.0),
        sub,
        "subnormals flushed under black_box"
    );
}

// ---------------------------------------------------------------------------
// 1. The twin is production (this is what gives the mutants their meaning)
// ---------------------------------------------------------------------------

#[test]
fn reference_is_bit_identical_to_production() {
    let mut corpus = adversarial_corpus();
    corpus.extend(tiny_normal_product_cases());
    for c in &corpus {
        let prod = eft_dot_f32(&c.a, &c.b).map(Certified::from);
        let twin = ref_dot(&c.a, &c.b, Inject::None);
        match (prod, twin) {
            (None, None) => {}
            (Some(p), Some(t)) => {
                assert_eq!(
                    p.value.to_bits(),
                    t.value.to_bits(),
                    "{}: twin value diverged from production",
                    c.name
                );
                assert_eq!(
                    p.err.to_bits(),
                    t.err.to_bits(),
                    "{}: twin err diverged from production",
                    c.name
                );
            }
            (p, t) => panic!(
                "{}: refusal disagreement production={p:?} twin={t:?}",
                c.name
            ),
        }
    }
}

#[test]
fn channel_value_path_is_the_plain_f32_fold() {
    // The channel is a pure ADDITION of an error estimate: it must never move
    // the value. If it did, every downstream bit-identity gate would be a lie.
    for c in adversarial_corpus() {
        let mut plain = 0.0f32;
        let mut overflowed = false;
        for (&x, &y) in c.a.iter().zip(c.b.iter()) {
            plain += x * y;
            if !plain.is_finite() {
                overflowed = true;
                break;
            }
        }
        if overflowed {
            continue;
        }
        if let Some(cert) = ref_dot(&c.a, &c.b, Inject::None) {
            assert_eq!(
                cert.value.to_bits(),
                plain.to_bits(),
                "{}: EFT channel perturbed the value path",
                c.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Enclosure — the moat property
// ---------------------------------------------------------------------------

#[test]
fn reference_encloses_every_adversarial_family() {
    let mut corpus = adversarial_corpus();
    corpus.extend(tiny_normal_product_cases());
    let bad = scan(|a, b| ref_dot(a, b, Inject::None), &corpus);
    assert!(
        bad.is_empty(),
        "the EFT reference published a NON-ENCLOSING certified radius:\n  {}",
        bad.join("\n  ")
    );
}

/// The unconditional statement, over EVERY family including the one that broke
/// the pre-fix channel. No carve-out.
#[test]
fn production_channel_encloses_every_family() {
    let mut corpus = adversarial_corpus();
    corpus.extend(tiny_normal_product_cases());
    let bad = scan(|a, b| eft_dot_f32(a, b).map(Certified::from), &corpus);
    assert!(
        bad.is_empty(),
        "ny_core::eft::eft_dot_f32 published a NON-ENCLOSING certified \
         radius:\n  {}",
        bad.join("\n  ")
    );
}

/// The SHIPPED channel — value plus `min(higham, eft)` — must enclose over the
/// same corpus. `min` publishes the smaller arm, so this is the statement that
/// actually governs a verdict, not the EFT arm on its own.
#[test]
fn shipped_downgrade_only_channel_encloses_every_family() {
    let mut corpus = adversarial_corpus();
    corpus.extend(tiny_normal_product_cases());
    let bad = scan(
        |a, b| eft_dot_f32_downgrade_only(a, b).map(Certified::from),
        &corpus,
    );
    assert!(
        bad.is_empty(),
        "the shipped min(higham, eft) channel published a NON-ENCLOSING \
         certified radius:\n  {}",
        bad.join("\n  ")
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Randomized enclosure over the same magnitude regimes the corpus pins by
    /// hand — the corpus catches what we thought of, this catches what we did not.
    #[test]
    fn reference_encloses_random_dots(
        pairs in proptest::collection::vec(
            (
                prop_oneof![
                    -1e3f32..1e3f32,
                    -1e-3f32..1e-3f32,
                    -1e6f32..1e6f32,
                    -1e20f32..1e20f32,
                    (1u32..=0x007f_ffffu32).prop_map(f32::from_bits),
                    Just(0.0f32),
                ],
                prop_oneof![
                    -1e3f32..1e3f32,
                    -1e-6f32..1e-6f32,
                    -1e-20f32..1e-20f32,
                    Just(1.0f32),
                    Just(-1.0f32),
                ],
            ),
            0..400,
        )
    ) {
        let a: Vec<f32> = pairs.iter().map(|p| p.0).collect();
        let b: Vec<f32> = pairs.iter().map(|p| p.1).collect();
        let cert = ref_dot(&a, &b, Inject::None).ok_or_else(|| {
            TestCaseError::fail(format!(
                "the bounded finite strategy must publish a certificate at n={}",
                a.len()
            ))
        })?;
        prop_assert!(
            enclosure_excess(&a, &b, cert).is_none(),
            "certified enclosure violated at n={}", a.len()
        );
    }

    /// The twin/production binding must hold on random input too, or the
    /// mutation results below do not transfer to production.
    #[test]
    fn reference_matches_production_on_random_dots(
        pairs in proptest::collection::vec(
            (-1e6f32..1e6f32, -1e6f32..1e6f32), 0..200,
        )
    ) {
        let a: Vec<f32> = pairs.iter().map(|p| p.0).collect();
        let b: Vec<f32> = pairs.iter().map(|p| p.1).collect();
        let prod = eft_dot_f32(&a, &b)
            .map(Certified::from)
            .ok_or_else(|| TestCaseError::fail(
                "production refused the bounded finite strategy"
            ))?;
        let twin = ref_dot(&a, &b, Inject::None).ok_or_else(|| {
            TestCaseError::fail(
                "reference refused the bounded finite strategy"
            )
        })?;
        prop_assert_eq!(
            (prod.value.to_bits(), prod.err.to_bits()),
            (twin.value.to_bits(), twin.err.to_bits()),
        );
    }

    /// The shipped channel is DOWNGRADE-ONLY: never a radius larger than the
    /// a-priori incumbent, and the value path is never perturbed.
    #[test]
    fn shipped_channel_never_weakens_the_incumbent(
        pairs in proptest::collection::vec(
            (-1e6f32..1e6f32, -1e-6f32..1e6f32), 1..200,
        )
    ) {
        let a: Vec<f32> = pairs.iter().map(|p| p.0).collect();
        let b: Vec<f32> = pairs.iter().map(|p| p.1).collect();
        let shipped = eft_dot_f32_downgrade_only(&a, &b).ok_or_else(|| {
            TestCaseError::fail(format!(
                "the bounded finite strategy must publish the shipped channel at n={}",
                a.len()
            ))
        })?;
        let eft = eft_dot_f32(&a, &b).expect("the shipped arm published, so this one does");
        let higham = higham_dot_err_f32(&a, &b);
        prop_assert_eq!(shipped.value.to_bits(), eft.value.to_bits());
        prop_assert!(shipped.err <= higham);
        prop_assert!(
            shipped.err.to_bits() == higham.to_bits() || shipped.err.to_bits() == eft.err.to_bits(),
            "the combinator synthesized a radius that is neither arm"
        );
    }
}

/// Unsupported non-finite folds are refusal cases, not discarded property
/// samples. Pin both ways a finite-input dot can become non-finite: an
/// overflowing product and an overflowing accumulator.
#[test]
fn non_finite_folds_are_refused_deterministically() {
    let cases: [(&str, &[f32], &[f32]); 2] = [
        ("product overflow", &[f32::MAX], &[2.0]),
        ("accumulator overflow", &[f32::MAX, f32::MAX], &[1.0, 1.0]),
    ];

    for (name, a, b) in cases {
        assert!(
            ref_dot(a, b, Inject::None).is_none(),
            "reference must refuse {name}"
        );
        assert!(eft_dot_f32(a, b).is_none(), "production must refuse {name}");
        assert!(
            eft_dot_f32_downgrade_only(a, b).is_none(),
            "shipped channel must refuse {name}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. TEETH — every injected under-charge must be caught
// ---------------------------------------------------------------------------

/// The load-bearing test of this file. Each mutant is a plausible one-line
/// mistake in an EFT channel; the oracle must reject every one of them. A
/// mutant that survives means the oracle is decoration and the artifact is
/// worthless — which is exactly the failure this campaign already shipped once.
#[test]
fn oracle_catches_every_injected_undercharge() {
    let mut corpus = adversarial_corpus();
    corpus.extend(tiny_normal_product_cases());

    let mutants = [
        (
            Inject::OneUlpUnderCharge,
            "one-ULP under-charge (next_down on err)",
        ),
        (
            Inject::ZeroResidual,
            "residual channel folded to 0.0 (reassociation)",
        ),
        (Inject::DropProductResiduals, "TwoProd residuals dropped"),
        (Inject::DropSumResiduals, "TwoSum residuals dropped"),
        (
            Inject::NoOutwardRounding,
            "f64->f32 cast rounds to NEAREST, not outward",
        ),
        (
            Inject::LegacyUnderflowGuard,
            "TwoProd exactness guard lowered back to 2^-126 (the fixed defect)",
        ),
    ];

    let mut survived = Vec::new();
    for (inject, label) in mutants {
        let bad = scan(|a, b| ref_dot(a, b, inject), &corpus);
        if bad.is_empty() {
            survived.push(label);
        } else {
            eprintln!("caught {label}: {} violating case(s)", bad.len());
            eprintln!("    first: {}", bad[0]);
        }
    }
    assert!(
        survived.is_empty(),
        "MUTANTS SURVIVED — this oracle has no teeth and must not be trusted \
         as a soundness gate: {survived:?}"
    );
}

/// The one-ULP demonstration on its own, pinned to the constructed case, so a
/// future corpus edit cannot silently remove the tightest witness.
#[test]
fn one_ulp_undercharge_is_caught_on_the_constructed_tight_case() {
    let c = teeth_case();
    let honest = ref_dot(&c.a, &c.b, Inject::None).expect("finite");
    let cut = ref_dot(&c.a, &c.b, Inject::OneUlpUnderCharge).expect("finite");

    // The injection really is exactly one ULP, and only on the error term.
    assert_eq!(honest.value.to_bits(), cut.value.to_bits());
    assert_eq!(
        cut.err.to_bits() + 1,
        honest.err.to_bits(),
        "not a 1-ULP cut"
    );

    // The honest radius encloses...
    assert!(
        enclosure_excess(&c.a, &c.b, honest).is_none(),
        "the honest channel must enclose the constructed case"
    );
    // ...and the case is tight enough that ONE ULP breaks it.
    let excess = enclosure_excess(&c.a, &c.b, cut)
        .expect("a 1-ULP under-charge must be caught: the case is not tight");

    // Pin the arithmetic of the construction (see `teeth_case` docs).
    let two_pow = |k: i32| BigRational::new(BigInt::from(1u8), BigInt::from(1u8) << (-k) as u32);
    let r = two_pow(-13) + two_pow(-23) + two_pow(-36) + two_pow(-40);
    assert_eq!(
        (exact_dot(&c.a, &c.b) - exact_f32(honest.value)).abs(),
        r,
        "the constructed true error is not R = 2^-13 + 2^-23 + 2^-36 + 2^-40"
    );
    // value = 2049 + 2^-12 exactly (ulp(2049) = 2^-12, so this is on-grid).
    assert_eq!(honest.value, 2049.0f32 + f32::from_bits(0x3980_0000));
    assert!(excess > BigRational::zero());
    eprintln!("1-ULP under-charge escapes the radius by {excess}");
}

// ---------------------------------------------------------------------------
// 4. Downgrade-only
// ---------------------------------------------------------------------------

#[test]
fn combinator_is_downgrade_only() {
    let mut corpus = adversarial_corpus();
    corpus.extend(tiny_normal_product_cases());
    let mut not_enclosing = Vec::new();
    let mut worse_than_higham = Vec::new();
    let mut higham_broken = Vec::new();
    for c in &corpus {
        let Some(eft) = ref_dot(&c.a, &c.b, Inject::None) else {
            continue;
        };
        let higham = higham_dot_err_f32(&c.a, &c.b);
        if !higham.is_finite() {
            continue;
        }
        // The Higham arm must itself be sound on this case, or `min` is
        // meaningless — it preferentially publishes the SMALLER arm. This is an
        // ASSERTION now, not a skip (see `higham_arm_encloses_under_underflow`).
        let higham_cert = Certified {
            value: eft.value,
            err: higham,
        };
        if enclosure_excess(&c.a, &c.b, higham_cert).is_some() {
            higham_broken.push(c.name);
            continue;
        }
        let combined = combine_downgrade_only(higham, eft.err);

        // (a) never tighter than the truth
        if enclosure_excess(
            &c.a,
            &c.b,
            Certified {
                value: eft.value,
                err: combined,
            },
        )
        .is_some()
        {
            not_enclosing.push(c.name);
        }
        // (b) never worse than the pure-Higham incumbent
        if combined > higham {
            worse_than_higham.push(c.name);
        }
    }
    assert!(
        higham_broken.is_empty(),
        "the a-priori arm is non-enclosing on {higham_broken:?} — min() would \
         publish the broken arm, so the downgrade-only argument does not hold"
    );
    assert!(
        not_enclosing.is_empty(),
        "combined radius does not enclose: {not_enclosing:?}"
    );
    assert!(
        worse_than_higham.is_empty(),
        "combined radius is WORSE than the incumbent Higham bound — the \
         downgrade-only property is violated: {worse_than_higham:?}"
    );
}

/// A refused or broken EFT arm must degrade to the incumbent byte-identically.
/// An implementation that computes only the new arm on the fast path has
/// silently made it load-bearing and voided the downgrade-only argument.
#[test]
fn combinator_refusal_degrades_byte_identically() {
    for higham in [1e-7f32, 1.0, 3.4e38, 0.0, f32::MIN_POSITIVE] {
        for broken in [f32::NAN, f32::INFINITY, -1.0f32, -0.0] {
            let out = combine_downgrade_only(higham, broken);
            if broken == -0.0 {
                // -0.0 is not < 0.0; min(h, -0.0) is a genuine (vacuous)
                // tightening to zero, which the enclosure test governs.
                continue;
            }
            assert_eq!(
                out.to_bits(),
                higham.to_bits(),
                "refusal path must be byte-identical to the incumbent"
            );
        }
    }
}

/// The brief writes the rule as `max(higham, eft)`. On the induced LOWER BOUND
/// that is right; applied literally to the RADII it is a strict no-op that
/// recovers nothing — the entire point of S2 is lost. Pinned so nobody
/// implements the ambiguous form.
#[test]
fn max_on_radii_is_sound_but_recovers_nothing() {
    let mut tightened = 0usize;
    let mut total = 0usize;
    for c in adversarial_corpus() {
        let Some(eft) = ref_dot(&c.a, &c.b, Inject::None) else {
            continue;
        };
        let higham = higham_dot_err_f32(&c.a, &c.b);
        if !higham.is_finite() || higham <= 0.0 {
            continue;
        }
        total += 1;
        assert!(
            higham.max(eft.err) >= higham,
            "max on radii can never beat the incumbent"
        );
        if combine_downgrade_only(higham, eft.err) < higham {
            tightened += 1;
        }
    }
    assert!(total > 0);
    assert!(
        tightened > 0,
        "min-on-radii recovered nothing anywhere — the corpus is not \
         exercising the regime S2 targets"
    );
    eprintln!("min-on-radii tightened {tightened}/{total} corpus cases; max-on-radii tightened 0");
}

// ---------------------------------------------------------------------------
// 5. Regression gates for the two defects this artifact caught (fixed 2026-08-02)
// ---------------------------------------------------------------------------

/// The guard/charge constants must stay in the relationship the exactness
/// derivation needs. `TWO_PROD_EXACT_FLOOR_F32` is the GUARD (`2^-101`, the
/// TwoProdFMA hypothesis `e_a + e_b ≥ −103` with a binade to spare);
/// `PROD_UNDERFLOW_FLOOR_F32` is the CHARGE (`2^-126 ≥ ½·ulp(p)` for every
/// `|p|` the guard diverts). Swapping them silently reintroduces the defect.
#[test]
fn underflow_guard_dominates_the_charged_floor() {
    assert_eq!(PROD_UNDERFLOW_FLOOR_F32, f32::MIN_POSITIVE);
    assert_eq!(TWO_PROD_EXACT_FLOOR_F32, 2f32.powi(-101));
    const { assert!(TWO_PROD_EXACT_FLOOR_F32 > PROD_UNDERFLOW_FLOOR_F32) };
}

/// The band `|p| ∈ [2^-126, 2^-101)` — normal products whose TwoProd residual
/// is itself rounded — is where the pre-fix channel published a radius that did
/// not enclose. Production must now cover it, and the OLD guard must still be
/// demonstrably broken on the same inputs (otherwise the fix is not the thing
/// doing the work, and [`oracle_catches_every_injected_undercharge`]'s
/// `LegacyUnderflowGuard` mutant is vacuous).
#[test]
fn tiny_normal_product_band_is_covered_and_the_old_guard_still_fails_it() {
    let cases = tiny_normal_product_cases();
    let bad = scan(|a, b| eft_dot_f32(a, b).map(Certified::from), &cases);
    assert!(
        bad.is_empty(),
        "production regressed into the tiny-normal-product band:\n  {}",
        bad.join("\n  ")
    );
    let legacy = scan(|a, b| ref_dot(a, b, Inject::LegacyUnderflowGuard), &cases);
    assert!(
        !legacy.is_empty(),
        "the legacy guard no longer fails this band — the corpus stopped \
         exercising the defect, so the regression gate is vacuous"
    );
    eprintln!(
        "legacy guard still non-enclosing on {} case(s); production clean",
        legacy.len()
    );
}

/// The Higham comparator arm charges `γ_{n+1}·Σ|aᵢbᵢ|`, a purely RELATIVE
/// model, and Higham's Thm 3.1 assumes NO underflow. Under gradual underflow
/// the f32 rounding error is ABSOLUTE (up to `2^-150`), which no relative
/// multiple of a subnormal-scale `Σ|ab|` can cover.
///
/// This is why the S2 doc's "safe by construction" claim was false as written:
/// `min(E_eft, E_higham)` is sound **iff both arms are sound**, and `min`
/// preferentially publishes the SMALLER — i.e. the broken — arm. The arm now
/// carries a `4n·2^-149` absolute floor; this is its gate.
#[test]
fn higham_arm_encloses_under_underflow() {
    // One subnormal-scale product: exact = 3·2^-149·(1+2^-23) style residue.
    let a = vec![f32::from_bits(3), 1.5e-30f32];
    let b = vec![1.0f32 + f32::from_bits(0x3400_0000), 1.5e-15];
    let value = {
        let mut acc = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            acc += x * y;
        }
        acc
    };
    let higham = higham_dot_err_f32(&a, &b);
    assert!(
        enclosure_excess(&a, &b, Certified { value, err: higham }).is_none(),
        "the Higham arm is STILL non-enclosing under underflow: \
         value={value:e} err={higham:e}"
    );
}

/// Every subnormal-reaching family in the corpus, not just the hand-built pair:
/// the Higham arm must enclose wherever it publishes a finite radius, because
/// `min` will prefer it whenever it is the smaller of the two.
#[test]
fn higham_arm_encloses_wherever_it_publishes() {
    let mut corpus = adversarial_corpus();
    corpus.extend(tiny_normal_product_cases());
    let bad = scan(
        |a, b| {
            let eft = eft_dot_f32(a, b)?;
            let higham = higham_dot_err_f32(a, b);
            higham.is_finite().then_some(Certified {
                value: eft.value,
                err: higham,
            })
        },
        &corpus,
    );
    assert!(
        bad.is_empty(),
        "the a-priori arm published a NON-ENCLOSING radius, so min() would \
         prefer it:\n  {}",
        bad.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// 6. Anti-vacuity — a corpus that never reaches the code under test proves
//    nothing. Every family must either publish a checked radius or exercise
//    the fail-closed refusal, and both outcomes must actually occur.
// ---------------------------------------------------------------------------

#[test]
fn corpus_is_not_vacuous() {
    let mut corpus = adversarial_corpus();
    corpus.extend(tiny_normal_product_cases());
    let mut published = 0usize;
    let mut refused = Vec::new();
    let mut max_n = 0usize;
    for c in &corpus {
        max_n = max_n.max(c.a.len());
        assert!(!c.a.is_empty(), "{}: empty case", c.name);
        match ref_dot(&c.a, &c.b, Inject::None) {
            Some(cert) => {
                assert!(cert.err >= 0.0 && cert.err.is_finite());
                published += 1;
            }
            None => refused.push(c.name),
        }
    }
    assert!(published >= 10, "only {published} cases reached the oracle");
    assert!(
        !refused.is_empty(),
        "the fail-closed refusal path is never exercised — add a case that \
         overflows the fold"
    );
    assert!(
        max_n >= 9216,
        "no deep-conv-length fold in the corpus (max n = {max_n})"
    );
    eprintln!(
        "corpus: {} cases, {published} published, {} refused ({:?}), max n = {max_n}",
        corpus.len(),
        refused.len(),
        refused
    );
}
