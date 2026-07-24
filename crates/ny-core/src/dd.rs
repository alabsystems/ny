// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Double-double (two-`f64`) compensated arithmetic and its certified
//! error envelope (`#dd-zonotope`).
//!
//! # Why this exists
//!
//! The `vggnet16_2022` VNN-COMP category has instances where only `k` input
//! pixels are perturbed (measured k = 1, 5, 10, 20, 100). A sparse-generator
//! DeepZ zonotope forward pass proves those specs *if and only if* the
//! rounding channel it carries stays far below the margin. The reference probe
//! (`scripts/vggnet16_zonotope_rounding_probe.py`, commit 49737525) MEASURED
//! that channel as a function of the working unit roundoff `u`:
//!
//! * `u = 2^-53` (plain f64): certified margin `[-29298, +29045]` on a true
//!   margin of `1.6375` — **vacuous**, and the inflated intermediate boxes
//!   manufacture 31121 spurious generator columns.
//! * `u = 2^-105` (double-double): certified half-width `1.1e-12` on the same
//!   spec — **12 orders of headroom**.
//!
//! The measured amplification `A = (rounding half-width)/u` is `~6e19 ≈ 2^66`,
//! stable across every k class. So the minimum non-vacuous significand for
//! VGG16 is ~65 bits: f64's 53 is 19 bits short and even an x87 80-bit
//! extended (64-bit significand) misses. Double-double's ~106 bits clears it.
//!
//! # The error envelope (`U_DD`) — derivation, not assertion
//!
//! [`dd_fma`] accumulates `acc += a*b` for plain-`f64` `a`, `b`:
//!
//! ```text
//! (p, e)  = two_prod(a, b)      // a*b = p + e            EXACT
//! (s, e2) = two_sum(acc.hi, p)  // acc.hi + p = s + e2    EXACT
//! t       = e2 + e + acc.lo     // <-- the ONLY rounding (2 roundings)
//! (h, l)  = two_sum(s, t)       // s + t = h + l          EXACT
//! ```
//!
//! So `h + l = s + fl(e2 + e + acc.lo)` exactly, and the exact step result is
//! `s + (e2 + e + acc.lo)`. The per-step error is therefore
//!
//! ```text
//! |err_i| <= 2u (|e2| + |e| + |acc.lo|)      (u = 2^-53)
//! ```
//!
//! with `|e2| <= u|s| <= u(|acc.hi| + |p|)`, `|e| <= u|p|`, and
//! `|acc.lo| <= u|acc.hi|` (the trailing `two_sum` normalizes). Hence
//! `|err_i| <= 4u^2 (|acc_i| + |p_i|)`. Since `|acc_i| <= S_i <= S` where
//! `S = sum_j |a_j b_j|`, summing over the `n` steps gives
//!
//! ```text
//! |dd_dot - exact| <= 4 (n + 1) u^2 S.
//! ```
//!
//! That is exactly Higham's `gamma_n(U_DD) * S` form with an effective unit
//! roundoff `U_DD_DERIVED = 4 u^2 = 2^-104`. [`U_DD`] is set to `2^-102`
//! (`16 u^2`) — a **4x safety factor** over the derived bound. At the measured
//! `A ~ 2^66` amplification that puts the certified half-width at ~`1e-11` on
//! a margin of `1.6`, i.e. still 11 orders of headroom, so the safety factor
//! is free.
//!
//! # The compiler hazard
//!
//! `two_sum`'s error term is *algebraically* zero. Any FP reassociation
//! collapses it to `0.0` and silently degrades double-double to plain f64 —
//! which the probe MEASURED to be vacuous, so the failure mode is a silently
//! **too tight** (unsound) bound. Rust does not enable fast-math, but LTO /
//! codegen changes are a standing hazard. [`crate::dd_selfcheck`] is therefore
//! a hard precondition of every consumer, not a nicety. Same precedent as
//! `ny-cuda/src/ieee_selfcheck.rs`.

/// f64 unit roundoff `2^-53`.
pub const U_F64: f64 = 1.0 / 9_007_199_254_740_992.0; // 2^-53, exactly

/// Certified effective unit roundoff of the [`Dd`] accumulator, `2^-102`.
///
/// The derivation above yields `4 u^2 = 2^-104`; this constant carries a 4x
/// safety factor on top. Consumers pair it with [`gamma_n_dd`] and the
/// sum-of-absolute-products `S` exactly as they would pair [`U_F64`] with
/// Higham's `gamma_n` for a plain f64 dot product.
pub const U_DD: f64 = 16.0 * U_F64 * U_F64; // 2^-102, exactly

/// Double-double: an unevaluated sum `hi + lo` with `|lo| <= u|hi|`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dd {
    /// Leading (rounded) component.
    pub hi: f64,
    /// Trailing correction; `|lo| <= u |hi|` after any of the constructors here.
    pub lo: f64,
}

impl Dd {
    /// The double-double zero.
    pub const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };

    /// Exact lift of a plain `f64`.
    #[inline]
    #[must_use]
    pub const fn from_f64(x: f64) -> Self {
        Dd { hi: x, lo: 0.0 }
    }

    /// Nearest `f64` to `hi + lo`.
    ///
    /// This collapse costs one f64 rounding (`<= u|value|`). Callers on a
    /// certified path must fold that into their error channel — which is why
    /// the zonotope keeps its center in [`Dd`] across every layer and collapses
    /// only once, at the final margin, where `u|margin| ~ 1e-16` is negligible
    /// against the ~`1e-11` certified rounding half-width.
    #[inline]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// `|hi| + |lo|`, an upper bound on `|hi + lo|` (no rounding down: the sum
    /// of two nonnegatives rounds to nearest, so widen by one ulp at the use
    /// site if an outward bound is required).
    #[inline]
    #[must_use]
    pub fn abs_upper(self) -> f64 {
        self.hi.abs() + self.lo.abs()
    }

    /// True when both components are finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }

    /// The double-double zero, as a function (sibling of the [`Dd::ZERO`]
    /// constant, kept for call sites that read better with a constructor).
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Dd::ZERO
    }

    /// Negation — exact, since negating an `f64` cannot round.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    /// Absolute value — exact, since the trailing word's sign follows the
    /// leading word's under the normalization invariant.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        if self.hi < 0.0 {
            self.neg()
        } else {
            self
        }
    }
}

impl PartialOrd for Dd {
    /// Compares the represented values. `hi` dominates; the trailing word only
    /// breaks a tie, which is exactly the ordering of `hi + lo` under the
    /// normalization invariant `|lo| <= u|hi|`.
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match self.hi.partial_cmp(&other.hi) {
            Some(core::cmp::Ordering::Equal) => self.lo.partial_cmp(&other.lo),
            non_equal => non_equal,
        }
    }
}

/// Knuth's `TwoSum`: returns `(s, e)` with `a + b == s + e` **exactly**,
/// `s == fl(a + b)`.
///
/// `#[inline]` (not `always`) and deliberately written without `mul_add` so the
/// backend cannot contract it. Verified at runtime by
/// [`crate::dd_selfcheck::dd_selfcheck_ok`].
#[inline]
#[must_use]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// `TwoProduct` via a fused multiply-add: returns `(p, e)` with
/// `a * b == p + e` **exactly**, `p == fl(a * b)`.
///
/// Requires a true single-rounding FMA. On aarch64 and x86-64+FMA
/// `f64::mul_add` lowers to `fmadd`/`vfmadd`; on a target without one it
/// becomes a libm call that is still correctly rounded. A platform that
/// silently substitutes `a*b + (-p)` computed in two roundings would return
/// `e == 0` and is caught by the self-check.
#[inline]
#[must_use]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// `x + y` in double-double (Dekker/"sloppy" `DWPlusDW`).
#[inline]
#[must_use]
pub fn dd_add(x: Dd, y: Dd) -> Dd {
    let (s, e) = two_sum(x.hi, y.hi);
    let e = e + x.lo + y.lo;
    let (hi, lo) = two_sum(s, e);
    Dd { hi, lo }
}

/// `x + y` for a plain-`f64` addend (`DWPlusFP`).
#[inline]
#[must_use]
pub fn dd_add_f64(x: Dd, y: f64) -> Dd {
    let (s, e) = two_sum(x.hi, y);
    let e = e + x.lo;
    let (hi, lo) = two_sum(s, e);
    Dd { hi, lo }
}

/// `acc + a*b` for plain-`f64` `a`, `b` — the accumulation step whose error is
/// bounded by [`U_DD`] in the module derivation.
#[inline]
#[must_use]
pub fn dd_fma(acc: Dd, a: f64, b: f64) -> Dd {
    let (p, e) = two_prod(a, b);
    let (s, e2) = two_sum(acc.hi, p);
    let e = e2 + e + acc.lo;
    let (hi, lo) = two_sum(s, e);
    Dd { hi, lo }
}

/// `x - y` in double-double.
#[inline]
#[must_use]
pub fn dd_sub(x: Dd, y: Dd) -> Dd {
    dd_add(x, y.neg())
}

/// `x * y` in double-double.
///
/// The `lo*lo` cross term is below the representable residual and is dropped;
/// the resulting relative error is within the [`U_DD`] envelope.
#[inline]
#[must_use]
pub fn dd_mul(x: Dd, y: Dd) -> Dd {
    let (p, e) = two_prod(x.hi, y.hi);
    let e = e + x.hi.mul_add(y.lo, x.lo * y.hi);
    let (hi, lo) = two_sum(p, e);
    Dd { hi, lo }
}

/// `x * y` for a plain-`f64` multiplier.
#[inline]
#[must_use]
pub fn dd_mul_f64(x: Dd, y: f64) -> Dd {
    let (p, e) = two_prod(x.hi, y);
    let e = e + x.lo * y;
    let (hi, lo) = two_sum(p, e);
    Dd { hi, lo }
}

/// Higham's `gamma_n = n u / (1 - n u)` at the plain-f64 unit roundoff.
///
/// Returns `+inf` when the denominator is not positive, mirroring
/// `crown_single_gamma_n_f64`'s clamp-to-infinity discipline: an infinite
/// error term makes every downstream certified bound vacuous, which is sound,
/// rather than negative, which would not be.
#[inline]
#[must_use]
pub fn gamma_n_f64(n: usize) -> f64 {
    gamma_n_at(n, U_F64)
}

/// Higham's `gamma_n` at the double-double effective unit roundoff [`U_DD`].
#[inline]
#[must_use]
pub fn gamma_n_dd(n: usize) -> f64 {
    gamma_n_at(n, U_DD)
}

/// Higham's `gamma_n = n u / (1 - n u)` at an explicit unit roundoff, rounded
/// outward (one ulp up) so the returned value is never below the exact ratio.
#[inline]
#[must_use]
pub fn gamma_n_at(n: usize, u: f64) -> f64 {
    // n u computed with one rounding then widened: `nu_hi >= n*u` exactly.
    let nu = (n as f64) * u;
    let nu_hi = next_up_f64(nu);
    let den = 1.0 - nu_hi;
    // `!(den > 0.0)` deliberately catches NaN as well as den <= 0 — do not
    // rewrite as `den <= 0.0`, which is false for NaN.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(den > 0.0) || !nu_hi.is_finite() {
        return f64::INFINITY;
    }
    next_up_f64(nu_hi / den)
}

/// Next representable `f64` above `x` (toward `+inf`).
///
/// Mirrors `ny_tensor::next_up_f32`; duplicated here so `ny-core` stays free of
/// a dependency on `ny-tensor` (which depends on `ny-core`).
#[inline]
#[must_use]
pub fn next_up_f64(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x.is_sign_positive() {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Next representable `f64` below `x` (toward `-inf`).
#[inline]
#[must_use]
pub fn next_down_f64(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(0x8000_0000_0000_0001);
    }
    let bits = x.to_bits();
    if x.is_sign_positive() {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u_constants_are_the_intended_powers_of_two() {
        assert_eq!(U_F64, 2.0_f64.powi(-53));
        assert_eq!(U_DD, 2.0_f64.powi(-102));
        // The DERIVED envelope is 4u^2 = 2^-104; U_DD carries 4x safety.
        // Deliberately a runtime assert in a test: documents the const relation.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(U_DD >= 4.0 * U_F64 * U_F64);
        }
        assert_eq!(U_DD, 16.0 * U_F64 * U_F64);
    }

    #[test]
    fn two_sum_is_exact_on_a_cancelling_pair() {
        // 1 + 2^-60: the sum rounds to 1.0 and the residual is exactly 2^-60.
        let (s, e) = two_sum(1.0, 2.0_f64.powi(-60));
        assert_eq!(s, 1.0);
        assert_eq!(e, 2.0_f64.powi(-60));
    }

    #[test]
    fn two_prod_recovers_the_exact_low_word() {
        // (1 + 2^-30)^2 = 1 + 2^-29 + 2^-60; the f64 product drops 2^-60.
        let a = 1.0 + 2.0_f64.powi(-30);
        let (p, e) = two_prod(a, a);
        assert_eq!(p, 1.0 + 2.0_f64.powi(-29));
        assert_eq!(e, 2.0_f64.powi(-60));
    }

    #[test]
    fn dd_dot_beats_naive_f64_on_a_cancelling_sum() {
        // sum of 1e17, 1.0 (x 100), -1e17 -> exact answer 100.
        let mut naive = 0.0_f64;
        let mut dd = Dd::ZERO;
        naive += 1e17;
        dd = dd_add_f64(dd, 1e17);
        for _ in 0..100 {
            naive += 1.0;
            dd = dd_add_f64(dd, 1.0);
        }
        naive -= 1e17;
        dd = dd_add_f64(dd, -1e17);
        assert_eq!(dd.to_f64(), 100.0, "double-double must be exact here");
        assert_ne!(naive, 100.0, "naive f64 must NOT be exact here");
    }

    #[test]
    fn dd_fma_matches_exact_rational_on_a_hard_dot() {
        // Deterministic pseudo-random dot product; the double-double result
        // must agree with an exact big-rational evaluation to <= U_DD relative.
        use num_bigint::BigInt;
        use num_rational::BigRational;

        let n = 4608usize; // VGG16 3x3x512 conv dot length.
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Values spanning ~8 binades so the accumulation genuinely cancels.
            let m = ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
            m * 2.0_f64.powi(((state % 17) as i32) - 8)
        };
        let a: Vec<f64> = (0..n).map(|_| next()).collect();
        let b: Vec<f64> = (0..n).map(|_| next()).collect();

        let mut dd = Dd::ZERO;
        for i in 0..n {
            dd = dd_fma(dd, a[i], b[i]);
        }

        let mut exact = BigRational::from(BigInt::from(0));
        let mut s_abs = 0.0_f64;
        for i in 0..n {
            let ra = BigRational::from_float(a[i]).expect("finite");
            let rb = BigRational::from_float(b[i]).expect("finite");
            exact += ra * rb;
            s_abs += (a[i] * b[i]).abs();
        }
        let got = BigRational::from_float(dd.hi).expect("finite")
            + BigRational::from_float(dd.lo).expect("finite");
        let diff = got - exact;
        let zero = BigRational::from(BigInt::from(0));
        let err = if diff < zero { -diff } else { diff };
        // The certified envelope: |err| <= gamma_n(U_DD) * S.
        let bound = BigRational::from_float(gamma_n_dd(n + 1) * s_abs).expect("finite");
        assert!(
            err <= bound,
            "dd dot error {err} exceeded the certified envelope {bound}"
        );
    }

    #[test]
    fn gamma_n_clamps_to_infinity_when_the_denominator_dies() {
        assert!(gamma_n_at(usize::MAX, U_F64).is_infinite());
        assert!(gamma_n_f64(4608).is_finite());
        assert!(gamma_n_dd(4608) > 0.0);
        // The DD gamma must be ~2^49 smaller than the f64 one.
        assert!(gamma_n_dd(4608) < gamma_n_f64(4608) * 1e-14);
    }

    /// The motivating property, stated as a behaviour: double-double retains a
    /// term that f64 drops entirely, and recovers it exactly on subtraction.
    #[test]
    fn dd_retains_a_term_f64_drops_entirely() {
        let tiny = 2.0_f64.powi(-60);
        assert_eq!(
            1.0_f64 + tiny,
            1.0,
            "f64 drops the term (this is the point)"
        );

        let sum = dd_add(Dd::from_f64(1.0), Dd::from_f64(tiny));
        assert_eq!(sum.hi, 1.0);
        assert_eq!(
            sum.lo, tiny,
            "double-double retains it in the trailing word"
        );
        assert_eq!(dd_sub(sum, Dd::from_f64(1.0)).to_f64(), tiny);
    }

    #[test]
    fn zero_neg_abs_and_ordering_follow_the_represented_value() {
        assert_eq!(Dd::zero(), Dd::ZERO);
        let a = Dd::from_f64(1.0);
        let b = dd_add(a, Dd::from_f64(2.0_f64.powi(-60)));
        assert!(b > a, "the trailing word breaks the tie");
        assert_eq!(b.neg().abs(), b);
        assert_eq!(b.abs().to_f64(), b.to_f64());
        assert!(!Dd::from_f64(f64::NAN).is_finite());
    }

    #[test]
    fn dd_mul_agrees_with_repeated_dd_fma_on_the_represented_value() {
        for (a, b) in [(3.0, 1.0 / 3.0), (1e-9, 1e9), (-2.5, 4.75)] {
            let via_mul = dd_mul(Dd::from_f64(a), Dd::from_f64(b));
            let via_fma = dd_fma(Dd::ZERO, a, b);
            assert_eq!(via_mul.to_f64(), via_fma.to_f64(), "a={a} b={b}");
        }
    }

    #[test]
    fn next_up_down_f64_bracket_the_value() {
        for &x in &[0.0_f64, 1.0, -1.0, 1e-300, -3.25e17] {
            assert!(next_down_f64(x) < x || x == 0.0);
            assert!(next_up_f64(x) > x || x == 0.0);
        }
        assert!(next_up_f64(0.0) > 0.0);
        assert!(next_down_f64(0.0) < 0.0);
        assert!(next_up_f64(f64::INFINITY).is_infinite());
    }
}
