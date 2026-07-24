// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact arbitrary-precision rational arithmetic for proof-carrying certificates.
//!
//! Certificates must be checkable by Clean's external-certificate verifier
//! (`clean-elab/src/cert/external/rational.rs`). Historically that verifier (and
//! this emitter) used an `i64` numerator/denominator pair with `i128`
//! intermediate arithmetic. Real benchmark networks (ACAS-Xu: f32 weights with
//! denominators up to `2^40`, exact preactivations needing 174–502 reduced
//! bits) overflow `i128` during interval-bound propagation. The only sound fix
//! is **arbitrary precision**.
//!
//! This module therefore backs [`Rat`] by [`num_rational::BigRational`]. To
//! preserve the *exact same `Copy` API* that the rest of `ny-cert` relies on
//! (the CROWN backward pass, the ReLU envelopes, the ONNX loader all pass
//! `Rat` by value and use `*r`), `Rat` is a small `Copy` **handle** (a `u32`
//! index) into a thread-local interning arena of canonicalised `BigRational`
//! values. Interning deduplicates structurally-equal values, so handle equality
//! coincides exactly with value equality — making the derived `PartialEq`/`Eq`/
//! `Hash` semantically correct. All arithmetic is performed in true bignum; the
//! former `i64` emission guard is gone, so `to_clean_string` always succeeds and
//! emits the full (possibly very large) `n/d` string.

use num_bigint::{BigInt, BigUint};
use num_rational::BigRational;
use num_traits::{CheckedDiv, One, Signed, ToPrimitive, Zero};
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::marker::PhantomData;
// `#[trust::requires/ensures]` use NY-owned no-op compatibility macros under
// stable rustc and become proof obligations under the external Trust verifier.
use trust as _;

/// Errors that can arise during exact rational arithmetic or emission.
///
/// The `Overflow` and `NotI64` variants are retained for API compatibility with
/// the i128-era callers (so existing `match`/`?` sites keep compiling), but with
/// arbitrary-precision arithmetic they are no longer produced by this module.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RatError {
    /// A denominator of zero was requested.
    #[error("rational denominator cannot be zero")]
    ZeroDenominator,
    /// An intermediate computation overflowed a fixed-width integer.
    ///
    /// Unreachable with bignum arithmetic; kept for source compatibility.
    #[error("exact rational arithmetic overflowed")]
    Overflow,
    /// A value cannot be represented in Clean's `i64` rational encoding.
    ///
    /// Unreachable now that certificates carry full bignum strings; kept for
    /// source compatibility with the i64-era schema code.
    #[error("rational {num}/{den} does not fit Clean's i64 certificate encoding")]
    NotI64 {
        /// Reduced numerator (decimal).
        num: String,
        /// Reduced denominator (decimal).
        den: String,
    },
    /// A point/vector was supplied with the wrong dimension. Fail-CLOSED guard
    /// (replaces a fail-loud `assert_eq!` on `x.len() == input_dim()` so callers
    /// that mis-size an input get a sound `Err` instead of a panic — total for
    /// the strict verifier without an unprovable interprocedural length
    /// precondition). Unreachable for every in-repo caller (all build `x` with
    /// exactly `input_dim()` entries).
    #[error("dimension mismatch: expected {expected}, got {got}")]
    Dimension {
        /// Expected length (`input_dim()`).
        expected: usize,
        /// Actual length supplied.
        got: usize,
    },
}

// ---------------------------------------------------------------------------
// Interning arena.
//
// Slot 0 is always 0/1, slot 1 is always 1/1 (so `Rat::ZERO`/`Rat::ONE` can be
// `const` handles). Every other distinct BigRational value gets a fresh slot;
// `dedup` maps a value back to its slot so equal values share a handle.
// ---------------------------------------------------------------------------
/// Dedup key wrapping a (reduced-form) `BigRational` with STRUCTURAL `Eq`/`Hash`.
///
/// `num_rational`'s `Hash for Ratio` hashes the mathematical equivalence class
/// by recursing through the continued-fraction (Euclidean) expansion — one
/// BigInt division per level, unbounded depth. On the huge-magnitude rationals
/// deep exact-CROWN accumulates (millions of bits), a single dedup probe
/// becomes minutes of divmods and a recursion-depth hazard: a beta-crown run
/// was observed pinning >78% of wall-clock inside that recursion during
/// certificate emission (acasxu maxdiff margins), stalling the CLI past every
/// deadline. Arena values are ALWAYS in reduced form with a positive
/// denominator (the `BigRational` construction/arithmetic invariant this arena
/// already documents and relies on), so structural `(numer, denom)` equality
/// coincides with mathematical equality here — and `BigInt`'s structural
/// `Hash` is a linear digit scan with no divisions. Structural `Eq` +
/// structural `Hash` are mutually consistent unconditionally, so the map
/// contract holds even without the reduced-form invariant.
#[derive(Debug, Clone)]
struct DedupKey(BigRational);

impl PartialEq for DedupKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.numer() == other.0.numer() && self.0.denom() == other.0.denom()
    }
}
impl Eq for DedupKey {}

impl std::hash::Hash for DedupKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.numer().hash(state);
        self.0.denom().hash(state);
    }
}

struct Arena {
    values: Vec<BigRational>,
    dedup: HashMap<DedupKey, u32>,
}

thread_local! {
    // `vec_init_then_push` is reported at block scope, so the allow must sit on
    // the item; the reason lives with the `Vec::new()` + push comment below.
    #[allow(clippy::vec_init_then_push)]
    static ARENA: RefCell<Arena> = RefCell::new({
        let zero = BigRational::zero();
        let one = BigRational::one();
        let mut dedup = HashMap::new();
        dedup.insert(DedupKey(zero.clone()), 0u32);
        dedup.insert(DedupKey(one.clone()), 1u32);
        // `Vec::new()` + push (not `vec![zero, one]`): the vec! macro's
        // boxed-slice expansion carries an alloc-internal Sub the strict
        // verifier refutes under havoc, charged (via the thread-local's lazy
        // init inlining) to whichever caller first touches the arena. Push
        // construction emits no macro-internal arithmetic; identical Vec.
        let mut values = Vec::new();
        values.push(zero);
        values.push(one);
        Arena { values, dedup }
    });
}

thread_local! {
    // Fail-CLOSED poison flag for the arena (same thread-local pattern as
    // `ARENA`). The four fallback arms in `intern`/`val` below are UNREACHABLE
    // by the arena's non-reentrancy/capacity arguments, but if one is ever
    // reached it silently substitutes the `0/1` ZERO value for an arbitrary
    // rational — fail-WRONG for a certificate emitter. Each such arm sets this
    // flag before returning its value fallback (totality is preserved);
    // certificate boundaries check [`poisoned`] and refuse to emit.
    static POISONED: Cell<bool> = const { Cell::new(false) };
}

/// Mark this thread's arena poisoned (a fallback arm was reached).
///
/// Free fn so each fallback arm is a single call; its only closure `|p|` is
/// concrete and `LocalKey::with` is total (see `val` for the rationale), and
/// `Cell::set` cannot panic — `poison` is TOTAL.
fn poison() {
    POISONED.with(|p| p.set(true));
}

/// True when any arena fallback arm has been reached on THIS thread.
///
/// Once set, every `Rat` produced or read on this thread since the poisoning
/// event is suspect (an arbitrary value may have been silently replaced by
/// `0/1`), so certificate producers must fail CLOSED: check this at entry and
/// again before returning a built certificate. The flag is thread-local (like
/// the arena itself) and is never cleared on non-test paths.
#[must_use]
pub fn poisoned() -> bool {
    POISONED.with(|p| p.get())
}

/// Test-only override of the poison flag, to exercise the fail-closed paths
/// (the real fallback arms are unreachable by construction and cannot be
/// forced from safe code).
#[cfg(test)]
pub(crate) fn set_poisoned_for_test(v: bool) {
    POISONED.with(|p| p.set(v));
}

/// Intern a canonicalised `BigRational`, returning its handle id.
fn intern(v: BigRational) -> u32 {
    ARENA.with(|a| {
        // `try_borrow_mut` is TOTAL (returns `Result`, never panics), unlike
        // `borrow_mut` which panics on a concurrent outstanding borrow. Arena
        // access is strictly scoped and non-reentrant, so the `Err` arm is
        // UNREACHABLE and, if ever reached, is POISON-MARKED (see `poisoned`)
        // before returning the `0/1` ZERO handle — the value fallback keeps
        // `intern` total, and the flag turns the silent wrong-value
        // substitution into a fail-CLOSED refusal at certificate boundaries.
        // Removes the panic boundary the strict verifier cannot discharge
        // without whole-program non-reentrancy reasoning.
        let Ok(mut a) = a.try_borrow_mut() else {
            poison();
            return 0;
        };
        let key = DedupKey(v);
        if let Some(&id) = a.dedup.get(&key) {
            return id;
        }
        // `try_from` fails only once the arena holds 2^32 distinct values —
        // UNREACHABLE before that: each slot owns a heap `BigRational` (plus a
        // dedup clone), so >2^32 live entries exceed hundreds of GiB and the
        // process aborts on allocation long before the index can overflow.
        // If ever reached, the arm is POISON-MARKED (see `poisoned`) and
        // returns the `0/1` ZERO handle WITHOUT growing the arena (state stays
        // consistent), mirroring the `try_borrow_mut` Err arm above — the flag
        // makes the wrong-value fallback fail-CLOSED at certificate
        // boundaries, while still removing the old
        // `.expect("rational arena exhausted (>2^32 distinct values)")` panic
        // boundary the strict verifier cannot discharge.
        let Ok(id) = u32::try_from(a.values.len()) else {
            poison();
            return 0;
        };
        a.values.push(key.0.clone());
        a.dedup.insert(key, id);
        id
    })
}

/// Fetch a clone of the `BigRational` value behind a handle.
///
/// MONOMORPHIC by design (it replaced a generic
/// `with_val<R>(id, f: impl FnOnce(&BigRational) -> R)` dispatcher). Verification
/// is a MIR pass over the GENERIC body: there the `f: impl FnOnce` type parameter
/// lowered `f(v)` to an unresolvable `<impl FnOnce.. as FnOnce>::call_once`
/// absent-callee obligation, and — because call-site names carry the concrete
/// generic args — every caller minted a distinct `with_val::<R, {closure}>`
/// absent-callee row (~45 runtime-checked rows, one per monomorphization). This
/// concrete accessor has NO type-parameter dispatch: its only closure `|a|` is
/// concrete (bundled; `LocalKey::with` is total), and `try_borrow` / slice `get`
/// / `BigRational::zero` / `Clone::clone` are each recognized-total — so `val`
/// verifies fully and every accessor routed through it discharges concretely.
/// It trusts NO closure.
fn val(id: u32) -> BigRational {
    ARENA.with(|a| {
        // `try_borrow` is TOTAL (never panics); the `Err` (reentrant-borrow)
        // arm is UNREACHABLE for the strictly-scoped, non-reentrant arena and,
        // if ever reached, is POISON-MARKED (see `poisoned`) before returning
        // a fresh `0/1`, exactly like the `None` (out-of-range id) arm. See
        // `intern` for the may-panic-boundary rationale.
        let Ok(arena) = a.try_borrow() else {
            poison();
            return BigRational::zero();
        };
        // `None` is unreachable for any live `Rat`: `intern` only ever returns
        // an in-range id, so every handle indexes a real slot. The
        // poison-marked `0/1` fallback (see `poisoned`) keeps the arena read
        // TOTAL — no panic boundary and no slice-bounds obligation on the
        // opaque interned id (which the verifier cannot otherwise bound) —
        // while making the wrong-value substitution fail-CLOSED at
        // certificate boundaries.
        match arena.values.get(id as usize) {
            Some(v) => v.clone(),
            None => {
                poison();
                BigRational::zero()
            }
        }
    })
}

/// An exact arbitrary-precision rational, represented as a `Copy` handle into a
/// thread-local interning arena. Always stored in reduced form with a positive
/// denominator (the `BigRational` invariant).
///
/// # Thread-locality contract
///
/// The arena is THREAD-LOCAL, so a handle is only meaningful on the thread
/// that interned it: on any other thread it would resolve against THAT
/// thread's arena — an unrelated value or the `0/1` fallback, with no error.
/// The contract is ENFORCED by the type system: the private
/// `PhantomData<*const ()>` marker field makes `Rat` (and every struct
/// embedding one) `!Send`/`!Sync`, so moving a handle across threads is a
/// compile error rather than a silent wrong value. To cross a thread
/// boundary, marshal values as strings (`to_clean_string`/`fmt`) instead, as
/// `certify_onnx`'s leaf workers do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rat {
    id: u32,
    // `*const ()` is `!Send`/`!Sync` while `PhantomData` of it stays
    // zero-sized and `Copy`/`Eq`/`Hash`-compatible — it exists solely to strip
    // the auto-traits (see the thread-locality contract above).
    _not_send: PhantomData<*const ()>,
}

impl Rat {
    /// The rational `0` (arena slot 0).
    pub const ZERO: Rat = Rat {
        id: 0,
        _not_send: PhantomData,
    };
    /// The rational `1` (arena slot 1).
    pub const ONE: Rat = Rat {
        id: 1,
        _not_send: PhantomData,
    };

    /// Construct a reduced rational `num/den` from machine integers.
    ///
    /// # Errors
    /// Returns [`RatError::ZeroDenominator`] when `den == 0`.
    pub fn new(num: i128, den: i128) -> Result<Self, RatError> {
        if den == 0 {
            return Err(RatError::ZeroDenominator);
        }
        // BigRational::new reduces and normalises the sign of the denominator.
        let r = BigRational::new(BigInt::from(num), BigInt::from(den));
        Ok(Rat {
            id: intern(r),
            _not_send: PhantomData,
        })
    }

    /// Construct a reduced rational from arbitrary-precision integers.
    ///
    /// # Errors
    /// Returns [`RatError::ZeroDenominator`] when `den` is zero.
    pub fn from_bigints(num: BigInt, den: BigInt) -> Result<Self, RatError> {
        if den.is_zero() {
            return Err(RatError::ZeroDenominator);
        }
        let r = BigRational::new(num, den);
        Ok(Rat {
            id: intern(r),
            _not_send: PhantomData,
        })
    }

    /// Construct a rational from an integer.
    #[must_use]
    pub fn from_int(n: i128) -> Self {
        let r = BigRational::from_integer(BigInt::from(n));
        Rat {
            id: intern(r),
            _not_send: PhantomData,
        }
    }

    /// The EXACT dyadic rational value of a finite `f32` (no rounding).
    ///
    /// IEEE-754 binary32 values are exactly `m · 2^e`, so the conversion is
    /// lossless: the returned rational denotes precisely the same real number
    /// as the float. Returns `None` for NaN/±∞ — the caller decides how a
    /// non-finite constant should fail (certificate producers fail closed).
    #[must_use]
    pub fn from_f32_exact(f: f32) -> Option<Self> {
        if !f.is_finite() {
            return None;
        }
        if f == 0.0 {
            return Some(Rat::ZERO);
        }
        let bits = f.to_bits();
        // behavior-identical: `bits` is u32 and 31 < 32, so `wrapping_shr(31)`
        // equals `>> 31`; it drops the spurious shift-overflow VC.
        let sign: i128 = if bits.wrapping_shr(31) == 0 { 1 } else { -1 };
        // Truncating cast keeps exactly the 8 exponent bits (`bits >> 23` has
        // at most 9 significant bits), so exp_field is in [0, 255] and the
        // subtraction below cannot overflow — but the model havocs cast
        // results entirely (the bound does not exist for the solver at ANY
        // width), so use `wrapping_sub`: exact for every real input, total
        // for the model's phantom values, and it emits no overflow VC.
        // behavior-identical: `bits` is u32 and 23 < 32, so `wrapping_shr(23)`
        // equals `>> 23`; it drops the spurious shift-overflow VC.
        let exp_field = i64::from(bits.wrapping_shr(23) as u8);
        let frac = i128::from(bits & 0x007f_ffff);
        let (mantissa, e2) = if exp_field == 0 {
            (frac, -149) // = -126 - 23: subnormal f32 base-2 exponent; literal avoids a Sub overflow VC on a compile-time constant
        } else {
            (8_388_608i128 | frac, exp_field.wrapping_sub(150)) // 1i128<<23 const-folded (= 2^23) to avoid a Shl overflow VC on a compile-time constant
        };
        // behavior-identical: `sign ∈ {−1,+1}` and `mantissa ∈ [0, 2^24)` (a 23-
        // or 24-bit magnitude), so `sign * mantissa` is a conditional negate whose
        // operand is never i128::MIN; `wrapping_neg` equals `-mantissa` exactly.
        // This models a width-128 Neg instead of an Unsupported width>64 i128 Mul.
        let signed = BigInt::from(if sign < 0 {
            mantissa.wrapping_neg()
        } else {
            mantissa
        });
        if e2 >= 0 {
            // Finite f32 exponents keep the shift far below BigInt limits.
            let num = signed << u32::try_from(e2).ok()?;
            Rat::from_bigints(num, BigInt::from(1)).ok()
        } else {
            // e2 < 0 on this branch, so `unsigned_abs` equals -e2 exactly;
            // unlike `-e2` it is total (no Neg-overflow VC). e2 is in
            // [-149, -1] here, so the u64 shift amount is tiny.
            let den = BigInt::from(1) << e2.unsigned_abs();
            Rat::from_bigints(signed, den).ok()
        }
    }

    /// Numerator of the reduced form (arbitrary precision).
    #[must_use]
    pub fn num(self) -> BigInt {
        val(self.id).numer().clone()
    }

    /// Denominator of the reduced form (arbitrary precision, always positive).
    #[must_use]
    pub fn den(self) -> BigInt {
        val(self.id).denom().clone()
    }

    /// The exact `BigRational` value behind this handle.
    #[must_use]
    pub fn to_big(self) -> BigRational {
        val(self.id)
    }

    /// True when the value is exactly zero. (Slot 0 by interning invariant.)
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.id == 0
    }

    /// True when the value is strictly positive.
    #[must_use]
    pub fn is_positive(self) -> bool {
        val(self.id).is_positive()
    }

    /// True when the value is strictly negative.
    #[must_use]
    pub fn is_negative(self) -> bool {
        val(self.id).is_negative()
    }

    /// Exact arbitrary-precision addition. Never overflows.
    ///
    /// # Errors
    /// Infallible; the `Result` is kept for source compatibility.
    #[allow(clippy::should_implement_trait, clippy::unnecessary_wraps)]
    pub fn add(self, other: Self) -> Result<Self, RatError> {
        let r = val(self.id) + val(other.id);
        Ok(Rat {
            id: intern(r),
            _not_send: PhantomData,
        })
    }

    /// Exact arbitrary-precision subtraction. Never overflows.
    ///
    /// # Errors
    /// Infallible; the `Result` is kept for source compatibility.
    #[allow(clippy::should_implement_trait, clippy::unnecessary_wraps)]
    pub fn sub(self, other: Self) -> Result<Self, RatError> {
        let r = val(self.id) - val(other.id);
        Ok(Rat {
            id: intern(r),
            _not_send: PhantomData,
        })
    }

    /// Exact arbitrary-precision multiplication. Never overflows.
    ///
    /// # Errors
    /// Infallible; the `Result` is kept for source compatibility.
    #[allow(clippy::should_implement_trait, clippy::unnecessary_wraps)]
    pub fn mul(self, other: Self) -> Result<Self, RatError> {
        let r = val(self.id) * val(other.id);
        Ok(Rat {
            id: intern(r),
            _not_send: PhantomData,
        })
    }

    /// Negation.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn neg(self) -> Self {
        let r = -val(self.id);
        Rat {
            id: intern(r),
            _not_send: PhantomData,
        }
    }

    /// Multiplicative inverse.
    ///
    /// # Errors
    /// Returns [`RatError::ZeroDenominator`] when the value is zero.
    pub fn inv(self) -> Result<Self, RatError> {
        // Value-local zero guard: test the SAME `BigRational` that `recip`
        // runs on, in one monomorphic body, so the nonzero fact is
        // intraprocedural and the verifier's `is_zero(x) == (x == 0)` axiom
        // discharges the `Ratio::recip` may-panic obligation. (The old shape
        // guarded on the handle — `self.id == 0` — which is equivalent by the
        // interning invariant but opaque to the verifier.) `val` releases its
        // arena borrow on return, so the later `intern` (which takes a fresh
        // borrow) cannot reentrant-conflict.
        let v = val(self.id);
        let r = if v.is_zero() { None } else { Some(v.recip()) };
        match r {
            None => Err(RatError::ZeroDenominator),
            Some(r) => Ok(Rat {
                id: intern(r),
                _not_send: PhantomData,
            }),
        }
    }

    /// Absolute value.
    #[must_use]
    pub fn abs(self) -> Self {
        if self.is_negative() {
            self.neg()
        } else {
            self
        }
    }

    /// Certified rational **upper bound on the square root**: returns `r`
    /// with `r · r ≥ self`, or `None` when `self` is negative.
    ///
    /// For `self = n/d` (reduced, `n, d > 0`), `√(n/d) = √(n·d)/d`. With
    /// `k = precision_bits` and `t = ⌊√(n·d·4^k)⌋` (exact integer square
    /// root):
    ///
    /// - if `t² = n·d·4^k` the value is a perfect square of the scaled grid
    ///   and `r = t/(d·2^k)` is **exact** (`r² = self`);
    /// - otherwise `r = (t+1)/(d·2^k)`, and `(t+1)² > n·d·4^k` certifies
    ///   `r² > self`.
    ///
    /// Either way the overestimate is at most one grid step:
    /// `r − √self ≤ 1/(d·2^k)`.
    #[must_use]
    pub fn sqrt_upper(self, precision_bits: u32) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        if self.is_zero() {
            return Some(Rat::ZERO);
        }
        // Positive by the checks above; BigRational keeps the denominator
        // positive, so both magnitudes convert losslessly.
        let n = self.num().to_biguint()?;
        let d = self.den().to_biguint()?;
        let scaled = (&n * &d) << (precision_bits as usize).saturating_mul(2);
        let t = isqrt_floor(&scaled);
        let num = if &t * &t == scaled { t } else { t + 1_u32 };
        let den = &d << (precision_bits as usize);
        let r = Rat::from_bigints(BigInt::from(num), BigInt::from(den)).ok()?;
        debug_assert!(r.mul(r).is_ok_and(|sq| sq >= self));
        Some(r)
    }

    /// Nearest-`f64` approximation of the exact value, for **display and
    /// diagnostics only** — the rounding direction is not certified. Values
    /// beyond `f64` range saturate to ±∞.
    #[must_use]
    pub fn to_f64_approx(self) -> f64 {
        val(self.id).to_f64().unwrap_or(f64::NAN)
    }

    /// Emit the canonical certificate string (`"n"` or `"n/d"`) with the full
    /// arbitrary-precision numerator and denominator. Always succeeds.
    ///
    /// # Errors
    /// Infallible; the `Result` is kept for source compatibility with the
    /// i64-era schema code.
    #[allow(clippy::unnecessary_wraps)]
    pub fn to_clean_string(self) -> Result<String, RatError> {
        let r = val(self.id);
        if r.denom().is_one() {
            Ok(r.numer().to_string())
        } else {
            Ok(format!("{}/{}", r.numer(), r.denom()))
        }
    }
}

/// Exact integer square-root floor: the unique `x` with `x² ≤ n < (x+1)²`.
///
/// Newton iteration `x ← (x + n/x)/2` starting from a guess `≥ √n`
/// (`2^⌈bits/2⌉`) converges to the floor; the trailing adjustment loops make
/// the floor property hold **by construction**, so callers' soundness never
/// rests on the convergence argument alone.
fn isqrt_floor(n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    // Over-estimating the initial exponent only widens the Newton isqrt guess
    // (it still converges to the exact floor via the trailing adjustments), so
    // the `usize::MAX` fallback is a total, sound substitute; the `None` branch
    // is unreachable on a 64-bit target (`u64` bit count always fits `usize`).
    let half_bits = usize::try_from(n.bits().div_ceil(2)).unwrap_or(usize::MAX);
    let mut x = BigUint::one() << half_bits;
    loop {
        // `checked_div` is TOTAL (`None` only on a zero divisor). `x >= 1` is
        // a loop invariant: x starts at `2^half_bits >= 1`, and with `n >= 1`
        // (the `is_zero` early return above) every Newton update
        // `y = (x + n/x)/2` from `x >= 1` is itself `>= 1` — so the `None` arm
        // is UNREACHABLE and fails safe to `0` (ny-cert's fail-soft idiom).
        // Removes the div-by-zero panic boundary the verifier cannot discharge
        // across the loop structure; quotient identical for every real input.
        let q = n.checked_div(&x).unwrap_or_else(BigUint::zero);
        let y = (&x + q) >> 1;
        if y >= x {
            break;
        }
        x = y;
    }
    while &x * &x > *n {
        x -= 1_u32;
    }
    while (&x + 1_u32) * (&x + 1_u32) <= *n {
        x += 1_u32;
    }
    x
}

impl PartialOrd for Rat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rat {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.id == other.id {
            return Ordering::Equal;
        }
        // Total arena reads via `.get()`: both ids are live `Rat` handles, so
        // both slots exist; the `None` arms are unreachable, POISON-MARKED
        // (see `intern`), and fall back to a fresh `0/1`, keeping `cmp`
        // panic-free and free of a slice-bounds obligation on the opaque
        // interned ids (see `with_val`). A reached arm can never silently
        // order a certificate: `poisoned()` refuses emission.
        ARENA.with(|a| {
            let zero = BigRational::zero();
            // `try_borrow` is TOTAL; the `Err` (reentrant-borrow) arm is
            // UNREACHABLE for the non-reentrant arena — poison-marked, then
            // falls back to `Equal`.
            let Ok(a) = a.try_borrow() else {
                poison();
                return Ordering::Equal;
            };
            let lhs = match a.values.get(self.id as usize) {
                Some(v) => v,
                None => {
                    poison();
                    &zero
                }
            };
            let rhs = match a.values.get(other.id as usize) {
                Some(v) => v,
                None => {
                    poison();
                    &zero
                }
            };
            lhs.cmp(rhs)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduces_on_construction() {
        let r = Rat::new(2, 4).unwrap();
        assert_eq!(r.num(), BigInt::from(1));
        assert_eq!(r.den(), BigInt::from(2));
    }

    #[test]
    fn normalizes_sign_to_numerator() {
        let r = Rat::new(1, -2).unwrap();
        assert_eq!(r.num(), BigInt::from(-1));
        assert_eq!(r.den(), BigInt::from(2));
    }

    #[test]
    fn arithmetic_is_exact() {
        let third = Rat::new(1, 3).unwrap();
        let sixth = Rat::new(1, 6).unwrap();
        assert_eq!(third.add(sixth).unwrap(), Rat::new(1, 2).unwrap());
        assert_eq!(third.mul(Rat::from_int(3)).unwrap(), Rat::ONE);
        assert_eq!(third.sub(third).unwrap(), Rat::ZERO);
    }

    #[test]
    fn emits_clean_strings() {
        assert_eq!(Rat::from_int(-5).to_clean_string().unwrap(), "-5");
        assert_eq!(Rat::new(-3, 4).unwrap().to_clean_string().unwrap(), "-3/4");
    }

    #[test]
    fn interning_makes_equal_values_share_a_handle() {
        let a = Rat::new(2, 4).unwrap();
        let b = Rat::new(1, 2).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.id, b.id);
    }

    #[test]
    fn rejects_zero_denominator() {
        assert_eq!(Rat::new(1, 0), Err(RatError::ZeroDenominator));
    }

    #[test]
    fn ordering_matches_value() {
        assert!(Rat::new(1, 3).unwrap() < Rat::new(1, 2).unwrap());
        assert!(Rat::new(-1, 2).unwrap() < Rat::ZERO);
    }

    #[test]
    fn huge_values_do_not_overflow() {
        // 2^200 / 3 — would overflow i128 instantly.
        let big = BigInt::from(2u8).pow(200);
        let r = Rat::from_bigints(big, BigInt::from(3)).unwrap();
        let s = r.add(r).unwrap(); // 2 * (2^200/3) = 2^201/3
        assert_eq!(s.num(), BigInt::from(2u8).pow(201));
        assert_eq!(s.den(), BigInt::from(3));
        // numerator has > 128 bits
        assert!(s.num().bits() > 128);
    }

    #[test]
    fn abs_negates_only_negatives() {
        assert_eq!(Rat::new(-3, 4).unwrap().abs(), Rat::new(3, 4).unwrap());
        assert_eq!(Rat::new(3, 4).unwrap().abs(), Rat::new(3, 4).unwrap());
        assert_eq!(Rat::ZERO.abs(), Rat::ZERO);
    }

    #[test]
    fn isqrt_floor_is_exact() {
        for n in [0_u32, 1, 2, 3, 4, 8, 9, 15, 16, 17, 24, 25, 26, 1_000_000] {
            let big = BigUint::from(n);
            let x = isqrt_floor(&big);
            assert!(&x * &x <= big, "isqrt({n})² must be ≤ {n}");
            assert!(
                (&x + 1_u32) * (&x + 1_u32) > big,
                "(isqrt({n})+1)² must be > {n}"
            );
        }
        // A value far beyond u128: 2^300 → isqrt = 2^150 exactly.
        let big = BigUint::from(1_u8) << 300;
        assert_eq!(isqrt_floor(&big), BigUint::from(1_u8) << 150);
    }

    #[test]
    fn sqrt_upper_is_exact_on_perfect_squares() {
        assert_eq!(Rat::from_int(25).sqrt_upper(16).unwrap(), Rat::from_int(5));
        assert_eq!(
            Rat::new(9, 4).unwrap().sqrt_upper(16).unwrap(),
            Rat::new(3, 2).unwrap()
        );
        assert_eq!(Rat::ZERO.sqrt_upper(16).unwrap(), Rat::ZERO);
        assert_eq!(Rat::ONE.sqrt_upper(16).unwrap(), Rat::ONE);
    }

    #[test]
    fn sqrt_upper_certifies_and_is_tight() {
        let k = 32_u32;
        for q in [
            Rat::from_int(2),
            Rat::new(1, 3).unwrap(),
            Rat::from_f32_exact(0.1).unwrap(),
            Rat::from_bigints(BigInt::from(10).pow(20), BigInt::from(7)).unwrap(),
        ] {
            let r = q.sqrt_upper(k).unwrap();
            // Certified upper bound: r² ≥ q.
            assert!(r.mul(r).unwrap() >= q, "sqrt_upper must satisfy r² ≥ q");
            // Tight to one grid step: (r − 1/(d·2^k))² < q.
            let step = Rat::from_bigints(BigInt::from(1), q.den() << (k as usize)).unwrap();
            let lo = r.sub(step).unwrap();
            assert!(
                lo.mul(lo).unwrap() < q,
                "sqrt_upper must be within one 1/(d·2^k) grid step of √q"
            );
        }
    }

    #[test]
    fn sqrt_upper_rejects_negatives() {
        assert_eq!(Rat::from_int(-1).sqrt_upper(16), None);
        assert_eq!(Rat::new(-1, 4).unwrap().sqrt_upper(16), None);
    }

    #[test]
    fn to_f64_approx_matches_small_values() {
        assert!((Rat::new(1, 2).unwrap().to_f64_approx() - 0.5).abs() < 1e-15);
        assert!((Rat::from_int(-3).to_f64_approx() + 3.0).abs() < 1e-15);
        assert!(Rat::ZERO.to_f64_approx().abs() < 1e-15);
    }

    #[test]
    fn poisoned_is_false_under_normal_operation() {
        // Exercise every arena path (intern hit + miss, val, cmp) — none of
        // the fallback arms is reachable in normal operation, so the poison
        // flag must stay clear.
        let a = Rat::new(1, 3).unwrap();
        let b = a.add(Rat::new(1, 6).unwrap()).unwrap();
        assert!(b < Rat::ONE);
        assert_eq!(b.to_clean_string().unwrap(), "1/2");
        assert!(!poisoned());
    }

    #[test]
    fn poison_flag_mechanics_set_and_clear() {
        // The real fallback arms cannot be forced from safe code (the arena is
        // non-reentrant by construction), so exercise the flag mechanics
        // directly through the test-only setter. Thread-local: this cannot
        // leak into tests on other threads, and we clear it before returning.
        assert!(!poisoned());
        set_poisoned_for_test(true);
        assert!(poisoned());
        set_poisoned_for_test(false);
        assert!(!poisoned());
    }

    #[test]
    fn from_f32_exact_is_lossless_and_fails_closed() {
        assert_eq!(Rat::from_f32_exact(0.0), Some(Rat::ZERO));
        assert_eq!(Rat::from_f32_exact(-0.0), Some(Rat::ZERO));
        assert_eq!(Rat::from_f32_exact(0.5), Some(Rat::new(1, 2).unwrap()));
        assert_eq!(Rat::from_f32_exact(-2.75), Some(Rat::new(-11, 4).unwrap()));
        // 0.1f32 is the dyadic 13421773/2^27, NOT 1/10 — exactness means the
        // float's true value, never the decimal it was parsed from.
        assert_eq!(
            Rat::from_f32_exact(0.1),
            Some(Rat::new(13_421_773, 1 << 27).unwrap())
        );
        // Extremes stay exact: f32::MAX and the smallest positive subnormal.
        let max = Rat::from_f32_exact(f32::MAX).unwrap();
        assert_eq!(max.den(), BigInt::from(1));
        let sub = Rat::from_f32_exact(f32::from_bits(1)).unwrap();
        assert_eq!(sub.num(), BigInt::from(1));
        assert_eq!(sub.den(), BigInt::from(2u8).pow(149));
        // Non-finite fails closed.
        assert_eq!(Rat::from_f32_exact(f32::NAN), None);
        assert_eq!(Rat::from_f32_exact(f32::INFINITY), None);
        assert_eq!(Rat::from_f32_exact(f32::NEG_INFINITY), None);
    }
}
