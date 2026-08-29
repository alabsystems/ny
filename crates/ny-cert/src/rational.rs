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
//! coincides exactly with value equality — making the derived `PartialEq`/`Eq`
//! semantically correct. All arithmetic is performed in true bignum; the
//! former `i64` emission guard is gone, so `to_clean_string` emits the full
//! (possibly very large) `n/d` string. The only remaining refusal is the
//! fail-closed arena-poison guard described below.

use num_bigint::{BigInt, BigUint};
use num_rational::BigRational;
use num_traits::{CheckedDiv, CheckedSub, One, Signed, ToPrimitive, Zero};
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::marker::PhantomData;
// `#[trust::requires/ensures]` use NY-owned no-op compatibility macros under
// stable rustc and become proof obligations under the external Trust verifier.
use trust as _;

/// Errors that can arise during exact rational arithmetic or emission.
///
/// The `Overflow` and `NotI64` variants are retained as legacy named variants
/// for callers that still reference them, but arbitrary-precision arithmetic
/// no longer produces either one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RatError {
    /// An impossible arena fallback was reached. A fallback substitutes `0/1`
    /// to keep the low-level accessor total, so no value from this thread may
    /// cross a proof/check/emission boundary after the flag is set.
    Poisoned,
    /// A denominator of zero was requested.
    ZeroDenominator,
    /// An intermediate computation overflowed a fixed-width integer.
    ///
    /// Unreachable with bignum arithmetic; kept for source compatibility.
    Overflow,
    /// A value cannot be represented in Clean's `i64` rational encoding.
    ///
    /// Unreachable now that certificates carry full bignum strings; kept for
    /// source compatibility with the i64-era schema code.
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
    Dimension {
        /// Expected length (`input_dim()`).
        expected: usize,
        /// Actual length supplied.
        got: usize,
    },
}

// Hand-written replacements for the `thiserror::Error` derive (removed so that
// ny-cert carries no proc-macro dependency and the bignum stack can itself be
// verified rather than assumed). The `Display` messages below are byte-identical
// to the former `#[error("...")]` strings; they appear in certificates and test
// assertions and must not be reworded.
impl core::fmt::Display for RatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RatError::Poisoned => write!(
                f,
                "rational arena is poisoned; refusing a potentially substituted value"
            ),
            RatError::ZeroDenominator => write!(f, "rational denominator cannot be zero"),
            RatError::Overflow => write!(f, "exact rational arithmetic overflowed"),
            RatError::NotI64 { num, den } => write!(
                f,
                "rational {num}/{den} does not fit Clean's i64 certificate encoding"
            ),
            RatError::Dimension { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl core::error::Error for RatError {
    /// No variant of `RatError` carried a `#[from]`/`#[source]` field, so the
    /// derived `source()` was `None` for every variant.
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        None
    }
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

#[cfg(test)]
thread_local! {
    // Const and non-Drop so it remains available while the drop-bearing ARENA
    // value is being destroyed.
    static ARENA_TEARDOWN_PROBE_ARMED: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
static ARENA_TEARDOWN_RESULT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
impl Drop for Arena {
    fn drop(&mut self) {
        let armed = ARENA_TEARDOWN_PROBE_ARMED
            .try_with(|probe| probe.replace(false))
            .unwrap_or(false);
        if !armed {
            return;
        }

        // std marks a lazy TLS key destroyed before dropping its value. This
        // makes each call below take ARENA.try_with's outer Err arm without
        // relying on destruction order between separate keys.
        let mut observed = 0_u8;

        set_poisoned_for_test(false);
        let _ = intern(BigRational::zero());
        if poisoned() {
            observed |= 1;
        }

        set_poisoned_for_test(false);
        let _ = val(0);
        if poisoned() {
            observed |= 2;
        }

        set_poisoned_for_test(false);
        let _ = Rat::ZERO.cmp(&Rat::ONE);
        if poisoned() {
            observed |= 4;
        }

        ARENA_TEARDOWN_RESULT.store(observed, std::sync::atomic::Ordering::Release);
    }
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
/// concrete, `Cell::set` cannot panic, and `LocalKey::try_with` is TOTAL
/// (returns `Result`, never panics) — so `poison` is TOTAL. `LocalKey::with` is
/// NOT total: it panics (`panic_access_error`) when the thread-local is touched
/// during its own destruction, a panic boundary the strict verifier cannot
/// discharge (same rationale as `intern`/`val` below).
///
/// The `Err` (thread-local already destroyed) arm deliberately does NOTHING,
/// and that is the fail-SAFE choice rather than a convenient one: if `POISONED`
/// is no longer writable then [`poisoned`] can no longer read it either, and
/// its own `Err` arm answers `true`. So the instant the flag becomes
/// inaccessible the thread is LATCHED poisoned — a mark can never be silently
/// lost by discarding it here.
fn poison() {
    // Result deliberately discarded; see the fn doc — an unwritable flag already
    // reads back as poisoned, so there is nothing to recover on the `Err` arm.
    let _ = POISONED.try_with(|p| p.set(true));
}

/// True when any arena fallback arm has been reached on THIS thread.
///
/// Once set, every `Rat` produced or read on this thread since the poisoning
/// event is suspect (an arbitrary value may have been silently replaced by
/// `0/1`), so certificate producers must fail CLOSED: check this at entry and
/// again before returning a built certificate. The flag is thread-local (like
/// the arena itself) and is never cleared on non-test paths.
///
/// `try_with` is TOTAL (returns `Result`, never panics), unlike `with`, which
/// panics (`panic_access_error`) if the thread-local is accessed during its own
/// destruction — a panic boundary the strict verifier cannot discharge.
///
/// The `Err` (thread-local already destroyed) arm answers `true`, i.e.
/// POISONED. That is the SEMANTICALLY conservative answer, not merely the one
/// that verifies: during TLS teardown there is no live arena, so `intern`/`val`/
/// `cmp` can only take their `Err`/`None` arms and substitute the `0/1` ZERO
/// value for an arbitrary rational — precisely the silent wrong-value condition
/// this flag exists to report. Answering `false` there would let a certificate
/// boundary accept values no arena backs; answering `true` can at worst refuse
/// a certificate on a thread that is already dying. Fail-CLOSED, so `true`.
///
/// Written as an explicit `match` rather than `.unwrap_or(true)`. The two are
/// semantically identical — `Result::unwrap_or` IS this match, and `true` is a
/// constant, so eager-vs-lazy does not arise — but the generic
/// `Result::<T, E>::unwrap_or` drops a CALLER-CHOSEN `E`, and some `E` has a
/// panicking `Drop`, so the verifier must treat the call as may-panic. Spelling
/// the match here monomorphises `E` to `AccessError`, a unit struct with no drop
/// glue, and the obligation is still emitted but now discharges. This is the
/// head of the arena cascade: `ensure_healthy` and ~30 further functions have
/// this call as their only unproved row.
#[must_use]
#[allow(
    clippy::manual_unwrap_or,
    reason = "the explicit match is what makes this function verifiable; see above"
)]
pub fn poisoned() -> bool {
    match POISONED.try_with(|p| p.get()) {
        Ok(v) => v,
        Err(_) => true,
    }
}

/// Refuse use of a thread whose arena has taken a totalizing fallback.
///
/// Kept as a small named function so every checked arithmetic and emission path
/// applies the same entry/exit policy.
pub(crate) fn ensure_healthy() -> Result<(), RatError> {
    if poisoned() {
        Err(RatError::Poisoned)
    } else {
        Ok(())
    }
}

/// Test-only override of the poison flag, to exercise the fail-closed paths.
/// Inner live-arena fallbacks are unreachable by construction; the teardown
/// regression deterministically exercises the outer key-unavailable arms.
#[cfg(test)]
pub(crate) fn set_poisoned_for_test(v: bool) {
    // `try_with` for the same totality reason as `poison`/`poisoned` above. A
    // dropped write cannot hide a failure: `poisoned()` reads an inaccessible
    // flag as `true`, so a lost `set(false)` reset makes tests fail loudly.
    let _ = POISONED.try_with(|p| p.set(v));
}

/// Intern a canonicalised `BigRational`, returning its handle id.
fn intern(v: BigRational) -> u32 {
    // `try_with` is TOTAL (returns `Result`, never panics), unlike `with` which
    // panics (`panic_access_error`) if the thread-local is accessed during its own
    // destruction — a panic boundary the strict verifier cannot discharge. The
    // `Err` (arena already destroyed) arm IS REACHABLE: the `cfg(test)` teardown
    // probe in `Arena::drop` re-enters `intern` after std has marked this key
    // destroyed, which is exactly what `destroyed_arena_outer_fallbacks_latch_poison`
    // pins. It therefore POISON-MARKS before returning the `0/1` ZERO handle, so the
    // wrong-value substitution is fail-CLOSED at certificate boundaries instead of
    // silent. The INNER `try_borrow_mut` arm below does stay unreachable, by the
    // arena's non-reentrancy argument, and is poison-marked for the same reason.
    let result = ARENA.try_with(|a| {
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
    });
    match result {
        Ok(id) => id,
        Err(_) => {
            // Returning ZERO without this latch would be fail-wrong: callers
            // could not distinguish a destroyed arena from a genuine zero.
            poison();
            0
        }
    }
}

/// Checked wrapper around [`intern`]. The second gate is essential: `intern`
/// remains total by returning slot zero on an impossible borrow/capacity
/// failure, and that substitution must never be reported as a successful
/// construction.
fn checked_intern(v: BigRational) -> Result<u32, RatError> {
    ensure_healthy()?;
    let id = intern(v);
    ensure_healthy()?;
    Ok(id)
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
/// concrete (bundled; `LocalKey::try_with` is total — `with` is NOT, see the
/// body), and `try_borrow` / slice `get` / `BigRational::zero` / `Clone::clone`
/// are each recognized-total — so `val`
/// verifies fully and every accessor routed through it discharges concretely.
/// It trusts NO closure.
fn val(id: u32) -> BigRational {
    // `try_with` is TOTAL (returns `Result`, never panics), unlike `with` which
    // panics if the thread-local is accessed during its own destruction — a panic
    // boundary the strict verifier cannot discharge. The `Err` (arena already
    // destroyed) arm IS REACHABLE: the `cfg(test)` teardown probe in `Arena::drop`
    // re-enters `val` from the destructor, which
    // `destroyed_arena_outer_fallbacks_latch_poison` pins. It POISON-MARKS before
    // falling back to a fresh `0/1`, exactly like the `try_borrow`/`None` arms
    // inside — those two INNER arms do stay unreachable, by non-reentrancy and by
    // `intern` only ever handing out in-range ids. Binding `result` first (rather
    // than `unwrap_or`) keeps the fallback `BigRational::zero()` LAZY.
    let result = ARENA.try_with(|a| {
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
    });
    match result {
        Ok(value) => value,
        Err(_) => {
            // Preserve the same poison-before-fallback invariant as every
            // inner borrow/index failure.
            poison();
            BigRational::zero()
        }
    }
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
// `Hash` intentionally NOT derived: `Rat` is never used as a hash-map/set key
// anywhere in the workspace (the arena dedup keys on `BigRational`; `Rat` appears
// only as a `BTreeMap` value via `Ord`), so the derived `<Rat as Hash>::hash<H>`
// was dead code whose panic-freedom is unprovable for a generic `H: Hasher`
// (`H::write_*` is arbitrary user code). Re-add `Hash` only if a hashed use appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rat {
    id: u32,
    // `*const ()` is `!Send`/`!Sync` while `PhantomData` of it stays
    // zero-sized and `Copy`/`Eq`-compatible — it exists solely to strip
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
    /// Returns [`RatError::ZeroDenominator`] when `den == 0`, or
    /// [`RatError::Poisoned`] after an arena fallback.
    pub fn new(num: i128, den: i128) -> Result<Self, RatError> {
        ensure_healthy()?;
        if den == 0 {
            return Err(RatError::ZeroDenominator);
        }
        // BigRational::new reduces and normalises the sign of the denominator.
        let r = BigRational::new(BigInt::from(num), BigInt::from(den));
        Ok(Rat {
            id: checked_intern(r)?,
            _not_send: PhantomData,
        })
    }

    /// Construct a reduced rational from arbitrary-precision integers.
    ///
    /// # Errors
    /// Returns [`RatError::ZeroDenominator`] when `den` is zero, or
    /// [`RatError::Poisoned`] after an arena fallback.
    pub fn from_bigints(num: BigInt, den: BigInt) -> Result<Self, RatError> {
        ensure_healthy()?;
        if den.is_zero() {
            return Err(RatError::ZeroDenominator);
        }
        let r = BigRational::new(num, den);
        Ok(Rat {
            id: checked_intern(r)?,
            _not_send: PhantomData,
        })
    }

    /// Construct a reduced rational from arbitrary-precision integers, yielding
    /// `None` for a zero denominator instead of an `Err`.
    ///
    /// The `Option`-returning twin of [`Rat::from_bigints`] — behaviourally identical to
    /// `Rat::from_bigints(num, den).ok()` (returns `None` iff `den` is zero, else `Some`
    /// of the identical reduced rational). Callers that already produce `Option<Self>`
    /// (the exact float conversions, where the denominator is nonzero by construction)
    /// use it to avoid a spurious `Result::ok()` error-discard boundary.
    #[must_use]
    pub fn from_bigints_opt(num: BigInt, den: BigInt) -> Option<Self> {
        ensure_healthy().ok()?;
        if den.is_zero() {
            return None;
        }
        let r = BigRational::new(num, den);
        Some(Rat {
            id: checked_intern(r).ok()?,
            _not_send: PhantomData,
        })
    }

    /// Construct a rational from an integer.
    #[must_use]
    pub fn from_int(n: i128) -> Self {
        // This legacy convenience API cannot return an error. Preserve its
        // signature, but never clear the poison: every checked consumer will
        // refuse the returned handle. Returning ZERO while already poisoned
        // also avoids presenting a newly interned value as trustworthy.
        if poisoned() {
            return Rat::ZERO;
        }
        let r = BigRational::from_integer(BigInt::from(n));
        let result = Rat {
            id: intern(r),
            _not_send: PhantomData,
        };
        if poisoned() {
            Rat::ZERO
        } else {
            result
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
        ensure_healthy().ok()?;
        // Classify entirely from the representation. A floating comparison
        // (`f == 0.0`) can treat a subnormal operand as zero when DAZ is live;
        // exact certificate lifting must remain independent of MXCSR state.
        let bits = f.to_bits();
        let magnitude = bits & 0x7fff_ffff;
        let exp_bits = bits.wrapping_shr(23) & 0xff;
        if exp_bits == 0xff {
            return None;
        }
        if magnitude == 0 {
            return Some(Rat::ZERO);
        }
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
        let exp_field = i64::from(exp_bits as u8);
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
            // Finite f32 exponents keep the shift far below BigInt limits. On this
            // branch e2 = exp_field − 150 with exp_field ∈ [0,255] and e2 ≥ 0, so
            // e2 ∈ [0,105] and `e2 as u32` is exact and total for every finite f32 —
            // the same total-cast idiom `crown_deep::f64_to_rat_exact` uses; the old
            // `u32::try_from(e2).ok()?` None path was unreachable here.
            let num = signed << (e2 as u32);
            // `from_bigints_opt` == `from_bigints(..).ok()`, den = 1 (literal) ≠ 0.
            Rat::from_bigints_opt(num, BigInt::from(1))
        } else {
            // e2 < 0 on this branch, so `unsigned_abs` equals -e2 exactly;
            // unlike `-e2` it is total (no Neg-overflow VC). e2 is in
            // [-149, -1] here, so the u64 shift amount is tiny.
            let den = BigInt::from(1) << e2.unsigned_abs();
            // den = 1 << |e2| ≥ 1 ≠ 0, so `from_bigints_opt` never yields None here.
            Rat::from_bigints_opt(signed, den)
        }
    }

    /// The EXACT dyadic rational value of a finite `f64` (no rounding).
    ///
    /// Classification and zero detection use the representation bits rather
    /// than floating-point comparisons, so subnormal inputs remain exact even
    /// when the host floating-point environment enables DAZ/FTZ behavior.
    /// Returns `None` for NaN/±∞ or a poisoned arena.
    #[must_use]
    pub fn from_f64_exact(f: f64) -> Option<Self> {
        ensure_healthy().ok()?;
        let bits = f.to_bits();
        let magnitude = bits & 0x7fff_ffff_ffff_ffff;
        let exp_bits = bits.wrapping_shr(52) & 0x7ff;
        if exp_bits == 0x7ff {
            return None;
        }
        if magnitude == 0 {
            return Some(Rat::ZERO);
        }

        let mantissa_field = bits & 0x000f_ffff_ffff_ffff;
        let (mantissa, exponent) = if exp_bits == 0 {
            (mantissa_field, -1074_i64)
        } else {
            (
                mantissa_field | 0x0010_0000_0000_0000,
                i64::try_from(exp_bits).ok()?.wrapping_sub(1075),
            )
        };
        let magnitude = BigInt::from(mantissa);
        let signed = if bits.wrapping_shr(63) == 0 {
            magnitude
        } else {
            -magnitude
        };
        if exponent >= 0 {
            Rat::from_bigints(signed << u32::try_from(exponent).ok()?, BigInt::from(1)).ok()
        } else {
            Rat::from_bigints(signed, BigInt::from(1) << exponent.unsigned_abs()).ok()
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

    /// Read both canonical components with one arena lookup and fail closed if
    /// that lookup took a totalizing fallback.
    ///
    /// This is the authoritative extraction API for arithmetic and certificate
    /// emission. The legacy [`Self::num`]/[`Self::den`] accessors remain for
    /// diagnostics and source compatibility.
    /// # Errors
    ///
    /// Returns [`RatError::Poisoned`] if the arena has taken a totalizing
    /// fallback before or during the lookup.
    pub fn checked_parts(self) -> Result<(BigInt, BigInt), RatError> {
        ensure_healthy()?;
        let (num, den) = val(self.id).into_raw();
        ensure_healthy()?;
        if !den.is_positive() {
            // Canonical BigRational denominators are positive. Treat a broken
            // invariant exactly like an arena fallback rather than allowing a
            // zero/negative denominator into a certificate.
            poison();
            return Err(RatError::Poisoned);
        }
        Ok((num, den))
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
    /// Returns [`RatError::Poisoned`] after an arena fallback.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Result<Self, RatError> {
        ensure_healthy()?;
        if self.id == 0 {
            ensure_healthy()?;
            return Ok(other);
        }
        if other.id == 0 {
            ensure_healthy()?;
            return Ok(self);
        }
        let (a, b) = self.checked_parts()?;
        let (c, d) = other.checked_parts()?;
        // Equal denominators are common for exact f32 dyadics. Avoiding the
        // general cross-product also prevents a large transient denominator.
        if b == d {
            return Rat::from_bigints(a + c, b);
        }
        // a/b + c/d = (a*d + c*b)/(b*d). BigInt arithmetic itself is total;
        // from_bigints guards the sole Ratio panic boundary (zero denominator).
        Rat::from_bigints(&a * &d + &c * &b, &b * &d)
    }

    /// Exact arbitrary-precision subtraction. Never overflows.
    ///
    /// # Errors
    /// Returns [`RatError::Poisoned`] after an arena fallback.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Result<Self, RatError> {
        ensure_healthy()?;
        if self.id == other.id {
            ensure_healthy()?;
            return Ok(Rat::ZERO);
        }
        if other.id == 0 {
            ensure_healthy()?;
            return Ok(self);
        }
        let (a, b) = self.checked_parts()?;
        let (c, d) = other.checked_parts()?;
        if b == d {
            return Rat::from_bigints(a - c, b);
        }
        Rat::from_bigints(&a * &d - &c * &b, &b * &d)
    }

    /// Exact arbitrary-precision multiplication. Never overflows.
    ///
    /// # Errors
    /// Returns [`RatError::Poisoned`] after an arena fallback.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Result<Self, RatError> {
        ensure_healthy()?;
        if self.id == 0 || other.id == 0 {
            ensure_healthy()?;
            return Ok(Rat::ZERO);
        }
        if self.id == 1 {
            ensure_healthy()?;
            return Ok(other);
        }
        if other.id == 1 {
            ensure_healthy()?;
            return Ok(self);
        }
        let (a, b) = self.checked_parts()?;
        let (c, d) = other.checked_parts()?;
        // Trust's RC-C proof deliberately avoids opaque gcd/division callees.
        // Still cancel the frequent exact reciprocal factors without division.
        if a == d {
            return Rat::from_bigints(c, b);
        }
        if c == b {
            return Rat::from_bigints(a, d);
        }
        if a == -&d {
            return Rat::from_bigints(-c, b);
        }
        if c == -&b {
            return Rat::from_bigints(-a, d);
        }
        Rat::from_bigints(a * c, b * d)
    }

    /// Negation.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn neg(self) -> Self {
        if poisoned() {
            return self;
        }
        if self.id == 0 {
            return self;
        }
        // -(a/b) = (-a)/b. `BigInt` Neg is total; b>0 by the arena invariant so
        // `from_bigints` never Errs on a healthy arena. If that invariant ever
        // breaks, retain totality but poison the thread before returning.
        let parts = self.checked_parts();
        if let Ok((num, den)) = parts {
            if let Ok(r) = Rat::from_bigints(-num, den) {
                return r;
            }
        }
        poison();
        self
    }

    /// Multiplicative inverse.
    ///
    /// # Errors
    /// Returns [`RatError::ZeroDenominator`] when the value is zero, or
    /// [`RatError::Poisoned`] after an arena fallback.
    pub fn inv(self) -> Result<Self, RatError> {
        ensure_healthy()?;
        if self.id == 0 {
            return Err(RatError::ZeroDenominator);
        }
        if self.id == 1 {
            return Ok(Rat::ONE);
        }
        let (num, den) = self.checked_parts()?;
        Rat::from_bigints(den, num)
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
        ensure_healthy().ok()?;
        if self.is_negative() {
            return None;
        }
        if self.is_zero() {
            ensure_healthy().ok()?;
            return Some(Rat::ZERO);
        }
        // Positive by the checks above; BigRational keeps the denominator
        // positive, so both magnitudes convert losslessly.
        let (num, den) = self.checked_parts().ok()?;
        let n = num.to_biguint()?;
        let d = den.to_biguint()?;
        let scaled = (&n * &d) << (precision_bits as usize).saturating_mul(2);
        let t = isqrt_floor(&scaled);
        let num = if &t * &t == scaled { t } else { t + 1_u32 };
        let den = &d << (precision_bits as usize);
        // den = `&d << precision_bits` with `d = self.den().to_biguint() >= 1` (self is
        // positive-nonzero by the guards above), so den >= 1 ≠ 0 and `from_bigints_opt`
        // (== `from_bigints(..).ok()`) never yields None here — removes the Result::ok
        // error-discard boundary without changing any value.
        let r = Rat::from_bigints_opt(BigInt::from(num), BigInt::from(den))?;
        debug_assert!(r.mul(r).is_ok_and(|sq| sq >= self));
        ensure_healthy().ok()?;
        Some(r)
    }

    /// Nearest-`f64` approximation of the exact value, for **display and
    /// diagnostics only** — the rounding direction is not certified. Values
    /// beyond `f64` range saturate to ±∞.
    #[must_use]
    pub fn to_f64_approx(self) -> f64 {
        if poisoned() {
            return f64::NAN;
        }
        let value = val(self.id).to_f64().unwrap_or(f64::NAN);
        if poisoned() {
            f64::NAN
        } else {
            value
        }
    }

    /// Emit the canonical certificate string (`"n"` or `"n/d"`) with the full
    /// arbitrary-precision numerator and denominator.
    ///
    /// # Errors
    /// Returns [`RatError::Poisoned`] after an arena fallback.
    pub fn to_clean_string(self) -> Result<String, RatError> {
        let (num, den) = self.checked_parts()?;
        if den.is_one() {
            Ok(num.to_string())
        } else {
            Ok(format!("{num}/{den}"))
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
        let q = n.checked_div(&x).unwrap_or(BigUint::zero());
        let y = (&x + q) >> 1;
        if y >= x {
            break;
        }
        x = y;
    }
    while &x * &x > *n {
        // `x >= 1` is a loop invariant here (x enters the loop `>= 1`, and the guard
        // `x*x > n` with `n >= 1` cannot hold at `x <= 1` — `1*1 = 1 <= n` — so this
        // body runs only for `x >= 2` and the decrement never underflows). The
        // verifier cannot discharge that BigUint loop invariant, so use the same
        // fail-soft idiom as `checked_div` above: `checked_sub` is TOTAL, its `None`
        // arm is UNREACHABLE and fails safe to `0` (still terminates: `0*0 = 0 <= n`).
        // Value identical for every real input; removes the SubAssign underflow panic.
        x = x.checked_sub(&BigUint::one()).unwrap_or(BigUint::zero());
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
        // `try_with` is TOTAL (returns `Result`, never panics), unlike `with` which
        // panics if the thread-local is accessed during its own destruction. The
        // `Err` (arena already destroyed) arm IS REACHABLE from the `cfg(test)`
        // teardown probe in `Arena::drop`, which re-enters `cmp` from the
        // destructor — see `destroyed_arena_outer_fallbacks_latch_poison`. It
        // POISON-MARKS before falling back to `Equal`, mirroring the `try_borrow`
        // fail-soft below (that INNER arm does stay unreachable for the
        // non-reentrant arena).
        let result = ARENA.try_with(|a| {
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
        });
        match result {
            Ok(ordering) => ordering,
            Err(_) => {
                // Equal is only a totalizing value; poison makes it impossible
                // for a certificate boundary to accept this substituted order.
                poison();
                Ordering::Equal
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destroyed_arena_outer_fallbacks_latch_poison() {
        ARENA_TEARDOWN_RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        std::thread::spawn(|| {
            ARENA_TEARDOWN_PROBE_ARMED.with(|probe| probe.set(true));
            set_poisoned_for_test(false);
            // Initialize the lazy, drop-bearing arena. Its cfg(test) Drop probe
            // runs after std has marked this LocalKey destroyed.
            let _ = intern(BigRational::one());
        })
        .join()
        .expect("arena teardown probe thread must not panic");

        assert_eq!(
            ARENA_TEARDOWN_RESULT.load(std::sync::atomic::Ordering::Acquire),
            0b111,
            "intern, val, and cmp outer fallbacks must each latch poison"
        );
    }

    /// Keep the thread-local poison flag from leaking when an assertion panics.
    struct PoisonReset;

    impl Drop for PoisonReset {
        fn drop(&mut self) {
            set_poisoned_for_test(false);
        }
    }

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
        assert_eq!(
            Rat::from_bigints_opt(BigInt::from(1), BigInt::from(0)),
            None
        );
    }

    #[test]
    fn bigint_option_constructor_matches_checked_constructor() {
        for (num, den) in [(6_i32, 8_i32), (-9, 12), (0, 7), (5, -11)] {
            let num = BigInt::from(num);
            let den = BigInt::from(den);
            assert_eq!(
                Rat::from_bigints_opt(num.clone(), den.clone()),
                Rat::from_bigints(num, den).ok()
            );
        }
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
    fn huge_arithmetic_matches_big_rational() {
        let pow = BigInt::from(2u8).pow(4096);
        let shared_den = pow.clone();
        let a_big = BigRational::new(BigInt::from(3), shared_den.clone());
        let b_big = BigRational::new(BigInt::from(5), shared_den);
        let a = Rat::from_bigints(a_big.numer().clone(), a_big.denom().clone()).unwrap();
        let b = Rat::from_bigints(b_big.numer().clone(), b_big.denom().clone()).unwrap();

        assert_eq!(a.add(b).unwrap().to_big(), &a_big + &b_big);
        assert_eq!(a.sub(b).unwrap().to_big(), &a_big - &b_big);

        let u_big = BigRational::new(&pow + 1_u8, &pow + 3_u8);
        let v_big = BigRational::new(&pow + 5_u8, &pow + 7_u8);
        let u = Rat::from_bigints(u_big.numer().clone(), u_big.denom().clone()).unwrap();
        let v = Rat::from_bigints(v_big.numer().clone(), v_big.denom().clone()).unwrap();
        assert_eq!(u.add(v).unwrap().to_big(), &u_big + &v_big);
        assert_eq!(u.sub(v).unwrap().to_big(), &u_big - &v_big);
        assert_eq!(u.mul(v).unwrap().to_big(), &u_big * &v_big);

        let reciprocal = u.inv().unwrap();
        assert_eq!(reciprocal.to_big(), u_big.recip());
        assert_eq!(u.mul(reciprocal).unwrap(), Rat::ONE);
        assert_eq!(u.neg().mul(reciprocal).unwrap(), Rat::from_int(-1));
        assert_eq!(Rat::ZERO.inv(), Err(RatError::ZeroDenominator));
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
        // directly through the test-only setter. Thread-local: this cannot leak
        // into tests on other threads, and the guard clears it even on panic.
        assert!(!poisoned());
        set_poisoned_for_test(true);
        let _reset = PoisonReset;
        assert!(poisoned());
    }

    #[test]
    fn checked_operations_fail_closed_after_poison() {
        let a = Rat::new(1, 3).unwrap();
        let b = Rat::new(1, 6).unwrap();
        set_poisoned_for_test(true);
        let _reset = PoisonReset;

        assert_eq!(Rat::new(1, 2), Err(RatError::Poisoned));
        assert_eq!(
            Rat::from_bigints(BigInt::from(1), BigInt::from(2)),
            Err(RatError::Poisoned)
        );
        assert_eq!(
            Rat::from_bigints_opt(BigInt::from(1), BigInt::from(2)),
            None
        );
        assert_eq!(Rat::from_f32_exact(0.5), None);
        assert_eq!(a.checked_parts(), Err(RatError::Poisoned));
        assert_eq!(a.add(b), Err(RatError::Poisoned));
        assert_eq!(a.sub(b), Err(RatError::Poisoned));
        assert_eq!(a.mul(b), Err(RatError::Poisoned));
        assert_eq!(a.inv(), Err(RatError::Poisoned));
        assert_eq!(a.to_clean_string(), Err(RatError::Poisoned));
        assert_eq!(a.sqrt_upper(32), None);
        assert!(a.to_f64_approx().is_nan());

        // The infallible legacy operations remain total, but can neither clear
        // poison nor make their results cross a checked boundary.
        assert_eq!(Rat::from_int(7), Rat::ZERO);
        assert_eq!(a.neg(), a);
        assert!(poisoned());
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
        // Extremes stay exact: f32::MAX and boundary normal/subnormal values.
        let max = Rat::from_f32_exact(f32::MAX).unwrap();
        assert_eq!(max.den(), BigInt::from(1));
        let min_sub = Rat::from_f32_exact(f32::from_bits(0x0000_0001)).unwrap();
        let neg_min_sub = Rat::from_f32_exact(f32::from_bits(0x8000_0001)).unwrap();
        assert_eq!(min_sub.num(), BigInt::from(1));
        assert_eq!(neg_min_sub.num(), BigInt::from(-1));
        assert_eq!(min_sub.den(), BigInt::from(2u8).pow(149));
        assert_eq!(neg_min_sub.den(), BigInt::from(2u8).pow(149));

        let max_sub = Rat::from_f32_exact(f32::from_bits(0x007f_ffff)).unwrap();
        let neg_max_sub = Rat::from_f32_exact(f32::from_bits(0x807f_ffff)).unwrap();
        let max_sub_num = BigInt::from((1_u32 << 23) - 1);
        assert_eq!(max_sub.num(), max_sub_num);
        assert_eq!(neg_max_sub.num(), -max_sub_num);
        assert_eq!(max_sub.den(), BigInt::from(2u8).pow(149));
        assert_eq!(neg_max_sub.den(), BigInt::from(2u8).pow(149));

        let min_normal = Rat::from_f32_exact(f32::from_bits(0x0080_0000)).unwrap();
        assert_eq!(min_normal.num(), BigInt::from(1));
        assert_eq!(min_normal.den(), BigInt::from(2u8).pow(126));
        // Non-finite fails closed.
        assert_eq!(Rat::from_f32_exact(f32::NAN), None);
        assert_eq!(Rat::from_f32_exact(f32::INFINITY), None);
        assert_eq!(Rat::from_f32_exact(f32::NEG_INFINITY), None);
    }

    #[test]
    fn from_f64_exact_is_lossless_and_fails_closed() {
        assert_eq!(Rat::from_f64_exact(0.0), Some(Rat::ZERO));
        assert_eq!(Rat::from_f64_exact(-0.0), Some(Rat::ZERO));
        assert_eq!(Rat::from_f64_exact(0.5), Some(Rat::new(1, 2).unwrap()));
        assert_eq!(Rat::from_f64_exact(-2.75), Some(Rat::new(-11, 4).unwrap()));

        let min_sub = Rat::from_f64_exact(f64::from_bits(0x0000_0000_0000_0001)).unwrap();
        let neg_min_sub = Rat::from_f64_exact(f64::from_bits(0x8000_0000_0000_0001)).unwrap();
        assert_eq!(min_sub.num(), BigInt::from(1));
        assert_eq!(neg_min_sub.num(), BigInt::from(-1));
        assert_eq!(min_sub.den(), BigInt::from(2_u8).pow(1074));
        assert_eq!(neg_min_sub.den(), BigInt::from(2_u8).pow(1074));

        let max_sub = Rat::from_f64_exact(f64::from_bits(0x000f_ffff_ffff_ffff)).unwrap();
        let neg_max_sub = Rat::from_f64_exact(f64::from_bits(0x800f_ffff_ffff_ffff)).unwrap();
        let max_sub_num = BigInt::from((1_u64 << 52) - 1);
        assert_eq!(max_sub.num(), max_sub_num);
        assert_eq!(neg_max_sub.num(), -max_sub_num);
        assert_eq!(max_sub.den(), BigInt::from(2_u8).pow(1074));
        assert_eq!(neg_max_sub.den(), BigInt::from(2_u8).pow(1074));

        let min_normal = Rat::from_f64_exact(f64::from_bits(0x0010_0000_0000_0000)).unwrap();
        assert_eq!(min_normal.num(), BigInt::from(1));
        assert_eq!(min_normal.den(), BigInt::from(2_u8).pow(1022));
        let max = Rat::from_f64_exact(f64::MAX).unwrap();
        assert_eq!(max.den(), BigInt::from(1));

        assert_eq!(Rat::from_f64_exact(f64::NAN), None);
        assert_eq!(Rat::from_f64_exact(f64::INFINITY), None);
        assert_eq!(Rat::from_f64_exact(f64::NEG_INFINITY), None);
    }
}
