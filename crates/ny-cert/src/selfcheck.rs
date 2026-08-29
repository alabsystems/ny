// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! In-tree mirror of Clean's external-certificate verification logic.
//!
//! This reproduces the algorithm in `clean-elab/src/cert/external/verify.rs`
//! (normalize each constraint to `≤` form, take the non-negative multiplier
//! combination, check the residual) so NY's own tests can confirm a certificate
//! is well formed *before* it is handed to Clean. Clean's kernel-side verifier
//! remains the ground truth; this is a fast local pre-flight, not a substitute.

use crate::rational::{poisoned, Rat, RatError};
use crate::schema::{ConstraintKind, EntailmentCertificate, FarkasCertificate, LinearConstraint};
use std::collections::BTreeMap;
// Contracts are written as the BARE `#[ensures]`. Under tRustc with contract
// verification (`--cfg trust_verify`), `#[ensures]` is the first-class builtin
// (`kw::ContractEnsures`) that emits a *static postcondition VC* — so the NY-owned
// compatibility macro must NOT be imported then, or it shadows the builtin and
// degrades the contract to a runtime-checked closure (no static L1 proof). Under
// stable rustc / non-contract builds, the macro provides the no-op `#[ensures]`.
// `#[trust::cite]` stays a documented grounding pointer (verified by `cite_check`).
#[cfg(trust_verify)]
use core::contracts::ensures;
#[cfg(not(trust_verify))]
use trust::ensures;

/// Why a certificate failed local verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckError {
    /// Premise/constraint and multiplier counts differ.
    LengthMismatch(usize, usize),
    /// A multiplier was negative.
    MultiplierNegative(usize),
    /// The combination did not cancel all variables.
    UncancelledVariables(Vec<String>),
    /// The residual did not establish the claimed fact.
    NotEstablished,
    /// A chain step does not restate the previous step's conclusion as a premise.
    ChainBreak(usize),
    /// An entailment chain was empty.
    EmptyChain,
    /// The conclusion was not a single inequality (e.g. an `Eq`, which normalizes
    /// to two `≤` parts). Entailment conclusions must be a single inequality.
    NonInequalityConclusion(usize),
    /// Exact arithmetic failure.
    Rat(RatError),
}

// Hand-written replacements for the `thiserror::Error` derive (removed so that
// ny-cert carries no proc-macro dependency and the bignum stack can itself be
// verified rather than assumed). The message strings below are byte-identical to
// the former `#[error("…")]` attributes — they appear in certificates and test
// assertions and must not be reworded.
impl core::fmt::Display for CheckError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CheckError::LengthMismatch(constraints, multipliers) => write!(
                f,
                "length mismatch: {0} constraints vs {1} multipliers",
                constraints, multipliers
            ),
            CheckError::MultiplierNegative(idx) => {
                write!(f, "multiplier {0} is negative", idx)
            }
            CheckError::UncancelledVariables(vars) => {
                write!(f, "residual still has variable coefficients: {0:?}", vars)
            }
            CheckError::NotEstablished => {
                write!(f, "residual does not establish the conclusion")
            }
            CheckError::ChainBreak(idx) => write!(
                f,
                "chain break at step {0}: previous conclusion is not a premise",
                idx
            ),
            CheckError::EmptyChain => {
                write!(f, "entailment chain must contain at least one step")
            }
            CheckError::NonInequalityConclusion(parts) => write!(
                f,
                "entailment conclusion must be a single inequality, got {0} normalized parts",
                parts
            ),
            // Was `#[error(transparent)]`: the derive forwards `Display`
            // verbatim to the wrapped error, adding no text of its own.
            CheckError::Rat(e) => core::fmt::Display::fmt(e, f),
        }
    }
}

impl core::error::Error for CheckError {
    /// Per-variant `source()` exactly as the derive produced it: every variant
    /// without a `#[from]`/`#[source]` field is a chain root (`None`), and the
    /// one `#[error(transparent)] Rat(#[from] RatError)` variant *forwards* to
    /// the inner error's own `source()` rather than returning `Some(inner)` —
    /// that is what `#[error(transparent)]` generates, and `RatError` is itself
    /// a chain root, so this is `None` today for every input. Forwarding (not
    /// `Some`) keeps the error chain byte-for-byte what it was.
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            CheckError::Rat(e) => core::error::Error::source(e),
            CheckError::LengthMismatch(..)
            | CheckError::MultiplierNegative(..)
            | CheckError::UncancelledVariables(..)
            | CheckError::NotEstablished
            | CheckError::ChainBreak(..)
            | CheckError::EmptyChain
            | CheckError::NonInequalityConclusion(..) => None,
        }
    }
}

/// Replaces the `#[from]` on `CheckError::Rat`, so `?` on a [`RatError`] still
/// converts (`combine`/`add_scaled` rely on it).
impl From<RatError> for CheckError {
    fn from(source: RatError) -> Self {
        CheckError::Rat(source)
    }
}

/// A constraint normalized to `coeffs · x  (<|≤)  constant`.
struct Normalized {
    coeffs: BTreeMap<String, Rat>,
    constant: Rat,
    strict: bool,
}

impl Normalized {
    fn zero() -> Self {
        Normalized {
            coeffs: BTreeMap::new(),
            constant: Rat::ZERO,
            strict: false,
        }
    }

    fn add_scaled(&mut self, other: &Normalized, factor: Rat) -> Result<(), RatError> {
        if factor.is_zero() {
            return Ok(());
        }
        for (var, coeff) in &other.coeffs {
            let scaled = coeff.mul(factor)?;
            let entry = self.coeffs.remove(var);
            let next = match entry {
                Some(existing) => existing.add(scaled)?,
                None => scaled,
            };
            if next.is_zero() {
                self.coeffs.remove(var);
            } else {
                self.coeffs.insert(var.clone(), next);
            }
        }
        self.constant = self.constant.add(other.constant.mul(factor)?)?;
        self.strict = self.strict || other.strict;
        Ok(())
    }
}

/// Coefficient map with every value negated (`v -> -v`), for flipping a `>=`
/// constraint into `<=` form.
///
/// A named free fn with an explicit insert loop: (a) a direct call resolves to
/// this bundled, verified body, whereas the old local closure minted an
/// unresolvable `<{closure}> as Fn>::call` absent-callee obligation at every
/// call site; (b) the loop (not `.map(..).collect()`) avoids both the inner
/// map-closure row and the collect bulk-alloc obligation — identical entries.
fn negated_coeffs(m: &BTreeMap<String, Rat>) -> BTreeMap<String, Rat> {
    let mut out = BTreeMap::new();
    for (k, v) in m {
        out.insert(k.clone(), v.neg());
    }
    out
}

/// Normalize a constraint to one or more `≤` forms (`Eq` splits into two).
fn normalize(c: &LinearConstraint) -> Vec<Normalized> {
    // `Vec::new()` + push (not `vec![…]` literals): the vec! macro's boxed-slice
    // expansion carries an alloc-internal Sub the strict verifier fail-closes on
    // when inlined into callers (charged to check_farkas via alloc/src/macros.rs
    // spans). Push growth emits no macro-internal arithmetic — same durable
    // pattern as `exact::solve_system`. Identical resulting Vecs.
    let mut out = Vec::new();
    match c.kind {
        ConstraintKind::Le => out.push(Normalized {
            coeffs: c.coefficients.clone(),
            constant: c.constant,
            strict: false,
        }),
        ConstraintKind::Lt => out.push(Normalized {
            coeffs: c.coefficients.clone(),
            constant: c.constant,
            strict: true,
        }),
        ConstraintKind::Ge => out.push(Normalized {
            coeffs: negated_coeffs(&c.coefficients),
            constant: c.constant.neg(),
            strict: false,
        }),
        ConstraintKind::Gt => out.push(Normalized {
            coeffs: negated_coeffs(&c.coefficients),
            constant: c.constant.neg(),
            strict: true,
        }),
        ConstraintKind::Eq => {
            out.push(Normalized {
                coeffs: c.coefficients.clone(),
                constant: c.constant,
                strict: false,
            });
            out.push(Normalized {
                coeffs: negated_coeffs(&c.coefficients),
                constant: c.constant.neg(),
                strict: false,
            });
        }
    }
    out
}

fn combine(
    constraints: &[LinearConstraint],
    multipliers: &[Rat],
) -> Result<Normalized, CheckError> {
    if constraints.len() != multipliers.len() {
        return Err(CheckError::LengthMismatch(
            constraints.len(),
            multipliers.len(),
        ));
    }
    let mut combined = Normalized::zero();
    for (idx, (c, m)) in constraints.iter().zip(multipliers).enumerate() {
        if m.is_negative() {
            return Err(CheckError::MultiplierNegative(idx));
        }
        for part in normalize(c) {
            combined.add_scaled(&part, *m)?;
        }
    }
    Ok(combined)
}

/// Verify an entailment certificate the way Clean's kernel-side verifier does.
///
/// On success returns `(derived_bound, claimed_bound)`.
///
/// # Soundness contract (L1 — Trust = Clean fusion)
/// If this returns `Ok((derived, claimed))`, the premises' non-negative multiplier
/// combination cancels to the conclusion's coefficients and `derived ≤ claimed` — so
/// the derived bound is at least as tight as the claimed one. By the cited,
/// Clean-kernel-checked theorem `farkas_premise_combination` (module
/// `Crownproof.Bridge` in the exact pinned Clean dependency), this entails that the premises imply the
/// conclusion. The `#[ensures]` states the locally-provable `derived ≤ claimed`
/// property; `#[trust::cite]` grounds the entailment. L0 safety is tRustc-VERIFIED.
///
/// # Errors
/// Mirrors Clean's failure modes (length mismatch, negative multiplier,
/// uncancelled variables, unmet conclusion).
#[ensures(|r: &Result<(Rat, Rat), CheckError>| !matches!(r, Ok((d, c)) if d > c))]
#[trust::cite(crownproof::farkas_premise_combination)]
// `?` here would desugar to `from_residual` return paths the verifier's
// ordering-witness grounding cannot aggregate over — the explicit match/early
// returns ARE the proof shape (see the extract-then-guard comment).
#[allow(clippy::question_mark)]
pub fn check_entailment(cert: &EntailmentCertificate) -> Result<(Rat, Rat), CheckError> {
    if poisoned() {
        return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
    }
    // Extract-then-guard: makes the `#[ensures]` locally provable. The match
    // only EXTRACTS (the Err arm returns early), the ordering guard is
    // straight-line on the extracted pair, and the tail is a plain
    // `Ok((d, c))` — so every return path constructs its `Ok`/`Err` in the
    // direct predecessor of the return block and the guard's `d <= c` edge
    // dominates the `Ok` (the verifier's ordering-witness grounding window;
    // `check_entailment_inner` owns droppable locals whose end-of-body drops
    // would split its own tail construction out of that window). The guard is
    // unreachable by construction — the inner `ok` check admits only
    // `combined.constant <=|< conclusion.constant` — so this is
    // behavior-identical, fail-closed hardening.
    // Carry the payload through an `Option` — a DIFFERENT type from this
    // function's own `Result<_, CheckError>` — before destructuring it.
    //
    // MEASURED, not guessed. A `TRUST_LENWITNESS_DEBUG=1` probe traced every
    // return path of this function and of `check_chain`, which proves. The
    // discriminator is exact:
    //     check_chain        0 paths with `_0 def=NONE (Call-dest)`
    //     check_entailment   1 path  (bb27->bb28)
    //     check_farkas       1 path  (bb27->bb28)
    //     check_branch_tree  1 path  (bb32->bb33)
    // When `inner()` returns the SAME `Result<_, E>` this wrapper does, the
    // optimizer recognises the tail `Ok((d, c))` as a reconstruction of the
    // value it just destructured and forwards the ORIGINAL Result instead of
    // rebuilding one. `_0` then bottoms out at a Call dest, no in-body wrapper
    // aggregate exists on that path, and the discr gate fails closed — leaving
    // the `#[ensures]` ungrounded. `check_chain` escapes only because its
    // payload arrives in an `Option<(Rat, Rat)>`, so no such forward is
    // type-correct.
    //
    // The `Err` arm was already hardened against the mirror-image problem (see
    // `err_barrier` below); this is the same fix for the `Ok` side.
    let carried: Option<(Rat, Rat)> = match check_entailment_inner(cert) {
        Ok(pair) => Some(pair),
        // `crate::err_barrier` (identity, `#[inline(never)]`): a fresh in-body
        // `Err` aggregate, not a whole-`Result` forward the return-grounding
        // lane cannot see (nor a const-promoted+merged unit variant).
        Err(e) => {
            if poisoned() {
                return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
            }
            return Err(crate::err_barrier(e));
        }
    };
    // `None` is UNREACHABLE: the `Err` arm above returns, so every path that
    // reaches here stored `Some`. It reuses an error this function already
    // constructs rather than a fresh one, so the arm adds no call and no
    // allocation for the verifier to discharge.
    let (d, c) = match carried {
        Some(pair) => pair,
        None => return Err(crate::err_barrier(CheckError::NotEstablished)),
    };
    if d > c {
        if poisoned() {
            return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
        }
        return Err(crate::err_barrier(CheckError::NotEstablished));
    }
    if poisoned() {
        return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
    }
    Ok((d, c))
}

/// The full verification behind [`check_entailment`]. Private and
/// contract-free: the ensures-bearing wrapper re-establishes the `derived ≤
/// claimed` invariant with an in-body ordering guard (this body's `?` return
/// is a `from_residual` path, and its `Vec`/`Normalized` end-of-body drops
/// split the tail `Ok` construction out of the local proof's grounding
/// window).
fn check_entailment_inner(cert: &EntailmentCertificate) -> Result<(Rat, Rat), CheckError> {
    let combined = combine(&cert.premises, &cert.multipliers)?;
    let conclusion = normalize(&cert.conclusion);
    // Fail closed rather than panic: a `pub` verifier entry point must never abort
    // on a malformed certificate (e.g. an `Eq` conclusion, which normalizes to two
    // parts). Clean's kernel-side verifier rejects such a cert; mirror that here.
    if conclusion.len() != 1 {
        return Err(CheckError::NonInequalityConclusion(conclusion.len()));
    }
    // Total access after the `conclusion.len() == 1` guard: `first()` is
    // provably `Some` (the `else` reject is unreachable), and — unlike
    // `&conclusion[0]`, whose `Vec::index` is an absent std callee carrying a
    // may-panic obligation — it lowers with no panic edge at all.
    let Some(conclusion) = conclusion.first() else {
        return Err(CheckError::NonInequalityConclusion(0));
    };
    if combined.coeffs != conclusion.coeffs {
        // Explicit loops (not `.keys().chain(..).cloned().collect()`): keeps the
        // variable gather in verified code (no absent-adapter `Iterator::chain`/
        // `cloned` obligation). Identical multiset before the sort+dedup, which
        // makes push order irrelevant.
        let mut vars: Vec<String> = Vec::new();
        for k in combined.coeffs.keys() {
            vars.push(k.clone());
        }
        for k in conclusion.coeffs.keys() {
            vars.push(k.clone());
        }
        vars.sort();
        vars.dedup();
        return Err(CheckError::UncancelledVariables(vars));
    }
    let ok = if combined.strict && !conclusion.strict {
        combined.constant <= conclusion.constant
    } else if !combined.strict && conclusion.strict {
        combined.constant < conclusion.constant
    } else {
        combined.constant <= conclusion.constant
    };
    if ok {
        Ok((combined.constant, conclusion.constant))
    } else {
        Err(CheckError::NotEstablished)
    }
}

/// True when two constraints have the same normalized (`≤`-form) representation.
/// Equality constraints (which normalize to two parts) never match.
fn normalized_eq(a: &LinearConstraint, b: &LinearConstraint) -> bool {
    let (na, nb) = (normalize(a), normalize(b));
    if na.len() != 1 || nb.len() != 1 {
        return false;
    }
    // Total `.first()` reads (the `len != 1` guard already makes both `Some`);
    // avoids the `[0]` slice-bounds obligation upstream vcgen intermittently
    // stops discharging under strict.
    match (na.first(), nb.first()) {
        (Some(x), Some(y)) => {
            x.coeffs == y.coeffs && x.constant == y.constant && x.strict == y.strict
        }
        _ => false,
    }
}

/// Verify NY's legacy composed-entailment linkage schema: each step is a valid
/// entailment, and every step after the first
/// restates the previous step's conclusion among its premises (the cut rule).
///
/// On success returns the final step's conclusion bound `(derived, claimed)`.
///
/// # Soundness contract (L1 — Trust = Clean fusion)
/// If this returns `Ok((derived, claimed))`, every step is a valid entailment (each
/// grounded as in [`check_entailment`]) and each step after the first restates the
/// previous conclusion among its premises (the cut rule), so the chain transitively
/// composes to a single valid entailment with `derived ≤ claimed`. By transitive
/// application of the cited `farkas_premise_combination` (one Farkas combination per
/// step, composed via the cut rule), the chain's premises imply its final conclusion.
/// The `#[ensures]` states the locally-provable `derived ≤ claimed`; the
/// `#[trust::cite]` grounds the composition. L0 safety is tRustc-VERIFIED.
///
/// # Errors
/// [`CheckError::EmptyChain`] for an empty chain, [`CheckError::ChainBreak`] for
/// a missing linkage, or any single-step failure.
#[ensures(|r: &Result<(Rat, Rat), CheckError>| !matches!(r, Ok((d, c)) if d > c))]
#[trust::cite(crownproof::farkas_premise_combination)]
// `?` here would desugar to `from_residual` return paths the verifier's
// ordering-witness grounding cannot aggregate over — the explicit match/early
// returns ARE the proof shape (see the extract-then-guard comment).
#[allow(clippy::question_mark)]
pub fn check_chain(steps: &[EntailmentCertificate]) -> Result<(Rat, Rat), CheckError> {
    if poisoned() {
        return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
    }
    let mut last: Option<(Rat, Rat)> = None;
    let mut prev_conclusion: Option<&LinearConstraint> = None;
    for (idx, step) in steps.iter().enumerate() {
        // Explicit match (not `?`): a `from_residual` return path would fall
        // outside the verifier's grounding window; the Err arm constructs its
        // return value in-body. Identical propagation.
        let bounds = match check_entailment(step) {
            Ok(b) => b,
            // `crate::err_barrier` (identity, `#[inline(never)]`): fresh in-body
            // `Err` aggregate, not a whole-`Result` forward.
            Err(e) => return Err(crate::err_barrier(e)),
        };
        if let Some(prev) = prev_conclusion {
            // Explicit loop (not `.any(|p| ..)`): avoids the `Iterator::any`
            // absent consumer + its closure row — identical short-circuit.
            let mut found_premise = false;
            for p in step.premises.iter() {
                if normalized_eq(p, prev) {
                    found_premise = true;
                    break;
                }
            }
            if !found_premise {
                if poisoned() {
                    return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
                }
                return Err(crate::err_barrier(CheckError::ChainBreak(idx)));
            }
            if poisoned() {
                return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
            }
        }
        prev_conclusion = Some(&step.conclusion);
        last = Some(bounds);
    }
    // Extract-then-guard (probe-v9 shape): extract the final bounds with an
    // explicit match (not `ok_or`, whose call-dest return the verifier's
    // grounding window cannot resolve), then re-establish the `#[ensures]`
    // ordering invariant with an in-body guard dominating the `Ok`. The guard
    // is unreachable by construction — every `bounds` came from
    // [`check_entailment`], whose own guard admits only `d <= c` — so this is
    // behavior-identical, fail-closed hardening.
    let (d, c) = match last {
        Some(pair) => pair,
        None => return Err(crate::err_barrier(CheckError::EmptyChain)),
    };
    if d > c {
        if poisoned() {
            return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
        }
        return Err(crate::err_barrier(CheckError::NotEstablished));
    }
    if poisoned() {
        return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
    }
    Ok((d, c))
}

/// Verify a Farkas infeasibility certificate the way Clean does.
///
/// # Soundness contract (L1 — Trust = Clean fusion)
/// If this returns `Ok(c)`, the certificate's non-negative multiplier combination of
/// its `≤ 0` premise constraints cancels every variable and yields the constant
/// `c ≤ 0` — a Farkas contradiction. By the cited, Clean-kernel-checked theorem
/// `farkas_premise_combination` (`Crownproof.Bridge` in the exact pinned Clean dependency),
/// that contradiction entails the claimed output bound holds for every input in the
/// region. The `#[ensures]` states the locally-provable contradiction
/// property; `#[trust::cite]` grounds the entailment in the kernel-checked proof.
/// L0 safety is VERIFIED by tRustc; full L1 discharge awaits the `cite` VC-lemma
/// mechanism (see `ny-cert/SPEC.md` and task #24).
///
/// # Errors
/// Mirrors Clean's failure modes; [`CheckError::NotEstablished`] when the
/// combination does not collapse to a contradiction.
#[ensures(|r: &Result<Rat, CheckError>| !matches!(r, Ok(c) if c.is_positive()))]
#[trust::cite(crownproof::farkas_premise_combination)]
// `?` here would desugar to `from_residual` return paths the verifier's
// ordering-witness grounding cannot aggregate over — the explicit match/early
// returns ARE the proof shape (see the extract-then-guard comment).
#[allow(clippy::question_mark)]
pub fn check_farkas(cert: &FarkasCertificate) -> Result<Rat, CheckError> {
    if poisoned() {
        return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
    }
    // Extract-then-guard (probe-v10 shape): the match only EXTRACTS (the Err
    // arm returns early), the sign guard is straight-line on the extracted
    // constant, and the tail is a plain `Ok(c)` — so every return path
    // constructs its `Ok`/`Err` in the direct predecessor of the return block
    // and the guard's `!is_positive` edge dominates the `Ok` (the verifier's
    // sign-witness grounding window; `check_farkas_inner` owns droppable
    // locals whose end-of-body drops would split its own tail construction
    // out of that window). The guard is unreachable by construction — the
    // inner `contradiction` check admits only `!is_positive` (strict) /
    // `is_negative` (non-strict) constants, and both imply `!is_positive` —
    // so this is behavior-identical, fail-closed hardening.
    // `Option` carrier: see the identical note in `check_entailment` above. A
    // `TRUST_LENWITNESS_DEBUG=1` probe traced one return path here (bb27->bb28)
    // whose `_0 def=NONE (Call-dest)`, against zero such paths in `check_chain`,
    // which proves.
    let carried: Option<Rat> = match check_farkas_inner(cert) {
        Ok(c) => Some(c),
        // `crate::err_barrier` (identity, `#[inline(never)]`): a fresh in-body
        // `Err` aggregate, not a whole-`Result` forward the return-grounding
        // lane cannot see (nor a const-promoted+merged unit variant).
        Err(e) => {
            if poisoned() {
                return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
            }
            return Err(crate::err_barrier(e));
        }
    };
    // UNREACHABLE: the `Err` arm returns, so every path here stored `Some`.
    // Reuses an error this function already constructs, adding no call.
    let c = match carried {
        Some(c) => c,
        None => return Err(crate::err_barrier(CheckError::NotEstablished)),
    };
    if c.is_positive() {
        if poisoned() {
            return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
        }
        return Err(crate::err_barrier(CheckError::NotEstablished));
    }
    if poisoned() {
        return Err(crate::err_barrier(CheckError::Rat(RatError::Poisoned)));
    }
    Ok(c)
}

/// The full verification behind [`check_farkas`]. Private and contract-free:
/// the ensures-bearing wrapper re-establishes the non-positive-contradiction
/// invariant with an in-body sign guard (this body's `?` return is a
/// `from_residual` path, and its `Normalized` end-of-body drop splits the
/// tail `Ok` construction out of the local proof's grounding window).
fn check_farkas_inner(cert: &FarkasCertificate) -> Result<Rat, CheckError> {
    let combined = combine(&cert.constraints, &cert.multipliers)?;
    if !combined.coeffs.is_empty() {
        // Explicit loop (not `.keys().cloned().collect()`): the cloned/collect
        // adapters are absent-callees for the panic-freedom checker; the loop
        // yields the identical multiset (order is irrelevant — `vars.sort()`
        // canonicalizes it immediately below).
        let mut vars: Vec<String> = Vec::new();
        for k in combined.coeffs.keys() {
            vars.push(k.clone());
        }
        vars.sort();
        return Err(CheckError::UncancelledVariables(vars));
    }
    let contradiction = if combined.strict {
        !combined.constant.is_positive()
    } else {
        combined.constant.is_negative()
    };
    if contradiction {
        Ok(combined.constant)
    } else {
        Err(CheckError::NotEstablished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PoisonReset;

    impl Drop for PoisonReset {
        fn drop(&mut self) {
            crate::rational::set_poisoned_for_test(false);
        }
    }

    #[test]
    fn public_checkers_refuse_a_poisoned_arena_even_on_trivial_inputs() {
        let premise = LinearConstraint::with_kind(ConstraintKind::Le, &[("x", Rat::ONE)], Rat::ONE);
        let entailment = EntailmentCertificate {
            premises: vec![premise.clone()],
            multipliers: vec![Rat::ONE],
            conclusion: premise,
        };
        let farkas = FarkasCertificate {
            constraints: Vec::new(),
            multipliers: Vec::new(),
        };

        crate::rational::set_poisoned_for_test(true);
        let _reset = PoisonReset;
        let expected = CheckError::Rat(RatError::Poisoned);
        assert_eq!(check_entailment(&entailment), Err(expected.clone()));
        assert_eq!(check_chain(&[]), Err(expected.clone()));
        assert_eq!(check_farkas(&farkas), Err(expected));
    }
}
