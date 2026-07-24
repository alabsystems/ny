// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SBAR: certificate-producing simplex-barycentric attention relaxation.
//!
//! This is the executable, Clean-checkable core of Pillar 2
//! (`docs/SBAR_ATTENTION_RELAXATION.md`). For a single attention (head, query)
//! the support function of the relaxed output in a readout direction `w` reduces
//! (after the *exact* inner-`V` corner step, §3.2) to a linear program over the
//! **box-truncated probability simplex**
//!
//! ```text
//!   maximize   Σ_j g_j · p_j
//!   subject to Σ_j p_j = 1,   p_lo_j ≤ p_j ≤ p_hi_j
//! ```
//!
//! where `g_j` is the per-unit-mass value contribution of attending to position
//! `j` and `[p_lo_j, p_hi_j]` are the softmax-monotone weight bounds (§2). The
//! optimum is found by **water-filling** (§3.3), and LP strong duality gives a
//! closed-form dual `(λ, μ⁺, μ⁻)` (§5) that *certifies* the bound without
//! re-solving. We emit that dual as an exact-rational entailment certificate
//! `Σ_j g_j p_j ≤ U` over the simplex+box premises — accepted by Clean's real
//! external-certificate verifier.
//!
//! The simplex equality `Σ p_j = 1` is supplied as an independent `le`/`ge`
//! pair (not a single `eq`) for the same reason as the affine layers in
//! [`crate::crown`]: Clean scales an `eq` premise's two halves by one multiplier
//! that cancels, so the free equality multiplier `λ` (which may be negative)
//! must be carried by whichever half matches its sign.

use crate::rational::{Rat, RatError};
use crate::schema::{ConstraintKind, EntailmentCertificate, LinearConstraint};
// Contracts are written as the BARE `#[ensures]` (see `selfcheck.rs` for the full
// rationale): under tRustc contract verification (`--cfg trust_verify`) it is the
// first-class builtin that emits a static postcondition VC, so the NY-owned
// compatibility macro must NOT be imported then or it shadows the builtin. Under
// stable rustc the macro provides the no-op `#[ensures]`. `#[trust::cite]`
// stays a documented grounding pointer (verified by `cite_check`).
#[cfg(trust_verify)]
use core::contracts::ensures;
#[cfg(not(trust_verify))]
use trust::ensures;

/// A box-truncated-simplex support LP for one attention (head, query) and
/// readout direction, in exact rationals.
#[derive(Debug, Clone)]
pub struct SimplexSupportLp {
    /// Per-position value contributions `g_j = Σ_c w_c · v̂_{j,c}` (§3.2).
    pub g: Vec<Rat>,
    /// Softmax-monotone lower weight bounds `p_lo_j` (§2).
    pub p_lo: Vec<Rat>,
    /// Softmax-monotone upper weight bounds `p_hi_j` (§2).
    pub p_hi: Vec<Rat>,
}

/// Errors from building or certifying an SBAR support bound.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SbarError {
    /// The `g`, `p_lo`, `p_hi` vectors had mismatched lengths.
    #[error("dimension mismatch")]
    Dimension,
    /// `Σ p_lo > 1` or `Σ p_hi < 1`: the truncated simplex is empty, so these
    /// are not valid softmax weight bounds.
    #[error("box-truncated simplex is infeasible (Σp_lo>1 or Σp_hi<1)")]
    Infeasible,
    /// A bound was malformed (`p_lo_j > p_hi_j`).
    #[error("p_lo[{0}] > p_hi[{0}]")]
    BadBox(usize),
    /// The LP dual objective did not equal the primal optimum exactly, an
    /// internal strong-duality invariant. Reported (fail-closed) instead of
    /// asserting so the certifier rejects rather than panicking; unreachable
    /// for valid simplex+box inputs.
    #[error("internal LP strong-duality check failed (dual != primal)")]
    DualityGap,
    /// Exact arithmetic failure.
    #[error(transparent)]
    Rat(#[from] RatError),
}

/// A certified SBAR upper bound on `Σ_j g_j p_j` over the truncated simplex.
#[derive(Debug, Clone)]
pub struct SbarUpperCert {
    /// The optimal (tight) bound `U = max Σ g_j p_j`.
    pub bound: Rat,
    /// Equality dual `λ` (free sign).
    pub lambda: Rat,
    /// Upper-bound duals `μ⁺_j = max(0, g_j − λ) ≥ 0`.
    pub mu_plus: Vec<Rat>,
    /// Lower-bound duals `μ⁻_j = max(0, λ − g_j) ≥ 0`.
    pub mu_minus: Vec<Rat>,
    /// Entailment certificate proving `Σ_j g_j p_j ≤ U` from the simplex+box.
    pub entailment: EntailmentCertificate,
}

/// Decimal variable name `p{j}` by manual ASCII construction (`String::from` +
/// divmod digit-push, no `format!`): extern `core::fmt` dispatch is opaque to
/// the verifier. Byte-identical to `format!("p{j}")` for every `usize` —
/// no leading zeros, and `j == 0` renders as `"p0"`.
fn pvar(j: usize) -> String {
    // total: `d = n % 10` is always in `0..=9`, so every arm is exact; the `_`
    // arm doubles as the (exact) `9` case, leaving no panic edge and no byte
    // arithmetic for the verifier to bound.
    fn digit_char(d: usize) -> char {
        match d {
            0 => '0',
            1 => '1',
            2 => '2',
            3 => '3',
            4 => '4',
            5 => '5',
            6 => '6',
            7 => '7',
            8 => '8',
            _ => '9',
        }
    }
    let mut s = String::from("p");
    if j == 0 {
        s.push('0');
        return s;
    }
    // Divmod digit-push, least-significant digit first (`/`/`%` by the nonzero
    // literal 10: no division panic edge; `n` strictly decreases, so the loop
    // terminates). `Vec::new()` (not `with_capacity`) per the allocation
    // convention in `build_entailment`.
    let mut digits: Vec<char> = Vec::new();
    let mut n = j;
    while n > 0 {
        // `checked_rem`/`checked_div` by the constant 10: behaviour-identical
        // (10 != 0, the fallbacks are unreachable) and they carry NO MIR
        // zero-divisor assert — the bare `%`/`/` raised divzero obligations
        // the solver lane left unknown (the range_i128 checked_rem lesson).
        digits.push(digit_char(n.checked_rem(10).unwrap_or(0)));
        n = n.checked_div(10).unwrap_or(0);
    }
    // Append most-significant-first via a descending index loop (no absent
    // `rev`/`reverse` adapters). `k < digits.len()` at every read, so the
    // `'0'` fallback is unreachable — total read, ny-cert idiom.
    let mut k = digits.len();
    while k > 0 {
        // `saturating_sub` (never saturates: `k > 0` inside the loop) instead
        // of `k -= 1`: no MIR Sub-overflow assert (the bare `-=` raised an
        // ArithmeticSafety(Sub) obligation the solver left unknown).
        k = k.saturating_sub(1);
        s.push(digits.get(k).copied().unwrap_or('0'));
    }
    s
}

impl SimplexSupportLp {
    fn validate(&self) -> Result<(), SbarError> {
        let m = self.g.len();
        if self.p_lo.len() != m || self.p_hi.len() != m {
            return Err(SbarError::Dimension);
        }
        // Total reads (fail-safe `Rat::ZERO`): `p_lo`/`p_hi` were just pinned to
        // length `m` above, so every `j < m` is in range — the `unwrap_or` arms
        // are unreachable. `.get()` keeps the compare free of a slice-bounds
        // obligation the intraprocedural verifier can't otherwise discharge.
        for j in 0..m {
            let plo = self.p_lo.get(j).copied().unwrap_or(Rat::ZERO);
            let phi = self.p_hi.get(j).copied().unwrap_or(Rat::ZERO);
            if plo > phi {
                return Err(SbarError::BadBox(j));
            }
        }
        Ok(())
    }

    fn sum(vals: &[Rat]) -> Result<Rat, RatError> {
        let mut acc = Rat::ZERO;
        for v in vals {
            acc = acc.add(*v)?;
        }
        Ok(acc)
    }

    /// Evaluate the objective `Σ_j g_j p_j` at a weight vector.
    ///
    /// `p` must have exactly `g.len()` entries; the in-crate callers evaluate
    /// `p_lo`/`p_hi`/vertex vectors only after `validate()` pinned their
    /// lengths to `m = g.len()`. A mismatch is reported as
    /// [`SbarError::Dimension`] rather than asserted, so there is no panic
    /// boundary for the verifier to refute. (Not a `#[trust::requires]`:
    /// method-call contract predicates are currently unparseable by the
    /// contract lowering and become their own FAILED "unverifiable spec"
    /// obligations.)
    ///
    /// # Errors
    /// [`SbarError::Dimension`] on a length mismatch, or exact-arithmetic
    /// overflow.
    pub fn objective(&self, p: &[Rat]) -> Result<Rat, SbarError> {
        if p.len() != self.g.len() {
            return Err(SbarError::Dimension);
        }
        let mut acc = Rat::ZERO;
        for (gj, pj) in self.g.iter().zip(p) {
            acc = acc.add(gj.mul(*pj)?)?;
        }
        Ok(acc)
    }

    /// Solve the support LP by water-filling and emit the certified upper bound
    /// with its closed-form dual entailment certificate.
    ///
    /// # Errors
    /// [`SbarError::Infeasible`] when the truncated simplex is empty,
    /// [`SbarError::Dimension`]/[`SbarError::BadBox`] for malformed input, or
    /// exact-arithmetic overflow.
    ///
    /// The `#[ensures]` states the locally-provable producer well-formedness
    /// invariant: on `Ok` the emitted dual is structurally consistent — equally
    /// many `μ⁺` and `μ⁻` duals (one of each per position) and one non-negative
    /// multiplier per entailment premise — which is what makes the certificate
    /// shape-consistent for the checker. It is stated over the result alone
    /// because the builtin `#[ensures]` closure must be `Copy + 'static` and so
    /// cannot capture `&self`/`self.g` (see `core::contracts::build_check_ensures`).
    /// `#[trust::cite]` grounds the completeness claim — that the water-filling
    /// dual `(λ, μ⁺, μ⁻)` certifies `Σ g_j p_j ≤ bound` over the box-truncated
    /// simplex — in the kernel-checked `sbar_support_sound` theorem (LP weak
    /// duality).
    #[ensures(|r: &Result<SbarUpperCert, SbarError>| !matches!(r, Ok(c) if c.mu_plus.len() != c.mu_minus.len() || c.entailment.premises.len() != c.entailment.multipliers.len()))]
    #[trust::cite(crownproof::sbar_support_sound)]
    pub fn certify_upper(&self) -> Result<SbarUpperCert, SbarError> {
        // Extract-then-guard wrapper. `certify_upper_inner` runs the water-fill
        // producer (with its `?`-carrying exact arithmetic); here two DOMINATING
        // length-equality guards re-establish the `#[ensures]` producer
        // well-formedness postcondition as an IN-BODY fact on the sole `Ok`
        // return path, so the intraprocedural verifier can ground it. The
        // producer's length equalities are established by PAIRED PUSH — `mu_plus`
        // and `mu_minus` get one element each per position, and `build_entailment`
        // pushes to `premises` and `multipliers` in lockstep — but that coupling
        // is interprocedural / loop-carried and invisible to the checker, so the
        // guards restate it locally over the returned aggregate. Both guards are
        // unreachable by construction (the lengths are always equal); fail closed
        // (`Dimension`) rather than assert, so no panic boundary is introduced.
        // No `?` in the wrapper: every return path is an in-body `Ok`/`Err`
        // aggregate, which the return-grounding lane requires to pin `_0_discr`
        // and the payload component lengths.
        // Crown-identical delegator shape (crown::certify PROVES with this exact
        // form): match-extract, straight-line guards, plain `Ok(cert)` tail. The
        // only structural difference from crown is TWO guards over TWO pairs (a
        // flat sibling pair `mu_plus`/`mu_minus` at `.2`/`.3` and a nested pair
        // `entailment.premises`/`multipliers` at `.4.0`/`.4.1`) vs crown's single
        // nested pair. Both guards are unreachable by construction (the producer
        // upholds both equalities by paired push); fail closed (`Dimension`).
        // Explicit match, not `?`: the in-body `Ok`/`Err` return paths are the
        // proof shape the return-grounding lane needs (pins `_0_discr` and the
        // payload component lengths); `?` would desugar to unaggregatable
        // `from_residual` edges.
        #[allow(clippy::question_mark)]
        let cert = match self.certify_upper_inner() {
            Ok(cert) => cert,
            // `crate::err_barrier` (identity): a fresh in-body `Err` aggregate,
            // not a whole-`Result` forward the return-grounding lane cannot see.
            Err(e) => return Err(crate::err_barrier(e)),
        };
        // SINGLE compound guard (not two): one Err return means no merge point
        // for drop-elaboration to route a promoted-Err move through above the
        // return block. Both len pairs still ground — `A` dominates the `Ok`
        // via its `else`, and `B` (checked only when `!A`) dominates it via its
        // own `else`. Barriered so the Err is a fresh in-body aggregate.
        if cert.mu_plus.len() != cert.mu_minus.len()
            || cert.entailment.premises.len() != cert.entailment.multipliers.len()
        {
            return Err(crate::err_barrier(SbarError::Dimension));
        }
        Ok(cert)
    }

    /// Water-fill producer for [`Self::certify_upper`]: solve the support LP and
    /// assemble the dual entailment certificate. The public wrapper adds the two
    /// dominating length-equality guards that discharge the `#[ensures]` producer
    /// well-formedness postcondition.
    fn certify_upper_inner(&self) -> Result<SbarUpperCert, SbarError> {
        self.validate()?;
        let m = self.g.len();
        let sum_lo = Self::sum(&self.p_lo)?;
        let sum_hi = Self::sum(&self.p_hi)?;
        if sum_lo > Rat::ONE || sum_hi < Rat::ONE {
            return Err(SbarError::Infeasible);
        }

        // Sort position indices by g descending (stable, exact comparisons).
        // Allocation cap (`.min(1_048_576)`): syntactic `min(m, C) <= C < 2^28`
        // bound for the checker. `m = g.len()` is the position count (a handful
        // per attention), so `.min` is the identity — behavior-preserving. Same
        // convention as `exact::solve_system`.
        let mut order: Vec<usize> = (0..m.min(1_048_576)).collect();
        // `Rat: Ord` (exact rationals are totally ordered, never NaN), so use the total
        // `cmp` directly — avoids a `partial_cmp(..).expect(..)` panic boundary the verifier
        // (correctly) cannot discharge without a NaN-freedom proof.
        // `a`/`b` range over `order = 0..m`, indices into `g` (len `m`); the
        // `Rat::ZERO` fallbacks are unreachable. Total reads keep the sort key
        // free of a slice-bounds obligation.
        order.sort_by(|&a, &b| {
            let ga = self.g.get(a).copied().unwrap_or(Rat::ZERO);
            let gb = self.g.get(b).copied().unwrap_or(Rat::ZERO);
            gb.cmp(&ga)
        });

        // Water-fill: start at p_lo, pour budget B = 1 − Σp_lo into highest g
        // first. The "water level" λ is the g of the position where the budget
        // runs out (the dual of the simplex equality).
        let budget = Rat::ONE.sub(sum_lo)?;
        let mut bound = self.objective(&self.p_lo)?; // value at the lower vertex
        let mut remaining = budget;
        // Default λ = min g (case: budget pours into everything / all-at-upper).
        // `m >= 1` here (m == 0 ⇒ sum_hi == 0 < 1 ⇒ Infeasible above), but the verifier
        // can't see that, so use `ok_or` instead of `.expect(..)` to discharge the panic
        // boundary — the None arm is an unreachable-but-sound Infeasible fallback.
        let last_idx = *order.last().ok_or(SbarError::Infeasible)?;
        // `last_idx ∈ order = 0..m`, so it indexes `g` (len `m`); the fallback is
        // unreachable.
        let mut lambda = self.g.get(last_idx).copied().unwrap_or(Rat::ZERO);
        let mut acc_slack = Rat::ZERO;
        // Every `t ∈ order = 0..m` indexes `p_hi`/`p_lo`/`g` (all len `m`); the
        // `.get()` fallbacks below are unreachable. Total reads discharge the
        // per-position slice-bounds obligations with no change to the water-fill.
        for &t in &order {
            let phi = self.p_hi.get(t).copied().unwrap_or(Rat::ZERO);
            let plo = self.p_lo.get(t).copied().unwrap_or(Rat::ZERO);
            let gt = self.g.get(t).copied().unwrap_or(Rat::ZERO);
            let slack = phi.sub(plo)?;
            if acc_slack.add(slack)? >= budget {
                // Budget runs out at (or before fully filling) position t.
                lambda = gt;
                let take = budget.sub(acc_slack)?; // ≤ slack, ≥ 0
                bound = bound.add(gt.mul(take)?)?;
                remaining = Rat::ZERO;
                break;
            }
            acc_slack = acc_slack.add(slack)?;
            bound = bound.add(gt.mul(slack)?)?;
            remaining = remaining.sub(slack)?;
        }
        // Feasibility was checked (sum_hi ≥ 1), so the budget is always filled.
        debug_assert!(remaining.is_zero());

        // Closed-form dual (§5): μ⁺_j = max(0, g_j − λ), μ⁻_j = max(0, λ − g_j).
        // Allocation caps (`.min(1_048_576)`, no-op — `m` is the position count):
        // syntactic per-site capacity bounds for the dual vectors.
        let mut mu_plus = Vec::with_capacity(m.min(1_048_576));
        let mut mu_minus = Vec::with_capacity(m.min(1_048_576));
        for gj in &self.g {
            let d = gj.sub(lambda)?;
            if d.is_positive() {
                mu_plus.push(d);
                mu_minus.push(Rat::ZERO);
            } else {
                mu_plus.push(Rat::ZERO);
                mu_minus.push(d.neg());
            }
        }

        // The dual objective λ + Σ μ⁺ p_hi − Σ μ⁻ p_lo equals the primal optimum
        // (LP strong duality); assert the exact match as an internal check.
        let mut dual_val = lambda;
        // `mu_plus`/`mu_minus` were built to length `m` (one push per `g`
        // element) and `p_hi`/`p_lo` are length `m`, so every `j < m` is in
        // range — the `Rat::ZERO` fallbacks are unreachable. Total reads keep the
        // duality cross-check free of slice-bounds obligations.
        for j in 0..m {
            let mup = mu_plus.get(j).copied().unwrap_or(Rat::ZERO);
            let mum = mu_minus.get(j).copied().unwrap_or(Rat::ZERO);
            let phi = self.p_hi.get(j).copied().unwrap_or(Rat::ZERO);
            let plo = self.p_lo.get(j).copied().unwrap_or(Rat::ZERO);
            dual_val = dual_val.add(mup.mul(phi)?)?;
            dual_val = dual_val.sub(mum.mul(plo)?)?;
        }
        // LP strong duality: the dual objective equals the primal optimum for
        // every valid simplex+box input, so this branch is unreachable. Fail
        // closed (reject) instead of asserting the equality, so the verifier
        // sees no panic boundary here.
        if dual_val != bound {
            return Err(SbarError::DualityGap);
        }

        let entailment = self.build_entailment(&mu_plus, &mu_minus, lambda, bound)?;
        Ok(SbarUpperCert {
            bound,
            lambda,
            mu_plus,
            mu_minus,
            entailment,
        })
    }

    /// Assemble the entailment certificate `Σ g_j p_j ≤ U` whose non-negative
    /// multipliers are the LP dual (the λ-half is selected by sign).
    fn build_entailment(
        &self,
        mu_plus: &[Rat],
        mu_minus: &[Rat],
        lambda: Rat,
        bound: Rat,
    ) -> Result<EntailmentCertificate, RatError> {
        let m = self.g.len();
        // Allocation caps (`.min(1_048_576)`, no-op — `m` is the position count,
        // so `2 + 2*m` is tiny): syntactic per-site bounds on the all-ones
        // collect and the premise/multiplier capacities.
        let all_ones: Vec<(String, Rat)> =
            (0..m.min(1_048_576)).map(|j| (pvar(j), Rat::ONE)).collect();
        // `Vec::new()` + push loop (not `collect`): the borrowing collect over
        // `all_ones` has no syntactic count bound at the allocation site, so it
        // carried a hardened allocation obligation. Same idiom as `premises`/
        // `mult` below; identical elements in identical order.
        let mut ones_refs: Vec<(&str, Rat)> = Vec::new();
        for (k, v) in &all_ones {
            ones_refs.push((k.as_str(), *v));
        }

        // `Vec::new()` (not `with_capacity`): the `2 + 2*m` capacity hint over the
        // unbounded `m = g.len()` carries a hardened allocation obligation the
        // checker fail-closes on (the saturating-composed count is not a form it
        // bounds); the growth cost is amortized noise. Same convention as
        // `exact::solve_system`'s augmented-matrix `Vec::new()`.
        let mut premises = Vec::new();
        let mut mult = Vec::new();

        // Premise 0: Σ p_j ≤ 1   (le, multiplier λ if λ ≥ 0 else 0).
        premises.push(LinearConstraint::with_kind(
            ConstraintKind::Le,
            &ones_refs,
            Rat::ONE,
        ));
        mult.push(if lambda.is_negative() {
            Rat::ZERO
        } else {
            lambda
        });
        // Premise 1: Σ p_j ≥ 1   (ge, multiplier −λ if λ < 0 else 0).
        premises.push(LinearConstraint::with_kind(
            ConstraintKind::Ge,
            &ones_refs,
            Rat::ONE,
        ));
        mult.push(if lambda.is_negative() {
            lambda.neg()
        } else {
            Rat::ZERO
        });

        // Box premises: p_j ≤ p_hi_j (μ⁺_j) and p_j ≥ p_lo_j (μ⁻_j).
        //
        // Total `.get()` reads with a fail-safe `Rat::ZERO` fallback: this
        // helper is only called from `certify_upper_inner` AFTER `validate()`
        // established `p_lo.len() == p_hi.len() == g.len() == m`, and the `mu`
        // vectors are built to length `m`, so every `j < m` indexes a real
        // element — the `unwrap_or` arms are unreachable. Reading via `.get()`
        // keeps the accesses TOTAL: the length equalities are INTERPROCEDURAL
        // (the guard lives in `validate`), which the intraprocedural verifier
        // cannot see, so a bare `[j]` would carry an unprovable slice-bounds
        // obligation. Fail-soft, ny-cert idiom; no behavior change for any
        // real input.
        for j in 0..m {
            let phi = self.p_hi.get(j).copied().unwrap_or(Rat::ZERO);
            let plo = self.p_lo.get(j).copied().unwrap_or(Rat::ZERO);
            let mup = mu_plus.get(j).copied().unwrap_or(Rat::ZERO);
            let mum = mu_minus.get(j).copied().unwrap_or(Rat::ZERO);
            premises.push(LinearConstraint::with_kind(
                ConstraintKind::Le,
                &[(pvar(j).as_str(), Rat::ONE)],
                phi,
            ));
            mult.push(mup);
            premises.push(LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[(pvar(j).as_str(), Rat::ONE)],
                plo,
            ));
            mult.push(mum);
        }

        // Conclusion: Σ g_j p_j ≤ U.
        // `Vec::new()` + push (not a `(0..m.min(C)).collect()`): the newer base
        // fail-closes the inline min-capped collect-count form; incremental push
        // growth carries no bulk-allocation obligation (same fix as
        // `exact::solve_system`'s solution vector).
        let mut g_terms: Vec<(String, Rat)> = Vec::new();
        for j in 0..m {
            g_terms.push((pvar(j), self.g.get(j).copied().unwrap_or(Rat::ZERO)));
        }
        let g_refs: Vec<(&str, Rat)> = g_terms.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let conclusion = LinearConstraint::with_kind(ConstraintKind::Le, &g_refs, bound);

        Ok(EntailmentCertificate {
            premises,
            multipliers: mult,
            conclusion,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check_entailment;

    fn r(n: i128, d: i128) -> Rat {
        Rat::new(n, d).unwrap()
    }

    /// The Section-1.2 worked example: n=2, weight box p∈[1/10,9/10] per
    /// position, values v_1∈[−1,1], v_2∈[9,11], readout w=1. For the UPPER bound
    /// (w>0) the exact inner-V corner (§3.2) picks the upper value endpoints, so
    /// g=(v̄_1, v̄_2)=(1, 11). Water-filling puts the budget on the larger g:
    /// p=(1/10, 9/10) ⇒ U = 1/10 + 99/10 = 10. This beats IBP's 10.8 (§1.2).
    #[test]
    fn section_1_2_example_certifies_and_self_checks() {
        let lp = SimplexSupportLp {
            g: vec![r(1, 1), r(11, 1)],
            p_lo: vec![r(1, 10), r(1, 10)],
            p_hi: vec![r(9, 10), r(9, 10)],
        };
        let cert = lp.certify_upper().unwrap();
        assert_eq!(cert.bound, r(10, 1)); // SBAR upper bound = 10.0
                                          // The IBP upper bound for the same data is 10.8 (doc §1.2); SBAR is
                                          // strictly tighter because it enforces Σp_j = 1.
        let ibp_upper = r(54, 5);
        assert!(cert.bound < ibp_upper);
        // All duals non-negative; the certificate passes Clean's check.
        for m in cert.mu_plus.iter().chain(&cert.mu_minus) {
            assert!(!m.is_negative());
        }
        let (derived, claimed) = check_entailment(&cert.entailment).unwrap();
        assert_eq!(derived, claimed); // tight: derived bound == claimed U
        assert_eq!(claimed, r(10, 1));
    }

    #[test]
    fn bound_dominates_all_feasible_weights() {
        // A 4-position LP; check the certified bound really is an upper bound
        // over many feasible weight vectors (vertices + interior points).
        let lp = SimplexSupportLp {
            g: vec![r(3, 1), r(-2, 1), r(5, 2), r(0, 1)],
            p_lo: vec![r(0, 1), r(1, 10), r(0, 1), r(1, 5)],
            p_hi: vec![r(7, 10), r(1, 2), r(3, 5), r(1, 1)],
        };
        let cert = lp.certify_upper().unwrap();
        check_entailment(&cert.entailment).unwrap();

        // Deterministically sample feasible p: start at p_lo, distribute the
        // budget across positions in every cyclic priority order.
        let m = lp.g.len();
        let sum_lo: Rat = {
            let mut a = Rat::ZERO;
            for v in &lp.p_lo {
                a = a.add(*v).unwrap();
            }
            a
        };
        let budget = Rat::ONE.sub(sum_lo).unwrap();
        for start in 0..m {
            let mut p = lp.p_lo.clone();
            let mut rem = budget;
            for off in 0..m {
                let j = (start + off) % m;
                let slack = lp.p_hi[j].sub(lp.p_lo[j]).unwrap();
                let take = if slack <= rem { slack } else { rem };
                p[j] = p[j].add(take).unwrap();
                rem = rem.sub(take).unwrap();
            }
            assert!(rem.is_zero(), "budget must fill");
            let obj = lp.objective(&p).unwrap();
            assert!(
                obj <= cert.bound,
                "feasible objective {obj:?} exceeds certified bound {:?}",
                cert.bound
            );
        }
    }

    #[test]
    fn degenerate_zero_budget_certifies() {
        // Σ p_lo = 1 already: the simplex collapses to the single point p_lo.
        let lp = SimplexSupportLp {
            g: vec![r(2, 1), r(-3, 1)],
            p_lo: vec![r(1, 2), r(1, 2)],
            p_hi: vec![r(1, 1), r(1, 1)],
        };
        let cert = lp.certify_upper().unwrap();
        // Only feasible point is (1/2,1/2): value = 1 − 3/2 = −1/2.
        assert_eq!(cert.bound, r(-1, 2));
        check_entailment(&cert.entailment).unwrap();
    }

    #[test]
    fn dual_satisfies_the_certificate_identities() {
        // Doc §5 test (iii): the closed-form dual must satisfy stationarity
        // λ·1 + μ⁺ − μ⁻ = g and value λ + p̄ᵀμ⁺ − p_loᵀμ⁻ = U, exactly.
        for (g, lo, hi) in &sample_lps() {
            let lp = SimplexSupportLp {
                g: g.clone(),
                p_lo: lo.clone(),
                p_hi: hi.clone(),
            };
            let c = lp.certify_upper().unwrap();
            // Stationarity, per position.
            for j in 0..g.len() {
                let recon = c
                    .lambda
                    .add(c.mu_plus[j])
                    .unwrap()
                    .sub(c.mu_minus[j])
                    .unwrap();
                assert_eq!(recon, g[j], "dual stationarity at {j}");
                assert!(!c.mu_plus[j].is_negative() && !c.mu_minus[j].is_negative());
            }
            // Dual value equals the primal bound.
            let mut dual = c.lambda;
            for j in 0..g.len() {
                dual = dual.add(c.mu_plus[j].mul(hi[j]).unwrap()).unwrap();
                dual = dual.sub(c.mu_minus[j].mul(lo[j]).unwrap()).unwrap();
            }
            assert_eq!(dual, c.bound, "dual value == primal optimum");
        }
    }

    #[allow(clippy::type_complexity)]
    fn sample_lps() -> Vec<(Vec<Rat>, Vec<Rat>, Vec<Rat>)> {
        vec![
            (
                vec![r(1, 1), r(11, 1)],
                vec![r(1, 10), r(1, 10)],
                vec![r(9, 10), r(9, 10)],
            ),
            (
                vec![r(3, 1), r(-2, 1), r(5, 2), r(0, 1)],
                vec![r(0, 1), r(1, 10), r(0, 1), r(1, 5)],
                vec![r(7, 10), r(1, 2), r(3, 5), r(1, 1)],
            ),
            (
                vec![r(-5, 2), r(-5, 2), r(7, 3)],
                vec![r(1, 4), r(1, 4), r(1, 4)],
                vec![r(1, 2), r(1, 2), r(1, 1)],
            ),
        ]
    }

    #[test]
    fn rejects_infeasible_box() {
        let lp = SimplexSupportLp {
            g: vec![r(1, 1), r(1, 1)],
            p_lo: vec![r(3, 5), r(3, 5)], // Σ = 6/5 > 1
            p_hi: vec![r(1, 1), r(1, 1)],
        };
        assert_eq!(lp.certify_upper().unwrap_err(), SbarError::Infeasible);
    }
}
