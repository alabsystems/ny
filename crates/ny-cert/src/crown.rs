// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact-rational CROWN certificate generation for one ReLU hidden layer.
//!
//! This is the constructive core of *Proof-Carrying Verification*: it shows that
//! a CROWN backward pass over a ReLU network is literally a Farkas / dual
//! multiplier derivation, and emits that derivation as a certificate Clean's
//! kernel-side verifier can check.
//!
//! ## The network
//!
//! ```text
//!   z = W₁·x + b₁        (affine pre-activations)
//!   a = ReLU(z)          (element-wise)
//!   y = W₂·a + b₂        (scalar affine output)
//! ```
//! over an input box `x ∈ [l, u]`. We verify a safety property `y ≥ t`.
//!
//! ## The certificate
//!
//! Every relaxation fact is a linear inequality over the variables
//! `xᵢ, zⱼ, aⱼ, y`:
//!
//! * box bounds `lᵢ ≤ xᵢ ≤ uᵢ`,
//! * the affine layers (each equality supplied as a `≤`/`≥` pair so the
//!   verifier can assign it a signed effective weight using two non-negative
//!   multipliers — Clean's entailment verifier scales an `eq` premise's two
//!   halves by a *single* multiplier, which cancels, so equalities must be
//!   split),
//! * the ReLU envelopes: a lower envelope `aⱼ ≥ pⱼ·zⱼ + qⱼ` and an upper
//!   envelope `aⱼ ≤ rⱼ·zⱼ + tⱼ`.
//!
//! The CROWN backward pass chooses, for each unit, the lower envelope when its
//! output weight is positive and the upper envelope when negative, then
//! back-substitutes through the affine layers and finally the box. Each
//! substitution *is* the choice of a non-negative multiplier on the
//! corresponding inequality. The accumulated combination is exactly
//! `−y ≤ −m`, where `m` is the CROWN lower bound on `y`.

use crate::rational::{poisoned, Rat, RatError};
use crate::schema::{ConstraintKind, EntailmentCertificate, FarkasCertificate, LinearConstraint};
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

/// Errors that can arise while building a certificate.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CrownError {
    /// A matrix/vector dimension did not line up.
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    /// An α slope was outside `[0, 1]` (would make the lower envelope unsound).
    #[error("alpha[{0}] must lie in [0, 1]")]
    AlphaOutOfRange(usize),
    /// The requested safety threshold exceeds the certified lower bound.
    #[error("threshold {threshold} exceeds certified lower bound {bound}")]
    ThresholdAboveBound {
        /// Requested threshold.
        threshold: String,
        /// Best certified lower bound.
        bound: String,
    },
    /// Exact arithmetic failure.
    #[error(transparent)]
    Rat(#[from] RatError),
}

/// A one-hidden-layer ReLU network plus an input box.
#[derive(Debug, Clone)]
pub struct Relu1Problem {
    /// Hidden weight matrix `W₁`, shape `[hidden][input]`.
    pub w1: Vec<Vec<Rat>>,
    /// Hidden bias `b₁`, length `hidden`.
    pub b1: Vec<Rat>,
    /// Output weight `W₂`, length `hidden` (scalar output).
    pub w2: Vec<Rat>,
    /// Output bias `b₂`.
    pub b2: Rat,
    /// Input lower bounds, length `input`.
    pub input_lower: Vec<Rat>,
    /// Input upper bounds, length `input`.
    pub input_upper: Vec<Rat>,
    /// Optional CROWN lower-envelope slopes `αⱼ ∈ [0, 1]`. When `None`, the
    /// adaptive default (`1` if `uⱼ ≥ −lⱼ`, else `0`) is used per unit.
    pub alpha: Option<Vec<Rat>>,
}

/// A certified verification result for [`Relu1Problem`].
#[derive(Debug, Clone)]
pub struct CertifiedRelu1 {
    /// Entailment certificate proving `y ≥ threshold`.
    pub entailment: EntailmentCertificate,
    /// Farkas certificate proving the unsafe region `y < threshold` is empty.
    pub farkas: FarkasCertificate,
    /// The CROWN lower bound `m` on the output (`y ≥ m`).
    pub lower_bound: Rat,
    /// Pre-activation lower bounds `lⱼ`.
    pub preact_lower: Vec<Rat>,
    /// Pre-activation upper bounds `uⱼ`.
    pub preact_upper: Vec<Rat>,
}

fn dot(weights: &[Rat], lo: &[Rat], hi: &[Rat], want_min: bool) -> Result<Rat, RatError> {
    let mut acc = Rat::ZERO;
    for ((w, l), u) in weights.iter().zip(lo).zip(hi) {
        // For a minimum, pick l where w>0 and u where w<0; flip for a maximum.
        let pick = if (w.is_positive() && want_min) || (w.is_negative() && !want_min) {
            *l
        } else {
            *u
        };
        acc = acc.add(w.mul(pick)?)?;
    }
    Ok(acc)
}

impl Relu1Problem {
    fn input_dim(&self) -> usize {
        self.input_lower.len()
    }

    fn hidden_dim(&self) -> usize {
        self.w1.len()
    }

    pub(crate) fn validate(&self) -> Result<(), CrownError> {
        let n = self.input_dim();
        let h = self.hidden_dim();
        if self.input_upper.len() != n {
            return Err(CrownError::Dimension("input bounds length differ".into()));
        }
        if self.b1.len() != h || self.w2.len() != h {
            return Err(CrownError::Dimension("hidden dimension mismatch".into()));
        }
        for (j, row) in self.w1.iter().enumerate() {
            if row.len() != n {
                return Err(CrownError::Dimension(format!(
                    "W1 row {j} width != input dim"
                )));
            }
        }
        Ok(())
    }

    /// Interval-bound-propagation pre-activation bounds `lⱼ, uⱼ` for each unit.
    ///
    /// # Errors
    /// Propagates exact-rational arena failures.
    pub fn preact_bounds(&self) -> Result<(Vec<Rat>, Vec<Rat>), CrownError> {
        crate::rational::ensure_healthy()?;
        self.validate()?;
        // `Vec::new()` (not `with_capacity(hidden_dim())`): the capacity hint
        // on a havoc-unbounded `w1.len()` carries a hardened allocation
        // obligation the model cannot bound (the cite_check/grid_min
        // precedent); amortized growth costs nothing at real hidden dims.
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for (row, b) in self.w1.iter().zip(&self.b1) {
            let zl = dot(row, &self.input_lower, &self.input_upper, true)?.add(*b)?;
            let zu = dot(row, &self.input_lower, &self.input_upper, false)?.add(*b)?;
            lo.push(zl);
            hi.push(zu);
        }
        crate::rational::ensure_healthy()?;
        Ok((lo, hi))
    }

    fn alpha_slopes(&self, lo: &[Rat], hi: &[Rat]) -> Result<Vec<Rat>, CrownError> {
        if let Some(alpha) = &self.alpha {
            if alpha.len() != self.hidden_dim() {
                return Err(CrownError::Dimension("alpha length mismatch".into()));
            }
            for (j, a) in alpha.iter().enumerate() {
                if a.is_negative() || *a > Rat::ONE {
                    return Err(CrownError::AlphaOutOfRange(j));
                }
            }
            return Ok(alpha.clone());
        }
        // Adaptive default: 1 if u ≥ −l, else 0.
        // Explicit Vec::new()+push (not `.collect()`): the length is `lo.len()`,
        // an input-derived count the intraprocedural verifier cannot bound, so a
        // bulk `.collect()` raises an UnboundedAllocation obligation. The loop
        // has no bulk-alloc obligation at all — identical elements and order.
        let mut slopes: Vec<Rat> = Vec::new();
        for (l, u) in lo.iter().zip(hi) {
            slopes.push(if *u >= l.neg() { Rat::ONE } else { Rat::ZERO });
        }
        Ok(slopes)
    }

    /// Build the proof-carrying certificate for the property `y ≥ threshold`.
    ///
    /// # Errors
    /// Returns [`CrownError`] on dimension mismatch, an out-of-range α, an
    /// infeasible threshold, or an exact-arithmetic/arena failure.
    ///
    /// The `#[ensures]` states the locally-provable producer well-formedness
    /// invariant: on `Ok` the emitted entailment certificate is a valid Farkas
    /// combination shape — exactly one non-negative multiplier per premise
    /// (`premises.len() == multipliers.len()`), the structural precondition the
    /// checker enforces (`CheckError::LengthMismatch` otherwise). It is stated
    /// over the result alone because the builtin `#[ensures]` closure must be
    /// `Copy + 'static` and so cannot capture `threshold`/`&self` (see
    /// `core::contracts::build_check_ensures`). `#[trust::cite]` grounds the
    /// deeper completeness claim — that this combination genuinely proves
    /// `y ≥ threshold` for the relaxed network — in the kernel-checked
    /// `crown_bridge` theorem.
    #[ensures(|r: &Result<CertifiedRelu1, CrownError>| !matches!(r, Ok(c) if c.entailment.premises.len() != c.entailment.multipliers.len()))]
    #[trust::cite(crownproof::crown_bridge)]
    // `?` here would desugar to `from_residual` return paths the verifier's
    // len-witness grounding cannot aggregate over — the explicit match/if-let
    // returns ARE the proof shape (see the extract-then-guard comment).
    #[allow(clippy::question_mark)]
    pub fn certify(&self, threshold: Rat) -> Result<CertifiedRelu1, CrownError> {
        if poisoned() {
            return Err(crate::err_barrier(CrownError::Rat(RatError::Poisoned)));
        }
        // Extract-then-guard: makes the `#[ensures]` locally provable. The
        // match only EXTRACTS (the Err arm returns early), the arity guard is
        // straight-line, and the tail is a plain `Ok(c)` — so every return
        // path constructs its `Ok`/`Err` in the direct predecessor of the
        // return block and the guard's equality edge dominates the `Ok`
        // (the verifier's len-witness grounding window; a guard INSIDE the
        // match arm splits the arm join from the return block and the
        // construction falls outside it). The guard is unreachable by
        // construction — `certify_inner` upholds the same invariant — so this
        // is behavior-identical, fail-closed hardening.
        let c = match self.certify_inner(threshold) {
            Ok(c) => c,
            // `crate::err_barrier` (identity, `#[inline(never)]`): a fresh in-body
            // `Err` aggregate, not a whole-`Result` forward the return-grounding
            // lane cannot see (nor a const-promoted+merged unit variant).
            Err(e) => {
                if poisoned() {
                    return Err(crate::err_barrier(CrownError::Rat(RatError::Poisoned)));
                }
                return Err(crate::err_barrier(e));
            }
        };
        if poisoned() {
            return Err(crate::err_barrier(CrownError::Rat(RatError::Poisoned)));
        }
        if c.entailment.premises.len() != c.entailment.multipliers.len() {
            return Err(crate::err_barrier(CrownError::Dimension(
                "certificate premise/multiplier arity mismatch".into(),
            )));
        }
        Ok(c)
    }

    /// The full certificate construction behind [`Self::certify`]. Private and
    /// contract-free: the ensures-bearing `certify` wrapper re-establishes the
    /// premise/multiplier arity invariant with an in-body guard (this body's
    /// pervasive `?` returns are `from_residual` paths the local proof cannot
    /// aggregate over).
    #[allow(clippy::too_many_lines)]
    fn certify_inner(&self, threshold: Rat) -> Result<CertifiedRelu1, CrownError> {
        self.validate()?;
        let n = self.input_dim();
        let h = self.hidden_dim();
        let (lo, hi) = self.preact_bounds()?;
        let alpha = self.alpha_slopes(&lo, &hi)?;

        // Free (nested) `fn`s, not `|i| ..` closures: called directly rather than
        // through an absent `<{closure} as Fn>::call` shim, so the var-name
        // builders stay in verified code. They capture nothing; `format!` over a
        // `usize` proves panic-free.
        fn xv(i: usize) -> String {
            format!("x{i}")
        }
        fn zv(j: usize) -> String {
            format!("z{j}")
        }
        fn av(j: usize) -> String {
            format!("a{j}")
        }

        // Total nested lookup into W₁ (row j, col i): `validate` pinned
        // `w1.len() == h` and every row width to `n`, so for `j < h`, `i < n`
        // the `.get`s are always `Some` — the `Rat::ZERO` fallback is
        // unreachable. Keeps the two W₁ reads free of slice-bounds obligations
        // without changing any value (the length facts live in `validate`, an
        // interprocedural fact the intraprocedural verifier cannot see).
        // Free (nested) `fn`, not a `|j, i|` closure: called directly rather
        // than through an absent `<{closure} as Fn>::call` shim (the captured
        // `self.w1` becomes the explicit `w1` parameter), and the row lookup is
        // a plain `match` rather than a closure passed to `Option::and_then` —
        // identical value on every input.
        fn w1_at(w1: &[Vec<Rat>], j: usize, i: usize) -> Rat {
            match w1.get(j) {
                Some(row) => row.get(i).copied().unwrap_or(Rat::ZERO),
                None => Rat::ZERO,
            }
        }
        // Total read-modify-add on an accumulator `Vec<Rat>`: every `idx` is
        // either a loop index `< the Vec's constructed length` (`h` or `n`) or a
        // premise index returned by `push` (`< premises.len() == mult.len()`), so
        // both the `.get`/`.get_mut` fallbacks are unreachable. Replaces
        // `v[idx] = v[idx].add(delta)?` — no `[]` panic / slice-bounds obligation,
        // identical result for every valid input.
        // Free (nested) `fn`, not a closure: called directly rather than through
        // an absent `<{closure} as Fn>::call` shim, so the accumulator update
        // stays in verified code. Captures nothing (only its params + `Rat::ZERO`).
        fn add_mult(v: &mut [Rat], idx: usize, delta: Rat) -> Result<(), RatError> {
            let cur = v.get(idx).copied().unwrap_or(Rat::ZERO);
            let next = cur.add(delta)?;
            if let Some(slot) = v.get_mut(idx) {
                *slot = next;
            }
            Ok(())
        }

        // --- Assemble the full relaxed-network constraint system (premises) ---
        // We record each premise's index so the backward pass can attach a
        // multiplier to it. Indices follow construction order.
        let mut premises: Vec<LinearConstraint> = Vec::new();
        let mut mult: Vec<Rat> = Vec::new();
        // Free (nested) `fn`, not a closure: called directly (no absent
        // `<{closure} as Fn>::call` shim); it captures nothing — the premise and
        // multiplier vectors are explicit `&mut` parameters.
        fn push(
            c: LinearConstraint,
            premises: &mut Vec<LinearConstraint>,
            mult: &mut Vec<Rat>,
        ) -> usize {
            // Index of the element we're about to push (pre-push len), avoiding a
            // `len() - 1` whose usize-underflow the verifier can't discharge locally.
            let idx = premises.len();
            premises.push(c);
            mult.push(Rat::ZERO);
            idx
        }

        // Box: xᵢ ≤ uᵢ and xᵢ ≥ lᵢ.
        // `Vec::new()` (not `with_capacity(n)`): the capacity hint on the
        // havoc-unbounded input dimension carries a hardened allocation
        // obligation the model cannot bound (the `preact_bounds` precedent);
        // push growth is behavior-identical.
        let mut box_u = Vec::new();
        let mut box_l = Vec::new();
        // `i < n`; `input_upper`/`input_lower` are length `n` (`validate`), so the
        // `Rat::ZERO` fallbacks are unreachable — total reads, no value change.
        for i in 0..n {
            box_u.push(push(
                LinearConstraint::with_kind(
                    ConstraintKind::Le,
                    &[(&xv(i), Rat::ONE)],
                    self.input_upper.get(i).copied().unwrap_or(Rat::ZERO),
                ),
                &mut premises,
                &mut mult,
            ));
            box_l.push(push(
                LinearConstraint::with_kind(
                    ConstraintKind::Ge,
                    &[(&xv(i), Rat::ONE)],
                    self.input_lower.get(i).copied().unwrap_or(Rat::ZERO),
                ),
                &mut premises,
                &mut mult,
            ));
        }

        // Affine pre-activations zⱼ = Σ W₁[j][i]·xᵢ + b₁[j], split into ≤ / ≥.
        // `Vec::new()` (not `with_capacity(h)`): same havoc-unbounded capacity
        // obligation as `box_u`/`box_l`; push growth is behavior-identical.
        let mut z_le = Vec::new();
        let mut z_ge = Vec::new();
        // `j < h`, `i < n`; `w1_at` totalizes the nested W₁ read and `b1` is
        // length `h` (`validate`), so the `Rat::ZERO` fallbacks are unreachable.
        for j in 0..h {
            let mut terms: Vec<(String, Rat)> = vec![(zv(j), Rat::ONE)];
            for i in 0..n {
                terms.push((xv(i), w1_at(&self.w1, j, i).neg()));
            }
            // Explicit loop (not `.map(..).collect()`): the `|(k, v)| ..` closure
            // would be invoked through an absent `<{closure} as Fn>::call` shim;
            // identical elements and order, and the push growth carries no
            // bulk-alloc obligation.
            let mut refs: Vec<(&str, Rat)> = Vec::new();
            for (k, v) in &terms {
                refs.push((k.as_str(), *v));
            }
            let bj = self.b1.get(j).copied().unwrap_or(Rat::ZERO);
            z_le.push(push(
                LinearConstraint::with_kind(ConstraintKind::Le, &refs, bj),
                &mut premises,
                &mut mult,
            ));
            z_ge.push(push(
                LinearConstraint::with_kind(ConstraintKind::Ge, &refs, bj),
                &mut premises,
                &mut mult,
            ));
        }

        // ReLU envelopes. Lower: aⱼ ≥ pⱼ·zⱼ + qⱼ. Upper: aⱼ ≤ rⱼ·zⱼ + tⱼ.
        // Choose per-unit shape from the pre-activation interval.
        // `Vec::new()` (not `with_capacity(h)`): same havoc-unbounded capacity
        // obligation as `box_u`/`box_l`; push growth is behavior-identical.
        let mut env_lower = Vec::new(); // (p, q, premise_idx)
        let mut env_upper = Vec::new(); // (r, t, premise_idx)
                                        // `j < h`; `lo`/`hi` have `h` entries (one per `w1` row) and `alpha` was
                                        // built/validated to length `h`, so the `Rat::ZERO` fallbacks are
                                        // unreachable — total reads, identical envelope selection.
        for j in 0..h {
            let l = lo.get(j).copied().unwrap_or(Rat::ZERO);
            let u = hi.get(j).copied().unwrap_or(Rat::ZERO);
            let (p, q, r, t) = if !l.is_negative() {
                // Always active: aⱼ = zⱼ.
                (Rat::ONE, Rat::ZERO, Rat::ONE, Rat::ZERO)
            } else if !u.is_positive() {
                // Always inactive: aⱼ = 0.
                (Rat::ZERO, Rat::ZERO, Rat::ZERO, Rat::ZERO)
            } else {
                // Unstable. Lower: aⱼ ≥ αⱼ·zⱼ. Upper: aⱼ ≤ s·(zⱼ − l), s = u/(u−l).
                let s = u.mul(u.sub(l)?.inv()?)?;
                (
                    alpha.get(j).copied().unwrap_or(Rat::ZERO),
                    Rat::ZERO,
                    s,
                    s.mul(l.neg())?,
                )
            };
            // Lower envelope aⱼ − p·zⱼ ≥ q   (kind Ge).
            let le_idx = push(
                LinearConstraint::with_kind(
                    ConstraintKind::Ge,
                    &[(&av(j), Rat::ONE), (&zv(j), p.neg())],
                    q,
                ),
                &mut premises,
                &mut mult,
            );
            // Upper envelope aⱼ − r·zⱼ ≤ t   (kind Le).
            let ue_idx = push(
                LinearConstraint::with_kind(
                    ConstraintKind::Le,
                    &[(&av(j), Rat::ONE), (&zv(j), r.neg())],
                    t,
                ),
                &mut premises,
                &mut mult,
            );
            env_lower.push((p, q, le_idx));
            env_upper.push((r, t, ue_idx));
        }

        // Output y = Σ W₂[j]·aⱼ + b₂, split into ≤ / ≥.
        let mut y_terms: Vec<(String, Rat)> = vec![("y".to_string(), Rat::ONE)];
        // `j < h`; `w2` is length `h` (`validate`), fallback unreachable.
        for j in 0..h {
            y_terms.push((av(j), self.w2.get(j).copied().unwrap_or(Rat::ZERO).neg()));
        }
        // Explicit loop (not `.map(..).collect()`): same absent `Fn::call` shim
        // rationale as `refs` above — identical elements and order.
        let mut y_refs: Vec<(&str, Rat)> = Vec::new();
        for (k, v) in &y_terms {
            y_refs.push((k.as_str(), *v));
        }
        let _y_le = push(
            LinearConstraint::with_kind(ConstraintKind::Le, &y_refs, self.b2),
            &mut premises,
            &mut mult,
        );
        let y_ge = push(
            LinearConstraint::with_kind(ConstraintKind::Ge, &y_refs, self.b2),
            &mut premises,
            &mut mult,
        );

        // --- CROWN backward pass = choosing the non-negative multipliers ---
        // Start: −y + Σ W₂[j]·aⱼ ≤ −b₂  (the normalized form of y_ge).
        add_mult(&mut mult, y_ge, Rat::ONE)?;
        let mut const_acc = self.b2.neg();

        // Eliminate each aⱼ with its sign-appropriate envelope; collect zⱼ coeff.
        // `j < h`; `w2` is length `h`, `env_lower`/`env_upper`/`z_coeff` each have
        // `h` entries, so the `.get()` guards below always match — the skip arms
        // are unreachable and the accumulation is identical.
        // `Vec::new()` + push (not `vec![Rat::ZERO; h]`): the `h`-count bulk
        // fill carries a hardened allocation obligation the model cannot bound
        // (the eval.rs grid precedent); the loop builds the identical all-zero
        // accumulator.
        let mut z_coeff: Vec<Rat> = Vec::new();
        for _ in 0..h {
            z_coeff.push(Rat::ZERO);
        }
        for j in 0..h {
            let w = self.w2.get(j).copied().unwrap_or(Rat::ZERO);
            if w.is_positive() {
                if let Some(&(p, q, idx)) = env_lower.get(j) {
                    add_mult(&mut mult, idx, w)?; // scale lower envelope by W₂[j]
                    add_mult(&mut z_coeff, j, w.mul(p)?)?;
                    const_acc = const_acc.add(w.mul(q.neg())?)?; // −w·q
                }
            } else if w.is_negative() {
                if let Some(&(r, t, idx)) = env_upper.get(j) {
                    let mag = w.neg();
                    add_mult(&mut mult, idx, mag)?; // scale upper envelope by |W₂[j]|
                    add_mult(&mut z_coeff, j, w.mul(r)?)?; // w·r (w<0)
                    const_acc = const_acc.add(mag.mul(t)?)?; // +|w|·t
                }
            }
        }

        // Eliminate each zⱼ through the affine layer; collect xᵢ coeff.
        // `j < h`, `i < n`; `z_coeff`/`z_ge`/`z_le`/`b1` are length `h`, `x_coeff`
        // length `n`, and `w1_at` totalizes the nested W₁ read — every `.get()`
        // guard matches, so the skip arms are unreachable and the result is
        // identical.
        // `Vec::new()` + push (not `vec![Rat::ZERO; n]`): same unbounded
        // bulk-fill obligation as `z_coeff` above; identical all-zero vec.
        let mut x_coeff: Vec<Rat> = Vec::new();
        for _ in 0..n {
            x_coeff.push(Rat::ZERO);
        }
        for j in 0..h {
            let c = z_coeff.get(j).copied().unwrap_or(Rat::ZERO);
            if c.is_zero() {
                continue;
            }
            if c.is_positive() {
                if let Some(&pidx) = z_ge.get(j) {
                    add_mult(&mut mult, pidx, c)?;
                }
            } else if let Some(&pidx) = z_le.get(j) {
                add_mult(&mut mult, pidx, c.neg())?;
            }
            for i in 0..n {
                add_mult(&mut x_coeff, i, c.mul(w1_at(&self.w1, j, i))?)?;
            }
            const_acc =
                const_acc.add(c.mul(self.b1.get(j).copied().unwrap_or(Rat::ZERO))?.neg())?;
            // −c·b₁
        }

        // Eliminate each xᵢ through the box.
        // `i < n`; `x_coeff`/`box_l`/`box_u`/`input_lower`/`input_upper` are all
        // length `n`, so every `.get()` guard matches — skip arms unreachable,
        // identical result.
        for i in 0..n {
            let d = x_coeff.get(i).copied().unwrap_or(Rat::ZERO);
            if d.is_zero() {
                continue;
            }
            if d.is_positive() {
                if let Some(&pidx) = box_l.get(i) {
                    add_mult(&mut mult, pidx, d)?;
                }
                let li = self.input_lower.get(i).copied().unwrap_or(Rat::ZERO);
                const_acc = const_acc.add(d.mul(li.neg())?)?; // −d·l
            } else {
                let mag = d.neg();
                if let Some(&pidx) = box_u.get(i) {
                    add_mult(&mut mult, pidx, mag)?;
                }
                let ui = self.input_upper.get(i).copied().unwrap_or(Rat::ZERO);
                const_acc = const_acc.add(mag.mul(ui)?)?; // |d|·u
            }
        }

        // Now the combination is exactly  −y ≤ const_acc, i.e. y ≥ −const_acc.
        let lower_bound = const_acc.neg();
        if threshold > lower_bound {
            return Err(CrownError::ThresholdAboveBound {
                threshold: format!("{}/{}", threshold.num(), threshold.den()),
                bound: format!("{}/{}", lower_bound.num(), lower_bound.den()),
            });
        }

        let entailment = EntailmentCertificate {
            premises: premises.clone(),
            multipliers: mult.clone(),
            conclusion: LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[("y", Rat::ONE)],
                threshold,
            ),
        };

        // Farkas: append the negated property y < threshold (strict) with
        // multiplier 1; the combination collapses to 0 < (threshold − m) ≤ 0.
        let mut f_constraints = premises;
        let mut f_mult = mult;
        f_constraints.push(LinearConstraint::with_kind(
            ConstraintKind::Lt,
            &[("y", Rat::ONE)],
            threshold,
        ));
        f_mult.push(Rat::ONE);
        let farkas = FarkasCertificate {
            constraints: f_constraints,
            multipliers: f_mult,
        };

        Ok(CertifiedRelu1 {
            entailment,
            farkas,
            lower_bound,
            preact_lower: lo,
            preact_upper: hi,
        })
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
    fn certify_refuses_poison_before_a_degenerate_identity_only_path() {
        // With no inputs or hidden units and zero bias, the legacy path can
        // assemble a structural certificate using only ZERO/ONE handles.
        // Poison must therefore be checked explicitly rather than relying on
        // a nontrivial arithmetic operation to notice it.
        let problem = Relu1Problem {
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Rat::ZERO,
            input_lower: Vec::new(),
            input_upper: Vec::new(),
            alpha: None,
        };
        crate::rational::set_poisoned_for_test(true);
        let _reset = PoisonReset;

        assert!(matches!(
            problem.preact_bounds(),
            Err(CrownError::Rat(RatError::Poisoned))
        ));
        assert!(matches!(
            problem.certify(Rat::ZERO),
            Err(CrownError::Rat(RatError::Poisoned))
        ));
    }
}
