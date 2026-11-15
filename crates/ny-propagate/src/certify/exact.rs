// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact SPECIFICATION-emptiness certificates: an `unsat` whose proof needs no
//! network bound at all (ENGINE_PROGRAM §1.6 U4-U6, routed through the same
//! token door as every covered lane).
//!
//! # The certificate
//!
//! Some properties are `unsat` because their unsafe region is empty AS A
//! FORMULA, whatever the networks compute. The isomorphic ε-equivalence
//! complement is the one in the tree today: over the single scalar term
//! `t = Y_g[i] - Y_f[i]` the unsafe conjunction carries `t > c₁ ∧ t < c₂`, which
//! no `t` satisfies once `c₁ ≥ c₂`. The certificate is a PAIR of atoms — one
//! lower (`>`/`≥`), one upper (`<`/`≤`) — whose Farkas combination with
//! multipliers `(1, 1)` is the contradiction `0 ⋈ c₂ - c₁` with `c₂ - c₁ ≤ 0`
//! (strict somewhere when `c₁ = c₂`). Checking it needs NO arithmetic: two exact
//! binary64 comparisons and one equality, so it is platform independent and has
//! no rounding to charge.
//!
//! This is the same fact the lane's own `ny_cert::check_farkas` replay proves
//! over exact rationals (every binary64 converts exactly), re-derived here so
//! the TOKEN rests on a check the certify facade performed itself. The lane
//! keeps its replay; `unsat` is published only when both agree.
//!
//! # Who may mint, and what the caller is trusted for
//!
//! [`scalar_emptiness`] builds an [`ExactEmptiness`] only after it has found
//! and checked a contradicting pair itself, and it is `pub(crate)` like every
//! other minting adapter (the crate's section-1.7 ratchet pins that). The one
//! public entry, [`authorize_scalar_emptiness`], passes EVERY group through it:
//! nothing a caller supplies is taken as proof, so no caller can launder a
//! number into a token.
//!
//! What a caller IS trusted for is that the atoms ARE the property — the parsed
//! VNN-LIB (U1 of §1.6), exactly the trust the margin-row lane places in its
//! objective rows — and that every atom of one group constrains the SAME term
//! `t`. A caller that passes atoms which are not the property obtains a token
//! for a different property; it can never obtain one for a region its atoms do
//! not make empty.

use std::fmt;

use super::token::{
    authorize_unsat, AuthorizeRefusal, ClassCover, GuardReport, TokenScope, UnsatToken,
};
use crate::margin_row::rounding::next_down;

/// The class id of a property whose unsafe region is ONE conjunction: the
/// single disjunct every [`ExactEmptiness`] of [`authorize_scalar_emptiness`]
/// empties.
pub const UNSAFE_CONJUNCTION: usize = 0;

/// The relation of one scalar atom `t ⋈ constant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarRel {
    /// `t < c`
    Lt,
    /// `t <= c`
    Le,
    /// `t > c`
    Gt,
    /// `t >= c`
    Ge,
}

impl ScalarRel {
    /// A LOWER bound on `t` (`>` or `>=`); otherwise an upper bound.
    #[must_use]
    pub const fn is_lower(self) -> bool {
        matches!(self, Self::Gt | Self::Ge)
    }

    /// A strict relation (`>` or `<`).
    #[must_use]
    pub const fn is_strict(self) -> bool {
        matches!(self, Self::Gt | Self::Lt)
    }
}

/// One atom `t ⋈ constant` over the scalar term `t` its group shares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarAtom {
    /// The relation.
    pub rel: ScalarRel,
    /// The right-hand side, exactly as parsed (signed, never normalised).
    pub constant: f64,
}

/// A CHECKED emptiness certificate: a lower atom and an upper atom of one
/// group that no value of `t` satisfies together. Private fields, and the only
/// constructor is [`scalar_emptiness`], so holding one means the check ran.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactEmptiness {
    lower: ScalarAtom,
    upper: ScalarAtom,
    slack: f64,
}

impl ExactEmptiness {
    /// The lower atom (`t > c` or `t >= c`) of the contradicting pair.
    #[must_use]
    pub fn lower(&self) -> ScalarAtom {
        self.lower
    }

    /// The upper atom (`t < c` or `t <= c`) of the contradicting pair.
    #[must_use]
    pub fn upper(&self) -> ScalarAtom {
        self.upper
    }

    /// `lower.constant - upper.constant`, rounded DOWN and never negative: the
    /// G5 margin the proof rests on. Zero only for a strict tie (`t > c ∧
    /// t <= c`), which is still empty.
    #[must_use]
    pub fn slack(&self) -> f64 {
        self.slack
    }
}

/// Whether `lower` (a `>`/`>=` atom) and `upper` (a `<`/`<=` atom) can hold
/// together for no `t`. Exact: comparisons only. A non-finite constant never
/// contradicts (`NaN` compares false; an infinite one is refused upstream).
fn contradicts(lower: ScalarAtom, upper: ScalarAtom) -> bool {
    lower.rel.is_lower()
        && !upper.rel.is_lower()
        && lower.constant.is_finite()
        && upper.constant.is_finite()
        && (lower.constant > upper.constant
            || (lower.constant == upper.constant
                && (lower.rel.is_strict() || upper.rel.is_strict())))
}

/// The pair's G5 slack: `lower - upper` rounded down, clamped at zero, and
/// `f64::MAX` if the subtraction overflows. Observational only — the check is
/// [`contradicts`], never this value.
fn pair_slack(lower: f64, upper: f64) -> f64 {
    let d = lower - upper;
    if !d.is_finite() {
        f64::MAX
    } else if d > 0.0 {
        next_down(d).max(0.0)
    } else {
        0.0
    }
}

/// MINT: the contradicting pair of largest slack among `atoms`, or `None` when
/// no lower atom contradicts an upper atom (the group is feasible, or it holds
/// a non-finite constant). `pub(crate)` — see the module docs.
#[must_use]
pub(crate) fn scalar_emptiness(atoms: &[ScalarAtom]) -> Option<ExactEmptiness> {
    if atoms.iter().any(|a| !a.constant.is_finite()) {
        return None;
    }
    let mut best: Option<ExactEmptiness> = None;
    for &lower in atoms.iter().filter(|a| a.rel.is_lower()) {
        for &upper in atoms.iter().filter(|a| !a.rel.is_lower()) {
            if !contradicts(lower, upper) {
                continue;
            }
            let slack = pair_slack(lower.constant, upper.constant);
            if !matches!(&best, Some(kept) if kept.slack >= slack) {
                best = Some(ExactEmptiness {
                    lower,
                    upper,
                    slack,
                });
            }
        }
    }
    best
}

/// Why [`authorize_scalar_emptiness`] refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactRefusal {
    /// No group was given: a vacuous proof is not a proof.
    NoGroups,
    /// A group has no atom.
    EmptyGroup {
        /// Group ordinal.
        group: usize,
    },
    /// A group carries a non-finite constant.
    NonFinite {
        /// Group ordinal.
        group: usize,
    },
    /// No pair of the group's atoms contradicts: the group is satisfiable.
    Feasible {
        /// Group ordinal.
        group: usize,
    },
    /// The aggregation refused (should not happen for a checked cover).
    Authorize(AuthorizeRefusal),
}

impl fmt::Display for ExactRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoGroups => f.write_str("no_groups"),
            Self::EmptyGroup { group } => write!(f, "empty_group group={group}"),
            Self::NonFinite { group } => write!(f, "non_finite group={group}"),
            Self::Feasible { group } => write!(f, "feasible group={group}"),
            Self::Authorize(refusal) => write!(f, "authorize {refusal}"),
        }
    }
}

/// The VERIFYING entry for a property whose unsafe region is ONE conjunction
/// and which the caller claims is empty by `groups` of scalar atoms (each group
/// constrains one term `t`; for the isomorphic complement, one output index).
///
/// EVERY group must be non-empty and proved empty by [`scalar_emptiness`] —
/// one would suffice for emptiness; requiring all keeps the certificate set
/// complete, exactly as the lane's own replay does. The covers are aggregated
/// by [`authorize_unsat`] at instance scope for [`UNSAFE_CONJUNCTION`], with
/// no guard to run (no network is involved, so G2/G3 have nothing to sample).
pub fn authorize_scalar_emptiness(groups: &[Vec<ScalarAtom>]) -> Result<UnsatToken, ExactRefusal> {
    if groups.is_empty() {
        return Err(ExactRefusal::NoGroups);
    }
    let mut covers = Vec::with_capacity(groups.len());
    for (group, atoms) in groups.iter().enumerate() {
        if atoms.is_empty() {
            return Err(ExactRefusal::EmptyGroup { group });
        }
        if atoms.iter().any(|a| !a.constant.is_finite()) {
            return Err(ExactRefusal::NonFinite { group });
        }
        let proof = scalar_emptiness(atoms).ok_or(ExactRefusal::Feasible { group })?;
        covers.push(ClassCover::Exact {
            class: UNSAFE_CONJUNCTION,
            proof,
        });
    }
    authorize_unsat(
        TokenScope::Instance,
        &[UNSAFE_CONJUNCTION],
        &covers,
        &GuardReport::new(),
    )
    .map_err(ExactRefusal::Authorize)
}
