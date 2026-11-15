// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Margin-row adapters: wrap values the lane's EXISTING certified evaluators
//! produced into `certify` types. No new arithmetic lives here — every value
//! passed in was computed by the Outward engine, the certified emptiness tests,
//! or the frozen-dual search, exactly as before.
//!
//! The adapters read the lane's own IR (`TrunkSplit`, `HeadFix`, the round
//! mode, the frozen-dual `ClassReport`), never a search representation.
//!
//! # Who may mint
//!
//! Every adapter that turns a raw `f64` (or a public `ClassReport`) into a
//! [`Certified`], a [`LeafEvidence`] or a [`ClassCover`] TRUSTS its caller to
//! pass a value the certified evaluator produced — it cannot re-derive it. So
//! those adapters are `pub(crate)`: only this crate's certified lanes (today
//! `margin_row/bab.rs`) may call them, and no other crate — in particular
//! `ny-cli`, the publication path — can build a `Certified`, a class cover or
//! an authorized `UnsatToken` from a number it chose (adversarial review of
//! #soundness-spine: with these `pub`, `authorize_unsat(Instance, …,
//! root_class(Outward, c, 1.0))` minted a publishable token from any float).
//! The crate's section-1.7 ratchet integration test pins the
//! visibility, and the compile-fail examples in `certify`'s docs pin it from
//! outside the crate. [`closes`] and [`leaf_path`] mint nothing and stay `pub`.

use super::cover::{CoverLiteral, CoverSite, LeafEvidence, LeafPath};
use super::token::ClassCover;
use super::value::{closed, CertReceipt, Certified, SearchValue};
use crate::margin_row::bab::{HeadFix, TrunkSplit};
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::zono_bab::{ledger_closes, ClassReport, SearchVerdict};

/// A margin-row bound, split by how it was computed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarginRowValue {
    /// Certified-outward engine output.
    Certified(Certified<f64>),
    /// Round-to-nearest parity output: search only, no closure predicate.
    Search(SearchValue),
}

/// Wrap an engine bound. Only `RoundMode::Outward` produces a [`Certified`];
/// a non-finite value is never certified.
#[must_use]
pub(crate) fn classify(mode: RoundMode, b: f64) -> MarginRowValue {
    match outward_bound(mode, b) {
        Some(c) => MarginRowValue::Certified(c),
        None => MarginRowValue::Search(SearchValue::new(b)),
    }
}

/// The certified form of an Outward engine bound, or `None` (Parity mode, or a
/// non-finite value).
#[must_use]
pub(crate) fn outward_bound(mode: RoundMode, b: f64) -> Option<Certified<f64>> {
    (mode.outward() && b.is_finite()).then(|| Certified::new(b, CertReceipt::MarginRowOutward))
}

/// The certified form of an emptiness slack (`certified lower − certified
/// upper` of one coordinate), or `None` unless it is Outward, finite and
/// strictly positive.
#[must_use]
pub(crate) fn outward_emptiness(mode: RoundMode, slack: f64) -> Option<Certified<f64>> {
    (mode.outward() && slack.is_finite() && slack > 0.0)
        .then(|| Certified::new(slack, CertReceipt::MarginRowEmptiness))
}

/// The margin-row closure decision: `closed` on the certified form, `false`
/// for a search value. Bit-identical to the historical Outward predicate
/// (`b.is_finite() && b > 0.0`); the historical Parity branch (`b >= 0.0`) is
/// gone — a parity value has no closure predicate.
#[must_use]
pub fn closes(mode: RoundMode, b: f64) -> bool {
    match classify(mode, b) {
        MarginRowValue::Certified(c) => closed(&c),
        MarginRowValue::Search(_) => false,
    }
}

/// Leaf evidence for a bound close: `Some` exactly when [`closes`] holds.
#[must_use]
pub(crate) fn bound_evidence(mode: RoundMode, b: f64) -> Option<LeafEvidence> {
    outward_bound(mode, b)
        .filter(closed)
        .map(LeafEvidence::Bound)
}

/// Leaf evidence for a certified-emptiness close.
#[must_use]
pub(crate) fn empty_evidence(mode: RoundMode, slack: f64) -> Option<LeafEvidence> {
    outward_emptiness(mode, slack).map(LeafEvidence::Empty)
}

/// A class discharged at the root: `Some` exactly when its certified root
/// bound closes.
#[must_use]
pub(crate) fn root_class(mode: RoundMode, class: usize, dj: f64) -> Option<ClassCover> {
    outward_bound(mode, dj)
        .filter(closed)
        .map(|bound| ClassCover::Root { class, bound })
}

/// A class the frozen-dual tier closed. Re-derives the closure from the
/// search's own report rather than trusting the tier's `closed` list: either
/// the state root pruned (no level was needed), or the level ledger closes
/// (`ledger_closes`, valid because the enumeration that fed it is the trusted
/// `zono_bab` loop itself).
#[must_use]
pub(crate) fn frozen_dual_class(report: &ClassReport) -> Option<ClassCover> {
    let run = report.result.as_ref().ok()?;
    let search = &run.search;
    if search.verdict != SearchVerdict::Closed {
        return None;
    }
    let value = if search.levels.is_empty() {
        search.root_certified
    } else if ledger_closes(&search.levels) {
        search.cert_min_slack
    } else {
        return None;
    };
    let cert = Certified::new(value, CertReceipt::FrozenDual);
    closed(&cert).then_some(ClassCover::FrozenDual {
        class: report.class,
        cert,
    })
}

/// The split that created a child leaf, when the leaf is a child closed at
/// its parent's expansion (not yet an entry of its own).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildSplit {
    /// A trunk ReLU split `(layer, pos, dir)`.
    Trunk(TrunkSplit),
    /// A head split `(index, dir)`.
    Head(HeadFix),
}

fn side(dir: i8) -> Option<bool> {
    match dir.signum() {
        1 => Some(true),
        -1 => Some(false),
        _ => None,
    }
}

fn trunk_literal(&(layer, pos, dir): &TrunkSplit) -> Option<CoverLiteral> {
    Some(CoverLiteral {
        site: CoverSite::Trunk {
            layer: u32::try_from(layer).ok()?,
            pos: u32::try_from(pos).ok()?,
        },
        positive: side(dir)?,
    })
}

fn head_literal(&(index, dir): &HeadFix) -> Option<CoverLiteral> {
    Some(CoverLiteral {
        site: CoverSite::Head {
            index: u32::try_from(index).ok()?,
        },
        positive: side(dir)?,
    })
}

/// A leaf's path below the frame root, from the lane's OWN domain record: the
/// trunk splits and head fixes its certified bound was computed under, minus
/// the head fixes the frame root was entered with, plus the child's split when
/// the leaf is a child closed at expansion. `None` when the head prefix is not
/// the frame root's, a split carries no side, or one site carries both sides.
#[must_use]
pub fn leaf_path(
    trunk: &[TrunkSplit],
    heads: &[HeadFix],
    root_heads: &[HeadFix],
    child: Option<ChildSplit>,
) -> Option<LeafPath> {
    if heads.get(..root_heads.len()) != Some(root_heads) {
        return None;
    }
    let below = &heads[root_heads.len()..];
    let mut lits = Vec::with_capacity(trunk.len() + below.len() + 1);
    for split in trunk {
        lits.push(trunk_literal(split)?);
    }
    for fix in below {
        lits.push(head_literal(fix)?);
    }
    match child {
        Some(ChildSplit::Trunk(split)) => lits.push(trunk_literal(&split)?),
        Some(ChildSplit::Head(fix)) => lits.push(head_literal(&fix)?),
        None => {}
    }
    LeafPath::from_literals(lits)
}

impl super::sink::SiteMap {
    /// A frame root's site translation and root α (ENGINE_PROGRAM §1.7 B1,
    /// proof bundle v2): `Trunk { layer, pos }` → the layer's Relu op and
    /// `unst[pos]`; `Head { index }` → the head Relu op (the op consuming the
    /// first head Gemm's output) and `index`. The α table is the root's lower
    /// slopes at every unstable trunk neuron plus the head α when installed.
    /// Built only at STREAM time, after the frame's token exists.
    #[must_use]
    pub fn from_root(
        root: &crate::margin_row::root::RootGates,
        net: &crate::margin_row::net::TwinNet,
    ) -> Self {
        let to32 = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        let trunk = root
            .layers
            .iter()
            .map(|l| (to32(l.op), l.unst.iter().map(|&i| to32(i)).collect()))
            .collect();
        let head_op = to32(net.i_gemm1 + 1);
        let mut alpha = Vec::new();
        for l in &root.layers {
            if l.unst.is_empty() {
                continue;
            }
            let idx: Vec<u32> = l.unst.iter().map(|&i| to32(i)).collect();
            let val: Vec<f64> = l
                .unst
                .iter()
                .map(|&i| l.alpha.get(i).copied().unwrap_or(f64::NAN))
                .collect();
            alpha.push((to32(l.op), idx, val));
        }
        if let Some(ha) = &root.head_alpha {
            alpha.push((head_op, (0..ha.len()).map(to32).collect(), ha.clone()));
        }
        Self::new(trunk, head_op, alpha)
    }
}
