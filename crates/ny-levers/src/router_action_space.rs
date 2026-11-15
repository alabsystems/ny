// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r1 (ROUTER_PROGRAM §5.4): the router's v0 ACTION SPACE — the only
//! typed keys a router- or optimizer-EMITTED value may ever name.
//!
//! Derived, never listed by hand: every [`crate::space::axes`] entry with
//! `Deliver::PresetKey`, class not `Unsafe`, whose declaration's moat risk is
//! `None` or `Low`, minus an explicit deny-list. The derivation is frozen in
//! `tests/data/router_action_space.tsv` by `tests/router_action_space_ratchet.rs`
//! (any delta fails, in both directions, with every row reported at once).
//!
//! The deny-list exists because the class/moat filter alone is one loosened
//! rule away from readmitting the keys that change what is proved or how it is
//! rounded: `root_f32`, `tableau_f32` and `err_envelope` are VerdictAffecting /
//! High, NOT Unsafe, so a future "High is fine inside a clamp" rule would pull
//! them in. They, the reserve / release family, batch-norm folding, attack
//! tolerances and the router's own two modes stay out forever.
//!
//! `High` knobs (`alpha_iters`, `interm_alpha`, `bi_topk`, ...) are never
//! router-emitted values: they move only inside a capability-owned rule with
//! its own LeverDecl, clamp-soundness audit and S1–S8.
//!
//! R1 emits NOTHING (`Plan::emitted` is always empty); this module is the
//! fence that R6's emitted values must stay inside.

use crate::space::{self, Class, Deliver};
use crate::MoatRisk;

/// Typed keys the router may never emit, whatever their class or moat risk.
/// Each must name a real preset-delivered axis (the ratchet checks), so the
/// list cannot rot into a placebo.
pub const DENY_KEYS: &[&str] = &[
    "margin_row.root_f32",
    "margin_row.tableau_f32",
    "margin_row.err_envelope",
    "margin_row.reserve_secs",
    "margin_row.adaptive_reserve",
    "margin_row.reserve_max_frac",
    "margin_row.release_frac",
    "bab.root_bab_reserve_frac",
    "bab.phase_budget.post_bab_reserve_secs",
    "bab.fast_f32_prewarm",
    "router.mode",
    "optimizer.mode",
];

/// Substrings no router-emitted key may contain: precision, rounding and
/// certified-error terms, BN folding, reserves and releases, tolerances, and
/// the router's own keys.
pub const DENY_PATTERNS: &[&str] = &[
    "f32",
    "f16",
    "precision",
    "err_envelope",
    "batch_norm",
    "reserve",
    "release",
    "tolerance",
    "ulps",
    "router.",
    "optimizer.",
];

/// One admissible router-emitted axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActionAxis {
    /// The typed preset key.
    pub key: &'static str,
    /// The lever it delivers.
    pub lever: &'static str,
    /// `safe-to-search` or `verdict-affecting`.
    pub class: &'static str,
    /// `none` or `low`.
    pub moat: &'static str,
}

const fn class_token(class: Class) -> &'static str {
    match class {
        Class::Unsafe => "unsafe",
        Class::VerdictAffecting => "verdict-affecting",
        Class::SafeToSearch => "safe-to-search",
    }
}

/// Whether `key` is denied by name or by pattern.
#[must_use]
pub fn denied(key: &str) -> bool {
    DENY_KEYS.contains(&key) || DENY_PATTERNS.iter().any(|pattern| key.contains(pattern))
}

/// Every preset-delivered, non-Unsafe axis whose moat risk is `None`/`Low`,
/// BEFORE the deny-list. Axes whose lever is undeclared are skipped (the space
/// module's own tests fail on them).
#[must_use]
pub fn candidates() -> Vec<ActionAxis> {
    let mut out: Vec<ActionAxis> = space::axes()
        .iter()
        .filter(|axis| axis.class != Class::Unsafe)
        .filter_map(|axis| {
            let Deliver::PresetKey(key) = axis.deliver else {
                return None;
            };
            let decl = crate::all().get(axis.name)?;
            let moat = match decl.moat {
                MoatRisk::None => "none",
                MoatRisk::Low => "low",
                MoatRisk::High => return None,
            };
            Some(ActionAxis {
                key,
                lever: axis.name,
                class: class_token(axis.class),
                moat,
            })
        })
        .collect();
    out.sort_unstable();
    out
}

/// The action space: [`candidates`] minus the deny-list, sorted by key.
#[must_use]
pub fn action_space() -> Vec<ActionAxis> {
    candidates()
        .into_iter()
        .filter(|axis| !denied(axis.key))
        .collect()
}

/// The manifest body (without its comment header): one
/// `key<TAB>lever<TAB>class<TAB>moat` row per axis.
#[must_use]
pub fn render_rows(axes: &[ActionAxis]) -> String {
    let mut out = String::new();
    for axis in axes {
        out.push_str(axis.key);
        out.push('\t');
        out.push_str(axis.lever);
        out.push('\t');
        out.push_str(axis.class);
        out.push('\t');
        out.push_str(axis.moat);
        out.push('\n');
    }
    out
}
