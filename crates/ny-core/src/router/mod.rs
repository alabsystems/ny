// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r1 / #router-r2 — the verdict-free runtime router and optimizer
//! (`docs/engine_program/router/ROUTER_PROGRAM.md`, steps R1: SHADOW and R2:
//! ACTING SET 1).
//!
//! ny decides at runtime, per row, which capabilities run and how the budget
//! is split, and tunes its own search/effort parameters from measured
//! progress. This module is the POLICY half: plain data and pure functions.
//!
//! * [`decl`] — the capability-declaration vocabulary (data only).
//! * [`facts`] — static row facts and the host envelope (data only).
//! * [`plan`] — the pure plan: admissions, order, contracts, reserves,
//!   checkpoints, and what the rules would do.
//! * [`acting`] — R2: the pure commitments of the seven acting rules (CR-1,
//!   CR-A, CR-C, CR-T, CR-H, CR-R, CR-K) — which lanes the ONE deadline
//!   authority governs, the attack-slice contracts, the terminal, hand-on,
//!   the publish grace and the derived caps.
//! * [`derive`] — O-DERIVE: caps from the deadline and the memory lease.
//! * [`ostop`] — R3: the pure commitments of the four O-STOP optimizer rules
//!   (`rule.governor`, `rule.ia_yield`, `rule.attack_defer` = CR-ATK,
//!   `rule.root_admission`) and CR-ATK's slope predicate.
//! * [`r4`] — R4: the pure commitments of acting set 3, CR-L (the memory
//!   lease, arithmetic in [`lease`]) and CR-F (the FAR hand-on routes).
//! * [`safepoint`] — R4: the enforceable safe-point rulings (`None` = never).
//! * [`route`] — R5: the pure commitments of acting set 4, CR-RS (route a
//!   capability only at SR.seal, name-blind, from facts and the host) and
//!   CR-G (gap-order contracts, arithmetic in [`gap_contract`]).
//! * [`otrial`] — R6: the pure commitments of the four O-TRIAL optimizer
//!   rules (`rule.trial_width`, `rule.trial_lr_backoff`,
//!   `rule.trial_split_restart`, `rule.trial_attack_halving`) and the sites
//!   each arms; [`trial`] — their pure state machines (AIMD width, lr
//!   backoff, the restart-form heuristic switch, successive halving).
//! * [`priors`] — R7: the compiled OFFLINE PRIORS v1 — per-phase COST models
//!   (time from static facts, keyed only by two operator bits, one shared
//!   size exponent) with the anti-fingerprint gates, read by the plan only
//!   under `optimizer.priors: true` and only on the host class they were
//!   measured on; Layer-B success curves ship uniform.
//! * [`spec`] — #admission-precision: the specification's routing shape
//!   (data, plus the pure archetype mapping).
//! * [`admission`] — #admission-precision: the lanes' own route checks as one
//!   callable implementation (the lanes call them inline; the router composes
//!   them into each routable consumer's site, `decl::AdmitSite`).
//! * [`checkpoint`] — the pure c1/c2/c3 checkpoints (no promotion variant)
//!   and, since R4, CR-F's per-capability `far_decision`.
//! * [`project`] — FAR / NOT-FAR projections in work units.
//! * [`hash`] — the static-input plan hash.
//! * [`bus`] — the write-only, never-blocking progress bus lanes push to.
//! * [`guard`] — `fail_open`: every router entry is panic-contained.
//!
//! The cells the acting rules write live OUTSIDE this module, in
//! `crate::deadline_authority` (layer L4 of the program: untrusted,
//! saturating, one monotone cell per lane) and, since R3,
//! `crate::stop_authority` (layer L5: which in-lane stop rules are armed and
//! the ledger of what each stop freed); this module only decides.
//!
//! SOUNDNESS. The router is verdict-free by construction: nothing here names a
//! verdict, a publisher or the proof profile (source ratchets in
//! `tests/router_ratchet.rs`), and it emits no action-space knob value
//! ([`plan::Plan::emitted`] stays empty). The R2 rules move TIME only — lane
//! deadlines within caps the route already keeps, a bounded publish grace for
//! an already-complete certificate, and rule-owned caps (depth, retry passes,
//! the warmup hand-on) whose soundness is argued in their own code and pinned
//! there. A wrong route or parameter can cost a proof, never mint one; every
//! rule is dark by default (`router.mode` defaults `off`), each has its own
//! ablation key, and any fault fails open to today's route.

pub mod acting;
pub mod admission;
pub mod bus;
pub mod checkpoint;
pub mod decl;
pub mod derive;
pub mod facts;
pub mod gap_contract;
pub mod guard;
pub mod hash;
pub mod lease;
pub mod ostop;
pub mod otrial;
pub mod plan;
pub mod priors;
pub mod project;
pub mod r4;
pub mod route;
pub mod safepoint;
pub mod spec;
pub mod trial;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

pub use decl::RuleId;

/// The router's record schema.
pub const ROUTER_SCHEMA: &str = "ny-router/v1";

/// The priors version a policy input carries when NO prior is read (the
/// default: `optimizer.priors` off, or the table unavailable). Layer-B success
/// curves are uniform in every v1 table (§5.5: every runtime-signal record is
/// one family); #router-r7's cost models, when read, replace this token with
/// [`priors::PriorTable::version_token`] (`cost-v1+<sha8>`).
pub const PRIORS_VERSION: &str = "uniform-v1";

/// Router rules that ACT in this binary when `router.mode` is `on`.
///
/// R2 (ROUTER_PROGRAM §10): acting set 1 — CR-1, CR-A, CR-C, CR-T, CR-H, CR-R,
/// CR-K ([`acting::R2_RULES`]). `router.mode` still defaults `off` and no
/// category yaml may set it (ROUTER_PROGRAM §2.5), so every scored run stays
/// byte-identical until the compiled default moves after the both-year moat;
/// `on` is a measurement arm until then, and each rule can be ablated by its
/// own `router.rules.*` key.
///
/// #router-r4 (ROUTER_PROGRAM §10 R4): acting set 3 joins it — CR-L, the
/// memory lease against the declared host's static envelope, and CR-F, the
/// FAR hand-on at a lane's own safe point to a STRICTLY sealed recipient
/// ([`r4::R4_RULES`]); [`r4::ACTING_SET`] is R2's seven followed by R4's two.
///
/// #router-r5 (ROUTER_PROGRAM §10 R5): acting set 4 joins it — CR-RS, which
/// ROUTES a capability (configures it on the row) only at SR.seal, and CR-G,
/// the gap-order contract inside multi-objective lanes ([`route::R5_RULES`]);
/// [`route::ACTING_SET`] is R4's nine followed by R5's two.
pub const ACTING_RULES: &[RuleId] = &route::ACTING_SET;

/// Optimizer rules that ACT in this binary when `optimizer.mode` is `on`.
///
/// R3 (ROUTER_PROGRAM §10): acting set 2, the O-STOP rules
/// ([`ostop::R3_RULES`]): `rule.governor`, `rule.ia_yield`,
/// `rule.attack_defer` (CR-ATK) and `rule.root_admission`. `optimizer.mode`
/// still defaults `off` and no category yaml may set it, so every scored run
/// stays byte-identical until the compiled default moves after the both-year
/// moat; each rule can be ablated by its own `optimizer.rules.*` key. R2's
/// O-DERIVE caps stay router rules (CR-K).
///
/// #router-r6 (ROUTER_PROGRAM §10 R6): the four O-TRIAL rules join it
/// ([`otrial::R6_RULES`]): `rule.trial_width`, `rule.trial_lr_backoff`,
/// `rule.trial_split_restart` and `rule.trial_attack_halving`, each ablated by
/// its own `optimizer.rules.trial_*` key; [`otrial::ACTING_SET`] is R3's four
/// followed by R6's four.
pub const ACTING_OPTIMIZER_RULES: &[RuleId] = &otrial::ACTING_SET;

/// The requested router mode (`router.mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouterMode {
    /// Dark: no facts, no plan, no bus, no output.
    #[default]
    Off,
    /// Compute, log and replay the plan; change nothing.
    Shadow,
    /// Run the acting rules of this binary (R2: [`ACTING_RULES`]).
    On,
}

impl RouterMode {
    /// The declared tokens, in declaration order.
    pub const TOKENS: &'static [&'static str] = &["off", "shadow", "on"];

    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::On => "on",
        }
    }

    /// Parse a token.
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "off" => Some(Self::Off),
            "shadow" => Some(Self::Shadow),
            "on" => Some(Self::On),
            _ => None,
        }
    }
}

/// The requested optimizer mode (`optimizer.mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerMode {
    /// Dark.
    #[default]
    Off,
    /// Log what the O-STOP / O-DERIVE rules would do; change nothing.
    Shadow,
    /// Run the acting optimizer rules (R3: [`ACTING_OPTIMIZER_RULES`]).
    On,
}

impl OptimizerMode {
    /// The declared tokens, in declaration order.
    pub const TOKENS: &'static [&'static str] = &["off", "shadow", "on"];

    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::On => "on",
        }
    }

    /// Parse a token.
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "off" => Some(Self::Off),
            "shadow" => Some(Self::Shadow),
            "on" => Some(Self::On),
            _ => None,
        }
    }
}

/// What a requested mode actually does in this binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EffectiveMode {
    /// Dark.
    Off,
    /// Observe and log only.
    Shadow,
    /// Acting rules run.
    Acting,
}

impl EffectiveMode {
    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Acting => "acting",
        }
    }

    /// Whether the router observes anything at all.
    #[must_use]
    pub const fn observes(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// `router.mode` in this binary: `on` with an empty acting set is `shadow`.
#[must_use]
pub const fn effective(requested: RouterMode, acting: &[RuleId]) -> EffectiveMode {
    match requested {
        RouterMode::Off => EffectiveMode::Off,
        RouterMode::Shadow => EffectiveMode::Shadow,
        RouterMode::On => {
            if acting.is_empty() {
                EffectiveMode::Shadow
            } else {
                EffectiveMode::Acting
            }
        }
    }
}

/// `optimizer.mode` in this binary: `on` with an empty acting set is `shadow`.
#[must_use]
pub const fn effective_optimizer(requested: OptimizerMode, acting: &[RuleId]) -> EffectiveMode {
    match requested {
        OptimizerMode::Off => EffectiveMode::Off,
        OptimizerMode::Shadow => EffectiveMode::Shadow,
        OptimizerMode::On => {
            if acting.is_empty() {
                EffectiveMode::Shadow
            } else {
                EffectiveMode::Acting
            }
        }
    }
}
