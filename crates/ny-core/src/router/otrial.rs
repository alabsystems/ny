// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r6 — the optimizer's O-TRIAL rules (ROUTER_PROGRAM §5.1, the §5.2
//! rows tagged R6, §5.3, §10 R6): the pure decision of which trial rules ACT
//! on a row, at which sites, with which constants. PURE: no clock, no
//! environment, no global; the same `(plan, input, r6, mode, switches)` gives
//! the same [`TrialCommitments`] bit for bit, which `ny vnncomp plan
//! --replay` re-checks.
//!
//! # The four rules, each its own capability and its own typed key
//!
//! | rule | key (`optimizer.rules.*`) | sites | machine |
//! |---|---|---|---|
//! | `rule.trial_width` | `trial_width` | the wave tree's width, the reordered input-split batch (dormancy-aware), the lowdim wave width | AIMD on certified closes / s inside the lane's static memory budget |
//! | `rule.trial_lr_backoff` | `trial_lr_backoff` | the root alpha ascent's Adam step | backoff only: 3 rejected iterates => x 0.5, floor lr0 / 8, revert on a falling gain rate |
//! | `rule.trial_split_restart` | `trial_split_restart` | the lowdim lane's split heuristic | restart form: FAR AND zero-lift > 0.9 => one restart with the other heuristic, the tree discarded |
//! | `rule.trial_attack_halving` | `trial_attack_halving` | the upfront APGD lane's initial step | successive halving over a fixed step pool by improvement rate |
//!
//! A site is armed only when `optimizer.mode` is `on` (effective `acting`),
//! its rule's key is on (the declaration default; the key is an ablation
//! switch), the row is not short (CR-S: no active probing on rows of 30 s or
//! less), its capability DECLARES safe points (a trial changes a value only
//! between units), its capability's ledger is SEALED (review fix: never
//! trial inside an `Open` lane, `blocked_by=ledger:<cap>` — today the wave
//! tree, until M6 seals it), and the row did not PIN its knob (a pinned knob is frozen,
//! `blocked_by=pin:<path>`). A rule acts when at least one of its sites is
//! armed. Every blocked site says why.
//!
//! # Name-blind
//!
//! The input is the plan, the policy input and the typed key paths the row
//! pinned; none carries a name or a path to a model or a property.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::acting::RuleAct;
use super::ostop::R3_RULES;
use super::plan::{Plan, PolicyInput, SHORT_ROW_MS};
use super::safepoint::{self, At};
use super::trial::{HalvingRule, LrRule, RestartRule, WidthRule};
use super::{EffectiveMode, RuleId};
use crate::trial_authority::{TrialDirectives, TrialSite};

/// Record schema of [`TrialCommitments`].
pub const OTRIAL_SCHEMA: &str = "ny-router-otrial/v1";

/// `rule.trial_width`: AIMD on a fast lane's width.
pub const RULE_TRIAL_WIDTH: RuleId = RuleId("rule.trial_width");
/// `rule.trial_lr_backoff`: an ascent's step backoff.
pub const RULE_TRIAL_LR_BACKOFF: RuleId = RuleId("rule.trial_lr_backoff");
/// `rule.trial_split_restart`: one restart with the other split heuristic.
pub const RULE_TRIAL_SPLIT_RESTART: RuleId = RuleId("rule.trial_split_restart");
/// `rule.trial_attack_halving`: successive halving over an attack's step pool.
pub const RULE_TRIAL_ATTACK_HALVING: RuleId = RuleId("rule.trial_attack_halving");

/// The R6 acting optimizer rules, in evaluation order.
pub const R6_RULES: [RuleId; 4] = [
    RULE_TRIAL_WIDTH,
    RULE_TRIAL_LR_BACKOFF,
    RULE_TRIAL_SPLIT_RESTART,
    RULE_TRIAL_ATTACK_HALVING,
];

/// The optimizer's acting set: R3's four O-STOP rules followed by R6's four
/// O-TRIAL rules.
pub const ACTING_SET: [RuleId; 8] = [
    R3_RULES[0],
    R3_RULES[1],
    R3_RULES[2],
    R3_RULES[3],
    R6_RULES[0],
    R6_RULES[1],
    R6_RULES[2],
    R6_RULES[3],
];

/// The four `optimizer.rules.trial_*` ablation switches. Each defaults ON —
/// it participates when the optimizer acts — so `optimizer.mode` is the only
/// arming decision, and a key set `false` ablates one rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrialSwitches {
    /// `rule.trial_width`.
    pub width: bool,
    /// `rule.trial_lr_backoff`.
    pub lr_backoff: bool,
    /// `rule.trial_split_restart`.
    pub split_restart: bool,
    /// `rule.trial_attack_halving`.
    pub attack_halving: bool,
}

impl TrialSwitches {
    /// Every rule participates (the declarations' defaults).
    pub const ALL_ON: Self = Self {
        width: true,
        lr_backoff: true,
        split_restart: true,
        attack_halving: true,
    };

    /// The switch for `rule`.
    #[must_use]
    pub fn get(&self, rule: RuleId) -> bool {
        match rule.as_str() {
            "rule.trial_width" => self.width,
            "rule.trial_lr_backoff" => self.lr_backoff,
            "rule.trial_split_restart" => self.split_restart,
            "rule.trial_attack_halving" => self.attack_halving,
            _ => false,
        }
    }
}

/// What the trial commit reads beyond the static policy input: the typed key
/// paths the row PINNED (its own file under any path, or a non-default lever
/// source). Recorded so a replay re-derives the commitments.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct R6Input {
    /// Pinned key paths, sorted.
    pub pinned_keys: Vec<String>,
}

/// One trial site's decision.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SiteAct {
    /// The site.
    pub site: TrialSite,
    /// Its rule.
    pub rule: String,
    /// The capability (registry id).
    pub cap: String,
    /// The knob.
    pub knob: String,
    /// Whether the site is armed.
    pub armed: bool,
    /// Why not (`mode:<token>`, `key`, `short_row`, `safe_points`,
    /// `ledger:<cap>`, `pin:<path>`).
    pub blocked_by: Option<String>,
}

/// What the optimizer's trial rules commit to on a row (pure; replayable).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialCommitments {
    /// [`OTRIAL_SCHEMA`].
    pub schema: String,
    /// Whether the optimizer acts at all on this row.
    pub acting: bool,
    /// Every R6 rule's decision, in [`R6_RULES`] order.
    pub rules: Vec<RuleAct>,
    /// Every site's decision, in [`TrialSite::ALL`] order.
    pub sites: Vec<SiteAct>,
    /// The directives the trial authority is armed with.
    pub directives: TrialDirectives,
    /// Hex sha256 of everything above.
    pub sha: String,
}

impl TrialCommitments {
    /// The first 8 hex digits of [`TrialCommitments::sha`].
    #[must_use]
    pub fn sha8(&self) -> &str {
        self.sha.get(..8).unwrap_or(&self.sha)
    }

    /// Whether `rule` acts on this row.
    #[must_use]
    pub fn acts(&self, rule: RuleId) -> bool {
        self.rules
            .iter()
            .any(|act| act.rule == rule.as_str() && act.acts)
    }

    /// The decision for `site`.
    #[must_use]
    pub fn site(&self, site: TrialSite) -> Option<&SiteAct> {
        self.sites.iter().find(|s| s.site == site)
    }
}

/// Review fix (router-r6): whether the capability `cap` is SEALED in the
/// row's input (`SealedAsShipped` or `Sealed`). A lane whose ledger is still
/// `Open` (S7+S8 empty) or `Shadow` is never tuned by a trial (§5.3: "knob in
/// action space or owned by a sealed capability rule"; §6.2: "lanes whose
/// ledger is Open" are never in the action space; R2's acting rules refuse
/// the same way with `ledger:<id>`). An undeclared capability is not sealed.
#[must_use]
pub fn cap_sealed(input: &PolicyInput, cap: &str) -> bool {
    input
        .capabilities
        .iter()
        .find(|c| c.id == cap)
        .is_some_and(|c| c.ledger.is_sealed())
}

/// Whether the capability `cap` declares a safe point at a unit boundary in
/// the row's input (an undeclared capability never does).
#[must_use]
pub fn cap_retunable(input: &PolicyInput, cap: &str) -> bool {
    let declared = safepoint::declared_or_never(
        input
            .capabilities
            .iter()
            .find(|c| c.id == cap)
            .map(|c| c.safe_points),
    );
    safepoint::may_suspend(declared, At::Unit(0)).allowed()
}

fn switch_for(site: TrialSite, switches: &TrialSwitches) -> bool {
    match site {
        TrialSite::WaveWidth | TrialSite::InputSplitWidth | TrialSite::LowdimWidth => {
            switches.width
        }
        TrialSite::AlphaLr => switches.lr_backoff,
        TrialSite::LowdimSplit => switches.split_restart,
        TrialSite::AttackStep => switches.attack_halving,
    }
}

fn what(rule: RuleId) -> String {
    match rule.as_str() {
        "rule.trial_width" => format!(
            "AIMD on certified closes/s per {}-unit window: x2 on >= {} rise with lease < {}, /2 \
             on >= {} fall; two falls after a raise => last good + freeze; lease > {} => revert \
             / halve; <= {} trials per knob, >= {} windows left, never past {} of the lane",
            WidthRule::SHIPPED.window_units,
            WidthRule::SHIPPED.raise_rise,
            WidthRule::SHIPPED.lease_raise_max,
            WidthRule::SHIPPED.fall_drop,
            WidthRule::SHIPPED.lease_guard,
            WidthRule::SHIPPED.max_trials,
            WidthRule::SHIPPED.windows_left_min,
            WidthRule::SHIPPED.late_frac,
        ),
        "rule.trial_lr_backoff" => format!(
            "backoff only: {} rejected iterates => lr x {}, floor lr0/{}; revert + tabu when the \
             next {} iterates' gain/s falls",
            LrRule::SHIPPED.reject_run,
            LrRule::SHIPPED.factor,
            LrRule::SHIPPED.floor_div,
            LrRule::SHIPPED.eval_iterates,
        ),
        "rule.trial_split_restart" => format!(
            "restart form: Volume FAR over two trailing windows and zero-lift fraction > {} => one \
             restart with the other split heuristic; the tree is discarded, no fact carried",
            RestartRule::SHIPPED.zero_lift_min,
        ),
        _ => format!(
            "successive halving over the step pool {:?} by improvement rate; restart 0 keeps \
             today's step",
            HalvingRule::SHIPPED.pool,
        ),
    }
}

/// The row's O-TRIAL commitments. Pure and total.
#[must_use]
pub fn commit(
    plan: &Plan,
    input: &PolicyInput,
    r6: &R6Input,
    mode: EffectiveMode,
    switches: &TrialSwitches,
) -> TrialCommitments {
    let acting = matches!(mode, EffectiveMode::Acting);
    let short_row = plan.short_row || input.budget.official_ms <= SHORT_ROW_MS;
    let sites: Vec<SiteAct> = TrialSite::ALL
        .into_iter()
        .map(|site| {
            let blocked_by = if !acting {
                Some(format!("mode:{}", mode.token()))
            } else if !switch_for(site, switches) {
                Some("key".to_owned())
            } else if short_row {
                Some("short_row".to_owned())
            } else if !cap_retunable(input, site.token()) {
                Some("safe_points".to_owned())
            } else if !cap_sealed(input, site.token()) {
                // Review fix (router-r6): never trial inside an unsealed lane
                // (the wave tree, `wave-beta-bab`, is `Open` until M6 seals
                // it — ROUTER_PROGRAM §10.1 R6 depends on "M6 gated").
                Some(format!("ledger:{}", site.token()))
            } else {
                site.pin_paths()
                    .iter()
                    .find(|path| r6.pinned_keys.iter().any(|k| k == *path))
                    .map(|path| format!("pin:{path}"))
            };
            SiteAct {
                site,
                rule: site.rule().to_owned(),
                cap: site.token().to_owned(),
                knob: site.knob().to_owned(),
                armed: blocked_by.is_none(),
                blocked_by,
            }
        })
        .collect();
    let rules: Vec<RuleAct> = R6_RULES
        .iter()
        .map(|rule| {
            let mine: Vec<&SiteAct> = sites.iter().filter(|s| s.rule == rule.as_str()).collect();
            let acts = mine.iter().any(|s| s.armed);
            let blocked_by = (!acts).then(|| {
                mine.first()
                    .and_then(|s| s.blocked_by.clone())
                    .unwrap_or_else(|| "no_site".to_owned())
            });
            RuleAct {
                rule: rule.as_str().to_owned(),
                acts,
                blocked_by,
                what: what(*rule),
            }
        })
        .collect();
    let mut bits = 0u32;
    for s in &sites {
        if s.armed {
            bits |= s.site.bit();
        }
    }
    let acts = |rule: RuleId| rules.iter().any(|a| a.rule == rule.as_str() && a.acts);
    let directives = TrialDirectives {
        width: acts(RULE_TRIAL_WIDTH).then_some(WidthRule::SHIPPED),
        lr_backoff: acts(RULE_TRIAL_LR_BACKOFF).then_some(LrRule::SHIPPED),
        split_restart: acts(RULE_TRIAL_SPLIT_RESTART).then_some(RestartRule::SHIPPED),
        attack_halving: acts(RULE_TRIAL_ATTACK_HALVING).then_some(HalvingRule::SHIPPED),
        sites: bits,
    };
    finish(TrialCommitments {
        schema: OTRIAL_SCHEMA.to_owned(),
        acting,
        rules,
        sites,
        directives,
        sha: String::new(),
    })
}

/// Stamp the digest (over everything but the digest itself).
fn finish(mut commitments: TrialCommitments) -> TrialCommitments {
    commitments.sha = String::new();
    let mut hash = Sha256::new();
    hash.update(format!("{commitments:?}").as_bytes());
    let digest: [u8; 32] = hash.finalize().into();
    commitments.sha = super::hash::hex(&digest);
    commitments
}
