// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r3 — ACTING SET 2, the optimizer's O-STOP rules (ROUTER_PROGRAM
//! §5.1, §5.2, §10 R3): the pure decision of which in-lane stop rules ACT on
//! a row, with which constants. PURE: no clock, no environment, no global;
//! the same `(plan, input, mode, switches, r2)` gives the same
//! [`StopCommitments`] bit for bit, which `ny vnncomp plan --replay` re-checks.
//!
//! # The four rules, each its own capability and its own typed key
//!
//! | rule | key (`optimizer.rules.*`) | what may stop early | rides on |
//! |---|---|---|---|
//! | `rule.governor` | `governor` | the margin-row root alpha ascent: yield when an iteration's gain `g_i < 0.15 g*` (its OWN best), after 5 samples; the best iterate is kept and certified once by the ascent's unchanged published guard | — |
//! | `rule.ia_yield` | `ia_yield` | the `#interm-alpha` candidate loop inside `backward_interm` (same 0.15, after 2 layers); the baseline shrink-only pass continues | CR-1: the ascent reads a CONTRACT that excludes the freed interm time, so it cannot inflate the ascent's `0.4 * remaining` share |
//! | `rule.attack_defer` (CR-ATK) | `attack_defer` | the upfront DLR-APGD attack at a restart boundary, when its best margin rose by no more than the noise floor over the trailing half of its slice (slope `<= eps / W`, `W >= 0.5` of the slice); the unspent part is DEFERRED to a late sat reserve held out of the verifier, never deleted (floors = today's caps) | CR-1 + the upfront slot governed (CR-A); never on a short row (CR-S) |
//! | `rule.root_admission` | `root_admission` | a root tightening phase whose padded predicted cost does not fit `phase_window::admit` against the window it would run in (the verifier's cell when governed), with the predicate's inputs logged | — |
//!
//! A rule acts only when `optimizer.mode` is `on` (effective `acting`), its
//! own key is on (the declaration default; the key is an ablation switch),
//! and the R2 rules it rides on act. A blocked rule says why (`blocked_by`).
//!
//! # One direction
//!
//! Every rule only ever ENDS a phase earlier than its own cap and hands the
//! rest downstream; none extends a phase, a lane or a cell. No rule touches
//! precision, a certified error term, a tolerance, an acceptance gate or the
//! publication of a result.
//!
//! # Name-blind
//!
//! The input is the plan and the policy input, which carry no name and no
//! path (type-level blindness).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::acting::{Commitments, RuleAct};
use super::plan::{Plan, PolicyInput, CR_1, SHORT_ROW_MS};
use super::safepoint::{self, At};
use super::{EffectiveMode, RuleId};
use crate::deadline_authority::LaneSlot;
use crate::stop_authority::{
    DeferRule, RatioRule, StopDirectives, StopSite, MIN_DEFER_WINDOW_FRAC,
};

/// Record schema of [`StopCommitments`].
pub const OSTOP_SCHEMA: &str = "ny-router-ostop/v1";

/// `rule.governor`: the alpha ascent's relative self-decay stop.
pub const RULE_GOVERNOR: RuleId = RuleId("rule.governor");
/// `rule.ia_yield`: the `#interm-alpha` candidate loop's relative self-decay stop.
pub const RULE_IA_YIELD: RuleId = RuleId("rule.ia_yield");
/// `rule.attack_defer` (CR-ATK): defer a flat upfront attack to a late reserve.
pub const RULE_ATTACK_DEFER: RuleId = RuleId("rule.attack_defer");
/// `rule.root_admission`: skip a root phase that does not fit its window.
pub const RULE_ROOT_ADMISSION: RuleId = RuleId("rule.root_admission");

/// The R3 acting optimizer rules, in evaluation order.
pub const R3_RULES: [RuleId; 4] = [
    RULE_GOVERNOR,
    RULE_IA_YIELD,
    RULE_ATTACK_DEFER,
    RULE_ROOT_ADMISSION,
];

/// `rule.governor`'s ratio. FITTED on one family (cifar100: the measured
/// ascent profile yields at iteration 12, the setting that converted
/// idx_1280 / idx_7886) — detector(n=1) until a second family reproduces a
/// converting stop (ROUTER_PROGRAM §10 R3 gate).
pub const GOVERNOR_RATIO: f64 = 0.15;

/// `rule.governor`'s minimum samples: the MOAT-CLEARED artifact's value (51
/// banked unsat rows held 51/51 with ratio 0.15 / 5 samples, yields at
/// iterations 5-13). ROUTER_PROGRAM §5.2 prints ">= 2"; the measured artifact
/// is what ships.
pub const GOVERNOR_MIN_SAMPLES: u32 = 5;

/// `rule.ia_yield`'s ratio (the same scale-free ratio; measured interm-alpha
/// profile 255.6 -> 20.4 -> 5.7 /s fires on the second layer).
pub const IA_YIELD_RATIO: f64 = 0.15;

/// `rule.ia_yield`'s minimum samples: the first layer sets the high-water rate
/// and can never itself trigger a stop.
pub const IA_YIELD_MIN_SAMPLES: u32 = 2;

/// CR-ATK's trailing window, as a fraction of the lane's slice (`W >= 0.5`).
pub const ATTACK_WINDOW_FRAC: f64 = MIN_DEFER_WINDOW_FRAC;

/// CR-ATK's smallest deferrable unspent time, milliseconds: the late
/// post-verifier attack's own minimum budget (`POSTBAB_ATTACK_MIN_BUDGET`,
/// 5 s). Below it the late attack would never start, so the defer would be
/// a cut — the lane runs on instead.
pub const ATTACK_MIN_DEFER_MS: u64 = 5_000;

/// The four `optimizer.rules.*` ablation switches. Each defaults ON — it
/// participates when the optimizer acts — so `optimizer.mode` is the only
/// arming decision, and a key set `false` ablates one rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OptimizerSwitches {
    /// `rule.governor`.
    pub governor: bool,
    /// `rule.ia_yield`.
    pub ia_yield: bool,
    /// `rule.attack_defer` (CR-ATK).
    pub attack_defer: bool,
    /// `rule.root_admission`.
    pub root_admission: bool,
}

impl OptimizerSwitches {
    /// Every rule participates (the declarations' defaults).
    pub const ALL_ON: Self = Self {
        governor: true,
        ia_yield: true,
        attack_defer: true,
        root_admission: true,
    };

    /// The switch for `rule`.
    #[must_use]
    pub fn get(&self, rule: RuleId) -> bool {
        match rule.as_str() {
            "rule.governor" => self.governor,
            "rule.ia_yield" => self.ia_yield,
            "rule.attack_defer" => self.attack_defer,
            "rule.root_admission" => self.root_admission,
            _ => false,
        }
    }
}

/// What the optimizer commits to on a row (pure; replayable).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StopCommitments {
    /// [`OSTOP_SCHEMA`].
    pub schema: String,
    /// Whether the optimizer acts at all on this row.
    pub acting: bool,
    /// Every R3 rule's decision, in [`R3_RULES`] order.
    pub rules: Vec<RuleAct>,
    /// The directives the stop authority is armed with.
    pub directives: StopDirectives,
    /// Hex sha256 of everything above.
    pub sha: String,
}

impl StopCommitments {
    /// The first 8 hex digits of [`StopCommitments::sha`].
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
}

fn act(rule: RuleId, acts: bool, blocked_by: &str, what: String) -> RuleAct {
    RuleAct {
        rule: rule.as_str().to_owned(),
        acts,
        blocked_by: (!acts).then(|| blocked_by.to_owned()),
        what,
    }
}

/// The governor's constants.
pub const GOVERNOR_RULE: RatioRule = RatioRule {
    ratio: GOVERNOR_RATIO,
    min_samples: GOVERNOR_MIN_SAMPLES,
};

/// The interm yield's constants.
pub const IA_YIELD_RULE: RatioRule = RatioRule {
    ratio: IA_YIELD_RATIO,
    min_samples: IA_YIELD_MIN_SAMPLES,
};

/// CR-ATK's constants.
pub const ATTACK_DEFER_RULE: DeferRule = DeferRule {
    window_frac: ATTACK_WINDOW_FRAC,
    min_defer_ms: ATTACK_MIN_DEFER_MS,
};

/// #router-r4: whether the capability a stop rule would END early declares a
/// safe point at a unit boundary in the row's input (an undeclared
/// capability never does). `rule.root_admission` is not asked: skipping a
/// phase BEFORE it starts is admission, not a suspension.
#[must_use]
pub fn site_suspendable(input: &PolicyInput, site: StopSite) -> bool {
    let declared = safepoint::declared_or_never(
        input
            .capabilities
            .iter()
            .find(|cap| cap.id == site.token())
            .map(|cap| cap.safe_points),
    );
    safepoint::may_suspend(declared, At::Unit(0)).allowed()
}

/// The row's O-STOP commitments. Pure and total. `r2` is the router's acting
/// commitments when the router acts on the row (`None` otherwise).
///
/// #router-r4: a rule that ends a phase early acts only when that phase's
/// capability DECLARES safe points (`blocked_by=safe_points` otherwise).
#[must_use]
pub fn commit(
    plan: &Plan,
    input: &PolicyInput,
    mode: EffectiveMode,
    switches: &OptimizerSwitches,
    r2: Option<&Commitments>,
) -> StopCommitments {
    let acting = matches!(mode, EffectiveMode::Acting);
    if !acting {
        let reason = format!("mode:{}", mode.token());
        let rules = R3_RULES
            .iter()
            .map(|rule| act(*rule, false, &reason, "optimizer not acting".to_owned()))
            .collect();
        return finish(StopCommitments {
            schema: OSTOP_SCHEMA.to_owned(),
            acting,
            rules,
            directives: StopDirectives::NONE,
            sha: String::new(),
        });
    }
    let cr1 = r2.is_some_and(|c| c.acts(CR_1));
    let upfront_governed = r2.is_some_and(|c| c.governed(LaneSlot::AttackUpfront));
    let short_row = plan.short_row || input.budget.official_ms <= SHORT_ROW_MS;
    let mut rules = Vec::with_capacity(R3_RULES.len());

    // rule.governor: the alpha ascent's relative self-decay stop.
    let governor = switches.governor && site_suspendable(input, StopSite::AlphaAscent);
    rules.push(act(
        RULE_GOVERNOR,
        governor,
        if switches.governor {
            "safe_points"
        } else {
            "key"
        },
        format!(
            "alpha ascent yields when gain < {GOVERNOR_RATIO} x own best after \
             {GOVERNOR_MIN_SAMPLES} samples; best iterate kept, certified once; freed time to \
             the tree; detector(n=1)"
        ),
    ));

    // rule.ia_yield: rides on CR-1 (the ascent reads a contract).
    let ia_yield = switches.ia_yield && cr1 && site_suspendable(input, StopSite::IntermAlpha);
    rules.push(act(
        RULE_IA_YIELD,
        ia_yield,
        if !switches.ia_yield {
            "key"
        } else if !cr1 {
            "CR-1"
        } else {
            "safe_points"
        },
        format!(
            "interm-alpha candidates yield when rate < {IA_YIELD_RATIO} x own best after \
             {IA_YIELD_MIN_SAMPLES} layers; baseline pass continues; ascent contract excludes the \
             freed interm time"
        ),
    ));

    // rule.attack_defer (CR-ATK): rides on CR-1 and the upfront slot.
    let upfront_suspendable = site_suspendable(input, StopSite::AttackUpfront);
    let attack_defer =
        switches.attack_defer && cr1 && upfront_governed && !short_row && upfront_suspendable;
    rules.push(act(
        RULE_ATTACK_DEFER,
        attack_defer,
        if !switches.attack_defer {
            "key"
        } else if !cr1 {
            "CR-1"
        } else if !upfront_governed {
            "attack_upfront_ungoverned"
        } else if short_row {
            "short_row"
        } else {
            "safe_points"
        },
        format!(
            "CR-ATK: upfront attack defers at a restart boundary when its best margin rose <= \
             noise over the trailing {ATTACK_WINDOW_FRAC} of its slice; unspent >= \
             {ATTACK_MIN_DEFER_MS}ms held as a late sat reserve (floors = today's caps)"
        ),
    ));

    // rule.root_admission.
    let root_admission = switches.root_admission;
    rules.push(act(
        RULE_ROOT_ADMISSION,
        root_admission,
        "key",
        "skip a root phase whose padded predicted cost does not fit phase_window::admit against \
         its window (the verifier cell when governed); inputs logged"
            .to_owned(),
    ));

    finish(StopCommitments {
        schema: OSTOP_SCHEMA.to_owned(),
        acting,
        rules,
        directives: StopDirectives {
            governor: governor.then_some(GOVERNOR_RULE),
            ia_yield: ia_yield.then_some(IA_YIELD_RULE),
            attack_defer: attack_defer.then_some(ATTACK_DEFER_RULE),
            root_admission,
        },
        sha: String::new(),
    })
}

/// Stamp the digest (over everything but the digest itself).
fn finish(mut commitments: StopCommitments) -> StopCommitments {
    commitments.sha = String::new();
    let mut hash = Sha256::new();
    hash.update(format!("{commitments:?}").as_bytes());
    let digest: [u8; 32] = hash.finalize().into();
    commitments.sha = super::hash::hex(&digest);
    commitments
}

/// CR-ATK's slope verdict over a lane's restart-boundary samples.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlopeVerdict {
    /// The best margin rose by at most `eps` over the whole trailing window:
    /// slope `<= eps / W`. The lane may defer.
    Flat {
        /// Its rise over the window.
        rise: f64,
        /// The noise floor it had to beat.
        eps: f64,
        /// The window, seconds.
        window_s: f64,
    },
    /// Still climbing, or already at the violation surface: keep attacking.
    Climbing,
    /// Not enough evidence (no finite sample, window not yet spanned,
    /// degenerate slice): keep attacking.
    Unknown(&'static str),
}

impl SlopeVerdict {
    /// Whether the verdict allows a defer.
    #[must_use]
    pub const fn is_flat(&self) -> bool {
        matches!(self, Self::Flat { .. })
    }
}

/// CR-ATK's pure slope predicate. `samples` are `(t, best)` pairs in
/// increasing `t` (seconds since the lane's slice started; `best` a running
/// MAX of the lane's closest-to-violation margin, so non-decreasing).
/// FLAT iff the trailing window `W = max(window_frac, 0.5) * slice` has fully
/// elapsed AND is spanned by samples, the best margin is still below the
/// violation surface, and it rose by at most `eps` (the confirmation path's
/// noise floor) across the window. The LEVEL is never judged: an unsat row
/// plateaued at its supremum can sit closer to violation than a sat row still
/// climbing, and only the rate separates them.
#[must_use]
pub fn attack_flat(
    samples: &[(f64, f64)],
    slice_s: f64,
    window_frac: f64,
    eps: f64,
) -> SlopeVerdict {
    if !(slice_s.is_finite() && slice_s > 0.0 && window_frac.is_finite()) {
        return SlopeVerdict::Unknown("degenerate_slice");
    }
    let frac = window_frac.clamp(MIN_DEFER_WINDOW_FRAC, 1.0);
    let window = frac * slice_s;
    let finite: Vec<(f64, f64)> = samples
        .iter()
        .copied()
        .filter(|(t, best)| t.is_finite() && best.is_finite())
        .collect();
    let Some(&(t_last, best_last)) = finite.last() else {
        return SlopeVerdict::Unknown("no_finite_sample");
    };
    if best_last >= 0.0 {
        // A point at the violation surface is in hand: the lane's own
        // candidate path owns it, never a stop rule.
        return SlopeVerdict::Climbing;
    }
    let start = t_last - window;
    if start.is_nan() || start < 0.0 {
        return SlopeVerdict::Unknown("window_not_elapsed");
    }
    // The running max as of the window's start: the latest sample at or
    // before it. None means no sample spans the window's start.
    let Some(&(_, best_start)) = finite.iter().rev().find(|(t, _)| *t <= start) else {
        return SlopeVerdict::Unknown("window_not_spanned");
    };
    let eps = if eps.is_finite() && eps > 0.0 {
        eps
    } else {
        0.0
    };
    let rise = best_last - best_start;
    if !rise.is_finite() {
        return SlopeVerdict::Unknown("non_finite_rise");
    }
    if rise <= eps {
        SlopeVerdict::Flat {
            rise,
            eps,
            window_s: window,
        }
    } else {
        SlopeVerdict::Climbing
    }
}
