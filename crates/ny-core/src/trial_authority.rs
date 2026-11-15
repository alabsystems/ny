// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r6 — O-TRIAL: THE TRIAL AUTHORITY (ROUTER_PROGRAM §5.1, §5.3,
//! §10 R6; layer L5 of §2.1).
//!
//! # What this is
//!
//! R3's stop authority ([`crate::stop_authority`]) lets a phase END early. R6
//! lets a fast lane TRY a different value of one of its own search / effort
//! knobs, measure it over whole windows of its own work units, and keep it or
//! revert it. Each acting trial rule owns its trial sites:
//!
//! | site | rule | knob | lane safe point |
//! |---|---|---|---|
//! | [`TrialSite::WaveWidth`] | `rule.trial_width` | `margin_row.tree_wave_batch` (the wave tree's width) | between waves |
//! | [`TrialSite::InputSplitWidth`] | `rule.trial_width` | `bab.batch_size` (the reordered input-split loop's LIVE batch; dormant without `reorder_bab`) | between batches |
//! | [`TrialSite::LowdimWidth`] | `rule.trial_width` | `lowdim.batch` (the lowdim lane's wave width, a code constant) | between waves |
//! | [`TrialSite::AlphaLr`] | `rule.trial_lr_backoff` | `margin_row.alpha_lr` (the root alpha ascent's Adam step) | between iterations |
//! | [`TrialSite::LowdimSplit`] | `rule.trial_split_restart` | `bab.branching.input_split.lowdim_split` | between waves |
//! | [`TrialSite::AttackStep`] | `rule.trial_attack_halving` | the upfront APGD lane's initial step (a code constant) | between restarts |
//!
//! This module holds the ARMING (which rules and sites the acting optimizer
//! committed, with which constants) and the LEDGER (every trial event). The
//! machines themselves are pure ([`crate::router::trial`]) and owned by the
//! lanes; the authority only answers "is this site armed, and with which
//! constants".
//!
//! # Dark by default, fail open
//!
//! Unarmed — every run without `optimizer.mode: on` — every read is `None`,
//! no lane builds a machine, and every lane uses today's value: an unarmed
//! process is byte-identical to a binary without this module. A router fault
//! disarms it (`router::guard::fail_open`); a lane that holds a machine drops
//! it at its next unit boundary (back to today's value), and a lane that
//! starts after the fault builds none. The per-row trial budget is kept here
//! ([`TrialAuthority::started`]), so a lane run more than once per row (per
//! clause, per ascent) never exceeds it.
//!
//! # Soundness
//!
//! Search and effort only, inside each site's declared range (see the pure
//! module). Nothing here touches a precision, a tolerance, an error term, an
//! acceptance gate or the publication of a result.
//!
//! # Never panics
//!
//! Every lock tolerates poison and every ledger is bounded ([`MAX_EVENTS`]).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use serde::{Deserialize, Serialize};

use crate::router::trial::{
    HalvingRule, LrRule, RestartRule, TrialDecision, TrialEvent, WidthRule,
};

/// Events one row records (counted past it, not stored).
pub const MAX_EVENTS: usize = 128;

/// A knob a trial rule may move, at its lane's own safe point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialSite {
    /// The margin-row wave tree's width (`rule.trial_width`).
    WaveWidth,
    /// The reordered input-split loop's batch (`rule.trial_width`).
    InputSplitWidth,
    /// The lowdim lane's wave width (`rule.trial_width`).
    LowdimWidth,
    /// The margin-row root alpha ascent's step (`rule.trial_lr_backoff`).
    AlphaLr,
    /// The lowdim lane's split heuristic (`rule.trial_split_restart`).
    LowdimSplit,
    /// The upfront APGD attack's initial step (`rule.trial_attack_halving`).
    AttackStep,
}

impl TrialSite {
    /// Every site, in rule order.
    pub const ALL: [Self; 6] = [
        Self::WaveWidth,
        Self::InputSplitWidth,
        Self::LowdimWidth,
        Self::AlphaLr,
        Self::LowdimSplit,
        Self::AttackStep,
    ];

    /// The capability the knob belongs to (registry ids).
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::WaveWidth => "wave-beta-bab",
            Self::InputSplitWidth => "input_split",
            Self::LowdimWidth | Self::LowdimSplit => "lowdim",
            Self::AlphaLr => "margin-row.alpha",
            Self::AttackStep => "attack.upfront_apgd",
        }
    }

    /// The knob (its typed key where one exists).
    #[must_use]
    pub const fn knob(self) -> &'static str {
        match self {
            Self::WaveWidth => "margin_row.tree_wave_batch",
            Self::InputSplitWidth => "bab.batch_size",
            Self::LowdimWidth => "lowdim.batch",
            Self::AlphaLr => "margin_row.alpha_lr",
            Self::LowdimSplit => "bab.branching.input_split.lowdim_split",
            Self::AttackStep => "attack.upfront_apgd.step0",
        }
    }

    /// Every path a category file may PIN the knob at (a pinned knob is
    /// frozen: the site is not armed). Empty for a code constant.
    #[must_use]
    pub const fn pin_paths(self) -> &'static [&'static str] {
        match self {
            Self::WaveWidth => &["margin_row.tree_wave_batch"],
            Self::InputSplitWidth => &["bab.batch_size", "solver.batch_size"],
            Self::AlphaLr => &["margin_row.alpha_lr"],
            Self::LowdimSplit => &["bab.branching.input_split.lowdim_split"],
            Self::LowdimWidth | Self::AttackStep => &[],
        }
    }

    /// The rule that owns the site (registry ids).
    #[must_use]
    pub const fn rule(self) -> &'static str {
        match self {
            Self::WaveWidth | Self::InputSplitWidth | Self::LowdimWidth => "rule.trial_width",
            Self::AlphaLr => "rule.trial_lr_backoff",
            Self::LowdimSplit => "rule.trial_split_restart",
            Self::AttackStep => "rule.trial_attack_halving",
        }
    }

    /// The site's bit in [`TrialDirectives::sites`].
    #[must_use]
    pub const fn bit(self) -> u32 {
        1u32 << (self as u32)
    }

    /// A stable token naming the site (`<capability>/<knob>`).
    #[must_use]
    pub fn label(self) -> String {
        format!("{}/{}", self.token(), self.knob())
    }
}

/// What the acting optimizer committed for one row (`router::otrial::commit`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrialDirectives {
    /// `rule.trial_width`.
    pub width: Option<WidthRule>,
    /// `rule.trial_lr_backoff`.
    pub lr_backoff: Option<LrRule>,
    /// `rule.trial_split_restart`.
    pub split_restart: Option<RestartRule>,
    /// `rule.trial_attack_halving`.
    pub attack_halving: Option<HalvingRule>,
    /// The armed sites ([`TrialSite::bit`]).
    pub sites: u32,
}

impl TrialDirectives {
    /// Nothing armed.
    pub const NONE: Self = Self {
        width: None,
        lr_backoff: None,
        split_restart: None,
        attack_halving: None,
        sites: 0,
    };

    /// Whether any site is armed.
    #[must_use]
    pub fn any(&self) -> bool {
        self.sites != 0
    }

    /// Whether `site` is armed AND its rule's constants are present.
    #[must_use]
    pub fn site_armed(&self, site: TrialSite) -> bool {
        if self.sites & site.bit() == 0 {
            return false;
        }
        match site {
            TrialSite::WaveWidth | TrialSite::InputSplitWidth | TrialSite::LowdimWidth => {
                self.width.is_some()
            }
            TrialSite::AlphaLr => self.lr_backoff.is_some(),
            TrialSite::LowdimSplit => self.split_restart.is_some(),
            TrialSite::AttackStep => self.attack_halving.is_some(),
        }
    }

    /// The armed sites, in order.
    #[must_use]
    pub fn armed_sites(&self) -> Vec<TrialSite> {
        TrialSite::ALL
            .into_iter()
            .filter(|site| self.site_armed(*site))
            .collect()
    }
}

/// Why directives were refused (the authority then stays unarmed: today).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialArmRefusal {
    /// A rule's constants are outside its protocol's domain.
    RuleOutOfRange,
    /// A site is armed without its rule's constants.
    SiteWithoutRule,
}

/// Counts of every recorded event, by decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrialCounts {
    /// `started`.
    pub started: u64,
    /// `kept`.
    pub kept: u64,
    /// `reverted`.
    pub reverted: u64,
    /// `tabu`.
    pub tabu: u64,
    /// `frozen`.
    pub frozen: u64,
    /// `guard`.
    pub guard: u64,
    /// `blocked`.
    pub blocked: u64,
}

impl TrialCounts {
    fn count(&mut self, decision: TrialDecision) {
        let slot = match decision {
            TrialDecision::Started => &mut self.started,
            TrialDecision::Kept => &mut self.kept,
            TrialDecision::Reverted => &mut self.reverted,
            TrialDecision::Tabu => &mut self.tabu,
            TrialDecision::Frozen => &mut self.frozen,
            TrialDecision::Guard => &mut self.guard,
            TrialDecision::Blocked => &mut self.blocked,
        };
        *slot = slot.saturating_add(1);
    }
}

/// The authority, as recorded in the flight `router` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialSnapshot {
    /// Whether it was armed when the snapshot was taken.
    pub armed: bool,
    /// The directives it held.
    pub directives: Option<TrialDirectives>,
    /// Every event, in order (bounded).
    pub events: Vec<TrialEvent>,
    /// Every event counted by decision (including dropped ones).
    pub counts: TrialCounts,
    /// Events past [`MAX_EVENTS`] (counted, not stored).
    pub dropped: u64,
}

#[derive(Debug, Default)]
struct Ledger {
    directives: Option<TrialDirectives>,
    events: Vec<TrialEvent>,
    counts: TrialCounts,
    /// Trials STARTED per site (in [`TrialSite::ALL`] order), counted even
    /// past [`MAX_EVENTS`]: the per-ROW trial budget.
    started: [u32; 6],
    dropped: u64,
}

/// The trial authority. One per process ([`global`]); instantiable for tests.
#[derive(Debug)]
pub struct TrialAuthority {
    armed: AtomicBool,
    ledger: Mutex<Ledger>,
}

impl Default for TrialAuthority {
    fn default() -> Self {
        Self::new()
    }
}

impl TrialAuthority {
    /// An unarmed authority.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
            ledger: Mutex::new(Ledger {
                directives: None,
                events: Vec::new(),
                counts: TrialCounts {
                    started: 0,
                    kept: 0,
                    reverted: 0,
                    tabu: 0,
                    frozen: 0,
                    guard: 0,
                    blocked: 0,
                },
                started: [0; 6],
                dropped: 0,
            }),
        }
    }

    fn ledger(&self) -> std::sync::MutexGuard<'_, Ledger> {
        self.ledger.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Arm the row's directives. Clears the ledger first. Refuses (and stays
    /// unarmed) when a rule's constants are outside their domain or a site is
    /// armed without its rule; empty directives arm nothing.
    pub fn arm(&self, directives: TrialDirectives) -> Result<(), TrialArmRefusal> {
        if directives.width.is_some_and(|r| !r.is_valid())
            || directives.lr_backoff.is_some_and(|r| !r.is_valid())
            || directives.split_restart.is_some_and(|r| !r.is_valid())
            || directives.attack_halving.is_some_and(|r| !r.is_valid())
        {
            return Err(TrialArmRefusal::RuleOutOfRange);
        }
        if TrialSite::ALL
            .into_iter()
            .any(|site| directives.sites & site.bit() != 0 && !directives.site_armed(site))
        {
            return Err(TrialArmRefusal::SiteWithoutRule);
        }
        self.disarm();
        {
            let mut ledger = self.ledger();
            *ledger = Ledger {
                directives: Some(directives),
                ..Ledger::default()
            };
        }
        if directives.any() {
            self.armed.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Disarm: every read answers `None` from here on. The ledger is kept for
    /// the snapshot.
    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Release);
    }

    /// Whether any site is armed.
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// The armed directives (`None` unarmed).
    #[must_use]
    pub fn directives(&self) -> Option<TrialDirectives> {
        if !self.is_armed() {
            return None;
        }
        self.ledger().directives
    }

    /// `rule.trial_width`'s constants at a width site (`None` unless that
    /// site is armed).
    #[must_use]
    pub fn width(&self, site: TrialSite) -> Option<WidthRule> {
        if !matches!(
            site,
            TrialSite::WaveWidth | TrialSite::InputSplitWidth | TrialSite::LowdimWidth
        ) {
            return None;
        }
        self.directives()
            .filter(|d| d.site_armed(site))
            .and_then(|d| d.width)
    }

    /// `rule.trial_lr_backoff`'s constants (`None` unless armed).
    #[must_use]
    pub fn lr_backoff(&self) -> Option<LrRule> {
        self.directives()
            .filter(|d| d.site_armed(TrialSite::AlphaLr))
            .and_then(|d| d.lr_backoff)
    }

    /// `rule.trial_split_restart`'s constants (`None` unless armed).
    #[must_use]
    pub fn split_restart(&self) -> Option<RestartRule> {
        self.directives()
            .filter(|d| d.site_armed(TrialSite::LowdimSplit))
            .and_then(|d| d.split_restart)
    }

    /// `rule.trial_attack_halving`'s constants (`None` unless armed).
    #[must_use]
    pub fn attack_halving(&self) -> Option<HalvingRule> {
        self.directives()
            .filter(|d| d.site_armed(TrialSite::AttackStep))
            .and_then(|d| d.attack_halving)
    }

    /// Whether `site` is armed.
    #[must_use]
    pub fn site_armed(&self, site: TrialSite) -> bool {
        self.directives().is_some_and(|d| d.site_armed(site))
    }

    /// Record one event. A no-op unless armed.
    pub fn record(&self, event: &TrialEvent) {
        if !self.is_armed() {
            return;
        }
        let event = event.clone().sanitized();
        let mut ledger = self.ledger();
        ledger.counts.count(event.decision);
        if event.decision == TrialDecision::Started {
            if let Some(i) = TrialSite::ALL
                .iter()
                .position(|s| s.token() == event.site && s.knob() == event.knob)
            {
                ledger.started[i] = ledger.started[i].saturating_add(1);
            }
        }
        if ledger.events.len() < MAX_EVENTS {
            ledger.events.push(event);
        } else {
            ledger.dropped = ledger.dropped.saturating_add(1);
        }
    }

    /// Trials already STARTED at `site` on this row (whatever lane run
    /// started them): a new machine at the site starts with this many spent,
    /// so the budget is per row, not per call.
    #[must_use]
    pub fn started(&self, site: TrialSite) -> u32 {
        let i = TrialSite::ALL.iter().position(|s| *s == site).unwrap_or(0);
        self.ledger().started[i]
    }

    /// Record and print each event (`[router] trial knob= dir= decision=`).
    /// A no-op unless armed (an unarmed lane holds no machine anyway).
    pub fn emit(&self, events: Vec<TrialEvent>) {
        if !self.is_armed() {
            return;
        }
        for event in events {
            eprintln!("{}", event.line());
            self.record(&event);
        }
    }

    /// The authority as recorded (whether or not armed now).
    #[must_use]
    pub fn snapshot(&self) -> TrialSnapshot {
        let ledger = self.ledger();
        TrialSnapshot {
            armed: self.is_armed(),
            directives: ledger.directives,
            events: ledger.events.clone(),
            counts: ledger.counts,
            dropped: ledger.dropped,
        }
    }
}

static GLOBAL: TrialAuthority = TrialAuthority::new();

/// The process's one trial authority. `ny vnncomp` runs one row per process.
#[must_use]
pub fn global() -> &'static TrialAuthority {
    &GLOBAL
}
