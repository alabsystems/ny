// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PURE checkpoints (ROUTER_PROGRAM §2.2 CHECKPOINTS, §4.2 CR-F, §4.3).
//!
//! At most three fixed checkpoints: c1 = lane warm-up (OBSERVATION ONLY — the
//! winners read 0.364–0.776 there, so nothing is classified), c2 = 50% and
//! c3 = 75% of the prover window. At c2/c3 a capability classified FAR from
//! its two trailing windows hands its unspent contract to a SEALED recipient
//! when it has safe points; otherwise nothing happens. Never kill; never
//! shorten a `SafePoints::None` lane.
//!
//! There is deliberately NO promotion amendment (§1.1: NEAR is not
//! observable); a source ratchet keeps it that way.
//!
//! In R1 the checkpoints run RETROSPECTIVELY, strictly after the verdict was
//! published, on the drained progress bus: they compute what CR-F would have
//! done at c2/c3 and change nothing.

use serde::{Deserialize, Serialize};

use super::bus::ProgressSample;
use super::decl::ProgressKind;
use super::plan::{Plan, PlannedCap};
use super::project::{self, FarClass};
use super::safepoint;

/// Amendments one checkpoint may return.
pub const MAX_AMENDMENTS_PER_CHECKPOINT: usize = 8;
/// Amendments one row may accumulate.
pub const MAX_AMENDMENTS_TOTAL: usize = 24;

/// The three checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckpointId {
    /// Lane warm-up: observe only.
    C1,
    /// 50% of the prover window.
    C2,
    /// 75% of the prover window.
    C3,
}

impl CheckpointId {
    /// All three, in order.
    pub const ALL: [Self; 3] = [Self::C1, Self::C2, Self::C3];

    /// Stable token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::C1 => "c1",
            Self::C2 => "c2",
            Self::C3 => "c3",
        }
    }

    /// Parse a token.
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "c1" => Some(Self::C1),
            "c2" => Some(Self::C2),
            "c3" => Some(Self::C3),
            _ => None,
        }
    }
}

/// What a checkpoint may change. Plain data; there is no promotion variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Amendment {
    /// Hand `ms` of `from`'s unspent contract to the sealed recipient `to`.
    HandOn {
        /// The FAR lane.
        from: String,
        /// The sealed recipient.
        to: String,
        /// Milliseconds handed on.
        ms: u64,
    },
    /// Defer the unspent part of `from`'s cap to a late reserve.
    Defer {
        /// The lane.
        from: String,
        /// Milliseconds deferred.
        ms: u64,
    },
    /// Arm the publish grace for a completed certificate inside the join margin.
    ArmPublishGrace {
        /// The lane.
        cap: String,
    },
    /// Derive a cap from the deadline and the lease.
    DeriveCap {
        /// The lane.
        cap: String,
        /// Which cap.
        what: String,
        /// The derived value.
        value: u64,
    },
}

/// One capability's observation at a checkpoint (logged; never a decision on
/// its own).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapObservation {
    /// The capability.
    pub cap: String,
    /// Its progress kind.
    pub kind: ProgressKind,
    /// Closes per unit in the older trailing window.
    pub w1: Option<f64>,
    /// Closes per unit in the newer trailing window.
    pub w2: Option<f64>,
    /// Open frontier at the newest sample.
    pub open: Option<u64>,
    /// Samples read.
    pub samples: usize,
    /// The classification.
    pub class: FarClass,
}

/// When `cap`'s contract ends on the plan's sequential schedule (the priced
/// contracts before it, in order, plus its own), in milliseconds since the row
/// started. `None` when it or any lane before it is unpriced.
#[must_use]
pub fn contract_end_ms(plan: &Plan, cap: &str) -> Option<u64> {
    let mut end = 0u64;
    for contract in &plan.contracts {
        end = end.saturating_add(contract.ms?);
        if contract.cap == cap {
            return Some(end);
        }
    }
    None
}

fn checkpoint_secs(plan: &Plan, at: CheckpointId) -> Option<f64> {
    plan.checkpoint_at_ms(at).map(|ms| ms as f64 / 1000.0)
}

fn trailing_for(cap: &PlannedCap, samples: &[ProgressSample], t_secs: f64) -> Vec<ProgressSample> {
    project::trailing(
        samples,
        &cap.id,
        t_secs,
        project::window_units(cap.progress),
    )
}

/// The samples a checkpoint reads: the trailing samples of every capability
/// in view. Logging exactly these makes the checkpoint replayable (the
/// selection is idempotent).
#[must_use]
pub fn select_samples(
    plan: &Plan,
    samples: &[ProgressSample],
    at: CheckpointId,
) -> Vec<ProgressSample> {
    let Some(t) = checkpoint_secs(plan, at) else {
        return Vec::new();
    };
    plan.caps
        .iter()
        .flat_map(|cap| trailing_for(cap, samples, t))
        .collect()
}

/// Every in-view capability's classification at `at` (c1 included: it is
/// logged, never acted on).
#[must_use]
pub fn observe(plan: &Plan, samples: &[ProgressSample], at: CheckpointId) -> Vec<CapObservation> {
    let Some(t) = checkpoint_secs(plan, at) else {
        return Vec::new();
    };
    plan.caps
        .iter()
        .filter_map(|cap| {
            let trailing = trailing_for(cap, samples, t);
            if trailing.is_empty() {
                return None;
            }
            let rates = |a: Option<&ProgressSample>, b: Option<&ProgressSample>| {
                a.zip(b).and_then(|(a, b)| {
                    project::window_rates(&project::window(a, b)).closes_per_unit
                })
            };
            let n = trailing.len();
            let (w1, w2) = if n >= 3 {
                (
                    rates(trailing.get(n - 3), trailing.get(n - 2)),
                    rates(trailing.get(n - 2), trailing.get(n - 1)),
                )
            } else if n == 2 {
                (None, rates(trailing.first(), trailing.get(1)))
            } else {
                (None, None)
            };
            Some(CapObservation {
                cap: cap.id.clone(),
                kind: cap.progress,
                w1,
                w2,
                open: trailing.last().map(|s| s.open),
                samples: n,
                class: project::classify(cap.progress, &trailing),
            })
        })
        .collect()
}

/// #router-r4 CR-F: why a capability cannot hand its contract on (pure).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FarBlock {
    /// The capability is not in the plan's view (a rule, or unregistered).
    Unplanned,
    /// It declares no safe point: the router never suspends it (§6.4).
    NoSafePoints,
    /// It does not accept being a hand-on source.
    NotHandOnSource,
    /// Its own predicate declined the row, or its first rung did not fit.
    Declined,
    /// Its progress kind is never classified FAR (opaque, dual, yield, screen).
    Unclassifiable,
    /// An attack's flat slope DEFERS to the late sat reserve (CR-ATK); its
    /// time is never handed to a prover ("defer the attack, never cut it").
    AttackDefers,
    /// No STRICTLY sealed (S7 + S8), admitted prover after it accepts a
    /// hand-on. In R4 no registry entry is `Sealed`, so this is every row.
    NoSealedRecipient,
    /// c1 is observation only (the winners read 0.364-0.776 there).
    ObserveOnly,
    /// CR-S: a short row takes no checkpoint action.
    ShortRow,
    /// The checkpoint is not planned.
    UnplannedCheckpoint,
}

impl FarBlock {
    /// Stable token for logs and records.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Unplanned => "unplanned",
            Self::NoSafePoints => "no_safe_points",
            Self::NotHandOnSource => "not_hand_on_source",
            Self::Declined => "declined",
            Self::Unclassifiable => "unclassifiable",
            Self::AttackDefers => "attack_defers",
            Self::NoSealedRecipient => "no_sealed_recipient",
            Self::ObserveOnly => "c1_observe_only",
            Self::ShortRow => "short_row",
            Self::UnplannedCheckpoint => "checkpoint_unplanned",
        }
    }
}

/// Whether the plan declined `cap` outright (its own predicate said no, or
/// its first rung did not fit). An UNKNOWN admission is not a decline: the
/// lane re-checks its own predicate at its site, and a lane asking at a safe
/// point is running, i.e. admitted itself.
fn declined(plan: &Plan, cap: &str) -> bool {
    plan.declined
        .iter()
        .any(|(id, why)| id == cap && !why.starts_with("unknown:"))
}

/// #router-r4 CR-F: the sealed recipient `from` would hand its contract to,
/// or why it cannot (pure, total). A source must be in view, declare safe
/// points (`safepoint::may_suspend` at a unit boundary), accept being a
/// hand-on source, not be declined, and report a progress kind the FAR rule
/// classifies — never an attack (CR-ATK defers it instead). The recipient is
/// the first capability AFTER it in the plan's structural (registry) order
/// whose ledger is STRICTLY `Sealed`, whose own predicate admitted the row,
/// which is a prover and which accepts a hand-on.
pub fn far_recipient<'p>(plan: &'p Plan, from: &str) -> Result<&'p PlannedCap, FarBlock> {
    let Some(index) = plan.caps.iter().position(|c| c.id == from) else {
        return Err(FarBlock::Unplanned);
    };
    let Some(source) = plan.caps.get(index) else {
        return Err(FarBlock::Unplanned);
    };
    if !safepoint::may_suspend(source.safe_points, safepoint::At::Unit(0)).allowed() {
        return Err(FarBlock::NoSafePoints);
    }
    if !source.accepts.hand_on_source {
        return Err(FarBlock::NotHandOnSource);
    }
    if declined(plan, from) {
        return Err(FarBlock::Declined);
    }
    match source.progress {
        ProgressKind::Margin => return Err(FarBlock::AttackDefers),
        ProgressKind::Opaque | ProgressKind::Dual | ProgressKind::Yield | ProgressKind::Screen => {
            return Err(FarBlock::Unclassifiable)
        }
        ProgressKind::FrontierBinary
        | ProgressKind::FrontierKary
        | ProgressKind::Bfs
        | ProgressKind::Volume => {}
    }
    plan.caps
        .iter()
        .skip(index + 1)
        .find(|c| c.sealed_s7s8 && c.admitted && c.prover && c.accepts.hand_on_sink)
        .ok_or(FarBlock::NoSealedRecipient)
}

/// #router-r4 CR-F: one capability's decision at one checkpoint (pure).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FarDecision {
    /// The capability.
    pub cap: String,
    /// The checkpoint.
    pub at: CheckpointId,
    /// Its FAR classification from its trailing windows (work units).
    pub class: FarClass,
    /// The sealed recipient, when one exists.
    pub to: Option<String>,
    /// Whether the hand-on is due (FAR, a recipient, an acting checkpoint).
    pub act: bool,
    /// Why not, when it is not.
    pub blocked_by: Option<FarBlock>,
}

/// #router-r4 CR-F: classify `cap` at checkpoint `at` from its trailing
/// samples and decide whether its unspent contract is handed on. The same
/// function serves the retrospective checkpoints (shadow) and the live
/// authority at the lane's safe point (acting), so replay re-derives both.
#[must_use]
pub fn far_decision(
    plan: &Plan,
    samples: &[ProgressSample],
    cap: &str,
    at: CheckpointId,
) -> FarDecision {
    let route = far_recipient(plan, cap);
    let view = plan.caps.iter().find(|c| c.id == cap);
    let class = match (view, checkpoint_secs(plan, at)) {
        (Some(view), Some(t)) => project::classify(view.progress, &trailing_for(view, samples, t)),
        _ => FarClass::Unknown,
    };
    let to = route.as_ref().ok().map(|c| c.id.clone());
    let blocked_by = if checkpoint_secs(plan, at).is_none() {
        Some(FarBlock::UnplannedCheckpoint)
    } else if at == CheckpointId::C1 {
        Some(FarBlock::ObserveOnly)
    } else if plan.short_row {
        Some(FarBlock::ShortRow)
    } else {
        route.err()
    };
    FarDecision {
        cap: cap.to_owned(),
        at,
        class,
        act: blocked_by.is_none() && class == FarClass::Far,
        to,
        blocked_by,
    }
}

/// The amendments checkpoint `at` makes. c1 returns nothing BY CONSTRUCTION;
/// short rows (CR-S) return nothing; a FAR capability hands its unspent
/// contract on only through [`far_decision`]: when it has safe points, may be
/// a hand-on source, and a STRICTLY sealed (#router-r4), admitted prover
/// recipient that accepts a hand-on exists after it in the plan's structural
/// order. Bounded by [`MAX_AMENDMENTS_PER_CHECKPOINT`].
#[must_use]
pub fn checkpoint(plan: &Plan, samples: &[ProgressSample], at: CheckpointId) -> Vec<Amendment> {
    if at == CheckpointId::C1 || plan.short_row {
        return Vec::new();
    }
    let Some(at_ms) = plan.checkpoint_at_ms(at) else {
        return Vec::new();
    };
    let observations = observe(plan, samples, at);
    let mut out = Vec::new();
    for obs in observations {
        if out.len() >= MAX_AMENDMENTS_PER_CHECKPOINT {
            break;
        }
        if obs.class != FarClass::Far {
            continue;
        }
        let decision = far_decision(plan, samples, &obs.cap, at);
        let (true, Some(to)) = (decision.act, decision.to) else {
            continue;
        };
        // The unspent part of a PRICED contract on the plan's sequential
        // schedule; an unpriced lane (today's cap) has no contract to hand on
        // in the plan's view (the live authority reads its deadline cell).
        let unspent = contract_end_ms(plan, &obs.cap).map(|end| end.saturating_sub(at_ms));
        if let Some(ms) = unspent.filter(|ms| *ms > 0) {
            out.push(Amendment::HandOn {
                from: obs.cap,
                to,
                ms,
            });
        }
    }
    out
}

/// The row's append-only amendment log: checkpoints may only add entries, in
/// checkpoint order, up to [`MAX_AMENDMENTS_TOTAL`]; nothing is ever rewritten.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmendmentLog {
    entries: Vec<(CheckpointId, Amendment)>,
    dropped: u64,
}

impl AmendmentLog {
    /// An empty log.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            dropped: 0,
        }
    }

    /// Append checkpoint `at`'s amendments. A checkpoint earlier than the last
    /// one appended, or entries past the caps, are dropped and counted, never
    /// reordered. Returns how many were appended.
    pub fn append(&mut self, at: CheckpointId, amendments: Vec<Amendment>) -> usize {
        let out_of_order = self.entries.last().is_some_and(|(last, _)| at < *last);
        let mut appended = 0usize;
        for (index, amendment) in amendments.into_iter().enumerate() {
            if out_of_order
                || index >= MAX_AMENDMENTS_PER_CHECKPOINT
                || self.entries.len() >= MAX_AMENDMENTS_TOTAL
            {
                self.dropped = self.dropped.saturating_add(1);
                continue;
            }
            self.entries.push((at, amendment));
            appended += 1;
        }
        appended
    }

    /// Every appended entry, in order.
    #[must_use]
    pub fn entries(&self) -> &[(CheckpointId, Amendment)] {
        &self.entries
    }

    /// Entries refused (out of order or over a cap).
    #[must_use]
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }
}
