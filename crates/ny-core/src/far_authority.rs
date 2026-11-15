// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r4 CR-F — THE FAR HAND-ON AUTHORITY (ROUTER_PROGRAM §2.2
//! CHECKPOINTS, §4.2 CR-F, §4.3, §6.4).
//!
//! # What this is
//!
//! R1 evaluated c2/c3 RETROSPECTIVELY, after publication, on the drained
//! progress bus. CR-F acts LIVE: a lane that declares safe points calls
//! [`FarAuthority::at_safe_point`] at each of its own unit boundaries with its
//! own cumulative counters. The authority keeps a short per-capability window
//! of those samples and, the first time the lane reaches a safe point after
//! c2 (and again after c3), runs the PURE decision
//! `router::checkpoint::far_decision` — the same function the retrospective
//! checkpoints run, on the same trailing-window selection — so a live
//! decision replays bit for bit from the samples the ledger records.
//!
//! When the decision acts (two trailing windows FAR in work units, a STRICTLY
//! sealed recipient, an acting checkpoint) AND the commitments carry a route
//! for the lane, the authority hands the lane's unspent deadline cell to the
//! recipient's cell (`deadline_authority::Authority::release_to`:
//! `credited <= unspent`, never past the recipient's cap) and answers
//! [`FarAction::HandOn`]; the lane then stops at this safe point and reports
//! undecided. Every other answer is [`FarAction::Continue`].
//!
//! # Safe points are enforced here
//!
//! The authority only ever acts at [`At::Unit`] (the lane's own boundary),
//! and only for a capability whose plan view declares `BetweenUnits`
//! (`router::safepoint::may_suspend`); any other request is refused and
//! counted. A `SafePoints::None` lane is never suspended.
//!
//! # Observe mode
//!
//! With no route (every R4 row: no registry entry is `Sealed`), CR-F arms in
//! OBSERVE mode: the authority still classifies at c2/c3 and records the
//! decision (`act=0 blocked_by=no_sealed_recipient`), which is the live
//! evidence that the mechanism engaged, and it never hands anything on.
//!
//! # Dark by default, fail open, never panics
//!
//! Unarmed — every run without `router.mode: on` — a call is one atomic load
//! and `Continue`. A router fault disarms it (`router::guard::fail_open`).
//! One poison-tolerant lock, bounded rings and ledger, saturating arithmetic.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::deadline_authority::{self, Authority, LaneSlot};
use crate::router::bus::{ProgressSample, SampleRecord};
use crate::router::checkpoint::{far_decision, CheckpointId, FarBlock, FarDecision};
use crate::router::decl::CapId;
use crate::router::plan::Plan;
use crate::router::project::FarClass;
use crate::router::r4::FarCommit;
use crate::router::safepoint::{self, At, Ruling};

/// Samples kept per capability (the trailing selection needs at most three,
/// spaced by the kind's window; the ring keeps far more than enough).
pub const RING: usize = 256;

/// Capabilities one row tracks.
pub const MAX_CAPS: usize = 16;

/// Decisions one row records (counted past it, not stored).
pub const MAX_EVENTS: usize = 64;

/// What the lane does at this safe point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FarAction {
    /// Keep going.
    Continue,
    /// Stop here: the unspent contract was handed to `to`.
    HandOn {
        /// The recipient capability.
        to: String,
        /// Unspent cell handed on, milliseconds.
        unspent_ms: u64,
        /// What the recipient was credited, milliseconds.
        credited_ms: u64,
    },
}

impl FarAction {
    /// Whether the lane must stop.
    #[must_use]
    pub const fn stops(&self) -> bool {
        matches!(self, Self::HandOn { .. })
    }
}

/// One live decision, as recorded (replayable from `samples`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FarEvent {
    /// The pure decision.
    pub decision: FarDecision,
    /// When the lane reached the safe point, ms since the row started.
    pub at_ms: u64,
    /// The unit boundary it reached.
    pub unit: u64,
    /// The samples the decision read (the capability's ring at that moment,
    /// up to the checkpoint instant).
    pub samples: Vec<SampleRecord>,
    /// Whether a hand-on was executed through the cells.
    pub handed: bool,
    /// Why an acting decision moved no time (no route, cells refused).
    pub not_handed: Option<String>,
    /// Unspent cell handed on, milliseconds.
    pub unspent_ms: u64,
    /// Credited to the recipient, milliseconds.
    pub credited_ms: u64,
}

/// One refused suspension request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafePointRefusal {
    /// The capability.
    pub cap: String,
    /// Where it was asked.
    pub at: At,
    /// Why.
    pub reason: String,
}

/// The authority, as recorded in the flight `router` record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FarSnapshot {
    /// Whether it was armed when the snapshot was taken.
    pub armed: bool,
    /// What it was armed with.
    pub commit: Option<FarCommit>,
    /// Safe-point calls received.
    pub calls: u64,
    /// Every checkpoint decision, in order (bounded).
    pub events: Vec<FarEvent>,
    /// Suspension requests refused at a non-safe point.
    pub refusals: Vec<SafePointRefusal>,
    /// Hand-ons executed.
    pub handed: u64,
    /// Decisions / refusals past [`MAX_EVENTS`].
    pub dropped: u64,
}

#[derive(Debug, Default)]
struct CapState {
    cap: String,
    ring: VecDeque<ProgressSample>,
    evaluated: Vec<CheckpointId>,
    handed: Option<FarAction>,
}

#[derive(Debug)]
struct State {
    plan: Option<Plan>,
    commit: Option<FarCommit>,
    anchor: Option<Instant>,
    caps: Vec<CapState>,
    calls: u64,
    events: Vec<FarEvent>,
    refusals: Vec<SafePointRefusal>,
    handed: u64,
    dropped: u64,
}

impl State {
    const EMPTY: Self = Self {
        plan: None,
        commit: None,
        anchor: None,
        caps: Vec::new(),
        calls: 0,
        events: Vec::new(),
        refusals: Vec::new(),
        handed: 0,
        dropped: 0,
    };
}

/// The FAR hand-on authority. One per process ([`global`]); instantiable for
/// tests.
#[derive(Debug)]
pub struct FarAuthority {
    armed: AtomicBool,
    state: Mutex<State>,
}

impl Default for FarAuthority {
    fn default() -> Self {
        Self::new()
    }
}

fn ms_of(secs: f64) -> u64 {
    if secs.is_finite() && secs > 0.0 {
        // Saturating float-to-int cast: never panics.
        (secs * 1000.0) as u64
    } else {
        0
    }
}

impl FarAuthority {
    /// An unarmed authority.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
            state: Mutex::new(State::EMPTY),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Arm the row's plan and CR-F commitment, anchored at the row's start
    /// (the same anchor the plan's checkpoint instants and the progress bus
    /// use). Clears every ring and the ledger.
    pub fn arm(&self, plan: Plan, commit: FarCommit, anchor: Instant) {
        self.disarm();
        *self.state() = State {
            plan: Some(plan),
            commit: Some(commit),
            anchor: Some(anchor),
            ..State::EMPTY
        };
        self.armed.store(true, Ordering::Release);
    }

    /// Disarm: every call answers `Continue` from here on. The ledger is kept
    /// for the snapshot.
    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Release);
    }

    /// Whether armed. One atomic load: the cost of a dark tap.
    #[inline]
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// The lane `sample.cap` reached its unit boundary `unit`: feed its
    /// counters and decide. Reads the wall clock against the armed anchor and
    /// moves time through the process's deadline authority. `Continue` when
    /// unarmed.
    pub fn at_safe_point(&self, sample: ProgressSample, unit: u64) -> FarAction {
        if !self.is_armed() {
            return FarAction::Continue;
        }
        let Some(anchor) = self.state().anchor else {
            return FarAction::Continue;
        };
        let now = Instant::now();
        let t = now.saturating_duration_since(anchor).as_secs_f64();
        self.decide_at(sample, At::Unit(unit), t, deadline_authority::global(), now)
    }

    /// The testable core of [`FarAuthority::at_safe_point`]: `at` is where
    /// the lane is, `t_secs` the row clock, `cells` the deadline authority a
    /// hand-on moves time through, `now` its instant.
    pub fn decide_at(
        &self,
        sample: ProgressSample,
        at: At,
        t_secs: f64,
        cells: &Authority,
        now: Instant,
    ) -> FarAction {
        if !self.is_armed() {
            return FarAction::Continue;
        }
        let mut state = self.state();
        state.calls = state.calls.saturating_add(1);
        let Some(plan) = state.plan.clone() else {
            return FarAction::Continue;
        };
        let cap_id = sample.cap.as_str().to_owned();
        // Safe points are enforced: only at a unit boundary of a capability
        // whose plan view declares them.
        let declared = safepoint::declared_or_never(
            plan.caps
                .iter()
                .find(|c| c.id == cap_id)
                .map(|c| c.safe_points),
        );
        if let Ruling::Refused(why) = safepoint::may_suspend(declared, at) {
            push_refusal(
                &mut state,
                SafePointRefusal {
                    cap: cap_id,
                    at,
                    reason: why.to_owned(),
                },
            );
            return FarAction::Continue;
        }
        let stamped = sample.at_secs(t_secs);
        let index = match state.caps.iter().position(|c| c.cap == cap_id) {
            Some(index) => index,
            None if state.caps.len() < MAX_CAPS => {
                state.caps.push(CapState {
                    cap: cap_id.clone(),
                    ..CapState::default()
                });
                state.caps.len() - 1
            }
            None => {
                state.dropped = state.dropped.saturating_add(1);
                return FarAction::Continue;
            }
        };
        let Some(cap) = state.caps.get_mut(index) else {
            return FarAction::Continue;
        };
        // A lane that handed its contract on stays stopped for the row.
        if let Some(done) = cap.handed.clone() {
            return done;
        }
        if cap.ring.len() >= RING {
            cap.ring.pop_front();
        }
        cap.ring.push_back(stamped);
        let due: Vec<CheckpointId> = [CheckpointId::C2, CheckpointId::C3]
            .into_iter()
            .filter(|id| !cap.evaluated.contains(id))
            .filter(|id| {
                plan.checkpoint_at_ms(*id)
                    .is_some_and(|at_ms| ms_of(t_secs) >= at_ms)
            })
            .collect();
        if due.is_empty() {
            return FarAction::Continue;
        }
        cap.evaluated.extend(due.iter().copied());
        let ring: Vec<ProgressSample> = cap.ring.iter().copied().collect();
        let commit = state.commit.clone();
        let mut outcome = FarAction::Continue;
        for id in due {
            let decision = far_decision(&plan, &ring, &cap_id, id);
            let cutoff = plan.checkpoint_at_ms(id).unwrap_or(0);
            let samples: Vec<SampleRecord> = ring
                .iter()
                .filter(|s| ms_of(s.secs()) <= cutoff)
                .map(ProgressSample::record)
                .collect();
            let mut event = FarEvent {
                decision: decision.clone(),
                at_ms: ms_of(t_secs),
                unit: match at {
                    At::Unit(n) => n,
                    At::Start | At::Inside => 0,
                },
                samples,
                handed: false,
                not_handed: None,
                unspent_ms: 0,
                credited_ms: 0,
            };
            if decision.act && !outcome.stops() {
                let route = commit.as_ref().and_then(|c| {
                    c.routes
                        .iter()
                        .find(|r| r.from == cap_id && Some(&r.to) == decision.to.as_ref())
                        .cloned()
                });
                match route {
                    None => {
                        event.not_handed = Some(FarBlock::NoSealedRecipient.token().to_owned());
                    }
                    Some(route) => match cells.release_to(route.from_slot, route.to_slot, now) {
                        None => event.not_handed = Some("cells_refused".to_owned()),
                        Some(hand) => {
                            event.handed = true;
                            event.unspent_ms = hand.unspent_ns / 1_000_000;
                            event.credited_ms = hand.credited_ns / 1_000_000;
                            outcome = FarAction::HandOn {
                                to: route.to,
                                unspent_ms: event.unspent_ms,
                                credited_ms: event.credited_ms,
                            };
                        }
                    },
                }
            }
            if state.events.len() < MAX_EVENTS {
                state.events.push(event);
            } else {
                state.dropped = state.dropped.saturating_add(1);
            }
        }
        if outcome.stops() {
            state.handed = state.handed.saturating_add(1);
            if let Some(cap) = state.caps.get_mut(index) {
                cap.handed = Some(outcome.clone());
            }
        }
        outcome
    }

    /// The authority as recorded (whether or not armed now).
    #[must_use]
    pub fn snapshot(&self) -> FarSnapshot {
        let state = self.state();
        FarSnapshot {
            armed: self.is_armed(),
            commit: state.commit.clone(),
            calls: state.calls,
            events: state.events.clone(),
            refusals: state.refusals.clone(),
            handed: state.handed,
            dropped: state.dropped,
        }
    }
}

fn push_refusal(state: &mut State, refusal: SafePointRefusal) {
    if state.refusals.len() < MAX_EVENTS {
        state.refusals.push(refusal);
    } else {
        state.dropped = state.dropped.saturating_add(1);
    }
}

static GLOBAL: FarAuthority = FarAuthority::new();

/// The process's one FAR authority. `ny vnncomp` runs one row per process.
#[must_use]
pub fn global() -> &'static FarAuthority {
    &GLOBAL
}

/// The lane-side tap: `cap` reached its unit boundary `unit` with these
/// cumulative counters. One atomic load and `Continue` when unarmed; a panic
/// inside is contained (`router::guard::fail_open`) and answers `Continue`.
#[inline]
pub fn at_safe_point(sample: ProgressSample, unit: u64) -> FarAction {
    let authority = global();
    if !authority.is_armed() {
        return FarAction::Continue;
    }
    crate::router::guard::fail_open("far", || authority.at_safe_point(sample, unit))
        .unwrap_or(FarAction::Continue)
}

/// A FAR-classifiable Volume sample for a lane that reports a certified
/// volume fraction `closed_fraction` and an open queue `open` after `units`
/// units of work (lowdim / input split).
#[must_use]
pub fn volume_sample(cap: CapId, units: u64, closed_fraction: f64, open: u64) -> ProgressSample {
    let mut sample = ProgressSample::scalar(cap, units, closed_fraction);
    sample.open = open;
    sample
}

/// The class token of a decision (logs).
#[must_use]
pub const fn class_token(class: FarClass) -> &'static str {
    class.token()
}

/// The slot a hand-on would move time out of, when `cap` owns exactly one.
#[must_use]
pub fn slot_of(cap: &str) -> Option<LaneSlot> {
    crate::router::r4::unique_slot(cap)
}
