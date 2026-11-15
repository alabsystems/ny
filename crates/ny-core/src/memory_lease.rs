// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r4 CR-L — THE MEMORY LEASE AUTHORITY (ROUTER_PROGRAM §2.2, §3.2
//! C7, §8, F6; layer L4 beside `deadline_authority`).
//!
//! # What this is
//!
//! One envelope per row (`router::lease::LeasePlan`, computed purely from the
//! DECLARED host's static envelope) and a ledger of the leases lanes hold
//! against it. The internal verifier holds its base lease from arming until
//! it returns; every graph-heavy optional lane ACQUIRES its lease before it
//! starts and RELEASES it when its working set is gone. A lease is granted
//! only while the sum of outstanding leases stays inside the envelope, so the
//! envelope is an AGGREGATE reservation, which the live per-call headroom gate
//! never was.
//!
//! # Dark by default, fail open
//!
//! Unarmed — every run without `router.mode: on` — every call returns `None`
//! and every gate keeps exactly today's live decision, so an unarmed process
//! is byte-identical to a binary without this module. A router fault disarms
//! it (`router::guard::fail_open`), which is the same `None` everywhere.
//!
//! # Never a verdict input
//!
//! The authority holds byte counts and nothing else. Its only output is an
//! admission decision; the glue (`ny-cli::router::lease`) turns it into the
//! same `bool` today's gate returns, AND-ed with today's live envelope (the
//! floor). It cannot reach a bound, a certificate or the publisher.
//!
//! # Never panics
//!
//! One poison-tolerant lock, saturating arithmetic, bounded ledger.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use serde::{Deserialize, Serialize};

use crate::router::lease::{decide, LeaseDecision, LeaseHolder, LeasePlan, HOLDERS};

/// Lease events one row records (counted past it, not stored).
pub const MAX_EVENTS: usize = 64;

/// What a lease event did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseOp {
    /// A new lease was decided (granted, narrowed or denied).
    Acquire,
    /// A held lease was returned.
    Release,
}

/// One lease event, as recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LeaseEvent {
    /// The holder.
    pub holder: LeaseHolder,
    /// What happened.
    pub op: LeaseOp,
    /// The decision (acquire) or the bytes returned (release, as `Granted`).
    pub decision: LeaseDecision,
    /// Bytes outstanding after the event.
    pub outstanding_bytes: u64,
}

/// One holder, as recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HolderSnapshot {
    /// The holder.
    pub holder: LeaseHolder,
    /// Bytes it holds now (0 = none).
    pub held_bytes: u64,
    /// Bytes its memory model asks for.
    pub want_bytes: u64,
}

/// The authority, as recorded in the flight `router` record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseSnapshot {
    /// Whether it was armed when the snapshot was taken.
    pub armed: bool,
    /// The plan it was armed with.
    pub plan: Option<LeasePlan>,
    /// Bytes outstanding now.
    pub outstanding_bytes: u64,
    /// The largest outstanding total seen this row.
    pub peak_bytes: u64,
    /// Every holder.
    pub holders: Vec<HolderSnapshot>,
    /// Every acquire / release, in order (bounded).
    pub events: Vec<LeaseEvent>,
    /// Events past [`MAX_EVENTS`].
    pub dropped: u64,
    /// Leases granted whole.
    pub granted: u64,
    /// Leases narrowed.
    pub narrowed: u64,
    /// Requests denied.
    pub denied: u64,
    /// Leases returned.
    pub released: u64,
}

/// Why arming was refused (the authority then stays unarmed: today).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseArmRefusal {
    /// The envelope is zero (unknown host memory).
    EnvelopeUnknown,
    /// The terminal's base lease does not fit the envelope.
    TerminalPastEnvelope,
}

#[derive(Debug, Default)]
struct State {
    plan: Option<LeasePlan>,
    held: [u64; HOLDERS],
    peak: u64,
    events: Vec<LeaseEvent>,
    dropped: u64,
    granted: u64,
    narrowed: u64,
    denied: u64,
    released: u64,
}

impl State {
    fn outstanding(&self) -> u64 {
        self.held.iter().fold(0u64, |sum, b| sum.saturating_add(*b))
    }

    fn available(&self) -> u64 {
        self.plan.map_or(0, |plan| {
            plan.envelope_bytes.saturating_sub(self.outstanding())
        })
    }

    fn push(&mut self, event: LeaseEvent) {
        if self.events.len() < MAX_EVENTS {
            self.events.push(event);
        } else {
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    fn held(&self, holder: LeaseHolder) -> u64 {
        self.held.get(holder.index()).copied().unwrap_or(0)
    }

    fn set_held(&mut self, holder: LeaseHolder, bytes: u64) {
        if let Some(slot) = self.held.get_mut(holder.index()) {
            *slot = bytes;
        }
    }
}

/// The lease authority. One per process ([`global`]); instantiable for tests.
#[derive(Debug)]
pub struct LeaseAuthority {
    armed: AtomicBool,
    state: Mutex<State>,
}

impl Default for LeaseAuthority {
    fn default() -> Self {
        Self::new()
    }
}

impl LeaseAuthority {
    /// An unarmed authority.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
            state: Mutex::new(State {
                plan: None,
                held: [0; HOLDERS],
                peak: 0,
                events: Vec::new(),
                dropped: 0,
                granted: 0,
                narrowed: 0,
                denied: 0,
                released: 0,
            }),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Arm the row's plan and reserve the terminal's base lease. Clears every
    /// lease and the ledger first. Refuses (and stays unarmed) on an unknown
    /// envelope or a terminal share past it. Returns the terminal's decision.
    pub fn arm(&self, plan: LeasePlan) -> Result<LeaseDecision, LeaseArmRefusal> {
        if plan.envelope_bytes == 0 {
            return Err(LeaseArmRefusal::EnvelopeUnknown);
        }
        if plan.terminal_bytes() > plan.envelope_bytes {
            return Err(LeaseArmRefusal::TerminalPastEnvelope);
        }
        self.disarm();
        let mut state = self.state();
        *state = State {
            plan: Some(plan),
            ..State::default()
        };
        let decision = decide(state.available(), plan.need(LeaseHolder::Terminal));
        state.set_held(LeaseHolder::Terminal, decision.bytes());
        state.granted = state.granted.saturating_add(1);
        let outstanding = state.outstanding();
        state.peak = outstanding;
        state.push(LeaseEvent {
            holder: LeaseHolder::Terminal,
            op: LeaseOp::Acquire,
            decision,
            outstanding_bytes: outstanding,
        });
        drop(state);
        self.armed.store(true, Ordering::Release);
        Ok(decision)
    }

    /// Disarm: every call returns `None` from here on (today's gates). The
    /// ledger is kept for the snapshot.
    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Release);
    }

    /// Whether a plan is armed. One atomic load: the cost of an unarmed gate.
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// The armed plan.
    #[must_use]
    pub fn plan(&self) -> Option<LeasePlan> {
        if !self.is_armed() {
            return None;
        }
        self.state().plan
    }

    /// `holder` asks for its lease. `None` when unarmed (the caller keeps
    /// today's gate). A holder that already holds a lease gets it back
    /// unchanged — a lane re-checking at each of its stages is never debited
    /// twice — and a denial is recorded and leaves nothing outstanding.
    pub fn acquire(&self, holder: LeaseHolder) -> Option<LeaseDecision> {
        if !self.is_armed() {
            return None;
        }
        let mut state = self.state();
        let plan = state.plan?;
        let held = state.held(holder);
        if held > 0 {
            let want = plan.need(holder).want_bytes;
            return Some(if held >= want {
                LeaseDecision::Granted { bytes: held }
            } else {
                LeaseDecision::Narrowed { bytes: held, want }
            });
        }
        let decision = decide(state.available(), plan.need(holder));
        match decision {
            LeaseDecision::Granted { .. } => state.granted = state.granted.saturating_add(1),
            LeaseDecision::Narrowed { .. } => state.narrowed = state.narrowed.saturating_add(1),
            LeaseDecision::Denied { .. } => state.denied = state.denied.saturating_add(1),
        }
        state.set_held(holder, decision.bytes());
        let outstanding = state.outstanding();
        state.peak = state.peak.max(outstanding);
        state.push(LeaseEvent {
            holder,
            op: LeaseOp::Acquire,
            decision,
            outstanding_bytes: outstanding,
        });
        Some(decision)
    }

    /// What [`LeaseAuthority::acquire`] would decide for `holder` now,
    /// without reserving anything (a lane that only decides whether to hold
    /// TIME for a later heavy stage). `None` when unarmed.
    #[must_use]
    pub fn peek(&self, holder: LeaseHolder) -> Option<LeaseDecision> {
        if !self.is_armed() {
            return None;
        }
        let state = self.state();
        let plan = state.plan?;
        let held = state.held(holder);
        if held > 0 {
            return Some(LeaseDecision::Granted { bytes: held });
        }
        Some(decide(state.available(), plan.need(holder)))
    }

    /// `holder`'s working set is gone: return its lease. Returns the bytes
    /// returned (`None` when unarmed or nothing was held). Releasing twice is
    /// a no-op.
    pub fn release(&self, holder: LeaseHolder) -> Option<u64> {
        if !self.is_armed() {
            return None;
        }
        let mut state = self.state();
        let held = state.held(holder);
        if held == 0 {
            return None;
        }
        state.set_held(holder, 0);
        state.released = state.released.saturating_add(1);
        let outstanding = state.outstanding();
        state.push(LeaseEvent {
            holder,
            op: LeaseOp::Release,
            decision: LeaseDecision::Granted { bytes: held },
            outstanding_bytes: outstanding,
        });
        Some(held)
    }

    /// Bytes `holder` holds now (0 when unarmed or none).
    #[must_use]
    pub fn held(&self, holder: LeaseHolder) -> u64 {
        if !self.is_armed() {
            return 0;
        }
        self.state().held(holder)
    }

    /// The authority as recorded (whether or not armed now).
    #[must_use]
    pub fn snapshot(&self) -> LeaseSnapshot {
        let state = self.state();
        let holders = LeaseHolder::ALL
            .into_iter()
            .map(|holder| HolderSnapshot {
                holder,
                held_bytes: state.held(holder),
                want_bytes: state.plan.map_or(0, |plan| plan.need(holder).want_bytes),
            })
            .collect();
        LeaseSnapshot {
            armed: self.is_armed(),
            plan: state.plan,
            outstanding_bytes: state.outstanding(),
            peak_bytes: state.peak,
            holders,
            events: state.events.clone(),
            dropped: state.dropped,
            granted: state.granted,
            narrowed: state.narrowed,
            denied: state.denied,
            released: state.released,
        }
    }
}

static GLOBAL: LeaseAuthority = LeaseAuthority::new();

/// The process's one lease authority. `ny vnncomp` runs one row per process.
#[must_use]
pub fn global() -> &'static LeaseAuthority {
    &GLOBAL
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::facts::{DeviceQual, HostClass, HostEnvelope};
    use crate::router::lease::{lease_plan, LeaseNeed, OPTIONAL_HEAVY_LEASE_BYTES};

    const GIB: u64 = 1 << 30;

    fn plan(mem_gib: u64, reserve_gib: u64) -> LeasePlan {
        lease_plan(&HostEnvelope {
            class: HostClass::Other,
            threads: 8,
            mem_bytes: mem_gib * GIB,
            reserve_bytes: reserve_gib * GIB,
            device: DeviceQual::None,
            declared: true,
        })
        .expect("a known envelope")
    }

    #[test]
    fn unarmed_lease_authority_answers_nothing() {
        let a = LeaseAuthority::new();
        assert!(!a.is_armed());
        assert_eq!(a.plan(), None);
        for holder in LeaseHolder::ALL {
            assert_eq!(a.acquire(holder), None);
            assert_eq!(a.peek(holder), None);
            assert_eq!(a.release(holder), None);
            assert_eq!(a.held(holder), 0);
        }
        let snap = a.snapshot();
        assert!(!snap.armed && snap.events.is_empty());
        assert_eq!(snap.outstanding_bytes, 0);
    }

    /// The aggregate contract: outstanding never exceeds the envelope, a
    /// denial reserves nothing, a release returns exactly what was held, and
    /// an idempotent re-acquire never debits twice.
    #[test]
    fn lease_outstanding_never_exceeds_the_envelope() {
        // 64 GiB host, 16 GiB reserve: E = 48 GiB, terminal 24, pool 24.
        let a = LeaseAuthority::new();
        let terminal = a.arm(plan(64, 16)).expect("arm");
        assert_eq!(terminal, LeaseDecision::Granted { bytes: 24 * GIB });
        let first = a.acquire(LeaseHolder::MarginRowConcurrent).expect("armed");
        assert_eq!(
            first,
            LeaseDecision::Granted {
                bytes: OPTIONAL_HEAVY_LEASE_BYTES
            }
        );
        // A second 16 GiB lease does not fit the 8 GiB left: denied, and the
        // denial reserves nothing.
        let second = a.acquire(LeaseHolder::PostVerifierTail).expect("armed");
        assert_eq!(
            second,
            LeaseDecision::Denied {
                available: 8 * GIB,
                want: 16 * GIB
            }
        );
        assert_eq!(a.held(LeaseHolder::PostVerifierTail), 0);
        assert_eq!(a.snapshot().outstanding_bytes, 40 * GIB);
        // Re-acquiring a held lease is idempotent.
        assert_eq!(
            a.acquire(LeaseHolder::MarginRowConcurrent),
            Some(LeaseDecision::Granted { bytes: 16 * GIB })
        );
        assert_eq!(a.snapshot().outstanding_bytes, 40 * GIB);
        // The terminal returns: the tail now fits.
        assert_eq!(a.release(LeaseHolder::Terminal), Some(24 * GIB));
        assert_eq!(a.release(LeaseHolder::Terminal), None, "released once");
        assert!(a
            .peek(LeaseHolder::PostVerifierTail)
            .is_some_and(LeaseDecision::admits));
        assert_eq!(
            a.snapshot().outstanding_bytes,
            16 * GIB,
            "a peek reserves nothing"
        );
        assert!(a
            .acquire(LeaseHolder::PostVerifierTail)
            .is_some_and(LeaseDecision::admits));
        let snap = a.snapshot();
        assert!(snap.peak_bytes <= 48 * GIB);
        assert_eq!(snap.outstanding_bytes, 32 * GIB);
        assert_eq!((snap.granted, snap.denied, snap.released), (3, 1, 1));
    }

    #[test]
    fn lease_arm_refuses_an_unknown_or_overfull_envelope() {
        let a = LeaseAuthority::new();
        let mut zero = plan(64, 16);
        zero.envelope_bytes = 0;
        assert_eq!(a.arm(zero), Err(LeaseArmRefusal::EnvelopeUnknown));
        let mut overfull = plan(64, 16);
        overfull.needs[LeaseHolder::Terminal.index()] = LeaseNeed::whole(u64::MAX);
        assert_eq!(a.arm(overfull), Err(LeaseArmRefusal::TerminalPastEnvelope));
        assert!(!a.is_armed());
        a.arm(plan(64, 16)).expect("positive control");
        assert!(a.is_armed());
        a.disarm();
        assert_eq!(a.acquire(LeaseHolder::MarginRowInline), None);
    }

    #[test]
    fn lease_extreme_values_never_panic() {
        let a = LeaseAuthority::new();
        let mut huge = plan(64, 16);
        huge.envelope_bytes = u64::MAX;
        huge.needs = [LeaseNeed::whole(u64::MAX); HOLDERS];
        huge.needs[LeaseHolder::Terminal.index()] = LeaseNeed::whole(u64::MAX);
        a.arm(huge).expect("terminal fits a u64::MAX envelope");
        for holder in LeaseHolder::ALL {
            let _ = a.acquire(holder);
            let _ = a.peek(holder);
        }
        for _ in 0..(MAX_EVENTS + 4) {
            let _ = a.release(LeaseHolder::Terminal);
            let _ = a.acquire(LeaseHolder::Terminal);
        }
        let snap = a.snapshot();
        assert!(snap.events.len() <= MAX_EVENTS);
        let text = serde_json::to_string(&snap).expect("serialize");
        let back: LeaseSnapshot = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, snap);
    }
}
