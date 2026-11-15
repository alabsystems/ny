// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r5 — THE ROUTE AUTHORITY (ROUTER_PROGRAM §3.5, §10 R5; layer L4
//! beside `deadline_authority`, `memory_lease` and `far_authority`).
//!
//! # What this is
//!
//! The process's record of what acting set 4 decided on this row: the
//! capabilities CR-RS ROUTED (each with the typed-key settings its sealed
//! evidence arm ran) and whether CR-G's gap-order contract funds
//! multi-objective classes. Each routed capability's own arming read site
//! asks [`RouteAuthority::routed`] beside its own key (`key || routed`), and
//! the frozen-dual tier asks [`RouteAuthority::gap_contract`] once per tier.
//! The authority counts every consult that answered `true` and keeps a
//! bounded ledger of what the glue applied and what each tier funded, so the
//! flight record shows whether a route ARRIVED, not only that it was armed.
//!
//! # Dark by default, fail open
//!
//! Unarmed — every run without `router.mode: on`, and every `on` run in which
//! no capability is sealed — every call is one atomic load answering `false`
//! / `None`, so an unarmed process is byte-identical to a binary without this
//! module. A router fault disarms it (`router::guard::fail_open`).
//!
//! # Never a verdict input
//!
//! It holds capability ids, typed-key paths and counters. A route only says
//! "run this sealed capability here, as its evidence arm ran"; the capability
//! still runs its own admission and its own certificate checks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use serde::{Deserialize, Serialize};

use crate::router::decl::CapId;

/// Ledger entries one row keeps per kind (counted past it, not stored).
pub const MAX_EVENTS: usize = 64;

/// One routed capability, as recorded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutedCap {
    /// The capability.
    pub cap: String,
    /// `(typed key path, value)` its route applies.
    pub settings: Vec<(String, String)>,
    /// Consults of [`RouteAuthority::routed`] that answered `true`.
    pub consults: u64,
}

/// One setting the glue applied (or found pinned) at a consumer's
/// configuration boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedSetting {
    /// The capability.
    pub cap: String,
    /// The typed key path.
    pub key: String,
    /// The value in force before.
    pub from: String,
    /// The value after (equal to `from` when blocked).
    pub to: String,
    /// `None` when applied; why not otherwise.
    pub blocked_by: Option<String>,
}

/// What one multi-objective tier funded under CR-G.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GapTier {
    /// Classes the tier was asked to fund.
    pub classes: u64,
    /// Classes funded with today's fair share.
    pub fair: u64,
    /// Classes funded with a contract larger than the fair share.
    pub contract: u64,
    /// Classes declined (their rung did not fit; they stay open in the tree).
    pub declined: u64,
    /// Classes whose build was cut by their window.
    pub cut: u64,
}

/// The authority, as recorded in the flight `router` record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteSnapshot {
    /// Whether it was armed when the snapshot was taken.
    pub armed: bool,
    /// The routed capabilities.
    pub routes: Vec<RoutedCap>,
    /// Whether CR-G funds multi-objective classes.
    pub gap_contract: bool,
    /// What the glue applied at configuration boundaries (bounded).
    pub applied: Vec<AppliedSetting>,
    /// Every CR-G tier (bounded).
    pub tiers: Vec<GapTier>,
    /// Ledger entries past [`MAX_EVENTS`].
    pub dropped: u64,
}

#[derive(Debug)]
struct State {
    routes: Vec<RoutedCap>,
    gap_contract: bool,
    applied: Vec<AppliedSetting>,
    tiers: Vec<GapTier>,
    dropped: u64,
}

impl State {
    const EMPTY: Self = Self {
        routes: Vec::new(),
        gap_contract: false,
        applied: Vec::new(),
        tiers: Vec::new(),
        dropped: 0,
    };
}

/// The route authority. One per process ([`global`]); instantiable for tests.
#[derive(Debug)]
pub struct RouteAuthority {
    armed: AtomicBool,
    state: Mutex<State>,
}

impl Default for RouteAuthority {
    fn default() -> Self {
        Self::new()
    }
}

impl RouteAuthority {
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

    /// Arm the row's routes and CR-G's flag, clearing the ledger first. With
    /// no route and no contract there is nothing to consult: the authority
    /// stays unarmed (every read is today's).
    pub fn arm(&self, routes: Vec<(String, Vec<(String, String)>)>, gap_contract: bool) {
        self.disarm();
        let mut state = self.state();
        *state = State {
            routes: routes
                .into_iter()
                .map(|(cap, settings)| RoutedCap {
                    cap,
                    settings,
                    consults: 0,
                })
                .collect(),
            gap_contract,
            ..State::EMPTY
        };
        let arm = !state.routes.is_empty() || state.gap_contract;
        drop(state);
        if arm {
            self.armed.store(true, Ordering::Release);
        }
    }

    /// Disarm: every read answers today's value from here on. The ledger is
    /// kept for the snapshot.
    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Release);
    }

    /// Whether armed. One atomic load: the cost of an unarmed consult.
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// Whether the router ROUTED `cap` on this row (counted). `false` when
    /// unarmed or not routed.
    pub fn routed(&self, cap: CapId) -> bool {
        if !self.is_armed() {
            return false;
        }
        let mut state = self.state();
        match state.routes.iter_mut().find(|r| r.cap == cap.as_str()) {
            Some(route) => {
                route.consults = route.consults.saturating_add(1);
                true
            }
            None => false,
        }
    }

    /// The settings `cap`'s route applies (counted as a consult), `None`
    /// when unarmed or not routed.
    pub fn settings(&self, cap: CapId) -> Option<Vec<(String, String)>> {
        if !self.is_armed() {
            return None;
        }
        let mut state = self.state();
        let route = state.routes.iter_mut().find(|r| r.cap == cap.as_str())?;
        route.consults = route.consults.saturating_add(1);
        Some(route.settings.clone())
    }

    /// Whether CR-G funds multi-objective classes on this row.
    #[must_use]
    pub fn gap_contract(&self) -> bool {
        self.is_armed() && self.state().gap_contract
    }

    /// Record a setting the glue applied (or found blocked). A no-op unarmed.
    pub fn note_applied(&self, applied: AppliedSetting) {
        if !self.is_armed() {
            return;
        }
        let mut state = self.state();
        if state.applied.len() < MAX_EVENTS {
            state.applied.push(applied);
        } else {
            state.dropped = state.dropped.saturating_add(1);
        }
    }

    /// Record what one tier funded under CR-G. A no-op unarmed.
    pub fn note_tier(&self, tier: GapTier) {
        if !self.is_armed() {
            return;
        }
        let mut state = self.state();
        if state.tiers.len() < MAX_EVENTS {
            state.tiers.push(tier);
        } else {
            state.dropped = state.dropped.saturating_add(1);
        }
    }

    /// The authority as recorded (whether or not armed now).
    #[must_use]
    pub fn snapshot(&self) -> RouteSnapshot {
        let state = self.state();
        RouteSnapshot {
            armed: self.is_armed(),
            routes: state.routes.clone(),
            gap_contract: state.gap_contract,
            applied: state.applied.clone(),
            tiers: state.tiers.clone(),
            dropped: state.dropped,
        }
    }
}

static GLOBAL: RouteAuthority = RouteAuthority::new();

/// The process's one route authority. `ny vnncomp` runs one row per process.
#[must_use]
pub fn global() -> &'static RouteAuthority {
    &GLOBAL
}

#[cfg(test)]
mod tests {
    use super::*;

    const FD: CapId = CapId("frozen-dual-bab");
    const WAVE: CapId = CapId("wave-beta-bab");

    #[test]
    fn unarmed_route_authority_answers_nothing() {
        let a = RouteAuthority::new();
        assert!(!a.is_armed());
        assert!(!a.routed(FD));
        assert_eq!(a.settings(FD), None);
        assert!(!a.gap_contract());
        a.note_applied(AppliedSetting {
            cap: FD.as_str().to_owned(),
            key: "k".to_owned(),
            from: "false".to_owned(),
            to: "true".to_owned(),
            blocked_by: None,
        });
        a.note_tier(GapTier::default());
        let snap = a.snapshot();
        assert!(!snap.armed && snap.routes.is_empty() && snap.applied.is_empty());
        assert_eq!(snap.tiers.len(), 0);
        // Arming with nothing to route and no contract stays unarmed.
        a.arm(Vec::new(), false);
        assert!(!a.is_armed());
    }

    #[test]
    fn route_authority_answers_only_for_routed_capabilities_and_counts_consults() {
        let a = RouteAuthority::new();
        a.arm(
            vec![(
                FD.as_str().to_owned(),
                vec![("margin_row.frozen_dual_bab".to_owned(), "true".to_owned())],
            )],
            false,
        );
        assert!(a.is_armed());
        assert!(a.routed(FD));
        assert!(!a.routed(WAVE), "an unrouted capability reads false");
        assert_eq!(
            a.settings(FD),
            Some(vec![(
                "margin_row.frozen_dual_bab".to_owned(),
                "true".to_owned()
            )])
        );
        assert!(!a.gap_contract());
        let snap = a.snapshot();
        assert_eq!(snap.routes[0].consults, 2);
        a.disarm();
        assert!(!a.routed(FD), "a disarmed authority is today's");
        assert_eq!(a.snapshot().routes[0].consults, 2, "the ledger is kept");
        // Re-arming clears the ledger.
        a.arm(Vec::new(), true);
        assert!(a.gap_contract());
        assert!(!a.routed(FD));
        assert_eq!(a.snapshot().routes.len(), 0);
    }

    #[test]
    fn route_authority_ledger_is_bounded() {
        let a = RouteAuthority::new();
        a.arm(Vec::new(), true);
        for _ in 0..(MAX_EVENTS + 5) {
            a.note_tier(GapTier::default());
        }
        let snap = a.snapshot();
        assert_eq!(snap.tiers.len(), MAX_EVENTS);
        assert_eq!(snap.dropped, 5);
        let text = serde_json::to_string(&snap).expect("serialize");
        let back: RouteSnapshot = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, snap);
    }
}
