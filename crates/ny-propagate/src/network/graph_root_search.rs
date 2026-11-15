// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-root-alpha (ENGINE_PROGRAM §2 V2): the injected UNTRUSTED search
//! backend of the graph lane's root α search
//! (`solver.alpha-crown.margin_alpha_search: joint_f32`), also consulted by
//! the graph β-CROWN lane's batched domain search (#graph-lane-search, M7,
//! `bab.graph_search: cpu_f32`) — one registry, one backend.
//!
//! This crate never depends on the search crate (or on any CUDA crate): it sees
//! the backend only as `&dyn ny_core::search::SearchBackend`, registered once
//! by ny-cli at startup with [`set_graph_root_search_backend_factory`]. The
//! factory runs at most once, on the first ARMED phase, so a dark lane
//! constructs nothing. A missing registration is a refusal of the phase (the
//! certified incumbent stands), never a verdict.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use ny_core::search::SearchBackend;

/// Process-wide outcome counters of the ARMED V2 phase.
///
/// TELEMETRY ONLY: nothing verdict-bearing reads them. They exist so a caller
/// that cannot see the lane's internals (ny-cli's end-to-end test, a probe
/// harness) can tell "the search ran and was certified" from "the phase
/// refused and the incumbent stood" — two outcomes whose published bounds can
/// be bit-identical, which is exactly how a VOID arm passes as a result.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphRootPhaseCounts {
    /// Armed phases that ran to their publication decision.
    pub entered: usize,
    /// Phases that refused (lowering, self-check, backend, deadline, ...).
    pub refused: usize,
    /// Candidate states the lane certified (one fold each).
    pub certified: usize,
    /// Candidates kept (strictly better than the incumbent).
    pub kept: usize,
    /// Disjoint-enclosure soundness alarms (publication withdrawn).
    pub alarms: usize,
}

static PHASE_ENTERED: AtomicUsize = AtomicUsize::new(0);
static PHASE_REFUSED: AtomicUsize = AtomicUsize::new(0);
static PHASE_CERTIFIED: AtomicUsize = AtomicUsize::new(0);
static PHASE_KEPT: AtomicUsize = AtomicUsize::new(0);
static PHASE_ALARMS: AtomicUsize = AtomicUsize::new(0);

/// A snapshot of the [`GraphRootPhaseCounts`] of this process.
#[must_use]
pub fn graph_root_phase_counts() -> GraphRootPhaseCounts {
    GraphRootPhaseCounts {
        entered: PHASE_ENTERED.load(Ordering::Relaxed),
        refused: PHASE_REFUSED.load(Ordering::Relaxed),
        certified: PHASE_CERTIFIED.load(Ordering::Relaxed),
        kept: PHASE_KEPT.load(Ordering::Relaxed),
        alarms: PHASE_ALARMS.load(Ordering::Relaxed),
    }
}

/// Record one armed phase's outcome (the lane, once per armed root).
pub(crate) fn record_graph_root_phase(refused: bool, certified: bool, kept: bool, alarm: bool) {
    PHASE_ENTERED.fetch_add(1, Ordering::Relaxed);
    for (hit, counter) in [
        (refused, &PHASE_REFUSED),
        (certified, &PHASE_CERTIFIED),
        (kept, &PHASE_KEPT),
        (alarm, &PHASE_ALARMS),
    ] {
        if hit {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Process-wide outcome counters of the ARMED M7 graph-lane search
/// (`bab.graph_search`, #graph-lane-search), which consults the SAME
/// registered backend as V2.
///
/// TELEMETRY ONLY, like [`GraphRootPhaseCounts`]: nothing verdict-bearing
/// reads them. They let a caller that cannot see the lane (ny-cli's
/// end-to-end test, a probe harness) tell "searched, certified and kept" from
/// "refused, the lane's own bounds stood" — two outcomes that can publish the
/// same verdict.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphLaneSearchCounts {
    /// Armed root phases entered.
    pub root_phases: usize,
    /// Root phases that refused before a certified candidate.
    pub root_refused: usize,
    /// Root candidates the lane certified (one evaluation each).
    pub root_certified: usize,
    /// Root phases that raised at least one row strictly.
    pub root_kept: usize,
    /// Armed waves entered.
    pub waves: usize,
    /// Waves that refused before any domain was certified.
    pub wave_refused: usize,
    /// Domains the backend returned a hint for.
    pub domains_searched: usize,
    /// Domain candidates the lane certified.
    pub domains_certified: usize,
    /// Domains whose bounds a certified candidate strictly raised.
    pub domains_kept: usize,
    /// Domains the raise closed (every row verified).
    pub domains_closed: usize,
    /// Rows strictly raised (root and domains).
    pub rows_raised: usize,
    /// Root disjoint-enclosure soundness alarms (publication withdrawn).
    pub alarms: usize,
}

const LANE_SLOTS: usize = 12;
#[allow(clippy::declare_interior_mutable_const)]
const LANE_ZERO: AtomicUsize = AtomicUsize::new(0);
static LANE: [AtomicUsize; LANE_SLOTS] = [LANE_ZERO; LANE_SLOTS];

/// A snapshot of the [`GraphLaneSearchCounts`] of this process.
#[must_use]
pub fn graph_lane_search_counts() -> GraphLaneSearchCounts {
    let at = |k: usize| LANE[k].load(Ordering::Relaxed);
    GraphLaneSearchCounts {
        root_phases: at(0),
        root_refused: at(1),
        root_certified: at(2),
        root_kept: at(3),
        waves: at(4),
        wave_refused: at(5),
        domains_searched: at(6),
        domains_certified: at(7),
        domains_kept: at(8),
        domains_closed: at(9),
        rows_raised: at(10),
        alarms: at(11),
    }
}

/// Add `delta` to the M7 counters (the lane, once per armed phase or wave).
pub(crate) fn record_graph_lane_search(delta: &GraphLaneSearchCounts) {
    let slots = [
        delta.root_phases,
        delta.root_refused,
        delta.root_certified,
        delta.root_kept,
        delta.waves,
        delta.wave_refused,
        delta.domains_searched,
        delta.domains_certified,
        delta.domains_kept,
        delta.domains_closed,
        delta.rows_raised,
        delta.alarms,
    ];
    for (slot, add) in LANE.iter().zip(slots) {
        if add > 0 {
            slot.fetch_add(add, Ordering::Relaxed);
        }
    }
}

type BackendFactory = Box<dyn Fn() -> Option<Arc<dyn SearchBackend>> + Send + Sync>;

static FACTORY: OnceLock<BackendFactory> = OnceLock::new();

/// Register the process's graph root-search backend factory (ny-cli, once,
/// at startup). First install wins.
pub fn set_graph_root_search_backend_factory<F>(factory: F)
where
    F: Fn() -> Option<Arc<dyn SearchBackend>> + Send + Sync + 'static,
{
    let _ = FACTORY.set(Box::new(factory));
}

/// The backend the armed phase consults: a test override on this thread when
/// one is installed, else the registered backend (built once per process).
/// `None` = no backend: the phase refuses and the incumbent stands.
pub(crate) fn graph_root_backend() -> Option<Arc<dyn SearchBackend>> {
    #[cfg(test)]
    if let Some(BackendOverride(overridden)) = test_override() {
        return overridden;
    }
    static HANDLE: OnceLock<Arc<dyn SearchBackend>> = OnceLock::new();
    if let Some(backend) = HANDLE.get() {
        return Some(Arc::clone(backend));
    }
    let built = FACTORY.get().and_then(|factory| factory())?;
    Some(Arc::clone(HANDLE.get_or_init(|| built)))
}

/// An installed test override of the registry: its `None` means "no backend
/// registered" (a named type rather than `Option<Option<_>>`, whose two
/// `None`s read alike).
#[cfg(test)]
#[derive(Clone)]
struct BackendOverride(Option<Arc<dyn SearchBackend>>);

#[cfg(test)]
thread_local! {
    /// The override on this thread, if one is installed.
    static TEST_BACKEND: std::cell::RefCell<Option<BackendOverride>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn test_override() -> Option<BackendOverride> {
    TEST_BACKEND.with(|cell| cell.borrow().clone())
}

/// Test-only: run `f` with `backend` standing in for the registry on this
/// thread (`None` = "no backend registered"), restoring the previous override
/// afterwards even if `f` panics.
#[cfg(test)]
pub(crate) fn with_test_backend<T>(
    backend: Option<Arc<dyn SearchBackend>>,
    f: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<BackendOverride>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            TEST_BACKEND.with(|cell| *cell.borrow_mut() = previous);
        }
    }
    let previous = TEST_BACKEND.with(|cell| cell.borrow_mut().replace(BackendOverride(backend)));
    let _restore = Restore(previous);
    f()
}
