// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #budget-abandonment FIX C: make the ReLU-split depth cap OBSERVABLE.
//!
//! `bab.max_depth` defaults to 100 and a too-deep domain is DROPPED, not
//! requeued (`domain_filter.rs`): the filter sets
//! `lifecycle.unresolved_due_to_depth` and `continue`s, so the queue drains
//! and the run returns `Unknown` ("Max depth N reached") with most of the
//! budget still on the clock. alpha-beta-CROWN has no depth cap at all.
//!
//! Raising the cap is a one-line yaml change, but before this module there was
//! NO WAY to tell "raised and still hit" from "raised and irrelevant" — the
//! reason string only says the cap was hit at all, and says nothing once it is
//! not. So the summary below is emitted UNCONDITIONALLY on every ReLU-split
//! entry, `dropped=0` included: an absent line means the binary has no lever
//! and a null from it is VOID, not negative.
//!
//! Telemetry only. Nothing here is read by a verdict.

use std::cell::Cell;

thread_local! {
    /// Domains dropped by the depth cap in the current ReLU-split entry.
    static DROPPED: Cell<usize> = const { Cell::new(0) };
    /// The deepest such domain's depth (0 when none was dropped).
    static DEEPEST: Cell<usize> = const { Cell::new(0) };
    /// Live scope nesting depth on this thread. Only the OUTERMOST scope
    /// resets and reports; a nested entry that zeroed the counters would make
    /// the outer scope publish `dropped=0` for a run that dropped domains,
    /// which is exactly the absence-read-as-evidence this line exists to stop.
    static NESTING: Cell<usize> = const { Cell::new(0) };
}

/// Record one domain dropped at `depth` by the `max_depth` cap.
pub(crate) fn record(depth: usize) {
    DROPPED.with(|c| c.set(c.get().saturating_add(1)));
    DEEPEST.with(|c| c.set(c.get().max(depth)));
}

/// Read the counters without disturbing them (tests, and the guard's own
/// report).
pub(crate) fn snapshot() -> (usize, usize) {
    (DROPPED.with(Cell::get), DEEPEST.with(Cell::get))
}

fn reset() {
    DROPPED.with(|c| c.set(0));
    DEEPEST.with(|c| c.set(0));
}

/// RAII scope around one ReLU-split verification: zeroes the counters on
/// entry and prints the summary on EVERY exit path, `?` returns included.
pub(crate) struct DepthDropScope {
    cap: usize,
    lane: &'static str,
    outermost: bool,
}

impl DepthDropScope {
    /// The single-objective ReLU-split entry point.
    pub(crate) fn new(cap: usize) -> Self {
        Self::for_lane(cap, "relu_split_single")
    }

    /// `lane` NAMES THE SCOPE OF THE COUNT. This module instruments the
    /// single-objective ReLU-split filter only; the multi-objective /
    /// domain-batch, input-split and GPU DomainList lanes drop at their own
    /// sites and are NOT covered, so `dropped=0` licenses no conclusion about
    /// them.
    pub(crate) fn for_lane(cap: usize, lane: &'static str) -> Self {
        let outermost = NESTING.with(|c| {
            let live = c.get();
            c.set(live.saturating_add(1));
            live == 0
        });
        if outermost {
            reset();
        }
        Self {
            cap,
            lane,
            outermost,
        }
    }
}

impl Drop for DepthDropScope {
    fn drop(&mut self) {
        NESTING.with(|c| c.set(c.get().saturating_sub(1)));
        if !self.outermost {
            return;
        }
        let (dropped, deepest) = snapshot();
        eprintln!(
            "[depth-drop] lane={} dropped={dropped} deepest={deepest} cap={}",
            self.lane, self.cap
        );
        reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nested_scope_does_not_zero_the_outer_scopes_counters() {
        let _outer = DepthDropScope::new(100);
        record(5);
        {
            let _inner = DepthDropScope::new(100);
            record(9);
            assert_eq!(
                snapshot(),
                (2, 9),
                "a nested entry must accumulate, never reset"
            );
        }
        assert_eq!(
            snapshot(),
            (2, 9),
            "the inner scope's Drop must not clear the outer scope's count"
        );
    }

    #[test]
    fn scope_zeroes_on_entry_and_accumulates_the_deepest_drop() {
        {
            let _scope = DepthDropScope::new(100);
            record(7);
            record(3);
            record(11);
            assert_eq!(snapshot(), (3, 11));
        }
        // Drop restored the zero state for the next entry on this thread.
        assert_eq!(snapshot(), (0, 0));

        let _scope = DepthDropScope::new(4);
        assert_eq!(snapshot(), (0, 0), "a new scope starts from zero");
    }
}
