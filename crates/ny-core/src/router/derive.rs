// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! O-DERIVE (ROUTER_PROGRAM §5.1, §5.2): caps derived from the deadline and
//! the memory lease instead of constants, and the publish grace. PURE: no
//! clock, no environment, no global; integer milliseconds and bytes only,
//! saturating everywhere.
//!
//! Every derivation here replaces a constant that caused a MEASURED early quit
//! (`study_routing_today.md` W2) and is bounded on both sides:
//!
//! * **per-clause passes** (`bab.disjunctive_clause_retry_passes`): the
//!   one-pass per-clause allocator back-loads the budget, so the FIRST clauses
//!   get the SMALLEST slice (14_ieee prop1: clause 1 got 10 s and died,
//!   clause 56 got 442 s and used 2.1 s; 470 s of 600 unspent). The loop
//!   itself already stops when a pass decides nothing and never runs past the
//!   Graph-MIP-reserving deadline, so passes are bounded by the clause count
//!   by construction; the derived ceiling adds the deadline's own bound — a
//!   pass costs at least one clause bootstrap (measured 6.4 s on the 118-bus
//!   net), so more than one pass per [`PASS_MIN_MS`] of window can never run;
//! * **tree depth** (`bab.max_depth`): the global default of 100 DROPS a
//!   domain at depth 100, which drains the queue into `unknown` with the
//!   budget unspent (118_ieee prop6: `unknown` at 9.9 s of 600; depth 100 000
//!   proved it `unsat` at 244 s). A path of depth `d` needs `d` sequential
//!   expansions, so a cap above [`DEPTH_PER_WINDOW_SEC`] per second of window
//!   is never reached before the deadline; the float-resolution floors
//!   (`min_branch_width`, the lowdim `min_width`) still end every path. Depth
//!   also bounded memory as a side effect — the same yaml that measured the
//!   conversion recorded that a byte budget on the resident queue is the real
//!   memory bound — so the lift is admitted only while such a budget is in
//!   force AND fits the host's static lease ([`depth_lift_gate`]);
//! * **publish grace** ([`publish_grace_ms`]): a completed certificate may use
//!   `0 < g < 3 s`, strictly below the join margin, leaving at least
//!   [`WRITE_RESERVE_MS`] for the result write (measured 9-18 ms).

use serde::{Deserialize, Serialize};

/// Kept after the last collection for the result write (measured 9-18 ms:
/// ~25x headroom).
pub const WRITE_RESERVE_MS: u64 = 500;

/// The publish grace must stay strictly below this (the margin-row lane's own
/// 3 s wrap-up reserve).
pub const PUBLISH_GRACE_CEILING_MS: u64 = 3_000;

/// The margin-row lanes stop searching this long before the scored deadline
/// (their own wrap-up reserve; unchanged by the router).
pub const MARGIN_ROW_TAIL_MS: u64 = 3_000;

/// A retry pass costs at least one clause bootstrap; one per second of window
/// is a generous upper bound on how many could ever run.
pub const PASS_MIN_MS: u64 = 1_000;

/// Hard ceiling on derived retry passes.
pub const PASS_CEILING: u64 = 1_024;

/// Expansions per second of window no path can exceed: a depth cap above this
/// is never reached before the deadline.
pub const DEPTH_PER_WINDOW_SEC: u64 = 1_000;

/// Hard ceiling on a derived depth cap.
pub const DEPTH_CEILING: u64 = 10_000_000;

/// The resident-queue byte budget may use at most this fraction (1/DEN) of
/// the static memory lease for the depth lift to be admitted.
pub const QUEUE_LEASE_DEN: u64 = 2;

/// CR-R: the publish grace for a completed certificate, from the join margin.
/// `None` when the margin leaves no room after the write reserve.
#[must_use]
pub const fn publish_grace_ms(join_ms: u64) -> Option<u64> {
    let room = join_ms.saturating_sub(WRITE_RESERVE_MS);
    let ceiling = PUBLISH_GRACE_CEILING_MS - 1;
    let grace = if room < ceiling { room } else { ceiling };
    if grace == 0 || grace >= join_ms {
        None
    } else {
        Some(grace)
    }
}

/// CR-K: the derived per-clause retry-pass ceiling for a prover window.
#[must_use]
pub const fn clause_passes(window_ms: u64) -> u64 {
    let passes = window_ms / PASS_MIN_MS;
    let passes = if passes < 1 { 1 } else { passes };
    if passes > PASS_CEILING {
        PASS_CEILING
    } else {
        passes
    }
}

/// CR-K: the derived depth cap for a prover window, never below `floor`
/// (today's value) and never above [`DEPTH_CEILING`].
#[must_use]
pub const fn max_depth(window_ms: u64, floor: u64) -> u64 {
    let derived = (window_ms / 1_000).saturating_mul(DEPTH_PER_WINDOW_SEC);
    let derived = if derived > DEPTH_CEILING {
        DEPTH_CEILING
    } else {
        derived
    };
    if derived < floor {
        floor
    } else {
        derived
    }
}

/// Why the depth lift may or may not apply on a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthGate {
    /// A resident-queue byte budget is in force and fits the lease.
    Admit,
    /// No byte budget bounds the resident queue: depth is still doing memory
    /// work, so it is not lifted.
    NoQueueBudget,
    /// The byte budget exceeds `lease / QUEUE_LEASE_DEN`.
    BudgetPastLease,
    /// The host's static lease is unknown.
    LeaseUnknown,
}

impl DepthGate {
    /// Stable token for logs.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Admit => "admit",
            Self::NoQueueBudget => "no_queue_byte_budget",
            Self::BudgetPastLease => "queue_budget_past_lease",
            Self::LeaseUnknown => "lease_unknown",
        }
    }
}

/// CR-K: whether the depth lift is admitted, from the resident-queue byte
/// budget in force (`0` = none) and the host's static lease.
#[must_use]
pub const fn depth_lift_gate(queue_budget_bytes: u64, lease_bytes: u64) -> DepthGate {
    if lease_bytes == 0 {
        DepthGate::LeaseUnknown
    } else if queue_budget_bytes == 0 {
        DepthGate::NoQueueBudget
    } else if queue_budget_bytes > lease_bytes / QUEUE_LEASE_DEN {
        DepthGate::BudgetPastLease
    } else {
        DepthGate::Admit
    }
}
