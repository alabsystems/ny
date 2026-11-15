// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Breakpoint dual search ingest. Search sees borrowed, immutable row data;
//! only multipliers cross back. The lane recomputes the reduced costs and
//! rounding charge from its own problem. Backend scores never enter a bound.

use ny_core::search::{DualSearchRequest, DualSearchRow, SearchBackend};

use super::{Candidate, FrozenDualProblem, NodeBound, NodeScratch, Side};

/// Per-worker admission, including row descriptors, the certifier scratch,
/// returned hints, and the backend's linear-size workspace. This deliberately
/// exceeds the CPU backend's own estimate; no dense depth-by-generators copy.
pub(super) fn work_bytes(n: usize, depth: usize) -> u64 {
    (n as u64)
        .saturating_mul(128)
        .saturating_add((depth as u64).saturating_mul(1024))
        .saturating_add(16 * 1024)
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Stats {
    pub skipped: u64,
    pub calls: u64,
    pub steps: u64,
    pub accepted: u64,
    pub prunes: u64,
    pub refused: u64,
    pub support: ny_core::search::SupportSearchStats,
}

pub(super) struct Accepted {
    pub bound: NodeBound,
    pub l0: Vec<f64>,
    pub l1: Vec<f64>,
}

/// An INGEST GATE. Length, finiteness and non-negativity are checked before
/// the unchanged certified evaluator accepts a strictly better candidate.
/// Missing backends, refusals, panics and malformed hints preserve the node.
///
/// The automatic route limits search time while leaving the certified replay
/// governed by the class deadline. A completed hint is still useful after its
/// five-millisecond search slice has ended.
pub(super) fn refine_with_policy(
    prob: &FrozenDualProblem,
    sides: &[Side],
    l0: &[f64],
    l1: &[f64],
    incumbent: f64,
    steps: usize,
    automatic: bool,
    deadline: Option<std::time::Instant>,
    backend: Option<&dyn SearchBackend>,
    scratch: &mut NodeScratch,
    stats: &mut Stats,
) -> Option<Accepted> {
    let backend = backend?;
    if steps == 0 || deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return None;
    }
    let warm = prob.eval_node(sides, l0, l1, scratch, None);
    if !warm.value.is_finite() || deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return None;
    }
    let rows: Vec<_> = sides
        .iter()
        .enumerate()
        .map(|(i, &side)| {
            let row = prob.split_row(i);
            let si = usize::from(side == Side::On);
            DualSearchRow {
                a: row.a,
                sign: side.sign(),
                b0: row.b0[si],
                b1: row.b1[si],
                vacuous: row.vacuous[si],
            }
        })
        .collect();
    let request = DualSearchRequest {
        rc: scratch.rc(),
        rows: &rows,
        l0,
        l1,
        steps: steps.min(32),
        deadline: if automatic {
            super::automatic::search_deadline(deadline, std::time::Instant::now())
        } else {
            deadline
        },
        mem_cap_bytes: work_bytes(prob.n(), sides.len()),
    };
    stats.calls += 1;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.refine_dual(&request)
    }));
    let Ok(Ok(result)) = result else {
        stats.refused += 1;
        return None;
    };
    // gain and stop are telemetry from search, never pruning evidence.
    stats.support.merge(result.support);
    stats.steps += result.steps.min(request.steps) as u64;
    let mut hint = result.multipliers.into_hint().into_ingest();
    let depth = sides.len();
    if hint.l0.len() != depth
        || hint.l1.len() != depth
        || hint.l0.iter().chain(&hint.l1).any(|v| !v.is_finite())
    {
        stats.refused += 1;
        return None;
    }
    for (i, row) in rows.iter().enumerate() {
        hint.l0[i] = hint.l0[i].max(0.0);
        hint.l1[i] = if row.vacuous {
            0.0
        } else {
            hint.l1[i].max(0.0)
        };
    }
    drop(rows);
    // A hint that arrives after the budget cannot affect this level's cover.
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return None;
    }
    let bound = prob.eval_node(sides, &hint.l0, &hint.l1, scratch, None);
    if bound.certified <= incumbent
        || !bound.certified.is_finite()
        || deadline.is_some_and(|d| std::time::Instant::now() >= d)
    {
        return None;
    }
    stats.accepted += 1;
    stats.prunes += u64::from(bound.prunes());
    Some(Accepted {
        bound,
        l0: hint.l0,
        l1: hint.l1,
    })
}

/// Refined candidates are recorded with their actual multiplier vectors.
pub(super) const CANDIDATE: Candidate = Candidate::Breakpoint;
