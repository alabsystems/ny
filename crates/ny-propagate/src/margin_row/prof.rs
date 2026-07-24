// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Zero-when-off phase profiler for the margin-row BaB (#twinwall).
//!
//! Coarse wall-clock accounting of the tree-loop hot phases, gated by
//! `NY_MARGIN_ROW_PROFILE=1`. Timers only READ the clock: they never touch a
//! coefficient, bound, or verdict, so enabling the profiler cannot move the
//! moat. Aggregation is a fixed array of atomics keyed by [`Phase`]; the
//! phases are entered from the (single) tree-loop driver thread, so contention
//! is nil even though each phase internally fans out over rayon.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Hot phases of the lane (index = discriminant).
#[derive(Copy, Clone, Debug)]
pub enum Phase {
    /// Root gate tableau build (`RootGates::build`).
    RootGate = 0,
    /// Root evaluation (`root_eval`: via-y all classes + direct fail subset).
    RootEval = 1,
    /// Per-domain y-row refresh (`rows_for`/`build_pack`: 2 backward passes).
    YRefresh = 2,
    /// Node re-bound (`eval_node`: seeded backward + compose_viay).
    EvalNode = 3,
    /// Head candidate pre-rank (`head_prerank`: variant_state + variants).
    HeadPrerank = 4,
    /// Trunk candidate shortlist.
    TrunkShortlist = 5,
    /// Batched candidate/child scoring (`score_candidates`: one wide pass).
    ScoreCands = 6,
    /// Everything else in the loop body (heap ops, push, bookkeeping).
    LoopOther = 7,
    /// Tier-0 unified candidate ranking (#epoch-bab: head variants + rank-1
    /// trunk variants over retained rows).
    Tier0Rank = 8,
    /// Tier-2 epoch rebuild + nested subtree run (#epoch-bab).
    EpochBuild = 9,
}

/// Number of tracked phases.
pub const NPHASE: usize = 10;

const NAMES: [&str; NPHASE] = [
    "root_gate_build",
    "root_eval",
    "y_refresh(2xbackward)",
    "eval_node(backward+compose)",
    "head_prerank(variants)",
    "trunk_shortlist",
    "score_candidates(batched)",
    "loop_other",
    "tier0_rank(variants)",
    "epoch(build+nested)",
];

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static NS: [AtomicU64; NPHASE] = [ZERO; NPHASE];
static CNT: [AtomicU64; NPHASE] = [ZERO; NPHASE];

static ON: OnceLock<bool> = OnceLock::new();
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// True when `NY_MARGIN_ROW_PROFILE=1` (checked once).
#[inline]
pub fn enabled() -> bool {
    *ON.get_or_init(|| {
        let on = std::env::var("NY_MARGIN_ROW_PROFILE").ok().as_deref() == Some("1");
        ACTIVE.store(on, Ordering::Relaxed);
        on
    })
}

/// RAII phase timer. `None` (and no allocation) when the profiler is off.
pub struct Timer {
    phase: usize,
    start: Instant,
}

impl Timer {
    /// Begin timing `phase`; returns `None` when profiling is disabled.
    #[inline]
    pub fn start(phase: Phase) -> Option<Self> {
        if ACTIVE.load(Ordering::Relaxed) || enabled() {
            Some(Self {
                phase: phase as usize,
                start: Instant::now(),
            })
        } else {
            None
        }
    }
}

impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        #[allow(clippy::cast_possible_truncation)]
        let ns = self.start.elapsed().as_nanos() as u64;
        NS[self.phase].fetch_add(ns, Ordering::Relaxed);
        CNT[self.phase].fetch_add(1, Ordering::Relaxed);
    }
}

/// Free-form event counters (index = [`Counter`]).
#[derive(Copy, Clone, Debug)]
pub enum Counter {
    /// y-row LRU hit.
    LruHit = 0,
    /// y-row LRU miss (rebuild).
    LruMiss = 1,
    /// Child pushed via a trunk split.
    TrunkSplit = 2,
    /// Child pushed via a head clamp.
    HeadSplit = 3,
    /// Frontier batches processed (parallel lane).
    FrontierBatch = 4,
    /// Domains popped into a frontier batch (parallel lane).
    FrontierPopped = 5,
    /// Tier-2 epoch attempted (#epoch-bab).
    EpochAttempt = 6,
    /// Tier-2 epoch closed its subtree.
    EpochClosed = 7,
}
const NCTR: usize = 8;
const CTR_NAMES: [&str; NCTR] = [
    "lru_hit",
    "lru_miss",
    "trunk_split",
    "head_split",
    "frontier_batch",
    "frontier_popped",
    "epoch_attempt",
    "epoch_closed",
];
static CTRS: [AtomicU64; NCTR] = [ZERO; NCTR];

/// Increment an event counter by `n` (no-op when profiling is off).
#[inline]
pub fn bump(c: Counter, n: u64) {
    if ACTIVE.load(Ordering::Relaxed) {
        CTRS[c as usize].fetch_add(n, Ordering::Relaxed);
    }
}

/// Reset all counters (call at the start of a profiled run).
pub fn reset() {
    for i in 0..NPHASE {
        NS[i].store(0, Ordering::Relaxed);
        CNT[i].store(0, Ordering::Relaxed);
    }
    for c in &CTRS {
        c.store(0, Ordering::Relaxed);
    }
}

/// Human-readable breakdown (seconds + call counts + % of tracked total).
pub fn dump() -> String {
    let mut total = 0u64;
    for a in &NS {
        total += a.load(Ordering::Relaxed);
    }
    let total_s = total.max(1) as f64 / 1e9;
    let mut out = format!("margin-row profile (tracked total {total_s:.2}s):\n");
    for i in 0..NPHASE {
        let ns = NS[i].load(Ordering::Relaxed);
        let cnt = CNT[i].load(Ordering::Relaxed);
        let s = ns as f64 / 1e9;
        let pct = 100.0 * ns as f64 / total.max(1) as f64;
        out.push_str(&format!(
            "  {:<28} {:>8.3}s  {:>6.1}%  n={:<7} {:.3}ms/call\n",
            NAMES[i],
            s,
            pct,
            cnt,
            if cnt > 0 { 1e3 * s / cnt as f64 } else { 0.0 }
        ));
    }
    out.push_str("  counters:");
    for i in 0..NCTR {
        out.push_str(&format!(
            " {}={}",
            CTR_NAMES[i],
            CTRS[i].load(Ordering::Relaxed)
        ));
    }
    out.push('\n');
    out
}
