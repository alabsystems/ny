// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Zero-when-off phase profiler for the margin-row BaB (#twinwall).
//!
//! Coarse wall-clock accounting of the tree-loop hot phases, gated by
//! `NY_MARGIN_ROW_PROFILE=1`. Timers do not directly supply a coefficient or
//! bound, but their clock reads and atomic updates can perturb a
//! deadline-sensitive verdict path. Aggregation is a fixed array of atomics keyed by [`Phase`].
//!
//! # The phases are NOT all driver-thread, and the totals are NOT all wall time
//!
//! This header used to claim "the phases are entered from the (single)
//! tree-loop driver thread, so contention is nil even though each phase
//! internally fans out over rayon." **That was false, and it silently inflated
//! every number this module has ever produced.**
//!
//! `Phase::EvalNode` starts its timer at `bab.rs:5011`, inside
//! `eval_with_pack`, reached from `expand_one` inside
//! `batch.par_iter().zip(packs.par_iter()).map(|(e, p)| this.expand_one(p, e))`
//! at `bab.rs:4207-4209` — a RAYON PARALLEL ITERATOR. `ScoreCands` is likewise
//! worker-entered. Since [`Timer`]'s `Drop` does `NS[phase].fetch_add(elapsed)`,
//! N concurrent workers each add their FULL elapsed time, so `NS[EvalNode]` is a
//! SUM OF OVERLAPPING INTERVALS, not occupied wall.
//!
//! The skew is systematic rather than uniform: worker-entered phases inflate by
//! roughly their concurrency while driver-entered phases (`RootGate`,
//! `RootEval`) do not, so the COMPARISON between phases is biased toward exactly
//! the phases this campaign has spent its effort optimising.
//!
//! [`dump`] therefore reports `max_inflight` per phase and labels each phase
//! CONCURRENT or DRIVER. A phase with `max_inflight > 1` must be read as a
//! summed duration; `sum / max_inflight` is a LOWER bound on its occupied wall
//! and the truth lies between the two. Never quote a percentage for a
//! CONCURRENT phase without saying which of the two you mean.

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

    // ---- #eval-subphase: NESTED breakdown of `EvalNode` (not in the total) ----
    //
    // Before this, ONE RAII timer at `bab.rs:eval_with_pack` covered
    // `domain_gates_acc`, the clip dual solves, `margin_seed`, the certified
    // backward pass, `per_class_direct`, `compose_viay`, and both ascents. The
    // certified pass in `engine.rs` carried ZERO `prof::` sites against 28 in
    // `bab.rs`, so 74.6% of the cifar100 lane was attributed entirely by
    // estimate. These variants make that attribution measurable.
    /// `domain_gates_acc` + `clip_tighten_gates` (gate rebuild per node).
    EvalGates = 10,
    /// `margin_seed` construction.
    EvalSeed = 11,
    /// The certified backward pass itself (`engine.rs` collect/seamed run).
    /// Runs 1x-11x per eval depending on the closure tier.
    EvalPass = 12,
    /// `compose_viay` — the only term cubic in all three shape parameters.
    EvalCompose = 13,
    /// The #margin-row-beta ascent, including its trial passes.
    EvalBeta = 14,
    /// The per-domain alpha ascent (`domain_alpha`).
    EvalAlpha = 15,
    /// `per_class_direct` -> `concretize_lower`.
    EvalDirect = 16,
}

/// Number of TOP-LEVEL tracked phases — the ones that partition the lane's
/// wall clock and therefore sum to the reported total.
pub const NPHASE_TOP: usize = 10;

/// Number of tracked phases including the #eval-subphase NESTED timers.
///
/// Sub-phases (index >= [`NPHASE_TOP`]) sit INSIDE `eval_node`, so they must not
/// be added to the total or every existing percentage would silently change.
/// `dump` reports them in their own section, as a breakdown OF `eval_node`
/// rather than a sibling of it, and the top-level figures stay bit-comparable
/// with every profile banked before this change.
pub const NPHASE: usize = 17;

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
    "  ├ eval:gates(dom+clip)",
    "  ├ eval:margin_seed",
    "  ├ eval:certified_pass",
    "  ├ eval:compose_viay",
    "  ├ eval:beta_ascent",
    "  ├ eval:alpha_ascent",
    "  └ eval:per_class_direct",
];

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static NS: [AtomicU64; NPHASE] = [ZERO; NPHASE];
static CNT: [AtomicU64; NPHASE] = [ZERO; NPHASE];
/// Timers of this phase currently in flight (incremented on start, decremented
/// on drop).
static INFLIGHT: [AtomicU64; NPHASE] = [ZERO; NPHASE];
/// High-water mark of [`INFLIGHT`] — the evidence that a phase's `NS` total sums
/// overlapping intervals. `1` means the total IS occupied wall.
static MAX_INFLIGHT: [AtomicU64; NPHASE] = [ZERO; NPHASE];

static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Latched RAW env string (lever-debt batch B1 preparation), read once through
/// the ny-levers chokepoint's raw view. [`Timer::start`] and [`bump`] sit in
/// the tree-loop hot path, so the STRING is latched; the DECISION is derived
/// per call in [`enabled`]. This remains process-wide; Phase 2 must replace it
/// with an injected per-run `LeverSet`.
fn env_raw() -> Option<&'static str> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    RAW.get_or_init(|| ny_levers::read_raw(&ny_levers::decls::telemetry::MARGIN_ROW_PROFILE))
        .as_deref()
}

/// True when `NY_MARGIN_ROW_PROFILE=1` (exact `"1"`, derived per call from the
/// latched raw string). Also refreshes the `ACTIVE` mirror that [`bump`]
/// polls lock-free.
#[inline]
pub fn enabled() -> bool {
    let on = env_raw() == Some("1");
    ACTIVE.store(on, Ordering::Relaxed);
    on
}

/// TEST-ONLY: force the `ACTIVE` mirror that [`bump`] polls.
///
/// The env latch is a process-wide `OnceLock`, so a test that needs to observe
/// the gated counters cannot get there by setting the variable. This touches
/// only the mirror — [`enabled`] still recomputes it from the latched string on
/// its next call, so no production path can be affected. Its only callers
/// are the uncharged-lane gpu_seam tests.
#[cfg(all(test, feature = "gpu-tests-uncharged"))]
pub(crate) fn force_active_for_test(on: bool) {
    ACTIVE.store(on, Ordering::Relaxed);
}

/// RAII phase timer. `None` (and no allocation) when both the profiler and
/// the #phase-cost census are off.
///
/// #phase-cost: every timer also opens the run-wide cost census span of the
/// census phase this lane phase maps to ([`census_phase`]) — the margin-row
/// lane's contribution to `<results>.flight.json`'s `phase_cost`. The census
/// span is `Send` and stack-neutral (the `EvalNode` timer rides inside an
/// `EvalSeeded` that rayon may finish on another worker), and it is
/// independent of `NY_MARGIN_ROW_PROFILE`: the profiler's own `NS`/`CNT`
/// accounting below still runs only when the profiler is on.
pub struct Timer {
    phase: usize,
    /// `Some` only while the `NY_MARGIN_ROW_PROFILE` profiler is on.
    start: Option<Instant>,
    _cost: ny_core::phase_cost::PhaseSpan<'static>,
}

/// The #phase-cost census phase a margin-row lane phase is charged to, or
/// `None` for a phase the census does not track separately (`loop_other`, the
/// epoch subtree, and the `eval_node` sub-phases other than the two ascents).
const fn census_phase(phase: Phase) -> Option<ny_core::phase_cost::CostPhase> {
    use ny_core::phase_cost::CostPhase;
    match phase {
        Phase::RootGate | Phase::RootEval => Some(CostPhase::RootBounds),
        Phase::YRefresh => Some(CostPhase::IntermRefresh),
        Phase::EvalNode => Some(CostPhase::ChildBound),
        Phase::HeadPrerank | Phase::TrunkShortlist | Phase::Tier0Rank => {
            Some(CostPhase::BranchPrescore)
        }
        Phase::ScoreCands => Some(CostPhase::BranchSim),
        Phase::EvalBeta => Some(CostPhase::BetaOpt),
        Phase::EvalAlpha => Some(CostPhase::AlphaOpt),
        Phase::LoopOther
        | Phase::EpochBuild
        | Phase::EvalGates
        | Phase::EvalSeed
        | Phase::EvalPass
        | Phase::EvalCompose
        | Phase::EvalDirect => None,
    }
}

impl Timer {
    /// Begin timing `phase`; returns `None` when both the profiler and the
    /// cost census are off for it.
    #[inline]
    pub fn start(phase: Phase) -> Option<Self> {
        let cost = census_phase(phase).map_or_else(
            ny_core::phase_cost::PhaseSpan::inert,
            ny_core::phase_cost::span,
        );
        if ACTIVE.load(Ordering::Relaxed) || enabled() {
            let idx = phase as usize;
            // Watermark the concurrency so `dump` can say whether this phase's
            // NS total is occupied wall or a sum of overlapping intervals.
            let now = INFLIGHT[idx].fetch_add(1, Ordering::Relaxed) + 1;
            MAX_INFLIGHT[idx].fetch_max(now, Ordering::Relaxed);
            Some(Self {
                phase: idx,
                start: Some(Instant::now()),
                _cost: cost,
            })
        } else if cost.is_armed() {
            Some(Self {
                phase: phase as usize,
                start: None,
                _cost: cost,
            })
        } else {
            None
        }
    }
}

impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        // Census-only timer: the census span closes itself.
        let Some(start) = self.start else {
            return;
        };
        #[allow(clippy::cast_possible_truncation)]
        let ns = start.elapsed().as_nanos() as u64;
        NS[self.phase].fetch_add(ns, Ordering::Relaxed);
        CNT[self.phase].fetch_add(1, Ordering::Relaxed);
        INFLIGHT[self.phase].fetch_sub(1, Ordering::Relaxed);
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
    /// GPU seam produced an AUTHORITATIVE pass (#margin-row-gpu).
    GpuSeamOk = 8,
    /// GPU seam refused; the caller ran the exact CPU pass (#margin-row-gpu).
    GpuSeamRefused = 9,
    /// GPU seam refused specifically because a SOUNDNESS GUARD tripped — the
    /// certified-error floor or the realization probe (#margin-row-gpu).
    ///
    /// This counter must stay at zero on a healthy device. A nonzero value is
    /// not a performance note; it means a dispatched certified payload failed a
    /// necessary condition and the seam must not be trusted until it is
    /// explained.
    GpuSeamGuardTrip = 10,
    /// DOMAIN-BATCHED certified GPU dispatches that published
    /// (#margin-row-gpu-batch). ONE increment per wide call, not per domain.
    ///
    /// This is the counter that distinguishes "the batched lane fired" from
    /// "the batched lane silently never ran" — the failure mode that cost a
    /// measurement cycle on the per-pass seam. Batched work does NOT bump
    /// `gpu_seam_ok`: that counter stays the per-pass seam's, so the two lanes
    /// remain separable in one profile.
    GpuBatchOk = 11,
    /// Domains SERVED by a published batched dispatch. `gpu_batch_domains /
    /// gpu_batch_ok` is the achieved mean batch width — the number the whole
    /// lane exists to raise, and the one to read before believing any timing.
    GpuBatchDomains = 12,
    /// Batched attempts that refused. Typed capacity refusals may be retried at
    /// a narrower width; a terminal refusal falls back to the per-pass seam and
    /// then to the exact CPU walk for those domains.
    GpuBatchRefused = 13,
    /// A batched dispatch was discarded because a SOUNDNESS GUARD tripped on
    /// one of its domains — that domain's certified-error floor or realization
    /// probe (#margin-row-gpu-batch).
    ///
    /// Like [`Counter::GpuSeamGuardTrip`] this must stay at zero on a healthy
    /// device, and it is recorded regardless of the profiling gate.
    GpuBatchGuardTrip = 14,
    /// Margin-row coefficient-batch trait calls attempted. This is deliberately
    /// separate from `ny_core::wide_lane_telemetry`, whose denominator belongs
    /// to the graph/BaB internal wide lane.
    GpuBatchAttempts = 15,
}
const NCTR: usize = 16;
const CTR_NAMES: [&str; NCTR] = [
    "lru_hit",
    "lru_miss",
    "trunk_split",
    "head_split",
    "frontier_batch",
    "frontier_popped",
    "epoch_attempt",
    "epoch_closed",
    "gpu_seam_ok",
    "gpu_seam_refused",
    "gpu_seam_guard_trip",
    "gpu_batch_ok",
    "gpu_batch_domains",
    "gpu_batch_refused",
    "gpu_batch_guard_trip",
    "gpu_batch_attempts",
];
static CTRS: [AtomicU64; NCTR] = [ZERO; NCTR];

/// Increment an event counter by `n` (no-op when profiling is off).
#[inline]
pub fn bump(c: Counter, n: u64) {
    if ACTIVE.load(Ordering::Relaxed) {
        CTRS[c as usize].fetch_add(n, Ordering::Relaxed);
    }
}

/// Increment an event counter REGARDLESS of the profiling gate.
///
/// Reserved for counters whose value is a SOUNDNESS signal rather than a
/// performance note ([`Counter::GpuSeamGuardTrip`],
/// [`Counter::GpuBatchGuardTrip`]): a guard trip must be observable in an
/// ordinary production run, not only under `NY_MARGIN_ROW_PROFILE=1`.
#[inline]
pub fn bump_always(c: Counter, n: u64) {
    CTRS[c as usize].fetch_add(n, Ordering::Relaxed);
}

/// Read one event counter.
#[inline]
pub fn counter(c: Counter) -> u64 {
    CTRS[c as usize].load(Ordering::Relaxed)
}

/// Reset all counters (call at the start of a profiled run).
pub fn reset() {
    for i in 0..NPHASE {
        NS[i].store(0, Ordering::Relaxed);
        CNT[i].store(0, Ordering::Relaxed);
        INFLIGHT[i].store(0, Ordering::Relaxed);
        MAX_INFLIGHT[i].store(0, Ordering::Relaxed);
    }
    for c in &CTRS {
        c.store(0, Ordering::Relaxed);
    }
}

/// Human-readable breakdown (seconds + call counts + % of tracked total).
pub fn dump() -> String {
    // #eval-subphase: the total is over TOP-LEVEL phases only. Sub-phases are
    // nested inside `eval_node`; adding them would double-count and move every
    // historical percentage.
    let mut total = 0u64;
    for a in NS.iter().take(NPHASE_TOP) {
        total += a.load(Ordering::Relaxed);
    }
    let total_s = total.max(1) as f64 / 1e9;
    let mut out = format!("margin-row profile (tracked total {total_s:.2}s):\n");
    for i in 0..NPHASE_TOP {
        let ns = NS[i].load(Ordering::Relaxed);
        let cnt = CNT[i].load(Ordering::Relaxed);
        let s = ns as f64 / 1e9;
        let pct = 100.0 * ns as f64 / total.max(1) as f64;
        let mx = MAX_INFLIGHT[i].load(Ordering::Relaxed).max(1);
        out.push_str(&format!(
            "  {:<28} {:>8.3}s  {:>6.1}%  n={:<7} {:.3}ms/call  {}\n",
            NAMES[i],
            s,
            pct,
            cnt,
            if cnt > 0 { 1e3 * s / cnt as f64 } else { 0.0 },
            if mx > 1 {
                format!(
                    "CONCURRENT max_inflight={mx} occupied_wall>={:.3}s ({:.3}ms/call)",
                    s / mx as f64,
                    if cnt > 0 {
                        1e3 * s / mx as f64 / cnt as f64
                    } else {
                        0.0
                    }
                )
            } else {
                "DRIVER".to_string()
            }
        ));
    }
    // Sub-phase breakdown, as a percentage OF `eval_node` rather than of the
    // lane, plus the residue that no sub-timer claimed. A large residue means
    // the breakdown is incomplete and must not be read as exhaustive.
    let eval_ns = NS[Phase::EvalNode as usize].load(Ordering::Relaxed);
    if eval_ns > 0 {
        let mut claimed = 0u64;
        for a in NS.iter().skip(NPHASE_TOP) {
            claimed += a.load(Ordering::Relaxed);
        }
        out.push_str(&format!(
            "  eval_node breakdown (nested; {:.1}% of eval_node claimed):\n",
            100.0 * claimed as f64 / eval_ns as f64
        ));
        for i in NPHASE_TOP..NPHASE {
            let ns = NS[i].load(Ordering::Relaxed);
            let cnt = CNT[i].load(Ordering::Relaxed);
            let sec = ns as f64 / 1e9;
            out.push_str(&format!(
                "  {:<28} {:>8.3}s  {:>6.1}%  n={:<7} {:.3}ms/call\n",
                NAMES[i],
                sec,
                100.0 * ns as f64 / eval_ns as f64,
                cnt,
                if cnt > 0 { 1e3 * sec / cnt as f64 } else { 0.0 }
            ));
        }
        let residue = eval_ns.saturating_sub(claimed);
        out.push_str(&format!(
            "  {:<28} {:>8.3}s  {:>6.1}%  (unattributed)\n",
            "  ? eval:residue",
            residue as f64 / 1e9,
            100.0 * residue as f64 / eval_ns as f64
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

#[cfg(test)]
mod concurrency_accounting_tests {
    use super::{Phase, Timer, ACTIVE, MAX_INFLIGHT, NS};
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Barrier};

    /// The watermark must actually catch overlap, or the CONCURRENT label is
    /// just another unverified prose claim like the one it replaced.
    ///
    /// A `Barrier` makes the overlap DETERMINISTIC — every worker holds a live
    /// `Timer` before any of them drops — so this cannot flake the way a
    /// sleep-based test would. The assertion is a LOWER bound (`>=`), which also
    /// makes it robust to other tests sharing these process-global atomics:
    /// interference can only ADD in-flight timers, never remove them.
    #[test]
    fn max_inflight_watermark_detects_overlapping_timers() {
        const WORKERS: usize = 4;
        let prev = ACTIVE.swap(true, Ordering::Relaxed);
        let idx = Phase::EvalNode as usize;
        MAX_INFLIGHT[idx].store(0, Ordering::Relaxed);

        let barrier = Arc::new(Barrier::new(WORKERS));
        let handles: Vec<_> = (0..WORKERS)
            .map(|_| {
                let b = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let _t = Timer::start(Phase::EvalNode);
                    // Every worker now holds a live timer; none may drop until
                    // all have started.
                    b.wait();
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker panicked");
        }

        let mx = MAX_INFLIGHT[idx].load(Ordering::Relaxed);
        ACTIVE.store(prev, Ordering::Relaxed);
        assert!(
            mx >= WORKERS as u64,
            "max_inflight was {mx}, expected >= {WORKERS}. Without this watermark \
             NS[EvalNode] is a sum of overlapping intervals that reads as occupied \
             wall — the exact defect that inflated the banked 2670 ms/call figure."
        );
        assert!(
            NS[idx].load(Ordering::Relaxed) > 0,
            "timers recorded no duration, so the watermark result is meaningless"
        );
    }

    /// `reset` must clear the new arrays too, or a second profiled run inherits
    /// the first run's watermark and silently reports the wrong concurrency.
    #[test]
    fn reset_clears_the_inflight_watermark() {
        let idx = Phase::ScoreCands as usize;
        MAX_INFLIGHT[idx].store(7, Ordering::Relaxed);
        super::reset();
        assert_eq!(
            MAX_INFLIGHT[idx].load(Ordering::Relaxed),
            0,
            "reset() left a stale watermark; the next run would misreport its concurrency"
        );
    }
}
