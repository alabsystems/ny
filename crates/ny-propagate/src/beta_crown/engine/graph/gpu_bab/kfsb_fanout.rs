// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #kfsb-production: the shared memory budget, the lease, the pool and the
//! capped observable of the #kfsb-single-shot candidate fan-out.
//!
//! # The defect this closes (review(dag-throughput), gap 3)
//!
//! The single-shot scorer simulates its `k` candidates in PARALLEL. Every
//! simulation is a constrained CROWN pass whose dense materializations are
//! admitted by `crown_memory::cpu_crown_dense_budget_bytes` — a share of LIVE headroom
//! (`headroom / 8`), sized for ONE pass: an advertised dense pair can peak at
//! four times its bytes, and `/ 8` leaves half the headroom after that peak.
//! Read `k` times at once (up to `2k` passes in flight), that figure let `k`
//! passes each claim it. On this unified-memory host — and on a competition
//! machine, which runs ny with no cgroup and so no governor — an
//! over-admission is a global OOM, not a refusal.
//!
//! # The contract
//!
//! The fan-out shares ONE budget ([`plan_kfsb_fanout`]):
//!
//! * `total` = the live dense budget, read ONCE per decision on the
//!   selecting thread — exactly what one dark pass may claim — or, when the
//!   router's R4 memory lease is armed, a LEASE from its envelope
//!   ([`LeaseHolder::KfsbFanout`], narrowable): `min(live, leased / 4)`. A
//!   denied lease runs ONE simulation at a time with the live budget: exactly
//!   the dark scorer's serial pass, inside the terminal's own lease.
//! * `width` = simulations in flight at once: `min(candidates, threads,
//!   total / min_share)`, at least 1 — the fan-out CAPS CONCURRENCY BY BYTES
//!   before it gives any simulation less than `min_share`.
//! * `share` = `total / width`: each simulation's dense budget, installed on
//!   EVERY worker of a pool the fan-out owns ([`run_shared_budget_fanout`]),
//!   so a nested Rayon job a pass spawns sees it too. The candidates run in
//!   waves of `width`, so at most `width` simulations are live at once even
//!   when a worker blocked in a nested join steals a top-level task.
//!
//! Invariant (pinned by test): `width * share <= total <= live`, and with a
//! lease `total * 4 <= leased`. The fan-out never claims more dense budget
//! than ONE dark pass does.
//!
//! # Scope of the guarantee (review(kfsb-production))
//!
//! The ONE budget is per DECISION, not per process. Both production callers
//! decide one domain at a time (the DomainList lane's per-domain loop and the
//! conjunctive multi-objective lane's sequential path), so there one decision
//! is in flight and the bound holds process-wide. The disjunctive batched
//! multi-objective lane builds children in a Rayon `par_iter`
//! (`batched_multi.rs`), and its kFSB route is reached only through the raw
//! `NY_MO_SCORER_FIX=1` (never in competition): there every concurrent
//! decision reads the live budget, plans its own `total`, and builds its own
//! pool — `n` decisions claim up to `n` budgets, as `n` dark passes would.
//! Likewise the R4 lease is HOLDER-keyed (`LeaseAuthority::acquire` hands an
//! already-held lease back unchanged): two lineages running armed lanes at
//! once would size their fan-outs from ONE 16 GiB lease, and the first to
//! finish returns it while the other still sizes by it. No production path
//! runs two armed lanes at once today; a caller that does needs a per-lane
//! holder or a reference count first.
//!
//! # Never a verdict input
//!
//! Budgets, leases and pools are admission and scheduling. A smaller share
//! can only make a simulation take its cheaper (looser, still sound)
//! representation, which moves an ADVISORY branching score. The selector
//! returns `(node, neuron)`; the committed children partition the parent for
//! any pick and are bounded by the unchanged pipeline.
//!
//! # The observable
//!
//! The first [`KFSB_SINGLE_SHOT_DECISION_LINES`] armed decisions print their
//! `[kfsb-single-shot] armed=1 ...` line; later ones are counted, not
//! printed. Each armed lane prints ONE `[kfsb-single-shot] summary ...` line
//! when it ends ([`KfsbSingleShotLaneScope`]), with the lane's totals: the
//! decisions, the printed / suppressed lines, the simulations by route (GPU,
//! CPU, GPU refused), the fan-out widths and shares, and the lease.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use ny_core::memory_lease::LeaseAuthority;
use ny_core::router::lease::{
    LeaseDecision, LeaseHolder, DENSE_PAIR_PEAK_FACTOR, KFSB_FANOUT_MIN_SHARE_BYTES,
};
use rayon::prelude::*;

use crate::network::crown_memory::DenseBudgetThreadCap;

use super::super::super::super::BetaCrownVerifier;

/// Armed decisions that print their own `[kfsb-single-shot] armed=1` line
/// per verifier lineage; later decisions are counted in the lane summary.
pub(crate) const KFSB_SINGLE_SHOT_DECISION_LINES: u64 = 32;

/// The least dense budget one simulation is given (bytes).
pub(crate) const KFSB_MIN_SHARE_BYTES: usize = KFSB_FANOUT_MIN_SHARE_BYTES as usize;

/// Where a fan-out's total came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FanoutBudgetSource {
    /// No lease armed (every run without `router.mode: on` and CR-L): the
    /// live dense budget.
    Live,
    /// An R4 lease was granted (`narrowed`: only part of the want fit).
    Lease {
        /// Resident bytes leased.
        leased_bytes: u64,
        /// Whether the lease was narrowed.
        narrowed: bool,
    },
    /// An R4 lease was denied: one simulation at a time, live budget.
    LeaseDenied,
}

impl FanoutBudgetSource {
    fn token(self) -> String {
        match self {
            Self::Live => "live".to_owned(),
            Self::Lease {
                leased_bytes,
                narrowed,
            } => format!(
                "{}:{:.2}GiB",
                if narrowed { "narrowed" } else { "granted" },
                leased_bytes as f64 / (1u64 << 30) as f64
            ),
            Self::LeaseDenied => "denied".to_owned(),
        }
    }
}

/// One decision's fan-out plan (pure arithmetic; see the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KfsbFanoutPlan {
    /// The live dense budget read once on the selecting thread.
    pub(crate) live_bytes: usize,
    /// The ONE budget the fan-out shares (`<= live_bytes`).
    pub(crate) total_bytes: usize,
    /// Simulations in flight at once (`>= 1`).
    pub(crate) width: usize,
    /// Each simulation's dense budget (`width * share_bytes <= total_bytes`).
    pub(crate) share_bytes: usize,
    /// Waves the candidates run in (`ceil(candidates / width)`).
    pub(crate) waves: usize,
    /// Where `total_bytes` came from.
    pub(crate) source: FanoutBudgetSource,
}

/// Plan one fan-out. PURE and total.
///
/// `live_bytes`: the live dense budget (one dark pass's claim). `lease`: the
/// R4 lease decision when the lease authority is armed. `candidates`: the
/// simulations to run. `threads`: the worker threads available.
/// `min_share_bytes`: the least share a simulation is given when more than
/// one runs at once.
#[must_use]
pub(crate) fn plan_kfsb_fanout(
    live_bytes: usize,
    lease: Option<LeaseDecision>,
    candidates: usize,
    threads: usize,
    min_share_bytes: usize,
) -> KfsbFanoutPlan {
    let (total_bytes, source, serial) = match lease {
        None => (live_bytes, FanoutBudgetSource::Live, false),
        Some(LeaseDecision::Denied { .. }) => (live_bytes, FanoutBudgetSource::LeaseDenied, true),
        Some(decision) => {
            let leased_bytes = decision.bytes();
            let covered =
                usize::try_from(leased_bytes / DENSE_PAIR_PEAK_FACTOR).unwrap_or(usize::MAX);
            (
                live_bytes.min(covered),
                FanoutBudgetSource::Lease {
                    leased_bytes,
                    narrowed: matches!(decision, LeaseDecision::Narrowed { .. }),
                },
                false,
            )
        }
    };
    let by_bytes = (total_bytes / min_share_bytes.max(1)).max(1);
    let width = if serial {
        1
    } else {
        candidates.min(threads.max(1)).min(by_bytes).max(1)
    };
    KfsbFanoutPlan {
        live_bytes,
        total_bytes,
        width,
        share_bytes: total_bytes / width,
        waves: candidates.div_ceil(width),
        source,
    }
}

/// Run `work` over `items` under `plan`: in waves of `plan.width` on a pool
/// the fan-out owns, with `plan.share_bytes` installed as the dense-budget
/// share on EVERY worker of that pool. Results come back in item order.
///
/// The pool has as many workers as the global pool (`threads`), so a
/// simulation's own nested parallelism is what it was before; the WAVES bound
/// how many simulations are live at once. If the pool cannot be built, the
/// items run serially on the caller with the share installed there.
pub(crate) fn run_shared_budget_fanout<T, R, F>(
    items: &[T],
    plan: &KfsbFanoutPlan,
    threads: usize,
    work: F,
) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    if items.is_empty() {
        return Vec::new();
    }
    let width = plan.width.max(1);
    let share = plan.share_bytes;
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(width).max(1))
        .thread_name(|index| format!("ny-kfsb-fanout-{index}"))
        .build_scoped(
            |thread| {
                let _share = DenseBudgetThreadCap::install(share);
                thread.run();
            },
            |pool| {
                pool.install(|| {
                    let mut out = Vec::with_capacity(items.len());
                    for wave in items.chunks(width) {
                        let results: Vec<R> = wave.par_iter().map(&work).collect();
                        out.extend(results);
                    }
                    out
                })
            },
        );
    match built {
        Ok(out) => out,
        Err(_) => {
            let _share = DenseBudgetThreadCap::install(share);
            items.iter().map(&work).collect()
        }
    }
}

/// The lease decision a decision's fan-out runs under, and whether this
/// decision must return it afterwards (no lane scope holds it).
pub(crate) struct FanoutLease {
    pub(crate) decision: Option<LeaseDecision>,
    release_after: bool,
}

/// Per-verifier-lineage counters of the armed single-shot scorer (shared
/// with every sub-verifier `with_config_from` builds), its line cap, and the
/// lane's R4 lease. Telemetry and admission only.
#[derive(Debug, Default)]
pub(crate) struct KfsbSingleShotTelemetry {
    decisions: AtomicU64,
    printed: AtomicU64,
    suppressed: AtomicU64,
    sims: AtomicU64,
    infeasible: AtomicU64,
    cpu_sims: AtomicU64,
    gpu_sims: AtomicU64,
    gpu_refused: AtomicU64,
    waves: AtomicU64,
    width_sum: AtomicU64,
    width_max: AtomicU64,
    /// Smallest share seen (0 = none yet).
    share_min_plus_one: AtomicU64,
    share_max: AtomicU64,
    prescore_ns: AtomicU64,
    sim_ns: AtomicU64,
    pool_ns: AtomicU64,
    lease: Mutex<LaneLease>,
}

#[derive(Debug, Default)]
struct LaneLease {
    /// Armed lane scopes open now (nested sub-verifier lanes share one).
    depth: usize,
    /// The lane's lease decision, asked once per outermost lane.
    decision: Option<LeaseDecision>,
}

/// A snapshot of the counters (deltas are per lane).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct KfsbSingleShotCounts {
    pub(crate) decisions: u64,
    pub(crate) printed: u64,
    pub(crate) suppressed: u64,
    pub(crate) sims: u64,
    pub(crate) infeasible: u64,
    pub(crate) cpu_sims: u64,
    pub(crate) gpu_sims: u64,
    pub(crate) gpu_refused: u64,
    pub(crate) waves: u64,
    pub(crate) width_sum: u64,
    pub(crate) prescore_ns: u64,
    pub(crate) sim_ns: u64,
    pub(crate) pool_ns: u64,
}

impl KfsbSingleShotCounts {
    fn since(self, before: Self) -> Self {
        Self {
            decisions: self.decisions.saturating_sub(before.decisions),
            printed: self.printed.saturating_sub(before.printed),
            suppressed: self.suppressed.saturating_sub(before.suppressed),
            sims: self.sims.saturating_sub(before.sims),
            infeasible: self.infeasible.saturating_sub(before.infeasible),
            cpu_sims: self.cpu_sims.saturating_sub(before.cpu_sims),
            gpu_sims: self.gpu_sims.saturating_sub(before.gpu_sims),
            gpu_refused: self.gpu_refused.saturating_sub(before.gpu_refused),
            waves: self.waves.saturating_sub(before.waves),
            width_sum: self.width_sum.saturating_sub(before.width_sum),
            prescore_ns: self.prescore_ns.saturating_sub(before.prescore_ns),
            sim_ns: self.sim_ns.saturating_sub(before.sim_ns),
            pool_ns: self.pool_ns.saturating_sub(before.pool_ns),
        }
    }
}

/// One decision, as recorded.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct KfsbDecisionRecord {
    pub(crate) sims: usize,
    pub(crate) infeasible: usize,
    pub(crate) cpu_sims: usize,
    pub(crate) gpu_sims: usize,
    pub(crate) gpu_refused: usize,
    pub(crate) prescore_secs: f64,
    pub(crate) sim_secs: f64,
    pub(crate) pool_secs: f64,
    pub(crate) plan: Option<KfsbFanoutPlan>,
    /// Test probe: the largest dense budget any simulation's worker saw.
    #[cfg(test)]
    pub(crate) max_budget_seen: usize,
}

fn nanos(secs: f64) -> u64 {
    if secs.is_finite() && secs > 0.0 {
        (secs * 1e9) as u64
    } else {
        0
    }
}

impl KfsbSingleShotTelemetry {
    /// The counters now.
    pub(crate) fn snapshot(&self) -> KfsbSingleShotCounts {
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        KfsbSingleShotCounts {
            decisions: load(&self.decisions),
            printed: load(&self.printed),
            suppressed: load(&self.suppressed),
            sims: load(&self.sims),
            infeasible: load(&self.infeasible),
            cpu_sims: load(&self.cpu_sims),
            gpu_sims: load(&self.gpu_sims),
            gpu_refused: load(&self.gpu_refused),
            waves: load(&self.waves),
            width_sum: load(&self.width_sum),
            prescore_ns: load(&self.prescore_ns),
            sim_ns: load(&self.sim_ns),
            pool_ns: load(&self.pool_ns),
        }
    }

    /// Record one decision. Returns whether its line may be printed (the
    /// first [`KFSB_SINGLE_SHOT_DECISION_LINES`] of this lineage).
    // trust-1.99 deprecates `fetch_update` (renamed `try_update`); the public
    // 1.95 pin lacks `try_update` — keep the spelling both toolchains accept.
    #[allow(deprecated)]
    pub(crate) fn record_decision(&self, record: &KfsbDecisionRecord) -> bool {
        let add = |a: &AtomicU64, v: usize| {
            a.fetch_add(u64::try_from(v).unwrap_or(u64::MAX), Ordering::Relaxed);
        };
        self.decisions.fetch_add(1, Ordering::Relaxed);
        add(&self.sims, record.sims);
        add(&self.infeasible, record.infeasible);
        add(&self.cpu_sims, record.cpu_sims);
        add(&self.gpu_sims, record.gpu_sims);
        add(&self.gpu_refused, record.gpu_refused);
        self.prescore_ns
            .fetch_add(nanos(record.prescore_secs), Ordering::Relaxed);
        self.sim_ns
            .fetch_add(nanos(record.sim_secs), Ordering::Relaxed);
        self.pool_ns
            .fetch_add(nanos(record.pool_secs), Ordering::Relaxed);
        if let Some(plan) = record.plan {
            add(&self.waves, plan.waves);
            add(&self.width_sum, plan.width);
            let width = u64::try_from(plan.width).unwrap_or(u64::MAX);
            self.width_max.fetch_max(width, Ordering::Relaxed);
            let share = u64::try_from(plan.share_bytes).unwrap_or(u64::MAX);
            self.share_max.fetch_max(share, Ordering::Relaxed);
            let _ = self.share_min_plus_one.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |current| {
                    let candidate = share.saturating_add(1);
                    (current == 0 || candidate < current).then_some(candidate)
                },
            );
        }
        let printed = self
            .printed
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n < KFSB_SINGLE_SHOT_DECISION_LINES).then_some(n + 1)
            })
            .is_ok();
        if !printed {
            self.suppressed.fetch_add(1, Ordering::Relaxed);
        }
        printed
    }

    /// The lease this decision's fan-out runs under. Unarmed authority (every
    /// run without `router.mode: on` and CR-L): `None`, nothing asked. Inside
    /// an armed lane scope the lane asks ONCE and keeps the answer (a denial
    /// is not re-asked per decision, so the ledger is not flooded); outside
    /// one, the decision acquires and must return it ([`Self::finish_lease`]).
    pub(crate) fn fanout_lease(&self, authority: &LeaseAuthority) -> FanoutLease {
        if !authority.is_armed() {
            return FanoutLease {
                decision: None,
                release_after: false,
            };
        }
        let mut lane = self.lease.lock().unwrap_or_else(PoisonError::into_inner);
        if lane.depth > 0 {
            if lane.decision.is_none() {
                lane.decision = authority.acquire(LeaseHolder::KfsbFanout);
            }
            return FanoutLease {
                decision: lane.decision,
                release_after: false,
            };
        }
        drop(lane);
        let decision = authority.acquire(LeaseHolder::KfsbFanout);
        FanoutLease {
            decision,
            release_after: decision.is_some_and(LeaseDecision::admits),
        }
    }

    /// Return a per-decision lease (a no-op for a lane-held one).
    pub(crate) fn finish_lease(&self, authority: &LeaseAuthority, lease: &FanoutLease) {
        if lease.release_after {
            let _ = authority.release(LeaseHolder::KfsbFanout);
        }
    }

    fn enter_lane(&self) {
        let mut lane = self.lease.lock().unwrap_or_else(PoisonError::into_inner);
        if lane.depth == 0 {
            lane.decision = None;
            // review(kfsb-production): the summary's width / share extremes
            // are PER LANE like its deltas. They are running extremes, which
            // a snapshot cannot subtract, so the outermost lane resets them
            // when it opens; before this, a lineage's second lane (the next
            // clause or disjunct on a `with_config_from` sub-verifier)
            // reported the first lane's widths and shares as its own. A
            // nested lane reports the enclosing lane's extremes so far.
            self.width_max.store(0, Ordering::Relaxed);
            self.share_min_plus_one.store(0, Ordering::Relaxed);
            self.share_max.store(0, Ordering::Relaxed);
        }
        lane.depth = lane.depth.saturating_add(1);
    }

    /// Leave a lane scope; the outermost one returns the lane's lease.
    /// Returns the lane's lease decision for the summary.
    fn leave_lane(&self, authority: &LeaseAuthority) -> Option<LeaseDecision> {
        let mut lane = self.lease.lock().unwrap_or_else(PoisonError::into_inner);
        let decision = lane.decision;
        lane.depth = lane.depth.saturating_sub(1);
        if lane.depth == 0 {
            if decision.is_some_and(LeaseDecision::admits) {
                let _ = authority.release(LeaseHolder::KfsbFanout);
            }
            lane.decision = None;
        }
        decision
    }

    fn share_range(&self) -> (u64, u64) {
        let min = self
            .share_min_plus_one
            .load(Ordering::Relaxed)
            .saturating_sub(1);
        (min, self.share_max.load(Ordering::Relaxed))
    }
}

/// The per-decision `[kfsb-single-shot] armed=1 ...` line.
pub(crate) fn decision_line(
    depth: usize,
    unstable: usize,
    scored: usize,
    candidates: usize,
    record: &KfsbDecisionRecord,
    pick: Option<(&str, usize)>,
) -> String {
    let pick = pick.map_or_else(|| "none".to_string(), |(name, idx)| format!("{name}:{idx}"));
    let mib = |bytes: usize| bytes as f64 / (1u64 << 20) as f64;
    let (width, share_mib, total_mib, live_mib, source) = record.plan.map_or_else(
        || (0, 0.0, 0.0, 0.0, "-".to_owned()),
        |plan| {
            (
                plan.width,
                mib(plan.share_bytes),
                mib(plan.total_bytes),
                mib(plan.live_bytes),
                plan.source.token(),
            )
        },
    );
    format!(
        "[kfsb-single-shot] armed=1 depth={depth} unstable={unstable} scored={scored} \
         candidates={candidates} sims={} infeasible={} prescore_ms={:.1} sim_ms={:.1} pick={pick} \
         gpu={} cpu={} gpu_refused={} width={width} share_mib={share_mib:.0} \
         total_mib={total_mib:.0} live_mib={live_mib:.0} budget={source} pool_ms={:.2}",
        record.sims,
        record.infeasible,
        record.prescore_secs * 1e3,
        record.sim_secs * 1e3,
        record.gpu_sims,
        record.cpu_sims,
        record.gpu_refused,
        record.pool_secs * 1e3,
    )
}

/// One `[kfsb-single-shot] summary lane=... ` line per armed lane, with this
/// lineage's deltas over the lane, when dropped. Built only when the key is
/// armed, so a dark run prints nothing and asks no lease.
pub(crate) struct KfsbSingleShotLaneScope {
    telemetry: Arc<KfsbSingleShotTelemetry>,
    before: KfsbSingleShotCounts,
    lane: &'static str,
    started: Instant,
}

impl KfsbSingleShotLaneScope {
    pub(crate) fn enter(verifier: &BetaCrownVerifier, lane: &'static str) -> Option<Self> {
        if !verifier.config.kfsb_single_shot {
            return None;
        }
        let telemetry = Arc::clone(&verifier.kfsb_single_shot_telemetry);
        telemetry.enter_lane();
        Some(Self {
            before: telemetry.snapshot(),
            telemetry,
            lane,
            started: Instant::now(),
        })
    }
}

/// The summary line (pure formatting; pinned by test).
pub(crate) fn summary_line(
    lane: &str,
    delta: &KfsbSingleShotCounts,
    share_range: (u64, u64),
    width_max: u64,
    lease: Option<LeaseDecision>,
    lane_secs: f64,
) -> String {
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    let mean_width = if delta.decisions == 0 {
        0.0
    } else {
        delta.width_sum as f64 / delta.decisions as f64
    };
    let lease = lease.map_or_else(
        || "none".to_owned(),
        |d| {
            format!(
                "{}:{:.2}GiB",
                d.token(),
                d.bytes() as f64 / (1u64 << 30) as f64
            )
        },
    );
    format!(
        "[kfsb-single-shot] summary lane={lane} armed=1 decisions={} printed={} suppressed={} \
         sims={} infeasible={} gpu={} cpu={} gpu_refused={} waves={} width_mean={mean_width:.1} \
         width_max={width_max} share_min_mib={:.0} share_max_mib={:.0} lease={lease} \
         prescore_s={:.2} sim_s={:.2} pool_s={:.3} lane_s={lane_secs:.1}",
        delta.decisions,
        delta.printed,
        delta.suppressed,
        delta.sims,
        delta.infeasible,
        delta.gpu_sims,
        delta.cpu_sims,
        delta.gpu_refused,
        delta.waves,
        mib(share_range.0),
        mib(share_range.1),
        delta.prescore_ns as f64 / 1e9,
        delta.sim_ns as f64 / 1e9,
        delta.pool_ns as f64 / 1e9,
    )
}

impl Drop for KfsbSingleShotLaneScope {
    fn drop(&mut self) {
        let lease = self.telemetry.leave_lane(ny_core::memory_lease::global());
        let delta = self.telemetry.snapshot().since(self.before);
        eprintln!(
            "{}",
            summary_line(
                self.lane,
                &delta,
                self.telemetry.share_range(),
                self.telemetry.width_max.load(Ordering::Relaxed),
                lease,
                self.started.elapsed().as_secs_f64(),
            )
        );
    }
}

#[cfg(test)]
impl KfsbSingleShotTelemetry {
    /// Test seam: enter / leave a lane against an explicit authority.
    pub(crate) fn enter_lane_for_tests(&self) {
        self.enter_lane();
    }

    pub(crate) fn leave_lane_for_tests(&self, authority: &LeaseAuthority) -> Option<LeaseDecision> {
        self.leave_lane(authority)
    }

    /// Test seam: the `(share_min, share_max)` and `width_max` the lane
    /// summary prints.
    pub(crate) fn lane_extremes_for_tests(&self) -> ((u64, u64), u64) {
        (self.share_range(), self.width_max.load(Ordering::Relaxed))
    }
}
