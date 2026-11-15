// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #iso-edge-cap: bound the SERIAL edge work of the grouped-disjunctive
//! input-split BaB — the α-slope edge pass and the edge-MILP escalation — by
//! the scored stop and by a per-wave wall allowance.
//!
//! # The defect (MEASURED on the Mac, `624477367`, 08-30 resweep)
//!
//! On `isomorphic_acasxu_2026` six of the eighteen gap rows show ONE batch
//! lasting 15-63 s (`i12` 63.2 s, `i10` 49.7 s, `i18` 18.1 s, `i19` 16.2 s,
//! `i39` 15.0 s, `i20` 14.7 s) where a healthy batch is ~1.6 s, and every stall
//! starts when the tree's head depth first reaches ~20 — the edge gate
//! (`edge_domain_rows`: depth >= `input_split_edge_milp_depth` and every open
//! row within `input_split_edge_milp_gap`). `i18`'s stall ran the process to
//! 100.96 s against a 100 s budget, because `BabWall::cannot_finish` predicts
//! the next batch from the PREVIOUS one and cannot see a stall coming.
//!
//! Both edge helpers took `lifecycle.deadline(bab_wall.horizon())` — the 1.4x
//! inflated EFFORT horizon, not the scored stop — and nothing bounded how much
//! of one wave they could spend. The in-tree warning at
//! `child_bound_deadline` ("STILL UNCAPPED ... the edge-α pass, the edge-MILP
//! oracle escalation ... re-opens the same defect") named exactly this.
//!
//! # What the key changes when armed (`bab.branching.input_split.edge_wave_cap`)
//!
//! 1. **The stop, not the horizon, is the edge work's AUTHORITY.** Every new
//!    edge step is refused once the scored stop has passed, every edge step's
//!    WORK deadline is at or before the stop, and an edge-MILP proof completed
//!    at or after the stop is declined.
//! 2. **A per-wave allowance** of `min(max_secs, rebound_share x T_rebound)`
//!    seconds of edge wall per wave, where `T_rebound` is the batched rebound
//!    that immediately precedes this wave's per-domain loop (the freshest
//!    measurement of what one wave of ordinary work costs on this row, this
//!    host). Defaults `max_secs = 2.0`, `rebound_share = 0.25` (WAVE6_PLAN
//!    §7.C6 change 2). A new edge step starts only while the wave's spent edge
//!    wall is below the allowance, and its work deadline is
//!    `min(stop, now + remaining allowance)`.
//! 3. **The second node-bound collection is gone when edge-α ran.** Edge-α's
//!    own `collect_intermediate_bounds` over the SAME exact input box is handed
//!    to the edge-MILP request instead of recomputing it, so the oracle's
//!    free-binary gate (`graph_mip_leaf.rs`, cap 96) runs without a second
//!    collection.
//!
//! BOUND QUALITY IS DELIBERATELY UNCHANGED. The per-node CROWN-IBP grants are
//! a share of the time remaining to the deadline they are handed and are
//! DROPPED under a 2 s floor (`budget_policy::compute_global_per_node_budget_secs`,
//! `MIN_PER_NODE_BUDGET_SECS`). Handing the collection a sub-2 s work deadline
//! would therefore silently degrade the edge pass to IBP-grade intermediates.
//! So the EFFORT deadline (the horizon, exactly as before) still shapes every
//! bound — the collection, the α-state init, the final per-row CROWN pass —
//! and only the ITERATIVE work (the α optimisation loop, the MILP solve slice)
//! and the decision to START a step are bounded by the allowance.
//!
//! KNOWN CONSEQUENCE, stated so a null is not misread: the ny-cli Graph-MIP
//! leaf oracle sizes its solve slice as `min(10 s, remaining / 4)` and declines
//! any slice under 1 s (`GraphMipLeafSolver::admit_slice`). A work deadline
//! under 4 s away is therefore always declined there, so with the default
//! `max_secs = 2` the edge-MILP consult is DE FACTO OFF under the cap (the
//! collection it used to pay is gone too when edge-α ran). That is the
//! intended trade: the MILP's own cumulative budget is `0.5 x` the first
//! remaining wall with no streak cap on this lane, i.e. up to ~half the row.
//! Raising `max_secs` above 4 s re-admits consults with proportionally
//! shorter slices.
//!
//! # Soundness
//!
//! Less edge work can only turn a would-be verify into a split. Edge-α only
//! TIGHTENS lower bounds, through the same monotone
//! `tighten_obj_lower_bounds` merge, and any α in `[0, 1]` yields a sound
//! bound, so an optimisation cut short by the work deadline returns a sound
//! (possibly looser) bound. Reusing edge-α's collection for the MILP request
//! hands the oracle the same kind of sound per-node boxes over the SAME exact
//! input box that it would otherwise recompute. The edge-MILP still counts a
//! domain only on the certified `VerifiedAllRows`. No verdict path changes:
//! a relational `unsat` still needs the unchanged Farkas formula-implication
//! token and a `sat` the unchanged ORT revalidation.
//!
//! # Observable
//!
//! Printed ONLY when the key is armed (dark runs keep byte-identical stderr):
//!
//! * `[edge-escalation] ARMED ...` once per BaB run;
//! * `[edge-escalation] wave=K t=..s alpha=N milp_collect=M milp_reuse=R
//!   milp_consult=C verified=V skipped=S secs=.. allowance=.. capped=0|1`
//!   for every wave in which an edge step was attempted or refused (rate
//!   limited to [`MAX_WAVE_LINES`] per process);
//! * `[edge-escalation] final armed=1 waves=.. engaged_waves=.. ...` on EVERY
//!   exit path of the run. `engaged_waves=0` means no domain ever reached the
//!   edge gate: armed but INERT, not negative.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;

use super::BabWall;

/// `bab.branching.input_split.edge_wave_cap.max_secs` when omitted.
pub(super) const EDGE_WAVE_CAP_DEFAULT_MAX_SECS: f64 = 2.0;
/// `bab.branching.input_split.edge_wave_cap.rebound_share` when omitted.
pub(super) const EDGE_WAVE_CAP_DEFAULT_REBOUND_SHARE: f64 = 0.25;
/// Defensive ceilings: `Duration::from_secs_f64` panics on a non-finite or
/// overflowing argument, and a preset is untrusted input.
const EDGE_WAVE_CAP_MAX_SECS_CEILING: f64 = 3_600.0;
const EDGE_WAVE_CAP_REBOUND_SHARE_CEILING: f64 = 64.0;
/// Per-wave lines printed per process before the rest are only counted.
pub(super) const MAX_WAVE_LINES: usize = 512;

/// The resolved, sanitized per-run policy. `None` from
/// [`resolve_edge_wave_cap`] is the shipped DARK state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct EdgeWaveCap {
    max: Duration,
    rebound_share: f64,
}

/// Resolve `BetaCrownConfig::input_split_edge_wave_cap`.
///
/// Absent (every shipped preset) → `None` → byte-identical to the pre-key
/// code. A non-finite or negative field also resolves to `None`: this lever
/// can only ever cost edge work, so a malformed number fails OPEN into today's
/// shipped behaviour rather than into a refusal. A ZERO field is honoured —
/// it is the typed way to measure "no edge work at all".
pub(super) fn resolve_edge_wave_cap(config: &BetaCrownConfig) -> Option<EdgeWaveCap> {
    let block = config.input_split_edge_wave_cap?;
    let max_secs = block.max_secs.unwrap_or(EDGE_WAVE_CAP_DEFAULT_MAX_SECS);
    let share = block
        .rebound_share
        .unwrap_or(EDGE_WAVE_CAP_DEFAULT_REBOUND_SHARE);
    if !max_secs.is_finite() || max_secs < 0.0 || !share.is_finite() || share < 0.0 {
        return None;
    }
    Some(EdgeWaveCap {
        max: Duration::from_secs_f64(max_secs.min(EDGE_WAVE_CAP_MAX_SECS_CEILING)),
        rebound_share: share.min(EDGE_WAVE_CAP_REBOUND_SHARE_CEILING),
    })
}

impl EdgeWaveCap {
    /// This wave's edge allowance: `min(max, rebound_share x rebound_secs)`.
    ///
    /// A non-finite or non-positive rebound measurement (e.g. a wave whose
    /// popped domains needed no deferred rebound) yields a ZERO allowance: no
    /// measurement, no edge budget. Never panics.
    pub(super) fn allowance(self, rebound_secs: f64) -> Duration {
        let rebound = if rebound_secs.is_finite() && rebound_secs > 0.0 {
            rebound_secs
        } else {
            0.0
        };
        let secs = (rebound * self.rebound_share).min(self.max.as_secs_f64());
        if secs.is_finite() && secs > 0.0 {
            Duration::from_secs_f64(secs)
        } else {
            Duration::ZERO
        }
    }

    /// The per-wave value the BaB loop hands the batch processors through
    /// [`BabWall::with_edge_wave`].
    pub(super) fn wave(self, index: usize, rebound_secs: f64) -> EdgeWave {
        EdgeWave {
            index,
            allowance: self.allowance(rebound_secs),
        }
    }

    pub(super) fn max_secs(self) -> f64 {
        self.max.as_secs_f64()
    }

    pub(super) fn rebound_share(self) -> f64 {
        self.rebound_share
    }
}

/// One wave's edge budget, carried by value inside [`BabWall`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct EdgeWave {
    /// The BaB loop's batch index (the `wave=` of the observable).
    pub(super) index: usize,
    /// Edge wall this wave may spend.
    pub(super) allowance: Duration,
}

// ---------------------------------------------------------------------------
// Process-global engagement counters (the `ChildBoundCapReport` pattern):
// the run report snapshots them at construction and prints the DELTA, so a
// restart inside one process reports its own numbers.
// ---------------------------------------------------------------------------

static WAVES: AtomicUsize = AtomicUsize::new(0);
static ENGAGED_WAVES: AtomicUsize = AtomicUsize::new(0);
static CAPPED_WAVES: AtomicUsize = AtomicUsize::new(0);
static ALPHA_PASSES: AtomicUsize = AtomicUsize::new(0);
static MILP_COLLECTS: AtomicUsize = AtomicUsize::new(0);
static MILP_REUSES: AtomicUsize = AtomicUsize::new(0);
static MILP_CONSULTS: AtomicUsize = AtomicUsize::new(0);
static VERIFIED: AtomicUsize = AtomicUsize::new(0);
static SKIPPED: AtomicUsize = AtomicUsize::new(0);
static SPENT_MICROS: AtomicU64 = AtomicU64::new(0);
static WAVE_LINES: AtomicUsize = AtomicUsize::new(0);
static WAVE_LINES_SUPPRESSED: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EdgeCounters {
    waves: usize,
    engaged_waves: usize,
    capped_waves: usize,
    alpha_passes: usize,
    milp_collects: usize,
    milp_reuses: usize,
    milp_consults: usize,
    verified: usize,
    skipped: usize,
    spent_micros: u64,
    lines_suppressed: usize,
}

impl EdgeCounters {
    fn load() -> Self {
        Self {
            waves: WAVES.load(Ordering::Relaxed),
            engaged_waves: ENGAGED_WAVES.load(Ordering::Relaxed),
            capped_waves: CAPPED_WAVES.load(Ordering::Relaxed),
            alpha_passes: ALPHA_PASSES.load(Ordering::Relaxed),
            milp_collects: MILP_COLLECTS.load(Ordering::Relaxed),
            milp_reuses: MILP_REUSES.load(Ordering::Relaxed),
            milp_consults: MILP_CONSULTS.load(Ordering::Relaxed),
            verified: VERIFIED.load(Ordering::Relaxed),
            skipped: SKIPPED.load(Ordering::Relaxed),
            spent_micros: SPENT_MICROS.load(Ordering::Relaxed),
            lines_suppressed: WAVE_LINES_SUPPRESSED.load(Ordering::Relaxed),
        }
    }

    fn since(self, base: Self) -> Self {
        Self {
            waves: self.waves.saturating_sub(base.waves),
            engaged_waves: self.engaged_waves.saturating_sub(base.engaged_waves),
            capped_waves: self.capped_waves.saturating_sub(base.capped_waves),
            alpha_passes: self.alpha_passes.saturating_sub(base.alpha_passes),
            milp_collects: self.milp_collects.saturating_sub(base.milp_collects),
            milp_reuses: self.milp_reuses.saturating_sub(base.milp_reuses),
            milp_consults: self.milp_consults.saturating_sub(base.milp_consults),
            verified: self.verified.saturating_sub(base.verified),
            skipped: self.skipped.saturating_sub(base.skipped),
            spent_micros: self.spent_micros.saturating_sub(base.spent_micros),
            lines_suppressed: self.lines_suppressed.saturating_sub(base.lines_suppressed),
        }
    }
}

fn duration_micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// One wave's edge governor, constructed at the top of each batch processor
/// from the [`BabWall`] the BaB loop handed it.
///
/// DARK (the wall carries no [`EdgeWave`]): [`Self::is_armed`] is false, the
/// callers take the historical code path statement for statement, and `Drop`
/// prints and records nothing.
#[derive(Debug)]
pub(super) struct EdgeWaveGovernor {
    wave: Option<EdgeWave>,
    run_start: Instant,
    stop_deadline: Instant,
    spent: Duration,
    alpha_passes: usize,
    milp_collects: usize,
    milp_reuses: usize,
    milp_consults: usize,
    verified: usize,
    skipped: usize,
}

impl EdgeWaveGovernor {
    /// The governor for the wave `bab_wall` describes, for a run that started
    /// at `run_start` (`lifecycle.start_time`).
    pub(super) fn for_wave(bab_wall: BabWall, run_start: Instant) -> Self {
        Self {
            wave: bab_wall.edge_wave(),
            run_start,
            // The scored stop as an absolute instant — never later than the
            // horizon, because `BabWall::clamped` clamps `stop <= horizon`.
            stop_deadline: GraphBabLifecycle::fail_closed_deadline(run_start, bab_wall.stop()),
            spent: Duration::ZERO,
            alpha_passes: 0,
            milp_collects: 0,
            milp_reuses: 0,
            milp_consults: 0,
            verified: 0,
            skipped: 0,
        }
    }

    /// Whether this wave runs the governed path at all.
    pub(super) fn is_armed(&self) -> bool {
        self.wave.is_some()
    }

    /// The edge work's AUTHORITY boundary: the scored stop.
    pub(super) fn stop_deadline(&self) -> Instant {
        self.stop_deadline
    }

    /// This wave's allowance (`ZERO` when dark; never consulted then).
    pub(super) fn allowance(&self) -> Duration {
        self.wave.map_or(Duration::ZERO, |wave| wave.allowance)
    }

    /// May a NEW edge step start at `now`? Always `true` when dark (the dark
    /// path never asks). Armed: only while the wave's spent edge wall is
    /// strictly below its allowance AND the scored stop has not passed.
    pub(super) fn may_start(&self, now: Instant) -> bool {
        match self.wave {
            None => true,
            Some(wave) => self.spent < wave.allowance && now < self.stop_deadline,
        }
    }

    /// The deadline a step started at `now` must finish its ITERATIVE work by:
    /// `min(stop, now + remaining allowance)`. Never later than the stop.
    pub(super) fn work_deadline(&self, now: Instant) -> Instant {
        let remaining = self.allowance().saturating_sub(self.spent);
        let by_allowance = GraphBabLifecycle::fail_closed_deadline(now, remaining);
        by_allowance.min(self.stop_deadline)
    }

    /// Charge `elapsed` edge wall to this wave.
    pub(super) fn charge(&mut self, elapsed: Duration) {
        self.spent = self.spent.saturating_add(elapsed);
    }

    pub(super) fn note_alpha_pass(&mut self) {
        self.alpha_passes += 1;
    }

    pub(super) fn note_milp_collect(&mut self) {
        self.milp_collects += 1;
    }

    pub(super) fn note_milp_reuse(&mut self) {
        self.milp_reuses += 1;
    }

    pub(super) fn note_milp_consult(&mut self) {
        self.milp_consults += 1;
    }

    pub(super) fn note_verified(&mut self) {
        self.verified += 1;
    }

    /// An eligible edge step was REFUSED because the allowance was spent or
    /// the stop had passed. This is the cap's own engagement count.
    pub(super) fn note_skipped(&mut self) {
        self.skipped += 1;
    }

    fn engaged(&self) -> bool {
        self.alpha_passes + self.milp_collects + self.milp_reuses + self.skipped > 0
    }

    fn capped(&self) -> bool {
        self.skipped > 0 || self.spent >= self.allowance()
    }

    #[cfg(test)]
    pub(super) fn spent(&self) -> Duration {
        self.spent
    }

    #[cfg(test)]
    pub(super) fn skipped(&self) -> usize {
        self.skipped
    }

    #[cfg(test)]
    pub(super) fn alpha_passes(&self) -> usize {
        self.alpha_passes
    }

    #[cfg(test)]
    pub(super) fn milp_collects(&self) -> usize {
        self.milp_collects
    }

    #[cfg(test)]
    pub(super) fn milp_reuses(&self) -> usize {
        self.milp_reuses
    }

    #[cfg(test)]
    pub(super) fn milp_consults(&self) -> usize {
        self.milp_consults
    }
}

impl Drop for EdgeWaveGovernor {
    fn drop(&mut self) {
        let Some(wave) = self.wave else {
            // Dark: nothing recorded, nothing printed.
            return;
        };
        WAVES.fetch_add(1, Ordering::Relaxed);
        if !self.engaged() {
            return;
        }
        let capped = self.capped();
        ENGAGED_WAVES.fetch_add(1, Ordering::Relaxed);
        if capped {
            CAPPED_WAVES.fetch_add(1, Ordering::Relaxed);
        }
        ALPHA_PASSES.fetch_add(self.alpha_passes, Ordering::Relaxed);
        MILP_COLLECTS.fetch_add(self.milp_collects, Ordering::Relaxed);
        MILP_REUSES.fetch_add(self.milp_reuses, Ordering::Relaxed);
        MILP_CONSULTS.fetch_add(self.milp_consults, Ordering::Relaxed);
        VERIFIED.fetch_add(self.verified, Ordering::Relaxed);
        SKIPPED.fetch_add(self.skipped, Ordering::Relaxed);
        SPENT_MICROS.fetch_add(duration_micros(self.spent), Ordering::Relaxed);
        if WAVE_LINES.fetch_add(1, Ordering::Relaxed) >= MAX_WAVE_LINES {
            WAVE_LINES_SUPPRESSED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        eprintln!(
            "[edge-escalation] wave={} t={:.1}s alpha={} milp_collect={} milp_reuse={} \
             milp_consult={} verified={} skipped={} secs={:.3} allowance={:.3} capped={}",
            wave.index,
            self.run_start.elapsed().as_secs_f64(),
            self.alpha_passes,
            self.milp_collects,
            self.milp_reuses,
            self.milp_consults,
            self.verified,
            self.skipped,
            self.spent.as_secs_f64(),
            wave.allowance.as_secs_f64(),
            u8::from(capped),
        );
    }
}

/// Run-scoped `[edge-escalation]` report: the ARMED line at construction and
/// the `final` line on drop, i.e. on EVERY exit path of the BaB loop
/// (verdict, timeout, domain limit, `?`). Constructed only when armed.
pub(super) struct EdgeEscalationReport {
    cap: EdgeWaveCap,
    base: EdgeCounters,
}

impl EdgeEscalationReport {
    pub(super) fn armed(cap: EdgeWaveCap, config: &BetaCrownConfig, bab_wall: BabWall) -> Self {
        eprintln!(
            "[edge-escalation] ARMED route=grouped-disjunctive max_secs={:.3} \
             rebound_share={:.3} edge_alpha={} edge_alpha_top={} edge_milp={} \
             edge_gap={} edge_depth={} horizon={:.3}s stop={:.3}s \
             source=bab.branching.input_split.edge_wave_cap (allowance per wave = \
             min(max_secs, rebound_share x the rebound that precedes it); edge work \
             never starts past the stop and never runs past min(stop, now + remaining \
             allowance); bound-shaping deadlines unchanged)",
            cap.max_secs(),
            cap.rebound_share(),
            u8::from(config.input_split_edge_alpha),
            config.input_split_edge_alpha_top,
            u8::from(config.input_split_edge_milp),
            config.input_split_edge_milp_gap,
            config.input_split_edge_milp_depth,
            bab_wall.horizon().as_secs_f64(),
            bab_wall.stop().as_secs_f64(),
        );
        Self {
            cap,
            base: EdgeCounters::load(),
        }
    }
}

impl Drop for EdgeEscalationReport {
    fn drop(&mut self) {
        let c = EdgeCounters::load().since(self.base);
        eprintln!(
            "[edge-escalation] final armed=1 max_secs={:.3} rebound_share={:.3} waves={} \
             engaged_waves={} capped_waves={} alpha={} milp_collect={} milp_reuse={} \
             milp_consult={} verified={} skipped={} secs={:.3} lines_suppressed={} \
             (engaged_waves=0 means no domain ever reached the edge gate: armed but INERT, \
             not negative; skipped>0 is the cap refusing edge work)",
            self.cap.max_secs(),
            self.cap.rebound_share(),
            c.waves,
            c.engaged_waves,
            c.capped_waves,
            c.alpha_passes,
            c.milp_collects,
            c.milp_reuses,
            c.milp_consults,
            c.verified,
            c.skipped,
            Duration::from_micros(c.spent_micros).as_secs_f64(),
            c.lines_suppressed,
        );
    }
}

#[cfg(test)]
#[path = "edge_cap_tests.rs"]
mod tests;
