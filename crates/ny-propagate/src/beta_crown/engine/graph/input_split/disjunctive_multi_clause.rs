// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-clause disjunctive (OR-of-AND) input-split BaB verifier.
//! Forked from `multi_objective.rs`. Uses `disjunctive_domain_verified()` (all
//! clauses satisfied) and `disjunctive_domain_priority()` (worst clause's best row).
//! Reference: `stop_criterion_general` (`auto_LiRPA/utils.py:115-137`).

mod child_batch;
mod edge_cap;
mod process_batch;
mod push_survivors;
mod screen_child;

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::Array2;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::beta_crown::branching::BranchingHeuristic;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::bounds::GraphAlphaState;
use crate::GraphNetwork;

use self::process_batch::process_disjunctive_domain_batch;
use super::super::shared::state::GraphBabLifecycle;
use super::batching::{
    bound_deferred_disjunctive_domains_batch, input_split_loop_batch_size,
    pop_multi_obj_input_domain_batch,
};
use super::build_batches::compute_crown_or_ibp_bounds_in_build_batches;
use super::fresh_domain_clip::FreshDomainClipTelemetry;
use super::grouped_semantics::{
    disjunctive_domain_priority, disjunctive_domain_verified, valid_disjunctive_layout,
};
use super::metrics::{should_log_batch, InputSplitBatchSummary};
use super::mul_binary::maybe_optimize_mul_binary_alphas;
use super::root_bounds::{
    collect_input_split_root_node_bounds, input_split_root_authority_deadline,
};
use super::shared::{
    compute_crown_or_ibp_bounds_with_node_bounds, extract_obj_bounds, graph_spec_ibp_fallback,
    MultiObjBounds, MultiObjInputDomain,
};
use crate::beta_crown::engine::BetaCrownVerifier;

/// #frontier-anatomy (DIAGNOSTIC, default-off, verdict-neutral): dump every
/// OPEN cell left in the BaB queue at exit.
///
/// Gated on the caller having supplied `--input-split-metrics-jsonl`, an
/// EXISTING CLI flag. `vnncomp_scripts/run_instance.sh` passes no such flag, so
/// this cannot fire in competition. It reads state and prints; it touches no
/// bound, no queue and no verdict.
fn dump_open_frontier(
    verifier: &BetaCrownVerifier,
    queue: &BinaryHeap<MultiObjInputDomain>,
    tag: &str,
    explored: usize,
    verified: usize,
    thresholds: &[f32],
    clause_sizes: &[usize],
) {
    if verifier.input_split_metrics_sink().is_none() {
        return;
    }
    let mut thr = String::new();
    for t in thresholds {
        thr.push_str(&format!("{t:.10e};"));
    }
    let mut cs = String::new();
    for c in clause_sizes {
        cs.push_str(&format!("{c};"));
    }
    eprintln!(
        "[frontier-dump] tag={tag} open={} explored={explored} verified={verified} clause_sizes={cs} thresholds={thr}",
        queue.len()
    );
    for (idx, domain) in queue.iter().enumerate() {
        let flat = domain.input_bounds.flatten();
        let mut boxes = String::new();
        for dim in 0..flat.len() {
            let lo = flat.lower()[[dim]];
            let hi = flat.upper()[[dim]];
            boxes.push_str(&format!("{lo:.10e}/{hi:.10e}/{:.10e};", hi - lo));
        }
        let mut objs = String::new();
        for (lo, hi) in &domain.obj_bounds {
            objs.push_str(&format!("{lo:.10e}/{hi:.10e};"));
        }
        eprintln!(
            "[frontier-cell] tag={tag} i={idx} depth={} prio={:.10e} needs_bounding={} box={boxes} obj={objs}",
            domain.depth, domain.priority, domain.needs_bounding
        );
    }
    eprintln!("[frontier-dump-end] tag={tag}");
}

/// The two DISTINCT deadlines a branch-and-bound run needs, kept apart
/// (#scored-wall-clamp).
///
/// Before this type there was one `bab_timeout: Duration`, and it was doing two
/// jobs that pull in opposite directions:
///
/// * **effort shaping** — every per-node CROWN-IBP grant in this lane is a
///   fraction of the budget still believed to be REMAINING
///   (`budget_policy::compute_global_per_node_budget_secs`: `share =
///   remaining / candidates`, dropped entirely when it falls under the floor).
///   A nearer deadline therefore buys LOOSER per-domain bounds, which produce
///   MORE splits, which is why this search is measurably NON-MONOTONIC in its
///   own timeout — the property `DEFAULT_REL_BAB_DEADLINE_MULT` exists to
///   exploit;
/// * **enforcement** — the instant past which the loop must not run, because
///   the process owes a verdict to a scored budget it does not control.
///
/// Conflating them means buying trajectory quality with wall-clock overrun.
/// MEASURED on `isomorphic_acasxu_2026` at the official 100 s budget: the
/// caller's 1.4x inflation put the single deadline at ~124.6 s absolute, and
/// EIGHTEEN of 50 rows ran until an in-process watchdog SIGKILLed them at
/// 105.0-105.1 s — 105 % of the official budget, and on the competition
/// harness an external kill at 100 s leaves those rows with no verdict token
/// at all.
///
/// Split in two, the inflation keeps shaping effort while the stop stays
/// inside the budget. `stop` is clamped to `horizon` on construction, so this
/// can only ever SHORTEN a run: no wall can extend a search past the timeout
/// its caller configured.
///
/// THE SERIAL EDGE WORK (#iso-edge-cap). The edge-α pass and the edge-MILP
/// escalation in `process_batch.rs` historically took
/// `lifecycle.deadline(horizon)` as BOTH their effort shape and their
/// authority, and nothing bounded how much of one wave they could spend —
/// MEASURED on `isomorphic_acasxu_2026` (Mac, `624477367`): single batches of
/// 15-63 s against a ~1.6 s norm, starting exactly at the edge gate's depth,
/// one of them running the process to 100.96 s of a 100 s budget. Armed by
/// `bab.branching.input_split.edge_wave_cap`, the BaB loop attaches a per-wave
/// [`edge_cap::EdgeWave`] allowance to the wall it hands each batch processor
/// ([`Self::with_edge_wave`]); the processors' edge governor then refuses to
/// START edge work past the stop or past the allowance, and bounds the
/// iterative part of each step by `min(stop, now + remaining allowance)`.
/// Dark (the shipped state), no wave is attached and the processors run the
/// historical code path statement for statement. See `edge_cap.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BabWall {
    /// Shapes bound effort. Every `lifecycle.deadline(..)` derives from this.
    horizon: Duration,
    /// Stops the loop. Every elapsed-time termination check derives from this.
    stop: Duration,
    /// #iso-edge-cap: this wave's edge-work allowance. `None` — every wall
    /// `clamped` builds, and every wave while the key is dark — is the
    /// historical, ungoverned edge path.
    edge_wave: Option<edge_cap::EdgeWave>,
}

impl BabWall {
    /// Shape effort against `horizon`, but stop at `wall` if `wall` is nearer.
    ///
    /// `wall = None` is the historical single-deadline behaviour — the horizon
    /// IS the stop — and is what every caller that supplies no scored wall
    /// gets. That arm is byte-identical to the pre-split code.
    ///
    /// `wall` is an ABSOLUTE instant — the caller derives it once from the
    /// scored budget at its own anchor, so nothing downstream can restate it
    /// as a duration and lose the seconds already spent. A wall that has
    /// already passed yields a zero stop (terminate at the next check), never
    /// a negative or wrapped one; a wall beyond the horizon is ignored.
    pub(super) fn clamped(horizon: Duration, wall: Option<Instant>, now: Instant) -> Self {
        let stop = match wall {
            Some(wall) => horizon.min(wall.saturating_duration_since(now)),
            None => horizon,
        };
        Self {
            horizon,
            stop,
            edge_wave: None,
        }
    }

    /// #iso-edge-cap: the same wall carrying one wave's edge allowance.
    /// `None` hands back a wall equal to one `clamped` built. Horizon and stop
    /// are untouched, so no termination check or bound deadline moves.
    fn with_edge_wave(self, edge_wave: Option<edge_cap::EdgeWave>) -> Self {
        Self { edge_wave, ..self }
    }

    /// #iso-edge-cap: this wave's edge allowance, `None` when dark.
    fn edge_wave(self) -> Option<edge_cap::EdgeWave> {
        self.edge_wave
    }

    /// The effort-shaping horizon. Pair with `lifecycle.deadline(..)`.
    pub(super) fn horizon(self) -> Duration {
        self.horizon
    }

    /// Whether the run started at `start` must stop now.
    ///
    /// Deliberately strict `>` and `Instant::elapsed`, exactly as the single
    /// `bab_timeout` check it replaces.
    pub(super) fn expired(self, start: Instant) -> bool {
        start.elapsed() > self.stop
    }

    /// Whether the next loop iteration must be SKIPPED because it cannot
    /// finish before the stop (#scored-wall-clamp).
    ///
    /// WHY THIS EXISTS. `expired` alone bounds the loop only to the granularity
    /// of one iteration, and an iteration is not interruptible all the way
    /// down: the batched rebound at the top of it runs to completion once
    /// entered. MEASURED on `isomorphic_acasxu_2026` at the official 100 s
    /// budget: with the stop honoured but no guard, the process exits 0.40 s to
    /// 2.54 s AFTER the wall — the spread is one rebound at `batch_size = 2048`.
    /// A 2.54 s tail against a 3 s reserve is 0.46 s of margin on the number
    /// that must never be exceeded.
    ///
    /// So the loop declines to START work it has measured itself unable to
    /// finish, and the tail past the wall becomes the ERROR in that estimate
    /// rather than a whole batch. MEASURED with the guard in place, same rows:
    /// the process exits 0.00-0.18 s after the wall.
    ///
    /// `previous_iteration` is the iteration just completed, NOT the longest
    /// seen. That choice was measured, not assumed: the longest-so-far estimate
    /// is monotone, so one slow batch permanently pessimises every later
    /// decision, and it cost a real row — `instance_15`, which the unguarded
    /// armed arm decides at 88.6 s and 95.0 s, became a 95.0 s `timeout`
    /// because the guard refused the batch that would have verified it. Batch
    /// cost is locally stable, so the previous iteration is the honest
    /// predictor; the reserve above the wall is what absorbs its error.
    ///
    /// Inert unless an absolute wall is actually binding: with `stop ==
    /// horizon` this returns `false` always, so no unclamped run changes.
    pub(super) fn cannot_finish(
        self,
        start: Instant,
        previous_iteration: Option<Duration>,
    ) -> bool {
        if !self.is_clamped() {
            return false;
        }
        match previous_iteration {
            // The first iteration has no estimate and is always attempted:
            // refusing it would turn a clamped run into no search at all.
            None => false,
            Some(iteration) => start.elapsed() + iteration > self.stop,
        }
    }

    /// True when an absolute wall is actually binding — i.e. the loop will stop
    /// before the horizon it is shaping effort against. Diagnostics only.
    pub(super) fn is_clamped(self) -> bool {
        self.stop < self.horizon
    }

    /// The stop, for logging and for tests that assert the derived deadline.
    pub(super) fn stop(self) -> Duration {
        self.stop
    }
}

fn format_optional_seconds(value: Option<f64>) -> String {
    value
        .map(|seconds| format!("{seconds:.3}"))
        .unwrap_or_else(|| "n/a".to_string())
}

/// Resolve the BaB wall slice for grouped disjunctive input splitting.
///
/// A caller-supplied deadline is already an authoritative phase boundary (the
/// CLI ledger has reserved its post-BaB slice). Only the convenience path with
/// no deadline derives that reservation from the configured timeout.
#[inline]
fn disjunctive_multi_clause_bab_timeout(
    configured_timeout: Duration,
    post_bab_pgd_fraction: f32,
    deadline: Option<Instant>,
    now: Instant,
) -> Duration {
    match deadline {
        Some(deadline) => deadline.saturating_duration_since(now),
        None => configured_timeout.mul_f32(1.0 - post_bab_pgd_fraction.clamp(0.0, 0.5)),
    }
}

/// Upper sanity clamp for the typed per-child cap, in seconds. Purely defensive:
/// `Duration::from_secs_f64` PANICS on a negative, non-finite or
/// `u64`-overflowing argument, and a preset is untrusted input.
const CHILD_BOUND_CAP_MAX_SECS: f64 = 86_400.0;

/// Resolve `bab.branching.input_split.child_bound_max_secs`
/// (`BetaCrownConfig::input_split_child_bound_max_secs`) into a per-call wall
/// slice (#cgan-dark-tail FIX-1).
///
/// `None` — THE SHIPPED DEFAULT — means NO CAP, and every consumer below is
/// then byte-for-byte identical to the pre-fix route. A non-finite or
/// non-positive value is also treated as "no cap" rather than as an error: this
/// lever can only ever cost bound quality, so refusing to start over a
/// malformed number would trade points for nothing.
///
/// `pub(super)` because the certified-f64 seams in `f64_tail.rs` resolve the
/// SAME key from the same `BetaCrownConfig` (see [`child_bound_deadline`]);
/// one resolver keeps "no cap" meaning exactly one thing lane-wide.
#[inline]
pub(super) fn resolve_child_bound_cap(config: &BetaCrownConfig) -> Option<Duration> {
    let secs = config.input_split_child_bound_max_secs?;
    if !secs.is_finite() || secs <= 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(secs.min(CHILD_BOUND_CAP_MAX_SECS)))
}

/// Engagement counters for the per-child cap (#cgan-dark-tail FIX-1, R7/R9).
///
/// A null from a lever whose own telemetry is off is VACUOUS, and "cap armed"
/// is not "cap engaged": a 20 s cap under an 895 s ceiling composes a nearer
/// deadline on EVERY call, but only the calls that actually reach it change
/// anything. `CALLS`/`HITS` prove the composition ran and bound; the calls that
/// truly ran out of slice name themselves in `[child-bound-slow]`
/// (`screen_child.rs`) with `elapsed` pinned at the cap — which is why a
/// measurement cap should be >= `CHILD_BOUND_SLOW_SECS` (2 s).
///
/// Process-global (matching `should_eprint_batch_trajectory` and the
/// `[f64-tail]` rate limiter); [`ChildBoundCapReport`] snapshots a baseline at
/// construction and reports the DELTA, so a restart run inside one process
/// reports its own numbers rather than the accumulated total.
static CHILD_BOUND_CAP_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static CHILD_BOUND_CAP_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Compose the deadline ONE per-child bound call may see (#cgan-dark-tail
/// FIX-1).
///
/// `ceiling` is the absolute BaB `crown_deadline` — the HARD limit, never
/// extended by this function. `cap` is the optional per-call wall slice.
///
/// * `cap == None` (the shipped default) returns `ceiling` UNCHANGED, so the
///   whole route stays byte-for-byte what it is today.
/// * `cap == Some(d)` returns `min(ceiling, now + d)`, i.e. the earlier of the
///   two. The result is therefore always `<= ceiling` when a ceiling exists.
///
/// SOUNDNESS: an earlier deadline can only make a bound LOOSER (or fail the
/// call outright, which `compute_child_bounds` already recovers from via the
/// frozen-root-α retry that records `frozen_fallback_completions`). It never
/// makes a bound wrong, because the deadline selects routes and fallbacks, not
/// bound math.
///
/// WHAT THE CAP DOES **NOT** BOUND — read before quoting a worst case. This is
/// a per-CALL slice, not a per-CHILD one, and one child is screened by up to
/// five bound seams (`screen_child.rs`: `pass0`, `post-relaxed-clip`,
/// `post-complete-clip`, `polish`, `f64-tail-hook`), each of which may in turn
/// spend a warm attempt AND a frozen-fallback retry — every one of them
/// re-entering here with a FRESH `Instant::now()`. So the honest per-child
/// ceiling is roughly `seams x 2 x cap`, not `cap`. That is still a bound
/// where today there is none (the measured tail is ONE call at 70-285 s
/// against a 0.13 s median over ~1640 calls), but a cap chosen as if it were
/// per-child will overshoot by nearly an order of magnitude. On the cgan
/// preset (`clip.relaxed: true`, polish and f64 tail dark) the live seams are
/// `pass0` and `post-relaxed-clip`, so the live worst case is ~`4 x cap`.
///
/// STILL UNCAPPED, deliberately, because every one of them is INERT on the
/// cgan preset and capping an unmeasured lane buys nothing: the edge-α pass,
/// the edge-MILP oracle escalation and the fresh-domain clip in
/// `process_batch.rs`, which each take `lifecycle.deadline(bab_wall.horizon())`
/// — the absolute ceiling — directly. Arming any of those on this lane
/// re-opens the same defect, so cap them at the same time.
///
/// #iso-edge-cap: the first two of those — the edge-α pass and the edge-MILP
/// escalation — now have a cap of their own,
/// `bab.branching.input_split.edge_wave_cap` (`edge_cap.rs`): the scored stop
/// as their authority plus a per-wave wall allowance. It ships DARK, so this
/// paragraph still describes every shipped preset. The fresh-domain clip is
/// untouched by it and remains uncapped.
///
/// `pub(super)` so the f64 seams in `f64_tail.rs` compose the identical law.
#[inline]
pub(super) fn child_bound_deadline(
    ceiling: Option<Instant>,
    cap: Option<Duration>,
) -> Option<Instant> {
    use std::sync::atomic::Ordering;
    let Some(cap) = cap else {
        // Shipped default: hand back exactly what the caller has today.
        return ceiling;
    };
    CHILD_BOUND_CAP_CALLS.fetch_add(1, Ordering::Relaxed);
    let capped = GraphBabLifecycle::fail_closed_deadline(Instant::now(), cap);
    // `min` is what makes the absolute ceiling a HARD limit: an over-long cap
    // can never push the deadline past it.
    let composed = ceiling.map_or(capped, |ceiling| ceiling.min(capped));
    // ENGAGEMENT, not arming: the cap bound this call only when what came out
    // is not the ceiling that went in.
    if ceiling != Some(composed) {
        CHILD_BOUND_CAP_HITS.fetch_add(1, Ordering::Relaxed);
    }
    Some(composed)
}

/// A call that spends at least this fraction of its allowance is counted as
/// TRUNCATED rather than completed.
///
/// It has to be a fraction, not an equality, because the interesting truncation
/// is INTERNAL: on `upsample` the capped calls returned `Ok` after
/// commit-what-you-have cut a 27-layer intermediate collection to 5-6 layers,
/// with `elapsed` pinned at a median 20.1 s against a 20.0 s cap. Waiting for
/// an `Err` would have classified all 37 of them as healthy completions and
/// pinned the estimator at exactly the allowance that was destroying the work.
const CHILD_BOUND_TRUNCATION_FRACTION: f64 = 0.95;

/// #cost-observer: the run-scoped authority for ONE per-child bound call's wall
/// allowance, and the only thing the closures below consult.
///
/// Two arms, selected by config and never by an env var:
///
/// * `Constant` — `bab.branching.input_split.child_bound_max_secs`, resolved
///   ONCE before the loop exactly as it is today. Byte-for-byte the shipped
///   route, including `None` = no cap.
/// * `Calibrated` — `bab.branching.input_split.child_bound_calibration`. The
///   allowance is derived PER CALL from the observer's measurement of this
///   row's own completed-call costs and the wall actually remaining.
///
/// SOUNDNESS. Both arms terminate in the SAME [`child_bound_deadline`]
/// composition, `min(ceiling, now + allowance)`. Nothing in this type can
/// produce a deadline later than the ceiling it was handed — the calibrated arm
/// does not even have a representation for one, because
/// `ny_core::cost_observer::Slice` is either a bounded allowance or "use the
/// caller's ceiling". So the calibration decides HOW MUCH WORK a bound call
/// gets and can never decide WHAT IS CONCLUDED from it: a shorter allowance
/// makes a bound looser, or fails the call into the frozen-root-α retry that
/// `screen_child.rs` already runs and already counts. Neither outcome can turn
/// a `timeout` into a wrong verdict, which is why this is verdict-neutral by
/// construction rather than by measurement.
// `large_enum_variant` is allowed deliberately. Exactly ONE `ChildBoundSlicer`
// exists per solve — `ChildBoundSlicer::new` is called once, outside the node
// loop, and the value lives on the stack for the whole descent. So the variant
// size difference costs one stack slot, not an allocation or a copy per node,
// and boxing the observer would put an indirection on the calibration path that
// IS per-call for no measurable gain.
#[allow(clippy::large_enum_variant)]
enum ChildBoundSlicer {
    Constant(Option<Duration>),
    Calibrated {
        observer: ny_core::cost_observer::CostObserver,
        /// What to use when the observer declines to bound a call — i.e. when
        /// MEASUREMENT IS UNAVAILABLE. This is `resolve_child_bound_cap`, the
        /// hand-set constant, so the uncalibrated behaviour is *exactly*
        /// today's whether or not a preset names it: `None` for every shipped
        /// preset (= the caller's ceiling), and the constant for one that sets
        /// both. That is what makes calibration a strict improvement rather
        /// than a different guess.
        fallback: Option<Duration>,
    },
}

impl ChildBoundSlicer {
    /// Select the arm. Calibration wins when its block is present; otherwise
    /// the constant governs, which is the shipped state for every preset.
    fn new(config: &BetaCrownConfig) -> Self {
        match config.input_split_child_bound_calibration {
            Some(calibration) => Self::Calibrated {
                observer: ny_core::cost_observer::CostObserver::new(calibration.to_policy()),
                fallback: resolve_child_bound_cap(config),
            },
            None => Self::Constant(resolve_child_bound_cap(config)),
        }
    }

    /// Run ONE per-child bound call under its own allowance, and measure it.
    ///
    /// This is the seam where a hand-set number used to be read. The closure
    /// receives the composed deadline and nothing else, so a caller cannot
    /// accidentally use the allowance for anything but a deadline.
    ///
    /// Classification of the outcome is deliberately three-way, not two:
    ///
    /// * spent >= [`CHILD_BOUND_TRUNCATION_FRACTION`] of the allowance —
    ///   TRUNCATED, whether it returned `Ok` or `Err`. This is the
    ///   commit-what-you-have case and it must not feed the typical cost.
    /// * returned `Ok` inside its allowance — COMPLETED, a real cost sample.
    /// * returned `Err` inside its allowance — recorded as NEITHER. It failed
    ///   for some reason other than its wall, so its duration is not a sample
    ///   of what this seam costs, and calling it a truncation would grow the
    ///   allowance in response to evidence that has nothing to do with time.
    fn measured<T>(
        &self,
        ceiling: Option<Instant>,
        call: impl FnOnce(Option<Instant>) -> Result<T>,
    ) -> Result<T> {
        let started = Instant::now();
        let allowance = match self {
            Self::Constant(cap) => *cap,
            // No ceiling means there is no wall to take a share of. Deriving
            // one anyway would INTRODUCE a deadline where the caller had none,
            // and deadline PRESENCE — not expiry — is its own measured
            // degradation on this codebase (`#deadline-presence`, four root
            // causes, two of which discard optimized alpha). Leave it absent.
            Self::Calibrated { .. } if ceiling.is_none() => None,
            Self::Calibrated { observer, fallback } => {
                // The LIVE remaining wall, measured now — not the budget the
                // run started with. This is what makes the allowance tighten
                // as the instance runs out, which a constant cannot express.
                let remaining =
                    ceiling.map_or(Duration::ZERO, |c| c.saturating_duration_since(started));
                match observer.slice(remaining) {
                    ny_core::cost_observer::Slice::Bounded(allowance) => Some(allowance),
                    // MEASUREMENT UNAVAILABLE — no evidence any finite allowance
                    // completes, or no usable wall left. Fall back to the
                    // hand-set constant, which is `None` for every shipped
                    // preset and therefore the caller's own ceiling. Either way
                    // the uncalibrated path is EXACTLY what ships today, which
                    // is what makes this a strict improvement and not a
                    // different guess.
                    ny_core::cost_observer::Slice::Ceiling => *fallback,
                }
            }
        };
        let outcome = call(child_bound_deadline(ceiling, allowance));
        if let Self::Calibrated { observer, .. } = self {
            let elapsed = started.elapsed();
            match allowance {
                Some(allowance)
                    if elapsed.as_secs_f64()
                        >= allowance.as_secs_f64() * CHILD_BOUND_TRUNCATION_FRACTION =>
                {
                    observer.observe(ny_core::cost_observer::Observation::Truncated(elapsed));
                }
                _ if outcome.is_ok() => {
                    observer.observe(ny_core::cost_observer::Observation::Completed(elapsed));
                }
                _ => {}
            }
        }
        outcome
    }

    /// The ARMED line: which arm, and what it will do.
    fn report_armed(&self, bab_timeout: Duration) {
        match self {
            Self::Constant(None) => {}
            Self::Constant(Some(cap)) => eprintln!(
                "[child-bound-cap] ARMED route=grouped-disjunctive cap={:.3}s \
                 ceiling={:.3}s source=bab.branching.input_split.child_bound_max_secs \
                 (per-call min(ceiling, now+cap); ceiling is never extended; \
                 per-CHILD worst case is ~seams x 2 x cap, NOT cap — see \
                 child_bound_deadline)",
                cap.as_secs_f64(),
                bab_timeout.as_secs_f64(),
            ),
            Self::Calibrated { observer, .. } => {
                let p = observer.policy();
                eprintln!(
                    "[wall-calib] ARMED seam=child-bound route=grouped-disjunctive \
                     ceiling={:.3}s source=bab.branching.input_split.child_bound_calibration \
                     headroom={} max_share_of_remaining={} probe_share={} min_samples={} \
                     window={} growth={} (allowance = min(headroom x median completed cost, \
                     share x REMAINING wall) x growth^consecutive_truncated, composed as \
                     min(ceiling, now+allowance); every policy number is dimensionless — the \
                     seconds all come from measurement)",
                    bab_timeout.as_secs_f64(),
                    p.headroom,
                    p.max_share_of_remaining,
                    p.probe_share,
                    p.min_samples,
                    p.window,
                    p.growth,
                );
            }
        }
    }
}

/// Run-scoped `[wall-calib]` report for the calibrated arm.
///
/// Prints on EVERY exit path of the BaB loop (verdict, timeout, domain limit,
/// `?`), like its `[child-bound-cap]` sibling and for the same reason. It
/// carries BOTH the measurement and the derived values, because a report
/// showing only the derived allowance cannot distinguish "measured a cheap seam
/// and bounded it correctly" from "measured nothing" — and this campaign has
/// already read one telemetry field as engagement when it was a failure count.
struct ChildBoundCalibrationReport<'a>(&'a ny_core::cost_observer::CostObserver);

impl Drop for ChildBoundCalibrationReport<'_> {
    fn drop(&mut self) {
        let report = self.0.report();
        eprintln!(
            "[wall-calib] final seam=child-bound route=grouped-disjunctive {report} \
             (completed=0 with derived>0 means NOTHING ever finished inside an allowance and \
             the observer fell back to the caller's ceiling — armed but INERT, not negative; \
             ceiling=0 with truncated=0 means every call finished inside its allowance and the \
             seam was never the bottleneck)"
        );
    }
}

/// Run-scoped `[child-bound-cap]` engagement report (#cgan-dark-tail FIX-1).
///
/// Constructed only when the cap is armed, and prints its DELTA on drop, so
/// every exit path of the BaB loop — verdict, timeout, domain limit, `?` — is
/// covered by one line instead of five. `eprintln!` deliberately: the scored
/// binary swallows `tracing::info!`, which is how a 32%-of-budget tail stayed
/// dark for weeks.
struct ChildBoundCapReport {
    cap: Duration,
    calls_at_start: usize,
    hits_at_start: usize,
}

impl ChildBoundCapReport {
    fn new(cap: Duration) -> Self {
        use std::sync::atomic::Ordering;
        Self {
            cap,
            calls_at_start: CHILD_BOUND_CAP_CALLS.load(Ordering::Relaxed),
            hits_at_start: CHILD_BOUND_CAP_HITS.load(Ordering::Relaxed),
        }
    }
}

impl Drop for ChildBoundCapReport {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        let calls = CHILD_BOUND_CAP_CALLS
            .load(Ordering::Relaxed)
            .saturating_sub(self.calls_at_start);
        let hits = CHILD_BOUND_CAP_HITS
            .load(Ordering::Relaxed)
            .saturating_sub(self.hits_at_start);
        eprintln!(
            "[child-bound-cap] final route=grouped-disjunctive cap={:.3}s composed_calls={} \
             capped_calls={} (capped_calls=0 with composed_calls>0 means the ceiling was \
             already nearer than the cap — the lever is armed but INERT; \
             see [child-bound-slow] for the calls that actually spent the slice)",
            self.cap.as_secs_f64(),
            calls,
            hits,
        );
    }
}

#[inline]
fn eager_warm_alpha_enabled(config: &BetaCrownConfig, root_alpha_available: bool) -> bool {
    !config.reorder_bab
        && config.input_split_alpha_iteration > 0
        && config.use_alpha_crown
        && root_alpha_available
}

/// #cgan-polish: total alpha iterations for the near-miss survivor re-bound.
/// 3 iterations = 2 real SPSA/Adam steps + the final evaluation. The eager
/// pass's 1 iteration performs ZERO steps: the warm-start optimization loop
/// (`warm_start.rs`) breaks at `iter == iterations - 1` BEFORE the first
/// gradient, so without this pass no per-domain alpha step ever happens.
const CGAN_POLISH_ALPHA_ITERATIONS: usize = 3;

/// #cgan-polish: knife-edge eligibility band. cgan row 2's asymptote sits at
/// worst-clause gaps of -2e-4..-3e-3 (banked trajectory logs), while healthy
/// splits arrive well below -1e-2 and keep the full-speed 1-iteration pass.
const CGAN_POLISH_BAND: f32 = 5e-3;

/// Pure arming rule: exact `"1"` arms, everything else (absence, `"0"`,
/// `"true"`, `" 1"`, empty) leaves the polish dark — the campaign's
/// exact-token convention, pinned by `cgan_polish_gate_tests`.
#[inline]
fn cgan_polish_token_arms(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Whether `NY_CGAN_POLISH=1` requested the near-miss survivor polish.
/// Env acquisition goes through the typed declaration's `read_raw` chokepoint
/// so the lever appears in the flight receipt.
///
/// DELIVERY SEAM (#lever-reachability): the pure predicate above still owns the
/// arming RULE and still decides whenever the environment speaks, in both
/// directions; absent, the typed `bab.cgan_polish` key decides.
fn cgan_polish_requested() -> bool {
    ny_levers::gate::resolve_raw_predicate(
        &ny_levers::decls::cgan_stacked::CGAN_POLISH,
        crate::preset_gates::cgan_polish_cell(),
        false,
        cgan_polish_token_arms,
    )
}

/// The polish is an EAGER-lane mechanism riding the warm-alpha closure: it can
/// only be active where the eager warm pass is (same seed availability, same
/// route). Pure so the composition is pinned by a unit test.
#[inline]
fn cgan_polish_active(requested: bool, warm_alpha_enabled: bool) -> bool {
    requested && warm_alpha_enabled
}

/// Seed deferred children independently of the eager-screen gate.
///
/// Reordered BaB deliberately skips eager child refinement, but its root still
/// needs to carry the optimized slopes into the deferred rebound overlay.
#[inline]
fn deferred_root_alpha_seed(
    config: &BetaCrownConfig,
    root_alpha_state: Option<&GraphAlphaState>,
) -> Option<Arc<GraphAlphaState>> {
    (config.input_split_alpha_iteration > 0 && config.use_alpha_crown)
        .then(|| root_alpha_state.cloned())
        .flatten()
        .map(Arc::new)
}

/// Conservatively disable reused-bound relaxed clipping whenever the graph
/// cannot soundly batch-stack input-split domains.
///
/// `num_clauses` is intentionally not part of the decision: the reused CROWN
/// bound can be unsound for a batch-unsafe graph even when the property has a
/// single clause. Keeping it at this policy boundary makes that invariant
/// explicit and regression-testable.
#[inline]
fn should_disable_disjunctive_relaxed_clip(
    _num_clauses: usize,
    batch_stack_safe: bool,
    relaxed_clip_enabled: bool,
) -> bool {
    // MEASURED 2026-07-24: naively re-enabling this clip for the batch-unsafe
    // class (behind an NY_LSNC_LEVELSET_CLIP experiment flag) reproduced the KNOWN
    // false-unsats — state_34 AND state_45 both returned `unsat` despite their
    // ORT-confirmed in-box counterexamples. The clip's reused-CROWN bound source
    // is genuinely unsound on this net class (807bf511's disable is correct); a
    // sound level-set clip requires a per-domain, non-batch-stacked bound source,
    // not this reused one. Keeping the sound disable.
    relaxed_clip_enabled && !batch_stack_safe
}

/// Arm only the mechanically sealed current-domain route.
///
/// Batch-safe graphs retain the established child clip. Batch-unsafe graphs
/// keep that legacy route disabled and may opt into this distinct
/// non-domain-stacked full-spec path, provided the typed config selected the
/// exact reordered/IBP-enhanced relaxed lifecycle validated by
/// `BetaCrownConfig::validate`.
#[inline]
fn should_enable_fresh_domain_clip(config: &BetaCrownConfig, batch_stack_safe: bool) -> bool {
    config.input_split_fresh_domain_clip
        && matches!(config.branching_heuristic, BranchingHeuristic::InputSplit)
        && config.enable_relaxed_clip
        && matches!(
            config.input_clip_type,
            crate::beta_crown::config::InputClipType::Relaxed
        )
        && config.reorder_bab
        && config.input_split_ibp_enhancement
        && config.relaxed_clip_iterations > 0
        && !batch_stack_safe
}

/// Is this engine the verdict-grade sound GPU CROWN backward?
///
/// The same three clauses `multi_objective/verify.rs`'s
/// `post_root_factory_engine_is_eligible` applies, restated here because that
/// one is private to its module.
#[inline]
fn engine_is_sound_gpu_crown(engine: &dyn GemmEngine) -> bool {
    engine.supports_deadline_safe_post_root_multi_objective_bab()
        && engine.as_gpu_crown_backward().is_some_and(|gpu| {
            gpu.provides_sound_gpu_crown() && gpu.honors_crown_backward_deadline()
        })
}

/// One-shot, print-only census of the root intermediate-bound map handed to
/// grouped disjunctive BaB (#cgan-root-census). Emitted UNCONDITIONALLY once
/// per BaB start so the per-node width/unstable profile can be diffed against
/// alpha-beta-CROWN's plain-CROWN root on the same input box. No behaviour
/// change: reads the map and the root objective bounds, writes stderr.
fn print_cgan_root_census(
    graph: &GraphNetwork,
    root_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    root_obj_bounds: &[(f32, f32)],
    thresholds: &[f32],
    clause_sizes: &[usize],
) {
    let Some(map) = root_node_bounds else {
        eprintln!("[cgan-root-census] root_node_bounds=None (no intermediate map; spec backward runs on IBP/forward fallback)");
        return;
    };
    // Graph order when the cached exec order resolves; insertion order otherwise.
    let order: Vec<String> = match graph.exec_order() {
        Ok(order) => order.to_vec(),
        Err(_) => graph.node_names().to_vec(),
    };
    // Which map entries feed a ReLU / LeakyReLU consumer (pre-activation nodes).
    let mut feeds_relu: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for name in graph.node_names() {
        if let Some(node) = graph.node(name) {
            if matches!(
                node.layer(),
                crate::Layer::ReLU(_) | crate::Layer::LeakyReLU(_)
            ) {
                for input in node.inputs() {
                    feeds_relu.insert(input.as_str());
                }
            }
        }
    }
    let emit = |name: &str, bt: &BoundedTensor| {
        let (kind, inputs) = match graph.node(name) {
            Some(node) => (node.layer().layer_type(), node.inputs().join(",")),
            None => ("<no-graph-node>", String::new()),
        };
        let lower = bt.lower();
        let upper = bt.upper();
        let numel = lower.len();
        let mut unstable = 0usize;
        let mut neg = 0usize;
        let mut pos = 0usize;
        let mut width_sum = 0f64;
        let mut max_width = f64::NEG_INFINITY;
        let mut min_lb = f64::INFINITY;
        let mut max_ub = f64::NEG_INFINITY;
        let mut nonfinite = 0usize;
        for (&l, &u) in lower.iter().zip(upper.iter()) {
            let (l, u) = (l as f64, u as f64);
            if !l.is_finite() || !u.is_finite() {
                nonfinite += 1;
            }
            if l < 0.0 && u > 0.0 {
                unstable += 1;
            } else if u <= 0.0 {
                neg += 1;
            } else {
                pos += 1;
            }
            let w = u - l;
            width_sum += w;
            if w > max_width {
                max_width = w;
            }
            if l < min_lb {
                min_lb = l;
            }
            if u > max_ub {
                max_ub = u;
            }
        }
        let mean_width = if numel > 0 {
            width_sum / numel as f64
        } else {
            0.0
        };
        eprintln!(
            "[cgan-root-census] node={name} kind={kind} numel={numel} unstable={unstable} neg={neg} pos={pos} mean_width={mean_width:.6e} max_width={max_width:.6e} min_lb={min_lb:.6e} max_ub={max_ub:.6e} nonfinite={nonfinite} feeds_relu={} shape={:?} inputs={inputs}",
            feeds_relu.contains(name),
            bt.shape(),
        );
    };
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut in_graph_order = 0usize;
    for name in &order {
        if let Some(bt) = map.get(name) {
            emit(name, bt);
            seen.insert(name.as_str());
            in_graph_order += 1;
        }
    }
    let mut rest: Vec<&String> = map.keys().filter(|k| !seen.contains(k.as_str())).collect();
    rest.sort();
    for name in &rest {
        emit(name, &map[*name]);
    }
    let graph_nodes_missing: Vec<&String> =
        order.iter().filter(|n| !map.contains_key(*n)).collect();
    let margins = root_obj_bounds
        .iter()
        .zip(thresholds.iter())
        .map(|((l, u), &t)| format!("[{l:.6},{u:.6}] thr={t:.6} margin(lb-thr)={:.6}", l - t))
        .collect::<Vec<_>>()
        .join(" | ");
    eprintln!(
        "[cgan-root-census] summary map_entries={} in_graph_order={} off_graph={} graph_nodes={} graph_nodes_missing_from_map={} missing_names={:?} output_node={} provenance=none(HashMap<String,BoundedTensor> carries no provenance) clauses={} clause_sizes={:?} root_obj={}",
        map.len(),
        in_graph_order,
        rest.len(),
        order.len(),
        graph_nodes_missing.len(),
        graph_nodes_missing,
        graph.output_name(),
        clause_sizes.len(),
        clause_sizes,
        margins,
    );
}

/// One-shot observable, printed on EVERY exit path with its reason.
///
/// Attempt 1 placed this AFTER three early returns, so `admitted=ABSENT` was
/// consistent with four different causes and proved nothing. An observable
/// behind an early return is not an observable.
fn disjunctive_engine_note(why: &str) {
    static ANNOUNCED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ANNOUNCED.get_or_init(|| eprintln!("[disjunctive-engine] {why}"));
}

impl BetaCrownVerifier {
    /// PREFER the sound GPU CROWN engine over a caller-supplied engine that is
    /// not one.
    ///
    /// WHY PREFER AND NOT FILL. Attempt 1 mirrored
    /// `resolve_post_root_multi_objective_engine`, whose first line returns the
    /// caller's engine when present. On this lane the caller ALWAYS supplies one
    /// (`engine_dispatch.rs:88` -> `verify_graph_input_split_with_engine`), so
    /// that mirror was dead by construction and never reached the GPU engine.
    ///
    /// WHY THIS IS WORTH DOING (measured 2026-09-01, `cGAN_imgSz32_nCh_3`
    /// prop_0): ny PROVES the row in 2430.85 s but is given 900 s — a **2.7x**
    /// wall shortfall at **3.42 domains/s**, with the queue draining 252 -> 32.
    /// The in-tree figure for the dominant GEMM is 257.56 s CPU vs 12.97 s GPU
    /// (~20x). The row is lost to WALL, not to convergence.
    ///
    /// CORRECTIONS (2026-09-10). (1) The 257.56 s / 12.97 s pair is NOT a cgan
    /// figure: it is the March-15-2026 WGPU measurement of a graph CROWN-IBP
    /// collection on a soundnessbench-shaped 6-Conv2d workload, recorded at
    /// `configs/vnncomp25/relusplitter.yaml:36-38` (see also
    /// `patches_target.rs`). No cgan GEMM has been timed CPU-vs-GPU. (2) The
    /// 2430.85 s proving wall included a 2500-domain restart-probe replay, so
    /// the proving TREE is ~1.9x the 900 s budget, not 2.7x. (3) The commit
    /// carrying this PREFER handoff on HEAD is `2af1fd80e`; `3bf356483` is an
    /// orphan of the same change.
    ///
    /// SOUNDNESS OF THE OVERRIDE. This replaces a caller's engine, which the
    /// multi-objective mirror never does, so it needs its own argument: the
    /// replacement happens ONLY when the incoming engine is not verdict-grade
    /// (`engine_is_sound_gpu_crown` false) and the preinitialized one IS — the
    /// same predicate `sound_gpu_gate` uses to authorize every verdict-carrying
    /// GPU CROWN route. A non-verdict-grade engine is a SPEED choice; swapping
    /// one for a verdict-grade engine cannot weaken a bound.
    ///
    /// Dark by default: `mo_cuda_factory_engine_handoff` is `false` and no
    /// shipped yaml sets it, so every current config is byte-identical.
    fn resolve_disjunctive_engine<'a>(
        &'a self,
        engine: Option<&'a dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Option<&'a dyn GemmEngine> {
        let caller = self.resolve_engine(engine);
        if !self.config.mo_cuda_factory_engine_handoff {
            disjunctive_engine_note("gate-off (bab.mo_cuda_factory_engine_handoff unset)");
            return caller;
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            disjunctive_engine_note("deadline-passed");
            return caller;
        }
        if caller.is_some_and(engine_is_sound_gpu_crown) {
            disjunctive_engine_note("caller-engine-already-sound-gpu-crown");
            return caller;
        }
        match crate::sound_f64_gemm::preinitialized_sound_gpu_engine() {
            Some(gpu) => {
                disjunctive_engine_note("ADMITTED: preferred sound GPU CROWN over caller engine");
                Some(gpu)
            }
            None => {
                disjunctive_engine_note(
                    "preinit-none (sound-CROWN slot empty: nothing satisfied \
                     provides_sound_gpu_crown)",
                );
                caller
            }
        }
    }
}

/// Build the verifier configuration used after the run-local dispatcher has
/// decided which clip route, if any, is authorized.
///
/// Both typed flags are cleared because the processing verifier itself runs
/// neither config-owned route: legacy child clipping is forbidden for this
/// graph class, while fresh current-domain authorization is carried solely by
/// the run-local telemetry capability.  Keeping the typed fresh flag here would
/// leave an internally invalid `BetaCrownConfig` after disabling the legacy
/// flag it depends on.
fn config_with_disjunctive_clip_routes_disabled(config: &BetaCrownConfig) -> BetaCrownConfig {
    let mut config = config.clone();
    config.enable_relaxed_clip = false;
    config.input_split_fresh_domain_clip = false;
    config
}

/// Rate limiter for the ALWAYS-ON stderr bound-trajectory line below.
///
/// The scored vnncomp entry point swallows `tracing::info!` (stated in-tree at
/// root_bounds.rs) and passes no `--input-split-metrics-jsonl` sink, so before
/// this line NO gap trajectory existed for the scored cgan route — every 900 s
/// run was blind on the ONE number that decides whether the tree is closing
/// (#cgan-reclaim runbook, docs/CGAN_BUDGET_RECLAIM_2026-08-20.md). Telemetry
/// is not a lever: this is unconditional, but WARN-equivalent rate-limited —
/// the first 3 batches, verdict-bearing batches, and otherwise at most one
/// line per 5 s, capped at 720 lines per process (≤ one 900 s run's worth).
fn should_eprint_batch_trajectory(force: bool, batch_index: usize) -> bool {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    static EMITTED: AtomicUsize = AtomicUsize::new(0);
    static LAST_EMIT_MS: AtomicU64 = AtomicU64::new(0);
    static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    const MAX_LINES: usize = 720;
    const MIN_INTERVAL_MS: u64 = 5_000;

    if EMITTED.load(Ordering::Relaxed) >= MAX_LINES {
        return false;
    }
    let now_ms = EPOCH
        .get_or_init(Instant::now)
        .elapsed()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    let timely = force || batch_index < 3 || {
        let last = LAST_EMIT_MS.load(Ordering::Relaxed);
        // First-ever emission (last == 0) always passes; afterwards enforce
        // the interval. A lost CAS race only skips one line — harmless.
        last == 0 || now_ms.saturating_sub(last) >= MIN_INTERVAL_MS
    };
    if !timely {
        return false;
    }
    LAST_EMIT_MS.store(now_ms.max(1), Ordering::Relaxed);
    EMITTED.fetch_add(1, Ordering::Relaxed);
    true
}

/// #cap-extend (DEFER-dont-DROP): near-closure gate for extending past the
/// domain cap instead of discarding a nearly-closed frontier.
///
/// The head priority is the gap `lower - threshold` of the most promising
/// unresolved domain (see `maybe_emit_batch_summary`); `closest_gap_seen` is
/// the maximum such gap observed across the run — "how close to closing did
/// the frontier ever get". The two conditions, BOTH required:
///
/// - the frontier reached within [`CAP_EXTEND_GAP_EPS`] of closure
///   (measured cgan2026 row 7: `best_gap=-0.0000` from t=196 s onward;
///   hopeless controls: row 10 base plateaued at `-1.61`, row 10 canary at
///   `-63.5` — three orders of magnitude on the same benchmark), and
/// - at least [`CAP_EXTEND_VERIFIED_NUM`]/[`CAP_EXTEND_VERIFIED_DEN`] of
///   explored domains actually verified (row 7 at its 2,500 trip: ~40%
///   verified; the row-10 stalls: 0%). This keeps a frontier that once
///   grazed zero but never closes anything from extending.
///
/// SCHEDULE-ONLY: granting or denying extension never touches bound math. A
/// wrong grant costs wall time already lost to the re-trace today; a wrong
/// denial is exactly today's behavior.
const CAP_EXTEND_GAP_EPS: f32 = 0.05;
/// Numerator of the minimum verified/explored fraction (see above).
const CAP_EXTEND_VERIFIED_NUM: usize = 1;
/// Denominator of the minimum verified/explored fraction (see above).
const CAP_EXTEND_VERIFIED_DEN: usize = 4;
/// One-time extension multiple of `max_domains`: DEFER, bounded — not a cap
/// lift. At row 7's measured ~6 domains/s the extended cap (2,500 → 20,000)
/// exceeds what any official wall can explore, so the wall governs; the
/// bound exists so a pathological fast-splitting frontier still terminates
/// on domain count (the FC cap-lift hazard,
/// docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md).
const CAP_EXTEND_FACTOR: usize = 8;

/// `#cap-extend-floor` — EXPERIMENT constant, `None` ships.
///
/// The productive test below demands `verified/explored >= 1/4`. Measured on
/// cgan `upsample_prop_0`, that is what denies a frontier sitting essentially
/// AT closure:
///
/// ```text
/// [cap-extend] denied: explored=2500 verified=91 open=2194
///                      closest_gap_seen=-0.0009 cap=2500
/// ```
///
/// **The GAP half passes with 55x margin (-0.0009 vs -0.05); only the 25%
/// verified test fails, at 3.6%.** So a tree with 2194 open domains, nine
/// ten-thousandths from closing, is discarded and REPLAYED from scratch purely
/// on a verification-rate heuristic — while the restart loop calls that same
/// seed "productive" in the same run. Two contradictory definitions of
/// productive, and the guard's own calibration (40% granted vs 0% denied)
/// never tested the 3.6% middle regime.
///
/// A prior session measured `0.01` here: the grant fired on the IDENTICAL
/// frontier it had denied, the whole-tree replay disappeared (restarts 2 -> 0),
/// explored went 1560 -> 4359, deepest 30 -> 74, and the frontier went from
/// verifying NOTHING per batch to 31-32 at `best_gap=-0.0000`. That change was
/// never committed; this restores the knob so it can be re-measured.
///
/// SCHEDULE-ONLY, as the doc above states: granting or denying never touches
/// bound math. A wrong grant costs wall already lost to the re-trace today.
/// `None` keeps the exact integer comparison, so the disarmed path is
/// bit-identical rather than merely equivalent.
pub(crate) const CAP_EXTEND_MIN_VERIFIED_FRAC: Option<f64> = None;

/// Decide whether a domain-cap trip may extend (true) or must stop exactly as
/// today (false). Pure so its truth table is directly testable.
fn cap_extend_decision(
    armed: bool,
    already_granted: bool,
    domains_explored: usize,
    domains_verified: usize,
    closest_gap_seen: f32,
) -> bool {
    if !armed || already_granted || domains_explored == 0 {
        return false;
    }
    // NaN closest_gap_seen fails the comparison and denies (fail-closed).
    let near_zero = closest_gap_seen >= -CAP_EXTEND_GAP_EPS;
    // #cap-extend-floor: `None` keeps the shipped exact integer comparison.
    let productive = match CAP_EXTEND_MIN_VERIFIED_FRAC {
        None => {
            domains_verified.saturating_mul(CAP_EXTEND_VERIFIED_DEN)
                >= domains_explored.saturating_mul(CAP_EXTEND_VERIFIED_NUM)
        }
        // A non-finite or negative floor would admit a stalled frontier, so it
        // falls back to the shipped rule rather than opening the gate.
        Some(frac) if frac.is_finite() && frac >= 0.0 => {
            domains_verified as f64 >= frac * domains_explored as f64
        }
        Some(_) => {
            domains_verified.saturating_mul(CAP_EXTEND_VERIFIED_DEN)
                >= domains_explored.saturating_mul(CAP_EXTEND_VERIFIED_NUM)
        }
    };
    near_zero && productive
}

/// #cap-extend: evaluate a domain-cap trip. Returns `true` when the one-time
/// extension has just been granted (the caller must CONTINUE searching);
/// `false` means stop exactly as today.
///
/// Every cap-enforcement site calls this, not just the loop head. With the
/// shipped `bab.batch_size: 64` the trip is almost always MID-BATCH — the
/// batch processors reach the cap and return before control returns to the
/// loop head. Measured on cgan2026 row 7 at the official 900 s: with the
/// grant wired only into the loop head, the armed canary produced NO
/// `[cap-extend]` line at all and timed out identically to production. An
/// extension that only the batch-size-1 unit fixture can reach is dead
/// capability.
fn try_grant_cap_extend(
    lifecycle: &mut GraphBabLifecycle,
    config: &BetaCrownConfig,
    open_domains: usize,
) -> bool {
    let armed = config.input_split_cap_extend_near_closure;
    let granted = cap_extend_decision(
        armed,
        lifecycle.cap_extend_effective_max_domains.is_some(),
        lifecycle.domains_explored,
        lifecycle.domains_verified,
        lifecycle.cap_extend_closest_gap_seen,
    );
    if granted {
        let extended = config.max_domains.saturating_mul(CAP_EXTEND_FACTOR);
        lifecycle.cap_extend_effective_max_domains = Some(extended);
        // The lever's own observable (armed-yaml reachability proof and A/B
        // marker; stderr, like the trajectory lines).
        eprintln!(
            "[cap-extend] granted: explored={} verified={} open={} \
             closest_gap_seen={:.4} cap {} -> {}",
            lifecycle.domains_explored,
            lifecycle.domains_verified,
            open_domains,
            lifecycle.cap_extend_closest_gap_seen,
            config.max_domains,
            extended,
        );
    } else if armed {
        eprintln!(
            "[cap-extend] denied: explored={} verified={} open={} \
             closest_gap_seen={:.4} cap={} granted_before={}",
            lifecycle.domains_explored,
            lifecycle.domains_verified,
            open_domains,
            lifecycle.cap_extend_closest_gap_seen,
            lifecycle.domain_cap(config.max_domains),
            lifecycle.cap_extend_effective_max_domains.is_some(),
        );
    }
    granted
}

fn maybe_emit_batch_summary(
    verifier: &BetaCrownVerifier,
    summary: &InputSplitBatchSummary,
    force: bool,
    queue_head: Option<(f32, usize)>,
    elapsed_s: f64,
) -> Result<()> {
    // Bound trajectory (#cgan lever-3 measurement): the queue head's priority
    // is the worst-clause best-row gap (`lower - threshold`) of the most
    // promising unresolved domain — the distance the bound must still climb
    // before the domain verifies. `best_gap`/`depth` across batches show
    // whether splits tighten the bound at all.
    let (best_gap, head_depth) = match queue_head {
        Some((gap, depth)) => (format!("{gap:.4}"), format!("{depth}")),
        None => ("n/a".to_string(), "n/a".to_string()),
    };

    // ALWAYS-ON stderr trajectory (see `should_eprint_batch_trajectory`).
    // Reads state and prints; touches no bound, no queue, no verdict.
    if should_eprint_batch_trajectory(force, summary.batch_index) {
        eprintln!(
            "[disjunctive-trajectory] t={elapsed_s:.1}s batch={} explored={} queue={} \
             best_gap={best_gap} head_depth={head_depth} verified_in_batch={} clipped={}",
            summary.batch_index,
            summary.domains_explored_after_batch,
            summary.queue_len_after_batch,
            summary.domains_verified_in_batch,
            summary.domains_clipped_in_batch,
        );
    }

    if !force && !should_log_batch(summary.batch_index) {
        return Ok(());
    }

    let record = summary.to_record();
    info!(
        "[disjunctive-multi-clause] batch={} popped={} queue={}->{} rebound={} rebound_s={:.3} forward_s={} backward_s={} materialize_s={} split_screen_s={:.3} dps={:.1} verified={} clipped={} best_gap={} head_depth={}",
        record.batch_index,
        record.popped_domains,
        record.queue_len_before_pop,
        record.queue_len_after_batch,
        record.rebound_mode.as_str(),
        record.rebound_total_s,
        format_optional_seconds(record.forward_s),
        format_optional_seconds(record.backward_s),
        format_optional_seconds(record.materialize_s),
        record.split_screen_s,
        record.domains_per_second,
        record.domains_verified_in_batch,
        record.domains_clipped_in_batch,
        best_gap,
        head_depth,
    );

    if let Some(sink) = verifier.input_split_metrics_sink() {
        sink.record_batch_summary(&record)?;
    }
    Ok(())
}

impl BetaCrownVerifier {
    /// LEVER 1 — the IMB early fast-path as a PUBLIC hook, so the CLI can run it AHEAD
    /// of the ~365 s CROWN-IBP per-output precheck (which the IMB neither needs nor
    /// consumes) instead of DOWNSTREAM of it. This is the SAME full-recheck logic as
    /// the in-lane `#imb-early` block below (`imb_multi_objective_floors` →
    /// `disjunctive_domain_verified`) — only the call POSITION moves.
    ///
    /// Returns `Some(Verified)` iff every disjunctive clause is IMB-refuted (→ the
    /// caller returns `unsat` immediately, skipping the precheck + collection + BaB);
    /// else `None` (→ the caller runs the UNCHANGED precheck + BaB). Gated
    /// `NY_IMB=1 && NY_IMB_WIRE=1` (opt-out `NY_IMB_EARLY=0`); on a disarmed/mismatched
    /// instance it is a no-op `None`. Sets `imb::mark_early_attempted()` so the in-lane
    /// block does NOT repeat the (expensive) leaf-BaB / tail-opt.
    ///
    /// SOUND: identical to the in-lane block — IMB only proposes an exact-cover input
    /// partition; each row is independently re-bounded on the original full network.
    /// `disjunctive_domain_verified` concludes UNSAT only when every clause has such a
    /// replay-certified row; any miss falls through to the standard pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn try_imb_early_disjunctive(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        clause_sizes: &[usize],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Option<BetaCrownResult> {
        // This public hook can return a final verdict before any of the normal
        // graph-verifier ingresses run. Reject quarantined/invalid
        // configuration before even consulting process-global IMB gates.
        if self.config.validate().is_err() {
            return None;
        }
        let imb_early_on = crate::imb::enabled()
            && matches!(std::env::var("NY_IMB_WIRE").ok().as_deref(), Some("1"))
            && !matches!(std::env::var("NY_IMB_EARLY").ok().as_deref(), Some("0"));
        self.try_imb_early_disjunctive_with_gate(
            graph,
            input,
            objectives,
            thresholds,
            clause_sizes,
            engine,
            deadline,
            imb_early_on,
        )
    }

    /// Gate-injected implementation so tests can exercise the armed preflight
    /// without mutating process-global environment variables.
    #[allow(clippy::too_many_arguments)]
    fn try_imb_early_disjunctive_with_gate(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        clause_sizes: &[usize],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        imb_early_on: bool,
    ) -> Option<BetaCrownResult> {
        // Defense in depth for the gate-injected implementation: future
        // internal callers and tests must not bypass the public preflight.
        if self.config.validate().is_err()
            || !imb_early_on
            || !valid_disjunctive_layout(objectives.len(), thresholds.len(), clause_sizes)
        {
            return None;
        }
        // Mark ATTEMPTED (verify or fall-through) so the downstream in-lane block skips.
        crate::imb::mark_early_attempted();
        let engine = self.resolve_engine(engine);
        // Configure the graph exactly as the in-lane path does (patches-mode + adopted
        // bound caches), so the seam-box CROWN and the IMB pipeline are byte-identical
        // to the in-lane fast-path — only the position differs.
        let configured = self.configured_graph_for_crown(graph);
        let graph = &configured;
        if !crate::imb::armed(graph, input) {
            return None;
        }
        let t_early = Instant::now();
        match graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline) {
            Ok(ibp_nb) => {
                let tightened = crate::imb::root_inject::imb_multi_objective_floors(
                    graph,
                    input,
                    objectives,
                    thresholds,
                    clause_sizes,
                    engine,
                    &ibp_nb,
                    deadline,
                );
                if disjunctive_domain_verified(&tightened, thresholds, clause_sizes) {
                    eprintln!(
                        "[imb] EARLY FAST-PATH (pre-precheck): VERIFIED in {:.1}s — skipping precheck + collection + BaB",
                        t_early.elapsed().as_secs_f64()
                    );
                    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
                    lifecycle.domains_explored = 1;
                    lifecycle.domains_verified = 1;
                    return Some(lifecycle.build_result(BabVerificationStatus::Verified));
                }
                eprintln!(
                    "[imb] EARLY FAST-PATH (pre-precheck): not verified in {:.1}s — falling through to precheck + BaB",
                    t_early.elapsed().as_secs_f64()
                );
            }
            Err(e) => {
                eprintln!(
                    "[imb] EARLY FAST-PATH (pre-precheck): IBP node-bounds failed ({e}); falling through"
                );
            }
        }
        None
    }

    /// Multi-clause disjunctive (OR-of-AND) input-split BaB.
    ///
    /// Spec rows packed clause-by-clause per `clause_sizes`. Verified when every
    /// clause has `lower > threshold` for at least one row.
    /// Reference: `stop_criterion_general` (`auto_LiRPA/utils.py:115-137`).
    /// `deadline`: If `Some`, the BaB engine derives its phase budgets from
    /// remaining wall-clock time instead of `self.config.timeout`.
    pub fn verify_graph_input_split_multi_clause_disjunctive(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        clause_sizes: &[usize],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        // #phase-cost: see `ny_core::phase_cost` (observability only).
        let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::BabLoop);
        graph.ensure_proof_eligible()?;
        self.config.validate()?;
        let engine = self.resolve_disjunctive_engine(engine, deadline);

        // Validate dimensions without allowing malformed packed layouts to
        // panic on overflow or introduce vacuous zero-width clauses.
        if objectives.is_empty() || objectives.len() != thresholds.len() {
            return Err(NyError::InvalidSpec(format!(
                "Disjunctive multi-clause: {} objectives vs {} thresholds",
                objectives.len(),
                thresholds.len()
            )));
        }
        if clause_sizes.is_empty() {
            return Err(NyError::InvalidSpec(
                "Disjunctive multi-clause: empty clause_sizes".to_string(),
            ));
        }
        if clause_sizes.contains(&0) {
            return Err(NyError::InvalidSpec(
                "Disjunctive multi-clause: zero-sized clause".to_string(),
            ));
        }
        let total_rows = clause_sizes
            .iter()
            .try_fold(0usize, |acc, &size| acc.checked_add(size))
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "Disjunctive multi-clause: clause_sizes total overflow".to_string(),
                )
            })?;
        if total_rows != objectives.len() {
            return Err(NyError::InvalidSpec(format!(
                "Disjunctive multi-clause: clause_sizes sum {} != {} objectives",
                total_rows,
                objectives.len()
            )));
        }

        // Reject non-finite thresholds early.
        for (i, &t) in thresholds.iter().enumerate() {
            if !t.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "Disjunctive multi-clause threshold[{}] is non-finite ({}); \
                     BaB cannot make progress with NaN/Inf thresholds",
                    i, t
                )));
            }
        }

        let graph = self.configured_graph_for_crown(graph);
        let graph = &graph;

        // #disj-cross-clause-clip-unsat: the RELAXED INPUT CLIP verifies domains
        // (box-infeasibility + the `concretize_postclip_lower_bounds` grouped
        // check) using the parent's REUSED CROWN linear bounds. On the
        // batch-stack-UNSAFE disjunctive tracks (lsnc_relu's MulBinary/Gather
        // difference nets, cgan's still-shape-unqualified Relu->BatchNorm,
        // linearizenn's Concat/Slice)
        // those reused bounds are NOT a sound lower bound over the clipped
        // sub-box — decisively reproduced on lsnc quadrotor2d_state_34: over the
        // sub-box around the ORT counterexample X the clip's concretize claims
        // `max Y_1 <= 0.40` while ORT gives `Y_1(X)=0.725`, `max Y_1 = 1.33`, so
        // every clause is spuriously refuted and the DOMAIN (which contains X) is
        // marked verified — a FALSE UNSAT that the upfront falsifier only
        // sometimes masks. Making the clip clause-aware keeps X inside the carried
        // UNION box but does NOT fix this, because the concretize bound itself is
        // unsound. Clause count is not a soundness precondition: the same reused
        // bound can be invalid for a batch-unsafe single-clause graph. So for the
        // entire batch-unsafe class we DISABLE the relaxed clip and let the SOUND
        // deferred re-bound
        // (`bound_deferred_disjunctive_domains_batch` + the top-of-loop grouped
        // check on freshly recomputed obj bounds) do all verification.
        //
        // This is STRICTLY BETTER than fail-closing the UNSAT verdict (the prior
        // #disj gate): the re-bound is sound and complete-in-the-limit, so real
        // UNSATs still resolve — measured with the clip off: cgan_2023 stays
        // `unsat` (169 s), linearizenn stays `unsat` (~0 s), lsnc_relu's
        // false-unsat becomes `unknown`/`sat`. Batch-SAFE disjunctions
        // (acasxu prop_5/6/9/10) keep the CLAUSE-AWARE clip — their CROWN bounds
        // ARE sound, the clip is load-bearing (acasxu prop_9 times out without
        // it), and the clause-aware carve fixes the cross-clause box shrink that
        // was latently unsound there.
        let disable_clip_unsound_class = should_disable_disjunctive_relaxed_clip(
            clause_sizes.len(),
            graph.is_input_split_batch_stack_safe(),
            self.config.enable_relaxed_clip,
        );
        let fresh_domain_clip_armed =
            should_enable_fresh_domain_clip(&self.config, graph.is_input_split_batch_stack_safe());
        crate::execution_telemetry::record_fresh_domain_clip_route(
            self.config.input_split_fresh_domain_clip,
            fresh_domain_clip_armed,
        );
        if self.config.input_split_fresh_domain_clip {
            eprintln!(
                "NY_FRESH_DOMAIN_CLIP route=grouped-disjunctive-current-domain \
                 status={} batch_stack_safe={} reorder_bab={} ibp_enhancement={}",
                if fresh_domain_clip_armed {
                    "armed"
                } else {
                    "inactive"
                },
                graph.is_input_split_batch_stack_safe(),
                self.config.reorder_bab,
                self.config.input_split_ibp_enhancement,
            );
        }
        let fresh_domain_clip_telemetry = FreshDomainClipTelemetry::new(fresh_domain_clip_armed);
        // #sb-plane: one unconditional line per lane start. `armed` is the
        // EFFECTIVE state — the key only changes anything while the fresh clip
        // is armed, because that is the only route that drops the plane. The
        // raw key and the clip state are printed beside it so "key set but
        // clip inactive" is readable rather than inferable.
        eprintln!(
            "[sb-plane] armed={} key={} fresh_clip_armed={}",
            fresh_domain_clip_armed && self.config.input_split_sb_keep_plane_under_fresh_clip,
            self.config.input_split_sb_keep_plane_under_fresh_clip,
            fresh_domain_clip_armed,
        );
        let clip_disabled_verifier;
        let bab: &BetaCrownVerifier = if disable_clip_unsound_class {
            let cfg = config_with_disjunctive_clip_routes_disabled(&self.config);
            debug_assert!(cfg.validate().is_ok());
            clip_disabled_verifier = self.with_config_from(cfg);
            &clip_disabled_verifier
        } else {
            self
        };

        let num_specs = objectives.len();
        let spec_dim = objectives[0].len();

        // Build multi-row spec matrix: each row is one C-matrix row (objective).
        let mut spec_data = Vec::with_capacity(num_specs * spec_dim);
        for obj in objectives {
            if obj.len() != spec_dim {
                return Err(NyError::InvalidSpec(format!(
                    "Objective dimension mismatch: {} vs {}",
                    obj.len(),
                    spec_dim
                )));
            }
            spec_data.extend_from_slice(obj);
        }
        let spec_matrix = Array2::from_shape_vec((num_specs, spec_dim), spec_data)
            .map_err(|e| NyError::InvalidSpec(format!("spec matrix: {}", e)))?;

        // EARLY IMB FAST-PATH (#imb-early). The self-contained Input-Manifold Bound
        // certificate is INDEPENDENT of the standard root collection (it rebuilds its
        // own tight prefix/tail anchors), but that collection is the pipeline's single
        // largest up-front cost (~313 s on cgan) and IMB needs ~250 s (180 s anchor +
        // ~40 s leaf-BaB + ~30 s tail-opt), so running IMB AFTER it can blow the wall
        // budget. Running IMB FIRST — before the collection — gives it the whole budget
        // and, when it refutes every clause, returns `Verified` immediately, skipping
        // the collection + BaB entirely.
        //
        // SOUND: IMB's sampled/decomposed floor only proposes terminal input boxes.
        // The wire validates their exact binary cover and independently re-bounds the
        // ORIGINAL full-network row on every leaf; only that replay lower can enter the
        // baseline. A candidate that fails any replay check returns baseline unchanged.
        // Therefore the tightened bounds are valid root lower bounds, and
        // `disjunctive_domain_verified` on them can only conclude `Verified` when every
        // clause is genuinely refuted. STRICTLY ADDITIVE: on any miss / disarmed / error
        // we fall through to the UNCHANGED standard pipeline (whose own late-stage
        // max-wiring stays as a second chance). Gate: `NY_IMB=1 && NY_IMB_WIRE=1`
        // (opt-out with `NY_IMB_EARLY=0`); default-OFF for every other benchmark.
        // SUPPRESSED when the CLI-level LEVER-1 hook already attempted the IMB (it runs
        // AHEAD of the precheck); re-running the leaf-BaB / tail-opt here would just
        // burn budget on the same result. When the hook never fired (non-Graph, etc.)
        // this stays the standalone in-lane fast-path.
        let imb_early_on = crate::imb::enabled()
            && matches!(std::env::var("NY_IMB_WIRE").ok().as_deref(), Some("1"))
            && !matches!(std::env::var("NY_IMB_EARLY").ok().as_deref(), Some("0"))
            && !crate::imb::early_attempted();
        if imb_early_on && crate::imb::armed(graph, input) {
            let t_early = Instant::now();
            match graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline) {
                Ok(ibp_nb) => {
                    // STEP 2 — MULTI-OBJECTIVE: use IMB to propose a partition for
                    // EVERY disjunctive clause's binding row, independently replay the
                    // original full-network row on all leaves, and raise only the
                    // replay-certified lowers, so
                    // `disjunctive_domain_verified` can conclude UNSAT only when every
                    // clause is genuinely refuted. Exact mode shares the prefix graph
                    // and tight anchor through one run-local graph/input-bound session;
                    // the tail (p,q) is per objective. For a single-clause prop
                    // (prop_0) this certifies exactly one objective — identical to the
                    // prior single-objective fast-path. A clause the IMB can't refute →
                    // not verified → the vacuous `-inf` baseline stands and we fall
                    // through unchanged.
                    let tightened = crate::imb::root_inject::imb_multi_objective_floors(
                        graph,
                        input,
                        objectives,
                        thresholds,
                        clause_sizes,
                        engine,
                        &ibp_nb,
                        deadline,
                    );
                    if disjunctive_domain_verified(&tightened, thresholds, clause_sizes) {
                        eprintln!(
                            "[imb] EARLY FAST-PATH: VERIFIED in {:.1}s — skipping standard collection + BaB",
                            t_early.elapsed().as_secs_f64()
                        );
                        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
                        lifecycle.domains_explored = 1;
                        lifecycle.domains_verified = 1;
                        return Ok(lifecycle.build_result(BabVerificationStatus::Verified));
                    }
                    eprintln!(
                        "[imb] EARLY FAST-PATH: not verified in {:.1}s — falling through to standard pipeline",
                        t_early.elapsed().as_secs_f64()
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[imb] EARLY FAST-PATH: IBP node-bounds failed ({e}); falling through"
                    );
                }
            }
        }

        let now = Instant::now();
        let mut lifecycle = GraphBabLifecycle::new(now);
        // A supplied absolute deadline is already the ledger-reserved BaB
        // boundary. Applying post_bab_pgd_fraction again here would reserve the
        // same post-BaB slice twice and make leaf deadlines unreachable.
        let pgd_frac = self
            .config
            .phase_budget
            .post_bab_pgd_fraction
            .clamp(0.0, 0.5);
        let bab_timeout =
            disjunctive_multi_clause_bab_timeout(self.config.timeout, pgd_frac, deadline, now);
        // #scored-wall-clamp. `bab_timeout` is the EFFORT HORIZON — the number
        // every per-node bound grant below is a fraction of, and the number a
        // caller may deliberately inflate past the scored budget to keep this
        // non-monotonic search on a converging trajectory. It is NOT permission
        // to run that long. `input_split_scored_wall`, when the frontend
        // supplies it, is the absolute instant this process owes a verdict by;
        // the loop stops there while effort keeps being shaped by the horizon.
        // Unset (the default, and every lane that has no inflation) leaves
        // `stop == horizon`, i.e. exactly the previous single-deadline code.
        let bab_wall = BabWall::clamped(bab_timeout, self.config.input_split_scored_wall, now);
        if bab_wall.is_clamped() {
            info!(
                horizon_secs = bab_wall.horizon().as_secs_f64(),
                stop_secs = bab_wall.stop().as_secs_f64(),
                "[disjunctive-multi-clause] scored wall binds: effort shaped against the \
                 horizon, loop stops at the wall"
            );
        }
        // #iso-edge-cap: the per-wave edge allowance policy. `None` — every
        // shipped preset — attaches nothing to the wall below and prints
        // nothing; armed, the report prints its ARMED line now and its `final`
        // line on every exit path of this function.
        let edge_wave_cap = edge_cap::resolve_edge_wave_cap(&self.config);
        let _edge_escalation_report = edge_wave_cap
            .map(|cap| edge_cap::EdgeEscalationReport::armed(cap, &self.config, bab_wall));
        let initial_deadline = {
            let frac = self
                .config
                .phase_budget
                .initial_bounds_fraction
                .clamp(0.0, 1.0);
            Some(GraphBabLifecycle::fail_closed_deadline(
                now,
                bab_timeout.mul_f32(frac),
            ))
        };
        // The ABSOLUTE BaB ceiling. Every per-child bound below is composed
        // against it, and no per-child call may ever run past it.
        let crown_deadline = Some(GraphBabLifecycle::fail_closed_deadline(now, bab_timeout));
        // #cgan-dark-tail FIX-1: the optional per-CHILD wall slice. `None` (the
        // shipped default for every preset) leaves `child_bound_deadline` a
        // pass-through and this whole lane byte-for-byte unchanged.
        //
        // HONEST CAVEAT — DO NOT READ THIS AS A CONVERSION LEVER. Recovering
        // the tail is NECESSARY, NOT SUFFICIENT. At the cliff the tree is
        // DIVERGING, measured on cgan row 2: `verified_in_batch=0`, `best_gap`
        // frozen, `queue` growing +1 per batch. Other cgan logs already explore
        // ~1.5x as many domains as the tail-stalled ones and STILL time out.
        // Handing the stalled ~290 s (32% of the official 900 s budget) back to
        // a search that is not closing children buys more expansions of the
        // same non-closing tree. What it does buy is a MEASURABLE run: with
        // 0-10 domains expanded in the last third of the budget, no downstream
        // lever can be A/B'd at all. Treat a converted row as a surprise, not
        // as the hypothesis.
        //
        // #cost-observer: the constant above is one of two arms. When the
        // preset names `child_bound_calibration` the allowance is DERIVED per
        // call from this row's own measured costs instead — see
        // [`ChildBoundSlicer`], which owns both arms and the soundness argument
        // they share.
        let child_bound_slicer = ChildBoundSlicer::new(&self.config);
        // Prints its engagement delta on EVERY exit path of the loop below
        // (verdict, timeout, domain limit, `?`); the inactive arm constructs
        // nothing and prints nothing.
        let _child_bound_cap_report = match &child_bound_slicer {
            ChildBoundSlicer::Constant(cap) => cap.map(ChildBoundCapReport::new),
            ChildBoundSlicer::Calibrated { .. } => None,
        };
        let _child_bound_calibration_report = match &child_bound_slicer {
            ChildBoundSlicer::Calibrated { observer, .. } => {
                Some(ChildBoundCalibrationReport(observer))
            }
            ChildBoundSlicer::Constant(_) => None,
        };
        child_bound_slicer.report_armed(bab_timeout);
        let mut domains_verified_by_clip = 0usize;

        let (root_node_bounds, root_alpha_state): (
            Option<HashMap<String, BoundedTensor>>,
            Option<GraphAlphaState>,
        ) = collect_input_split_root_node_bounds(
            graph,
            input,
            &self.config,
            engine,
            initial_deadline,
            input_split_root_authority_deadline(
                crown_deadline,
                self.config.input_split_scored_wall,
            ),
            "disjunctive-multi-clause input splitting",
            self.disjunctive_restart_root_cache(objectives, thresholds, clause_sizes, deadline)
                .map(|cache| (cache, deadline)),
        )?;

        // The root collector's map includes a certified enclosure for the output
        // node. In particular, the complete cGAN transaction can make that box
        // decisive even when a subsequent spec-CROWN backward is looser. Project
        // the already-paid-for output box through the exact packed spec and apply
        // the ordinary grouped stop criterion before starting any further root or
        // child work. `graph_spec_ibp_fallback` is the existing verdict-safe
        // reduction: it sanitizes non-finite values and accumulates in f64 with
        // outward f32 rounding. Any malformed-map error simply retains the
        // historical spec-CROWN path below.
        if let Some(root_map) = root_node_bounds.as_ref() {
            match graph_spec_ibp_fallback(graph, input, &spec_matrix, engine, Some(root_map))
                .and_then(|(bounds, _linear)| extract_obj_bounds(&bounds, num_specs))
            {
                Ok(root_map_obj_bounds)
                    if disjunctive_domain_verified(
                        &root_map_obj_bounds,
                        thresholds,
                        clause_sizes,
                    ) =>
                {
                    info!(
                        "[disjunctive-multi-clause] certified root-map output box verifies all clauses; skipping fresh spec-CROWN and child bounding"
                    );
                    lifecycle.domains_explored = 1;
                    lifecycle.domains_verified = 1;
                    return Ok(lifecycle.build_result(BabVerificationStatus::Verified));
                }
                Ok(_) => {}
                Err(err) => tracing::debug!(
                    error = %err,
                    "root-map clause retest declined; retaining fresh spec-CROWN path"
                ),
            }
        }

        // Phase 4: MulBinary SPSA alpha optimization.
        let mul_binary_alphas_multi = maybe_optimize_mul_binary_alphas(
            graph,
            input,
            &spec_matrix,
            engine,
            initial_deadline,
            self.config.crown_backward_layers,
            "Graph input split (disjunctive-multi-clause)",
        )?;

        let crown_bkwd = self.config.crown_backward_layers;
        let compute_bounds = |input_bounds: &BoundedTensor,
                              node_bounds: Option<&HashMap<String, BoundedTensor>>|
         -> Result<MultiObjBounds> {
            // #cgan-dark-tail FIX-1 / #cost-observer: this closure is invoked
            // ONCE PER CHILD bound, so the slicer's `Instant::now()` is that
            // call's own start and its `elapsed()` is that call's own cost.
            // An inactive slicer returns `crown_deadline` verbatim and records
            // nothing.
            child_bound_slicer.measured(crown_deadline, |deadline| {
                let (bounds, linear) = compute_crown_or_ibp_bounds_with_node_bounds(
                    graph,
                    input_bounds,
                    &spec_matrix,
                    engine,
                    root_node_bounds.as_ref(),
                    node_bounds,
                    root_alpha_state.as_ref(),
                    mul_binary_alphas_multi.as_ref(),
                    deadline,
                    crown_bkwd,
                    self.config.input_split_ibp_enhancement,
                )?;
                Ok((extract_obj_bounds(&bounds, num_specs)?, linear))
            })
        };

        // First safe slice of grouped-disjunctive per-domain alpha refinement:
        // eager BaB only. The reordered rebound is intentionally excluded because
        // its dense batch request currently carries one shared root alpha state,
        // not one inherited state per domain.
        let warm_alpha_enabled = eager_warm_alpha_enabled(&self.config, root_alpha_state.is_some());
        if self.config.input_split_alpha_iteration > 0 {
            if warm_alpha_enabled {
                eprintln!(
                    "NY_WARM_ALPHA route=grouped-disjunctive-eager status=enabled \
                     iterations={} lr={}",
                    self.config.input_split_alpha_iteration, self.config.input_split_lr_alpha
                );
            } else {
                eprintln!(
                    "NY_WARM_ALPHA route=grouped-disjunctive-eager status=inactive iterations={} \
                     reorder_bab={} use_alpha_crown={} root_alpha_available={}",
                    self.config.input_split_alpha_iteration,
                    self.config.reorder_bab,
                    self.config.use_alpha_crown,
                    root_alpha_state.is_some()
                );
            }
        }
        let warm_alpha_telemetry = screen_child::WarmAlphaTelemetry::new(warm_alpha_enabled);
        let warm_compute_bounds = |input_bounds: &BoundedTensor,
                                   node_bounds: Option<&HashMap<String, BoundedTensor>>,
                                   parent_alpha: &GraphAlphaState|
         -> Result<screen_child::WarmDisjunctiveBoundsResult> {
            // #cgan-dark-tail FIX-1 (warm step) / #cost-observer: per-call
            // allowance, same absolute ceiling.
            child_bound_slicer.measured(crown_deadline, |deadline| {
                let (bounds, linear, refined_alpha, anchors) =
                    super::shared::compute_warm_start_crown_bounds_with_refined_alpha(
                        graph,
                        input_bounds,
                        &spec_matrix,
                        engine,
                        node_bounds,
                        parent_alpha,
                        mul_binary_alphas_multi.as_ref(),
                        deadline,
                        crown_bkwd,
                        &self.config,
                    )?;
                Ok((
                    extract_obj_bounds(&bounds, num_specs)?,
                    linear,
                    refined_alpha,
                    anchors,
                ))
            })
        };
        let warm_compute_bounds_opt: Option<&screen_child::WarmDisjunctiveComputeBoundsFn<'_>> =
            if warm_alpha_enabled {
                Some(&warm_compute_bounds)
            } else {
                None
            };

        // #cgan-polish (NY_CGAN_POLISH=1): near-miss survivor alpha polish.
        // Same certified machinery as the warm closure above, with enough
        // iterations for real SPSA steps; invoked by `screen_disjunctive_child`
        // only for children already inside the knife-edge band.
        let polish_requested = cgan_polish_requested();
        let polish_config = cgan_polish_active(polish_requested, warm_alpha_enabled).then(|| {
            let mut cfg = self.config.clone();
            cfg.input_split_alpha_iteration = CGAN_POLISH_ALPHA_ITERATIONS;
            cfg
        });
        if polish_requested {
            match &polish_config {
                Some(_) => eprintln!(
                    "[cgan-polish] NY_CGAN_POLISH ARMED route=grouped-disjunctive-eager \
                     band={CGAN_POLISH_BAND:e} alpha_iterations={CGAN_POLISH_ALPHA_ITERATIONS} \
                     (near-miss survivors get real SPSA steps; the eager pass at iterations=1 \
                     performs zero)"
                ),
                None => eprintln!(
                    "[cgan-polish] NY_CGAN_POLISH INERT: eager warm-alpha lane not active \
                     (reorder_bab={} alpha_iteration={} use_alpha_crown={} \
                     root_alpha_available={})",
                    self.config.reorder_bab,
                    self.config.input_split_alpha_iteration,
                    self.config.use_alpha_crown,
                    root_alpha_state.is_some(),
                ),
            }
        }
        let polish_compute_bounds = |input_bounds: &BoundedTensor,
                                     node_bounds: Option<&HashMap<String, BoundedTensor>>,
                                     parent_alpha: &GraphAlphaState|
         -> Result<screen_child::WarmDisjunctiveBoundsResult> {
            let cfg = polish_config
                .as_ref()
                .expect("polish closure invoked without an armed polish config");
            // #cgan-dark-tail FIX-1 (polish step) / #cost-observer: per-call
            // allowance, same absolute ceiling.
            child_bound_slicer.measured(crown_deadline, |deadline| {
                let (bounds, linear, refined_alpha, anchors) =
                    super::shared::compute_warm_start_crown_bounds_with_refined_alpha(
                        graph,
                        input_bounds,
                        &spec_matrix,
                        engine,
                        node_bounds,
                        parent_alpha,
                        mul_binary_alphas_multi.as_ref(),
                        deadline,
                        crown_bkwd,
                        cfg,
                    )?;
                Ok((
                    extract_obj_bounds(&bounds, num_specs)?,
                    linear,
                    refined_alpha,
                    anchors,
                ))
            })
        };
        let survivor_polish: Option<screen_child::SurvivorPolish<'_>> =
            polish_config.is_some().then(|| {
                screen_child::SurvivorPolish::new(
                    screen_child::PolishParams {
                        band: CGAN_POLISH_BAND,
                        alpha_iterations: CGAN_POLISH_ALPHA_ITERATIONS,
                    },
                    &polish_compute_bounds,
                )
            });

        // #cgan-f64-tail (design B): run-scoped settings + ONE escalation
        // budget shared by the batch seam and the survivor-polish hook
        // (FIX B5). The hook itself is eager-lane only (the reordered path
        // has no polish seam) and additionally requires warm-refined anchors
        // at the call site — without them it never fires (anchor-source
        // vacuity guard, FIX B4/risk 4). Dark default: settings.enabled is
        // false unless the typed key or `NY_F64_TAIL=1` arms it, and then
        // every line below is stderr-visible (FIX B2).
        let f64_tail_settings = super::f64_tail::resolve_f64_tail_settings(&self.config);
        let f64_tail_budget = super::f64_tail::F64TailRunBudget::new();
        if f64_tail_settings.enabled {
            eprintln!(
                "[f64-tail] ARMED source={} band={:e} max_escalations={} route=grouped-disjunctive reorder_bab={}",
                f64_tail_settings.source,
                f64_tail_settings.band,
                f64_tail_settings.max_escalations,
                self.config.reorder_bab,
            );
        }
        let f64_tail_hook: Option<super::f64_tail::F64TailHook<'_>> =
            (f64_tail_settings.enabled && !self.config.reorder_bab).then(|| {
                super::f64_tail::F64TailHook::new(
                    graph,
                    &spec_matrix,
                    thresholds,
                    clause_sizes,
                    mul_binary_alphas_multi.as_ref(),
                    engine,
                    // The hook keeps the ABSOLUTE ceiling here on purpose: it
                    // composes `min(ceiling, now + call_slice)` itself, per
                    // call. #cgan-dark-tail FIX-1 therefore threads the cap as
                    // a DURATION below, never as a pre-composed `Instant` —
                    // this constructor runs once, at BaB start.
                    crown_deadline,
                    f64_tail_settings,
                    &f64_tail_budget,
                )
                // #cost-observer: the f64-tail hook deliberately keeps the
                // CONSTANT, and is NOT routed through `ChildBoundSlicer`.
                //
                // Three reasons, in order of how much they matter:
                //
                // 1. MIXING THE TWO SEAMS WOULD POISON THE ESTIMATE. The
                //    observer keys everything on the MEDIAN of one seam's own
                //    completed calls. f64-tail escalations cost a measured
                //    0.1235 s mean (363 escalations, 5.68 s total); per-child
                //    bounds on the same row run a 0.13 s median with a 70-285 s
                //    tail. Feeding both into one window would produce a median
                //    describing NEITHER, and the derived allowance would be
                //    wrong for both. One observer per cost distribution.
                // 2. THE BUILDER SHAPE CANNOT CARRY A DERIVED VALUE. This
                //    constructor runs ONCE, at BaB start; `with_child_bound_cap`
                //    folds a static `Duration` into `call_slice` via `min`. A
                //    per-call allowance has nowhere to live here.
                // 3. IT IS NOT THE BOTTLENECK. At 0.1235 s a mean, a cap of
                //    seconds never binds — this is a ceiling on a pathological
                //    case, not a schedule.
                //
                // WHICH CONFIGURATIONS REACH THIS SITE: `f64_tail.enabled` AND
                // `reorder_bab == false`. The f64 tail ships dark, so today it
                // is inert everywhere. Note the per-child slicer above is NOT
                // gated this way — its `compute_bounds` closure is
                // unconditional, so `[wall-calib] ARMED` prints and the seam is
                // live whatever `reorder_bab` says. Only `warm_compute_bounds`
                // and the polish closure additionally require
                // `!reorder_bab`.
                .with_child_bound_cap(resolve_child_bound_cap(&self.config))
            });

        // Root domain bounds.
        let (root_bounds, root_linear) = compute_crown_or_ibp_bounds_in_build_batches(
            graph,
            input,
            &spec_matrix,
            self.config.build_batch_size,
            engine,
            root_node_bounds.as_ref(),
            root_alpha_state.as_ref(),
            mul_binary_alphas_multi.as_ref(),
            initial_deadline,
            crown_bkwd,
            self.config.input_split_ibp_enhancement,
        )?;

        // #split-hint: publish the ROOT plane's per-input influence vector once,
        // so descendants that carry no plane of their own score by OUTPUT
        // SENSITIVITY instead of falling back to pure width. Scoring only —
        // `root_linear` itself still travels to `domain.linear_bounds` for the
        // certified clip paths, untouched.
        if let Some(plane) = root_linear.as_ref() {
            self.install_root_split_hint(plane, self.config.verify_upper_bound);
        }

        let root_obj_bounds = extract_obj_bounds(&root_bounds, num_specs)?;

        // Multi-neuron (k-ReLU) ROOT injection (#multineuron, NY_MULTINEURON=1;
        // conv-fed discriminator ReLUs via NY_MULTINEURON_CONV=1). Wires the proven
        // cifar100 root injection (multi_objective/root.rs) into the DISJUNCTIVE
        // input-split root so the cgan objective slack — which lives in the conv-fed
        // discriminator ReLUs — can be tightened BEFORE BaB (measured: prop_0 is
        // margin-bound, ~2e-4 short and uniform, so a tighter root is the lever, not
        // more splitting). SOUND-BY-CONSTRUCTION (Invariant MN): the function returns
        // the per-objective MAX over the baseline and every injected β-candidate, so
        // it can only tighten the margin, never over-claim; it returns the baseline
        // unchanged when disarmed, on a non-conv net, without alpha, or when no
        // facet-carrying group is found. No-op unless NY_MULTINEURON=1.
        let root_obj_bounds = match root_node_bounds.as_ref() {
            Some(nb) => crate::multineuron::root_inject::tighten_root_objective_bounds(
                graph,
                input,
                objectives,
                engine,
                nb,
                root_alpha_state.as_ref(),
                &root_obj_bounds,
                initial_deadline,
            ),
            None => root_obj_bounds,
        };

        // Input-Manifold Bound (IMB) ROOT floor (#imb, NY_IMB=1; default-OFF).
        // Proposes a partition for the hardest objective by keeping the generator
        // + first-disc-block prefix
        // EXACT and relaxing the disc tail to ONE alpha-optimized affine lower
        // functional `p·h(x)+q`, then certifying `min_x[p·h(x)+q]` via per-leaf
        // backward-CROWN input-split BaB over the free input dims. Without
        // `NY_IMB_WIRE=1` this remains log-only. With wiring, the decomposed floor
        // still has no authority: every terminal box is replayed against the
        // original full-network objective before any lower bound can be raised.
        // No-op unless NY_IMB=1 (the `else` arm is the byte-identical default).
        let root_obj_bounds = if crate::imb::enabled() {
            match root_node_bounds.as_ref() {
                Some(nb) => crate::imb::root_inject::tighten_root_objective_bounds_imb(
                    graph,
                    input,
                    objectives,
                    thresholds,
                    engine,
                    nb,
                    root_alpha_state.as_ref(),
                    &root_obj_bounds,
                    initial_deadline,
                ),
                None => root_obj_bounds,
            }
        } else {
            root_obj_bounds
        };

        // #cgan-root-census: unconditional, print-only, once per BaB start.
        print_cgan_root_census(
            graph,
            root_node_bounds.as_ref(),
            &root_obj_bounds,
            thresholds,
            clause_sizes,
        );

        info!(
            "[disjunctive-multi-clause] {} clauses, {} total rows, root bounds (alpha={}, forward_bounds={}): {}",
            clause_sizes.len(),
            num_specs,
            self.config.use_alpha_crown,
            self.config.use_forward_bounds,
            root_obj_bounds
                .iter()
                .zip(thresholds.iter())
                .map(|((l, u), &t)| format!("[{:.6}, {:.6}] thr={:.6}", l, u, t))
                .collect::<Vec<_>>()
                .join(", ")
        );

        if disjunctive_domain_verified(&root_obj_bounds, thresholds, clause_sizes) {
            lifecycle.domains_explored = 1;
            lifecycle.domains_verified = 1;
            return Ok(lifecycle.build_result(BabVerificationStatus::Verified));
        }
        if bab_wall.expired(lifecycle.start_time) {
            return Ok(lifecycle.timeout_result());
        }

        let root_priority = disjunctive_domain_priority(&root_obj_bounds, thresholds, clause_sizes);
        // Seed the root with its optimized α state for the deferred reordered
        // rebound. This gate is deliberately separate from `warm_alpha_enabled`:
        // that eager lane requires `!reorder_bab`, while reordered domains must
        // carry the seed without also running the eager refinement.
        let root_inherited_alpha =
            deferred_root_alpha_seed(&self.config, root_alpha_state.as_ref());
        let mut queue: BinaryHeap<MultiObjInputDomain> = BinaryHeap::new();
        queue.push(MultiObjInputDomain {
            input_bounds: Arc::new(input.clone()),
            obj_bounds: root_obj_bounds,
            linear_bounds: root_linear,
            depth: 0,
            priority: root_priority,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: root_inherited_alpha,
        });

        if self.config.reorder_bab {
            let loop_batch = input_split_loop_batch_size(self.config.batch_size, input.len())?;
            info!(
                requested_batch_size = loop_batch.requested_batch_size,
                effective_batch_size = loop_batch.effective_batch_size,
                clamp_reason = loop_batch.clamp_reason.as_str(),
                input_elems = input.len(),
                "[disjunctive-multi-clause] using reordered BaB (bound -> filter -> split -> clip)"
            );
        }

        let loop_batch_size = if self.config.reorder_bab {
            input_split_loop_batch_size(self.config.batch_size, input.len())?.effective_batch_size
        } else {
            1
        };
        let mut batch_index = 0usize;
        // #cap-extend: seed the closest-to-closure head gap with the root's own
        // gap. Both this and the grant live on the LIFECYCLE, because the cap
        // is enforced at three further per-domain sites inside the batch
        // processors and those are where the trip actually happens.
        lifecycle.cap_extend_closest_gap_seen = root_priority;
        // #scored-wall-clamp: the duration of the iteration just completed,
        // used to decline one that cannot finish inside the wall. `None` until
        // the first has completed; see `BabWall::cannot_finish`.
        let mut previous_iteration: Option<Duration> = None;

        while !queue.is_empty() {
            if bab_wall.expired(lifecycle.start_time) {
                dump_open_frontier(
                    self,
                    &queue,
                    "wall-expired",
                    lifecycle.domains_explored,
                    lifecycle.domains_verified,
                    thresholds,
                    clause_sizes,
                );
                return Ok(lifecycle.timeout_result());
            }
            if bab_wall.cannot_finish(lifecycle.start_time, previous_iteration) {
                info!(
                    elapsed_secs = lifecycle.start_time.elapsed().as_secs_f64(),
                    stop_secs = bab_wall.stop().as_secs_f64(),
                    previous_iteration_secs = previous_iteration.unwrap_or_default().as_secs_f64(),
                    "[disjunctive-multi-clause] declining a batch that cannot finish \
                     inside the scored wall"
                );
                dump_open_frontier(
                    self,
                    &queue,
                    "scored-wall-decline",
                    lifecycle.domains_explored,
                    lifecycle.domains_verified,
                    thresholds,
                    clause_sizes,
                );
                return Ok(lifecycle.timeout_result());
            }
            let iteration_start = Instant::now();
            // #cap-extend: with the lever DARK no grant is ever recorded, the
            // effective cap is exactly `max_domains`, and this arm is
            // byte-for-byte today's stop.
            let effective_max_domains = lifecycle.domain_cap(self.config.max_domains);
            if lifecycle.domains_explored >= effective_max_domains {
                if try_grant_cap_extend(&mut lifecycle, &self.config, queue.len()) {
                    continue;
                }
                dump_open_frontier(
                    self,
                    &queue,
                    "domain-limit",
                    lifecycle.domains_explored,
                    lifecycle.domains_verified,
                    thresholds,
                    clause_sizes,
                );
                return Ok(lifecycle.build_result(BabVerificationStatus::Unknown {
                    reason: format!(
                        "Domain limit {}: {}/{} verified",
                        effective_max_domains,
                        lifecycle.domains_verified,
                        lifecycle.domains_explored
                    ),
                }));
            }

            let queue_len_before_pop = queue.len();
            let mut domains = pop_multi_obj_input_domain_batch(&mut queue, loop_batch_size);
            let popped_domains = domains.len();
            let verified_before_batch = lifecycle.domains_verified;
            let clipped_before_batch = domains_verified_by_clip;
            let rebound = bound_deferred_disjunctive_domains_batch(
                &mut domains,
                graph,
                &spec_matrix,
                thresholds,
                clause_sizes,
                engine,
                root_node_bounds.as_ref(),
                root_alpha_state.as_ref(),
                mul_binary_alphas_multi.as_ref(),
                crown_deadline,
                crown_bkwd,
                &self.config,
                self.graph_domain_batch_metrics_sink(),
                batch_index,
                Some(&f64_tail_budget),
            )?;
            let split_screen_start = Instant::now();
            // #iso-edge-cap: this wave's edge allowance, derived from the
            // rebound that just completed (the freshest measurement of what a
            // wave of ordinary work costs on this row). Dark hands the batch
            // processors `bab_wall` itself.
            let wave_wall = match edge_wave_cap {
                Some(cap) => {
                    bab_wall.with_edge_wave(Some(cap.wave(batch_index, rebound.total_elapsed_s)))
                }
                None => bab_wall,
            };

            let batch_result = process_disjunctive_domain_batch(
                bab,
                graph,
                domains,
                &spec_matrix,
                thresholds,
                clause_sizes,
                engine,
                &compute_bounds,
                warm_compute_bounds_opt,
                &warm_alpha_telemetry,
                survivor_polish.as_ref(),
                f64_tail_hook.as_ref(),
                &fresh_domain_clip_telemetry,
                mul_binary_alphas_multi.as_ref(),
                wave_wall,
                &mut queue,
                &mut lifecycle,
                &mut domains_verified_by_clip,
            )?;

            let summary = InputSplitBatchSummary {
                batch_index,
                queue_len_before_pop,
                queue_len_after_batch: queue.len(),
                popped_domains,
                domains_explored_after_batch: lifecycle.domains_explored,
                domains_verified_in_batch: lifecycle.domains_verified - verified_before_batch,
                domains_clipped_in_batch: domains_verified_by_clip - clipped_before_batch,
                rebound,
                split_screen_elapsed_s: split_screen_start.elapsed().as_secs_f64(),
            };
            let should_force_emit = batch_result.is_some();
            let queue_head = queue.peek().map(|d| (d.priority, d.depth));
            // #cap-extend: NaN never advances the maximum (`>` is false).
            if let Some((gap, _)) = queue_head {
                if gap > lifecycle.cap_extend_closest_gap_seen {
                    lifecycle.cap_extend_closest_gap_seen = gap;
                }
            }
            maybe_emit_batch_summary(
                self,
                &summary,
                should_force_emit,
                queue_head,
                lifecycle.start_time.elapsed().as_secs_f64(),
            )?;
            batch_index += 1;
            previous_iteration = Some(iteration_start.elapsed());

            if let Some(result) = batch_result {
                return Ok(result);
            }
        }

        if domains_verified_by_clip > 0 {
            info!(
                "[disjunctive-multi-clause] domains_verified_by_clip={} out of {} verified ({} explored)",
                domains_verified_by_clip, lifecycle.domains_verified, lifecycle.domains_explored
            );
        }

        Ok(lifecycle.build_final_result())
    }
}

#[cfg(test)]
mod cap_extend_tests {
    use super::*;

    /// Row-7 shape at its measured 2,500 trip: gap grazed zero, ~40%
    /// verified. Flipping either predicate constant or the conjunction to a
    /// disjunction fails one of these.
    #[test]
    fn armed_near_closure_extends() {
        assert!(cap_extend_decision(true, false, 2_500, 1_000, -0.0001));
        // Boundary: exactly the verified fraction and exactly -eps still grant.
        assert!(cap_extend_decision(
            true,
            false,
            2_000,
            500,
            -CAP_EXTEND_GAP_EPS
        ));
    }

    /// Cap trips FAR from closure (row-10 base shape: gap plateau -1.61,
    /// verified 0) — must stop exactly as today even when armed.
    #[test]
    fn armed_far_from_closure_still_stops() {
        assert!(!cap_extend_decision(true, false, 2_500, 0, -1.6093));
        // Gap near zero but nothing ever verified: deny.
        assert!(!cap_extend_decision(true, false, 2_500, 0, -0.0001));
        // Productive but the frontier never approached closure: deny.
        assert!(!cap_extend_decision(true, false, 2_500, 1_000, -0.5));
        // Just under the verified-fraction floor: deny.
        assert!(!cap_extend_decision(true, false, 2_001, 500, -0.0001));
    }

    /// Dark (default) never extends, whatever the frontier looks like.
    #[test]
    fn dark_never_extends() {
        assert!(!cap_extend_decision(false, false, 2_500, 2_499, -0.0000001));
    }

    /// The grant is one-time: a second trip at the extended cap stops.
    #[test]
    fn extension_is_granted_at_most_once() {
        assert!(!cap_extend_decision(true, true, 20_000, 19_000, -0.0001));
    }

    /// Non-finite gap tracking fails closed.
    #[test]
    fn nan_gap_denies() {
        assert!(!cap_extend_decision(true, false, 2_500, 1_000, f32::NAN));
        assert!(!cap_extend_decision(
            true,
            false,
            2_500,
            1_000,
            f32::NEG_INFINITY
        ));
    }
}

#[cfg(test)]
mod deadline_budget_tests {
    use super::*;

    #[test]
    fn supplied_deadline_is_preserved_without_a_second_post_bab_reservation() {
        let now = Instant::now();
        let remaining = Duration::new(73, 123_456_789);
        let deadline = now + remaining;

        let timeout = disjunctive_multi_clause_bab_timeout(
            Duration::from_secs(100),
            0.25,
            Some(deadline),
            now,
        );

        assert_eq!(timeout, remaining);
        assert_eq!(now + timeout, deadline);
        assert!(
            now + timeout <= deadline,
            "derived boundary must never be later"
        );
    }

    #[test]
    fn absent_deadline_reserves_post_bab_slice_from_configured_timeout() {
        assert_eq!(
            disjunctive_multi_clause_bab_timeout(
                Duration::from_secs(100),
                0.25,
                None,
                Instant::now(),
            ),
            Duration::from_secs(75)
        );
    }

    // ── #scored-wall-clamp ────────────────────────────────────────────────
    //
    // These pin the invariant the whole change rests on: the derived stop is
    // never later than the scored wall, and never later than the horizon the
    // caller configured, and with no wall supplied nothing moves at all.

    /// DEFAULTS UNCHANGED. No wall ⇒ the stop IS the horizon, so every lane
    /// and every caller that supplies nothing runs the pre-split code.
    #[test]
    fn absent_scored_wall_leaves_stop_equal_to_horizon() {
        let now = Instant::now();
        let horizon = Duration::from_secs(72);
        let wall = BabWall::clamped(horizon, None, now);

        assert_eq!(wall.horizon(), horizon);
        assert_eq!(wall.stop(), horizon);
        assert!(
            !wall.is_clamped(),
            "no wall can be binding when there is none"
        );
    }

    /// THE DEFECT. An inflated horizon (the measured 1.4x relational
    /// multiplier: 72 s of real budget shaped as 100.8 s) keeps shaping
    /// effort, but the loop stops at the wall — never at the horizon.
    #[test]
    fn an_inflated_horizon_still_stops_at_the_scored_wall() {
        let now = Instant::now();
        let real_remaining = Duration::from_secs(72);
        let horizon = real_remaining.mul_f64(1.4);
        let wall = BabWall::clamped(horizon, Some(now + real_remaining), now);

        assert_eq!(
            wall.horizon(),
            horizon,
            "effort must still be shaped against the inflated horizon"
        );
        assert_eq!(
            wall.stop(),
            real_remaining,
            "the loop must stop at the wall"
        );
        assert!(wall.is_clamped());
        assert!(
            wall.stop() <= wall.horizon(),
            "the derived stop may never exceed the horizon"
        );
    }

    /// A WALL CANNOT EXTEND A RUN. Handed a wall beyond the configured
    /// horizon, the horizon still wins — this mechanism only ever shortens.
    #[test]
    fn a_wall_past_the_horizon_never_extends_the_run() {
        let now = Instant::now();
        let horizon = Duration::from_secs(30);
        let wall = BabWall::clamped(horizon, Some(now + Duration::from_mins(5)), now);

        assert_eq!(wall.stop(), horizon);
        assert!(!wall.is_clamped());
    }

    /// A WALL ALREADY PAST is a zero stop, not a wrap. `Instant` arithmetic
    /// panics on overflow in debug and saturates in release; neither is an
    /// acceptable way to compute a deadline, so the constructor saturates
    /// explicitly and the next termination check fires immediately.
    #[test]
    fn a_wall_already_passed_yields_an_immediate_stop() {
        let now = Instant::now();
        let horizon = Duration::from_secs(72);
        let wall = BabWall::clamped(
            horizon,
            Some(now.checked_sub(Duration::from_secs(5)).unwrap()),
            now,
        );

        assert_eq!(wall.stop(), Duration::ZERO);
        assert!(
            wall.expired(now.checked_sub(Duration::from_millis(1)).unwrap()),
            "a run that has already started is over the moment the wall has passed"
        );
    }

    /// THE GUARD IS INERT WITHOUT A WALL. `cannot_finish` must never change an
    /// unclamped run, however long its iterations are — otherwise every family
    /// that does not arm the clamp would silently start truncating its search.
    #[test]
    fn the_cannot_finish_guard_is_inert_without_a_wall() {
        let now = Instant::now();
        let wall = BabWall::clamped(Duration::from_secs(10), None, now);
        assert!(!wall.cannot_finish(now, Some(Duration::from_hours(1))));
    }

    /// THE ESTIMATE IS THE PREVIOUS ITERATION, NOT THE LONGEST. A single slow
    /// batch must not pessimise every later decision — measured, that cost a
    /// row that was otherwise decided inside the budget.
    #[test]
    fn the_cannot_finish_guard_forgets_an_earlier_slow_iteration() {
        let now = Instant::now();
        let wall = BabWall::clamped(
            Duration::from_secs(100),
            Some(now + Duration::from_secs(10)),
            now,
        );
        let start = now.checked_sub(Duration::from_secs(7)).unwrap(); // ~3 s left of the 10 s stop
                                                                      // A 5 s iteration happened earlier and is now behind us; the iteration
                                                                      // just completed took 1 s. The guard must consult the latter.
        assert!(
            !wall.cannot_finish(start, Some(Duration::from_secs(1))),
            "a slow batch in the past must not veto a fast one now"
        );
    }

    /// THE FIRST ITERATION IS ALWAYS ATTEMPTED. With no measurement yet there
    /// is nothing to predict with, and refusing would turn a clamped run into
    /// no search at all.
    #[test]
    fn the_cannot_finish_guard_never_refuses_the_first_iteration() {
        let now = Instant::now();
        let wall = BabWall::clamped(
            Duration::from_secs(100),
            Some(now + Duration::from_millis(1)),
            now,
        );
        assert!(wall.is_clamped());
        assert!(!wall.cannot_finish(now, None));
    }

    /// AN ITERATION THAT DOES NOT FIT IS DECLINED, one that does is not.
    /// This is what bounds the tail past the wall to the ERROR in the estimate
    /// rather than to a whole batch.
    #[test]
    fn the_cannot_finish_guard_declines_only_what_does_not_fit() {
        let now = Instant::now();
        let wall = BabWall::clamped(
            Duration::from_secs(100),
            Some(now + Duration::from_secs(10)),
            now,
        );
        let start = now.checked_sub(Duration::from_secs(7)).unwrap(); // ~3 s left of the 10 s stop
        assert!(
            !wall.cannot_finish(start, Some(Duration::from_secs(1))),
            "a 1 s iteration fits in the ~3 s remaining"
        );
        assert!(
            wall.cannot_finish(start, Some(Duration::from_secs(5))),
            "a 5 s iteration does not, and must be declined rather than started"
        );
    }

    /// A DECISION LANDING JUST INSIDE THE WALL IS KEPT. `expired` is strict
    /// `>`, so a run whose elapsed time equals the stop is still live; only
    /// time PAST the wall terminates it. This is the property that keeps a row
    /// deciding in its final second from being demoted to `timeout`.
    #[test]
    fn a_run_just_inside_the_wall_is_not_expired() {
        let now = Instant::now();
        let wall = BabWall::clamped(
            Duration::from_secs(100),
            Some(now + Duration::from_secs(10)),
            now,
        );

        // start_time far enough back that elapsed() is ~9.99 s: inside.
        assert!(!wall.expired(now.checked_sub(Duration::from_millis(9_990)).unwrap()));
        // ... and elapsed() ~10.5 s: outside.
        assert!(wall.expired(now.checked_sub(Duration::from_millis(10_500)).unwrap()));
    }
}

/// #cgan-dark-tail FIX-1: the per-CHILD bound wall cap.
///
/// The property under test is a composition law, not a bound: the value handed
/// to a per-child bound call must be `min(absolute ceiling, now + cap)`, and
/// with no cap it must be the ceiling ITSELF — not a recomputed approximation
/// of it — so that the shipped default is byte-for-byte today's behaviour.
#[cfg(test)]
mod child_bound_cap_tests {
    use super::*;

    fn cfg_with_cap(secs: Option<f64>) -> BetaCrownConfig {
        BetaCrownConfig {
            input_split_child_bound_max_secs: secs,
            ..Default::default()
        }
    }

    fn cfg_calibrated() -> BetaCrownConfig {
        BetaCrownConfig {
            input_split_child_bound_calibration: Some(
                crate::beta_crown::config::ChildBoundCalibrationConfig::default(),
            ),
            ..Default::default()
        }
    }

    /// #cost-observer, ARM SELECTION: naming the calibration block selects the
    /// calibrated arm; leaving it out keeps the constant, including its `None`.
    #[test]
    fn the_calibration_block_selects_the_calibrated_arm() {
        assert!(matches!(
            ChildBoundSlicer::new(&BetaCrownConfig::default()),
            ChildBoundSlicer::Constant(None)
        ));
        assert!(matches!(
            ChildBoundSlicer::new(&cfg_with_cap(Some(20.0))),
            ChildBoundSlicer::Constant(Some(_))
        ));
        assert!(matches!(
            ChildBoundSlicer::new(&cfg_calibrated()),
            ChildBoundSlicer::Calibrated { .. }
        ));
    }

    /// #cost-observer, THE STRICT-IMPROVEMENT LEG: when the observer declines
    /// to bound a call — measurement unavailable — the allowance is the HAND-SET
    /// CONSTANT, not something new.
    ///
    /// For every shipped preset that constant is `None`, so the uncalibrated
    /// path is the caller's own ceiling, i.e. byte-for-byte today. A preset that
    /// sets both gets the constant it asked for. Neither is a fresh guess, which
    /// is the difference between calibrating a budget and re-tuning one.
    #[test]
    fn an_uncalibrated_call_falls_back_to_the_hand_set_constant() {
        // Constant absent (every shipped preset): fall back to the ceiling.
        //
        // NOTE: no warm-up `measured` call before forcing the state. The first
        // version of this test made one "harmless" `Ok(())` call here, and the
        // slicer — correctly, per its own three-way classification — recorded
        // that fast `Ok` as a COMPLETED cost sample, so the observer was never
        // in the "nothing ever completed" state this test exercises and it
        // calibrated a share-term slice (ceiling - 450 s) instead of falling
        // back. The mechanism was right and the test was wrong; keep the state
        // injection direct.
        let bare = ChildBoundSlicer::new(&cfg_calibrated());
        let ChildBoundSlicer::Calibrated { observer, fallback } = &bare else {
            panic!("expected the calibrated arm");
        };
        assert_eq!(
            *fallback, None,
            "no constant named => no cap when uncalibrated"
        );
        // Force the "nothing ever completes" state and confirm the pass-through.
        observer.observe(ny_core::cost_observer::Observation::Truncated(
            Duration::from_secs(1),
        ));
        let ceil = Instant::now() + Duration::from_mins(15);
        let seen = bare.measured(Some(ceil), Ok::<_, NyError>).unwrap();
        assert_eq!(seen, Some(ceil), "uncalibrated + no constant == today");

        // Constant ALSO named: that is what the uncalibrated path uses.
        let both = ChildBoundSlicer::new(&BetaCrownConfig {
            input_split_child_bound_max_secs: Some(20.0),
            input_split_child_bound_calibration: Some(
                crate::beta_crown::config::ChildBoundCalibrationConfig::default(),
            ),
            ..Default::default()
        });
        let ChildBoundSlicer::Calibrated { observer, fallback } = &both else {
            panic!("expected the calibrated arm");
        };
        assert_eq!(*fallback, Some(Duration::from_secs(20)));
        observer.observe(ny_core::cost_observer::Observation::Truncated(
            Duration::from_secs(1),
        ));
        let ceil = Instant::now() + Duration::from_mins(15);
        let seen = both
            .measured(Some(ceil), Ok::<_, NyError>)
            .unwrap()
            .expect("a ceiling in must produce a deadline out");
        assert!(
            seen < ceil,
            "the named constant must bind while measurement is unavailable"
        );
    }

    /// #cost-observer, THE DEADLINE-PRESENCE GUARD: with no ceiling the
    /// calibrated arm must not INVENT one. Deadline presence — not expiry — is
    /// its own measured degradation here (`#deadline-presence`: four root
    /// causes, two of which discard optimized alpha), so a resource lever must
    /// not create a deadline as a side effect of budgeting.
    #[test]
    fn the_calibrated_arm_never_invents_a_deadline_where_there_was_none() {
        let slicer = ChildBoundSlicer::new(&cfg_calibrated());
        let seen = slicer.measured(None, Ok::<_, NyError>).unwrap();
        assert_eq!(seen, None, "no ceiling in, no deadline out");
    }

    /// #cost-observer, BYTE-IDENTITY LEG: with no calibration block the slicer
    /// hands the closure exactly the ceiling it was given and records nothing,
    /// so the shipped route is unchanged.
    #[test]
    fn the_unarmed_slicer_passes_the_ceiling_through_untouched() {
        let slicer = ChildBoundSlicer::new(&BetaCrownConfig::default());
        let ceiling = Some(Instant::now() + Duration::from_mins(15));
        let seen = slicer.measured(ceiling, Ok::<_, NyError>).unwrap();
        assert_eq!(seen, ceiling, "no cap, no calibration: pass-through");
    }

    /// #cost-observer, THE MOAT ARGUMENT AS A TEST: whatever the observer has
    /// learned, and however many truncations it has seen, the deadline handed
    /// to a bound call is NEVER later than the ceiling.
    ///
    /// This is the property that makes the change verdict-neutral by
    /// construction. Budgets decide how much work a bound gets; they never
    /// decide what is concluded from it, and an allowance that only ever brings
    /// a deadline FORWARD can only make a bound looser — which cannot certify
    /// anything a tighter bound would not have.
    #[test]
    fn no_calibrated_state_can_ever_push_a_deadline_past_the_ceiling() {
        let slicer = ChildBoundSlicer::new(&cfg_calibrated());
        // Drive the observer through completions, truncations and the growth
        // escalation that follows a run of truncations.
        for round in 0..64u32 {
            let ceiling_at = Instant::now() + Duration::from_millis(50);
            let ceiling = Some(ceiling_at);
            let seen = slicer
                .measured(ceiling, |deadline| {
                    if round % 7 == 0 {
                        std::thread::sleep(Duration::from_micros(200));
                    }
                    Ok::<_, NyError>(deadline)
                })
                .unwrap();
            let seen = seen.expect("a ceiling in must produce a deadline out");
            assert!(
                seen <= ceiling_at,
                "round {round}: calibrated deadline ran past the absolute ceiling"
            );
        }
    }

    /// #cost-observer: an `Err` that did NOT spend its allowance is not a cost
    /// sample and not a truncation. Calling it either would corrupt the median
    /// with a failure that has nothing to do with wall time, or grow the
    /// allowance in response to evidence about a different problem entirely.
    #[test]
    fn a_fast_error_is_recorded_as_neither_completed_nor_truncated() {
        let slicer = ChildBoundSlicer::new(&cfg_calibrated());
        let ceiling = Some(Instant::now() + Duration::from_mins(15));
        for _ in 0..8 {
            let outcome: Result<()> = slicer.measured(ceiling, |_| {
                Err(NyError::InternalError("not a deadline problem".into()))
            });
            assert!(outcome.is_err());
        }
        let ChildBoundSlicer::Calibrated { observer, .. } = &slicer else {
            panic!("expected the calibrated arm");
        };
        let report = observer.report();
        assert_eq!(report.completed, 0, "an error is not a cost sample");
        assert_eq!(report.truncated, 0, "and it is not a truncation either");
        assert_eq!(report.derived, 8, "but the allowance was still derived");
    }

    /// #cost-observer: a call that spends its whole allowance is a TRUNCATION
    /// even when it returns `Ok`.
    ///
    /// This is the `upsample` shape and the single most important
    /// classification in the mechanism: those 37 calls all returned `Ok` after
    /// commit-what-you-have cut a 27-layer collection to 5-6 layers, with
    /// `elapsed` pinned at the cap. Counting them as healthy completions would
    /// have pinned the estimator at exactly the allowance that was destroying
    /// the work, and the allowance could never have grown.
    #[test]
    fn a_call_that_spends_its_whole_allowance_is_truncated_even_when_it_returns_ok() {
        let slicer = ChildBoundSlicer::new(&BetaCrownConfig {
            input_split_child_bound_calibration: Some(
                crate::beta_crown::config::ChildBoundCalibrationConfig {
                    // A probe share large enough that the derived allowance is
                    // a handful of milliseconds against the ceiling below.
                    probe_share: Some(0.5),
                    ..Default::default()
                },
            ),
            ..Default::default()
        });
        let ceiling = Some(Instant::now() + Duration::from_millis(20));
        let _ = slicer.measured(ceiling, |_| {
            std::thread::sleep(Duration::from_millis(30));
            Ok::<_, NyError>(())
        });
        let ChildBoundSlicer::Calibrated { observer, .. } = &slicer else {
            panic!("expected the calibrated arm");
        };
        let report = observer.report();
        assert_eq!(report.truncated, 1, "Ok + allowance spent == truncated");
        assert_eq!(report.completed, 0);
    }

    /// The shipped default: no key named anywhere => no cap.
    #[test]
    fn default_config_resolves_to_no_cap() {
        assert_eq!(resolve_child_bound_cap(&BetaCrownConfig::default()), None);
        assert_eq!(resolve_child_bound_cap(&cfg_with_cap(None)), None);
    }

    /// A named, well-formed value becomes exactly that slice.
    #[test]
    fn named_value_resolves_to_that_duration() {
        assert_eq!(
            resolve_child_bound_cap(&cfg_with_cap(Some(20.0))),
            Some(Duration::from_secs(20))
        );
        assert_eq!(
            resolve_child_bound_cap(&cfg_with_cap(Some(0.25))),
            Some(Duration::from_millis(250))
        );
    }

    /// A malformed value degrades to "no cap" rather than panicking:
    /// `Duration::from_secs_f64` PANICS on negative / non-finite / overflowing
    /// input and a preset is untrusted. The clamp keeps the overflow case away
    /// from that panic.
    #[test]
    fn malformed_values_degrade_to_no_cap_and_never_panic() {
        for bad in [0.0, -1.0, f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            assert_eq!(
                resolve_child_bound_cap(&cfg_with_cap(Some(bad))),
                None,
                "{bad} must resolve to no cap"
            );
        }
        // Finite but absurd: clamped, still constructible.
        assert_eq!(
            resolve_child_bound_cap(&cfg_with_cap(Some(1.0e30))),
            Some(Duration::from_secs_f64(CHILD_BOUND_CAP_MAX_SECS))
        );
    }

    /// `None` reproduces today's value BIT-FOR-BIT: the very same `Instant`
    /// object, not a re-derived one.
    #[test]
    fn no_cap_returns_the_ceiling_unchanged() {
        let ceiling = Instant::now() + Duration::from_secs(895);
        assert_eq!(child_bound_deadline(Some(ceiling), None), Some(ceiling));
        assert_eq!(child_bound_deadline(None, None), None);
    }

    /// The composition law, plus the TEETH: a cap that is silently ignored
    /// would return the far ceiling and fail the strict-inequality assert.
    #[test]
    fn cap_composes_as_min_and_has_teeth() {
        let cap = Duration::from_secs(5);
        let before = Instant::now();
        let ceiling = before + Duration::from_secs(895);

        let composed = child_bound_deadline(Some(ceiling), Some(cap))
            .expect("a cap always yields a finite deadline");
        let after = Instant::now();

        // TEETH: an ignored cap returns `ceiling`, which is 890 s later.
        assert!(
            composed < ceiling,
            "a 5 s cap under an 895 s ceiling MUST bring the deadline forward; \
             got the ceiling back, so the cap was ignored"
        );
        // It is exactly `now + cap` for some `now` inside the call.
        assert!(composed >= before + cap);
        assert!(composed <= after + cap);
    }

    /// The ceiling is a HARD limit: a cap larger than the remaining budget
    /// must NOT extend it.
    #[test]
    fn cap_never_exceeds_the_absolute_ceiling() {
        let ceiling = Instant::now() + Duration::from_millis(10);
        let composed = child_bound_deadline(Some(ceiling), Some(Duration::from_hours(1)))
            .expect("a cap always yields a finite deadline");
        assert_eq!(
            composed, ceiling,
            "an over-long cap must leave the absolute ceiling in force"
        );
        assert!(composed <= ceiling);
    }

    /// An already-reached ceiling stays reached — the cap cannot resurrect it.
    /// (Built by capturing `now` rather than subtracting from it: `Instant -
    /// Duration` panics on underflow and `Instant`'s epoch is not portable.)
    #[test]
    fn reached_ceiling_stays_reached_under_a_cap() {
        let ceiling = Instant::now();
        let composed = child_bound_deadline(Some(ceiling), Some(Duration::from_secs(30)))
            .expect("a cap always yields a finite deadline");
        assert_eq!(composed, ceiling);
        assert!(composed <= Instant::now());
    }

    /// With no ceiling at all, the cap still bounds the call. This lane always
    /// supplies a ceiling today, so this only pins the helper's contract.
    #[test]
    fn cap_applies_even_without_a_ceiling() {
        let before = Instant::now();
        let composed = child_bound_deadline(None, Some(Duration::from_secs(7)))
            .expect("a cap always yields a finite deadline");
        assert!(composed >= before + Duration::from_secs(7));
    }

    /// R7/R9: the lever's OWN engagement telemetry must move when it engages.
    ///
    /// The counters are process-global on purpose (see their doc comment), and
    /// `cargo test` runs this module's cases concurrently in one process, so
    /// only concurrency-safe claims are asserted: deltas as LOWER BOUNDS.
    /// Notably NOT asserted: `hits <= calls`. It is true of the counters, but
    /// reading them takes two non-atomic loads, so a sibling thread
    /// incrementing between them can invert the observation. The exact per-run
    /// numbers come from [`ChildBoundCapReport`], which snapshots its own
    /// baseline in a single-threaded BaB run.
    #[test]
    fn engagement_counters_move_when_the_cap_binds() {
        use std::sync::atomic::Ordering;

        let calls_before = CHILD_BOUND_CAP_CALLS.load(Ordering::Relaxed);
        let hits_before = CHILD_BOUND_CAP_HITS.load(Ordering::Relaxed);

        // Binds: a 1 s cap under a 900 s ceiling.
        let far = Instant::now() + Duration::from_mins(15);
        let _ = child_bound_deadline(Some(far), Some(Duration::from_secs(1)));

        let calls_delta = CHILD_BOUND_CAP_CALLS
            .load(Ordering::Relaxed)
            .saturating_sub(calls_before);
        let hits_delta = CHILD_BOUND_CAP_HITS
            .load(Ordering::Relaxed)
            .saturating_sub(hits_before);
        // Lower bounds only: sibling test threads share these statics, so an
        // equality (or a cross-counter comparison, whose two loads cannot be
        // taken atomically) would be flaky by construction.
        assert!(
            calls_delta >= 1,
            "a capped composition must be counted as a call"
        );
        assert!(
            hits_delta >= 1,
            "a cap nearer than the ceiling must be counted as a HIT — without \
             this the armed/engaged distinction is unobservable and any null is \
             vacuous"
        );
    }

    /// The dark default must not even touch the counters: `None` is the
    /// pass-through arm, and a lever that ships off must leave NO trace.
    #[test]
    fn no_cap_leaves_the_engagement_counters_alone() {
        use std::sync::atomic::Ordering;

        // Serialized against nothing — this asserts an equality that only
        // holds if no OTHER thread increments, so it reads the counters
        // around a no-cap call and requires the delta to be attributable
        // entirely to concurrent capped calls, never to this one. Expressed
        // as: a `None` call returns the ceiling ITSELF, which is the
        // observable that proves it took the pass-through arm.
        let far = Instant::now() + Duration::from_mins(15);
        let before = CHILD_BOUND_CAP_HITS.load(Ordering::Relaxed);
        assert_eq!(child_bound_deadline(Some(far), None), Some(far));
        // Weak but non-flaky: the pass-through arm cannot DECREASE anything.
        assert!(CHILD_BOUND_CAP_HITS.load(Ordering::Relaxed) >= before);
    }

    /// The run report is a DELTA reporter: a restart run inside one process
    /// must not inherit the previous run's counts.
    #[test]
    fn report_snapshots_a_baseline_rather_than_the_total() {
        use std::sync::atomic::Ordering;

        let far = Instant::now() + Duration::from_mins(15);
        // Bank some history first.
        let _ = child_bound_deadline(Some(far), Some(Duration::from_secs(1)));
        let report = ChildBoundCapReport::new(Duration::from_secs(1));
        assert!(
            report.calls_at_start >= 1,
            "the baseline is taken at construction, NOT at zero — a run that \
             starts after history exists must not re-report it"
        );
        assert!(
            CHILD_BOUND_CAP_CALLS.load(Ordering::Relaxed) >= report.calls_at_start,
            "the counters are monotone, so the reported delta is never negative"
        );
        assert_eq!(report.cap, Duration::from_secs(1));
    }
}

#[cfg(test)]
mod cgan_polish_gate_tests {
    use super::*;

    /// The campaign's exact-token convention: only `"1"` arms.
    #[test]
    fn polish_token_is_exact_one() {
        assert!(cgan_polish_token_arms(Some("1")));
        assert!(!cgan_polish_token_arms(Some("0")));
        assert!(!cgan_polish_token_arms(Some("true")));
        assert!(!cgan_polish_token_arms(Some("")));
        assert!(!cgan_polish_token_arms(Some(" 1")));
        assert!(!cgan_polish_token_arms(Some("01")));
        assert!(!cgan_polish_token_arms(None));
    }

    /// The polish rides the eager warm-alpha lane: requested without the lane
    /// is INERT (and announced as such), never a silent semi-armed state.
    #[test]
    fn polish_requires_the_eager_warm_alpha_lane() {
        assert!(cgan_polish_active(true, true));
        assert!(!cgan_polish_active(true, false));
        assert!(!cgan_polish_active(false, true));
        assert!(!cgan_polish_active(false, false));
    }

    /// The polish must perform REAL optimizer steps: the warm-start loop
    /// breaks at `iter == iterations - 1` before the first SPSA gradient, so
    /// any iteration count <= 1 would reproduce the zero-step eager pass and
    /// make the lever vacuous (R7).
    #[test]
    fn polish_iteration_count_exceeds_the_zero_step_threshold() {
        // `const {}` rather than a plain assert: these are invariants on tuning
        // CONSTANTS, so a violation is decidable at compile time and there is no
        // reason to let a binary that breaks them link at all. The test keeps its
        // name so the R7 rationale above still has something to hang on.
        const { assert!(CGAN_POLISH_ALPHA_ITERATIONS >= 2) };
        const { assert!(CGAN_POLISH_BAND > 0.0) };
    }
}

#[cfg(test)]
mod warm_alpha_gate_tests {
    use super::*;

    #[test]
    fn grouped_disjunctive_warm_alpha_gate_is_explicitly_eager_only() {
        let enabled = BetaCrownConfig {
            reorder_bab: false,
            use_alpha_crown: true,
            input_split_alpha_iteration: 5,
            ..Default::default()
        };
        assert!(eager_warm_alpha_enabled(&enabled, true));

        let cases = [
            BetaCrownConfig {
                reorder_bab: true,
                ..enabled.clone()
            },
            BetaCrownConfig {
                use_alpha_crown: false,
                ..enabled.clone()
            },
            BetaCrownConfig {
                input_split_alpha_iteration: 0,
                ..enabled.clone()
            },
        ];
        for config in cases {
            assert!(!eager_warm_alpha_enabled(&config, true));
        }
        assert!(!eager_warm_alpha_enabled(&enabled, false));
    }

    #[test]
    fn grouped_disjunctive_root_seed_includes_reorder_but_eager_gate_does_not_f8() {
        let root_alpha = GraphAlphaState::new();
        let reordered = BetaCrownConfig {
            reorder_bab: true,
            use_alpha_crown: true,
            input_split_alpha_iteration: 5,
            ..Default::default()
        };

        assert!(deferred_root_alpha_seed(&reordered, Some(&root_alpha)).is_some());
        assert!(!eager_warm_alpha_enabled(&reordered, true));

        let eager = BetaCrownConfig {
            reorder_bab: false,
            ..reordered.clone()
        };
        assert!(deferred_root_alpha_seed(&eager, Some(&root_alpha)).is_some());
        assert!(eager_warm_alpha_enabled(&eager, true));

        for disabled in [
            BetaCrownConfig {
                input_split_alpha_iteration: 0,
                ..reordered.clone()
            },
            BetaCrownConfig {
                use_alpha_crown: false,
                ..reordered.clone()
            },
        ] {
            assert!(deferred_root_alpha_seed(&disabled, Some(&root_alpha)).is_none());
        }
        assert!(deferred_root_alpha_seed(&reordered, None).is_none());
    }
}

#[cfg(test)]
mod relaxed_clip_safety_tests {
    use super::{
        config_with_disjunctive_clip_routes_disabled, should_disable_disjunctive_relaxed_clip,
        should_enable_fresh_domain_clip,
    };
    use crate::beta_crown::branching::BranchingHeuristic;
    use crate::beta_crown::config::{BetaCrownConfig, InputClipType};
    use crate::beta_crown::engine::graph::input_split::fresh_domain_clip::FreshDomainClipTelemetry;
    use crate::layers::{GatherLayer, Layer, ReLULayer};
    use crate::{GraphNetwork, GraphNode};

    #[test]
    fn batch_unsafe_graph_disables_relaxed_clip_regardless_of_clause_count() {
        for num_clauses in [1, 2, 8] {
            assert!(should_disable_disjunctive_relaxed_clip(
                num_clauses,
                false,
                true
            ));
        }
    }

    #[test]
    fn safe_or_already_disabled_relaxed_clip_keeps_existing_verifier() {
        for num_clauses in [1, 2, 8] {
            assert!(!should_disable_disjunctive_relaxed_clip(
                num_clauses,
                true,
                true
            ));
            assert!(!should_disable_disjunctive_relaxed_clip(
                num_clauses,
                false,
                false
            ));
            assert!(!should_disable_disjunctive_relaxed_clip(
                num_clauses,
                true,
                false
            ));
        }
    }

    #[test]
    fn fresh_domain_clip_gate_is_typed_default_dark_and_batch_unsafe_only() {
        let armed = BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::InputSplit,
            reorder_bab: true,
            input_split_ibp_enhancement: true,
            enable_relaxed_clip: true,
            input_clip_type: InputClipType::Relaxed,
            input_split_fresh_domain_clip: true,
            ..Default::default()
        };
        assert!(should_enable_fresh_domain_clip(&armed, false));
        assert!(!should_enable_fresh_domain_clip(&armed, true));
        assert!(!should_enable_fresh_domain_clip(
            &BetaCrownConfig::default(),
            false
        ));

        for incompatible in [
            BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::LargestBoundWidth,
                ..armed.clone()
            },
            BetaCrownConfig {
                reorder_bab: false,
                ..armed.clone()
            },
            BetaCrownConfig {
                input_split_ibp_enhancement: false,
                ..armed.clone()
            },
            BetaCrownConfig {
                enable_relaxed_clip: false,
                ..armed.clone()
            },
            BetaCrownConfig {
                input_clip_type: InputClipType::Complete,
                ..armed.clone()
            },
            BetaCrownConfig {
                relaxed_clip_iterations: 0,
                ..armed
            },
        ] {
            assert!(!should_enable_fresh_domain_clip(&incompatible, false));
        }
    }

    #[test]
    fn effective_batch_unsafe_processing_config_remains_valid() {
        let requested = BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::InputSplit,
            reorder_bab: true,
            input_split_ibp_enhancement: true,
            enable_relaxed_clip: true,
            input_clip_type: InputClipType::Relaxed,
            input_split_fresh_domain_clip: true,
            ..Default::default()
        };
        requested.validate().expect("requested route must be valid");

        let effective = config_with_disjunctive_clip_routes_disabled(&requested);
        assert!(!effective.enable_relaxed_clip);
        assert!(!effective.input_split_fresh_domain_clip);
        effective
            .validate()
            .expect("effective processing verifier must remain valid");
    }

    #[test]
    fn typed_request_on_real_gather_graph_arms_only_run_local_capability() {
        // Gather on an absolute axis is the concrete LSNC graph class whose
        // prepended domain axis makes domain stacking unsafe.
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "gather",
            Layer::Gather(GatherLayer::new(0, None, vec![2])),
            vec!["relu".to_string()],
        ));
        graph.set_output("gather");
        assert!(!graph.is_input_split_batch_stack_safe());

        let mut safe_graph = GraphNetwork::new();
        safe_graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        safe_graph.set_output("relu");
        assert!(safe_graph.is_input_split_batch_stack_safe());

        let requested = BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::InputSplit,
            reorder_bab: true,
            input_split_ibp_enhancement: true,
            enable_relaxed_clip: true,
            input_clip_type: InputClipType::Relaxed,
            input_split_fresh_domain_clip: true,
            ..Default::default()
        };
        requested.validate().expect("typed request must validate");
        let armed =
            should_enable_fresh_domain_clip(&requested, graph.is_input_split_batch_stack_safe());
        assert!(!should_enable_fresh_domain_clip(
            &requested,
            safe_graph.is_input_split_batch_stack_safe(),
        ));
        let capability = FreshDomainClipTelemetry::new(armed);
        assert!(capability.enabled());

        let effective = config_with_disjunctive_clip_routes_disabled(&requested);
        assert!(!effective.enable_relaxed_clip);
        assert!(!effective.input_split_fresh_domain_clip);
        effective
            .validate()
            .expect("capability-bearing processing config stays internally valid");
    }
}

#[cfg(test)]
mod imb_early_layout_tests {
    use super::*;
    use ndarray::array;

    fn assert_armed_preflight_rejects(
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        clause_sizes: &[usize],
    ) {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let graph = GraphNetwork::new();
        let input = BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn())
            .expect("valid test input");
        crate::imb::reset_early_attempted();

        let result = verifier.try_imb_early_disjunctive_with_gate(
            &graph,
            &input,
            objectives,
            thresholds,
            clause_sizes,
            None,
            None,
            true,
        );

        assert!(result.is_none(), "malformed layout must never verify");
        assert!(
            !crate::imb::early_attempted(),
            "malformed layout must return before marking or expensive collection"
        );
    }

    #[test]
    fn armed_early_imb_rejects_empty_clauses_and_empty_objectives() {
        assert_armed_preflight_rejects(&[], &[], &[]);
    }

    #[test]
    fn armed_early_imb_rejects_empty_clauses_with_nonempty_objectives() {
        assert_armed_preflight_rejects(&[vec![1.0]], &[0.0], &[]);
    }

    #[test]
    fn armed_early_imb_rejects_empty_objectives_with_nonempty_clause() {
        assert_armed_preflight_rejects(&[], &[], &[1]);
    }

    #[test]
    fn armed_early_imb_rejects_zero_sized_clause() {
        assert_armed_preflight_rejects(&[vec![1.0]], &[0.0], &[0, 1]);
    }

    #[test]
    fn armed_early_imb_rejects_threshold_and_clause_total_mismatches() {
        assert_armed_preflight_rejects(&[vec![1.0]], &[], &[1]);
        assert_armed_preflight_rejects(&[vec![1.0]], &[0.0], &[2]);
    }

    #[test]
    fn armed_early_imb_rejects_clause_total_overflow() {
        assert_armed_preflight_rejects(&[vec![1.0]], &[0.0], &[usize::MAX, 1]);
    }

    #[test]
    fn armed_early_imb_rejects_quarantined_cut_config_before_attempt() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_cuts: true,
            ..BetaCrownConfig::default()
        });
        let graph = GraphNetwork::new();
        let input = BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn())
            .expect("valid test input");
        crate::imb::reset_early_attempted();

        let result = verifier.try_imb_early_disjunctive_with_gate(
            &graph,
            &input,
            &[vec![1.0]],
            &[0.0],
            &[1],
            None,
            None,
            true,
        );

        assert!(
            result.is_none(),
            "quarantined cut configuration must never publish an early verdict"
        );
        assert!(
            !crate::imb::early_attempted(),
            "invalid configuration must be refused before gate or graph work"
        );
    }
}

#[cfg(test)]
mod cap_extend_floor_tests {
    use super::{cap_extend_decision, CAP_EXTEND_MIN_VERIFIED_FRAC};

    /// The exact frontier the shipped rule denies on cgan `upsample_prop_0`.
    const EXPLORED: usize = 2500;
    const VERIFIED: usize = 91; // 3.6%
    const GAP: f32 = -0.0009; // 55x inside CAP_EXTEND_GAP_EPS

    #[test]
    fn the_measured_upsample_frontier_is_denied_by_the_shipped_floor() {
        assert!(
            !cap_extend_decision(true, false, EXPLORED, VERIFIED, GAP),
            "the shipped 25% floor must still deny 3.6%; this pins the defect"
        );
    }

    #[test]
    fn the_gap_half_passes_with_a_wide_margin_on_that_frontier() {
        // Isolate WHICH half denies: at a 0% floor only the gap test remains,
        // and it must pass — otherwise the floor is not the blocker.
        let decide = |frac: f64| {
            let productive = VERIFIED as f64 >= frac * EXPLORED as f64;
            let near_zero = GAP >= -0.05f32;
            near_zero && productive
        };
        assert!(
            decide(0.0),
            "gap half must pass; it is 55x inside the epsilon"
        );
        assert!(decide(0.01), "3.6% clears a 1% floor");
        assert!(!decide(0.25), "3.6% cannot clear the shipped 25% floor");
    }

    #[test]
    fn a_stalled_frontier_is_denied_even_at_a_one_percent_floor() {
        // The floor exists to stop a tree that grazed zero but closes nothing.
        // Lowering it must NOT admit that case.
        let productive = 0f64 >= 0.01 * EXPLORED as f64;
        assert!(!productive, "verified=0 must fail any positive floor");
    }

    #[test]
    fn a_far_frontier_is_denied_however_low_the_floor_goes() {
        // The measured hopeless control: best_gap -0.7603.
        assert!(
            !cap_extend_decision(true, false, EXPLORED, VERIFIED, -0.7603),
            "the gap half is independent of the floor and must still deny"
        );
    }

    #[test]
    fn the_disarmed_and_already_granted_paths_are_untouched() {
        assert!(
            !cap_extend_decision(false, false, EXPLORED, VERIFIED, GAP),
            "unarmed"
        );
        assert!(
            !cap_extend_decision(true, true, EXPLORED, VERIFIED, GAP),
            "one-time only"
        );
        assert!(!cap_extend_decision(true, false, 0, 0, GAP), "explored=0");
    }

    #[test]
    fn a_forty_percent_frontier_is_granted_under_the_shipped_rule() {
        // The guard's own calibration point (cgan2026 row 7, ~40% verified).
        // Proves the patch does not disturb the case the shipped floor exists
        // to admit.
        assert!(cap_extend_decision(true, false, 2500, 1000, -0.0000));
    }

    #[test]
    fn cap_extend_floor_ships_off() {
        assert!(
            CAP_EXTEND_MIN_VERIFIED_FRAC.is_none(),
            "`None` keeps the shipped EXACT INTEGER comparison, so the disarmed \
path is bit-identical rather than merely equivalent"
        );
    }
}
