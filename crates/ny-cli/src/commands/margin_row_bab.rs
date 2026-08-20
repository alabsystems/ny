// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Margin-row twin-wall UNSAT lane — vnncomp glue (#twinwall).
//!
//! ONNX -> [`TwinSpec`] extractor (f64 BN fold with certified error
//! bookkeeping, Python-reference parity: `core.py::Net.__init__`), the
//! robustness vnnlib parser, and the strictly-additive vnncomp hook: after
//! the generic verifier returns unknown/timeout and the attack found no CE,
//! run the margin-row BaB with the remaining budget. The authority gate is a
//! trusted code constant after the tier sweep and enclosure obligations passed;
//! the performance recipe remains independently kill-switchable.
//!
//! Everything here fails CLOSED: any structural deviation from the twin-wall
//! family (Conv+BN trunk, ReLU/Add, head Gemm->ReLU->Gemm) returns `None` and
//! the caller keeps its existing verdict.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, Instant};

use prost::Message;

use ny_onnx::onnx_proto::{GraphProto, ModelProto, NodeProto, TensorProto};
use ny_propagate::margin_row::{
    run_margin_row_lane, BabStats, MarginRowOutcome, TwinOpSpec, TwinSpec,
};

use super::vnncomp::VnncompResult;

/// f64 unit roundoff.
const U: f64 = 1.110_223_024_625_156_5e-16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MarginRowReserveRoute {
    NotApplicable,
    Fixed,
    AdaptivePreserved,
    AdaptiveReleasedAlphaBetaTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MarginRowReserveDecision {
    /// Configured reserve before the adaptive policy.
    pub(crate) configured_secs: u64,
    /// Reserve actually subtracted from the internal verifier.
    pub(crate) reserve_secs: u64,
    pub(crate) route: MarginRowReserveRoute,
}

impl MarginRowReserveDecision {
    /// The sealed adaptive route hands the internal verifier the full window;
    /// that is the only production route authorized to arm sparse root CROWN.
    pub(crate) fn enables_scored_sparse_crown(self) -> bool {
        matches!(
            self.route,
            MarginRowReserveRoute::AdaptiveReleasedAlphaBetaTier
        )
    }

    /// Apply the dark budget-proportional ceiling to an already-resolved
    /// decision. `route` is deliberately UNCHANGED: the sealed adaptive route
    /// is the only thing authorized to arm scored sparse root CROWN
    /// ([`Self::enables_scored_sparse_crown`]), and a scheduling ceiling must
    /// not smuggle that in.
    pub(crate) fn capped_to_internal_budget(
        mut self,
        internal_budget_secs: u64,
        preset: Option<&Path>,
    ) -> Self {
        // Nothing to cap: skip the preset load entirely. `min` against any
        // ceiling would return 0 anyway, so this is byte-identical and keeps
        // the non-reserving categories off the filesystem.
        if self.reserve_secs == 0 {
            return self;
        }
        self.reserve_secs = capped_reserve_secs(
            self.reserve_secs,
            internal_budget_secs,
            margin_row_reserve_max_frac(preset),
        );
        self
    }
}

/// Dark opt-in: cap the margin-row reserve at this fraction of the INTERNAL
/// budget. Absent or invalid ⇒ no ceiling ⇒ byte-identical to the shipped
/// fixed-seconds policy.
pub(crate) const RESERVE_MAX_FRAC_ENV: &str = "NY_MARGIN_ROW_RESERVE_MAX_FRAC";

/// Accept a ceiling fraction only if it is finite and strictly inside
/// `(0, 1)`. `1.0` is rejected on purpose — it is a no-op ceiling and
/// accepting it would invite "0 means release" confusion with
/// `NY_MARGIN_ROW_RESERVE_SECS`, which is the existing full-release knob.
fn valid_reserve_fraction(fraction: f64) -> Option<f64> {
    (fraction.is_finite() && fraction > 0.0 && fraction < 1.0).then_some(fraction)
}

/// Parse the ceiling fraction from its environment form. Everything the
/// parser or [`valid_reserve_fraction`] rejects (absent, malformed, `0`,
/// `>= 1`, non-finite) declines the ceiling and keeps the shipped policy.
fn reserve_max_fraction(raw: Option<&str>) -> Option<f64> {
    raw.and_then(|value| value.parse::<f64>().ok())
        .and_then(valid_reserve_fraction)
}

/// Resolve the reserve ceiling: `NY_MARGIN_ROW_RESERVE_MAX_FRAC` > per-category
/// `margin_row.reserve_max_frac` in the preset > no ceiling.
///
/// The environment wins wherever it is PRESENT, and a present-but-declined
/// value resolves to "no ceiling" WITHOUT consulting the preset. That is
/// deliberate: it keeps `NY_MARGIN_ROW_RESERVE_MAX_FRAC=0` usable as an exact
/// kill switch for a ceiling a shipped preset asked for, which is what the
/// dark-gate discipline requires of a scheduling knob. (It differs from
/// `margin_row_reserve_secs`, where an unparseable env value falls through —
/// there the fallback is a nonzero default, so falling through is the
/// conservative direction; here the fallback IS the shipped policy.)
///
/// Sound by construction: same argument as [`capped_reserve_secs`] — this only
/// schedules wall time between two independently sound lanes, so it is
/// verdict-neutral and can at worst make a lane fail to prove.
pub(crate) fn margin_row_reserve_max_frac(preset: Option<&Path>) -> Option<f64> {
    resolve_reserve_max_frac(
        std::env::var(RESERVE_MAX_FRAC_ENV).ok().as_deref(),
        preset
            .and_then(|p| crate::preset::load_preset(p).ok())
            .and_then(|c| c.margin_row.reserve_max_frac),
    )
}

/// The pure half of [`margin_row_reserve_max_frac`], kept separate so the
/// precedence rules are testable without mutating process-global environment.
fn resolve_reserve_max_frac(env_raw: Option<&str>, typed: Option<f32>) -> Option<f64> {
    match env_raw {
        // PRESENT (even if declined) ⇒ the environment decides, full stop.
        Some(raw) => reserve_max_fraction(Some(raw)),
        None => typed.and_then(|fraction| valid_reserve_fraction(f64::from(fraction))),
    }
}

/// Clamp a fixed-seconds reserve to a fraction of the internal budget.
///
/// WHY (measured 2026-07-26, cifar100_2024 `prop_idx_9502_sidx_7197` — a
/// winnable-60 row — via `NY_PHASE_TELEMETRY=1`): the reserve is a FIXED
/// number of seconds, so its share of the budget GROWS as the budget shrinks.
///
/// | scored budget | internal tier | ledger after reserve | effective BaB |
/// |---|---|---|---|
/// | 100 s | 95 s | 46 s | **34.2 s** |
/// | 200 s | 190 s | 141 s | **90.0 s** |
///
/// Doubling the scored budget yields 2.63x the BaB time, because the fixed
/// 45 s is 47% of the 95 s internal tier but only 24% of the 190 s one. That
/// non-proportionality — not per-domain GPU throughput — is why a scored 100 s
/// run behaves nothing like the first 100 s of a 200 s run. On that same run
/// the lane the reserve paid for reported `inline worker exceeded its 51.1s
/// hard slice cap; abandoning the detached worker`: 45 s bought nothing, while
/// the verifier that might have closed the row was left unable to finish its
/// root bootstrap.
///
/// Scope: 53 of the 60 cifar100 rows alpha-beta-CROWN proves and NY does not
/// pay the full fixed reserve.
///
/// The second half of this paragraph used to read "the shipped
/// `adaptive_reserve` release covers exactly 7 hard-coded rows (and none of
/// NY's 41 banked unsats)". That has been FALSE since #6569bfdc replaced the
/// seven-filename allowlist with pure budget arithmetic
/// ([`adaptive_release_target`]). On the scored cifar100 path every input is
/// fixed — reserve 45 s ([`margin_row_reserve_secs`], the preset sets no
/// `reserve_secs`), internal tier 95 s (`100 - max(5, 100/20)`), release_frac
/// 0.40 ([`DEFAULT_ADAPTIVE_RELEASE_FRAC`], not overridden) — so `45 >= 38`
/// holds for EVERY structurally admissible row, not seven. The banked-unsat
/// carve-out is therefore no longer true by construction; it can only be
/// established by measurement, and #6569bfdc's own message asks for exactly
/// that A/B ("must be A/B'd on the GB10 before the next bank of that
/// category") before the category is banked again.
///
/// Sound by construction: this only schedules wall time between two lanes that
/// are each independently sound. Shrinking the margin-row slice can only make
/// that lane fail to prove (fail-closed, verdict-neutral); it can never
/// produce a wrong verdict. Same argument as the surrounding reserve logic.
fn capped_reserve_secs(
    reserve_secs: u64,
    internal_budget_secs: u64,
    max_fraction: Option<f64>,
) -> u64 {
    let Some(fraction) = max_fraction else {
        return reserve_secs;
    };
    // f64 -> u64 via a saturating floor: the product is bounded by
    // `internal_budget_secs` for any accepted fraction (< 1.0), so this cannot
    // exceed u64, but keep the conversion total rather than relying on that.
    let ceiling = (internal_budget_secs as f64 * fraction).floor();
    let ceiling = if ceiling.is_finite() && ceiling >= 0.0 {
        ceiling as u64
    } else {
        return reserve_secs;
    };
    reserve_secs.min(ceiling)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdaptiveReserveGateEnv<'a> {
    Absent,
    Unicode(&'a str),
    NonUnicode,
}

fn adaptive_reserve_gate_env(raw: Option<&std::ffi::OsStr>) -> AdaptiveReserveGateEnv<'_> {
    match raw {
        None => AdaptiveReserveGateEnv::Absent,
        Some(value) => value.to_str().map_or(
            AdaptiveReserveGateEnv::NonUnicode,
            AdaptiveReserveGateEnv::Unicode,
        ),
    }
}

/// Resolve the typed production policy plus an exact sealed A/B override.
/// Only raw Unicode `"1"` enables and `"0"` disables. Any other present value
/// fails closed to the fixed-reserve policy, even if the preset is typed on.
fn adaptive_reserve_enabled(typed_enabled: bool, gate: AdaptiveReserveGateEnv<'_>) -> bool {
    match gate {
        AdaptiveReserveGateEnv::Absent => typed_enabled,
        AdaptiveReserveGateEnv::Unicode("1") => true,
        AdaptiveReserveGateEnv::Unicode(_) | AdaptiveReserveGateEnv::NonUnicode => false,
    }
}

/// Dark opt-in: the reserve share of the INTERNAL budget at or above which the
/// adaptive route releases the reserve entirely. Absent or invalid ⇒
/// [`DEFAULT_ADAPTIVE_RELEASE_FRAC`].
pub(crate) const RELEASE_FRAC_ENV: &str = "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC";

/// Release when the fixed reserve would take >= 40% of the internal budget.
///
/// Chosen to sit between the two measured regimes rather than to hit a row
/// list: the shipped 45 s reserve is 47.4% of the 95 s internal tier at a
/// scored 100 s (release) and 23.7% of the 190 s tier at a scored 200 s
/// (preserve). See [`capped_reserve_secs`] for the measurement.
const DEFAULT_ADAPTIVE_RELEASE_FRAC: f64 = 0.40;

/// Resolve the release threshold: `NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC` >
/// per-category `margin_row.release_frac` > [`DEFAULT_ADAPTIVE_RELEASE_FRAC`].
///
/// Shares [`valid_reserve_fraction`]'s admission rule, so a malformed or
/// out-of-range value in either source falls back to the shipped default
/// rather than silently disabling or universalizing the release.
pub(crate) fn margin_row_release_frac(preset: Option<&Path>) -> f64 {
    resolve_release_frac(
        std::env::var(RELEASE_FRAC_ENV).ok().as_deref(),
        preset
            .and_then(|p| crate::preset::load_preset(p).ok())
            .and_then(|c| c.margin_row.release_frac),
    )
}

/// The pure half of [`margin_row_release_frac`].
fn resolve_release_frac(env_raw: Option<&str>, typed: Option<f32>) -> f64 {
    env_raw
        .and_then(|raw| raw.parse::<f64>().ok())
        .and_then(valid_reserve_fraction)
        .or_else(|| typed.map(f64::from).and_then(valid_reserve_fraction))
        .unwrap_or(DEFAULT_ADAPTIVE_RELEASE_FRAC)
}

/// Should the adaptive route hand the internal verifier its whole window?
///
/// STRUCTURAL, not identity-based. The predicate is pure budget arithmetic:
/// release exactly when the configured fixed reserve would consume at least
/// `release_frac` of the internal budget. It therefore generalizes to every
/// category, model, and instance — including ones never measured — and depends
/// on nothing but the two numbers the scheduler already has.
///
/// This replaces a hardcoded seven-filename CIFAR100 allowlist. That list
/// encoded a real measurement (those rows paid a reserve that bought nothing
/// while the verifier missed its root bootstrap) but keyed it to the identity
/// of specific public benchmark files, so it could not fire on an unseen
/// instance with the same pathology and would silently stop firing if a file
/// were renamed. The share-of-budget condition is the actual mechanism behind
/// that measurement, expressed directly.
///
/// Sound by construction: this only schedules wall time between two
/// independently sound lanes, so it is verdict-neutral — at worst a lane fails
/// to prove. Same argument as [`capped_reserve_secs`].
fn adaptive_release_target(
    configured_secs: u64,
    internal_budget_secs: u64,
    release_frac: f64,
) -> bool {
    if internal_budget_secs == 0 {
        // No budget to take a share of; keep the established fixed lane.
        return false;
    }
    configured_secs as f64 >= internal_budget_secs as f64 * release_frac
}

fn reserve_policy(
    configured_secs: u64,
    adaptive_enabled: bool,
    release_target: bool,
) -> MarginRowReserveDecision {
    if configured_secs == 0 {
        return MarginRowReserveDecision {
            configured_secs,
            reserve_secs: 0,
            route: MarginRowReserveRoute::NotApplicable,
        };
    }
    if !adaptive_enabled {
        return MarginRowReserveDecision {
            configured_secs,
            reserve_secs: configured_secs,
            route: MarginRowReserveRoute::Fixed,
        };
    }
    if release_target {
        MarginRowReserveDecision {
            configured_secs,
            reserve_secs: 0,
            route: MarginRowReserveRoute::AdaptiveReleasedAlphaBetaTier,
        }
    } else {
        MarginRowReserveDecision {
            configured_secs,
            reserve_secs: configured_secs,
            route: MarginRowReserveRoute::AdaptivePreserved,
        }
    }
}

/// Resolve the per-instance reserve after structural admission.
///
/// A typed category preset or exact `NY_MARGIN_ROW_ADAPTIVE_RESERVE=1` enables
/// the adaptive route; within it, [`adaptive_release_target`] decides purely
/// from budget arithmetic. All parse, extraction, and environment uncertainty
/// retains the fixed policy. No verifier or margin-row bound/verdict code is
/// changed by this decision.
pub(crate) fn margin_row_reserve_decision(
    onnx: &Path,
    vnnlib: &Path,
    preset: Option<&Path>,
    internal_budget_secs: u64,
) -> MarginRowReserveDecision {
    let configured_secs = margin_row_reserve_secs(preset);
    // No reserve unless this category opted in: skip the expensive
    // structural probe entirely when there is nothing to reserve.
    if configured_secs == 0
        || !ny_propagate::margin_row::margin_row_bab_enabled()
        || parse_vnnlib_robustness(vnnlib).is_none()
        || extract_twin_spec(onnx).is_none()
    {
        return MarginRowReserveDecision {
            configured_secs,
            reserve_secs: 0,
            route: MarginRowReserveRoute::NotApplicable,
        };
    }
    let typed_adaptive = preset
        .and_then(|path| crate::preset::load_preset(path).ok())
        .and_then(|config| config.margin_row.adaptive_reserve)
        .unwrap_or(false);
    let gate = std::env::var_os("NY_MARGIN_ROW_ADAPTIVE_RESERVE");
    reserve_policy(
        configured_secs,
        adaptive_reserve_enabled(typed_adaptive, adaptive_reserve_gate_env(gate.as_deref())),
        adaptive_release_target(
            configured_secs,
            internal_budget_secs,
            margin_row_release_frac(preset),
        ),
    )
}

/// Seconds held back from the internal verifier for the margin-row lane.
///
/// **DEFAULT 45** — main's shipped value. The historical CIFAR100 +23
/// bookkeeping appears to have used that default, but the TinyImageNet +67
/// commit bodies explicitly record a non-default 82 s reserve. A later
/// blocked-backward 0/70 scratch rerun also used 82 s. Neither TinyImageNet run
/// has sealed provenance, so neither establishes a production reserve.
/// Resolution order:
/// `NY_MARGIN_ROW_RESERVE_SECS` > per-category `margin_row.reserve_secs` in the
/// preset > 45.
///
/// I briefly defaulted this to 0 on the theory that a 45 s reserve forfeits
/// solves — 28 on `sat_relu` and 10 on `cifar100_2024` by my reading of the
/// scorecards. That was WRONG on both counts and is corrected here:
///   * `sat_relu` never took the reserve at all: `extract_twin_spec` rejects
///     those nets (structural mismatch), so `margin_row_reserve_decision`
///     returns `NotApplicable` and the budget was never touched. 0 at risk,
///     not 28.
///   * Historical CIFAR100 bookkeeping records +23 certified UNSAT and appears
///     to use the 45 s default, but it is not a sealed A/B and therefore does
///     not establish the reserve's net production effect.
///
pub(crate) fn margin_row_reserve_secs(preset: Option<&Path>) -> u64 {
    if let Ok(v) = std::env::var("NY_MARGIN_ROW_RESERVE_SECS") {
        if let Ok(n) = v.parse() {
            return n; // explicit override wins
        }
    }
    // Per-category opt-in: `margin_row: { reserve_secs: N }` in the preset.
    preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.margin_row.reserve_secs)
        .unwrap_or(45)
}

fn log_classwise_stats(stats: &BabStats) {
    if stats.class_runs.is_empty() {
        return;
    }
    let order: Vec<usize> = stats.class_runs.iter().map(|run| run.class).collect();
    eprintln!(
        "margin-row classwise: order={order:?}, root_closed={}, completed={}/{}, stop={}",
        stats.root_closed_classes,
        stats.class_runs.iter().filter(|run| run.verified).count(),
        stats.tree_classes.len(),
        stats.stop,
    );
    for run in &stats.class_runs {
        eprintln!(
            "margin-row classwise class {}: verified={}, root_bound={:.6}, expansions={}, \
             domains={}, maxDepth={}, dips={}, epochs={}/{}, ledger={:?}, elapsed={:.3}s, stop={}",
            run.class,
            run.verified,
            run.root_bound,
            run.expansions,
            run.domains_created,
            run.max_depth,
            run.mono_raw_dips,
            run.epochs_closed,
            run.epochs_attempted,
            run.ledger_ok,
            run.elapsed_secs,
            run.stop,
        );
    }
}

#[derive(Debug, PartialEq, Eq)]
enum DetachedTake<T> {
    Finished(T),
    StillRunning,
    Failed,
}

/// Run `work` on a detached worker and bound how long the caller waits at
/// `cap` (#twinwall watchdog-kill class, banked 99ed4d42).
///
/// Distinguishes a cap expiry from a completed/failed worker.  The abandoned
/// worker keeps running until its own cooperative deadline fires or process
/// teardown; callers must not start another graph-heavy tail in that state.
///
/// WHY: `run_margin_row_lane`'s internal deadline checks are cooperative and
/// can be arbitrarily coarse between expensive tableau builds (measured on
/// metaroom: 113.1s against a 45s slice; one run overran until the process
/// watchdog killed it at budget+grace — the error-scoring risk class). The
/// only sound external enforcement is to abandon the work at the boundary.
fn run_capped_detached<T: Send + 'static>(
    cap: Duration,
    label: &'static str,
    work: impl FnOnce() -> T + Send + 'static,
) -> DetachedTake<T> {
    // Capacity-1 channel: the send never blocks, and a receiver gone after
    // the cap makes it fail — the expected abandoned-worker case.
    let (tx, rx) = std::sync::mpsc::sync_channel::<T>(1);
    let handle = match std::thread::Builder::new()
        .name(format!("margin-row-{label}"))
        .spawn(move || {
            // Evaluate the consumed FnOnce before publishing. Its captured
            // graph state is consequently dropped before `recv_timeout`
            // observes completion; joining below only waits for this bounded
            // send/epilogue, not for another proof phase.
            let out = work();
            let _ = tx.send(out);
        }) {
        Ok(handle) => handle,
        Err(_) => {
            eprintln!(
                "margin-row BaB: could not spawn {label} worker; skipping lane (fail-closed)"
            );
            return DetachedTake::Failed;
        }
    };
    match rx.recv_timeout(cap) {
        Ok(out) => match handle.join() {
            Ok(()) => DetachedTake::Finished(out),
            Err(_) => {
                eprintln!(
                    "margin-row BaB: {label} worker panicked after producing a result; \
                     discarding it fail-closed"
                );
                DetachedTake::Failed
            }
        },
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            eprintln!(
                "margin-row BaB: {label} worker exceeded its {:.1}s hard slice cap; abandoning \
                 the detached worker and returning in-budget (fail-closed, verdict-neutral)",
                cap.as_secs_f64()
            );
            // #margin-row-profile-on-abandon: dump the phase profile HERE.
            //
            // `prof::dump()` normally runs at the end of `lane_impl`, but an
            // abandoned worker never returns, so the profiler could not report
            // on the one run you actually need to profile -- the overrun. The
            // counters are process-global atomics, so the abandoning thread can
            // read them while the detached worker is still going; the numbers
            // are a snapshot at the cap, which is exactly the question ("where
            // did the slice go?").
            if ny_propagate::margin_row::prof::enabled() {
                eprint!(
                    "margin-row profile AT SLICE CAP ({label}, {:.1}s, worker still running):\n{}",
                    cap.as_secs_f64(),
                    ny_propagate::margin_row::prof::dump()
                );
            }
            // Dropping the JoinHandle detaches the still-running worker.
            DetachedTake::StillRunning
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            let _ = handle.join();
            eprintln!(
                "margin-row BaB: {label} worker exited without a result (panicked); fail-closed"
            );
            DetachedTake::Failed
        }
    }
}

pub(crate) enum MarginRowTry {
    Verdict(VnncompResult),
    FinishedWithoutVerdict,
    StillRunning,
}

/// Strictly-additive vnncomp hook: only ever turns unknown/timeout into
/// `Unsat`. The caller can distinguish an abandoned worker so it never starts
/// another graph-heavy optional tail while that worker remains live.
pub(crate) fn try_margin_row_unsat(
    onnx: &Path,
    vnnlib: &Path,
    instance_deadline: Option<Instant>,
) -> MarginRowTry {
    if !ny_propagate::margin_row::margin_row_bab_enabled() {
        return MarginRowTry::FinishedWithoutVerdict;
    }
    // No scored deadline (interactive runs): cap the lane at 10 min.
    let instance_deadline =
        instance_deadline.unwrap_or_else(|| Instant::now() + Duration::from_mins(10));
    let remaining = instance_deadline.saturating_duration_since(Instant::now());
    if remaining < Duration::from_secs(10) {
        return MarginRowTry::FinishedWithoutVerdict; // not enough budget for root gates + a useful tree
    }
    let deadline = instance_deadline
        .checked_sub(Duration::from_secs(3))
        .unwrap_or(instance_deadline);
    let dbg = std::env::var("NY_MARGIN_ROW_DEBUG").is_ok();
    let spec = match extract_twin_spec(onnx) {
        Some(s) => s,
        None => {
            if dbg {
                eprintln!("margin-row: extract_twin_spec returned None (structural mismatch)");
            }
            return MarginRowTry::FinishedWithoutVerdict;
        }
    };
    let (lo, hi, t, adv) = match parse_vnnlib_robustness(vnnlib) {
        Some(v) => v,
        None => {
            if dbg {
                eprintln!("margin-row: parse_vnnlib_robustness returned None");
            }
            return MarginRowTry::FinishedWithoutVerdict;
        }
    };
    if lo.len() != spec.n_in {
        if dbg {
            eprintln!(
                "margin-row: dim mismatch lo.len()={} != spec.n_in={}",
                lo.len(),
                spec.n_in
            );
        }
        return MarginRowTry::FinishedWithoutVerdict;
    }
    if dbg {
        eprintln!(
            "margin-row: spec OK (n_in={}, ops={}), running lane...",
            spec.n_in,
            spec.ops.len()
        );
    }
    let t0 = Instant::now();
    // HARD SLICE ENFORCEMENT (banked 99ed4d42): the lane's cooperative
    // deadline checks are too coarse to trust with the caller's tail (they
    // overran a 45s slice to 113.1s on metaroom and once ran to the watchdog
    // kill). Detached worker + bounded join: on expiry the lane returns a
    // distinct `StillRunning` state in-budget, and the caller suppresses its
    // post-BaB graph tail. The worker still gets `Some(deadline)` so it normally
    // stops cooperatively well before the external cap fires.
    let cap = deadline.saturating_duration_since(t0);
    let out = match run_capped_detached(cap, "inline", move || {
        run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000)
    }) {
        DetachedTake::Finished(out) => out,
        DetachedTake::StillRunning => return MarginRowTry::StillRunning,
        DetachedTake::Failed => return MarginRowTry::FinishedWithoutVerdict,
    };
    match out {
        MarginRowOutcome::Unsat(stats) => {
            eprintln!(
                "margin-row BaB: UNSAT in {:.1}s (root_bound={:.4}, tree_classes={:?}, \
                 expansions={}, domains={}, maxDepth={}, dips={})",
                t0.elapsed().as_secs_f64(),
                stats.root_bound,
                stats.tree_classes,
                stats.expansions,
                stats.domains_created,
                stats.max_depth,
                stats.mono_raw_dips,
            );
            log_classwise_stats(&stats);
            MarginRowTry::Verdict(VnncompResult::Unsat)
        }
        MarginRowOutcome::Unknown { reason, stats } => {
            eprintln!(
                "margin-row BaB: no verdict in {:.1}s ({reason}{})",
                t0.elapsed().as_secs_f64(),
                stats
                    .as_ref()
                    .map(|s| format!(
                        ", root_bound={:.4}, expansions={}",
                        s.root_bound, s.expansions
                    ))
                    .unwrap_or_default(),
            );
            if let Some(stats) = &stats {
                log_classwise_stats(stats);
            }
            MarginRowTry::FinishedWithoutVerdict
        }
    }
}

/// Handle to a margin-row lane running CONCURRENTLY with the internal
/// verifier (#epoch-bab).
pub(crate) struct ConcurrentLane {
    rx: std::sync::mpsc::Receiver<Option<VnncompResult>>,
    handle: std::thread::JoinHandle<()>,
}

pub(crate) enum ConcurrentLaneTake {
    Verdict(VnncompResult),
    FinishedWithoutVerdict,
    StillRunning,
}

impl ConcurrentLane {
    /// Collect the concurrent lane's verdict, waiting up to `grace` for it to
    /// Peek for a verdict WITHOUT consuming the handle (#twinwall-join).
    /// The same lane is consulted twice per instance: once on the short grace
    /// right after the internal verifier returns, and once at the very end of
    /// the post-BaB tail. A message the first call did not wait long enough
    /// for stays queued and is still there for the second — the receiver is
    /// only drained by an actual `recv`. Moving the handle at the first
    /// consult is what silently discarded a certified proof on cifar100_2024.
    ///
    /// Because it borrows, this CANNOT join, so unlike [`Self::take`] it makes
    /// no promise that worker-owned graph memory is released. Callers needing
    /// that guarantee must still reach the consuming `take`.
    pub(crate) fn peek(&self, grace: Duration) -> ConcurrentLaneTake {
        match self.rx.recv_timeout(grace) {
            Ok(Some(verdict)) => ConcurrentLaneTake::Verdict(verdict),
            Ok(None) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                ConcurrentLaneTake::FinishedWithoutVerdict
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => ConcurrentLaneTake::StillRunning,
        }
    }

    /// land.  Keep timeout distinct from a completed no-verdict worker: only
    /// the former can still own/allocate graph memory after this handle drops.
    pub(crate) fn take(self, grace: Duration) -> ConcurrentLaneTake {
        match self.rx.recv_timeout(grace) {
            Ok(verdict) => {
                // The message is sent before the closure's captured model is
                // destroyed. Join on every completed path so `Finished*`
                // literally means all worker-owned graph memory is gone.
                if self.handle.join().is_err() {
                    return ConcurrentLaneTake::FinishedWithoutVerdict;
                }
                match verdict {
                    Some(verdict) => ConcurrentLaneTake::Verdict(verdict),
                    None => ConcurrentLaneTake::FinishedWithoutVerdict,
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                let _ = self.handle.join();
                ConcurrentLaneTake::FinishedWithoutVerdict
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Dropping the handle detaches the worker; the caller must
                // suppress every later graph-heavy tail for this instance.
                ConcurrentLaneTake::StillRunning
            }
        }
    }
}

/// Resolve the concurrent-lane gate: exact `NY_MARGIN_ROW_CONCURRENT=1|0`
/// wins wherever the variable is PRESENT (so it stays a kill switch for a
/// preset that asks for the lane); otherwise the typed
/// `margin_row.concurrent` preset key decides; absent from both ⇒ OFF, i.e.
/// byte-identical to the shipped reserve-only path.
pub(crate) fn concurrent_lane_armed(preset: Option<&Path>) -> bool {
    if let Ok(raw) = std::env::var("NY_MARGIN_ROW_CONCURRENT") {
        return raw.trim() == "1";
    }
    preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.margin_row.concurrent)
        .unwrap_or(false)
}

/// Resolve the typed f32-root-tableau gate. Returns `None` when the preset
/// stays silent, so the caller leaves the propagate-side flag untouched.
pub(crate) fn root_f32_from_preset(preset: Option<&Path>) -> Option<bool> {
    preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.margin_row.root_f32)
}

/// Resolve the typed branch-width keys (#margin-row-branch-width). `None`s
/// leave the propagate-side defaults untouched.
// STAGED AHEAD OF ITS CONSUMER. Added by 043d7ff75 but nothing calls it yet,
// so the crate fails -D warnings on dead_code. DELETE THIS ATTRIBUTE with the
// commit that wires the preset key through; it is not a permanent exemption.
#[allow(dead_code)]
pub(crate) fn branch_width_from_preset(
    preset: Option<&Path>,
) -> (Option<usize>, Option<usize>, Option<bool>) {
    let Some(c) = preset.and_then(|p| crate::preset::load_preset(p).ok()) else {
        return (None, None, None);
    };
    (
        c.margin_row.k_head,
        c.margin_row.k_trunk,
        c.margin_row.k_adaptive,
    )
}

/// Resolve the typed backward-interm key (#backward-interm).
// STAGED AHEAD OF ITS CONSUMER. Added by 043d7ff75 but nothing calls it yet,
// so the crate fails -D warnings on dead_code. DELETE THIS ATTRIBUTE with the
// commit that wires the preset key through; it is not a permanent exemption.
#[allow(dead_code)]
pub(crate) fn backward_interm_from_preset(preset: Option<&Path>) -> Option<bool> {
    preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.margin_row.backward_interm)
}

/// Start the margin-row lane on a BACKGROUND THREAD, concurrently with the
/// internal verifier (#epoch-bab).
///
/// WHY THIS IS ~FREE, and why it beats the budget reserve it replaces: the
/// wall presets (`cifar100_2024`, `tinyimagenet_2024`, `metaroom_2023`) all
/// run the internal verifier on `device: wgpu` — the GPU — while this lane is
/// pure CPU/rayon. They contend for memory bandwidth and host threads, not
/// for the resource the verifier is actually bound on. A reserve BUYS the
/// lane time by TAKING it from the verifier (measured: 45 s would forfeit 28
/// `sat_relu` and 10 `cifar100` solves); running concurrently gives the lane
/// the whole instance budget while the verifier keeps its own.
///
/// Soundness is unchanged: this is the same fail-closed lane
/// (`Unsat`/`Unknown` only, never `Sat`), its verdict is consumed ONLY when
/// the internal verifier came back undecided, and it is bounded by the same
/// instance deadline. `None` = not started (lane disabled, structural
/// mismatch, or too little budget), in which case the caller falls back to
/// the inline post-verifier attempt exactly as before.
pub(crate) fn spawn_concurrent_lane(
    onnx: &Path,
    vnnlib: &Path,
    preset: Option<&Path>,
    instance_deadline: Option<Instant>,
) -> Option<ConcurrentLane> {
    if !ny_propagate::margin_row::margin_row_bab_enabled() {
        return None;
    }
    // OPT-IN (`NY_MARGIN_ROW_CONCURRENT=1`). Default OFF so the shipped path is
    // reserve-only: the 45s default feeds a lane that runs after the verifier.
    // Enabling both would tax the verifier twice — it would lose the reserved
    // 45s AND contend with the concurrent lane for the rest — and that
    // combination has no sealed measurement. Historical bookkeeping differs:
    // CIFAR100 +23 appears to use 45s, while the TinyImageNet +67 commit bodies
    // explicitly record 82s. Flip this on only with a sealed A/B that watches
    // the retained rows.
    //
    // MEASURED on cifar100_2024, 2026-08-02 (RTX 5080 + WSL2, 10-CPU cgroup,
    // rows 1..12 at the official 100 s budget, same binary, arms interleaved):
    //
    //   stock       solved 2/12 (sat 1, unsat 1)  timeout 10
    //   concurrent  solved 2/12 (sat 1, unsat 1)  timeout 10
    //
    // Identical -- no conversion and no regression. The premise below HELD, so
    // this is a fair test of the mechanism rather than a premise failure: the
    // verifier's sound f64 A.W GEMMs run on cuBLAS while the collector/alpha
    // work is CPU-bound. It simply does not pay on this category. Default OFF.
    //
    // TYPED ROUTE (#twinwall-provenance, 2026-08-03): `margin_row.concurrent`
    // in the category preset arms the same lane. tinyimagenet_2024 needs it,
    // because its whole +67 UNSAT bank was gathered through `sweep_targets`,
    // which hands the lane the FULL per-instance budget — a budget the
    // reserve-only route cannot reproduce at ANY reserve value (every banked
    // row recorded >= 50 s of lane time against a 45 s shipped reserve, and
    // the reserve slice only starts after the internal verifier returns).
    // The env var still wins wherever it is present, in both directions.
    if !concurrent_lane_armed(preset) {
        return None;
    }
    // THE PREMISE MUST HOLD. Running concurrently is ~free only because the
    // internal verifier is on the GPU while this lane is on the CPU. On a
    // CPU-device category they contend for the SAME resource and the lane
    // would slow the verifier down for nothing — and `sat_relu` (no device
    // set, so CPU) is in the twin-wall structural class, scores 101/101 for
    // ny, and has 28 solves within 45s of its budget wall. Spawn only where
    // the premise is true; elsewhere the inline post-verifier attempt still
    // runs, exactly as before.
    let device = preset
        .and_then(|p| crate::preset::load_preset(p).ok())
        .and_then(|c| c.general.device);
    let gpu = device
        .as_deref()
        .is_some_and(|d| matches!(d, "wgpu" | "cuda" | "metal" | "gpu"));
    if !gpu {
        return None;
    }
    let deadline = instance_deadline?;
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining < Duration::from_secs(10) {
        return None;
    }
    // Structural admission happens on THIS thread (cheap, and keeps the
    // fail-closed decision on the same path the inline lane uses).
    let spec = extract_twin_spec(onnx)?;
    let (lo, hi, t, adv) = parse_vnnlib_robustness(vnnlib)?;
    if lo.len() != spec.n_in {
        return None;
    }
    // Leave the tail to the caller's own wrap-up (result write, witness
    // gating) exactly as the inline lane does.
    let lane_deadline = deadline
        .checked_sub(Duration::from_secs(3))
        .unwrap_or(deadline);
    // BOUND THE CONTENTION. The lane is rayon-parallel and would otherwise
    // take the global pool, competing with the verifier's host-side threads
    // (the GPU is what the verifier is bound on, but it still needs CPU to
    // feed it). Confining the lane to a quarter of the cores keeps it useful
    // while leaving the verifier its orchestration headroom. Measured
    // unbounded: the lane runs ~2.4x slower concurrently than standalone
    // (prop_1498, 16.4s -> 40.1s), and a 69.5s standalone closure
    // (resnet_large prop_idx_4486) misses the budget entirely.
    // Default 0 = use the global rayon pool (all cores), which is the
    // configuration the conversion evidence was gathered under. Bounding the
    // lane cuts contention with the verifier BUT also slows the lane, and the
    // lane's speed is what decides whether it closes inside the budget — so
    // the trade is not obviously a win in either direction and is left as a
    // knob until measured. NY_MARGIN_ROW_CONCURRENT_THREADS=n to bound it.
    let lane_threads = std::env::var("NY_MARGIN_ROW_CONCURRENT_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("margin-row-concurrent".into())
        .spawn(move || {
            let t0 = Instant::now();
            let run = || run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(lane_deadline), 20_000);
            let out = if lane_threads == 0 {
                run()
            } else {
                match rayon::ThreadPoolBuilder::new()
                    .num_threads(lane_threads)
                    .thread_name(|i| format!("margin-row-{i}"))
                    .build()
                {
                    Ok(pool) => pool.install(run),
                    // Pool creation failed: fall back to the global pool
                    // rather than silently dropping the lane.
                    Err(_) => run(),
                }
            };
            let verdict = match out {
                MarginRowOutcome::Unsat(stats) => {
                    eprintln!(
                        "margin-row BaB (concurrent): UNSAT in {:.1}s (root_bound={:.4}, \
                         expansions={}, maxDepth={}, ledger={:?})",
                        t0.elapsed().as_secs_f64(),
                        stats.root_bound,
                        stats.expansions,
                        stats.max_depth,
                        stats.ledger_ok
                    );
                    Some(VnncompResult::Unsat)
                }
                MarginRowOutcome::Unknown { reason, .. } => {
                    eprintln!(
                        "margin-row BaB (concurrent): no verdict in {:.1}s ({reason})",
                        t0.elapsed().as_secs_f64()
                    );
                    None
                }
            };
            let _ = tx.send(verdict);
        })
        .ok()?;
    eprintln!("margin-row BaB: started CONCURRENT lane alongside the internal verifier");
    Some(ConcurrentLane { rx, handle })
}

// --------------------------------------------------------------------------
//  ONNX -> TwinSpec (parity port of core.py::Net.__init__)
// --------------------------------------------------------------------------

fn tensor_f32(t: &TensorProto) -> Option<Vec<f64>> {
    // data_type 1 = FLOAT; reject external payloads.
    if t.data_type != 1 || t.data_location != 0 {
        return None;
    }
    if !t.raw_data.is_empty() {
        if !t.raw_data.len().is_multiple_of(4) {
            return None;
        }
        Some(
            t.raw_data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f64::from(f32::from_le_bytes(*c)))
                .collect(),
        )
    } else {
        Some(t.float_data.iter().map(|&v| f64::from(v)).collect())
    }
}

/// INT64 tensor payload (shape constants). Rejects external payloads.
fn tensor_i64(t: &TensorProto) -> Option<Vec<i64>> {
    // data_type 7 = INT64.
    if t.data_type != 7 || t.data_location != 0 {
        return None;
    }
    if !t.raw_data.is_empty() {
        if !t.raw_data.len().is_multiple_of(8) {
            return None;
        }
        Some(
            t.raw_data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect(),
        )
    } else {
        Some(t.int64_data.clone())
    }
}

fn attr_ints<'a>(n: &'a NodeProto, name: &str) -> Option<&'a [i64]> {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.as_slice())
}

fn attr_f(n: &NodeProto, name: &str, default: f64) -> f64 {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .map_or(default, |a| f64::from(a.f_value()))
}

fn attr_i(n: &NodeProto, name: &str, default: i64) -> i64 {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .map_or(default, |a| a.i_value())
}

/// Extract the twin-wall family net; `None` on ANY structural deviation.
pub(crate) fn extract_twin_spec(onnx: &Path) -> Option<TwinSpec> {
    let bytes = std::fs::read(onnx).ok()?;
    let model = ModelProto::decode(bytes.as_slice()).ok()?;
    let g: &GraphProto = model.graph.as_ref()?;
    let inits: HashMap<&str, &TensorProto> =
        g.initializer.iter().map(|t| (t.name.as_str(), t)).collect();

    // Input (C, H, W).
    let inp = g.input.first()?;
    let dims: Vec<i64> = inp
        .r#type
        .as_ref()?
        .tensor_type
        .as_ref()?
        .shape
        .as_ref()?
        .dim
        .iter()
        .map(|d| match &d.value {
            Some(ny_onnx::onnx_proto::tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
            _ => 0,
        })
        .collect();
    if dims.len() != 4 {
        return None;
    }
    let chw = (
        usize::try_from(dims[1]).ok()?,
        usize::try_from(dims[2]).ok()?,
        usize::try_from(dims[3]).ok()?,
    );
    if chw.0 == 0 || chw.1 == 0 || chw.2 == 0 {
        return None;
    }
    let n_in = chw.0 * chw.1 * chw.2;

    // Consumers of activation tensors (for the Conv+BN fold check).
    let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (ni, n) in g.node.iter().enumerate() {
        for i in &n.input {
            if !inits.contains_key(i.as_str()) {
                consumers.entry(i.as_str()).or_default().push(ni);
            }
        }
    }

    // Tensor name -> id; shapes per id.
    let mut tid: HashMap<&str, usize> = HashMap::new();
    tid.insert(inp.name.as_str(), 0);
    let mut shape_chw: Vec<Option<(usize, usize, usize)>> = vec![Some(chw)];
    let mut flat: Vec<usize> = vec![n_in];
    let mut ops: Vec<TwinOpSpec> = Vec::new();
    let mut skip_bn: HashSet<usize> = HashSet::new();
    // Constant nodes referenced as Reshape shape inputs (name -> payload).
    let mut consts: HashMap<&str, &TensorProto> = HashMap::new();

    for (ni, n) in g.node.iter().enumerate() {
        if skip_bn.contains(&ni) {
            continue;
        }
        let out_id = ops.len() + 1;
        match n.op_type.as_str() {
            "Conv" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                let ishape = shape_chw.get(in_id).copied().flatten()?;
                let w_t = inits.get(n.input.get(1)?.as_str())?;
                if w_t.dims.len() != 4 {
                    return None;
                }
                let (co, ci, kh, kw) = (
                    usize::try_from(w_t.dims[0]).ok()?,
                    usize::try_from(w_t.dims[1]).ok()?,
                    usize::try_from(w_t.dims[2]).ok()?,
                    usize::try_from(w_t.dims[3]).ok()?,
                );
                let mut weight = tensor_f32(w_t)?;
                if weight.len() != co * ci * kh * kw || ci != ishape.0 {
                    return None;
                }
                let cbias = match n.input.get(2) {
                    Some(name) => {
                        let bt = inits.get(name.as_str())?;
                        let b = tensor_f32(bt)?;
                        if b.len() != co {
                            return None;
                        }
                        Some(b)
                    }
                    None => None,
                };
                let dil = attr_ints(n, "dilations").unwrap_or(&[1, 1]);
                if dil.iter().any(|&d| d != 1) || attr_i(n, "group", 1) != 1 {
                    return None;
                }
                let strides = attr_ints(n, "strides").unwrap_or(&[1, 1]);
                let pads = attr_ints(n, "pads").unwrap_or(&[0, 0, 0, 0]);
                if strides.len() != 2 || pads.len() != 4 {
                    return None;
                }
                let stride = (
                    usize::try_from(strides[0]).ok()?,
                    usize::try_from(strides[1]).ok()?,
                );
                let pads = (
                    usize::try_from(pads[0]).ok()?,
                    usize::try_from(pads[1]).ok()?,
                    usize::try_from(pads[2]).ok()?,
                    usize::try_from(pads[3]).ok()?,
                );
                // Fold the (unique-consumer) BatchNormalization, if present.
                let conv_out = n.output.first()?;
                let mut out_name = conv_out.as_str();
                let (mut bias, mut bias_err, mut w_rel) = (
                    cbias.clone().unwrap_or_else(|| vec![0.0; co]),
                    vec![0.0; co],
                    0.0,
                );
                let cons = consumers.get(conv_out.as_str());
                let bn_idx = cons.and_then(|c| {
                    (c.len() == 1 && g.node[c[0]].op_type == "BatchNormalization").then(|| c[0])
                });
                if let Some(bi) = bn_idx {
                    let bn = &g.node[bi];
                    if bn.input.len() < 5 {
                        return None;
                    }
                    let w_bn = tensor_f32(inits.get(bn.input[1].as_str())?)?;
                    let b_bn = tensor_f32(inits.get(bn.input[2].as_str())?)?;
                    let mean = tensor_f32(inits.get(bn.input[3].as_str())?)?;
                    let var = tensor_f32(inits.get(bn.input[4].as_str())?)?;
                    if [&w_bn, &b_bn, &mean, &var].iter().any(|v| v.len() != co) {
                        return None;
                    }
                    let eps = attr_f(bn, "epsilon", 1e-5);
                    let mut a_c = vec![0.0; co];
                    for ch in 0..co {
                        let t1 = var[ch] + eps;
                        // NaN-rejecting positivity check (NaN compares false).
                        if t1 <= 0.0 || t1.is_nan() {
                            return None;
                        }
                        // Parity with core.py: a_c = w / sqrt(var + eps).
                        a_c[ch] = w_bn[ch] / t1.sqrt();
                        // b_c = b - a_c * mean (+ a_c * conv bias).
                        let m1 = a_c[ch] * mean[ch];
                        let mut b_c = b_bn[ch] - m1;
                        let mut cb_term = 0.0;
                        if let Some(cb) = &cbias {
                            cb_term = a_c[ch] * cb[ch];
                            b_c += cb_term;
                        }
                        bias[ch] = b_c;
                        // Certified absolute error: each of the ~5 rounded ops
                        // contributes <= u * |operand magnitude|.
                        bias_err[ch] = 8.0 * U * (m1.abs() + cb_term.abs() + b_c.abs()) + 1e-300;
                    }
                    for (i, w) in weight.iter_mut().enumerate() {
                        *w *= a_c[i / (ci * kh * kw)];
                    }
                    // Multiplicative chain (add, sqrt, div, mul): rel < 5u.
                    w_rel = 1e-15;
                    skip_bn.insert(bi);
                    out_name = bn.output.first()?.as_str();
                }
                let oh = (ishape.1 + pads.0 + pads.2).checked_sub(kh)? / stride.0 + 1;
                let ow = (ishape.2 + pads.1 + pads.3).checked_sub(kw)? / stride.1 + 1;
                let oshape = (co, oh, ow);
                ops.push(TwinOpSpec::Conv {
                    input: in_id,
                    weight,
                    bias,
                    bias_err,
                    weight_rel_err: w_rel,
                    kernel: (co, ci, kh, kw),
                    stride,
                    pads,
                    ishape,
                    oshape,
                });
                tid.insert(out_name, out_id);
                shape_chw.push(Some(oshape));
                flat.push(co * oh * ow);
            }
            // Standalone BN (not folded into a preceding Conv) becomes a
            // certified per-channel affine (#epoch-bab Phase D). Needs the
            // CHW shape of its input (else fail-closed).
            "BatchNormalization" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                let shape = shape_chw.get(in_id).copied().flatten()?;
                let c = shape.0;
                if n.input.len() < 5 {
                    return None;
                }
                let w_bn = tensor_f32(inits.get(n.input[1].as_str())?)?;
                let b_bn = tensor_f32(inits.get(n.input[2].as_str())?)?;
                let mean = tensor_f32(inits.get(n.input[3].as_str())?)?;
                let var = tensor_f32(inits.get(n.input[4].as_str())?)?;
                if [&w_bn, &b_bn, &mean, &var].iter().any(|v| v.len() != c) {
                    return None;
                }
                let eps = attr_f(n, "epsilon", 1e-5);
                let mut scale = vec![0.0; c];
                let mut shift = vec![0.0; c];
                let mut shift_err = vec![0.0; c];
                for ch in 0..c {
                    let t1 = var[ch] + eps;
                    if t1 <= 0.0 || t1.is_nan() {
                        return None;
                    }
                    scale[ch] = w_bn[ch] / t1.sqrt();
                    let m1 = scale[ch] * mean[ch];
                    shift[ch] = b_bn[ch] - m1;
                    // Certified absolute error: ~4 rounded ops on operands of
                    // magnitude |m1| / |shift| (mirrors the Conv+BN fold).
                    shift_err[ch] = 8.0 * U * (m1.abs() + shift[ch].abs()) + 1e-300;
                    if !scale[ch].is_finite() || !shift[ch].is_finite() {
                        return None;
                    }
                }
                ops.push(TwinOpSpec::ChannelAffine {
                    input: in_id,
                    scale,
                    shift,
                    // Multiplicative chain (add, sqrt, div): rel < 5u.
                    scale_rel_err: 1e-15,
                    shift_err,
                    shape,
                });
                tid.insert(n.output.first()?.as_str(), out_id);
                shape_chw.push(Some(shape));
                flat.push(flat[in_id]);
            }
            "ConvTranspose" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                let ishape = shape_chw.get(in_id).copied().flatten()?;
                let w_t = inits.get(n.input.get(1)?.as_str())?;
                if w_t.dims.len() != 4 {
                    return None;
                }
                // ONNX ConvTranspose weight layout is [cin][cout][kh][kw];
                // TwinOpSpec wants [cout][cin][kh][kw].
                let (ci, co, kh, kw) = (
                    usize::try_from(w_t.dims[0]).ok()?,
                    usize::try_from(w_t.dims[1]).ok()?,
                    usize::try_from(w_t.dims[2]).ok()?,
                    usize::try_from(w_t.dims[3]).ok()?,
                );
                let w_raw = tensor_f32(w_t)?;
                if w_raw.len() != ci * co * kh * kw || ci != ishape.0 {
                    return None;
                }
                let mut weight = vec![0.0; co * ci * kh * kw];
                for c in 0..ci {
                    for o in 0..co {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                weight[((o * ci + c) * kh + ky) * kw + kx] =
                                    w_raw[((c * co + o) * kh + ky) * kw + kx];
                            }
                        }
                    }
                }
                let cbias = match n.input.get(2) {
                    Some(name) => {
                        let b = tensor_f32(inits.get(name.as_str())?)?;
                        if b.len() != co {
                            return None;
                        }
                        b
                    }
                    None => vec![0.0; co],
                };
                let dil = attr_ints(n, "dilations").unwrap_or(&[1, 1]);
                if dil.iter().any(|&d| d != 1) || attr_i(n, "group", 1) != 1 {
                    return None;
                }
                let strides = attr_ints(n, "strides").unwrap_or(&[1, 1]);
                let pads_a = attr_ints(n, "pads").unwrap_or(&[0, 0, 0, 0]);
                let opad = attr_ints(n, "output_padding").unwrap_or(&[0, 0]);
                if strides.len() != 2 || pads_a.len() != 4 || opad.len() != 2 {
                    return None;
                }
                // output_shape / auto_pad are not modelled: fail closed.
                if attr_ints(n, "output_shape").is_some()
                    || n.attribute.iter().any(|a| a.name == "auto_pad")
                {
                    return None;
                }
                let stride = (
                    usize::try_from(strides[0]).ok()?,
                    usize::try_from(strides[1]).ok()?,
                );
                let pads = (
                    usize::try_from(pads_a[0]).ok()?,
                    usize::try_from(pads_a[1]).ok()?,
                    usize::try_from(pads_a[2]).ok()?,
                    usize::try_from(pads_a[3]).ok()?,
                );
                let out_pad = (
                    usize::try_from(opad[0]).ok()?,
                    usize::try_from(opad[1]).ok()?,
                );
                if stride.0 == 0 || stride.1 == 0 || ishape.1 == 0 || ishape.2 == 0 {
                    return None;
                }
                let oh =
                    ((ishape.1 - 1) * stride.0 + kh + out_pad.0).checked_sub(pads.0 + pads.2)?;
                let ow =
                    ((ishape.2 - 1) * stride.1 + kw + out_pad.1).checked_sub(pads.1 + pads.3)?;
                let oshape = (co, oh, ow);
                ops.push(TwinOpSpec::ConvTranspose {
                    input: in_id,
                    weight,
                    bias: cbias,
                    bias_err: vec![0.0; co],
                    weight_rel_err: 0.0,
                    kernel: (co, ci, kh, kw),
                    stride,
                    pads,
                    ishape,
                    oshape,
                    out_pad,
                });
                tid.insert(n.output.first()?.as_str(), out_id);
                shape_chw.push(Some(oshape));
                flat.push(co * oh * ow);
            }
            "Relu" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                ops.push(TwinOpSpec::Relu { input: in_id });
                tid.insert(n.output.first()?.as_str(), out_id);
                shape_chw.push(shape_chw[in_id]);
                flat.push(flat[in_id]);
            }
            "Add" => {
                let a = *tid.get(n.input.first()?.as_str())?;
                let b = *tid.get(n.input.get(1)?.as_str())?;
                if flat[a] != flat[b] {
                    return None;
                }
                ops.push(TwinOpSpec::Add { lhs: a, rhs: b });
                tid.insert(n.output.first()?.as_str(), out_id);
                shape_chw.push(shape_chw[a]);
                flat.push(flat[a]);
            }
            "Flatten" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                ops.push(TwinOpSpec::Flatten { input: in_id });
                tid.insert(n.output.first()?.as_str(), out_id);
                shape_chw.push(None);
                flat.push(flat[in_id]);
            }
            // Constant nodes only feed Reshape shape inputs here; record the
            // payload and emit no op (fail-closed: a Constant consumed as an
            // ACTIVATION would be an unknown tensor id downstream -> None).
            "Constant" => {
                let val = n
                    .attribute
                    .iter()
                    .find(|a| a.name == "value")
                    .and_then(|a| a.t.as_ref())?;
                consts.insert(n.output.first()?.as_str(), val);
            }
            // Reshape with a STATIC element-count-preserving target is a flat
            // identity in this row-major representation (ONNX Reshape is
            // C-order). Track the CHW shape when the target is [1, C, H, W]
            // (a conv may follow, e.g. the cgan generator); else None.
            "Reshape" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                let shape_name = n.input.get(1)?.as_str();
                let shape_t = inits
                    .get(shape_name)
                    .copied()
                    .or_else(|| consts.get(shape_name).copied())?;
                let dims = tensor_i64(shape_t)?;
                // Resolve: 0 = copy input dim (only allowed at axis 0 with
                // batch 1 here), -1 = infer; all others positive.
                let mut resolved: Vec<i64> = Vec::with_capacity(dims.len());
                let mut infer = None;
                for (ax, &d) in dims.iter().enumerate() {
                    match d {
                        -1 => {
                            if infer.is_some() {
                                return None;
                            }
                            infer = Some(ax);
                            resolved.push(1);
                        }
                        0 => {
                            if ax != 0 {
                                return None;
                            }
                            resolved.push(1);
                        }
                        d if d > 0 => resolved.push(d),
                        _ => return None,
                    }
                }
                let known: i64 = resolved.iter().product();
                let total = i64::try_from(flat[in_id]).ok()?;
                if let Some(ax) = infer {
                    if known == 0 || total % known != 0 {
                        return None;
                    }
                    resolved[ax] = total / known;
                }
                if resolved.iter().product::<i64>() != total {
                    return None;
                }
                ops.push(TwinOpSpec::Flatten { input: in_id });
                tid.insert(n.output.first()?.as_str(), out_id);
                let chw_out = if resolved.len() == 4 && resolved[0] == 1 {
                    Some((
                        usize::try_from(resolved[1]).ok()?,
                        usize::try_from(resolved[2]).ok()?,
                        usize::try_from(resolved[3]).ok()?,
                    ))
                } else {
                    None
                };
                shape_chw.push(chw_out);
                flat.push(flat[in_id]);
            }
            "Gemm" => {
                let in_id = *tid.get(n.input.first()?.as_str())?;
                if (attr_f(n, "alpha", 1.0) - 1.0).abs() > 0.0
                    || (attr_f(n, "beta", 1.0) - 1.0).abs() > 0.0
                    || attr_i(n, "transA", 0) != 0
                {
                    return None;
                }
                let w_t = inits.get(n.input.get(1)?.as_str())?;
                if w_t.dims.len() != 2 {
                    return None;
                }
                let (d0, d1) = (
                    usize::try_from(w_t.dims[0]).ok()?,
                    usize::try_from(w_t.dims[1]).ok()?,
                );
                let w_raw = tensor_f32(w_t)?;
                if w_raw.len() != d0 * d1 {
                    return None;
                }
                let trans_b = attr_i(n, "transB", 0) != 0;
                let (weight, no, nin) = if trans_b {
                    (w_raw, d0, d1)
                } else {
                    // W stored (n_in, n_out): transpose to (n_out, n_in).
                    let mut w = vec![0.0; d0 * d1];
                    for i in 0..d0 {
                        for j in 0..d1 {
                            w[j * d0 + i] = w_raw[i * d1 + j];
                        }
                    }
                    (w, d1, d0)
                };
                if nin != flat[in_id] {
                    return None;
                }
                let bias = match n.input.get(2) {
                    Some(name) => {
                        let b = tensor_f32(inits.get(name.as_str())?)?;
                        if b.len() != no {
                            return None;
                        }
                        b
                    }
                    None => vec![0.0; no],
                };
                ops.push(TwinOpSpec::Gemm {
                    input: in_id,
                    weight,
                    bias,
                    shape: (no, nin),
                });
                tid.insert(n.output.first()?.as_str(), out_id);
                shape_chw.push(None);
                flat.push(no);
            }
            other => {
                if std::env::var("NY_MARGIN_ROW_DEBUG").is_ok() {
                    eprintln!(
                        "margin-row extract_twin_spec: UNSUPPORTED op '{other}' at node {ni} \
                         (in={:?} out={:?})",
                        n.input, n.output
                    );
                }
                return None;
            }
        }
    }
    Some(TwinSpec { n_in, ops })
}

// --------------------------------------------------------------------------
//  vnnlib robustness parser (parity port of core.py::parse_vnnlib)
// --------------------------------------------------------------------------

/// NCHW `(H, W)` from a `(declare-input X float32 [1, C, H, W])` header, needed to
/// flatten VNNLIB-2.0 bracket indices `X[0,c,h,w]` into the net's row-major input.
/// `None` when there is no 4-D declare-input (the flat `X_<idx>` form, where the
/// header is unused).
fn parse_declare_input_hw(text: &str) -> Option<(usize, usize)> {
    for line in text.lines() {
        let l = line.trim();
        if l.starts_with("(declare-input") {
            let lb = l.find('[')?;
            let rb = l[lb..].find(']')? + lb;
            let dims: Vec<usize> = l[lb + 1..rb]
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if dims.len() == 4 {
                return Some((dims[2], dims[3]));
            }
        }
    }
    None
}

/// Row-major flat index for a bracket tuple `"0,c,h,w"` given the NCHW `(H, W)`:
/// `c*H*W + h*W + w`. Matches the net's Flatten ordering in [`extract_twin_spec`].
fn flat_index_nchw(idx_str: &str, hw: Option<(usize, usize)>) -> Option<usize> {
    let parts: Vec<usize> = idx_str
        .split(',')
        .map(str::trim)
        .filter_map(|s| s.parse().ok())
        .collect();
    if parts.len() != 4 {
        return None;
    }
    let (h, w) = hw?;
    Some(parts[1] * h * w + parts[2] * w + parts[3])
}

/// Last comma-separated index in a bracket tuple, e.g. `"0,142"` -> `142`.
fn last_bracket_index(s: &str) -> Option<usize> {
    s.split(',').next_back()?.trim().parse().ok()
}

/// Parse the cifar100/tinyimagenet robustness form: per-pixel X bounds and a
/// disjunction of `(and (>= Y_j Y_t))` clauses. Returns `(lo, hi, t, adv)`.
///
/// Accepts BOTH VNNLIB encodings of the identical property: the flat 2025 form
/// (`X_<flatidx>`, `Y_<j>`) and the VNNLIB-2.0 bracket form (`X[0,c,h,w]`,
/// `Y[0,j]`). Bracket input indices are flattened row-major (NCHW) using the
/// `declare-input` header, which is exactly the net's Flatten ordering, so the two
/// encodings yield identical `(lo, hi, t, adv)`.
#[allow(clippy::type_complexity)]
pub(crate) fn parse_vnnlib_robustness(
    path: &Path,
) -> Option<(Vec<f64>, Vec<f64>, usize, Vec<usize>)> {
    let text = std::fs::read_to_string(path).ok()?;
    let hw = parse_declare_input_hw(&text);
    let mut lo: HashMap<usize, f64> = HashMap::new();
    let mut hi: HashMap<usize, f64> = HashMap::new();
    let mut adv: Vec<usize> = Vec::new();
    let mut true_c: HashSet<usize> = HashSet::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line
            .strip_prefix("(assert (<= X_")
            .or_else(|| line.strip_prefix("(assert (>= X_"))
        {
            // Flat 2025 form: `X_<flatidx>`.
            let upper = line.starts_with("(assert (<=");
            let mut parts = rest.split_whitespace();
            let idx: usize = parts.next()?.parse().ok()?;
            let val_s = parts.next()?;
            let val: f64 = val_s.trim_end_matches(')').parse().ok()?;
            if upper {
                hi.insert(idx, val);
            } else {
                lo.insert(idx, val);
            }
        } else if let Some(rest) = line
            .strip_prefix("(assert (<= X[")
            .or_else(|| line.strip_prefix("(assert (>= X["))
        {
            // VNNLIB-2.0 bracket form: `X[0,c,h,w] <val>`.
            let upper = line.starts_with("(assert (<=");
            let (idx_str, after) = rest.split_once(']')?;
            let idx = flat_index_nchw(idx_str, hw)?;
            let val: f64 = after
                .split_whitespace()
                .next()?
                .trim_end_matches(')')
                .parse()
                .ok()?;
            if upper {
                hi.insert(idx, val);
            } else {
                lo.insert(idx, val);
            }
        } else if let Some(rest) = line.strip_prefix("(and (>= Y_") {
            // Flat form: `(and (>= Y_j Y_t))`.
            let mut parts = rest.split_whitespace();
            let j: usize = parts.next()?.parse().ok()?;
            let t_s = parts.next()?;
            let t: usize = t_s
                .trim_start_matches("Y_")
                .trim_end_matches(')')
                .parse()
                .ok()?;
            adv.push(j);
            true_c.insert(t);
        } else if let Some(rest) = line.strip_prefix("(and (>= Y[") {
            // Bracket form: `(and (>= Y[0,j] Y[0,t]))`.
            let (j_str, after) = rest.split_once(']')?;
            let j = last_bracket_index(j_str)?;
            let t_inner = after.trim_start().strip_prefix("Y[")?;
            let (t_str, _) = t_inner.split_once(']')?;
            let t = last_bracket_index(t_str)?;
            adv.push(j);
            true_c.insert(t);
        }
    }
    if lo.is_empty() || lo.len() != hi.len() || true_c.len() != 1 || adv.is_empty() {
        return None;
    }
    let n = lo.keys().max()? + 1;
    if lo.len() != n {
        return None;
    }
    let mut lov = vec![0.0; n];
    let mut hiv = vec![0.0; n];
    for i in 0..n {
        lov[i] = *lo.get(&i)?;
        hiv[i] = *hi.get(&i)?;
        if lov[i] > hiv[i] || lov[i].is_nan() || hiv[i].is_nan() {
            return None;
        }
    }
    adv.sort_unstable();
    adv.dedup();
    let t = true_c.into_iter().next()?;

    // SOUNDNESS GATE (#twinwall): the margin-row lane emits a CERTIFIED `unsat`, so a
    // mis-flattened or permuted input box parsed here would be a false verdict. Cross-
    // check the whole parsed box against ny's PRODUCTION VNNLIB parser (`load_vnnlib`,
    // the parser that drives the real verdicts and is fuzzed/tested for both the flat
    // 2025 and bracket 2.0 encodings) and FAIL-CLOSED on any divergence: the lane then
    // simply does not fire. This makes the fast hand-rolled parser above safe — its
    // certified output is trusted only when an independent, authoritative parser agrees
    // element-for-element. Class indices are additionally range-checked.
    match ny_onnx::vnnlib::load_vnnlib(path) {
        Ok(spec) => {
            if spec.input_bounds.len() != n {
                return None;
            }
            for (i, &(bl, bu)) in spec.input_bounds.iter().enumerate() {
                if (bl - lov[i]).abs() > 1e-6 || (bu - hiv[i]).abs() > 1e-6 {
                    return None; // input-box disagreement — never emit a certified unsat
                }
            }
            if t >= spec.num_outputs || adv.iter().any(|&j| j >= spec.num_outputs) {
                return None;
            }
            // OBLIGATION 2 — OUTPUT-SIDE fail-closed. The input-box cross-check
            // above proves the parsed X region matches the trusted parser, but
            // says NOTHING about the output property: a hand parser that silently
            // DROPS a `(and (>= Y_j Y_t))` disjunct would emit a certified `unsat`
            // against FEWER adversarial classes than the spec actually forbids —
            // a false verdict. Reconstruct (true class, adversarial set) from the
            // trusted spec's output constraints and require it to match `(t, adv)`
            // EXACTLY. Any divergence, or an output form that is not the recognized
            // robustness OR-of-single-comparisons, fails closed (lane does not fire).
            use ny_onnx::vnnlib::OutputConstraint;
            if !spec.is_disjunction {
                return None; // robustness unsafe region is an OR of disjuncts
            }
            if spec.output_constraint_clauses.is_empty() {
                return None; // no reconstructable disjuncts — fail closed
            }
            let mut rec_true: Option<usize> = None;
            let mut rec_adv: HashSet<usize> = HashSet::new();
            for clause in &spec.output_constraint_clauses {
                // A robustness disjunct is a SINGLE comparison Y_j vs Y_t.
                if clause.len() != 1 {
                    return None;
                }
                let (adv_class, true_class) = match &clause[0] {
                    // Y_a >= Y_t / Y_a > Y_t: a is adversarial, t is true.
                    OutputConstraint::GreaterEq(a, b) | OutputConstraint::GreaterThan(a, b) => {
                        (*a, *b)
                    }
                    // Y_t <= Y_a / Y_t < Y_a: a is adversarial, t is true.
                    OutputConstraint::LessEq(a, b) | OutputConstraint::LessThan(a, b) => (*b, *a),
                    // Constant/threshold forms are NOT robustness disjuncts.
                    _ => return None,
                };
                if adv_class == true_class {
                    return None; // degenerate Y_t vs Y_t
                }
                match rec_true {
                    None => rec_true = Some(true_class),
                    Some(tc) if tc != true_class => return None, // inconsistent true class
                    _ => {}
                }
                if !rec_adv.insert(adv_class) {
                    return None; // duplicate adversarial disjunct — not the clean OR form
                }
            }
            let rec_true = rec_true?;
            if rec_true != t {
                return None; // reconstructed true class disagrees with hand parse
            }
            // Set-equality of the adversarial set: a hand parse capturing a strict
            // SUBSET (or superset) of the trusted disjuncts diverges here → None.
            let adv_set: HashSet<usize> = adv.iter().copied().collect();
            if rec_adv != adv_set {
                return None;
            }
        }
        Err(_) => return None, // trusted parser could not read it — fail closed
    }

    Some((lov, hiv, t, adv))
}

// --------------------------------------------------------------------------
//  Differential gates vs the verified Python reference (#twinwall INC2a/b/c).
//  Real-corpus measurements are exposed through the explicit
//  `ny vnncomp-research margin-row` lane and must run serially.
// --------------------------------------------------------------------------

pub(crate) mod research {
    use super::*;
    use ny_propagate::margin_row::bab::{BabConfig, MarginRowBab};
    use ny_propagate::margin_row::bounds::{
        compose_viay, head_gates, margin_seed, per_class_direct, row_dots, MarginBatch, YBox,
    };
    use ny_propagate::margin_row::engine::{domain_gates, BackwardEngine, LaneDir, Seed};
    use ny_propagate::margin_row::{RootGates, RoundMode, TwinNet};

    // --- Hard slice cap (banked 99ed4d42): the inline lane must RETURN at the
    // cap even when the underlying work ignores its cooperative deadline
    // entirely (the metaroom overrun class: 113.1s against a 45s slice, one
    // run killed by the watchdog at budget+grace). ---
    #[test]
    fn capped_detached_abandons_slow_work_at_the_cap() {
        let t0 = Instant::now();
        // Worker sleeps 30s — a stand-in for a lane whose internal deadline
        // checks are too coarse to fire. The cap is 200ms.
        let out = run_capped_detached(Duration::from_millis(200), "test-slow", || {
            std::thread::sleep(Duration::from_secs(30));
            42_u32
        });
        let elapsed = t0.elapsed();
        // Fail-closed abandon: no value, and the caller got control back at
        // the cap (generous 5s bound for CI schedulers), NOT after 30s.
        assert_eq!(out, DetachedTake::StillRunning);
        assert!(
            elapsed < Duration::from_secs(5),
            "cap not enforced: returned after {elapsed:?}"
        );
    }

    #[test]
    fn capped_detached_passes_through_fast_work() {
        let out = run_capped_detached(Duration::from_secs(30), "test-fast", || 7_u32);
        assert_eq!(out, DetachedTake::Finished(7));
    }

    #[test]
    fn capped_detached_panicking_worker_is_fail_closed() {
        let out = run_capped_detached(Duration::from_secs(30), "test-panic", || -> u32 {
            panic!("worker panic must map to None, never a verdict")
        });
        assert_eq!(out, DetachedTake::Failed);
    }

    #[test]
    fn concurrent_lane_joins_completed_worker_before_reporting_finished() {
        let (tx, rx) = std::sync::mpsc::channel();
        let epilogue_finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_flag = std::sync::Arc::clone(&epilogue_finished);
        let handle = std::thread::spawn(move || {
            tx.send(Some(VnncompResult::Unsat)).expect("receiver alive");
            std::thread::sleep(Duration::from_millis(20));
            worker_flag.store(true, std::sync::atomic::Ordering::Release);
        });
        let lane = ConcurrentLane { rx, handle };
        assert!(matches!(
            lane.take(Duration::from_secs(1)),
            ConcurrentLaneTake::Verdict(VnncompResult::Unsat)
        ));
        assert!(
            epilogue_finished.load(std::sync::atomic::Ordering::Acquire),
            "a finished result must mean the worker and its captured graph are gone"
        );
    }

    #[test]
    fn concurrent_lane_joins_disconnected_worker_fail_closed() {
        let (tx, rx) = std::sync::mpsc::channel::<Option<VnncompResult>>();
        let handle = std::thread::spawn(move || drop(tx));
        let lane = ConcurrentLane { rx, handle };
        assert!(matches!(
            lane.take(Duration::from_secs(1)),
            ConcurrentLaneTake::FinishedWithoutVerdict
        ));
    }

    #[test]
    fn concurrent_lane_timeout_remains_distinct_from_completion() {
        let (tx, rx) = std::sync::mpsc::channel::<Option<VnncompResult>>();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            let _ = tx.send(None);
        });
        let lane = ConcurrentLane { rx, handle };
        assert!(matches!(
            lane.take(Duration::from_millis(5)),
            ConcurrentLaneTake::StillRunning
        ));
    }

    // --- Bracket/flat VNNLIB index mapping (self-contained, no external files) ---
    // Guards the soundness-critical flatten used to parse VNNLIB-2.0 bracket specs
    // into the net's row-major NCHW input order (a mis-flatten would be a false unsat).
    #[test]
    fn bracket_flatten_is_row_major_nchw() {
        // tinyimagenet: C=3, H=W=56 -> n_in = 9408.
        assert_eq!(flat_index_nchw("0,0,0,0", Some((56, 56))), Some(0));
        assert_eq!(flat_index_nchw("0,1,0,0", Some((56, 56))), Some(56 * 56));
        assert_eq!(flat_index_nchw("0,0,1,0", Some((56, 56))), Some(56));
        assert_eq!(flat_index_nchw("0,0,0,1", Some((56, 56))), Some(1));
        assert_eq!(flat_index_nchw("0,2,55,55", Some((56, 56))), Some(9407));
        // cifar100: C=3, H=W=32 -> n_in = 3072; last pixel = 3071.
        assert_eq!(flat_index_nchw("0,2,31,31", Some((32, 32))), Some(3071));
        // Missing shape / wrong arity -> None (fail-closed, no unsat).
        assert_eq!(flat_index_nchw("0,2,31,31", None), None);
        assert_eq!(flat_index_nchw("0,2,31", Some((32, 32))), None);
    }

    #[test]
    fn declare_input_hw_and_class_index() {
        let hdr = "(declare-input  X float32 [1, 3, 56, 56])";
        assert_eq!(parse_declare_input_hw(hdr), Some((56, 56)));
        assert_eq!(
            parse_declare_input_hw("(declare-input X float32 [1, 3, 32, 32])"),
            Some((32, 32))
        );
        assert_eq!(parse_declare_input_hw("no header here"), None);
        assert_eq!(last_bracket_index("0,142"), Some(142));
        assert_eq!(last_bracket_index("0,0"), Some(0));
    }

    /// OBLIGATION 2 (output-side fail-closed): a hand parse that captures a
    /// STRICT SUBSET of the trusted spec's adversarial disjuncts must return
    /// `None` (never a spec), because a certified `unsat` against fewer
    /// adversarial classes than the property forbids is a FALSE verdict.
    ///
    /// The `(and (<= Y_5 Y_2))` disjunct below is semantically a real robustness
    /// disjunct (Y_5 <= Y_2, i.e. class 2 beats true class 5), and the trusted
    /// `load_vnnlib` parser records it — but the fast hand parser only matches
    /// lines beginning `(and (>= Y_...`, so it silently DROPS it. The output-side
    /// set-equality cross-check must catch the divergence and fail closed.
    #[test]
    fn output_side_subset_disjunct_fails_closed() {
        let dir = std::env::temp_dir();
        let stamp = format!(
            "{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let header = "\
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(declare-const Y_3 Real)
(declare-const Y_4 Real)
(declare-const Y_5 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
";

        // POSITIVE CONTROL: all three disjuncts in the `>=` form the hand parser
        // recognizes, so hand parse == trusted spec == {adv 0,1,2 vs true 5}.
        let full = format!(
            "{header}(assert (or\n\
             (and (>= Y_0 Y_5))\n\
             (and (>= Y_1 Y_5))\n\
             (and (>= Y_2 Y_5))\n\
             ))\n"
        );
        let full_path = dir.join(format!("mr_full_{stamp}.vnnlib"));
        std::fs::write(&full_path, full).unwrap();
        let full_parsed = parse_vnnlib_robustness(&full_path);
        std::fs::remove_file(&full_path).ok();
        let (_, _, t, adv) =
            full_parsed.expect("full (matching) disjunction should parse to a spec");
        assert_eq!(t, 5, "true class");
        assert_eq!(adv, vec![0, 1, 2], "adversarial set");

        // SUBSET CASE: the third disjunct is written `(<= Y_5 Y_2)` — a real
        // disjunct load_vnnlib captures, but the hand parser (matching only
        // `(and (>= Y_...`) drops, so its adv set is the strict subset {0,1}.
        let subset = format!(
            "{header}(assert (or\n\
             (and (>= Y_0 Y_5))\n\
             (and (>= Y_1 Y_5))\n\
             (and (<= Y_5 Y_2))\n\
             ))\n"
        );
        let subset_path = dir.join(format!("mr_subset_{stamp}.vnnlib"));
        std::fs::write(&subset_path, subset).unwrap();
        let subset_parsed = parse_vnnlib_robustness(&subset_path);
        std::fs::remove_file(&subset_path).ok();
        assert!(
            subset_parsed.is_none(),
            "hand parse dropped a disjunct (captured strict subset {{0,1}} of the trusted \
             {{0,1,2}}); the output-side cross-check must fail closed, got {subset_parsed:?}"
        );
    }

    const BENCH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benchmarks/vnncomp2025/benchmarks/cifar100_2024"
    );
    const BENCH_ROOT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benchmarks/vnncomp2025/benchmarks"
    );

    /// Build an instance from an explicit onnx + vnnlib path pair.
    fn instance_at(onnx: &str, vnnlib: &str) -> (TwinSpec, Vec<f64>, Vec<f64>, usize, Vec<usize>) {
        let spec = extract_twin_spec(Path::new(onnx)).expect("extract twin spec");
        let (lo, hi, t, adv) = parse_vnnlib_robustness(Path::new(vnnlib)).expect("parse vnnlib");
        (spec, lo, hi, t, adv)
    }

    /// PROFILE / MEASURE harness driven by env (works for cifar100 +
    /// tinyimagenet). Set NY_MARGIN_ROW_PROFILE=1 for the phase breakdown.
    ///   NY_PROBE_CAT   = cifar100_2024 (default) | tinyimagenet_2024
    ///   NY_PROBE_ONNX  = onnx basename (default per category)
    ///   NY_PROBE_VNNLIB= vnnlib basename (required)
    ///   NY_PROBE_SECS  = wall budget (default 100)
    pub(crate) fn probe_env_instance() {
        let cat = std::env::var("NY_PROBE_CAT").unwrap_or_else(|_| "cifar100_2024".into());
        let default_onnx = if cat.starts_with("tiny") {
            "TinyImageNet_resnet_medium.onnx"
        } else {
            "CIFAR100_resnet_medium.onnx"
        };
        let onnx_base = std::env::var("NY_PROBE_ONNX").unwrap_or_else(|_| default_onnx.into());
        let vnnlib_base = std::env::var("NY_PROBE_VNNLIB").expect("set NY_PROBE_VNNLIB");
        let secs: u64 = std::env::var("NY_PROBE_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let onnx = format!("{BENCH_ROOT}/{cat}/onnx/{onnx_base}");
        let vnnlib = format!("{BENCH_ROOT}/{cat}/vnnlib/{vnnlib_base}");
        let (spec, lo, hi, t, adv) = instance_at(&onnx, &vnnlib);
        let net = TwinNet::compile(&spec).expect("compile");
        eprintln!(
            "[probe] {cat}/{vnnlib_base}: n_in={} n_y={} n_out={} trunk_relus={} adv={} budget={secs}s cores={}",
            net.n_in, net.n_y, net.n_out, net.trunk_relus.len(), adv.len(),
            rayon::current_num_threads()
        );
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_secs(secs);
        let out = run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000);
        let el = t0.elapsed().as_secs_f64();
        match out {
            MarginRowOutcome::Unsat(s) => eprintln!(
                "[probe] {vnnlib_base} -> UNSAT in {el:.1}s (root={:.4} exp={} domains={} closed={} maxD={} dips={} stop={})",
                s.root_bound, s.expansions, s.domains_created, s.closed, s.max_depth, s.mono_raw_dips, s.stop
            ),
            MarginRowOutcome::Unknown { reason, stats } => eprintln!(
                "[probe] {vnnlib_base} -> UNKNOWN({reason}) in {el:.1}s ({:?})",
                stats.map(|s| (s.root_bound, s.expansions, s.domains_created, s.max_depth, s.stop))
            ),
        }
    }

    /// ROOT-BUILD profile + ORACLE digest. Times `RootGates::build` (median of
    /// NY_ROOT_REPS, default 3), prints the per-op internal breakdown when
    /// NY_MARGIN_ROW_ROOT_TIMING=1, and emits a bit-exact FNV digest of EVERY
    /// frozen gate array (mid/rad/xabs + per-layer l/u/alpha/s/c/ms/unst) plus
    /// the verdict-feeding root_bound. Compare the digest across code versions:
    /// EQUAL => the entire downstream is bit-identical (strongest oracle). Set
    /// `NY_ROOT_DUMP=<path>` to also write every (l,u) as LE f64 for an
    /// outward-enclosure check (BLAS path). NY_ROOT_BLAS=1 selects the DGEMM
    /// forward-conv (provably-outward, not bit-identical).
    ///   NY_PROBE_CAT / NY_PROBE_ONNX / NY_PROBE_VNNLIB, NY_ROOT_REPS
    pub(crate) fn probe_root_build() {
        use ny_propagate::margin_row::bab::root_eval;
        let cat = std::env::var("NY_PROBE_CAT").unwrap_or_else(|_| "cifar100_2024".into());
        let default_onnx = if cat.starts_with("tiny") {
            "TinyImageNet_resnet_medium.onnx"
        } else {
            "CIFAR100_resnet_medium.onnx"
        };
        let onnx_base = std::env::var("NY_PROBE_ONNX").unwrap_or_else(|_| default_onnx.into());
        let vnnlib_base = std::env::var("NY_PROBE_VNNLIB").expect("set NY_PROBE_VNNLIB");
        let reps: usize = std::env::var("NY_ROOT_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let onnx = format!("{BENCH_ROOT}/{cat}/onnx/{onnx_base}");
        let vnnlib = format!("{BENCH_ROOT}/{cat}/vnnlib/{vnnlib_base}");
        let (spec, lo, hi, t, adv) = instance_at(&onnx, &vnnlib);
        let net = TwinNet::compile(&spec).expect("compile");
        eprintln!(
            "[rootbuild] {cat}/{vnnlib_base}: n_in={} n_y={} n_out={} trunk_relus={} cores={}",
            net.n_in,
            net.n_y,
            net.n_out,
            net.trunk_relus.len(),
            rayon::current_num_threads()
        );
        // ONE warm build (also emits the per-op breakdown when profiling is on).
        // The digest + root_bound below come from THIS build and are printed
        // BEFORE the timing reps, so a wall-capped run still yields the oracle.
        let t_warm = Instant::now();
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let warm_secs = t_warm.elapsed().as_secs_f64();
        eprintln!("[rootbuild] warm build wall {warm_secs:.3}s");
        // Bit-exact digest of the full frozen-gate product (FNV-1a over bits).
        fn fnv(mut h: u64, xs: impl Iterator<Item = u64>) -> u64 {
            for b in xs {
                h ^= b;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            h
        }
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        h = fnv(h, root.mid.iter().map(|v| v.to_bits()));
        h = fnv(h, root.rad.iter().map(|v| v.to_bits()));
        h = fnv(h, root.xabs.iter().map(|v| v.to_bits()));
        let mut total_neurons = 0usize;
        for lg in &root.layers {
            total_neurons += lg.n;
            h = fnv(h, lg.l.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.u.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.alpha.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.s.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.c.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.ms.iter().map(|v| v.to_bits()));
            h = fnv(h, lg.unst.iter().map(|&i| i as u64));
        }
        eprintln!(
            "[rootbuild] GATE_DIGEST = {h:#018x}  layers={} total_neurons={total_neurons}",
            root.layers.len()
        );
        for (i, lg) in root.layers.iter().enumerate() {
            let minl = lg.l.iter().copied().fold(f64::INFINITY, f64::min);
            let maxu = lg.u.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "[rootbuild]   L{i:>2} n={:>7} unst={:>7} minl={minl:>12.6} maxu={maxu:>12.6}",
                lg.n,
                lg.unst.len()
            );
        }
        if let Ok(path) = std::env::var("NY_ROOT_DUMP") {
            use std::io::Write;
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("dump create"));
            for lg in &root.layers {
                for &v in &lg.l {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
                for &v in &lg.u {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
            }
            f.flush().unwrap();
            eprintln!("[rootbuild] dumped per-layer l|u (LE f64) to {path}");
        }
        // Verdict-feeding root_bound (ties RootGates to the moat number).
        let eng = BackwardEngine::new(&net, &root);
        match root_eval(&eng, &net, t, &adv) {
            Ok(re) => {
                let rb = re
                    .dj
                    .iter()
                    .zip(&adv)
                    .filter(|(_, j)| re.tree_classes.contains(j))
                    .map(|(b, _)| *b)
                    .fold(f64::INFINITY, f64::min);
                eprintln!(
                    "[rootbuild] root_eval: adv={} tree_classes={} closed_at_root={} root_bound={rb:.17} bits={:#018x}",
                    adv.len(), re.tree_classes.len(), re.tree_classes.is_empty(), rb.to_bits()
                );
            }
            Err(e) => eprintln!("[rootbuild] root_eval err: {e}"),
        }
        // Best-effort timing (median of NY_ROOT_REPS extra builds). Comes LAST
        // so a wall-capped run still emitted the digest + root_bound above.
        if reps > 0 {
            let mut times = vec![warm_secs];
            for _ in 0..reps {
                let t0 = Instant::now();
                let r = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
                times.push(t0.elapsed().as_secs_f64());
                std::hint::black_box(&r.layers[0].l[0]);
            }
            times.sort_by(f64::total_cmp);
            eprintln!(
                "[rootbuild] build median {:.3}s over {} builds (min {:.3}s max {:.3}s)",
                times[times.len() / 2],
                times.len(),
                times[0],
                times[times.len() - 1]
            );
        }
    }

    /// Pass-cost scaling: is a backward pass cost proportional to column count
    /// R (batching only fills cores) or dominated by a fixed per-pass constant
    /// (batching amortizes real work)? Times single Lower passes at several R.
    pub(crate) fn probe_pass_scaling() {
        use ndarray::Array2;
        let vnnlib_base = std::env::var("NY_PROBE_VNNLIB").expect("set NY_PROBE_VNNLIB");
        let cat = std::env::var("NY_PROBE_CAT").unwrap_or_else(|_| "cifar100_2024".into());
        let default_onnx = if cat.starts_with("tiny") {
            "TinyImageNet_resnet_medium.onnx"
        } else {
            "CIFAR100_resnet_medium.onnx"
        };
        let onnx_base = std::env::var("NY_PROBE_ONNX").unwrap_or_else(|_| default_onnx.into());
        let onnx = format!("{BENCH_ROOT}/{cat}/onnx/{onnx_base}");
        let vnnlib = format!("{BENCH_ROOT}/{cat}/vnnlib/{vnnlib_base}");
        let (spec, lo, hi, _t, _adv) = instance_at(&onnx, &vnnlib);
        let net = TwinNet::compile(&spec).expect("compile");
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let eng = BackwardEngine::new(&net, &root);
        let n_y = net.n_y;
        eprintln!(
            "[scaling] {vnnlib_base} n_y={n_y} cores={}",
            rayon::current_num_threads()
        );
        eprintln!("[scaling]   R      ms   ms/col   (single Lower backward, outward)");
        for &r in &[1usize, 4, 16, 64, 128, 256] {
            // Seed (n_y, r): column c activates head neuron c%n_y (finite, dense-ish).
            let mut s = Array2::<f64>::zeros((n_y, r));
            for c in 0..r {
                s[[c % n_y, c]] = 1.0;
            }
            let e = Array2::<f64>::zeros((n_y, r));
            let seed = Seed { s, e: Some(e) };
            // Warm once, then median of 5.
            let _ = eng
                .run(&seed, None, LaneDir::Lower, None, false)
                .expect("run");
            let mut times = Vec::new();
            for _ in 0..5 {
                let t0 = Instant::now();
                let _ = eng
                    .run(&seed, None, LaneDir::Lower, None, false)
                    .expect("run");
                times.push(t0.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.total_cmp(b));
            let ms = times[2] * 1e3;
            eprintln!("[scaling] {r:>4} {ms:>8.2} {:>8.3}", ms / r as f64);
        }
    }

    /// ORACLE: the serial lane is the verified truth. Run serial (frontier=1)
    /// and parallel (frontier=N) from the SAME root and assert the root_bound
    /// is BIT-IDENTICAL and the verdict matches. Drive via NY_PROBE_VNNLIB /
    /// NY_ORACLE_FRONTIER / NY_PROBE_SECS.
    pub(crate) fn oracle_serial_vs_parallel() {
        let cat = std::env::var("NY_PROBE_CAT").unwrap_or_else(|_| "cifar100_2024".into());
        let default_onnx = if cat.starts_with("tiny") {
            "TinyImageNet_resnet_medium.onnx"
        } else {
            "CIFAR100_resnet_medium.onnx"
        };
        let onnx_base = std::env::var("NY_PROBE_ONNX").unwrap_or_else(|_| default_onnx.into());
        let vnnlib_base = std::env::var("NY_PROBE_VNNLIB").expect("set NY_PROBE_VNNLIB");
        let secs: u64 = std::env::var("NY_PROBE_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let frontier: usize = std::env::var("NY_ORACLE_FRONTIER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(14);
        let onnx = format!("{BENCH_ROOT}/{cat}/onnx/{onnx_base}");
        let vnnlib = format!("{BENCH_ROOT}/{cat}/vnnlib/{vnnlib_base}");
        let (spec, lo, hi, t, adv) = instance_at(&onnx, &vnnlib);
        let net = TwinNet::compile(&spec).expect("compile");
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let run = |frontier: usize| {
            let deadline = Instant::now() + Duration::from_secs(secs);
            // #margin-row-branch-width: the candidate-width knobs live in
            // `margin_row/mod.rs`, NOT here. This diagnostic entry point builds
            // its own config; the SCORED concurrent lane does not come through
            // it, so an override added here is inert — measured exactly that way
            // (expansions identical at k=8, 4 and 2).
            let cfg = BabConfig {
                max_expansions: 20_000,
                deadline: Some(deadline),
                lru_cap: 64,
                frontier,
                ..BabConfig::default()
            };
            match MarginRowBab::run(&net, &root, t, &adv, cfg) {
                MarginRowOutcome::Unsat(s) => {
                    ("UNSAT".to_string(), s.root_bound, s.expansions, s.stop)
                }
                MarginRowOutcome::Unknown { reason, stats } => {
                    let rb = stats.as_ref().map_or(f64::NAN, |s| s.root_bound);
                    let exp = stats.as_ref().map_or(0, |s| s.expansions);
                    (format!("UNKNOWN({reason})"), rb, exp, "n/a".to_string())
                }
            }
        };
        let (sv, srb, sexp, sstop) = run(1);
        let (pv, prb, pexp, pstop) = run(frontier);
        eprintln!("[oracle] {vnnlib_base} frontier={frontier}");
        eprintln!("[oracle]   serial  : {sv:>10} root_bound={srb:.17} bits={:#018x} exp={sexp} stop={sstop}", srb.to_bits());
        eprintln!("[oracle]   parallel: {pv:>10} root_bound={prb:.17} bits={:#018x} exp={pexp} stop={pstop}", prb.to_bits());
        eprintln!(
            "[oracle]   root_bound diff = {:.3e}  bit-identical={}",
            (srb - prb).abs(),
            srb.to_bits() == prb.to_bits()
        );
        assert_eq!(
            srb.to_bits(),
            prb.to_bits(),
            "root_bound must be bit-identical (serial vs parallel)"
        );
        let serial_unsat = sv == "UNSAT";
        let parallel_unsat = pv == "UNSAT";
        // MOAT-safe: parallel may only be UNSAT if serial is (both sound); it
        // must NEVER claim UNSAT where serial did not.
        assert!(
            !parallel_unsat || serial_unsat,
            "parallel UNSAT but serial not: {pv} vs {sv}"
        );
        eprintln!("[oracle]   VERDICT MATCH: serial={sv} parallel={pv}");
    }

    fn instance(name: &str) -> (TwinSpec, Vec<f64>, Vec<f64>, usize, Vec<usize>) {
        let onnx = format!("{BENCH}/onnx/CIFAR100_resnet_medium.onnx");
        let vnnlib = format!("{BENCH}/vnnlib/{name}");
        let spec = extract_twin_spec(Path::new(&onnx)).expect("extract twin spec");
        let (lo, hi, t, adv) = parse_vnnlib_robustness(Path::new(&vnnlib)).expect("parse vnnlib");
        (spec, lo, hi, t, adv)
    }

    /// Probe-semantics bound at a (splits, head_clamp) prefix:
    /// `B = max(min_j direct_j, min_j max(m1_j, m2v_j))` over the root-fail
    /// class set (mirrors probe_backward.py's per-depth bound).
    #[allow(clippy::type_complexity)]
    struct Probe<'a> {
        eng: BackwardEngine<'a>,
        mb: MarginBatch,
    }

    impl<'a> Probe<'a> {
        fn new(net: &'a TwinNet, root: &'a RootGates, t: usize, adv: &[usize]) -> Self {
            let eng = BackwardEngine::new(net, root);
            // Root-fail set from the via-y bound over ALL classes (probe line
            // 62-64: b0 = mb_all.bounds(Al0, Au0); fail = b0 < 0).
            let (al0, au0) = eng.y_rows(None).expect("y_rows");
            let ybox = YBox::from_rows(&eng, &al0, &au0);
            let mb_all = MarginBatch::new(net, t, adv).expect("mb_all");
            let gates = head_gates(&ybox, root.mode);
            let ms = margin_seed(&mb_all, &gates, &ybox, root.mode);
            let ald = row_dots(root, &al0);
            let aud = row_dots(root, &au0);
            let m2v = compose_viay(&eng, &mb_all, &gates, &al0, &au0, &ald, &aud, root.mode);
            let fail: Vec<usize> = (0..adv.len())
                .filter(|&k| ms.m1[k].max(m2v[k]) < 0.0)
                .map(|k| adv[k])
                .collect();
            let mb = MarginBatch::new(net, t, &fail).expect("mb fail");
            Self { eng, mb }
        }

        fn bound_at(
            &self,
            splits: &[(usize, usize, i8)],
            head_clamp: &[(usize, i8)],
        ) -> (f64, f64) {
            let root = self.eng.root;
            let dom = domain_gates(root, splits);
            let dom_opt = (!splits.is_empty()).then_some(&dom);
            let (al, au) = self.eng.y_rows(dom_opt).expect("y_rows");
            let mut ybox = YBox::from_rows(&self.eng, &al, &au);
            ybox.clamp(head_clamp);
            assert!(!ybox.is_empty(), "probe domain must be feasible");
            let gates = head_gates(&ybox, root.mode);
            let ms = margin_seed(&self.mb, &gates, &ybox, root.mode);
            let ald = row_dots(root, &al);
            let aud = row_dots(root, &au);
            let m2v = compose_viay(&self.eng, &self.mb, &gates, &al, &au, &ald, &aud, root.mode);
            let b_viay = (0..self.mb.nf())
                .map(|k| ms.m1[k].max(m2v[k]))
                .fold(f64::INFINITY, f64::min);
            let pass = self
                .eng
                .run(&ms.seed, dom_opt, LaneDir::Lower, None, false)
                .expect("seeded pass");
            let direct = per_class_direct(&self.eng, &pass, &ms, 0..self.mb.nf());
            let b_direct = direct.iter().copied().fold(f64::INFINITY, f64::min);
            (b_direct.max(b_viay), b_viay)
        }
    }

    const P1498: &str = "CIFAR100_resnet_medium_prop_idx_1498_sidx_792_eps_0.0039.vnnlib";
    const P4429: &str = "CIFAR100_resnet_medium_prop_idx_4429_sidx_1471_eps_0.0039.vnnlib";
    const P2551: &str = "CIFAR100_resnet_medium_prop_idx_2551_sidx_9941_eps_0.0039.vnnlib";
    const P6232_SAT: &str = "CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib";

    /// Recorded 1498 worst path (out/probe_bw_prop_idx_1498_sidx_792_*.json).
    /// Events in chosen order; expected (bound, bound_viay) per depth.
    const TRAJ_1498: &[(Option<(&str, usize, usize, i8)>, f64, f64)] = &[
        (
            Some(("head", 23, 0, -1)),
            -0.127_445_187_049_127_2,
            -1.510_236_772_280_93,
        ),
        (
            Some(("head", 10, 0, 1)),
            -0.099_167_847_802_682_06,
            -1.472_712_865_338_950_3,
        ),
        (
            Some(("trunk", 5, 9, 1)),
            -0.073_089_561_916_132_14,
            -1.457_823_829_617_386_2,
        ),
        (
            Some(("head", 53, 0, -1)),
            -0.053_729_815_336_940_27,
            -1.469_321_617_079_468_3,
        ),
        (
            Some(("trunk", 5, 10, -1)),
            -0.037_234_894_794_964_585,
            -1.445_943_919_376_340_4,
        ),
        (
            Some(("head", 32, 0, 1)),
            -0.022_401_641_733_078_015,
            -1.425_016_855_247_147_6,
        ),
        (
            Some(("trunk", 5, 52, -1)),
            -0.008_241_903_027_566_416,
            -1.415_053_943_968_675_1,
        ),
        (None, 0.002_599_361_254_453_106, -1.384_457_873_124_884_9),
    ];

    /// INC2a gate (i): parity-mode root direct bound on prop_1498 must
    /// reproduce the Python reference to <= 1e-9.
    pub(crate) fn inc2a_root_parity_1498() {
        let (spec, lo, hi, t, adv) = instance(P1498);
        let net = TwinNet::compile(&spec).expect("compile");
        let t0 = Instant::now();
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Parity, None).expect("root");
        eprintln!(
            "[inc2a] parity root gates build: {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        let probe = Probe::new(&net, &root, t, &adv);
        let (b, b_viay) = probe.bound_at(&[], &[]);
        eprintln!("[inc2a] root direct={b:.12} viay={b_viay:.12}");
        assert!(
            (b - (-0.127_445_187_049_127_2)).abs() <= 1e-9,
            "root direct {b:.12} vs reference -0.127445187049 (diff {})",
            (b - (-0.127_445_187_049_127_2)).abs()
        );
        assert!(
            (b_viay - (-1.510_236_772_280_93)).abs() <= 1e-9,
            "root viay {b_viay:.12} vs reference -1.51023677228093"
        );
    }

    /// INC2a gates (ii)+(iii): outward bound <= parity bound, within 1e-3;
    /// enclosure vs 200 sampled in-box margins (membership-filtered form is
    /// exercised in the INC2b replay; the root has no splits).
    pub(crate) fn inc2a_outward_bound_and_enclosure_1498() {
        use ndarray::Array2;
        let (spec, lo, hi, t, adv) = instance(P1498);
        let net = TwinNet::compile(&spec).expect("compile");
        let root_p = RootGates::build(&net, &lo, &hi, RoundMode::Parity, None).expect("root");
        let probe_p = Probe::new(&net, &root_p, t, &adv);
        let (b_par, _) = probe_p.bound_at(&[], &[]);
        let root_o = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let probe_o = Probe::new(&net, &root_o, t, &adv);
        let (b_out, _) = probe_o.bound_at(&[], &[]);
        eprintln!(
            "[inc2a] parity={b_par:.12} outward={b_out:.12} gap={:.3e}",
            b_par - b_out
        );
        assert!(b_out <= b_par, "outward {b_out} must be <= parity {b_par}");
        assert!(
            b_par - b_out <= 1e-3,
            "outward gap {:.3e} exceeds 1e-3",
            b_par - b_out
        );
        // Enclosure: min sampled margin over the fail set >= outward bound.
        let mut state = 271_828_182_u64;
        let mut next_unit = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
        };
        let n_in = net.n_in;
        let bsz = 200;
        let mut x = Array2::<f64>::zeros((n_in, bsz));
        for i in 0..n_in {
            for b in 0..bsz {
                let u = next_unit();
                x[[i, b]] = lo[i] + u * (hi[i] - lo[i]);
            }
        }
        let (y, _) = net
            .forward_points(&x, &std::collections::BTreeMap::new())
            .expect("forward");
        let (w2, b2, (_, n_y)) = net.gemm2();
        let fail = &probe_o.mb.adv;
        let mut min_margin = f64::INFINITY;
        for &j in fail {
            for b in 0..bsz {
                let mut m = b2[t] - b2[j];
                for k in 0..n_y {
                    m += (w2[t * n_y + k] - w2[j * n_y + k]) * y[[k, b]].max(0.0);
                }
                min_margin = min_margin.min(m);
            }
        }
        eprintln!("[inc2a] sampled min margin {min_margin:.6} vs outward bound {b_out:.6}");
        assert!(
            b_out <= min_margin,
            "outward bound {b_out} above sampled margin {min_margin}"
        );
    }

    /// INC2b gate: replay the RECORDED 1498 worst path. Parity mode must match
    /// every depth to <= 1e-9; outward mode must still cross 0 at the end.
    pub(crate) fn inc2b_trajectory_replay_1498() {
        let (spec, lo, hi, t, adv) = instance(P1498);
        let net = TwinNet::compile(&spec).expect("compile");
        for mode in [RoundMode::Parity, RoundMode::Outward] {
            let root = RootGates::build(&net, &lo, &hi, mode, None).expect("root");
            let probe = Probe::new(&net, &root, t, &adv);
            let mut splits: Vec<(usize, usize, i8)> = Vec::new();
            let mut clamp: Vec<(usize, i8)> = Vec::new();
            let mut final_bound = f64::NEG_INFINITY;
            for (depth, (event, want_b, want_viay)) in TRAJ_1498.iter().enumerate() {
                // Outward mode: recorded positions index the PARITY unstable
                // list; translate through neuron ids (outward unstable is a
                // superset).
                let (b, b_viay) = probe.bound_at(&splits, &clamp);
                eprintln!(
                    "[inc2b {mode:?}] d={depth} B={b:.12} (ref {want_b:.12}) viay={b_viay:.12}"
                );
                if mode == RoundMode::Parity {
                    assert!(
                        (b - want_b).abs() <= 1e-9,
                        "depth {depth}: bound {b:.12} vs recorded {want_b:.12} (diff {:.3e})",
                        (b - want_b).abs()
                    );
                    assert!(
                        (b_viay - want_viay).abs() <= 1e-9,
                        "depth {depth}: viay {b_viay:.12} vs recorded {want_viay:.12}"
                    );
                } else {
                    assert!(
                        b <= want_b + 1e-9,
                        "outward bound {b} above parity-recorded {want_b}"
                    );
                    assert!(
                        (b - want_b).abs() <= 1e-3,
                        "outward drifted {:.3e} from recorded at depth {depth}",
                        (b - want_b).abs()
                    );
                }
                final_bound = b;
                if let Some((kind, a, pos, dir)) = event {
                    if *kind == "trunk" {
                        splits.push((*a, *pos, *dir));
                    } else {
                        clamp.push((*a, *dir));
                    }
                }
            }
            if mode == RoundMode::Outward {
                assert!(
                    final_bound > 0.0,
                    "OUTWARD MODE LOST THE CROSSING: final bound {final_bound:.9} \
                     (parity reference +0.002599361254)"
                );
                eprintln!(
                    "[inc2b] outward final bound {final_bound:.12} still crosses 0 \
                     (margin kept: {:.3e} of +2.599e-3)",
                    final_bound
                );
            }
        }
    }

    /// Recorded positions are into the PARITY unstable list; the outward
    /// replay above relies on the unstable sets agreeing at those positions.
    /// This cross-checks that the (layer 5) ids match between modes.
    pub(crate) fn inc2b_unstable_position_parity_l5() {
        let (spec, lo, hi, _, _) = instance(P1498);
        let net = TwinNet::compile(&spec).expect("compile");
        let par = RootGates::build(&net, &lo, &hi, RoundMode::Parity, None).expect("root");
        let out = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        for li in 0..par.layers.len() {
            assert_eq!(
                par.layers[li].unst, out.layers[li].unst,
                "unstable sets diverge at layer {li}: recorded (li,pos) splits \
                 would not transfer between modes"
            );
        }
    }

    /// INC2c gate: end-to-end certified closures on the three probed
    /// pyrat-easy instances, and the MOAT gate on a known-SAT instance.
    pub(crate) fn inc2c_closes_4429() {
        assert_closes(P4429, 100);
    }

    pub(crate) fn inc2c_closes_1498() {
        assert_closes(P1498, 100);
    }

    pub(crate) fn inc2c_closes_2551() {
        assert_closes(P2551, 100);
    }

    /// MOAT: a banked-SAT instance must NEVER verify.
    pub(crate) fn inc2c_moat_sat_6232_not_verified() {
        let (spec, lo, hi, t, adv) = instance(P6232_SAT);
        let deadline = Instant::now() + Duration::from_secs(100);
        match run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000) {
            MarginRowOutcome::Unsat(stats) => panic!(
                "MOAT VIOLATION: SAT instance verified (root_bound={}, stop={})",
                stats.root_bound, stats.stop
            ),
            MarginRowOutcome::Unknown { reason, stats } => {
                eprintln!(
                    "[inc2c moat] 6232 correctly NOT verified: {reason} \
                     (root_bound={:?})",
                    stats.map(|s| s.root_bound)
                );
            }
        }
    }

    fn assert_closes(name: &str, budget_secs: u64) {
        let (spec, lo, hi, t, adv) = instance(name);
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_secs(budget_secs);
        match run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000) {
            MarginRowOutcome::Unsat(stats) => {
                eprintln!(
                    "[inc2c] {name}: UNSAT in {:.1}s (root={:.4}, exp={}, domains={}, \
                     closed={}, maxD={}, dips={}, stop={})",
                    t0.elapsed().as_secs_f64(),
                    stats.root_bound,
                    stats.expansions,
                    stats.domains_created,
                    stats.closed,
                    stats.max_depth,
                    stats.mono_raw_dips,
                    stats.stop
                );
                assert!(
                    t0.elapsed().as_secs() < budget_secs,
                    "closure exceeded the {budget_secs}s budget"
                );
            }
            MarginRowOutcome::Unknown { reason, stats } => panic!(
                "{name} did not close: {reason} (stats: {:?})",
                stats.map(|s| (s.root_bound, s.expansions, s.stop))
            ),
        }
    }

    /// Extractor + parser smoke on the real files (fast; not ignored gating
    /// logic — still needs the benchmark checkout).
    pub(crate) fn extractor_and_parser_smoke() {
        let (spec, lo, hi, t, adv) = instance(P1498);
        assert_eq!(spec.n_in, 3072);
        assert_eq!(spec.ops.len(), 19 + 9 + 8 + 1 + 2 + 1); // conv+relu+add+flat+gemms+head relu
        assert_eq!(lo.len(), 3072);
        assert!(hi.iter().zip(&lo).all(|(h, l)| h >= l));
        assert_eq!(t, 72);
        assert_eq!(adv.len(), 99);
        let net = TwinNet::compile(&spec).expect("compile");
        assert_eq!(net.n_y, 100);
        assert_eq!(net.n_out, 100);
        assert_eq!(net.trunk_relus.len(), 9);
    }

    // ===================== ADVERSARIAL MOAT + CLOSURE HARNESS ==============
    // Ground truth: official 2025 results. GT-SAT set = clean-sat (abc/pyrat/
    // neuralsat, ORT-gated) PLUS conflict-sat (those tools sat, nnv the lone
    // false-unsat). The lane must return NON-Unsat on EVERY GT-sat row. UNSAT
    // tiers from pyrat(unsat)=pyrat-easy, abc(unsat)&!pyrat=abc-only.

    /// 14 GT-SAT CIFAR100_resnet_medium instances (7 clean, 7 nnv-conflict).
    const GT_SAT: &[(&str, &str)] = &[
        (
            "CIFAR100_resnet_medium_prop_idx_2697_sidx_4836_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5001_sidx_1154_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5634_sidx_9441_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5894_sidx_5577_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_6573_sidx_5525_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_7431_sidx_7329_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_9845_sidx_200_eps_0.0039.vnnlib",
            "clean",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1592_sidx_3741_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2352_sidx_2704_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4752_sidx_1800_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5973_sidx_7841_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_6049_sidx_4719_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_7569_sidx_2906_eps_0.0039.vnnlib",
            "nnv-conflict",
        ),
    ];

    /// GT-UNSAT rows spanning both tiers (pyrat-easy + abc-only).
    const GT_UNSAT: &[(&str, &str)] = &[
        (
            "CIFAR100_resnet_medium_prop_idx_1498_sidx_792_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2551_sidx_9941_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4429_sidx_1471_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_54_sidx_9735_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1588_sidx_6654_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1642_sidx_7494_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1798_sidx_1061_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_3343_sidx_1406_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_3418_sidx_4225_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4605_sidx_6225_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5127_sidx_993_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5242_sidx_1208_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_5831_sidx_5506_eps_0.0039.vnnlib",
            "pyrat-easy",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1190_sidx_8846_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_1761_sidx_3933_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2050_sidx_8228_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2132_sidx_6868_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_230_sidx_1968_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2477_sidx_388_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2779_sidx_6416_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_2925_sidx_8815_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_3006_sidx_928_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4191_sidx_3301_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4760_sidx_4651_eps_0.0039.vnnlib",
            "abc-only",
        ),
        (
            "CIFAR100_resnet_medium_prop_idx_4921_sidx_3617_eps_0.0039.vnnlib",
            "abc-only",
        ),
    ];

    fn moat_deadline_secs() -> u64 {
        std::env::var("NY_MOAT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60)
    }

    /// Run the lane directly (the decisive path — no attack short-circuit).
    /// Returns (verdict, root_bound, stop, expansions, max_depth, dips, secs).
    fn run_named(name: &str, secs: u64) -> (String, f64, String, usize, usize, usize, f64) {
        let (spec, lo, hi, t, adv) = instance(name);
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_secs(secs);
        match run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000) {
            MarginRowOutcome::Unsat(s) => (
                "UNSAT".into(),
                s.root_bound,
                s.stop,
                s.expansions,
                s.max_depth,
                s.mono_raw_dips,
                t0.elapsed().as_secs_f64(),
            ),
            MarginRowOutcome::Unknown { reason, stats } => match stats {
                Some(s) => (
                    format!("UNKNOWN({reason})"),
                    s.root_bound,
                    s.stop,
                    s.expansions,
                    s.max_depth,
                    s.mono_raw_dips,
                    t0.elapsed().as_secs_f64(),
                ),
                None => (
                    format!("UNKNOWN({reason})"),
                    f64::NAN,
                    "no-stats".into(),
                    0,
                    0,
                    0,
                    t0.elapsed().as_secs_f64(),
                ),
            },
        }
    }

    /// THE MOAT: every GT-sat row must return NON-Unsat. A single Unsat is
    /// catastrophic (a false-unsat where a real counterexample exists).
    pub(crate) fn moat_gt_sat_never_unsat() {
        let secs = moat_deadline_secs();
        let mut violations = Vec::new();
        for (name, kind) in GT_SAT {
            let (v, rb, stop, exp, md, dips, el) = run_named(name, secs);
            let short = name.split("prop_idx_").nth(1).unwrap_or(name);
            eprintln!(
                "[moat {kind:>12}] {short:>28} -> {v:>10} root_bound={rb:+.5} \
                 stop={stop} exp={exp} maxD={md} dips={dips} ({el:.1}s)"
            );
            if v == "UNSAT" {
                violations.push((*name, *kind, rb));
            }
            assert!(
                rb <= 0.0 || v.starts_with("UNKNOWN"),
                "GT-sat {short}: positive root_bound {rb} would imply a false close"
            );
        }
        assert!(
            violations.is_empty(),
            "MOAT BROKEN: lane returned UNSAT on GT-sat instances {violations:?}"
        );
        eprintln!(
            "[moat] PASS: 0 false-unsat across {} GT-sat rows",
            GT_SAT.len()
        );
    }

    /// Closure + cross-check contingency over GT-unsat tiers. Records how many
    /// each tier closes; every UNSAT here is GT-correct (all rows are GT-unsat).
    pub(crate) fn closure_gt_unsat_contingency() {
        let secs = moat_deadline_secs();
        let mut tally: std::collections::BTreeMap<&str, (usize, usize, usize)> = Default::default(); // tier -> (unsat, unknown, total)
        for (name, tier) in GT_UNSAT {
            let (v, rb, stop, exp, md, dips, el) = run_named(name, secs);
            let short = name.split("prop_idx_").nth(1).unwrap_or(name);
            eprintln!(
                "[closure {tier:>10}] {short:>28} -> {v:>10} root_bound={rb:+.5} \
                 stop={stop} exp={exp} maxD={md} dips={dips} ({el:.1}s)"
            );
            let e = tally.entry(tier).or_default();
            e.2 += 1;
            if v == "UNSAT" {
                e.0 += 1;
            } else {
                e.1 += 1;
            }
        }
        eprintln!("[closure] contingency (tier -> unsat/unknown/total):");
        for (tier, (u, k, tot)) in &tally {
            eprintln!("   {tier:>10}: unsat={u} unknown={k} total={tot}");
        }
    }

    /// Extraction safety on the NON-target onnx models: the lane must either
    /// extract them as valid twin-wall nets OR fail-closed (None). Either way,
    /// a GT-sat row on those models can never be falsely closed because the
    /// vnncomp hook only runs the lane when extraction succeeds AND the verdict
    /// is unknown/timeout. Reports which models extract.
    pub(crate) fn extraction_safety_other_models() {
        let base = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benchmarks/vnncomp2025/benchmarks"
        );
        for (bench, onnx) in [
            ("cifar100_2024", "CIFAR100_resnet_large.onnx"),
            ("cifar100_2024", "test_nano.onnx"),
            ("tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx"),
            ("tinyimagenet_2024", "test_nano.onnx"),
        ] {
            let p = format!("{base}/{bench}/onnx/{onnx}");
            let got = extract_twin_spec(Path::new(&p));
            eprintln!(
                "[extract] {bench}/{onnx}: {}",
                got.as_ref()
                    .map_or("None (fail-closed)".to_string(), |s| format!(
                        "Some(n_in={}, ops={})",
                        s.n_in,
                        s.ops.len()
                    ))
            );
        }
    }
    /// EPOCH GAIN PROBE (#epoch-bab Phase C): on a real wall instance,
    /// measure per-layer epoch-rebuild gain — bake k same-direction splits
    /// of layer L's top unstable neurons (by retention score c*(u-l)) and
    /// compare the rebuilt-root domain bound against the frozen-gates
    /// domain-override bound for the SAME splits. Env:
    ///   NY_PROBE_VNNLIB, NY_GAIN_K (default 4), NY_GAIN_LAYERS (e.g. "0,2,4").
    pub(crate) fn epoch_gain_probe_real() {
        use ny_propagate::margin_row::root::RetainCfg;
        let vnnlib_base = std::env::var("NY_PROBE_VNNLIB").expect("set NY_PROBE_VNNLIB");
        let k: usize = std::env::var("NY_GAIN_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let layers_env = std::env::var("NY_GAIN_LAYERS").unwrap_or_else(|_| "0,2,4,6,8".into());
        let (spec, lo, hi, t, adv) = instance(&vnnlib_base);
        let net = TwinNet::compile(&spec).expect("compile");
        let t0 = Instant::now();
        let cfg = RetainCfg {
            per_layer: 64,
            budget_bytes: 1 << 30,
        };
        let (root, _) =
            RootGates::build_retaining(&net, &lo, &hi, RoundMode::Outward, None, Some(&cfg), &[])
                .expect("root");
        eprintln!(
            "[gain] root build {:.1}s, {} trunk layers",
            t0.elapsed().as_secs_f64(),
            root.layers.len()
        );
        // Root-fail classes via the probe helper.
        let probe = Probe::new(&net, &root, t, &adv);
        let (b_root, _) = probe.bound_at(&[], &[]);
        eprintln!("[gain] {vnnlib_base} root bound {b_root:+.5}");
        for ls in layers_env.split(',') {
            let li: usize = ls.trim().parse().expect("layer index");
            let lg = &root.layers[li];
            // Top-k unstable by c*(u-l).
            let mut scored: Vec<(f64, usize, usize)> = lg
                .unst
                .iter()
                .enumerate()
                .map(|(pos, &j)| (lg.c[j] * (lg.u[j] - lg.l[j]), j, pos))
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            scored.truncate(k);
            if scored.is_empty() {
                eprintln!("[gain] layer {li}: no unstable neurons");
                continue;
            }
            for dir in [1i8, -1] {
                let splits_pos: Vec<(usize, usize, i8)> =
                    scored.iter().map(|&(_, _, pos)| (li, pos, dir)).collect();
                let splits_abs: Vec<(usize, usize, i8)> =
                    scored.iter().map(|&(_, j, _)| (li, j, dir)).collect();
                let (bf, _) = probe.bound_at(&splits_pos, &[]);
                let tb = Instant::now();
                let (eroot, _) = RootGates::build_retaining(
                    &net,
                    &lo,
                    &hi,
                    RoundMode::Outward,
                    None,
                    None,
                    &splits_abs,
                )
                .expect("epoch root");
                let eprobe = Probe::new(&net, &eroot, t, &adv);
                let (be, _) = eprobe.bound_at(&[], &[]);
                eprintln!(
                    "[gain] layer {li} k={} dir {dir:+}: frozen {bf:+.5} epoch {be:+.5} \
                     gain {:+.4} (rebuild {:.1}s)",
                    scored.len(),
                    be - bf,
                    tb.elapsed().as_secs_f64()
                );
            }
        }
    }
    // ================= METAROOM BANKED GAINS (#epoch-bab Phase D) ==========
    // The Constant/Reshape extractor extension unlocked the metaroom_2023
    // class (Conv-Relu...-Reshape-Gemm-Relu-Gemm). Measured 2026-07-18 over
    // the full 106-row instances.csv (M5 Max, release, official per-instance
    // budgets): 85 UNSAT, 0 incorrect — every one CONFIRMED against the
    // official 2025 results (>=1 tool unsat, 0 tools sat; cross-checked
    // against all 7 metaroom-reporting tools).
    //
    // FIVE of those are instances NY's PRODUCTION verifier does not decide
    // (reports/measured/metaroom_2023.csv: timeout). They are the banked
    // competitive gain of this arc — the lane closes each in ~8-9s where
    // production burns its budget and yields nothing.
    const METAROOM_PROD_TIMEOUT_GAINS: &[(&str, &str)] = &[
        (
            "6cnn_ry_1_0_no_custom_OP.onnx",
            "spec_idx_108_eps_0.00000436.vnnlib",
        ),
        (
            "6cnn_ry_64_10_no_custom_OP.onnx",
            "spec_idx_132_eps_0.00000436.vnnlib",
        ),
        (
            "6cnn_ry_4_0_no_custom_OP.onnx",
            "spec_idx_125_eps_0.00000436.vnnlib",
        ),
        (
            "6cnn_ry_52_8_no_custom_OP.onnx",
            "spec_idx_126_eps_0.00000436.vnnlib",
        ),
        (
            "6cnn_ry_28_4_no_custom_OP.onnx",
            "spec_idx_113_eps_0.00000436.vnnlib",
        ),
    ];

    /// The 5 banked metaroom gains still close (regression gate for the
    /// extractor + lane). Each is GT-unsat by >=6 official tools.
    pub(crate) fn metaroom_banked_gains_still_close() {
        let mut failures = Vec::new();
        for (onnx_base, vnnlib_base) in METAROOM_PROD_TIMEOUT_GAINS {
            let onnx = format!("{BENCH_ROOT}/metaroom_2023/onnx/{onnx_base}");
            let vnnlib = format!("{BENCH_ROOT}/metaroom_2023/vnnlib/{vnnlib_base}");
            let (spec, lo, hi, t, adv) = instance_at(&onnx, &vnnlib);
            let t0 = Instant::now();
            let deadline = t0 + Duration::from_mins(1);
            let out = run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000);
            match out {
                MarginRowOutcome::Unsat(s) => eprintln!(
                    "[metaroom] {vnnlib_base} -> UNSAT in {:.1}s (exp={} ledger={:?})",
                    t0.elapsed().as_secs_f64(),
                    s.expansions,
                    s.ledger_ok
                ),
                MarginRowOutcome::Unknown { reason, .. } => {
                    eprintln!("[metaroom] {vnnlib_base} -> UNKNOWN({reason})");
                    failures.push(*vnnlib_base);
                }
            }
        }
        assert!(
            failures.is_empty(),
            "banked metaroom gains regressed: {failures:?}"
        );
    }

    /// MOAT on the metaroom class: every GT-SAT metaroom row must return
    /// NON-Unsat. (Official 2025: 5 sat rows on this benchmark.)
    pub(crate) fn metaroom_moat_gt_sat_never_unsat() {
        const GT_SAT: &[(&str, &str)] = &[
            (
                "4cnn_ry_99_16_no_custom_OP.onnx",
                "spec_idx_43_eps_0.00000436.vnnlib",
            ),
            (
                "6cnn_ry_117_19_no_custom_OP.onnx",
                "spec_idx_101_eps_0.00000436.vnnlib",
            ),
            (
                "6cnn_ry_57_9_no_custom_OP.onnx",
                "spec_idx_129_eps_0.00000436.vnnlib",
            ),
            (
                "6cnn_ry_81_13_no_custom_OP.onnx",
                "spec_idx_144_eps_0.00000436.vnnlib",
            ),
            (
                "6cnn_ry_93_15_no_custom_OP.onnx",
                "spec_idx_148_eps_0.00000436.vnnlib",
            ),
        ];
        let mut violations = Vec::new();
        for (onnx_base, vnnlib_base) in GT_SAT {
            let onnx = format!("{BENCH_ROOT}/metaroom_2023/onnx/{onnx_base}");
            let vnnlib = format!("{BENCH_ROOT}/metaroom_2023/vnnlib/{vnnlib_base}");
            let Some(spec) = extract_twin_spec(Path::new(&onnx)) else {
                eprintln!("[metaroom moat] {vnnlib_base}: extraction fail-closed (safe)");
                continue;
            };
            let Some((lo, hi, t, adv)) = parse_vnnlib_robustness(Path::new(&vnnlib)) else {
                eprintln!("[metaroom moat] {vnnlib_base}: parse fail-closed (safe)");
                continue;
            };
            let deadline = Instant::now() + Duration::from_mins(1);
            match run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000) {
                MarginRowOutcome::Unsat(s) => {
                    violations.push((*vnnlib_base, s.root_bound));
                }
                MarginRowOutcome::Unknown { reason, .. } => {
                    eprintln!("[metaroom moat] {vnnlib_base}: correctly NOT verified ({reason})");
                }
            }
        }
        assert!(
            violations.is_empty(),
            "MOAT BROKEN on metaroom GT-sat rows: {violations:?}"
        );
    }
    /// The ceiling declines for every input that is not a finite fraction
    /// strictly inside `(0, 1)`, so an unset/typo'd/hostile value can never
    /// silently change the shipped scheduling policy.
    #[test]
    fn reserve_max_fraction_accepts_only_proper_fractions() {
        assert_eq!(reserve_max_fraction(Some("0.25")), Some(0.25));
        assert_eq!(reserve_max_fraction(Some("0.5")), Some(0.5));
        for declined in [
            None,
            Some(""),
            Some("0"),
            Some("0.0"),
            Some("1"),
            Some("1.0"),
            Some("1.5"),
            Some("-0.25"),
            Some("nan"),
            Some("inf"),
            Some("25%"),
            Some("a quarter"),
        ] {
            assert_eq!(
                reserve_max_fraction(declined),
                None,
                "{declined:?} must decline the ceiling"
            );
        }
    }

    /// No ceiling ⇒ the reserve is returned untouched. This is the shipped
    /// default path and it must stay byte-identical.
    #[test]
    fn capped_reserve_without_a_fraction_is_the_identity() {
        for budget in [0, 10, 95, 190, 600] {
            assert_eq!(capped_reserve_secs(45, budget, None), 45);
        }
    }

    /// The ceiling binds only when it is BELOW the configured reserve — it can
    /// never hand the lane MORE time than the fixed policy asked for.
    #[test]
    fn capped_reserve_only_ever_shrinks_the_reserve() {
        // The measured cifar100 case: 45 s of a 95 s internal tier is 47%.
        assert_eq!(capped_reserve_secs(45, 95, Some(0.25)), 23);
        // Same fixed reserve, double the budget: the ceiling stops binding.
        assert_eq!(capped_reserve_secs(45, 190, Some(0.25)), 45);
        // A generous ceiling is inert at either budget.
        assert_eq!(capped_reserve_secs(45, 95, Some(0.9)), 45);
        // Already-released reserves stay released.
        assert_eq!(capped_reserve_secs(0, 95, Some(0.25)), 0);
        // Degenerate budgets floor to zero rather than panicking.
        assert_eq!(capped_reserve_secs(45, 0, Some(0.25)), 0);
    }

    /// The typed `margin_row.reserve_max_frac` key arms the ceiling only for
    /// proper fractions, exactly like the environment form. A malformed or
    /// out-of-range typed value must NOT arm the gate — it declines and the
    /// shipped fixed-seconds policy stands.
    #[test]
    fn typed_reserve_max_frac_arms_only_on_proper_fractions() {
        assert_eq!(resolve_reserve_max_frac(None, Some(0.25)), Some(0.25));
        assert_eq!(resolve_reserve_max_frac(None, Some(0.5)), Some(0.5));
        for declined in [
            None,
            Some(0.0f32),
            Some(-0.0),
            Some(-0.25),
            Some(1.0),
            Some(1.5),
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
        ] {
            assert_eq!(
                resolve_reserve_max_frac(None, declined),
                None,
                "typed {declined:?} must decline the ceiling"
            );
        }
    }

    /// Precedence: the environment wins wherever it is PRESENT, and a present
    /// value the parser declines resolves to "no ceiling" WITHOUT falling back
    /// to the preset — that is the exact kill switch for a shipped preset key.
    #[test]
    fn reserve_max_frac_env_overrides_and_kills_the_typed_key() {
        // Env absent ⇒ the typed key decides.
        assert_eq!(resolve_reserve_max_frac(None, Some(0.25)), Some(0.25));
        // Env present and valid ⇒ env wins over the typed key.
        assert_eq!(resolve_reserve_max_frac(Some("0.5"), Some(0.25)), Some(0.5));
        // Env present but declined ⇒ NO ceiling, typed key is not consulted.
        for kill in ["0", "0.0", "1", "1.0", "-0.25", "nan", "", "a quarter"] {
            assert_eq!(
                resolve_reserve_max_frac(Some(kill), Some(0.25)),
                None,
                "{kill:?} must kill the typed ceiling rather than fall through"
            );
        }
    }

    /// End-to-end through a real preset file: absent key ⇒ no ceiling and the
    /// reserve is untouched; a proper fraction ⇒ the ceiling binds. Reads the
    /// environment, which is unset in the test process (nothing here sets it).
    #[test]
    fn typed_reserve_max_frac_round_trips_through_a_preset_file() {
        let dir = std::env::temp_dir().join(format!("ny_mr_frac_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmpdir");

        // No preset at all ⇒ no ceiling.
        assert_eq!(margin_row_reserve_max_frac(None), None);

        // A preset without the key ⇒ still no ceiling (byte-identical path).
        let plain = dir.join("plain.yaml");
        std::fs::write(&plain, "margin_row:\n  reserve_secs: 45\n").expect("write");
        assert_eq!(margin_row_reserve_max_frac(Some(&plain)), None);
        let decision = MarginRowReserveDecision {
            configured_secs: 45,
            reserve_secs: 45,
            route: MarginRowReserveRoute::Fixed,
        };
        assert_eq!(
            decision
                .capped_to_internal_budget(95, Some(&plain))
                .reserve_secs,
            45,
            "an absent reserve_max_frac must leave the shipped reserve alone"
        );

        // A preset naming the key ⇒ the ceiling binds, route untouched.
        let capped = dir.join("capped.yaml");
        std::fs::write(
            &capped,
            "margin_row:\n  reserve_secs: 45\n  reserve_max_frac: 0.25\n",
        )
        .expect("write");
        assert_eq!(margin_row_reserve_max_frac(Some(&capped)), Some(0.25));
        let applied = decision.capped_to_internal_budget(95, Some(&capped));
        assert_eq!(applied.reserve_secs, 23);
        assert_eq!(applied.configured_secs, 45);
        assert_eq!(applied.route, MarginRowReserveRoute::Fixed);

        // A malformed typed value declines rather than arming anything.
        let bogus = dir.join("bogus.yaml");
        std::fs::write(&bogus, "margin_row:\n  reserve_max_frac: 1.0\n").expect("write");
        assert_eq!(margin_row_reserve_max_frac(Some(&bogus)), None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Landing the KEY is the deliverable; ARMING it needs a measured A/B we
    /// have not run. No shipped preset may name `margin_row.reserve_max_frac`
    /// until that exists, so every shipped category stays byte-identical.
    #[test]
    fn no_shipped_preset_arms_the_reserve_ceiling() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
        let mut armed: Vec<String> = Vec::new();
        for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
            let dir = root.join(year);
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries {
                let path = entry.expect("dir entry").path();
                if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).expect("readable preset");
                let preset: crate::preset::PresetConfig = serde_yaml::from_str(&text)
                    .unwrap_or_else(|e| panic!("preset {} must parse: {e}", path.display()));
                if preset.margin_row.reserve_max_frac.is_some() {
                    armed.push(format!("{year}/{}", path.display()));
                }
            }
        }
        assert!(
            armed.is_empty(),
            "shipped presets must not arm the reserve ceiling without a measured A/B: {armed:?}"
        );
    }

    /// An already-released reserve skips the ceiling path entirely (and with it
    /// the preset load) and stays released.
    #[test]
    fn capped_to_internal_budget_is_inert_on_a_released_reserve() {
        let released = MarginRowReserveDecision {
            configured_secs: 45,
            reserve_secs: 0,
            route: MarginRowReserveRoute::AdaptiveReleasedAlphaBetaTier,
        };
        let capped = released.capped_to_internal_budget(95, Some(Path::new("/nonexistent.yaml")));
        assert_eq!(capped.reserve_secs, 0);
        assert!(capped.enables_scored_sparse_crown());
    }

    /// The ceiling is a SCHEDULING knob: it must not alter the route, because
    /// the route is what authorizes scored sparse root CROWN.
    #[test]
    fn capped_reserve_preserves_the_route_and_configured_seconds() {
        let decision = MarginRowReserveDecision {
            configured_secs: 45,
            reserve_secs: 45,
            route: MarginRowReserveRoute::AdaptivePreserved,
        };
        let capped = decision.capped_to_internal_budget(95, None);
        assert_eq!(capped.route, MarginRowReserveRoute::AdaptivePreserved);
        assert_eq!(capped.configured_secs, 45);
        assert!(!capped.enables_scored_sparse_crown());
        // With the env unset (the default in the test process) the ceiling
        // declines, so this is also an identity check on the shipped path.
        assert_eq!(capped.reserve_secs, 45);
    }

    /// The reserve resolves to the shipped 45 s default when neither the
    /// environment nor a category preset overrides it. This pins that
    /// resolution behavior (#epoch-bab); it does not establish score impact.
    #[test]
    fn margin_row_reserve_defaults_to_shipped_value_and_is_preset_scoped() {
        // No preset -> the shipped value.
        assert_eq!(margin_row_reserve_secs(None), 45);

        let dir = std::env::temp_dir().join(format!("ny_mr_preset_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmpdir");

        // A preset without a margin_row section -> still the default.
        let plain = dir.join("plain.yaml");
        std::fs::write(&plain, "general: {}\n").expect("write");
        assert_eq!(margin_row_reserve_secs(Some(&plain)), 45);

        // Opt-in is read from the preset.
        let opted = dir.join("opted.yaml");
        std::fs::write(&opted, "margin_row:\n  reserve_secs: 25\n").expect("write");
        assert_eq!(margin_row_reserve_secs(Some(&opted)), 25);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn adaptive_reserve_policy_is_typed_exact_and_fail_closed() {
        assert!(!adaptive_reserve_enabled(
            false,
            AdaptiveReserveGateEnv::Absent
        ));
        assert!(adaptive_reserve_enabled(
            true,
            AdaptiveReserveGateEnv::Absent
        ));
        assert!(adaptive_reserve_enabled(
            false,
            AdaptiveReserveGateEnv::Unicode("1")
        ));
        for malformed in ["0", "", " 1", "1 ", "true", "ON", "yes", "01"] {
            assert!(
                !adaptive_reserve_enabled(true, AdaptiveReserveGateEnv::Unicode(malformed)),
                "present non-exact gate {malformed:?} must retain the fixed reserve"
            );
        }
        assert!(!adaptive_reserve_enabled(
            true,
            AdaptiveReserveGateEnv::NonUnicode
        ));
        assert_eq!(
            adaptive_reserve_gate_env(None),
            AdaptiveReserveGateEnv::Absent
        );
    }

    /// The release predicate is budget arithmetic, so it must reproduce the
    /// two MEASURED regimes that motivated it and must not depend on the
    /// identity of any model or property file.
    #[test]
    fn adaptive_release_is_structural_budget_share_not_instance_identity() {
        let frac = DEFAULT_ADAPTIVE_RELEASE_FRAC;

        // Measured 2026-07-26 (cifar100_2024 prop_idx_9502_sidx_7197):
        // scored 100 s -> 95 s internal tier, 45 s reserve = 47.4% -> RELEASE.
        assert!(
            adaptive_release_target(45, 95, frac),
            "45s of a 95s internal tier is 47% — the regime where the reserve bought nothing"
        );
        // scored 200 s -> 190 s internal tier, 45 s reserve = 23.7% -> PRESERVE.
        assert!(
            !adaptive_release_target(45, 190, frac),
            "45s of a 190s internal tier is 24% — the regime where the lane had airtime"
        );

        // Exactly at the threshold releases; a hair under preserves.
        assert!(adaptive_release_target(40, 100, frac));
        assert!(!adaptive_release_target(39, 100, frac));

        // Degenerate budgets keep the established fixed lane.
        assert!(!adaptive_release_target(45, 0, frac));
        // A zero reserve is never a release target (the caller short-circuits
        // it to NotApplicable, but the predicate must agree).
        assert!(!adaptive_release_target(0, 95, frac));

        // Threshold resolution: env > preset > default, each fail-safe.
        assert_eq!(resolve_release_frac(None, None), frac);
        assert_eq!(resolve_release_frac(Some("0.25"), None), 0.25);
        assert_eq!(resolve_release_frac(None, Some(0.25)), 0.25);
        assert_eq!(resolve_release_frac(Some("0.25"), Some(0.9)), 0.25);
        for bad in ["", "abc", "0", "1", "1.5", "-0.2", "NaN", "inf"] {
            assert_eq!(
                resolve_release_frac(Some(bad), None),
                frac,
                "malformed threshold {bad:?} must fall back to the shipped default"
            );
        }
        for bad in [0.0f32, 1.0, 1.5, -0.2, f32::NAN, f32::INFINITY] {
            assert_eq!(resolve_release_frac(None, Some(bad)), frac);
        }
    }

    /// The removed allowlist keyed on `CIFAR100_resnet_medium.onnx` plus seven
    /// exact property basenames. Nothing in the reserve path may consult a
    /// model or property NAME again — the decision is budget-only.
    #[test]
    fn reserve_path_carries_no_instance_name_literals() {
        let src = include_str!("margin_row_bab.rs");
        let reserve_region = src
            .split_once("pub(crate) const RELEASE_FRAC_ENV")
            .expect("release threshold block present")
            .1
            .split_once("pub(crate) fn margin_row_reserve_secs")
            .expect("fixed-reserve resolver boundary present")
            .0;
        for needle in [
            "CIFAR100_resnet_medium.onnx",
            "prop_idx_",
            "ADAPTIVE_RELEASE_CIFAR100_MEDIUM",
        ] {
            assert!(
                !reserve_region.contains(needle),
                "instance-identity literal {needle:?} reintroduced on the reserve path"
            );
        }
    }

    #[test]
    fn adaptive_reserve_is_default_off_fail_open_and_preserves_closure_sentinels() {
        assert_eq!(
            reserve_policy(45, false, true),
            MarginRowReserveDecision {
                configured_secs: 45,
                reserve_secs: 45,
                route: MarginRowReserveRoute::Fixed,
            },
            "unset/default policy must be byte-for-policy identical even on a target row"
        );
        assert_eq!(
            reserve_policy(45, true, false),
            MarginRowReserveDecision {
                configured_secs: 45,
                reserve_secs: 45,
                route: MarginRowReserveRoute::AdaptivePreserved,
            },
            "unknown adaptive rows retain the established lane budget"
        );
        assert_eq!(
            reserve_policy(45, true, true),
            MarginRowReserveDecision {
                configured_secs: 45,
                reserve_secs: 0,
                route: MarginRowReserveRoute::AdaptiveReleasedAlphaBetaTier,
            }
        );
        assert!(reserve_policy(45, true, true).enables_scored_sparse_crown());
        assert!(!reserve_policy(45, true, false).enables_scored_sparse_crown());
        assert!(!reserve_policy(45, false, true).enables_scored_sparse_crown());
        assert_eq!(
            reserve_policy(0, true, true),
            MarginRowReserveDecision {
                configured_secs: 0,
                reserve_secs: 0,
                route: MarginRowReserveRoute::NotApplicable,
            }
        );

        // The historical margin-row closures were all measured at LONG scored
        // budgets, where the reserve is a small share of the internal tier and
        // the lane demonstrably had airtime. The structural predicate must
        // leave that whole regime on the fixed route, whatever the instance is
        // called. cifar100/tinyimagenet run a 200 s scored budget (190 s
        // internal) and metaroom 210 s (199 s internal).
        for internal_budget in [190u64, 199, 300, 600] {
            assert!(
                !adaptive_release_target(45, internal_budget, DEFAULT_ADAPTIVE_RELEASE_FRAC),
                "long-budget closure regime must retain the reserve ({internal_budget}s internal)"
            );
        }
    }

    /// The concurrent-lane gate: env wins in BOTH directions wherever present,
    /// the typed preset key decides when it is absent, and absent-from-both is
    /// OFF (byte-identical to the shipped reserve-only path).
    #[test]
    fn concurrent_lane_gate_resolution_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let on = dir.path().join("on.yaml");
        std::fs::write(&on, "margin_row:\n  concurrent: true\n").expect("write");
        let off = dir.path().join("off.yaml");
        std::fs::write(&off, "margin_row:\n  concurrent: false\n").expect("write");
        let silent = dir.path().join("silent.yaml");
        std::fs::write(&silent, "margin_row:\n  reserve_secs: 45\n").expect("write");

        // Env absent: the preset decides; a preset that never mentions the key
        // keeps the shipped OFF behaviour.
        assert!(concurrent_lane_armed(Some(&on)));
        assert!(!concurrent_lane_armed(Some(&off)));
        assert!(!concurrent_lane_armed(Some(&silent)));
        assert!(!concurrent_lane_armed(None));
    }

    /// The typed f32 route must never be able to TIGHTEN anything: it only
    /// selects a tableau precision whose rounding is charged into a certified
    /// additive slack. Here we only pin the plumbing — that the preset key is
    /// read, and that a preset which stays silent leaves the flag alone.
    #[test]
    fn root_f32_preset_key_is_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        let on = dir.path().join("on.yaml");
        std::fs::write(&on, "margin_row:\n  root_f32: true\n").expect("write");
        let off = dir.path().join("off.yaml");
        std::fs::write(&off, "margin_row:\n  root_f32: false\n").expect("write");
        let silent = dir.path().join("silent.yaml");
        std::fs::write(&silent, "margin_row:\n  reserve_secs: 0\n").expect("write");

        assert_eq!(root_f32_from_preset(Some(&on)), Some(true));
        assert_eq!(root_f32_from_preset(Some(&off)), Some(false));
        assert_eq!(root_f32_from_preset(Some(&silent)), None);
        assert_eq!(root_f32_from_preset(None), None);
    }

    /// A preset that arms the concurrent lane must not ALSO bill the internal
    /// verifier for a reserve: it would lose the reserved slice AND contend
    /// with the lane, the untested combination 31949bcc refused.
    ///
    /// This checks the EFFECTIVE reserve at the category's official scored
    /// budget, through the real [`reserve_policy`] — not one preset key — so
    /// both shipped ways of satisfying the invariant are covered and neither
    /// can drift:
    ///   * `reserve_secs: 0` (tinyimagenet_2024) never takes a reserve at all;
    ///   * `adaptive_reserve: true` (cifar100_2024) RELEASES the whole reserve
    ///     whenever the configured seconds reach `release_frac` of the internal
    ///     budget — for the shipped 45 s / 0.40 pair that is every internal
    ///     tier up to 112 s, and a scored 100 s budget is a 95 s tier.
    ///
    /// Requiring the budget to be listed here is deliberate: a new concurrent
    /// preset must state the budget its evidence was gathered at, because the
    /// adaptive release is budget-dependent and silently stops firing on long
    /// tiers.
    #[test]
    fn concurrent_presets_never_also_hold_a_reserve() {
        // Official scored per-instance budget from each category's
        // instances.csv (uniform within these categories).
        const OFFICIAL_SCORED_SECS: &[(&str, u64)] =
            &[("cifar100_2024", 100), ("tinyimagenet_2024", 100)];

        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let dir = root.join("configs/vnncomp25");
        let entries = std::fs::read_dir(&dir).unwrap_or_else(|error| {
            panic!("read shipped preset directory {}: {error}", dir.display())
        });
        let mut concurrent_presets_checked = 0usize;
        for entry in entries {
            let entry =
                entry.unwrap_or_else(|error| panic!("read entry under {}: {error}", dir.display()));
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "yaml") {
                continue;
            }
            let cfg = crate::preset::load_preset(&path)
                .unwrap_or_else(|error| panic!("load shipped preset {}: {error}", path.display()));
            if cfg.margin_row.concurrent != Some(true) {
                continue;
            }
            concurrent_presets_checked += 1;
            let category = path
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("preset file stem");
            let scored = OFFICIAL_SCORED_SECS
                .iter()
                .find(|(name, _)| *name == category)
                .map(|(_, secs)| *secs)
                .unwrap_or_else(|| {
                    panic!(
                        "{} arms the concurrent lane but has no official scored budget \
                         recorded here; the adaptive reserve release is budget-dependent, \
                         so the budget the evidence was gathered at must be stated",
                        path.display()
                    )
                });
            // Bind to the PRODUCTION tier, not a local mirror. This test used to
            // call a `#[cfg(test)] internal_timeout_tier` copy that main deleted;
            // the copy was a drift hazard anyway (its own doc called it a "mirror
            // of vnncomp::internal_timeout_secs"), and a stale mirror would let
            // this reserve assertion pass against a rule production no longer uses.
            let internal = crate::commands::vnncomp::internal_timeout_secs(scored);
            let configured = cfg.margin_row.reserve_secs.unwrap_or(45);
            let decision = reserve_policy(
                configured,
                cfg.margin_row.adaptive_reserve.unwrap_or(false),
                adaptive_release_target(
                    configured,
                    internal,
                    resolve_release_frac(None, cfg.margin_row.release_frac),
                ),
            );
            assert_eq!(
                decision.reserve_secs,
                0,
                "{} arms the concurrent lane but still holds {}s of the {internal}s \
                 internal tier (scored {scored}s, route={:?}); the pair taxes the \
                 internal verifier twice",
                path.display(),
                decision.reserve_secs,
                decision.route,
            );
        }
        assert_eq!(
            concurrent_presets_checked,
            OFFICIAL_SCORED_SECS.len(),
            "the concurrent-preset invariant checked {concurrent_presets_checked} shipped \
             preset(s), but the sealed budget registry names {}; a missing/unreadable preset or \
             stale registry must not turn the test into a vacuous pass",
            OFFICIAL_SCORED_SECS.len()
        );
    }

    /// Shipped presets must retain the global reserve until a sealed A/B
    /// justifies a category override. Historical scorecards are not sealed
    /// production evidence, so merely fitting an override beneath an old
    /// solve time is not authority to ship it.
    #[test]
    fn shipped_presets_do_not_claim_unmeasured_reserve_overrides() {
        const PRESETS: &[&str] = &[
            "configs/vnncomp25/metaroom_2023.yaml",
            "configs/vnncomp25/sat_relu.yaml",
            "configs/vnncomp25/cifar100_2024.yaml",
        ];
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root")
            .to_path_buf();
        for preset_rel in PRESETS {
            let preset = root.join(preset_rel);
            assert!(
                preset.is_file(),
                "required shipped preset is missing: {}",
                preset.display()
            );
            let config = crate::preset::load_preset(&preset).unwrap_or_else(|error| {
                panic!("load shipped preset {}: {error}", preset.display())
            });
            assert_eq!(
                config.margin_row.reserve_secs, None,
                "{preset_rel} claims a category reserve override without a sealed A/B"
            );
        }
    }
    /// BATCH SWEEP (#epoch-bab): run the lane over a target list, one line of
    /// evidence per instance. Env:
    ///   NY_SWEEP_TARGETS = file of `onnx_basename,vnnlib_basename` lines
    ///   NY_SWEEP_CAT     = benchmark category dir (default cifar100_2024)
    ///   NY_SWEEP_SECS    = per-instance budget (default 100)
    ///   NY_SWEEP_SKIP/NY_SWEEP_TAKE = shard the list
    pub(crate) fn sweep_targets() {
        let targets = std::env::var("NY_SWEEP_TARGETS").expect("set NY_SWEEP_TARGETS");
        let cat = std::env::var("NY_SWEEP_CAT").unwrap_or_else(|_| "cifar100_2024".into());
        let secs: u64 = std::env::var("NY_SWEEP_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let skip: usize = std::env::var("NY_SWEEP_SKIP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let take: usize = std::env::var("NY_SWEEP_TAKE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX);
        let text = std::fs::read_to_string(&targets).expect("target list");
        let rows: Vec<(String, String)> = text
            .lines()
            .filter_map(|l| l.split_once(','))
            .map(|(a, b)| (a.trim().to_string(), b.trim().to_string()))
            .skip(skip)
            .take(take)
            .collect();
        let (mut unsat, mut unknown, mut skipped) = (0usize, 0usize, 0usize);
        for (i, (ob, vb)) in rows.iter().enumerate() {
            let onnx = format!("{BENCH_ROOT}/{cat}/onnx/{ob}");
            let vnnlib = format!("{BENCH_ROOT}/{cat}/vnnlib/{vb}");
            if !Path::new(&onnx).exists() || !Path::new(&vnnlib).exists() {
                eprintln!("[sweep {i}] MISSING {ob} / {vb}");
                skipped += 1;
                continue;
            }
            let Some(spec) = extract_twin_spec(Path::new(&onnx)) else {
                eprintln!("[sweep {i}] {vb} -> extract fail-closed");
                skipped += 1;
                continue;
            };
            let Some((lo, hi, t, adv)) = parse_vnnlib_robustness(Path::new(&vnnlib)) else {
                eprintln!("[sweep {i}] {vb} -> parse fail-closed");
                skipped += 1;
                continue;
            };
            let t0 = Instant::now();
            let deadline = t0 + Duration::from_secs(secs);
            let out = run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000);
            let el = t0.elapsed().as_secs_f64();
            match out {
                MarginRowOutcome::Unsat(st) => {
                    unsat += 1;
                    eprintln!(
                        "[sweep {i}] {vb} -> UNSAT {el:.1}s root={:.4} exp={} ledger={:?}",
                        st.root_bound, st.expansions, st.ledger_ok
                    );
                }
                MarginRowOutcome::Unknown { reason, stats } => {
                    unknown += 1;
                    eprintln!(
                        "[sweep {i}] {vb} -> unknown({reason}) {el:.1}s root={:?}",
                        stats.map(|s| s.root_bound)
                    );
                }
            }
        }
        eprintln!(
            "[sweep] SUMMARY cat={cat} budget={secs}s rows={} UNSAT={unsat} unknown={unknown} skipped={skipped}",
            rows.len()
        );
    }

    fn require_probe_instance() -> anyhow::Result<()> {
        let category = std::env::var("NY_PROBE_CAT").unwrap_or_else(|_| "cifar100_2024".to_owned());
        let default_onnx = if category.starts_with("tiny") {
            "TinyImageNet_resnet_medium.onnx"
        } else {
            "CIFAR100_resnet_medium.onnx"
        };
        let onnx = std::env::var("NY_PROBE_ONNX").unwrap_or_else(|_| default_onnx.to_owned());
        let vnnlib = std::env::var("NY_PROBE_VNNLIB")
            .map_err(|_| anyhow::anyhow!("this probe requires NY_PROBE_VNNLIB"))?;
        for path in [
            Path::new(BENCH_ROOT)
                .join(&category)
                .join("onnx")
                .join(onnx),
            Path::new(BENCH_ROOT)
                .join(category)
                .join("vnnlib")
                .join(vnnlib),
        ] {
            anyhow::ensure!(
                path.is_file(),
                "missing margin-row prerequisite {}",
                path.display()
            );
        }
        Ok(())
    }

    /// Execute one explicit, serial real-corpus margin-row measurement.
    pub(crate) fn run(probe: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            Path::new(BENCH_ROOT).is_dir(),
            "missing VNN-COMP benchmark root {BENCH_ROOT}"
        );
        match probe {
            "instance" => {
                require_probe_instance()?;
                probe_env_instance();
            }
            "root-build" => {
                require_probe_instance()?;
                probe_root_build();
            }
            "pass-scaling" => {
                require_probe_instance()?;
                probe_pass_scaling();
            }
            "serial-parallel-oracle" => {
                require_probe_instance()?;
                oracle_serial_vs_parallel();
            }
            "inc2a-root-parity-1498" => inc2a_root_parity_1498(),
            "inc2a-enclosure-1498" => inc2a_outward_bound_and_enclosure_1498(),
            "inc2b-trajectory-1498" => inc2b_trajectory_replay_1498(),
            "inc2b-l5-parity" => inc2b_unstable_position_parity_l5(),
            "inc2c-close-4429" => inc2c_closes_4429(),
            "inc2c-close-1498" => inc2c_closes_1498(),
            "inc2c-close-2551" => inc2c_closes_2551(),
            "inc2c-moat-6232" => inc2c_moat_sat_6232_not_verified(),
            "extractor-smoke" => extractor_and_parser_smoke(),
            "cifar-moat" => moat_gt_sat_never_unsat(),
            "cifar-closure" => closure_gt_unsat_contingency(),
            "extraction-safety" => extraction_safety_other_models(),
            "epoch-gain" => {
                require_probe_instance()?;
                epoch_gain_probe_real();
            }
            "metaroom-gains" => metaroom_banked_gains_still_close(),
            "metaroom-moat" => metaroom_moat_gt_sat_never_unsat(),
            "sweep" => {
                let targets = std::env::var("NY_SWEEP_TARGETS")
                    .map_err(|_| anyhow::anyhow!("sweep requires NY_SWEEP_TARGETS"))?;
                anyhow::ensure!(
                    Path::new(&targets).is_file(),
                    "missing sweep target list {targets}"
                );
                sweep_targets();
            }
            _ => anyhow::bail!(
                "unknown margin-row probe {probe:?}; expected instance, root-build, \
                 pass-scaling, serial-parallel-oracle, inc2a-root-parity-1498, \
                 inc2a-enclosure-1498, inc2b-trajectory-1498, inc2b-l5-parity, \
                 inc2c-close-4429, inc2c-close-1498, inc2c-close-2551, \
                 inc2c-moat-6232, extractor-smoke, cifar-moat, cifar-closure, \
                 extraction-safety, epoch-gain, metaroom-gains, metaroom-moat, or sweep"
            ),
        }
        Ok(())
    }
}
