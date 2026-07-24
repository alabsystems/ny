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

/// The seven still-open CIFAR100 medium rows isolated by the 2026-07-19
/// barrier-1 campaign. On these exact rows the margin-row root was observed to
/// be noncompetitive while the verifier did not reach useful BaB airtime after
/// paying the fixed reserve. This list is intentionally exact and narrow: an
/// unknown row, a renamed file, or the same property index on another model
/// keeps the reserve (fail-open toward the historically established lane).
const ADAPTIVE_RELEASE_CIFAR100_MEDIUM: &[&str] = &[
    "CIFAR100_resnet_medium_prop_idx_815_sidx_1902_eps_0.0039.vnnlib",
    "CIFAR100_resnet_medium_prop_idx_966_sidx_2330_eps_0.0039.vnnlib",
    "CIFAR100_resnet_medium_prop_idx_1190_sidx_8846_eps_0.0039.vnnlib",
    "CIFAR100_resnet_medium_prop_idx_1761_sidx_3933_eps_0.0039.vnnlib",
    "CIFAR100_resnet_medium_prop_idx_1798_sidx_1061_eps_0.0039.vnnlib",
    "CIFAR100_resnet_medium_prop_idx_2050_sidx_8228_eps_0.0039.vnnlib",
    "CIFAR100_resnet_medium_prop_idx_2477_sidx_388_eps_0.0039.vnnlib",
];

fn adaptive_release_target(onnx: &Path, vnnlib: &Path) -> bool {
    onnx.file_name().and_then(|name| name.to_str()) == Some("CIFAR100_resnet_medium.onnx")
        && vnnlib
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| ADAPTIVE_RELEASE_CIFAR100_MEDIUM.contains(&name))
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
/// release on the exact open αβ-tier rows above. All parse, extraction,
/// environment, and identity uncertainty retains the fixed policy. No verifier
/// or margin-row bound/verdict code is changed by this decision.
pub(crate) fn margin_row_reserve_decision(
    onnx: &Path,
    vnnlib: &Path,
    preset: Option<&Path>,
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
        adaptive_release_target(onnx, vnnlib),
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
/// Mirror of `vnncomp::internal_timeout_secs`: the internal verifier's budget
/// is the scored budget minus a grace tier. A reserve comes out of THAT, not
/// the scored budget — getting this wrong once made a 25s reserve look safe
/// against 28s of headroom when the real tail was 18s.
#[cfg(test)]
pub(crate) fn internal_timeout_tier(timeout_secs: u64) -> u64 {
    let grace = (timeout_secs / 20).max(5);
    timeout_secs
        .checked_sub(grace)
        .filter(|&t| t >= 1)
        .unwrap_or(timeout_secs)
}

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

/// Run `work` on a DETACHED worker thread and bound the join at `cap`
/// (#twinwall watchdog-kill class, banked 99ed4d42) — the same posture as
/// ny-mip's `run_with_hard_deadline` slice enforcement.
///
/// WHY: `run_margin_row_lane`'s internal deadline checks are cooperative and
/// can be arbitrarily coarse between expensive tableau builds (measured on
/// metaroom: 113.1s against a 45s slice; one run overran until the process
/// watchdog killed it at budget+grace — the error-scoring risk class). The
/// only sound external enforcement is to abandon the work at the boundary.
///
/// Returns `None` when the cap expires, the worker cannot be spawned, or the
/// worker panics — all VERDICT-NEUTRAL, fail-closed outcomes. The abandoned
/// worker keeps running until its own cooperative deadline fires or process
/// teardown (accepted cost, identical to the MIP lane's detached workers).
fn run_capped_detached<T: Send + 'static>(
    cap: Duration,
    label: &'static str,
    work: impl FnOnce() -> T + Send + 'static,
) -> Option<T> {
    // Capacity-1 channel: the send never blocks, and a receiver gone after
    // the cap makes it fail — the expected abandoned-worker case.
    let (tx, rx) = std::sync::mpsc::sync_channel::<T>(1);
    if std::thread::Builder::new()
        .name(format!("margin-row-{label}"))
        .spawn(move || {
            let _ = tx.send(work());
        })
        .is_err()
    {
        eprintln!("margin-row BaB: could not spawn {label} worker; skipping lane (fail-closed)");
        return None;
    }
    match rx.recv_timeout(cap) {
        Ok(out) => Some(out),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            eprintln!(
                "margin-row BaB: {label} worker exceeded its {:.1}s hard slice cap; abandoning \
                 the detached worker and returning in-budget (fail-closed, verdict-neutral)",
                cap.as_secs_f64()
            );
            None
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            eprintln!(
                "margin-row BaB: {label} worker exited without a result (panicked); fail-closed"
            );
            None
        }
    }
}

/// Strictly-additive vnncomp hook: only ever turns unknown/timeout into
/// `Unsat`. Returns `None` on any mismatch, budget shortfall, or non-verdict.
pub(crate) fn try_margin_row_unsat(
    onnx: &Path,
    vnnlib: &Path,
    instance_deadline: Option<Instant>,
) -> Option<VnncompResult> {
    if !ny_propagate::margin_row::margin_row_bab_enabled() {
        return None;
    }
    // No scored deadline (interactive runs): cap the lane at 10 min.
    let instance_deadline =
        instance_deadline.unwrap_or_else(|| Instant::now() + Duration::from_mins(10));
    let remaining = instance_deadline.saturating_duration_since(Instant::now());
    if remaining < Duration::from_secs(10) {
        return None; // not enough budget for root gates + a useful tree
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
            return None;
        }
    };
    let (lo, hi, t, adv) = match parse_vnnlib_robustness(vnnlib) {
        Some(v) => v,
        None => {
            if dbg {
                eprintln!("margin-row: parse_vnnlib_robustness returned None");
            }
            return None;
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
        return None;
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
    // kill). Detached worker + bounded join: on expiry the lane returns
    // `None` in-budget and the caller's post-BaB reserve stays intact. The
    // worker still gets `Some(deadline)` so it normally stops cooperatively
    // well before the external cap fires.
    let cap = deadline.saturating_duration_since(t0);
    let out = run_capped_detached(cap, "inline", move || {
        run_margin_row_lane(&spec, &lo, &hi, t, &adv, Some(deadline), 20_000)
    })?;
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
            Some(VnncompResult::Unsat)
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
            None
        }
    }
}

/// Handle to a margin-row lane running CONCURRENTLY with the internal
/// verifier (#epoch-bab).
pub(crate) struct ConcurrentLane {
    rx: std::sync::mpsc::Receiver<Option<VnncompResult>>,
}

impl ConcurrentLane {
    /// Collect the concurrent lane's verdict, waiting up to `grace` for it to
    /// land. `None` = no verdict (still running, failed, or not decided): the
    /// caller keeps whatever it had.
    pub(crate) fn take(self, grace: Duration) -> Option<VnncompResult> {
        self.rx.recv_timeout(grace).unwrap_or_default()
    }
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
    if std::env::var("NY_MARGIN_ROW_CONCURRENT").ok().as_deref() != Some("1") {
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
    std::thread::Builder::new()
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
    Some(ConcurrentLane { rx })
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
        .map_or(default, |a| f64::from(a.f))
}

fn attr_i(n: &NodeProto, name: &str, default: i64) -> i64 {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .map_or(default, |a| a.i)
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
//  All #[ignore]: they need benchmarks/vnncomp2025 locally and real minutes.
//  Run explicitly, serially:
//    cargo test -p ny-cli margin_row_bab -- --ignored --test-threads=1
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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
        assert_eq!(out, None);
        assert!(
            elapsed < Duration::from_secs(5),
            "cap not enforced: returned after {elapsed:?}"
        );
    }

    #[test]
    fn capped_detached_passes_through_fast_work() {
        let out = run_capped_detached(Duration::from_secs(30), "test-fast", || 7_u32);
        assert_eq!(out, Some(7));
    }

    #[test]
    fn capped_detached_panicking_worker_is_fail_closed() {
        let out = run_capped_detached(Duration::from_secs(30), "test-panic", || -> u32 {
            panic!("worker panic must map to None, never a verdict")
        });
        assert_eq!(out, None);
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
    #[test]
    #[ignore = "measurement harness; drive via NY_PROBE_* env; run solo"]
    fn probe_env_instance() {
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
    /// NY_ROOT_DUMP=<path> to also write every (l,u) as LE f64 for an
    /// outward-enclosure check (BLAS path). NY_ROOT_BLAS=1 selects the DGEMM
    /// forward-conv (provably-outward, not bit-identical).
    ///   NY_PROBE_CAT / NY_PROBE_ONNX / NY_PROBE_VNNLIB, NY_ROOT_REPS
    #[test]
    #[ignore = "root-build profile+oracle; NY_PROBE_VNNLIB=<inst>; run solo"]
    fn probe_root_build() {
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
    #[test]
    #[ignore = "measurement; NY_PROBE_VNNLIB=<inst>; run solo"]
    fn probe_pass_scaling() {
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
    #[test]
    #[ignore = "oracle diff; NY_PROBE_VNNLIB=<inst>; run solo"]
    fn oracle_serial_vs_parallel() {
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
    #[test]
    #[ignore = "needs benchmarks/vnncomp2025 + ~1 min; run serially"]
    fn inc2a_root_parity_1498() {
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
    #[test]
    #[ignore = "needs benchmarks/vnncomp2025 + ~2 min; run serially"]
    fn inc2a_outward_bound_and_enclosure_1498() {
        use ndarray::Array2;
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};
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
        let mut rng = StdRng::seed_from_u64(271_828_182);
        let n_in = net.n_in;
        let bsz = 200;
        let mut x = Array2::<f64>::zeros((n_in, bsz));
        for i in 0..n_in {
            for b in 0..bsz {
                let u: f64 = rng.random_range(0.0..1.0);
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
    #[test]
    #[ignore = "needs benchmarks/vnncomp2025 + ~5 min; run serially"]
    fn inc2b_trajectory_replay_1498() {
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
    #[test]
    #[ignore = "needs benchmarks/vnncomp2025 + ~1 min; run serially"]
    fn inc2b_unstable_position_parity_l5() {
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
    #[test]
    #[ignore = "needs benchmarks/vnncomp2025; ~100s per instance; run solo at the END"]
    fn inc2c_closes_4429() {
        assert_closes(P4429, 100);
    }

    #[test]
    #[ignore = "needs benchmarks/vnncomp2025; ~100s; run solo at the END"]
    fn inc2c_closes_1498() {
        assert_closes(P1498, 100);
    }

    #[test]
    #[ignore = "needs benchmarks/vnncomp2025; ~100s; run solo at the END"]
    fn inc2c_closes_2551() {
        assert_closes(P2551, 100);
    }

    /// MOAT: a banked-SAT instance must NEVER verify.
    #[test]
    #[ignore = "needs benchmarks/vnncomp2025; ~100s; run solo at the END"]
    fn inc2c_moat_sat_6232_not_verified() {
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
    #[test]
    #[ignore = "needs benchmarks/vnncomp2025"]
    fn extractor_and_parser_smoke() {
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
    #[test]
    #[ignore = "moat scan: 14 GT-sat instances; run solo, NY_MOAT_SECS=60"]
    fn moat_gt_sat_never_unsat() {
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
    #[test]
    #[ignore = "closure scan: GT-unsat tiers; run solo, NY_MOAT_SECS=100"]
    fn closure_gt_unsat_contingency() {
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
    #[test]
    #[ignore = "needs benchmarks/vnncomp2025"]
    fn extraction_safety_other_models() {
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
    #[test]
    #[ignore = "measurement; needs benchmarks/vnncomp2025; run solo"]
    fn epoch_gain_probe_real() {
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
    #[test]
    #[ignore = "needs benchmarks/vnncomp2025 metaroom; ~1 min; run solo"]
    fn metaroom_banked_gains_still_close() {
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
    #[test]
    #[ignore = "needs benchmarks/vnncomp2025 metaroom; run solo"]
    fn metaroom_moat_gt_sat_never_unsat() {
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

    #[test]
    fn adaptive_reserve_targets_exactly_the_open_alpha_beta_tier() {
        let model = Path::new("/bench/onnx/CIFAR100_resnet_medium.onnx");
        assert_eq!(ADAPTIVE_RELEASE_CIFAR100_MEDIUM.len(), 7);
        for name in ADAPTIVE_RELEASE_CIFAR100_MEDIUM {
            assert!(
                adaptive_release_target(model, Path::new(name)),
                "open tier row must engage: {name}"
            );
            assert!(
                !adaptive_release_target(
                    Path::new("TinyImageNet_resnet_medium.onnx"),
                    Path::new(name)
                ),
                "the same property basename on another model must retain its reserve"
            );
            let renamed = format!("copy-{name}");
            assert!(
                !adaptive_release_target(model, Path::new(&renamed)),
                "renamed/unknown evidence must fail open to the fixed reserve"
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

        // Representative historical margin-row closures across the affected
        // families. Exact model+property matching keeps every one on the fixed
        // route; similarly numbered TinyImageNet rows cannot collide.
        const CLOSURE_SENTINELS: &[(&str, &str)] = &[
            (
                "CIFAR100_resnet_medium.onnx",
                "CIFAR100_resnet_medium_prop_idx_54_sidx_9735_eps_0.0039.vnnlib",
            ),
            (
                "CIFAR100_resnet_medium.onnx",
                "CIFAR100_resnet_medium_prop_idx_1498_sidx_792_eps_0.0039.vnnlib",
            ),
            (
                "CIFAR100_resnet_medium.onnx",
                "CIFAR100_resnet_medium_prop_idx_1588_sidx_6654_eps_0.0039.vnnlib",
            ),
            (
                "CIFAR100_resnet_medium.onnx",
                "CIFAR100_resnet_medium_prop_idx_4605_sidx_6225_eps_0.0039.vnnlib",
            ),
            (
                "CIFAR100_resnet_medium.onnx",
                "CIFAR100_resnet_medium_prop_idx_7176_sidx_1522_eps_0.0039.vnnlib",
            ),
            (
                "CIFAR100_resnet_large.onnx",
                "CIFAR100_resnet_large_prop_idx_5308_sidx_1650_eps_0.0039.vnnlib",
            ),
            (
                "TinyImageNet_resnet_medium.onnx",
                "TinyImageNet_resnet_medium_prop_idx_1126_sidx_4974_eps_0.0039.vnnlib",
            ),
            (
                "TinyImageNet_resnet_medium.onnx",
                "CIFAR100_resnet_medium_prop_idx_815_sidx_1902_eps_0.0039.vnnlib",
            ),
            (
                "6cnn_ry_64_10_no_custom_OP.onnx",
                "spec_idx_132_eps_0.00000436.vnnlib",
            ),
        ];
        for (model, property) in CLOSURE_SENTINELS {
            assert!(
                !adaptive_release_target(Path::new(model), Path::new(property)),
                "historical closure sentinel must retain reserve: {model}/{property}"
            );
        }
    }

    /// Legacy-scorecard tripwire for explicit future preset reserves. It
    /// cross-checks each preset's declared `reserve_secs` against historical
    /// solve rows; it does not validate the global default or constitute
    /// sealed production evidence.
    #[test]
    fn shipped_preset_reserves_never_threaten_a_measured_solve() {
        // (preset, legacy scorecard, official per-instance budget)
        const CATS: &[(&str, &str, u64)] = &[
            (
                "configs/vnncomp25/metaroom_2023.yaml",
                "reports/measured/metaroom_2023.csv",
                210,
            ),
            (
                "configs/vnncomp25/sat_relu.yaml",
                "reports/measured/sat_relu.csv",
                100,
            ),
            (
                "configs/vnncomp25/cifar100_2024.yaml",
                "reports/measured/cifar100_2024.csv",
                100,
            ),
        ];
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root")
            .to_path_buf();
        for (preset_rel, csv_rel, budget) in CATS {
            let preset = root.join(preset_rel);
            let csv = root.join(csv_rel);
            if !preset.exists() || !csv.exists() {
                continue; // category not present in this checkout
            }
            // Only PRESET-DECLARED reserves are checked here. This invariant
            // does not validate the global 45s default: historical CIFAR100
            // +23 bookkeeping appears to use 45s, but the TinyImageNet +67
            // commit bodies explicitly record an 82s override, and neither is
            // sealed production evidence. Re-litigating the default from
            // per-row solve times would also trip on stale rows — e.g.
            // metaroom's 182.0s `spec_idx_28`,
            // which on the current binary times out with a 45s reserve, a 5s
            // reserve, and NO reserve alike.
            let declared = crate::preset::load_preset(&preset)
                .ok()
                .and_then(|c| c.margin_row.reserve_secs);
            let Some(reserve) = declared else {
                continue; // takes the shipped default; outside this preset-only invariant
            };
            if reserve == 0 {
                continue; // explicitly opted out
            }
            let text = std::fs::read_to_string(&csv).expect("scorecard");
            let mut worst = 0.0f64;
            for line in text.lines() {
                let f: Vec<&str> = line.split(',').collect();
                if f.len() < 6 {
                    continue;
                }
                let verdict = f[4].trim();
                if verdict != "sat" && verdict != "unsat" {
                    continue;
                }
                if let Ok(t) = f[5].trim().parse::<f64>() {
                    worst = worst.max(t);
                }
            }
            // The internal verifier never gets the raw budget — it gets
            // `internal_timeout_secs(budget)` (budget minus a grace tier of
            // budget/20, min 5s). The reserve comes out of THAT, so the true
            // unused tail is measured against the internal tier. Getting this
            // wrong once cost metaroom its 182.0s solve on paper (25s reserve
            // vs 18.0s of real headroom).
            let internal = internal_timeout_tier(*budget);
            #[allow(clippy::cast_precision_loss)]
            let headroom = internal as f64 - worst;
            #[allow(clippy::cast_precision_loss)]
            let reserve_f = reserve as f64;
            assert!(
                reserve_f <= headroom,
                "{preset_rel} reserves {reserve}s but the slowest MEASURED solve on \
                 {csv_rel} takes {worst:.1}s and the internal verifier only gets \
                 {internal}s of the {budget}s budget (just {headroom:.1}s of unused \
                 tail) — this reserve would forfeit a solve NY currently lands"
            );
        }
    }
    /// BATCH SWEEP (#epoch-bab): run the lane over a target list, one line of
    /// evidence per instance. Env:
    ///   NY_SWEEP_TARGETS = file of `onnx_basename,vnnlib_basename` lines
    ///   NY_SWEEP_CAT     = benchmark category dir (default cifar100_2024)
    ///   NY_SWEEP_SECS    = per-instance budget (default 100)
    ///   NY_SWEEP_SKIP/NY_SWEEP_TAKE = shard the list
    #[test]
    #[ignore = "measurement sweep; needs benchmarks/vnncomp2025; run solo"]
    fn sweep_targets() {
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
}
