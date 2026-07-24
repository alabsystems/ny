// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// In-process ay backend: `ir::MilpProblem` -> `ay_milp::Model` and back.
//
// R3 of the ay-as-library plan (ay repo,
// designs/2026-07-12-ay-as-library-for-ny.md): replaces the P0 subprocess
// call-out (~25 ms/solve of spawn + SMT-LIB text) with typed in-process
// calls (µs). The subprocess lane survives as `MipBackend::AyProc`
// (debug/bootstrap only, frozen P0 code).
//
// Soundness posture is unchanged from P0: any Sat is revalidated by the
// concrete forward pass downstream; Unsat requires all phase-split
// subproblems Unsat; anything inconclusive degrades to Timeout, never
// Unsat. New over P0: ay-milp verdicts can carry exact model-level
// certificates (Farkas / optimality); we verify them here when present and
// surface them through verdict admission (LG3). An UNSAT whose exported
// Farkas witness does not re-check is an error, never a verdict.

use ay_milp::{BabSession, Model, Outcome, Sense as AySense, SolveOpts};
use num_rational::BigRational;
use num_traits::ToPrimitive;

use crate::ay::{ObjSense, ObjectiveSpec};
use crate::error::MipError;
use crate::ir::{Col, MilpProblem};
use crate::solver::MipResult;

type Result<T> = std::result::Result<T, MipError>;

/// BLESSED ENV CHOKE POINT (clippy env wall) for AY's process-global solver
/// configuration.
///
/// The exact Git-pinned ay-milp engine is configured through `AY_MILP_*` environment
/// variables (its `SolveOpts` carries no engine-economics knobs at the pinned
/// rev), so NY's fail-closed posture — lattice quarantine, saturation-stop
/// kill switches, flip-LNS schedule, window recipe — is DELIBERATELY sticky,
/// process-global state reasserted before every solve. That is the one
/// production-side use of `std::env::set_var`/`remove_var` in the workspace;
/// every write is serialized behind one mutex here so concurrent solver
/// sessions and env-contract tests can never interleave half-applied
/// schedules. Verdicts never depend on these knobs (advice-lane only; every
/// answer still rests on ay's exact certificates + `check_point`).
mod ay_env {
    use std::sync::{Mutex, MutexGuard};

    static AY_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> MutexGuard<'static, ()> {
        AY_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Serialized process-global set (the blessed choke point).
    pub(super) fn set(key: &str, value: &str) {
        let _g = lock();
        // Blessed choke point: the one place raw set_var is allowed in ny-mip.
        // (`env_mutation` is the Trust toolchain's deny-by-default env wall;
        // stock rustc doesn't know it, hence `unknown_lints`.)
        #[allow(clippy::disallowed_methods)]
        #[allow(unknown_lints, env_mutation)]
        std::env::set_var(key, value);
    }

    /// Serialized process-global remove (the blessed choke point).
    pub(super) fn remove(key: &str) {
        let _g = lock();
        // Blessed choke point: the one place raw remove_var is allowed in ny-mip.
        // (`env_mutation`: Trust-only deny-by-default env wall.)
        #[allow(clippy::disallowed_methods)]
        #[allow(unknown_lints, env_mutation)]
        std::env::remove_var(key);
    }

    /// Serialized read-then-set: a knob already present in the environment is
    /// never overwritten (user/session overrides always win), and the
    /// check+write pair is atomic under the lock.
    pub(super) fn set_if_unset(key: &str, value: &str) {
        let _g = lock();
        if std::env::var_os(key).is_none() {
            // Blessed choke point (see `set`).
            // (`env_mutation`: Trust-only deny-by-default env wall.)
            #[allow(clippy::disallowed_methods)]
            #[allow(unknown_lints, env_mutation)]
            std::env::set_var(key, value);
        }
    }
}

const HISTORICAL_AY_FLIP_SHARE: &str = "0.75";
const NATIVE_AY_FLIP_CAP_SECS: &str = "18";

/// Choose whether NY admits AY's default-on tall-LP flip-LNS cap.
///
/// This is deliberately fail-closed: only an exact `1` after trimming enables
/// the new schedule. Missing, non-Unicode, and malformed values retain NY's
/// previously pinned 75% flip-LNS share.
fn use_ay_tall_flip_cap(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

/// Apply NY's flip-LNS scheduling policy to AY's process-global environment.
///
/// AY interprets an explicit `AY_MILP_FLIP_SHARE` as an opt-out from its tall
/// absolute cap. NY therefore pins the historical share by default and removes
/// that variable only for the provenance-captured canary. The canary also pins
/// the eta-lane cap to 18 seconds and disables AY's independently tunable warm-LU
/// lane, so inherited AY settings cannot silently turn the recorded canary into
/// a custom or 12-second schedule. Environment mutation is process-global:
/// `NY_AY_MILP_TALL_FLIP_CAP` is startup configuration and must not be toggled
/// concurrently with solver calls. Reapplying this policy before every branch-
/// and-bound session prevents inherited AY settings from silently changing
/// either posture.
fn configure_ay_flip_schedule() {
    let raw = std::env::var("NY_AY_MILP_TALL_FLIP_CAP").ok();
    let native_tall_cap = use_ay_tall_flip_cap(raw.as_deref());
    ay_env::set("AY_MILP_FLIP_CAP_SECS", NATIVE_AY_FLIP_CAP_SECS);
    ay_env::remove("AY_MILP_WARM_LU");
    if native_tall_cap {
        ay_env::remove("AY_MILP_FLIP_SHARE");
    } else {
        ay_env::set("AY_MILP_FLIP_SHARE", HISTORICAL_AY_FLIP_SHARE);
    }
}

/// Lower the solver-neutral IR to an `ay_milp::Model`.
///
/// Column order is preserved (IR `Col(i)` == ay-milp column `i`). Integer
/// columns must be ReLU binaries (`[0,1]`, possibly `fix_col`-pinned to a
/// single value) — the same contract the P0 SMT-LIB lowering enforced.
/// `ColSpec::obj` is ignored, matching P0: feasibility solves have no
/// objective, and optimization passes the objective separately.
pub(crate) fn to_ay_model(problem: &MilpProblem) -> Result<Model> {
    let mut model = Model::new();
    for (i, spec) in problem.cols().iter().enumerate() {
        if spec.integer {
            let col = model.add_binary_col();
            let pinned_zero = spec.lb == 0.0 && spec.ub == 0.0;
            let pinned_one = spec.lb == 1.0 && spec.ub == 1.0;
            let full = spec.lb == 0.0 && spec.ub == 1.0;
            if pinned_zero || pinned_one {
                model.fix_col(col, spec.lb);
            } else if !full {
                return Err(MipError::Encoding(format!(
                    "integer column {i} must be a ReLU binary in [0,1] or pinned, got [{}, {}]",
                    spec.lb, spec.ub
                )));
            }
        } else {
            let _ = model.add_col(spec.lb, spec.ub);
        }
    }
    for row in problem.rows() {
        let coeffs: Vec<(ay_milp::Col, f64)> = row
            .coeffs
            .iter()
            .map(|&(c, a)| {
                model
                    .col_at(c)
                    .map(|col| (col, a))
                    .ok_or_else(|| MipError::Encoding(format!("row references column {c}")))
            })
            .collect::<Result<_>>()?;
        model.add_row(row.lb, row.ub, &coeffs);
    }
    if let Some(row) = problem.margin_row() {
        let ay_row = model.row_at(row.0).ok_or_else(|| {
            MipError::Encoding(format!(
                "marked margin row {} disappeared during AY lowering",
                row.0
            ))
        })?;
        model.mark_margin_row(ay_row).map_err(|e| {
            MipError::Encoding(format!(
                "AY rejected marked margin row {} during lowering: {e}",
                row.0
            ))
        })?;
    }
    Ok(model)
}

/// Lower the IR to the CONTINUOUS LP relaxation of `problem`: every integer
/// (ReLU-binary) column becomes a continuous variable over its declared box
/// (`[0,1]`, or a pinned point). Everything else is byte-for-byte identical to
/// [`to_ay_model`] — same column order, same rows, same margin marker.
///
/// SOUNDNESS: the feasible set of the LP relaxation CONTAINS the MILP's feasible
/// set (relaxing integrality only adds points), so any rigorous min/max over a
/// column of this model is a valid OUTER bound on that column's reachable value
/// in the true MILP — and hence on the true reachable pre-activation set the
/// MILP over-approximates. Used only for OBBT bound tightening, never to decide
/// feasibility (a relaxed model must not be handed to the certified-UNSAT path).
pub(crate) fn to_ay_model_relaxed(problem: &MilpProblem) -> Result<Model> {
    let mut model = Model::new();
    for (i, spec) in problem.cols().iter().enumerate() {
        if spec.integer {
            let pinned = spec.lb == spec.ub;
            let full = spec.lb == 0.0 && spec.ub == 1.0;
            if !pinned && !full {
                return Err(MipError::Encoding(format!(
                    "integer column {i} must be a ReLU binary in [0,1] or pinned, got [{}, {}]",
                    spec.lb, spec.ub
                )));
            }
            // Continuous relaxation: the same box, no integrality.
            let _ = model.add_col(spec.lb, spec.ub);
        } else {
            let _ = model.add_col(spec.lb, spec.ub);
        }
    }
    for row in problem.rows() {
        let coeffs: Vec<(ay_milp::Col, f64)> = row
            .coeffs
            .iter()
            .map(|&(c, a)| {
                model
                    .col_at(c)
                    .map(|col| (col, a))
                    .ok_or_else(|| MipError::Encoding(format!("row references column {c}")))
            })
            .collect::<Result<_>>()?;
        model.add_row(row.lb, row.ub, &coeffs);
    }
    // No margin marker on the relaxation: OBBT optimizes each target column
    // directly; the marker only matters for the feasibility/decision path.
    Ok(model)
}

#[cfg(test)]
mod margin_row_mapping_tests {
    use super::*;

    #[test]
    fn lowering_preserves_exact_marked_row_identity() {
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, 0.0, 2.0);
        problem.add_row(1.5, f64::INFINITY, [(x, 1.0)]);
        let margin = problem.add_row(f64::NEG_INFINITY, 1.0, [(x, 1.0)]);
        problem.mark_margin_row(margin).expect("valid margin row");

        let model = to_ay_model(&problem).expect("lowering");
        let mapped = model.margin_row().expect("AY model must carry marker");
        assert_eq!(mapped.index(), margin.0);
        let (coeffs, lb, ub) = model.row(mapped);
        assert_eq!(coeffs, &[(0, 1.0)]);
        assert_eq!(lb, f64::NEG_INFINITY);
        assert_eq!(ub, 1.0);
    }

    #[test]
    fn unmarked_one_sided_row_stays_unmarked() {
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, 0.0, 2.0);
        problem.add_row(f64::NEG_INFINITY, 1.0, [(x, 1.0)]);

        let model = to_ay_model(&problem).expect("lowering");
        assert_eq!(
            model.margin_row(),
            None,
            "lowering must never infer a margin"
        );
    }
}

/// Stack reservation for the detached solve worker. Matches ay's own
/// solver-thread headroom (`SMT_FILE_THREAD_STACK_SIZE`, ay
/// crates/ay/src/run.rs — the reason its CLI re-execs itself), so moving the
/// in-process solve off the caller's thread can never LOWER the stack the
/// engine had before this wrapper existed.
const SOLVE_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Run an in-process ay solve on a detached worker thread and enforce the MIP
/// slice from OUTSIDE the ay session.
///
/// WHY (vnncomp timeout arc, 2026-07-18): when ay's native B&B goes Unknown
/// near the slice deadline, its SMT fallback (ay-dpll persistent loop) grants
/// itself a fresh budget with deadline checks too coarse for BigRational
/// pivots — `session.check()` can overshoot the slice by minutes, so the
/// harness watchdog kills the process and scores `timeout`/`error` instead of
/// a clean in-budget `unknown`. `SolveOpts` at the pinned rev exposes no
/// cancel/interrupt hook (deadline/time_limit are advisory to the engine's
/// own checks), so the only sound external enforcement is to abandon the
/// solve at the boundary — the exact posture the `AyProc` subprocess lane
/// already has (`ay/run.rs` polls and kills the process tree at the slice).
///
/// Returns `Ok(None)` when the slice expires — VERDICT-NEUTRAL: callers map
/// it to `MipResult::Timeout` / a fail-closed decline, never a verdict.
///
/// ACCEPTED COST: the abandoned worker keeps running (it still holds ay's
/// internal deadline, which usually fires late but does fire; a pathological
/// hang is bounded by process teardown — for vnncomp, the per-instance
/// process exit). It is detached precisely so a spinning solve can never
/// stall the caller past its budget again.
pub(crate) fn run_with_hard_deadline<T, F>(
    timeout_secs: f64,
    label: &'static str,
    solve: F,
) -> Result<Option<T>>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    // Same clamp as `solve_opts`, so the external deadline and the session's
    // internal time limit describe the same slice.
    let slice = std::time::Duration::from_secs_f64(timeout_secs.clamp(0.001, 86_400.0));
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<T>>(1);
    std::thread::Builder::new()
        .name(format!("ny-mip-ay-{label}"))
        .stack_size(SOLVE_THREAD_STACK_BYTES)
        .spawn(move || {
            // A receiver gone after the deadline makes this send fail; that
            // is the expected abandoned-solve case and is deliberately
            // ignored (capacity 1, so the send itself never blocks).
            let _ = tx.send(solve());
        })
        .map_err(|e| MipError::Solver(format!("spawning ay solve worker ({label}): {e}")))?;
    match rx.recv_timeout(slice) {
        Ok(result) => result.map(Some),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            tracing::warn!(
                "ay in-process solve ({label}) exceeded its {:.3}s slice; abandoning the \
                 worker thread and returning in-budget (the worker may keep running until \
                 ay's internal deadline or process teardown)",
                slice.as_secs_f64()
            );
            Ok(None)
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(MipError::Solver(format!(
            "ay solve worker ({label}) exited without a result (panicked)"
        ))),
    }
}

pub(crate) fn solve_opts(timeout_secs: f64) -> SolveOpts {
    // AY 6b9f4be8 repaired the market-split lattice detector's near-integer
    // normalization and interval enumeration. Keep the lane quarantined as
    // defense in depth, however: it can still return `Optimal { cert: None }`,
    // while NY consumes optimal values as rigorous bounds. Re-admit it only
    // after a separate proof review and discriminating A/B campaign. Reassert
    // on every solve so a caller cannot accidentally enable it between
    // detached worker sessions.
    ay_env::set("AY_MILP_NO_LATTICE", "1");

    // AY 2a1b5545 adds two default-on tall-LP throughput heuristics without
    // dedicated regression coverage: a flip-LNS saturation stop and a
    // bloom-cap relaxation. Both retained candidates and pruning bounds are
    // postchecked, but NY keeps new heuristic scheduling fail-closed until AY
    // has discriminating kill-switch/classification tests. The saturation
    // tuning values are parsed even when its kill switch is present, so pin
    // finite values as well; malformed inherited values must never panic an
    // in-process verifier worker.
    ay_env::set("AY_MILP_NO_SAT_STOP", "1");
    ay_env::set("AY_MILP_SAT_STOP_SECS", "15");
    ay_env::set("AY_MILP_SAT_STOP_MULT", "1.5");
    ay_env::set("AY_MILP_NO_BLOOM_RELAX", "1");

    // AY fb172576 adds an 18s absolute flip-LNS cap for every `tall_lu()`
    // model (currently just rows >= 1,000), a class that includes NY's long
    // w2/w5 ReLU windows. Preserve the prior fractional schedule until a
    // sealed A/B explicitly arms the provenance-recorded NY canary.
    configure_ay_flip_schedule();
    let clamped = timeout_secs.clamp(0.001, 86_400.0);
    // tree_cert_leaves: ay's default (256) silently poisons the tree-cert
    // capture on any B&B tree past ~256 leaves — the k=95 ACAS diff-leaf
    // DECIDED infeasible in 56s but returned (None, None) certs, which ny's
    // certified-only admission rightly refuses (measured 2026-07-18, ay
    // designs/2026-07-18-cert-emission-scoping.md). Raising the cap only
    // ever ADDS certified verdicts and stays fail-closed (finalize that
    // can't complete in budget → uncertified, exactly as before). MEASURED
    // LIMIT (same day): finalize throughput binds well before the cap —
    // k=63 (19 leaves) certifies in ~12.5s, but k=95 (~2^18-leaf tree)
    // consumed 600s of cert lane without completing (wall = cap+5s at every
    // cap tried). Until ay's finalize is faster at that scale (named lever:
    // the redundant cert double-verify in tree_cert.rs), the certified band
    // is k≈63-class trees regardless of this cap.
    SolveOpts::new()
        .with_time_limit(std::time::Duration::from_secs_f64(clamped))
        .with_tree_cert_leaves(65_536)
}

/// Auto-apply the measured cifar100-window solver recipe for window-class models.
///
/// Tuned on the two solved-and-verified cifar100 w5 windows (prop8945
/// r99-67 / r99-73: 26,831 cols, 18,692 rows, 53 ReLU binaries; ay repo,
/// designs/2026-07-12-gurobi-class-milp-for-ny.md G4 ledger): presolve capped
/// at 2% of the budget, root cuts off, the feasibility pump off (655–939 s per
/// attempt on the windows, never landed), and the dive's terminal salvage
/// fixed at 16 pins. `AY_MILP_DIVE_MAX_PINS=16` is instance-family-tuned; a
/// principled auto-cap inside ay is the recorded follow-up.
/// `AY_MILP_REFACTOR_EVERY` is deliberately not set — the eta-refactor cadence
/// belongs to ay's in-code defaults, not this recipe.
///
/// Env-var wiring because `SolveOpts` carries no engine-economics knobs at the
/// pinned rev. `set_var` is process-global and sticky after the first window
/// solve — acceptable: window solves are this lane's workload, and sub-gate
/// models (the whole 144-instance mip-diff corpus among them) never reach the
/// gate. A knob already present in the environment is never overwritten, so
/// user/session overrides always win. ADVICE-lane only: these steer search
/// economics (which incumbent the float lane finds, and how fast); every
/// verdict still rests on ay's exact certificates and `check_point`
/// (fail-closed), so the recipe can change speed, never an answer.
fn apply_window_recipe(problem: &MilpProblem) {
    if !is_window_class(problem) {
        return;
    }
    for (key, val) in [
        ("AY_MILP_PRESOLVE_SHARE", "0.02"),
        ("AY_MILP_NO_CUTS", "1"),
        ("AY_MILP_PUMP_RESTARTS", "0"),
        ("AY_MILP_DIVE_MAX_PINS", "16"),
    ] {
        ay_env::set_if_unset(key, val);
    }
}

/// The w5 windows are 18,692 rows; the corpus models are hundreds. Gate
/// well above the latter and comfortably below the former.
const WINDOW_ROWS_GATE: usize = 8192;

fn is_window_class(problem: &MilpProblem) -> bool {
    problem.rows().len() >= WINDOW_ROWS_GATE
}

/// Opt-in budget FLOOR for window-class solves: `NY_MIP_WINDOW_TIMEOUT_SECS`.
///
/// The solved w5 windows are measured at ~25 min–2 h wall (warm-salvage
/// dependent); every existing ny call site budgets seconds, so without an
/// explicit grant the window chain (root LP → dive → pinned salvage → exact
/// refine) can never run to an incumbent through ny's own path. When the env
/// is set AND the model is window-class (same gate as the recipe), the
/// effective budget becomes max(caller's, floor) — the complete-verifier
/// lane's legitimate way to grant window time. UNSET = byte-identical
/// behavior: a caller asking 20 s keeps getting 20 s; no silent contract
/// change. Never lowers a caller's larger budget, never touches sub-gate
/// models. Budget is pure economics — verdicts stay fail-closed either way
/// (a too-small budget yields Timeout/unknown, never a wrong answer).
fn window_budget_floor(problem: &MilpProblem, timeout_secs: f64) -> f64 {
    if !is_window_class(problem) {
        return timeout_secs;
    }
    match std::env::var("NY_MIP_WINDOW_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(floor) if floor.is_finite() && floor > timeout_secs => floor,
        _ => timeout_secs,
    }
}

/// Extract f64 values for the given IR columns from an exact witness.
/// Best-effort conversion by design: every Sat is revalidated downstream by
/// the concrete forward pass (same posture as the P0 text parser).
fn extract_cols(values: &[BigRational], cols: &[Col]) -> Vec<f64> {
    cols.iter()
        .map(|c| values.get(c.0).and_then(ToPrimitive::to_f64).unwrap_or(0.0))
        .collect()
}

/// IEEE-754 successor of a finite `f64` (toward `+inf`). Bit-manipulation so
/// it is correct at the MSRV (predating stable `f64::next_up`), mirroring
/// ny-core's f32 helpers.
fn next_up_f64(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1); // smallest positive subnormal
    }
    let bits = x.to_bits();
    f64::from_bits(if x > 0.0 { bits + 1 } else { bits - 1 })
}

/// IEEE-754 predecessor of a finite `f64` (toward `-inf`).
fn next_down_f64(x: f64) -> f64 {
    -next_up_f64(-x)
}

/// OUTWARD-rounded f64 of an exact rational dual bound.
///
/// For Minimize the dual is a LOWER bound on the optimum, so the f64 must not
/// exceed the exact rational (round toward `-inf`); for Maximize it is an
/// UPPER bound (round toward `+inf`). `to_f64` rounds to NEAREST, which can
/// land half an ulp on the over-claiming side; when the conversion is not
/// exactly the rational and landed on the wrong side, step ONE ulp outward.
/// Exactly-representable bounds pass through untouched (no gratuitous
/// loosening). Non-finite conversions yield `None` — an infinite bound
/// carries no pruning information.
fn rigorous_dual_bound_f64(bound: &BigRational, sense: ObjSense) -> Option<f64> {
    let f = bound.to_f64()?;
    if !f.is_finite() {
        return None;
    }
    // `from_float` on a finite f64 is exact, so the comparison below is exact.
    let back = BigRational::from_float(f)?;
    let safe = match sense {
        ObjSense::Minimize => {
            if back <= *bound {
                f
            } else {
                next_down_f64(f)
            }
        }
        ObjSense::Maximize => {
            if back >= *bound {
                f
            } else {
                next_up_f64(f)
            }
        }
    };
    safe.is_finite().then_some(safe)
}

/// Map an `ay_milp::Outcome` to a `MipResult`.
///
/// `objective` is the optimization target when this was an `optimize_col`
/// solve (`None` for feasibility checks): it selects the column whose model
/// value is reported as the incumbent objective for `Feasible`, and the
/// direction the rigorous dual bound is OUTWARD-rounded in.
fn map_outcome(
    outcome: Outcome,
    model: &Model,
    input_vars: &[Col],
    output_vars: &[Col],
    objective: Option<ObjectiveSpec>,
) -> MipResult {
    match outcome {
        Outcome::Optimal {
            value,
            model_values,
            cert,
        } => {
            if let Some(cert) = &cert {
                match cert.verify(model) {
                    Ok(()) => tracing::debug!("ay-milp optimality certificate verified"),
                    Err(e) => tracing::warn!(
                        "ay-milp optimality certificate FAILED verification: {e} \
                         (verdict stands on the exact solver; certificate discarded)"
                    ),
                }
            }
            // A proven optimum IS a rigorous dual bound on itself (the exact
            // solver's completed tree); surface it outward-rounded so callers
            // can prune/tighten on it uniformly with the Feasible arm.
            let dual_bound = objective.and_then(|o| rigorous_dual_bound_f64(&value, o.sense));
            MipResult::Sat {
                objective: value.to_f64().unwrap_or(f64::NAN),
                output_values: extract_cols(&model_values, output_vars),
                input_values: extract_cols(&model_values, input_vars),
                dual_bound,
            }
        }
        Outcome::Feasible {
            model_values,
            incumbent_only,
            dual_bound,
        } => {
            // Surface the exactly-certified incumbent (w2): its objective is
            // the objective column's value at the feasible point (the model
            // is always built with a single unit-coefficient objective column
            // and no offset), and the rigorous interrupted-tree dual bound is
            // OUTWARD-rounded. `Outcome::Feasible.dual_bound` is rigorous BY
            // CONTRACT when present (`None` whenever any part of the tree was
            // discarded without proof — ay contract property 3), so `Some`
            // here always means "safe to prune on". Sat stays a candidate:
            // callers revalidate the point downstream, unchanged.
            let objective_value = objective
                .map(|o| {
                    model_values
                        .get(o.col.0)
                        .and_then(ToPrimitive::to_f64)
                        .unwrap_or(f64::NAN)
                })
                .unwrap_or(0.0);
            if incumbent_only {
                tracing::debug!(
                    "ay-milp returned an incumbent-only feasible point \
                     (objective {objective_value}); a better point may exist"
                );
            }
            let dual = match (&dual_bound, objective) {
                (Some(b), Some(o)) => rigorous_dual_bound_f64(b, o.sense),
                _ => None,
            };
            MipResult::Sat {
                objective: objective_value,
                output_values: extract_cols(&model_values, output_vars),
                input_values: extract_cols(&model_values, input_vars),
                dual_bound: dual,
            }
        }
        Outcome::Infeasible { cert, tree_cert } => {
            // LG3: UNSAT evidence is verified at the admission seam. The
            // `certified` flag records that independent evidence checked out
            // here, closing the "trusted MIP solver" hole
            // (NyVerdictAdmission) for every certificate-carrying UNSAT.
            //
            // Two evidence lanes, root Farkas preferred:
            // - `cert`: the exact Farkas witness for relaxation-level (root-LP)
            //   infeasibility.
            // - `tree_cert` (P2 MilpInfeasibilityCertificate, ay be3bae2f): a
            //   branch skeleton whose splits cover by construction (integral
            //   column, integer cut ⇒ x≤cut ∨ x≥cut+1 covers ℤ) with an exact
            //   per-leaf Farkas re-derived in THIS caller's model frame. Its
            //   `verify(model)` re-checks coverage and every leaf in pure
            //   rational arithmetic — the same independent-evidence posture as
            //   the root Farkas, extended to case-split UNSAT.
            // A lane that verifies ⇒ certified; NO lane present ⇒ the verdict
            // stands on ay's exact solving alone (`certified: false`), which
            // ny's admission then refuses to mint.
            //
            // A witness that does NOT re-check is a different state entirely,
            // and never a bare UNSAT: ay's own session already re-validates
            // every witness it returns and withholds the verdict when one
            // fails (`UnknownReason::WitnessRejected`), so a rejection arriving
            // here means two exact checks disagree about the same model. That
            // is a hard failure of the certificate lane, reported as such — it
            // keeps the rejection distinguishable from absence and is never
            // shopped to the other lane for a second opinion.
            match (&cert, &tree_cert) {
                (Some(cert), _) => match cert.verify(model) {
                    Ok(()) => {
                        tracing::debug!("ay-milp Farkas certificate verified");
                        MipResult::Unsat { certified: true }
                    }
                    Err(e) => MipResult::Error(format!(
                        "ay-milp Farkas certificate FAILED verification: {e}"
                    )),
                },
                (None, Some(tree)) => match tree.verify(model) {
                    Ok(()) => {
                        tracing::debug!("ay-milp case-split tree certificate verified");
                        MipResult::Unsat { certified: true }
                    }
                    Err(e) => MipResult::Error(format!(
                        "ay-milp tree certificate FAILED verification: {e}"
                    )),
                },
                (None, None) => MipResult::Unsat { certified: false },
            }
        }
        // Sound degrade, same as the P0 lane's Unknown -> Timeout.
        Outcome::Unknown { reason } => {
            tracing::debug!("ay-milp answered unknown: {reason:?}");
            MipResult::Timeout
        }
        // A dual bound without a feasible point (deadline-obeying interrupted
        // tree, no incumbent): no verdict — degrade to Timeout like Unknown
        // (the caller keeps its original sound bound), instead of erroring.
        // Surfacing this bound for pruning needs a MipResult variant of its
        // own; deferred until a caller wants it.
        Outcome::Bound {
            dual_bound,
            rigorous,
        } => {
            tracing::debug!(
                "ay-milp answered with a bound only (dual_bound={dual_bound}, \
                 rigorous={rigorous}); no feasible point -> Timeout"
            );
            MipResult::Timeout
        }
        Outcome::Unbounded => MipResult::Error("objective unbounded".to_string()),
        other => MipResult::Error(format!("unexpected ay-milp outcome: {other:?}")),
    }
}

/// Feasibility check on the in-process backend.
///
/// Both `warm_start_cols` (the PGD counter-example candidate, seeded as the
/// session incumbent) and non-empty `branch_hints` (NY's ranked unstable-ReLU
/// binaries) are ADVICE consumed by the native branch-and-cut engine (P2/P3):
/// they steer incumbent discovery and branch order but never change a verdict
/// or a certificate. NY forwards an empty slice by default; only the exact
/// `NY_AY_BRANCH_HINTS=1` canary makes this API live.
pub(crate) fn check_feasibility(
    problem: &MilpProblem,
    timeout_secs: f64,
    input_vars: &[Col],
    output_vars: &[Col],
    warm_start_cols: Option<&[f64]>,
    branch_hints: &[Col],
) -> Result<MipResult> {
    apply_window_recipe(problem);
    let timeout_secs = window_budget_floor(problem, timeout_secs);
    // Owned copies for the detached worker (the whole solve — model lowering
    // included — runs inside the externally enforced slice; see
    // `run_with_hard_deadline`). The IR clone is trivial next to the solve.
    let problem = problem.clone();
    let input_vars = input_vars.to_vec();
    let output_vars = output_vars.to_vec();
    let warm_start_cols = warm_start_cols.map(<[f64]>::to_vec);
    let branch_hints = branch_hints.to_vec();
    let result = run_with_hard_deadline(timeout_secs, "check", move || {
        let model = to_ay_model(&problem)?;
        let mut session = BabSession::new(model, &solve_opts(timeout_secs))
            .map_err(|e| MipError::Solver(e.to_string()))?;
        if let Some(seed) = &warm_start_cols {
            session.seed_incumbent(seed);
        }
        if !branch_hints.is_empty() {
            let hint_cols: Vec<ay_milp::Col> = branch_hints
                .iter()
                .filter_map(|c| session.model().col_at(c.0))
                .collect();
            session.hint_branch_order(&hint_cols);
        }
        let outcome = session
            .check()
            .map_err(|e| MipError::Solver(e.to_string()))?;
        Ok(map_outcome(
            outcome,
            session.model(),
            &input_vars,
            &output_vars,
            None,
        ))
    })?;
    // Slice expired with the solve still running: Timeout, never a verdict.
    Ok(result.unwrap_or(MipResult::Timeout))
}

/// Optimize a single column on the in-process backend.
///
/// Unlike the P0 subprocess lane, Maximize needs no aux-column rewrite: the
/// upstream OMT wrong-optimum chain is fixed at the root (ay R1, commit
/// 6d2bc529) and the in-process lane reports exact optima directly.
pub(crate) fn optimize_col(
    problem: &MilpProblem,
    timeout_secs: f64,
    objective: ObjectiveSpec,
    input_vars: &[Col],
    output_vars: &[Col],
) -> Result<MipResult> {
    apply_window_recipe(problem);
    let timeout_secs = window_budget_floor(problem, timeout_secs);
    // Same externally enforced slice as `check_feasibility` — this seam feeds
    // the tighten lane, where one hung `session.check()` would equally blow
    // the whole instance budget.
    let problem = problem.clone();
    let input_vars = input_vars.to_vec();
    let output_vars = output_vars.to_vec();
    let result = run_with_hard_deadline(timeout_secs, "optimize", move || {
        let mut model = to_ay_model(&problem)?;
        let target = model
            .col_at(objective.col.0)
            .ok_or_else(|| MipError::Encoding(format!("objective column {}", objective.col.0)))?;
        let sense = match objective.sense {
            ObjSense::Minimize => AySense::Minimize,
            ObjSense::Maximize => AySense::Maximize,
        };
        model.set_objective(&[(target, 1.0)], sense);
        let mut session = BabSession::new(model, &solve_opts(timeout_secs))
            .map_err(|e| MipError::Solver(e.to_string()))?;
        let outcome = session
            .check()
            .map_err(|e| MipError::Solver(e.to_string()))?;
        Ok(map_outcome(
            outcome,
            session.model(),
            &input_vars,
            &output_vars,
            Some(objective),
        ))
    })?;
    Ok(result.unwrap_or(MipResult::Timeout))
}

#[cfg(test)]
mod outward_rounding_tests {
    use super::*;

    /// Exact rational n/d (built from exactly-representable f64s; the
    /// division is exact rational arithmetic).
    fn rat(n: f64, d: f64) -> BigRational {
        BigRational::from_float(n).unwrap() / BigRational::from_float(d).unwrap()
    }

    /// Exactly-representable bounds pass through untouched in both senses.
    #[test]
    fn exact_bound_round_trips() {
        let b = rat(5.0, 2.0); // 2.5, exact in f64
        assert_eq!(rigorous_dual_bound_f64(&b, ObjSense::Minimize), Some(2.5));
        assert_eq!(rigorous_dual_bound_f64(&b, ObjSense::Maximize), Some(2.5));
    }

    /// A non-representable rational must land on the SOUND side: `<=` the
    /// exact value for Minimize (lower bound), `>=` for Maximize (upper).
    #[test]
    fn inexact_bound_rounds_outward() {
        let b = rat(1.0, 3.0); // 0.333..., not representable
        let lo = rigorous_dual_bound_f64(&b, ObjSense::Minimize).unwrap();
        let hi = rigorous_dual_bound_f64(&b, ObjSense::Maximize).unwrap();
        let back_lo = BigRational::from_float(lo).unwrap();
        let back_hi = BigRational::from_float(hi).unwrap();
        assert!(
            back_lo <= b,
            "Minimize dual bound must not exceed the exact value"
        );
        assert!(
            back_hi >= b,
            "Maximize dual bound must not undercut the exact value"
        );
        // And the pair brackets the exact value tightly (within one ulp).
        assert!(hi - lo <= f64::EPSILON, "bracket should be ulp-tight");
    }

    /// Negative side (sign-symmetric ulp stepping).
    #[test]
    fn negative_inexact_bound_rounds_outward() {
        let b = rat(-1.0, 3.0);
        let lo = rigorous_dual_bound_f64(&b, ObjSense::Minimize).unwrap();
        let hi = rigorous_dual_bound_f64(&b, ObjSense::Maximize).unwrap();
        assert!(BigRational::from_float(lo).unwrap() <= b);
        assert!(BigRational::from_float(hi).unwrap() >= b);
    }

    /// A bound too large for f64 yields None (no false pruning information).
    #[test]
    fn overflowing_bound_is_none() {
        let big = BigRational::from_float(1e300).unwrap();
        let huge = &big * &big; // 1e600: overflows f64
        assert_eq!(rigorous_dual_bound_f64(&huge, ObjSense::Minimize), None);
        assert_eq!(rigorous_dual_bound_f64(&huge, ObjSense::Maximize), None);
    }
}

#[cfg(test)]
mod ay_safety_guard_tests {
    use super::*;

    #[test]
    fn tall_flip_cap_opt_in_is_exact_and_fail_closed() {
        assert!(!use_ay_tall_flip_cap(None));
        for malformed in ["", "0", "01", "1.0", "true", "yes", "1x"] {
            assert!(
                !use_ay_tall_flip_cap(Some(malformed)),
                "malformed value {malformed:?} must retain historical scheduling"
            );
        }
        assert!(use_ay_tall_flip_cap(Some("1")));
        assert!(use_ay_tall_flip_cap(Some("  1\n")));
    }

    #[test]
    fn tall_flip_cap_policy_defaults_old_and_exact_one_arms_native() {
        for case in ["default", "malformed", "canary"] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "ay_lib::ay_safety_guard_tests::tall_flip_cap_policy_child",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("NY_INTERNAL_AY_FLIP_POLICY_TEST", case)
                .env("AY_MILP_FLIP_CAP_SECS", "3")
                .env("AY_MILP_WARM_LU", "1");
            match case {
                "default" => {
                    command
                        .env_remove("NY_AY_MILP_TALL_FLIP_CAP")
                        .env_remove("AY_MILP_FLIP_SHARE");
                }
                "malformed" => {
                    command
                        .env("NY_AY_MILP_TALL_FLIP_CAP", "malformed")
                        .env_remove("AY_MILP_FLIP_SHARE");
                }
                "canary" => {
                    command
                        .env("NY_AY_MILP_TALL_FLIP_CAP", " 1 ")
                        .env("AY_MILP_FLIP_SHARE", "0.25")
                        .env("AY_MILP_FLIP_CAP_SECS", "900");
                }
                _ => unreachable!(),
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "isolated {case} policy probe failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn tall_flip_cap_policy_child() {
        let Ok(case) = std::env::var("NY_INTERNAL_AY_FLIP_POLICY_TEST") else {
            return;
        };
        configure_ay_flip_schedule();
        match case.as_str() {
            "default" | "malformed" => assert_eq!(
                std::env::var("AY_MILP_FLIP_SHARE").as_deref(),
                Ok(HISTORICAL_AY_FLIP_SHARE)
            ),
            "canary" => assert!(std::env::var_os("AY_MILP_FLIP_SHARE").is_none()),
            _ => panic!("unknown isolated policy case {case:?}"),
        }
        assert_eq!(
            std::env::var("AY_MILP_FLIP_CAP_SECS").as_deref(),
            Ok(NATIVE_AY_FLIP_CAP_SECS)
        );
        assert!(std::env::var_os("AY_MILP_WARM_LU").is_none());
    }

    /// Defense-in-depth regression for NY's lattice quarantine. AY 6b9f4be8
    /// repaired this near-integer detector case, but NY keeps the lane disabled
    /// until its certificate-free `Optimal` path is independently readmitted.
    #[test]
    fn near_integer_lattice_shortcut_is_disabled_before_optimization() {
        // Adversarial pre-state routed through the crate's blessed choke
        // point (clippy env wall); `solve_opts` must reassert every guard.
        ay_env::remove("AY_MILP_NO_LATTICE");
        ay_env::remove("AY_MILP_NO_SAT_STOP");
        ay_env::set("AY_MILP_SAT_STOP_SECS", "NaN");
        ay_env::set("AY_MILP_SAT_STOP_MULT", "NaN");
        ay_env::remove("AY_MILP_NO_BLOOM_RELAX");

        let mut problem = MilpProblem::new();
        let x0 = problem.add_integer_col(0.0, 0.0, 1.0);
        let x1 = problem.add_integer_col(0.0, 0.0, 1.0);
        let x2 = problem.add_integer_col(0.0, 0.0, 1.0);
        let slack = problem.add_col(0.0, 0.0, f64::INFINITY);
        problem.add_row(
            1.0,
            1.0,
            [(x0, 2.0), (x1, 3.0), (x2, 2f64.powi(-21)), (slack, 1.0)],
        );

        let result = optimize_col(
            &problem,
            10.0,
            ObjectiveSpec {
                col: slack,
                sense: ObjSense::Minimize,
            },
            &[],
            &[slack],
        )
        .unwrap();
        let MipResult::Sat { objective, .. } = result else {
            panic!("guarded exact model must solve, got {result:?}");
        };
        let expected = 1.0 - 2f64.powi(-21);
        assert_eq!(objective.to_bits(), expected.to_bits());
        assert_eq!(std::env::var("AY_MILP_NO_LATTICE").as_deref(), Ok("1"));
        assert_eq!(std::env::var("AY_MILP_NO_SAT_STOP").as_deref(), Ok("1"));
        assert_eq!(std::env::var("AY_MILP_SAT_STOP_SECS").as_deref(), Ok("15"));
        assert_eq!(std::env::var("AY_MILP_SAT_STOP_MULT").as_deref(), Ok("1.5"));
        assert_eq!(std::env::var("AY_MILP_NO_BLOOM_RELAX").as_deref(), Ok("1"));
    }
}

#[cfg(test)]
mod hard_deadline_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// A solve that outlives its slice is abandoned AT the slice boundary and
    /// surfaces as `None` (mapped to Timeout by every caller) — never as a
    /// hang and never as a verdict. This is the vnncomp fix's contract: the
    /// SMT-fallback overshoot (deadline checks too coarse for BigRational
    /// pivots) must not stall the caller past its MIP slice.
    #[test]
    fn slow_solve_returns_none_at_the_slice_boundary() {
        let started = Instant::now();
        let result: Result<Option<MipResult>> =
            run_with_hard_deadline(0.05, "test-slow", move || {
                // A worker that ignores every deadline for far longer than
                // the slice (stands in for the ay-dpll persistent loop).
                std::thread::sleep(Duration::from_secs(10));
                Ok(MipResult::Unsat { certified: false })
            });
        let elapsed = started.elapsed();
        assert!(
            matches!(result, Ok(None)),
            "expired slice must yield Ok(None), got {result:?}"
        );
        // Well before the worker's 10s — the caller is back in budget. The
        // generous margin keeps the assertion robust on loaded CI machines.
        assert!(
            elapsed < Duration::from_secs(5),
            "caller stalled {elapsed:?} past a 0.05s slice"
        );
    }

    /// An in-budget solve passes its result through untouched.
    #[test]
    fn fast_solve_passes_through() {
        let result = run_with_hard_deadline(30.0, "test-fast", || Ok(42usize));
        assert!(matches!(result, Ok(Some(42))));
    }

    /// A worker-side error is surfaced as an error, not a timeout.
    #[test]
    fn worker_error_passes_through() {
        let result: Result<Option<usize>> = run_with_hard_deadline(30.0, "test-err", || {
            Err(MipError::Solver("boom".to_string()))
        });
        assert!(matches!(result, Err(MipError::Solver(_))));
    }

    /// A panicking worker is reported as a solver error (disconnected channel),
    /// never silently swallowed and never a verdict.
    #[test]
    fn worker_panic_is_an_error() {
        let result: Result<Option<usize>> =
            run_with_hard_deadline(30.0, "test-panic", || panic!("worker died"));
        assert!(matches!(result, Err(MipError::Solver(_))));
    }
}

/// Which oriented side of a one-sided row a Farkas multiplier scales.
///
/// Mirrors `ay_milp::BoundSide` at the ny-mip API boundary so consumers do
/// not depend on ay-milp types directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowSide {
    /// The `>= lb` side.
    Lower,
    /// The `<= ub` side.
    Upper,
}

/// Prove an LP (continuous columns only) INFEASIBLE and export the VERIFIED
/// Farkas evidence as row-indexed multipliers.
///
/// Built for the relational formula-implication check (hardening the
/// relational `unsat` gate from shape matching to a semantic implication
/// proof): each implication
/// obligation `C ∧ N` is lowered to rows over free columns; ay's exact solver
/// answers `Infeasible { cert }`; the certificate is INDEPENDENTLY verified
/// against the model here (`cert.verify`, exact arithmetic); and the
/// multipliers are returned keyed by `(row_index, side, coefficient)` so the
/// caller can rebuild — and RE-CHECK with `ny_cert::check_farkas`, a separate
/// checker — the same combination over its own constraint representation.
///
/// FAIL-CLOSED `Ok(None)` whenever the evidence is not exportable in that
/// shape: not infeasible, no certificate, certificate fails verification, the
/// problem has integer columns, or any multiplier references a COLUMN bound
/// (callers must encode every fact as a row over free columns). `Err` only
/// for solver-level failures.
pub fn prove_infeasible_with_row_farkas(
    problem: &MilpProblem,
    timeout_secs: f64,
) -> Result<Option<Vec<(usize, RowSide, BigRational)>>> {
    if problem.cols().iter().any(|c| c.integer) {
        return Ok(None); // LP-only lane: fail closed on binaries
    }
    // Same externally enforced slice as the other `session.check()` seams; a
    // slice that expires mid-solve is the existing fail-closed decline.
    let problem = problem.clone();
    let result = run_with_hard_deadline(timeout_secs, "farkas", move || {
        let model = to_ay_model(&problem)?;
        let mut session = BabSession::new(model, &solve_opts(timeout_secs))
            .map_err(|e| MipError::Solver(e.to_string()))?;
        let outcome = session
            .check()
            .map_err(|e| MipError::Solver(e.to_string()))?;
        let Outcome::Infeasible {
            cert: Some(cert), ..
        } = outcome
        else {
            return Ok(None);
        };
        // Independent exact verification against the model (never trust the
        // solver's word for the evidence).
        if let Err(e) = cert.verify(session.model()) {
            tracing::warn!(
                "ay-milp Farkas certificate FAILED verification in the row-export lane: {e}; \
                 declining (fail-closed)"
            );
            return Ok(None);
        }
        let mut rows = Vec::with_capacity(cert.multipliers.len());
        for m in &cert.multipliers {
            match m.fact {
                ay_milp::FactRef::RowBound { row, side } => {
                    let side = match side {
                        ay_milp::BoundSide::Lower => RowSide::Lower,
                        ay_milp::BoundSide::Upper => RowSide::Upper,
                    };
                    rows.push((row.index(), side, m.coeff.clone()));
                }
                // Column-bound facts cannot arise for free columns; if one
                // does, the caller's row mapping would be wrong — fail closed.
                _ => return Ok(None),
            }
        }
        Ok(Some(rows))
    })?;
    Ok(result.flatten())
}
