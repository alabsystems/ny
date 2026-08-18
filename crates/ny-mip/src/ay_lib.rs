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

use ay_milp::{
    BabSession, EngineEconomics, Model, Outcome, Sense as AySense, SolveOpts, TargetFsbPrefixOpts,
    Trust,
};
use num_rational::BigRational;
use num_traits::ToPrimitive;

use crate::ay::{ObjSense, ObjectiveSpec};
use crate::error::MipError;
use crate::ir::{Col, MilpProblem};
use crate::solver::{MipResult, OneSidedSatDecline, OneSidedSatProbe, OneSidedSatWitness, Sense};

type Result<T> = std::result::Result<T, MipError>;

// NY'S ENGINE ECONOMICS TRAVEL ON THE SOLVE, NOT IN THE PROCESS ENVIRONMENT.
//
// AY M1 (`e1c347ec9`, in the pinned rev) put every knob NY pins onto a typed
// `EngineEconomics` carried by `SolveOpts::with_engine`, and resolves
// `caller (SolveOpts) > environment > policy > compiled default` (ay
// `tune.rs:794-802`). NY therefore configures AY by VALUE and writes no process
// environment at all.
//
// WHY THIS REPLACED THE OLD `set_var` CHOKE POINT — two reasons, one of them a
// live defect the pin bump introduced:
//
// 1. THE ENVIRONMENT IS NOW READ ONCE. The pinned rev added `ay-milp`'s
//    `tune.rs`, whose release-build env layer is a `OnceLock<EnvSnapshot>`
//    captured on the FIRST knob read (`tune.rs:700-706`), and
//    `tune::on(Knob::NoCuts)` fires in `add_root_cuts` on essentially every
//    MILP solve (`bab.rs:4181`). The old window recipe wrote its four knobs
//    LAZILY — only when a window-class model first arrived — so in any process
//    that solved a sub-gate model first (the 144-instance mip-diff corpus, a
//    Graph-MIP leaf, an ACAS diff-leaf), the snapshot froze those variables as
//    ABSENT and the measured cifar100 w5 recipe never reached AY again. Typed
//    options are resolved per solve from the caller layer and cannot be beaten
//    by ordering.
// 2. THE WRITES RACED REAL SOLVES. NY runs solves on detached worker threads
//    that keep running after abandonment (see `run_with_hard_deadline`), so a
//    live AY solve genuinely raced NY's next `set_var`. Serializing NY's own
//    writes behind a mutex never fixed that, because AY's reads were not behind
//    it.
//
// What changes is that the posture is now GUARANTEED rather than
// order-dependent, and an inherited `AY_MILP_*` can no longer beat what NY
// configured.
//
// These knobs are NOT verdict-neutral in the unqualified sense NY used to claim
// — see the correction on `window_recipe`, from AY `f36f19a5b`. They cannot
// reach the certified-UNSAT or revalidated-SAT directions, but on an INTEGRAL
// model a search-changing setting can move an `Optimal` value that no
// certificate refuses. `map_outcome` gates that surface on `Outcome::trust`.

/// Fraction of the remaining budget NY gives AY's flip-LNS incumbent walk.
const NY_FLIP_SHARE: f64 = 0.75;
/// NY's absolute cap on AY's flip-LNS window for `tall_lu` models.
const NY_FLIP_CAP: std::time::Duration = std::time::Duration::from_secs(18);

/// Choose whether NY admits AY's default-on tall-LP flip-LNS cap.
///
/// This is deliberately fail-closed: only an exact `1` after trimming enables
/// the new schedule. Missing, non-Unicode, and malformed values retain NY's
/// previously pinned 75% flip-LNS share.
fn use_ay_tall_flip_cap(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

/// Fold NY's flip-LNS scheduling policy onto the typed engine carrier.
///
/// AY treats an explicit flip-LNS share as an opt-out from its tall absolute
/// cap — `with_flip_lns_share` "also disables the absolute cap, restoring the
/// pure fractional schedule" (ay `opts.rs:224-227`), exactly the coupling the
/// environment variables always had. NY therefore pins the historical share by
/// default and LEAVES THE FIELD UNSET only for the provenance-captured canary.
/// Both arms pin the eta-lane cap to 18 s and explicitly disable AY's
/// independently tunable warm-LU lane, so an inherited setting cannot silently
/// turn the recorded canary into a custom or 12-second schedule.
///
/// This function MUTATES NOTHING. `NY_AY_MILP_TALL_FLIP_CAP` is still read here
/// (it is NY's own knob, not one of AY's), but it is now an ordinary per-solve
/// read: because the result lands on the solve's own options rather than on
/// process state, the old "startup-only, never toggle concurrently with MIP
/// solves" restriction no longer applies.
fn flip_schedule(engine: &EngineEconomics) -> EngineEconomics {
    let raw = std::env::var("NY_AY_MILP_TALL_FLIP_CAP").ok();
    let inherited_share = std::env::var_os("AY_MILP_FLIP_SHARE").is_some();
    flip_schedule_with(engine, raw.as_deref(), inherited_share)
}

/// [`flip_schedule`]'s policy as a pure function of its two environment
/// inputs, so every arm is testable without touching process state.
fn flip_schedule_with(
    engine: &EngineEconomics,
    tall_flip_cap_raw: Option<&str>,
    inherited_share: bool,
) -> EngineEconomics {
    let engine = (*engine).with_flip_lns_cap(NY_FLIP_CAP).with_warm_lu(false);
    if use_ay_tall_flip_cap(tall_flip_cap_raw) {
        // CANARY ARM: leave `flip_lns_share` unset so AY's own tall absolute
        // cap governs. Under the old env lane NY could REMOVE an inherited
        // `AY_MILP_FLIP_SHARE`; by value it can only decline to set one, and
        // AY's environment layer still sits below the caller layer — so an
        // inherited share would silently survive and mislabel the A/B.
        //
        // Fail closed on the MEASUREMENT: refuse to arm, and fall back to the
        // pinned default arm. Declining to run a mislabelled canary loses a
        // measurement arm, never a verdict (both arms are advice-lane
        // scheduling), and a mislabelled arm would corrupt the provenance
        // ledger `docs/SOLVER_POLICY.md` rests on.
        if !inherited_share {
            return engine;
        }
        tracing::warn!(
            "NY_AY_MILP_TALL_FLIP_CAP=1 NOT armed: an inherited AY_MILP_FLIP_SHARE would \
             outlive the typed carrier and mislabel the recorded A/B; using the pinned \
             {NY_FLIP_SHARE} share instead"
        );
    }
    engine
        .with_flip_lns_share(NY_FLIP_SHARE)
        .expect("NY flip-LNS share is a literal in [0, 1]")
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

    #[test]
    fn finite_support_aux_roundtrips_and_prices_a_snapped_dual_residual() {
        let mut problem = MilpProblem::new();
        let y = problem.add_col(0.0, -2.0, 3.0);
        let t = problem.add_col(0.0, -10.0, 10.0);
        problem.add_row(0.0, 0.0, [(t, 1.0), (y, -2.0)]);

        let text = crate::dump::to_milp_text(&problem);
        let roundtrip =
            crate::dump::from_milp_text(&text).expect("support-column dump must roundtrip");
        assert_eq!(roundtrip.cols()[t.0].lb.to_bits(), (-10.0_f64).to_bits());
        assert_eq!(roundtrip.cols()[t.0].ub.to_bits(), 10.0_f64.to_bits());

        let model = to_ay_model(&roundtrip).expect("AY must admit a bounded support auxiliary");
        let ay_t = model.col_at(t.0).expect("column order is preserved");
        let (lb, ub) = model.col_bounds(ay_t);
        assert_eq!(lb.to_bits(), (-10.0_f64).to_bits());
        assert_eq!(ub.to_bits(), 10.0_f64.to_bits());
        let ay_row = model.row_at(0).expect("support equality is preserved");
        let (coeffs, row_lb, row_ub) = model.row(ay_row);
        assert_eq!(coeffs, &[(y.0 as u32, -2.0), (t.0 as u32, 1.0)]);
        assert_eq!(row_lb.to_bits(), 0.0_f64.to_bits());
        assert_eq!(row_ub.to_bits(), 0.0_f64.to_bits());

        // Model the exact residual left when equality and sparse-row duals
        // independently snap one 2^-30 grid unit apart. The equality row,
        // t-upper bound, and y-lower bound combine as
        //
        //   eps(t - 2y) + eps(10 - t) + 2eps(y + 2) = 14eps.
        //
        // Thus even a deliberately non-cancelling t residual remains a valid,
        // independently checked weak bound. With a free t column, the required
        // upper-bound fact would be unavailable and this proof would decline.
        let eps = BigRational::new(1.into(), (1_i64 << 30).into());
        let certificate = ay_milp::OptimalityCertificate {
            sense: AySense::Minimize,
            objective: Vec::new(),
            bound: -(BigRational::from_integer(14.into()) * &eps),
            multipliers: vec![
                ay_milp::Multiplier {
                    fact: ay_milp::FactRef::RowBound {
                        row: ay_row,
                        side: ay_milp::BoundSide::Lower,
                    },
                    coeff: eps.clone(),
                },
                ay_milp::Multiplier {
                    fact: ay_milp::FactRef::ColBound {
                        col: ay_t,
                        side: ay_milp::BoundSide::Upper,
                    },
                    coeff: eps.clone(),
                },
                ay_milp::Multiplier {
                    fact: ay_milp::FactRef::ColBound {
                        col: model.col_at(y.0).expect("y column"),
                        side: ay_milp::BoundSide::Lower,
                    },
                    coeff: BigRational::from_integer(2.into()) * eps,
                },
            ],
        };
        certificate
            .verify(&model)
            .expect("finite t bounds must absorb an exact snapped-dual residual");
    }
}

/// Stack reservation for the detached solve worker. Matches ay's own
/// solver-thread headroom (`SMT_FILE_THREAD_STACK_SIZE`, ay
/// crates/ay/src/run.rs — the reason its CLI re-execs itself), so moving the
/// in-process solve off the caller's thread can never LOWER the stack the
/// engine had before this wrapper existed.
pub(crate) const SOLVE_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

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
fn run_with_hard_deadline_at_clock<T, F, N>(
    deadline: std::time::Instant,
    label: &'static str,
    solve: F,
    now_after_spawn: N,
) -> Result<Option<T>>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
    N: FnOnce() -> std::time::Instant,
{
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

    // Sample only AFTER thread/channel setup. A relative timeout computed
    // before `spawn` would grant coordinator latency as a fresh clock.
    let Some(remaining) = deadline
        .checked_duration_since(now_after_spawn())
        .filter(|remaining| !remaining.is_zero())
    else {
        tracing::warn!(
            "ay in-process solve ({label}) reached its absolute deadline during worker setup; \
             abandoning the worker thread"
        );
        return Ok(None);
    };
    match rx.recv_timeout(remaining) {
        Ok(result) if std::time::Instant::now() < deadline => result.map(Some),
        Ok(_) => {
            tracing::warn!(
                "ay in-process solve ({label}) completed after its absolute deadline; \
                 abandoning the late result"
            );
            Ok(None)
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            tracing::warn!(
                "ay in-process solve ({label}) reached its absolute deadline; abandoning the \
                 worker thread and returning in-budget (the worker may keep running until \
                 ay's internal deadline or process teardown)"
            );
            Ok(None)
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(MipError::Solver(format!(
            "ay solve worker ({label}) exited without a result (panicked)"
        ))),
    }
}

/// Run an in-process AY solve against one caller-owned absolute deadline.
///
/// Unlike [`run_with_hard_deadline`], this entry point does not create a
/// relative slice. Worker/channel setup is charged to `deadline`, which is
/// sampled only after the detached worker has been spawned.
pub(crate) fn run_with_hard_deadline_at<T, F>(
    deadline: std::time::Instant,
    label: &'static str,
    solve: F,
) -> Result<Option<T>>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    run_with_hard_deadline_at_clock(deadline, label, solve, std::time::Instant::now)
}

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
    // internal time limit describe the same slice. Converting it to an
    // absolute Instant before spawning charges worker setup too.
    let slice = std::time::Duration::from_secs_f64(hard_timeout_slice_secs(timeout_secs));
    let deadline = std::time::Instant::now()
        .checked_add(slice)
        .ok_or_else(|| MipError::Solver(format!("ay solve deadline overflow ({label})")))?;
    run_with_hard_deadline_at(deadline, label, solve)
}

/// NY's fail-closed engine posture, as a typed value.
///
/// Every pin here is advice-lane: AY classifies the whole carrier as unable to
/// make a value, bound, verdict, or certificate wrong (ay `opts.rs:100-107`).
/// What they buy NY is determinism and quarantine, not correctness — and now
/// they are honoured per solve instead of depending on write ordering against
/// AY's frozen environment snapshot.
fn ny_base_engine() -> EngineEconomics {
    EngineEconomics::new()
        // AY 6b9f4be8 repaired the market-split lattice detector's near-integer
        // normalization and interval enumeration. Keep the lane quarantined as
        // defense in depth, however: it can still return `Optimal { cert: None }`,
        // while NY consumes optimal values as rigorous bounds. Re-admit it only
        // after a separate proof review and discriminating A/B campaign. Carried
        // on every solve, so a caller cannot accidentally enable it between
        // detached worker sessions.
        .with_lattice(false)
        // AY 2a1b5545 adds two default-on tall-LP throughput heuristics without
        // dedicated regression coverage: a flip-LNS saturation stop and a
        // bloom-cap relaxation. Both retained candidates and pruning bounds are
        // postchecked, but NY keeps new heuristic scheduling fail-closed until AY
        // has discriminating kill-switch/classification tests. (The pinned rev
        // ships AY's M2 *class* declaration but not those tests, so the
        // quarantine stands.)
        .with_saturation_stop(false)
        .with_bloom_cap_relaxation(false)
        // The saturation tuning values are parsed even when the kill switch is
        // present, so pin them too. This is now belt-and-braces rather than a
        // defence against a live abort: AY's `tune::parse_real` REJECTS
        // non-finite / negative / >1e15 values and falls back to the compiled
        // default (`tune.rs:989-1006`, "rejection rather than clamping"), so a
        // malformed inherited value can no longer reach `Duration::mul_f64` and
        // abort an in-process verifier worker. Pinning by value also puts these
        // above AY's environment layer entirely.
        .with_saturation_stop_floor(std::time::Duration::from_secs(15))
        .with_saturation_stop_multiplier(1.5)
        .expect("NY saturation-stop multiplier is a finite literal in range")
}

pub(crate) fn solve_opts(timeout_secs: f64) -> SolveOpts {
    let clamped = hard_timeout_slice_secs(timeout_secs);
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
    // AY fb172576 adds an 18s absolute flip-LNS cap for every `tall_lu()`
    // model (currently just rows >= 1,000), a class that includes NY's long
    // w2/w5 ReLU windows. Preserve the prior fractional schedule until a
    // sealed A/B explicitly arms the provenance-recorded NY canary.
    SolveOpts::new()
        .with_time_limit(std::time::Duration::from_secs_f64(clamped))
        .with_tree_cert_leaves(65_536)
        .with_engine(flip_schedule(&ny_base_engine()))
}

/// Add the typed, per-session node-warm ceiling to the canonical NY options.
///
/// This constructor is used only by exact neural Graph-MIP feasibility
/// checks. Every other authority path continues to call [`solve_opts`] and
/// therefore remains uncapped.
fn graph_mip_solve_opts(
    timeout_secs: f64,
    node_warm_time_limit: Option<std::time::Duration>,
) -> SolveOpts {
    let opts = solve_opts(timeout_secs);
    match node_warm_time_limit {
        Some(limit) => opts.with_node_warm_time_limit(Some(limit)),
        None => opts,
    }
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
/// Carried on the solve's own options, scoped to window-class models. This is
/// the lane the pin bump silently broke: the recipe used to be written to the
/// environment on first window arrival, which AY's frozen snapshot no longer
/// observes (see the module note above). These steer search economics: which
/// incumbent the float lane finds, and how fast.
///
/// A CORRECTION NY MUST NOT LOSE (AY `f36f19a5b`). NY used to justify tolerating
/// these settings freely with "every verdict still rests on ay's exact
/// certificates and `check_point`, so the recipe can change speed, never an
/// answer." **That sentence is FALSE for the outcome shape NY's own workload
/// produces.** `LpSession::verify` asserts the certified dual bound MEETS the
/// primal only under `!model.has_integrality()`; on an INTEGRAL model — every NY
/// model with ReLU binaries — the certificate is checked for NON-CROSSING and
/// nothing more, and the integrality gap is closed by the search's
/// exhaustiveness, which is NOT certified. So a search-changing setting CAN move
/// an `Optimal` value here, with no certificate refusing it.
///
/// What actually keeps NY safe is narrower, and worth stating exactly:
/// - The UNSAT/Verified direction is unaffected: it admits only
///   `Unsat { certified: true }`, whose Farkas or tree certificate IS re-checked
///   against NY's own model at the seam.
/// - The SAT direction is unaffected: every Sat is revalidated downstream by the
///   concrete forward pass.
/// - The exposed surface was `Optimal` consumed as a prunable bound. That is now
///   gated on `Outcome::trust` in `map_outcome`, so a search-trusted optimum no
///   longer becomes a `dual_bound` a caller may prune on.
///
/// THE OVERRIDE CONTRACT IS PRESERVED BY READING, NEVER WRITING. The old
/// `set_if_unset` let a user/session `AY_MILP_*` win. The caller layer now
/// outranks the environment, so honouring an operator override means declining
/// to set the field and letting AY's own environment layer resolve it.
fn window_recipe(engine: &EngineEconomics, problem: &MilpProblem) -> EngineEconomics {
    window_recipe_with(engine, problem, WindowOverrides::from_env())
}

/// Which window knobs the operator has already pinned in the environment.
///
/// Reading these is what preserves the old `set_if_unset` contract: a knob the
/// operator set is one NY declines to set, leaving AY's environment layer to
/// resolve it.
#[derive(Clone, Copy, Default)]
struct WindowOverrides {
    presolve_share: bool,
    cuts: bool,
    pump_restarts: bool,
    dive_max_pins: bool,
    /// The `NY_AY_WINDOW_CUTS=1` measurement canary (see [`use_window_cuts`]).
    cuts_canary: bool,
}

impl WindowOverrides {
    fn from_env() -> Self {
        Self {
            presolve_share: std::env::var_os("AY_MILP_PRESOLVE_SHARE").is_some(),
            cuts: std::env::var_os("AY_MILP_NO_CUTS").is_some(),
            pump_restarts: std::env::var_os("AY_MILP_PUMP_RESTARTS").is_some(),
            dive_max_pins: std::env::var_os("AY_MILP_DIVE_MAX_PINS").is_some(),
            cuts_canary: use_window_cuts(std::env::var("NY_AY_WINDOW_CUTS").ok().as_deref()),
        }
    }
}

/// Opt-in canary that ADMITS AY's root cut separation on window-class models.
///
/// WHY A CANARY IS NEEDED AT ALL. AY's `AY_MILP_NO_CUTS` is PRESENCE-based
/// (`tune::on` = `var_os(..).is_some()`, ay `tune.rs:814-822`), so the
/// environment cannot express "cuts ON": presence means off, and absence means
/// NY pins off itself. Cuts were therefore unconditionally off for this class
/// both before and after the typed migration, and no A/B was runnable. This is
/// the only lever that arms the cuts-ON arm.
///
/// WHY IT IS WORTH RUNNING NOW. NY's window models are COSTLESS (the objective
/// is dropped when lowering), so the root cut loop's improvement is identically
/// zero and AY's OLD strict `>` adoption gate adopted no cut regardless of how
/// many were separated — `NO_CUTS=1` cost NY nothing. AY `70d0f38e3` ("price
/// the last cut round, adopt on an exact tie") changed that gate to `>=`, so a
/// zero-improvement cut can now be adopted, and the pin is a real choice for
/// the first time.
///
/// WHAT THIS CANARY CAN AND CANNOT CHANGE. Cuts tighten the relaxation and
/// steer search economics. They cannot touch the UNSAT/Verified direction (which
/// admits only `Unsat { certified: true }`, re-checked against NY's model at the
/// seam) or the SAT direction (revalidated downstream by the concrete forward
/// pass). They CAN, per AY `f36f19a5b`, move an `Optimal` value on an integral
/// model, because there the optimality certificate is checked for non-crossing
/// only — see `window_recipe` for the full correction. That surface is gated on
/// `Outcome::trust` in `map_outcome`, so this canary changes search economics and
/// nothing NY prunes on.
///
/// Fail-closed like NY's other canaries: only an exact `1` after trimming arms
/// it; missing, non-Unicode, and malformed values keep cuts off.
fn use_window_cuts(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

/// [`window_recipe`]'s policy as a pure function of the operator overrides.
fn window_recipe_with(
    engine: &EngineEconomics,
    problem: &MilpProblem,
    overrides: WindowOverrides,
) -> EngineEconomics {
    if !is_window_class(problem) {
        return *engine;
    }
    let mut engine = *engine;
    if !overrides.presolve_share {
        engine = engine
            .with_presolve_share(0.02)
            .expect("NY presolve share is a literal in [0, 1]");
    }
    // The canary is the only way to reach cuts-ON (see `use_window_cuts`), so
    // it outranks both NY's default pin and an operator's presence-based
    // `AY_MILP_NO_CUTS` — otherwise the arm it exists to run is unreachable.
    if overrides.cuts_canary {
        engine = engine.with_cuts(true);
    } else if !overrides.cuts {
        engine = engine.with_cuts(false);
    }
    if !overrides.pump_restarts {
        engine = engine.with_pump_restarts(0);
    }
    if !overrides.dive_max_pins {
        engine = engine.with_dive_max_pins(16);
    }
    engine
}

/// Overlay the window recipe onto an already-built options value.
///
/// Always starts from `opts.engine()` so the base guards ([`ny_base_engine`])
/// and the flip schedule can never be dropped by the overlay.
fn window_scoped_opts(opts: SolveOpts, problem: &MilpProblem) -> SolveOpts {
    let engine = window_recipe(&opts.engine(), problem);
    opts.with_engine(engine)
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
pub(crate) fn window_budget_floor_from_value(
    problem: &MilpProblem,
    timeout_secs: f64,
    value: Option<&str>,
) -> f64 {
    if !is_window_class(problem) {
        return timeout_secs;
    }
    match value.and_then(|v| v.parse::<f64>().ok()) {
        Some(floor) if floor.is_finite() && floor > timeout_secs => floor,
        _ => timeout_secs,
    }
}

pub(crate) fn window_budget_floor(problem: &MilpProblem, timeout_secs: f64) -> f64 {
    let value = std::env::var("NY_MIP_WINDOW_TIMEOUT_SECS").ok();
    window_budget_floor_from_value(problem, timeout_secs, value.as_deref())
}

/// Exact relative hard slice enforced by AY's detached wall-clock wrapper.
///
/// Keep callers that compare an armed schedule with historical behavior on
/// the same clamp as [`run_with_hard_deadline`].
pub(crate) fn hard_timeout_slice_secs(timeout_secs: f64) -> f64 {
    timeout_secs.clamp(0.001, 86_400.0)
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
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude > f64::INFINITY.to_bits() || bits == f64::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return f64::from_bits(1); // smallest positive subnormal
    }
    f64::from_bits(if bits & 0x8000_0000_0000_0000 == 0 {
        bits + 1
    } else {
        bits - 1
    })
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
fn map_outcome_with_sat_relu_certificate(
    outcome: Outcome,
    model: &Model,
    input_vars: &[Col],
    output_vars: &[Col],
    objective: Option<ObjectiveSpec>,
    sat_relu_certificate: Option<&ay_milp::SatReluInfeasibilityCertificate>,
    sat_relu_replay_deadline: Option<std::time::Instant>,
    strict_infeasibility_evidence: bool,
) -> MipResult {
    // Classify BEFORE the match consumes the outcome (AY `f36f19a5b`). This is
    // a method, not a field, so it must be read against the model NY handed AY.
    let outcome_trust = outcome.trust(model);
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
            // A proven optimum is a rigorous dual bound on itself ONLY when the
            // rim closed on it. AY `f36f19a5b` corrected the sentence NY had
            // been relying on: `LpSession::verify` asserts the certified dual
            // bound MEETS the primal only under `!model.has_integrality()`. On
            // an INTEGRAL model — which is every NY model carrying ReLU
            // binaries — the certificate is checked for NON-CROSSING and
            // nothing more; the integrality gap is closed by the search's
            // exhaustiveness, which is not certified. So a search-changing
            // setting CAN move an `Optimal` value on NY's own workload, and no
            // certificate refuses it.
            //
            // `Outcome::trust` makes that readable from the API instead of
            // inferable from AY's source. Gate the prunable bound on
            // `RimClosed`: `MipResult::Sat::dual_bound` promises callers a
            // bound they "may prune / tighten on directly" (solver.rs), and a
            // search-trusted number does not meet that promise. This costs NY
            // nothing today — no production caller reads this field (the only
            // binding site is `tests/ay_backend.rs`) — so it is pure hygiene
            // now and a closed hole later, rather than a bound a future caller
            // inherits believing it is rim-closed.
            //
            // The Sat VERDICT is unaffected and stays exactly as sound as
            // before: every Sat is revalidated downstream by the concrete
            // forward pass.
            let trust = outcome_trust;
            let dual_bound = match trust {
                Trust::RimClosed => {
                    objective.and_then(|o| rigorous_dual_bound_f64(&value, o.sense))
                }
                Trust::SearchTrusted { why } => {
                    tracing::debug!(
                        "ay-milp optimum is search-trusted, not rim-closed ({why}); \
                         withholding the prunable dual bound"
                    );
                    None
                }
            };
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
            // Four evidence lanes, root Farkas preferred:
            // - `cert`: the exact Farkas witness for relaxation-level (root-LP)
            //   infeasibility.
            // - `tree_cert` (P2 MilpInfeasibilityCertificate, ay be3bae2f): a
            //   branch skeleton whose splits cover by construction (integral
            //   column, integer cut ⇒ x≤cut ∨ x≥cut+1 covers ℤ) with an exact
            //   per-leaf Farkas re-derived in THIS caller's model frame. Its
            //   `verify(model)` re-checks coverage and every leaf in pure
            //   rational arithmetic — the same independent-evidence posture as
            //   the root Farkas, extended to case-split UNSAT.
            // - `sat_relu_certificate`: a session-side, model-bound CDCL/RUP
            //   refutation. It lives beside the outcome because it is neither
            //   a linear combination nor a B&B tree. NY reconstructs the exact
            //   SAT projection from THIS model and independently replays the
            //   proof under the caller's remaining absolute deadline.
            // - `strict_infeasibility_evidence`: the session was constructed
            //   with AY's `require_certificates` policy. Under that public
            //   contract, a bare `Outcome::Infeasible` can survive only when an
            //   exact-reduction side artifact (hybrid PB/LP, parity, PB-DP,
            //   network design, or a later typed lane) was independently
            //   rebuilt and replayed against THIS session's model before AY
            //   returned it. This closes the cross-repo API gap without
            //   duplicating every evolving side-certificate type in NY.
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
                // A strict session already rebuilt and replayed whichever
                // typed side artifact admitted this otherwise-bare outcome.
                // Take that public contract before attempting NY's historical
                // SAT/ReLU replay: AY may legitimately have spent nearly the
                // whole shared deadline producing the proof, and a redundant
                // second replay must not turn a proved result into a timeout.
                (None, None) if strict_infeasibility_evidence => {
                    tracing::debug!(
                        "ay-milp strict certificate policy admitted a replayed typed \
                         infeasibility side artifact"
                    );
                    MipResult::Unsat { certified: true }
                }
                (None, None) => match sat_relu_certificate {
                    Some(certificate) => {
                        match ay_milp::verify_sat_relu_infeasibility_certificate(
                            model,
                            certificate,
                            sat_relu_replay_deadline,
                        ) {
                            Ok(()) => {
                                tracing::debug!("ay-milp SAT/ReLU resolution certificate verified");
                                MipResult::Unsat { certified: true }
                            }
                            Err(error) => MipResult::Error(format!(
                                "ay-milp SAT/ReLU certificate FAILED verification: {error}"
                            )),
                        }
                    }
                    None => MipResult::Unsat { certified: false },
                },
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

/// Map an ordinary [`BabSession`] outcome together with its typed side evidence.
///
/// SAT/ReLU refutations deliberately do not masquerade as a root Farkas row or
/// a branch-and-bound tree, so their certificate lives on the session rather
/// than in [`Outcome::Infeasible`].  Read it before the session is dropped and
/// independently rebuild/replay it against the exact model NY handed to AY.
fn map_bab_outcome(
    outcome: Outcome,
    session: &BabSession,
    input_vars: &[Col],
    output_vars: &[Col],
    objective: Option<ObjectiveSpec>,
    sat_relu_replay_deadline: Option<std::time::Instant>,
    strict_infeasibility_evidence: bool,
) -> MipResult {
    map_outcome_with_sat_relu_certificate(
        outcome,
        session.model(),
        input_vars,
        output_vars,
        objective,
        session.sat_relu_infeasibility_certificate(),
        sat_relu_replay_deadline,
        strict_infeasibility_evidence,
    )
}

/// Map an outcome that has no session-side evidence carrier.
fn map_outcome(
    outcome: Outcome,
    model: &Model,
    input_vars: &[Col],
    output_vars: &[Col],
    objective: Option<ObjectiveSpec>,
) -> MipResult {
    map_outcome_with_sat_relu_certificate(
        outcome,
        model,
        input_vars,
        output_vars,
        objective,
        None,
        None,
        false,
    )
}

#[cfg(test)]
mod sat_relu_certificate_tests {
    use super::*;

    const EPSILON: f64 = 1.0 / 1_048_576.0;

    /// Build the exact SAT-to-Big-M-ReLU layout recognized by AY's typed
    /// SAT/ReLU route. The two unit clauses `x` and `not x` make it UNSAT while
    /// leaving generic branch-and-bound as a possible fallback if the route is
    /// accidentally disabled.
    fn contradictory_sat_relu_problem(epsilon: f64) -> (MilpProblem, Vec<Col>) {
        let variable_count = 1usize;
        let clauses = [vec![(0usize, true)], vec![(0usize, false)]];
        let clause_count = clauses.len();
        let relu_count = clause_count + 2 * variable_count;

        let mut definitions: Vec<(Vec<(usize, f64)>, f64)> = Vec::with_capacity(relu_count);
        for clause in &clauses {
            let negative_count = clause.iter().filter(|(_, positive)| !positive).count();
            definitions.push((
                clause
                    .iter()
                    .map(|&(variable, positive)| (variable, if positive { -1.0 } else { 1.0 }))
                    .collect(),
                negative_count as f64 - 1.0,
            ));
        }
        definitions.push((vec![(0, 1.0)], 0.0));
        definitions.push((vec![(0, 2.0)], 1.0));

        let input_lb = -epsilon;
        let input_ub = 1.0 + epsilon;
        let interval = |terms: &[(usize, f64)], rhs: f64| {
            let mut lower = -rhs;
            let mut upper = -rhs;
            for &(_, coefficient) in terms {
                if coefficient > 0.0 {
                    lower += coefficient * input_lb;
                    upper += coefficient * input_ub;
                } else {
                    lower += coefficient * input_ub;
                    upper += coefficient * input_lb;
                }
            }
            (lower, upper)
        };

        let mut problem = MilpProblem::new();
        let inputs: Vec<Col> = (0..variable_count)
            .map(|_| problem.add_col(0.0, input_lb, input_ub))
            .collect();
        let preactivations: Vec<Col> = (0..relu_count)
            .map(|_| problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY))
            .collect();
        let mut relus = Vec::with_capacity(relu_count);
        let mut phases = Vec::with_capacity(relu_count);
        let mut intervals = Vec::with_capacity(relu_count);
        for (terms, rhs) in &definitions {
            let (lower, upper) = interval(terms, *rhs);
            assert!(lower < 0.0 && upper > 0.0);
            relus.push(problem.add_col(0.0, 0.0, upper));
            phases.push(problem.add_integer_col(0.0, 0.0, 1.0));
            intervals.push((lower, upper));
        }
        let output0 = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        let output1 = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);

        for (index, (terms, rhs)) in definitions.iter().enumerate() {
            let mut row: Vec<(Col, f64)> = terms
                .iter()
                .map(|&(input, coefficient)| (inputs[input], coefficient))
                .collect();
            row.push((preactivations[index], -1.0));
            problem.add_row(*rhs, *rhs, row);
        }
        for index in 0..relu_count {
            let (lower, upper) = intervals[index];
            let big_m = -lower;
            problem.add_row(
                0.0,
                f64::INFINITY,
                [(relus[index], 1.0), (preactivations[index], -1.0)],
            );
            problem.add_row(
                f64::NEG_INFINITY,
                big_m,
                [
                    (relus[index], 1.0),
                    (preactivations[index], -1.0),
                    (phases[index], big_m),
                ],
            );
            problem.add_row(
                f64::NEG_INFINITY,
                0.0,
                [(relus[index], 1.0), (phases[index], -upper)],
            );
        }

        let mut clause_output: Vec<(Col, f64)> = relus[..clause_count]
            .iter()
            .map(|&relu| (relu, -1.0))
            .collect();
        clause_output.push((output0, -1.0));
        problem.add_row(-1.0, -1.0, clause_output);

        let mut boolean_output: Vec<(Col, f64)> = relus
            [clause_count..clause_count + variable_count]
            .iter()
            .map(|&relu| (relu, 1.0))
            .collect();
        boolean_output.extend(
            relus[clause_count + variable_count..]
                .iter()
                .map(|&relu| (relu, -1.0)),
        );
        boolean_output.push((output1, -1.0));
        problem.add_row(0.0, 0.0, boolean_output);
        problem.add_row(1.0, f64::INFINITY, [(output0, 1.0)]);
        problem.add_row(f64::NEG_INFINITY, 0.0, [(output1, 1.0)]);
        (problem, phases)
    }

    fn solved_sat_relu_session_with_policy(
        require_certificates: bool,
    ) -> (BabSession, Outcome, Option<std::time::Instant>) {
        let (problem, _) = contradictory_sat_relu_problem(EPSILON);
        let model = to_ay_model(&problem).expect("SAT/ReLU test model lowers");
        let opts = SolveOpts::new()
            .with_time_limit(std::time::Duration::from_secs(5))
            .with_require_certificates(require_certificates);
        let replay_deadline = opts.effective_deadline(std::time::Instant::now());
        let mut session = BabSession::new(model, &opts).expect("SAT/ReLU test session");
        let outcome = session.check().expect("SAT/ReLU route decides");
        assert!(
            matches!(
                &outcome,
                Outcome::Infeasible {
                    cert: None,
                    tree_cert: None
                }
            ),
            "typed SAT/ReLU proof must use the session sidecar: {outcome:?}"
        );
        assert!(
            session.sat_relu_infeasibility_certificate().is_some(),
            "recognized contradiction must retain its typed proof"
        );
        (session, outcome, replay_deadline)
    }

    fn solved_sat_relu_session() -> (BabSession, Outcome, Option<std::time::Instant>) {
        solved_sat_relu_session_with_policy(false)
    }

    #[test]
    fn sat_relu_session_sidecar_replay_certifies_infeasibility() {
        let (session, outcome, replay_deadline) = solved_sat_relu_session();
        let result = map_bab_outcome(outcome, &session, &[], &[], None, replay_deadline, false);
        assert!(
            matches!(result, MipResult::Unsat { certified: true }),
            "an independently replayed SAT/ReLU proof must carry authority: {result:?}"
        );
    }

    #[test]
    fn ordinary_check_feasibility_consumes_the_sat_relu_sidecar() {
        let (problem, _) = contradictory_sat_relu_problem(EPSILON);
        let all_columns: Vec<Col> = (0..problem.num_cols()).map(Col).collect();
        let result = check_feasibility(
            &problem,
            5.0,
            None,
            None,
            &all_columns,
            &all_columns,
            None,
            &[],
        )
        .expect("ordinary AY check");
        assert!(
            matches!(result, MipResult::Unsat { certified: true }),
            "the production session mapping must consume and replay the sidecar: {result:?}"
        );
    }

    #[test]
    fn bare_infeasibility_needs_the_strict_session_contract() {
        let model = Model::new();
        let bare = || Outcome::Infeasible {
            cert: None,
            tree_cert: None,
        };
        let historical = map_outcome_with_sat_relu_certificate(
            bare(),
            &model,
            &[],
            &[],
            None,
            None,
            None,
            false,
        );
        assert!(matches!(historical, MipResult::Unsat { certified: false }));

        // This boolean is valid only for an outcome returned by a session
        // whose SolveOpts had `require_certificates=true`. AY's public policy
        // degrades the same bare outcome to Unknown unless a typed side proof
        // was independently replayed before return.
        let strict =
            map_outcome_with_sat_relu_certificate(bare(), &model, &[], &[], None, None, None, true);
        assert!(matches!(strict, MipResult::Unsat { certified: true }));
    }

    #[test]
    fn sat_relu_certificate_from_a_mutated_model_fails_closed() {
        let (session, _, replay_deadline) = solved_sat_relu_session();
        let certificate = session
            .sat_relu_infeasibility_certificate()
            .expect("typed proof")
            .clone();
        let (mutated_problem, _) = contradictory_sat_relu_problem(2.0 * EPSILON);
        let mutated_model = to_ay_model(&mutated_problem).expect("mutated model still lowers");
        assert!(matches!(
            ay_milp::verify_sat_relu_infeasibility_certificate(
                &mutated_model,
                &certificate,
                replay_deadline,
            ),
            Err(ay_milp::SatReluInfeasibilityVerificationError::ModelDigestMismatch)
        ));

        let result = map_outcome_with_sat_relu_certificate(
            Outcome::Infeasible {
                cert: None,
                tree_cert: None,
            },
            &mutated_model,
            &[],
            &[],
            None,
            Some(&certificate),
            replay_deadline,
            false,
        );
        let MipResult::Error(error) = result else {
            panic!("a proof forged from another model must fail closed: {result:?}");
        };
        assert!(
            error.contains("model digest differs"),
            "the fail-closed result must preserve the replay rejection: {error}"
        );
    }

    #[test]
    fn sat_relu_certificate_replay_deadline_fails_closed() {
        let (session, _, _) = solved_sat_relu_session();
        let result = map_outcome_with_sat_relu_certificate(
            Outcome::Infeasible {
                cert: None,
                tree_cert: None,
            },
            session.model(),
            &[],
            &[],
            None,
            session.sat_relu_infeasibility_certificate(),
            std::time::Instant::now().checked_sub(std::time::Duration::from_millis(1)),
            false,
        );
        let MipResult::Error(error) = result else {
            panic!("an out-of-budget replay must fail closed: {result:?}");
        };
        assert!(
            error.contains("resource envelope"),
            "the fail-closed result must preserve the deadline rejection: {error}"
        );
    }

    #[test]
    fn strict_session_does_not_replay_typed_sidecar_after_shared_deadline() {
        let (session, _, _) = solved_sat_relu_session_with_policy(true);
        let result = map_outcome_with_sat_relu_certificate(
            Outcome::Infeasible {
                cert: None,
                tree_cert: None,
            },
            session.model(),
            &[],
            &[],
            None,
            session.sat_relu_infeasibility_certificate(),
            std::time::Instant::now().checked_sub(std::time::Duration::from_millis(1)),
            true,
        );
        assert!(
            matches!(result, MipResult::Unsat { certified: true }),
            "AY's strict policy already replayed the side artifact inside the shared deadline: \
             {result:?}"
        );
    }
}

/// Map AY's shared-prefix result with a deliberately narrower UNSAT authority
/// rule than the ordinary backend seam.
///
/// A root Farkas certificate is useful diagnostic evidence, but this canary is
/// qualifying the one-root *whole prefix tree*.  It therefore grants
/// `certified: true` only when AY exported a whole-tree certificate and NY
/// independently replayed that tree against the exact lowered caller model.
/// Bare/root-only infeasibility stays an uncertified decline.
fn map_shared_binary_prefix_outcome(
    outcome: Outcome,
    model: &Model,
    input_vars: &[Col],
    output_vars: &[Col],
    mode: SharedBinaryPrefixMode,
) -> MipResult {
    match outcome {
        Outcome::Infeasible {
            tree_cert: Some(tree),
            ..
        } => match tree.verify(model) {
            Ok(()) => {
                tracing::debug!(
                    "AY shared-prefix whole-tree certificate passed NY independent replay"
                );
                MipResult::Unsat { certified: true }
            }
            Err(error) => MipResult::Error(format!(
                "AY shared-prefix whole-tree certificate FAILED NY replay: {error}"
            )),
        },
        Outcome::Infeasible { .. } => {
            tracing::debug!(
                "AY shared-prefix infeasibility had no replayed whole-tree certificate; \
                 retaining verdict_authority=false"
            );
            MipResult::Unsat { certified: false }
        }
        // The explicit marked API maps every decided optimization result back
        // into the ORIGINAL feasibility frame before returning. A bare
        // optimization result here would therefore be an API-contract break,
        // not authority to interpret the margin value inside NY.
        Outcome::Optimal { .. } if mode == SharedBinaryPrefixMode::MarkedMargin => {
            MipResult::Error(
                "AY marked-margin shared-prefix API returned a bare optimal outcome".to_string(),
            )
        }
        // A rigorous margin bound is only a trigger for AY's original-frame
        // whole-tree replay. It carries no verdict authority by itself.
        Outcome::Bound { .. } if mode == SharedBinaryPrefixMode::MarkedMargin => MipResult::Timeout,
        other => map_outcome(other, model, input_vars, output_vars, None),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SharedBinaryPrefixMode {
    Plain,
    MarkedMargin,
}

impl SharedBinaryPrefixMode {
    fn canary_name(self) -> &'static str {
        match self {
            Self::Plain => "safenlp-shared-binary-prefix",
            Self::MarkedMargin => "safenlp-marked-margin-shared-binary-prefix",
        }
    }

    fn hard_deadline_label(self) -> &'static str {
        match self {
            Self::Plain => "shared-prefix",
            Self::MarkedMargin => "marked-margin-shared-prefix",
        }
    }
}

/// Read AY's optional process-global bounded-LP profiler without enabling it.
///
/// The pinned typed API does not expose per-session LP counts.  When an
/// operator has independently enabled AY's diagnostic profiler,
/// `SBPROFILE sb_solves=N ...` lets this canary report a best-effort
/// process-global snapshot delta.  Concurrent or detached AY work can
/// contaminate that delta, so it is never labelled exact-per-call.  Missing
/// telemetry remains `None` and can never affect a result.
fn profiled_bounded_lp_calls() -> Option<u64> {
    ay_milp::sb_profile_line()
        .split_whitespace()
        .find_map(|field| field.strip_prefix("sb_solves=")?.parse().ok())
}

fn profiled_bounded_lp_call_delta(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    match (before, after) {
        (Some(before), Some(after)) => Some(after.saturating_sub(before)),
        (None, Some(after)) => Some(after),
        (_, None) => None,
    }
}

fn shared_prefix_outcome_tag(result: &MipResult) -> &'static str {
    match result {
        MipResult::Sat { .. } => "sat-candidate",
        MipResult::Unsat { certified: true } => "unsat-tree-replay-verified",
        MipResult::Unsat { certified: false } => "unsat-no-tree-authority",
        MipResult::Timeout => "timeout",
        MipResult::Error(_) => "error",
    }
}

fn shared_prefix_phase_telemetry_gate_on(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Resolved per call through the ny-levers chokepoint's raw view (lever-debt
/// batch B1 preparation). The gate is consulted once per shared-prefix session
/// start/completion, so a per-call env read costs nothing; the unit-tested pure
/// predicate above stays the arming rule. This remains live process state until
/// Phase 2 injects a per-run `LeverSet`.
fn shared_prefix_phase_telemetry_enabled() -> bool {
    shared_prefix_phase_telemetry_gate_on(
        ny_levers::read_raw(&ny_levers::decls::telemetry::PHASE_TELEMETRY).as_deref(),
    )
}

fn shared_prefix_session_start_marker_if(enabled: bool, regions_total: usize) -> Option<String> {
    if !enabled {
        return None;
    }
    Some(format!(
        "NY_MIP_SAFENLP_SHARED_PREFIX_V1 event=session-start regions_total={regions_total}"
    ))
}

fn shared_prefix_session_completion_marker_if(
    enabled: bool,
    regions_total: usize,
    admitted_result: Option<&MipResult>,
) -> Option<String> {
    if !enabled {
        return None;
    }
    let result = admitted_result?;
    let outcome = shared_prefix_outcome_tag(result);
    let verdict_authority = matches!(result, MipResult::Unsat { certified: true });
    Some(format!(
        "NY_MIP_SAFENLP_SHARED_PREFIX_V1 event=session-complete \
         regions_total={regions_total} outcome={outcome} verdict_authority={verdict_authority}"
    ))
}

fn marked_margin_shared_prefix_session_start_marker_if(
    enabled: bool,
    regions_total: usize,
    margin_row: usize,
    target_fsb_prefix: bool,
) -> Option<String> {
    enabled.then(|| {
        format!(
            "NY_MIP_SAFENLP_MARKED_MARGIN_PREFIX_V1 event=session-start \
             regions_total={regions_total} margin_row={margin_row} \
             target_fsb_prefix={target_fsb_prefix}"
        )
    })
}

fn marked_margin_shared_prefix_session_completion_marker_if(
    enabled: bool,
    regions_total: usize,
    margin_row: usize,
    target_fsb_prefix: bool,
    admitted_result: Option<&MipResult>,
) -> Option<String> {
    if !enabled {
        return None;
    }
    let result = admitted_result?;
    let outcome = shared_prefix_outcome_tag(result);
    let verdict_authority = matches!(result, MipResult::Unsat { certified: true });
    Some(format!(
        "NY_MIP_SAFENLP_MARKED_MARGIN_PREFIX_V1 event=session-complete \
         regions_total={regions_total} margin_row={margin_row} \
         target_fsb_prefix={target_fsb_prefix} outcome={outcome} \
         original_tree_replay_authority={verdict_authority}"
    ))
}

/// Check plain feasibility through one AY shared binary-prefix session.
///
/// This is the default-dark SafeNLP canary's backend seam.  One admitted
/// invocation lowers one model, creates one [`BabSession`], and seats all
/// `2^k` regions in that session instead of cloning and re-lowering one model
/// per assignment.  The pinned AY revision disables symmetry before a serial
/// nonempty-prefix root, so certificate harvesting cannot hide a second root
/// preparation inside this invocation.
///
/// The caller must provide one through four distinct live `[0, 1]` binaries
/// and an absolute deadline.  The same deadline covers lowering, root work,
/// every prefix region, tree finalization, and NY's detached hard-stop
/// wrapper.  Because the prefix passed to AY is nonempty, pinned AY
/// mechanically skips both its margin reframe and its exact SMT fallback.
///
/// Once this function starts a shared session, errors and incomplete results
/// stay fail-closed; it never starts a second historical session.  Admission
/// declines are handled before this call by [`crate::solver::MipSolver`].
/// This local guarantee is not a whole-verifier session-count claim: an
/// objective-first probe can precede this seam, and a SAT candidate rejected
/// by concrete replay can cause later violation-slack solves.
pub(crate) fn check_feasibility_shared_binary_prefix(
    problem: &MilpProblem,
    timeout_secs: f64,
    hard_deadline: Option<std::time::Instant>,
    node_warm_time_limit: Option<std::time::Duration>,
    input_vars: &[Col],
    output_vars: &[Col],
    split_cols: &[Col],
    branch_hints: &[Col],
) -> Result<MipResult> {
    check_feasibility_shared_binary_prefix_inner(
        problem,
        timeout_secs,
        hard_deadline,
        node_warm_time_limit,
        input_vars,
        output_vars,
        split_cols,
        None,
        branch_hints,
        SharedBinaryPrefixMode::Plain,
    )
}

/// Check one explicitly marked decision margin through one AY shared-prefix
/// optimization session.
///
/// AY may use a strict exact interrupted-tree bound only to trigger replay of
/// every open region against the original feasibility model. NY independently
/// replays the returned whole tree against the same lowered original model
/// before granting UNSAT authority. A bound, optimum without that mapped tree,
/// incomplete result, or backend error never launches a second session.
pub(crate) fn check_feasibility_marked_margin_shared_binary_prefix(
    problem: &MilpProblem,
    timeout_secs: f64,
    hard_deadline: Option<std::time::Instant>,
    node_warm_time_limit: Option<std::time::Duration>,
    input_vars: &[Col],
    output_vars: &[Col],
    split_cols: &[Col],
    branch_hints: &[Col],
) -> Result<MipResult> {
    check_feasibility_shared_binary_prefix_inner(
        problem,
        timeout_secs,
        hard_deadline,
        node_warm_time_limit,
        input_vars,
        output_vars,
        split_cols,
        None,
        branch_hints,
        SharedBinaryPrefixMode::MarkedMargin,
    )
}

/// Check one explicitly marked decision margin with bounded objective-aware
/// prefix selection inside the same AY session.
///
/// `fallback_prefix` is the exact existing four-column split plan. AY may
/// replace it only after a complete target-FSB scan of `candidates`; every
/// selector decline executes the whole fallback unchanged in this same
/// [`BabSession`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_feasibility_marked_margin_target_fsb_shared_binary_prefix(
    problem: &MilpProblem,
    timeout_secs: f64,
    hard_deadline: Option<std::time::Instant>,
    node_warm_time_limit: Option<std::time::Duration>,
    input_vars: &[Col],
    output_vars: &[Col],
    fallback_prefix: &[Col; 4],
    candidates: &[Col],
    branch_hints: &[Col],
) -> Result<MipResult> {
    check_feasibility_shared_binary_prefix_inner(
        problem,
        timeout_secs,
        hard_deadline,
        node_warm_time_limit,
        input_vars,
        output_vars,
        fallback_prefix,
        Some(candidates),
        branch_hints,
        SharedBinaryPrefixMode::MarkedMargin,
    )
}

#[allow(clippy::too_many_arguments)]
fn check_feasibility_shared_binary_prefix_inner(
    problem: &MilpProblem,
    timeout_secs: f64,
    hard_deadline: Option<std::time::Instant>,
    node_warm_time_limit: Option<std::time::Duration>,
    input_vars: &[Col],
    output_vars: &[Col],
    split_cols: &[Col],
    target_fsb_candidates: Option<&[Col]>,
    branch_hints: &[Col],
    mode: SharedBinaryPrefixMode,
) -> Result<MipResult> {
    let margin_row = match (mode, problem.margin_row()) {
        (SharedBinaryPrefixMode::Plain, None) => None,
        (SharedBinaryPrefixMode::Plain, Some(_)) => {
            return Err(MipError::Solver(
                "plain SafeNLP shared-prefix entry rejects a marked margin row".to_string(),
            ));
        }
        (SharedBinaryPrefixMode::MarkedMargin, Some(row)) => Some(row.0),
        (SharedBinaryPrefixMode::MarkedMargin, None) => {
            return Err(MipError::Solver(
                "marked-margin SafeNLP shared-prefix entry requires a marked row".to_string(),
            ));
        }
    };
    if mode == SharedBinaryPrefixMode::MarkedMargin {
        if std::env::var_os("AY_MILP_NO_MARGIN_REFRAME").is_some() {
            return Err(MipError::Solver(
                "required marked-margin shared prefix is disabled by \
                 AY_MILP_NO_MARGIN_REFRAME"
                    .to_string(),
            ));
        }
        if std::env::var_os("AY_MILP_SMT").is_some() {
            return Err(MipError::Solver(
                "required marked-margin shared prefix rejects forced non-native AY_MILP_SMT"
                    .to_string(),
            ));
        }
    }
    let caller_deadline = hard_deadline.ok_or_else(|| {
        MipError::Solver(
            "SafeNLP shared-prefix canary requires an absolute AY deadline".to_string(),
        )
    })?;
    if !(1..=4).contains(&split_cols.len()) {
        return Err(MipError::Solver(format!(
            "SafeNLP shared-prefix canary needs 1..=4 split columns, got {}",
            split_cols.len()
        )));
    }
    let mut seen = vec![false; problem.num_cols()];
    for &col in split_cols {
        let Some(spec) = problem.cols().get(col.0) else {
            return Err(MipError::Encoding(format!(
                "shared-prefix column {} is out of range",
                col.0
            )));
        };
        if !spec.integer
            || spec.lb.to_bits() != 0.0_f64.to_bits()
            || spec.ub.to_bits() != 1.0_f64.to_bits()
            || std::mem::replace(&mut seen[col.0], true)
        {
            return Err(MipError::Encoding(format!(
                "shared-prefix column {} is not a distinct live [0, 1] binary",
                col.0
            )));
        }
    }

    let target_fsb_prefix = target_fsb_candidates.is_some();
    if let Some(candidates) = target_fsb_candidates {
        if mode != SharedBinaryPrefixMode::MarkedMargin {
            return Err(MipError::Encoding(
                "target-FSB prefix selection requires the marked-margin entry".to_string(),
            ));
        }
        if split_cols.len() != 4 {
            return Err(MipError::Encoding(format!(
                "target-FSB prefix selection needs an exact four-column fallback, got {}",
                split_cols.len()
            )));
        }
        if !(4..=8).contains(&candidates.len()) {
            return Err(MipError::Encoding(format!(
                "target-FSB prefix selection needs 4..=8 candidates, got {}",
                candidates.len()
            )));
        }
        let mut candidate_seen = vec![false; problem.num_cols()];
        for &col in candidates {
            let Some(spec) = problem.cols().get(col.0) else {
                return Err(MipError::Encoding(format!(
                    "target-FSB candidate column {} is out of range",
                    col.0
                )));
            };
            if !spec.integer
                || spec.lb.to_bits() != 0.0_f64.to_bits()
                || spec.ub.to_bits() != 1.0_f64.to_bits()
                || std::mem::replace(&mut candidate_seen[col.0], true)
            {
                return Err(MipError::Encoding(format!(
                    "target-FSB candidate column {} is not a distinct live [0, 1] binary",
                    col.0
                )));
            }
        }
    }

    let now = std::time::Instant::now();
    let relative_deadline = now
        .checked_add(std::time::Duration::from_secs_f64(hard_timeout_slice_secs(
            timeout_secs,
        )))
        .ok_or_else(|| MipError::Solver("AY shared-prefix deadline overflow".to_string()))?;
    let deadline = caller_deadline.min(relative_deadline);
    let Some(remaining) = deadline
        .checked_duration_since(now)
        .filter(|remaining| !remaining.is_zero())
    else {
        return Ok(MipResult::Timeout);
    };

    // All owned setup is charged to the caller deadline.  The one IR clone is
    // for NY's detached hard-stop worker, not one clone per prefix assignment.
    let problem = problem.clone();
    let input_vars = input_vars.to_vec();
    let output_vars = output_vars.to_vec();
    let split_cols = split_cols.to_vec();
    let target_fsb_candidates = target_fsb_candidates.map(<[Col]>::to_vec);
    let branch_hints = branch_hints.to_vec();
    let regions_total = 1usize << split_cols.len();
    let nodes_before = ay_milp::nodes_explored();
    let lp_calls_before = profiled_bounded_lp_calls();

    let solve = move || {
        let model = to_ay_model(&problem)?;
        let ay_splits = split_cols
            .iter()
            .map(|col| {
                model
                    .col_at(col.0)
                    .ok_or_else(|| MipError::Encoding(format!("shared-prefix column {}", col.0)))
            })
            .collect::<Result<Vec<_>>>()?;
        let ay_target_fsb_candidates = target_fsb_candidates
            .as_ref()
            .map(|candidates| {
                candidates
                    .iter()
                    .map(|col| {
                        model.col_at(col.0).ok_or_else(|| {
                            MipError::Encoding(format!("target-FSB candidate column {}", col.0))
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let opts = window_scoped_opts(
            graph_mip_solve_opts(remaining.as_secs_f64(), node_warm_time_limit),
            &problem,
        )
        .with_deadline(deadline)
        .with_require_certificates(true);
        let mut session =
            BabSession::new(model, &opts).map_err(|error| MipError::Solver(error.to_string()))?;
        if !branch_hints.is_empty() {
            let hint_cols = branch_hints
                .iter()
                .filter_map(|col| session.model().col_at(col.0))
                .collect::<Vec<_>>();
            session.hint_branch_order(&hint_cols);
        }
        let phase_telemetry = shared_prefix_phase_telemetry_enabled();
        let marker = match (mode, margin_row) {
            (SharedBinaryPrefixMode::Plain, _) => {
                shared_prefix_session_start_marker_if(phase_telemetry, regions_total)
            }
            (SharedBinaryPrefixMode::MarkedMargin, Some(row)) => {
                marked_margin_shared_prefix_session_start_marker_if(
                    phase_telemetry,
                    regions_total,
                    row,
                    target_fsb_prefix,
                )
            }
            (SharedBinaryPrefixMode::MarkedMargin, None) => None,
        };
        if let Some(marker) = marker {
            eprintln!("{marker}");
        }
        let outcome = match (mode, ay_target_fsb_candidates.as_deref()) {
            (SharedBinaryPrefixMode::Plain, None) => session.check_shared_binary_prefix(&ay_splits),
            (SharedBinaryPrefixMode::MarkedMargin, None) => {
                session.check_marked_margin_shared_binary_prefix(&ay_splits)
            }
            (SharedBinaryPrefixMode::MarkedMargin, Some(candidates)) => {
                let fallback: [ay_milp::Col; 4] =
                    ay_splits.as_slice().try_into().map_err(|_| {
                        MipError::Encoding(
                            "target-FSB prefix fallback lost its exact four-column width"
                                .to_string(),
                        )
                    })?;
                session.check_marked_margin_target_fsb_shared_binary_prefix(
                    &fallback,
                    candidates,
                    &TargetFsbPrefixOpts::new(),
                )
            }
            (SharedBinaryPrefixMode::Plain, Some(_)) => unreachable!(
                "target-FSB candidate validation rejects the plain entry before session creation"
            ),
        }
        .map_err(|error| MipError::Solver(error.to_string()))?;
        Ok(map_shared_binary_prefix_outcome(
            outcome,
            session.model(),
            &input_vars,
            &output_vars,
            mode,
        ))
    };
    let result = match run_with_hard_deadline_at(deadline, mode.hard_deadline_label(), solve)? {
        Some(result) => {
            let phase_telemetry = shared_prefix_phase_telemetry_enabled();
            let marker = match (mode, margin_row) {
                (SharedBinaryPrefixMode::Plain, _) => shared_prefix_session_completion_marker_if(
                    phase_telemetry,
                    regions_total,
                    Some(&result),
                ),
                (SharedBinaryPrefixMode::MarkedMargin, Some(row)) => {
                    marked_margin_shared_prefix_session_completion_marker_if(
                        phase_telemetry,
                        regions_total,
                        row,
                        target_fsb_prefix,
                        Some(&result),
                    )
                }
                (SharedBinaryPrefixMode::MarkedMargin, None) => None,
            };
            if let Some(marker) = marker {
                eprintln!("{marker}");
            }
            result
        }
        None => MipResult::Timeout,
    };

    let nodes_process_global_snapshot_delta =
        ay_milp::nodes_explored().saturating_sub(nodes_before);
    let lp_calls_process_global_snapshot_delta =
        profiled_bounded_lp_call_delta(lp_calls_before, profiled_bounded_lp_calls());
    let outcome = shared_prefix_outcome_tag(&result);
    let verdict_authority = matches!(&result, MipResult::Unsat { certified: true });
    tracing::info!(
        canary = mode.canary_name(),
        marked_margin = mode == SharedBinaryPrefixMode::MarkedMargin,
        margin_row = ?margin_row,
        target_fsb_prefix,
        admitted_seam_model_lowerings_max = 1,
        admitted_seam_bab_sessions_max =
            if mode == SharedBinaryPrefixMode::MarkedMargin { 2 } else { 1 },
        admitted_seam_native_prefix_search_sessions_max = 1,
        admitted_seam_root_preparations_max = 1,
        lp_calls_process_global_snapshot_delta = ?lp_calls_process_global_snapshot_delta,
        lp_calls_exact_per_call = false,
        regions_total,
        regions_entered = "not-exported-by-pinned-ay-typed-api",
        regions_entered_exact_per_call = false,
        nodes_process_global_snapshot_delta,
        nodes_exact_per_call = false,
        outcome,
        verdict_authority,
        "AY shared-prefix feasibility canary completed"
    );
    Ok(result)
}

#[cfg(test)]
mod shared_binary_prefix_tests {
    use super::*;

    #[test]
    fn phase_telemetry_markers_are_exact_and_default_dark() {
        for raw in [None, Some(""), Some("0"), Some("true"), Some(" 1")] {
            assert!(!shared_prefix_phase_telemetry_gate_on(raw));
        }
        assert!(shared_prefix_phase_telemetry_gate_on(Some("1")));

        let result = MipResult::Unsat { certified: true };
        assert_eq!(shared_prefix_session_start_marker_if(false, 8), None);
        assert_eq!(
            shared_prefix_session_completion_marker_if(false, 8, Some(&result)),
            None
        );
        assert_eq!(
            shared_prefix_session_completion_marker_if(true, 8, None),
            None,
            "a wrapper-generated timeout must not claim session completion"
        );
        assert_eq!(
            shared_prefix_session_start_marker_if(true, 8).as_deref(),
            Some("NY_MIP_SAFENLP_SHARED_PREFIX_V1 event=session-start regions_total=8")
        );
        assert_eq!(
            shared_prefix_session_completion_marker_if(true, 8, Some(&result)).as_deref(),
            Some(
                "NY_MIP_SAFENLP_SHARED_PREFIX_V1 event=session-complete \
                 regions_total=8 outcome=unsat-tree-replay-verified verdict_authority=true"
            )
        );

        assert_eq!(
            marked_margin_shared_prefix_session_start_marker_if(false, 8, 42, false),
            None
        );
        assert_eq!(
            marked_margin_shared_prefix_session_completion_marker_if(true, 8, 42, false, None),
            None,
            "a wrapper timeout must not claim marked-session completion"
        );
        assert_eq!(
            marked_margin_shared_prefix_session_start_marker_if(true, 8, 42, false).as_deref(),
            Some(
                "NY_MIP_SAFENLP_MARKED_MARGIN_PREFIX_V1 event=session-start \
                 regions_total=8 margin_row=42 target_fsb_prefix=false"
            )
        );
        assert_eq!(
            marked_margin_shared_prefix_session_completion_marker_if(
                true,
                8,
                42,
                false,
                Some(&result),
            )
            .as_deref(),
            Some(
                "NY_MIP_SAFENLP_MARKED_MARGIN_PREFIX_V1 event=session-complete \
                 regions_total=8 margin_row=42 target_fsb_prefix=false \
                 outcome=unsat-tree-replay-verified \
                 original_tree_replay_authority=true"
            )
        );
        assert_eq!(
            marked_margin_shared_prefix_session_start_marker_if(true, 8, 42, true).as_deref(),
            Some(
                "NY_MIP_SAFENLP_MARKED_MARGIN_PREFIX_V1 event=session-start \
                 regions_total=8 margin_row=42 target_fsb_prefix=true"
            )
        );
        assert_eq!(
            marked_margin_shared_prefix_session_completion_marker_if(
                true,
                8,
                42,
                true,
                Some(&result),
            )
            .as_deref(),
            Some(
                "NY_MIP_SAFENLP_MARKED_MARGIN_PREFIX_V1 event=session-complete \
                 regions_total=8 margin_row=42 target_fsb_prefix=true \
                 outcome=unsat-tree-replay-verified \
                 original_tree_replay_authority=true"
            )
        );
    }

    fn deadline() -> std::time::Instant {
        std::time::Instant::now() + std::time::Duration::from_secs(10)
    }

    fn run(problem: &MilpProblem, splits: &[Col]) -> MipResult {
        check_feasibility_shared_binary_prefix(
            problem,
            10.0,
            Some(deadline()),
            None,
            &[],
            &[],
            splits,
            &[],
        )
        .expect("shared-prefix backend call")
    }

    fn run_marked(problem: &MilpProblem, splits: &[Col]) -> MipResult {
        check_feasibility_marked_margin_shared_binary_prefix(
            problem,
            10.0,
            Some(deadline()),
            None,
            &[],
            &[],
            splits,
            &[],
        )
        .expect("marked-margin shared-prefix backend call")
    }

    fn target_fsb_selection_fixture() -> (MilpProblem, [Col; 4], [Col; 8]) {
        let mut problem = MilpProblem::new();
        let candidates: [Col; 8] = std::array::from_fn(|_| problem.add_integer_col(0.0, 0.0, 1.0));
        let fallback_only: [Col; 2] =
            std::array::from_fn(|_| problem.add_integer_col(0.0, 0.0, 1.0));
        let useful = [candidates[1], candidates[3], candidates[5], candidates[7]];
        let mut objective = Vec::with_capacity(useful.len());
        for &split in &useful {
            let epigraph = problem.add_col(0.0, 0.0, 1.0);
            // p >= max(x, 1-x). Fixing a useful binary raises its target
            // contribution from 1/2 to 1; a distractor leaves it unchanged.
            problem.add_row(0.0, f64::INFINITY, [(epigraph, 1.0), (split, -1.0)]);
            problem.add_row(1.0, f64::INFINITY, [(epigraph, 1.0), (split, 1.0)]);
            objective.push((epigraph, 1.0));
        }
        problem
            .add_margin_row(f64::NEG_INFINITY, 3.0, objective)
            .expect("one-sided target-FSB fixture margin");
        let fallback = [
            candidates[0],
            candidates[2],
            fallback_only[0],
            fallback_only[1],
        ];
        (problem, fallback, candidates)
    }

    #[test]
    fn marked_margin_target_fsb_entry_reaches_a_successful_ay_selection() {
        const CHILD_MODE: &str = "NY_TEST_SAFENLP_TARGET_FSB_PREFIX_SELECTED";
        if std::env::var_os(CHILD_MODE).is_some() {
            let (problem, fallback, candidates) = target_fsb_selection_fixture();
            let result = check_feasibility_marked_margin_target_fsb_shared_binary_prefix(
                &problem,
                10.0,
                Some(deadline()),
                None,
                &[],
                &[],
                &fallback,
                &candidates,
                &[],
            )
            .expect("NY target-FSB entry must reach AY without a second-session fallback");
            assert!(
                !matches!(result, MipResult::Error(_)),
                "a completed target selection must retain ordinary mapped outcomes: {result:?}"
            );
            return;
        }

        let test_name = "ay_lib::shared_binary_prefix_tests::\
                         marked_margin_target_fsb_entry_reaches_a_successful_ay_selection";
        let mut command = std::process::Command::new(
            std::env::current_exe().expect("current ny-mip test executable"),
        );
        command
            .arg("--exact")
            .arg(test_name)
            .arg("--nocapture")
            .env(CHILD_MODE, "1")
            .env("AY_MILP_TRACE", "1")
            .env_remove("AY_MILP_MAX_NODES")
            .env_remove("AY_MILP_NO_CUTS")
            .env_remove("AY_MILP_NO_MARGIN_REFRAME")
            .env_remove("AY_MILP_SMT");
        let output = command.output().expect("spawn isolated target-FSB test");
        assert!(
            output.status.success(),
            "isolated target-FSB integration check failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("AY_MILP_TRACE target-fsb-prefix selected:"),
            "valid NY columns did not reach AY's successful selector:\n{stderr}"
        );
    }

    #[test]
    fn shared_prefix_plain_feasibility_returns_a_sat_candidate() {
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        let y = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(1.0, 1.0, [(x, 1.0), (y, 1.0)]);

        assert!(matches!(run(&problem, &[x, y]), MipResult::Sat { .. }));
    }

    #[test]
    fn shared_prefix_authority_requires_replayed_whole_tree() {
        // The unique LP point is (1/2, 1/2, 1/2), but no binary point can
        // satisfy all three equations.  Root Farkas is impossible, forcing a
        // genuine case-split tree whose caller-frame replay is the only
        // authority admitted by this canary.
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        let y = problem.add_integer_col(0.0, 0.0, 1.0);
        let z = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(1.0, 1.0, [(x, 1.0), (y, 1.0)]);
        problem.add_row(1.0, 1.0, [(y, 1.0), (z, 1.0)]);
        problem.add_row(1.0, 1.0, [(x, 1.0), (z, 1.0)]);

        assert!(matches!(
            run(&problem, &[x, y, z]),
            MipResult::Unsat { certified: true }
        ));
    }

    #[test]
    fn root_only_infeasibility_has_no_shared_tree_authority() {
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(f64::NEG_INFINITY, 0.0, [(x, 1.0)]);
        problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);

        assert!(matches!(
            run(&problem, &[x]),
            MipResult::Unsat { certified: false }
        ));
    }

    #[test]
    fn marked_margin_shared_prefix_returns_only_a_sat_candidate() {
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        let y = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(1.0, f64::INFINITY, [(x, 1.0), (y, 1.0)]);
        let margin = problem.add_row(f64::NEG_INFINITY, 1.0, [(x, 1.0), (y, 1.0)]);
        problem.mark_margin_row(margin).expect("one-sided margin");

        assert!(matches!(run_marked(&problem, &[x]), MipResult::Sat { .. }));
    }

    #[test]
    fn marked_margin_strict_frontier_with_a_verified_whole_tree_is_unsat() {
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);
        let margin = problem.add_row(f64::NEG_INFINITY, 0.0, [(x, 1.0)]);
        problem.mark_margin_row(margin).expect("one-sided margin");

        let result = run_marked(&problem, &[x]);
        assert!(
            matches!(result, MipResult::Unsat { certified: true }),
            "this complete two-leaf prefix must replay a whole tree against the \
             contradictory original model, got {result:?}"
        );
    }

    #[test]
    fn interrupted_marked_margin_tree_replays_through_ny_mapping() {
        const CHILD_MODE: &str = "NY_TEST_SAFENLP_MARKED_PREFIX_INTERRUPTED_TREE";
        if std::env::var_os(CHILD_MODE).is_some() {
            // Four binaries constrained to a half-integral sum are
            // integer-infeasible, while both sides of the one-column prefix
            // remain LP-feasible. A zero-node cap therefore interrupts a
            // genuinely open marked tree whose strict upper margin bound can
            // only become authoritative through original-frame region replay.
            let mut problem = MilpProblem::new();
            let cols = [
                problem.add_integer_col(0.0, 0.0, 1.0),
                problem.add_integer_col(0.0, 0.0, 1.0),
                problem.add_integer_col(0.0, 0.0, 1.0),
                problem.add_integer_col(0.0, 0.0, 1.0),
            ];
            let sum = [
                (cols[0], 1.0),
                (cols[1], 1.0),
                (cols[2], 1.0),
                (cols[3], 1.0),
            ];
            problem.add_row(1.5, 1.5, sum);
            let margin = problem.add_row(f64::NEG_INFINITY, 1.0, sum);
            problem
                .mark_margin_row(margin)
                .expect("one-sided marked margin");

            let result = run_marked(&problem, &cols[..1]);
            assert!(
                matches!(result, MipResult::Unsat { certified: true }),
                "the interrupted marked tree must cross NY's original-model replay seam, \
                 got {result:?}"
            );
            return;
        }

        let test_name = "ay_lib::shared_binary_prefix_tests::\
                         interrupted_marked_margin_tree_replays_through_ny_mapping";
        let mut command = std::process::Command::new(
            std::env::current_exe().expect("current ny-mip test binary"),
        );
        command
            .arg("--exact")
            .arg(test_name)
            .arg("--nocapture")
            .env(CHILD_MODE, "1")
            .env("AY_MILP_MAX_NODES", "0")
            .env("AY_MILP_NO_CUTS", "1")
            .env_remove("AY_MILP_NO_MARGIN_REFRAME")
            .env_remove("AY_MILP_SMT")
            .env_remove("NY_AY_WINDOW_CUTS");
        let output = command
            .output()
            .expect("spawn isolated interrupted-tree mapping test");
        assert!(
            output.status.success(),
            "isolated interrupted-tree mapping check failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    #[test]
    fn plain_and_marked_backend_entries_require_their_exact_marker_shape() {
        let mut marked = MilpProblem::new();
        let x = marked.add_integer_col(0.0, 0.0, 1.0);
        let margin = marked.add_row(f64::NEG_INFINITY, 0.0, [(x, 1.0)]);
        marked.mark_margin_row(margin).expect("one-sided margin");
        let plain_error = check_feasibility_shared_binary_prefix(
            &marked,
            10.0,
            Some(deadline()),
            None,
            &[],
            &[],
            &[x],
            &[],
        )
        .expect_err("plain entry must reject a marker");
        assert!(plain_error.to_string().contains("rejects a marked margin"));

        let mut plain = MilpProblem::new();
        let x = plain.add_integer_col(0.0, 0.0, 1.0);
        let marked_error = check_feasibility_marked_margin_shared_binary_prefix(
            &plain,
            10.0,
            Some(deadline()),
            None,
            &[],
            &[],
            &[x],
            &[],
        )
        .expect_err("marked entry must require a marker");
        assert!(marked_error.to_string().contains("requires a marked row"));
    }

    #[test]
    fn bare_marked_margin_optimum_and_bound_have_no_unsat_authority() {
        let mut model = Model::new();
        model.add_binary_col();
        let zero = BigRational::from_integer(0.into());
        let optimal = Outcome::Optimal {
            value: zero.clone(),
            model_values: vec![zero.clone()],
            cert: None,
        };
        assert!(matches!(
            map_shared_binary_prefix_outcome(
                optimal,
                &model,
                &[],
                &[],
                SharedBinaryPrefixMode::MarkedMargin,
            ),
            MipResult::Error(_)
        ));

        let bound = Outcome::Bound {
            dual_bound: zero,
            rigorous: true,
        };
        assert!(matches!(
            map_shared_binary_prefix_outcome(
                bound,
                &model,
                &[],
                &[],
                SharedBinaryPrefixMode::MarkedMargin,
            ),
            MipResult::Timeout
        ));
    }

    #[test]
    fn inherited_ay_controls_fail_closed_before_any_retry() {
        const CHILD_MODE: &str = "NY_TEST_SAFENLP_MARKED_PREFIX_FAIL_CLOSED";
        if let Ok(mode) = std::env::var(CHILD_MODE) {
            let mut problem = MilpProblem::new();
            let x = problem.add_integer_col(0.0, 0.0, 1.0);
            let margin = problem.add_row(f64::NEG_INFINITY, 0.0, [(x, 1.0)]);
            problem.mark_margin_row(margin).expect("one-sided margin");
            let error = check_feasibility_marked_margin_shared_binary_prefix(
                &problem,
                10.0,
                Some(deadline()),
                None,
                &[],
                &[],
                &[x],
                &[],
            )
            .expect_err("an incompatible inherited AY control must fail closed");
            let message = error.to_string();
            match mode.as_str() {
                "margin-disabled" => assert!(
                    message.contains("disabled by AY_MILP_NO_MARGIN_REFRAME"),
                    "unexpected margin-disable error: {message}"
                ),
                "forced-smt" => assert!(
                    message.contains("rejects forced non-native AY_MILP_SMT"),
                    "unexpected forced-SMT error: {message}"
                ),
                other => panic!("unknown child mode {other:?}"),
            }
            return;
        }

        let test_name =
            "ay_lib::shared_binary_prefix_tests::inherited_ay_controls_fail_closed_before_any_retry";
        for mode in ["margin-disabled", "forced-smt"] {
            let mut command = std::process::Command::new(
                std::env::current_exe().expect("current ny-mip test executable"),
            );
            command
                .arg("--exact")
                .arg(test_name)
                .arg("--nocapture")
                .env(CHILD_MODE, mode)
                .env_remove("AY_MILP_NO_MARGIN_REFRAME")
                .env_remove("AY_MILP_SMT");
            match mode {
                "margin-disabled" => {
                    command.env("AY_MILP_NO_MARGIN_REFRAME", "1");
                }
                "forced-smt" => {
                    command.env("AY_MILP_SMT", "1");
                }
                _ => unreachable!(),
            }
            let output = command.output().expect("spawn isolated control test");
            assert!(
                output.status.success(),
                "isolated {mode} fail-closed check failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
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
    hard_deadline: Option<std::time::Instant>,
    node_warm_time_limit: Option<std::time::Duration>,
    input_vars: &[Col],
    output_vars: &[Col],
    warm_start_cols: Option<&[f64]>,
    branch_hints: &[Col],
) -> Result<MipResult> {
    // An absolute caller deadline means preliminary work has already consumed
    // part of this solve's historical envelope.  Never reapply the optional
    // window floor in that mode.  The relative slice remains an independent
    // per-call ceiling, so violation-slack retries can still subdivide the
    // remaining ledger.
    let (timeout_secs, hard_deadline) = match hard_deadline {
        Some(caller_deadline) => {
            let now = std::time::Instant::now();
            let relative = now
                .checked_add(std::time::Duration::from_secs_f64(hard_timeout_slice_secs(
                    timeout_secs,
                )))
                .ok_or_else(|| {
                    MipError::Solver("ay solve deadline overflow (check)".to_string())
                })?;
            let deadline = caller_deadline.min(relative);
            let Some(remaining) = deadline
                .checked_duration_since(now)
                .filter(|remaining| !remaining.is_zero())
            else {
                return Ok(MipResult::Timeout);
            };
            (remaining.as_secs_f64(), Some(deadline))
        }
        None => (window_budget_floor(problem, timeout_secs), None),
    };
    // Owned copies for the detached worker (the whole solve — model lowering
    // included — runs inside the externally enforced slice; see
    // `run_with_hard_deadline`). With a caller deadline, the deadline was
    // fixed before these copies, so their setup cost is charged too.
    let problem = problem.clone();
    let input_vars = input_vars.to_vec();
    let output_vars = output_vars.to_vec();
    let warm_start_cols = warm_start_cols.map(<[f64]>::to_vec);
    let branch_hints = branch_hints.to_vec();
    if hard_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
        return Ok(MipResult::Timeout);
    }
    let solve = move || {
        let model = to_ay_model(&problem)?;
        let opts = window_scoped_opts(
            graph_mip_solve_opts(timeout_secs, node_warm_time_limit),
            &problem,
        )
        // AY now has several independently replayed infeasibility artifacts
        // that live beside `Outcome` (hybrid PB/LP, parity, PB-DP, and others),
        // not only the Farkas/tree fields NY historically inspected. Requiring
        // certificates at the session boundary makes a surviving bare
        // `Infeasible` an explicit proof-bearing API state; an unsupported or
        // unproved UNSAT degrades to `Unknown` before it reaches NY.
        .with_require_certificates(true);
        // Seat search and independent side-certificate replay inside one
        // absolute envelope.  A caller-owned deadline wins over the relative
        // AY slice; without one, this is the same effective deadline AY fixes
        // at the start of `BabSession::check` (sampled slightly earlier, hence
        // never granting replay extra time).
        let sat_relu_replay_deadline = opts
            .effective_deadline(std::time::Instant::now())
            .map(|deadline| hard_deadline.map_or(deadline, |outer| deadline.min(outer)))
            .or(hard_deadline);
        let mut session =
            BabSession::new(model, &opts).map_err(|e| MipError::Solver(e.to_string()))?;
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
        Ok(map_bab_outcome(
            outcome,
            &session,
            &input_vars,
            &output_vars,
            None,
            sat_relu_replay_deadline,
            opts.require_certificates,
        ))
    };
    let result = match hard_deadline {
        Some(deadline) => run_with_hard_deadline_at(deadline, "check", solve),
        None => run_with_hard_deadline(timeout_secs, "check", solve),
    }?;
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
    hard_deadline: Option<std::time::Instant>,
    objective: ObjectiveSpec,
    input_vars: &[Col],
    output_vars: &[Col],
) -> Result<MipResult> {
    let (timeout_secs, hard_deadline) = match hard_deadline {
        Some(caller_deadline) => {
            let now = std::time::Instant::now();
            let relative = now
                .checked_add(std::time::Duration::from_secs_f64(hard_timeout_slice_secs(
                    timeout_secs,
                )))
                .ok_or_else(|| {
                    MipError::Solver("ay solve deadline overflow (optimize)".to_string())
                })?;
            let deadline = caller_deadline.min(relative);
            let Some(remaining) = deadline
                .checked_duration_since(now)
                .filter(|remaining| !remaining.is_zero())
            else {
                return Ok(MipResult::Timeout);
            };
            (remaining.as_secs_f64(), Some(deadline))
        }
        None => (window_budget_floor(problem, timeout_secs), None),
    };
    // Same externally enforced slice as `check_feasibility` — this seam feeds
    // the tighten lane, where one hung `session.check()` would equally blow
    // the whole instance budget.
    let problem = problem.clone();
    let input_vars = input_vars.to_vec();
    let output_vars = output_vars.to_vec();
    if hard_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
        return Ok(MipResult::Timeout);
    }
    let solve = move || {
        let mut model = to_ay_model(&problem)?;
        let target = model
            .col_at(objective.col.0)
            .ok_or_else(|| MipError::Encoding(format!("objective column {}", objective.col.0)))?;
        let sense = match objective.sense {
            ObjSense::Minimize => AySense::Minimize,
            ObjSense::Maximize => AySense::Maximize,
        };
        model.set_objective(&[(target, 1.0)], sense);
        let opts = window_scoped_opts(solve_opts(timeout_secs), &problem);
        let sat_relu_replay_deadline = opts
            .effective_deadline(std::time::Instant::now())
            .map(|deadline| hard_deadline.map_or(deadline, |outer| deadline.min(outer)))
            .or(hard_deadline);
        let mut session =
            BabSession::new(model, &opts).map_err(|e| MipError::Solver(e.to_string()))?;
        let outcome = session
            .check()
            .map_err(|e| MipError::Solver(e.to_string()))?;
        Ok(map_bab_outcome(
            outcome,
            &session,
            &input_vars,
            &output_vars,
            Some(objective),
            sat_relu_replay_deadline,
            opts.require_certificates,
        ))
    };
    let result = match hard_deadline {
        Some(deadline) => run_with_hard_deadline_at(deadline, "optimize", solve),
        None => run_with_hard_deadline(timeout_secs, "optimize", solve),
    }?;
    Ok(result.unwrap_or(MipResult::Timeout))
}

/// Extract a candidate column vector without the permissive zero-fill used by
/// the ordinary `MipResult` seam.
///
/// A witness-only probe must never manufacture a missing/non-representable
/// coordinate and leave the concrete replay to discover it later.
fn extract_candidate_cols(
    values: &[BigRational],
    cols: &[Col],
) -> std::result::Result<Vec<f64>, String> {
    cols.iter()
        .map(|col| {
            let value = values
                .get(col.0)
                .ok_or_else(|| format!("candidate omits column {}", col.0))?;
            let value = value
                .to_f64()
                .ok_or_else(|| format!("candidate column {} is not representable", col.0))?;
            value
                .is_finite()
                .then_some(value)
                .ok_or_else(|| format!("candidate column {} is non-finite", col.0))
        })
        .collect()
}

/// Admit only the feasible-point arms of an AY optimization outcome.
///
/// The exact `check_point` call is deliberately repeated here even though
/// `BabSession::check` already finishes through AY's point gate.  It makes the
/// SAT-only authority boundary local and testable: a forged, truncated, or
/// stale point can only become `ReplayRejected`, never a witness.
fn map_one_sided_sat_outcome(
    outcome: Outcome,
    model: &Model,
    input_vars: &[Col],
    output_vars: &[Col],
) -> OneSidedSatProbe {
    let model_values = match outcome {
        Outcome::Optimal { model_values, .. } | Outcome::Feasible { model_values, .. } => {
            model_values
        }
        Outcome::Infeasible { .. } => {
            return OneSidedSatProbe::Declined(OneSidedSatDecline::InfeasibleIgnored);
        }
        Outcome::Unknown { .. } | Outcome::Bound { .. } | Outcome::Unbounded => {
            return OneSidedSatProbe::Declined(OneSidedSatDecline::NoWitness);
        }
        other => {
            return OneSidedSatProbe::Declined(OneSidedSatDecline::SolverError(format!(
                "unexpected AY objective outcome: {other:?}"
            )));
        }
    };

    if model_values.len() != model.num_cols() {
        return OneSidedSatProbe::Declined(OneSidedSatDecline::ReplayRejected(format!(
            "candidate arity {} != model columns {}",
            model_values.len(),
            model.num_cols()
        )));
    }
    if let Err(error) = model.check_point(&model_values) {
        return OneSidedSatProbe::Declined(OneSidedSatDecline::ReplayRejected(format!(
            "{error:?}"
        )));
    }
    let objective = match model.objective_value_at(&model_values).to_f64() {
        Some(value) if value.is_finite() => value,
        _ => {
            return OneSidedSatProbe::Declined(OneSidedSatDecline::ReplayRejected(
                "objective value is not a finite f64".to_string(),
            ));
        }
    };
    let input_values = match extract_candidate_cols(&model_values, input_vars) {
        Ok(values) => values,
        Err(error) => {
            return OneSidedSatProbe::Declined(OneSidedSatDecline::ReplayRejected(error));
        }
    };
    let output_values = match extract_candidate_cols(&model_values, output_vars) {
        Ok(values) => values,
        Err(error) => {
            return OneSidedSatProbe::Declined(OneSidedSatDecline::ReplayRejected(error));
        }
    };
    OneSidedSatProbe::Witness(OneSidedSatWitness {
        objective,
        output_values,
        input_values,
    })
}

fn finish_one_sided_sat_run(result: Result<Option<OneSidedSatProbe>>) -> OneSidedSatProbe {
    match result {
        Ok(Some(probe)) => probe,
        Ok(None) => OneSidedSatProbe::Declined(OneSidedSatDecline::Deadline),
        Err(error) => {
            OneSidedSatProbe::Declined(OneSidedSatDecline::SolverError(error.to_string()))
        }
    }
}

/// Optimize an already-constrained one-sided row solely to propose a SAT point.
///
/// The row remains present in the model.  Its sparse linear form becomes the
/// objective (minimize for `<=`, maximize for `>=`) so AY searches toward a
/// deeper, more replay-stable violation.  Any non-witness result is collapsed
/// into [`OneSidedSatProbe::Declined`]; in particular, an exact infeasibility
/// result is intentionally discarded and can never reach NY's UNSAT admission.
pub(crate) fn probe_one_sided_sat(
    problem: &MilpProblem,
    timeout_secs: f64,
    row: crate::ir::Row,
    sense: Sense,
    node_warm_time_limit: Option<std::time::Duration>,
    input_vars: &[Col],
    output_vars: &[Col],
    branch_hints: &[Col],
    hard_deadline: Option<std::time::Instant>,
) -> OneSidedSatProbe {
    let now = std::time::Instant::now();
    let slice = std::time::Duration::from_secs_f64(hard_timeout_slice_secs(timeout_secs));
    let Some(relative_deadline) = now.checked_add(slice) else {
        return OneSidedSatProbe::Declined(OneSidedSatDecline::SolverError(
            "SAT-objective deadline overflow".to_string(),
        ));
    };
    let deadline = hard_deadline.map_or(relative_deadline, |absolute| {
        absolute.min(relative_deadline)
    });
    if deadline <= now {
        return OneSidedSatProbe::Declined(OneSidedSatDecline::Deadline);
    }
    // Unlike the complete-verifier path, this caller-owned probe slice is
    // absolute.  Never apply the optional window budget floor here: doing so
    // would steal time reserved for the historical fallback.  The deadline is
    // fixed BEFORE the owned IR/input copies, so setup cannot buy a fresh
    // solver clock.
    let problem = problem.clone();
    let input_vars = input_vars.to_vec();
    let output_vars = output_vars.to_vec();
    let branch_hints = branch_hints.to_vec();
    let result = run_with_hard_deadline_at(deadline, "sat-objective", move || {
        let mut model = to_ay_model(&problem)?;
        let objective_row = model
            .row_at(row.0)
            .ok_or_else(|| MipError::Encoding(format!("objective row {}", row.0)))?;
        let (coeffs, lb, ub) = model.row(objective_row);
        let shape_matches = match sense {
            Sense::Minimise => lb == f64::NEG_INFINITY && ub.is_finite(),
            Sense::Maximise => lb.is_finite() && ub == f64::INFINITY,
        };
        if !shape_matches || coeffs.is_empty() {
            return Ok(OneSidedSatProbe::Declined(OneSidedSatDecline::InvalidRow(
                format!("objective row {} changed shape during AY lowering", row.0),
            )));
        }
        let objective: Vec<(ay_milp::Col, f64)> = coeffs
            .iter()
            .map(|&(col, coefficient)| {
                model
                    .col_at(col as usize)
                    .map(|column| (column, coefficient))
                    .ok_or_else(|| {
                        MipError::Encoding(format!(
                            "objective row {} references missing column {col}",
                            row.0
                        ))
                    })
            })
            .collect::<Result<_>>()?;
        // This is a separate SAT-only schedule, not AY's verdict-preserving
        // margin reframe.  Clear any legacy marker defensively, keep the row
        // constrained, and install its sparse form directly as the objective.
        model.clear_margin();
        model.set_objective(
            &objective,
            match sense {
                Sense::Minimise => AySense::Minimize,
                Sense::Maximise => AySense::Maximize,
            },
        );
        let mut session = BabSession::new(
            model,
            &window_scoped_opts(
                graph_mip_solve_opts(timeout_secs, node_warm_time_limit),
                &problem,
            ),
        )
        .map_err(|error| MipError::Solver(error.to_string()))?;
        if !branch_hints.is_empty() {
            let hint_cols: Vec<ay_milp::Col> = branch_hints
                .iter()
                .filter_map(|col| session.model().col_at(col.0))
                .collect();
            session.hint_branch_order(&hint_cols);
        }
        let outcome = session
            .check()
            .map_err(|error| MipError::Solver(error.to_string()))?;
        Ok(map_one_sided_sat_outcome(
            outcome,
            session.model(),
            &input_vars,
            &output_vars,
        ))
    });
    finish_one_sided_sat_run(result)
}

#[cfg(test)]
mod one_sided_sat_tests {
    use super::*;

    #[test]
    fn forged_out_of_model_point_is_replay_rejected() {
        let mut model = Model::new();
        let x = model.add_col(0.0, 1.0);
        model.set_objective(&[(x, 1.0)], AySense::Maximize);
        let two = BigRational::from_integer(2.into());
        let forged = Outcome::Optimal {
            value: two.clone(),
            model_values: vec![two],
            cert: None,
        };

        assert!(matches!(
            map_one_sided_sat_outcome(forged, &model, &[Col(0)], &[Col(0)]),
            OneSidedSatProbe::Declined(OneSidedSatDecline::ReplayRejected(_))
        ));
    }

    #[test]
    fn truncated_point_is_replay_rejected_before_zero_fill() {
        let mut model = Model::new();
        let x = model.add_col(0.0, 1.0);
        model.set_objective(&[(x, 1.0)], AySense::Minimize);
        let forged = Outcome::Optimal {
            value: BigRational::from_integer(0.into()),
            model_values: vec![],
            cert: None,
        };

        assert!(matches!(
            map_one_sided_sat_outcome(forged, &model, &[Col(0)], &[Col(0)]),
            OneSidedSatProbe::Declined(OneSidedSatDecline::ReplayRejected(_))
        ));
    }

    #[test]
    fn infeasible_outcome_has_no_unsat_surface() {
        let model = Model::new();
        let outcome = Outcome::Infeasible {
            cert: None,
            tree_cert: None,
        };
        assert_eq!(
            map_one_sided_sat_outcome(outcome, &model, &[], &[]),
            OneSidedSatProbe::Declined(OneSidedSatDecline::InfeasibleIgnored)
        );
    }

    #[test]
    fn deadline_and_error_runs_are_verdict_neutral_declines() {
        assert_eq!(
            finish_one_sided_sat_run(Ok(None)),
            OneSidedSatProbe::Declined(OneSidedSatDecline::Deadline)
        );
        assert!(matches!(
            finish_one_sided_sat_run(Err(MipError::Solver("boom".to_string()))),
            OneSidedSatProbe::Declined(OneSidedSatDecline::SolverError(_))
        ));
    }
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
    fn window_floor_and_hard_clamp_define_the_historical_envelope() {
        let mut sub_gate = MilpProblem::new();
        let x = sub_gate.add_col(0.0, 0.0, 1.0);
        sub_gate.add_row(0.0, 1.0, [(x, 1.0)]);
        assert_eq!(
            window_budget_floor_from_value(&sub_gate, 20.0, Some("100")),
            20.0,
            "sub-gate models must ignore the window floor"
        );

        let mut window = MilpProblem::new();
        let x = window.add_col(0.0, 0.0, 1.0);
        for _ in 0..WINDOW_ROWS_GATE {
            window.add_row(0.0, 1.0, [(x, 1.0)]);
        }
        assert_eq!(
            window_budget_floor_from_value(&window, 20.0, Some("100")),
            100.0
        );
        assert_eq!(
            hard_timeout_slice_secs(window_budget_floor_from_value(
                &window,
                20.0,
                Some("100000")
            )),
            86_400.0,
            "the armed envelope must include the historical wrapper's 24h clamp"
        );
        for malformed in ["", "NaN", "inf", "-1", "not-a-number"] {
            assert_eq!(
                window_budget_floor_from_value(&window, 20.0, Some(malformed)),
                20.0
            );
        }
    }

    #[test]
    fn typed_node_warm_limit_is_per_session_and_default_authority_stays_uncapped() {
        let cap = std::time::Duration::from_secs(5);
        let capped = graph_mip_solve_opts(30.0, Some(cap));
        let authority = solve_opts(30.0);
        let later_uncapped_graph = graph_mip_solve_opts(30.0, None);

        assert_eq!(capped.node_warm_time_limit, Some(cap));
        assert_eq!(
            authority.node_warm_time_limit, None,
            "certified-linear, OBBT, and other solve_opts users must stay uncapped"
        );
        assert_eq!(
            later_uncapped_graph.node_warm_time_limit, None,
            "one capped session must not mutate a later session"
        );
    }

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

    /// The flip-LNS policy, every arm, with no process state involved.
    ///
    /// This replaces a subprocess-isolation harness
    /// (`NY_INTERNAL_AY_FLIP_POLICY_TEST` + `--test-threads=1` + `Command::env`)
    /// that existed ONLY because the assertions read process-global state. The
    /// assertions are also strictly stronger now: they check the value AY will
    /// actually resolve from (the caller layer), not merely that NY's env
    /// writes landed somewhere AY might or might not read.
    #[test]
    fn tall_flip_cap_policy_defaults_old_and_exact_one_arms_native() {
        let base = EngineEconomics::new();
        for (case, raw, inherited) in [
            ("default", None, false),
            ("malformed", Some("malformed"), false),
        ] {
            let engine = flip_schedule_with(&base, raw, inherited);
            assert_eq!(
                engine.flip_lns_share(),
                Some(NY_FLIP_SHARE),
                "{case}: the historical fractional schedule must be pinned"
            );
        }

        // Canary, environment clean: leave the share UNSET so AY's own tall
        // absolute cap governs.
        let armed = flip_schedule_with(&base, Some(" 1 "), false);
        assert_eq!(armed.flip_lns_share(), None);

        // Canary with an inherited share: REFUSE to arm. NY can no longer
        // remove the variable, and AY's env layer sits below the caller layer,
        // so arming would silently mislabel the recorded A/B.
        let refused = flip_schedule_with(&base, Some("1"), true);
        assert_eq!(
            refused.flip_lns_share(),
            Some(NY_FLIP_SHARE),
            "an inherited AY_MILP_FLIP_SHARE must prevent the canary from arming"
        );

        // Both arms pin the cap and disable warm-LU, so an inherited setting
        // cannot turn a recorded canary into a custom or 12-second schedule.
        for engine in [
            flip_schedule_with(&base, None, false),
            armed,
            refused,
            flip_schedule_with(&base, Some("1"), true),
        ] {
            assert_eq!(engine.flip_lns_cap(), Some(NY_FLIP_CAP));
            assert_eq!(engine.warm_lu(), Some(false));
        }
    }

    /// The window recipe reaches window-class models, only window-class models,
    /// and never displaces the base guards.
    #[test]
    fn window_recipe_is_scoped_and_preserves_the_operator_override() {
        let mut window = MilpProblem::new();
        let c = window.add_col(0.0, 0.0, 1.0);
        for _ in 0..WINDOW_ROWS_GATE {
            window.add_row(0.0, 1.0, [(c, 1.0)]);
        }
        assert!(is_window_class(&window));

        let mut small = MilpProblem::new();
        let s = small.add_col(0.0, 0.0, 1.0);
        small.add_row(0.0, 1.0, [(s, 1.0)]);
        assert!(!is_window_class(&small));

        let base = flip_schedule(&ny_base_engine());
        let none = WindowOverrides::default();

        let w = window_recipe_with(&base, &window, none);
        assert_eq!(w.presolve_share(), Some(0.02));
        assert_eq!(w.cuts(), Some(false));
        assert_eq!(w.pump_restarts(), Some(0));
        assert_eq!(w.dive_max_pins(), Some(16));

        // Sub-gate models (the 144-instance mip-diff corpus among them) must
        // never pick the recipe up — that scoping is what makes it safe to
        // apply automatically.
        let m = window_recipe_with(&base, &small, none);
        assert_eq!(m.presolve_share(), None);
        assert_eq!(m.cuts(), None);
        assert_eq!(m.pump_restarts(), None);
        assert_eq!(m.dive_max_pins(), None);

        // The base guards survive the overlay in both cases.
        for engine in [w, m] {
            assert_eq!(engine.lattice(), Some(false));
            assert_eq!(engine.saturation_stop(), Some(false));
            assert_eq!(engine.bloom_cap_relaxation(), Some(false));
        }

        // The old `set_if_unset` contract: a knob the operator pinned is one NY
        // declines to set, so AY's environment layer resolves it.
        let overridden = window_recipe_with(
            &base,
            &window,
            WindowOverrides {
                cuts: true,
                ..WindowOverrides::default()
            },
        );
        assert_eq!(overridden.cuts(), None);
        assert_eq!(overridden.presolve_share(), Some(0.02));

        // The canary is the only reachable cuts-ON arm, so it outranks both
        // NY's pin and an operator's presence-based AY_MILP_NO_CUTS.
        let armed = window_recipe_with(
            &base,
            &window,
            WindowOverrides {
                cuts_canary: true,
                ..WindowOverrides::default()
            },
        );
        assert_eq!(armed.cuts(), Some(true));
        let armed_over_operator = window_recipe_with(
            &base,
            &window,
            WindowOverrides {
                cuts: true,
                cuts_canary: true,
                ..WindowOverrides::default()
            },
        );
        assert_eq!(armed_over_operator.cuts(), Some(true));

        // It stays scoped to window-class models like the rest of the recipe.
        let armed_small = window_recipe_with(
            &base,
            &small,
            WindowOverrides {
                cuts_canary: true,
                ..WindowOverrides::default()
            },
        );
        assert_eq!(armed_small.cuts(), None);
    }

    #[test]
    fn window_cuts_canary_opt_in_is_exact_and_fail_closed() {
        assert!(!use_window_cuts(None));
        for malformed in ["", "0", "01", "1.0", "true", "yes", "1x", "on"] {
            assert!(
                !use_window_cuts(Some(malformed)),
                "malformed value {malformed:?} must keep window cuts off"
            );
        }
        assert!(use_window_cuts(Some("1")));
        assert!(use_window_cuts(Some("  1\n")));
    }

    /// Every constructor carries the full guard set on the value AY resolves
    /// from — the structural replacement for "reasserted before every solve".
    #[test]
    fn every_solve_opts_constructor_carries_the_guards() {
        for opts in [
            solve_opts(30.0),
            graph_mip_solve_opts(30.0, None),
            graph_mip_solve_opts(30.0, Some(std::time::Duration::from_millis(500))),
        ] {
            let engine = opts.engine();
            assert_eq!(engine.lattice(), Some(false));
            assert_eq!(engine.saturation_stop(), Some(false));
            assert_eq!(
                engine.saturation_stop_floor(),
                Some(std::time::Duration::from_secs(15))
            );
            assert_eq!(engine.saturation_stop_multiplier(), Some(1.5));
            assert_eq!(engine.bloom_cap_relaxation(), Some(false));
            assert_eq!(engine.flip_lns_cap(), Some(NY_FLIP_CAP));
            assert_eq!(engine.warm_lu(), Some(false));
        }
    }

    /// Defense-in-depth regression for NY's lattice quarantine. AY 6b9f4be8
    /// repaired this near-integer detector case, but NY keeps the lane disabled
    /// until its certificate-free `Optimal` path is independently readmitted.
    #[test]
    fn near_integer_lattice_shortcut_is_disabled_before_optimization() {
        // The guards now travel on the solve's own options, so the adversarial
        // pre-state this test used to install (removing NO_LATTICE, poisoning
        // SAT_STOP_* with "NaN") is no longer expressible as env at all — and
        // no longer relevant: the caller layer outranks AY's environment layer
        // (ay tune.rs:794-802), so an inherited value cannot reach this solve.
        // The NaN case is doubly dead: AY's `tune::parse_real` rejects
        // non-finite values and falls back to its compiled default
        // (tune.rs:989-1006), which is the process-abort fix that made NY's
        // finite-value pins stop being load-bearing.
        assert_eq!(solve_opts(10.0).engine().lattice(), Some(false));

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
            None,
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
        // NY writes no process environment for AY any more; the guards are
        // asserted on the value AY resolves from, above and in
        // `every_solve_opts_constructor_carries_the_guards`.
        assert!(std::env::var_os("AY_MILP_NO_LATTICE").is_none());
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

    /// Coordinator setup is part of an absolute budget. Simulate the clock
    /// advancing past the deadline immediately after a successful spawn: even
    /// an already queued worker result must be declined rather than receiving
    /// a newly sampled relative slice.
    #[test]
    fn absolute_deadline_resamples_after_injected_spawn_delay() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let worker_entered = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_flag = std::sync::Arc::clone(&worker_entered);
        let clock_flag = std::sync::Arc::clone(&worker_entered);
        let result = run_with_hard_deadline_at_clock(
            deadline,
            "test-post-spawn-delay",
            move || {
                worker_flag.store(true, std::sync::atomic::Ordering::Release);
                Ok(42usize)
            },
            move || {
                let wait_until = Instant::now() + Duration::from_secs(2);
                while !clock_flag.load(std::sync::atomic::Ordering::Acquire) {
                    assert!(
                        Instant::now() < wait_until,
                        "deadline clock was sampled before the worker spawned"
                    );
                    std::thread::yield_now();
                }
                deadline + Duration::from_millis(1)
            },
        );
        assert!(
            matches!(result, Ok(None)),
            "post-spawn coordinator delay must consume the absolute budget"
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
