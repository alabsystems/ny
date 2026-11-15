// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-attack-twin — exact-gradient projected PGD on a point-folded ATTACK
//! twin, in the post-BaB tail, for nets the exact loader leaves outside the
//! point-VJP fragment.
//!
//! # The gap this closes
//!
//! On `lsnc_relu` the exact loader fails four constant-only `MatMul` cones
//! closed (a rounded binary32 product is not the authored real product), so the
//! real graph carries `OpaqueSkip` nodes, every exact-gradient lane is
//! STATICALLY dead (`Attack lanes: … outside the exact-gradient fragment`) and
//! the post-BaB tail prints `DECLINED` with ~17 s of a 25 s budget unspent. The
//! field's counterexamples for 2026 rows 14/17/58 sit ON the `X_5 = +2` face
//! with margins ~1.5e-4 — a set random sampling essentially never hits (1.5 M
//! ORT samples found none) but that a projected gradient walks into, because
//! `∂Y_7/∂X_5 ≈ 1` pushes `X_5` onto its upper face.
//!
//! # The lane
//!
//! 1. [`ny_onnx::attack_twin::PointFoldedAttackTwin`] folds every constant cone
//!    ONCE through ONNX Runtime; the folded graph is inside the exact point-VJP
//!    fragment (widened by the exact linear `ReduceSum`). The twin is a distinct
//!    type with no route to any bound API.
//! 2. Per restart: pick a disjunct (clause), start uniformly in its box — every
//!    other restart snapped onto the face its DISTINCTIVE constraints' gradient
//!    points at (constraints shared by every clause, like a distributed top-level
//!    band, are not distinctive) — and ascend the hinge sum of the clause's
//!    unsatisfied constraint margins with Adam (per-coordinate step
//!    `STEP_FRACTION × width`, decay `LR_DECAY`), projected onto the box.
//! 3. Every iterate is evaluated by the trusted ONNX Runtime forward of the
//!    ORIGINAL model, and the search returns a point ONLY when that forward
//!    violates the property at zero tolerance (`property_violated_f64`).
//!
//! # Soundness
//!
//! Attack-only. The twin supplies DIRECTIONS; the original model's ORT forward
//! supplies every value that decides anything, and the caller re-runs the
//! unchanged `gate_sat_with_trusted_oracle` (ORT re-confirm + true-f64) before
//! a `sat` is published. The lane has no verdict-bearing return variant, so it
//! cannot produce `unsat`. It keeps no state across rows: no witness is carried
//! between instances (2025 and 2026 `lsnc_relu` share file names but not bands).

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{f32_to_f64_exact, f64_to_f32_down, f64_to_f32_up};
use ny_levers::decls::vnncomp_entry::ATTACK_POINT_FOLDED_TWIN;
use ny_onnx::attack_twin::PointFoldedAttackTwin;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};

use super::{
    build_search_box, clamp_inside_box, constraint_margin, format_smtlib_witness_f64,
    postbab_attack_window, postbab_restart_row_seed, property_violated_f64, refine_emit_view,
    SimpleRng,
};

/// Screening: every clause gets this many restarts (one uniform, one
/// face-snapped when it is 2) of [`SCREEN_STEPS`] steps before the cycle order
/// is fixed.
const SCREEN_ROUNDS: usize = 2;
/// Steps per screening restart.
const SCREEN_STEPS: usize = 30;
/// Steps per ordinary restart. The numpy prototype's clause-12 hits on rows
/// 14/17/58 landed at a median of 32-50 steps; 200 steps added almost nothing.
const RESTART_STEPS: usize = 100;
/// Adam learning rate as a fraction of each coordinate's box width — the
/// numpy prototype's tuned value, NOT alpha-beta-CROWN's lsnc recipe. abc's
/// `vnncomp25/lsnc_relu.yaml` sets `pgd_alpha: 0.01` with `pgd_alpha_scale`,
/// and `attack/attack_interface.py:104-108` turns that into a per-coordinate
/// step of `0.01 × width` with `use_adam = False` (sign-gradient PGD). Only
/// the width scaling is shared.
const STEP_FRACTION: f64 = 0.03;
/// Per-step learning-rate decay (alpha-beta-CROWN's `pgd_lr_decay`).
const LR_DECAY: f64 = 0.99;
const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;
const ADAM_EPS: f64 = 1e-12;
/// Consecutive gradient refusals, before ANY gradient succeeded, that retire
/// the lane as unavailable on this net.
const GRADIENT_UNAVAILABLE_AFTER: usize = 3;
/// A restart whose clause margin has not improved for this many steps has
/// converged (or is oscillating on a boundary it cannot cross): the next
/// restart is a better use of the window.
const STALL_STEPS: usize = 25;

/// The frozen admission of the twin lane (`attack.point_folded_twin`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PointFoldedTwinPolicy {
    /// Whether the lane may run in the post-BaB tail.
    pub(super) armed: bool,
    /// Which layer decided `armed` (the receipt's resolution source).
    pub(super) source: ny_levers::Source,
}

impl PointFoldedTwinPolicy {
    /// Resolve from the SAME frozen scored-run LeverSet as the flight receipt.
    pub(super) fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        let resolved = levers.resolved(&ATTACK_POINT_FOLDED_TWIN).ok_or_else(|| {
            format!(
                "{} is absent from the frozen scored-run LeverSet",
                ATTACK_POINT_FOLDED_TWIN.name
            )
        })?;
        match &resolved.value {
            ny_levers::LeverValue::Bool(armed) => Ok(Self {
                armed: *armed,
                source: resolved.source,
            }),
            other => Err(format!(
                "{} resolved to non-boolean value {other:?}",
                ATTACK_POINT_FOLDED_TWIN.name
            )),
        }
    }

    /// The lever's arming observable, printed once at scored-run entry when
    /// armed. The lane's own `[lsnc-attack-twin] armed folded=` line follows
    /// only when the post-BaB tail is reached on an out-of-fragment net.
    pub(super) fn announce(self) {
        if self.armed {
            eprintln!(
                "[lsnc-attack-twin] policy armed=1 source={:?} (post-BaB tail; engages only \
                 when the real net is outside the exact-gradient fragment)",
                self.source
            );
        }
    }
}

/// Run the twin lane in the post-BaB tail.
///
/// Returns the SMT-LIB witness of a point whose ORT forward on the ORIGINAL
/// model violates the property, or `None`. The caller re-gates it through
/// `gate_sat_with_trusted_oracle` before anything is published. With
/// `has_seed`, the ordinary tail has seeded stages of its own to run, so this
/// lane takes at most half of the window.
///
/// The search runs on up to [`MAX_WORKERS`] threads. Every helper worker builds
/// its OWN spec, twin and trusted forward from the two file paths, so nothing
/// but the paths and a stop flag crosses a thread boundary; the first worker to
/// confirm a violation raises the flag and the others return at their next
/// step.
pub(super) fn try_point_folded_twin_falsify(
    onnx: &Path,
    vnnlib: &Path,
    scored_deadline: Instant,
    has_seed: bool,
) -> Option<String> {
    let lane_start = Instant::now();
    let Some((window_deadline, window)) = postbab_attack_window(scored_deadline, lane_start) else {
        eprintln!(
            "[lsnc-attack-twin] armed folded=0 declined=no-window remaining_s={:.1}",
            scored_deadline
                .saturating_duration_since(lane_start)
                .as_secs_f64()
        );
        return None;
    };
    let deadline = if has_seed {
        lane_start
            .checked_add(window / 2)
            .unwrap_or(window_deadline)
            .min(window_deadline)
    } else {
        window_deadline
    };
    let mut lane = match LaneSetup::build(onnx, vnnlib) {
        Ok(lane) => lane,
        Err(reason) => {
            eprintln!("[lsnc-attack-twin] armed folded=0 declined={reason}");
            return None;
        }
    };
    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .clamp(1, MAX_WORKERS);
    let fragment = match lane.twin.first_gradient_refusal() {
        None => "admitted".to_string(),
        Some((node, layer)) => format!("refused({node}:{layer})"),
    };
    eprintln!(
        "[lsnc-attack-twin] armed folded={} frontier={} excised={} nodes={} fragment={fragment} \
         budget_s={:.1} seeded_tail={} workers={workers} setup_s={:.3}",
        lane.twin.folded_tensors(),
        lane.twin.frontier_tensors(),
        lane.twin.excised_constant_nodes(),
        lane.twin.node_count(),
        deadline.saturating_duration_since(lane_start).as_secs_f64(),
        u8::from(has_seed),
        lane_start.elapsed().as_secs_f64(),
    );
    if !lane.twin.supports_gradient() {
        eprintln!("[lsnc-attack-twin] done found=0 reason=outside-fragment");
        return None;
    }
    let stop = AtomicBool::new(false);
    let row_seed = postbab_restart_row_seed(onnx, vnnlib);
    let search_start = Instant::now();
    let results: Vec<(Option<Vec<f32>>, TwinPgdStats)> = std::thread::scope(|scope| {
        let helpers: Vec<_> = (1..workers)
            .map(|worker| {
                let stop = &stop;
                scope.spawn(move || match LaneSetup::build(onnx, vnnlib) {
                    Ok(mut helper) => helper.search(deadline, row_seed, worker, Some(stop)),
                    Err(_) => (
                        None,
                        TwinPgdStats {
                            stop: "worker-setup-failed",
                            ..TwinPgdStats::default()
                        },
                    ),
                })
            })
            .collect();
        let mut results = vec![lane.search(deadline, row_seed, 0, Some(&stop))];
        for helper in helpers {
            results.push(helper.join().unwrap_or_else(|_| {
                (
                    None,
                    TwinPgdStats {
                        stop: "worker-panicked",
                        ..TwinPgdStats::default()
                    },
                )
            }));
        }
        results
    });
    // Every worker's OWN stop reason, in worker order. The lane total must not
    // flatten them: a run whose workers all stopped `gradient-unavailable` or
    // `worker-setup-failed` is VOID, and printing it as `stop=deadline` would
    // read as "searched the whole window and found nothing" — a negative.
    let worker_stops: Vec<&'static str> = results.iter().map(|(_, stats)| stats.stop).collect();
    let mut total = TwinPgdStats::default();
    let mut found: Option<(usize, Vec<f32>)> = None;
    for (worker, (point, stats)) in results.into_iter().enumerate() {
        total.absorb(&stats);
        if found.is_none() {
            if let Some(point) = point {
                total.found_clause = stats.found_clause;
                total.found_restart = stats.found_restart;
                found = Some((worker, point));
            }
        }
    }
    total.stop = lane_stop(found.is_some(), &worker_stops);
    eprintln!(
        "[lsnc-attack-twin] done found={} finder={} workers={workers} {} worker_stops={} \
         secs={:.3}",
        u8::from(found.is_some()),
        found
            .as_ref()
            .map_or_else(|| "none".to_string(), |(worker, _)| worker.to_string()),
        total.receipt(),
        worker_stops.join(","),
        search_start.elapsed().as_secs_f64(),
    );
    let (_, found) = found?;
    let output = lane.forward.run(&found).ok()?;
    let found64 = refine_emit_view(&found, &lane.emit_pin);
    Some(format_smtlib_witness_f64(&found64, &output))
}

/// The lane's terminal `stop=` token from its workers' own stop reasons.
///
/// `found` wins. Otherwise a reason every worker shares is reported as is, and
/// disagreeing workers report `mixed` (the `worker_stops=` field then lists
/// each one). Never defaults to `deadline`: a lane whose workers could not get
/// a gradient or could not build is VOID, not a searched-and-empty negative.
fn lane_stop(found: bool, worker_stops: &[&'static str]) -> &'static str {
    if found {
        return "found";
    }
    match worker_stops.split_first() {
        Some((&first, rest)) if rest.iter().all(|&stop| stop == first) => first,
        Some(_) => "mixed",
        None => "no-worker",
    }
}

/// Upper bound on the lane's concurrent search workers. The post-BaB tail
/// runs after the verifier has returned, so these cores are otherwise idle;
/// the cap keeps a many-core host from paying one setup per core.
const MAX_WORKERS: usize = 4;

/// Everything one search worker owns: the row's spec and search box, its own
/// attack twin, and its own handle on the trusted forward.
struct LaneSetup {
    spec: VnnLibSpec,
    box_lo: Vec<f32>,
    box_hi: Vec<f32>,
    emit_pin: Vec<Option<f64>>,
    twin: PointFoldedAttackTwin,
    input_shape: Vec<usize>,
    forward: ny_onnx::diff::OrtForward,
}

impl LaneSetup {
    /// Build a worker from the row's files. The error is the observable's
    /// `declined=` reason.
    fn build(onnx: &Path, vnnlib: &Path) -> Result<Self, String> {
        let spec =
            ny_onnx::vnnlib::load_vnnlib(vnnlib).map_err(|error| format!("vnnlib ({error})"))?;
        let (box_lo, box_hi, emit_pin) =
            build_search_box(&spec).ok_or_else(|| "search-box".to_string())?;
        let dim = box_lo.len();
        // The SAME loader configuration the real graph is loaded with, so the
        // twin differs from it only in the point-folded constants.
        let load_config = ny_onnx::OnnxLoadConfig::default()
            .with_shape_infer_backend(crate::commands::cli_shape_infer_backend());
        let twin = PointFoldedAttackTwin::build(onnx, dim, &load_config)
            .map_err(|error| format!("build ({error})"))?;
        let (_, declared) = ny_onnx::diff::read_input_shape_maybe_gzip(onnx, dim)
            .map_err(|error| format!("input-shape ({error})"))?;
        let input_shape = twin.input_shape(&declared);
        let forward = ny_onnx::diff::OrtForward::from_path(onnx, dim)
            .map_err(|error| format!("trusted-forward ({error})"))?;
        Ok(Self {
            spec,
            box_lo,
            box_hi,
            emit_pin,
            twin,
            input_shape,
            forward,
        })
    }

    fn search(
        &mut self,
        deadline: Instant,
        row_seed: u64,
        worker: usize,
        stop: Option<&AtomicBool>,
    ) -> (Option<Vec<f32>>, TwinPgdStats) {
        let mut oracle = TwinOracle {
            twin: &self.twin,
            forward: &mut self.forward,
            input_shape: &self.input_shape,
            deadline,
        };
        twin_pgd_search(
            &mut oracle,
            &self.spec,
            &self.box_lo,
            &self.box_hi,
            &self.emit_pin,
            deadline,
            row_seed,
            worker,
            stop,
        )
    }
}

/// What the search needs from the outside world: a TRUSTED value oracle (the
/// original model) and a DIRECTION oracle (the attack twin).
pub(super) trait TwinPgdOracle {
    /// The trusted forward of the ORIGINAL model at `x`, as the exact f64 view
    /// of its binary32 outputs; `None` when it cannot be evaluated.
    fn trusted_outputs(&mut self, x: &[f32]) -> Option<Vec<f64>>;
    /// `d(row · y(x)) / dx` from the attack twin; `None` when there is no
    /// direction at this point.
    fn twin_gradient(&mut self, x: &[f32], row: &[f32]) -> Option<Vec<f32>>;
}

struct TwinOracle<'a> {
    twin: &'a PointFoldedAttackTwin,
    forward: &'a mut ny_onnx::diff::OrtForward,
    input_shape: &'a [usize],
    deadline: Instant,
}

impl TwinPgdOracle for TwinOracle<'_> {
    fn trusted_outputs(&mut self, x: &[f32]) -> Option<Vec<f64>> {
        let out = self.forward.run(x).ok()?;
        Some(out.iter().map(|&v| f32_to_f64_exact(v)).collect())
    }

    fn twin_gradient(&mut self, x: &[f32], row: &[f32]) -> Option<Vec<f32>> {
        let point = ArrayD::from_shape_vec(IxDyn(self.input_shape), x.to_vec()).ok()?;
        let row = Array2::from_shape_vec((1, row.len()), row.to_vec()).ok()?;
        let grad = self
            .twin
            .point_gradient(&point, &row, Some(self.deadline))
            .ok()??;
        let grad: Vec<f32> = grad.iter().copied().collect();
        (grad.len() == x.len() && grad.iter().all(|g| g.is_finite())).then_some(grad)
    }
}

/// Counters for the lane's terminal receipt line.
#[derive(Debug, Clone, Default)]
pub(super) struct TwinPgdStats {
    pub(super) clauses: usize,
    pub(super) restarts: usize,
    pub(super) face_restarts: usize,
    pub(super) steps: usize,
    pub(super) ort_evals: usize,
    pub(super) grad_evals: usize,
    pub(super) grad_failures: usize,
    pub(super) stalled_restarts: usize,
    pub(super) best_margin: Option<f64>,
    pub(super) best_clause: Option<usize>,
    pub(super) found_clause: Option<usize>,
    pub(super) found_restart: Option<usize>,
    pub(super) stop: &'static str,
}

impl TwinPgdStats {
    /// Fold one worker's counters into the lane total.
    fn absorb(&mut self, other: &Self) {
        self.clauses = self.clauses.max(other.clauses);
        self.restarts += other.restarts;
        self.face_restarts += other.face_restarts;
        self.stalled_restarts += other.stalled_restarts;
        self.steps += other.steps;
        self.ort_evals += other.ort_evals;
        self.grad_evals += other.grad_evals;
        self.grad_failures += other.grad_failures;
        if let Some(margin) = other.best_margin {
            if self.best_margin.is_none_or(|best| margin > best) {
                self.best_margin = Some(margin);
                self.best_clause = other.best_clause;
            }
        }
    }

    fn receipt(&self) -> String {
        let best_margin = self
            .best_margin
            .map_or_else(|| "none".to_string(), |m| format!("{m:+.6e}"));
        let clause = |c: Option<usize>| c.map_or_else(|| "none".to_string(), |c| c.to_string());
        format!(
            "clause={} restart={} clauses={} restarts={} face_restarts={} stalled_restarts={} \
             steps={} ort_evals={} grad_evals={} grad_failures={} best_margin={best_margin} \
             best_clause={} stop={}",
            clause(self.found_clause),
            clause(self.found_restart),
            self.clauses,
            self.restarts,
            self.face_restarts,
            self.stalled_restarts,
            self.steps,
            self.ort_evals,
            self.grad_evals,
            self.grad_failures,
            clause(self.best_clause),
            self.stop,
        )
    }
}

/// One disjunct of the property as the search sees it: its conjunction, the
/// constraints that distinguish it from the other disjuncts, and its box.
struct SearchClause {
    index: usize,
    constraints: Vec<OutputConstraint>,
    distinctive: Vec<OutputConstraint>,
    lo: Vec<f32>,
    hi: Vec<f32>,
}

/// The search's clause list. A top-level disjunction searches each disjunct
/// separately (inside its own input box, when it has one); a conjunction of
/// clauses, or a flat constraint list, is ONE clause.
fn search_clauses(spec: &VnnLibSpec, box_lo: &[f32], box_hi: &[f32]) -> Vec<SearchClause> {
    let groups: Vec<Vec<OutputConstraint>> = if spec.output_constraint_clauses.is_empty() {
        vec![spec.output_constraints.clone()]
    } else if spec.is_disjunction {
        spec.output_constraint_clauses.clone()
    } else {
        vec![spec
            .output_constraint_clauses
            .iter()
            .flatten()
            .cloned()
            .collect()]
    };
    let per_clause_boxes = spec.is_disjunction && !spec.per_clause_input_bounds.is_empty();
    let mut clauses = Vec::with_capacity(groups.len());
    for (index, constraints) in groups.iter().enumerate() {
        if constraints.is_empty() {
            continue;
        }
        let mut lo = box_lo.to_vec();
        let mut hi = box_hi.to_vec();
        if per_clause_boxes {
            if let Some(map) = spec.per_clause_input_bounds.get(index) {
                for (&dim, &(clause_lo, clause_hi)) in map {
                    if dim >= lo.len() {
                        continue;
                    }
                    // Inward rounding: the clause box stays inside its declared
                    // f64 box, as the global search box does.
                    lo[dim] = lo[dim].max(f64_to_f32_up(clause_lo));
                    hi[dim] = hi[dim].min(f64_to_f32_down(clause_hi));
                }
            }
            if lo.iter().zip(&hi).any(|(l, h)| l > h) {
                continue; // empty clause box: nothing to search
            }
        }
        // A constraint every disjunct carries (a distributed top-level
        // conjunct) cannot say which face THIS disjunct needs.
        let distinctive: Vec<OutputConstraint> = if groups.len() > 1 {
            constraints
                .iter()
                .filter(|c| !groups.iter().all(|group| group.contains(c)))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        let distinctive = if distinctive.is_empty() {
            constraints.clone()
        } else {
            distinctive
        };
        clauses.push(SearchClause {
            index,
            constraints: constraints.clone(),
            distinctive,
            lo,
            hi,
        });
    }
    clauses
}

/// Add `d margin(c) / dy` to `row`. `false` when the constraint has no margin
/// the search knows how to ascend (a future `OutputConstraint` variant).
fn add_margin_gradient(row: &mut [f32], constraint: &OutputConstraint) -> bool {
    use OutputConstraint as OC;
    let mut add = |index: usize, sign: f32| match row.get_mut(index) {
        Some(slot) => {
            *slot += sign;
            true
        }
        None => false,
    };
    match *constraint {
        // margin = y_j - y_i
        OC::LessEq(i, j) | OC::LessThan(i, j) => add(j, 1.0) & add(i, -1.0),
        // margin = y_i - y_j
        OC::GreaterEq(i, j) | OC::GreaterThan(i, j) => add(i, 1.0) & add(j, -1.0),
        // margin = k - y_i
        OC::LessEqConst(i, _) | OC::LessThanConst(i, _) => add(i, -1.0),
        // margin = y_i - k
        OC::GreaterEqConst(i, _) | OC::GreaterThanConst(i, _) => add(i, 1.0),
        _ => false,
    }
}

/// Whether a constraint still needs ascent at this margin (it is not yet
/// satisfied under the property's exact comparison semantics).
fn constraint_unsatisfied(constraint: &OutputConstraint, margin: f64) -> bool {
    if constraint.is_strict() {
        margin <= 0.0
    } else {
        margin < 0.0
    }
}

/// The per-restart start point: uniform in the clause box, optionally snapped
/// onto the face the clause's distinctive gradient points at.
fn restart_start(
    oracle: &mut impl TwinPgdOracle,
    clause: &SearchClause,
    num_outputs: usize,
    face: bool,
    rng: &mut SimpleRng,
    stats: &mut TwinPgdStats,
) -> Vec<f32> {
    let mut x: Vec<f32> = clause
        .lo
        .iter()
        .zip(&clause.hi)
        .map(|(&lo, &hi)| {
            let v = lo + rng.next_f32() * (hi - lo);
            v.clamp(lo, hi)
        })
        .collect();
    if !face || num_outputs == 0 {
        return x;
    }
    let mut row = vec![0.0_f32; num_outputs];
    let mut any = false;
    for constraint in &clause.distinctive {
        any |= add_margin_gradient(&mut row, constraint);
    }
    if !any || row.iter().all(|&r| r == 0.0) {
        return x;
    }
    let Some(grad) = oracle.twin_gradient(&x, &row) else {
        stats.grad_failures += 1;
        return x;
    };
    stats.grad_evals += 1;
    let best = grad
        .iter()
        .enumerate()
        .filter(|&(d, _)| clause.hi[d] > clause.lo[d])
        .map(|(d, &g)| {
            (
                d,
                f64::from(g.abs()) * f64::from(clause.hi[d] - clause.lo[d]),
            )
        })
        .filter(|&(_, score)| score > 0.0 && score.is_finite())
        .max_by(|a, b| a.1.total_cmp(&b.1));
    if let Some((d, _)) = best {
        x[d] = if grad[d] > 0.0 {
            clause.hi[d]
        } else {
            clause.lo[d]
        };
        stats.face_restarts += 1;
    }
    x
}

/// Projected Adam PGD over the clauses, until the deadline or an ORT-confirmed
/// violation. Returns the violating point (inside the global search box) and
/// the lane's counters.
///
/// Every point the search returns satisfies `property_violated_f64` on the
/// trusted outputs at the point it returns; nothing else leaves this function.
#[allow(clippy::too_many_arguments)]
pub(super) fn twin_pgd_search(
    oracle: &mut impl TwinPgdOracle,
    spec: &VnnLibSpec,
    box_lo: &[f32],
    box_hi: &[f32],
    emit_pin: &[Option<f64>],
    deadline: Instant,
    seed: u64,
    worker: usize,
    stop: Option<&AtomicBool>,
) -> (Option<Vec<f32>>, TwinPgdStats) {
    let clauses = search_clauses(spec, box_lo, box_hi);
    let mut stats = TwinPgdStats {
        clauses: clauses.len(),
        stop: "deadline",
        ..TwinPgdStats::default()
    };
    if clauses.is_empty() {
        stats.stop = "no-clause";
        return (None, stats);
    }
    // Workers draw independent streams and start their screening at different
    // clauses, so parallel workers cover the clause cycle rather than repeat it.
    let worker_salt = u64::try_from(worker)
        .unwrap_or(u64::MAX)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut rng = SimpleRng::new(seed ^ worker_salt ^ 0x7A1F_0DED_5EED_0001);
    let stopped = || stop.is_some_and(|flag| flag.load(Ordering::Relaxed));
    let mut best_by_clause = vec![f64::NEG_INFINITY; clauses.len()];
    // Cycle order, fixed after screening: most promising clause first. Every
    // clause keeps an equal share — the field's own attack spends equal effort
    // per disjunct, and a clause that never crosses in screening can still be
    // the only satisfiable one.
    let mut order: Vec<usize> = (0..clauses.len()).collect();
    let screen_restarts = SCREEN_ROUNDS * clauses.len();
    let mut restart = 0usize;
    loop {
        if Instant::now() >= deadline {
            break;
        }
        if stopped() {
            stats.stop = "stopped";
            break;
        }
        if restart == screen_restarts {
            order.sort_by(|&a, &b| best_by_clause[b].total_cmp(&best_by_clause[a]));
        }
        let (slot, steps) = if restart < screen_restarts {
            ((restart + worker) % clauses.len(), SCREEN_STEPS)
        } else {
            (
                order[(restart - screen_restarts) % clauses.len()],
                RESTART_STEPS,
            )
        };
        let clause = &clauses[slot];
        let face = restart % 2 == 1;
        stats.restarts += 1;
        let num_outputs = spec.num_outputs;
        let mut x = restart_start(oracle, clause, num_outputs, face, &mut rng, &mut stats);
        if stats.grad_evals == 0 && stats.grad_failures >= GRADIENT_UNAVAILABLE_AFTER {
            stats.stop = "gradient-unavailable";
            break;
        }
        let dims = x.len();
        let mut first_moment = vec![0.0_f64; dims];
        let mut second_moment = vec![0.0_f64; dims];
        let mut lr = STEP_FRACTION;
        let mut restart_best = f64::NEG_INFINITY;
        let mut last_improvement = 0usize;
        for step in 0..steps {
            if Instant::now() >= deadline || stopped() {
                break;
            }
            let Some(outputs) = oracle.trusted_outputs(&x) else {
                break;
            };
            stats.ort_evals += 1;
            stats.steps += 1;
            if property_violated_f64(spec, &refine_emit_view(&x, emit_pin), &outputs) {
                stats.found_clause = Some(clause.index);
                stats.found_restart = Some(stats.restarts - 1);
                stats.stop = "found";
                if let Some(flag) = stop {
                    flag.store(true, Ordering::Relaxed);
                }
                let found = clamp_inside_box(&x, box_lo, box_hi);
                return (found, stats);
            }
            let margins: Vec<f64> = clause
                .constraints
                .iter()
                .map(|c| constraint_margin(c, &outputs))
                .collect();
            let clause_margin = margins.iter().copied().fold(f64::INFINITY, f64::min);
            if clause_margin > best_by_clause[slot] {
                best_by_clause[slot] = clause_margin;
            }
            if stats.best_margin.is_none_or(|best| clause_margin > best) {
                stats.best_margin = Some(clause_margin);
                stats.best_clause = Some(clause.index);
            }
            if clause_margin > restart_best {
                restart_best = clause_margin;
                last_improvement = step;
            } else if step - last_improvement >= STALL_STEPS {
                stats.stalled_restarts += 1;
                break;
            }
            let mut row = vec![0.0_f32; outputs.len()];
            let mut active = false;
            for (constraint, &margin) in clause.constraints.iter().zip(&margins) {
                if constraint_unsatisfied(constraint, margin) {
                    active |= add_margin_gradient(&mut row, constraint);
                }
            }
            if !active || row.iter().all(|&r| r == 0.0) {
                break; // satisfied here yet not violated (e.g. a clause box): next restart
            }
            let Some(grad) = oracle.twin_gradient(&x, &row) else {
                stats.grad_failures += 1;
                break;
            };
            stats.grad_evals += 1;
            let t = i32::try_from(step + 1).unwrap_or(i32::MAX);
            let bias1 = 1.0 - ADAM_BETA1.powi(t);
            let bias2 = 1.0 - ADAM_BETA2.powi(t);
            let mut moved = false;
            for d in 0..dims {
                let (lo, hi) = (clause.lo[d], clause.hi[d]);
                if hi <= lo {
                    continue;
                }
                let g = f64::from(grad[d]);
                first_moment[d] = ADAM_BETA1 * first_moment[d] + (1.0 - ADAM_BETA1) * g;
                second_moment[d] = ADAM_BETA2 * second_moment[d] + (1.0 - ADAM_BETA2) * g * g;
                let m_hat = first_moment[d] / bias1;
                let v_hat = second_moment[d] / bias2;
                let width = f64::from(hi) - f64::from(lo);
                let delta = lr * width * m_hat / (v_hat.sqrt() + ADAM_EPS);
                if !delta.is_finite() {
                    continue;
                }
                let next = ((f64::from(x[d]) + delta) as f32).clamp(lo, hi);
                if next != x[d] {
                    moved = true;
                    x[d] = next;
                }
            }
            if !moved {
                break; // pinned on the faces along every ascent coordinate
            }
            lr *= LR_DECAY;
        }
        restart += 1;
    }
    (None, stats)
}

#[cfg(test)]
mod tests;
