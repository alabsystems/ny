// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// MIP solver wrapper for neural network verification: lowers the
// solver-neutral IR to the ay backend at solve time (SOLVER POLICY:
// docs/SOLVER_POLICY.md — ay is the only solver in ny; HiGHS was deleted
// at LG3 once verified ay certificates replaced it as the independent
// cross-check).
//
// Two use cases:
// 1. Complete verification: check feasibility of constrained region
//    (SAT = counterexample exists, UNSAT = property verified)
// 2. LP bound tightening: minimize/maximize neuron values subject to
//    network constraints (see `tighten`)

use crate::config::{MipBackend, MipConfig};
use crate::encoder::MipParts;
use crate::error::MipError;
use crate::ir;

use std::collections::HashSet;

type Result<T> = std::result::Result<T, MipError>;

/// Optimization direction for [`MipSolver::minimize_output`] /
/// [`MipSolver::maximize_output`].
///
/// Owned by ny-mip (not a re-export of a solver crate's type) so the public
/// API is backend-neutral.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sense {
    /// Minimize the target column.
    Minimise,
    /// Maximize the target column.
    Maximise,
}

/// Result of a MIP/LP solve.
#[derive(Debug)]
pub enum MipResult {
    /// Feasible solution found (counterexample exists).
    Sat {
        /// Objective value AT THE RETURNED POINT: the proven optimum for a
        /// completed optimization; the exactly-feasible INCUMBENT's objective
        /// for an interrupted one (ay `Outcome::Feasible { incumbent_only:
        /// true, .. }` — a better point may exist); 0.0 for pure feasibility
        /// checks. NEVER use this as a bound on the optimum — that is what
        /// [`dual_bound`](Self::Sat::dual_bound) is for.
        objective: f64,
        /// Output neuron values extracted from the solution.
        output_values: Vec<f64>,
        /// Input values from the solution (the counterexample inputs).
        input_values: Vec<f64>,
        /// RIGOROUS dual bound on the true optimum (a lower bound for
        /// Minimize, an upper bound for Maximize), rounded OUTWARD to f64 so
        /// the float never over-claims. `Some` only when the bound is
        /// rigorous (ay contract property 3): the exact optimum of a
        /// completed solve, or `Outcome::Feasible`'s Neumaier–Shcherbina /
        /// exact interrupted-tree bound. `None` for feasibility checks, the
        /// subprocess lane, and any non-rigorous or unavailable bound.
        /// Callers may prune / tighten on it directly.
        dual_bound: Option<f64>,
    },
    /// Proven infeasible (property verified). `certified` records that an
    /// independent exact certificate (Farkas or case-split) was verified at
    /// the backend seam (LG3, ay repo designs/2026-07-12-ay-as-library-for-ny.md).
    Unsat {
        /// Whether verified certificate evidence accompanied the verdict.
        certified: bool,
    },
    /// Solver timed out or hit iteration limit.
    Timeout,
    /// Solver error.
    Error(String),
}

/// Cap on the number of binaries fixed by phase-split racing: 2^4 = 16
/// subproblems, matching the core counts this targets (designs/scip.md).
const MAX_SPLIT_K: usize = 4;

/// Exact opt-in gate for the NeuralSAT-style near-stable ReLU ordering.
///
/// Only the literal value `1` enables the canary. Unset, non-Unicode, and all
/// other values preserve the historical widest-first ordering exactly.
fn mip_stability_hints_enabled_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

fn mip_stability_hints_enabled() -> bool {
    mip_stability_hints_enabled_from_value(std::env::var("NY_MIP_STABILITY_HINTS").ok().as_deref())
}

/// Private, once-resolved AY hint-consumption state. There is deliberately no
/// public/programmatic enable: only the exact environment canary may make the
/// search advice live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AyBranchHintState {
    Disabled,
    Enabled,
}

fn ay_branch_hint_state_from_value(value: Option<&str>) -> AyBranchHintState {
    if value == Some("1") {
        AyBranchHintState::Enabled
    } else {
        AyBranchHintState::Disabled
    }
}

fn resolved_ay_branch_hint_state() -> AyBranchHintState {
    ay_branch_hint_state_from_value(std::env::var("NY_AY_BRANCH_HINTS").ok().as_deref())
}

/// Recover `min(-l, u)` for each unstable ReLU from the canonical Big-M rows.
///
/// This mirrors NeuralSAT's `largest=False` tightening candidate score: a
/// small value means one phase boundary is close and is therefore a promising
/// early decision. Reference: `Verified-Intelligence/NeuralSAT`,
/// `src/tightener/cpu_tightener.py` (`d96e64a5a9755dcd9059a5bd7e3d0b0537e26451`,
/// audited 2026-07-22). Recovery is deliberately fail-closed as an
/// optimization: only the exact encoder row pair
///
/// * `y - x - l*z <= -l`, and
/// * `y - u*z <= 0`
///
/// with a shared `y` is accepted. Missing, duplicate-conflicting, or altered
/// rows yield `None`, which the canary ranks after recovered ReLUs using the
/// historical width key. This metadata can only move search order; it never
/// changes the model, the exhaustive phase partition, or verdict admission.
fn relu_stability_scores(problem: &ir::MilpProblem, binary_vars: &[ir::Col]) -> Vec<Option<f64>> {
    fn record_candidate(
        slot: &mut Option<(usize, f64)>,
        candidate: (usize, f64),
        ambiguous: &mut bool,
    ) {
        match *slot {
            None => *slot = Some(candidate),
            Some(existing)
                if existing.0 == candidate.0 && existing.1.to_bits() == candidate.1.to_bits() => {}
            Some(_) => *ambiguous = true,
        }
    }

    // Dense column lookup keeps extraction O(rows + nnz + binaries), avoiding
    // a binary-by-row scan on large Graph-MIP models.
    let mut binary_index = vec![None; problem.num_cols()];
    let mut ambiguous = vec![false; binary_vars.len()];
    for (index, &col) in binary_vars.iter().enumerate() {
        let Some(slot) = binary_index.get_mut(col.0) else {
            ambiguous[index] = true;
            continue;
        };
        if let Some(previous) = *slot {
            ambiguous[previous] = true;
            ambiguous[index] = true;
        } else {
            *slot = Some(index);
        }
    }

    // `(y column, magnitude)` for `-l` and `u`, respectively.
    let mut lower_magnitudes = vec![None; binary_vars.len()];
    let mut upper_magnitudes = vec![None; binary_vars.len()];

    for row in problem.rows() {
        if row.lb != f64::NEG_INFINITY || !row.ub.is_finite() {
            continue;
        }

        let mut tracked = None;
        let mut multiple_tracked = false;
        for &(col, weight) in &row.coeffs {
            let Some(Some(index)) = binary_index.get(col) else {
                continue;
            };
            if tracked.is_some() {
                multiple_tracked = true;
                break;
            }
            tracked = Some((*index, col, weight));
        }
        let Some((index, z_col, z_weight)) = tracked else {
            continue;
        };
        if multiple_tracked || !z_weight.is_finite() {
            continue;
        }

        // y - x - l*z <= -l: three exact nonzero terms, `-l > 0`.
        if row.coeffs.len() == 3 && z_weight > 0.0 && z_weight.to_bits() == row.ub.to_bits() {
            let mut y_col = None;
            let mut saw_minus_one = false;
            let mut canonical = true;
            for &(col, weight) in &row.coeffs {
                if col == z_col {
                    continue;
                }
                if weight == 1.0 && y_col.is_none() {
                    y_col = Some(col);
                } else if weight == -1.0 && !saw_minus_one {
                    saw_minus_one = true;
                } else {
                    canonical = false;
                }
            }
            if canonical && saw_minus_one {
                if let Some(y_col) = y_col {
                    record_candidate(
                        &mut lower_magnitudes[index],
                        (y_col, z_weight),
                        &mut ambiguous[index],
                    );
                }
            }
            continue;
        }

        // y - u*z <= 0: two exact nonzero terms, `u > 0`.
        if row.coeffs.len() == 2 && row.ub == 0.0 && z_weight < 0.0 {
            let y_col = row
                .coeffs
                .iter()
                .find_map(|&(col, weight)| (col != z_col && weight == 1.0).then_some(col));
            if let Some(y_col) = y_col {
                record_candidate(
                    &mut upper_magnitudes[index],
                    (y_col, -z_weight),
                    &mut ambiguous[index],
                );
            }
        }
    }

    lower_magnitudes
        .into_iter()
        .zip(upper_magnitudes)
        .enumerate()
        .map(|(index, (lower, upper))| {
            if ambiguous[index] {
                return None;
            }
            let ((lower_y, lower), (upper_y, upper)) = (lower?, upper?);
            if lower_y != upper_y || !lower.is_finite() || !upper.is_finite() {
                return None;
            }
            let score = lower.min(upper);
            (score > 0.0).then_some(score)
        })
        .collect()
}

/// Fingerprint identifying the exact problem a [`SplitUnsatCache`] memo is
/// valid for: `(split_cols, num_subproblems, num_rows, num_cols,
/// ir_content_hash)`.
type SplitFingerprint = (Vec<ir::Col>, usize, usize, usize, u64);

/// Cheap deterministic content hash over the FULL IR: every column bound /
/// objective / integrality flag and every row bound / sparse coefficient
/// triplet, hashed via f64 bit patterns (`to_bits`: NaN-safe, and
/// `-0.0 != 0.0` — over-sensitivity is the fail-closed direction). Any change
/// to the encoded problem changes the hash, which clears the memo.
///
/// `DefaultHasher::new()` is deterministic (fixed keys), so equal IRs hash
/// equal across calls within a process — all the memo needs.
fn ir_content_hash(problem: &ir::MilpProblem) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut h = DefaultHasher::new();
    problem.num_cols().hash(&mut h);
    for col in problem.cols() {
        col.lb.to_bits().hash(&mut h);
        col.ub.to_bits().hash(&mut h);
        col.obj.to_bits().hash(&mut h);
        col.integer.hash(&mut h);
    }
    problem.num_rows().hash(&mut h);
    for row in problem.rows() {
        row.lb.to_bits().hash(&mut h);
        row.ub.to_bits().hash(&mut h);
        row.coeffs.len().hash(&mut h);
        for &(col_idx, weight) in &row.coeffs {
            col_idx.hash(&mut h);
            weight.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

/// Certified-UNSAT memo for phase-split racing across REPEATED solves of the
/// same problem.
///
/// ny-cli's disjunctive MIP path re-solves a timed-out clause in a later
/// round with a larger slice; without the memo, a race abandoned at "15 of 16
/// subproblem results" throws all 15 certified-Unsat proofs away and the
/// retry starts from zero. With it, the retry pre-seeds the proven
/// assignments and spends its whole slice on the still-open ones.
///
/// SOUNDNESS (fail-closed): only `Unsat { certified: true }` sub-verdicts are
/// recorded, and they are replayed only when the memo's fingerprint — split
/// columns, subproblem count, IR shape, and a content hash over every bound /
/// coefficient of the IR — matches the new problem EXACTLY. Any drift
/// (different clause, re-encoded bounds, changed split plan) clears the memo.
/// Sat, uncertified Unsat, Timeout, and Error results are never cached.
#[derive(Debug, Default)]
pub struct SplitUnsatCache {
    /// Identity of the problem the `proven` set is valid for; `None` until
    /// the first race reconciles.
    fingerprint: Option<SplitFingerprint>,
    /// Assignments (fixed-binary bit patterns in `0..num_subproblems`) proven
    /// `Unsat { certified: true }` for the fingerprinted problem.
    proven: HashSet<usize>,
}

impl SplitUnsatCache {
    /// Reconcile the memo with the problem about to be raced: on ANY
    /// fingerprint mismatch (including a fresh `None`) the proven set is
    /// cleared BEFORE the new fingerprint is adopted — fail-closed.
    fn reconcile(&mut self, fingerprint: SplitFingerprint) {
        if self.fingerprint.as_ref() != Some(&fingerprint) {
            self.proven.clear();
            self.fingerprint = Some(fingerprint);
        }
    }

    /// Whether `assignment` is already proven certified-Unsat for the
    /// fingerprinted problem.
    fn is_proven(&self, assignment: usize) -> bool {
        self.proven.contains(&assignment)
    }

    /// Record a sub-verdict: ONLY `Unsat { certified: true }` is memoized;
    /// Sat, uncertified Unsat, Timeout, and Error are ignored.
    fn record(&mut self, assignment: usize, result: &MipResult) {
        if matches!(result, MipResult::Unsat { certified: true }) {
            self.proven.insert(assignment);
        }
    }
}

/// Aggregate phase-split subproblem results into the parent verdict.
///
/// SOUNDNESS CONTRACT (designs/scip.md Phase C): the subproblems exactly
/// partition the parent's feasible set, so:
/// - any `Sat` is a feasible parent point -> return it (first wins);
/// - `Unsat` requires EXACTLY `expected` Unsat sub-verdicts — a missing,
///   Timeout, or Error sub-result forces `Timeout`, never `Unsat`.
fn aggregate_split_results(results: Vec<MipResult>, expected: usize) -> MipResult {
    let total = results.len();
    let mut num_unsat = 0usize;
    let mut all_certified = true;
    for result in results {
        match result {
            sat @ MipResult::Sat { .. } => return sat,
            MipResult::Unsat { certified } => {
                num_unsat += 1;
                all_certified &= certified;
            }
            MipResult::Timeout => {}
            MipResult::Error(e) => {
                tracing::warn!("phase-split subproblem error (treated as timeout): {e}");
            }
        }
    }
    if num_unsat == expected && total == expected {
        // Certified only when EVERY split carried verified evidence (the
        // full cross-split partition certificate is assembled inside
        // ay-milp's native racing lane; this flag reports per-split
        // verification for ny's own thread racing).
        MipResult::Unsat {
            certified: all_certified,
        }
    } else {
        MipResult::Timeout
    }
}

/// Status breakdown for one phase-split race.
///
/// A worker that returns `Timeout` or `Error` has produced a channel message,
/// but it has not closed its partition.  Keep those outcomes separate from
/// certified UNSAT so deadline telemetry cannot make an inconclusive race look
/// closer to a proof than it is.
#[derive(Debug, Default, PartialEq, Eq)]
struct SplitStatusCounts {
    certified_unsat: usize,
    uncertified_unsat: usize,
    sat: usize,
    timeout: usize,
    error: usize,
    missing: usize,
}

fn split_status_counts(slots: &[Option<MipResult>]) -> SplitStatusCounts {
    let mut counts = SplitStatusCounts::default();
    for slot in slots {
        match slot {
            Some(MipResult::Unsat { certified: true }) => counts.certified_unsat += 1,
            Some(MipResult::Unsat { certified: false }) => counts.uncertified_unsat += 1,
            Some(MipResult::Sat { .. }) => counts.sat += 1,
            Some(MipResult::Timeout) => counts.timeout += 1,
            Some(MipResult::Error(_)) => counts.error += 1,
            None => counts.missing += 1,
        }
    }
    counts
}

/// MIP solver for neural network verification.
///
/// Wraps an encoded MILP IR (from `MipEncoder`) with solver configuration.
pub struct MipSolver {
    parts: MipParts,
    config: MipConfig,
    ay_branch_hints: AyBranchHintState,
}

impl MipSolver {
    /// Create a solver from encoded network parts and configuration.
    pub fn new(parts: MipParts, config: MipConfig) -> Self {
        // Startup policy: resolve the exact environment canary once, before
        // any serial solve or detached split worker is created. Later process
        // environment mutation cannot change this solver's search policy.
        let ay_branch_hints = resolved_ay_branch_hint_state();
        Self {
            parts,
            config,
            ay_branch_hints,
        }
    }

    /// Check feasibility: is the constrained region non-empty?
    ///
    /// For complete verification with output property negation:
    /// - SAT means a counterexample exists (property violated)
    /// - UNSAT means no counterexample (property verified)
    ///
    /// Uses a trivial objective (minimize 0) since we only care about
    /// feasibility, not optimization.
    pub fn check_feasibility(&self) -> Result<MipResult> {
        self.check_feasibility_with_warm_start(None)
    }

    /// Check feasibility with an optional warm-start primal solution.
    ///
    /// If `warm_start_cols` is provided, the solver attempts to seed the
    /// backend with the primal column values before solving. If the backend
    /// rejects the seed (e.g., wrong length, infeasible point), the solve
    /// proceeds cold — warm-starting is a performance hint, not a correctness
    /// requirement.
    ///
    /// When phase-split racing is enabled (`MipConfig::parallel_split`, the
    /// default) and the problem has unstable ReLU binaries, the check races
    /// the 2^k fixed-prefix subproblems across threads (designs/scip.md
    /// Phase C); otherwise it is a single serial solve.
    ///
    /// Part of #3865: PGD-to-HiGHS warm start.
    pub fn check_feasibility_with_warm_start(
        &self,
        warm_start_cols: Option<&[f64]>,
    ) -> Result<MipResult> {
        match self.split_plan() {
            Some(split_cols) => self.check_feasibility_split(&split_cols, warm_start_cols, None),
            None => self.solve_ir(&self.parts.problem, warm_start_cols),
        }
    }

    /// Check feasibility with a certified-UNSAT phase-split memo (and an
    /// optional warm start), for callers that re-solve the SAME problem with
    /// a growing time slice (ny-cli's multi-round disjunctive clause
    /// schedule).
    ///
    /// Subproblems the memo has already proven `Unsat { certified: true }`
    /// are pre-seeded and not re-solved; everything else runs exactly as
    /// [`Self::check_feasibility_with_warm_start`]. The memo is keyed by a
    /// full problem fingerprint and clears itself on ANY drift (fail-closed)
    /// — see [`SplitUnsatCache`].
    pub fn check_feasibility_cached(
        &self,
        warm_start_cols: Option<&[f64]>,
        cache: &mut SplitUnsatCache,
    ) -> Result<MipResult> {
        match self.split_plan() {
            Some(split_cols) => {
                self.check_feasibility_split(&split_cols, warm_start_cols, Some(cache))
            }
            None => self.solve_ir(&self.parts.problem, warm_start_cols),
        }
    }

    /// Solve one concrete IR instance on the configured backend.
    fn solve_ir(
        &self,
        problem: &ir::MilpProblem,
        warm_start_cols: Option<&[f64]>,
    ) -> Result<MipResult> {
        crate::dump::maybe_dump(problem);
        match self.config.backend {
            MipBackend::Ay => crate::ay_lib::check_feasibility(
                problem,
                self.config.timeout_secs,
                &self.parts.input_vars,
                &self.parts.output_vars,
                warm_start_cols,
                &self.ay_branch_hint_order(),
            ),
            MipBackend::AyProc => crate::ay::check_feasibility(
                problem,
                self.config.timeout_secs,
                &self.parts.input_vars,
                &self.parts.output_vars,
                warm_start_cols,
            ),
        }
    }

    /// All ReLU indicator binaries ranked by the selected advice policy. The
    /// default is historical DESCENDING pre-activation width (widest first).
    /// `NY_MIP_STABILITY_HINTS=1` opts into ASCENDING `min(-l, u)`
    /// (closest-to-stable first), following NeuralSAT's exact-MIP tightening
    /// candidate order. The native branch-and-cut engine takes this as branch
    /// hints (P3: advice only, verdicts and certificates unchanged; only
    /// search order moves). Same ranking key as [`Self::split_plan`], but the
    /// full order rather than the top-k phase-split prefix.
    fn branch_hint_order(&self) -> Vec<ir::Col> {
        self.ranked_binaries()
            .into_iter()
            .map(|i| self.parts.binary_vars[i])
            .collect()
    }

    /// AY-facing hint payload. Disabled is the allocation-free empty vector,
    /// so the backend does not call `BabSession::hint_branch_order` and AY
    /// takes its historical unhinted entrypoint. Ranking metadata is not even
    /// recovered until the exact environment gate is live.
    fn ay_branch_hint_order(&self) -> Vec<ir::Col> {
        self.ay_branch_hint_order_with_gate(self.ay_branch_hints == AyBranchHintState::Enabled)
    }

    fn ay_branch_hint_order_with_gate(&self, enabled: bool) -> Vec<ir::Col> {
        if enabled {
            self.branch_hint_order()
        } else {
            Vec::new()
        }
    }

    /// Indices into `binary_vars`, ordered by the active advice policy.
    fn ranked_binaries(&self) -> Vec<usize> {
        self.ranked_binaries_with_stability(mip_stability_hints_enabled())
    }

    fn ranked_binaries_with_stability(&self, stability_hints: bool) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.parts.binary_vars.len()).collect();

        // Keep this disabled branch identical to the historical comparator:
        // unset/default behavior must not move even a tie.
        if !stability_hints {
            order.sort_by(|&a, &b| {
                let wa = self.parts.binary_widths.get(a).copied().unwrap_or(0.0);
                let wb = self.parts.binary_widths.get(b).copied().unwrap_or(0.0);
                wb.partial_cmp(&wa)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            return order;
        }

        let scores = relu_stability_scores(&self.parts.problem, &self.parts.binary_vars);
        order.sort_by(|&a, &b| {
            let wa = self.parts.binary_widths.get(a).copied().unwrap_or(0.0);
            let wb = self.parts.binary_widths.get(b).copied().unwrap_or(0.0);
            match (
                scores.get(a).copied().flatten(),
                scores.get(b).copied().flatten(),
            ) {
                (Some(sa), Some(sb)) => sa
                    .partial_cmp(&sb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal))
                    .then(a.cmp(&b)),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => wb
                    .partial_cmp(&wa)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b)),
            }
        });
        order
    }

    /// Decide the phase-split branching set, or `None` for a serial solve.
    ///
    /// Splits on the first k unstable ReLU indicator binaries under the active
    /// advice policy: widest-first by default, closest-to-stable first under
    /// `NY_MIP_STABILITY_HINTS=1`. Here `k = ceil(log2(threads))` capped at
    /// [`MAX_SPLIT_K`] and at the number of available binaries.
    /// designs/scip.md Phase C.
    fn split_plan(&self) -> Option<Vec<ir::Col>> {
        let threads = match self.config.parallel_split {
            0 => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            n => n,
        };
        if threads <= 1 {
            return None; // parallel_split = 1 is the explicit disable path
        }
        let num_binaries = self.parts.binary_vars.len();
        if num_binaries == 0 {
            return None; // pure LP: nothing to split on
        }
        // ceil(log2(threads)), then cap.
        let k = (usize::BITS - (threads - 1).leading_zeros()) as usize;
        let k = k.min(MAX_SPLIT_K).min(num_binaries);
        if k == 0 {
            return None;
        }

        // Rank binaries using the advice policy and take the top k.
        let order = self.ranked_binaries();
        Some(
            order[..k]
                .iter()
                .map(|&i| self.parts.binary_vars[i])
                .collect(),
        )
    }

    /// Phase-split parallel racing (designs/scip.md Phase C).
    ///
    /// Enumerates all 2^k assignments of the chosen ReLU indicator binaries,
    /// clones the IR per assignment with those binaries FIXED (bounds pinned
    /// to the assignment value), and solves the subproblems concurrently —
    /// one solver model per thread, built from the Send+Sync IR. Each
    /// subproblem gets the full remaining time limit (they run concurrently).
    ///
    /// SOUNDNESS: `{0,1}^k` exactly partitions the binary space, so the union
    /// of the subproblems' feasible sets equals the parent's. Any Sat is a
    /// feasible point of the parent (witness still revalidated downstream);
    /// Unsat requires ALL 2^k subproblems Unsat; any Timeout/Error without a
    /// Sat aggregates to Timeout — never Unsat. A subproblem whose result has
    /// not arrived by the slice deadline counts as Timeout — never Unsat.
    ///
    /// SLICE ENFORCEMENT (vnncomp timeout arc, 2026-07-18): the workers are
    /// DETACHED threads and the joins are deadline-bounded `recv_timeout`s on
    /// a result channel, so one hung backend solve (the ay SMT-fallback
    /// overshoot — see `ay_lib::run_with_hard_deadline`) can never stall this
    /// call past `timeout_secs`. This outer deadline is deliberately
    /// REDUNDANT with the per-solve enforcement inside both backends (ay_lib
    /// wrapper / AyProc process kill): a regression in either seam leaves the
    /// slice still enforced here. Abandoned workers keep running detached —
    /// accepted cost, bounded by the backends' own deadlines and ultimately
    /// by process teardown.
    ///
    /// CERTIFIED-UNSAT MEMO (`cache`): when a [`SplitUnsatCache`] is
    /// supplied, subproblems previously proven `Unsat { certified: true }`
    /// for this EXACT problem (full fingerprint match, see
    /// [`SplitUnsatCache`]) are pre-seeded instead of re-solved, so a re-race
    /// after a slice timeout only spends its budget on the still-open
    /// assignments. Only certified Unsat is ever recorded — Sat, uncertified
    /// Unsat, Timeout, and Error never are — and recording happens only after
    /// the race, so the memo can never influence the round that produced it.
    fn check_feasibility_split(
        &self,
        split_cols: &[ir::Col],
        warm_start_cols: Option<&[f64]>,
        cache: Option<&mut SplitUnsatCache>,
    ) -> Result<MipResult> {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        let k = split_cols.len();
        let num_subproblems = 1usize << k;
        tracing::info!(
            "MIP phase-split racing: fixing {k} widest ReLU binaries -> {num_subproblems} \
             concurrent subproblems ({:?} backend)",
            self.config.backend
        );

        // Every subproblem gets the full slice (they run concurrently), and
        // the whole race is bounded by the same slice from OUTSIDE the
        // backends. Clamp mirrors the backends' own budget clamp.
        let deadline = Instant::now()
            + Duration::from_secs_f64(self.config.timeout_secs.clamp(0.001, 86_400.0));

        // Owned context for the detached workers.
        let backend = self.config.backend;
        let timeout_secs = self.config.timeout_secs;
        let input_vars = std::sync::Arc::new(self.parts.input_vars.clone());
        let output_vars = std::sync::Arc::new(self.parts.output_vars.clone());
        let branch_hints = std::sync::Arc::new(self.ay_branch_hint_order());
        let warm_start = std::sync::Arc::new(warm_start_cols.map(<[f64]>::to_vec));

        // Certified-UNSAT memo (fail-closed): reconcile clears the proven set
        // unless the fingerprint — split set, subproblem count, IR shape, and
        // the full IR content hash — matches this exact problem.
        let mut cache = cache;
        if let Some(cache) = cache.as_deref_mut() {
            cache.reconcile((
                split_cols.to_vec(),
                num_subproblems,
                self.parts.problem.num_rows(),
                self.parts.problem.num_cols(),
                ir_content_hash(&self.parts.problem),
            ));
        }

        // One result slot per assignment (indexed by the fixed-binary bit
        // pattern). Assignments the memo already proved Unsat{certified:true}
        // for this exact problem are PRE-SEEDED and their workers are never
        // spawned.
        let mut slots: Vec<Option<MipResult>> = (0..num_subproblems)
            .map(|assignment| {
                cache
                    .as_ref()
                    .is_some_and(|cache| cache.is_proven(assignment))
                    .then_some(MipResult::Unsat { certified: true })
            })
            .collect();
        let num_preseeded = slots.iter().filter(|slot| slot.is_some()).count();
        if num_preseeded > 0 {
            tracing::info!(
                "phase-split memo: {num_preseeded} of {num_subproblems} subproblems already \
                 proven Unsat (certified) for this exact problem; racing only the rest"
            );
        }

        let (tx, rx) = mpsc::channel::<(usize, MipResult)>();
        for assignment in 0..num_subproblems {
            if slots[assignment].is_some() {
                continue; // pre-seeded certified Unsat: nothing to solve
            }
            let mut sub = self.parts.problem.clone();
            for (bit, &col) in split_cols.iter().enumerate() {
                sub.fix_col(col, ((assignment >> bit) & 1) as f64);
            }
            let tx = tx.clone();
            let input_vars = std::sync::Arc::clone(&input_vars);
            let output_vars = std::sync::Arc::clone(&output_vars);
            let branch_hints = std::sync::Arc::clone(&branch_hints);
            let warm_start = std::sync::Arc::clone(&warm_start);
            std::thread::Builder::new()
                .name(format!("ny-mip-split-{assignment}"))
                .spawn(move || {
                    crate::dump::maybe_dump(&sub);
                    let result = match backend {
                        MipBackend::Ay => crate::ay_lib::check_feasibility(
                            &sub,
                            timeout_secs,
                            &input_vars,
                            &output_vars,
                            warm_start.as_deref(),
                            &branch_hints,
                        ),
                        MipBackend::AyProc => crate::ay::check_feasibility(
                            &sub,
                            timeout_secs,
                            &input_vars,
                            &output_vars,
                            warm_start.as_deref(),
                        ),
                    };
                    // A receiver gone after the deadline makes this send
                    // fail; expected for an abandoned race.
                    let _ = tx.send((
                        assignment,
                        match result {
                            Ok(result) => result,
                            Err(e) => MipResult::Error(format!("subproblem solve failed: {e}")),
                        },
                    ));
                })
                .map_err(|e| {
                    MipError::Solver(format!("spawning phase-split worker {assignment}: {e}"))
                })?;
        }
        drop(tx); // workers hold the remaining senders

        let mut num_results = num_preseeded;
        while num_results < num_subproblems {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(remaining) {
                Ok((assignment, result)) => {
                    let slot = &mut slots[assignment];
                    // One worker per assignment, and pre-seeded slots never
                    // spawn one, so a filled slot is unreachable; keep the
                    // first result if it ever happens (never double-count).
                    debug_assert!(
                        slot.is_none(),
                        "duplicate result for assignment {assignment}"
                    );
                    if slot.is_none() {
                        *slot = Some(result);
                        num_results += 1;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let statuses = split_status_counts(&slots);
                    tracing::warn!(
                        "phase-split race hit the {timeout_secs}s slice deadline with \
                         {num_results} of {num_subproblems} worker replies; \
                         certified_unsat={} uncertified_unsat={} sat={} timeout={} error={} \
                         missing={}; abandoning the rest",
                        statuses.certified_unsat,
                        statuses.uncertified_unsat,
                        statuses.sat,
                        statuses.timeout,
                        statuses.error,
                        statuses.missing,
                    );
                    break;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Every remaining worker exited without sending (panic).
                    tracing::warn!(
                        "phase-split workers exited without results ({num_results} of \
                         {num_subproblems} collected)"
                    );
                    break;
                }
            }
        }

        let statuses = split_status_counts(&slots);
        tracing::info!(
            "phase-split race status: certified_unsat={} uncertified_unsat={} sat={} \
             timeout={} error={} missing={} total={num_subproblems}",
            statuses.certified_unsat,
            statuses.uncertified_unsat,
            statuses.sat,
            statuses.timeout,
            statuses.error,
            statuses.missing,
        );

        // Memoize ONLY certified Unsat sub-verdicts — proofs about this exact
        // fingerprinted subproblem, replayable verbatim on a re-solve. Sat,
        // uncertified Unsat, Timeout, and Error are NEVER cached. Recording
        // happens strictly AFTER the race: the memo cannot influence the
        // round that produced it.
        if let Some(cache) = cache {
            for (assignment, slot) in slots.iter().enumerate() {
                if let Some(result) = slot {
                    cache.record(assignment, result);
                }
            }
        }

        // Missing results count as Timeout in the aggregation — never Unsat.
        let results: Vec<MipResult> = slots
            .into_iter()
            .map(|slot| slot.unwrap_or(MipResult::Timeout))
            .collect();

        Ok(aggregate_split_results(results, num_subproblems))
    }

    /// Minimize a specific output neuron subject to all network constraints.
    ///
    /// `output_idx` is the index into the encoder's output variables (0-based).
    /// Used for LP bound tightening: the minimum value of neuron i
    /// gives a tighter lower bound.
    pub fn minimize_output(&self, output_idx: usize) -> Result<MipResult> {
        self.optimize_output(output_idx, Sense::Minimise)
    }

    /// Maximize a specific output neuron subject to all network constraints.
    pub fn maximize_output(&self, output_idx: usize) -> Result<MipResult> {
        self.optimize_output(output_idx, Sense::Maximise)
    }

    /// Optimize a single output neuron in the given direction.
    fn optimize_output(&self, output_idx: usize, sense: Sense) -> Result<MipResult> {
        if output_idx >= self.parts.output_vars.len() {
            return Err(MipError::Encoding(format!(
                "output index {} out of range (max {})",
                output_idx,
                self.parts.output_vars.len()
            )));
        }

        let spec = crate::ay::ObjectiveSpec {
            col: self.parts.output_vars[output_idx],
            sense: match sense {
                Sense::Minimise => crate::ay::ObjSense::Minimize,
                Sense::Maximise => crate::ay::ObjSense::Maximize,
            },
        };
        match self.config.backend {
            MipBackend::Ay => crate::ay_lib::optimize_col(
                &self.parts.problem,
                self.config.timeout_secs,
                spec,
                &self.parts.input_vars,
                &self.parts.output_vars,
            ),
            MipBackend::AyProc => crate::ay::optimize_col(
                &self.parts.problem,
                self.config.timeout_secs,
                spec,
                &self.parts.input_vars,
                &self.parts.output_vars,
            ),
        }
    }

    /// Get the number of output neurons.
    pub fn num_outputs(&self) -> usize {
        self.parts.output_vars.len()
    }

    /// Get the number of input neurons.
    pub fn num_inputs(&self) -> usize {
        self.parts.input_vars.len()
    }

    /// Get the total number of columns (variables) in the MIP problem.
    ///
    /// This is the required length of the warm-start dense vector for
    /// `check_feasibility_with_warm_start`.
    pub fn num_cols(&self) -> usize {
        self.parts.num_cols
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;

    fn add_synthetic_relu(problem: &mut ir::MilpProblem, lb: f64, ub: f64) -> ir::Col {
        assert!(lb < 0.0 && ub > 0.0);
        let x = problem.add_col(0.0, lb, ub);
        let y = problem.add_col(0.0, 0.0, ub);
        let z = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(0.0, f64::INFINITY, [(y, 1.0), (x, -1.0)]);
        problem.add_row(f64::NEG_INFINITY, -lb, [(y, 1.0), (x, -1.0), (z, -lb)]);
        problem.add_row(f64::NEG_INFINITY, 0.0, [(y, 1.0), (z, -ub)]);
        z
    }

    fn synthetic_solver(
        problem: ir::MilpProblem,
        binary_vars: Vec<ir::Col>,
        binary_widths: Vec<f64>,
    ) -> MipSolver {
        let num_cols = problem.num_cols();
        MipSolver::new(
            MipParts {
                problem,
                input_vars: Vec::new(),
                output_vars: Vec::new(),
                binary_vars,
                binary_widths,
                num_cols,
            },
            MipConfig::default(),
        )
    }

    #[test]
    fn stability_hint_gate_accepts_only_literal_one() {
        assert!(mip_stability_hints_enabled_from_value(Some("1")));
        for value in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some(" 1 "),
            Some("1\n"),
        ] {
            assert!(
                !mip_stability_hints_enabled_from_value(value),
                "unexpectedly enabled for {value:?}"
            );
        }
    }

    #[test]
    fn ay_branch_hint_gate_is_typed_and_accepts_only_literal_one() {
        assert_eq!(
            ay_branch_hint_state_from_value(Some("1")),
            AyBranchHintState::Enabled
        );
        for value in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("yes"),
            Some("01"),
            Some(" 1 "),
            Some("1\n"),
        ] {
            assert_eq!(
                ay_branch_hint_state_from_value(value),
                AyBranchHintState::Disabled,
                "unexpectedly enabled for {value:?}"
            );
        }
    }

    #[test]
    fn recovers_neuralsat_stability_score_from_exact_bigm_pair() {
        let mut problem = ir::MilpProblem::new();
        let z = add_synthetic_relu(&mut problem, -0.25, 1.5);
        assert_eq!(relu_stability_scores(&problem, &[z]), vec![Some(0.25)]);
    }

    #[test]
    fn altered_or_incomplete_bigm_pair_falls_back() {
        let mut problem = ir::MilpProblem::new();
        let x = problem.add_col(0.0, -0.25, 1.5);
        let lower_y = problem.add_col(0.0, 0.0, 1.5);
        let upper_y = problem.add_col(0.0, 0.0, 1.5);
        let z = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(
            f64::NEG_INFINITY,
            0.25,
            [(lower_y, 1.0), (x, -1.0), (z, 0.25)],
        );
        // A different y in the upper row is not a canonical ReLU pair.
        problem.add_row(f64::NEG_INFINITY, 0.0, [(upper_y, 1.0), (z, -1.5)]);
        assert_eq!(relu_stability_scores(&problem, &[z]), vec![None]);

        let orphan = problem.add_integer_col(0.0, 0.0, 1.0);
        assert_eq!(relu_stability_scores(&problem, &[orphan]), vec![None]);
    }

    #[test]
    fn disabled_ranking_preserves_historical_widest_first_order() {
        let mut problem = ir::MilpProblem::new();
        let first = add_synthetic_relu(&mut problem, -0.75, 1.25);
        let second = add_synthetic_relu(&mut problem, -0.03125, 0.96875);
        let solver = synthetic_solver(problem, vec![first, second], vec![2.0, 1.0]);

        assert_eq!(solver.ranked_binaries_with_stability(false), vec![0, 1]);
        assert!(
            solver.ay_branch_hint_order_with_gate(false).is_empty(),
            "the default-off AY seam must not materialize or forward advice"
        );
        assert_eq!(
            solver.ay_branch_hint_order_with_gate(true),
            vec![first, second]
        );
    }

    #[test]
    fn enabled_ranking_prefers_closest_to_stable_relu() {
        let mut problem = ir::MilpProblem::new();
        let wide = add_synthetic_relu(&mut problem, -0.75, 1.25);
        let near_stable = add_synthetic_relu(&mut problem, -0.03125, 0.96875);
        let solver = synthetic_solver(problem, vec![wide, near_stable], vec![2.0, 1.0]);

        assert_eq!(solver.ranked_binaries_with_stability(true), vec![1, 0]);
    }

    #[test]
    fn enabled_ranking_puts_unrecovered_rows_last_with_width_fallback() {
        let mut problem = ir::MilpProblem::new();
        let recovered = add_synthetic_relu(&mut problem, -0.5, 0.5);
        let orphan_wide = problem.add_integer_col(0.0, 0.0, 1.0);
        let orphan_narrow = problem.add_integer_col(0.0, 0.0, 1.0);
        let solver = synthetic_solver(
            problem,
            vec![orphan_narrow, recovered, orphan_wide],
            vec![2.0, 1.0, 4.0],
        );

        assert_eq!(solver.ranked_binaries_with_stability(true), vec![1, 2, 0]);
    }

    fn sat() -> MipResult {
        MipResult::Sat {
            objective: 0.0,
            output_values: vec![1.0],
            input_values: vec![0.5],
            dual_bound: None,
        }
    }

    #[test]
    fn split_status_telemetry_does_not_count_replies_as_proofs() {
        let slots = vec![
            Some(MipResult::Unsat { certified: true }),
            Some(MipResult::Unsat { certified: false }),
            Some(sat()),
            Some(MipResult::Timeout),
            Some(MipResult::Error("boom".into())),
            None,
            Some(MipResult::Timeout),
        ];

        assert_eq!(
            split_status_counts(&slots),
            SplitStatusCounts {
                certified_unsat: 1,
                uncertified_unsat: 1,
                sat: 1,
                timeout: 2,
                error: 1,
                missing: 1,
            }
        );
    }

    /// Unsat aggregation requires EXACTLY 2^k Unsat sub-verdicts.
    #[test]
    fn all_unsat_aggregates_to_unsat() {
        let results = vec![
            MipResult::Unsat { certified: true },
            MipResult::Unsat { certified: true },
            MipResult::Unsat { certified: true },
            MipResult::Unsat { certified: true },
        ];
        assert!(matches!(
            aggregate_split_results(results, 4),
            MipResult::Unsat { .. }
        ));
    }

    /// Any Sat wins regardless of sibling verdicts (witness revalidated later).
    #[test]
    fn any_sat_aggregates_to_sat() {
        let results = vec![
            MipResult::Unsat { certified: true },
            sat(),
            MipResult::Timeout,
            MipResult::Unsat { certified: true },
        ];
        assert!(matches!(
            aggregate_split_results(results, 4),
            MipResult::Sat { .. }
        ));
    }

    /// A Timeout sub-result forces Timeout, never Unsat (soundness guard c).
    #[test]
    fn timeout_subresult_forces_timeout() {
        let results = vec![
            MipResult::Unsat { certified: true },
            MipResult::Unsat { certified: true },
            MipResult::Timeout,
            MipResult::Unsat { certified: true },
        ];
        assert!(matches!(
            aggregate_split_results(results, 4),
            MipResult::Timeout
        ));
    }

    /// An Error sub-result forces Timeout, never Unsat (soundness guard c).
    #[test]
    fn error_subresult_forces_timeout() {
        let results = vec![
            MipResult::Unsat { certified: true },
            MipResult::Unsat { certified: true },
            MipResult::Error("boom".into()),
            MipResult::Unsat { certified: true },
        ];
        assert!(matches!(
            aggregate_split_results(results, 4),
            MipResult::Timeout
        ));
    }

    /// A missing sub-result (fewer than 2^k) forces Timeout, never Unsat.
    #[test]
    fn missing_subresult_forces_timeout() {
        let results = vec![
            MipResult::Unsat { certified: true },
            MipResult::Unsat { certified: true },
            MipResult::Unsat { certified: false },
        ];
        assert!(matches!(
            aggregate_split_results(results, 4),
            MipResult::Timeout
        ));
    }

    /// The 2^k assignment enumeration is exhaustive and distinct: fixing k
    /// binaries by the bit pattern of `assignment in 0..2^k` covers every
    /// {0,1}^k vector exactly once (soundness: exact partition by construction).
    #[test]
    fn assignment_enumeration_is_exhaustive_and_distinct() {
        let k = 4;
        let mut seen = HashSet::new();
        for assignment in 0..(1usize << k) {
            let bits: Vec<u8> = (0..k).map(|bit| ((assignment >> bit) & 1) as u8).collect();
            assert!(seen.insert(bits), "duplicate assignment {assignment}");
        }
        assert_eq!(seen.len(), 1 << k);
    }

    fn fp(hash: u64) -> SplitFingerprint {
        (vec![ir::Col(3), ir::Col(7)], 4, 10, 20, hash)
    }

    /// A recorded certified-Unsat assignment reads back as proven — the seed
    /// for the pre-seed-and-skip path in `check_feasibility_split` (slot
    /// filled with `Unsat { certified: true }`, worker never spawned).
    #[test]
    fn cache_records_certified_unsat_and_reports_proven() {
        let mut cache = SplitUnsatCache::default();
        cache.reconcile(fp(0xfeed));
        cache.record(2, &MipResult::Unsat { certified: true });
        assert!(cache.is_proven(2));
        assert!(!cache.is_proven(0));
        assert!(!cache.is_proven(1));
        assert!(!cache.is_proven(3));
        // A same-fingerprint reconcile (next round, identical problem) keeps
        // the proven set.
        cache.reconcile(fp(0xfeed));
        assert!(cache.is_proven(2));
    }

    /// ANY fingerprint drift clears the proven set (fail-closed): differing
    /// content hash, split set, or subproblem count.
    #[test]
    fn cache_fingerprint_mismatch_clears_proven() {
        // Content-hash drift.
        let mut cache = SplitUnsatCache::default();
        cache.reconcile(fp(0xfeed));
        cache.record(2, &MipResult::Unsat { certified: true });
        cache.reconcile(fp(0xbeef));
        assert!(!cache.is_proven(2));

        // Split-set drift at identical hash.
        let mut cache = SplitUnsatCache::default();
        cache.reconcile(fp(0xfeed));
        cache.record(1, &MipResult::Unsat { certified: true });
        cache.reconcile((vec![ir::Col(3), ir::Col(8)], 4, 10, 20, 0xfeed));
        assert!(!cache.is_proven(1));

        // Subproblem-count drift at identical hash.
        let mut cache = SplitUnsatCache::default();
        cache.reconcile(fp(0xfeed));
        cache.record(1, &MipResult::Unsat { certified: true });
        cache.reconcile((vec![ir::Col(3), ir::Col(7)], 8, 10, 20, 0xfeed));
        assert!(!cache.is_proven(1));
    }

    /// Sat, uncertified Unsat, Timeout, and Error are NEVER memoized.
    #[test]
    fn cache_never_records_sat_uncertified_timeout_or_error() {
        let mut cache = SplitUnsatCache::default();
        cache.reconcile(fp(0xfeed));
        cache.record(0, &sat());
        cache.record(1, &MipResult::Unsat { certified: false });
        cache.record(2, &MipResult::Timeout);
        cache.record(3, &MipResult::Error("boom".into()));
        for assignment in 0..4 {
            assert!(
                !cache.is_proven(assignment),
                "assignment {assignment} must not be cached"
            );
        }
    }

    /// The IR content hash sees every bound: two IRs identical except one
    /// column bound (or one row coefficient) hash differently, so the memo
    /// clears on any encoded-problem drift.
    #[test]
    fn ir_content_hash_detects_bound_and_coefficient_drift() {
        let build = |ub: f64, weight: f64| {
            let mut p = ir::MilpProblem::new();
            let x = p.add_col(0.0, 0.0, ub);
            let z = p.add_integer_col(0.0, 0.0, 1.0);
            p.add_row(f64::NEG_INFINITY, 1.5, vec![(x, weight), (z, 1.0)]);
            p
        };
        let base = ir_content_hash(&build(1.0, 2.0));
        assert_eq!(
            base,
            ir_content_hash(&build(1.0, 2.0)),
            "hash must be deterministic"
        );
        assert_ne!(
            base,
            ir_content_hash(&build(1.5, 2.0)),
            "column-bound drift must change the hash"
        );
        assert_ne!(
            base,
            ir_content_hash(&build(1.0, 2.5)),
            "row-coefficient drift must change the hash"
        );
    }
}
