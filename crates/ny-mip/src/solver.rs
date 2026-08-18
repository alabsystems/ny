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

use crate::config::{MipBackend, MipConfig, MipFeasibilityIngress};
use crate::encoder::MipParts;
use crate::error::MipError;
use crate::ir;

use std::collections::{HashMap, HashSet};

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

/// A SAT candidate produced by direct optimization of one already-constrained
/// one-sided unsafe-region row.
///
/// This type deliberately carries no proof or verdict.  The point has passed
/// AY's exact model replay, but a caller must still run the original network
/// and property checker (and, in VNN-COMP, the trusted-runtime replay) before
/// it may emit a counterexample.
#[derive(Debug, Clone, PartialEq)]
pub struct OneSidedSatWitness {
    /// Objective value at the returned point.
    pub objective: f64,
    /// Output columns extracted from the exact AY point.
    pub output_values: Vec<f64>,
    /// Input columns extracted from the exact AY point.
    pub input_values: Vec<f64>,
}

/// Why the direct one-sided objective probe produced no usable SAT candidate.
///
/// Even [`Self::InfeasibleIgnored`] is only a decline: this lane has no UNSAT
/// variant by construction and can never authorize a verified verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OneSidedSatDecline {
    /// Only the pinned in-process AY API exposes the sparse objective surface.
    UnsupportedBackend,
    /// The caller supplied no positive finite slice.
    InvalidTimeout,
    /// The requested row is missing or is not one finite one-sided inequality.
    InvalidRow(String),
    /// The externally enforced hard deadline expired.
    Deadline,
    /// AY proved the constrained model infeasible; intentionally discarded.
    InfeasibleIgnored,
    /// AY returned no feasible incumbent (unknown/bound-only/unbounded).
    NoWitness,
    /// AY returned a point that failed exact replay against the same model.
    ReplayRejected(String),
    /// Lowering/session infrastructure failed; the caller must fall back.
    SolverError(String),
}

/// Witness-only result of [`MipSolver::probe_one_sided_sat`].
#[derive(Debug, Clone, PartialEq)]
pub enum OneSidedSatProbe {
    /// An exactly model-feasible candidate, still requiring concrete replay.
    Witness(OneSidedSatWitness),
    /// No candidate.  Every reason is verdict-neutral.
    Declined(OneSidedSatDecline),
}

/// Cap on the number of binaries fixed by phase-split racing: 2^4 = 16
/// subproblems, matching the core counts this targets (designs/scip.md).
const MAX_SPLIT_K: usize = 4;

/// Validate and classify a caller-identified one-sided row.
///
/// This is stricter than merely testing `is_finite() ^ is_finite()`: the open
/// side must be the correctly signed infinity, every referenced column and
/// coefficient must be valid, and duplicate cancellation must not erase the
/// effective linear form.
fn one_sided_row_sense(
    problem: &ir::MilpProblem,
    row: ir::Row,
) -> std::result::Result<Sense, String> {
    let spec = problem
        .rows()
        .get(row.0)
        .ok_or_else(|| format!("row {} is out of range", row.0))?;
    let sense = if spec.lb == f64::NEG_INFINITY && spec.ub.is_finite() {
        Sense::Minimise
    } else if spec.lb.is_finite() && spec.ub == f64::INFINITY {
        Sense::Maximise
    } else {
        return Err(format!(
            "row {} is not a finite one-sided inequality",
            row.0
        ));
    };

    let mut coeffs = spec.coeffs.clone();
    if coeffs
        .iter()
        .any(|&(col, coeff)| col >= problem.num_cols() || !coeff.is_finite())
    {
        return Err(format!(
            "row {} has an invalid column or non-finite coefficient",
            row.0
        ));
    }
    coeffs.sort_unstable_by_key(|&(col, _)| col);
    coeffs.dedup_by(|later, first| {
        if later.0 == first.0 {
            first.1 += later.1;
            true
        } else {
            false
        }
    });
    if coeffs.iter().any(|&(_, coeff)| !coeff.is_finite()) {
        return Err(format!("row {} coefficient sum is non-finite", row.0));
    }
    if !coeffs.iter().any(|&(_, coeff)| coeff != 0.0) {
        return Err(format!("row {} has no effective coefficient", row.0));
    }
    Ok(sense)
}

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

/// Default-dark routing state for the SafeNLP shared-prefix feasibility canary.
///
/// Only the literal value `1` arms the experiment.  In particular, `0`,
/// malformed Unicode strings, and non-Unicode values all preserve the
/// historical cloned phase-split race.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SafeNlpSharedPrefixState {
    Disabled,
    Enabled,
}

fn safenlp_shared_prefix_state_from_value(value: Option<&str>) -> SafeNlpSharedPrefixState {
    if value == Some("1") {
        SafeNlpSharedPrefixState::Enabled
    } else {
        SafeNlpSharedPrefixState::Disabled
    }
}

fn resolved_safenlp_shared_prefix_state() -> SafeNlpSharedPrefixState {
    safenlp_shared_prefix_state_from_value(
        std::env::var("NY_MIP_SAFENLP_SHARED_PREFIX")
            .ok()
            .as_deref(),
    )
}

/// Default-dark selection state for the objective-aware marked-margin prefix.
///
/// This is deliberately separate from the shared-prefix ingress gate: the
/// selector may only refine an already-required, explicitly marked session.
/// Only the literal value `1` enables it; all other spellings preserve the
/// existing fixed four-column prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SafeNlpTargetFsbPrefixState {
    Disabled,
    Enabled,
}

fn safenlp_target_fsb_prefix_state_from_value(value: Option<&str>) -> SafeNlpTargetFsbPrefixState {
    if value == Some("1") {
        SafeNlpTargetFsbPrefixState::Enabled
    } else {
        SafeNlpTargetFsbPrefixState::Disabled
    }
}

fn resolved_safenlp_target_fsb_prefix_state() -> SafeNlpTargetFsbPrefixState {
    safenlp_target_fsb_prefix_state_from_value(
        std::env::var("NY_MIP_SAFENLP_TARGET_FSB_PREFIX")
            .ok()
            .as_deref(),
    )
}

/// Recover the marked decision row as a linear form to minimize.
///
/// The identity must be explicit and the open side must be the correctly
/// signed infinity. For `c*x <= u` the lower form is `c*x`; for `c*x >= l`
/// it is `-c*x`. Sparse column identities and order are retained exactly so
/// the fail-closed canonical-ReLU ranker, rather than this routing seam,
/// remains responsible for rejecting duplicate or otherwise malformed input.
fn marked_margin_lower_form(problem: &ir::MilpProblem) -> Option<Vec<(ir::Col, f64)>> {
    let margin = problem.margin_row()?;
    let row = problem.rows().get(margin.0)?;
    let negate = if row.lb == f64::NEG_INFINITY && row.ub.is_finite() {
        false
    } else if row.lb.is_finite() && row.ub == f64::INFINITY {
        true
    } else {
        return None;
    };
    if row.coeffs.is_empty() {
        return None;
    }
    row.coeffs
        .iter()
        .map(|&(col, coefficient)| {
            if col >= problem.num_cols() || !coefficient.is_finite() {
                return None;
            }
            let coefficient = if negate { -coefficient } else { coefficient };
            coefficient
                .is_finite()
                .then_some((ir::Col(col), coefficient))
        })
        .collect()
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

#[derive(Debug, Clone, Copy)]
struct CanonicalReluAdviceEncoding {
    binary: ir::Col,
    input: ir::Col,
    output: ir::Col,
    upper_slope: f64,
    intercept_gap: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct PartialCanonicalReluAdviceEncoding {
    lower_upper_row: Option<(ir::Col, ir::Col, f64)>,
    zero_upper_row: Option<(ir::Col, f64)>,
}

#[derive(Debug, Clone)]
struct AppendOnlyAffineDefinition {
    output_weight: f64,
    inputs: Vec<(ir::Col, f64)>,
}

/// Rank canonical unstable-ReLU binaries for a lower-bound linear form.
///
/// This is an unwired, advice-only BaBSR primitive. `objective` is the sparse
/// linear form to MINIMIZE; the IR's stored objective coefficients are
/// deliberately ignored because certified decision models carry the target
/// only in a final one-sided row. Coefficients are propagated backwards
/// through append-only affine equality rows and through each ReLU's triangle
/// upper slope. A ReLU with downstream coefficient `c` receives the
/// intercept-only score
///
/// `max(-c, 0) * (-l * u) / (u - l)`.
///
/// Only exact encoder triples (`y >= x`, `y - x - l*z <= -l`,
/// `y - u*z <= 0`) and unique, finite, append-only affine definitions are
/// accepted. Any malformed candidate, duplicate identity/definition, invalid
/// sparse term, or non-finite intermediate makes the whole helper return an
/// empty vector. Zero-score ReLUs are omitted, and positive scores are sorted
/// descending with caller order as the deterministic tie-break.
///
/// The result can only advise a later search policy. It does not mutate the
/// model, establish a bound, admit a verdict, or participate in certificate
/// replay.
pub fn rank_canonical_relu_binaries_for_lower_form(
    problem: &ir::MilpProblem,
    objective: &[(ir::Col, f64)],
    binary_vars: &[ir::Col],
) -> Vec<ir::Col> {
    fn extract(
        problem: &ir::MilpProblem,
        binary_vars: &[ir::Col],
    ) -> Option<(
        Vec<CanonicalReluAdviceEncoding>,
        Vec<Option<AppendOnlyAffineDefinition>>,
    )> {
        let num_cols = problem.num_cols();
        if binary_vars.is_empty() {
            return None;
        }

        let mut binary_index = vec![None; num_cols];
        for (index, &binary) in binary_vars.iter().enumerate() {
            let spec = problem.cols().get(binary.0)?;
            if !spec.integer
                || spec.lb.to_bits() != 0.0_f64.to_bits()
                || spec.ub.to_bits() != 1.0_f64.to_bits()
                || binary_index[binary.0].replace(index).is_some()
            {
                return None;
            }
        }

        let mut partial = vec![PartialCanonicalReluAdviceEncoding::default(); binary_vars.len()];
        let mut lower_rows: HashMap<(ir::Col, ir::Col), usize> = HashMap::new();
        let mut affine_definitions: Vec<Option<AppendOnlyAffineDefinition>> = vec![None; num_cols];

        for row in problem.rows() {
            if row
                .coeffs
                .iter()
                .any(|&(col, weight)| col >= num_cols || !weight.is_finite())
            {
                return None;
            }

            // Collect exact `y - x >= 0` rows in the same pass. A duplicate
            // exact row is retained as a count and rejects that candidate
            // below rather than being silently accepted.
            if row.lb.to_bits() == 0.0_f64.to_bits()
                && row.ub == f64::INFINITY
                && row.coeffs.len() == 2
            {
                let positive = row
                    .coeffs
                    .iter()
                    .find_map(|&(col, weight)| (weight == 1.0).then_some(ir::Col(col)));
                let negative = row
                    .coeffs
                    .iter()
                    .find_map(|&(col, weight)| (weight == -1.0).then_some(ir::Col(col)));
                if let (Some(output), Some(input)) = (positive, negative) {
                    if output != input {
                        let count = lower_rows.entry((input, output)).or_default();
                        *count = count.checked_add(1)?;
                    }
                }
            }

            let mut tracked_binary = None;
            for &(col, weight) in &row.coeffs {
                let Some(index) = binary_index[col] else {
                    continue;
                };
                if tracked_binary.is_some() {
                    return None;
                }
                tracked_binary = Some((index, ir::Col(col), weight));
            }

            if let Some((index, binary, binary_weight)) = tracked_binary {
                let mut seen_cols = HashSet::with_capacity(row.coeffs.len());
                if !row
                    .coeffs
                    .iter()
                    .all(|&(col, weight)| weight != 0.0 && seen_cols.insert(col))
                {
                    return None;
                }

                // `y - x - l*z <= -l`, where `-l > 0`.
                if row.lb == f64::NEG_INFINITY
                    && row.ub.is_finite()
                    && row.ub > 0.0
                    && row.coeffs.len() == 3
                    && binary_weight.to_bits() == row.ub.to_bits()
                {
                    let output = row.coeffs.iter().find_map(|&(col, weight)| {
                        (col != binary.0 && weight == 1.0).then_some(ir::Col(col))
                    });
                    let input = row.coeffs.iter().find_map(|&(col, weight)| {
                        (col != binary.0 && weight == -1.0).then_some(ir::Col(col))
                    });
                    let (Some(input), Some(output)) = (input, output) else {
                        return None;
                    };
                    if partial[index]
                        .lower_upper_row
                        .replace((input, output, row.ub))
                        .is_some()
                    {
                        return None;
                    }
                    continue;
                }

                // `y - u*z <= 0`, where `u > 0`.
                if row.lb == f64::NEG_INFINITY
                    && row.ub.to_bits() == 0.0_f64.to_bits()
                    && row.coeffs.len() == 2
                    && binary_weight < 0.0
                {
                    let output = row.coeffs.iter().find_map(|&(col, weight)| {
                        (col != binary.0 && weight == 1.0).then_some(ir::Col(col))
                    });
                    let output = output?;
                    if partial[index]
                        .zero_upper_row
                        .replace((output, -binary_weight))
                        .is_some()
                    {
                        return None;
                    }
                    continue;
                }

                // A supplied binary occurring in any other row is not an
                // unmodified canonical ReLU indicator.
                return None;
            }

            if row.lb.is_finite() && row.lb.to_bits() == row.ub.to_bits() {
                if row.coeffs.is_empty() {
                    continue;
                }
                let mut seen_cols = HashSet::with_capacity(row.coeffs.len());
                if !row
                    .coeffs
                    .iter()
                    .all(|&(col, weight)| weight != 0.0 && seen_cols.insert(col))
                {
                    return None;
                }
                let &(output, output_weight) = row.coeffs.iter().max_by_key(|&&(col, _)| col)?;
                if problem.cols()[output].integer
                    || output_weight == 0.0
                    || affine_definitions[output].is_some()
                {
                    return None;
                }
                let inputs = row
                    .coeffs
                    .iter()
                    .filter_map(|&(col, weight)| (col != output).then_some((ir::Col(col), weight)))
                    .collect();
                affine_definitions[output] = Some(AppendOnlyAffineDefinition {
                    output_weight,
                    inputs,
                });
            }
        }

        let mut relus = Vec::with_capacity(binary_vars.len());
        let mut output_owner = vec![None; num_cols];
        for (index, (&binary, rows)) in binary_vars.iter().zip(partial).enumerate() {
            let (input, lower_output, lower_magnitude) = rows.lower_upper_row?;
            let (upper_output, upper) = rows.zero_upper_row?;
            if lower_output != upper_output
                || !(input.0 < lower_output.0 && lower_output.0 < binary.0)
                || lower_rows.get(&(input, lower_output)).copied() != Some(1)
            {
                return None;
            }

            let input_spec = problem.cols().get(input.0)?;
            let output_spec = problem.cols().get(lower_output.0)?;
            if input_spec.integer
                || output_spec.integer
                || output_spec.lb.to_bits() != 0.0_f64.to_bits()
                || output_spec.ub.to_bits() != upper.to_bits()
                || affine_definitions[lower_output.0].is_some()
                || affine_definitions[binary.0].is_some()
                || output_owner[lower_output.0].replace(index).is_some()
            {
                return None;
            }

            let width = lower_magnitude + upper;
            let upper_slope = upper / width;
            let intercept_gap = lower_magnitude * upper_slope;
            if !lower_magnitude.is_finite()
                || lower_magnitude <= 0.0
                || !upper.is_finite()
                || upper <= 0.0
                || !width.is_finite()
                || width <= 0.0
                || !upper_slope.is_finite()
                || !intercept_gap.is_finite()
                || intercept_gap <= 0.0
            {
                return None;
            }

            relus.push(CanonicalReluAdviceEncoding {
                binary,
                input,
                output: lower_output,
                upper_slope,
                intercept_gap,
            });
        }

        Some((relus, affine_definitions))
    }

    let Some((relus, affine_definitions)) = extract(problem, binary_vars) else {
        return Vec::new();
    };

    let mut coefficients = vec![0.0_f64; problem.num_cols()];
    let mut objective_cols = HashSet::with_capacity(objective.len());
    for &(col, coefficient) in objective {
        if col.0 >= coefficients.len() || !coefficient.is_finite() || !objective_cols.insert(col.0)
        {
            return Vec::new();
        }
        coefficients[col.0] = coefficient;
    }

    let mut relu_by_output = vec![None; problem.num_cols()];
    for (index, relu) in relus.iter().enumerate() {
        relu_by_output[relu.output.0] = Some(index);
    }
    let mut scores = vec![0.0_f64; relus.len()];

    for col in (0..coefficients.len()).rev() {
        let coefficient = coefficients[col];
        if coefficient == 0.0 {
            continue;
        }
        if !coefficient.is_finite() {
            return Vec::new();
        }

        if let Some(index) = relu_by_output[col] {
            let relu = relus[index];
            let score = (-coefficient).max(0.0) * relu.intercept_gap;
            let propagated = coefficient * relu.upper_slope;
            let next = coefficients[relu.input.0] + propagated;
            if !score.is_finite() || !propagated.is_finite() || !next.is_finite() {
                return Vec::new();
            }
            scores[index] = score;
            coefficients[relu.input.0] = next;
            coefficients[col] = 0.0;
            continue;
        }

        if let Some(definition) = &affine_definitions[col] {
            coefficients[col] = 0.0;
            for &(input, input_weight) in &definition.inputs {
                // a_out*out + sum(a_i*x_i) = b
                // => dL/dx_i += -(dL/dout)*a_i/a_out.
                let propagated = -coefficient * (input_weight / definition.output_weight);
                let next = coefficients[input.0] + propagated;
                if !propagated.is_finite() || !next.is_finite() {
                    return Vec::new();
                }
                coefficients[input.0] = next;
            }
            continue;
        }

        // A coefficient that reaches an unrecognized integer is inconsistent
        // with a pure affine/ReLU value graph. Decline all advice.
        if problem.cols()[col].integer {
            return Vec::new();
        }
    }

    let mut ranked: Vec<(usize, f64)> = scores
        .into_iter()
        .enumerate()
        .filter(|&(_, score)| score > 0.0)
        .collect();
    ranked.sort_by(|&(left_index, left_score), &(right_index, right_score)| {
        right_score
            .total_cmp(&left_score)
            .then(left_index.cmp(&right_index))
    });
    ranked
        .into_iter()
        .map(|(index, _)| relus[index].binary)
        .collect()
}

#[derive(Debug, Clone)]
struct BiasAwareAppendOnlyAffineDefinition {
    output_weight: f64,
    bias: f64,
    inputs: Vec<(ir::Col, f64)>,
}

fn extract_bias_aware_canonical_relu_advice_graph(
    problem: &ir::MilpProblem,
    binary_vars: &[ir::Col],
) -> Option<(
    Vec<CanonicalReluAdviceEncoding>,
    Vec<Option<BiasAwareAppendOnlyAffineDefinition>>,
)> {
    let num_cols = problem.num_cols();
    if binary_vars.is_empty() {
        return None;
    }

    let mut binary_index = vec![None; num_cols];
    for (index, &binary) in binary_vars.iter().enumerate() {
        let spec = problem.cols().get(binary.0)?;
        if !spec.integer
            || spec.lb.to_bits() != 0.0_f64.to_bits()
            || spec.ub.to_bits() != 1.0_f64.to_bits()
            || binary_index[binary.0].replace(index).is_some()
        {
            return None;
        }
    }

    let mut partial = vec![PartialCanonicalReluAdviceEncoding::default(); binary_vars.len()];
    let mut lower_rows: HashMap<(ir::Col, ir::Col), usize> = HashMap::new();
    let mut affine_definitions: Vec<Option<BiasAwareAppendOnlyAffineDefinition>> =
        vec![None; num_cols];

    for row in problem.rows() {
        if row
            .coeffs
            .iter()
            .any(|&(col, weight)| col >= num_cols || !weight.is_finite())
        {
            return None;
        }

        if row.lb.to_bits() == 0.0_f64.to_bits() && row.ub == f64::INFINITY && row.coeffs.len() == 2
        {
            let positive = row
                .coeffs
                .iter()
                .find_map(|&(col, weight)| (weight == 1.0).then_some(ir::Col(col)));
            let negative = row
                .coeffs
                .iter()
                .find_map(|&(col, weight)| (weight == -1.0).then_some(ir::Col(col)));
            if let (Some(output), Some(input)) = (positive, negative) {
                if output != input {
                    let count = lower_rows.entry((input, output)).or_default();
                    *count = count.checked_add(1)?;
                }
            }
        }

        let mut tracked_binary = None;
        for &(col, weight) in &row.coeffs {
            let Some(index) = binary_index[col] else {
                continue;
            };
            if tracked_binary.is_some() {
                return None;
            }
            tracked_binary = Some((index, ir::Col(col), weight));
        }

        if let Some((index, binary, binary_weight)) = tracked_binary {
            let mut seen_cols = HashSet::with_capacity(row.coeffs.len());
            if !row
                .coeffs
                .iter()
                .all(|&(col, weight)| weight != 0.0 && seen_cols.insert(col))
            {
                return None;
            }

            if row.lb == f64::NEG_INFINITY
                && row.ub.is_finite()
                && row.ub > 0.0
                && row.coeffs.len() == 3
                && binary_weight.to_bits() == row.ub.to_bits()
            {
                let output = row.coeffs.iter().find_map(|&(col, weight)| {
                    (col != binary.0 && weight == 1.0).then_some(ir::Col(col))
                });
                let input = row.coeffs.iter().find_map(|&(col, weight)| {
                    (col != binary.0 && weight == -1.0).then_some(ir::Col(col))
                });
                let (Some(input), Some(output)) = (input, output) else {
                    return None;
                };
                if partial[index]
                    .lower_upper_row
                    .replace((input, output, row.ub))
                    .is_some()
                {
                    return None;
                }
                continue;
            }

            if row.lb == f64::NEG_INFINITY
                && row.ub.to_bits() == 0.0_f64.to_bits()
                && row.coeffs.len() == 2
                && binary_weight < 0.0
            {
                let output = row.coeffs.iter().find_map(|&(col, weight)| {
                    (col != binary.0 && weight == 1.0).then_some(ir::Col(col))
                });
                let output = output?;
                if partial[index]
                    .zero_upper_row
                    .replace((output, -binary_weight))
                    .is_some()
                {
                    return None;
                }
                continue;
            }

            return None;
        }

        if row.lb.is_finite() && row.lb.to_bits() == row.ub.to_bits() {
            if row.coeffs.is_empty() {
                continue;
            }
            let mut seen_cols = HashSet::with_capacity(row.coeffs.len());
            if !row
                .coeffs
                .iter()
                .all(|&(col, weight)| weight != 0.0 && seen_cols.insert(col))
            {
                return None;
            }
            let &(output, output_weight) = row.coeffs.iter().max_by_key(|&&(col, _)| col)?;
            if problem.cols()[output].integer
                || output_weight == 0.0
                || affine_definitions[output].is_some()
            {
                return None;
            }
            let bias = row.lb / output_weight;
            if !bias.is_finite() {
                return None;
            }
            let inputs = row
                .coeffs
                .iter()
                .filter_map(|&(col, weight)| (col != output).then_some((ir::Col(col), weight)))
                .collect();
            affine_definitions[output] = Some(BiasAwareAppendOnlyAffineDefinition {
                output_weight,
                bias,
                inputs,
            });
        }
    }

    let mut relus = Vec::with_capacity(binary_vars.len());
    let mut output_owner = vec![None; num_cols];
    for (index, (&binary, rows)) in binary_vars.iter().zip(partial).enumerate() {
        let (input, lower_output, lower_magnitude) = rows.lower_upper_row?;
        let (upper_output, upper) = rows.zero_upper_row?;
        if lower_output != upper_output
            || !(input.0 < lower_output.0 && lower_output.0 < binary.0)
            || lower_rows.get(&(input, lower_output)).copied() != Some(1)
        {
            return None;
        }

        let input_spec = problem.cols().get(input.0)?;
        let output_spec = problem.cols().get(lower_output.0)?;
        if input_spec.integer
            || output_spec.integer
            || output_spec.lb.to_bits() != 0.0_f64.to_bits()
            || output_spec.ub.to_bits() != upper.to_bits()
            || affine_definitions[lower_output.0].is_some()
            || affine_definitions[binary.0].is_some()
            || output_owner[lower_output.0].replace(index).is_some()
        {
            return None;
        }

        let width = lower_magnitude + upper;
        let upper_slope = upper / width;
        let intercept_gap = lower_magnitude * upper_slope;
        if !lower_magnitude.is_finite()
            || lower_magnitude <= 0.0
            || !upper.is_finite()
            || upper <= 0.0
            || !width.is_finite()
            || width <= 0.0
            || !upper_slope.is_finite()
            || !intercept_gap.is_finite()
            || intercept_gap <= 0.0
        {
            return None;
        }

        relus.push(CanonicalReluAdviceEncoding {
            binary,
            input,
            output: lower_output,
            upper_slope,
            intercept_gap,
        });
    }

    Some((relus, affine_definitions))
}

fn rank_canonical_relu_binaries_for_lower_form_full_babsr(
    problem: &ir::MilpProblem,
    objective: &[(ir::Col, f64)],
    binary_vars: &[ir::Col],
) -> Option<Vec<ir::Col>> {
    let (relus, affine_definitions) =
        extract_bias_aware_canonical_relu_advice_graph(problem, binary_vars)?;
    let mut coefficients = vec![0.0_f64; problem.num_cols()];
    let mut objective_cols = HashSet::with_capacity(objective.len());
    for &(col, coefficient) in objective {
        if col.0 >= coefficients.len() || !coefficient.is_finite() || !objective_cols.insert(col.0)
        {
            return None;
        }
        coefficients[col.0] = coefficient;
    }

    let mut relu_by_output = vec![None; problem.num_cols()];
    for (index, relu) in relus.iter().enumerate() {
        relu_by_output[relu.output.0] = Some(index);
    }
    let mut scores = vec![0.0_f64; relus.len()];

    for col in (0..coefficients.len()).rev() {
        let coefficient = coefficients[col];
        if coefficient == 0.0 {
            continue;
        }
        if !coefficient.is_finite() {
            return None;
        }

        if let Some(index) = relu_by_output[col] {
            let relu = relus[index];
            let producer_bias = affine_definitions[relu.input.0]
                .as_ref()
                .map_or(0.0, |definition| definition.bias);
            let weighted_bias = coefficient * producer_bias;
            let inactive_delta = weighted_bias * (relu.upper_slope - 1.0);
            let active_delta = weighted_bias * relu.upper_slope;
            let intercept = coefficient.min(0.0) * relu.intercept_gap;
            if !weighted_bias.is_finite()
                || !inactive_delta.is_finite()
                || !active_delta.is_finite()
                || !intercept.is_finite()
            {
                return None;
            }
            let score = (inactive_delta.max(active_delta) + intercept).abs();
            let propagated = coefficient * relu.upper_slope;
            let next = coefficients[relu.input.0] + propagated;
            if !score.is_finite() || !propagated.is_finite() || !next.is_finite() {
                return None;
            }
            scores[index] = score;
            coefficients[relu.input.0] = next;
            coefficients[col] = 0.0;
            continue;
        }

        if let Some(definition) = &affine_definitions[col] {
            coefficients[col] = 0.0;
            for &(input, input_weight) in &definition.inputs {
                let propagated = -coefficient * (input_weight / definition.output_weight);
                let next = coefficients[input.0] + propagated;
                if !propagated.is_finite() || !next.is_finite() {
                    return None;
                }
                coefficients[input.0] = next;
            }
            continue;
        }

        if problem.cols()[col].integer {
            return None;
        }
    }

    let mut ranked: Vec<(usize, f64)> = scores
        .into_iter()
        .enumerate()
        .filter(|&(_, score)| score > 0.0)
        .collect();
    ranked.sort_by(|&(left_index, left_score), &(right_index, right_score)| {
        right_score
            .total_cmp(&left_score)
            .then(left_index.cmp(&right_index))
    });
    Some(
        ranked
            .into_iter()
            .map(|(index, _)| relus[index].binary)
            .collect(),
    )
}

/// Rank a bounded union of full-BaBSR and intercept-only ReLU candidates.
///
/// The full score is the winner-style, bias-aware BaBSR proxy
///
/// `abs(max((c*b)*(alpha - 1), (c*b)*alpha) + min(c, 0)*beta)`,
///
/// where `c` is the back-propagated lower-form coefficient, `b` is the
/// append-only producer equality's exact floating RHS divided by its output
/// coefficient, and `(alpha, beta)` are the triangle upper-line slope and
/// intercept. This is the `reduceop: max` variant used by the VNN-COMP 2025
/// alpha-beta-CROWN configurations. A raw input has `b = 0`. Full-score candidates come first,
/// followed by the existing intercept-only order as a backup; each source is
/// truncated to `candidates_per_score` and the combined order is stably
/// deduplicated, so the result has at most `2 * candidates_per_score` entries.
///
/// Extraction and arithmetic fail closed: malformed canonical triples,
/// ambiguous affine definitions, invalid handles, duplicate sparse identities,
/// non-finite biases, or non-finite intermediate values return no advice. This
/// The only production consumer is the exact default-dark, typed required
/// marked-margin SafeNLP prefix selector. There the bounded union can replace
/// only search order inside AY's complete shared-prefix partition; the
/// original fixed prefix remains the whole fallback, and neither this advice
/// nor its scores can alter the model or establish a verdict.
pub fn rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
    problem: &ir::MilpProblem,
    objective: &[(ir::Col, f64)],
    binary_vars: &[ir::Col],
    candidates_per_score: usize,
) -> Vec<ir::Col> {
    if candidates_per_score == 0 {
        return Vec::new();
    }
    let Some(full) =
        rank_canonical_relu_binaries_for_lower_form_full_babsr(problem, objective, binary_vars)
    else {
        return Vec::new();
    };
    let intercept = rank_canonical_relu_binaries_for_lower_form(problem, objective, binary_vars);
    let union_capacity = candidates_per_score
        .saturating_mul(2)
        .min(binary_vars.len());
    let mut union = Vec::with_capacity(union_capacity);
    let mut seen = HashSet::with_capacity(union_capacity);
    for binary in full
        .into_iter()
        .take(candidates_per_score)
        .chain(intercept.into_iter().take(candidates_per_score))
    {
        if seen.insert(binary.0) {
            union.push(binary);
        }
    }
    union
}

/// Exact identity of the problem a [`SplitUnsatCache`] memo is valid for.
///
/// Verdict authority must not rest on a finite-width digest: even a
/// vanishingly unlikely collision could replay a certified UNSAT result for a
/// different subproblem. Retaining the immutable solver IR makes cache replay
/// conditional on structural equality of every bound, coefficient,
/// integrality bit, objective, and decision-row identity.
#[derive(Debug, Clone, PartialEq)]
struct SplitFingerprint {
    split_cols: Vec<ir::Col>,
    num_subproblems: usize,
    problem: ir::MilpProblem,
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
/// columns, subproblem count, and a structural copy of the full IR — matches
/// the new problem EXACTLY. Any drift
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
    fn reconcile(
        &mut self,
        split_cols: &[ir::Col],
        num_subproblems: usize,
        problem: &ir::MilpProblem,
    ) {
        let matches = self.fingerprint.as_ref().is_some_and(|fingerprint| {
            fingerprint.split_cols == split_cols
                && fingerprint.num_subproblems == num_subproblems
                && fingerprint.problem == *problem
        });
        if !matches {
            self.proven.clear();
            self.fingerprint = Some(SplitFingerprint {
                split_cols: split_cols.to_vec(),
                num_subproblems,
                problem: problem.clone(),
            });
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
    safenlp_shared_prefix: SafeNlpSharedPrefixState,
    safenlp_target_fsb_prefix: SafeNlpTargetFsbPrefixState,
}

/// Exact backend entry owned by one admitted SafeNLP shared-prefix attempt.
///
/// Keeping the marked form distinct prevents a required marked-margin caller
/// from silently entering plain feasibility when its row identity is lost.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SafeNlpSharedPrefixPlan {
    Plain(Vec<ir::Col>),
    MarkedMargin(Vec<ir::Col>),
    MarkedMarginTargetFsb {
        fallback: [ir::Col; MAX_SPLIT_K],
        candidates: Vec<ir::Col>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SafeNlpSharedPrefixMode {
    Plain,
    MarkedMargin,
}

impl MipSolver {
    /// Create a solver from encoded network parts and configuration.
    pub fn new(parts: MipParts, config: MipConfig) -> Self {
        // Startup policy: resolve the exact environment canary once, before
        // any serial solve or detached split worker is created. Later process
        // environment mutation cannot change this solver's search policy.
        let ay_branch_hints = resolved_ay_branch_hint_state();
        let safenlp_shared_prefix = resolved_safenlp_shared_prefix_state();
        let safenlp_target_fsb_prefix = resolved_safenlp_target_fsb_prefix_state();
        Self {
            parts,
            config,
            ay_branch_hints,
            safenlp_shared_prefix,
            safenlp_target_fsb_prefix,
        }
    }

    /// Historical wall slice that an immediate feasibility call would own.
    ///
    /// This mirrors the actual authority path rather than merely returning
    /// [`MipConfig::timeout_secs`]: serial AY window models include the
    /// configured window floor, phase-split races retain their outer caller
    /// slice, and every relative path includes AY's `[1ms, 24h]` hard clamp.
    /// A caller that schedules preliminary work can use this as one shared
    /// envelope and then install an absolute cap with
    /// [`Self::set_ay_hard_deadline`].
    pub fn effective_feasibility_timeout_secs(&self) -> f64 {
        let window_floor = std::env::var("NY_MIP_WINDOW_TIMEOUT_SECS").ok();
        self.effective_feasibility_timeout_secs_with_window_value(window_floor.as_deref())
    }

    fn effective_feasibility_timeout_secs_with_window_value(
        &self,
        window_floor: Option<&str>,
    ) -> f64 {
        let phase_split = self.split_plan().is_some();
        let relative = if self.config.backend == MipBackend::Ay
            && !phase_split
            && self.config.ay_hard_deadline.is_none()
        {
            crate::ay_lib::window_budget_floor_from_value(
                &self.parts.problem,
                self.config.timeout_secs,
                window_floor,
            )
        } else {
            self.config.timeout_secs
        };
        let relative = crate::ay_lib::hard_timeout_slice_secs(relative);
        match self.config.ay_hard_deadline {
            Some(deadline) => deadline
                .checked_duration_since(std::time::Instant::now())
                .map_or(0.0, |remaining| remaining.as_secs_f64().min(relative)),
            None => relative,
        }
    }

    /// Put all subsequent AY work on one absolute caller-owned deadline.
    ///
    /// The relative timeout remains a per-call ceiling.  The absolute cap is
    /// forwarded through serial solves, phase-split workers, cached re-solves,
    /// and solver instances rebuilt from the same [`MipConfig`]; AY's window
    /// timeout floor is suppressed while it is present.
    pub fn set_ay_hard_deadline(
        &mut self,
        timeout_secs: f64,
        deadline: std::time::Instant,
    ) -> Result<()> {
        if self.config.backend != MipBackend::Ay {
            return Err(MipError::Solver(
                "an AY hard deadline requires the in-process AY backend".to_string(),
            ));
        }
        if !timeout_secs.is_finite() || timeout_secs < 0.001 {
            return Err(MipError::Solver(format!(
                "invalid AY hard timeout slice {timeout_secs}"
            )));
        }
        self.config.timeout_secs = timeout_secs;
        self.config.ay_hard_deadline = Some(deadline);
        Ok(())
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
        if let Some(plan) = self.safenlp_shared_prefix_plan(warm_start_cols.is_some())? {
            crate::dump::maybe_dump(&self.parts.problem);
            let branch_hints = self.ay_branch_hint_order();
            return match plan {
                SafeNlpSharedPrefixPlan::Plain(split_cols) => {
                    crate::ay_lib::check_feasibility_shared_binary_prefix(
                        &self.parts.problem,
                        self.config.timeout_secs,
                        self.config.ay_hard_deadline,
                        self.config.ay_node_warm_time_limit,
                        &self.parts.input_vars,
                        &self.parts.output_vars,
                        &split_cols,
                        &branch_hints,
                    )
                }
                SafeNlpSharedPrefixPlan::MarkedMargin(split_cols) => {
                    crate::ay_lib::check_feasibility_marked_margin_shared_binary_prefix(
                        &self.parts.problem,
                        self.config.timeout_secs,
                        self.config.ay_hard_deadline,
                        self.config.ay_node_warm_time_limit,
                        &self.parts.input_vars,
                        &self.parts.output_vars,
                        &split_cols,
                        &branch_hints,
                    )
                }
                SafeNlpSharedPrefixPlan::MarkedMarginTargetFsb {
                    fallback,
                    candidates,
                } => {
                    crate::ay_lib::check_feasibility_marked_margin_target_fsb_shared_binary_prefix(
                        &self.parts.problem,
                        self.config.timeout_secs,
                        self.config.ay_hard_deadline,
                        self.config.ay_node_warm_time_limit,
                        &self.parts.input_vars,
                        &self.parts.output_vars,
                        &fallback,
                        &candidates,
                        &branch_hints,
                    )
                }
            };
        }
        match self.split_plan() {
            Some(split_cols) => self.check_feasibility_split(&split_cols, warm_start_cols, None),
            None => self.solve_ir(&self.parts.problem, warm_start_cols),
        }
    }

    /// Select the existing phase-split prefix for one shared AY session.
    ///
    /// Admission is intentionally narrower than AY's typed API.  The canary
    /// needs the in-process backend, an already-installed caller deadline, and
    /// no external incumbent seed. Historical and required-plain ingress
    /// require no margin marker; required-marked ingress requires one.
    /// Failing a historical check returns `None` before AY is entered, so the
    /// caller takes the byte-for-byte historical serial/phase-split path.
    /// Required ingress instead errors before AY and cannot fall through.
    fn safenlp_shared_prefix_plan(
        &self,
        has_warm_start: bool,
    ) -> Result<Option<SafeNlpSharedPrefixPlan>> {
        let (required, mode) = match self.config.feasibility_ingress {
            MipFeasibilityIngress::Historical => (false, SafeNlpSharedPrefixMode::Plain),
            MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix => {
                (true, SafeNlpSharedPrefixMode::Plain)
            }
            MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix => {
                (true, SafeNlpSharedPrefixMode::MarkedMargin)
            }
        };
        let enabled = required || self.safenlp_shared_prefix == SafeNlpSharedPrefixState::Enabled;
        match self.safenlp_shared_prefix_plan_with_gate_result(enabled, has_warm_start, mode) {
            Ok(plan) => Ok(Some(plan)),
            Err(reason) if required => Err(MipError::Solver(format!(
                "required SafeNLP shared-binary-prefix session was not admitted: {reason}"
            ))),
            Err(_) => Ok(None),
        }
    }

    #[cfg(test)]
    fn safenlp_shared_prefix_plan_with_gate(
        &self,
        enabled: bool,
        has_warm_start: bool,
    ) -> Option<Vec<ir::Col>> {
        match self
            .safenlp_shared_prefix_plan_with_gate_result(
                enabled,
                has_warm_start,
                SafeNlpSharedPrefixMode::Plain,
            )
            .ok()?
        {
            SafeNlpSharedPrefixPlan::Plain(split_cols) => Some(split_cols),
            SafeNlpSharedPrefixPlan::MarkedMargin(_)
            | SafeNlpSharedPrefixPlan::MarkedMarginTargetFsb { .. } => None,
        }
    }

    fn safenlp_shared_prefix_plan_with_gate_result(
        &self,
        enabled: bool,
        has_warm_start: bool,
        mode: SafeNlpSharedPrefixMode,
    ) -> std::result::Result<SafeNlpSharedPrefixPlan, &'static str> {
        if !enabled {
            return Err("route is disabled");
        }
        if self.config.backend != MipBackend::Ay {
            return Err("backend is not in-process AY");
        }
        if has_warm_start {
            return Err("an external warm start is present");
        }
        match (mode, self.parts.problem.margin_row()) {
            (SafeNlpSharedPrefixMode::Plain, Some(_)) => {
                return Err("the plain problem has a marked margin row");
            }
            (SafeNlpSharedPrefixMode::MarkedMargin, None) => {
                return Err("the marked-margin problem has no marked margin row");
            }
            _ => {}
        }
        if self
            .config
            .ay_hard_deadline
            .is_none_or(|deadline| std::time::Instant::now() >= deadline)
        {
            return Err("the caller deadline is absent or expired");
        }
        let split_cols = self
            .split_plan()
            .ok_or("no live phase-split prefix is available")?;
        if !(1..=MAX_SPLIT_K).contains(&split_cols.len()) {
            return Err("the phase-split prefix width is outside 1..=4");
        }

        let mut seen = vec![false; self.parts.problem.num_cols()];
        for &col in &split_cols {
            let spec = self
                .parts
                .problem
                .cols()
                .get(col.0)
                .ok_or("a phase-split column is out of range")?;
            if !spec.integer
                || spec.lb.to_bits() != 0.0_f64.to_bits()
                || spec.ub.to_bits() != 1.0_f64.to_bits()
            {
                return Err("a phase-split column is not a live binary");
            }
            let seen = seen
                .get_mut(col.0)
                .ok_or("a phase-split column is out of range")?;
            if std::mem::replace(seen, true) {
                return Err("the phase-split prefix contains a duplicate column");
            }
        }
        Ok(match mode {
            SafeNlpSharedPrefixMode::Plain => SafeNlpSharedPrefixPlan::Plain(split_cols),
            SafeNlpSharedPrefixMode::MarkedMargin => {
                if self.safenlp_target_fsb_prefix == SafeNlpTargetFsbPrefixState::Enabled {
                    if let Some((fallback, candidates)) =
                        self.marked_margin_target_fsb_prefix_plan(&split_cols)
                    {
                        return Ok(SafeNlpSharedPrefixPlan::MarkedMarginTargetFsb {
                            fallback,
                            candidates,
                        });
                    }
                }
                SafeNlpSharedPrefixPlan::MarkedMargin(split_cols)
            }
        })
    }

    /// Build bounded target-FSB advice without weakening the fixed-prefix
    /// contract. Any ranking/admission problem declines to the caller's exact
    /// `split_cols` fallback; this helper never turns optional advice into an
    /// error and never starts a backend session.
    fn marked_margin_target_fsb_prefix_plan(
        &self,
        split_cols: &[ir::Col],
    ) -> Option<([ir::Col; MAX_SPLIT_K], Vec<ir::Col>)> {
        let fallback = <[ir::Col; MAX_SPLIT_K]>::try_from(split_cols).ok()?;
        let objective = marked_margin_lower_form(&self.parts.problem)?;
        let candidates = rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
            &self.parts.problem,
            &objective,
            &self.parts.binary_vars,
            MAX_SPLIT_K,
        );
        if !(MAX_SPLIT_K..=2 * MAX_SPLIT_K).contains(&candidates.len()) {
            return None;
        }

        let mut seen = vec![false; self.parts.problem.num_cols()];
        for &candidate in &candidates {
            let spec = self.parts.problem.cols().get(candidate.0)?;
            if !spec.integer
                || spec.lb.to_bits() != 0.0_f64.to_bits()
                || spec.ub.to_bits() != 1.0_f64.to_bits()
                || std::mem::replace(seen.get_mut(candidate.0)?, true)
            {
                return None;
            }
        }
        Some((fallback, candidates))
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
        if self.config.feasibility_ingress != MipFeasibilityIngress::Historical {
            // Required ingress owns one shared AY session.  A cloned
            // phase-split memo is inapplicable, but the typed route must
            // still be honored rather than silently entering the cached
            // historical implementation below.
            return self.check_feasibility_with_warm_start(warm_start_cols);
        }
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
        let _ = crate::dump::maybe_dump(problem);
        match self.config.backend {
            MipBackend::Ay => crate::ay_lib::check_feasibility(
                problem,
                self.config.timeout_secs,
                self.config.ay_hard_deadline,
                self.config.ay_node_warm_time_limit,
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
        let relative_deadline = Instant::now()
            + Duration::from_secs_f64(crate::ay_lib::hard_timeout_slice_secs(
                self.config.timeout_secs,
            ));
        let deadline = self
            .config
            .ay_hard_deadline
            .map_or(relative_deadline, |hard| hard.min(relative_deadline));
        if Instant::now() >= deadline {
            return Ok(MipResult::Timeout);
        }

        // Owned context for the detached workers.
        let backend = self.config.backend;
        let timeout_secs = self.config.timeout_secs;
        let ay_hard_deadline = self.config.ay_hard_deadline;
        let ay_node_warm_time_limit = self.config.ay_node_warm_time_limit;
        let input_vars = std::sync::Arc::new(self.parts.input_vars.clone());
        let output_vars = std::sync::Arc::new(self.parts.output_vars.clone());
        let branch_hints = std::sync::Arc::new(self.ay_branch_hint_order());
        let warm_start = std::sync::Arc::new(warm_start_cols.map(<[f64]>::to_vec));

        // Certified-UNSAT memo (fail-closed): reconcile clears the proven set
        // unless the fingerprint — split set, subproblem count, and a
        // structural copy of the full IR — matches this exact problem.
        let mut cache = cache;
        if let Some(cache) = cache.as_deref_mut() {
            cache.reconcile(split_cols, num_subproblems, &self.parts.problem);
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
            if ay_hard_deadline.is_some() && Instant::now() >= deadline {
                return Ok(MipResult::Timeout);
            }
            let mut sub = self.parts.problem.clone();
            for (bit, &col) in split_cols.iter().enumerate() {
                sub.fix_col(col, ((assignment >> bit) & 1) as f64);
            }
            // A large IR clone is part of the caller's absolute ledger. Do
            // not keep spawning detached workers after setup consumed it.
            if ay_hard_deadline.is_some() && Instant::now() >= deadline {
                return Ok(MipResult::Timeout);
            }
            let tx = tx.clone();
            let input_vars = std::sync::Arc::clone(&input_vars);
            let output_vars = std::sync::Arc::clone(&output_vars);
            let branch_hints = std::sync::Arc::clone(&branch_hints);
            let warm_start = std::sync::Arc::clone(&warm_start);
            std::thread::Builder::new()
                .name(format!("ny-mip-split-{assignment}"))
                .spawn(move || {
                    let _ = crate::dump::maybe_dump(&sub);
                    let result = match backend {
                        MipBackend::Ay => crate::ay_lib::check_feasibility(
                            &sub,
                            timeout_secs,
                            ay_hard_deadline,
                            ay_node_warm_time_limit,
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

    /// Directly optimize one existing one-sided row for a robust SAT point.
    ///
    /// For `c·x <= t` this minimizes `c·x`; for `c·x >= t` it maximizes
    /// `c·x`.  The row remains in the model, so every returned point satisfies
    /// the original unsafe-region system.  AY exact-replays the point before
    /// it leaves this seam.  No other solver outcome can escape as authority:
    /// infeasible, bound-only, unknown, unbounded, error, and deadline states
    /// all become [`OneSidedSatProbe::Declined`].
    ///
    /// Callers still MUST replay [`OneSidedSatWitness::input_values`] through
    /// the original concrete network/property implementation.  This method is
    /// intentionally serial and independently hard-deadline-bounded; it never
    /// consumes or mutates the historical feasibility solver state.
    pub fn probe_one_sided_sat(&self, row: ir::Row, timeout_secs: f64) -> OneSidedSatProbe {
        self.probe_one_sided_sat_with_deadline(row, timeout_secs, None)
    }

    /// Probe one row while also respecting a caller-owned absolute deadline.
    ///
    /// The effective deadline is the earlier of `hard_deadline` and the
    /// relative `timeout_secs` slice anchored inside the AY entry point. This
    /// prevents logging, scheduling, or other caller work from buying a fresh
    /// solver clock.
    pub fn probe_one_sided_sat_until(
        &self,
        row: ir::Row,
        timeout_secs: f64,
        hard_deadline: std::time::Instant,
    ) -> OneSidedSatProbe {
        self.probe_one_sided_sat_with_deadline(row, timeout_secs, Some(hard_deadline))
    }

    fn probe_one_sided_sat_with_deadline(
        &self,
        row: ir::Row,
        timeout_secs: f64,
        hard_deadline: Option<std::time::Instant>,
    ) -> OneSidedSatProbe {
        if self.config.backend != MipBackend::Ay {
            return OneSidedSatProbe::Declined(OneSidedSatDecline::UnsupportedBackend);
        }
        if !timeout_secs.is_finite() || timeout_secs < 0.001 {
            return OneSidedSatProbe::Declined(OneSidedSatDecline::InvalidTimeout);
        }
        let sense = match one_sided_row_sense(&self.parts.problem, row) {
            Ok(sense) => sense,
            Err(reason) => {
                return OneSidedSatProbe::Declined(OneSidedSatDecline::InvalidRow(reason));
            }
        };
        crate::ay_lib::probe_one_sided_sat(
            &self.parts.problem,
            timeout_secs,
            row,
            sense,
            self.config.ay_node_warm_time_limit,
            &self.parts.input_vars,
            &self.parts.output_vars,
            &self.ay_branch_hint_order(),
            hard_deadline,
        )
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
                self.config.ay_hard_deadline,
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

    fn timeout_envelope_solver(
        rows: usize,
        with_binary: bool,
        timeout_secs: f64,
        parallel_split: usize,
    ) -> MipSolver {
        let mut problem = ir::MilpProblem::new();
        let x = if with_binary {
            problem.add_integer_col(0.0, 0.0, 1.0)
        } else {
            problem.add_col(0.0, 0.0, 1.0)
        };
        for _ in 0..rows {
            problem.add_row(0.0, 1.0, [(x, 1.0)]);
        }
        let num_cols = problem.num_cols();
        MipSolver::new(
            MipParts {
                problem,
                input_vars: vec![x],
                output_vars: vec![x],
                binary_vars: with_binary.then_some(x).into_iter().collect(),
                binary_widths: with_binary.then_some(1.0).into_iter().collect(),
                num_cols,
            },
            MipConfig {
                backend: MipBackend::Ay,
                parallel_split,
                timeout_secs,
                ..MipConfig::default()
            },
        )
    }

    #[test]
    fn effective_timeout_matches_serial_window_split_outer_and_hard_clamp() {
        let serial_window = timeout_envelope_solver(8192, false, 20.0, 1);
        assert_eq!(
            serial_window.effective_feasibility_timeout_secs_with_window_value(Some("100")),
            100.0,
            "serial requested20/window100 historical wall is 100s"
        );

        let split_window = timeout_envelope_solver(8192, true, 20.0, 2);
        assert_eq!(
            split_window.effective_feasibility_timeout_secs_with_window_value(Some("100")),
            20.0,
            "phase-split's historical outer ledger dominates worker window floors"
        );

        let clamped = timeout_envelope_solver(8192, false, 100_000.0, 1);
        assert_eq!(
            clamped.effective_feasibility_timeout_secs_with_window_value(Some("200000")),
            86_400.0,
            "historical detached-wrapper clamp must bound the armed envelope"
        );

        let sub_gate = timeout_envelope_solver(1, false, 20.0, 1);
        assert_eq!(
            sub_gate.effective_feasibility_timeout_secs_with_window_value(Some("100")),
            20.0
        );
    }

    fn add_synthetic_relu(problem: &mut ir::MilpProblem, lb: f64, ub: f64) -> ir::Col {
        assert!(lb < 0.0 && ub > 0.0);
        let x = problem.add_col(0.0, lb, ub);
        add_synthetic_relu_for_input(problem, x, lb, ub).1
    }

    fn add_synthetic_relu_for_input(
        problem: &mut ir::MilpProblem,
        x: ir::Col,
        lb: f64,
        ub: f64,
    ) -> (ir::Col, ir::Col) {
        assert!(lb < 0.0 && ub > 0.0);
        let y = problem.add_col(0.0, 0.0, ub);
        let z = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(0.0, f64::INFINITY, [(y, 1.0), (x, -1.0)]);
        problem.add_row(f64::NEG_INFINITY, -lb, [(y, 1.0), (x, -1.0), (z, -lb)]);
        problem.add_row(f64::NEG_INFINITY, 0.0, [(y, 1.0), (z, -ub)]);
        (y, z)
    }

    fn add_synthetic_biased_relu(problem: &mut ir::MilpProblem, bias: f64) -> (ir::Col, ir::Col) {
        let source = problem.add_col(0.0, -4.0, 4.0);
        // Match `MipEncoder::encode_linear`: affine outputs are free columns.
        // The canonical Big-M rows themselves imply the supplied [-1, 1]
        // preactivation interval once the ReLU indicator is in [0, 1].
        let input = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        problem.add_row(bias, bias, [(input, 1.0), (source, -1.0)]);
        add_synthetic_relu_for_input(problem, input, -1.0, 1.0)
    }

    fn full_babsr_union_fixture(
        positive_coefficient_bias: f64,
    ) -> (ir::MilpProblem, ir::Col, ir::Col, ir::Col) {
        let mut problem = ir::MilpProblem::new();
        let (positive_output, positive_binary) =
            add_synthetic_biased_relu(&mut problem, positive_coefficient_bias);
        let (negative_output, negative_binary) = add_synthetic_biased_relu(&mut problem, 0.0);
        let objective = problem.add_col(0.0, -1.0, 1.0);
        // objective = positive_output - negative_output. The existing
        // intercept score can see only the negative-coefficient ReLU.
        problem.add_row(
            0.0,
            0.0,
            [
                (objective, 1.0),
                (positive_output, -1.0),
                (negative_output, 1.0),
            ],
        );
        (problem, objective, positive_binary, negative_binary)
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
    fn safenlp_shared_prefix_gate_accepts_only_literal_one() {
        assert_eq!(
            safenlp_shared_prefix_state_from_value(Some("1")),
            SafeNlpSharedPrefixState::Enabled
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
                safenlp_shared_prefix_state_from_value(value),
                SafeNlpSharedPrefixState::Disabled,
                "unexpectedly enabled for {value:?}"
            );
        }
    }

    #[test]
    fn safenlp_target_fsb_prefix_gate_accepts_only_literal_one() {
        assert_eq!(
            safenlp_target_fsb_prefix_state_from_value(Some("1")),
            SafeNlpTargetFsbPrefixState::Enabled
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
                safenlp_target_fsb_prefix_state_from_value(value),
                SafeNlpTargetFsbPrefixState::Disabled,
                "unexpectedly enabled for {value:?}"
            );
        }
    }

    #[test]
    fn marked_margin_lower_form_preserves_upper_and_negates_lower_orientation() {
        let mut upper = ir::MilpProblem::new();
        let upper_x = upper.add_col(0.0, -1.0, 1.0);
        let upper_y = upper.add_col(0.0, -1.0, 1.0);
        upper
            .add_margin_row(f64::NEG_INFINITY, 0.25, [(upper_x, 2.0), (upper_y, -3.0)])
            .expect("upper-bounded marked row");
        assert_eq!(
            marked_margin_lower_form(&upper),
            Some(vec![(upper_x, 2.0), (upper_y, -3.0)])
        );

        let mut lower = ir::MilpProblem::new();
        let lower_x = lower.add_col(0.0, -1.0, 1.0);
        let lower_y = lower.add_col(0.0, -1.0, 1.0);
        lower
            .add_margin_row(-0.25, f64::INFINITY, [(lower_x, 2.0), (lower_y, -3.0)])
            .expect("lower-bounded marked row");
        assert_eq!(
            marked_margin_lower_form(&lower),
            Some(vec![(lower_x, -2.0), (lower_y, 3.0)])
        );
    }

    fn shared_prefix_plan_solver(
        backend: MipBackend,
        hard_deadline: Option<std::time::Instant>,
        mark_margin: bool,
    ) -> (MipSolver, Vec<ir::Col>) {
        let mut problem = ir::MilpProblem::new();
        let binaries = (0..5)
            .map(|_| problem.add_integer_col(0.0, 0.0, 1.0))
            .collect::<Vec<_>>();
        if mark_margin {
            problem
                .add_margin_row(f64::NEG_INFINITY, 0.0, [(binaries[0], 1.0)])
                .expect("synthetic one-sided margin");
        }
        let num_cols = problem.num_cols();
        let solver = MipSolver::new(
            MipParts {
                problem,
                input_vars: Vec::new(),
                output_vars: Vec::new(),
                binary_vars: binaries.clone(),
                binary_widths: vec![1.0, 5.0, 2.0, 4.0, 3.0],
                num_cols,
            },
            MipConfig {
                backend,
                parallel_split: 16,
                timeout_secs: 10.0,
                ay_hard_deadline: hard_deadline,
                ..MipConfig::default()
            },
        );
        (solver, binaries)
    }

    fn target_fsb_prefix_plan_solver(
        scored_candidates: usize,
        duplicate_objective_identity: bool,
    ) -> (MipSolver, Vec<ir::Col>) {
        assert!((1..=4).contains(&scored_candidates));
        let mut problem = ir::MilpProblem::new();
        let mut outputs = Vec::new();
        let mut binaries = Vec::new();
        for _ in 0..8 {
            let input = problem.add_col(0.0, -1.0, 1.0);
            let (output, binary) = add_synthetic_relu_for_input(&mut problem, input, -1.0, 1.0);
            outputs.push(output);
            binaries.push(binary);
        }
        let mut objective = outputs[4..4 + scored_candidates]
            .iter()
            .copied()
            .map(|output| (output, -1.0))
            .collect::<Vec<_>>();
        if duplicate_objective_identity {
            objective.push((outputs[4], -1.0));
        }
        problem
            .add_margin_row(f64::NEG_INFINITY, 0.0, objective)
            .expect("synthetic one-sided target row");
        let num_cols = problem.num_cols();
        let mut solver = MipSolver::new(
            MipParts {
                problem,
                input_vars: Vec::new(),
                output_vars: Vec::new(),
                binary_vars: binaries.clone(),
                binary_widths: vec![8.0, 7.0, 6.0, 5.0, 1.0, 1.0, 1.0, 1.0],
                num_cols,
            },
            MipConfig {
                backend: MipBackend::Ay,
                parallel_split: 16,
                timeout_secs: 10.0,
                ay_hard_deadline: Some(
                    std::time::Instant::now() + std::time::Duration::from_secs(30),
                ),
                feasibility_ingress:
                    MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix,
                ..MipConfig::default()
            },
        );
        solver.safenlp_target_fsb_prefix = SafeNlpTargetFsbPrefixState::Enabled;
        (solver, binaries)
    }

    fn target_fsb_row45_union_plan_solver() -> (MipSolver, Vec<ir::Col>) {
        let mut problem = ir::MilpProblem::new();
        let mut binaries = Vec::new();
        for _ in 0..4 {
            let input = problem.add_col(0.0, -1.0, 1.0);
            let (_, binary) = add_synthetic_relu_for_input(&mut problem, input, -1.0, 1.0);
            binaries.push(binary);
        }

        // Full BaBSR sees this positive-coefficient, biased producer while
        // the intercept-only backup does not. The next four negative-form
        // candidates occur in both lists, so top4+top4 stably unions to five.
        let (biased_output, biased_binary) = add_synthetic_biased_relu(&mut problem, 2.0);
        binaries.push(biased_binary);
        let mut objective = vec![(biased_output, 1.0)];
        for _ in 0..4 {
            let input = problem.add_col(0.0, -1.0, 1.0);
            let (output, binary) = add_synthetic_relu_for_input(&mut problem, input, -1.0, 1.0);
            objective.push((output, -1.0));
            binaries.push(binary);
        }
        problem
            .add_margin_row(f64::NEG_INFINITY, 0.0, objective)
            .expect("synthetic row45-style target row");
        let num_cols = problem.num_cols();
        let mut solver = MipSolver::new(
            MipParts {
                problem,
                input_vars: Vec::new(),
                output_vars: Vec::new(),
                binary_vars: binaries.clone(),
                binary_widths: vec![9.0, 8.0, 7.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                num_cols,
            },
            MipConfig {
                backend: MipBackend::Ay,
                parallel_split: 16,
                timeout_secs: 10.0,
                ay_hard_deadline: Some(
                    std::time::Instant::now() + std::time::Duration::from_secs(30),
                ),
                feasibility_ingress:
                    MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix,
                ..MipConfig::default()
            },
        );
        solver.safenlp_target_fsb_prefix = SafeNlpTargetFsbPrefixState::Enabled;
        (solver, binaries)
    }

    #[test]
    fn safenlp_shared_prefix_reuses_ranked_live_split_handles() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let (solver, binaries) = shared_prefix_plan_solver(MipBackend::Ay, Some(deadline), false);

        assert!(
            solver
                .safenlp_shared_prefix_plan_with_gate(false, false)
                .is_none(),
            "the default-dark arm must retain the historical race"
        );
        assert_eq!(
            solver.safenlp_shared_prefix_plan_with_gate(true, false),
            Some(vec![binaries[1], binaries[3], binaries[4], binaries[2]]),
            "the canary must reuse the historical widest-first split plan"
        );
        assert!(
            solver
                .safenlp_shared_prefix_plan_with_gate(true, true)
                .is_none(),
            "AY rejects mixed external-incumbent ownership"
        );
    }

    #[test]
    fn safenlp_shared_prefix_admission_requires_ay_deadline_and_plain_model() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let (wrong_backend, _) =
            shared_prefix_plan_solver(MipBackend::AyProc, Some(deadline), false);
        assert!(wrong_backend
            .safenlp_shared_prefix_plan_with_gate(true, false)
            .is_none());

        let (no_deadline, _) = shared_prefix_plan_solver(MipBackend::Ay, None, false);
        assert!(no_deadline
            .safenlp_shared_prefix_plan_with_gate(true, false)
            .is_none());

        let (mut marked_margin, _) =
            shared_prefix_plan_solver(MipBackend::Ay, Some(deadline), true);
        assert!(
            marked_margin
                .safenlp_shared_prefix_plan_with_gate(true, false)
                .is_none(),
            "the shared-prefix canary must not silently compose with margin reframe"
        );
        marked_margin.safenlp_shared_prefix = SafeNlpSharedPrefixState::Enabled;
        assert!(
            marked_margin
                .safenlp_shared_prefix_plan(false)
                .expect("historical marker mismatch must decline, not error")
                .is_none(),
            "historical shared+marked behavior must retain its serial margin fallback"
        );
    }

    #[test]
    fn required_shared_prefix_ingress_is_typed_and_does_not_need_a_second_env_read() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let (mut solver, binaries) =
            shared_prefix_plan_solver(MipBackend::Ay, Some(deadline), false);
        solver.safenlp_shared_prefix = SafeNlpSharedPrefixState::Disabled;
        solver.config.feasibility_ingress = MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix;

        assert_eq!(
            solver
                .safenlp_shared_prefix_plan(false)
                .expect("valid required ingress"),
            Some(SafeNlpSharedPrefixPlan::Plain(vec![
                binaries[1],
                binaries[3],
                binaries[4],
                binaries[2],
            ]))
        );
    }

    #[test]
    fn required_marked_margin_ingress_is_typed_and_reuses_the_same_prefix() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let (mut solver, binaries) =
            shared_prefix_plan_solver(MipBackend::Ay, Some(deadline), true);
        solver.safenlp_shared_prefix = SafeNlpSharedPrefixState::Disabled;
        solver.safenlp_target_fsb_prefix = SafeNlpTargetFsbPrefixState::Disabled;
        solver.config.feasibility_ingress =
            MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix;

        assert_eq!(
            solver
                .safenlp_shared_prefix_plan(false)
                .expect("valid required marked-margin ingress"),
            Some(SafeNlpSharedPrefixPlan::MarkedMargin(vec![
                binaries[1],
                binaries[3],
                binaries[4],
                binaries[2],
            ])),
            "the marked lane must reuse the historical ranked split handles"
        );
    }

    #[test]
    fn target_fsb_prefix_uses_row_objective_union_disjoint_from_exact_fallback() {
        let (solver, binaries) = target_fsb_row45_union_plan_solver();

        assert_eq!(
            solver
                .safenlp_shared_prefix_plan(false)
                .expect("valid required marked-margin ingress"),
            Some(SafeNlpSharedPrefixPlan::MarkedMarginTargetFsb {
                fallback: [binaries[0], binaries[1], binaries[2], binaries[3]],
                candidates: vec![
                    binaries[4],
                    binaries[5],
                    binaries[6],
                    binaries[7],
                    binaries[8],
                ],
            }),
            "the row45-style full+intercept union may be independent of the exact old prefix"
        );
    }

    #[test]
    fn target_fsb_prefix_short_or_malformed_ranking_keeps_fixed_marked_api() {
        for (scored_candidates, duplicate_objective_identity) in [(3, false), (4, true)] {
            let (solver, binaries) =
                target_fsb_prefix_plan_solver(scored_candidates, duplicate_objective_identity);
            assert_eq!(
                solver
                    .safenlp_shared_prefix_plan(false)
                    .expect("valid required marked-margin ingress"),
                Some(SafeNlpSharedPrefixPlan::MarkedMargin(vec![
                    binaries[0],
                    binaries[1],
                    binaries[2],
                    binaries[3],
                ])),
                "optional selector decline must retain the exact existing fixed-prefix API"
            );
        }
    }

    #[test]
    fn target_fsb_prefix_does_not_change_plain_required_ingress() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let (mut solver, binaries) =
            shared_prefix_plan_solver(MipBackend::Ay, Some(deadline), false);
        solver.config.feasibility_ingress = MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix;
        solver.safenlp_target_fsb_prefix = SafeNlpTargetFsbPrefixState::Enabled;

        assert_eq!(
            solver
                .safenlp_shared_prefix_plan(false)
                .expect("valid required plain ingress"),
            Some(SafeNlpSharedPrefixPlan::Plain(vec![
                binaries[1],
                binaries[3],
                binaries[4],
                binaries[2],
            ]))
        );
    }

    #[test]
    fn required_plain_and_marked_ingress_reject_marker_identity_drift() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let (mut marked, _) = shared_prefix_plan_solver(MipBackend::Ay, Some(deadline), true);
        marked.config.feasibility_ingress = MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix;
        let error = marked
            .safenlp_shared_prefix_plan(false)
            .expect_err("required plain ingress must reject a marker");
        assert!(error
            .to_string()
            .contains("plain problem has a marked margin row"));

        let (mut unmarked, _) = shared_prefix_plan_solver(MipBackend::Ay, Some(deadline), false);
        unmarked.config.feasibility_ingress =
            MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix;
        let error = unmarked
            .safenlp_shared_prefix_plan(false)
            .expect_err("required marked ingress must reject a missing marker");
        assert!(error
            .to_string()
            .contains("marked-margin problem has no marked margin row"));
    }

    #[test]
    fn required_shared_prefix_decline_errors_while_historical_mode_falls_back() {
        let (mut solver, _) = shared_prefix_plan_solver(MipBackend::Ay, None, false);
        solver.safenlp_shared_prefix = SafeNlpSharedPrefixState::Enabled;

        assert_eq!(
            solver.config.feasibility_ingress,
            MipFeasibilityIngress::Historical
        );
        assert!(
            solver
                .safenlp_shared_prefix_plan(false)
                .expect("historical admission decline is not an error")
                .is_none(),
            "historical mode must retain its serial/phase-split fallback"
        );

        solver.config.feasibility_ingress = MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix;
        let error = solver
            .check_feasibility_with_warm_start(None)
            .expect_err("required mode must fail before any fallback solver is entered");
        assert!(
            error
                .to_string()
                .contains("required SafeNLP shared-binary-prefix session was not admitted"),
            "unexpected error: {error}"
        );
        assert!(
            error
                .to_string()
                .contains("caller deadline is absent or expired"),
            "the backend decline must remain distinguishable: {error}"
        );

        let mut cache = SplitUnsatCache::default();
        let cached_error = solver
            .check_feasibility_cached(None, &mut cache)
            .expect_err("cached entry must honor required ingress too");
        assert!(
            cached_error
                .to_string()
                .contains("required SafeNLP shared-binary-prefix session was not admitted"),
            "cached entry silently escaped required ingress: {cached_error}"
        );
    }

    #[test]
    fn required_shared_prefix_rejects_warm_start_before_backend_entry() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let (mut solver, _) = shared_prefix_plan_solver(MipBackend::Ay, Some(deadline), false);
        solver.config.feasibility_ingress = MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix;

        let error = solver
            .check_feasibility_with_warm_start(Some(&[]))
            .expect_err("required shared-prefix ingress owns a cold session");
        assert!(
            error.to_string().contains("external warm start is present"),
            "unexpected error: {error}"
        );

        let (mut marked, _) = shared_prefix_plan_solver(MipBackend::Ay, Some(deadline), true);
        marked.config.feasibility_ingress =
            MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix;
        let error = marked
            .check_feasibility_with_warm_start(Some(&[]))
            .expect_err("required marked shared-prefix ingress owns a cold session");
        assert!(
            error.to_string().contains("external warm start is present"),
            "unexpected error: {error}"
        );
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

    #[test]
    fn lower_form_ranking_backpropagates_through_affine_relu_chain() {
        let mut problem = ir::MilpProblem::new();
        let x1 = problem.add_col(0.0, -1.0, 1.0);
        let (y1, z1) = add_synthetic_relu_for_input(&mut problem, x1, -1.0, 1.0);

        // x2 = y1 - 0.25, hence x2 is in [-0.25, 0.75].
        let x2 = problem.add_col(0.0, -0.25, 0.75);
        problem.add_row(-0.25, -0.25, [(x2, 1.0), (y1, -1.0)]);
        let (y2, z2) = add_synthetic_relu_for_input(&mut problem, x2, -0.25, 0.75);

        // Minimize q = -y2. ReLU2 scores 1 * 0.1875. Its -0.75
        // upper-slope coefficient reaches ReLU1, which scores 0.75 * 0.5.
        let q = problem.add_col(0.0, -0.75, 0.0);
        problem.add_row(0.0, 0.0, [(q, 1.0), (y2, 1.0)]);

        assert_eq!(
            rank_canonical_relu_binaries_for_lower_form(
                &problem,
                &[(q, 1.0)],
                // Deliberately reverse caller order: scores, not input order,
                // must put the upstream high-impact ReLU first.
                &[z2, z1],
            ),
            vec![z1, z2]
        );
    }

    #[test]
    fn lower_form_affine_back_substitution_uses_the_negative_sign() {
        let mut problem = ir::MilpProblem::new();
        let x_positive = problem.add_col(0.0, -1.0, 1.0);
        let (y_positive, z_positive) =
            add_synthetic_relu_for_input(&mut problem, x_positive, -1.0, 1.0);
        let x_negative = problem.add_col(0.0, -1.0, 1.0);
        let (y_negative, z_negative) =
            add_synthetic_relu_for_input(&mut problem, x_negative, -1.0, 1.0);

        // q - 2*y_positive + y_negative = 0, so minimizing q gives
        // coefficients +2 and -1 respectively. Intercept-only BaBSR must
        // omit the positive-coefficient ReLU and retain the negative one.
        let q = problem.add_col(0.0, -1.0, 2.0);
        problem.add_row(0.0, 0.0, [(q, 1.0), (y_positive, -2.0), (y_negative, 1.0)]);

        assert_eq!(
            rank_canonical_relu_binaries_for_lower_form(
                &problem,
                &[(q, 1.0)],
                &[z_positive, z_negative],
            ),
            vec![z_negative]
        );
    }

    #[test]
    fn lower_form_ranking_fails_closed_on_malformed_nonfinite_or_duplicate_input() {
        let mut malformed = ir::MilpProblem::new();
        let x = malformed.add_col(0.0, -0.5, 1.0);
        let y = malformed.add_col(0.0, 0.0, 1.0);
        let z = malformed.add_integer_col(0.0, 0.0, 1.0);
        malformed.add_row(0.0, f64::INFINITY, [(y, 1.0), (x, -1.0)]);
        malformed.add_row(f64::NEG_INFINITY, 0.5, [(y, 1.0), (x, -1.0), (z, 0.5)]);
        assert!(
            rank_canonical_relu_binaries_for_lower_form(&malformed, &[(y, -1.0)], &[z]).is_empty(),
            "an incomplete canonical triple must produce no partial advice"
        );

        let mut canonical = ir::MilpProblem::new();
        let x = canonical.add_col(0.0, -0.5, 1.0);
        let (y, z) = add_synthetic_relu_for_input(&mut canonical, x, -0.5, 1.0);
        assert!(
            rank_canonical_relu_binaries_for_lower_form(&canonical, &[(y, f64::NAN)], &[z],)
                .is_empty(),
            "a non-finite objective must produce no advice"
        );
        assert!(
            rank_canonical_relu_binaries_for_lower_form(&canonical, &[(y, -1.0)], &[z, z],)
                .is_empty(),
            "duplicate binary identities must produce no advice"
        );
        assert!(
            rank_canonical_relu_binaries_for_lower_form(&canonical, &[(y, -1.0), (y, 0.0)], &[z],)
                .is_empty(),
            "duplicate objective identities must produce no advice"
        );

        // Even a byte-identical duplicate Big-M row is ambiguous metadata.
        canonical.add_row(f64::NEG_INFINITY, 0.0, [(y, 1.0), (z, -1.0)]);
        assert!(
            rank_canonical_relu_binaries_for_lower_form(&canonical, &[(y, -1.0)], &[z]).is_empty(),
            "duplicate canonical rows must produce no advice"
        );
    }

    #[test]
    fn full_babsr_union_uses_affine_bias_and_retains_intercept_backup() {
        let (zero_bias, objective, positive, negative) = full_babsr_union_fixture(0.0);
        assert_eq!(
            rank_canonical_relu_binaries_for_lower_form(
                &zero_bias,
                &[(objective, 1.0)],
                &[positive, negative],
            ),
            vec![negative],
            "the historical intercept scorer must remain bias-blind"
        );
        assert_eq!(
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                &zero_bias,
                &[(objective, 1.0)],
                &[positive, negative],
                1,
            ),
            vec![negative],
            "zero producer bias gives the positive-coefficient ReLU no full score"
        );

        let (biased, objective, positive, negative) = full_babsr_union_fixture(2.0);
        assert_eq!(
            rank_canonical_relu_binaries_for_lower_form(
                &biased,
                &[(objective, 1.0)],
                &[positive, negative],
            ),
            vec![negative],
            "changing only the producer bias must not change the old scorer"
        );
        assert_eq!(
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                &biased,
                &[(objective, 1.0)],
                &[positive, negative],
                1,
            ),
            vec![positive, negative],
            "full BaBSR should lead with the bias-sensitive candidate and union the intercept backup"
        );
        assert_eq!(
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                &biased,
                &[(objective, 1.0)],
                &[positive, negative],
                2,
            ),
            vec![positive, negative],
            "overlapping full/intercept lists must be stably deduplicated"
        );
        assert!(
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                &biased,
                &[(objective, 1.0)],
                &[positive, negative],
                0,
            )
            .is_empty(),
            "a zero per-score budget must request no candidates"
        );

        let mut tied = ir::MilpProblem::new();
        let (first_output, first_binary) = add_synthetic_biased_relu(&mut tied, 0.0);
        let (second_output, second_binary) = add_synthetic_biased_relu(&mut tied, 0.0);
        let objective = tied.add_col(0.0, -2.0, 0.0);
        tied.add_row(
            0.0,
            0.0,
            [(objective, 1.0), (first_output, 1.0), (second_output, 1.0)],
        );
        assert_eq!(
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                &tied,
                &[(objective, 1.0)],
                &[second_binary, first_binary],
                2,
            ),
            vec![second_binary, first_binary],
            "equal full and intercept scores must preserve caller order after deduplication"
        );

        let mut max_reduceop = ir::MilpProblem::new();
        let (zero_bias_output, zero_bias_binary) =
            add_synthetic_biased_relu(&mut max_reduceop, 0.0);
        let (biased_output, biased_binary) = add_synthetic_biased_relu(&mut max_reduceop, 2.0);
        let objective = max_reduceop.add_col(0.0, -2.0, 0.0);
        max_reduceop.add_row(
            0.0,
            0.0,
            [
                (objective, 1.0),
                (zero_bias_output, 1.0),
                (biased_output, 1.0),
            ],
        );
        assert_eq!(
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                &max_reduceop,
                &[(objective, 1.0)],
                &[zero_bias_binary, biased_binary],
                2,
            ),
            vec![zero_bias_binary, biased_binary],
            "the VNN-COMP max reduceop makes these full scores tie; a min reduceop would incorrectly promote the biased candidate"
        );
    }

    #[test]
    fn full_babsr_union_fails_closed_on_malformed_or_nonfinite_bias_metadata() {
        let mut incomplete = ir::MilpProblem::new();
        let input = incomplete.add_col(0.0, -0.5, 1.0);
        let output = incomplete.add_col(0.0, 0.0, 1.0);
        let binary = incomplete.add_integer_col(0.0, 0.0, 1.0);
        incomplete.add_row(0.0, f64::INFINITY, [(output, 1.0), (input, -1.0)]);
        incomplete.add_row(
            f64::NEG_INFINITY,
            0.5,
            [(output, 1.0), (input, -1.0), (binary, 0.5)],
        );
        assert!(
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                &incomplete,
                &[(output, -1.0)],
                &[binary],
                1,
            )
            .is_empty(),
            "an incomplete canonical ReLU must yield no union advice"
        );

        let mut overflow = ir::MilpProblem::new();
        let source = overflow.add_col(0.0, -4.0, 4.0);
        let input = overflow.add_col(0.0, -1.0, 1.0);
        overflow.add_row(1.0, 1.0, [(input, f64::from_bits(1)), (source, -1.0)]);
        let (output, binary) = add_synthetic_relu_for_input(&mut overflow, input, -1.0, 1.0);
        assert!(
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                &overflow,
                &[(output, -1.0)],
                &[binary],
                1,
            )
            .is_empty(),
            "an overflowing rhs/output-weight producer bias must fail closed"
        );

        let mut canonical = ir::MilpProblem::new();
        let input = canonical.add_col(0.0, -1.0, 1.0);
        let (output, binary) = add_synthetic_relu_for_input(&mut canonical, input, -1.0, 1.0);
        assert!(
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                &canonical,
                &[(output, -1.0)],
                &[binary, binary],
                1,
            )
            .is_empty(),
            "duplicate binary identities must fail closed"
        );
        assert!(
            rank_canonical_relu_binaries_for_lower_form_full_babsr_union(
                &canonical,
                &[(output, f64::NAN)],
                &[binary],
                1,
            )
            .is_empty(),
            "a non-finite target coefficient must fail closed"
        );
    }

    #[test]
    fn appended_decision_row_preserves_lower_form_advice_handles() {
        let mut problem = ir::MilpProblem::new();
        let x = problem.add_col(0.0, -1.0, 1.0);
        let (y, z) = add_synthetic_relu_for_input(&mut problem, x, -1.0, 1.0);
        let q = problem.add_col(0.0, -1.0, 0.0);
        problem.add_row(0.0, 0.0, [(q, 1.0), (y, 1.0)]);

        let before = rank_canonical_relu_binaries_for_lower_form(&problem, &[(q, 1.0)], &[z]);
        problem
            .add_margin_row(f64::NEG_INFINITY, -0.25, [(q, 1.0)])
            .expect("one-sided q decision row is a valid explicit margin");
        let after = rank_canonical_relu_binaries_for_lower_form(&problem, &[(q, 1.0)], &[z]);

        assert_eq!(before, vec![z]);
        assert_eq!(after, before);
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

    fn cache_problem(ub: f64, weight: f64) -> ir::MilpProblem {
        let mut problem = ir::MilpProblem::new();
        let x = problem.add_col(0.0, 0.0, ub);
        let z = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(f64::NEG_INFINITY, 1.5, vec![(x, weight), (z, 1.0)]);
        problem
    }

    fn fp(ub: f64, weight: f64) -> SplitFingerprint {
        SplitFingerprint {
            split_cols: vec![ir::Col(0), ir::Col(1)],
            num_subproblems: 4,
            problem: cache_problem(ub, weight),
        }
    }

    fn reconcile_fp(cache: &mut SplitUnsatCache, fingerprint: &SplitFingerprint) {
        cache.reconcile(
            &fingerprint.split_cols,
            fingerprint.num_subproblems,
            &fingerprint.problem,
        );
    }

    /// A recorded certified-Unsat assignment reads back as proven — the seed
    /// for the pre-seed-and-skip path in `check_feasibility_split` (slot
    /// filled with `Unsat { certified: true }`, worker never spawned).
    #[test]
    fn cache_records_certified_unsat_and_reports_proven() {
        let mut cache = SplitUnsatCache::default();
        reconcile_fp(&mut cache, &fp(1.0, 2.0));
        cache.record(2, &MipResult::Unsat { certified: true });
        assert!(cache.is_proven(2));
        assert!(!cache.is_proven(0));
        assert!(!cache.is_proven(1));
        assert!(!cache.is_proven(3));
        // A same-fingerprint reconcile (next round, identical problem) keeps
        // the proven set.
        reconcile_fp(&mut cache, &fp(1.0, 2.0));
        assert!(cache.is_proven(2));
    }

    /// ANY fingerprint drift clears the proven set (fail-closed): differing
    /// exact IR, split set, or subproblem count.
    #[test]
    fn cache_fingerprint_mismatch_clears_proven() {
        // Exact IR drift.
        let mut cache = SplitUnsatCache::default();
        reconcile_fp(&mut cache, &fp(1.0, 2.0));
        cache.record(2, &MipResult::Unsat { certified: true });
        reconcile_fp(&mut cache, &fp(1.5, 2.0));
        assert!(!cache.is_proven(2));

        // Split-set drift with the identical IR.
        let mut cache = SplitUnsatCache::default();
        reconcile_fp(&mut cache, &fp(1.0, 2.0));
        cache.record(1, &MipResult::Unsat { certified: true });
        let mut split_drift = fp(1.0, 2.0);
        split_drift.split_cols = vec![ir::Col(1), ir::Col(0)];
        reconcile_fp(&mut cache, &split_drift);
        assert!(!cache.is_proven(1));

        // Subproblem-count drift with the identical IR.
        let mut cache = SplitUnsatCache::default();
        reconcile_fp(&mut cache, &fp(1.0, 2.0));
        cache.record(1, &MipResult::Unsat { certified: true });
        let mut count_drift = fp(1.0, 2.0);
        count_drift.num_subproblems = 8;
        reconcile_fp(&mut cache, &count_drift);
        assert!(!cache.is_proven(1));
    }

    /// Sat, uncertified Unsat, Timeout, and Error are NEVER memoized.
    #[test]
    fn cache_never_records_sat_uncertified_timeout_or_error() {
        let mut cache = SplitUnsatCache::default();
        reconcile_fp(&mut cache, &fp(1.0, 2.0));
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

    /// Structural IR identity sees every bound and coefficient, so the memo
    /// clears on any encoded-problem drift without trusting a hash.
    #[test]
    fn exact_ir_identity_detects_bound_and_coefficient_drift() {
        let base = cache_problem(1.0, 2.0);
        assert_eq!(base, cache_problem(1.0, 2.0));
        assert_ne!(base, cache_problem(1.5, 2.0));
        assert_ne!(base, cache_problem(1.0, 2.5));
    }
}
