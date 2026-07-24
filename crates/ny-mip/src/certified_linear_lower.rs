// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof-carrying lower bounds for a fixed linear form over a small MILP.
//!
//! This is the reusable authority seam needed by post-prefix neural-network
//! oracles. The proposal route first asks AY to optimize only to *propose* a
//! binary32 lower bound, rounds it strictly downward, and checks it with a
//! separate decision solve. The decision-only route skips that
//! non-authoritative optimization and checks an explicit caller-selected
//! binary32 threshold directly:
//!
//! ```text
//! model ∧ linear_form <= proposed_lower
//! ```
//!
//! Before branch-and-bound, the decision route relaxes integrality and asks AY
//! for an exact certified lower row on the requested form. If that relaxation
//! lower is strictly above the selected threshold, its exact entailment is
//! enough: the relaxed feasible set contains the MILP feasible set. Otherwise
//! the original decision MILP remains the fail-closed fallback.
//!
//! A bound is returned only when one of those routes establishes the strict
//! separation and:
//!
//! 1. AY's relaxation entailment, root Farkas, or whole branch-tree
//!    certificate verifies exactly against the caller's lowered model; and
//! 2. every linear obligation is independently reconstructed from the
//!    original [`crate::ir::MilpProblem`] and accepted by
//!    [`ny_cert::check_entailment`] or [`ny_cert::check_farkas`].
//!
//! The optimization answer therefore has no authority.  It may be arbitrarily
//! wrong without producing a bound; only an independently replayed linear
//! proof can cross this API.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ay_milp::{
    BabSession, BoundSide, CertifiedRow as AyCertifiedRow, FactRef,
    FarkasCertificate as AyFarkasCertificate, LpSession, Outcome, Sense as AySense, SolveOpts,
    TreeNode,
};
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use ny_cert::{
    check_entailment, check_farkas, ConstraintKind, EntailmentCertificate,
    FarkasCertificate as NyFarkasCertificate, LinearConstraint, Rat,
};

use crate::ay_lib::{run_with_hard_deadline, solve_opts, to_ay_model};
use crate::error::MipError;
use crate::ir::{Col, MilpProblem, RowSpec};

/// Hard ceiling on the branch-tree certificate admitted by this API.
///
/// The tail oracle this was built for has roughly 18 unstable ReLUs.  A proof
/// larger than this is no longer a bounded tail proof and is declined before
/// ny-cert replay can turn it into an unpriced exact-arithmetic workload.
pub const CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES: usize = 4_096;

/// Explicit budgets for one proposal-and-proof attempt.
///
/// There is intentionally no `Default`: a caller must price both the
/// non-authoritative optimization and the authoritative decision proof.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertifiedLinearLowerBoundConfig {
    /// Wall-clock slice for AY's proposal optimization.
    pub proposal_timeout_secs: f64,
    /// Wall-clock slice for the separately solved exact relaxation entailment
    /// or MILP infeasibility proof.
    pub proof_timeout_secs: f64,
    /// Maximum AY branch-tree leaves accepted and replayed.
    pub max_tree_leaves: usize,
}

/// Explicit budget for certifying one caller-selected lower threshold.
///
/// Unlike [`CertifiedLinearLowerBoundConfig`], this decision-only route does
/// not spend time optimizing a non-authoritative proposal. The caller chooses
/// `q`; AY must prove either that the continuous relaxation entails a stronger
/// lower row or that `model ∧ linear_form <= q` is infeasible. Every proof
/// obligation is replayed exactly before `q` is returned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertifiedLinearLowerDecisionConfig {
    /// Wall-clock slice for the exact relaxation entailment or MILP
    /// infeasibility proof.
    pub proof_timeout_secs: f64,
    /// Maximum AY branch-tree leaves accepted and replayed.
    pub max_tree_leaves: usize,
}

/// A finite binary32 lower bound carrying two independent exact replays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertifiedLinearLowerBound {
    /// Certified binary32 lower bound. The optimization route rounds its
    /// proposal strictly downward; the decision-only route returns the exact
    /// caller-selected binary32 threshold proved by the certificate.
    pub lower: f32,
    /// Exact proof route that authorized this bound.
    pub proof_route: CertifiedLinearLowerProofRoute,
    /// Number of leaves in AY's case-split certificate (zero for relaxation
    /// entailment and root Farkas).
    pub ay_tree_leaves: usize,
    /// Number of linear proof obligations independently accepted by ny-cert.
    ///
    /// The legacy field name includes the root/tree Farkas obligations this
    /// API originally admitted. A relaxation-fast-path result instead counts
    /// its one exact entailment replay here; both are non-negative linear
    /// combinations checked independently from AY.
    pub ny_cert_farkas_replays: usize,
}

/// Exact proof route that authorized a certified linear lower bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertifiedLinearLowerProofRoute {
    /// The continuous relaxation entailed a strictly stronger lower row.
    RelaxationEntailment,
    /// The original decision MILP had an exact root Farkas certificate.
    RootFarkas,
    /// The original decision MILP had an exact branch-tree certificate.
    TreeFarkas,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReplayStats {
    proof_route: CertifiedLinearLowerProofRoute,
    tree_leaves: usize,
    linear_replays: usize,
}

/// Process-wide admission for the exact lower-bound worker.
///
/// [`run_with_hard_deadline`] deliberately detaches an AY worker that exceeds
/// its wall-clock slice.  Without a lease retained by that detached worker,
/// repeated region/objective calls could accumulate several large exact MILPs
/// after timeouts and turn a fail-closed performance lane into an RSS hazard.
/// A new attempt therefore declines while the preceding worker is still alive.
static CERTIFIED_LINEAR_LOWER_WORKER_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Opaque process-wide admission for one exact lower-bound worker.
///
/// Callers that must perform substantial model construction can acquire this
/// *before* encoding and pass it to
/// [`certify_linear_lower_bound_with_ay_admission`] or
/// [`certify_linear_lower_bound_at_with_ay_admission`]. If a worker detached
/// at a prior hard deadline is still alive, acquisition fails immediately and
/// the caller can shed the entire encoding/RSS workload.
///
/// The private field prevents callers from forging or dropping an unacquired
/// admission and reopening the lane underneath a live worker:
///
/// ```compile_fail
/// let _forged = ny_mip::CertifiedLinearLowerWorkerAdmission { _private: () };
/// ```
#[must_use = "dropping the admission reopens the exact-worker lane"]
pub struct CertifiedLinearLowerWorkerAdmission {
    _private: (),
}

impl CertifiedLinearLowerWorkerAdmission {
    /// Try to reserve the single process-wide exact AY worker slot.
    pub fn try_acquire() -> Option<Self> {
        CERTIFIED_LINEAR_LOWER_WORKER_ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self { _private: () })
    }
}

impl Drop for CertifiedLinearLowerWorkerAdmission {
    fn drop(&mut self) {
        CERTIFIED_LINEAR_LOWER_WORKER_ACTIVE.store(false, Ordering::Release);
    }
}

fn next_down_f32(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f32::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

/// Convert an exact rational proposal to a *strictly* smaller finite f32.
///
/// `ToPrimitive::to_f32` may round either way.  Moving one representable value
/// below a small numerical separation and checking the result again in exact
/// arithmetic removes any dependency on that rounding direction.  The
/// separation is authority-neutral (the exact proof remains mandatory), but
/// keeps AY's floating search from accepting an attained optimum through its
/// feasibility tolerance before it can construct the exact linear proof.
fn strict_outward_f32_lower(value: &BigRational) -> Option<f32> {
    let nearest = value.to_f32()?;
    if !nearest.is_finite() {
        return None;
    }
    let nearest64 = f64::from(nearest);
    let separation = 1.0e-6 * (1.0 + nearest64.abs());
    let separated = nearest64 - separation;
    if !separated.is_finite() {
        return None;
    }
    let mut lower = next_down_f32(separated as f32);
    for _ in 0..2 {
        if !lower.is_finite() {
            return None;
        }
        let exact = BigRational::from_float(f64::from(lower))?;
        if exact < *value {
            return Some(lower);
        }
        lower = next_down_f32(lower);
    }
    None
}

fn validate_config(config: CertifiedLinearLowerBoundConfig) -> Result<(), MipError> {
    for (field, value) in [
        ("proposal_timeout_secs", config.proposal_timeout_secs),
        ("proof_timeout_secs", config.proof_timeout_secs),
    ] {
        if !value.is_finite() || value <= 0.0 || value > 300.0 {
            return Err(MipError::Encoding(format!(
                "{field} must be finite and in (0, 300], got {value}"
            )));
        }
    }
    if config.max_tree_leaves == 0
        || config.max_tree_leaves > CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES
    {
        return Err(MipError::Encoding(format!(
            "max_tree_leaves must be in 1..={CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES}, got {}",
            config.max_tree_leaves
        )));
    }
    Ok(())
}

fn validate_decision_config(config: CertifiedLinearLowerDecisionConfig) -> Result<(), MipError> {
    if !config.proof_timeout_secs.is_finite()
        || config.proof_timeout_secs <= 0.0
        || config.proof_timeout_secs > 300.0
    {
        return Err(MipError::Encoding(format!(
            "proof_timeout_secs must be finite and in (0, 300], got {}",
            config.proof_timeout_secs
        )));
    }
    if config.max_tree_leaves == 0
        || config.max_tree_leaves > CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES
    {
        return Err(MipError::Encoding(format!(
            "max_tree_leaves must be in 1..={CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES}, got {}",
            config.max_tree_leaves
        )));
    }
    Ok(())
}

fn canonical_objective(
    problem: &MilpProblem,
    terms: &[(Col, f64)],
) -> Result<Vec<(Col, f64)>, MipError> {
    if terms.is_empty() {
        return Err(MipError::Encoding(
            "certified linear lower bound requires a nonempty objective".to_owned(),
        ));
    }
    let mut canonical = terms.to_vec();
    canonical.sort_unstable_by_key(|(col, _)| col.0);
    for (index, &(col, coeff)) in canonical.iter().enumerate() {
        if col.0 >= problem.num_cols() {
            return Err(MipError::Encoding(format!(
                "objective term {index} references column {}, but the model has {} columns",
                col.0,
                problem.num_cols()
            )));
        }
        if !coeff.is_finite() {
            return Err(MipError::Encoding(format!(
                "objective term {index} has non-finite coefficient"
            )));
        }
        if coeff == 0.0 {
            return Err(MipError::Encoding(format!(
                "objective term {index} has a zero coefficient"
            )));
        }
    }
    if canonical.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(MipError::Encoding(
            "certified linear objective contains duplicate columns".to_owned(),
        ));
    }
    Ok(canonical)
}

fn exact_f64(value: f64, what: &str) -> Result<BigRational, MipError> {
    BigRational::from_float(value)
        .ok_or_else(|| MipError::Encoding(format!("{what} must be finite")))
}

fn rat(value: &BigRational) -> Result<Rat, MipError> {
    Rat::from_bigints(value.numer().clone(), value.denom().clone())
        .map_err(|error| MipError::Solver(format!("ny-cert rational conversion failed: {error}")))
}

fn variable_name(index: usize) -> String {
    format!("x{index}")
}

/// Match `ay_milp::Model::add_row`'s duplicate merge exactly.
fn canonical_row_coefficients(row: &RowSpec) -> Result<Vec<(usize, f64)>, MipError> {
    let mut coeffs = row.coeffs.clone();
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
        return Err(MipError::Encoding(
            "row duplicate merge produced a non-finite coefficient".to_owned(),
        ));
    }
    coeffs.retain(|&(_, coeff)| coeff != 0.0);
    Ok(coeffs)
}

fn linear_constraint(
    kind: ConstraintKind,
    coeffs: impl IntoIterator<Item = (usize, BigRational)>,
    constant: &BigRational,
) -> Result<LinearConstraint, MipError> {
    let mut coefficients = BTreeMap::new();
    for (index, coeff) in coeffs {
        coefficients.insert(variable_name(index), rat(&coeff)?);
    }
    Ok(LinearConstraint {
        kind,
        coefficients,
        constant: rat(constant)?,
    })
}

fn fact_constraint(
    problem: &MilpProblem,
    fact: FactRef,
    effective_lb: &[Option<BigRational>],
    effective_ub: &[Option<BigRational>],
) -> Result<LinearConstraint, MipError> {
    match fact {
        FactRef::RowBound { row, side } => {
            let row_index = row.index();
            let row = problem.rows().get(row_index).ok_or_else(|| {
                MipError::Solver(format!("AY certificate references missing row {row_index}"))
            })?;
            let coeffs = canonical_row_coefficients(row)?
                .into_iter()
                .map(|(index, coeff)| Ok((index, exact_f64(coeff, "row coefficient")?)))
                .collect::<Result<Vec<_>, MipError>>()?;
            match side {
                BoundSide::Lower => linear_constraint(
                    ConstraintKind::Ge,
                    coeffs,
                    &exact_f64(row.lb, "row lower bound")?,
                ),
                BoundSide::Upper => linear_constraint(
                    ConstraintKind::Le,
                    coeffs,
                    &exact_f64(row.ub, "row upper bound")?,
                ),
            }
        }
        FactRef::ColBound { col, side } => {
            let index = col.index();
            if index >= problem.num_cols() {
                return Err(MipError::Solver(format!(
                    "AY certificate references missing column {index}"
                )));
            }
            let bound = match side {
                BoundSide::Lower => effective_lb.get(index).and_then(Clone::clone),
                BoundSide::Upper => effective_ub.get(index).and_then(Clone::clone),
            }
            .ok_or_else(|| {
                MipError::Solver(format!(
                    "AY certificate references an infinite effective column bound at {index}"
                ))
            })?;
            linear_constraint(
                match side {
                    BoundSide::Lower => ConstraintKind::Ge,
                    BoundSide::Upper => ConstraintKind::Le,
                },
                [(index, BigRational::from_integer(1.into()))],
                &bound,
            )
        }
        _ => Err(MipError::Solver(
            "AY certificate references an unsupported fact kind".to_owned(),
        )),
    }
}

fn replay_farkas_with_ny_cert(
    problem: &MilpProblem,
    cert: &AyFarkasCertificate,
    effective_lb: &[Option<BigRational>],
    effective_ub: &[Option<BigRational>],
) -> Result<(), MipError> {
    if ny_cert::rational::poisoned() {
        return Err(MipError::Solver(
            "ny-cert rational arena was already poisoned".to_owned(),
        ));
    }
    let mut constraints = Vec::with_capacity(cert.multipliers.len());
    let mut multipliers = Vec::with_capacity(cert.multipliers.len());
    for multiplier in &cert.multipliers {
        if multiplier.coeff <= BigRational::zero() {
            return Err(MipError::Solver(
                "AY certificate contains a nonpositive multiplier".to_owned(),
            ));
        }
        constraints.push(fact_constraint(
            problem,
            multiplier.fact,
            effective_lb,
            effective_ub,
        )?);
        multipliers.push(rat(&multiplier.coeff)?);
    }
    let cert = NyFarkasCertificate {
        constraints,
        multipliers,
    };
    check_farkas(&cert)
        .map_err(|error| MipError::Solver(format!("ny-cert Farkas replay failed: {error}")))?;
    if ny_cert::rational::poisoned() {
        return Err(MipError::Solver(
            "ny-cert rational arena became poisoned during Farkas replay".to_owned(),
        ));
    }
    Ok(())
}

fn base_column_bounds(
    problem: &MilpProblem,
) -> Result<(Vec<Option<BigRational>>, Vec<Option<BigRational>>), MipError> {
    let mut lower = Vec::with_capacity(problem.num_cols());
    let mut upper = Vec::with_capacity(problem.num_cols());
    for (index, col) in problem.cols().iter().enumerate() {
        lower.push(if col.lb.is_finite() {
            Some(exact_f64(col.lb, &format!("column {index} lower bound"))?)
        } else {
            None
        });
        upper.push(if col.ub.is_finite() {
            Some(exact_f64(col.ub, &format!("column {index} upper bound"))?)
        } else {
            None
        });
    }
    Ok((lower, upper))
}

fn canonical_exact_objective(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
) -> Result<BTreeMap<usize, BigRational>, MipError> {
    let mut exact = BTreeMap::new();
    for &(col, coefficient) in objective {
        if col.0 >= problem.num_cols() {
            return Err(MipError::Encoding(format!(
                "relaxation objective references missing column {}",
                col.0
            )));
        }
        let coefficient = exact_f64(coefficient, "relaxation objective coefficient")?;
        if exact.insert(col.0, coefficient).is_some() {
            return Err(MipError::Encoding(
                "relaxation objective contains duplicate columns".to_owned(),
            ));
        }
    }
    Ok(exact)
}

fn canonical_certified_row(
    problem: &MilpProblem,
    row: &AyCertifiedRow,
) -> Result<BTreeMap<usize, BigRational>, MipError> {
    let mut exact = BTreeMap::<usize, BigRational>::new();
    for &(col, ref coefficient) in &row.coeffs {
        let col = usize::try_from(col).map_err(|_| {
            MipError::Solver("AY certified-row column does not fit usize".to_owned())
        })?;
        if col >= problem.num_cols() {
            return Err(MipError::Solver(format!(
                "AY certified row references missing column {col}"
            )));
        }
        *exact.entry(col).or_default() += coefficient;
    }
    exact.retain(|_, coefficient| !coefficient.is_zero());
    Ok(exact)
}

fn replay_entailment_with_ny_cert(
    problem: &MilpProblem,
    row: &AyCertifiedRow,
) -> Result<(), MipError> {
    if ny_cert::rational::poisoned() {
        return Err(MipError::Solver(
            "ny-cert rational arena was already poisoned".to_owned(),
        ));
    }
    let (lower, upper) = base_column_bounds(problem)?;
    let mut premises = Vec::with_capacity(row.multipliers.len());
    let mut multipliers = Vec::with_capacity(row.multipliers.len());
    for multiplier in &row.multipliers {
        if multiplier.coeff < BigRational::zero() {
            return Err(MipError::Solver(
                "AY certified row contains a negative multiplier".to_owned(),
            ));
        }
        if multiplier.coeff.is_zero() {
            continue;
        }
        premises.push(fact_constraint(problem, multiplier.fact, &lower, &upper)?);
        multipliers.push(rat(&multiplier.coeff)?);
    }
    let conclusion = linear_constraint(
        ConstraintKind::Ge,
        canonical_certified_row(problem, row)?,
        &row.lb,
    )?;
    let certificate = EntailmentCertificate {
        premises,
        multipliers,
        conclusion,
    };
    check_entailment(&certificate)
        .map_err(|error| MipError::Solver(format!("ny-cert entailment replay failed: {error}")))?;
    if ny_cert::rational::poisoned() {
        return Err(MipError::Solver(
            "ny-cert rational arena became poisoned during entailment replay".to_owned(),
        ));
    }
    Ok(())
}

fn replay_root_farkas(
    problem: &MilpProblem,
    cert: &AyFarkasCertificate,
) -> Result<ReplayStats, MipError> {
    let (lower, upper) = base_column_bounds(problem)?;
    replay_farkas_with_ny_cert(problem, cert, &lower, &upper)?;
    Ok(ReplayStats {
        proof_route: CertifiedLinearLowerProofRoute::RootFarkas,
        tree_leaves: 0,
        linear_replays: 1,
    })
}

fn replay_tree_farkas(
    problem: &MilpProblem,
    root: &TreeNode,
    max_tree_leaves: usize,
) -> Result<ReplayStats, MipError> {
    let (mut lower, mut upper) = base_column_bounds(problem)?;
    enum Step<'a> {
        Visit(&'a TreeNode),
        Tighten {
            col: usize,
            upper: bool,
            to: Box<BigRational>,
            child: &'a TreeNode,
        },
        Restore {
            col: usize,
            upper: bool,
        },
    }
    let mut undo: Vec<Option<BigRational>> = Vec::new();
    let mut stack = vec![Step::Visit(root)];
    let mut leaves = 0usize;
    while let Some(step) = stack.pop() {
        match step {
            Step::Visit(TreeNode::Leaf { farkas }) => {
                leaves = leaves
                    .checked_add(1)
                    .ok_or_else(|| MipError::Solver("AY tree leaf count overflow".to_owned()))?;
                if leaves > max_tree_leaves {
                    return Err(MipError::Solver(format!(
                        "AY tree has more than the admitted {max_tree_leaves} leaves"
                    )));
                }
                replay_farkas_with_ny_cert(problem, farkas, &lower, &upper)?;
            }
            Step::Visit(TreeNode::Split { col, cut, lo, hi }) => {
                let index = col.index();
                let Some(spec) = problem.cols().get(index) else {
                    return Err(MipError::Solver(format!(
                        "AY tree splits on missing column {index}"
                    )));
                };
                if !spec.integer || !cut.is_integer() {
                    return Err(MipError::Solver(format!(
                        "AY tree has an invalid integer split on column {index}"
                    )));
                }
                stack.push(Step::Restore {
                    col: index,
                    upper: false,
                });
                stack.push(Step::Tighten {
                    col: index,
                    upper: false,
                    to: Box::new(cut.clone() + BigRational::from_integer(1.into())),
                    child: hi,
                });
                stack.push(Step::Restore {
                    col: index,
                    upper: true,
                });
                stack.push(Step::Tighten {
                    col: index,
                    upper: true,
                    to: Box::new(cut.clone()),
                    child: lo,
                });
            }
            Step::Tighten {
                col,
                upper: is_upper,
                to,
                child,
            } => {
                let to = *to;
                let slot = if is_upper {
                    &mut upper[col]
                } else {
                    &mut lower[col]
                };
                undo.push(slot.clone());
                *slot = Some(match slot.take() {
                    Some(previous) => {
                        if is_upper {
                            previous.min(to)
                        } else {
                            previous.max(to)
                        }
                    }
                    None => to,
                });
                stack.push(Step::Visit(child));
            }
            Step::Restore {
                col,
                upper: is_upper,
            } => {
                let previous = undo.pop().ok_or_else(|| {
                    MipError::Solver("unbalanced AY tree replay stack".to_owned())
                })?;
                if is_upper {
                    upper[col] = previous;
                } else {
                    lower[col] = previous;
                }
            }
        }
    }
    if leaves == 0 || !undo.is_empty() {
        return Err(MipError::Solver(
            "AY tree replay was empty or structurally unbalanced".to_owned(),
        ));
    }
    Ok(ReplayStats {
        proof_route: CertifiedLinearLowerProofRoute::TreeFarkas,
        tree_leaves: leaves,
        linear_replays: leaves,
    })
}

fn solve_proposal(
    problem: MilpProblem,
    objective: Vec<(Col, f64)>,
    timeout_secs: f64,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<BigRational>, MipError> {
    run_with_hard_deadline(timeout_secs, "linear-lower-proposal", move || {
        // If the hard deadline detaches this worker, retain process-wide
        // admission until the AY session actually exits.
        let _worker_lease = worker_lease;
        let mut model = to_ay_model(&problem)?;
        let mut ay_objective = Vec::with_capacity(objective.len());
        for (col, coeff) in objective {
            let ay_col = model.col_at(col.0).ok_or_else(|| {
                MipError::Encoding(format!("objective column {} disappeared", col.0))
            })?;
            ay_objective.push((ay_col, coeff));
        }
        model.set_objective(&ay_objective, AySense::Minimize);
        let opts = solve_opts(timeout_secs).with_tree_cert_leaves(0);
        let mut session =
            BabSession::new(model, &opts).map_err(|error| MipError::Solver(error.to_string()))?;
        match session
            .check()
            .map_err(|error| MipError::Solver(error.to_string()))?
        {
            Outcome::Optimal { value, .. } => Ok(Some(value)),
            _ => Ok(None),
        }
    })
    .map(Option::flatten)
}

fn try_relaxed_linear_lower_proof(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    opts: &SolveOpts,
) -> Result<Option<ReplayStats>, MipError> {
    let mut relaxed_problem = problem.clone();
    relaxed_problem.relax_integrality();
    let relaxed_model = to_ay_model(&relaxed_problem)?;
    let mut ay_objective = Vec::with_capacity(objective.len());
    for &(col, coefficient) in objective {
        let ay_col = relaxed_model.col_at(col.0).ok_or_else(|| {
            MipError::Encoding(format!("relaxation objective column {} disappeared", col.0))
        })?;
        ay_objective.push((ay_col, coefficient));
    }
    let mut session = LpSession::new(&relaxed_model, opts)
        .map_err(|error| MipError::Solver(error.to_string()))?;
    let Some(row) = session.harvest_cut(&ay_objective, AySense::Minimize) else {
        return Ok(None);
    };
    row.verify(&relaxed_model).map_err(|error| {
        MipError::Solver(format!(
            "AY relaxation entailment failed independent verification: {error}"
        ))
    })?;
    // The relaxed copy differs only in integrality metadata. Match and replay
    // the certified row against the caller's original rows and bounds so that
    // integrality never becomes an implicit linear premise.
    let expected = canonical_exact_objective(problem, objective)?;
    let actual = canonical_certified_row(problem, &row)?;
    if actual != expected {
        return Err(MipError::Solver(
            "AY relaxation entailment does not match the requested objective".to_owned(),
        ));
    }
    let requested = exact_f64(
        f64::from(requested_lower),
        "requested certified lower threshold",
    )?;
    if row.lb <= requested {
        return Ok(None);
    }
    replay_entailment_with_ny_cert(problem, &row)?;
    Ok(Some(ReplayStats {
        proof_route: CertifiedLinearLowerProofRoute::RelaxationEntailment,
        tree_leaves: 0,
        linear_replays: 1,
    }))
}

fn solve_and_replay_milp_proof(
    problem: &MilpProblem,
    opts: &SolveOpts,
    max_tree_leaves: usize,
) -> Result<Option<ReplayStats>, MipError> {
    let model = to_ay_model(problem)?;
    let mut session =
        BabSession::new(model, opts).map_err(|error| MipError::Solver(error.to_string()))?;
    let outcome = session
        .check()
        .map_err(|error| MipError::Solver(error.to_string()))?;
    match outcome {
        Outcome::Infeasible {
            cert: Some(cert), ..
        } => {
            cert.verify(session.model()).map_err(|error| {
                MipError::Solver(format!(
                    "AY root Farkas certificate failed independent verification: {error}"
                ))
            })?;
            replay_root_farkas(problem, &cert).map(Some)
        }
        Outcome::Infeasible {
            cert: None,
            tree_cert: Some(tree),
        } => {
            if tree.num_leaves() > max_tree_leaves {
                return Ok(None);
            }
            tree.verify(session.model()).map_err(|error| {
                MipError::Solver(format!(
                    "AY tree certificate failed independent verification: {error}"
                ))
            })?;
            replay_tree_farkas(problem, &tree.root, max_tree_leaves).map(Some)
        }
        _ => Ok(None),
    }
}

fn solve_and_replay_decision_proof(
    problem: MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    timeout_secs: f64,
    max_tree_leaves: usize,
    worker_lease: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<ReplayStats>, MipError> {
    let wall = Duration::from_secs_f64(timeout_secs);
    let proof_deadline = Instant::now()
        .checked_add(wall)
        .ok_or_else(|| MipError::Encoding("linear-lower proof deadline overflow".to_owned()))?;
    run_with_hard_deadline(timeout_secs, "linear-lower-proof", move || {
        // Exact replay can outlive the caller's hard deadline too.  Keep new
        // proof attempts shed until this detached worker has really stopped.
        let _worker_lease = worker_lease;
        let relaxation_opts = solve_opts(timeout_secs)
            .with_deadline(proof_deadline)
            .with_tree_cert_leaves(0)
            .with_require_certificates(true);
        if let Some(replay) =
            try_relaxed_linear_lower_proof(&problem, &objective, requested_lower, &relaxation_opts)?
        {
            return Ok(Some(replay));
        }

        let Some(remaining) = proof_deadline.checked_duration_since(Instant::now()) else {
            return Ok(None);
        };
        if remaining.is_zero() {
            return Ok(None);
        }
        let mut decision = problem.clone();
        decision.add_row(
            f64::NEG_INFINITY,
            f64::from(requested_lower),
            objective.iter().copied(),
        );
        let fallback_opts = solve_opts(remaining.as_secs_f64())
            .with_deadline(proof_deadline)
            .with_tree_cert_leaves(max_tree_leaves)
            .with_require_certificates(true);
        solve_and_replay_milp_proof(&decision, &fallback_opts, max_tree_leaves)
    })
    .map(Option::flatten)
}

/// Propose and independently certify a lower bound on a fixed linear form.
///
/// `problem` may contain continuous and binary columns, but must not carry a
/// marked margin row: this function appends its own typed decision row and
/// refuses any competing solver-routing identity.  `objective` is interpreted
/// over the existing columns and may not contain duplicates or zeros.
///
/// `Ok(None)` is the ordinary fail-closed outcome: AY did not complete the
/// proposal, could not export a strict relaxation entailment or bounded proof
/// tree, or the separate decision problem was feasible/inconclusive. A
/// malformed call or a disagreement between exact checkers is an error.
pub fn certify_linear_lower_bound_with_ay(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    config: CertifiedLinearLowerBoundConfig,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    let Some(proposal_admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "certified linear lower bound declined: a prior exact AY worker is still active"
        );
        return Ok(None);
    };
    certify_linear_lower_bound_with_ay_prepared(problem, objective, config, proposal_admission)
}

/// Certify a fixed linear lower bound using an admission acquired before model
/// construction.
///
/// This is equivalent to [`certify_linear_lower_bound_with_ay`], except the
/// caller supplies the opaque exact-worker slot. It exists so large encoders can
/// decline before allocating their model when a prior hard-deadline worker is
/// still alive.
pub fn certify_linear_lower_bound_with_ay_admission(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    config: CertifiedLinearLowerBoundConfig,
    proposal_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    certify_linear_lower_bound_with_ay_prepared(problem, objective, config, proposal_admission)
}

/// Prove a caller-selected lower threshold without first optimizing a proposal.
///
/// The returned `lower` is exactly `requested_lower`. It has authority only
/// because AY either derived a strictly stronger objective row from the
/// continuous relaxation or proved
/// `problem ∧ objective <= requested_lower` infeasible. The resulting exact
/// linear obligation is independently replayed by ny-cert. A feasible
/// equality, timeout, missing bounded certificate, or replay disagreement
/// returns no bound (or an error for malformed/checker-divergent input).
pub fn certify_linear_lower_bound_at_with_ay(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    config: CertifiedLinearLowerDecisionConfig,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_decision_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    let Some(proof_admission) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "certified linear lower threshold declined: a prior exact AY worker is still active"
        );
        return Ok(None);
    };
    certify_linear_lower_bound_at_with_ay_prepared(
        problem,
        objective,
        requested_lower,
        config,
        proof_admission,
    )
}

/// Decision-only counterpart to
/// [`certify_linear_lower_bound_with_ay_admission`].
///
/// The caller supplies an admission acquired before potentially large model
/// construction. No separate proposal worker is launched: the lease is
/// retained by the exact proof worker until that worker really exits.
pub fn certify_linear_lower_bound_at_with_ay_admission(
    problem: &MilpProblem,
    objective: &[(Col, f64)],
    requested_lower: f32,
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    validate_decision_config(config)?;
    if problem.margin_row().is_some() {
        return Err(MipError::Encoding(
            "certified linear lower bound refuses a pre-marked decision row".to_owned(),
        ));
    }
    if !requested_lower.is_finite() {
        return Err(MipError::Encoding(
            "requested certified lower threshold must be finite".to_owned(),
        ));
    }
    let objective = canonical_objective(problem, objective)?;
    certify_linear_lower_bound_at_with_ay_prepared(
        problem,
        objective,
        requested_lower,
        config,
        proof_admission,
    )
}

fn certify_linear_lower_bound_at_with_ay_prepared(
    problem: &MilpProblem,
    objective: Vec<(Col, f64)>,
    requested_lower: f32,
    config: CertifiedLinearLowerDecisionConfig,
    proof_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let Some(replay) = solve_and_replay_decision_proof(
        problem.clone(),
        objective,
        requested_lower,
        config.proof_timeout_secs,
        config.max_tree_leaves,
        proof_admission,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(CertifiedLinearLowerBound {
        lower: requested_lower,
        proof_route: replay.proof_route,
        ay_tree_leaves: replay.tree_leaves,
        ny_cert_farkas_replays: replay.linear_replays,
    }))
}

fn certify_linear_lower_bound_with_ay_prepared(
    problem: &MilpProblem,
    objective: Vec<(Col, f64)>,
    config: CertifiedLinearLowerBoundConfig,
    proposal_admission: CertifiedLinearLowerWorkerAdmission,
) -> Result<Option<CertifiedLinearLowerBound>, MipError> {
    let Some(optimum) = solve_proposal(
        problem.clone(),
        objective.clone(),
        config.proposal_timeout_secs,
        proposal_admission,
    )?
    else {
        return Ok(None);
    };
    let Some(lower) = strict_outward_f32_lower(&optimum) else {
        return Ok(None);
    };

    let Some(proof_lease) = CertifiedLinearLowerWorkerAdmission::try_acquire() else {
        tracing::warn!(
            "certified linear lower bound declined before replay: another exact AY worker is active"
        );
        return Ok(None);
    };
    let Some(replay) = solve_and_replay_decision_proof(
        problem.clone(),
        objective,
        lower,
        config.proof_timeout_secs,
        config.max_tree_leaves,
        proof_lease,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(CertifiedLinearLowerBound {
        lower,
        proof_route: replay.proof_route,
        ay_tree_leaves: replay.tree_leaves,
        ny_cert_farkas_replays: replay.linear_replays,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    static CERTIFIED_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn config() -> CertifiedLinearLowerBoundConfig {
        CertifiedLinearLowerBoundConfig {
            proposal_timeout_secs: 10.0,
            proof_timeout_secs: 10.0,
            max_tree_leaves: 64,
        }
    }

    fn decision_config() -> CertifiedLinearLowerDecisionConfig {
        CertifiedLinearLowerDecisionConfig {
            proof_timeout_secs: 10.0,
            max_tree_leaves: 64,
        }
    }

    #[test]
    fn exact_worker_lease_sheds_overlap_and_reopens_after_exit() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let lease =
            CertifiedLinearLowerWorkerAdmission::try_acquire().expect("first worker admitted");
        assert!(
            CertifiedLinearLowerWorkerAdmission::try_acquire().is_none(),
            "an overlapping exact worker must be shed"
        );
        drop(lease);
        let reopened = CertifiedLinearLowerWorkerAdmission::try_acquire()
            .expect("admission reopens after exit");
        drop(reopened);
    }

    #[test]
    fn detached_worker_retains_admission_until_actual_exit() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let admission =
            CertifiedLinearLowerWorkerAdmission::try_acquire().expect("worker admitted");
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let result: Result<Option<()>, MipError> =
            run_with_hard_deadline(0.01, "linear-lower-lease-test", move || {
                let _admission = admission;
                let _ = release_rx.recv();
                Ok(())
            });
        assert!(
            matches!(result, Ok(None)),
            "the caller must detach at its hard deadline"
        );
        assert!(
            CertifiedLinearLowerWorkerAdmission::try_acquire().is_none(),
            "the detached worker must retain admission"
        );
        release_tx.send(()).expect("release detached worker");

        let wait_until = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(reopened) = CertifiedLinearLowerWorkerAdmission::try_acquire() {
                drop(reopened);
                break;
            }
            assert!(
                Instant::now() < wait_until,
                "admission did not reopen after the detached worker exited"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn continuous_lower_bound_requires_exact_linear_replay() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);

        let certified = certify_linear_lower_bound_with_ay(&problem, &[(x, 1.0)], config())
            .expect("solver/checkers agree")
            .expect("root proof");
        assert!(certified.lower < 1.0);
        assert!(certified.lower > 0.99);
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::RelaxationEntailment
        );
        assert_eq!(certified.ay_tree_leaves, 0);
        assert_eq!(certified.ny_cert_farkas_replays, 1);
    }

    #[test]
    fn relaxation_entailment_requires_strict_exact_separation() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);
        let objective = canonical_objective(&problem, &[(x, 1.0)]).unwrap();
        let opts = solve_opts(10.0).with_require_certificates(true);

        let replay = try_relaxed_linear_lower_proof(&problem, &objective, 0.99, &opts)
            .expect("AY and ny-cert agree")
            .expect("the relaxed optimum strictly exceeds 0.99");
        assert_eq!(
            replay.proof_route,
            CertifiedLinearLowerProofRoute::RelaxationEntailment
        );
        assert_eq!(replay.tree_leaves, 0);
        assert_eq!(replay.linear_replays, 1);

        assert!(
            try_relaxed_linear_lower_proof(&problem, &objective, 1.0, &opts)
                .expect("equality is an ordinary non-proof")
                .is_none(),
            "an attained optimum must not prove the non-strict decision row infeasible"
        );
    }

    #[test]
    fn selected_continuous_threshold_skips_proposal_and_replays_linear_proof() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);

        let certified =
            certify_linear_lower_bound_at_with_ay(&problem, &[(x, 1.0)], 0.99, decision_config())
                .expect("solver/checkers agree")
                .expect("strictly separated root proof");
        assert_eq!(certified.lower.to_bits(), 0.99_f32.to_bits());
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::RelaxationEntailment
        );
        assert_eq!(certified.ay_tree_leaves, 0);
        assert_eq!(certified.ny_cert_farkas_replays, 1);

        assert!(
            certify_linear_lower_bound_at_with_ay(&problem, &[(x, 1.0)], 1.0, decision_config(),)
                .expect("feasible equality is an ordinary non-certificate")
                .is_none(),
            "x=1 satisfies the non-strict decision row, so q=1 is not certified"
        );
    }

    #[test]
    fn infeasible_relaxation_falls_back_to_root_farkas() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);
        problem.add_row(f64::NEG_INFINITY, 0.0, [(x, 1.0)]);

        let certified =
            certify_linear_lower_bound_at_with_ay(&problem, &[(x, 1.0)], 0.0, decision_config())
                .expect("solver/checkers agree")
                .expect("infeasible model has a root proof");
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::RootFarkas
        );
        assert_eq!(certified.ay_tree_leaves, 0);
        assert_eq!(certified.ny_cert_farkas_replays, 1);
    }

    #[test]
    fn selected_integral_threshold_replays_every_tree_leaf() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        let y = problem.add_integer_col(0.0, 0.0, 1.0);
        let z = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(1.5, f64::INFINITY, [(x, 1.0), (y, 1.0), (z, 1.0)]);
        let objective = canonical_objective(&problem, &[(x, 1.0), (y, 1.0), (z, 1.0)]).unwrap();
        let opts = solve_opts(10.0).with_require_certificates(true);
        assert!(
            try_relaxed_linear_lower_proof(&problem, &objective, 1.75, &opts)
                .expect("relaxation solve/checkers agree")
                .is_none(),
            "LP optimum 1.5 cannot certify an integer-only lower threshold of 1.75"
        );

        let certified = certify_linear_lower_bound_at_with_ay(
            &problem,
            &[(x, 1.0), (y, 1.0), (z, 1.0)],
            1.75,
            decision_config(),
        )
        .expect("solver/checkers agree")
        .expect("integer-separated tree proof");
        assert_eq!(certified.lower, 1.75);
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert!(certified.ay_tree_leaves >= 2);
        assert_eq!(certified.ny_cert_farkas_replays, certified.ay_tree_leaves);
    }

    #[test]
    fn integrality_gap_lower_bound_replays_every_tree_leaf() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        // LP relaxation: x=y=z=0.5 gives x+y+z=1.5.
        // Binary model: x+y+z>=1.5 forces at least two variables on, so
        // min(x+y+z)=2.
        // The downward q is between the two and therefore needs an integer
        // case split; a root-LP Farkas proof cannot exist.
        let mut problem = MilpProblem::new();
        let x = problem.add_integer_col(0.0, 0.0, 1.0);
        let y = problem.add_integer_col(0.0, 0.0, 1.0);
        let z = problem.add_integer_col(0.0, 0.0, 1.0);
        problem.add_row(1.5, f64::INFINITY, [(x, 1.0), (y, 1.0), (z, 1.0)]);

        let certified =
            certify_linear_lower_bound_with_ay(&problem, &[(x, 1.0), (y, 1.0), (z, 1.0)], config())
                .expect("solver/checkers agree")
                .expect("tree proof");
        assert!(certified.lower < 2.0);
        assert!(certified.lower > 1.5);
        assert_eq!(
            certified.proof_route,
            CertifiedLinearLowerProofRoute::TreeFarkas
        );
        assert!(certified.ay_tree_leaves >= 2);
        assert_eq!(certified.ny_cert_farkas_replays, certified.ay_tree_leaves);
    }

    #[test]
    fn malformed_objective_and_unpriced_tree_are_rejected() {
        let _test_guard = CERTIFIED_TEST_LOCK.lock().unwrap();
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, 0.0, 1.0);
        assert!(certify_linear_lower_bound_with_ay(&problem, &[], config()).is_err());
        assert!(
            certify_linear_lower_bound_with_ay(&problem, &[(x, 1.0), (x, 2.0)], config()).is_err()
        );
        let mut invalid = config();
        invalid.max_tree_leaves = CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES + 1;
        assert!(certify_linear_lower_bound_with_ay(&problem, &[(x, 1.0)], invalid).is_err());

        for nonfinite in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            assert!(certify_linear_lower_bound_at_with_ay(
                &problem,
                &[(x, 1.0)],
                nonfinite,
                decision_config(),
            )
            .is_err());
        }
        let mut invalid_decision = decision_config();
        invalid_decision.proof_timeout_secs = 0.0;
        assert!(certify_linear_lower_bound_at_with_ay(
            &problem,
            &[(x, 1.0)],
            0.5,
            invalid_decision,
        )
        .is_err());
    }
}
