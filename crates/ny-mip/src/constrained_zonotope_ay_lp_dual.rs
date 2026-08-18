// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unwired exact-AY LP proposals for constrained-zonotope dual bounds.
//!
//! For a constrained zonotope
//!
//! ```text
//! x = c + G alpha + e,  alpha in [-1, 1],  C alpha <= d,  |e_i| <= r_i,
//! ```
//!
//! and a direction `q`, this module asks AY to minimize and maximize
//! `(G^T q) alpha`.  AY's independently verified optimality certificates name
//! positive multipliers on the upper sides of the rows `C alpha <= d`.  Those
//! row multipliers are useful candidates for the constrained-zonotope dual.
//!
//! # Trust boundary
//!
//! AY is used only as a candidate generator here.  A returned optimality
//! certificate must first pass AY's exact [`ay_milp::OptimalityCertificate`]
//! verifier against the exact-dyadic model which was solved.  Even then, its
//! row multipliers have no authority: they are converted to finite `f64` and
//! replayed through [`ConstrainedZonotope64::evaluate_dual`], whose outward
//! arithmetic is the only bound-admission path.  A direction is retained only
//! on a strict certified improvement over the mandatory zero-multiplier
//! baseline.
//!
//! Configuration rejection, resource exhaustion, a deadline, AY failure,
//! infeasibility, an absent or malformed certificate, multiplier conversion,
//! and candidate replay failure all retain that baseline.  This module is
//! deliberately **unwired** and cannot emit a verifier verdict.

use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use ay_milp::{
    BoundSide, FactRef, LpSession, Model, OptimalityCertificate, Outcome, Sense, SolveOpts,
};
use num_rational::BigRational;
use num_traits::ToPrimitive;

use crate::{ConstrainedZonotope64, ConstrainedZonotope64Error, ConstrainedZonotopeDualBounds};

/// Absolute row cap for this unwired proposal engine.
pub const AY_LP_DUAL_HARD_MAX_CONSTRAINTS: usize = 8_192;
/// Absolute alpha-symbol cap.
pub const AY_LP_DUAL_HARD_MAX_ALPHA_DIM: usize = 4_096;
/// Absolute dense predicate scan cap.
pub const AY_LP_DUAL_HARD_MAX_CONSTRAINT_ELEMENTS: usize = 32_000_000;
/// Absolute predicate nonzero cap in the AY model.
pub const AY_LP_DUAL_HARD_MAX_CONSTRAINT_NONZEROS: usize = 24_000_000;
/// Absolute sparse-generator scan cap.
pub const AY_LP_DUAL_HARD_MAX_GENERATOR_NONZEROS: usize = 8_000_000;
/// Absolute cap on one AY optimality certificate's multiplier count.
pub const AY_LP_DUAL_HARD_MAX_CERTIFICATE_MULTIPLIERS: usize = 32_768;
/// Absolute bit-length cap on every exact rational in an AY certificate.
pub const AY_LP_DUAL_HARD_MAX_CERTIFICATE_RATIONAL_BITS: u64 = 32_768;
/// Absolute aggregate exact-rational bit cap for one AY certificate.
pub const AY_LP_DUAL_HARD_MAX_CERTIFICATE_TOTAL_BITS: u64 = 16_777_216;
/// Largest AY retained-memory budget accepted by this proposal engine.
pub const AY_LP_DUAL_HARD_MAX_MEMORY_BYTES: usize = 2_147_483_648;

/// Stack headroom for the deadline-isolated AY worker.
const AY_LP_DUAL_SOLVE_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;
/// Poll the proposal deadline at least this often during bounded linear scans.
const AY_LP_DUAL_MAX_ITEMS_PER_POLL: usize = 16_384;
/// Bound process-wide accumulation when AY outlives an external hard timeout.
const AY_LP_DUAL_HARD_MAX_ACTIVE_WORKERS: usize = 8;

// Caller-side retained-workspace admission estimate.  These weights
// deliberately cover both copies of the AY model (the caller's `Model` plus
// `LpSession`'s clone), AY's CSC/CSR and possible dense float mirrors, its
// exact-rational tableau, per-row/per-column solver vectors, and certificate
// metadata.  The formula is:
//
// fixed
// + alpha_dim                 * PER_ALPHA
// + constraint_count         * PER_CONSTRAINT
// + constraint_elements      * PER_CONSTRAINT_ELEMENT
// + constraint_nonzeros      * PER_CONSTRAINT_NONZERO
// + max_certificate_count    * PER_CERTIFICATE_MULTIPLIER
// + ceil(max_certificate_total_bits / 8).
//
// This is a deterministic, overflow-checked admission estimate, not an
// allocator quota: arbitrary-precision pivots may transiently exceed it.
// Pinned AY de03 applies `SolveOpts::memory_budget` only to branch-and-bound's
// open set, not `LpSession`, so this caller-side check is load-bearing.
const AY_LP_DUAL_WORKSPACE_FIXED_BYTES: usize = 1 << 20;
const AY_LP_DUAL_WORKSPACE_BYTES_PER_ALPHA: usize = 1_024;
const AY_LP_DUAL_WORKSPACE_BYTES_PER_CONSTRAINT: usize = 1_024;
const AY_LP_DUAL_WORKSPACE_BYTES_PER_CONSTRAINT_ELEMENT: usize = 8;
const AY_LP_DUAL_WORKSPACE_BYTES_PER_CONSTRAINT_NONZERO: usize = 384;
const AY_LP_DUAL_WORKSPACE_BYTES_PER_CERTIFICATE_MULTIPLIER: usize = 128;

static AY_LP_DUAL_ACTIVE_WORKERS: AtomicUsize = AtomicUsize::new(0);

/// Caller-tightenable resource limits.
///
/// Values above the hard ceilings are malformed rather than permission to do
/// more work.  Any malformed or exceeded proposal limit retains the certified
/// zero-multiplier baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AyLpDualLimits {
    pub max_constraints: usize,
    pub max_alpha_dim: usize,
    pub max_constraint_elements: usize,
    pub max_constraint_nonzeros: usize,
    pub max_generator_nonzeros: usize,
    pub max_certificate_multipliers: usize,
    pub max_certificate_rational_bits: u64,
    pub max_certificate_total_bits: u64,
}

impl Default for AyLpDualLimits {
    fn default() -> Self {
        Self {
            max_constraints: AY_LP_DUAL_HARD_MAX_CONSTRAINTS,
            max_alpha_dim: AY_LP_DUAL_HARD_MAX_ALPHA_DIM,
            max_constraint_elements: AY_LP_DUAL_HARD_MAX_CONSTRAINT_ELEMENTS,
            max_constraint_nonzeros: AY_LP_DUAL_HARD_MAX_CONSTRAINT_NONZEROS,
            max_generator_nonzeros: AY_LP_DUAL_HARD_MAX_GENERATOR_NONZEROS,
            max_certificate_multipliers: AY_LP_DUAL_HARD_MAX_CERTIFICATE_MULTIPLIERS,
            max_certificate_rational_bits: AY_LP_DUAL_HARD_MAX_CERTIFICATE_RATIONAL_BITS,
            max_certificate_total_bits: AY_LP_DUAL_HARD_MAX_CERTIFICATE_TOTAL_BITS,
        }
    }
}

/// Explicit execution policy for the unwired AY proposal.
///
/// The absolute deadline covers proposal planning, model construction, both
/// solves, certificate extraction, and candidate replay.  Constructing and
/// outward-evaluating the mandatory zero baseline is authority work and is
/// always attempted first, even if this proposal deadline has already passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AyLpDualConfig {
    pub deadline: Instant,
    /// Caller-side retained-workspace admission budget.
    ///
    /// With `A` alpha symbols, `K` constraint rows, `E = A*K` dense elements,
    /// `N` nonzeros, certificate-count cap `M`, and certificate aggregate-bit
    /// cap `B`, the exact admission estimate is
    /// `1_048_576 + 1_024*A + 1_024*K + 8*E + 384*N + 128*M + ceil(B/8)`
    /// bytes.  All operations are checked. Existing domain storage and the
    /// worker's reserved stack address space are excluded. The value is also
    /// forwarded to AY for compatibility with future LP memory accounting,
    /// but this caller-side estimate is the operative guard for pinned AY
    /// de03.
    pub ay_memory_budget_bytes: usize,
    pub limits: AyLpDualLimits,
}

/// Independently replayed lower and upper AY-LP proposals.
#[derive(Clone, Debug, PartialEq)]
pub struct AyLpDualProposal {
    /// Accepted certified bounds.  Either side may remain at its zero baseline.
    pub bounds: ConstrainedZonotopeDualBounds,
    /// Accepted lower multipliers, or zeros when no strict improvement passed.
    pub lower_multipliers: Vec<f64>,
    /// Accepted upper multipliers, or zeros when no strict improvement passed.
    pub upper_multipliers: Vec<f64>,
    /// Whether an exact AY certificate produced a structurally valid candidate.
    pub lower_ay_certificate_verified: bool,
    /// Whether an exact AY certificate produced a structurally valid candidate.
    pub upper_ay_certificate_verified: bool,
    /// Whether outward replay strictly raised the lower bound.
    pub lower_improved: bool,
    /// Whether outward replay strictly lowered the upper bound.
    pub upper_improved: bool,
}

/// Failure of the mandatory zero-baseline authority path.
///
/// Every AY/proposal failure is deliberately swallowed in favor of that
/// baseline.  Only inability to create or outward-evaluate the baseline is an
/// error.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AyLpDualProposerError {
    /// The outward zero-multiplier evaluator failed closed.
    #[error(transparent)]
    Baseline(#[from] ConstrainedZonotope64Error),
    /// Storage for mandatory zero multipliers could not be reserved.
    #[error("unable to reserve zero multipliers for {direction}")]
    BaselineAllocation {
        /// Which independent result vector could not be allocated.
        direction: &'static str,
    },
}

#[derive(Clone, Copy, Debug)]
struct SearchPlan {
    alpha_dim: usize,
    constraint_count: usize,
    constraint_elements: usize,
    constraint_nonzeros: usize,
    generator_nonzeros: usize,
    estimated_workspace_bytes: usize,
}

#[derive(Debug)]
struct AyCandidates {
    lower: Option<Vec<f64>>,
    upper: Option<Vec<f64>>,
}

/// Ask exact AY for lower/upper row-multiplier candidates and independently
/// replay any candidates through NY's outward constrained-zonotope evaluator.
///
/// # Errors
///
/// Returns [`AyLpDualProposerError`] only when the mandatory zero baseline
/// cannot be allocated or certified.  Every proposal failure keeps the
/// baseline instead.
pub fn propose_ay_lp_dual_unwired(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    config: AyLpDualConfig,
) -> Result<AyLpDualProposal, AyLpDualProposerError> {
    let constraint_count = domain.constraint_count();
    let lower_zero = try_zero_multipliers(constraint_count, "lower")?;
    let upper_zero = try_zero_multipliers(constraint_count, "upper")?;

    // Authority comes first.  Invalid directions and unsupported arithmetic
    // must not be hidden behind a heuristic deadline or resource fallback.
    let baseline = domain.evaluate_dual(direction, &lower_zero)?;
    let mut proposal = baseline_proposal(baseline, lower_zero, upper_zero);

    let Some(plan) = SearchPlan::checked(domain, &config) else {
        return Ok(proposal);
    };
    if deadline_expired(config.deadline) {
        return Ok(proposal);
    }

    let Some(projected_generators) = project_generators(domain, direction, plan, config.deadline)
    else {
        return Ok(proposal);
    };
    let Some((model, objective)) =
        build_model(domain, &projected_generators, plan, config.deadline)
    else {
        return Ok(proposal);
    };
    if deadline_expired(config.deadline) {
        return Ok(proposal);
    }

    let Some(candidates) =
        solve_with_hard_deadline(model, objective, plan.constraint_count, config)
    else {
        return Ok(proposal);
    };

    proposal.lower_ay_certificate_verified = candidates.lower.is_some();
    proposal.upper_ay_certificate_verified = candidates.upper.is_some();

    if !deadline_expired(config.deadline) {
        if let Some(candidate) = candidates.lower {
            // The AY candidate has no authority.  Swallow outward-evaluator
            // failure and retain the already-certified zero baseline.
            if let Ok(candidate_bounds) = domain.evaluate_dual(direction, &candidate) {
                if candidate_bounds.lower > baseline.lower {
                    proposal.bounds.lower = candidate_bounds.lower;
                    proposal.lower_multipliers = candidate;
                    proposal.lower_improved = true;
                }
            }
        }
    }

    if !deadline_expired(config.deadline) {
        if let Some(candidate) = candidates.upper {
            if let Ok(candidate_bounds) = domain.evaluate_dual(direction, &candidate) {
                if candidate_bounds.upper < baseline.upper {
                    proposal.bounds.upper = candidate_bounds.upper;
                    proposal.upper_multipliers = candidate;
                    proposal.upper_improved = true;
                }
            }
        }
    }

    Ok(proposal)
}

impl SearchPlan {
    fn checked(domain: &ConstrainedZonotope64, config: &AyLpDualConfig) -> Option<Self> {
        let limits = config.limits;
        if limits.max_constraints > AY_LP_DUAL_HARD_MAX_CONSTRAINTS
            || limits.max_alpha_dim > AY_LP_DUAL_HARD_MAX_ALPHA_DIM
            || limits.max_constraint_elements > AY_LP_DUAL_HARD_MAX_CONSTRAINT_ELEMENTS
            || limits.max_constraint_nonzeros > AY_LP_DUAL_HARD_MAX_CONSTRAINT_NONZEROS
            || limits.max_generator_nonzeros > AY_LP_DUAL_HARD_MAX_GENERATOR_NONZEROS
            || limits.max_certificate_multipliers > AY_LP_DUAL_HARD_MAX_CERTIFICATE_MULTIPLIERS
            || limits.max_certificate_rational_bits > AY_LP_DUAL_HARD_MAX_CERTIFICATE_RATIONAL_BITS
            || limits.max_certificate_total_bits > AY_LP_DUAL_HARD_MAX_CERTIFICATE_TOTAL_BITS
            || config.ay_memory_budget_bytes == 0
            || config.ay_memory_budget_bytes > AY_LP_DUAL_HARD_MAX_MEMORY_BYTES
        {
            return None;
        }

        let alpha_dim = domain.alpha_dim();
        let constraint_count = domain.constraint_count();
        let constraint_elements = constraint_count.checked_mul(alpha_dim)?;
        if constraint_count > limits.max_constraints
            || alpha_dim > limits.max_alpha_dim
            || constraint_elements > limits.max_constraint_elements
        {
            return None;
        }

        let mut generator_nonzeros = 0_usize;
        for generator in domain.generators() {
            generator_nonzeros = generator_nonzeros.checked_add(generator.nnz())?;
            if generator_nonzeros > limits.max_generator_nonzeros {
                return None;
            }
        }

        let mut constraint_nonzeros = 0_usize;
        for (index, coefficient) in domain.constraints().iter().enumerate() {
            if index.is_multiple_of(AY_LP_DUAL_MAX_ITEMS_PER_POLL)
                && deadline_expired(config.deadline)
            {
                return None;
            }
            if *coefficient != 0.0 {
                constraint_nonzeros = constraint_nonzeros.checked_add(1)?;
                if constraint_nonzeros > limits.max_constraint_nonzeros {
                    return None;
                }
            }
        }

        let mut plan = Self {
            alpha_dim,
            constraint_count,
            constraint_elements,
            constraint_nonzeros,
            generator_nonzeros,
            estimated_workspace_bytes: 0,
        };
        plan.estimated_workspace_bytes = plan.estimate_workspace_bytes(limits)?;
        if plan.estimated_workspace_bytes > config.ay_memory_budget_bytes {
            return None;
        }
        Some(plan)
    }

    fn estimate_workspace_bytes(self, limits: AyLpDualLimits) -> Option<usize> {
        let mut bytes = AY_LP_DUAL_WORKSPACE_FIXED_BYTES;
        bytes = checked_weighted_add(bytes, self.alpha_dim, AY_LP_DUAL_WORKSPACE_BYTES_PER_ALPHA)?;
        bytes = checked_weighted_add(
            bytes,
            self.constraint_count,
            AY_LP_DUAL_WORKSPACE_BYTES_PER_CONSTRAINT,
        )?;
        bytes = checked_weighted_add(
            bytes,
            self.constraint_elements,
            AY_LP_DUAL_WORKSPACE_BYTES_PER_CONSTRAINT_ELEMENT,
        )?;
        bytes = checked_weighted_add(
            bytes,
            self.constraint_nonzeros,
            AY_LP_DUAL_WORKSPACE_BYTES_PER_CONSTRAINT_NONZERO,
        )?;
        bytes = checked_weighted_add(
            bytes,
            limits.max_certificate_multipliers,
            AY_LP_DUAL_WORKSPACE_BYTES_PER_CERTIFICATE_MULTIPLIER,
        )?;
        let certificate_bits = limits.max_certificate_total_bits.checked_add(7)?;
        bytes.checked_add(usize::try_from(certificate_bits / 8).ok()?)
    }
}

fn checked_weighted_add(base: usize, items: usize, bytes_per_item: usize) -> Option<usize> {
    base.checked_add(items.checked_mul(bytes_per_item)?)
}

fn try_zero_multipliers(
    constraint_count: usize,
    direction: &'static str,
) -> Result<Vec<f64>, AyLpDualProposerError> {
    let mut multipliers = Vec::new();
    multipliers
        .try_reserve_exact(constraint_count)
        .map_err(|_| AyLpDualProposerError::BaselineAllocation { direction })?;
    multipliers.resize(constraint_count, 0.0);
    Ok(multipliers)
}

fn baseline_proposal(
    bounds: ConstrainedZonotopeDualBounds,
    lower_multipliers: Vec<f64>,
    upper_multipliers: Vec<f64>,
) -> AyLpDualProposal {
    AyLpDualProposal {
        bounds,
        lower_multipliers,
        upper_multipliers,
        lower_ay_certificate_verified: false,
        upper_ay_certificate_verified: false,
        lower_improved: false,
        upper_improved: false,
    }
}

fn project_generators(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    plan: SearchPlan,
    deadline: Instant,
) -> Option<Vec<f64>> {
    debug_assert_eq!(direction.len(), domain.value_dim());
    let mut projected = Vec::new();
    projected.try_reserve_exact(plan.alpha_dim).ok()?;

    let mut visited = 0_usize;
    for generator in domain.generators() {
        let mut sum = 0.0_f64;
        for (coordinate, coefficient) in generator.entries() {
            if visited.is_multiple_of(AY_LP_DUAL_MAX_ITEMS_PER_POLL) && deadline_expired(deadline) {
                return None;
            }
            visited = visited.checked_add(1)?;
            let term = direction[coordinate] * coefficient;
            if !term.is_finite() {
                return None;
            }
            sum += term;
            if !sum.is_finite() {
                return None;
            }
        }
        projected.push(sum);
    }
    debug_assert_eq!(visited, plan.generator_nonzeros);
    Some(projected)
}

fn build_model(
    domain: &ConstrainedZonotope64,
    projected_generators: &[f64],
    plan: SearchPlan,
    deadline: Instant,
) -> Option<(Model, Vec<(ay_milp::Col, f64)>)> {
    let mut model = Model::new();
    let mut columns = Vec::new();
    columns.try_reserve_exact(plan.alpha_dim).ok()?;
    for alpha in 0..plan.alpha_dim {
        if alpha.is_multiple_of(AY_LP_DUAL_MAX_ITEMS_PER_POLL) && deadline_expired(deadline) {
            return None;
        }
        columns.push(model.add_col(-1.0, 1.0));
    }

    let mut objective = Vec::new();
    objective.try_reserve_exact(plan.alpha_dim).ok()?;
    for (&column, &coefficient) in columns.iter().zip(projected_generators) {
        if coefficient != 0.0 {
            objective.push((column, coefficient));
        }
    }

    let constraints = domain.constraints();
    let mut visited = 0_usize;
    let mut nonzeros = 0_usize;
    for row in 0..plan.constraint_count {
        let mut terms = Vec::new();
        terms.try_reserve_exact(plan.alpha_dim).ok()?;
        for alpha in 0..plan.alpha_dim {
            if visited.is_multiple_of(AY_LP_DUAL_MAX_ITEMS_PER_POLL) && deadline_expired(deadline) {
                return None;
            }
            visited = visited.checked_add(1)?;
            let coefficient = constraints[[row, alpha]];
            if coefficient != 0.0 {
                nonzeros = nonzeros.checked_add(1)?;
                if nonzeros > plan.constraint_nonzeros {
                    return None;
                }
                terms.push((columns[alpha], coefficient));
            }
        }
        model.add_row(f64::NEG_INFINITY, domain.rhs()[row], &terms);
    }
    debug_assert_eq!(visited, plan.constraint_elements);
    debug_assert_eq!(nonzeros, plan.constraint_nonzeros);
    Some((model, objective))
}

fn solve_with_hard_deadline(
    model: Model,
    objective: Vec<(ay_milp::Col, f64)>,
    constraint_count: usize,
    config: AyLpDualConfig,
) -> Option<AyCandidates> {
    if deadline_expired(config.deadline) {
        return None;
    }
    let worker_permit = try_acquire_worker(
        &AY_LP_DUAL_ACTIVE_WORKERS,
        AY_LP_DUAL_HARD_MAX_ACTIVE_WORKERS,
    )?;
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("ny-mip-cz-ay-lp-dual".to_owned())
        .stack_size(AY_LP_DUAL_SOLVE_THREAD_STACK_BYTES)
        .spawn(move || {
            // The permit deliberately lives until this worker exits.  If the
            // receiver abandons it at the hard deadline, subsequent callers
            // can create at most the remaining process-wide worker slots.
            let _worker_permit = worker_permit;
            let candidates = solve_owned(
                model,
                &objective,
                constraint_count,
                config.deadline,
                config.ay_memory_budget_bytes,
                config.limits.max_certificate_multipliers,
                config.limits.max_certificate_rational_bits,
                config.limits.max_certificate_total_bits,
            );
            let _ = sender.send(candidates);
        })
        .ok()?;

    let remaining = config
        .deadline
        .checked_duration_since(Instant::now())
        .unwrap_or(Duration::ZERO);
    receiver.recv_timeout(remaining).ok()
}

struct ActiveWorkerPermit {
    counter: &'static AtomicUsize,
}

impl Drop for ActiveWorkerPermit {
    fn drop(&mut self) {
        let previous = self.counter.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
    }
}

fn try_acquire_worker(
    counter: &'static AtomicUsize,
    hard_max_active: usize,
) -> Option<ActiveWorkerPermit> {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
            (active < hard_max_active).then_some(active + 1)
        })
        .ok()?;
    Some(ActiveWorkerPermit { counter })
}

fn solve_owned(
    mut model: Model,
    objective: &[(ay_milp::Col, f64)],
    constraint_count: usize,
    deadline: Instant,
    ay_memory_budget_bytes: usize,
    max_certificate_multipliers: usize,
    max_certificate_rational_bits: u64,
    max_certificate_total_bits: u64,
) -> AyCandidates {
    let lower = solve_direction(
        &mut model,
        objective,
        Sense::Minimize,
        constraint_count,
        deadline,
        ay_memory_budget_bytes,
        max_certificate_multipliers,
        max_certificate_rational_bits,
        max_certificate_total_bits,
    );
    let upper = if deadline_expired(deadline) {
        None
    } else {
        solve_direction(
            &mut model,
            objective,
            Sense::Maximize,
            constraint_count,
            deadline,
            ay_memory_budget_bytes,
            max_certificate_multipliers,
            max_certificate_rational_bits,
            max_certificate_total_bits,
        )
    };
    AyCandidates { lower, upper }
}

fn solve_direction(
    model: &mut Model,
    objective: &[(ay_milp::Col, f64)],
    sense: Sense,
    constraint_count: usize,
    deadline: Instant,
    ay_memory_budget_bytes: usize,
    max_certificate_multipliers: usize,
    max_certificate_rational_bits: u64,
    max_certificate_total_bits: u64,
) -> Option<Vec<f64>> {
    if deadline_expired(deadline) {
        return None;
    }
    model.set_objective(objective, sense);
    let opts = SolveOpts::new()
        .with_deadline(deadline)
        .with_threads(1)
        .with_determinism(true)
        .with_require_certificates(true)
        .with_memory_budget(Some(ay_memory_budget_bytes))
        .with_tree_cert_leaves(0);
    let mut session = LpSession::new(model, &opts).ok()?;
    let outcome = session.optimize_model_objective().ok()?;
    let Outcome::Optimal {
        cert: Some(certificate),
        ..
    } = outcome
    else {
        return None;
    };
    if deadline_expired(deadline)
        || certificate.multipliers.len() > max_certificate_multipliers
        || certificate.sense != sense
        || certificate.objective.len() != objective.len()
        || !certificate_rationals_within_limits(
            &certificate,
            max_certificate_rational_bits,
            max_certificate_total_bits,
            deadline,
        )
        || !certificate_matches_objective(&certificate, objective, sense)
        || certificate.verify(model).is_err()
    {
        return None;
    }
    extract_row_upper_multipliers(&certificate, constraint_count, deadline)
}

fn certificate_rationals_within_limits(
    certificate: &OptimalityCertificate,
    max_rational_bits: u64,
    max_total_bits: u64,
    deadline: Instant,
) -> bool {
    let mut total_bits = 0_u64;
    if !charge_certificate_rational(
        &certificate.bound,
        max_rational_bits,
        max_total_bits,
        &mut total_bits,
    ) {
        return false;
    }
    for (index, (_, coefficient)) in certificate.objective.iter().enumerate() {
        if index.is_multiple_of(AY_LP_DUAL_MAX_ITEMS_PER_POLL) && deadline_expired(deadline) {
            return false;
        }
        if !charge_certificate_rational(
            coefficient,
            max_rational_bits,
            max_total_bits,
            &mut total_bits,
        ) {
            return false;
        }
    }
    for (index, multiplier) in certificate.multipliers.iter().enumerate() {
        if index.is_multiple_of(AY_LP_DUAL_MAX_ITEMS_PER_POLL) && deadline_expired(deadline) {
            return false;
        }
        if !charge_certificate_rational(
            &multiplier.coeff,
            max_rational_bits,
            max_total_bits,
            &mut total_bits,
        ) {
            return false;
        }
    }
    true
}

fn charge_certificate_rational(
    value: &BigRational,
    max_rational_bits: u64,
    max_total_bits: u64,
    total_bits: &mut u64,
) -> bool {
    let bits = value.numer().bits().max(value.denom().bits());
    if bits > max_rational_bits {
        return false;
    }
    let Some(next_total) = total_bits.checked_add(bits) else {
        return false;
    };
    if next_total > max_total_bits {
        return false;
    }
    *total_bits = next_total;
    true
}

fn certificate_matches_objective(
    certificate: &OptimalityCertificate,
    objective: &[(ay_milp::Col, f64)],
    sense: Sense,
) -> bool {
    certificate.sense == sense
        && certificate.objective.len() == objective.len()
        && certificate.objective.iter().zip(objective).all(
            |(&(cert_column, ref cert_coefficient), &(column, coefficient))| {
                usize::try_from(cert_column).ok() == Some(column.index())
                    && BigRational::from_float(coefficient)
                        .is_some_and(|expected| expected == *cert_coefficient)
            },
        )
}

fn extract_row_upper_multipliers(
    certificate: &OptimalityCertificate,
    constraint_count: usize,
    deadline: Instant,
) -> Option<Vec<f64>> {
    let mut candidate = Vec::new();
    candidate.try_reserve_exact(constraint_count).ok()?;
    candidate.resize(constraint_count, 0.0);

    for (index, multiplier) in certificate.multipliers.iter().enumerate() {
        if index.is_multiple_of(AY_LP_DUAL_MAX_ITEMS_PER_POLL) && deadline_expired(deadline) {
            return None;
        }
        match multiplier.fact {
            FactRef::RowBound {
                row,
                side: BoundSide::Upper,
            } => {
                let slot = candidate.get_mut(row.index())?;
                let coefficient = multiplier.coeff.to_f64()?;
                if !coefficient.is_finite() || coefficient <= 0.0 {
                    return None;
                }
                *slot += coefficient;
                if !slot.is_finite() {
                    return None;
                }
            }
            FactRef::ColBound { .. } => {}
            // The constructed model has no finite row-lower facts.  Reject
            // rather than silently reinterpret any future/foreign row kind.
            FactRef::RowBound { .. } => return None,
            _ => return None,
        }
    }
    Some(candidate)
}

fn deadline_expired(deadline: Instant) -> bool {
    Instant::now() >= deadline
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use ay_milp::{Multiplier, Row};
    use ndarray::{array, Array2};
    use num_rational::BigRational;

    use super::*;

    fn config() -> AyLpDualConfig {
        AyLpDualConfig {
            deadline: Instant::now() + Duration::from_secs(10),
            ay_memory_budget_bytes: 64 * 1024 * 1024,
            limits: AyLpDualLimits {
                max_constraints: 8,
                max_alpha_dim: 8,
                max_constraint_elements: 64,
                max_constraint_nonzeros: 64,
                max_generator_nonzeros: 64,
                max_certificate_multipliers: 64,
                max_certificate_rational_bits: 8_192,
                max_certificate_total_bits: 65_536,
            },
        }
    }

    fn one_alpha_domain(
        constraint: Option<(f64, f64)>,
        center: f64,
        remainder: f64,
    ) -> ConstrainedZonotope64 {
        let (constraints, rhs) = match constraint {
            Some((coefficient, bound)) => (array![[coefficient]], vec![bound]),
            None => (Array2::zeros((0, 1)), Vec::new()),
        };
        ConstrainedZonotope64::try_new(
            vec![center],
            vec![vec![(0, 1.0)]],
            constraints,
            rhs,
            vec![remainder],
        )
        .expect("valid one-alpha constrained zonotope")
    }

    #[test]
    fn ay_row_multiplier_strictly_improves_lower_bound_after_outward_replay() {
        // -alpha <= 0, so alpha >= 0.
        let domain = one_alpha_domain(Some((-1.0, 0.0)), 0.0, 0.0);
        let proposal = propose_ay_lp_dual_unwired(&domain, &[1.0], config()).expect("AY proposal");

        assert!(proposal.lower_ay_certificate_verified);
        assert!(proposal.lower_improved);
        assert_eq!(proposal.bounds.lower.to_bits(), 1_u64 << 63 | 1);
        assert!(proposal.lower_multipliers[0] > 0.0);
        assert!(!proposal.upper_improved);
        assert_eq!(proposal.bounds.upper, 1.0);
    }

    #[test]
    fn ay_row_multiplier_strictly_improves_upper_bound_after_outward_replay() {
        // alpha <= 0.
        let domain = one_alpha_domain(Some((1.0, 0.0)), 2.0, 0.25);
        let proposal = propose_ay_lp_dual_unwired(&domain, &[1.0], config()).expect("AY proposal");

        assert!(proposal.upper_ay_certificate_verified);
        assert!(proposal.upper_improved);
        assert_eq!(proposal.bounds.upper, 2.25_f64.next_up().next_up());
        assert!(proposal.upper_multipliers[0] > 0.0);
        assert!(!proposal.lower_improved);
        assert_eq!(proposal.bounds.lower, 0.75_f64.next_down().next_down());
    }

    #[test]
    fn no_constraints_retains_the_exact_zero_baseline() {
        let domain = one_alpha_domain(None, -3.0, 0.5);
        let zero = domain.evaluate_dual(&[2.0], &[]).expect("zero baseline");
        let proposal = propose_ay_lp_dual_unwired(&domain, &[2.0], config()).expect("AY proposal");

        assert_eq!(proposal.bounds, zero);
        assert!(!proposal.lower_improved);
        assert!(!proposal.upper_improved);
        assert!(proposal.lower_multipliers.is_empty());
        assert!(proposal.upper_multipliers.is_empty());
    }

    #[test]
    fn coupled_constraint_multiplier_maps_back_to_its_original_row() {
        let domain = ConstrainedZonotope64::try_new(
            vec![0.0, 0.0],
            vec![vec![(0, 1.0)], vec![(1, 1.0)]],
            array![[1.0, 1.0]],
            vec![0.0],
            vec![0.0, 0.0],
        )
        .expect("valid coupled constrained zonotope");
        let baseline = domain
            .evaluate_dual(&[1.0, 1.0], &[0.0])
            .expect("zero baseline");
        let proposal =
            propose_ay_lp_dual_unwired(&domain, &[1.0, 1.0], config()).expect("AY proposal");

        assert!(proposal.upper_ay_certificate_verified);
        assert!(proposal.upper_improved);
        assert!(proposal.bounds.upper < baseline.upper);
        assert!(proposal.bounds.upper.abs() <= 4.0 * f64::EPSILON);
        assert!(proposal.upper_multipliers[0] > 0.0);
    }

    #[test]
    fn expired_deadline_and_tight_limits_retain_baseline() {
        let domain = one_alpha_domain(Some((1.0, 0.0)), 0.0, 0.0);
        let baseline = domain.evaluate_dual(&[1.0], &[0.0]).expect("zero baseline");

        let mut expired = config();
        expired.deadline = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("monotonic clock supports lookback");
        let expired_proposal =
            propose_ay_lp_dual_unwired(&domain, &[1.0], expired).expect("expired fallback");
        assert_eq!(expired_proposal.bounds, baseline);
        assert!(!expired_proposal.upper_ay_certificate_verified);

        let mut tight = config();
        tight.limits.max_constraints = 0;
        let tight_proposal =
            propose_ay_lp_dual_unwired(&domain, &[1.0], tight).expect("limit fallback");
        assert_eq!(tight_proposal.bounds, baseline);
        assert!(!tight_proposal.upper_ay_certificate_verified);

        let mut sparse_limit = config();
        sparse_limit.limits.max_constraint_nonzeros = 0;
        let sparse_proposal = propose_ay_lp_dual_unwired(&domain, &[1.0], sparse_limit)
            .expect("constraint-nonzero fallback");
        assert_eq!(sparse_proposal.bounds, baseline);
        assert!(!sparse_proposal.upper_ay_certificate_verified);
    }

    #[test]
    fn zero_memory_budget_retains_baseline() {
        let domain = one_alpha_domain(Some((1.0, 0.0)), 0.0, 0.0);
        let baseline = domain.evaluate_dual(&[1.0], &[0.0]).expect("zero baseline");
        let mut zero_budget = config();
        zero_budget.ay_memory_budget_bytes = 0;

        let proposal =
            propose_ay_lp_dual_unwired(&domain, &[1.0], zero_budget).expect("zero-budget fallback");
        assert_eq!(proposal.bounds, baseline);
        assert!(!proposal.lower_ay_certificate_verified);
        assert!(!proposal.upper_ay_certificate_verified);
    }

    #[test]
    fn caller_memory_estimate_is_exact_and_load_bearing() {
        let domain = one_alpha_domain(Some((1.0, 0.0)), 0.0, 0.0);
        let baseline = domain.evaluate_dual(&[1.0], &[0.0]).expect("zero baseline");
        let admitted = config();
        let plan = SearchPlan::checked(&domain, &admitted).expect("admitted search plan");
        assert_eq!(plan.alpha_dim, 1);
        assert_eq!(plan.constraint_count, 1);
        assert_eq!(plan.constraint_elements, 1);
        assert_eq!(plan.constraint_nonzeros, 1);
        assert_eq!(plan.estimated_workspace_bytes, 1_067_400);

        let mut one_byte_short = admitted;
        one_byte_short.ay_memory_budget_bytes = plan.estimated_workspace_bytes - 1;
        let declined = propose_ay_lp_dual_unwired(&domain, &[1.0], one_byte_short)
            .expect("under-budget fallback");
        assert_eq!(declined.bounds, baseline);
        assert!(!declined.upper_ay_certificate_verified);

        let mut exact_budget = admitted;
        exact_budget.ay_memory_budget_bytes = plan.estimated_workspace_bytes;
        let accepted = propose_ay_lp_dual_unwired(&domain, &[1.0], exact_budget)
            .expect("exact-budget proposal");
        assert!(accepted.upper_ay_certificate_verified);
        assert!(accepted.upper_improved);
    }

    #[test]
    fn malformed_super_hard_limit_and_infeasible_model_retain_baseline() {
        let domain = one_alpha_domain(Some((1.0, -2.0)), 0.0, 0.0);
        let baseline = domain.evaluate_dual(&[1.0], &[0.0]).expect("zero baseline");

        let infeasible =
            propose_ay_lp_dual_unwired(&domain, &[1.0], config()).expect("infeasible fallback");
        assert_eq!(infeasible.bounds, baseline);
        assert!(!infeasible.lower_ay_certificate_verified);
        assert!(!infeasible.upper_ay_certificate_verified);

        let mut malformed = config();
        malformed.limits.max_constraints = AY_LP_DUAL_HARD_MAX_CONSTRAINTS + 1;
        let malformed_proposal = propose_ay_lp_dual_unwired(&domain, &[1.0], malformed)
            .expect("malformed config fallback");
        assert_eq!(malformed_proposal.bounds, baseline);

        let mut malformed_bits = config();
        malformed_bits.limits.max_certificate_rational_bits =
            AY_LP_DUAL_HARD_MAX_CERTIFICATE_RATIONAL_BITS + 1;
        let malformed_bits_proposal = propose_ay_lp_dual_unwired(&domain, &[1.0], malformed_bits)
            .expect("malformed rational-bit fallback");
        assert_eq!(malformed_bits_proposal.bounds, baseline);

        let mut malformed_total = config();
        malformed_total.limits.max_certificate_total_bits =
            AY_LP_DUAL_HARD_MAX_CERTIFICATE_TOTAL_BITS + 1;
        let malformed_total_proposal = propose_ay_lp_dual_unwired(&domain, &[1.0], malformed_total)
            .expect("malformed total-bit fallback");
        assert_eq!(malformed_total_proposal.bounds, baseline);
    }

    #[test]
    fn invalid_direction_is_a_baseline_error_not_a_hidden_fallback() {
        let domain = one_alpha_domain(Some((1.0, 0.0)), 0.0, 0.0);
        assert!(matches!(
            propose_ay_lp_dual_unwired(&domain, &[], config()),
            Err(AyLpDualProposerError::Baseline(_))
        ));
        assert!(matches!(
            propose_ay_lp_dual_unwired(&domain, &[f64::NAN], config()),
            Err(AyLpDualProposerError::Baseline(_))
        ));
    }

    fn exact(value: i64) -> BigRational {
        BigRational::from_integer(value.into())
    }

    fn max_certificate(row: Row, row_coefficients: Vec<BigRational>) -> OptimalityCertificate {
        OptimalityCertificate {
            sense: Sense::Maximize,
            objective: vec![(0, exact(1))],
            bound: exact(0),
            multipliers: row_coefficients
                .into_iter()
                .map(|coeff| Multiplier {
                    fact: FactRef::RowBound {
                        row,
                        side: BoundSide::Upper,
                    },
                    coeff,
                })
                .collect(),
        }
    }

    #[test]
    fn exact_certificate_verification_precedes_duplicate_row_extraction() {
        let mut model = Model::new();
        let alpha = model.add_col(-1.0, 1.0);
        let row = model.add_row(f64::NEG_INFINITY, 0.0, &[(alpha, 1.0)]);
        model.set_objective(&[(alpha, 1.0)], Sense::Maximize);
        let certificate = max_certificate(
            row,
            vec![
                BigRational::new(1.into(), 2.into()),
                BigRational::new(1.into(), 2.into()),
            ],
        );
        certificate.verify(&model).expect("exact certificate");

        let extracted =
            extract_row_upper_multipliers(&certificate, 1, Instant::now() + Duration::from_secs(1))
                .expect("row multiplier");
        assert_eq!(extracted, vec![1.0]);
    }

    #[test]
    fn certificate_objective_and_sense_are_bound_to_the_requested_solve() {
        let mut model = Model::new();
        let alpha = model.add_col(-1.0, 1.0);
        let other = model.add_col(-1.0, 1.0);
        let alpha_row = model.add_row(f64::NEG_INFINITY, 0.0, &[(alpha, 1.0)]);
        let other_row = model.add_row(f64::NEG_INFINITY, 0.0, &[(other, 1.0)]);
        let expected = vec![(alpha, 1.0)];

        let matching = max_certificate(alpha_row, vec![exact(1)]);
        assert!(certificate_matches_objective(
            &matching,
            &expected,
            Sense::Maximize
        ));
        assert!(!certificate_matches_objective(
            &matching,
            &expected,
            Sense::Minimize
        ));

        let foreign = OptimalityCertificate {
            sense: Sense::Maximize,
            objective: vec![(1, exact(1))],
            bound: exact(0),
            multipliers: vec![Multiplier {
                fact: FactRef::RowBound {
                    row: other_row,
                    side: BoundSide::Upper,
                },
                coeff: exact(1),
            }],
        };
        foreign
            .verify(&model)
            .expect("foreign-objective certificate is internally valid");
        assert!(!certificate_matches_objective(
            &foreign,
            &expected,
            Sense::Maximize
        ));
    }

    #[test]
    fn extraction_rejects_foreign_or_lower_row_facts() {
        let mut model = Model::new();
        let alpha = model.add_col(-1.0, 1.0);
        let first = model.add_row(f64::NEG_INFINITY, 0.0, &[(alpha, 1.0)]);
        let foreign = model.add_row(f64::NEG_INFINITY, 0.0, &[(alpha, 1.0)]);
        let foreign_certificate = max_certificate(foreign, vec![exact(1)]);
        foreign_certificate
            .verify(&model)
            .expect("valid but foreign certificate");
        assert!(extract_row_upper_multipliers(
            &foreign_certificate,
            1,
            Instant::now() + Duration::from_secs(1)
        )
        .is_none());

        let lower_certificate = OptimalityCertificate {
            sense: Sense::Minimize,
            objective: vec![(0, exact(1))],
            bound: exact(0),
            multipliers: vec![Multiplier {
                fact: FactRef::RowBound {
                    row: first,
                    side: BoundSide::Lower,
                },
                coeff: exact(1),
            }],
        };
        assert!(extract_row_upper_multipliers(
            &lower_certificate,
            1,
            Instant::now() + Duration::from_secs(1)
        )
        .is_none());
    }

    #[test]
    fn certificate_rational_caps_precede_exact_verification() {
        let mut model = Model::new();
        let alpha = model.add_col(-1.0, 1.0);
        let row = model.add_row(f64::NEG_INFINITY, 0.0, &[(alpha, 1.0)]);
        let matching = max_certificate(row, vec![exact(1)]);
        let deadline = Instant::now() + Duration::from_secs(1);

        assert!(certificate_rationals_within_limits(
            &matching, 1, 3, deadline
        ));
        assert!(!certificate_rationals_within_limits(
            &matching, 1, 2, deadline
        ));

        let oversized_multiplier = max_certificate(row, vec![exact(2)]);
        assert!(!certificate_rationals_within_limits(
            &oversized_multiplier,
            1,
            64,
            deadline
        ));
    }

    #[test]
    fn active_worker_permits_bound_detached_accumulation() {
        static TEST_ACTIVE_WORKERS: AtomicUsize = AtomicUsize::new(0);

        let first = try_acquire_worker(&TEST_ACTIVE_WORKERS, 2).expect("first permit");
        let second = try_acquire_worker(&TEST_ACTIVE_WORKERS, 2).expect("second permit");
        assert!(try_acquire_worker(&TEST_ACTIVE_WORKERS, 2).is_none());
        assert_eq!(TEST_ACTIVE_WORKERS.load(Ordering::Acquire), 2);

        drop(first);
        let replacement =
            try_acquire_worker(&TEST_ACTIVE_WORKERS, 2).expect("released permit is reusable");
        drop(second);
        drop(replacement);
        assert_eq!(TEST_ACTIVE_WORKERS.load(Ordering::Acquire), 0);
    }

    #[test]
    fn accepted_proposals_can_never_weaken_the_zero_baseline() {
        for coefficient in [-1.0, 1.0] {
            let domain = one_alpha_domain(Some((coefficient, 0.0)), 7.0, 0.125);
            let baseline = domain
                .evaluate_dual(&[-3.0], &[0.0])
                .expect("zero baseline");
            let proposal =
                propose_ay_lp_dual_unwired(&domain, &[-3.0], config()).expect("AY proposal");
            assert!(proposal.bounds.lower >= baseline.lower);
            assert!(proposal.bounds.upper <= baseline.upper);
        }
    }
}
