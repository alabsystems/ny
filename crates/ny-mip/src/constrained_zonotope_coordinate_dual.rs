// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unwired coordinate-ascent proposals for constrained-zonotope dual bounds.
//!
//! For `C alpha <= d` and `g = G^T q`, the lower-certificate candidate problem
//! (up to terms independent of `lambda`) is
//!
//! ```text
//! maximize  -lambda d - ||g + C^T lambda||_1,  lambda >= 0.
//! ```
//!
//! The upper problem is the same maximization after replacing `g` by `-g`:
//! minimizing `lambda d + ||g - C^T lambda||_1` is equivalent to maximizing
//! `-lambda d - ||-g + C^T lambda||_1`.
//!
//! With every multiplier except `lambda_k` fixed, remove its old contribution
//! and write the coordinate objective as
//!
//! ```text
//! phi(t) = -d_k t - sum_j |h_j + C_kj t|,  t >= 0.
//! ```
//!
//! This is concave and piecewise linear.  Its nonnegative breakpoints are
//! `-h_j / C_kj`, plus the boundary `0`; the right slope decreases by
//! `2 |C_kj|` at every positive breakpoint.  The proposer sorts those points
//! and selects the first slope change from positive to nonpositive.
//!
//! # Trust boundary
//!
//! All coordinate arithmetic is untrusted candidate search.  The public entry
//! point first evaluates `lambda = 0` and independently replays each finite
//! proposal through [`ConstrainedZonotope64::evaluate_dual`].  A direction is
//! retained only when that outward evaluator reports a strict certified
//! improvement over the zero baseline.  Malformed configuration, malformed
//! warm starts, resource-limit rejection, non-finite proposal arithmetic, and
//! a numerically unbounded coordinate all retain the zero baseline.
//!
//! This module is deliberately **unwired**.  It cannot emit a verifier verdict
//! and is not called by any scored path.  A future CUDA implementation may
//! batch the untrusted breakpoint search, but CPU outward replay remains a
//! mandatory authority boundary.

use std::cmp::Ordering;

use crate::constrained_zonotope_call_budget::{
    ConstrainedZonotopeCallBudget, ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallGate,
    ConstrainedZonotopeCallOutcome, ConstrainedZonotopeCallTracker,
    ConstrainedZonotopePeakLiveBytes, InertConstrainedZonotopeCallGate,
};
use crate::constrained_zonotope_dual::{
    evaluate_constrained_zonotope64_dual_with_call_gate, ConstrainedZonotopeDualBudgetError,
    DUAL_SHAPE_ERROR_LIVE_BYTES,
};
use crate::{ConstrainedZonotope64, ConstrainedZonotope64Error, ConstrainedZonotopeDualBounds};

/// Maximum number of deterministic coordinate sweeps accepted by this M2
/// prototype.
pub const COORDINATE_DUAL_MAX_SWEEPS: u8 = 4;

/// Hard ceilings which caller-supplied limits may only tighten.
pub const COORDINATE_DUAL_HARD_MAX_CONSTRAINTS: usize = 8_192;
pub const COORDINATE_DUAL_HARD_MAX_ALPHA_DIM: usize = 4_096;
pub const COORDINATE_DUAL_HARD_MAX_BREAKPOINTS: usize = 4_097;
pub const COORDINATE_DUAL_HARD_MAX_WORK: usize = 268_435_456;

/// Caller-tightenable resource limits for coordinate proposal search.
///
/// Values above the hard ceilings are malformed rather than permission to do
/// more work.  A domain which exceeds a valid limit receives the certified
/// zero-multiplier baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoordinateDualLimits {
    /// Maximum rows in `C alpha <= d`.
    pub max_constraints: usize,
    /// Maximum number of alpha symbols.
    pub max_alpha_dim: usize,
    /// Maximum points in one sorted list, including the boundary `0`.
    pub max_breakpoints: usize,
    /// Conservative cap on scalar proposal/search work across both directions.
    pub max_work: usize,
}

impl Default for CoordinateDualLimits {
    fn default() -> Self {
        Self {
            max_constraints: COORDINATE_DUAL_HARD_MAX_CONSTRAINTS,
            max_alpha_dim: COORDINATE_DUAL_HARD_MAX_ALPHA_DIM,
            max_breakpoints: COORDINATE_DUAL_HARD_MAX_BREAKPOINTS,
            max_work: COORDINATE_DUAL_HARD_MAX_WORK,
        }
    }
}

/// Deterministic coordinate-ascent configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoordinateDualConfig {
    /// Number of complete row-order sweeps; valid values are `1..=4`.
    pub sweeps: u8,
    /// Caller-tightenable, hard-bounded resource limits.
    pub limits: CoordinateDualLimits,
}

impl Default for CoordinateDualConfig {
    fn default() -> Self {
        Self {
            sweeps: 2,
            limits: CoordinateDualLimits::default(),
        }
    }
}

/// Independently replayed lower and upper coordinate-dual proposals.
#[derive(Clone, Debug, PartialEq)]
pub struct CoordinateDualProposal {
    /// The accepted lower and upper certificates.  These may come from
    /// different multiplier vectors.
    pub bounds: ConstrainedZonotopeDualBounds,
    /// Accepted lower multipliers, or all zero when no strict improvement was
    /// certified.
    pub lower_multipliers: Vec<f64>,
    /// Accepted upper multipliers, or all zero when no strict improvement was
    /// certified.
    pub upper_multipliers: Vec<f64>,
    /// Whether outward replay strictly raised the lower certificate.
    pub lower_improved: bool,
    /// Whether outward replay strictly lowered the upper certificate.
    pub upper_improved: bool,
}

/// Failure of the mandatory zero-baseline authority path.
///
/// Heuristic proposal failures are deliberately not errors: they return the
/// zero baseline.  This error is reserved for cases where even that baseline
/// cannot be allocated or outward-evaluated.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CoordinateDualProposerError {
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

/// Mandatory baseline or call-firewall refusal from budgeted coordinate search.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CoordinateDualBudgetError {
    /// The mandatory zero baseline could not be allocated or certified.
    #[error(transparent)]
    Proposal(#[from] CoordinateDualProposerError),

    /// The caller's deadline or aggregate peak-memory ceiling refused work.
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
}

/// Propose lower and upper multipliers, then independently replay them through
/// the outward evaluator.
///
/// `lower_warm_start` and `upper_warm_start` are optional, independent vectors
/// of finite nonnegative multipliers.  A malformed warm start rejects all
/// proposal search and returns the zero baseline; it never affects authority.
///
/// # Errors
///
/// Returns [`CoordinateDualProposerError`] only when the mandatory zero
/// baseline cannot be allocated or certified.  Every heuristic failure keeps
/// that baseline instead.
pub fn propose_coordinate_dual_unwired(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    config: CoordinateDualConfig,
    lower_warm_start: Option<&[f64]>,
    upper_warm_start: Option<&[f64]>,
) -> Result<CoordinateDualProposal, CoordinateDualProposerError> {
    let mut gate = InertConstrainedZonotopeCallGate;
    match propose_coordinate_dual_impl(
        domain,
        direction,
        config,
        lower_warm_start,
        upper_warm_start,
        &mut gate,
    ) {
        Ok(proposal) => Ok(proposal),
        Err(CoordinateDualBudgetError::Proposal(error)) => Err(error),
        Err(CoordinateDualBudgetError::Budget(_)) => {
            unreachable!("the inert coordinate-dual call gate cannot refuse work")
        }
    }
}

/// Coordinate-dual proposal behind the shared synchronous execution firewall.
///
/// The mandatory zero baseline and every accepted heuristic candidate replay
/// on the same outward evaluator and the same absolute deadline. Peak
/// admission includes all simultaneously retained proposal vectors, sort
/// storage, and nested replay diagnostics.
///
/// # Errors
///
/// Returns [`CoordinateDualBudgetError::Proposal`] when the mandatory zero
/// baseline cannot be allocated or certified. Returns
/// [`CoordinateDualBudgetError::Budget`] before publishing a result when the
/// deadline, peak-memory ceiling, or checked resource accounting refuses work.
pub fn propose_coordinate_dual_unwired_with_budget(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    config: CoordinateDualConfig,
    lower_warm_start: Option<&[f64]>,
    upper_warm_start: Option<&[f64]>,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<ConstrainedZonotopeCallOutcome<CoordinateDualProposal>, CoordinateDualBudgetError> {
    let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    let proposal = propose_coordinate_dual_impl(
        domain,
        direction,
        config,
        lower_warm_start,
        upper_warm_start,
        &mut gate,
    )?;
    Ok(ConstrainedZonotopeCallOutcome::new(proposal, gate.report()))
}

#[cfg(test)]
fn propose_coordinate_dual_unwired_with_clock<N>(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    config: CoordinateDualConfig,
    lower_warm_start: Option<&[f64]>,
    upper_warm_start: Option<&[f64]>,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<ConstrainedZonotopeCallOutcome<CoordinateDualProposal>, CoordinateDualBudgetError>
where
    N: FnMut(&'static str) -> std::time::Instant,
{
    let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
    let proposal = propose_coordinate_dual_impl(
        domain,
        direction,
        config,
        lower_warm_start,
        upper_warm_start,
        &mut gate,
    )?;
    Ok(ConstrainedZonotopeCallOutcome::new(proposal, gate.report()))
}

fn propose_coordinate_dual_impl<G>(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    config: CoordinateDualConfig,
    lower_warm_start: Option<&[f64]>,
    upper_warm_start: Option<&[f64]>,
    gate: &mut G,
) -> Result<CoordinateDualProposal, CoordinateDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let constraint_count = domain.constraint_count();
    gate.preflight_peak_live_bytes(coordinate_dual_baseline_peak_live_bytes(constraint_count)?)?;
    gate.checkpoint("coordinate-dual baseline allocation")?;
    let lower_zero = try_zero_multipliers(constraint_count, "lower", gate)?;
    let upper_zero = try_zero_multipliers(constraint_count, "upper", gate)?;

    // This is intentionally first and mandatory.  In particular, invalid
    // direction data or an unsupported floating-point environment must not be
    // hidden behind a proposer resource fallback.
    let baseline = mandatory_dual_replay(domain, direction, &lower_zero, gate)?;
    gate.checkpoint("coordinate-dual mandatory baseline complete")?;

    let Some(plan) = SearchPlan::checked_with_gate(domain, config, gate)? else {
        gate.checkpoint("coordinate-dual publication")?;
        return Ok(baseline_proposal(baseline, lower_zero, upper_zero));
    };
    if !valid_warm_start(lower_warm_start, constraint_count, gate)?
        || !valid_warm_start(upper_warm_start, constraint_count, gate)?
    {
        gate.checkpoint("coordinate-dual publication")?;
        return Ok(baseline_proposal(baseline, lower_zero, upper_zero));
    }
    gate.preflight_peak_live_bytes(coordinate_dual_search_peak_live_bytes(
        constraint_count,
        plan.alpha_dim,
    )?)?;
    gate.checkpoint("coordinate-dual search-memory preflight complete")?;

    let projected_generators = match project_generators(domain, direction, plan.alpha_dim, gate) {
        Ok(projected) => projected,
        Err(CoordinateSearchError::Candidate(_)) => {
            gate.checkpoint("coordinate-dual publication")?;
            return Ok(baseline_proposal(baseline, lower_zero, upper_zero));
        }
        Err(CoordinateSearchError::Budget(error)) => return Err(error.into()),
    };

    let lower_seed = lower_warm_start.unwrap_or(&lower_zero);
    let upper_seed = upper_warm_start.unwrap_or(&upper_zero);
    let lower_candidate =
        optional_coordinate_candidate(domain, &projected_generators, 1.0, lower_seed, plan, gate)?;
    let upper_candidate =
        optional_coordinate_candidate(domain, &projected_generators, -1.0, upper_seed, plan, gate)?;

    let mut proposal = baseline_proposal(baseline, lower_zero, upper_zero);

    if let Some(candidate) = lower_candidate {
        // The proposer has no authority.  Swallow candidate-evaluation failure
        // and retain the already-certified zero baseline.
        if let Some(candidate_bounds) = optional_dual_replay(domain, direction, &candidate, gate)? {
            if candidate_bounds.lower > baseline.lower {
                proposal.bounds.lower = candidate_bounds.lower;
                proposal.lower_multipliers = candidate;
                proposal.lower_improved = true;
            }
        }
    }

    if let Some(candidate) = upper_candidate {
        if let Some(candidate_bounds) = optional_dual_replay(domain, direction, &candidate, gate)? {
            if candidate_bounds.upper < baseline.upper {
                proposal.bounds.upper = candidate_bounds.upper;
                proposal.upper_multipliers = candidate;
                proposal.upper_improved = true;
            }
        }
    }

    gate.checkpoint("coordinate-dual publication")?;
    Ok(proposal)
}

fn try_zero_multipliers<G>(
    constraint_count: usize,
    direction: &'static str,
    gate: &mut G,
) -> Result<Vec<f64>, CoordinateDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut multipliers = Vec::new();
    gate.checkpoint("coordinate-dual zero-multiplier allocation")?;
    multipliers
        .try_reserve_exact(constraint_count)
        .map_err(|_| CoordinateDualProposerError::BaselineAllocation { direction })?;
    for _ in 0..constraint_count {
        gate.charge_items(1, "coordinate-dual zero-multiplier initialization")?;
        multipliers.push(0.0);
    }
    Ok(multipliers)
}

fn baseline_proposal(
    bounds: ConstrainedZonotopeDualBounds,
    lower_multipliers: Vec<f64>,
    upper_multipliers: Vec<f64>,
) -> CoordinateDualProposal {
    CoordinateDualProposal {
        bounds,
        lower_multipliers,
        upper_multipliers,
        lower_improved: false,
        upper_improved: false,
    }
}

fn coordinate_dual_baseline_peak_live_bytes(
    constraint_count: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_elements::<f64>(
        constraint_count.checked_mul(2).ok_or(
            ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "coordinate-dual baseline multiplier count",
            },
        )?,
        "coordinate-dual baseline multiplier bytes",
    )?;
    peak.add_bytes(
        DUAL_SHAPE_ERROR_LIVE_BYTES,
        "coordinate-dual replay diagnostic bytes",
    )?;
    Ok(peak.finish())
}

fn coordinate_dual_search_peak_live_bytes(
    constraint_count: usize,
    alpha_dim: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let retained_multiplier_count = constraint_count.checked_mul(4).ok_or(
        ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "coordinate-dual peak multiplier count",
        },
    )?;

    // Candidate construction retains the two zero-baseline vectors, the
    // completed candidate from the other direction, and the candidate under
    // construction. Its projected/base/breakpoint scratch is gone before
    // either authoritative replay begins.
    let mut candidate_peak = ConstrainedZonotopePeakLiveBytes::new();
    candidate_peak.add_elements::<f64>(
        retained_multiplier_count,
        "coordinate-dual retained multiplier bytes",
    )?;
    candidate_peak.add_elements::<f64>(
        alpha_dim
            .checked_mul(3)
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "coordinate-dual peak alpha scratch count",
            })?,
        "coordinate-dual projected/base scratch bytes",
    )?;
    candidate_peak.add_elements::<Breakpoint>(
        alpha_dim
            .checked_add(1)
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "coordinate-dual breakpoint capacity",
            })?,
        "coordinate-dual breakpoint bytes",
    )?;

    // Replay retains the same four multiplier vectors plus the shared
    // projected-generator vector and the nested evaluator's complete
    // diagnostic allowance. Candidate sort scratch cannot overlap this phase,
    // so the exact simultaneous peak is the maximum, not the sum.
    let mut replay_peak = ConstrainedZonotopePeakLiveBytes::new();
    replay_peak.add_elements::<f64>(
        retained_multiplier_count,
        "coordinate-dual replay multiplier bytes",
    )?;
    replay_peak.add_elements::<f64>(
        alpha_dim,
        "coordinate-dual replay projected-generator bytes",
    )?;
    replay_peak.add_bytes(
        DUAL_SHAPE_ERROR_LIVE_BYTES,
        "coordinate-dual replay diagnostic bytes",
    )?;

    Ok(candidate_peak.finish().max(replay_peak.finish()))
}

fn mandatory_dual_replay<G>(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    multipliers: &[f64],
    gate: &mut G,
) -> Result<ConstrainedZonotopeDualBounds, CoordinateDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    match evaluate_constrained_zonotope64_dual_with_call_gate(domain, direction, multipliers, gate)
    {
        Ok(bounds) => Ok(bounds),
        Err(ConstrainedZonotopeDualBudgetError::Evaluation(error)) => Err(
            CoordinateDualProposerError::Baseline(ConstrainedZonotope64Error::from(error)).into(),
        ),
        Err(ConstrainedZonotopeDualBudgetError::Budget(error)) => Err(error.into()),
    }
}

fn optional_dual_replay<G>(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    multipliers: &[f64],
    gate: &mut G,
) -> Result<Option<ConstrainedZonotopeDualBounds>, CoordinateDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    match evaluate_constrained_zonotope64_dual_with_call_gate(domain, direction, multipliers, gate)
    {
        Ok(bounds) => Ok(Some(bounds)),
        Err(ConstrainedZonotopeDualBudgetError::Evaluation(_)) => Ok(None),
        Err(ConstrainedZonotopeDualBudgetError::Budget(error)) => Err(error.into()),
    }
}

fn valid_warm_start<G>(
    warm_start: Option<&[f64]>,
    constraint_count: usize,
    gate: &mut G,
) -> Result<bool, ConstrainedZonotopeCallBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let Some(warm_start) = warm_start else {
        return Ok(true);
    };
    if warm_start.len() != constraint_count {
        return Ok(false);
    }
    for value in warm_start {
        gate.charge_items(1, "coordinate-dual warm-start validation")?;
        if !value.is_finite() || *value < 0.0 {
            return Ok(false);
        }
    }
    Ok(true)
}

#[derive(Clone, Copy, Debug)]
struct SearchPlan {
    sweeps: usize,
    alpha_dim: usize,
    breakpoint_capacity: usize,
}

impl SearchPlan {
    fn checked_with_gate<G>(
        domain: &ConstrainedZonotope64,
        config: CoordinateDualConfig,
        gate: &mut G,
    ) -> Result<Option<Self>, ConstrainedZonotopeCallBudgetError>
    where
        G: ConstrainedZonotopeCallGate,
    {
        let limits = config.limits;
        if !(1..=COORDINATE_DUAL_MAX_SWEEPS).contains(&config.sweeps)
            || limits.max_constraints == 0
            || limits.max_alpha_dim == 0
            || limits.max_breakpoints == 0
            || limits.max_work == 0
            || limits.max_constraints > COORDINATE_DUAL_HARD_MAX_CONSTRAINTS
            || limits.max_alpha_dim > COORDINATE_DUAL_HARD_MAX_ALPHA_DIM
            || limits.max_breakpoints > COORDINATE_DUAL_HARD_MAX_BREAKPOINTS
            || limits.max_work > COORDINATE_DUAL_HARD_MAX_WORK
        {
            return Ok(None);
        }

        let constraint_count = domain.constraint_count();
        let alpha_dim = domain.alpha_dim();
        if constraint_count > limits.max_constraints || alpha_dim > limits.max_alpha_dim {
            return Ok(None);
        }
        let Some(breakpoint_capacity) = alpha_dim.checked_add(1) else {
            return Ok(None);
        };
        if breakpoint_capacity > limits.max_breakpoints {
            return Ok(None);
        }

        let mut generator_nonzeros = 0_usize;
        for generator in domain.generators() {
            gate.charge_items(1, "coordinate-dual search-plan generator scan")?;
            let Some(count) = generator_nonzeros.checked_add(generator.nnz()) else {
                return Ok(None);
            };
            generator_nonzeros = count;
        }

        let Some(total_work) = checked_search_work(
            generator_nonzeros,
            constraint_count,
            alpha_dim,
            usize::from(config.sweeps),
            breakpoint_capacity,
        ) else {
            return Ok(None);
        };
        if total_work > limits.max_work {
            return Ok(None);
        }

        Ok(Some(Self {
            sweeps: usize::from(config.sweeps),
            alpha_dim,
            breakpoint_capacity,
        }))
    }
}

/// Conservative comparison/scalar-work accounting with no wrapping path.
fn checked_search_work(
    generator_nonzeros: usize,
    constraint_count: usize,
    alpha_dim: usize,
    sweeps: usize,
    breakpoint_capacity: usize,
) -> Option<usize> {
    // Deterministic sort plus scans/update: 3*m +
    // (m+1)*ceil(log2(m+1)) + 1 per coordinate.  Account for both directional
    // searches and both initial C^T lambda projections.  The independently
    // authoritative evaluator has its own fail-closed arithmetic and is
    // outside this heuristic work budget.
    let sort_factor = ceil_log2(breakpoint_capacity);
    // A fourfold comparison allowance avoids pretending that the asymptotic
    // `n log2(n)` term is an exact implementation count.
    let sort_work = breakpoint_capacity
        .checked_mul(sort_factor)?
        .checked_mul(4)?;
    let scan_work = alpha_dim
        .checked_mul(3)?
        .checked_add(sort_work)?
        .checked_add(1)?;
    let coordinate_work = constraint_count
        .checked_mul(sweeps)?
        .checked_mul(scan_work)?
        .checked_mul(2)?;
    let initial_projection_work = constraint_count.checked_mul(alpha_dim)?.checked_mul(2)?;
    generator_nonzeros
        .checked_add(initial_projection_work)?
        .checked_add(coordinate_work)
}

fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::try_from(usize::BITS - (value - 1).leading_zeros())
            .expect("usize bit width fits usize")
    }
}

#[derive(Clone, Copy, Debug)]
struct Breakpoint {
    location: f64,
    slope_drop: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateFailure {
    Allocation,
    NonFiniteArithmetic,
    UnboundedCoordinate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CoordinateSearchError {
    Candidate(CandidateFailure),
    Budget(ConstrainedZonotopeCallBudgetError),
}

impl From<CandidateFailure> for CoordinateSearchError {
    fn from(error: CandidateFailure) -> Self {
        Self::Candidate(error)
    }
}

impl From<ConstrainedZonotopeCallBudgetError> for CoordinateSearchError {
    fn from(error: ConstrainedZonotopeCallBudgetError) -> Self {
        Self::Budget(error)
    }
}

fn project_generators<G>(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    alpha_dim: usize,
    gate: &mut G,
) -> Result<Vec<f64>, CoordinateSearchError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut projected = Vec::new();
    gate.checkpoint("coordinate-dual projected-generator allocation")?;
    projected
        .try_reserve_exact(alpha_dim)
        .map_err(|_| CandidateFailure::Allocation)?;
    for _ in 0..alpha_dim {
        gate.charge_items(1, "coordinate-dual projected-generator initialization")?;
        projected.push(0.0);
    }

    for (column, generator) in domain.generators().iter().enumerate() {
        gate.charge_items(1, "coordinate-dual projected-generator column")?;
        let mut sum = 0.0;
        for (value_index, coefficient) in generator.entries() {
            gate.charge_items(1, "coordinate-dual projected-generator entry")?;
            let product = finite_mul(direction[value_index], coefficient)?;
            sum = finite_add(sum, product)?;
        }
        projected[column] = sum;
    }
    Ok(projected)
}

fn optional_coordinate_candidate<G>(
    domain: &ConstrainedZonotope64,
    projected_generators: &[f64],
    generator_sign: f64,
    warm_start: &[f64],
    plan: SearchPlan,
    gate: &mut G,
) -> Result<Option<Vec<f64>>, CoordinateDualBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    match coordinate_candidate(
        domain,
        projected_generators,
        generator_sign,
        warm_start,
        plan,
        gate,
    ) {
        Ok(candidate) => Ok(Some(candidate)),
        Err(CoordinateSearchError::Candidate(_)) => Ok(None),
        Err(CoordinateSearchError::Budget(error)) => Err(error.into()),
    }
}

fn coordinate_candidate<G>(
    domain: &ConstrainedZonotope64,
    projected_generators: &[f64],
    generator_sign: f64,
    warm_start: &[f64],
    plan: SearchPlan,
    gate: &mut G,
) -> Result<Vec<f64>, CoordinateSearchError>
where
    G: ConstrainedZonotopeCallGate,
{
    let constraint_count = domain.constraint_count();
    let constraints = domain.constraints();
    let mut multipliers = copy_finite_vector(warm_start, gate)?;
    let mut projected = Vec::new();
    gate.checkpoint("coordinate-dual candidate projection allocation")?;
    projected
        .try_reserve_exact(plan.alpha_dim)
        .map_err(|_| CandidateFailure::Allocation)?;
    for &value in projected_generators {
        gate.charge_items(1, "coordinate-dual candidate projection initialization")?;
        projected.push(finite_mul(generator_sign, value)?);
    }

    for row in 0..constraint_count {
        gate.charge_items(1, "coordinate-dual warm projection row")?;
        let multiplier = multipliers[row];
        if multiplier == 0.0 {
            continue;
        }
        for column in 0..plan.alpha_dim {
            gate.charge_items(1, "coordinate-dual warm projection entry")?;
            let contribution = finite_mul(constraints[[row, column]], multiplier)?;
            projected[column] = finite_add(projected[column], contribution)?;
        }
    }

    let mut base = Vec::new();
    gate.checkpoint("coordinate-dual base allocation")?;
    base.try_reserve_exact(plan.alpha_dim)
        .map_err(|_| CandidateFailure::Allocation)?;
    for _ in 0..plan.alpha_dim {
        gate.charge_items(1, "coordinate-dual base initialization")?;
        base.push(0.0);
    }
    let mut breakpoints = Vec::new();
    gate.checkpoint("coordinate-dual breakpoint allocation")?;
    breakpoints
        .try_reserve_exact(plan.breakpoint_capacity)
        .map_err(|_| CandidateFailure::Allocation)?;

    for _ in 0..plan.sweeps {
        gate.charge_items(1, "coordinate-dual candidate sweep")?;
        for row in 0..constraint_count {
            gate.charge_items(1, "coordinate-dual candidate row")?;
            let old = multipliers[row];
            breakpoints.clear();
            // The nonnegative boundary is a real member of the sorted
            // candidate list, not a post-hoc clamp.
            breakpoints.push(Breakpoint {
                location: 0.0,
                slope_drop: 0.0,
            });

            let mut right_slope = finite_neg(domain.rhs()[row])?;
            for column in 0..plan.alpha_dim {
                gate.charge_items(1, "coordinate-dual candidate column")?;
                let coefficient = constraints[[row, column]];
                let removed = finite_mul(coefficient, old)?;
                let base_value = finite_sub(projected[column], removed)?;
                base[column] = base_value;

                if coefficient == 0.0 {
                    continue;
                }

                // Right derivative at zero.  A zero argument immediately
                // takes sign(coefficient) for t > 0.
                let signed_coefficient = if base_value > 0.0 {
                    coefficient
                } else if base_value < 0.0 {
                    finite_neg(coefficient)?
                } else {
                    coefficient.abs()
                };
                right_slope = finite_sub(right_slope, signed_coefficient)?;

                // A strictly positive root exists exactly when base and row
                // coefficient have opposite signs.  Negative roots do not
                // affect the feasible half-line and are never divided out.
                if (base_value < 0.0 && coefficient > 0.0)
                    || (base_value > 0.0 && coefficient < 0.0)
                {
                    let location = finite_div(finite_neg(base_value)?, coefficient)?;
                    if location <= 0.0 {
                        return Err(CandidateFailure::NonFiniteArithmetic.into());
                    }
                    let slope_drop = finite_mul(2.0, coefficient.abs())?;
                    if breakpoints.len() >= plan.breakpoint_capacity {
                        return Err(CandidateFailure::Allocation.into());
                    }
                    breakpoints.push(Breakpoint {
                        location,
                        slope_drop,
                    });
                }
            }

            // The explicit in-place heapsort has no hidden allocation and
            // polls inside its comparison/swap loops.  The total comparator
            // keeps the groups deterministic even though stability is unused.
            sort_breakpoints_with_gate(&mut breakpoints, gate)?;

            let replacement = if right_slope <= 0.0 {
                0.0
            } else {
                first_nonpositive_slope_breakpoint(&breakpoints, right_slope, gate)?
            };
            multipliers[row] = replacement;
            for column in 0..plan.alpha_dim {
                gate.charge_items(1, "coordinate-dual candidate update")?;
                let contribution = finite_mul(constraints[[row, column]], replacement)?;
                projected[column] = finite_add(base[column], contribution)?;
            }
        }
    }

    for value in &multipliers {
        gate.charge_items(1, "coordinate-dual candidate validation")?;
        if !value.is_finite() || *value < 0.0 {
            return Err(CandidateFailure::NonFiniteArithmetic.into());
        }
    }
    Ok(multipliers)
}

fn breakpoint_order(left: &Breakpoint, right: &Breakpoint) -> Ordering {
    let order = left.location.total_cmp(&right.location);
    if order == Ordering::Equal {
        left.slope_drop.total_cmp(&right.slope_drop)
    } else {
        order
    }
}

fn sort_breakpoints_with_gate<G>(
    breakpoints: &mut [Breakpoint],
    gate: &mut G,
) -> Result<(), ConstrainedZonotopeCallBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let len = breakpoints.len();
    if len < 2 {
        return Ok(());
    }

    for root in (0..(len / 2)).rev() {
        sift_breakpoints_down(breakpoints, root, len, gate)?;
    }
    for end in (1..len).rev() {
        gate.charge_items(1, "coordinate-dual breakpoint sort swap")?;
        breakpoints.swap(0, end);
        sift_breakpoints_down(breakpoints, 0, end, gate)?;
    }
    Ok(())
}

fn sift_breakpoints_down<G>(
    breakpoints: &mut [Breakpoint],
    mut root: usize,
    end: usize,
    gate: &mut G,
) -> Result<(), ConstrainedZonotopeCallBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    loop {
        let Some(left_child) = root.checked_mul(2).and_then(|value| value.checked_add(1)) else {
            return Ok(());
        };
        if left_child >= end {
            return Ok(());
        }

        let mut greater_child = left_child;
        let right_child = left_child + 1;
        if right_child < end {
            gate.charge_items(1, "coordinate-dual breakpoint sort comparison")?;
            if breakpoint_order(&breakpoints[greater_child], &breakpoints[right_child])
                == Ordering::Less
            {
                greater_child = right_child;
            }
        }

        gate.charge_items(1, "coordinate-dual breakpoint sort comparison")?;
        if breakpoint_order(&breakpoints[root], &breakpoints[greater_child]) != Ordering::Less {
            return Ok(());
        }
        gate.charge_items(1, "coordinate-dual breakpoint sort swap")?;
        breakpoints.swap(root, greater_child);
        root = greater_child;
    }
}

fn first_nonpositive_slope_breakpoint<G>(
    breakpoints: &[Breakpoint],
    mut slope: f64,
    gate: &mut G,
) -> Result<f64, CoordinateSearchError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut index = 1;
    while index < breakpoints.len() {
        gate.charge_items(1, "coordinate-dual breakpoint group")?;
        let location = breakpoints[index].location;
        let mut total_drop = 0.0;
        while index < breakpoints.len() && breakpoints[index].location == location {
            gate.charge_items(1, "coordinate-dual breakpoint slope scan")?;
            total_drop = finite_add(total_drop, breakpoints[index].slope_drop)?;
            index += 1;
        }
        slope = finite_sub(slope, total_drop)?;
        if slope <= 0.0 {
            return Ok(location);
        }
    }

    // A positive asymptotic slope denotes an unbounded heuristic dual.  It can
    // expose an empty predicate, but this unwired proposer has no authority to
    // use that fact and must retain the finite zero baseline.
    Err(CandidateFailure::UnboundedCoordinate.into())
}

fn copy_finite_vector<G>(source: &[f64], gate: &mut G) -> Result<Vec<f64>, CoordinateSearchError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut copied = Vec::new();
    gate.checkpoint("coordinate-dual multiplier allocation")?;
    copied
        .try_reserve_exact(source.len())
        .map_err(|_| CandidateFailure::Allocation)?;
    for &value in source {
        gate.charge_items(1, "coordinate-dual multiplier copy")?;
        copied.push(value);
    }
    Ok(copied)
}

fn finite_add(left: f64, right: f64) -> Result<f64, CandidateFailure> {
    let value = left + right;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CandidateFailure::NonFiniteArithmetic)
    }
}

fn finite_sub(left: f64, right: f64) -> Result<f64, CandidateFailure> {
    let value = left - right;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CandidateFailure::NonFiniteArithmetic)
    }
}

fn finite_mul(left: f64, right: f64) -> Result<f64, CandidateFailure> {
    let value = left * right;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CandidateFailure::NonFiniteArithmetic)
    }
}

fn finite_div(left: f64, right: f64) -> Result<f64, CandidateFailure> {
    let value = left / right;
    if right != 0.0 && value.is_finite() {
        Ok(value)
    } else {
        Err(CandidateFailure::NonFiniteArithmetic)
    }
}

fn finite_neg(value: f64) -> Result<f64, CandidateFailure> {
    let negated = -value;
    if negated.is_finite() {
        Ok(negated)
    } else {
        Err(CandidateFailure::NonFiniteArithmetic)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::mem::size_of;
    use std::time::{Duration, Instant};

    use ndarray::{array, Array2};
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};
    use proptest::prelude::*;

    use super::*;

    fn domain(
        center: Vec<f64>,
        generators: Vec<Vec<(usize, f64)>>,
        constraints: Array2<f64>,
        rhs: Vec<f64>,
        remainder: Vec<f64>,
    ) -> ConstrainedZonotope64 {
        ConstrainedZonotope64::try_new(center, generators, constraints, rhs, remainder).unwrap()
    }

    fn rat(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite dyadic")
    }

    fn exact_dual_endpoint(
        domain: &ConstrainedZonotope64,
        direction: &[f64],
        multipliers: &[f64],
        lower: bool,
    ) -> BigRational {
        let mut center = BigRational::zero();
        let mut box_charge = BigRational::zero();
        for ((&q, &c), &r) in direction
            .iter()
            .zip(domain.center())
            .zip(domain.box_remainder())
        {
            center += rat(q) * rat(c);
            box_charge += rat(q).abs() * rat(r);
        }
        let lambda_d = multipliers
            .iter()
            .zip(domain.rhs())
            .fold(BigRational::zero(), |sum, (&lambda, &rhs)| {
                sum + rat(lambda) * rat(rhs)
            });
        let mut norm = BigRational::zero();
        for column in 0..domain.alpha_dim() {
            let mut g = BigRational::zero();
            for (value_index, coefficient) in domain.generators()[column].entries() {
                g += rat(direction[value_index]) * rat(coefficient);
            }
            let c_lambda = (0..domain.constraint_count()).fold(BigRational::zero(), |sum, row| {
                sum + rat(domain.constraints()[[row, column]]) * rat(multipliers[row])
            });
            norm += if lower { g + c_lambda } else { g - c_lambda }.abs();
        }
        if lower {
            center - lambda_d - norm - box_charge
        } else {
            center + lambda_d + norm + box_charge
        }
    }

    fn assert_baseline(proposal: &CoordinateDualProposal) {
        assert!(!proposal.lower_improved);
        assert!(!proposal.upper_improved);
        assert!(proposal.lower_multipliers.iter().all(|&value| value == 0.0));
        assert!(proposal.upper_multipliers.iter().all(|&value| value == 0.0));
    }

    fn budget_toy() -> ConstrainedZonotope64 {
        domain(
            vec![0.25],
            vec![vec![(0, 1.0)], vec![(0, -0.5)]],
            array![[-1.0, 0.5], [0.25, -1.0]],
            vec![0.0, 0.5],
            vec![0.125],
        )
    }

    fn assert_proposal_bits_equal(left: &CoordinateDualProposal, right: &CoordinateDualProposal) {
        assert_eq!(left.bounds.lower.to_bits(), right.bounds.lower.to_bits());
        assert_eq!(left.bounds.upper.to_bits(), right.bounds.upper.to_bits());
        assert_eq!(left.lower_improved, right.lower_improved);
        assert_eq!(left.upper_improved, right.upper_improved);
        assert_eq!(
            left.lower_multipliers
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            right
                .lower_multipliers
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            left.upper_multipliers
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            right
                .upper_multipliers
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    struct RefuseAtCharge {
        target: &'static str,
        inner: InertConstrainedZonotopeCallGate,
    }

    impl RefuseAtCharge {
        fn new(target: &'static str) -> Self {
            Self {
                target,
                inner: InertConstrainedZonotopeCallGate,
            }
        }
    }

    impl ConstrainedZonotopeCallGate for RefuseAtCharge {
        fn is_enforcing(&self) -> bool {
            true
        }

        fn checkpoint(
            &mut self,
            checkpoint: &'static str,
        ) -> Result<(), ConstrainedZonotopeCallBudgetError> {
            self.inner.checkpoint(checkpoint)
        }

        fn charge_items(
            &mut self,
            items: usize,
            checkpoint: &'static str,
        ) -> Result<(), ConstrainedZonotopeCallBudgetError> {
            if checkpoint == self.target {
                return Err(ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint });
            }
            self.inner.charge_items(items, checkpoint)
        }

        fn preflight_peak_live_bytes(
            &mut self,
            transform_owned_bytes: usize,
        ) -> Result<(), ConstrainedZonotopeCallBudgetError> {
            self.inner.preflight_peak_live_bytes(transform_owned_bytes)
        }

        fn report(&self) -> crate::ConstrainedZonotopeCallReport {
            self.inner.report()
        }
    }

    #[test]
    fn exact_lower_and_upper_toys_use_opposite_sign_transforms() {
        let alpha_nonnegative = domain(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            array![[-1.0]],
            vec![0.0],
            vec![0.125],
        );
        let positive = propose_coordinate_dual_unwired(
            &alpha_nonnegative,
            &[1.0],
            CoordinateDualConfig::default(),
            None,
            None,
        )
        .unwrap();
        assert!(positive.lower_improved);
        assert!(!positive.upper_improved);
        assert_eq!(positive.lower_multipliers, vec![1.0]);
        assert!(positive.bounds.lower > -1.125);
        assert!(positive.bounds.lower <= -0.125);
        assert_eq!(positive.bounds.upper, 1.125_f64.next_up());

        let negative = propose_coordinate_dual_unwired(
            &alpha_nonnegative,
            &[-1.0],
            CoordinateDualConfig::default(),
            None,
            None,
        )
        .unwrap();
        assert!(!negative.lower_improved);
        assert!(negative.upper_improved);
        assert_eq!(negative.upper_multipliers, vec![1.0]);
        assert!(negative.bounds.upper < 1.125);
        assert!(negative.bounds.upper >= 0.125);
    }

    #[test]
    fn independent_upper_toy_tightens_alpha_le_zero() {
        let alpha_nonpositive = domain(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            array![[1.0]],
            vec![0.0],
            vec![0.0],
        );
        let proposal = propose_coordinate_dual_unwired(
            &alpha_nonpositive,
            &[1.0],
            CoordinateDualConfig {
                sweeps: 1,
                ..CoordinateDualConfig::default()
            },
            None,
            None,
        )
        .unwrap();
        assert!(!proposal.lower_improved);
        assert!(proposal.upper_improved);
        assert_eq!(proposal.upper_multipliers, vec![1.0]);
        assert!(proposal.bounds.upper.abs() <= f64::from_bits(8));
    }

    #[test]
    fn warm_start_is_used_and_duplicate_breakpoints_are_deterministic() {
        let duplicate_rows = domain(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            array![[-1.0], [-1.0]],
            vec![0.0, 0.0],
            vec![0.0],
        );
        let config = CoordinateDualConfig {
            sweeps: 1,
            ..CoordinateDualConfig::default()
        };
        let cold =
            propose_coordinate_dual_unwired(&duplicate_rows, &[1.0], config, None, None).unwrap();
        let warm = [0.5, 0.5];
        let warm_first =
            propose_coordinate_dual_unwired(&duplicate_rows, &[1.0], config, Some(&warm), None)
                .unwrap();
        let warm_second =
            propose_coordinate_dual_unwired(&duplicate_rows, &[1.0], config, Some(&warm), None)
                .unwrap();

        assert_eq!(cold.lower_multipliers, vec![1.0, 0.0]);
        assert_eq!(warm_first.lower_multipliers, vec![0.5, 0.5]);
        assert_eq!(warm_first, warm_second);
        assert_eq!(
            exact_dual_endpoint(&duplicate_rows, &[1.0], &cold.lower_multipliers, true,),
            exact_dual_endpoint(&duplicate_rows, &[1.0], &warm_first.lower_multipliers, true,)
        );
        assert!(cold.bounds.lower > -1.0e-12);
        assert!(warm_first.bounds.lower > -1.0e-12);
    }

    #[test]
    fn every_supported_sweep_count_is_deterministic_and_non_regressing() {
        let toy = domain(
            vec![0.25],
            vec![vec![(0, 1.0)], vec![(0, -0.5)]],
            array![[-1.0, 0.5], [0.25, -1.0]],
            vec![0.0, 0.5],
            vec![0.125],
        );
        let zero = [0.0, 0.0];
        let baseline = toy.evaluate_dual(&[1.0], &zero).unwrap();
        for sweeps in 1..=COORDINATE_DUAL_MAX_SWEEPS {
            let config = CoordinateDualConfig {
                sweeps,
                ..CoordinateDualConfig::default()
            };
            let first = propose_coordinate_dual_unwired(&toy, &[1.0], config, None, None).unwrap();
            let second = propose_coordinate_dual_unwired(&toy, &[1.0], config, None, None).unwrap();
            assert_eq!(first, second);
            assert!(first.bounds.lower >= baseline.lower);
            assert!(first.bounds.upper <= baseline.upper);
        }
    }

    #[test]
    fn zero_constraints_generators_and_dimensions_are_supported() {
        let empty = domain(
            Vec::new(),
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            Vec::new(),
        );
        let empty_proposal = propose_coordinate_dual_unwired(
            &empty,
            &[],
            CoordinateDualConfig::default(),
            None,
            None,
        )
        .unwrap();
        assert_baseline(&empty_proposal);
        assert_eq!(empty_proposal.bounds.lower, 0.0);
        assert_eq!(empty_proposal.bounds.upper, 0.0);

        let no_generators = domain(
            vec![2.0],
            Vec::new(),
            Array2::zeros((1, 0)),
            vec![1.0],
            vec![0.25],
        );
        let proposal = propose_coordinate_dual_unwired(
            &no_generators,
            &[1.0],
            CoordinateDualConfig::default(),
            None,
            None,
        )
        .unwrap();
        assert_baseline(&proposal);

        let no_constraints = domain(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0],
        );
        let proposal = propose_coordinate_dual_unwired(
            &no_constraints,
            &[1.0],
            CoordinateDualConfig::default(),
            None,
            None,
        )
        .unwrap();
        assert_baseline(&proposal);
    }

    #[test]
    fn malformed_configuration_and_warm_starts_retain_baseline() {
        let toy = domain(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            array![[-1.0]],
            vec![0.0],
            vec![0.0],
        );
        let invalid_configs = [
            CoordinateDualConfig {
                sweeps: 0,
                ..CoordinateDualConfig::default()
            },
            CoordinateDualConfig {
                sweeps: 5,
                ..CoordinateDualConfig::default()
            },
            CoordinateDualConfig {
                limits: CoordinateDualLimits {
                    max_alpha_dim: 0,
                    ..CoordinateDualLimits::default()
                },
                ..CoordinateDualConfig::default()
            },
            CoordinateDualConfig {
                limits: CoordinateDualLimits {
                    max_constraints: COORDINATE_DUAL_HARD_MAX_CONSTRAINTS + 1,
                    ..CoordinateDualLimits::default()
                },
                ..CoordinateDualConfig::default()
            },
            CoordinateDualConfig {
                limits: CoordinateDualLimits {
                    max_work: 1,
                    ..CoordinateDualLimits::default()
                },
                ..CoordinateDualConfig::default()
            },
        ];
        for config in invalid_configs {
            assert_baseline(
                &propose_coordinate_dual_unwired(&toy, &[1.0], config, None, None).unwrap(),
            );
        }

        for warm in [&[][..], &[f64::NAN][..], &[-1.0][..], &[0.0, 0.0][..]] {
            assert_baseline(
                &propose_coordinate_dual_unwired(
                    &toy,
                    &[1.0],
                    CoordinateDualConfig::default(),
                    Some(warm),
                    None,
                )
                .unwrap(),
            );
        }
    }

    #[test]
    fn caller_resource_caps_retain_baseline() {
        let toy = domain(
            vec![0.0],
            vec![vec![(0, 1.0)], vec![(0, 0.5)]],
            array![[-1.0, 0.0], [0.0, -1.0]],
            vec![0.0, 0.0],
            vec![0.0],
        );
        for limits in [
            CoordinateDualLimits {
                max_constraints: 1,
                ..CoordinateDualLimits::default()
            },
            CoordinateDualLimits {
                max_alpha_dim: 1,
                ..CoordinateDualLimits::default()
            },
            CoordinateDualLimits {
                max_breakpoints: 2,
                ..CoordinateDualLimits::default()
            },
        ] {
            assert_baseline(
                &propose_coordinate_dual_unwired(
                    &toy,
                    &[1.0],
                    CoordinateDualConfig { sweeps: 1, limits },
                    None,
                    None,
                )
                .unwrap(),
            );
        }
    }

    #[test]
    fn nonfinite_and_unbounded_proposals_retain_baseline() {
        let overflow = domain(
            vec![0.0],
            vec![Vec::new()],
            array![[f64::MAX]],
            vec![0.0],
            vec![0.0],
        );
        let huge = [f64::MAX];
        assert_baseline(
            &propose_coordinate_dual_unwired(
                &overflow,
                &[0.0],
                CoordinateDualConfig::default(),
                Some(&huge),
                Some(&huge),
            )
            .unwrap(),
        );

        let unbounded = domain(
            Vec::new(),
            Vec::new(),
            Array2::zeros((1, 0)),
            vec![-1.0],
            Vec::new(),
        );
        assert_baseline(
            &propose_coordinate_dual_unwired(
                &unbounded,
                &[],
                CoordinateDualConfig::default(),
                None,
                None,
            )
            .unwrap(),
        );
    }

    #[test]
    fn invalid_direction_fails_at_mandatory_baseline() {
        let toy = domain(
            vec![0.0],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0],
        );
        assert!(matches!(
            propose_coordinate_dual_unwired(
                &toy,
                &[f64::NAN],
                CoordinateDualConfig::default(),
                None,
                None,
            ),
            Err(CoordinateDualProposerError::Baseline(
                ConstrainedZonotope64Error::Dual(_)
            ))
        ));
    }

    #[test]
    fn mandatory_error_order_matches_legacy_and_optional_failures_stay_private() {
        let toy = domain(
            vec![0.0],
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            vec![0.0],
        );
        let malformed_config = CoordinateDualConfig {
            sweeps: 0,
            ..CoordinateDualConfig::default()
        };
        let legacy_error =
            propose_coordinate_dual_unwired(&toy, &[f64::NAN], malformed_config, None, None)
                .unwrap_err();
        let deadline = Instant::now() + Duration::from_mins(1);
        assert_eq!(
            propose_coordinate_dual_unwired_with_budget(
                &toy,
                &[f64::NAN],
                malformed_config,
                None,
                None,
                ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
            )
            .unwrap_err(),
            CoordinateDualBudgetError::Proposal(legacy_error),
            "mandatory direction/FTZ authority must precede heuristic fallback"
        );

        // The zero baseline for this domain is finite, while replaying the
        // supplied optional multiplier overflows at lambda*d. Evaluation
        // failures are private heuristic failures; budget refusals are not.
        let replay_overflow = domain(
            Vec::new(),
            Vec::new(),
            Array2::zeros((1, 0)),
            vec![f64::MAX],
            Vec::new(),
        );
        let mut inert = InertConstrainedZonotopeCallGate;
        assert!(matches!(
            mandatory_dual_replay(&replay_overflow, &[], &[2.0], &mut inert),
            Err(CoordinateDualBudgetError::Proposal(
                CoordinateDualProposerError::Baseline(ConstrainedZonotope64Error::Dual(
                    crate::ConstrainedZonotopeDualError::NonFiniteArithmetic { .. }
                ))
            ))
        ));
        assert_eq!(
            optional_dual_replay(&replay_overflow, &[], &[2.0], &mut inert).unwrap(),
            None
        );

        let mut refused = RefuseAtCharge::new("dual finite-input validation");
        assert!(matches!(
            optional_dual_replay(&replay_overflow, &[], &[2.0], &mut refused),
            Err(CoordinateDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "dual finite-input validation"
                }
            ))
        ));
    }

    #[test]
    fn budgeted_search_is_bit_identical_and_reports_exact_complete_peak() {
        let toy = budget_toy();
        let config = CoordinateDualConfig {
            sweeps: 1,
            ..CoordinateDualConfig::default()
        };
        let warm = [0.125, -0.0];
        let legacy =
            propose_coordinate_dual_unwired(&toy, &[1.0], config, Some(&warm), Some(&warm))
                .unwrap();

        let constraint_count = toy.constraint_count();
        let alpha_dim = toy.alpha_dim();
        let candidate_peak = constraint_count * 4 * size_of::<f64>()
            + alpha_dim * 3 * size_of::<f64>()
            + (alpha_dim + 1) * size_of::<Breakpoint>();
        let replay_peak = constraint_count * 4 * size_of::<f64>()
            + alpha_dim * size_of::<f64>()
            + DUAL_SHAPE_ERROR_LIVE_BYTES;
        let transform_peak = candidate_peak.max(replay_peak);
        let baseline_live_bytes = 13;
        let exact_peak = baseline_live_bytes + transform_peak;
        assert_eq!(
            coordinate_dual_search_peak_live_bytes(constraint_count, alpha_dim).unwrap(),
            transform_peak
        );
        assert_eq!(
            coordinate_dual_search_peak_live_bytes(1, 0).unwrap(),
            4 * size_of::<f64>() + DUAL_SHAPE_ERROR_LIVE_BYTES,
            "nested replay dominates when there is no alpha scratch"
        );
        assert_eq!(
            coordinate_dual_search_peak_live_bytes(0, 2).unwrap(),
            6 * size_of::<f64>() + 3 * size_of::<Breakpoint>(),
            "candidate scratch dominates when there are no multiplier vectors"
        );

        let deadline = Instant::now() + Duration::from_mins(1);
        let outcome = propose_coordinate_dual_unwired_with_budget(
            &toy,
            &[1.0],
            config,
            Some(&warm),
            Some(&warm),
            ConstrainedZonotopeCallBudget::new(deadline, baseline_live_bytes, exact_peak),
        )
        .unwrap();
        assert_proposal_bits_equal(outcome.value(), &legacy);
        assert_eq!(outcome.report().peak_live_bytes(), exact_peak);
        assert!(outcome.report().charged_items() > 0);
        assert!(outcome.report().deadline_polls() > 0);

        assert!(matches!(
            propose_coordinate_dual_unwired_with_budget(
                &toy,
                &[1.0],
                config,
                Some(&warm),
                Some(&warm),
                ConstrainedZonotopeCallBudget::new(
                    deadline,
                    baseline_live_bytes,
                    exact_peak - 1,
                ),
            ),
            Err(CoordinateDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                    required,
                    limit,
                }
            )) if required == exact_peak && limit == exact_peak - 1
        ));
    }

    #[test]
    fn baseline_peak_and_aggregate_overflow_fail_before_allocation() {
        let toy = budget_toy();
        let expected_baseline =
            toy.constraint_count() * 2 * size_of::<f64>() + DUAL_SHAPE_ERROR_LIVE_BYTES;
        assert_eq!(
            coordinate_dual_baseline_peak_live_bytes(toy.constraint_count()).unwrap(),
            expected_baseline
        );
        assert!(matches!(
            coordinate_dual_baseline_peak_live_bytes(usize::MAX),
            Err(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "coordinate-dual baseline multiplier count"
            })
        ));
        assert!(matches!(
            coordinate_dual_search_peak_live_bytes(0, usize::MAX),
            Err(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "coordinate-dual peak alpha scratch count"
            })
        ));

        let start = Instant::now();
        let reads = Cell::new(0_usize);
        let result = propose_coordinate_dual_unwired_with_clock(
            &toy,
            &[1.0],
            CoordinateDualConfig::default(),
            None,
            None,
            ConstrainedZonotopeCallBudget::new(
                start + Duration::from_secs(1),
                usize::MAX,
                usize::MAX,
            ),
            |_| {
                reads.set(reads.get() + 1);
                start
            },
        );
        assert!(matches!(
            result,
            Err(CoordinateDualBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                    operation: "aggregate peak-live bytes"
                }
            ))
        ));
        assert_eq!(
            reads.get(),
            1,
            "aggregate overflow must precede coordinate allocation"
        );
    }

    #[test]
    fn deadline_refuses_nested_replay_and_every_coordinate_publication_seam() {
        let toy = budget_toy();
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        let deadline = start + Duration::from_secs(1);

        for seam in [
            "sparse dual input validation",
            "coordinate-dual mandatory baseline complete",
            "coordinate-dual search-memory preflight complete",
            "coordinate-dual multiplier allocation",
            "coordinate-dual base allocation",
            "coordinate-dual breakpoint allocation",
            "coordinate-dual publication",
        ] {
            let result = propose_coordinate_dual_unwired_with_clock(
                &toy,
                &[1.0],
                CoordinateDualConfig {
                    sweeps: 1,
                    ..CoordinateDualConfig::default()
                },
                None,
                None,
                ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
                |checkpoint| {
                    if checkpoint == seam {
                        expired
                    } else {
                        start
                    }
                },
            );
            assert!(
                matches!(
                    result,
                    Err(CoordinateDualBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == seam
                ),
                "deadline seam {seam} must keep the proposal private"
            );
        }
    }

    #[test]
    fn every_search_loop_is_gate_charged_and_large_sort_chunk_polls() {
        let empty_generator = domain(
            Vec::new(),
            vec![Vec::new()],
            Array2::zeros((0, 1)),
            Vec::new(),
            Vec::new(),
        );
        let mut generator_gate = RefuseAtCharge::new("coordinate-dual projected-generator column");
        assert!(matches!(
            project_generators(&empty_generator, &[], 1, &mut generator_gate),
            Err(CoordinateSearchError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "coordinate-dual projected-generator column"
                }
            ))
        ));

        let dense = domain(
            vec![0.0],
            vec![vec![(0, 1.0)], vec![(0, -1.0)]],
            array![[-1.0, 1.0]],
            vec![0.0],
            vec![0.0],
        );
        let plan = SearchPlan {
            sweeps: 1,
            alpha_dim: 2,
            breakpoint_capacity: 3,
        };
        let mut dense_gate = RefuseAtCharge::new("coordinate-dual candidate column");
        assert!(matches!(
            coordinate_candidate(&dense, &[1.0, -1.0], 1.0, &[0.0], plan, &mut dense_gate,),
            Err(CoordinateSearchError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "coordinate-dual candidate column"
                }
            ))
        ));

        for target in [
            "coordinate-dual zero-multiplier initialization",
            "coordinate-dual search-plan generator scan",
            "coordinate-dual warm-start validation",
        ] {
            let mut gate = RefuseAtCharge::new(target);
            let result = match target {
                "coordinate-dual zero-multiplier initialization" => {
                    try_zero_multipliers(1, "test", &mut gate).map(|_| ())
                }
                "coordinate-dual search-plan generator scan" => SearchPlan::checked_with_gate(
                    &dense,
                    CoordinateDualConfig::default(),
                    &mut gate,
                )
                .map(|_| ())
                .map_err(CoordinateDualBudgetError::from),
                "coordinate-dual warm-start validation" => {
                    valid_warm_start(Some(&[0.25]), 1, &mut gate)
                        .map(|_| ())
                        .map_err(CoordinateDualBudgetError::from)
                }
                _ => unreachable!(),
            };
            assert!(
                matches!(
                    result,
                    Err(CoordinateDualBudgetError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == target
                ),
                "deadline charge {target} must refuse"
            );
        }

        for target in [
            "coordinate-dual projected-generator initialization",
            "coordinate-dual projected-generator column",
            "coordinate-dual projected-generator entry",
        ] {
            let mut gate = RefuseAtCharge::new(target);
            assert!(
                matches!(
                    project_generators(&dense, &[1.0], 2, &mut gate),
                    Err(CoordinateSearchError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == target
                ),
                "deadline charge {target} must refuse"
            );
        }

        for target in [
            "coordinate-dual multiplier copy",
            "coordinate-dual candidate projection initialization",
            "coordinate-dual warm projection row",
            "coordinate-dual warm projection entry",
            "coordinate-dual base initialization",
            "coordinate-dual candidate sweep",
            "coordinate-dual candidate row",
            "coordinate-dual candidate column",
            "coordinate-dual breakpoint sort comparison",
            "coordinate-dual breakpoint sort swap",
            "coordinate-dual breakpoint group",
            "coordinate-dual breakpoint slope scan",
            "coordinate-dual candidate update",
            "coordinate-dual candidate validation",
        ] {
            let mut gate = RefuseAtCharge::new(target);
            assert!(
                matches!(
                    coordinate_candidate(
                        &dense,
                        &[1.0, -1.0],
                        1.0,
                        &[0.25],
                        plan,
                        &mut gate,
                    ),
                    Err(CoordinateSearchError::Budget(
                        ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
                    )) if checkpoint == target
                ),
                "deadline charge {target} must refuse"
            );
        }

        let mut breakpoints = (0..COORDINATE_DUAL_HARD_MAX_BREAKPOINTS)
            .rev()
            .map(|index| Breakpoint {
                location: index as f64,
                slope_drop: (index % 7) as f64,
            })
            .collect::<Vec<_>>();
        let start = Instant::now();
        let expired = start + Duration::from_secs(2);
        let mut tracker = ConstrainedZonotopeCallTracker::with_clock(
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |checkpoint| {
                if checkpoint.starts_with("coordinate-dual breakpoint sort") {
                    expired
                } else {
                    start
                }
            },
        )
        .unwrap();
        let result = sort_breakpoints_with_gate(&mut breakpoints, &mut tracker);
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "coordinate-dual breakpoint sort comparison"
                    | "coordinate-dual breakpoint sort swap"
            })
        ));
    }

    #[test]
    fn checked_work_arithmetic_rejects_overflow() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(checked_search_work(7, 2, 3, 4, 4), Some(691));
        assert_eq!(checked_search_work(usize::MAX, 1, 1, 1, 2), None);
        assert_eq!(checked_search_work(0, usize::MAX, 2, 4, 3), None);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn polled_heapsort_is_bit_identical_to_the_legacy_total_order(
            raw in prop::collection::vec((any::<u64>(), any::<u64>()), 0..96),
        ) {
            let mut expected = raw
                .iter()
                .map(|&(location, slope_drop)| Breakpoint {
                    location: f64::from_bits(location),
                    slope_drop: f64::from_bits(slope_drop),
                })
                .collect::<Vec<_>>();
            let mut actual = expected.clone();
            expected.sort_unstable_by(breakpoint_order);
            sort_breakpoints_with_gate(
                &mut actual,
                &mut InertConstrainedZonotopeCallGate,
            )
            .unwrap();
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                prop_assert_eq!(actual.location.to_bits(), expected.location.to_bits());
                prop_assert_eq!(actual.slope_drop.to_bits(), expected.slope_drop.to_bits());
            }
        }

        #[test]
        fn random_dyadic_proposals_are_exactly_certified_and_never_regress(
            generator_seed in prop::collection::vec(-8i16..=8, 3),
            constraint_seed in prop::collection::vec(-8i16..=8, 4),
            rhs_seed in prop::collection::vec(0i16..=8, 2),
            direction_seed in prop::collection::vec(-8i16..=8, 2),
            remainder_seed in prop::collection::vec(0i16..=4, 2),
            sweeps in 1u8..=4,
        ) {
            let scale = |value: i16| f64::from(value) / 8.0;
            let domain = domain(
                vec![0.25, -0.5],
                vec![
                    vec![(0, scale(generator_seed[0])), (1, scale(generator_seed[1]))]
                        .into_iter().filter(|(_, value)| *value != 0.0).collect(),
                    vec![(1, scale(generator_seed[2]))]
                        .into_iter().filter(|(_, value)| *value != 0.0).collect(),
                ],
                Array2::from_shape_vec(
                    (2, 2),
                    constraint_seed[..4].iter().copied().map(scale).collect(),
                ).unwrap(),
                rhs_seed.iter().copied().map(scale).collect(),
                remainder_seed.iter().copied().map(scale).collect(),
            );
            let direction: Vec<_> = direction_seed.iter().copied().map(scale).collect();
            let zero = vec![0.0; 2];
            let baseline = domain.evaluate_dual(&direction, &zero).unwrap();
            let proposal = propose_coordinate_dual_unwired(
                &domain,
                &direction,
                CoordinateDualConfig {
                    sweeps,
                    ..CoordinateDualConfig::default()
                },
                None,
                None,
            ).unwrap();

            prop_assert!(proposal.bounds.lower >= baseline.lower);
            prop_assert!(proposal.bounds.upper <= baseline.upper);
            prop_assert!(proposal.lower_multipliers.iter().all(|v| v.is_finite() && *v >= 0.0));
            prop_assert!(proposal.upper_multipliers.iter().all(|v| v.is_finite() && *v >= 0.0));

            let exact_lower = exact_dual_endpoint(
                &domain,
                &direction,
                &proposal.lower_multipliers,
                true,
            );
            let exact_upper = exact_dual_endpoint(
                &domain,
                &direction,
                &proposal.upper_multipliers,
                false,
            );
            prop_assert!(rat(proposal.bounds.lower) <= exact_lower);
            prop_assert!(rat(proposal.bounds.upper) >= exact_upper);

            // Every 1/4-grid feasible alpha witness is contained by the two
            // independently selected certified endpoints.
            for a0_index in -4..=4 {
                for a1_index in -4..=4 {
                    let alpha = [f64::from(a0_index) / 4.0, f64::from(a1_index) / 4.0];
                    let feasible = (0..2).all(|row| {
                        domain.constraints()[[row, 0]] * alpha[0]
                            + domain.constraints()[[row, 1]] * alpha[1]
                            <= domain.rhs()[row]
                    });
                    if !feasible {
                        continue;
                    }
                    let mut value = domain.center().to_vec();
                    for (column, &alpha_value) in alpha.iter().enumerate() {
                        for (value_index, coefficient) in domain.generators()[column].entries() {
                            value[value_index] += coefficient * alpha_value;
                        }
                    }
                    let concrete = direction.iter().zip(value).map(|(&q, x)| q * x).sum::<f64>();
                    let box_charge = direction.iter().zip(domain.box_remainder())
                        .map(|(&q, &r)| q.abs() * r).sum::<f64>();
                    prop_assert!(proposal.bounds.lower <= concrete - box_charge);
                    prop_assert!(proposal.bounds.upper >= concrete + box_charge);
                }
            }
        }
    }
}
