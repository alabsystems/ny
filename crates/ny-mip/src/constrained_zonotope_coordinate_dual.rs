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
    let constraint_count = domain.constraint_count();
    let lower_zero = try_zero_multipliers(constraint_count, "lower")?;
    let upper_zero = try_zero_multipliers(constraint_count, "upper")?;

    // This is intentionally first and mandatory.  In particular, invalid
    // direction data or an unsupported floating-point environment must not be
    // hidden behind a proposer resource fallback.
    let baseline = domain.evaluate_dual(direction, &lower_zero)?;

    let Some(plan) = SearchPlan::checked(domain, config) else {
        return Ok(baseline_proposal(baseline, lower_zero, upper_zero));
    };
    if !valid_warm_start(lower_warm_start, constraint_count)
        || !valid_warm_start(upper_warm_start, constraint_count)
    {
        return Ok(baseline_proposal(baseline, lower_zero, upper_zero));
    }

    let Ok(projected_generators) = project_generators(domain, direction, plan.alpha_dim) else {
        return Ok(baseline_proposal(baseline, lower_zero, upper_zero));
    };

    let lower_seed = lower_warm_start.unwrap_or(&lower_zero);
    let upper_seed = upper_warm_start.unwrap_or(&upper_zero);
    let lower_candidate =
        coordinate_candidate(domain, &projected_generators, 1.0, lower_seed, plan);
    let upper_candidate =
        coordinate_candidate(domain, &projected_generators, -1.0, upper_seed, plan);

    let mut proposal = baseline_proposal(baseline, lower_zero, upper_zero);

    if let Ok(candidate) = lower_candidate {
        // The proposer has no authority.  Swallow candidate-evaluation failure
        // and retain the already-certified zero baseline.
        if let Ok(candidate_bounds) = domain.evaluate_dual(direction, &candidate) {
            if candidate_bounds.lower > baseline.lower {
                proposal.bounds.lower = candidate_bounds.lower;
                proposal.lower_multipliers = candidate;
                proposal.lower_improved = true;
            }
        }
    }

    if let Ok(candidate) = upper_candidate {
        if let Ok(candidate_bounds) = domain.evaluate_dual(direction, &candidate) {
            if candidate_bounds.upper < baseline.upper {
                proposal.bounds.upper = candidate_bounds.upper;
                proposal.upper_multipliers = candidate;
                proposal.upper_improved = true;
            }
        }
    }

    Ok(proposal)
}

fn try_zero_multipliers(
    constraint_count: usize,
    direction: &'static str,
) -> Result<Vec<f64>, CoordinateDualProposerError> {
    let mut multipliers = Vec::new();
    multipliers
        .try_reserve_exact(constraint_count)
        .map_err(|_| CoordinateDualProposerError::BaselineAllocation { direction })?;
    multipliers.resize(constraint_count, 0.0);
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

fn valid_warm_start(warm_start: Option<&[f64]>, constraint_count: usize) -> bool {
    let Some(warm_start) = warm_start else {
        return true;
    };
    warm_start.len() == constraint_count
        && warm_start
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
}

#[derive(Clone, Copy, Debug)]
struct SearchPlan {
    sweeps: usize,
    alpha_dim: usize,
    breakpoint_capacity: usize,
}

impl SearchPlan {
    fn checked(domain: &ConstrainedZonotope64, config: CoordinateDualConfig) -> Option<Self> {
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
            return None;
        }

        let constraint_count = domain.constraint_count();
        let alpha_dim = domain.alpha_dim();
        if constraint_count > limits.max_constraints || alpha_dim > limits.max_alpha_dim {
            return None;
        }
        let breakpoint_capacity = alpha_dim.checked_add(1)?;
        if breakpoint_capacity > limits.max_breakpoints {
            return None;
        }

        let mut generator_nonzeros = 0_usize;
        for generator in domain.generators() {
            generator_nonzeros = generator_nonzeros.checked_add(generator.nnz())?;
        }

        let total_work = checked_search_work(
            generator_nonzeros,
            constraint_count,
            alpha_dim,
            usize::from(config.sweeps),
            breakpoint_capacity,
        )?;
        if total_work > limits.max_work {
            return None;
        }

        Some(Self {
            sweeps: usize::from(config.sweeps),
            alpha_dim,
            breakpoint_capacity,
        })
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

fn project_generators(
    domain: &ConstrainedZonotope64,
    direction: &[f64],
    alpha_dim: usize,
) -> Result<Vec<f64>, CandidateFailure> {
    let mut projected = Vec::new();
    projected
        .try_reserve_exact(alpha_dim)
        .map_err(|_| CandidateFailure::Allocation)?;
    projected.resize(alpha_dim, 0.0);

    for (column, generator) in domain.generators().iter().enumerate() {
        let mut sum = 0.0;
        for (value_index, coefficient) in generator.entries() {
            let product = finite_mul(direction[value_index], coefficient)?;
            sum = finite_add(sum, product)?;
        }
        projected[column] = sum;
    }
    Ok(projected)
}

fn coordinate_candidate(
    domain: &ConstrainedZonotope64,
    projected_generators: &[f64],
    generator_sign: f64,
    warm_start: &[f64],
    plan: SearchPlan,
) -> Result<Vec<f64>, CandidateFailure> {
    let constraint_count = domain.constraint_count();
    let constraints = domain.constraints();
    let mut multipliers = copy_finite_vector(warm_start)?;
    let mut projected = Vec::new();
    projected
        .try_reserve_exact(plan.alpha_dim)
        .map_err(|_| CandidateFailure::Allocation)?;
    for &value in projected_generators {
        projected.push(finite_mul(generator_sign, value)?);
    }

    for row in 0..constraint_count {
        let multiplier = multipliers[row];
        if multiplier == 0.0 {
            continue;
        }
        for column in 0..plan.alpha_dim {
            let contribution = finite_mul(constraints[[row, column]], multiplier)?;
            projected[column] = finite_add(projected[column], contribution)?;
        }
    }

    let mut base = Vec::new();
    base.try_reserve_exact(plan.alpha_dim)
        .map_err(|_| CandidateFailure::Allocation)?;
    base.resize(plan.alpha_dim, 0.0);
    let mut breakpoints = Vec::new();
    breakpoints
        .try_reserve_exact(plan.breakpoint_capacity)
        .map_err(|_| CandidateFailure::Allocation)?;

    for _ in 0..plan.sweeps {
        for row in 0..constraint_count {
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
                        return Err(CandidateFailure::NonFiniteArithmetic);
                    }
                    let slope_drop = finite_mul(2.0, coefficient.abs())?;
                    if breakpoints.len() >= plan.breakpoint_capacity {
                        return Err(CandidateFailure::Allocation);
                    }
                    breakpoints.push(Breakpoint {
                        location,
                        slope_drop,
                    });
                }
            }

            // This in-place sort never invokes an infallible hidden scratch
            // allocation after the checked reserve.  The total comparator
            // keeps the groups deterministic even though stability is unused.
            breakpoints.sort_unstable_by(|left, right| {
                let order = left.location.total_cmp(&right.location);
                if order == Ordering::Equal {
                    left.slope_drop.total_cmp(&right.slope_drop)
                } else {
                    order
                }
            });

            let replacement = if right_slope <= 0.0 {
                0.0
            } else {
                first_nonpositive_slope_breakpoint(&breakpoints, right_slope)?
            };
            multipliers[row] = replacement;
            for column in 0..plan.alpha_dim {
                let contribution = finite_mul(constraints[[row, column]], replacement)?;
                projected[column] = finite_add(base[column], contribution)?;
            }
        }
    }

    if multipliers
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(CandidateFailure::NonFiniteArithmetic);
    }
    Ok(multipliers)
}

fn first_nonpositive_slope_breakpoint(
    breakpoints: &[Breakpoint],
    mut slope: f64,
) -> Result<f64, CandidateFailure> {
    let mut index = 1;
    while index < breakpoints.len() {
        let location = breakpoints[index].location;
        let mut total_drop = 0.0;
        while index < breakpoints.len() && breakpoints[index].location == location {
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
    Err(CandidateFailure::UnboundedCoordinate)
}

fn copy_finite_vector(source: &[f64]) -> Result<Vec<f64>, CandidateFailure> {
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(source.len())
        .map_err(|_| CandidateFailure::Allocation)?;
    copied.extend_from_slice(source);
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
