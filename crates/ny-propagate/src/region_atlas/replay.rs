// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Replay supplied nonnegative multipliers against original affine premises.
//! Nonzero coefficient residuals are bounded on the original box; rounded
//! cancellation is never assumed to be an exact Farkas identity.

use super::numeric::Interval;
use super::*;
use crate::certify::scope::NumericalSemantics;
use crate::certify::work::{LeaseClass, WorkBudget, WorkLease, WorkRequest};

pub(super) fn shape(
    geometry: AffineGeometry<'_>,
    options: ReplayOptions,
) -> Result<(usize, usize), AtlasRefusal> {
    let dimensions = geometry.scope.region().lo().len();
    let terms = dimensions
        .checked_mul(geometry.inequalities.len())
        .ok_or(AtlasRefusal::Overflow)?;
    if options.max_dimensions > 65_536
        || options.max_inequalities > 4096
        || options.max_terms > 1_000_000
        || dimensions == 0
        || dimensions > options.max_dimensions
        || geometry.inequalities.len() > options.max_inequalities
        || terms > options.max_terms
    {
        return Err(AtlasRefusal::ResourceLimit);
    }
    Ok((dimensions, terms))
}

pub(super) fn validate_and_hash(
    geometry: AffineGeometry<'_>,
    dimensions: usize,
    lease: &WorkLease,
) -> Result<Digest, AtlasRefusal> {
    lease.checkpoint()?;
    geometry.scope.validate(dimensions)?;
    if geometry.scope.identity().numerical != NumericalSemantics::ExactReal {
        // A contract digest alone is not an execution-error enclosure.
        return Err(AtlasRefusal::UnsupportedSemantics);
    }
    let mut hash = Sha256::new();
    hash.update(b"ny-atlas/original-affine-premises/v1");
    hash.update((dimensions as u64).to_le_bytes());
    hash.update((geometry.inequalities.len() as u64).to_le_bytes());
    for row in geometry.inequalities {
        lease.checkpoint()?;
        validate_row(row.coefficients, row.upper, dimensions, lease)?;
        for (j, &value) in row.coefficients.iter().enumerate() {
            if j & 255 == 0 {
                lease.checkpoint()?;
            }
            hash.update(float_bits(value).to_le_bytes());
        }
        hash.update(float_bits(row.upper).to_le_bytes());
    }
    lease.checkpoint()?;
    Ok(Digest(hash.finalize().into()))
}

fn validate_row(
    coefficients: &[f64],
    constant: f64,
    dimensions: usize,
    lease: &WorkLease,
) -> Result<(), AtlasRefusal> {
    if coefficients.len() != dimensions {
        return Err(AtlasRefusal::Shape);
    }
    if !constant.is_finite() {
        return Err(AtlasRefusal::NonFinite);
    }
    for (j, &value) in coefficients.iter().enumerate() {
        if j & 255 == 0 {
            lease.checkpoint()?;
        }
        if !value.is_finite() {
            return Err(AtlasRefusal::NonFinite);
        }
    }
    Ok(())
}

pub(super) fn checked_direction_digest(
    geometry: AffineGeometry<'_>,
    coefficients: &[f64],
    constant: f64,
    domain: &[u8],
    options: ReplayOptions,
    budget: &WorkBudget,
) -> Result<Digest, AtlasRefusal> {
    let (dimensions, _) = shape(geometry, options)?;
    if coefficients.len() != dimensions {
        return Err(AtlasRefusal::Shape);
    }
    let mut lease = budget.lease_for(
        LeaseClass::Replay,
        WorkRequest {
            scratch_bytes: 1024,
            work: (dimensions as u64)
                .checked_mul(4)
                .ok_or(AtlasRefusal::Overflow)?,
            ..WorkRequest::default()
        },
    )?;
    validate_row(coefficients, constant, dimensions, &lease)?;
    let result = row_digest(domain, coefficients, constant);
    lease.complete()?;
    Ok(result)
}

fn prepare(
    geometry: AffineGeometry<'_>,
    multipliers: &[f64],
    options: ReplayOptions,
    budget: &WorkBudget,
    retain_direction: bool,
) -> Result<(WorkLease, Digest, Vec<Interval>, Interval), AtlasRefusal> {
    let (dimensions, terms) = shape(geometry, options)?;
    if multipliers.len() != geometry.inequalities.len() {
        return Err(AtlasRefusal::Shape);
    }
    // Only one D-vector of coefficient intervals, plus hash/stack scratch.
    // Proof copies include original box endpoints and optionally one direction.
    let scratch_bytes = dimensions
        .checked_mul(16)
        .and_then(|n| n.checked_add(4096))
        .ok_or(AtlasRefusal::Overflow)?;
    let persistent_bytes = dimensions
        .checked_mul(if retain_direction { 24 } else { 16 })
        .and_then(|n| n.checked_add(2048))
        .ok_or(AtlasRefusal::Overflow)?;
    let work = terms
        .checked_mul(12)
        .and_then(|n| n.checked_add(dimensions.checked_mul(32)?))
        .and_then(|n| n.checked_add(geometry.inequalities.len()))
        .ok_or(AtlasRefusal::Overflow)?;
    let lease = budget.lease_for(
        LeaseClass::Replay,
        WorkRequest {
            scratch_bytes,
            persistent_bytes,
            work: u64::try_from(work).map_err(|_| AtlasRefusal::Overflow)?,
            proposals: 1,
        },
    )?;
    let premises = validate_and_hash(geometry, dimensions, &lease)?;
    for &value in multipliers {
        lease.checkpoint()?;
        if !value.is_finite() || value < 0.0 {
            return Err(AtlasRefusal::InvalidMultipliers);
        }
    }
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(dimensions)
        .map_err(|_| AtlasRefusal::Allocation)?;
    coefficients.resize(dimensions, Interval::ZERO);
    let mut upper = Interval::ZERO;
    for (row, &multiplier) in geometry.inequalities.iter().zip(multipliers) {
        lease.checkpoint()?;
        if multiplier == 0.0 {
            continue;
        }
        let multiplier = Interval::point(multiplier);
        upper = upper.add(multiplier.mul(Interval::point(row.upper))?)?;
        for (j, (&coefficient, sum)) in row.coefficients.iter().zip(&mut coefficients).enumerate() {
            if j & 255 == 0 {
                lease.checkpoint()?;
            }
            *sum = sum.add(multiplier.mul(Interval::point(coefficient))?)?;
        }
    }
    lease.checkpoint()?;
    Ok((lease, premises, coefficients, upper))
}

fn box_range(
    coefficients: &[Interval],
    geometry: AffineGeometry<'_>,
    lease: &WorkLease,
) -> Result<Interval, AtlasRefusal> {
    let mut range = Interval::ZERO;
    for (j, ((coefficient, &lo), &hi)) in coefficients
        .iter()
        .zip(geometry.scope.region().lo())
        .zip(geometry.scope.region().hi())
        .enumerate()
    {
        if j & 255 == 0 {
            lease.checkpoint()?;
        }
        range = range.add(coefficient.mul(Interval { lo, hi })?)?;
    }
    Ok(range)
}

fn common(
    geometry: AffineGeometry<'_>,
    premises: Digest,
    lease: WorkLease,
) -> Result<CommonProof, AtlasRefusal> {
    // Clone only AFTER the lease; endpoints and immutable proof metadata are
    // charged until the final shared proof owner drops its reservation.
    lease.checkpoint()?;
    let scope = geometry.scope.clone();
    let reservation = lease.commit_persistent()?;
    Ok(CommonProof {
        scope,
        premises,
        reservation,
    })
}

/// Check `min_box(sum lambda_i*a_i*x) > sum lambda_i*b_i` for lambda>=0.
/// Residual coefficients are retained and bounded; exact zero is not assumed.
pub fn replay_conflict(
    geometry: AffineGeometry<'_>,
    multipliers: &[f64],
    options: ReplayOptions,
    budget: &WorkBudget,
) -> Result<ScopeGeometryConflict, AtlasRefusal> {
    let (lease, premises, coefficients, upper) =
        prepare(geometry, multipliers, options, budget, false)?;
    let range = box_range(&coefficients, geometry, &lease)?;
    let slack = range.sub(upper)?.lo;
    if slack <= 0.0 {
        return Err(AtlasRefusal::NoContradiction);
    }
    drop(coefficients);
    Ok(ScopeGeometryConflict(Arc::new(ConflictProof {
        common: common(geometry, premises, lease)?,
        slack_lower: slack,
    })))
}

/// Certify q*x+c >= min_box((q+sum lambda*a)*x)+c-sum lambda*b.
/// The requested property's digest must bind this exact direction and unsafe
/// relation. A positive result cannot exclude a different class's objective.
pub fn replay_margin(
    geometry: AffineGeometry<'_>,
    objective: AffineObjective<'_>,
    multipliers: &[f64],
    options: ReplayOptions,
    budget: &WorkBudget,
) -> Result<RegionMargin, AtlasRefusal> {
    let (lease, premises, mut coefficients, upper) =
        prepare(geometry, multipliers, options, budget, true)?;
    validate_row(
        objective.coefficients,
        objective.constant,
        coefficients.len(),
        &lease,
    )?;
    let digest = objective.digest();
    if geometry.scope.property() != digest {
        return Err(AtlasRefusal::PropertyMismatch);
    }
    for (j, (sum, &value)) in coefficients
        .iter_mut()
        .zip(objective.coefficients)
        .enumerate()
    {
        if j & 255 == 0 {
            lease.checkpoint()?;
        }
        *sum = sum.add(Interval::point(value))?;
    }
    let lower = box_range(&coefficients, geometry, &lease)?
        .add(Interval::point(objective.constant))?
        .sub(upper)?
        .lo;
    lease.checkpoint()?;
    let mut retained = Vec::new();
    retained
        .try_reserve_exact(objective.coefficients.len())
        .map_err(|_| AtlasRefusal::Allocation)?;
    retained.extend_from_slice(objective.coefficients);
    drop(coefficients);
    Ok(RegionMargin(Arc::new(MarginProof {
        common: common(geometry, premises, lease)?,
        objective: digest,
        coefficients: retained,
        constant: objective.constant,
        lower,
        parents: Vec::new(),
        depth: 0,
    })))
}

/// Check the facet via `sum lambda*b + max_box((facet-sum lambda*a)*x)`.
/// The multipliers are proposals; their residual decomposition is replayed.
pub fn replay_facet(
    geometry: AffineGeometry<'_>,
    facet: AffineInequality<'_>,
    multipliers: &[f64],
    options: ReplayOptions,
    budget: &WorkBudget,
) -> Result<ReusableFacet, AtlasRefusal> {
    let (lease, premises, mut coefficients, upper) =
        prepare(geometry, multipliers, options, budget, true)?;
    validate_row(facet.coefficients, facet.upper, coefficients.len(), &lease)?;
    for (j, (sum, &value)) in coefficients.iter_mut().zip(facet.coefficients).enumerate() {
        if j & 255 == 0 {
            lease.checkpoint()?;
        }
        *sum = Interval::point(value).sub(*sum)?;
    }
    if upper.add(box_range(&coefficients, geometry, &lease)?)?.hi > facet.upper {
        return Err(AtlasRefusal::InvalidFacet);
    }
    lease.checkpoint()?;
    let digest = row_digest(b"ny-atlas/facet-le/v1", facet.coefficients, facet.upper);
    let mut retained = Vec::new();
    retained
        .try_reserve_exact(facet.coefficients.len())
        .map_err(|_| AtlasRefusal::Allocation)?;
    retained.extend_from_slice(facet.coefficients);
    drop(coefficients);
    Ok(ReusableFacet(Arc::new(FacetProof {
        common: common(geometry, premises, lease)?,
        digest,
        coefficients: retained,
        upper: facet.upper,
    })))
}

/// Nonnegative cone transfer with an independently bounded affine residual.
/// Each source proof must cover this ORIGINAL-coordinate region and premise
/// set. Direction/constant decomposition is recomputed with outward intervals;
/// approximate coefficient equality never substitutes for residual discharge.
pub fn replay_conic_transfer(
    geometry: AffineGeometry<'_>,
    objective: AffineObjective<'_>,
    contributors: &[(&RegionMargin, f64)],
    options: ReplayOptions,
    budget: &WorkBudget,
) -> Result<RegionMargin, AtlasRefusal> {
    let (dimensions, terms) = shape(geometry, options)?;
    if contributors.is_empty() || contributors.len() > 128 {
        return Err(AtlasRefusal::ResourceLimit);
    }
    let scratch_bytes = dimensions
        .checked_mul(16)
        .and_then(|n| n.checked_add(4096))
        .ok_or(AtlasRefusal::Overflow)?;
    let persistent_bytes = dimensions
        .checked_mul(24)
        .and_then(|n| {
            n.checked_add(
                contributors
                    .len()
                    .checked_mul(size_of::<(RegionMargin, f64)>())?,
            )
        })
        .and_then(|n| n.checked_add(2048))
        .ok_or(AtlasRefusal::Overflow)?;
    let work = dimensions
        .checked_mul(contributors.len())
        .and_then(|n| n.checked_mul(16))
        .and_then(|n| n.checked_add(terms.checked_mul(4)?))
        .and_then(|n| n.checked_add(dimensions.checked_mul(64)?))
        .ok_or(AtlasRefusal::Overflow)?;
    let lease = budget.lease_for(
        LeaseClass::Replay,
        WorkRequest {
            scratch_bytes,
            persistent_bytes,
            work: u64::try_from(work).map_err(|_| AtlasRefusal::Overflow)?,
            proposals: 1,
        },
    )?;
    let premises = validate_and_hash(geometry, dimensions, &lease)?;
    validate_row(
        objective.coefficients,
        objective.constant,
        dimensions,
        &lease,
    )?;
    let digest = objective.digest();
    if geometry.scope.property() != digest {
        return Err(AtlasRefusal::PropertyMismatch);
    }
    let mut residual = Vec::new();
    residual
        .try_reserve_exact(dimensions)
        .map_err(|_| AtlasRefusal::Allocation)?;
    residual.extend(objective.coefficients.iter().copied().map(Interval::point));
    let mut constant = Interval::point(objective.constant);
    let mut lower = Interval::ZERO;
    let mut depth = 0usize;
    for &(proof, weight) in contributors {
        lease.checkpoint()?;
        if !weight.is_finite() || weight < 0.0 {
            return Err(AtlasRefusal::InvalidMultipliers);
        }
        // Validate scope even for zero weights: unbound proof handles do not
        // silently become accepted dependencies.
        if !proof.0.common.covers(geometry, premises)? {
            return Err(AtlasRefusal::ScopeMismatch);
        }
        depth = depth.max(proof.0.depth.checked_add(1).ok_or(AtlasRefusal::Overflow)?);
        if depth > 64 {
            return Err(AtlasRefusal::ResourceLimit);
        }
        let weight = Interval::point(weight);
        lower = lower.add(weight.mul(Interval::point(proof.0.lower))?)?;
        constant = constant.sub(weight.mul(Interval::point(proof.0.constant))?)?;
        for (j, (residual, &coefficient)) in
            residual.iter_mut().zip(&proof.0.coefficients).enumerate()
        {
            if j & 255 == 0 {
                lease.checkpoint()?;
            }
            *residual = residual.sub(weight.mul(Interval::point(coefficient))?)?;
        }
    }
    let lower = lower
        .add(box_range(&residual, geometry, &lease)?)?
        .add(constant)?
        .lo;
    let mut retained = Vec::new();
    retained
        .try_reserve_exact(dimensions)
        .map_err(|_| AtlasRefusal::Allocation)?;
    retained.extend_from_slice(objective.coefficients);
    drop(residual);
    let mut parents = Vec::new();
    parents
        .try_reserve_exact(contributors.len())
        .map_err(|_| AtlasRefusal::Allocation)?;
    parents.extend(
        contributors
            .iter()
            .map(|&(proof, weight)| (proof.clone(), weight)),
    );
    Ok(RegionMargin(Arc::new(MarginProof {
        common: common(geometry, premises, lease)?,
        objective: digest,
        coefficients: retained,
        constant: objective.constant,
        lower,
        parents,
        depth,
    })))
}
