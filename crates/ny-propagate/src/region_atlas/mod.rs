// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Checked ORIGINAL-coordinate affine evidence for a bounded region atlas.
//!
//! This foundation proves statements about the supplied affine inequalities,
//! not that those inequalities describe an arbitrary neural network. A network
//! adapter must establish that connection independently. Frozen generator
//! charts and nonlinear relations cannot be relabelled as original coordinates.
//! Every replay rehashes the actual premises, checks exact scope containment,
//! and recomputes arithmetic. Identity/box checks alone are never a proof.
//!
//! Sealed evidence has no serde constructor, conversion from a search score,
//! `Certified` constructor, or UNSAT-token door. Atlas lookup returns a checked
//! local statement; existing complete-cover and certificate authority remain
//! responsible for verification. Eviction cannot delete an unresolved branch.

mod numeric;
mod pool;
mod replay;

pub use pool::{AtlasEntry, AtlasStats, RegionAtlas};
pub use replay::{replay_conflict, replay_conic_transfer, replay_facet, replay_margin};

use crate::certify::scope::{CertificateScope, Digest, ScopeRefusal};
use crate::certify::work::{PersistentReservation, WorkRefusal};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

/// An original-coordinate premise `coefficients · x <= upper`.
/// Borrowed original data is revalidated and rehashed during each replay.
#[derive(Debug, Clone, Copy)]
pub struct AffineInequality<'a> {
    pub coefficients: &'a [f64],
    pub upper: f64,
}

/// A scalar affine objective, with the constant as a homogeneous coordinate.
/// The associated unsafe query is `coefficients · x + constant <= 0`.
#[derive(Debug, Clone, Copy)]
pub struct AffineObjective<'a> {
    pub coefficients: &'a [f64],
    pub constant: f64,
}

impl AffineObjective<'_> {
    /// Bind the direction, constant and non-strict unsafe relation. This hash
    /// is an identity only; replay still validates and evaluates every float.
    pub fn digest(&self) -> Digest {
        row_digest(
            b"ny-atlas/objective-le-zero/v1",
            self.coefficients,
            self.constant,
        )
    }
}

/// Scope and premises over the SAME original coordinates. Premise order is
/// bound as well: deleting or changing an antecedent requires fresh replay.
#[derive(Debug, Clone, Copy)]
pub struct AffineGeometry<'a> {
    pub scope: &'a CertificateScope,
    pub inequalities: &'a [AffineInequality<'a>],
}

#[derive(Debug, Clone, Copy)]
pub struct ReplayOptions {
    pub max_dimensions: usize,
    pub max_inequalities: usize,
    pub max_terms: usize,
}

impl Default for ReplayOptions {
    fn default() -> Self {
        Self {
            max_dimensions: 4096,
            max_inequalities: 512,
            max_terms: 250_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtlasRefusal {
    Scope(ScopeRefusal),
    Budget(WorkRefusal),
    ResourceLimit,
    Shape,
    NonFinite,
    UnsupportedSemantics,
    InvalidMultipliers,
    PropertyMismatch,
    ScopeMismatch,
    NoContradiction,
    InvalidFacet,
    Allocation,
    Overflow,
}

impl From<ScopeRefusal> for AtlasRefusal {
    fn from(value: ScopeRefusal) -> Self {
        Self::Scope(value)
    }
}
impl From<WorkRefusal> for AtlasRefusal {
    fn from(value: WorkRefusal) -> Self {
        Self::Budget(value)
    }
}

#[derive(Debug)]
struct CommonProof {
    scope: CertificateScope,
    premises: Digest,
    reservation: PersistentReservation,
}

impl CommonProof {
    fn covers(&self, geometry: AffineGeometry<'_>, premises: Digest) -> Result<bool, AtlasRefusal> {
        self.scope.validate(usize::MAX)?;
        geometry.scope.validate(usize::MAX)?;
        Ok(self.premises == premises
            && self.scope.identity() == geometry.scope.identity()
            && self.scope.region().contains(geometry.scope.region())?)
    }
}

/// Replayed emptiness of the original-coordinate box plus exact premises.
/// Its property-independent conclusion can transfer across objective queries.
#[derive(Debug, Clone)]
pub struct ScopeGeometryConflict(Arc<ConflictProof>);

#[derive(Debug)]
struct ConflictProof {
    common: CommonProof,
    slack_lower: f64,
}

impl ScopeGeometryConflict {
    pub fn scope(&self) -> &CertificateScope {
        &self.0.common.scope
    }
    pub fn slack_lower(&self) -> f64 {
        self.0.slack_lower
    }
    pub fn reserved_bytes(&self) -> usize {
        self.0.common.reservation.bytes()
    }
}

/// Checked affine direction lower bound on one region and exact premise set.
/// This is not an objective-independent emptiness statement.
#[derive(Clone)]
pub struct RegionMargin(Arc<MarginProof>);

#[derive(Debug)]
struct MarginProof {
    common: CommonProof,
    objective: Digest,
    coefficients: Vec<f64>,
    constant: f64,
    lower: f64,
    parents: Vec<(RegionMargin, f64)>,
    depth: usize,
}

impl std::fmt::Debug for RegionMargin {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RegionMargin")
            .field("objective", &self.0.objective)
            .field("lower", &self.0.lower)
            .field("dependencies", &self.0.parents.len())
            .field("depth", &self.0.depth)
            .finish()
    }
}

impl RegionMargin {
    pub fn scope(&self) -> &CertificateScope {
        &self.0.common.scope
    }
    pub fn lower(&self) -> f64 {
        self.0.lower
    }
    pub fn objective(&self) -> AffineObjective<'_> {
        AffineObjective {
            coefficients: &self.0.coefficients,
            constant: self.0.constant,
        }
    }
    pub fn reserved_bytes(&self) -> usize {
        self.0.common.reservation.bytes()
    }
    pub fn dependency_count(&self) -> usize {
        self.0.parents.len()
    }
    /// Strict positivity excludes this bound's `objective <= 0` query only.
    pub fn exclusion(&self) -> Option<ObjectiveExclusion> {
        (self.0.lower > 0.0).then(|| ObjectiveExclusion(self.clone()))
    }
}

/// A property-relative exclusion. No API converts it to a geometry conflict.
#[derive(Debug, Clone)]
pub struct ObjectiveExclusion(RegionMargin);

impl ObjectiveExclusion {
    pub fn margin(&self) -> &RegionMargin {
        &self.0
    }
}

/// A checked affine inequality, reusable only under its original premises and
/// region. A nonlinear facet needs an additional complete operator proof.
#[derive(Debug, Clone)]
pub struct ReusableFacet(Arc<FacetProof>);

#[derive(Debug)]
struct FacetProof {
    common: CommonProof,
    digest: Digest,
    coefficients: Vec<f64>,
    upper: f64,
}

impl ReusableFacet {
    pub fn scope(&self) -> &CertificateScope {
        &self.0.common.scope
    }
    pub fn inequality(&self) -> AffineInequality<'_> {
        AffineInequality {
            coefficients: &self.0.coefficients,
            upper: self.0.upper,
        }
    }
    pub fn reserved_bytes(&self) -> usize {
        self.0.common.reservation.bytes()
    }
}

fn row_digest(domain: &[u8], coefficients: &[f64], constant: f64) -> Digest {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update((coefficients.len() as u64).to_le_bytes());
    for &value in coefficients.iter().chain(std::iter::once(&constant)) {
        hash.update(float_bits(value).to_le_bytes());
    }
    Digest(hash.finalize().into())
}

fn float_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

#[cfg(test)]
mod tests;
