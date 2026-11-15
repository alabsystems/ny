// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Identity and region checks shared by reusable certificates.
//!
//! A matching scope is necessary for proof reuse; it is never arithmetic
//! evidence and cannot mint a `Certified` value or an UNSAT token. Deserialized
//! scopes are proposals and are validated again by every public comparison.

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};

/// Current serialized scope format. Unknown versions fail closed.
pub const SCOPE_SCHEMA_VERSION: u32 = 1;

/// Cryptographic identity of original data, not a proof of that data's meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    /// Length-delimited, domain-separated hash. Large consumers should hash
    /// their original tensors incrementally rather than copy them to a buffer.
    #[must_use]
    pub fn of_bytes(domain: &[u8], bytes: &[u8]) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"ny-certificate-scope/v1");
        hash.update((domain.len() as u64).to_le_bytes());
        hash.update(domain);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
        Self(hash.finalize().into())
    }
}

/// The interpretation that a checker actually proves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum NumericalSemantics {
    /// Exact real interpretation of the serialized finite coefficients.
    ExactReal,
    /// Execution is enclosed under a separately checked error contract.
    /// Merely supplying this digest does not validate that contract.
    EnclosedExecution { contract: Digest },
}

/// Every identity that must match before a certificate can be reused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeIdentity {
    pub schema_version: u32,
    pub model: Digest,
    pub lowering: Digest,
    pub coordinates: Digest,
    pub assumptions: Digest,
    pub numerical: NumericalSemantics,
}

/// Refusing scope reuse cannot change an already checked result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeRefusal {
    Schema,
    Dimension,
    NonFinite,
    ReversedInterval,
    ResourceLimit,
    DuplicateProof,
    MissingDependency,
    CyclicDependency,
}

/// An original-coordinate box. Fields are private; JSON remains untrusted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxRegion {
    lo: Vec<f64>,
    hi: Vec<f64>,
}

impl BoxRegion {
    pub fn new(lo: Vec<f64>, hi: Vec<f64>) -> Result<Self, ScopeRefusal> {
        let result = Self { lo, hi };
        result.validate(usize::MAX)?;
        Ok(result)
    }

    /// Validate before any subsequent allocation proportional to the box.
    pub fn validate(&self, max_dimensions: usize) -> Result<(), ScopeRefusal> {
        if self.lo.is_empty() || self.lo.len() != self.hi.len() {
            return Err(ScopeRefusal::Dimension);
        }
        if self.lo.len() > max_dimensions {
            return Err(ScopeRefusal::ResourceLimit);
        }
        for (&lo, &hi) in self.lo.iter().zip(&self.hi) {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(ScopeRefusal::NonFinite);
            }
            if lo > hi {
                return Err(ScopeRefusal::ReversedInterval);
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn lo(&self) -> &[f64] {
        &self.lo
    }

    #[must_use]
    pub fn hi(&self) -> &[f64] {
        &self.hi
    }

    /// Exact endpoint containment. No tolerance can enlarge a proof's region.
    pub fn contains(&self, child: &Self) -> Result<bool, ScopeRefusal> {
        self.validate(usize::MAX)?;
        child.validate(usize::MAX)?;
        if self.lo.len() != child.lo.len() {
            return Err(ScopeRefusal::Dimension);
        }
        Ok(self
            .lo
            .iter()
            .zip(&self.hi)
            .zip(child.lo.iter().zip(&child.hi))
            .all(|((&lo, &hi), (&child_lo, &child_hi))| lo <= child_lo && child_hi <= hi))
    }
}

/// An arithmetic proof may be used only on this identity, property, and region.
/// The property digest includes directions, constants, target, and strictness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificateScope {
    identity: ScopeIdentity,
    region: BoxRegion,
    property: Digest,
}

impl CertificateScope {
    pub fn new(
        identity: ScopeIdentity,
        region: BoxRegion,
        property: Digest,
    ) -> Result<Self, ScopeRefusal> {
        let result = Self {
            identity,
            region,
            property,
        };
        result.validate(usize::MAX)?;
        Ok(result)
    }

    pub fn validate(&self, max_dimensions: usize) -> Result<(), ScopeRefusal> {
        if self.identity.schema_version != SCOPE_SCHEMA_VERSION {
            return Err(ScopeRefusal::Schema);
        }
        self.region.validate(max_dimensions)
    }

    #[must_use]
    pub fn identity(&self) -> &ScopeIdentity {
        &self.identity
    }

    #[must_use]
    pub fn region(&self) -> &BoxRegion {
        &self.region
    }

    #[must_use]
    pub fn property(&self) -> Digest {
        self.property
    }

    /// Match all semantics and the requested property before checking region
    /// refinement. A different objective is not an infeasibility statement.
    pub fn covers(&self, request: &Self) -> Result<bool, ScopeRefusal> {
        self.validate(usize::MAX)?;
        request.validate(usize::MAX)?;
        if self.identity != request.identity || self.property != request.property {
            return Ok(false);
        }
        self.region.contains(&request.region)
    }
}

/// Untrusted proof-dependency record. Graph validation does not replay its
/// theorem; consumers must check each node's actual evidence independently.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofDependency {
    pub id: Digest,
    pub depends_on: Vec<Digest>,
}

/// Reject duplicate IDs, missing ancestors, and cycles before a replay walk.
/// Iterative traversal avoids a stack overflow on hostile serialized chains.
pub fn validate_proof_dag(
    nodes: &[ProofDependency],
    max_nodes: usize,
    max_edges: usize,
) -> Result<(), ScopeRefusal> {
    if nodes.len() > max_nodes {
        return Err(ScopeRefusal::ResourceLimit);
    }
    let edge_count = nodes
        .iter()
        .try_fold(0usize, |sum, node| sum.checked_add(node.depends_on.len()))
        .ok_or(ScopeRefusal::ResourceLimit)?;
    if edge_count > max_edges {
        return Err(ScopeRefusal::ResourceLimit);
    }
    let mut ids = HashMap::with_capacity(nodes.len());
    for (index, node) in nodes.iter().enumerate() {
        if ids.insert(node.id, index).is_some() {
            return Err(ScopeRefusal::DuplicateProof);
        }
    }
    let mut waiting = vec![0usize; nodes.len()];
    let mut children = vec![Vec::new(); nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        let mut unique = HashSet::new();
        for dependency in &node.depends_on {
            let &parent = ids.get(dependency).ok_or(ScopeRefusal::MissingDependency)?;
            if !unique.insert(parent) {
                return Err(ScopeRefusal::DuplicateProof);
            }
            waiting[index] += 1;
            children[parent].push(index);
        }
    }
    let mut ready: VecDeque<_> = waiting
        .iter()
        .enumerate()
        .filter_map(|(index, &count)| (count == 0).then_some(index))
        .collect();
    let mut visited = 0usize;
    while let Some(parent) = ready.pop_front() {
        visited += 1;
        for &child in &children[parent] {
            waiting[child] -= 1;
            if waiting[child] == 0 {
                ready.push_back(child);
            }
        }
    }
    if visited != nodes.len() {
        return Err(ScopeRefusal::CyclicDependency);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(lo: f64, hi: f64) -> CertificateScope {
        let digest = Digest::of_bytes(b"test", b"model");
        CertificateScope::new(
            ScopeIdentity {
                schema_version: SCOPE_SCHEMA_VERSION,
                model: digest,
                lowering: digest,
                coordinates: digest,
                assumptions: digest,
                numerical: NumericalSemantics::ExactReal,
            },
            BoxRegion::new(vec![lo], vec![hi]).unwrap(),
            digest,
        )
        .unwrap()
    }

    #[test]
    fn scoped_reuse_requires_region_and_every_semantic_identity() {
        let parent = scope(-1.0, 1.0);
        assert_eq!(parent.covers(&scope(-0.5, 0.5)), Ok(true));
        assert_eq!(parent.covers(&scope(-1.0000000000000002, 0.5)), Ok(false));
        let mut child = scope(-0.5, 0.5);
        child.property = Digest::of_bytes(b"test", b"other-class");
        assert_eq!(parent.covers(&child), Ok(false));
        child = scope(-0.5, 0.5);
        child.identity.numerical = NumericalSemantics::EnclosedExecution {
            contract: child.property,
        };
        assert_eq!(parent.covers(&child), Ok(false));
        child = scope(-0.5, 0.5);
        child.identity.schema_version += 1;
        assert_eq!(parent.covers(&child), Err(ScopeRefusal::Schema));
    }

    #[test]
    fn deserialized_invalid_regions_never_authorize_reuse() {
        let mut proposal = serde_json::to_value(scope(-1.0, 1.0)).unwrap();
        proposal["region"]["lo"] = serde_json::json!([2.0]);
        let poisoned: CertificateScope = serde_json::from_value(proposal).unwrap();
        assert_eq!(
            poisoned.covers(&scope(0.0, 0.0)),
            Err(ScopeRefusal::ReversedInterval)
        );
        assert_eq!(
            BoxRegion::new(vec![f64::NAN], vec![1.0]),
            Err(ScopeRefusal::NonFinite)
        );
    }

    #[test]
    fn proof_dag_rejects_cycles_missing_and_duplicate_dependencies() {
        let a = Digest::of_bytes(b"test", b"a");
        let b = Digest::of_bytes(b"test", b"b");
        let root = ProofDependency {
            id: a,
            depends_on: vec![],
        };
        let leaf = ProofDependency {
            id: b,
            depends_on: vec![a],
        };
        assert_eq!(
            validate_proof_dag(&[leaf.clone(), root.clone()], 2, 1),
            Ok(())
        );
        assert_eq!(
            validate_proof_dag(&[leaf.clone()], 2, 1),
            Err(ScopeRefusal::MissingDependency)
        );
        assert_eq!(
            validate_proof_dag(&[root.clone(), root], 2, 1),
            Err(ScopeRefusal::DuplicateProof)
        );
        let cyclic = ProofDependency {
            id: a,
            depends_on: vec![b],
        };
        assert_eq!(
            validate_proof_dag(&[cyclic, leaf], 2, 2),
            Err(ScopeRefusal::CyclicDependency)
        );
    }
    #[test]
    fn numerical_semantics_refuses_unrecognized_execution_claim_fields() {
        let contract = Digest::of_bytes(b"execution-contract", b"separate-binary64");
        let valid = NumericalSemantics::EnclosedExecution { contract };
        let mut json = serde_json::to_value(valid).unwrap();
        json["EnclosedExecution"]["unvalidated_kernel"] = serde_json::json!("quantized");
        assert!(serde_json::from_value::<NumericalSemantics>(json).is_err());
    }
}
