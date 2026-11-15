// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

struct ForwardOwner {
    facts: Vec<Fact>,
    original: Arc<ChainOwner>,
    positions: usize,
    input_digest: Digest,
    binding: OriginalChainBinding,
    reservation: PersistentReservation,
}

/// Checked original forward facts. Clones share the actual arrays/reservations
/// and original owners. A supplied input box is a local assumption, not an
/// independently certified upstream enclosure or a complete network token.
#[derive(Clone)]
pub struct OriginalChainForwardReport(Arc<ForwardOwner>);

impl OriginalChainForwardReport {
    #[cfg(test)]
    pub(super) fn retained_storage_observation(&self) -> (usize, usize, usize, usize) {
        (
            size_of::<ForwardOwner>(),
            size_of::<ChainOwner>(),
            self.0.facts.capacity(),
            size_of::<Fact>(),
        )
    }

    pub(super) fn finish(
        facts: Vec<Fact>,
        original: Arc<ChainOwner>,
        positions: usize,
        input_digest: Digest,
        binding: OriginalChainBinding,
        phase: OriginalAffinePhase,
    ) -> Result<Self> {
        if size_of::<ForwardOwner>() + 4 * size_of::<usize>() > META {
            return Err(invalid("forward report metadata capacity"));
        }
        // The caller already retained META + exact Vec capacity before fill.
        let reservation = phase.commit()?;
        Ok(Self(Arc::new(ForwardOwner {
            facts,
            original,
            positions,
            input_digest,
            binding,
            reservation,
        })))
    }
    pub fn bounds(&self) -> &BoundedTensor {
        self.0
            .facts
            .last()
            .expect("private constructor requires a terminal affine")
            .bounds()
    }
    pub fn scope(&self) -> &ScopeIdentity {
        &self.0.original.scope
    }
    pub fn binding(&self) -> OriginalChainBinding {
        self.0.binding
    }
    pub fn num_nodes(&self) -> usize {
        self.0.facts.len()
    }
    /// Sum of this report's leased capacities, including retained forward facts
    /// and chain/per-node preparation metadata. Original coefficient owner
    /// capacities are reported separately by `original_owner_bytes`.
    /// Shared clones do not create additive process allocations or RSS claims.
    pub fn reserved_bytes(&self) -> usize {
        let preparation = self
            .0
            .original
            .reservation
            .bytes()
            .saturating_add(self.0.original.preparation_metadata_bytes);
        self.0.facts.iter().fold(
            self.0.reservation.bytes().saturating_add(preparation),
            |n, fact| n.saturating_add(fact.bytes()),
        )
    }
    pub fn original_owner_bytes(&self) -> usize {
        self.0
            .original
            .owners
            .iter()
            .fold(0usize, |n, owner| n.saturating_add(owner.reserved_bytes()))
    }
    pub(super) fn owner(&self) -> &Arc<ChainOwner> {
        &self.0.original
    }
    pub(super) fn facts(&self) -> &[Fact] {
        &self.0.facts
    }
    pub(super) fn positions(&self) -> usize {
        self.0.positions
    }
    pub(super) fn input_digest(&self) -> Digest {
        self.0.input_digest
    }
}

struct LinearOwner {
    output: OriginalAffineLinearReport,
    facts: OriginalChainForwardReport,
    binding: OriginalChainBinding,
    reservation: PersistentReservation,
}

/// Checked signed local rows over the chain input, with complete stored errors.
/// Actual forward facts and original coefficients survive every output clone.
/// There is no owned-array extraction, graph eligibility or full-cover claim.
#[derive(Clone)]
pub struct OriginalChainLinearReport(Arc<LinearOwner>);

impl OriginalChainLinearReport {
    pub(super) fn finish(
        output: OriginalAffineLinearReport,
        facts: OriginalChainForwardReport,
        binding: OriginalChainBinding,
        mut phase: OriginalAffinePhase,
    ) -> Result<Self> {
        if size_of::<LinearOwner>() + 4 * size_of::<usize>() > META {
            return Err(invalid("linear report metadata capacity"));
        }
        phase.retain_metadata(META)?;
        let reservation = phase.commit()?;
        Ok(Self(Arc::new(LinearOwner {
            output,
            facts,
            binding,
            reservation,
        })))
    }
    pub fn bounds(&self) -> &LinearBounds {
        self.0.output.bounds()
    }
    pub fn scope(&self) -> &ScopeIdentity {
        self.0.facts.scope()
    }
    pub fn binding(&self) -> OriginalChainBinding {
        self.0.binding
    }
    pub fn forward_facts(&self) -> &OriginalChainForwardReport {
        &self.0.facts
    }
    pub fn reserved_bytes(&self) -> usize {
        self.0
            .reservation
            .bytes()
            .saturating_add(self.0.output.reserved_bytes())
            .saturating_add(self.0.facts.reserved_bytes())
    }
    pub fn original_owner_bytes(&self) -> usize {
        self.0.facts.original_owner_bytes()
    }
}
