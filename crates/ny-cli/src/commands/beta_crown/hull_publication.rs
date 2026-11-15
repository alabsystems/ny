// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Participating hull allocations only. Loaded models and legacy LP/BaB have
//! not joined this census. One caller owner is shared by all phase children.

use anyhow::{anyhow, Result};
use ny_mip::single_hidden_hull::resource::{
    HullCheckedCertificate, HullLeaseClass, HullProofOrigin, HullQueryContext,
};
use ny_propagate::certify::{
    residency::{CancellationToken, MemoryAuthority},
    work::{LeaseClass, PersistentReservation, WorkBudget, WorkLimits, WorkRequest, WorkReserve},
};
use std::{
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use super::single_hidden_hull::resource::{hull_context_for_budget, HullBudgetedRootResult};

pub(super) const HULL_PARTICIPATING_BYTES: usize = 32 << 20;
pub(super) const HULL_JSON_BYTES: usize = 64 << 10;
const OWNER_BYTES: usize = 4096;

#[derive(Debug)]
pub(crate) struct InstanceVerificationResources {
    budget: WorkBudget,
    cancellation: CancellationToken,
    original_deadline: Option<Instant>,
    _resident: PersistentReservation,
}

/// A caller refusal is inherited too: a nested handler must not replace it
/// with a fresh ledger and another allowance.
#[derive(Debug, Clone)]
pub(crate) enum CallerVerificationResources {
    Ready(Arc<InstanceVerificationResources>),
    Refused(String),
}

impl InstanceVerificationResources {
    /// Called at the caller's started dispatch boundary, never in a leaf.
    pub(crate) fn start(original_deadline: Option<Instant>) -> Result<Arc<Self>> {
        let instance = ny_core::instance_budget::deadline();
        let original_deadline = match (original_deadline, instance) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let bounded = original_deadline.unwrap_or_else(|| Instant::now() + Duration::from_secs(20));
        let memory = MemoryAuthority::new(HULL_PARTICIPATING_BYTES);
        let cancellation = CancellationToken::default();
        let budget = WorkBudget::new_in(
            &memory,
            &cancellation,
            WorkLimits {
                deadline: Some(bounded),
                scratch_bytes: 16 << 20,
                persistent_bytes: HULL_PARTICIPATING_BYTES,
                max_work: 1 << 30,
                max_proposals: 1,
                certificate_reserve: Duration::from_millis(100),
                reserve: WorkReserve::default(),
            },
        )?;
        budget.tighten_total_byte_limit(HULL_PARTICIPATING_BYTES)?;
        let mut lease = budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: OWNER_BYTES,
                ..WorkRequest::default()
            },
        )?;
        lease.mark_materialized()?;
        let resident = lease.commit_persistent()?;
        Ok(Arc::new(Self {
            budget,
            cancellation,
            original_deadline,
            _resident: resident,
        }))
    }

    pub(super) fn context(self: &Arc<Self>, phase_deadline: Instant) -> HullRootContext {
        let deadline = self
            .original_deadline
            .map_or(phase_deadline, |d| d.min(phase_deadline));
        HullRootContext {
            context: hull_context_for_budget(&self.budget, deadline),
            resources: self.clone(),
        }
    }

    #[cfg(test)]
    pub(crate) fn cancel(&self) {
        self.cancellation.cancel();
    }
    pub(super) fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    pub(crate) fn budget(&self) -> &WorkBudget {
        &self.budget
    }
}

/// Constructible only by the exact caller resource owner. An arbitrary query
/// adapter cannot be paired with an unrelated equal-limit budget at publication.
pub(super) struct HullRootContext {
    context: HullQueryContext,
    resources: Arc<InstanceVerificationResources>,
}
impl HullRootContext {
    pub(super) fn query(&self) -> &HullQueryContext {
        &self.context
    }
}

/// Private producer binding: scope owns all original values and its one full
/// property objective; capture identity prevents cross-invocation reuse.
pub(crate) struct HullPublicationProof {
    replay: HullCheckedCertificate,
    context: HullQueryContext,
    resources: Arc<InstanceVerificationResources>,
    expected_scope: [u8; 32],
    capture: Option<Arc<()>>,
    _resident: PersistentReservation,
}

impl fmt::Debug for HullPublicationProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HullPublicationProof")
            .field("scope", &self.expected_scope)
            .field("lower_bound", &self.replay.lower_bound())
            .finish_non_exhaustive()
    }
}

impl HullPublicationProof {
    pub(super) fn from_checked(
        checked: HullBudgetedRootResult,
        root_context: HullRootContext,
    ) -> Result<(ny_propagate::BetaCrownResult, Arc<Self>)> {
        let HullRootContext { context, resources } = root_context;
        let original = checked.proof.original();
        anyhow::ensure!(
            checked.proof.origin() == HullProofOrigin::OriginalReplay
                && checked.proof.verified()
                && matches!(
                    checked.result.result,
                    ny_propagate::BabVerificationStatus::Verified
                )
                && checked.proof.scope() == original.scope()
                && original.context().shares_authority(&context),
            "foreign or unverified hull replay"
        );
        context.checkpoint(HullLeaseClass::Replay)?;
        let mut lease = resources.budget.lease_for(
            LeaseClass::Replay,
            WorkRequest {
                persistent_bytes: OWNER_BYTES,
                work: 64,
                ..WorkRequest::default()
            },
        )?;
        lease.mark_materialized()?;
        let resident = lease.commit_persistent()?;
        let expected_scope = original.scope();
        let proof = Arc::new(Self {
            replay: checked.proof,
            context,
            resources,
            expected_scope,
            capture: super::output::capture_identity(),
            _resident: resident,
        });
        proof.checkpoint()?;
        Ok((checked.result, proof))
    }

    pub(crate) fn checkpoint(&self) -> Result<()> {
        anyhow::ensure!(
            self.replay.origin() == HullProofOrigin::OriginalReplay
                && self.replay.verified()
                && self.replay.scope() == self.expected_scope
                && self.replay.original().scope() == self.expected_scope
                && self
                    .replay
                    .original()
                    .context()
                    .shares_authority(&self.context),
            "hull publication scope mismatch"
        );
        self.context.checkpoint(HullLeaseClass::Replay)?;
        Ok(())
    }

    pub(super) fn capture_matches(&self, capture: Option<&Arc<()>>) -> bool {
        match (&self.capture, capture) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }

    pub(crate) fn scope(&self) -> [u8; 32] {
        self.expected_scope
    }
    pub(crate) fn lower_bound(&self) -> f64 {
        self.replay.lower_bound()
    }
    pub(super) fn search(&self) -> ny_mip::single_hidden_hull::resource::HullTriangleSearchStats {
        self.replay.triangle_search()
    }
    #[cfg(test)]
    pub(super) fn resources(&self) -> &Arc<InstanceVerificationResources> {
        &self.resources
    }

    pub(crate) fn reserve_publication(&self, bytes: usize) -> Result<PersistentReservation> {
        self.checkpoint()?;
        let mut lease = self.resources.budget.lease_for(
            LeaseClass::Replay,
            WorkRequest {
                persistent_bytes: bytes,
                // Bounded JSON source visits, escaping, numeric formatting and
                // path/metadata scans: at most128 logical steps/quoted byte plus
                // fixed serializer/channel overhead. No additional proposal.
                work: u64::try_from(bytes)?
                    .checked_mul(128)
                    .and_then(|n| n.checked_add(8192))
                    .ok_or_else(|| anyhow!("hull publication work overflow"))?,
                ..WorkRequest::default()
            },
        )?;
        lease.mark_materialized()?;
        lease.checkpoint()?;
        Ok(lease.commit_persistent()?)
    }
}

#[derive(Debug)]
pub(crate) struct HullPublicationAuthority {
    proof: Arc<HullPublicationProof>,
    // Keeps serialized/captured and receipt metadata capacity paid until the
    // last staged/concurrent publication owner has actually gone away.
    _storage: PersistentReservation,
}

impl HullPublicationAuthority {
    pub(super) fn new(
        proof: Arc<HullPublicationProof>,
        storage: PersistentReservation,
    ) -> Result<Arc<Self>> {
        anyhow::ensure!(
            storage.shares_ledger(proof.resources.budget()),
            "foreign hull publication storage"
        );
        anyhow::ensure!(
            storage.bytes() >= HULL_JSON_BYTES * 2 + OWNER_BYTES,
            "hull publication storage is underpaid"
        );
        proof.checkpoint()?;
        Ok(Arc::new(Self {
            proof,
            _storage: storage,
        }))
    }
    pub(crate) fn checkpoint(&self) -> Result<()> {
        self.proof.checkpoint()
    }
    #[cfg(test)]
    pub(crate) fn scope(&self) -> [u8; 32] {
        self.proof.scope()
    }
    pub(crate) fn lower_bound(&self) -> f64 {
        self.proof.lower_bound()
    }
    pub(crate) fn reserve_publication(&self, bytes: usize) -> Result<PersistentReservation> {
        self.proof.reserve_publication(bytes)
    }
}

/// Bounded serializer: every byte is admitted before allocation; no growth
/// beyond the reconciled actual capacity is possible.
pub(super) struct JsonWriter {
    bytes: Vec<u8>,
}
impl JsonWriter {
    pub(super) fn new() -> Result<Self> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(HULL_JSON_BYTES)?;
        anyhow::ensure!(
            bytes.capacity() <= HULL_JSON_BYTES * 2,
            "hull JSON allocator capacity exceeds admission"
        );
        Ok(Self { bytes })
    }
    pub(super) fn finish(self) -> Result<String> {
        Ok(String::from_utf8(self.bytes)?)
    }
}
impl std::io::Write for JsonWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let Some(next) = self.bytes.len().checked_add(bytes.len()) else {
            return Err(std::io::Error::other("hull JSON overflow"));
        };
        if next > HULL_JSON_BYTES || next > self.bytes.capacity() {
            return Err(std::io::Error::other("hull JSON exceeds prepaid capacity"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "hull_publication_tests.rs"]
pub(crate) mod tests;
