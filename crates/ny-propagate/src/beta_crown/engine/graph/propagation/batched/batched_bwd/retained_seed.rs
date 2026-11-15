// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Immutable custody and existing-query admission of the EXACT owned seed.
//! This component accounts the seed, not the original graph or domain maps.
//! It never certifies the seed's meaning or enables finite collection.

use super::bounded_storage::{BoundedBatchPending, SoaStorageBudget};
use super::{CrownDispatchPlan, LinearBounds};
use crate::bounds::patches::PatchesMaterializationDeadline;
use crate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkRefusal, WorkRequest,
};
use crate::layers::linear::allocation_provenance::TrackedLinearArray;
use ny_core::{NyError, Result};
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

const SITE: &str = "retained original SoA seed";

fn invalid(reason: &str) -> NyError {
    NyError::InvalidSpec(format!("{SITE}: {reason}"))
}
fn refused(reason: WorkRefusal) -> NyError {
    match reason {
        WorkRefusal::Deadline | WorkRefusal::Cancelled => {
            NyError::DeadlineExceeded(format!("{SITE}: original query {reason}"))
        }
        _ => NyError::UnsupportedOp(format!("{SITE}: original query {reason}")),
    }
}
fn narrower(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}
fn checkpoint(query: &WorkBudget, class: LeaseClass, deadline: Option<Instant>) -> Result<()> {
    query.checkpoint_for(class).map_err(refused)?;
    PatchesMaterializationDeadline::new(deadline).checkpoint(SITE)
}

/// Every refusal retains the same original arrays for the caller's existing
/// fallback. Only UnsupportedLayout is a clean allocation-capability miss.
#[derive(Debug)]
pub(super) struct SeedAdoptionFailure {
    original: LinearBounds,
    error: NyError,
    unsupported_layout: bool,
}
impl SeedAdoptionFailure {
    fn new(original: LinearBounds, error: NyError, unsupported_layout: bool) -> Self {
        Self {
            original,
            error,
            unsupported_layout,
        }
    }
    pub(super) fn into_parts(self) -> (LinearBounds, NyError, bool) {
        (self.original, self.error, self.unsupported_layout)
    }
}

/// Move-only owner: no mutable seed or detachable claim is exposed. Arrays
/// drop before the claim. Binding cannot switch query ledger, class or clock.
pub(super) struct RetainedSoaSeed {
    original: LinearBounds,
    claim: PersistentReservation,
    class: LeaseClass,
    deadline: Option<Instant>,
}

/// Reuse the original ndarray ownership tracker. All six allocation
/// outcomes are inspected, so an unsupported role cannot hide an invalid one.
/// Fixed-rank metadata reconstruction retains each original Vec and offset.
fn observe_seed(original: LinearBounds) -> (LinearBounds, Result<Option<usize>>) {
    let LinearBounds {
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        lower_a_err,
        upper_a_err,
    } = original;
    let lower_a = TrackedLinearArray::new(lower_a);
    let lower_b = TrackedLinearArray::new(lower_b);
    let upper_a = TrackedLinearArray::new(upper_a);
    let upper_b = TrackedLinearArray::new(upper_b);
    let lower_a_err = lower_a_err.map(TrackedLinearArray::new);
    let upper_a_err = upper_a_err.map(TrackedLinearArray::new);
    let facts = [
        lower_a.observed_payload_bytes_v1(),
        lower_b.observed_payload_bytes_v1(),
        upper_a.observed_payload_bytes_v1(),
        upper_b.observed_payload_bytes_v1(),
        lower_a_err
            .as_ref()
            .map_or(Ok(Some(0)), |a| a.observed_payload_bytes_v1()),
        upper_a_err
            .as_ref()
            .map_or(Ok(Some(0)), |a| a.observed_payload_bytes_v1()),
        PersistentReservation::allocation_request_bytes()
            .map(Some)
            .map_err(refused),
    ];
    let quote = quote_seed(facts);
    let original = LinearBounds {
        lower_a: lower_a.into_array(),
        lower_b: lower_b.into_array(),
        upper_a: upper_a.into_array(),
        upper_b: upper_b.into_array(),
        lower_a_err: lower_a_err.map(TrackedLinearArray::into_array),
        upper_a_err: upper_a_err.map(TrackedLinearArray::into_array),
    };
    (original, quote)
}

fn quote_seed(facts: [Result<Option<usize>>; 7]) -> Result<Option<usize>> {
    // Conservative inline owner charge, not allocator metadata or RSS. There
    // is no copied payload or extra heap wrapper. The reservation's own Arc
    // body/counters are included above before commit_persistent allocates it.
    let mut bytes = size_of::<RetainedSoaSeed>();
    let mut unsupported = false;
    let mut first_error = None;
    for fact in facts {
        match fact {
            Ok(Some(n)) => match bytes.checked_add(n) {
                Some(total) => bytes = total,
                None => {
                    first_error.get_or_insert_with(|| invalid("capacity sum overflow"));
                }
            },
            Ok(None) => unsupported = true,
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    match first_error {
        Some(error) => Err(error),
        None if unsupported => Ok(None),
        None => Ok(Some(bytes)),
    }
}

impl RetainedSoaSeed {
    /// Adopt a previously uncharged original seed. Its existing query must
    /// already join the resident graph's MemoryAuthority. This does not count
    /// source memory before adoption or replace a complete capture census.
    #[expect(
        clippy::result_large_err,
        reason = "Failure returns the original seed custody without allocating"
    )]
    pub(super) fn try_adopt(
        original: LinearBounds,
        query: &WorkBudget,
        class: LeaseClass,
        requested: Option<Instant>,
    ) -> std::result::Result<Self, SeedAdoptionFailure> {
        if query.memory_authority().is_none() {
            return Err(SeedAdoptionFailure::new(
                original,
                invalid("query has no original shared memory authority"),
                false,
            ));
        }
        if let Err(error) = checkpoint(query, class, requested) {
            return Err(SeedAdoptionFailure::new(original, error, false));
        }
        // Six fixed-rank ownership observations, never a scalar-value scan.
        // Admit that work before reconstructing any ndarray metadata.
        let mut work = match query.lease_for(
            class,
            WorkRequest {
                work: 64,
                ..WorkRequest::default()
            },
        ) {
            Ok(work) => work,
            Err(error) => return Err(SeedAdoptionFailure::new(original, refused(error), false)),
        };
        let deadline = narrower(work.deadline(), requested);
        let (original, quote) = observe_seed(original);
        let bytes = match quote {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                return Err(SeedAdoptionFailure::new(
                    original,
                    NyError::UnsupportedOp(format!("{SITE}: unsupported ndarray layout")),
                    true,
                ))
            }
            Err(error) => return Err(SeedAdoptionFailure::new(original, error, false)),
        };
        if let Err(error) = checkpoint(query, class, deadline) {
            return Err(SeedAdoptionFailure::new(original, error, false));
        }
        let lease = match query.lease_for(
            class,
            WorkRequest {
                persistent_bytes: bytes,
                ..WorkRequest::default()
            },
        ) {
            Ok(lease) => lease,
            Err(error) => return Err(SeedAdoptionFailure::new(original, refused(error), false)),
        };
        let claim = match lease.commit_persistent() {
            Ok(claim) => claim,
            Err(error) => return Err(SeedAdoptionFailure::new(original, refused(error), false)),
        };
        if let Err(error) =
            checkpoint(query, class, deadline).and_then(|()| work.complete().map_err(refused))
        {
            drop(claim);
            return Err(SeedAdoptionFailure::new(original, error, false));
        }
        Ok(Self {
            original,
            claim,
            class,
            deadline,
        })
    }

    pub(super) fn seed(&self) -> &LinearBounds {
        &self.original
    }
    pub(super) fn retained_bytes(&self) -> usize {
        self.claim.bytes()
    }
    pub(super) fn storage_budget(
        &self,
        query: &WorkBudget,
        requested: Option<Instant>,
    ) -> Result<Arc<SoaStorageBudget>> {
        if !self.claim.shares_ledger(query) {
            return Err(invalid(
                "seed and caller use different original query ledgers",
            ));
        }
        let deadline = narrower(self.deadline, requested);
        checkpoint(query, self.class, deadline)?;
        SoaStorageBudget::from_query(query, self.claim.clone(), self.class, deadline)
    }

    /// Seed only from this exact original owner; no unrelated LinearBounds can
    /// be paired with its claim. Graph/target/domain correspondence remains a
    /// separate capture obligation and finite traversal remains refused.
    pub(super) fn seed_pending(
        &self,
        query: &WorkBudget,
        plan: &CrownDispatchPlan,
        domains: usize,
        idx: usize,
        requested: Option<Instant>,
    ) -> Result<BoundedBatchPending> {
        let owner = self.storage_budget(query, requested)?;
        let deadline = narrower(self.deadline, requested);
        BoundedBatchPending::try_new(plan, domains, &owner, deadline)?
            .seed_broadcast(idx, &self.original)
    }

    /// Caller-controlled end of declared participation. The exact original
    /// ndarray owners are moved back, with no clone, repair or revalidation.
    pub(super) fn into_original(self) -> LinearBounds {
        let Self {
            original, claim, ..
        } = self;
        drop(claim);
        original
    }
}

#[cfg(test)]
#[path = "retained_seed_tests.rs"]
mod tests;
