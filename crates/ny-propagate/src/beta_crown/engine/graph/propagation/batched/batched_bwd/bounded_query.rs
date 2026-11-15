// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bind storage to an existing admitted query and retained capacity owner.
//! The retained claim must own the external graph/maps/seed capture. Identity
//! and byte checks do not prove that capture's source correspondence; the
//! future collector still needs its original-input ownership/census binding.

use super::{invalid, narrower, Poll, SoaStorageBudget, ENTRY, PUBLISH, SITE};
use crate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRefusal, WorkRequest,
};
use ny_core::{NyError, Result};
use std::mem::size_of;
use std::sync::{atomic::AtomicUsize, Arc, Mutex};
use std::time::Instant;

pub(super) struct QueryAuthority {
    _retained: PersistentReservation,
    query: WorkBudget,
    class: LeaseClass,
    phase: Mutex<WorkLease>,
}

fn refused(error: WorkRefusal) -> NyError {
    match error {
        WorkRefusal::Deadline | WorkRefusal::Cancelled => {
            NyError::DeadlineExceeded(format!("{SITE}: original query {error}"))
        }
        _ => NyError::UnsupportedOp(format!("{SITE}: original query authority refused {error}")),
    }
}

impl QueryAuthority {
    pub(super) fn checkpoint(&self) -> Result<()> {
        self.query.checkpoint_for(self.class).map_err(refused)
    }

    pub(super) fn bytes(&self, bytes: usize) -> Result<WorkLease> {
        self.query
            .lease_for(
                self.class,
                WorkRequest {
                    scratch_bytes: bytes,
                    ..WorkRequest::default()
                },
            )
            .map_err(refused)
    }

    pub(super) fn work(&self, units: usize) -> Result<()> {
        if units == 0 {
            return self.checkpoint();
        }
        let units = u64::try_from(units).map_err(|_| invalid("query work count overflows u64"))?;
        self.phase
            .lock()
            .map_err(|_| invalid("query work phase poisoned"))?
            .reserve_more_work(units)
            .map_err(refused)
    }
}

// The long-lived work phase is never marked complete by Drop. The eventual
// complete collector needs its own explicit success boundary; a refusal must
// not become a successful completion merely because its storage is released.

impl SoaStorageBudget {
    /// Join the caller's query and the ORIGINAL retained allocation claim.
    /// No memory authority or work/deadline allowance is constructed here.
    /// The complete retained claim counts in the local dense envelope and is
    /// cloned/moved without charging its bytes a second time globally.
    pub(in super::super) fn from_query(
        query: &WorkBudget,
        retained: PersistentReservation,
        class: LeaseClass,
        requested: Option<Instant>,
    ) -> Result<Arc<Self>> {
        Poll::new(requested).checkpoint(ENTRY)?;
        if !retained.shares_memory_authority(query) {
            return Err(invalid(
                "retained payload and query lack the same memory authority",
            ));
        }
        let retained_bytes = retained.bytes();
        // Participating owner bodies plus the two Arc counter pairs. This is
        // declared capacity, not an allocator-metadata or RSS measurement.
        let headers = size_of::<Self>()
            .checked_add(size_of::<QueryAuthority>())
            .and_then(|n| n.checked_add(4 * size_of::<usize>()))
            .ok_or_else(|| invalid("query storage owner header overflow"))?;
        let required = retained_bytes
            .checked_add(headers)
            .ok_or_else(|| invalid("query retained bytes overflow"))?;
        let ceiling_bytes = super::cpu_crown_dense_budget_bytes();
        if required > ceiling_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: required,
                budget_bytes: ceiling_bytes,
                site: SITE,
            });
        }
        let phase = query
            .lease_for(
                class,
                WorkRequest {
                    scratch_bytes: headers,
                    work: 64,
                    ..WorkRequest::default()
                },
            )
            .map_err(refused)?;
        let deadline = narrower(phase.deadline(), requested);
        Poll::new(deadline).checkpoint(ENTRY)?;
        let authority = Arc::new(QueryAuthority {
            _retained: retained,
            query: query.clone(),
            class,
            phase: Mutex::new(phase),
        });
        let owner = Arc::new(Self {
            retained_bytes,
            ceiling_bytes,
            reserved_bytes: AtomicUsize::new(headers),
            deadline,
            query: Some(authority),
        });
        owner
            .query
            .as_ref()
            .expect("query authority set above")
            .phase
            .lock()
            .map_err(|_| invalid("query work phase poisoned"))?
            .mark_materialized()
            .map_err(refused)?;
        let mut poll = Poll::for_owner(&owner, deadline);
        owner.check_live(&mut poll)?;
        poll.checkpoint(PUBLISH)?;
        Ok(owner)
    }
}

impl super::Charge {
    pub(super) fn materialize(&mut self) -> Result<()> {
        if let Some(lease) = &mut self.query_lease {
            lease.mark_materialized().map_err(refused)?;
            lease.complete().map_err(refused)?;
        }
        Ok(())
    }

    pub(super) fn reconcile_query(&mut self, additional: usize) -> Result<()> {
        if let Some(lease) = &mut self.query_lease {
            lease.reserve_more_scratch(additional).map_err(refused)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "bounded_query_tests.rs"]
mod tests;
