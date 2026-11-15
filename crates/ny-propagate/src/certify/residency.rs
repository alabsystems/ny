// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared configured capacity authority for explicitly participating ledgers.
//!
//! This counts requested allocation capacities, including pending commitments.
//! Materialized means the participating owner has committed its capacity; it
//! does not mean physically resident pages or measured RSS. All claims continue
//! to count against the same cap. Untracked allocations and reconciliation with
//! measured process/system headroom require their own adapters.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use super::work::WorkRefusal;

/// Monotone, clone-shared cancellation for one finite query. Cancelling never
/// releases a worker's live reservations; its actual owners must drop them.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryReceipt {
    pub attempted: u64,
    pub admitted: u64,
    pub refused: u64,
    pub live_pending_bytes: usize,
    pub live_materialized_bytes: usize,
    pub peak_leased_bytes: usize,
}

#[derive(Debug)]
struct MemoryState {
    cap: usize,
    receipt: MemoryReceipt,
}

/// Distinct work ledgers may join this one byte authority. Clones preserve
/// identity; equal caps on another authority do not grant shared admission.
#[derive(Clone, Debug)]
pub struct MemoryAuthority(Arc<Mutex<MemoryState>>);

impl MemoryAuthority {
    pub fn new(capacity_bytes: usize) -> Self {
        Self(Arc::new(Mutex::new(MemoryState {
            cap: capacity_bytes,
            receipt: MemoryReceipt::default(),
        })))
    }

    pub fn shares_authority(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub fn receipt(&self) -> Result<MemoryReceipt, WorkRefusal> {
        self.0
            .lock()
            .map(|state| state.receipt)
            .map_err(|_| WorkRefusal::Unavailable)
    }

    pub(crate) fn reserve(&self, bytes: usize) -> Result<MemoryClaim, WorkRefusal> {
        let mut state = self.0.lock().map_err(|_| WorkRefusal::Unavailable)?;
        admit_pending_capacity(&mut state, bytes)?;
        Ok(MemoryClaim {
            authority: self.clone(),
            bytes,
            materialized: false,
        })
    }
}

/// Common admission for a new claim or an actual-capacity overage. Both spend
/// the same aggregate cap, including all other pending/materialized owners.
/// Receipt admissions count capacity decisions, including successful growth.
fn admit_pending_capacity(state: &mut MemoryState, bytes: usize) -> Result<(), WorkRefusal> {
    state.receipt.attempted = state.receipt.attempted.saturating_add(1);
    let pending = state.receipt.live_pending_bytes.checked_add(bytes);
    let total =
        pending.and_then(|pending| pending.checked_add(state.receipt.live_materialized_bytes));
    let total = match total {
        Some(total) if total <= state.cap => total,
        Some(_) => {
            state.receipt.refused = state.receipt.refused.saturating_add(1);
            return Err(WorkRefusal::MemoryLimit);
        }
        None => {
            state.receipt.refused = state.receipt.refused.saturating_add(1);
            return Err(WorkRefusal::Overflow);
        }
    };
    state.receipt.live_pending_bytes = pending.ok_or(WorkRefusal::Overflow)?;
    state.receipt.peak_leased_bytes = state.receipt.peak_leased_bytes.max(total);
    state.receipt.admitted = state.receipt.admitted.saturating_add(1);
    Ok(())
}

/// A single claim; it is transferred into the last-owner persistent wrapper.
/// This type deliberately has no Clone implementation.
#[derive(Debug)]
pub(crate) struct MemoryClaim {
    authority: MemoryAuthority,
    bytes: usize,
    materialized: bool,
}

impl MemoryClaim {
    /// Admit an overage while this entire allocation commitment is still
    /// pending. Call before touching the new capacity, then materialize once
    /// every participating buffer has reconciled. Sealed claims cannot grow.
    pub(crate) fn grow_pending(&mut self, additional: usize) -> Result<(), WorkRefusal> {
        if self.materialized {
            return Err(WorkRefusal::Unavailable);
        }
        let next = self
            .bytes
            .checked_add(additional)
            .ok_or(WorkRefusal::Overflow)?;
        if additional == 0 {
            return Ok(());
        }
        let mut state = self
            .authority
            .0
            .lock()
            .map_err(|_| WorkRefusal::Unavailable)?;
        admit_pending_capacity(&mut state, additional)?;
        self.bytes = next;
        Ok(())
    }

    pub(crate) fn materialize(&mut self) -> Result<(), WorkRefusal> {
        if self.materialized {
            return Ok(());
        }
        let mut state = self
            .authority
            .0
            .lock()
            .map_err(|_| WorkRefusal::Unavailable)?;
        let pending = state
            .receipt
            .live_pending_bytes
            .checked_sub(self.bytes)
            .ok_or(WorkRefusal::Unavailable)?;
        let materialized = state
            .receipt
            .live_materialized_bytes
            .checked_add(self.bytes)
            .ok_or(WorkRefusal::Overflow)?;
        state.receipt.live_pending_bytes = pending;
        state.receipt.live_materialized_bytes = materialized;
        self.materialized = true;
        Ok(())
    }

    /// Release scratch while retaining the original claim for returned owners.
    /// Shrinking never creates or replenishes a work/memory allowance.
    pub(crate) fn retain_only(&mut self, bytes: usize) -> Result<(), WorkRefusal> {
        let release = self.bytes.checked_sub(bytes).ok_or(WorkRefusal::Overflow)?;
        let mut state = self
            .authority
            .0
            .lock()
            .map_err(|_| WorkRefusal::Unavailable)?;
        let live = if self.materialized {
            &mut state.receipt.live_materialized_bytes
        } else {
            &mut state.receipt.live_pending_bytes
        };
        *live = live.checked_sub(release).ok_or(WorkRefusal::Unavailable)?;
        self.bytes = bytes;
        Ok(())
    }
}

impl Drop for MemoryClaim {
    fn drop(&mut self) {
        if let Ok(mut state) = self.authority.0.lock() {
            let live = if self.materialized {
                &mut state.receipt.live_materialized_bytes
            } else {
                &mut state.receipt.live_pending_bytes
            };
            // Never underflow a capacity counter into a fresh allowance. On
            // an inconsistent claim, retain the conservative byte charge.
            if let Some(remaining) = live.checked_sub(self.bytes) {
                *live = remaining;
            }
        }
    }
}

#[cfg(test)]
#[path = "residency/tests.rs"]
mod tests;
