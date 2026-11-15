// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! A shared admission ledger for optional search and certificate work.
//!
//! Clone the budget, not its allowance: all clones charge the same counters.
//! Leases are acquired before allocation and release resident bytes on drop.
//! Work/proposal allowances are consumed even if a proposal fails. Reserving
//! memory does not prove a bound, and refusals never revoke prior evidence.

use super::residency::{CancellationToken, MemoryAuthority, MemoryClaim};
use std::alloc::Layout;
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkRefusal {
    Deadline,
    ScratchLimit,
    PersistentLimit,
    WorkLimit,
    ProposalLimit,
    Overflow,
    Unavailable,
    Cancelled,
    MemoryLimit,
}
impl std::fmt::Display for WorkRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for WorkRefusal {}

/// Conservative predictions, including allocation capacity and metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkRequest {
    pub scratch_bytes: usize,
    pub persistent_bytes: usize,
    pub work: u64,
    pub proposals: u64,
}

/// Caller-predicted capacity held for replay/complete fallback. This does not
/// grant additional resources: replay and fallback still share the total caps.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkReserve {
    pub persistent_bytes: usize,
    pub work: u64,
    pub proposals: u64,
}

/// Explicit finite limits for a caller-admitted query or model construction.
/// A new query joins its resident model's memory authority; nested phases clone
/// this same work ledger and cannot replace its work/deadline/cancellation.
#[derive(Debug, Clone, Copy)]
pub struct WorkLimits {
    pub deadline: Option<Instant>,
    pub scratch_bytes: usize,
    pub persistent_bytes: usize,
    pub max_work: u64,
    pub max_proposals: u64,
    pub certificate_reserve: Duration,
    pub reserve: WorkReserve,
}

/// Admission and completion counts are distinct. Work counts are reserved
/// allowances; byte counts are leased capacities, not measured RSS.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkReceipt {
    pub attempted: u64,
    pub admitted: u64,
    pub refused: u64,
    pub completed: u64,
    pub work_reserved: u64,
    pub proposals_reserved: u64,
    pub live_scratch_bytes: usize,
    pub live_persistent_bytes: usize,
    pub peak_leased_bytes: usize,
}

/// Deadline class; all classes still spend the same resource counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseClass {
    /// Optional proposals stop before the certificate reserve.
    Optional,
    /// Original-evidence replay may use the reserved time.
    Replay,
    /// Complete verification fallback may use the reserved time.
    Fallback,
}

#[derive(Debug)]
struct State {
    max_scratch: usize,
    max_persistent: usize,
    max_total_bytes: usize,
    max_work: u64,
    max_proposals: u64,
    reserve: WorkReserve,
    receipt: WorkReceipt,
}

/// One query's shared budget. The reserve is time held for replay/fallback.
#[derive(Debug, Clone)]
pub struct WorkBudget(Arc<BudgetInner>);

#[derive(Debug)]
struct BudgetInner {
    candidate_deadline: Option<Instant>,
    final_deadline: Option<Instant>,
    state: Mutex<State>,
    memory: Option<MemoryAuthority>,
    cancellation: Option<CancellationToken>,
}

impl WorkBudget {
    pub fn new(
        deadline: Option<Instant>,
        scratch_bytes: usize,
        persistent_bytes: usize,
        max_work: u64,
        max_proposals: u64,
        certificate_reserve: Duration,
    ) -> Result<Self, WorkRefusal> {
        Self::new_reserved(
            deadline,
            scratch_bytes,
            persistent_bytes,
            max_work,
            max_proposals,
            certificate_reserve,
            WorkReserve::default(),
        )
    }

    pub fn new_reserved(
        deadline: Option<Instant>,
        scratch_bytes: usize,
        persistent_bytes: usize,
        max_work: u64,
        max_proposals: u64,
        certificate_reserve: Duration,
        reserve: WorkReserve,
    ) -> Result<Self, WorkRefusal> {
        if reserve.persistent_bytes > persistent_bytes {
            return Err(WorkRefusal::PersistentLimit);
        }
        if reserve.work > max_work {
            return Err(WorkRefusal::WorkLimit);
        }
        if reserve.proposals > max_proposals {
            return Err(WorkRefusal::ProposalLimit);
        }
        let candidate_deadline = deadline
            .map(|deadline| {
                deadline
                    .checked_sub(certificate_reserve)
                    .ok_or(WorkRefusal::Overflow)
            })
            .transpose()?;
        let max_total_bytes = scratch_bytes
            .checked_add(persistent_bytes)
            .ok_or(WorkRefusal::Overflow)?;
        Ok(Self(Arc::new(BudgetInner {
            candidate_deadline,
            final_deadline: deadline,
            memory: None,
            cancellation: None,
            state: Mutex::new(State {
                max_scratch: scratch_bytes,
                max_persistent: persistent_bytes,
                max_total_bytes,
                max_work,
                max_proposals,
                reserve,
                receipt: WorkReceipt::default(),
            }),
        })))
    }

    /// Caller-created finite work authority, sharing configured byte admission
    /// with distinct resident/model/query ledgers. Existing constructors retain
    /// their original bounded-session semantics.
    pub fn new_in(
        memory: &MemoryAuthority,
        cancellation: &CancellationToken,
        limits: WorkLimits,
    ) -> Result<Self, WorkRefusal> {
        let mut budget = Self::new_reserved(
            limits.deadline,
            limits.scratch_bytes,
            limits.persistent_bytes,
            limits.max_work,
            limits.max_proposals,
            limits.certificate_reserve,
            limits.reserve,
        )?;
        let inner = Arc::get_mut(&mut budget.0).ok_or(WorkRefusal::Unavailable)?;
        inner.memory = Some(memory.clone());
        inner.cancellation = Some(cancellation.clone());
        Ok(budget)
    }

    pub fn memory_authority(&self) -> Option<MemoryAuthority> {
        self.0.memory.clone()
    }

    pub fn shares_memory_authority(&self, other: &Self) -> bool {
        match (&self.0.memory, &other.0.memory) {
            (Some(left), Some(right)) => left.shares_authority(right),
            _ => false,
        }
    }

    /// True only for clones of this same resource authority. Equal limits do
    /// not make independently constructed ledgers interchangeable.
    pub fn shares_ledger(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Monotonically restrict combined scratch and persistent capacity on this
    /// exact shared ledger. Restriction and every later admission use the same
    /// mutex, so concurrent phases cannot each spend the same remaining peak.
    /// Existing live claims must fit; refusal leaves the limit and claims
    /// unchanged. This never renews work, proposals, time or cancellation.
    /// Optional byte admissions preserve the configured replay reserve inside
    /// this combined ceiling. A ceiling below that reserve leaves no optional
    /// byte capacity; replay/fallback still use only the restricted ceiling.
    pub fn tighten_total_byte_limit(&self, limit: usize) -> Result<usize, WorkRefusal> {
        let mut state = self.0.state.lock().map_err(|_| WorkRefusal::Unavailable)?;
        self.checkpoint_for(LeaseClass::Fallback)?;
        let restricted = state.max_total_bytes.min(limit);
        let live = state
            .receipt
            .live_scratch_bytes
            .checked_add(state.receipt.live_persistent_bytes)
            .ok_or(WorkRefusal::Overflow)?;
        if live > restricted {
            return Err(WorkRefusal::MemoryLimit);
        }
        state.max_total_bytes = restricted;
        Ok(restricted)
    }

    /// Atomically admit all resources. No partial reservation on refusal.
    pub fn lease(&self, request: WorkRequest) -> Result<WorkLease, WorkRefusal> {
        self.lease_for(LeaseClass::Optional, request)
    }

    /// Reserve time is accessible only through an explicit replay/fallback lane.
    /// Resource caps remain common; this is not a separate resource allowance.
    pub fn lease_for(
        &self,
        class: LeaseClass,
        request: WorkRequest,
    ) -> Result<WorkLease, WorkRefusal> {
        let mut state = self.0.state.lock().map_err(|_| WorkRefusal::Unavailable)?;
        state.receipt.attempted = state.receipt.attempted.saturating_add(1);
        let result = self
            .checkpoint_for(class)
            .and_then(|()| admit(&state, class, request));
        let (scratch, persistent, work, proposals) = match result {
            Ok(resources) => resources,
            Err(refusal) => {
                state.receipt.refused = state.receipt.refused.saturating_add(1);
                return Err(refusal);
            }
        };
        // Check the peak before mutating any reservation counters.
        let resident = scratch
            .checked_add(persistent)
            .ok_or(WorkRefusal::Overflow)?;
        let memory_claim = match &self.0.memory {
            Some(authority) => {
                let bytes = request
                    .scratch_bytes
                    .checked_add(request.persistent_bytes)
                    .ok_or(WorkRefusal::Overflow)?;
                match authority.reserve(bytes) {
                    Ok(claim) => Some(claim),
                    Err(refusal) => {
                        state.receipt.refused = state.receipt.refused.saturating_add(1);
                        return Err(refusal);
                    }
                }
            }
            None => None,
        };
        state.receipt.live_scratch_bytes = scratch;
        state.receipt.live_persistent_bytes = persistent;
        state.receipt.work_reserved = work;
        state.receipt.proposals_reserved = proposals;
        state.receipt.admitted = state.receipt.admitted.saturating_add(1);
        state.receipt.peak_leased_bytes = state.receipt.peak_leased_bytes.max(resident);
        Ok(WorkLease {
            budget: self.clone(),
            class,
            request,
            live: true,
            completed: false,
            materialized: false,
            memory_claim,
        })
    }

    pub fn receipt(&self) -> Result<WorkReceipt, WorkRefusal> {
        self.0
            .state
            .lock()
            .map(|state| state.receipt)
            .map_err(|_| WorkRefusal::Unavailable)
    }

    /// Optional-work checkpoint. Deadlines are immutable and need no mutex.
    pub fn checkpoint(&self) -> Result<(), WorkRefusal> {
        self.checkpoint_for(LeaseClass::Optional)
    }

    pub fn checkpoint_for(&self, class: LeaseClass) -> Result<(), WorkRefusal> {
        self.checkpoint_at(class, Instant::now())
    }

    fn checkpoint_at(&self, class: LeaseClass, now: Instant) -> Result<(), WorkRefusal> {
        if self
            .0
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(WorkRefusal::Cancelled);
        }
        let deadline = match class {
            LeaseClass::Optional => self.0.candidate_deadline,
            LeaseClass::Replay | LeaseClass::Fallback => self.0.final_deadline,
        };
        if deadline.is_some_and(|deadline| now >= deadline) {
            Err(WorkRefusal::Deadline)
        } else {
            Ok(())
        }
    }
}

fn admit(
    state: &State,
    class: LeaseClass,
    request: WorkRequest,
) -> Result<(usize, usize, u64, u64), WorkRefusal> {
    let scratch = state
        .receipt
        .live_scratch_bytes
        .checked_add(request.scratch_bytes)
        .ok_or(WorkRefusal::Overflow)?;
    let persistent = state
        .receipt
        .live_persistent_bytes
        .checked_add(request.persistent_bytes)
        .ok_or(WorkRefusal::Overflow)?;
    scratch
        .checked_add(persistent)
        .ok_or(WorkRefusal::Overflow)?;
    let work = state
        .receipt
        .work_reserved
        .checked_add(request.work)
        .ok_or(WorkRefusal::Overflow)?;
    let proposals = state
        .receipt
        .proposals_reserved
        .checked_add(request.proposals)
        .ok_or(WorkRefusal::Overflow)?;
    let reserve = if class == LeaseClass::Optional {
        state.reserve
    } else {
        WorkReserve::default()
    };
    admit_byte_totals(state, class, scratch, persistent)?;
    if work > state.max_work - reserve.work {
        return Err(WorkRefusal::WorkLimit);
    }
    if proposals > state.max_proposals - reserve.proposals {
        return Err(WorkRefusal::ProposalLimit);
    }
    Ok((scratch, persistent, work, proposals))
}

fn admit_byte_totals(
    state: &State,
    class: LeaseClass,
    scratch: usize,
    persistent: usize,
) -> Result<(), WorkRefusal> {
    let reserve = if class == LeaseClass::Optional {
        state.reserve
    } else {
        WorkReserve::default()
    };
    if scratch > state.max_scratch {
        return Err(WorkRefusal::ScratchLimit);
    }
    if persistent > state.max_persistent - reserve.persistent_bytes {
        return Err(WorkRefusal::PersistentLimit);
    }
    if scratch
        .checked_add(persistent)
        .ok_or(WorkRefusal::Overflow)?
        > state
            .max_total_bytes
            .saturating_sub(reserve.persistent_bytes)
    {
        return Err(WorkRefusal::MemoryLimit);
    }
    Ok(())
}

#[derive(Debug)]
pub struct WorkLease {
    budget: WorkBudget,
    class: LeaseClass,
    request: WorkRequest,
    live: bool,
    completed: bool,
    materialized: bool,
    memory_claim: Option<MemoryClaim>,
}

impl WorkLease {
    /// Reconcile allocation-capacity overages on this same pending lease.
    ///
    /// The caller must reconcile actual capacity before touching it and mark
    /// the lease materialized only after all participating buffers reconcile.
    /// Local scratch/combined limits, protected replay bytes, and the shared
    /// resident/model/query authority must all admit the addition atomically.
    /// Refusal preserves every existing claim; the caller discards private
    /// buffers before dropping their lease. Work/proposals/deadlines/classes
    /// and phase counts are unchanged. Materialized/completed leases are sealed.
    pub fn reserve_more_scratch(&mut self, additional: usize) -> Result<(), WorkRefusal> {
        if !self.live || self.completed || self.materialized {
            return Err(WorkRefusal::Unavailable);
        }
        self.checkpoint()?;
        if additional == 0 {
            return Ok(());
        }
        let request_scratch = self
            .request
            .scratch_bytes
            .checked_add(additional)
            .ok_or(WorkRefusal::Overflow)?;
        let mut state = self
            .budget
            .0
            .state
            .lock()
            .map_err(|_| WorkRefusal::Unavailable)?;
        self.checkpoint()?;
        let scratch = state
            .receipt
            .live_scratch_bytes
            .checked_add(additional)
            .ok_or(WorkRefusal::Overflow)?;
        let persistent = state.receipt.live_persistent_bytes;
        admit_byte_totals(&state, self.class, scratch, persistent)?;
        let resident = scratch
            .checked_add(persistent)
            .ok_or(WorkRefusal::Overflow)?;
        // Same budget -> memory-authority lock order as the original lease
        // admission. A global refusal occurs before any local debit.
        if let Some(claim) = &mut self.memory_claim {
            claim.grow_pending(additional)?;
        }
        state.receipt.live_scratch_bytes = scratch;
        state.receipt.peak_leased_bytes = state.receipt.peak_leased_bytes.max(resident);
        self.request.scratch_bytes = request_scratch;
        Ok(())
    }

    /// Reserve more primitive work within this same live phase before it is
    /// performed. Byte/proposal claims, phase counters and the original class
    /// deadline remain unchanged. Work is never refunded after completion,
    /// cancellation or drop; another phase spends the same aggregate limit.
    pub fn reserve_more_work(&mut self, additional: u64) -> Result<(), WorkRefusal> {
        if !self.live || self.completed {
            return Err(WorkRefusal::Unavailable);
        }
        self.checkpoint()?;
        let request_work = self
            .request
            .work
            .checked_add(additional)
            .ok_or(WorkRefusal::Overflow)?;
        let mut state = self
            .budget
            .0
            .state
            .lock()
            .map_err(|_| WorkRefusal::Unavailable)?;
        self.checkpoint()?;
        let (_, _, work, _) = admit(
            &state,
            self.class,
            WorkRequest {
                work: additional,
                ..WorkRequest::default()
            },
        )?;
        state.receipt.work_reserved = work;
        self.request.work = request_work;
        Ok(())
    }

    /// Record participating allocations as committed capacities. This does not
    /// release memory or assert physical page residency/RSS. A failure keeps the
    /// lease and all claims owned until the worker drops them.
    pub fn mark_materialized(&mut self) -> Result<(), WorkRefusal> {
        self.checkpoint()?;
        if let Some(claim) = &mut self.memory_claim {
            claim.materialize()?;
        }
        self.materialized = true;
        Ok(())
    }
    /// This lease's existing immutable deadline. Optional work sees the
    /// candidate deadline; replay/fallback see the original final deadline.
    /// Passing it into cooperative kernels grants no new time allowance.
    pub fn deadline(&self) -> Option<Instant> {
        match self.class {
            LeaseClass::Optional => self.budget.0.candidate_deadline,
            LeaseClass::Replay | LeaseClass::Fallback => self.budget.0.final_deadline,
        }
    }

    pub fn checkpoint(&self) -> Result<(), WorkRefusal> {
        self.budget.checkpoint_for(self.class)
    }

    /// Record completed work before dropping its scratch.
    pub fn complete(&mut self) -> Result<(), WorkRefusal> {
        self.checkpoint()?;
        if !self.completed {
            let mut state = self
                .budget
                .0
                .state
                .lock()
                .map_err(|_| WorkRefusal::Unavailable)?;
            state.receipt.completed = state.receipt.completed.saturating_add(1);
            self.completed = true;
        }
        Ok(())
    }

    /// Transfer retained bytes to a separately owned immutable result/cache.
    /// Scratch is released; persistent bytes stay charged until the last owner
    /// drops its reservation. Consuming a lease cannot duplicate an allowance.
    pub fn commit_persistent(mut self) -> Result<PersistentReservation, WorkRefusal> {
        let resident = self.persistent_guard()?;
        Ok(PersistentReservation(Arc::new(resident)))
    }

    /// Attach pre-priced immutable data to this same reservation allocation.
    /// Caller admission covers every data buffer and this owner's metadata.
    pub(crate) fn commit_retained_with<T>(
        mut self,
        build: impl FnOnce() -> T,
    ) -> Result<RetainedOwner<T>, WorkRefusal> {
        RetainedOwner::<T>::check_metadata(self.request.persistent_bytes)?;
        let resident = self.persistent_guard()?;
        // new_cyclic allocates the same Arc control/data block before calling
        // build, preserving post-allocation inherited-clock observation. Its
        // temporary Weak is ignored and cannot escape to a payload/caller.
        Ok(RetainedOwner(Arc::new_cyclic(move |_| RetainedData {
            data: build(),
            resident,
        })))
    }

    // Borrow the lease until its caller constructs the Arc, preserving the
    // original lease-budget lifetime across allocation and every refusal.
    fn persistent_guard(&mut self) -> Result<PersistentInner, WorkRefusal> {
        self.complete()?;
        {
            let mut state = self
                .budget
                .0
                .state
                .lock()
                .map_err(|_| WorkRefusal::Unavailable)?;
            if let Some(claim) = &mut self.memory_claim {
                claim.retain_only(self.request.persistent_bytes)?;
                claim.materialize()?;
            }
            release_bytes(
                &mut state.receipt.live_scratch_bytes,
                self.request.scratch_bytes,
            );
        }
        self.live = false;
        Ok(PersistentInner {
            budget: self.budget.clone(),
            bytes: self.request.persistent_bytes,
            _memory_claim: self.memory_claim.take(),
        })
    }
}

#[cfg(test)]
#[path = "work/capacity_tests.rs"]
mod capacity_tests;

impl Drop for WorkLease {
    fn drop(&mut self) {
        if self.live {
            if let Ok(mut state) = self.budget.0.state.lock() {
                release_bytes(
                    &mut state.receipt.live_scratch_bytes,
                    self.request.scratch_bytes,
                );
                release_bytes(
                    &mut state.receipt.live_persistent_bytes,
                    self.request.persistent_bytes,
                );
            }
        }
    }
}

#[cfg(test)]
mod incremental_work_tests {
    use super::*;

    #[test]
    fn incremental_work_same_phase_preserves_capacity_counts_and_protected_replay() {
        let budget = WorkBudget::new_reserved(
            None,
            8,
            0,
            10,
            0,
            Duration::ZERO,
            WorkReserve {
                work: 3,
                ..WorkReserve::default()
            },
        )
        .unwrap();
        let mut scan = budget
            .lease(WorkRequest {
                scratch_bytes: 8,
                work: 2,
                ..WorkRequest::default()
            })
            .unwrap();
        scan.reserve_more_work(5).unwrap();
        assert_eq!(scan.reserve_more_work(1), Err(WorkRefusal::WorkLimit));
        let receipt = budget.receipt().unwrap();
        assert_eq!(receipt.admitted, 1);
        assert_eq!(receipt.attempted, 1);
        assert_eq!(receipt.completed, 0);
        assert_eq!(receipt.work_reserved, 7);
        assert_eq!(receipt.live_scratch_bytes, 8);
        assert_eq!(receipt.proposals_reserved, 0);
        scan.complete().unwrap();
        assert_eq!(scan.reserve_more_work(0), Err(WorkRefusal::Unavailable));
        drop(scan);
        assert_eq!(budget.receipt().unwrap().work_reserved, 7);
        let mut replay = budget
            .lease_for(LeaseClass::Replay, WorkRequest::default())
            .unwrap();
        replay.reserve_more_work(3).unwrap();
        assert_eq!(replay.reserve_more_work(1), Err(WorkRefusal::WorkLimit));
        assert_eq!(budget.receipt().unwrap().work_reserved, 10);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    }

    #[test]
    fn incremental_work_cancellation_preserves_spent_work_and_live_claim_until_drop() {
        let memory = MemoryAuthority::new(8);
        let cancel = CancellationToken::default();
        let budget = WorkBudget::new_in(
            &memory,
            &cancel,
            WorkLimits {
                deadline: None,
                scratch_bytes: 8,
                persistent_bytes: 0,
                max_work: 10,
                max_proposals: 0,
                certificate_reserve: Duration::ZERO,
                reserve: WorkReserve::default(),
            },
        )
        .unwrap();
        let mut scan = budget
            .lease(WorkRequest {
                scratch_bytes: 8,
                work: 2,
                ..WorkRequest::default()
            })
            .unwrap();
        scan.reserve_more_work(3).unwrap();
        cancel.cancel();
        assert_eq!(scan.reserve_more_work(1), Err(WorkRefusal::Cancelled));
        assert_eq!(budget.receipt().unwrap().work_reserved, 5);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 8);
        drop(scan);
        assert_eq!(budget.receipt().unwrap().work_reserved, 5);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        let receipt = memory.receipt().unwrap();
        assert_eq!(receipt.live_pending_bytes, 0);
        assert_eq!(receipt.live_materialized_bytes, 0);
    }

    #[test]
    fn incremental_work_overflow_refuses_without_changing_original_ledger() {
        let budget = WorkBudget::new(None, 0, 0, u64::MAX, 0, Duration::ZERO).unwrap();
        let mut scan = budget
            .lease(WorkRequest {
                work: u64::MAX,
                ..WorkRequest::default()
            })
            .unwrap();
        assert_eq!(scan.reserve_more_work(1), Err(WorkRefusal::Overflow));
        assert_eq!(budget.receipt().unwrap().work_reserved, u64::MAX);
        assert_eq!(budget.receipt().unwrap().admitted, 1);
    }

    #[test]
    fn incremental_work_parallel_phases_cannot_each_spend_the_same_remaining_work() {
        let budget = WorkBudget::new(None, 0, 0, 10, 0, Duration::ZERO).unwrap();
        let first = budget.lease(WorkRequest::default()).unwrap();
        let second = budget.lease(WorkRequest::default()).unwrap();
        let results = std::thread::scope(|scope| {
            let a = scope.spawn(move || {
                let mut phase = first;
                phase.reserve_more_work(7)
            });
            let b = scope.spawn(move || {
                let mut phase = second;
                phase.reserve_more_work(7)
            });
            [a.join().unwrap(), b.join().unwrap()]
        });
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|r| **r == Err(WorkRefusal::WorkLimit))
                .count(),
            1
        );
        assert_eq!(budget.receipt().unwrap().work_reserved, 7);
        assert_eq!(budget.receipt().unwrap().admitted, 2);
    }
}

/// Clone shares ownership of one reservation, not another byte allocation.
#[derive(Debug, Clone)]
pub struct PersistentReservation(Arc<PersistentInner>);

impl PersistentReservation {
    /// Declared Arc owner/counter allocation made by `commit_persistent`.
    /// This follows this pinned toolchain's ArcInner layout (two AtomicUsize
    /// counters followed by the payload), not allocator bookkeeping or RSS.
    /// A producer can add this quote before minting its retained claim.
    pub fn allocation_request_bytes() -> Result<usize, WorkRefusal> {
        Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(Layout::new::<PersistentInner>())
            .map(|(layout, _)| layout.pad_to_align().size())
            .map_err(|_| WorkRefusal::Overflow)
    }

    #[must_use]
    pub fn bytes(&self) -> usize {
        self.0.bytes
    }

    /// A retained allocation must keep the original producer's work authority.
    /// Equal byte limits on an unrelated ledger cannot pay for its ownership.
    #[must_use]
    pub fn shares_ledger(&self, budget: &WorkBudget) -> bool {
        self.0.budget.shares_ledger(budget)
    }

    /// Retain an immutable producer's capacity while another admitted query
    /// uses the same global memory authority. This does not transfer or renew
    /// the producer's work ledger; callers requiring that identity still use
    /// `shares_ledger`. Unjoined or merely equal-limit authorities refuse.
    #[must_use]
    pub fn shares_memory_authority(&self, budget: &WorkBudget) -> bool {
        self.0.budget.shares_memory_authority(budget)
    }
}

#[derive(Debug)]
struct PersistentInner {
    budget: WorkBudget,
    bytes: usize,
    _memory_claim: Option<MemoryClaim>,
}

impl Drop for PersistentInner {
    fn drop(&mut self) {
        if let Ok(mut state) = self.budget.0.state.lock() {
            release_bytes(&mut state.receipt.live_persistent_bytes, self.bytes);
        }
    }
}

/// One enclosing allocation replaces the reservation's existing Arc payload.
/// Data drops before the unchanged guard, keeping all buffers charged through
/// their destruction. Cloning shares both data and the original byte claim.
pub(crate) struct RetainedOwner<T>(Arc<RetainedData<T>>);

struct RetainedData<T> {
    data: T,
    resident: PersistentInner,
}

impl<T> Clone for RetainedOwner<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}
impl<T> Deref for RetainedOwner<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0.data
    }
}
impl<T> RetainedOwner<T> {
    /// Checked owner/control/alignment bound inside the caller's existing
    /// fixed metadata. Four atomic control words preserve the existing report-owner
    /// allowance; Layout also covers alignment padding rather than assuming
    /// all retained types have pointer alignment. This is not a new byte fee.
    pub(crate) fn metadata_bytes() -> Result<usize, WorkRefusal> {
        let (layout, _) = Layout::new::<[std::sync::atomic::AtomicUsize; 4]>()
            .extend(Layout::new::<RetainedData<T>>())
            .map_err(|_| WorkRefusal::Overflow)?;
        Ok(layout.pad_to_align().size())
    }

    pub(crate) fn check_metadata(available: usize) -> Result<(), WorkRefusal> {
        if Self::metadata_bytes()? > available {
            return Err(WorkRefusal::PersistentLimit);
        }
        Ok(())
    }

    pub(crate) fn bytes(&self) -> usize {
        self.0.resident.bytes
    }

    pub(crate) fn shares_ledger(&self, budget: &WorkBudget) -> bool {
        self.0.resident.budget.shares_ledger(budget)
    }

    #[cfg(test)]
    pub(crate) fn get_mut(&mut self) -> Option<&mut T> {
        Arc::get_mut(&mut self.0).map(|owner| &mut owner.data)
    }
}

// Never panic during unwinding. An invariant failure keeps the ledger closed
// (all subsequent admissions refuse) instead of wrapping or under-accounting.
fn release_bytes(live: &mut usize, bytes: usize) {
    *live = live.checked_sub(bytes).unwrap_or(usize::MAX);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_lease_refuses_double_spending_and_preserves_completed_work() {
        let budget = WorkBudget::new(None, 100, 20, 10, 2, Duration::ZERO).unwrap();
        let sibling = budget.clone();
        let mut first = budget
            .lease(WorkRequest {
                scratch_bytes: 100,
                persistent_bytes: 20,
                work: 6,
                proposals: 1,
            })
            .unwrap();
        assert_eq!(
            sibling
                .lease(WorkRequest {
                    scratch_bytes: 1,
                    ..WorkRequest::default()
                })
                .unwrap_err(),
            WorkRefusal::ScratchLimit
        );
        first.complete().unwrap();
        drop(first);
        assert_eq!(
            sibling
                .lease(WorkRequest {
                    work: 5,
                    ..WorkRequest::default()
                })
                .unwrap_err(),
            WorkRefusal::WorkLimit
        );
        let second = sibling
            .lease(WorkRequest {
                work: 4,
                proposals: 1,
                ..WorkRequest::default()
            })
            .unwrap();
        drop(second);
        let receipt = budget.receipt().unwrap();
        assert_eq!(receipt.completed, 1);
        assert_eq!(receipt.work_reserved, 10);
        assert_eq!(receipt.peak_leased_bytes, 120);
        assert_eq!(receipt.live_scratch_bytes, 0);
        assert_eq!(receipt.live_persistent_bytes, 0);
    }

    #[test]
    #[expect(
        clippy::redundant_clone,
        reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
    )]
    fn persistent_reservations_release_only_after_the_last_owner() {
        let budget = WorkBudget::new(None, 4, 8, 1, 1, Duration::ZERO).unwrap();
        let lease = budget
            .lease(WorkRequest {
                scratch_bytes: 4,
                persistent_bytes: 8,
                ..WorkRequest::default()
            })
            .unwrap();
        let retained = lease.commit_persistent().unwrap();
        let equal_limits = WorkBudget::new(None, 4, 8, 1, 1, Duration::ZERO).unwrap();
        assert!(retained.shares_ledger(&budget));
        assert!(retained.shares_ledger(&budget.clone()));
        assert!(!retained.shares_ledger(&equal_limits));
        let second_owner = retained.clone();
        drop(retained);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 8);
        drop(second_owner);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    }

    #[test]
    fn certificate_reserve_stops_optional_work_before_the_final_deadline() {
        let final_deadline = Instant::now().checked_add(Duration::from_secs(30)).unwrap();
        let budget =
            WorkBudget::new(Some(final_deadline), 10, 10, 1, 1, Duration::from_secs(60)).unwrap();
        assert_eq!(budget.checkpoint(), Err(WorkRefusal::Deadline));
        assert_eq!(
            budget.lease(WorkRequest::default()).unwrap_err(),
            WorkRefusal::Deadline
        );
        assert_eq!(budget.receipt().unwrap().admitted, 0);
    }

    #[test]
    fn checked_resource_arithmetic_refuses_overflow() {
        assert_eq!(
            WorkBudget::new(None, usize::MAX, 1, 0, 0, Duration::ZERO).unwrap_err(),
            WorkRefusal::Overflow
        );
        let budget =
            WorkBudget::new(None, usize::MAX, 0, u64::MAX, u64::MAX, Duration::ZERO).unwrap();
        let first = budget
            .lease(WorkRequest {
                scratch_bytes: usize::MAX,
                work: u64::MAX,
                ..WorkRequest::default()
            })
            .unwrap();
        assert_eq!(
            budget
                .lease(WorkRequest {
                    scratch_bytes: 1,
                    ..WorkRequest::default()
                })
                .unwrap_err(),
            WorkRefusal::Overflow
        );
        drop(first);
        assert_eq!(
            budget
                .lease(WorkRequest {
                    work: 1,
                    ..WorkRequest::default()
                })
                .unwrap_err(),
            WorkRefusal::Overflow
        );
    }
    #[test]
    fn replay_and_fallback_can_use_reserve_but_not_extend_the_final_deadline() {
        let now = Instant::now();
        let final_deadline = now.checked_add(Duration::from_secs(30)).unwrap();
        let budget =
            WorkBudget::new(Some(final_deadline), 10, 10, 4, 4, Duration::from_secs(60)).unwrap();
        assert_eq!(budget.checkpoint(), Err(WorkRefusal::Deadline));
        for class in [LeaseClass::Replay, LeaseClass::Fallback] {
            let mut lease = budget
                .lease_for(
                    class,
                    WorkRequest {
                        work: 1,
                        ..WorkRequest::default()
                    },
                )
                .unwrap();
            lease.complete().unwrap();
            assert_eq!(
                budget.checkpoint_at(class, final_deadline),
                Err(WorkRefusal::Deadline)
            );
        }
        assert_eq!(budget.receipt().unwrap().completed, 2);
        assert_eq!(budget.receipt().unwrap().work_reserved, 2);
    }

    #[test]
    fn deterministic_deadline_boundaries_and_unwind_release_are_checked() {
        let now = Instant::now();
        let final_deadline = now.checked_add(Duration::from_secs(60)).unwrap();
        let budget =
            WorkBudget::new(Some(final_deadline), 10, 10, 4, 4, Duration::from_secs(30)).unwrap();
        let optional = final_deadline.checked_sub(Duration::from_secs(30)).unwrap();
        assert!(budget.checkpoint_at(LeaseClass::Optional, now).is_ok());
        assert_eq!(
            budget.checkpoint_at(LeaseClass::Optional, optional),
            Err(WorkRefusal::Deadline)
        );
        assert!(budget.checkpoint_at(LeaseClass::Replay, optional).is_ok());
        let result = std::panic::catch_unwind(|| {
            let _lease = budget
                .lease(WorkRequest {
                    scratch_bytes: 10,
                    persistent_bytes: 10,
                    ..WorkRequest::default()
                })
                .unwrap();
            panic!("adversarial proposal panic");
        });
        assert!(result.is_err());
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    }

    #[test]
    fn lease_deadline_preserves_selected_class_and_original_reserve() {
        let final_deadline = Instant::now() + Duration::from_secs(60);
        let reserve = Duration::from_secs(20);
        let budget = WorkBudget::new(Some(final_deadline), 1, 1, 1, 1, reserve).unwrap();
        for class in [
            LeaseClass::Optional,
            LeaseClass::Replay,
            LeaseClass::Fallback,
        ] {
            let lease = budget.lease_for(class, WorkRequest::default()).unwrap();
            let expected = if class == LeaseClass::Optional {
                Some(final_deadline.checked_sub(reserve).unwrap())
            } else {
                Some(final_deadline)
            };
            assert_eq!(lease.deadline(), expected);
            drop(lease);
        }
        let budget = WorkBudget::new(None, 1, 1, 1, 1, reserve).unwrap();
        assert_eq!(
            budget.lease(WorkRequest::default()).unwrap().deadline(),
            None
        );
    }
    #[test]
    fn optional_work_cannot_spend_predicted_replay_capacity() {
        let budget = WorkBudget::new_reserved(
            None,
            16,
            20,
            10,
            3,
            Duration::ZERO,
            WorkReserve {
                persistent_bytes: 8,
                work: 4,
                proposals: 1,
            },
        )
        .unwrap();
        let retained = budget
            .lease(WorkRequest {
                persistent_bytes: 12,
                work: 6,
                proposals: 2,
                ..WorkRequest::default()
            })
            .unwrap()
            .commit_persistent()
            .unwrap();
        assert_eq!(
            budget
                .lease(WorkRequest {
                    work: 1,
                    ..WorkRequest::default()
                })
                .unwrap_err(),
            WorkRefusal::WorkLimit
        );
        let mut replay = budget
            .lease_for(
                LeaseClass::Replay,
                WorkRequest {
                    persistent_bytes: 8,
                    work: 4,
                    proposals: 1,
                    ..WorkRequest::default()
                },
            )
            .unwrap();
        replay.complete().unwrap();
        assert_eq!(budget.receipt().unwrap().work_reserved, 10);
        assert_eq!(budget.receipt().unwrap().proposals_reserved, 3);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 20);
        assert_eq!(
            budget
                .lease_for(
                    LeaseClass::Fallback,
                    WorkRequest {
                        work: 1,
                        ..WorkRequest::default()
                    }
                )
                .unwrap_err(),
            WorkRefusal::WorkLimit
        );
        drop(replay);
        drop(retained);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    }

    #[test]
    fn combined_byte_ceiling_counts_both_lanes_and_retained_last_owners() {
        let budget = WorkBudget::new(None, 100, 100, 10, 2, Duration::ZERO).unwrap();
        let mut first = budget
            .lease(WorkRequest {
                scratch_bytes: 80,
                persistent_bytes: 40,
                work: 1,
                ..WorkRequest::default()
            })
            .unwrap();
        first.mark_materialized().unwrap();
        let sibling = budget.clone();
        assert_eq!(sibling.tighten_total_byte_limit(120).unwrap(), 120);
        assert_eq!(
            sibling
                .lease(WorkRequest {
                    scratch_bytes: 1,
                    ..WorkRequest::default()
                })
                .unwrap_err(),
            WorkRefusal::MemoryLimit
        );
        let retained = first.commit_persistent().unwrap();
        let last = retained.clone();
        drop(retained);
        let scratch = sibling
            .lease(WorkRequest {
                scratch_bytes: 80,
                ..WorkRequest::default()
            })
            .unwrap();
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 40);
        assert_eq!(
            budget
                .lease(WorkRequest {
                    persistent_bytes: 1,
                    ..WorkRequest::default()
                })
                .unwrap_err(),
            WorkRefusal::MemoryLimit
        );
        drop(last);
        let remainder = budget
            .lease(WorkRequest {
                persistent_bytes: 40,
                ..WorkRequest::default()
            })
            .unwrap();
        drop(remainder);
        drop(scratch);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(budget.receipt().unwrap().work_reserved, 1);
    }

    #[test]
    #[expect(
        clippy::redundant_clone,
        reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
    )]
    fn combined_byte_ceiling_refusal_is_atomic_and_restriction_cannot_be_relaxed() {
        let budget = WorkBudget::new(None, 100, 100, 10, 2, Duration::ZERO).unwrap();
        let first = budget
            .lease(WorkRequest {
                scratch_bytes: 80,
                persistent_bytes: 40,
                ..WorkRequest::default()
            })
            .unwrap();
        let before = budget.receipt().unwrap();
        assert_eq!(
            budget.tighten_total_byte_limit(119),
            Err(WorkRefusal::MemoryLimit)
        );
        assert_eq!(budget.receipt().unwrap(), before);
        let allowed = budget
            .lease(WorkRequest {
                persistent_bytes: 1,
                ..WorkRequest::default()
            })
            .unwrap();
        drop(allowed);
        assert_eq!(budget.tighten_total_byte_limit(120).unwrap(), 120);
        assert_eq!(budget.clone().tighten_total_byte_limit(200).unwrap(), 120);
        assert_eq!(
            budget
                .lease(WorkRequest {
                    persistent_bytes: 1,
                    ..WorkRequest::default()
                })
                .unwrap_err(),
            WorkRefusal::MemoryLimit
        );
        drop(first);
        assert_eq!(budget.tighten_total_byte_limit(0).unwrap(), 0);
        assert_eq!(budget.tighten_total_byte_limit(200).unwrap(), 0);
        let work_only = budget
            .lease(WorkRequest {
                work: 1,
                ..WorkRequest::default()
            })
            .unwrap();
        drop(work_only);
        let expired = WorkBudget::new(Some(Instant::now()), 10, 10, 0, 0, Duration::ZERO).unwrap();
        assert_eq!(
            expired.tighten_total_byte_limit(1),
            Err(WorkRefusal::Deadline)
        );
        assert_eq!(expired.0.state.lock().unwrap().max_total_bytes, 20);
    }

    #[test]
    fn combined_byte_ceiling_is_atomic_across_concurrent_clones() {
        use std::sync::{Arc, Barrier};
        const WORKERS: usize = 16;
        let budget = WorkBudget::new(None, 1000, 1000, 100, 0, Duration::ZERO).unwrap();
        budget.tighten_total_byte_limit(30).unwrap();
        let start = Arc::new(Barrier::new(WORKERS));
        let attempted = Arc::new(Barrier::new(WORKERS));
        let admitted = std::thread::scope(|scope| {
            let mut joins = Vec::new();
            for _ in 0..WORKERS {
                let own = budget.clone();
                let start = start.clone();
                let attempted = attempted.clone();
                joins.push(scope.spawn(move || {
                    start.wait();
                    let lease = own.lease(WorkRequest {
                        scratch_bytes: 7,
                        persistent_bytes: 3,
                        work: 1,
                        ..WorkRequest::default()
                    });
                    let admitted = lease.is_ok();
                    attempted.wait();
                    drop(lease);
                    usize::from(admitted)
                }));
            }
            joins
                .into_iter()
                .map(|join| join.join().unwrap())
                .sum::<usize>()
        });
        assert_eq!(admitted, 3);
        let receipt = budget.receipt().unwrap();
        assert_eq!(receipt.peak_leased_bytes, 30);
        assert_eq!(receipt.admitted, 3);
        assert_eq!(receipt.refused, 13);
        assert_eq!(receipt.work_reserved, 3);
        assert_eq!(receipt.live_scratch_bytes, 0);
        assert_eq!(receipt.live_persistent_bytes, 0);
    }

    #[test]
    #[expect(
        clippy::redundant_clone,
        reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
    )]
    fn combined_byte_ceiling_preserves_replay_reserve_across_both_byte_lanes() {
        let budget = WorkBudget::new_reserved(
            None,
            100,
            100,
            100,
            0,
            Duration::ZERO,
            WorkReserve {
                persistent_bytes: 40,
                ..WorkReserve::default()
            },
        )
        .unwrap();
        assert_eq!(budget.tighten_total_byte_limit(120).unwrap(), 120);
        assert_eq!(
            budget
                .lease(WorkRequest {
                    scratch_bytes: 100,
                    persistent_bytes: 20,
                    ..WorkRequest::default()
                })
                .unwrap_err(),
            WorkRefusal::MemoryLimit
        );
        let optional = budget
            .lease(WorkRequest {
                scratch_bytes: 60,
                persistent_bytes: 20,
                ..WorkRequest::default()
            })
            .unwrap();
        let replay = budget
            .clone()
            .lease_for(
                LeaseClass::Replay,
                WorkRequest {
                    persistent_bytes: 40,
                    ..WorkRequest::default()
                },
            )
            .unwrap();
        assert_eq!(budget.receipt().unwrap().peak_leased_bytes, 120);
        assert_eq!(
            budget
                .lease_for(
                    LeaseClass::Fallback,
                    WorkRequest {
                        scratch_bytes: 1,
                        ..WorkRequest::default()
                    },
                )
                .unwrap_err(),
            WorkRefusal::MemoryLimit
        );
        drop(replay);
        drop(optional);

        assert_eq!(budget.tighten_total_byte_limit(20).unwrap(), 20);
        assert_eq!(
            budget
                .lease(WorkRequest {
                    scratch_bytes: 1,
                    ..WorkRequest::default()
                })
                .unwrap_err(),
            WorkRefusal::MemoryLimit
        );
        let work_only = budget
            .lease(WorkRequest {
                work: 1,
                ..WorkRequest::default()
            })
            .unwrap();
        let fallback = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: 20,
                    ..WorkRequest::default()
                },
            )
            .unwrap();
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 20);
        drop(fallback);
        drop(work_only);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    }
}

#[cfg(test)]
#[path = "work_retained_tests.rs"]
mod retained_tests;
