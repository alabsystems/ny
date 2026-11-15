// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Allocator controls for a source-bound PRIVATE primitive test instance.
//! The included modules are not ny_propagate's library types. Public library
//! commit is compared separately; real whole-chain owners still need their
//! own allocator totals, numerical/resource controls and final qualification.

#[path = "support/certificate_allocations.rs"]
mod allocations;
#[path = "../src/certify/residency.rs"]
mod residency;
#[path = "../src/certify/work.rs"]
mod work;

// Preserve the included units' original crate paths. These aliases refer to
// this private source instance; the public-library commit is compared separately.
mod certify {
    pub(crate) mod residency {
        pub(crate) use crate::residency::{CancellationToken, MemoryAuthority};
    }
    pub(crate) mod work {
        pub(crate) use crate::work::{
            LeaseClass, WorkBudget, WorkLimits, WorkRequest, WorkReserve,
        };
    }
}

use allocations::{measure, snapshot, watch_deallocation, Allocation, Counts};
use ny_propagate::certify::{residency as library_residency, work as library_work};
use residency::{CancellationToken, MemoryAuthority};
use std::cell::Cell;
use std::hint::black_box;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::time::Duration;
use work::{LeaseClass, RetainedOwner, WorkBudget, WorkLimits, WorkRequest, WorkReserve};

// Existing original_affine_chain::META, not a new owner allowance. These
// controls measure the generic primitive, not private paid-cover allocations.
const FACT_METADATA: usize = 16 * 1024;
const SCRATCH: usize = 32;
const WORK: u64 = 7;
const PROPOSALS: u64 = 1;

fn request() -> WorkRequest {
    WorkRequest {
        scratch_bytes: SCRATCH,
        persistent_bytes: FACT_METADATA,
        work: WORK,
        proposals: PROPOSALS,
    }
}

fn resources() -> (WorkBudget, MemoryAuthority) {
    let memory = MemoryAuthority::new(FACT_METADATA + SCRATCH);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: SCRATCH,
            persistent_bytes: FACT_METADATA,
            max_work: WORK,
            max_proposals: PROPOSALS,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    assert!(budget.memory_authority().unwrap().shares_authority(&memory));
    (budget, memory)
}

struct Payload<'a> {
    budget: &'a WorkBudget,
    memory: &'a MemoryAuthority,
    drops: &'a Cell<usize>,
    charged_at_drop: &'a Cell<bool>,
    words: [u64; 4],
}
impl Drop for Payload<'_> {
    fn drop(&mut self) {
        // Record instead of panicking from a destructor during unwinding.
        let charged = self.budget.receipt().is_ok_and(|state| {
            state.live_persistent_bytes == FACT_METADATA && state.live_scratch_bytes == 0
        }) && self.memory.receipt().is_ok_and(|state| {
            state.live_materialized_bytes == FACT_METADATA && state.live_pending_bytes == 0
        });
        self.charged_at_drop.set(charged);
        self.drops.set(self.drops.get().saturating_add(1));
    }
}

#[repr(align(4096))]
struct HighAligned<'a> {
    payload: Payload<'a>,
}

fn assert_committed(budget: &WorkBudget, memory: &MemoryAuthority) {
    let work = budget.receipt().unwrap();
    assert_eq!(work.completed, 1);
    assert_eq!(work.work_reserved, WORK);
    assert_eq!(work.proposals_reserved, PROPOSALS);
    assert_eq!(work.live_scratch_bytes, 0);
    assert_eq!(work.live_persistent_bytes, FACT_METADATA);
    let bytes = memory.receipt().unwrap();
    assert_eq!(bytes.live_pending_bytes, 0);
    assert_eq!(bytes.live_materialized_bytes, FACT_METADATA);
}

fn assert_released(budget: &WorkBudget, memory: &MemoryAuthority, completed: u64) {
    let work = budget.receipt().unwrap();
    assert_eq!(work.completed, completed);
    assert_eq!(work.work_reserved, WORK);
    assert_eq!(work.proposals_reserved, PROPOSALS);
    assert_eq!(work.live_scratch_bytes, 0);
    assert_eq!(work.live_persistent_bytes, 0);
    let bytes = memory.receipt().unwrap();
    assert_eq!(bytes.live_pending_bytes, 0);
    assert_eq!(bytes.live_materialized_bytes, 0);
}

fn watch_single_allocated_owner() -> (Counts, Allocation) {
    let entered = snapshot().unwrap();
    assert_eq!(entered.allocations, 1);
    assert_eq!(entered.reallocations, 0);
    assert_eq!(entered.deallocations, 0);
    assert_eq!(entered.live_allocations(), 1);
    assert!(!entered.overflowed);
    let owner = entered.single_live_allocation().unwrap();
    assert_eq!(entered.requested_bytes, owner.bytes);
    assert!(watch_deallocation(owner.address));
    (entered, owner)
}

fn actual_owner_case<T>(
    label: &str,
    budget: &WorkBudget,
    memory: &MemoryAuthority,
    build: impl FnOnce() -> T,
    inspect: impl FnOnce(&T),
) {
    let bound = RetainedOwner::<T>::metadata_bytes().unwrap();
    assert!(bound <= FACT_METADATA);
    let lease = budget.lease_for(LeaseClass::Fallback, request()).unwrap();
    assert_eq!(
        memory.receipt().unwrap().live_pending_bytes,
        FACT_METADATA + SCRATCH
    );
    let factory = Cell::new(None);
    let ((record, payload_offset), counts) = measure(|| {
        let owner = lease
            .commit_retained_with(|| {
                // First factory action: prove an actual sole allocator request
                // already exists, before constructing or observing payload data.
                factory.set(Some(watch_single_allocated_owner()));
                assert_committed(budget, memory);
                build()
            })
            .unwrap();
        let (entered, record) = factory.get().unwrap();
        let payload_address = std::ptr::from_ref(&*owner).addr();
        let minted = snapshot().unwrap();
        assert_eq!(minted.allocations, entered.allocations);
        assert_eq!(minted.reallocations, 0);
        assert_eq!(minted.allocation_containing(payload_address), Some(record));
        assert_eq!(payload_address % align_of::<T>(), 0);
        assert!(
            payload_address.checked_add(size_of::<T>()).unwrap()
                <= record.address.checked_add(record.bytes).unwrap()
        );
        assert!(record.bytes <= bound);
        assert_eq!(owner.bytes(), FACT_METADATA);
        assert!(owner.shares_ledger(budget));
        inspect(&owner);
        let before_clone = snapshot().unwrap();
        let alias = black_box(&owner).clone();
        assert!(std::ptr::eq(&raw const *owner, &raw const *alias));
        assert_eq!(snapshot().unwrap(), before_clone);
        drop(owner);
        assert_eq!(snapshot().unwrap(), before_clone);
        assert_committed(budget, memory);
        drop(alias);
        assert_released(budget, memory, 1);
        let released = snapshot().unwrap();
        assert_eq!(released.watched_deallocations(record.address), Some(1));
        assert!(released.allocation_containing(record.address).is_none());
        (record, payload_address - record.address)
    })
    .unwrap();
    assert_eq!(counts.allocations, 1);
    assert_eq!(counts.reallocations, 0);
    assert_eq!(counts.deallocations, 1);
    assert_eq!(counts.live_allocations(), 0);
    assert_eq!(counts.live_requested_bytes(), Some(0));
    assert_eq!(counts.watched_deallocations(record.address), Some(1));
    assert!(!counts.overflowed);
    println!(
        "{{\"kind\":\"source_instance_retained_owner\",\"case\":\"{label}\",\"owner_request\":{},\"payload_bytes\":{},\"payload_alignment\":{},\"payload_offset\":{payload_offset},\"owner_handle_bytes\":{},\"handle_plus_owner_request\":{},\"checked_metadata_bound\":{bound},\"existing_fact_metadata\":{FACT_METADATA},\"factory_entry_allocations\":1,\"clone_allocations\":0,\"watched_owner_deallocations\":1}}",
        record.bytes,
        size_of::<T>(),
        align_of::<T>(),
        size_of::<RetainedOwner<T>>(),
        record.bytes + size_of::<RetainedOwner<T>>(),
    );
}

#[test]
fn source_instance_normal_and_high_alignment_owner_allocate_before_factory_and_share_last_claim() {
    let (budget, memory) = resources();
    let drops = Cell::new(0);
    let charged = Cell::new(false);
    actual_owner_case(
        "normal",
        &budget,
        &memory,
        || Payload {
            budget: &budget,
            memory: &memory,
            drops: &drops,
            charged_at_drop: &charged,
            words: [2, 3, 5, 7],
        },
        |payload| assert_eq!(payload.words, [2, 3, 5, 7]),
    );
    assert_eq!(drops.get(), 1);
    assert!(charged.get());

    let (budget, memory) = resources();
    let drops = Cell::new(0);
    let charged = Cell::new(false);
    actual_owner_case(
        "align4096",
        &budget,
        &memory,
        || HighAligned {
            payload: Payload {
                budget: &budget,
                memory: &memory,
                drops: &drops,
                charged_at_drop: &charged,
                words: [11, 13, 17, 19],
            },
        },
        |payload| assert_eq!(payload.payload.words, [11, 13, 17, 19]),
    );
    assert_eq!(drops.get(), 1);
    assert!(charged.get());
}

#[test]
fn source_instance_panicking_factory_releases_watched_owner_once_without_refunding_spent_work() {
    let (budget, memory) = resources();
    let lease = budget.lease_for(LeaseClass::Fallback, request()).unwrap();
    let drops = Cell::new(0);
    let charged = Cell::new(false);
    let factory = Cell::new(None);
    // Preallocate the panic payload and bypass the process-global panic hook.
    // Unwinder allocations remain visible; only the watched owner is called
    // an owner release, rather than subtracting or ignoring runtime events.
    let panic_payload: Box<dyn std::any::Any + Send> = Box::new("retained owner factory control");
    let (outcome, counts) = measure(|| {
        catch_unwind(AssertUnwindSafe(|| {
            let _: RetainedOwner<Payload<'_>> = lease
                .commit_retained_with(|| {
                    factory.set(Some(watch_single_allocated_owner()));
                    assert_committed(&budget, &memory);
                    let _payload = Payload {
                        budget: &budget,
                        memory: &memory,
                        drops: &drops,
                        charged_at_drop: &charged,
                        words: [2, 3, 5, 7],
                    };
                    resume_unwind(panic_payload);
                })
                .unwrap();
        }))
    })
    .unwrap();
    assert!(outcome.is_err());
    let (entered, owner) = factory.get().unwrap();
    assert_eq!(entered.allocations, 1);
    assert_eq!(counts.watched_deallocations(owner.address), Some(1));
    // An unrelated unwind allocation may legally reuse the retired address.
    // The generation-specific watch, not current address occupancy, is decisive.
    assert_eq!(drops.get(), 1);
    assert!(charged.get());
    assert_released(&budget, &memory, 1);
    assert!(!counts.overflowed);
    println!(
        "{{\"kind\":\"source_instance_factory_panic\",\"owner_request\":{},\"factory_entry_allocations\":1,\"watched_owner_deallocations\":1,\"all_observed_allocations\":{},\"all_observed_deallocations\":{},\"live_observed_allocations\":{}}}",
        owner.bytes,
        counts.allocations,
        counts.deallocations,
        counts.live_allocations(),
    );
    drop(outcome);
}

#[test]
fn source_instance_insufficient_existing_metadata_refuses_before_factory_or_owner_allocation() {
    #[repr(align(16384))]
    struct TooAligned;
    assert!(RetainedOwner::<TooAligned>::metadata_bytes().unwrap() > FACT_METADATA);
    let (budget, memory) = resources();
    let lease = budget.lease_for(LeaseClass::Fallback, request()).unwrap();
    let called = Cell::new(false);
    let (refused, counts) = measure(|| {
        let result = lease.commit_retained_with(|| {
            called.set(true);
            TooAligned
        });
        matches!(result, Err(work::WorkRefusal::PersistentLimit))
    })
    .unwrap();
    assert!(refused);
    assert!(!called.get());
    assert!(counts.is_empty());
    assert_released(&budget, &memory, 0);
}

// This macro invokes two DIFFERENT concrete public APIs; it neither converts
// private source-instance owners into library owners nor substitutes receipts.
macro_rules! public_commit_observation {
    ($w:ident, $r:ident) => {{
        let memory = $r::MemoryAuthority::new(FACT_METADATA + SCRATCH);
        let cancel = $r::CancellationToken::default();
        let budget = $w::WorkBudget::new_in(
            &memory,
            &cancel,
            $w::WorkLimits {
                deadline: None,
                scratch_bytes: SCRATCH,
                persistent_bytes: FACT_METADATA,
                max_work: WORK,
                max_proposals: PROPOSALS,
                certificate_reserve: Duration::ZERO,
                reserve: $w::WorkReserve::default(),
            },
        )
        .unwrap();
        let lease = budget
            .lease_for(
                $w::LeaseClass::Fallback,
                $w::WorkRequest {
                    scratch_bytes: SCRATCH,
                    persistent_bytes: FACT_METADATA,
                    work: WORK,
                    proposals: PROPOSALS,
                },
            )
            .unwrap();
        let (owner_record, counts) = measure(|| {
            let owner = lease.commit_persistent().unwrap();
            let (minted, record) = watch_single_allocated_owner();
            assert_eq!(
                $w::PersistentReservation::allocation_request_bytes().unwrap(),
                record.bytes
            );
            assert!(owner.shares_memory_authority(&budget));
            assert_eq!(owner.bytes(), FACT_METADATA);
            assert!(owner.shares_ledger(&budget));
            let committed = budget.receipt().unwrap();
            assert_eq!(committed.completed, 1);
            assert_eq!(committed.live_scratch_bytes, 0);
            assert_eq!(committed.live_persistent_bytes, FACT_METADATA);
            assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
            assert_eq!(
                memory.receipt().unwrap().live_materialized_bytes,
                FACT_METADATA
            );
            let before_clone = snapshot().unwrap();
            let alias = black_box(&owner).clone();
            assert_eq!(snapshot().unwrap(), before_clone);
            drop(owner);
            assert_eq!(snapshot().unwrap(), before_clone);
            assert_eq!(
                budget.receipt().unwrap().live_persistent_bytes,
                FACT_METADATA
            );
            assert_eq!(
                memory.receipt().unwrap().live_materialized_bytes,
                FACT_METADATA
            );
            assert!(alias.shares_ledger(&budget));
            drop(alias);
            assert_eq!(
                snapshot().unwrap().watched_deallocations(record.address),
                Some(1)
            );
            assert_eq!(minted.allocations, 1);
            record
        })
        .unwrap();
        assert_eq!(counts.allocations, 1);
        assert_eq!(counts.reallocations, 0);
        assert_eq!(counts.deallocations, 1);
        assert_eq!(counts.live_allocations(), 0);
        assert_eq!(counts.live_requested_bytes(), Some(0));
        assert!(!counts.overflowed);
        let w = budget.receipt().unwrap();
        let m = memory.receipt().unwrap();
        assert_eq!(w.live_scratch_bytes, 0);
        assert_eq!(w.live_persistent_bytes, 0);
        assert_eq!(w.work_reserved, WORK);
        assert_eq!(w.proposals_reserved, PROPOSALS);
        assert_eq!(m.live_pending_bytes, 0);
        assert_eq!(m.live_materialized_bytes, 0);
        (
            owner_record.bytes,
            (
                w.attempted,
                w.admitted,
                w.refused,
                w.completed,
                w.work_reserved,
                w.proposals_reserved,
                w.live_scratch_bytes,
                w.live_persistent_bytes,
                w.peak_leased_bytes,
            ),
            (
                m.attempted,
                m.admitted,
                m.refused,
                m.live_pending_bytes,
                m.live_materialized_bytes,
                m.peak_leased_bytes,
            ),
        )
    }};
}

#[test]
fn source_instance_public_commit_matches_real_library_public_commit_allocations_and_receipts() {
    let included = public_commit_observation!(work, residency);
    let library = public_commit_observation!(library_work, library_residency);
    assert_eq!(included, library);
    println!(
        "{{\"kind\":\"public_commit_comparison\",\"included_source_owner_request\":{},\"real_library_owner_request\":{},\"allocations_per_commit\":1,\"clone_allocations\":0,\"last_alias_deallocations\":1,\"receipts_equal\":true}}",
        included.0, library.0,
    );
}
