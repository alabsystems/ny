// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

fn resources<T>() -> (WorkBudget, MemoryAuthority, CancellationToken, WorkRequest) {
    let request = WorkRequest {
        scratch_bytes: 32,
        persistent_bytes: RetainedOwner::<T>::metadata_bytes().unwrap(),
        work: 7,
        proposals: 1,
    };
    let memory = MemoryAuthority::new(request.scratch_bytes + request.persistent_bytes);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: request.scratch_bytes,
            persistent_bytes: request.persistent_bytes,
            max_work: request.work,
            max_proposals: request.proposals,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, cancel, request)
}

#[test]
fn retained_owner_shared_claim_keeps_original_ledger_and_last_alias() {
    let (budget, memory, _, request) = resources::<[u8; 3]>();
    let foreign = WorkBudget::new(
        None,
        request.scratch_bytes,
        request.persistent_bytes,
        request.work,
        request.proposals,
        Duration::ZERO,
    )
    .unwrap();
    let lease = budget.lease(request).unwrap();
    assert_eq!(
        memory.receipt().unwrap().live_pending_bytes,
        request.scratch_bytes + request.persistent_bytes
    );
    let owner = lease.commit_retained_with(|| [1_u8, 2, 3]).unwrap();
    assert_eq!(&*owner, &[1, 2, 3]);
    assert_eq!(owner.bytes(), request.persistent_bytes);
    assert!(owner.shares_ledger(&budget));
    let same_budget = budget.clone();
    assert!(owner.shares_ledger(&same_budget));
    drop(same_budget);
    assert!(!owner.shares_ledger(&foreign));
    let last = owner.clone();
    assert!(std::ptr::eq(&raw const *owner, &raw const *last));
    assert_eq!(budget.receipt().unwrap().completed, 1);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        request.persistent_bytes
    );
    drop(owner);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        request.persistent_bytes
    );
    drop(last);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    assert_eq!(budget.receipt().unwrap().work_reserved, request.work);
    assert_eq!(
        budget.receipt().unwrap().proposals_reserved,
        request.proposals
    );
}

#[test]
fn retained_owner_public_commit_and_typed_commit_keep_identical_receipts() {
    for materialized in [false, true] {
        let (old_budget, old_memory, _, request) = resources::<[u8; 3]>();
        let (new_budget, new_memory, _, _) = resources::<[u8; 3]>();
        let mut old = old_budget.lease(request).unwrap();
        let mut new = new_budget.lease(request).unwrap();
        if materialized {
            old.mark_materialized().unwrap();
            new.mark_materialized().unwrap();
        }
        let old = old.commit_persistent().unwrap();
        let new = new.commit_retained_with(|| [1_u8, 2, 3]).unwrap();
        assert_eq!(old_budget.receipt().unwrap(), new_budget.receipt().unwrap());
        assert_eq!(old_memory.receipt().unwrap(), new_memory.receipt().unwrap());
        drop(old);
        drop(new);
        assert_eq!(old_budget.receipt().unwrap(), new_budget.receipt().unwrap());
        assert_eq!(old_memory.receipt().unwrap(), new_memory.receipt().unwrap());
    }
}

struct DropProbe {
    budget: WorkBudget,
    memory: MemoryAuthority,
    bytes: usize,
    observed: Arc<AtomicBool>,
}
impl Drop for DropProbe {
    fn drop(&mut self) {
        assert_eq!(
            self.budget.receipt().unwrap().live_persistent_bytes,
            self.bytes
        );
        assert_eq!(
            self.memory.receipt().unwrap().live_materialized_bytes,
            self.bytes
        );
        self.observed.store(true, Ordering::SeqCst);
    }
}

#[test]
fn retained_owner_payload_drops_before_the_same_original_guard() {
    let (budget, memory, _, request) = resources::<DropProbe>();
    let observed = Arc::new(AtomicBool::new(false));
    let owner = budget
        .lease(request)
        .unwrap()
        .commit_retained_with(|| DropProbe {
            budget: budget.clone(),
            memory: memory.clone(),
            bytes: request.persistent_bytes,
            observed: observed.clone(),
        })
        .unwrap();
    let last = owner.clone();
    drop(owner);
    assert!(!observed.load(Ordering::SeqCst));
    drop(last);
    assert!(observed.load(Ordering::SeqCst));
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}

#[test]
fn retained_owner_bound_and_alignment_refuse_without_completion_or_factory() {
    #[repr(align(256))]
    struct Aligned([u8; 1]);
    let (budget, memory, _, request) = resources::<Aligned>();
    assert_eq!(request.persistent_bytes % align_of::<Aligned>(), 0);
    assert!(request.persistent_bytes >= size_of::<RetainedData<Aligned>>());
    assert_eq!(
        RetainedOwner::<Aligned>::check_metadata(request.persistent_bytes - 1),
        Err(WorkRefusal::PersistentLimit)
    );
    assert_eq!(budget.receipt().unwrap().admitted, 0);
    let refused = WorkRequest {
        persistent_bytes: request.persistent_bytes - 1,
        ..request
    };
    let ran = AtomicBool::new(false);
    let result = budget.lease(refused).unwrap().commit_retained_with(|| {
        ran.store(true, Ordering::SeqCst);
        Aligned([3])
    });
    assert!(matches!(result, Err(WorkRefusal::PersistentLimit)));
    assert!(!ran.load(Ordering::SeqCst));
    assert_eq!(budget.receipt().unwrap().completed, 0);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(budget.receipt().unwrap().work_reserved, request.work);
    let (budget, _, _, request) = resources::<Aligned>();
    let owner = budget
        .lease(request)
        .unwrap()
        .commit_retained_with(|| Aligned([3]))
        .unwrap();
    assert_eq!(owner.0.data.0[0], 3);
    assert_eq!(
        std::ptr::from_ref(&*owner).addr() % align_of::<Aligned>(),
        0
    );
}

#[test]
fn retained_owner_cancelled_commit_never_calls_factory_or_renews_work() {
    for materialized in [false, true] {
        let (budget, memory, cancel, request) = resources::<[u8; 3]>();
        let mut lease = budget.lease(request).unwrap();
        if materialized {
            lease.mark_materialized().unwrap();
        }
        cancel.cancel();
        let ran = AtomicBool::new(false);
        let result = lease.commit_retained_with(|| {
            ran.store(true, Ordering::SeqCst);
            [1_u8, 2, 3]
        });
        assert!(matches!(result, Err(WorkRefusal::Cancelled)));
        assert!(!ran.load(Ordering::SeqCst));
        assert_eq!(budget.receipt().unwrap().completed, 0);
        assert_eq!(budget.receipt().unwrap().work_reserved, request.work);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    }
}

#[test]
fn retained_owner_unavailable_ledger_keeps_poisoned_authority_closed() {
    let (budget, memory, _, request) = resources::<[u8; 3]>();
    let lease = budget.lease(request).unwrap();
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = budget.0.state.lock().unwrap();
        panic!("isolated retained-owner ledger refusal control");
    }));
    assert!(poisoned.is_err());
    assert!(matches!(
        lease.commit_retained_with(|| [1_u8, 2, 3]),
        Err(WorkRefusal::Unavailable)
    ));
    assert_eq!(budget.receipt(), Err(WorkRefusal::Unavailable));
    let state = budget.0.state.lock().unwrap_err().into_inner();
    assert_eq!(state.receipt.completed, 0);
    assert_eq!(state.receipt.live_scratch_bytes, request.scratch_bytes);
    assert_eq!(
        state.receipt.live_persistent_bytes,
        request.persistent_bytes
    );
    assert_eq!(state.receipt.work_reserved, request.work);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}

#[test]
fn retained_owner_panicking_factory_releases_claim_once_without_refunding_work() {
    let (budget, memory, _, request) = resources::<DropProbe>();
    let observed = Arc::new(AtomicBool::new(false));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: RetainedOwner<DropProbe> = budget
            .lease(request)
            .unwrap()
            .commit_retained_with(|| {
                assert_eq!(budget.receipt().unwrap().completed, 1);
                assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
                assert_eq!(
                    memory.receipt().unwrap().live_materialized_bytes,
                    request.persistent_bytes
                );
                let _data = DropProbe {
                    budget: budget.clone(),
                    memory: memory.clone(),
                    bytes: request.persistent_bytes,
                    observed: observed.clone(),
                };
                panic!("isolated retained-owner factory panic control");
            })
            .unwrap();
    }));
    assert!(result.is_err());
    assert!(observed.load(Ordering::SeqCst));
    assert_eq!(budget.receipt().unwrap().completed, 1);
    assert_eq!(budget.receipt().unwrap().work_reserved, request.work);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}

#[test]
fn retained_owner_unique_test_mutation_cannot_change_shared_payload() {
    let (budget, _, _, request) = resources::<[u8; 3]>();
    let mut first = budget
        .lease(request)
        .unwrap()
        .commit_retained_with(|| [1_u8, 2, 3])
        .unwrap();
    let second = first.clone();
    assert!(first.get_mut().is_none());
    drop(second);
    first.get_mut().unwrap()[0] = 7;
    assert_eq!(first[0], 7);
    assert_eq!(first.bytes(), request.persistent_bytes);
}
