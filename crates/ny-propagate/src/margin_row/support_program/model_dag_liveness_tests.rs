// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Public diagnostic flows use authentic compilation and completed sessions.
//! These controls do not manufacture model owners, schedules or proof tokens.

use super::OWNER_WORK;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{
    LeaseClass, WorkBudget, WorkLimits, WorkRefusal, WorkRequest, WorkReserve,
};
use crate::margin_row::{
    PreparedRootCompilation, PreparedRootProofOptions, PreparedRootSession,
    RootPrecisionRuntimePolicy, TwinOpSpec, TwinSpec,
};
use std::time::{Duration, Instant};

const CAP: usize = 128 << 20;
const MAX_WORK: u64 = 100_000_000_000_000;

fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(2 * CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: MAX_WORK,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, cancel)
}

fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}

fn fixture() -> TwinSpec {
    TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 2),
                oshape: (1, 1, 2),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Add { lhs: 2, rhs: 2 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![0.5],
                shift: vec![0.0],
                shift_err: vec![0.0],
                scale_rel_err: 0.0,
                shape: (1, 1, 2),
            },
            TwinOpSpec::Flatten { input: 3 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.5, -0.25],
                bias: vec![0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, 2),
            },
            TwinOpSpec::Relu { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![1.0, -1.0],
                bias: vec![0.0, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    }
}

fn compilation(budget: &WorkBudget) -> PreparedRootCompilation {
    PreparedRootCompilation::compile_declared_twin(fixture(), budget, Some(at())).unwrap()
}

fn domain(
    compiled: PreparedRootCompilation,
    budget: &WorkBudget,
    width: f64,
) -> PreparedRootSession {
    compiled
        .prepare_declared_domain(
            &[-width; 2],
            &[width; 2],
            RootPrecisionRuntimePolicy::default(),
            PreparedRootProofOptions {
                prefer_patch: false,
                dense_tile: 1,
                ..PreparedRootProofOptions::default()
            },
            budget,
            None,
        )
        .unwrap()
}

fn session(budget: &WorkBudget) -> PreparedRootSession {
    domain(compilation(budget), budget, 1.0)
}

#[test]
fn prepared_model_dag_liveness_reports_duplicate_edges_unused_outputs_and_full_head_finish() {
    let _env = crate::tests::lock_env_shared();
    assert!(!PreparedRootProofOptions::default().keep_input_rows);
    assert!(!PreparedRootProofOptions::default().row_f32);
    let (budget, memory, _) = resources();
    let source = session(&budget);
    let before = budget.receipt().unwrap();
    let schedule = source.model_dag_liveness(None).unwrap();
    schedule.validate_session(&source, None).unwrap();
    assert_eq!(schedule.operation_count(), 8);
    assert_eq!(schedule.tensor_count(), 9);
    let counts = [1, 1, 3, 1, 0, 1, 1, 1, 0];
    for (tensor, count) in counts.into_iter().enumerate() {
        assert_eq!(schedule.consumer_count(tensor, None).unwrap(), count);
    }
    let releases: [&[usize]; 8] = [&[0], &[1], &[], &[2, 4], &[3], &[5], &[6], &[7]];
    let after_creation = budget.receipt().unwrap();
    let bytes = live(&memory);
    let mut released = [0usize; 9];
    for (op, expected) in releases.into_iter().enumerate() {
        assert_eq!(schedule.release_after(op, None).unwrap(), expected);
        for &tensor in schedule.release_after(op, None).unwrap() {
            released[tensor] += 1;
        }
    }
    // Trunk construction ends before the first head Gemm (op 5). The full
    // original DAG still has op 5, ReLU 6 and Gemm 7, then finish tensor 8.
    assert_eq!(schedule.release_after(5, None).unwrap(), &[5]);
    assert_eq!(schedule.release_after(6, None).unwrap(), &[6]);
    assert_eq!(schedule.release_after(7, None).unwrap(), &[7]);
    assert_eq!(schedule.finish_releases(None).unwrap(), &[8]);
    for &tensor in schedule.finish_releases(None).unwrap() {
        released[tensor] += 1;
    }
    assert_eq!(released, [1; 9]);
    assert!(schedule.consumer_count(9, None).is_err());
    assert!(schedule.release_after(8, None).is_err());
    assert!(after_creation.admitted > before.admitted);
    assert!(after_creation.work_reserved > before.work_reserved);
    let after_reads = budget.receipt().unwrap();
    assert_eq!(after_reads.admitted, after_creation.admitted);
    assert_eq!(after_reads.work_reserved, after_creation.work_reserved);
    assert_eq!(live(&memory), bytes);
}

#[test]
fn prepared_model_dag_liveness_pairs_same_compilation_domains_and_refuses_equal_foreign_models() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let compiled = compilation(&budget);
    let first = domain(compiled.clone(), &budget, 1.0);
    let other_domain = domain(compiled, &budget, 0.5);
    assert!(!first.same_source(&other_domain));
    let schedule = first.model_dag_liveness(None).unwrap();
    schedule.validate_session(&other_domain, None).unwrap();
    let independently_compiled = session(&budget);
    assert_eq!(
        first.compiled_digest(),
        independently_compiled.compiled_digest()
    );
    assert_eq!(
        first.source_digest(),
        independently_compiled.source_digest()
    );
    assert!(schedule
        .validate_session(&independently_compiled, None)
        .is_err());
    let (foreign_budget, _, _) = resources();
    let foreign = session(&foreign_budget);
    assert_eq!(first.compiled_digest(), foreign.compiled_digest());
    assert!(!budget.shares_ledger(&foreign_budget));
    assert!(schedule.validate_session(&foreign, None).is_err());
}

#[test]
fn prepared_model_dag_liveness_clones_share_claims_and_release_only_at_last_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(&budget);
    let source_bytes = live(&memory);
    let schedule = source.model_dag_liveness(None).unwrap();
    let after_creation = live(&memory);
    assert!(after_creation > source_bytes);
    let admitted = budget.receipt().unwrap().admitted;
    let work = budget.receipt().unwrap().work_reserved;
    let alias = schedule.clone();
    assert_eq!(
        schedule.finish_releases(None).unwrap().as_ptr(),
        alias.finish_releases(None).unwrap().as_ptr()
    );
    assert_eq!(live(&memory), after_creation);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    assert_eq!(budget.receipt().unwrap().work_reserved, work);
    drop(source);
    let graph_only = live(&memory);
    assert!(graph_only > 0);
    assert!(graph_only < after_creation);
    alias.checkpoint(None).unwrap();
    assert_eq!(alias.consumer_count(2, None).unwrap(), 3);
    drop(schedule);
    assert_eq!(live(&memory), graph_only);
    drop(alias);
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}

#[test]
fn prepared_model_dag_liveness_preserves_session_request_clocks_and_shared_cancellation() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let source = session(&budget);
    let session_at = source
        .deadline(None)
        .unwrap()
        .checked_sub(Duration::from_secs(10))
        .expect("test clock supports an earlier deadline");
    let restricted = source.restricted(Some(session_at)).unwrap();
    let schedule = restricted
        .model_dag_liveness(Some(session_at + Duration::from_secs(1)))
        .unwrap();
    assert_eq!(schedule.deadline(None).unwrap(), session_at);
    assert_eq!(
        schedule
            .deadline(Some(session_at + Duration::from_secs(2)))
            .unwrap(),
        session_at
    );
    let request_at = session_at
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an earlier deadline");
    let earlier = source.model_dag_liveness(Some(request_at)).unwrap();
    assert_eq!(earlier.deadline(None).unwrap(), request_at);
    let expired = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("test clock supports an expired deadline"),
    );
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    assert!(restricted.model_dag_liveness(expired).is_err());
    assert!(schedule.consumer_count(2, expired).is_err());
    assert!(schedule.release_after(7, expired).is_err());
    assert!(schedule.finish_releases(expired).is_err());
    assert!(schedule.validate_session(&source, expired).is_err());
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    cancel.cancel();
    assert!(source.model_dag_liveness(None).is_err());
    assert!(schedule.consumer_count(2, None).is_err());
    assert!(schedule.release_after(7, None).is_err());
    assert!(schedule.finish_releases(None).is_err());
    assert!(schedule.validate_session(&source, None).is_err());
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
}

#[test]
fn prepared_model_dag_liveness_header_refuses_occupied_original_capacity_before_graph_admission() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(&budget);
    // Occupy this genuine session's already capped shared ledger. No session
    // above the construction ceiling, integer substitute or foreign claim.
    // Keep the combined ceiling below the per-persistent limit so the actual
    // failure specifically exercises the shared combined-capacity guard.
    let cap = budget.tighten_total_byte_limit(CAP / 2).unwrap();
    let original = budget.receipt().unwrap();
    let bytes = super::owner_bytes().unwrap();
    let occupied = cap
        .checked_sub(original.live_persistent_bytes + original.live_scratch_bytes)
        .and_then(|remaining| remaining.checked_sub(bytes - 1))
        .unwrap();
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: occupied,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let before = budget.receipt().unwrap();
    let memory_before = live(&memory);
    let error = source.model_dag_liveness(None).err().unwrap();
    assert!(error.to_string().contains("MemoryLimit"));
    let refused = budget.receipt().unwrap();
    assert_eq!(refused.admitted, before.admitted);
    assert_eq!(refused.work_reserved, before.work_reserved);
    assert_eq!(refused.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(refused.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(live(&memory), memory_before);
    drop(blocker);
    let schedule = source.model_dag_liveness(None).unwrap();
    assert_eq!(schedule.finish_releases(None).unwrap(), &[8]);
}

#[test]
fn prepared_model_dag_liveness_graph_work_refusal_releases_header_without_refilling_work() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(&budget);
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    // Leave enough real shared work for the wrapper, but none for the
    // unchanged OriginalLiveness builder. A failed second admission must not
    // leak the first header claim or refund its spent work.
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: MAX_WORK - before.work_reserved - OWNER_WORK,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let occupied = budget.receipt().unwrap();
    let error = source.model_dag_liveness(None).err().unwrap();
    assert!(error.to_string().contains("WorkLimit"));
    let refused = budget.receipt().unwrap();
    assert_eq!(refused.admitted, occupied.admitted + 1);
    assert_eq!(refused.work_reserved, MAX_WORK);
    assert_eq!(refused.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(refused.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(live(&memory), bytes);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    drop(blocker);
    assert!(matches!(
        budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: 1,
                ..WorkRequest::default()
            }
        ),
        Err(WorkRefusal::WorkLimit)
    ));
    assert_eq!(budget.receipt().unwrap().work_reserved, MAX_WORK);
    source.checkpoint(None).unwrap();
}
