// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::{
    residency::{CancellationToken, MemoryAuthority},
    work::{LeaseClass, WorkLimits, WorkRequest, WorkReserve},
};
use crate::margin_row::TwinOpSpec;
use std::time::Duration;

fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources(cap: usize) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(cap);
    let cancellation = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: Some(at()),
            scratch_bytes: cap,
            persistent_bytes: cap,
            max_work: 1_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(cap).unwrap();
    (budget, memory, cancellation)
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}
fn fixture() -> TwinSpec {
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.25],
                bias: vec![-0.25],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![0.5, -0.25],
                bias: vec![0.5, 0.75],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.0; 6],
                bias: vec![0.0, -0.25, 2.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (3, 2),
            },
        ],
    }
}
fn prove(
    owner: PreparedRootCompilation,
    budget: &WorkBudget,
    requested: Option<Instant>,
) -> Result<PreparedRootProof> {
    owner.prove_declared_domain(
        &[-1.0],
        &[1.0],
        2,
        &[1, 0],
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        budget,
        requested,
    )
}

#[test]
fn prepared_staged_compilation_finishes_actual_original_model_without_recompiling_parameters() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(64 << 20);
    let compiled =
        PreparedRootCompilation::compile_declared_twin(fixture(), &budget, Some(at())).unwrap();
    let declaration = *compiled.source_digest();
    let actual = *compiled.compiled_digest();
    let alias = compiled.clone();
    let proof = prove(compiled, &budget, Some(at())).unwrap();
    assert_eq!(*proof.source_digest(), declaration);
    assert_eq!(*proof.compiled_digest(), actual);
    assert_eq!(proof.target(), 2);
    assert_eq!(proof.classes(), [1, 0]);
    // Independent integer original-coordinate control: regardless of both
    // nonnegative hidden activations, final original zero weights give
    // y=[0,-1/4,2], hence margins [9/4,2] on the entire original box.
    assert!(proof.min_margin() > 1.99 && proof.min_margin() <= 2.0);
    assert!(proof.budget().shares_ledger(&budget));
    drop(proof);
    assert_eq!(live(&memory), alias.resident_bytes());
    drop(alias);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_staged_compilation_releases_only_completed_decoder_overlap_and_keeps_last_model_owner()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(64 << 20);
    let mut decoder = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: 8 << 20,
                work: 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let compiled =
        PreparedRootCompilation::compile_declared_twin(fixture(), &budget, Some(at())).unwrap();
    let with_decoder = live(&memory);
    let alias = compiled.clone();
    assert_eq!(live(&memory), with_decoder);
    decoder.complete().unwrap();
    drop(decoder);
    assert_eq!(live(&memory), with_decoder - (8 << 20));
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(compiled);
    assert_eq!(live(&memory), alias.resident_bytes());
    alias.checkpoint(&budget, None).unwrap();
    drop(alias);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_staged_compilation_equal_limits_on_foreign_ledger_cannot_adopt_actual_model() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(64 << 20);
    let (foreign, foreign_memory, _) = resources(64 << 20);
    let compiled =
        PreparedRootCompilation::compile_declared_twin(fixture(), &budget, Some(at())).unwrap();
    let before = foreign.receipt().unwrap().admitted;
    assert!(compiled.checkpoint(&foreign, None).is_err());
    assert!(prove(compiled, &foreign, Some(at())).is_err());
    assert_eq!(foreign.receipt().unwrap().admitted, before);
    assert_eq!(live(&memory), 0);
    assert_eq!(live(&foreign_memory), 0);
}

#[test]
fn prepared_staged_compilation_consuming_transfer_persists_earliest_original_clock() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(64 << 20);
    let original = at();
    let early = original.checked_sub(Duration::from_secs(10)).unwrap();
    let compiled =
        PreparedRootCompilation::compile_declared_twin(fixture(), &budget, Some(original))
            .unwrap()
            .restrict_deadline(&budget, Some(early))
            .unwrap();
    assert_eq!(compiled.deadline(Some(original)).unwrap(), Some(early));
    let proof = prove(compiled, &budget, Some(original)).unwrap();
    assert_eq!(proof.deadline(Some(original)).unwrap(), early);
}

#[test]
fn prepared_staged_compilation_expired_entry_and_cancelled_continuation_publish_nothing() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources(64 << 20);
    let before = budget.receipt().unwrap().admitted;
    assert!(PreparedRootCompilation::compile_declared_twin(
        fixture(),
        &budget,
        Some(Instant::now())
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before);
    let compiled =
        PreparedRootCompilation::compile_declared_twin(fixture(), &budget, Some(at())).unwrap();
    let before = budget.receipt().unwrap().admitted;
    cancel.cancel();
    assert!(prove(compiled, &budget, Some(at())).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_staged_compilation_actual_same_ledger_held_quota_refuses_new_model_without_cap_raise() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(64 << 20);
    let held = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: (64 << 20) - 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let before = live(&memory);
    let peak = memory.receipt().unwrap().peak_leased_bytes;
    assert!(
        PreparedRootCompilation::compile_declared_twin(fixture(), &budget, Some(at())).is_err()
    );
    assert_eq!(live(&memory), before);
    assert_eq!(memory.receipt().unwrap().peak_leased_bytes, peak);
    drop(held);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_staged_compilation_requested_f32_cannot_relabel_actual_f64_model_delivery() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(64 << 20);
    let compiled =
        PreparedRootCompilation::compile_declared_twin(fixture(), &budget, Some(at())).unwrap();
    for policy in [
        RootPrecisionRuntimePolicy {
            root_f32: true,
            ..RootPrecisionRuntimePolicy::default()
        },
        RootPrecisionRuntimePolicy {
            tableau_f32: true,
            ..RootPrecisionRuntimePolicy::default()
        },
    ] {
        assert!(compiled
            .clone()
            .prove_declared_domain(
                &[-1.0],
                &[1.0],
                2,
                &[1, 0],
                policy,
                PreparedRootProofOptions::default(),
                &budget,
                Some(at())
            )
            .is_err());
        assert_eq!(live(&memory), compiled.resident_bytes());
    }
    drop(compiled);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_staged_compilation_original_signed_zero_and_subnormal_parameter_bits_bind_identity() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(64 << 20);
    let mut positive = fixture();
    let mut negative = fixture();
    let mut tiny = fixture();
    for (spec, value) in [
        (&mut positive, 0.0),
        (&mut negative, -0.0),
        (&mut tiny, -f64::from_bits(1)),
    ] {
        if let TwinOpSpec::Gemm { weight, .. } = spec.ops.last_mut().unwrap() {
            weight[0] = value;
        }
    }
    let a = PreparedRootCompilation::compile_declared_twin(positive, &budget, Some(at())).unwrap();
    let b = PreparedRootCompilation::compile_declared_twin(negative, &budget, Some(at())).unwrap();
    let c = PreparedRootCompilation::compile_declared_twin(tiny, &budget, Some(at())).unwrap();
    assert_ne!(a.source_digest(), b.source_digest());
    assert_ne!(a.source_digest(), c.source_digest());
    assert_ne!(b.compiled_digest(), c.compiled_digest());
}

#[test]
fn prepared_staged_compilation_debug_reports_original_contract_and_distinct_capacity_claims() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(64 << 20);
    let compiled =
        PreparedRootCompilation::compile_declared_twin(fixture(), &budget, Some(at())).unwrap();
    let source = *compiled.source_digest();
    let actual = *compiled.compiled_digest();
    let capacity = compiled.model.actual_bytes();
    let paid = compiled.resident_bytes();
    assert!(capacity <= paid);
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    let observed = format!("{compiled:?}");
    assert!(observed.contains("parameter_contract: Binary64WithFoldUncertainty"));
    assert!(observed.contains(&format!(
        "compiled_capacity_with_owner_allowance_bytes: {capacity}"
    )));
    assert!(observed.contains(&format!("resident_bytes: {paid}")));
    assert_eq!(*compiled.source_digest(), source);
    assert_eq!(*compiled.compiled_digest(), actual);
    let after = budget.receipt().unwrap();
    assert_eq!(after.work_reserved, before.work_reserved);
    assert_eq!(after.proposals_reserved, before.proposals_reserved);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(live(&memory), bytes);
    drop(compiled);
    assert_eq!(live(&memory), 0);
}
