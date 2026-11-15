// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::{
    PreparedRootOutcome, PreparedRootProofOptions, PreparedRootSession, RootPrecisionRuntimePolicy,
    TwinOpSpec, TwinSpec,
};
use std::time::Duration;

fn clock() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(256 << 20);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(clock()),
            scratch_bytes: 128 << 20,
            persistent_bytes: 128 << 20,
            max_work: 100_000_000_000_000,
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
fn open(budget: &WorkBudget) -> PreparedRootOpen {
    let spec = TwinSpec {
        n_in: 1,
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
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![1.0, 1.0],
                bias: vec![-1.0, -1.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0, -1.0, 0.0, 0.0],
                bias: vec![0.75, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    };
    let session = PreparedRootSession::compile_declared_twin(
        spec,
        &[-1.0],
        &[3.0],
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            prefer_patch: false,
            ..PreparedRootProofOptions::default()
        },
        budget,
        Some(clock()),
    )
    .unwrap();
    match session.try_prove_root(0, &[1], None).unwrap() {
        PreparedRootOutcome::Open(open) => open,
        PreparedRootOutcome::Proven(_) => panic!("original duplicate-head relaxation must be open"),
    }
}

#[test]
fn prepared_domain_actual_original_phases_sparse_frame_never_copy_source_weights_or_baseline_gates()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = PreparedDomain::from_open(open(&budget), None).unwrap();
    assert_eq!(
        root.next_split(None).unwrap(),
        Some(PreparedSplitSite::Head { neuron: 1 })
    );
    let base_alpha = root.adapter().gates().layers[0].alpha.clone();
    let off = root
        .split_trunk(1, 0, PreparedPhaseSide::Inactive, None)
        .unwrap();
    let on = root
        .split_trunk(1, 0, PreparedPhaseSide::Active, None)
        .unwrap();
    let off_frame = off.frame(None).unwrap();
    let on_frame = on.frame(None).unwrap();
    assert!(std::ptr::eq(root.adapter().net(), off.adapter().net()));
    assert!(std::ptr::eq(
        FrozenGateFrame::layer(&off_frame, 0).unwrap(),
        &raw const root.adapter().gates().layers[0]
    ));
    assert_eq!(off_frame.overrides().layers[&0].alpha, [0.0]);
    assert_eq!(on_frame.overrides().layers[&0].s, [1.0]);
    assert_eq!(root.adapter().gates().layers[0].alpha, base_alpha);
    assert!(off.same_root(&on) && !off.same_domain(&on));
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop((off_frame, on_frame, off, on, root));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_foreign_equal_models_and_original_id_duplicate_refusals_preserve_source() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = PreparedDomain::from_open(open(&budget), None).unwrap();
    let b = PreparedDomain::from_open(open(&budget), None).unwrap();
    assert!(!a.same_root(&b));
    let child = a.split_head(1, PreparedPhaseSide::Inactive, None).unwrap();
    let before = live(&memory);
    assert!(child
        .split_head(1, PreparedPhaseSide::Active, None)
        .is_err());
    assert_eq!(live(&memory), before);
    assert!(a.split_head(2, PreparedPhaseSide::Active, None).is_err());
    assert!(a
        .split_trunk(3, 0, PreparedPhaseSide::Inactive, None)
        .is_err());
    assert!(a
        .split_trunk(1, 1, PreparedPhaseSide::Inactive, None)
        .is_err());
    a.checkpoint(None).unwrap();
    drop((a, b, child));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_paid_chain_and_cover_path_bind_original_absolute_ids_and_last_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = PreparedDomain::from_open(open(&budget), None).unwrap();
    let child = root
        .split_head(1, PreparedPhaseSide::Active, None)
        .unwrap()
        .split_trunk(1, 0, PreparedPhaseSide::Inactive, None)
        .unwrap();
    assert_eq!(child.depth(), 2);
    assert!(child
        .matches_legacy_splits(&[(0, 0, -1)], &[(1, 1)])
        .unwrap());
    assert!(!child
        .matches_legacy_splits(&[(0, 0, 1)], &[(1, 1)])
        .unwrap());
    let path = child.cover_path(None).unwrap();
    assert_eq!(path.path(None).unwrap().depth(), 2);
    assert_eq!(
        path.path(None).unwrap().literals(),
        [
            CoverLiteral {
                site: CoverSite::Trunk { layer: 0, pos: 0 },
                positive: false
            },
            CoverLiteral {
                site: CoverSite::Head { index: 1 },
                positive: true
            },
        ]
    );
    assert!(path.domain().same_domain(&child));
    drop((root, child));
    let held = live(&memory);
    assert!(held > 0);
    path.path(None).unwrap();
    drop(path);
    assert_eq!(live(&memory), 0);
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn prepared_domain_successful_shorter_child_clock_is_kept_by_alias_frame_and_path() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = PreparedDomain::from_open(open(&budget), None).unwrap();
    let early = Instant::now() + Duration::from_secs(30);
    let child = root
        .split_head(1, PreparedPhaseSide::Active, Some(early))
        .unwrap();
    assert!(child.deadline(None).unwrap() <= early);
    assert!(child.clone().deadline(None).unwrap() <= early);
    let frame = child.frame(Some(early)).unwrap();
    assert!(frame.deadline(None).unwrap() <= early);
    let before = budget.receipt().unwrap().admitted;
    assert!(child
        .split_trunk(
            1,
            0,
            PreparedPhaseSide::Active,
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
        )
        .is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before);
    let path = child.cover_path(Some(early)).unwrap();
    path.path(None).unwrap();
    drop((root, child, frame, path));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_original_cancellation_blocks_next_phase_and_frame_before_allocation() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let root = PreparedDomain::from_open(open(&budget), None).unwrap();
    let before = budget.receipt().unwrap().admitted;
    cancel.cancel();
    assert!(root.split_head(1, PreparedPhaseSide::Active, None).is_err());
    assert!(root.frame(None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before);
    drop(root);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_original_zero_free_capacity_refuses_node_without_losing_parent_claim() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = PreparedDomain::from_open(open(&budget), None).unwrap();
    let r = budget.receipt().unwrap();
    budget
        .tighten_total_byte_limit(r.live_scratch_bytes + r.live_persistent_bytes)
        .unwrap();
    let before = live(&memory);
    assert!(root.split_head(1, PreparedPhaseSide::Active, None).is_err());
    assert_eq!(live(&memory), before);
    root.checkpoint(None).unwrap();
    drop(root);
    assert_eq!(live(&memory), 0);
}
