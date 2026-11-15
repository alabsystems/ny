// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Original prepared producer, actual containment and original token controls.
//! No fixture supplies certified bounds, a completed frame or token authority.

use super::super::terminal::build_prepared_terminal;
use super::super::{build_prepared_root, OriginalModel, OriginalRoot, PreparedConstructionOptions};
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::certify::{GuardId, UnsatSource};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 128 << 20;
fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources(
    cap: usize,
    deadline: Option<Instant>,
) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(cap * 2);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline,
            scratch_bytes: cap,
            persistent_bytes: cap,
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
fn fixture(fold: bool, positive: bool) -> TwinSpec {
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.25],
                bias: vec![-0.25],
                bias_err: vec![if fold { 1.0 / 256.0 } else { 0.0 }],
                weight_rel_err: if fold { 1.0 / 128.0 } else { 0.0 },
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
                bias: vec![0.75, 0.5],
                bias_err: if fold { vec![1.0 / 512.0; 2] } else { vec![] },
                weight_rel_err: if fold { 1.0 / 256.0 } else { 0.0 },
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.75, 0.25, 0.5, -0.25, -0.25, 0.5],
                bias: vec![if positive { 2.0 } else { -2.0 }, 0.0, 0.125],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (3, 2),
            },
        ],
    }
}
fn source(spec: TwinSpec, budget: &WorkBudget, deadline: Option<Instant>) -> PreparedRootSource {
    share_prepared_root(unshared_source(spec, budget, deadline), deadline).unwrap()
}
fn unshared_source(
    spec: TwinSpec,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> PreparedRootConstruction {
    let n = spec.n_in;
    let model = OriginalModel::compile(spec, budget, deadline).unwrap();
    let root = OriginalRoot::new(
        model,
        &vec![-1.0; n],
        &vec![1.0; n],
        RootPrecisionRuntimePolicy::default(),
        deadline,
    )
    .unwrap();
    build_prepared_root(root, PreparedConstructionOptions::default(), Some(at())).unwrap()
}
fn pair(
    spec: TwinSpec,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> (PreparedRootSource, PreparedTerminal) {
    let src = source(spec, budget, deadline);
    let terminal = build_prepared_terminal(
        src.construction(),
        0,
        &[2, 1],
        Some(src.deadline(Some(at())).unwrap()),
    )
    .unwrap();
    (src, terminal)
}
fn r(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}
fn independent_margins(spec: &TwinSpec, x: f64, bits: usize) -> [BigRational; 2] {
    let s = |bit| {
        if bits & (1_usize << bit) == 0_usize {
            r(-1.0)
        } else {
            r(1.0)
        }
    };
    let (w, b, rho, eb) = match &spec.ops[0] {
        TwinOpSpec::Conv {
            weight,
            bias,
            weight_rel_err,
            bias_err,
            ..
        } => (weight[0], bias[0], *weight_rel_err, bias_err[0]),
        _ => unreachable!(),
    };
    let h = (r(w) * (r(1.0) + s(0) * r(rho)) * r(x) + r(b) + s(1) * r(eb)).max(BigRational::zero());
    let (weights, bias, rho, errors) = match &spec.ops[3] {
        TwinOpSpec::Gemm {
            weight,
            bias,
            weight_rel_err,
            bias_err,
            ..
        } => (weight, bias, *weight_rel_err, bias_err),
        _ => unreachable!(),
    };
    let y: Vec<_> = (0..2)
        .map(|j| {
            (r(weights[j]) * (r(1.0) + s(2 + j) * r(rho)) * &h
                + r(bias[j])
                + s(4 + j) * r(if errors.is_empty() { 0.0 } else { errors[j] }))
            .max(BigRational::zero())
        })
        .collect();
    let (weights, bias) = match &spec.ops[5] {
        TwinOpSpec::Gemm { weight, bias, .. } => (weight, bias),
        _ => unreachable!(),
    };
    let outputs: Vec<_> = (0..3)
        .map(|o| (0..2).fold(r(bias[o]), |sum, j| sum + r(weights[o * 2 + j]) * &y[j]))
        .collect();
    [&outputs[0] - &outputs[2], &outputs[0] - &outputs[1]]
}

#[test]
fn prepared_publication_actual_positive_producer_enters_original_authorizer_with_exact_sorted_and_original_ordered_classes(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let spec = fixture(false, true);
    let (src, terminal) = pair(spec.clone(), &budget, None);
    let original = terminal.receipt();
    let pubn = authorize_prepared_terminal(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    assert_eq!(pubn.token().source(), UnsatSource::Authorized);
    assert_eq!(pubn.token().scope(), TokenScope::Instance);
    assert_eq!(pubn.token().classes(), [1, 2]);
    assert_eq!(pubn.terminal().adversaries(), [2, 1]);
    assert_eq!(pubn.receipt().terminal, original);
    assert_eq!(pubn.receipt().classes, 2);
    assert!(pubn.receipt().containment_points > 0);
    assert!(pubn.receipt().containment_checks > 0);
    assert!(!pubn.receipt().containment_skipped);
    for x in [-1.0, -0.25, 0.0, 0.2, 1.0] {
        for (column, margin) in independent_margins(&spec, x, 0).iter().enumerate() {
            assert!(margin > &BigRational::zero());
            assert!(r(pubn.terminal().lower()[column]) <= *margin);
            assert!(r(pubn.token().min_margin()) <= *margin);
        }
    }
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    pubn.checkpoint(Some(at())).unwrap();
    drop(pubn);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_publication_fold_uncertainty_original_rational_corners_remain_positive_not_point_oracle_authority(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP, None);
    let spec = fixture(true, true);
    let (src, terminal) = pair(spec.clone(), &budget, None);
    let pubn = authorize_prepared_terminal(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    for bits in 0..64 {
        for x in [-1.0, -0.25, 0.0, 0.5, 1.0] {
            for (column, margin) in independent_margins(&spec, x, bits).iter().enumerate() {
                assert!(margin > &BigRational::zero());
                assert!(r(pubn.terminal().lower()[column]) <= *margin);
            }
        }
    }
    assert!(pubn.receipt().containment_points > 0);
    assert_eq!(pubn.receipt().terminal.margin_columns, 2);
}

#[test]
fn prepared_publication_nonpositive_original_bounds_never_publish_even_when_reference_can_run() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let (src, terminal) = pair(fixture(false, false), &budget, None);
    assert!(terminal.lower().iter().any(|&v| v <= 0.0));
    assert!(authorize_prepared_terminal(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at())
    )
    .is_err());
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn prepared_publication_partial_classes_and_relabelled_parent_target_cannot_mint_an_instance_token()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let src = source(fixture(false, true), &budget, None);
    let partial = build_prepared_terminal(src.construction(), 0, &[1], Some(at())).unwrap();
    assert!(partial.lower()[0] > 0.0);
    assert!(authorize_prepared_terminal(
        src,
        partial,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at())
    )
    .is_err());
    assert_eq!(live(&memory), 0);
    let (src, complete) = pair(fixture(false, true), &budget, None);
    assert!(authorize_prepared_terminal(
        src,
        complete,
        1,
        RootPrecisionRuntimePolicy::default(),
        Some(at())
    )
    .is_err());
    assert_eq!(live(&memory), 0);
    let (src, complete) = pair(fixture(false, true), &budget, None);
    let pubn = authorize_prepared_terminal(
        src,
        complete,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    assert!(pubn.validate_objective(1, &[2, 1], Some(at())).is_err());
    assert!(pubn.validate_objective(0, &[1, 2], Some(at())).is_err());
    pubn.validate_objective(0, &[2, 1], Some(at())).unwrap();
    drop(pubn);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_publication_foreign_completed_prefix_same_ledger_and_equal_limit_other_ledger_are_refused(
) {
    let _env = crate::tests::lock_env_shared();
    let (first, memory, _) = resources(CAP, None);
    let (src, terminal) = pair(fixture(false, true), &first, None);
    let foreign = source(fixture(false, true), &first, None);
    assert!(authorize_prepared_terminal(
        foreign,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at())
    )
    .is_err());
    let terminal = build_prepared_terminal(
        src.construction(),
        0,
        &[2, 1],
        Some(src.deadline(Some(at())).unwrap()),
    )
    .unwrap();
    let (second, foreign_memory, _) = resources(CAP, None);
    let foreign = source(fixture(false, true), &second, None);
    assert!(authorize_prepared_terminal(
        foreign,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at())
    )
    .is_err());
    assert_eq!(live(&foreign_memory), 0);
    drop(src);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_publication_original_nonzero_target_is_structurally_excluded_once_from_complete_class_domain(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    for target in [1usize, 2] {
        let mut spec = fixture(false, true);
        if let TwinOpSpec::Gemm { weight, bias, .. } = &mut spec.ops[5] {
            weight.fill(0.0);
            bias.fill(0.0);
            bias[target] = 2.0;
        }
        let src = source(spec, &budget, None);
        let classes: Vec<_> = (0..3).rev().filter(|&c| c != target).collect();
        let terminal =
            build_prepared_terminal(src.construction(), target, &classes, Some(at())).unwrap();
        let pubn = authorize_prepared_terminal(
            src,
            terminal,
            target,
            RootPrecisionRuntimePolicy::default(),
            Some(at()),
        )
        .unwrap();
        assert_eq!(pubn.receipt().target, target);
        assert_eq!(pubn.terminal().adversaries(), classes);
        assert_eq!(
            pubn.token().classes(),
            (0..3).filter(|&c| c != target).collect::<Vec<_>>()
        );
        assert!(pubn.token().min_margin() > 0.0 && r(pubn.token().min_margin()) <= r(2.0));
        pubn.validate_objective(target, &classes, Some(at()))
            .unwrap();
        drop(pubn);
        assert_eq!(live(&memory), 0);
    }
}

#[test]
fn prepared_publication_requested_precision_is_exact_actual_policy_not_row_storage_or_false_capability(
) {
    let _env = crate::tests::lock_env_shared();
    for policy in [
        RootPrecisionRuntimePolicy {
            root_f32: true,
            ..RootPrecisionRuntimePolicy::default()
        },
        RootPrecisionRuntimePolicy {
            tableau_f32: true,
            ..RootPrecisionRuntimePolicy::default()
        },
        RootPrecisionRuntimePolicy {
            ibp_seed: true,
            ..RootPrecisionRuntimePolicy::default()
        },
    ] {
        let (budget, memory, _) = resources(CAP, None);
        let (src, terminal) = pair(fixture(false, true), &budget, None);
        assert!(authorize_prepared_terminal(src, terminal, 0, policy, Some(at())).is_err());
        assert_eq!(live(&memory), 0);
    }
    let (budget, _, _) = resources(CAP, None);
    let (src, terminal) = pair(fixture(false, true), &budget, None);
    let pubn = authorize_prepared_terminal(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    assert!(pubn
        .validate_policy(
            RootPrecisionRuntimePolicy {
                root_f32: true,
                ..RootPrecisionRuntimePolicy::default()
            },
            Some(at())
        )
        .is_err());
    pubn.validate_policy(RootPrecisionRuntimePolicy::default(), Some(at()))
        .unwrap();
}

fn blocker(budget: &WorkBudget, bytes: usize) -> PersistentReservation {
    let mut lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: bytes,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    lease.mark_materialized().unwrap();
    lease.commit_persistent().unwrap()
}
fn publication_request(classes: usize) -> usize {
    owner_bytes()
        + classes * size_of::<usize>()
        + classes * size_of::<ClassCover>()
        + classes * 5 * size_of::<usize>()
        + META
}

#[test]
fn prepared_publication_one_past_original_fixed_cap_refuses_new_owner_before_allocation_with_global_headroom(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(ROOT_SOURCE_BYTES_CAP + CAP, None);
    let (src, terminal) = pair(fixture(false, true), &budget, None);
    let before = budget.receipt().unwrap();
    let old = before.live_scratch_bytes + before.live_persistent_bytes;
    let held = blocker(
        &budget,
        ROOT_SOURCE_BYTES_CAP - old - publication_request(2) + 1,
    );
    let bytes = held.bytes();
    assert!(
        matches!(authorize_prepared_terminal(src,terminal,0,RootPrecisionRuntimePolicy::default(),Some(at())),Err(NyError::CpuMemoryExceeded{required_bytes,budget_bytes,..})if required_bytes==ROOT_SOURCE_BYTES_CAP+1&&budget_bytes==ROOT_SOURCE_BYTES_CAP)
    );
    assert_eq!(live(&memory), bytes);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(held);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_publication_unaffordable_downgrade_sampling_is_explicit_skipped_not_pass_or_proof_authority(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(ROOT_SOURCE_BYTES_CAP + CAP, None);
    let (src, terminal) = pair(fixture(false, true), &budget, None);
    let before = budget.receipt().unwrap();
    let old = before.live_scratch_bytes + before.live_persistent_bytes;
    let held = blocker(
        &budget,
        ROOT_SOURCE_BYTES_CAP - old - publication_request(2),
    );
    let pubn = authorize_prepared_terminal(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    assert!(pubn.receipt().containment_skipped);
    assert_eq!(pubn.receipt().containment_points, 0);
    assert_eq!(pubn.receipt().containment_checks, 0);
    assert!(pubn.token().min_margin() > 0.0);
    pubn.checkpoint(Some(at())).unwrap();
    drop(pubn);
    drop(held);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_publication_clones_retain_original_source_terminal_and_paid_token_until_exact_last_owner(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let (src, terminal) = pair(fixture(false, true), &budget, None);
    let pubn = authorize_prepared_terminal(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    let bytes = live(&memory);
    assert!(bytes > pubn.resident_bytes() + pubn.terminal().resident_bytes());
    assert!(pubn.0.resident.shares_ledger(&budget));
    let clone = pubn.clone();
    assert!(Arc::ptr_eq(&clone.0, &pubn.0));
    assert_eq!(live(&memory), bytes);
    drop(pubn);
    assert_eq!(live(&memory), bytes);
    clone.checkpoint(Some(at())).unwrap();
    drop(clone);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_publication_shared_source_independent_last_owner_keeps_original_rows_and_header_after_proof_drops(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let (src, terminal) = pair(fixture(false, true), &budget, None);
    let source_bytes = live(&memory) - terminal.resident_bytes();
    let retained = src.clone();
    assert!(retained.same_source(&src));
    assert_eq!(retained.resident_bytes(), shared_source_bytes());
    let pubn = authorize_prepared_terminal(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    assert!(retained.same_source(pubn.source_owner()));
    drop(pubn);
    assert_eq!(live(&memory), source_bytes);
    retained.checkpoint(Some(at())).unwrap();
    assert_eq!(retained.construction().prefix().layer_count(), 1);
    let clone = retained.clone();
    drop(retained);
    assert_eq!(live(&memory), source_bytes);
    drop(clone);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_publication_shared_source_header_is_priced_before_allocation_at_exact_original_cap_and_deadline(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(ROOT_SOURCE_BYTES_CAP + CAP, None);
    let src = unshared_source(fixture(false, true), &budget, None);
    let old = budget.receipt().unwrap();
    let held = blocker(
        &budget,
        ROOT_SOURCE_BYTES_CAP
            - old.live_scratch_bytes
            - old.live_persistent_bytes
            - shared_source_bytes()
            + 1,
    );
    let before = budget.receipt().unwrap();
    assert!(
        matches!(share_prepared_root(src,Some(at())),Err(NyError::CpuMemoryExceeded{required_bytes,budget_bytes,..})if required_bytes==ROOT_SOURCE_BYTES_CAP+1&&budget_bytes==ROOT_SOURCE_BYTES_CAP)
    );
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(live(&memory), held.bytes());
    drop(held);
    assert_eq!(live(&memory), 0);
    let (budget, memory, _) = resources(CAP, None);
    let src = unshared_source(fixture(false, true), &budget, None);
    let before = budget.receipt().unwrap();
    assert!(matches!(
        share_prepared_root(src, Some(Instant::now())),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_publication_late_guard_refuses_real_token_while_clear_guard_keeps_original_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let (src, terminal) = pair(fixture(false, true), &budget, None);
    let pubn = authorize_prepared_terminal(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    let kept = pubn.clone();
    let mut trip = GuardReport::new();
    trip.trip(GuardId::SpecAgreement, "actual property disagreement");
    assert!(pubn.ratified(&trip, Some(at())).is_err());
    assert!(live(&memory) > 0);
    let kept = kept.ratified(&GuardReport::new(), Some(at())).unwrap();
    kept.checkpoint(Some(at())).unwrap();
    drop(kept);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_publication_cancellation_after_actual_original_authorizer_and_containment_discards_token_before_transfer(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources(CAP, None);
    let (src, terminal) = pair(fixture(false, true), &budget, None);
    let before = budget.receipt().unwrap();
    let mut observed = false;
    let mut hook = || {
        observed = true;
        cancel.cancel();
    };
    assert!(authorize_inner(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
        Some(&mut hook)
    )
    .is_err());
    assert!(observed);
    assert_eq!(live(&memory), 0);
    let after = budget.receipt().unwrap();
    assert!(after.work_reserved > before.work_reserved);
    assert_eq!(after.live_scratch_bytes, 0);
    assert_eq!(after.live_persistent_bytes, 0);
}

#[test]
fn prepared_publication_earliest_original_deadline_and_post_authorizer_expiry_cannot_be_renewed() {
    let _env = crate::tests::lock_env_shared();
    let limit = at();
    let (budget, memory, _) = resources(CAP, Some(limit));
    let (src, terminal) = pair(fixture(false, true), &budget, Some(limit));
    let pubn = authorize_prepared_terminal(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(limit + Duration::from_secs(600)),
    )
    .unwrap();
    assert_eq!(
        pubn.deadline(Some(limit + Duration::from_mins(15)))
            .unwrap(),
        limit
    );
    assert!(pubn.checkpoint(Some(Instant::now())).is_err());
    drop(pubn);
    assert_eq!(live(&memory), 0);
    let (budget, memory, _) = resources(CAP, None);
    let (src, terminal) = pair(fixture(false, true), &budget, None);
    let limit = Instant::now() + Duration::from_secs(2);
    let mut observed = false;
    let mut hook = || {
        observed = true;
        while Instant::now() < limit {
            std::hint::spin_loop();
        }
    };
    assert!(matches!(
        authorize_inner(
            src,
            terminal,
            0,
            RootPrecisionRuntimePolicy::default(),
            Some(limit),
            Some(&mut hook)
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(observed);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_publication_real_transposed_padding_residual_reference_uses_original_coordinates_and_paid_within_walk(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let spec = TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::ConvTranspose {
                input: 0,
                weight: vec![1.0, -0.5],
                bias: vec![0.25],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 2),
                stride: (1, 2),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 2),
                oshape: (1, 1, 4),
                out_pad: (0, 0),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![-0.25],
                shift: vec![0.5],
                shift_err: vec![0.0],
                scale_rel_err: 0.0,
                shape: (1, 1, 4),
            },
            TwinOpSpec::Add { lhs: 3, rhs: 2 },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: vec![1.0, -0.5, 0.25, -0.125],
                bias: vec![0.5],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, 4),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: vec![0.5; 3],
                bias: vec![2.0, 0.0, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (3, 1),
            },
        ],
    };
    let (src, terminal) = pair(spec, &budget, None);
    let pubn = authorize_prepared_terminal(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    let point = Array2::from_shape_vec((2, 1), vec![-1.0, 0.5]).unwrap();
    let mut audit = ContainmentAudit::default();
    pubn.0.frame.audit(&point, &mut audit).unwrap();
    assert!(audit.checks() > 0);
    assert_eq!(audit.violations(), 0);
    // Authored original transpose: pre=[−.75,.75,.75,0],
    // shared residual=.75*ReLU(pre)+.5; original head is43/64.
    assert!(r(pubn.terminal().head_lo()[0]) <= r(43.0 / 64.0));
    assert!(r(pubn.terminal().head_hi()[0]) >= r(43.0 / 64.0));
    assert!(pubn
        .terminal()
        .lower()
        .iter()
        .all(|&x| x > 0.0 && r(x) <= r(2.0)));
    let mut outside = ContainmentAudit::default();
    let point = Array2::from_shape_vec((2, 1), vec![2.0, 0.5]).unwrap();
    assert!(pubn.0.frame.audit(&point, &mut outside).is_err());
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(pubn);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_publication_paid_sampling_keeps_extreme_signed_zero_and_subnormal_declared_coordinates_finite(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP, None);
    let limit = at();
    let lo = [-f64::MAX, -0.0, -f64::from_bits(1), 0.0];
    let hi = [f64::MAX, 0.0, f64::from_bits(1), f64::from_bits(1)];
    let count = 16;
    let mut lease = admit(
        &budget,
        WorkRequest {
            scratch_bytes: lo.len() * count * 8 + META,
            work: work(lo.len() * count, 64).unwrap(),
            ..WorkRequest::default()
        },
        limit,
    )
    .unwrap();
    let points = sample_points_paid(&lo, &hi, count, &budget, limit).unwrap();
    lease.mark_materialized().unwrap();
    for i in 0..lo.len() {
        for s in 0..count {
            let x = points[[i, s]];
            assert!(x.is_finite() && x >= lo[i] && x <= hi[i]);
        }
    }
    assert_eq!(points[[0, 0]], 0.0);
    assert_eq!(points[[0, 1]], -f64::MAX);
    assert_eq!(points[[0, 2]], f64::MAX);
    assert_eq!(points[[2, 1]].to_bits(), lo[2].to_bits());
    assert_eq!(points[[2, 2]].to_bits(), hi[2].to_bits());
    drop(points);
    lease.complete().unwrap();
    drop(lease);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn prepared_publication_partial_sampling_cannot_erase_an_observed_downgrade_violation() {
    let mut audit = ContainmentAudit::default();
    // A reference value1 exceeds this deliberately invalid diagnostic box.
    // This is only a guard control: no numeric certificate/token is minted.
    audit.check_box("head_y", 0, 0, 0.0, 0.25, 1.0);
    assert_eq!(audit.violations(), 1);
    let out = containment_completion(
        Err("later original phase deadline".to_owned()),
        audit,
        32,
        0.0,
    );
    assert!(out.skipped.is_some());
    assert_eq!(out.points, 0);
    assert_eq!(out.audit.checks(), 1);
    assert_eq!(out.audit.violations(), 1);
    assert_eq!(out.report().trips()[0].guard, GuardId::Containment);
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct DirectedRounding {
    #[cfg(target_arch = "aarch64")]
    saved: u64,
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl DirectedRounding {
    #[allow(unsafe_code)]
    fn enter() -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let saved: u64;
            std::arch::asm!("mrs {saved}, fpcr",saved=out(reg)saved,options(nomem,nostack));
            let directed = (saved & !(3u64 << 22)) | (1u64 << 22);
            std::arch::asm!("msr fpcr, {directed}","isb",directed=in(reg)directed,options(nomem,nostack));
            Self { saved }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let saved = std::arch::x86_64::_mm_getcsr();
            std::arch::x86_64::_mm_setcsr((saved & !(3 << 13)) | (1 << 13));
            Self { saved }
        }
    }
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl Drop for DirectedRounding {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("msr fpcr, {saved}","isb",saved=in(reg)self.saved,options(nomem,nostack));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved);
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn prepared_publication_live_unqualified_environment_refuses_existing_proof_before_any_new_work_admission(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let (src, terminal) = pair(fixture(false, true), &budget, None);
    let pubn = authorize_prepared_terminal(
        src,
        terminal,
        0,
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    let before = budget.receipt().unwrap();
    let bytes = memory.receipt().unwrap();
    {
        let _directed = DirectedRounding::enter();
        assert!(!ny_core::has_f64_interval_proof_environment());
        assert!(matches!(
            pubn.checkpoint(Some(at())),
            Err(NyError::SoundnessRefusal(_))
        ));
    }
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(memory.receipt().unwrap(), bytes);
    pubn.checkpoint(Some(at())).unwrap();
    drop(pubn);
    assert_eq!(live(&memory), 0);
}
