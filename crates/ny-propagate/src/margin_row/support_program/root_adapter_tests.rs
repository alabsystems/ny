// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Genuine source projection, policy, resource and original-model controls.

use super::super::compiler::OriginalModel;
use super::super::construction::{build_prepared_root, PreparedConstructionOptions};
use super::super::prefix::OriginalRoot;
use super::super::publication::share_prepared_root;
use super::super::retained::PreparedSplit;
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkBudget, WorkLimits, WorkReserve};
use crate::margin_row::engine::BackwardEngine;
use crate::margin_row::input_clipping::InputSplitSide;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 128 << 20;
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
fn fixture(fold: bool) -> TwinSpec {
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.25],
                bias: vec![-0.25],
                bias_err: vec![if fold { 1.0 / 512.0 } else { 0.0 }],
                weight_rel_err: if fold { 1.0 / 256.0 } else { 0.0 },
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![-0.5],
                shift: vec![-0.1],
                shift_err: vec![if fold { 1.0 / 1024.0 } else { 0.0 }],
                scale_rel_err: if fold { 1.0 / 512.0 } else { 0.0 },
                shape: (1, 1, 1),
            },
            TwinOpSpec::Add { lhs: 3, rhs: 2 },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: vec![0.5, -0.25],
                bias: vec![0.5, 0.75],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: vec![0.75, 0.25, -0.25, 0.5],
                bias: vec![0.5, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    }
}
fn source(budget: &WorkBudget, fold: bool, keep: bool, f32: bool) -> PreparedRootSource {
    source_with_reference(
        budget,
        fold,
        keep,
        f32,
        super::super::PreparedReferenceArithmetic::F64,
    )
}
fn source_with_reference(
    budget: &WorkBudget,
    fold: bool,
    keep: bool,
    f32: bool,
    reference_arithmetic: super::super::PreparedReferenceArithmetic,
) -> PreparedRootSource {
    let model = OriginalModel::compile(fixture(fold), budget, Some(at())).unwrap();
    let root = OriginalRoot::new(
        model,
        &[-1.0],
        &[1.0],
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    let constructed = build_prepared_root(
        root,
        PreparedConstructionOptions {
            reference_arithmetic,
            keep_input_rows: keep,
            row_f32: f32,
            prefer_patch: false,
            ..PreparedConstructionOptions::default()
        },
        Some(at()),
    )
    .unwrap();
    share_prepared_root(constructed, Some(at())).unwrap()
}

#[test]
fn prepared_root_binary32_reference_keeps_every_tableau_counter_and_request_honest() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = source_with_reference(
        &budget,
        true,
        true,
        true,
        super::super::PreparedReferenceArithmetic::Binary32Certified,
    );
    let reference = source.construction().reference_receipt();
    let adapter = project(source, true, true).unwrap();
    assert_eq!(adapter.receipt().reference, reference);
    assert!(reference.multiply_add_pairs > 0);
    assert_eq!(
        (reference.completed_conv_ops, reference.completed_affine_ops),
        (1, 1)
    );
    assert_eq!(
        adapter.gates().root_precision_execution,
        RootPrecisionExecutionEvidence {
            outward_eligible: true,
            ..RootPrecisionExecutionEvidence::default()
        }
    );
    assert!(!adapter.receipt().precision.root_f32 && !adapter.receipt().precision.tableau_f32);
    assert!(adapter.receipt().source.f32_tiles > 0);
    adapter.checkpoint(None).unwrap();
    drop(adapter);
    assert_eq!(live(&memory), 0);
}
fn project(source: PreparedRootSource, clip: bool, retain: bool) -> Result<PreparedRootAdapter> {
    project_prepared_root(
        source,
        RootPrecisionRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
        clip,
        retain.then(RetainCfg::default),
        Some(at()),
    )
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn r(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap()
}

#[test]
fn prepared_root_adapter_actual_model_borrow_and_every_gate_domain_bit_are_preserved() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let src = source(&budget, true, true, false);
    let adapter = project(src.clone(), true, true).unwrap();
    adapter.validate_source(&src, None).unwrap();
    assert!(std::ptr::eq(
        adapter.net(),
        src.construction().prefix().root().model().net()
    ));
    let original = src.construction().prefix().root();
    for (copied, genuine) in [
        (adapter.gates().lo.as_slice(), original.lo()),
        (&adapter.gates().hi, original.hi()),
        (&adapter.gates().mid, original.mid()),
        (&adapter.gates().rad, original.rad()),
        (&adapter.gates().xabs, original.xabs()),
    ] {
        assert_eq!(bits(copied), bits(genuine));
        assert_ne!(copied.as_ptr(), genuine.as_ptr());
    }
    let frame = src.construction().prefix().frame(None).unwrap();
    for (copy, genuine) in adapter.gates().layers.iter().zip(frame.layers()) {
        assert_eq!((copy.op, copy.n), (genuine.op, genuine.n));
        assert_eq!(copy.unst, genuine.unst);
        for (a, b) in [
            (&copy.l, &genuine.l),
            (&copy.u, &genuine.u),
            (&copy.alpha, &genuine.alpha),
            (&copy.s, &genuine.s),
            (&copy.c, &genuine.c),
            (&copy.ms, &genuine.ms),
        ] {
            assert_eq!(bits(a), bits(b));
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        assert!(copy.clip_rows.is_none());
    }
    assert!(adapter.gates().head_group_source.is_none());
    assert_eq!(adapter.receipt().reference_conv_ops, 1);
    assert_eq!(
        adapter.gates().root_precision_execution.root_f64_conv_ops,
        0
    );
    assert_eq!(
        adapter
            .gates()
            .root_precision_execution
            .tableau_f64_store_ops,
        0
    );
    assert_eq!(
        adapter.gates().backward_interm_execution.columns_completed,
        0
    );
    assert_eq!(adapter.receipt().source.derived_ops, adapter.net().i_gemm1);
}

#[test]
fn prepared_root_adapter_keeps_actual_f32_row_errors_separate_from_f64_gate_compute() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let src = source(&budget, true, true, true);
    let adapter = project(src, true, true).unwrap();
    let clip = adapter.clip_rows().unwrap();
    let (rows, column) = clip.row(1, 0, None).unwrap();
    assert_eq!(
        rows.backing(),
        crate::margin_row::input_rows::InputRowsBacking::F32
    );
    assert!(rows.lower(column).unwrap().pointwise_error(at()).unwrap() > 0.0);
    assert!(rows.upper(column).unwrap().pointwise_error(at()).unwrap() > 0.0);
    assert!(!adapter.receipt().precision.root_f32 && !adapter.receipt().precision.tableau_f32);
    assert_eq!(
        adapter.gates().root_precision_execution.root_f32_conv_ops,
        0
    );
    assert_eq!(
        adapter
            .gates()
            .root_precision_execution
            .tableau_f32_store_ops,
        0
    );
    assert!(adapter.receipt().source.f32_tiles > 0);
    assert_eq!(
        adapter.receipt().retained_payload_bytes,
        adapter.retained_rows().unwrap().payload_bytes()
    );
}

#[test]
fn prepared_root_adapter_original_signed_fold_family_remains_enclosed_through_ordinary_kernel() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let adapter = project(source(&budget, true, true, false), true, false).unwrap();
    // Explicitly admit the legacy test walk on the same budget. Production
    // per-pass/memo integration is an ordinary caller obligation, not a claim
    // granted by the immutable projection's initial array allocation.
    let mut walk = admit(adapter.source(), 0, 16 << 20, 1_000_000_000, at()).unwrap();
    let engine = BackwardEngine::new(adapter.net(), adapter.gates());
    let (lower, upper) = engine.y_rows(None).unwrap();
    let lo = engine.concretize_lower(&lower);
    let hi = engine.concretize_upper(&upper);
    for x in [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0] {
        for signs in 0_usize..16 {
            let sign = |bit| {
                if signs & (1_usize << bit) == 0_usize {
                    r(-1.0)
                } else {
                    r(1.0)
                }
            };
            let h = (r(1.25) * (r(1.0) + sign(0) * r(1.0 / 256.0)) * r(x) - r(0.25)
                + sign(1) * r(1.0 / 512.0))
            .max(BigRational::zero());
            let affine = r(-0.5) * (r(1.0) + sign(2) * r(1.0 / 512.0)) * &h - r(0.1)
                + sign(3) * r(1.0 / 1024.0);
            let z = (affine + h).max(BigRational::zero());
            let exact = [r(0.5) * &z + r(0.5), r(-0.25) * z + r(0.75)];
            for j in 0..2 {
                assert!(r(lo[j]) <= exact[j] && exact[j] <= r(hi[j]));
            }
        }
    }
    drop((lower, upper, lo, hi));
    walk.complete().unwrap();
}

#[test]
fn prepared_root_adapter_requested_precision_and_enabled_bi_dose_refuse_before_new_admission() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let src = source(&budget, false, true, false);
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    for precision in [
        RootPrecisionRuntimePolicy {
            root_f32: true,
            ..RootPrecisionRuntimePolicy::default()
        },
        RootPrecisionRuntimePolicy {
            tableau_f32: true,
            ..RootPrecisionRuntimePolicy::default()
        },
        RootPrecisionRuntimePolicy {
            err_lane: ErrLane::Envelope,
            ..RootPrecisionRuntimePolicy::default()
        },
    ] {
        assert!(project_prepared_root(
            src.clone(),
            precision,
            BackwardIntermRuntimePolicy::default(),
            true,
            None,
            None
        )
        .is_err());
    }
    let bi = BackwardIntermRuntimePolicy {
        enabled: true,
        ..BackwardIntermRuntimePolicy::default()
    };
    assert!(project_prepared_root(
        src.clone(),
        RootPrecisionRuntimePolicy::default(),
        bi,
        true,
        None,
        None
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    assert_eq!(live(&memory), before);
    let disabled = BackwardIntermRuntimePolicy {
        enabled: false,
        topk: 17,
        chunk: 3,
        ..bi
    };
    let adapter = project_prepared_root(
        src,
        RootPrecisionRuntimePolicy::default(),
        disabled,
        false,
        None,
        None,
    )
    .unwrap();
    assert_eq!(adapter.receipt().backward_interm, disabled);
    assert_eq!(
        adapter.gates().backward_interm_execution.configured_topk,
        17
    );
    assert_eq!(
        adapter.gates().backward_interm_execution.configured_chunk,
        3
    );
    assert_eq!(
        adapter.gates().backward_interm_execution.chunks_completed,
        0
    );
}

#[test]
fn prepared_root_adapter_requested_rows_cannot_disappear_on_unretained_source() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let src = source(&budget, false, false, false);
    assert!(project(src.clone(), true, false).is_err());
    assert!(project(src.clone(), false, true).is_err());
    let adapter = project(src, false, false).unwrap();
    assert!(adapter.clip_rows().is_none() && adapter.retained_rows().is_none());
    assert!(!adapter.receipt().clipping_requested && !adapter.receipt().retained_requested);
    adapter.checkpoint(None).unwrap();
}

#[test]
fn prepared_root_adapter_sharing_and_detached_clip_batch_retain_exact_last_owner_claims() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let src = source(&budget, false, true, false);
    let adapter = project(src.clone(), true, true).unwrap();
    let before = live(&memory);
    let alias = adapter.clone();
    assert!(adapter.same_owner(&alias));
    assert!(std::ptr::eq(adapter.gates(), alias.gates()));
    assert_eq!(live(&memory), before);
    drop(src);
    assert_eq!(live(&memory), before);
    let batch = adapter
        .clip_rows()
        .unwrap()
        .halfspaces(
            &[PreparedSplit {
                layer: 0,
                target: 0,
                side: InputSplitSide::Active,
            }],
            None,
        )
        .unwrap();
    let with_batch = live(&memory);
    let removable = adapter.resident_bytes()
        + adapter.clip_rows().unwrap().resident_bytes()
        + adapter.retained_rows().unwrap().resident_bytes();
    drop(adapter);
    assert_eq!(live(&memory), with_batch);
    alias.checkpoint(None).unwrap();
    drop(alias);
    assert_eq!(live(&memory), with_batch - removable);
    batch.validate_source(batch.source(), None).unwrap();
    drop(batch);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_root_adapter_original_cap_and_expired_cancelled_clock_refuse_before_projection_allocation(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let src = source(&budget, false, false, false);
    let before = live(&memory);
    let peak = memory.receipt().unwrap().peak_leased_bytes;
    let admitted = budget.receipt().unwrap().admitted;
    assert!(project_prepared_root(
        src.clone(),
        RootPrecisionRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
        false,
        None,
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap()
        )
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    budget.tighten_total_byte_limit(before).unwrap();
    assert!(project(src.clone(), false, false).is_err());
    assert_eq!(live(&memory), before);
    assert_eq!(memory.receipt().unwrap().peak_leased_bytes, peak);
    cancel.cancel();
    assert!(project(src, false, false).is_err());
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_root_adapter_cancellation_after_actual_first_gate_copy_discards_the_partial_projection()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let src = source(&budget, false, false, false);
    let before = live(&memory);
    let mut copied = false;
    let mut observe = |layer| {
        assert_eq!(layer, 0);
        copied = true;
        cancel.cancel();
    };
    assert!(project_inner(
        src.clone(),
        RootProductDemand {
            precision: RootPrecisionRuntimePolicy::default(),
            clipping: false,
            retention: None,
        },
        BackwardIntermRuntimePolicy::default(),
        Some(at()),
        Some(&mut observe)
    )
    .is_err());
    assert!(copied);
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(src);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_root_adapter_scope_and_successful_earliest_deadline_are_exact() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let first = source(&budget, false, true, false);
    let second = source(&budget, false, true, false);
    let early = first
        .deadline(None)
        .unwrap()
        .checked_sub(Duration::from_secs(10))
        .unwrap();
    let adapter = project_prepared_root(
        first.clone(),
        RootPrecisionRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
        true,
        Some(RetainCfg::default()),
        Some(early),
    )
    .unwrap();
    assert_eq!(adapter.deadline(None).unwrap(), early);
    assert_eq!(adapter.clip_rows().unwrap().deadline(None).unwrap(), early);
    assert_eq!(
        adapter.retained_rows().unwrap().deadline(None).unwrap(),
        early
    );
    adapter
        .validate_source(&first, Some(early + Duration::from_secs(1)))
        .unwrap();
    assert!(adapter.validate_source(&second, None).is_err());
}

#[test]
fn prepared_root_product_demand_keeps_none_and_requested_zero_retention_distinct_on_original_budget(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let src = source(&budget, false, true, false);
    let source_bytes = live(&memory);
    let policy = RootPrecisionRuntimePolicy::default();
    let bi = BackwardIntermRuntimePolicy::default();
    let zero = RetainCfg {
        per_layer: 0,
        budget_bytes: 0,
    };
    let absent = project_prepared_root(src.clone(), policy, bi, false, None, Some(at())).unwrap();
    assert!(!absent.receipt().retained_requested);
    assert!(absent.retained_rows().is_none());
    assert_eq!(live(&memory), source_bytes + absent.resident_bytes());
    let absent_bytes = absent.resident_bytes();
    let present =
        project_prepared_root(src.clone(), policy, bi, false, Some(zero), Some(at())).unwrap();
    assert!(present.receipt().retained_requested);
    assert_eq!(present.receipt().retained_per_layer, 0);
    assert_eq!(present.receipt().retained_budget_bytes, 0);
    assert_eq!(present.receipt().retained_payload_bytes, 0);
    assert!(present.source().same_source(&src));
    assert!(absent.source().same_source(&src));
    assert!(!present.same_owner(&absent));
    let retained = present.retained_rows().unwrap().clone();
    assert_eq!(retained.cfg(), zero);
    assert_eq!(retained.payload_bytes(), 0);
    assert_eq!(retained.layers().len(), src.construction().layers().len());
    assert!(retained.layers().iter().all(|layer| layer.ids().is_empty()));
    assert!(retained.source().budget().shares_ledger(&budget));
    retained.validate_source(&src, None).unwrap();
    present.checkpoint(None).unwrap();
    absent.checkpoint(None).unwrap();
    let retained_bytes = retained.resident_bytes();
    assert!(retained_bytes > 0);
    assert_eq!(
        live(&memory),
        source_bytes + absent_bytes + present.resident_bytes() + retained_bytes
    );
    let spent = budget.receipt().unwrap();
    drop(present);
    assert_eq!(live(&memory), source_bytes + absent_bytes + retained_bytes);
    drop(retained);
    assert_eq!(live(&memory), source_bytes + absent_bytes);
    drop(absent);
    assert_eq!(live(&memory), source_bytes);
    drop(src);
    assert_eq!(live(&memory), 0);
    let released = budget.receipt().unwrap();
    assert_eq!(released.work_reserved, spent.work_reserved);
    assert_eq!(released.proposals_reserved, spent.proposals_reserved);

    // Reuse the SAME original ledger, including all already-spent allowance.
    // A zero-sized request still requires genuine complete retained coverage.
    let src = source(&budget, false, false, false);
    let absent = project_prepared_root(src.clone(), policy, bi, false, None, Some(at())).unwrap();
    let before = budget.receipt().unwrap();
    let before_bytes = live(&memory);
    let refused = project_prepared_root(src.clone(), policy, bi, false, Some(zero), Some(at()));
    assert!(matches!(
        refused,
        Err(NyError::InvalidSpec(message))
            if message == "original support program: prepared retention requires complete requested input rows"
    ));
    assert_eq!(live(&memory), before_bytes);
    let after = budget.receipt().unwrap();
    assert!(after.work_reserved >= before.work_reserved);
    assert!(before.work_reserved >= spent.work_reserved);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, 0);
    drop(absent);
    drop(src);
    assert_eq!(live(&memory), 0);
    let released = budget.receipt().unwrap();
    assert_eq!(released.work_reserved, after.work_reserved);
    assert_eq!(released.proposals_reserved, after.proposals_reserved);
}
