// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Actual producer-row refinement transactions, allocation moves and failures.
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::engine::run_prepared_prefix_input_rows;
use crate::margin_row::patch::{run_prepared_prefix_patch_input_rows, PreparedPatchPlan};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;
const CAP: usize = 128 << 20;
fn authority(cap: usize) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(cap.checked_mul(2).unwrap());
    let token = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &token,
        WorkLimits {
            deadline: None,
            scratch_bytes: cap,
            persistent_bytes: cap,
            max_work: 1_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, token)
}
fn at() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn live(memory: &MemoryAuthority) -> usize {
    let receipt = memory.receipt().unwrap();
    receipt.live_pending_bytes + receipt.live_materialized_bytes
}
fn fixture(shifts: &[f64], fold: bool, spatial: bool) -> TwinSpec {
    let n = shifts.len();
    let rel = if fold { 0.003 } else { 0.0 };
    let (kernel, shape, weight) = if spatial {
        ((1, 1, 1, 1), (1, 1, n), vec![1.0])
    } else {
        (
            (n, n, 1, 1),
            (n, 1, 1),
            (0..n * n)
                .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
                .collect(),
        )
    };
    TwinSpec {
        n_in: n,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight,
                bias: vec![0.0; kernel.0],
                bias_err: vec![if fold { 0.001 } else { 0.0 }; kernel.0],
                weight_rel_err: rel,
                kernel,
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: shape,
                oshape: shape,
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![-1.0; if spatial { 1 } else { n }],
                shift: if spatial {
                    vec![shifts[0]]
                } else {
                    shifts.to_vec()
                },
                shift_err: vec![if fold { 0.002 } else { 0.0 }; if spatial { 1 } else { n }],
                scale_rel_err: rel,
                shape,
            },
            TwinOpSpec::Add { lhs: 3, rhs: 2 },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: vec![0.25; n],
                bias: vec![0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, n),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: vec![0.5; 2],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    }
}
fn prepare(
    shifts: &[f64],
    fold: bool,
    spatial: bool,
    budget: &WorkBudget,
) -> (PreparedPrefix, CertifiedTensorBox) {
    let model = OriginalModel::compile(fixture(shifts, fold, spatial), budget, None).unwrap();
    let root = OriginalRoot::new(
        model,
        &vec![-1.0; shifts.len()],
        &vec![1.0; shifts.len()],
        RootPrecisionRuntimePolicy::default(),
        None,
    )
    .unwrap();
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let (prefix, relu) = root
        .empty_prefix()
        .unwrap()
        .commit_relu(&conv, 1, None)
        .unwrap();
    let affine = root.derive(2, &relu, None).unwrap();
    let add = root.derive(3, &affine, Some(&relu)).unwrap();
    (prefix, add)
}
fn rows(prefix: &PreparedPrefix, targets: &[usize], deadline: Instant) -> InputRowPair<'static> {
    run_prepared_prefix_input_rows(prefix, targets, Some(deadline)).unwrap()
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn rat(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}
#[test]
fn refinement_transaction_moves_endpoint_allocations_and_claim_once_across_ordered_partial_tiles() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP);
    let (prefix, pre) = prepare(&[0.1, -0.2, 0.3, -0.4], false, false, &budget);
    let d = at();
    let lp = pre.lo.as_ptr();
    let up = pre.hi.as_ptr();
    let lc = pre.lo.capacity();
    let uc = pre.hi.capacity();
    let claim = pre._resident.bytes();
    let reference = prefix
        .refine_relu(&pre, 4, &rows(&prefix, &[2, 0, 3, 1], d), Some(d))
        .unwrap();
    let mut accumulator = prefix.begin_refinement(pre, 4, Some(d)).unwrap();
    assert_eq!(accumulator.preactivation().lo.as_ptr(), lp);
    assert_eq!(accumulator.preactivation().hi.as_ptr(), up);
    for target in [2, 0, 3, 1] {
        let tile = rows(&prefix, &[target], d);
        let persistent = budget.receipt().unwrap().live_persistent_bytes;
        accumulator.apply_rows(&tile, Some(d)).unwrap();
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, persistent);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(accumulator.preactivation().lo.as_ptr(), lp);
        assert_eq!(accumulator.preactivation().hi.as_ptr(), up);
    }
    let refined = accumulator.finish().unwrap();
    assert_eq!(refined.lo.as_ptr(), lp);
    assert_eq!(refined.hi.as_ptr(), up);
    assert_eq!(refined.lo.capacity(), lc);
    assert_eq!(refined.hi.capacity(), uc);
    assert_eq!(refined._resident.bytes(), claim);
    assert_eq!(bits(refined.lo()), bits(reference.lo()));
    assert_eq!(bits(refined.hi()), bits(reference.hi()));
    let (next, _) = prefix.commit_relu(&refined, 4, None).unwrap();
    assert_eq!(next.gate(1).unwrap().unst.len(), 0);
}
#[test]
fn refinement_transaction_signed_fold_rows_contain_exact_original_residual_values() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP);
    let (prefix, pre) = prepare(&[0.1, -0.2], true, false, &budget);
    let d = at();
    let tile = rows(&prefix, &[1, 0], d);
    let mut accumulator = prefix.begin_refinement(pre, 4, None).unwrap();
    accumulator.apply_rows(&tile, Some(d)).unwrap();
    let refined = accumulator.finish().unwrap();
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        for b in 0..16 {
            for i in 0..2 {
                let sign = |k: usize| if b & (1_usize << k) == 0 { -1.0 } else { 1.0 };
                let h = ((rat(1.0) + rat(sign(0)) * rat(0.003)) * rat(x)
                    + rat(sign(1)) * rat(0.001))
                .max(BigRational::zero());
                let exact = -(rat(1.0) + rat(sign(2)) * rat(0.003)) * &h
                    + rat([0.1, -0.2][i])
                    + rat(sign(3)) * rat(0.002)
                    + h;
                assert!(rat(refined.lo[i]) <= exact && exact <= rat(refined.hi[i]));
            }
        }
    }
}
#[test]
fn refinement_transaction_genuine_patch_products_tighten_without_repacking_or_dense_scatter() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP);
    let (prefix, pre) = prepare(&[0.1; 5], false, true, &budget);
    let d = at();
    let plan = PreparedPatchPlan::build(&prefix, 1.0, Some(d)).unwrap();
    let tile = run_prepared_prefix_patch_input_rows(&plan, &[4, 0, 2], Some(d)).unwrap();
    let before = bits(pre.lo());
    let lp = pre.lo.as_ptr();
    let mut accumulator = prefix.begin_refinement(pre, 4, None).unwrap();
    accumulator.apply_rows(&tile, None).unwrap();
    drop(plan);
    drop(tile);
    let refined = accumulator.finish().unwrap();
    assert_eq!(refined.lo.as_ptr(), lp);
    for i in [4, 0, 2] {
        assert!(refined.lo[i] > 0.0 && refined.hi[i] < 0.1000001);
    }
    for i in [1, 3] {
        assert_eq!(refined.lo[i].to_bits(), before[i]);
    }
}
#[test]
fn refinement_transaction_rejects_foreign_prefix_tensor_and_op_before_mutation() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP);
    let (prefix, pre) = prepare(&[0.1, -0.2], false, false, &budget);
    let (foreign, foreign_pre) = prepare(&[0.1, -0.2], false, false, &budget);
    let d = at();
    assert!(prefix.begin_refinement(foreign_pre, 4, None).is_err());
    let mut accumulator = prefix.begin_refinement(pre, 4, None).unwrap();
    let before = (
        bits(accumulator.preactivation().lo()),
        bits(accumulator.preactivation().hi()),
    );
    let tile = rows(&foreign, &[0], d);
    assert!(accumulator.apply_rows(&tile, None).is_err());
    assert_eq!(bits(accumulator.preactivation().lo()), before.0);
    assert_eq!(bits(accumulator.preactivation().hi()), before.1);
    let input = prefix.root().input_box().unwrap();
    assert!(prefix.begin_refinement(input, 4, None).is_err());
    let output = accumulator.finish().unwrap();
    assert!(prefix.begin_refinement(output, 1, None).is_err());
}
#[test]
fn refinement_transaction_cancellation_during_commit_rolls_back_all_endpoint_bits() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, token) = authority(CAP);
    let (prefix, pre) = prepare(&[0.1, -0.2], false, false, &budget);
    let d = at();
    let tile = rows(&prefix, &[0, 1], d);
    let mut accumulator = prefix.begin_refinement(pre, 4, None).unwrap();
    let old = (
        bits(accumulator.preactivation().lo()),
        bits(accumulator.preactivation().hi()),
        accumulator.deadline,
    );
    let lp = accumulator.preactivation().lo.as_ptr();
    let before = live(&memory);
    let mut writes = 0;
    assert!(accumulator
        .apply_rows_inner(
            &tile,
            None,
            Some(&mut |_| {
                writes += 1;
                token.cancel();
            })
        )
        .is_err());
    assert!(writes > 0);
    assert_eq!(bits(accumulator.preactivation().lo()), old.0);
    assert_eq!(bits(accumulator.preactivation().hi()), old.1);
    assert_eq!(accumulator.deadline, old.2);
    assert_eq!(accumulator.preactivation().lo.as_ptr(), lp);
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}
#[test]
fn refinement_transaction_expiry_is_atomic_and_successful_noop_inherits_earliest_rows_deadline() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP);
    let (prefix, pre) = prepare(&[0.1, -0.2], false, false, &budget);
    let late = at() + Duration::from_secs(30);
    let early = at();
    let tile = rows(&prefix, &[0, 1], early);
    let mut accumulator = prefix.begin_refinement(pre, 4, Some(late)).unwrap();
    assert_eq!(accumulator.preactivation().deadline().unwrap(), Some(late));
    let before = bits(accumulator.preactivation().lo());
    assert!(accumulator
        .apply_rows(
            &tile,
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
        )
        .is_err());
    assert_eq!(bits(accumulator.preactivation().lo()), before);
    assert_eq!(accumulator.deadline, Some(late));
    accumulator.apply_rows(&tile, Some(late)).unwrap();
    let first = (
        bits(accumulator.preactivation().lo()),
        bits(accumulator.preactivation().hi()),
    );
    accumulator.apply_rows(&tile, Some(late)).unwrap();
    assert_eq!(bits(accumulator.preactivation().lo()), first.0);
    assert_eq!(bits(accumulator.preactivation().hi()), first.1);
    assert_eq!(accumulator.deadline, Some(early));
    assert_eq!(
        accumulator.finish().unwrap().deadline().unwrap(),
        Some(early)
    );
}
#[test]
fn refinement_transaction_large_global_cap_cannot_allocate_tile_scratch_past_local_source_ceiling()
{
    let _env = crate::tests::lock_env_shared();
    let cap = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
    let (budget, memory, _) = authority(cap * 2);
    let (prefix, pre) = prepare(&[0.1, -0.2], false, false, &budget);
    let d = at();
    let tile = rows(&prefix, &[0, 1], d);
    let mut accumulator = prefix.begin_refinement(pre, 4, None).unwrap();
    let scratch = 2 * size_of::<RefinementProposal>() + OWNER_METADATA;
    let current = live(&memory);
    let mut blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: cap - current - scratch + 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    blocker.mark_materialized().unwrap();
    let blocker = blocker.commit_persistent().unwrap();
    let before = memory.receipt().unwrap();
    let lo = bits(accumulator.preactivation().lo());
    assert!(matches!(
        accumulator.apply_rows(&tile, None),
        Err(NyError::CpuMemoryExceeded { .. })
    ));
    assert_eq!(
        memory.receipt().unwrap().peak_leased_bytes,
        before.peak_leased_bytes
    );
    assert_eq!(
        live(&memory),
        before.live_materialized_bytes + before.live_pending_bytes
    );
    assert_eq!(bits(accumulator.preactivation().lo()), lo);
    drop(blocker);
    accumulator.apply_rows(&tile, None).unwrap();
}
#[test]
fn refinement_transaction_last_owner_release_keeps_original_model_prefix_and_pre_until_finish_drop()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority(CAP);
    let (prefix, pre) = prepare(&[0.1, -0.2], false, false, &budget);
    let d = at();
    let tile = rows(&prefix, &[0], d);
    let mut accumulator = prefix.begin_refinement(pre, 4, None).unwrap();
    accumulator.apply_rows(&tile, None).unwrap();
    drop(tile);
    drop(prefix);
    assert!(live(&memory) > 0);
    let pre = accumulator.finish().unwrap();
    assert!(pre.lo[0] > 0.0);
    assert!(live(&memory) > 0);
    drop(pre);
    assert_eq!(live(&memory), 0);
}
#[test]
fn refinement_transaction_zero_ties_and_subnormal_inputs_preserve_exact_rational_zero() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP);
    let tiny = f64::from_bits(1);
    let model = OriginalModel::compile(fixture(&[0.0, -0.0], false, false), &budget, None).unwrap();
    let root = OriginalRoot::new(
        model,
        &[-tiny, -tiny],
        &[tiny, tiny],
        RootPrecisionRuntimePolicy::default(),
        None,
    )
    .unwrap();
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let (prefix, relu) = root
        .empty_prefix()
        .unwrap()
        .commit_relu(&conv, 1, None)
        .unwrap();
    let affine = root.derive(2, &relu, None).unwrap();
    let pre = root.derive(3, &affine, Some(&relu)).unwrap();
    let d = at();
    let tile = rows(&prefix, &[1, 0], d);
    let mut accumulator = prefix.begin_refinement(pre, 4, None).unwrap();
    accumulator.apply_rows(&tile, None).unwrap();
    let refined = accumulator.finish().unwrap();
    for i in 0..2 {
        assert!(rat(refined.lo[i]) <= BigRational::zero());
        assert!(BigRational::zero() <= rat(refined.hi[i]));
    }
}
#[test]
fn refinement_transaction_crossed_intersection_fault_is_detected_before_any_target_writes() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP);
    let (prefix, pre) = prepare(&[0.1, -0.2], false, false, &budget);
    let d = at();
    let tile = rows(&prefix, &[0, 1], d);
    let mut accumulator = prefix.begin_refinement(pre, 4, None).unwrap();
    // Deliberate private endpoint fault, never supplied as source authority.
    // The row pair itself is the actual original producer's complete result.
    // A correctly certified unsplit box and rows cannot mathematically cross.
    accumulator.pre.as_mut().unwrap().lo[1] = 2.0;
    accumulator.pre.as_mut().unwrap().hi[1] = 3.0;
    let old = (
        bits(accumulator.preactivation().lo()),
        bits(accumulator.preactivation().hi()),
    );
    assert!(accumulator.apply_rows(&tile, None).is_err());
    assert_eq!(bits(accumulator.preactivation().lo()), old.0);
    assert_eq!(bits(accumulator.preactivation().hi()), old.1);
}
