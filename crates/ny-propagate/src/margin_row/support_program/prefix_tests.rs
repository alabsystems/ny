// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Independent rational containment, scope and shared-resource controls.

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::engine::run_prepared_prefix_input_rows;
use crate::margin_row::input_rows::{InputRowPair, InputRowsBacking};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 128 << 20;

fn budget_with(deadline: Option<Instant>) -> (WorkBudget, CancellationToken, MemoryAuthority) {
    let memory = MemoryAuthority::new(CAP * 2);
    let cancellation = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline,
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 1_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, cancellation, memory)
}

fn oversized_budget() -> (WorkBudget, CancellationToken, MemoryAuthority) {
    let cap = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP + CAP;
    let memory = MemoryAuthority::new(cap + CAP);
    let cancellation = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: None,
            scratch_bytes: CAP,
            persistent_bytes: cap,
            max_work: 1_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, cancellation, memory)
}

fn paid_claim(budget: &WorkBudget, bytes: usize) -> PersistentReservation {
    budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: bytes,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap()
}

// Resource-only overpayments on genuine producer owners. Actual numbers,
// model/root/tensor identity and immutable deadlines never change. No large
// allocation, new ledger, fabricated intervals or gate completion is involved.
fn reprice_box(value: &mut CertifiedTensorBox, bytes: usize) {
    let claim = paid_claim(value.root().budget(), bytes);
    value._resident = claim;
}

fn reprice_root(value: &mut OriginalRoot, bytes: usize) {
    let claim = paid_claim(value.budget(), bytes);
    Arc::get_mut(&mut value.0)
        .expect("sole original root owner")
        ._resident = claim;
}

fn reprice_prefix(value: &mut PreparedPrefix, bytes: usize) {
    let claim = paid_claim(value.root().budget(), bytes);
    Arc::get_mut(value.inner.as_mut().unwrap())
        .expect("sole prefix generation owner")
        ._resident = claim;
}

fn head(ops: &mut Vec<TwinOpSpec>, input: usize, n: usize, outputs: usize) {
    let k = ops.len();
    assert_eq!(input, k);
    ops.push(TwinOpSpec::Gemm {
        input,
        weight: vec![0.25; n],
        bias: vec![0.0],
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (1, n),
    });
    ops.push(TwinOpSpec::Relu { input: k + 1 });
    ops.push(TwinOpSpec::Gemm {
        input: k + 2,
        weight: vec![0.125; outputs],
        bias: vec![0.0; outputs],
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (outputs, 1),
    });
}

fn fixture() -> TwinSpec {
    let mut ops = vec![
        TwinOpSpec::Conv {
            input: 0,
            weight: vec![-0.75],
            bias: vec![0.125],
            bias_err: vec![0.001],
            weight_rel_err: 0.003,
            kernel: (1, 1, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (1, 1, 2),
            oshape: (1, 1, 2),
        },
        TwinOpSpec::Relu { input: 1 },
        TwinOpSpec::ChannelAffine {
            input: 2,
            scale: vec![-1.25],
            shift: vec![0.4],
            scale_rel_err: 0.003,
            shift_err: vec![0.002],
            shape: (1, 1, 2),
        },
        TwinOpSpec::Add { lhs: 3, rhs: 2 },
        TwinOpSpec::Relu { input: 4 },
        TwinOpSpec::Flatten { input: 5 },
    ];
    head(&mut ops, 6, 2, 2);
    TwinSpec { n_in: 2, ops }
}

fn cancelling_fixture(shifts: [f64; 2]) -> TwinSpec {
    let mut ops = vec![
        TwinOpSpec::Conv {
            input: 0,
            weight: vec![1.0, 0.0, 0.0, 1.0],
            bias: vec![0.0; 2],
            bias_err: vec![0.0; 2],
            weight_rel_err: 0.0,
            kernel: (2, 2, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (2, 1, 1),
            oshape: (2, 1, 1),
        },
        TwinOpSpec::Relu { input: 1 },
        TwinOpSpec::ChannelAffine {
            input: 2,
            scale: vec![-1.0; 2],
            shift: shifts.to_vec(),
            scale_rel_err: 0.0,
            shift_err: vec![0.0; 2],
            shape: (2, 1, 1),
        },
        TwinOpSpec::Add { lhs: 3, rhs: 2 },
        TwinOpSpec::Relu { input: 4 },
        TwinOpSpec::Flatten { input: 5 },
    ];
    head(&mut ops, 6, 2, 2);
    TwinSpec { n_in: 2, ops }
}

fn cancelling_prefix(root: &OriginalRoot) -> (PreparedPrefix, CertifiedTensorBox) {
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let (one, relu) = root
        .empty_prefix()
        .unwrap()
        .commit_relu(&conv, 1, None)
        .unwrap();
    let affine = root.derive(2, &relu, None).unwrap();
    let add = root.derive(3, &affine, Some(&relu)).unwrap();
    (one, add)
}

fn products(prefix: &PreparedPrefix, targets: &[usize]) -> InputRowPair<'static> {
    run_prepared_prefix_input_rows(
        prefix,
        targets,
        Some(Instant::now() + Duration::from_secs(30)),
    )
    .unwrap()
}

fn root(spec: TwinSpec, lo: &[f64], hi: &[f64], budget: &WorkBudget) -> OriginalRoot {
    let model = OriginalModel::compile(spec, budget, None).unwrap();
    OriginalRoot::new(model, lo, hi, RootPrecisionRuntimePolicy::default(), None).unwrap()
}

fn rat(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite exact float")
}

fn contains(value: &CertifiedTensorBox, i: usize, exact: &BigRational) {
    assert!(
        rat(value.lo()[i]) <= *exact,
        "lower {} excludes {exact}",
        value.lo()[i]
    );
    assert!(
        *exact <= rat(value.hi()[i]),
        "upper {} excludes {exact}",
        value.hi()[i]
    );
}

fn line_contains(gate: &LayerGates, i: usize, x: &BigRational) {
    let y = x.clone().max(BigRational::zero());
    assert!(rat(gate.alpha[i]) * x <= y);
    assert!(y <= rat(gate.s[i]) * x + rat(gate.c[i]));
}

#[test]
fn folded_signed_reference_and_completed_lines_contain_exact_rational_members() {
    let (budget, _, _) = budget_with(None);
    let root = root(fixture(), &[-1.0, -0.75], &[1.25, 1.5], &budget);
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let empty = root.empty_prefix().unwrap();
    let (one, relu) = empty.commit_relu(&conv, 1, Some(&[0.37, 0.63])).unwrap();
    let affine = root.derive(2, &relu, None).unwrap();
    let add = root.derive(3, &affine, Some(&relu)).unwrap();
    let (two, final_relu) = one.commit_relu(&add, 4, None).unwrap();
    let flat = root.derive(5, &final_relu, None).unwrap();
    for sw in [-1.0, 1.0] {
        for sb in [-1.0, 1.0] {
            for ss in [-1.0, 1.0] {
                for st in [-1.0, 1.0] {
                    let w = rat(-0.75) + rat(sw) * rat(0.003) * rat(0.75);
                    let b = rat(0.125) + rat(sb) * rat(0.001);
                    let scale = rat(-1.25) + rat(ss) * rat(0.003) * rat(1.25);
                    let shift = rat(0.4) + rat(st) * rat(0.002);
                    for i in 0..2 {
                        for x in [root.lo()[i], 0.0, root.hi()[i]] {
                            let pre = &w * rat(x) + &b;
                            contains(&conv, i, &pre);
                            line_contains(one.gate(0).unwrap(), i, &pre);
                            let y = pre.max(BigRational::zero());
                            contains(&relu, i, &y);
                            let a = &scale * &y + &shift;
                            contains(&affine, i, &a);
                            let z = a + y;
                            contains(&add, i, &z);
                            line_contains(two.gate(1).unwrap(), i, &z);
                            contains(&flat, i, &z.max(BigRational::zero()));
                        }
                    }
                }
            }
        }
    }
    assert_eq!(two.layer_count(), 2);
    assert_eq!(one.gate(0).unwrap().alpha, vec![0.37, 0.63]);
    assert_eq!(root.region(), OriginalRegion::DeclaredBox);
    assert_eq!(root.mode(), RoundMode::Outward);
    assert!(root.budget().shares_ledger(&budget));
}

#[test]
fn convolution_spatial_padding_and_channel_mapping_match_independent_rationals() {
    let (budget, _, _) = budget_with(None);
    let mut ops = vec![
        TwinOpSpec::Conv {
            input: 0,
            weight: vec![0.5, -1.0, -0.25, 0.75, -0.5, 0.25, 1.25, -0.125],
            bias: vec![0.125, -0.375],
            bias_err: vec![0.0; 2],
            weight_rel_err: 0.0,
            kernel: (2, 2, 1, 2),
            stride: (1, 1),
            pads: (0, 1, 0, 0),
            ishape: (2, 1, 2),
            oshape: (2, 1, 2),
        },
        TwinOpSpec::Relu { input: 1 },
        TwinOpSpec::Flatten { input: 2 },
    ];
    head(&mut ops, 3, 4, 2);
    let lo = [-0.75, -0.25, -1.0, 0.125];
    let hi = [1.25, 0.5, 0.75, 1.0];
    let root = root(TwinSpec { n_in: 4, ops }, &lo, &hi, &budget);
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let weights = [0.5, -1.0, -0.25, 0.75, -0.5, 0.25, 1.25, -0.125];
    for mask in 0..16 {
        let x: Vec<_> = (0..4)
            .map(|i| rat(if mask & (1 << i) == 0 { lo[i] } else { hi[i] }))
            .collect();
        for ch in 0..2 {
            for outx in 0..2 {
                let mut exact = rat([0.125, -0.375][ch]);
                for input_ch in 0..2 {
                    for kernel_x in 0..2 {
                        let source_x = outx as isize + kernel_x as isize - 1;
                        if (0..2).contains(&source_x) {
                            exact += rat(weights[ch * 4 + input_ch * 2 + kernel_x])
                                * &x[input_ch * 2 + source_x as usize];
                        }
                    }
                }
                contains(&conv, ch * 2 + outx, &exact);
            }
        }
    }
}

#[test]
fn subnormal_multiplication_cancellation_and_signed_zero_are_enclosed() {
    let (budget, _, _) = budget_with(None);
    let tiny = f64::from_bits(1);
    let mut ops = vec![
        TwinOpSpec::Conv {
            input: 0,
            weight: vec![0.5, -0.5],
            bias: vec![-0.0],
            bias_err: vec![0.0],
            weight_rel_err: 0.0,
            kernel: (1, 2, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (2, 1, 1),
            oshape: (1, 1, 1),
        },
        TwinOpSpec::Relu { input: 1 },
        TwinOpSpec::Flatten { input: 2 },
    ];
    head(&mut ops, 3, 1, 2);
    let root = root(
        TwinSpec { n_in: 2, ops },
        &[tiny, -tiny],
        &[tiny, -tiny],
        &budget,
    );
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let exact = rat(0.5) * rat(tiny) + rat(-0.5) * rat(-tiny);
    contains(&conv, 0, &exact);
    assert!(conv.hi()[0] >= tiny);
    let (_, relu) = root
        .empty_prefix()
        .unwrap()
        .commit_relu(&conv, 1, None)
        .unwrap();
    contains(&relu, 0, &exact);
    let (zero, _) = uncertain_product(-0.0, 0.0, -tiny, tiny).unwrap();
    assert_eq!(zero, 0.0);
    assert_eq!(mul_down(-0.0, 2.0).to_bits(), (-0.0f64).to_bits());
    assert!(rat(mul_down(0.5, tiny)) <= rat(0.5) * rat(tiny));
    assert!(rat(0.5) * rat(tiny) <= rat(mul_up(0.5, tiny)));
    assert!(rat(add_down(1.0, -1.0)) <= BigRational::zero());
    assert!(BigRational::zero() <= rat(add_up(1.0, -1.0)));
}

#[test]
fn scope_and_original_order_refuse_relabelled_edges_and_gate_completion() {
    let (budget, _, _) = budget_with(None);
    let model = OriginalModel::compile(fixture(), &budget, None).unwrap();
    let first = OriginalRoot::new(
        model.clone(),
        &[-1.0; 2],
        &[1.0; 2],
        RootPrecisionRuntimePolicy::default(),
        None,
    )
    .unwrap();
    let other = OriginalRoot::new(
        model,
        &[-1.0; 2],
        &[1.0; 2],
        RootPrecisionRuntimePolicy::default(),
        None,
    )
    .unwrap();
    let input = first.input_box().unwrap();
    let other_input = other.input_box().unwrap();
    assert!(first.derive(0, &other_input, None).is_err());
    let conv = first.derive(0, &input, None).unwrap();
    let empty = first.empty_prefix().unwrap();
    assert!(empty.for_relu(4).is_err());
    assert!(empty.commit_relu(&input, 1, None).is_err());
    assert!(empty.commit_relu(&conv, 1, Some(&[0.5])).is_err());
    let (one, relu) = empty.commit_relu(&conv, 1, Some(&[f64::NAN, 0.5])).unwrap();
    assert!(one.for_relu(1).is_err());
    assert!(one.gate(0).unwrap().alpha.iter().all(|x| x.is_finite()));
    assert!(first.derive(2, &input, None).is_err());
    let affine = first.derive(2, &relu, None).unwrap();
    assert!(first.derive(3, &affine, Some(&conv)).is_err());
    assert!(first.derive(3, &affine, None).is_err());
    assert!(first.derive(2, &relu, Some(&relu)).is_err());
    assert!(first.derive(6, &relu, None).is_err());
    assert!(first.derive(7, &relu, None).is_err());
    assert!(!first.same_root(&other));
}

#[test]
fn immutable_frames_share_gates_and_do_not_retain_dead_reference_buffers() {
    let (budget, _, memory) = budget_with(None);
    let root = root(fixture(), &[-1.0; 2], &[1.0; 2], &budget);
    let model_bytes = root.model().resident_bytes();
    let root_bytes = root.resident_bytes();
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let empty = root.empty_prefix().unwrap();
    let empty_bytes = empty.node()._resident.bytes();
    let (one, relu) = empty.commit_relu(&conv, 1, None).unwrap();
    let gate_bytes = one.node()._resident.bytes();
    drop(input);
    drop(conv);
    drop(relu);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        model_bytes + root_bytes + empty_bytes + gate_bytes
    );
    let twin = one.clone();
    assert!(one.same_prefix(&twin));
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        model_bytes + root_bytes + empty_bytes + gate_bytes
    );
    {
        let frame = one.frame(None).unwrap();
        assert_eq!(frame.layers().len(), 1);
        assert!(std::ptr::eq(frame.gate(0).unwrap(), one.gate(0).unwrap()));
        assert!(frame.prefix().same_prefix(&one));
        assert_eq!(frame.mode(), RoundMode::Outward);
        assert!(frame.root().same_root(&root));
        assert!(budget.receipt().unwrap().live_scratch_bytes > 0);
    }
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(empty);
    drop(root);
    drop(one);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        model_bytes + root_bytes + empty_bytes + gate_bytes
    );
    drop(twin);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}

#[test]
fn gate_and_output_overlap_is_admitted_atomically_without_losing_prior_prefix() {
    let (budget, _, _) = budget_with(None);
    let root = root(fixture(), &[-1.0; 2], &[1.0; 2], &budget);
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let empty = root.empty_prefix().unwrap();
    let existing = budget.receipt().unwrap().live_persistent_bytes;
    let gate_bytes = payload(2, 6, 8).unwrap()
        + payload(2, 1, size_of::<usize>()).unwrap()
        + owner_bytes::<PrefixNode>().unwrap();
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: CAP - existing - gate_bytes,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let before = budget.receipt().unwrap();
    assert!(empty.commit_relu(&conv, 1, None).is_err());
    let after = budget.receipt().unwrap();
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
    assert!(after.refused > before.refused);
    assert_eq!(empty.layer_count(), 0);
    assert!(empty.gate(0).is_none());
    assert!(conv.lo().iter().zip(conv.hi()).all(|(l, u)| l <= u));
    drop(blocker);
    let (one, _) = empty.commit_relu(&conv, 1, None).unwrap();
    assert_eq!(one.layer_count(), 1);
}

#[test]
fn inherited_cancellation_refuses_all_steps_and_preserves_owned_proof() {
    let (budget, cancellation, _) = budget_with(None);
    let root = root(fixture(), &[-1.0; 2], &[1.0; 2], &budget);
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let (one, relu) = root
        .empty_prefix()
        .unwrap()
        .commit_relu(&conv, 1, None)
        .unwrap();
    let before = budget.receipt().unwrap();
    cancellation.cancel();
    assert!(root.input_box().is_err());
    assert!(root.derive(2, &relu, None).is_err());
    assert!(one.frame(None).is_err());
    assert!(one.for_relu(4).is_err());
    assert!(root.empty_prefix().is_err());
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    assert_eq!(
        budget.receipt().unwrap().live_scratch_bytes,
        before.live_scratch_bytes
    );
    assert!(one.gate(0).is_some());
}

#[test]
fn requested_and_builder_deadlines_cannot_be_renewed_by_new_views() {
    let (budget, _, _) = budget_with(None);
    let built_deadline = Instant::now() + Duration::from_secs(60);
    let model = OriginalModel::compile(fixture(), &budget, Some(built_deadline)).unwrap();
    let root_deadline = Instant::now() + Duration::from_secs(30);
    let root = OriginalRoot::new(
        model,
        &[-1.0; 2],
        &[1.0; 2],
        RootPrecisionRuntimePolicy::default(),
        Some(root_deadline),
    )
    .unwrap();
    let prefix = root.empty_prefix().unwrap();
    let later = Instant::now() + Duration::from_secs(120);
    assert_eq!(
        prefix.frame(Some(later)).unwrap().deadline(),
        Some(root_deadline)
    );
    let shorter = Instant::now() + Duration::from_secs(15);
    assert_eq!(
        prefix.frame(Some(shorter)).unwrap().deadline(),
        Some(shorter)
    );
    let before = budget.receipt().unwrap();
    assert!(prefix
        .frame(Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap()
        ))
        .is_err());
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    assert_eq!(
        budget.receipt().unwrap().live_scratch_bytes,
        before.live_scratch_bytes
    );
    assert_eq!(root.deadline(None).unwrap(), Some(root_deadline));
}

#[test]
fn numerical_box_and_completed_prefix_keep_an_earlier_producer_deadline() {
    let (budget, _, _) = budget_with(None);
    let root = root(fixture(), &[-1.0; 2], &[1.0; 2], &budget);
    let input = root.input_box().unwrap();
    let at = Instant::now() + Duration::from_secs(30);
    let conv = root
        .derive_with_deadline(0, &input, None, Some(at))
        .unwrap();
    assert_eq!(conv.deadline().unwrap(), Some(at));
    let empty = root.empty_prefix().unwrap();
    let (one, relu) = empty.commit_relu(&conv, 1, None).unwrap();
    assert_eq!(one.deadline(None).unwrap(), Some(at));
    assert_eq!(relu.deadline().unwrap(), Some(at));
    let affine = root.derive(2, &relu, None).unwrap();
    let add = root.derive(3, &affine, Some(&relu)).unwrap();
    assert_eq!(add.deadline().unwrap(), Some(at));
    let (two, output) = one.commit_relu(&add, 4, None).unwrap();
    let later = Instant::now() + Duration::from_secs(120);
    assert_eq!(two.frame(Some(later)).unwrap().deadline(), Some(at));
    assert_eq!(output.deadline().unwrap(), Some(at));
    let source_floor = root.model().resident_bytes()
        + root.resident_bytes()
        + empty.node()._resident.bytes()
        + one.node()._resident.bytes()
        + two.node()._resident.bytes();
    assert_eq!(two.resident_bytes(Some(later)).unwrap(), source_floor);
}

#[test]
fn input_rows_refinement_recovers_residual_cancellation_and_repairs_only_selected_gate() {
    let (budget, _, _) = budget_with(None);
    let root = root(
        cancelling_fixture([0.1, -0.2]),
        &[-1.0, -0.5],
        &[1.0, 1.25],
        &budget,
    );
    let (prefix, pre) = cancelling_prefix(&root);
    assert!(pre.lo().iter().all(|&l| l < 0.0));
    assert!(pre.hi().iter().all(|&u| u > 0.0));
    let rows = products(&prefix, &[0]);
    let before = budget.receipt().unwrap().live_persistent_bytes;
    let source_bias = rows.lower(0).unwrap().bias();
    let refined = prefix.refine_relu(&pre, 4, &rows, None).unwrap();
    assert!(refined.lo()[0] > 0.0);
    assert!(refined.hi()[0] - refined.lo()[0] < (pre.hi()[0] - pre.lo()[0]) * 0.01);
    assert_eq!(refined.lo()[1].to_bits(), pre.lo()[1].to_bits());
    assert_eq!(refined.hi()[1].to_bits(), pre.hi()[1].to_bits());
    contains(&refined, 0, &rat(0.1));
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before + refined.resident_bytes()
    );
    assert_eq!(
        rows.lower(0).unwrap().bias().to_bits(),
        source_bias.to_bits()
    );
    assert!(rows.validate_prepared_prefix(&prefix, 4, &[0]).is_ok());
    let (completed, output) = prefix.commit_relu(&refined, 4, None).unwrap();
    let gate = completed.gate(1).unwrap();
    assert_eq!(gate.alpha[0], 1.0);
    assert_eq!(gate.s[0], 1.0);
    assert_eq!(gate.c[0], 0.0);
    assert_eq!(gate.unst, vec![1]);
    contains(&output, 0, &rat(0.1));
}

#[test]
fn input_rows_refinement_uses_ordered_original_ids_and_independent_rational_semantics() {
    let (budget, _, _) = budget_with(None);
    let shifts = [0.1, -0.2];
    let root = root(
        cancelling_fixture(shifts),
        &[-1.0, -0.5],
        &[1.0, 1.25],
        &budget,
    );
    let (prefix, pre) = cancelling_prefix(&root);
    let rows = products(&prefix, &[1, 0]);
    assert_eq!(rows.targets(), &[1, 0]);
    assert!(rows.validate_prepared_prefix(&prefix, 4, &[0, 1]).is_err());
    let refined = prefix.refine_relu(&pre, 4, &rows, None).unwrap();
    assert!(refined.lo()[0] > 0.0);
    assert!(refined.hi()[1] < 0.0);
    let (completed, output) = prefix.commit_relu(&refined, 4, None).unwrap();
    let gate = completed.gate(1).unwrap();
    assert_eq!((gate.unst).len(), 0);
    assert_eq!(gate.alpha, vec![1.0, 0.0]);
    assert_eq!(gate.s, vec![1.0, 0.0]);
    for mask in 0..4 {
        for i in 0..2 {
            let x = rat(if mask & (1 << i) == 0 {
                root.lo()[i]
            } else {
                root.hi()[i]
            });
            let y = x.max(BigRational::zero());
            let actual = rat(-1.0) * &y + rat(shifts[i]) + y;
            contains(&refined, i, &actual);
            line_contains(gate, i, &actual);
            contains(&output, i, &actual.max(BigRational::zero()));
        }
    }
}

#[test]
fn input_rows_refinement_charges_error_at_zero_and_subnormal_clipping_boundaries() {
    let (budget, _, _) = budget_with(None);
    let tiny = f64::from_bits(1);
    let root = root(
        cancelling_fixture([0.0, tiny]),
        &[-1.0, -0.5],
        &[1.0, 1.25],
        &budget,
    );
    let (prefix, pre) = cancelling_prefix(&root);
    let rows = products(&prefix, &[0, 1]);
    let refined = prefix.refine_relu(&pre, 4, &rows, None).unwrap();
    contains(&refined, 0, &BigRational::zero());
    contains(&refined, 1, &rat(tiny));
    for (column, exact) in [BigRational::zero(), rat(tiny)].iter().enumerate() {
        let lower = rows.lower(column).unwrap();
        let upper = rows.upper(column).unwrap();
        assert!(rat(input_row_bound(&root, &lower, true, rows.deadline()).unwrap()) <= *exact);
        assert!(*exact <= rat(input_row_bound(&root, &upper, false, rows.deadline()).unwrap()));
    }
    let (completed, output) = prefix.commit_relu(&refined, 4, None).unwrap();
    line_contains(completed.gate(1).unwrap(), 0, &BigRational::zero());
    line_contains(completed.gate(1).unwrap(), 1, &rat(tiny));
    contains(&output, 0, &BigRational::zero());
    contains(&output, 1, &rat(tiny));
}

#[test]
fn input_rows_refinement_refuses_foreign_root_prefix_operation_and_borrowed_diagnostics() {
    let (budget, _, _) = budget_with(None);
    let spec = cancelling_fixture([0.1, -0.2]);
    let root = root(spec.clone(), &[-1.0, -0.5], &[1.0, 1.25], &budget);
    let (prefix, pre) = cancelling_prefix(&root);
    let rows = products(&prefix, &[0, 1]);
    assert!(prefix.refine_relu(&pre, 1, &rows, None).is_err());
    let wrong_tensor = root.input_box().unwrap();
    assert!(prefix.refine_relu(&wrong_tensor, 4, &rows, None).is_err());
    let (sibling, _) = cancelling_prefix(&root);
    assert!(!prefix.same_prefix(&sibling));
    assert!(sibling.refine_relu(&pre, 4, &rows, None).is_err());
    let foreign = OriginalRoot::new(
        root.model().clone(),
        root.lo(),
        root.hi(),
        Default::default(),
        None,
    )
    .unwrap();
    let (foreign_prefix, foreign_pre) = cancelling_prefix(&foreign);
    assert!(prefix.refine_relu(&foreign_pre, 4, &rows, None).is_err());
    assert!(foreign_prefix
        .refine_relu(&foreign_pre, 4, &rows, None)
        .is_err());

    // A successful legacy producer is still borrowed-root evidence. It cannot
    // acquire ownership authority merely from identical numbers or geometry.
    let net = crate::margin_row::net::TwinNet::compile(&spec).unwrap();
    let _native = crate::margin_row::head_group_source::HeadGroupSourceGuard::enter(true);
    let legacy = crate::margin_row::root::RootGates::build(
        &net,
        root.lo(),
        root.hi(),
        RoundMode::Outward,
        None,
    )
    .unwrap();
    let seed = crate::margin_row::engine::Seed {
        s: ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        e: None,
    };
    let diagnostics = crate::margin_row::engine::BackwardEngine::new(&net, &legacy)
        .with_gpu_conv_route(crate::margin_row::gpu_conv::GpuConvRoute::Off)
        .run_prefix_input_rows(
            &seed,
            4,
            &[0, 1],
            0,
            CAP,
            Instant::now() + Duration::from_secs(30),
        )
        .unwrap();
    let before = budget.receipt().unwrap().live_persistent_bytes;
    assert!(prefix.refine_relu(&pre, 4, &diagnostics, None).is_err());
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, before);
    assert_eq!(prefix.layer_count(), 1);
}

#[test]
fn input_rows_refinement_cap_deadline_and_cancellation_refusal_preserve_all_owners() {
    let (budget, cancellation, _) = budget_with(None);
    let root = root(
        cancelling_fixture([0.1, -0.2]),
        &[-1.0, -0.5],
        &[1.0, 1.25],
        &budget,
    );
    let (prefix, pre) = cancelling_prefix(&root);
    let rows = products(&prefix, &[1, 0]);
    let before = budget.receipt().unwrap();
    let bytes = payload(2, 2, 8).unwrap() + owner_bytes::<CertifiedTensorBox>().unwrap();
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: CAP - before.live_persistent_bytes - bytes + 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let held = budget.receipt().unwrap();
    assert!(prefix.refine_relu(&pre, 4, &rows, None).is_err());
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        held.live_persistent_bytes
    );
    assert_eq!(
        budget.receipt().unwrap().live_scratch_bytes,
        held.live_scratch_bytes
    );
    drop(blocker);
    let past = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap();
    assert!(matches!(
        prefix.refine_relu(&pre, 4, &rows, Some(past)),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    let refined = prefix.refine_relu(&pre, 4, &rows, None).unwrap();
    assert_eq!(refined.deadline().unwrap(), Some(rows.deadline()));
    let (completed, output) = prefix.commit_relu(&refined, 4, None).unwrap();
    let later = rows.deadline() + Duration::from_secs(60);
    assert_eq!(
        completed.frame(Some(later)).unwrap().deadline(),
        Some(rows.deadline())
    );
    assert_eq!(output.deadline().unwrap(), Some(rows.deadline()));
    let live = budget.receipt().unwrap().live_persistent_bytes;
    cancellation.cancel();
    assert!(prefix.refine_relu(&pre, 4, &rows, Some(later)).is_err());
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, live);
    assert_eq!(rows.targets(), &[1, 0]);
    assert_eq!(prefix.layer_count(), 1);
}

#[test]
fn input_rows_refinement_preserves_fixed_local_ceiling_under_larger_shared_authority() {
    let fixed_cap = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
    let global_cap = fixed_cap.checked_add(CAP).unwrap();
    let memory = MemoryAuthority::new(global_cap.checked_add(CAP).unwrap());
    let cancellation = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: None,
            scratch_bytes: CAP,
            persistent_bytes: global_cap,
            max_work: 1_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let root = root(
        cancelling_fixture([0.1, -0.2]),
        &[-1.0, -0.5],
        &[1.0, 1.25],
        &budget,
    );
    let (prefix, mut pre) = cancelling_prefix(&root);
    let rows = products(&prefix, &[1, 0]);
    let original_pre_claim = pre.resident_bytes();
    // Overpay this genuine producer box's existing storage on the SAME
    // authority. Endpoints, original scope, numerical proof and producer
    // deadline stay intact; no arrays or gate completion are fabricated.
    // Requested capacity receipts are not physical allocations or RSS.
    let conservative = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: fixed_cap,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap();
    let original = std::mem::replace(&mut pre._resident, conservative);
    drop(original);
    let before = budget.receipt().unwrap();
    assert!(before.live_persistent_bytes > fixed_cap);
    let expected = rows.resident_floor(rows.deadline()).unwrap()
        + fixed_cap
        + payload(2, 2, 8).unwrap()
        + owner_bytes::<CertifiedTensorBox>().unwrap();
    let refusal = prefix.refine_relu(&pre, 4, &rows, None);
    assert!(matches!(refusal, Err(NyError::CpuMemoryExceeded {
        required_bytes, budget_bytes, site: "prepared prefix refinement visible overlap"
    }) if required_bytes == expected && budget_bytes == fixed_cap));
    let after = budget.receipt().unwrap();
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(prefix.layer_count(), 1);
    assert_eq!(rows.targets(), &[1, 0]);
    contains(&pre, 0, &rat(0.1));
    contains(&pre, 1, &rat(-0.2));
    // Restore the original price with another paid reservation, then the same
    // authenticated rows and unchanged numerical box refine successfully.
    let ordinary = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: original_pre_claim,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap();
    let oversized = std::mem::replace(&mut pre._resident, ordinary);
    drop(oversized);
    assert!(prefix.refine_relu(&pre, 4, &rows, None).is_ok());
}

#[test]
fn original_root_fixed_ceiling_prices_owned_planes_and_distinct_borrowed_endpoints() {
    let (budget, _, _) = oversized_budget();
    let cap = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
    let bytes = payload(2, 5, 8).unwrap() + owner_bytes::<RootInner>().unwrap();
    let model = OriginalModel::compile(fixture(), &budget, None).unwrap();
    let model = model
        .reprice_for_test(paid_claim(&budget, cap - bytes - 32 + 1))
        .unwrap();
    let before = budget.receipt().unwrap();
    let refused = OriginalRoot::new(
        model.clone(),
        &[-1.0; 2],
        &[1.0; 2],
        Default::default(),
        None,
    );
    assert!(
        matches!(refused, Err(NyError::CpuMemoryExceeded { required_bytes,
        budget_bytes, site: "prepared original root visible overlap" })
        if required_bytes == cap + 1 && budget_bytes == cap)
    );
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    assert_eq!(
        budget.receipt().unwrap().live_scratch_bytes,
        before.live_scratch_bytes
    );
    let point = [0.0; 2];
    // The same borrowed input span counts once. All five NEW owned planes and
    // metadata still count; the genuinely declared point domain remains valid.
    let admitted = OriginalRoot::new(model, &point, &point, Default::default(), None).unwrap();
    assert_eq!(admitted.lo(), &point);
    assert_eq!(admitted.hi(), &point);
}

#[test]
fn original_input_box_and_empty_prefix_enforce_fixed_ceiling_before_allocation() {
    let (budget, _, _) = oversized_budget();
    let cap = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
    let mut root = root(fixture(), &[-1.0; 2], &[1.0; 2], &budget);
    let original = root.resident_bytes();
    let model = root.model().resident_bytes();
    let output = payload(2, 2, 8).unwrap() + owner_bytes::<CertifiedTensorBox>().unwrap();
    reprice_root(&mut root, cap - model - output + 1);
    let before = budget.receipt().unwrap();
    assert!(matches!(root.input_box(), Err(NyError::CpuMemoryExceeded {
        required_bytes, budget_bytes, site: "prepared original input box visible overlap"
    }) if required_bytes == cap + 1 && budget_bytes == cap));
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    reprice_root(&mut root, original);
    let empty = owner_bytes::<PrefixNode>().unwrap();
    reprice_root(&mut root, cap - model - empty + 1);
    let before = budget.receipt().unwrap();
    assert!(
        matches!(root.empty_prefix(), Err(NyError::CpuMemoryExceeded {
        required_bytes, budget_bytes, site: "prepared empty prefix visible overlap"
    }) if required_bytes == cap + 1 && budget_bytes == cap)
    );
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    reprice_root(&mut root, original);
    assert!(root.input_box().is_ok());
    assert!(root.empty_prefix().is_ok());
}

#[test]
fn original_reference_fixed_ceiling_counts_unary_and_distinct_residual_operands_once() {
    let (budget, _, _) = oversized_budget();
    let cap = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
    let root = root(fixture(), &[-1.0; 2], &[1.0; 2], &budget);
    let base = root.visible_resident_bytes().unwrap();
    let output = payload(2, 2, 8).unwrap() + owner_bytes::<CertifiedTensorBox>().unwrap();
    let mut input = root.input_box().unwrap();
    let input_claim = input.resident_bytes();
    reprice_box(&mut input, cap - base - output + 1);
    let before = budget.receipt().unwrap();
    assert!(
        matches!(root.derive(0, &input, None), Err(NyError::CpuMemoryExceeded {
        required_bytes, budget_bytes, site: "prepared original reference box visible overlap"
    }) if required_bytes == cap + 1 && budget_bytes == cap)
    );
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    reprice_box(&mut input, input_claim);
    let conv = root.derive(0, &input, None).unwrap();
    let mut relu = root.derive(1, &conv, None).unwrap();
    let relu_claim = relu.resident_bytes();
    let affine = root.derive(2, &relu, None).unwrap();
    reprice_box(&mut relu, cap - base - affine.resident_bytes() - output + 1);
    let before = budget.receipt().unwrap();
    assert!(
        matches!(root.derive(3, &affine, Some(&relu)), Err(NyError::CpuMemoryExceeded {
        required_bytes, budget_bytes, site: "prepared original reference box visible overlap"
    }) if required_bytes == cap + 1 && budget_bytes == cap)
    );
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    reprice_box(&mut relu, relu_claim);
    assert!(root.derive(3, &affine, Some(&relu)).is_ok());

    let mut spec = fixture();
    spec.ops[3] = TwinOpSpec::Add { lhs: 2, rhs: 2 };
    let shared = tests::root(spec, &[-1.0; 2], &[1.0; 2], &budget);
    let i = shared.input_box().unwrap();
    let c = shared.derive(0, &i, None).unwrap();
    let mut y = shared.derive(1, &c, None).unwrap();
    let shared_base = shared.visible_resident_bytes().unwrap();
    reprice_box(&mut y, cap - shared_base - output);
    // The identical actual operand has one storage owner even though the Add
    // uses it twice. Exact local-ceiling equality must remain admissible.
    let doubled = shared.derive(3, &y, Some(&y)).unwrap();
    contains(&doubled, 0, &(rat(2.0) * rat(0.875)));
}

#[test]
fn completed_gate_fixed_ceiling_includes_prior_prefix_and_new_gate_output_overlap() {
    let (budget, _, _) = oversized_budget();
    let cap = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
    let root = root(
        cancelling_fixture([0.1, -0.2]),
        &[-1.0, -0.5],
        &[1.0, 1.25],
        &budget,
    );
    let (mut prefix, pre) = cancelling_prefix(&root);
    let original = prefix.node()._resident.bytes();
    let base = prefix.resident_bytes(None).unwrap() - original;
    let gate = payload(2, 6, 8).unwrap()
        + payload(2, 1, size_of::<usize>()).unwrap()
        + owner_bytes::<PrefixNode>().unwrap();
    let output = payload(2, 2, 8).unwrap() + owner_bytes::<CertifiedTensorBox>().unwrap();
    reprice_prefix(
        &mut prefix,
        cap - base - pre.resident_bytes() - gate - output + 1,
    );
    let before = budget.receipt().unwrap();
    assert!(
        matches!(prefix.commit_relu(&pre, 4, None), Err(NyError::CpuMemoryExceeded {
        required_bytes, budget_bytes, site: "prepared completed gate visible overlap"
    }) if required_bytes == cap + 1 && budget_bytes == cap)
    );
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    assert_eq!(
        budget.receipt().unwrap().live_scratch_bytes,
        before.live_scratch_bytes
    );
    assert_eq!(prefix.layer_count(), 1);
    reprice_prefix(&mut prefix, original);
    assert!(prefix.commit_relu(&pre, 4, None).is_ok());
}

#[test]
fn prepared_frame_fixed_ceiling_prices_real_borrowed_reference_metadata() {
    let (budget, _, _) = oversized_budget();
    let cap = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
    let root = root(fixture(), &[-1.0; 2], &[1.0; 2], &budget);
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let (mut prefix, _) = root
        .empty_prefix()
        .unwrap()
        .commit_relu(&conv, 1, None)
        .unwrap();
    let original = prefix.node()._resident.bytes();
    let base = prefix.resident_bytes(None).unwrap() - original;
    let scratch = size_of::<&LayerGates>() + size_of::<PreparedPrefixFrame<'_>>();
    reprice_prefix(&mut prefix, cap - base - scratch + 1);
    let before = budget.receipt().unwrap();
    assert!(
        matches!(prefix.frame(None), Err(NyError::CpuMemoryExceeded {
        required_bytes, budget_bytes, site: "prepared prefix frame visible overlap"
    }) if required_bytes == cap + 1 && budget_bytes == cap)
    );
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    assert_eq!(
        budget.receipt().unwrap().live_scratch_bytes,
        before.live_scratch_bytes
    );
    reprice_prefix(&mut prefix, original);
    assert!(prefix.frame(None).is_ok());
}

#[test]
fn input_rows_refinement_preserves_real_f32_storage_and_full_cast_gap() {
    let (budget, _, _) = budget_with(None);
    let root = root(
        cancelling_fixture([0.1, -0.2]),
        &[-1.0, -0.5],
        &[1.0, 1.25],
        &budget,
    );
    let (prefix, pre) = cancelling_prefix(&root);
    let rows = products(&prefix, &[1, 0]);
    let deadline = rows.deadline();
    let floor = rows.resident_floor(deadline).unwrap();
    let stored = rows.into_f32(floor, CAP, deadline).unwrap();
    assert_eq!(stored.rows.backing(), InputRowsBacking::F32);
    assert!(stored.receipt.changed_cells > 0);
    let before = budget.receipt().unwrap().live_persistent_bytes;
    let refined = prefix.refine_relu(&pre, 4, &stored.rows, None).unwrap();
    contains(&refined, 0, &rat(0.1));
    contains(&refined, 1, &rat(-0.2));
    assert!(refined.lo()[0] > 0.0 && refined.hi()[1] < 0.0);
    assert_eq!(stored.rows.backing(), InputRowsBacking::F32);
    assert_eq!(stored.rows.targets(), &[1, 0]);
    assert_eq!(stored.rows.deadline(), deadline);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before + refined.resident_bytes()
    );
    let lower = stored.rows.lower(1).unwrap();
    let upper = stored.rows.upper(1).unwrap();
    // A bias-only nominal lower evaluation would exclude the true positive
    // constant after nearest f32 storage. Its full cast gap must be charged.
    assert!(lower.entries().all(|(_, a, _)| a == 0.0));
    assert!(rat(lower.bias()) > rat(0.1));
    assert!(rat(lower.pointwise_error(deadline).unwrap()) >= rat(lower.bias()) - rat(0.1));
    assert!(rat(input_row_bound(&root, &lower, true, deadline).unwrap()) <= rat(0.1));
    assert!(rat(0.1) <= rat(input_row_bound(&root, &upper, false, deadline).unwrap()));
    let negative_upper = stored.rows.upper(0).unwrap();
    assert!(negative_upper.entries().all(|(_, a, _)| a == 0.0));
    assert!(rat(negative_upper.bias()) < rat(-0.2));
    assert!(
        rat(negative_upper.pointwise_error(deadline).unwrap())
            >= rat(-0.2) - rat(negative_upper.bias())
    );
}

#[test]
fn f32_reference_and_nonfinite_declared_boxes_refuse_before_publishing_owners() {
    let (budget, _, _) = budget_with(None);
    let model = OriginalModel::compile(fixture(), &budget, None).unwrap();
    let before = budget.receipt().unwrap().live_persistent_bytes;
    for policy in [
        RootPrecisionRuntimePolicy {
            root_f32: true,
            ..Default::default()
        },
        RootPrecisionRuntimePolicy {
            tableau_f32: true,
            ..Default::default()
        },
    ] {
        assert!(OriginalRoot::new(model.clone(), &[-1.0; 2], &[1.0; 2], policy, None).is_err());
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, before);
    }
    for (lo, hi) in [
        (vec![f64::NAN, 0.0], vec![1.0; 2]),
        (vec![0.0; 2], vec![f64::INFINITY, 1.0]),
        (vec![1.0; 2], vec![0.0; 2]),
        (vec![f64::MAX; 2], vec![f64::MAX; 2]),
    ] {
        assert!(OriginalRoot::new(model.clone(), &lo, &hi, Default::default(), None).is_err());
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, before);
    }
}

#[test]
fn original_prefix_has_no_small_head_or_original_input_dimension_arming_cap() {
    let (budget, _, _) = budget_with(None);
    let n = 9408;
    let mut ops = vec![
        TwinOpSpec::Relu { input: 0 },
        TwinOpSpec::Flatten { input: 1 },
    ];
    head(&mut ops, 2, n, 200);
    let root = root(
        TwinSpec { n_in: n, ops },
        &vec![-1.0; n],
        &vec![1.0; n],
        &budget,
    );
    assert_eq!(root.model().net().n_out, 200);
    let input = root.input_box().unwrap();
    let (one, output) = root
        .empty_prefix()
        .unwrap()
        .commit_relu(&input, 0, None)
        .unwrap();
    assert_eq!(one.gate(0).unwrap().n, n);
    assert_eq!(output.lo().len(), n);
    assert!(one.frame(None).is_ok());
}

#[test]
fn deep_prefix_last_owner_release_is_iterative_and_does_not_copy_history() {
    let (budget, _, _) = budget_with(None);
    let depth = 2048;
    let mut ops = Vec::with_capacity(depth + 4);
    for op in 0..depth {
        ops.push(TwinOpSpec::Relu { input: op });
    }
    ops.push(TwinOpSpec::Flatten { input: depth });
    head(&mut ops, depth + 1, 1, 2);
    let root = root(TwinSpec { n_in: 1, ops }, &[-1.0], &[1.0], &budget);
    let mut prefix = root.empty_prefix().unwrap();
    let mut value = root.input_box().unwrap();
    for op in 0..depth {
        let (next, output) = prefix.commit_relu(&value, op, None).unwrap();
        prefix = next;
        value = output;
    }
    assert_eq!(prefix.layer_count(), depth);
    assert!(prefix.gate(depth - 1).is_some());
    let live = budget.receipt().unwrap().live_persistent_bytes;
    let one_gate = payload(1, 6, 8).unwrap()
        + payload(1, 1, size_of::<usize>()).unwrap()
        + owner_bytes::<PrefixNode>().unwrap();
    assert_eq!(
        live,
        root.model().resident_bytes()
            + root.resident_bytes()
            + owner_bytes::<PrefixNode>().unwrap()
            + depth * one_gate
            + value.resident_bytes()
    );
    drop(value);
    drop(root);
    drop(prefix);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct UnqualifiedEnvironment {
    #[cfg(target_arch = "aarch64")]
    saved: u64,
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl UnqualifiedEnvironment {
    // The register is thread-local. This test-only guard saves it before
    // alteration and restores it during normal return and unwinding.
    #[allow(unsafe_code)]
    fn enter(flush: bool) -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let saved: u64;
            std::arch::asm!("mrs {saved}, fpcr", saved = out(reg) saved, options(nomem, nostack));
            let altered = if flush {
                saved | (1_u64 << 24)
            } else {
                (saved & !(3_u64 << 22)) | (1_u64 << 22)
            };
            std::arch::asm!("msr fpcr, {altered}", "isb", altered = in(reg) altered, options(nomem, nostack));
            Self { saved }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let saved = std::arch::x86_64::_mm_getcsr();
            let altered = if flush {
                saved | (1 << 15) | (1 << 6)
            } else {
                (saved & !(3 << 13)) | (1 << 13)
            };
            std::arch::x86_64::_mm_setcsr(altered);
            Self { saved }
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl Drop for UnqualifiedEnvironment {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("msr fpcr, {saved}", "isb", saved = in(reg) self.saved, options(nomem, nostack));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved);
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn directed_rounding_and_flush_cannot_publish_original_numerical_proofs() {
    let (budget, _, _) = budget_with(None);
    let root = root(fixture(), &[-1.0; 2], &[1.0; 2], &budget);
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let prefix = root.empty_prefix().unwrap();
    let rows = products(&prefix, &[0, 1]);
    let before = budget.receipt().unwrap();
    for flush in [false, true] {
        let _environment = UnqualifiedEnvironment::enter(flush);
        assert!(!ny_core::has_f64_interval_proof_environment());
        assert!(matches!(
            OriginalRoot::new(
                root.model().clone(),
                &[-1.0; 2],
                &[1.0; 2],
                Default::default(),
                None
            ),
            Err(NyError::SoundnessRefusal(_))
        ));
        assert!(matches!(
            root.derive(0, &input, None),
            Err(NyError::SoundnessRefusal(_))
        ));
        assert!(matches!(
            prefix.commit_relu(&conv, 1, None),
            Err(NyError::SoundnessRefusal(_))
        ));
        assert!(matches!(
            prefix.refine_relu(&conv, 1, &rows, None),
            Err(NyError::SoundnessRefusal(_))
        ));
    }
    assert!(ny_core::has_f64_interval_proof_environment());
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    assert_eq!(
        budget.receipt().unwrap().live_scratch_bytes,
        before.live_scratch_bytes
    );
}

#[test]
fn construction_request_clock_is_attached_before_input_prefix_and_original_derivation() {
    let (budget, _, _) = budget_with(None);
    let root = root(fixture(), &[-1.0; 2], &[1.0; 2], &budget);
    let input_at = Instant::now() + Duration::from_secs(120);
    let prefix_at = input_at.checked_sub(Duration::from_secs(30)).unwrap();
    let input = root.input_box_with_deadline(Some(input_at)).unwrap();
    assert_eq!(input.deadline().unwrap(), Some(input_at));
    let conv = root.derive_with_deadline(0, &input, None, None).unwrap();
    assert_eq!(conv.deadline().unwrap(), Some(input_at));
    let prefix = root.empty_prefix_with_deadline(Some(prefix_at)).unwrap();
    assert_eq!(prefix.deadline(None).unwrap(), Some(prefix_at));
    let (next, output) = prefix.commit_relu(&conv, 1, None).unwrap();
    assert_eq!(next.deadline(Some(input_at)).unwrap(), Some(prefix_at));
    assert_eq!(output.deadline().unwrap(), Some(prefix_at));
}

#[test]
fn expired_construction_request_refuses_before_any_new_input_prefix_or_derive_claim() {
    let (budget, _, _) = budget_with(None);
    let root = root(fixture(), &[-1.0; 2], &[1.0; 2], &budget);
    let input = root.input_box().unwrap();
    let before = budget.receipt().unwrap();
    let expired = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    assert!(matches!(
        root.input_box_with_deadline(expired),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        root.empty_prefix_with_deadline(expired),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        root.derive_with_deadline(0, &input, None, expired),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(budget.receipt().unwrap(), before);
    assert_eq!(input.deadline().unwrap(), None);
}

#[test]
fn computed_f32_reference_transfers_paid_endpoints_into_genuine_prefix_gates() {
    let (budget, _, memory) = budget_with(None);
    let root = root(fixture(), &[-1.0, -0.75], &[1.25, 1.5], &budget);
    let at = Instant::now() + Duration::from_secs(30);
    let input = root.input_box_with_deadline(Some(at)).unwrap();
    let computed = super::super::precision::derive_reference_f32(&root, 0, &input, None).unwrap();
    let lo_pointer = computed.lo().as_ptr();
    let hi_pointer = computed.hi().as_ptr();
    let before_budget = budget.receipt().unwrap();
    let before_memory = memory.receipt().unwrap();
    let (conv, receipt) = root.certify_reference_f32(0, computed).unwrap();
    assert_eq!(conv.lo().as_ptr(), lo_pointer);
    assert_eq!(conv.hi().as_ptr(), hi_pointer);
    assert_eq!(budget.receipt().unwrap(), before_budget);
    assert_eq!(memory.receipt().unwrap(), before_memory);
    assert_eq!(conv.deadline().unwrap(), Some(at));
    assert_eq!(receipt.conv_ops, 1);
    assert!(receipt.multiply_add_pairs > 0);
    for (i, (&lo, &hi)) in input.lo().iter().zip(input.hi()).enumerate() {
        for x in [lo, 0.0, hi] {
            for rho in [-0.003, 0.003] {
                for bias_error in [-0.001, 0.001] {
                    let exact =
                        rat(-0.75) * rat(x) * (rat(1.0) + rat(rho)) + rat(0.125) + rat(bias_error);
                    contains(&conv, i, &exact);
                }
            }
        }
    }
    let prefix = root.empty_prefix_with_deadline(Some(at)).unwrap();
    let (completed, relu) = prefix.commit_relu(&conv, 1, None).unwrap();
    for i in 0..conv.lo().len() {
        for x in [conv.lo()[i], 0.0, conv.hi()[i]] {
            line_contains(completed.gate(0).unwrap(), i, &rat(x));
        }
    }
    let (affine, receipt) = root
        .derive_reference_f32_with_deadline(2, &relu, None)
        .unwrap();
    assert_eq!(receipt.affine_ops, 1);
    assert_eq!(affine.tensor(), 3);
    assert_eq!(affine.deadline().unwrap(), Some(at));
    let rows = products(&completed, &[0, 1]);
    rows.validate_prepared_prefix(&completed, 4, &[0, 1])
        .unwrap();
}
