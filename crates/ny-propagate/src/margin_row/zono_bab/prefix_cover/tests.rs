// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use num_rational::BigRational as Q;
use num_traits::Zero;

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::net::TwinNet;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::zono_state::{
    class_margin, FrameInput, FrozenDualFrame, FrozenDualOptions, FrozenDualTrunk, LayerInput,
    TrunkRelax,
};

fn budget() -> WorkBudget {
    WorkBudget::new(None, 0, 64 << 10, 50_000_000, 10_000, Duration::ZERO).unwrap()
}

fn managed(
    memory: &MemoryAuthority,
    cancellation: &CancellationToken,
    persistent_bytes: usize,
    max_work: u64,
    max_proposals: u64,
) -> WorkBudget {
    WorkBudget::new_in(
        memory,
        cancellation,
        WorkLimits {
            deadline: None,
            scratch_bytes: 0,
            persistent_bytes,
            max_work,
            max_proposals,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap()
}

fn phases(depth: usize, bits: usize) -> Vec<Side> {
    (0..depth)
        .map(|i| {
            if bits & (1 << i) == 0 {
                Side::Off
            } else {
                Side::On
            }
        })
        .collect()
}

fn request<'a>(
    problem: &'a FrozenDualProblem,
    parent: &'a [Side],
    chosen: &'a [Side],
) -> PrefixCoverRequest<'a> {
    PrefixCoverRequest {
        ordered_rows: &problem.order()[..parent.len() + chosen.len()],
        parent,
        chosen_path: chosen,
        deadline: None,
    }
}

#[test]
fn fd_prefix_cover_exhaustive_binary_assignments_have_exactly_one_actual_prefix() {
    let budget = budget();
    for depth in 1..=6 {
        let problem = FrozenDualProblem::synthetic(depth, depth, depth as u64);
        for parent_depth in 0..depth {
            for chosen_bits in 0..1 << depth {
                let path = phases(depth, chosen_bits);
                let parent = &path[..parent_depth];
                let cover = build(
                    &problem,
                    &request(&problem, parent, &path[parent_depth..]),
                    &budget,
                )
                .unwrap();
                assert_eq!(cover.parent().sides().copied().collect::<Vec<_>>(), parent);
                let member_depths: Vec<_> = cover.members().map(PrefixMember::depth).collect();
                assert_eq!(member_depths[0], depth);
                assert_eq!(
                    &member_depths[1..],
                    &(parent_depth + 1..=depth).collect::<Vec<_>>()
                );
                for assignment_bits in 0..1 << depth {
                    let assignment = phases(depth, assignment_bits);
                    let matching = cover
                        .members()
                        .filter(|member| {
                            member
                                .sides()
                                .copied()
                                .eq(assignment[..member.depth()].iter().copied())
                        })
                        .count();
                    assert_eq!(matching, usize::from(assignment[..parent_depth] == *parent));
                }
                cover.replay(&problem, parent, &budget, None).unwrap();
            }
        }
    }
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn fd_prefix_cover_closed_zero_boundaries_are_covered_and_can_overlap() {
    let budget = budget();
    let problem = FrozenDualProblem::synthetic_degenerate(4, 4, 0);
    let path = [Side::On, Side::Off, Side::On, Side::Off];
    let cover = build(
        &problem,
        &request(&problem, &path[..1], &path[1..]),
        &budget,
    )
    .unwrap();
    // These signs model actual preactivation signs; a zero satisfies BOTH
    // phases. A nonzero assignment has the usual first disagreement.
    for bits in 0..3usize.pow(4) {
        let mut digits = bits;
        let mut signs = [0i32; 4];
        for sign in &mut signs {
            *sign = (digits % 3) as i32 - 1;
            digits /= 3;
        }
        let compatible = |side: Side, sign: i32| match side {
            Side::Off => sign <= 0,
            Side::On => sign >= 0,
        };
        let parent_contains = compatible(path[0], signs[0]);
        let count = cover
            .members()
            .filter(|member| {
                member
                    .sides()
                    .enumerate()
                    .all(|(i, &s)| compatible(s, signs[i]))
            })
            .count();
        assert_eq!(count > 0, parent_contains);
    }
    assert_eq!(cover.members().count(), 4);
    assert!(cover.members().all(|member| {
        member
            .ordered_splits()
            .all(|(row, _)| row.a.iter().all(|&a| a == 0.0))
    }));
    // At the all-zero point ALL shorter siblings and the destination occur.
    assert_eq!(
        cover.members().filter(|m| m.sides().all(|_| true)).count(),
        4
    );
}

fn original_problem() -> FrozenDualProblem {
    let gemm = |input, weight, bias, shape| TwinOpSpec::Gemm {
        input,
        weight,
        bias,
        shape,
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
    };
    let net = TwinNet::compile(&TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0, 1.0],
                bias: vec![0.25, -0.25],
                bias_err: vec![0.0; 2],
                weight_rel_err: 0.0,
                kernel: (2, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (2, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            gemm(2, vec![1.0, -1.0, -1.0, 1.0], vec![-0.125, 0.125], (2, 2)),
            TwinOpSpec::Relu { input: 3 },
            gemm(4, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], (2, 2)),
        ],
    })
    .unwrap();
    let frame = FrozenDualFrame::new(
        &net,
        FrameInput {
            lo: vec![-1.0],
            hi: vec![1.0],
            trunk: vec![LayerInput {
                l: vec![-0.75, -1.25],
                u: vec![1.25, 0.75],
                alpha: vec![0.5; 2],
                s: vec![0.625, 0.375],
            }],
            head: LayerInput {
                l: vec![-0.125, -0.375],
                u: vec![0.375, 0.125],
                alpha: vec![0.5; 2],
                s: vec![0.75, 0.25],
            },
        },
        0,
    )
    .unwrap();
    let margin = class_margin(&net, &frame, 1, None).unwrap();
    let trunk = FrozenDualTrunk::build(
        &net,
        &frame,
        TrunkRelax::anchor(&frame, &margin),
        &FrozenDualOptions::default(),
    )
    .unwrap();
    let tail = trunk.relax().tail(&net, &frame, 1, None).unwrap();
    let state = trunk.state(&frame, 1, &tail).unwrap();
    FrozenDualProblem::new(&state, &[1, 0, 3, 2]).unwrap()
}

#[test]
fn fd_prefix_cover_original_reordered_split_rows_and_rational_sign_cuts_are_borrowed() {
    let problem = original_problem();
    let budget = budget();
    let path = [Side::On, Side::Off, Side::On, Side::Off];
    let cover = build(&problem, &request(&problem, &[], &path), &budget).unwrap();
    for member in cover.members() {
        for (i, (row, _side)) in member.ordered_splits().enumerate() {
            let original = problem.split_row(i);
            assert_eq!(row.state_row, [1, 0, 3, 2][i]);
            assert_eq!(row.enc, original.enc);
            assert_eq!(row.a.as_ptr(), original.a.as_ptr());
            assert_eq!(row.b0.map(f64::to_bits), original.b0.map(f64::to_bits));
        }
    }
    let q = |x| Q::from_float(x).unwrap();
    // Actual original first-layer preactivations x-1/4 and x+1/4,
    // with exact rational boundary points. Their interval-enlarged sign cuts
    // must retain those points on both closed phases at zero.
    for x in [-1.0, -0.25, 0.0, 0.25, 1.0] {
        for (i, center) in [-0.25, 0.25].into_iter().enumerate() {
            let row = problem.split_row(i);
            let affine = q(row.a[0]) * q(x);
            let z = q(x) + q(center);
            if z <= Q::zero() {
                assert!(affine <= q(row.b0[0]));
            }
            if z >= Q::zero() {
                assert!(-affine <= q(row.b0[1]));
            }
        }
    }
    cover.replay(&problem, &[], &budget, None).unwrap();
}

#[test]
fn fd_prefix_cover_rejects_labels_duplicates_empty_paths_and_depth_before_output() {
    let problem = FrozenDualProblem::synthetic(4, 4, 1);
    let budget = budget();
    let chosen = [Side::Off; 2];
    for rows in [&[1, 0][..], &[0, 0][..], &[0, usize::MAX][..]] {
        let request = PrefixCoverRequest {
            ordered_rows: rows,
            parent: &[],
            chosen_path: &chosen,
            deadline: None,
        };
        assert!(matches!(
            build(&problem, &request, &budget),
            Err(PrefixCoverRefusal::Order | PrefixCoverRefusal::DuplicateRow)
        ));
    }
    assert_eq!(
        build(&problem, &request(&problem, &[Side::Off], &[]), &budget).unwrap_err(),
        PrefixCoverRefusal::Shape
    );
    let wrong_length = PrefixCoverRequest {
        ordered_rows: &[],
        parent: &[],
        chosen_path: &chosen,
        deadline: None,
    };
    assert_eq!(
        build(&problem, &wrong_length, &budget).unwrap_err(),
        PrefixCoverRefusal::Shape
    );
    let beyond_chart = PrefixCoverRequest {
        ordered_rows: &[0, 1, 2, 3, 4],
        parent: &[],
        chosen_path: &[Side::On; 5],
        deadline: None,
    };
    assert_eq!(
        build(&problem, &beyond_chart, &budget).unwrap_err(),
        PrefixCoverRefusal::Depth
    );
    assert_eq!(problem.order(), &[0, 1, 2, 3]);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn fd_prefix_cover_fixed_depth_cap_admits_512_and_refuses_513() {
    let problem = FrozenDualProblem::synthetic(513, 513, 2);
    let budget = budget();
    let path = [Side::On; 513];
    let cover = build(&problem, &request(&problem, &[], &path[..512]), &budget).unwrap();
    assert_eq!(cover.members().count(), 513);
    assert_eq!(cover.members().last().unwrap().depth(), 512);
    assert_eq!(
        admission(&problem, &request(&problem, &[], &path)).unwrap_err(),
        PrefixCoverRefusal::Depth
    );
    cover.replay(&problem, &[], &budget, None).unwrap();
}

#[test]
fn fd_prefix_cover_replay_binds_actual_problem_parent_and_original_ledger() {
    let problem = FrozenDualProblem::synthetic(3, 3, 1);
    let budget = budget();
    let path = [Side::Off, Side::On, Side::Off];
    let cover = build(
        &problem,
        &request(&problem, &path[..1], &path[1..]),
        &budget,
    )
    .unwrap();
    let copied_problem = problem.clone();
    assert_eq!(
        cover.replay(&copied_problem, &path[..1], &budget, None),
        Err(PrefixCoverRefusal::ForeignProblem)
    );
    let foreign = tests::budget();
    let before = foreign.receipt().unwrap();
    assert_eq!(
        cover.replay(&problem, &path[..1], &foreign, None),
        Err(PrefixCoverRefusal::ForeignLedger)
    );
    assert_eq!(foreign.receipt().unwrap(), before);
    assert_eq!(
        cover.replay(&problem, &[], &budget, None),
        Err(PrefixCoverRefusal::Parent)
    );
    assert_eq!(
        cover.replay(&problem, &[Side::On], &budget, None),
        Err(PrefixCoverRefusal::Parent)
    );
    cover
        .replay(&problem, &path[..1], &budget.clone(), None)
        .unwrap();
}

#[test]
fn fd_prefix_cover_fresh_replay_rejects_incomplete_wrong_depth_flip_and_order() {
    let problem = FrozenDualProblem::synthetic(4, 4, 3);
    let budget = budget();
    for poison in 0..4 {
        let path = [Side::Off, Side::On, Side::On, Side::Off];
        let mut cover = build(
            &problem,
            &request(&problem, &path[..1], &path[1..]),
            &budget,
        )
        .unwrap();
        let storage = Arc::get_mut(&mut cover.storage).unwrap();
        match poison {
            0 => {
                storage.siblings.pop();
            }
            1 => storage.siblings[0].depth += 1,
            2 => storage.siblings[1].flipped = opposite(storage.siblings[1].flipped),
            _ => storage.rows.swap(0, 1),
        }
        assert!(matches!(
            cover.replay(&problem, &path[..1], &budget, None),
            Err(PrefixCoverRefusal::Incomplete | PrefixCoverRefusal::Order)
        ));
    }
}

#[test]
fn fd_prefix_cover_preprices_all_metadata_work_proposals_and_shared_capacity() {
    let problem = FrozenDualProblem::synthetic(4, 4, 4);
    let path = [Side::On; 4];
    let request = request(&problem, &[], &path);
    let predicted = admission(&problem, &request).unwrap();
    for (persistent, work, proposals, reason) in [
        (
            predicted.persistent_bytes - 1,
            predicted.work,
            1,
            WorkRefusal::PersistentLimit,
        ),
        (
            predicted.persistent_bytes,
            predicted.work - 1,
            1,
            WorkRefusal::WorkLimit,
        ),
        (
            predicted.persistent_bytes,
            predicted.work,
            0,
            WorkRefusal::ProposalLimit,
        ),
    ] {
        let budget = WorkBudget::new(None, 0, persistent, work, proposals, Duration::ZERO).unwrap();
        assert_eq!(
            build(&problem, &request, &budget).unwrap_err(),
            PrefixCoverRefusal::Budget(reason)
        );
        let receipt = budget.receipt().unwrap();
        assert_eq!(receipt.admitted, 0);
        assert_eq!(receipt.live_persistent_bytes, 0);
    }
    let memory = MemoryAuthority::new(predicted.persistent_bytes - 1);
    let token = CancellationToken::default();
    let budget = managed(
        &memory,
        &token,
        predicted.persistent_bytes,
        predicted.work,
        1,
    );
    assert_eq!(
        build(&problem, &request, &budget).unwrap_err(),
        PrefixCoverRefusal::Budget(WorkRefusal::MemoryLimit)
    );
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    // An exact finite admission suffices; no allowance refill on replay.
    let exact = WorkBudget::new(
        None,
        0,
        predicted.persistent_bytes,
        predicted.work,
        1,
        Duration::ZERO,
    )
    .unwrap();
    let cover = build(&problem, &request, &exact).unwrap();
    assert_eq!(
        cover.replay(&problem, &[], &exact, None),
        Err(PrefixCoverRefusal::Budget(WorkRefusal::WorkLimit))
    );
    drop(cover);
    assert_eq!(exact.receipt().unwrap().work_reserved, predicted.work);
}

#[test]
fn fd_prefix_cover_last_clone_holds_claim_after_cancellation_until_actual_drop() {
    let problem = FrozenDualProblem::synthetic(5, 5, 5);
    let path = [Side::On; 5];
    let request = request(&problem, &[], &path);
    let predicted = admission(&problem, &request).unwrap();
    let memory = MemoryAuthority::new(predicted.persistent_bytes);
    let token = CancellationToken::default();
    let budget = managed(
        &memory,
        &token,
        predicted.persistent_bytes,
        predicted.work * 2,
        1,
    );
    let cover = build(&problem, &request, &budget).unwrap();
    let clone = cover.clone();
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        predicted.persistent_bytes
    );
    token.cancel();
    assert_eq!(
        cover.replay(&problem, &[], &budget, None),
        Err(PrefixCoverRefusal::Budget(WorkRefusal::Cancelled))
    );
    drop(cover);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        predicted.persistent_bytes
    );
    assert_eq!(clone.parent().depth(), 0);
    drop(clone);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}

#[test]
fn fd_prefix_cover_cancel_during_copy_consumes_admission_releases_partial_claims_and_preserves_parent(
) {
    let problem = FrozenDualProblem::synthetic(96, 96, 6);
    let path = [Side::Off; 96];
    let parent = &path[..3];
    let request = request(&problem, parent, &path[3..]);
    let predicted = admission(&problem, &request).unwrap();
    let memory = MemoryAuthority::new(predicted.persistent_bytes);
    let token = CancellationToken::default();
    let budget = managed(
        &memory,
        &token,
        predicted.persistent_bytes,
        predicted.work,
        1,
    );
    let mut checkpoints = 0;
    let outcome = build_inner(&problem, &request, &budget, || {
        checkpoints += 1;
        if checkpoints == 2 {
            token.cancel();
        }
    });
    assert_eq!(
        outcome.unwrap_err(),
        PrefixCoverRefusal::Budget(WorkRefusal::Cancelled)
    );
    assert_eq!(checkpoints, 2);
    assert_eq!(parent, &[Side::Off; 3]);
    assert_eq!(problem.order().len(), 96);
    let receipt = budget.receipt().unwrap();
    assert_eq!(receipt.admitted, 1);
    assert_eq!(receipt.completed, 0);
    assert_eq!(receipt.work_reserved, predicted.work);
    assert_eq!(receipt.live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}

#[test]
fn fd_prefix_cover_earlier_absolute_and_class_deadlines_never_grant_new_time() {
    let problem = FrozenDualProblem::synthetic(2, 2, 7);
    let path = [Side::Off; 2];
    let budget = budget();
    let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let mut request = request(&problem, &[], &path);
    request.deadline = Some(past);
    assert_eq!(
        build(&problem, &request, &budget).unwrap_err(),
        PrefixCoverRefusal::Budget(WorkRefusal::Deadline)
    );
    assert_eq!(budget.receipt().unwrap().attempted, 0);
    request.deadline = None;
    let mut cover = build(&problem, &request, &budget).unwrap();
    assert_eq!(
        cover.replay(&problem, &[], &budget, Some(past)),
        Err(PrefixCoverRefusal::Budget(WorkRefusal::Deadline))
    );
    // A deterministic already-expired inherited deadline cannot be cleared by
    // passing None or a later replay deadline. No wall-clock sleep is needed.
    Arc::get_mut(&mut cover.storage).unwrap().deadline = Some(past);
    assert_eq!(
        cover.replay(&problem, &[], &budget, None),
        Err(PrefixCoverRefusal::Budget(WorkRefusal::Deadline))
    );
    assert_eq!(
        cover.replay(
            &problem,
            &[],
            &budget,
            Some(Instant::now() + Duration::from_secs(10))
        ),
        Err(PrefixCoverRefusal::Budget(WorkRefusal::Deadline))
    );
    let optional_expired = WorkBudget::new(
        Some(Instant::now() + Duration::from_secs(10)),
        0,
        64 << 10,
        100_000,
        1,
        Duration::from_secs(20),
    )
    .unwrap();
    assert_eq!(
        build(&problem, &request, &optional_expired).unwrap_err(),
        PrefixCoverRefusal::Budget(WorkRefusal::Deadline)
    );
    // That ledger's original final lane remains live, but construction cannot
    // borrow it to evade its already expired Optional deadline.
    optional_expired.checkpoint_for(LeaseClass::Replay).unwrap();
}
