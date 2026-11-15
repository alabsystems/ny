// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use std::time::Duration;

fn budget(memory: &MemoryAuthority, token: &CancellationToken, persistent: usize) -> WorkBudget {
    WorkBudget::new_in(
        memory,
        token,
        WorkLimits {
            deadline: None,
            scratch_bytes: 16 << 20,
            persistent_bytes: persistent,
            max_work: 1_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap()
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
            // An unused tensor still releases at its producer. The earlier
            // residual value must survive this last, distinct consumption.
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
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (1, 2),
            },
            TwinOpSpec::Relu { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![1.0, -1.0],
                bias: vec![0.0, 0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    }
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise shared-model ownership across independent liveness schedules."
)]
fn residual_duplicate_edges_last_use_and_unused_outputs_have_exact_release_boundaries() {
    let memory = MemoryAuthority::new(32 << 20);
    let token = CancellationToken::default();
    let budget = budget(&memory, &token, 16 << 20);
    let model = OriginalModel::compile(fixture(), &budget, None).unwrap();
    let schedule = OriginalLiveness::new(model.clone(), None).unwrap();
    let uses = [1, 1, 3, 1, 0, 1, 1, 1, 0];
    for (tensor, &count) in uses.iter().enumerate() {
        assert_eq!(schedule.consumer_count(tensor), Some(count));
    }
    assert_eq!(schedule.consumer_count(9), None);
    let releases: [&[usize]; 8] = [&[0], &[1], &[], &[2, 4], &[3], &[5], &[6], &[7]];
    for (op, &expected) in releases.iter().enumerate() {
        assert_eq!(schedule.release_after(op), Some(expected));
    }
    assert_eq!(schedule.release_after(8), None);
    assert_eq!(schedule.finish_releases(), &[8]);
    let mut census = [0usize; 9];
    for op in 0..8 {
        for &tensor in schedule.release_after(op).unwrap() {
            census[tensor] += 1;
        }
    }
    for &tensor in schedule.finish_releases() {
        census[tensor] += 1;
    }
    assert_eq!(census, [1; 9]);
}

#[test]
fn schedule_keeps_shared_model_alive_and_drops_only_its_last_claim() {
    let memory = MemoryAuthority::new(32 << 20);
    let token = CancellationToken::default();
    let budget = budget(&memory, &token, 16 << 20);
    let model = OriginalModel::compile(fixture(), &budget, None).unwrap();
    let model_bytes = model.resident_bytes();
    let schedule = OriginalLiveness::new(model.clone(), None).unwrap();
    assert!(schedule.same_model(&model));
    let total = budget.receipt().unwrap().live_persistent_bytes;
    assert!(total > model_bytes);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, total);
    drop(model);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, total);
    drop(schedule);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}

#[test]
fn independent_identical_model_cannot_impersonate_schedule_owner() {
    let memory = MemoryAuthority::new(32 << 20);
    let token = CancellationToken::default();
    let budget = budget(&memory, &token, 16 << 20);
    let a = OriginalModel::compile(fixture(), &budget, None).unwrap();
    let b = OriginalModel::compile(fixture(), &budget, None).unwrap();
    assert_eq!(a.identity(), b.identity());
    let schedule = OriginalLiveness::new(a.clone(), None).unwrap();
    assert!(schedule.same_model(&a));
    assert!(!schedule.same_model(&b));
}

#[test]
fn liveness_one_byte_cap_refusal_preserves_existing_model() {
    let inspect_memory = MemoryAuthority::new(32 << 20);
    let token = CancellationToken::default();
    let inspect_budget = budget(&inspect_memory, &token, 16 << 20);
    let inspect_model = OriginalModel::compile(fixture(), &inspect_budget, None).unwrap();
    let model_bytes = inspect_model.resident_bytes();
    let n = inspect_model.net().ops.len() + 1;
    let schedule_bytes = 512 + n * 3 * size_of::<usize>() + size_of::<usize>();
    let memory = MemoryAuthority::new(32 << 20);
    let budget = budget(&memory, &token, model_bytes + schedule_bytes - 1);
    let model = OriginalModel::compile(fixture(), &budget, None).unwrap();
    assert!(OriginalLiveness::new(model.clone(), None).is_err());
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, model_bytes);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        model_bytes
    );
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    drop(model);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Retain the original paid owner while a cloned attempt is refused."
)]
fn liveness_rejects_expiry_and_shared_cancellation_without_releasing_model() {
    let memory = MemoryAuthority::new(32 << 20);
    let token = CancellationToken::default();
    let budget = budget(&memory, &token, 16 << 20);
    let model = OriginalModel::compile(fixture(), &budget, None).unwrap();
    let bytes = model.resident_bytes();
    let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(OriginalLiveness::new(model.clone(), Some(past)).is_err());
    token.cancel();
    assert!(OriginalLiveness::new(model.clone(), None).is_err());
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, bytes);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, bytes);
}
