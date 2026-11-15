// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::super::compiler::OriginalModel;
use super::*;
use crate::certify::{
    residency::{CancellationToken, MemoryAuthority},
    work::{WorkBudget, WorkLimits, WorkReserve},
};
use crate::margin_row::{RootPrecisionRuntimePolicy, TwinOpSpec, TwinSpec};
use std::time::Duration;

fn fixture() -> TwinSpec {
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![0.1],
                bias: vec![0.2],
                bias_err: vec![0.001],
                weight_rel_err: 0.003,
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
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0, 0.0],
                bias: vec![1.0, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    }
}

fn authority(at: Instant) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(32 << 20);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(at),
            scratch_bytes: 16 << 20,
            persistent_bytes: 16 << 20,
            max_work: 1_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, cancel)
}
fn root_on(budget: &WorkBudget, at: Instant) -> OriginalRoot {
    OriginalRoot::new(
        OriginalModel::compile(fixture(), budget, Some(at)).unwrap(),
        &[-0.3],
        &[0.4],
        RootPrecisionRuntimePolicy::default(),
        Some(at),
    )
    .unwrap()
}

#[test]
fn reference_execution_rejects_foreign_missing_duplicate_and_copied_counter_authority() {
    let _env = crate::tests::lock_env_shared();
    let at = Instant::now() + Duration::from_secs(30);
    let (budget, memory, cancel) = authority(at);
    let root = root_on(&budget, at);
    let foreign = root_on(&budget, at);
    let (foreign_budget, foreign_memory, _) = authority(at);
    let foreign_ledger = root_on(&foreign_budget, at);
    assert_eq!(root.model().identity(), foreign.model().identity());
    let mut lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: ReferenceExecution::allocation_bytes(2).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
    let mut execution = ReferenceExecution::new(
        root.clone(),
        PreparedReferenceArithmetic::Binary32Certified,
        2,
        at,
    )
    .unwrap();
    lease.mark_materialized().unwrap();
    let resident = lease.commit_persistent().unwrap();
    let input = root.input_box_with_deadline(Some(at)).unwrap();
    let (output, receipt) = root
        .derive_reference_f32_with_deadline(0, &input, Some(at))
        .unwrap();
    let foreign_input = foreign_ledger.input_box_with_deadline(Some(at)).unwrap();
    assert!(execution
        .record(0, &foreign_input, &output, Some(receipt))
        .is_err());
    drop(foreign_input);
    execution.record(0, &input, &output, Some(receipt)).unwrap();
    assert!(execution.record(0, &input, &output, Some(receipt)).is_err());
    execution.finish(&root, at).unwrap();
    let before_checkpoint = budget.receipt().unwrap();
    execution.checkpoint(&root, at).unwrap();
    assert_eq!(budget.receipt().unwrap(), before_checkpoint);
    execution.validate(&root, at).unwrap();
    assert!(execution.receipt().multiply_add_pairs > 0);
    assert!(execution.validate(&foreign, at).is_err());
    assert!(execution.validate(&foreign_ledger, at).is_err());
    execution.steps[0].input = 1;
    assert!(execution.validate(&root, at).is_err());
    execution.steps[0].input = 0;
    execution.steps[0].tensor = 2;
    assert!(execution.validate(&root, at).is_err());
    execution.steps[0].tensor = 1;
    let step = execution.steps.pop().unwrap();
    assert!(execution.validate(&root, at).is_err());
    execution.steps.push(step);
    execution.steps.push(Step {
        op: 0,
        input: 0,
        tensor: 1,
        conv: true,
        binary32: Some(receipt),
    });
    assert!(execution.validate(&root, at).is_err());
    execution.steps.pop();
    execution.receipt.multiply_add_pairs += 1;
    assert!(execution.validate(&root, at).is_err());
    execution.receipt.multiply_add_pairs -= 1;
    execution.arithmetic = PreparedReferenceArithmetic::F64;
    assert!(execution.validate(&root, at).is_err());
    execution.arithmetic = PreparedReferenceArithmetic::Binary32Certified;
    execution.validate(&root, at).unwrap();
    execution.deadline = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    assert!(execution.validate(&root, at).is_err());
    execution.deadline = at;
    cancel.cancel();
    assert!(execution.validate(&root, at).is_err());
    drop((
        execution,
        resident,
        input,
        output,
        root,
        foreign,
        foreign_ledger,
    ));
    let foreign_live = foreign_memory.receipt().unwrap();
    assert_eq!(
        foreign_live.live_pending_bytes + foreign_live.live_materialized_bytes,
        0
    );
    let r = memory.receipt().unwrap();
    assert_eq!(r.live_pending_bytes + r.live_materialized_bytes, 0);
}
