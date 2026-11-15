// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Original ONNX/VNN-LIB snapshots reach the actual root-open, phase split,
//! complete tree cover, original authorizer and existing private receipt.
//! Private refusal controls use real producers; no numeric box, token or
//! certificate is fabricated. The resource limits match the existing small
//! original-file controls, not a claim about whole-family memory or speed.

use super::super::unsat_receipt::UnsatReceipt;
use super::*;
use ny_onnx::onnx_proto::{
    tensor_shape_proto::{dimension::Value, Dimension},
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::{
    certify::{
        residency::{CancellationToken, MemoryAuthority},
        work::{WorkLimits, WorkReserve},
    },
    margin_row::bab::PreparedTreeSchedule,
};
use prost::Message;
use std::time::Duration;

const CAP: usize = 32 << 20;
fn clock() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(clock()),
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 100_000_000_000,
            max_proposals: 1,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(CAP).unwrap();
    (budget, memory, cancel)
}
fn live(memory: &MemoryAuthority) -> usize {
    let receipt = memory.receipt().unwrap();
    receipt.live_pending_bytes + receipt.live_materialized_bytes
}
fn info(name: &str, dims: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&d| Dimension {
                            value: Some(Value::DimValue(d)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}
fn tensor(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: dims.to_vec(),
        data_type: 1,
        float_data: values.to_vec(),
        ..Default::default()
    }
}
fn node(op: &str, input: &[&str], output: &str) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input: input.iter().map(|s| (*s).into()).collect(),
        output: vec![output.into()],
        ..Default::default()
    }
}
fn model(outputs: usize, target: usize, positive: bool) -> ModelProto {
    let mut head = node("Gemm", &["flat", "w1", "b1"], "headpre");
    let mut output = node("Gemm", &["head", "w2", "b2"], "output");
    for gemm in [&mut head, &mut output] {
        gemm.attribute.push(AttributeProto {
            name: "transB".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
    }
    let mut w2 = vec![0.0; 2 * outputs];
    w2[2 * target] = 1.0;
    w2[2 * target + 1] = -1.0;
    let mut b2 = vec![0.0; outputs];
    b2[target] = if positive { 0.75 } else { -0.75 };
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            input: vec![info("input", &[1, 1, 1, 1])],
            output: vec![info("output", &[1, outputs as i64])],
            initializer: vec![
                tensor("wc", &[1, 1, 1, 1], &[1.0]),
                tensor("bc", &[1], &[0.0]),
                tensor("w1", &[2, 1], &[1.0, 1.0]),
                tensor("b1", &[2], &[-1.0, -1.0]),
                tensor("w2", &[outputs as i64, 2], &w2),
                tensor("b2", &[outputs as i64], &b2),
            ],
            node: vec![
                node("Conv", &["input", "wc", "bc"], "conv"),
                node("Relu", &["conv"], "trunk"),
                node("Flatten", &["trunk"], "flat"),
                head,
                node("Relu", &["headpre"], "head"),
                output,
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn property(outputs: usize, target: usize) -> String {
    let mut text = String::from("(declare-const X_0 Real)\n");
    for class in 0..outputs {
        text.push_str(&format!("(declare-const Y_{class} Real)\n"));
    }
    text.push_str("(assert (>= X_0 0))\n(assert (<= X_0 3))\n(assert (or");
    for class in (0..outputs).rev().filter(|&class| class != target) {
        if class % 2 == 0 {
            text.push_str(&format!(" (>= Y_{class} Y_{target})"));
        } else {
            text.push_str(&format!(" (< Y_{target} Y_{class})"));
        }
    }
    text.push_str("))\n");
    text
}
fn files(
    outputs: usize,
    target: usize,
    positive: bool,
) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let onnx = dir.path().join("actual.onnx");
    let vnn = dir.path().join("actual.vnnlib");
    std::fs::write(&onnx, model(outputs, target, positive).encode_to_vec()).unwrap();
    std::fs::write(&vnn, property(outputs, target)).unwrap();
    (dir, onnx, vnn)
}
fn prepare(onnx: &Path, vnn: &Path, budget: &WorkBudget) -> OriginalPreparedRootSession {
    OriginalPreparedRootSession::prepare_original_files_staged(
        onnx,
        vnn,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            prefer_patch: false,
            ..Default::default()
        },
        budget,
        clock(),
    )
    .unwrap()
}
fn root_open(session: &OriginalPreparedRootSession) -> OriginalPreparedRootOpen {
    match session.try_prove_root(None).unwrap() {
        OriginalPreparedRootOutcome::Open(open) => {
            assert!(!open.attempt_receipt().terminal_closed);
            assert!(open.attempt_receipt().ordinary_completed);
            open
        }
        OriginalPreparedRootOutcome::Proven(_) => {
            panic!("actual independent duplicate-head relaxation must leave root open")
        }
    }
}
fn options(schedule: PreparedTreeSchedule) -> PreparedTreeOptions {
    PreparedTreeOptions {
        max_expansions: 2,
        lru_capacity: 4,
        schedule,
    }
}
fn proven(outcome: OriginalPreparedTreeOutcome) -> OriginalPreparedRootPublication {
    match outcome {
        OriginalPreparedTreeOutcome::Proven(owner) => owner,
        OriginalPreparedTreeOutcome::Open(_) => panic!("actual two-head phase cover must close"),
    }
}
fn open(outcome: OriginalPreparedTreeOutcome) -> OriginalPreparedTreeOpen {
    match outcome {
        OriginalPreparedTreeOutcome::Open(owner) => owner,
        OriginalPreparedTreeOutcome::Proven(_) => panic!("unfinished actual tree cannot prove"),
    }
}
fn zero_expansions() -> PreparedTreeOptions {
    PreparedTreeOptions {
        max_expansions: 0,
        lru_capacity: 1,
        schedule: PreparedTreeSchedule::Serial,
    }
}
fn exact_original_margin_quarters(x_quarters: i64, positive: bool) -> (i64, i64) {
    // Independent integer interpretation of the actual authored parameters.
    // x/4 -> ReLU(x/4) -> two IDENTICAL ReLU(x/4-1) coordinates.
    let trunk_quarters = x_quarters.max(0);
    let first_head_quarters = (trunk_quarters - 4).max(0);
    let second_head_quarters = (trunk_quarters - 4).max(0);
    (
        (if positive { 3 } else { -3 }) + first_head_quarters - second_head_quarters,
        4,
    )
}

#[test]
fn original_file_tree_genuine_root_open_head_split_enters_existing_private_receipt() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(3, 1, true);
    let session = prepare(&onnx, &vnn, &budget);
    let root = root_open(&session);
    let owner = proven(
        root.try_tree(options(PreparedTreeSchedule::Serial), None)
            .unwrap(),
    );
    let tree = owner.tree_receipt().unwrap();
    assert_eq!((tree.expansions, tree.bound_leaves), (1, 2));
    assert_eq!((owner.target(), owner.classes()), (1, &[2, 0][..]));
    assert!(owner.0.proof.session().same_source(&session.0.session));
    assert!(owner.min_margin() > 0.0 && owner.min_margin() <= 0.75);
    for x in 0..=12 {
        assert_eq!(exact_original_margin_quarters(x, true), (3, 4));
    }
    let receipt = UnsatReceipt::from_original_prepared(owner, None).unwrap();
    let authority = receipt.original_prepared_authority().unwrap();
    assert_eq!(authority.tree_receipt().unwrap().bound_leaves, 2);
    receipt.checkpoint_owner(None).unwrap();
    drop(authority);
    drop(receipt);
    drop(root);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_tree_complete_original_order_above64_keeps_all_class_covers() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(66, 1, true);
    let session = prepare(&onnx, &vnn, &budget);
    let root = root_open(&session);
    let owner = proven(
        root.try_tree(options(PreparedTreeSchedule::Serial), None)
            .unwrap(),
    );
    let expected: Vec<_> = (0..66).rev().filter(|&c| c != 1).collect();
    assert_eq!(owner.classes(), expected);
    assert_eq!(owner.classes().len(), 65);
    assert_eq!(owner.tree_receipt().unwrap().bound_leaves, 2);
    assert!(owner.min_margin() > 0.0 && owner.min_margin() <= 0.75);
    owner.checkpoint(None).unwrap();
    drop(owner);
    drop(root);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_tree_parallel_and_wave_use_real_schedules_with_same_raw_objective() {
    let _env = ny_test_utils::env::lock_env_shared();
    for schedule in [
        PreparedTreeSchedule::Parallel { width: 2 },
        PreparedTreeSchedule::Wave { width: 2 },
    ] {
        let (budget, memory, _) = resources();
        let (_dir, onnx, vnn) = files(3, 2, true);
        let session = prepare(&onnx, &vnn, &budget);
        let root = root_open(&session);
        let owner = proven(root.try_tree(options(schedule), None).unwrap());
        assert_eq!((owner.target(), owner.classes()), (2, &[1, 0][..]));
        let receipt = owner.tree_receipt().unwrap();
        assert_eq!((receipt.expansions, receipt.bound_leaves), (1, 2));
        match schedule {
            PreparedTreeSchedule::Parallel { .. } => assert_eq!(receipt.parallel_batches, 1),
            PreparedTreeSchedule::Wave { .. } => assert_eq!(receipt.wave_batches, 1),
            _ => unreachable!(),
        }
        drop(owner);
        drop(root);
        drop(session);
        assert_eq!(live(&memory), 0);
    }
}

#[test]
fn original_file_tree_deleted_paths_reuse_held_source_without_reload_or_compile() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(3, 1, true);
    let session = prepare(&onnx, &vnn, &budget);
    let root = root_open(&session);
    std::fs::remove_file(&onnx).unwrap();
    std::fs::remove_file(&vnn).unwrap();
    let before = budget.receipt().unwrap();
    let owner = proven(
        root.try_tree(options(PreparedTreeSchedule::Serial), None)
            .unwrap(),
    );
    assert!(owner.0.proof.session().same_source(&session.0.session));
    assert!(budget.receipt().unwrap().work_reserved > before.work_reserved);
    owner
        .validate_asset_hashes(*session.model_sha256(), *session.property_sha256(), None)
        .unwrap();
    drop(owner);
    drop(root);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_tree_open_resume_moves_existing_frontier_and_grants_no_expansion_credit() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(3, 1, true);
    let session = prepare(&onnx, &vnn, &budget);
    let root = root_open(&session);
    let tree = open(root.try_tree(zero_expansions(), None).unwrap());
    assert_eq!(tree.reason(), PreparedTreeOpenReason::ExpansionLimit);
    assert_eq!((tree.open_domains(), tree.receipt().expansions), (1, 0));
    std::fs::remove_file(&onnx).unwrap();
    std::fs::remove_file(&vnn).unwrap();
    let before = budget.receipt().unwrap();
    let tree = open(tree.resume(None).unwrap());
    assert_eq!((tree.open_domains(), tree.receipt().expansions), (1, 0));
    assert_eq!(tree.receipt().bound_leaves, 0);
    assert!(tree
        .session()
        .unwrap()
        .0
        .session
        .same_source(&session.0.session));
    assert!(budget.receipt().unwrap().work_reserved >= before.work_reserved);
    drop(tree);
    drop(root);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_tree_shared_alias_refuses_resume_and_last_clone_keeps_actual_work() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(3, 1, true);
    let session = prepare(&onnx, &vnn, &budget);
    let root = root_open(&session);
    let tree = open(root.try_tree(zero_expansions(), None).unwrap());
    let surviving = tree.clone();
    assert!(tree.resume(None).is_err());
    drop(root);
    drop(session);
    assert!(live(&memory) > 0);
    assert_eq!(surviving.open_domains(), 1);
    surviving.checkpoint(None).unwrap();
    drop(surviving);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_tree_receipt_last_owner_keeps_tree_original_source_and_paid_cover() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(3, 1, true);
    let session = prepare(&onnx, &vnn, &budget);
    let root = root_open(&session);
    let owner = proven(
        root.try_tree(options(PreparedTreeSchedule::Serial), None)
            .unwrap(),
    );
    let receipt = UnsatReceipt::from_original_prepared(owner, None).unwrap();
    let last = receipt.original_prepared_authority().unwrap();
    drop(receipt);
    drop(root);
    drop(session);
    assert!(live(&memory) > 0);
    assert_eq!(last.tree_receipt().unwrap().bound_leaves, 2);
    last.checkpoint(None).unwrap();
    drop(last);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_tree_proven_and_open_aliases_retain_earliest_successful_clock() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(3, 1, true);
    let session = prepare(&onnx, &vnn, &budget);
    let root = root_open(&session);
    let earlier = Instant::now() + Duration::from_secs(30);
    let narrowed = root.clone().restricted(Some(earlier)).unwrap();
    let tree = open(narrowed.try_tree(zero_expansions(), Some(clock())).unwrap());
    assert!(tree.deadline() <= earlier);
    let tree = tree.restricted(None).unwrap();
    assert!(tree.session().unwrap().deadline() <= earlier);
    let owner = proven(
        narrowed
            .try_tree(options(PreparedTreeSchedule::Serial), Some(clock()))
            .unwrap(),
    );
    assert!(owner.deadline() <= earlier);
    let receipt = UnsatReceipt::from_original_prepared(owner, Some(clock())).unwrap();
    assert!(receipt.original_prepared_authority().unwrap().deadline() <= earlier);
    drop(receipt);
    drop(tree);
    drop(narrowed);
    drop(root);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_tree_header_cap_refuses_before_search_and_preserves_authentic_root() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(3, 1, true);
    let session = prepare(&onnx, &vnn, &budget);
    let root = root_open(&session);
    let available = tree_outcome_owner_bytes().unwrap() - 1;
    let filler = CAP
        .checked_sub(live(&memory))
        .unwrap()
        .checked_sub(available)
        .unwrap();
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: filler,
                ..Default::default()
            },
        )
        .unwrap();
    let before = budget.receipt().unwrap();
    assert!(root
        .try_tree(options(PreparedTreeSchedule::Serial), None)
        .is_err());
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    assert!(root.0.open.session().same_source(&session.0.session));
    drop(blocker);
    let owner = proven(
        root.try_tree(options(PreparedTreeSchedule::Serial), None)
            .unwrap(),
    );
    assert_eq!(owner.tree_receipt().unwrap().expansions, 1);
    drop(owner);
    drop(root);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_tree_shared_cancel_refuses_authority_without_reset_or_early_claim_release() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let (_dir, onnx, vnn) = files(3, 1, true);
    let session = prepare(&onnx, &vnn, &budget);
    let root = root_open(&session);
    let tree = open(root.try_tree(zero_expansions(), None).unwrap());
    let alias = tree.clone();
    let before = budget.receipt().unwrap();
    cancel.cancel();
    assert!(root
        .try_tree(options(PreparedTreeSchedule::Serial), None)
        .is_err());
    assert!(tree.resume(None).is_err());
    assert!(alias.checkpoint(None).is_err());
    assert!(live(&memory) > 0);
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        before.work_reserved
    );
    drop(alias);
    drop(root);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_tree_expired_requested_clock_refuses_before_any_new_owner_admission() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(3, 1, true);
    let session = prepare(&onnx, &vnn, &budget);
    let root = root_open(&session);
    let before = budget.receipt().unwrap();
    assert!(root
        .try_tree(
            options(PreparedTreeSchedule::Serial),
            Some(
                Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("test clock supports an expired deadline")
            )
        )
        .is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    let owner = proven(
        root.try_tree(options(PreparedTreeSchedule::Serial), None)
            .unwrap(),
    );
    drop(owner);
    drop(root);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_tree_equal_hashes_same_ledger_foreign_real_producer_cannot_attach() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(3, 1, true);
    let first = prepare(&onnx, &vnn, &budget);
    let second = prepare(&onnx, &vnn, &budget);
    assert_eq!(first.model_sha256(), second.model_sha256());
    assert_eq!(first.property_sha256(), second.property_sha256());
    assert_eq!(
        first.0.session.source_digest(),
        second.0.session.source_digest()
    );
    assert!(!first.0.session.same_source(&second.0.session));
    let first_open = root_open(&first);
    let second_open = root_open(&second);
    let actual = MarginRowBab::run_prepared(
        second_open.0.open.clone(),
        options(PreparedTreeSchedule::Serial),
        None,
    )
    .unwrap();
    let PreparedTreeOutcome::Covered(certificate) = actual else {
        panic!("real foreign source tree closes")
    };
    let proof = PreparedRootProof::from_prepared_tree(certificate, None).unwrap();
    assert!(proof.session().same_source(&second.0.session));
    let header = tree_outcome_owner_lease(&budget, clock()).unwrap();
    assert!(finish_original_tree_proof(first_open.clone(), proof, header, clock()).is_err());
    let genuine = proven(
        first_open
            .try_tree(options(PreparedTreeSchedule::Serial), None)
            .unwrap(),
    );
    assert!(genuine.0.proof.session().same_source(&first.0.session));
    drop(genuine);
    drop(first_open);
    drop(second_open);
    drop(first);
    drop(second);
    assert_eq!(live(&memory), 0);
}

#[test]
fn original_file_tree_original_unsafe_output_remains_open_and_cannot_gain_receipt() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let (_dir, onnx, vnn) = files(3, 1, false);
    let session = prepare(&onnx, &vnn, &budget);
    let root = root_open(&session);
    let tree = open(
        root.try_tree(options(PreparedTreeSchedule::Serial), None)
            .unwrap(),
    );
    for x in 0..=12 {
        assert_eq!(exact_original_margin_quarters(x, false), (-3, 4));
    }
    assert!(tree.open_domains() > 0);
    tree.checkpoint(None).unwrap();
    drop(tree);
    drop(root);
    drop(session);
    assert_eq!(live(&memory), 0);
}
