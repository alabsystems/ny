// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Genuine original native files -> prepared root/tree -> actual async v2
//! writer -> independent original-file arithmetic/cover checker. Mutations
//! below are adversarial OFFLINE inputs; they grant no producer authority.

use super::*;
use crate::commands::{
    prepared_root_publication::{
        OriginalPreparedRootOutcome, OriginalPreparedRootSession, OriginalPreparedTreeOutcome,
    },
    unsat_receipt::UnsatReceipt,
};
use ny_core::proof_profile::{EntryPoint, ProofProfile, ResolvedProofProfile};
use ny_onnx::onnx_proto::{
    tensor_shape_proto::{dimension::Value, Dimension},
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto,
    TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::{
    certify::{
        residency::{CancellationToken, MemoryAuthority},
        work::{LeaseClass, WorkBudget, WorkLimits, WorkRequest, WorkReserve},
    },
    margin_row::{
        bab::{PreparedTreeOptions, PreparedTreeSchedule},
        PreparedRootProofOptions, RootPrecisionRuntimePolicy,
    },
};
use prost::Message;
use std::{
    fs,
    time::{Duration, Instant},
};

const CAP: usize = 32 << 20;
fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(at()),
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 100_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(CAP).unwrap();
    (budget, memory, cancel)
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}
fn tensor(name: &str, dims: &[i64], data: &[f32]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: dims.to_vec(),
        data_type: 1,
        float_data: data.to_vec(),
        ..Default::default()
    }
}
fn info(name: &str, width: usize) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        Dimension {
                            value: Some(Value::DimParam("original_batch".into())),
                        },
                        Dimension {
                            value: Some(Value::DimValue(width as i64)),
                        },
                    ],
                }),
            }),
        }),
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
fn model(outputs: usize, target: usize, tree: bool, margin: f32) -> ModelProto {
    let mut weights = vec![0.0; 2 * outputs];
    if tree {
        weights[target] = 1.0;
        weights[outputs + target] = -1.0;
    }
    let mut bias = vec![0.0; outputs];
    bias[target] = margin;
    ModelProto {
        ir_version: 8,
        opset_import: vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            },
            OperatorSetIdProto {
                domain: "ai.onnx.ml".into(),
                version: 2,
            },
        ],
        graph: Some(GraphProto {
            input: vec![info("input", 1)],
            output: vec![info("output", outputs)],
            initializer: vec![
                tensor("w1", &[1, 2], &[1.0, 1.0]),
                tensor("b1", &[2], &[-1.0, -1.0]),
                tensor("w2", &[2, outputs as i64], &weights),
                tensor("b2", &[outputs as i64], &bias),
            ],
            node: vec![
                node("MatMul", &["input", "w1"], "matmul_pre"),
                node("Add", &["matmul_pre", "b1"], "original_pre"),
                node("Relu", &["original_pre"], "hidden"),
                node("MatMul", &["hidden", "w2"], "matmul_output"),
                node("Add", &["matmul_output", "b2"], "output"),
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn property(outputs: usize, target: usize) -> String {
    let mut s = String::from("(declare-const X_0 Real)\n");
    for class in 0..outputs {
        s.push_str(&format!("(declare-const Y_{class} Real)\n"));
    }
    s.push_str("(assert (>= X_0 0))\n(assert (<= X_0 3))\n(assert (or");
    for class in (0..outputs).rev().filter(|&c| c != target) {
        s.push_str(&if class % 2 == 0 {
            format!(" (< Y_{target} Y_{class})")
        } else {
            format!(" (>= Y_{class} Y_{target})")
        });
    }
    s.push_str("))\n");
    s
}
fn files(outputs: usize, target: usize, tree: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("original.onnx"),
        model(outputs, target, tree, 0.75).encode_to_vec(),
    )
    .unwrap();
    fs::write(
        dir.path().join("original.vnnlib"),
        property(outputs, target),
    )
    .unwrap();
    dir
}
fn receipt(dir: &Path, budget: &WorkBudget) -> UnsatReceipt {
    let session = OriginalPreparedRootSession::prepare_original_files_staged(
        &dir.join("original.onnx"),
        &dir.join("original.vnnlib"),
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        budget,
        at(),
    )
    .unwrap();
    let owner = match session.try_prove_root(None).unwrap() {
        OriginalPreparedRootOutcome::Proven(owner) => owner,
        OriginalPreparedRootOutcome::Open(open) => {
            match open
                .try_tree(
                    PreparedTreeOptions {
                        max_expansions: 2,
                        lru_capacity: 4,
                        schedule: PreparedTreeSchedule::Serial,
                    },
                    None,
                )
                .unwrap()
            {
                OriginalPreparedTreeOutcome::Proven(owner) => owner,
                OriginalPreparedTreeOutcome::Open(_) => {
                    panic!("real duplicate native head tree must close")
                }
            }
        }
    };
    UnsatReceipt::from_original_prepared(owner, None).unwrap()
}
fn writer(dir: &Path, name: &str) -> BundleWriter {
    let (writer, sink) = BundleWriter::create(
        &dir.join(name),
        Header {
            profile: ResolvedProofProfile::resolve(
                EntryPoint::VnncompV1,
                Some(ProofProfile::Proof),
            ),
            onnx: dir.join("original.onnx"),
            vnnlib: dir.join("original.vnnlib"),
        },
    )
    .unwrap();
    drop(sink);
    writer
}
fn export(dir: &Path, receipt: &UnsatReceipt, name: &str) -> PathBuf {
    let writer = writer(dir, name);
    let path = writer.final_path.clone();
    let owner = retained_bundle_owner(&writer, receipt).unwrap();
    finish_publication(
        writer,
        "unsat",
        Some(receipt.published_summary()),
        Some(owner),
    );
    assert!(
        path.is_file(),
        "actual final prepared bundle must be atomically written"
    );
    path
}
fn records(path: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}
fn mutated(dir: &Path, name: &str, mut lines: Vec<serde_json::Value>) -> PathBuf {
    lines.pop();
    let mut bytes = Vec::new();
    for line in &lines {
        bytes.extend(serde_json::to_vec(line).unwrap());
        bytes.push(b'\n');
    }
    let end = json!({"kind":"end", "lines":lines.len()+1, "sha256":hex(&Sha256::digest(&bytes))});
    bytes.extend(serde_json::to_vec(&end).unwrap());
    bytes.push(b'\n');
    let path = dir.join(name);
    fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn prepared_replay_actual_native_root_writes_strict_original_clauses_and_complete_node() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let dir = files(3, 2, false);
    let receipt = receipt(dir.path(), &budget);
    assert!(receipt
        .original_prepared_authority()
        .unwrap()
        .tree_receipt()
        .is_none());
    let path = export(dir.path(), &receipt, "root-result");
    let lines = records(&path);
    assert_eq!(lines.iter().filter(|r| r["kind"] == "node").count(), 1);
    assert!(lines
        .iter()
        .any(|r| r["kind"] == "sites" && r["tensors"][0]["name"] == "original_pre"));
    let clauses = lines.iter().find(|r| r["kind"] == "clauses").unwrap();
    assert!(clauses["list"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c["id"] == 0 && c["atoms"][0] == "Y0:-1,Y2:1<0"));
    drop(receipt);
    assert_eq!(live(&memory), 0);
}
#[test]
fn prepared_replay_actual_native_two_child_tree_keeps_real_complementary_phase_nodes() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let dir = files(3, 2, true);
    let receipt = receipt(dir.path(), &budget);
    assert_eq!(
        receipt
            .original_prepared_authority()
            .unwrap()
            .tree_receipt()
            .unwrap()
            .bound_leaves,
        2
    );
    let path = export(dir.path(), &receipt, "tree-result");
    let lines = records(&path);
    let nodes: Vec<_> = lines.iter().filter(|r| r["kind"] == "node").collect();
    assert_eq!(nodes.len(), 2);
    assert!(nodes
        .iter()
        .all(|n| n["lits"].as_array().unwrap().len() == 1
            && n["lits"][0][0] == 1
            && n["lits"][0][1] == 1));
    assert_ne!(nodes[0]["lits"][0][2], nodes[1]["lits"][0][2]);
    drop(receipt);
    assert_eq!(live(&memory), 0);
}
#[test]
fn prepared_replay_full_65_original_classes_nonzero_target_and_exact_order_survive() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let dir = files(66, 17, true);
    let receipt = receipt(dir.path(), &budget);
    let path = export(dir.path(), &receipt, "wide-result");
    let lines = records(&path);
    let expected: Vec<_> = (0..66usize).rev().filter(|&c| c != 17).collect();
    let verdict = lines.iter().find(|r| r["kind"] == "verdict").unwrap();
    // The merged writer's `required` is the clause-id SET of the bundle's
    // `clauses` (sorted; the checker reads it as a set); the export's exact
    // original order survives in every cover node.
    let mut sorted = expected.clone();
    sorted.sort_unstable();
    assert_eq!(verdict["required"], json!(sorted));
    assert!(lines
        .iter()
        .filter(|r| r["kind"] == "node")
        .all(|r| r["dis"] == json!(expected)));
    drop(receipt);
    assert_eq!(live(&memory), 0);
}
#[test]
fn prepared_replay_offline_incomplete_cover_mutations_preserve_rehashed_artifacts() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, _, _) = resources();
    let dir = files(3, 2, true);
    let receipt = receipt(dir.path(), &budget);
    let path = export(dir.path(), &receipt, "cover-result");
    let base = records(&path);
    for case in 0..3 {
        let mut changed = base.clone();
        let positions: Vec<_> = changed
            .iter()
            .enumerate()
            .filter(|(_, r)| r["kind"] == "node")
            .map(|(i, _)| i)
            .collect();
        match case {
            0 => {
                changed.remove(positions[1]);
                for r in &mut changed {
                    if r["kind"] == "verdict" {
                        r["nodes"] = json!(1);
                    }
                }
            }
            1 => changed[positions[1]]["lits"] = changed[positions[0]]["lits"].clone(),
            2 => {
                for &i in &positions {
                    changed[i]["dis"] = json!([1]);
                }
            }
            _ => unreachable!(),
        }
        let altered = mutated(dir.path(), &format!("bad-cover-{case}"), changed);
        assert_ne!(records(&altered), base);
    }
}
#[test]
fn prepared_replay_offline_negative_original_and_cached_margin_mutation_is_rehashed() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, _, _) = resources();
    let dir = files(3, 2, false);
    let receipt = receipt(dir.path(), &budget);
    let path = export(dir.path(), &receipt, "positive-result");
    let mut changed = records(&path);
    let negative = model(3, 2, false, -0.75).encode_to_vec();
    fs::write(dir.path().join("original.onnx"), &negative).unwrap();
    changed[0]["onnx"]["sha256"] = json!(hex(&Sha256::digest(&negative)));
    changed[0]["onnx"]["bytes"] = json!(negative.len());
    for r in &mut changed {
        if r["kind"] == "verdict" {
            r["source"] = json!("authorized");
            r["min_margin"] = json!(1e300);
        }
        if r["kind"] == "node" {
            r["hint"] = json!({"margin":1e300});
        }
    }
    let altered = mutated(dir.path(), "negative-hostile-bundle", changed);
    assert!(altered.is_file());
}
#[test]
fn prepared_replay_offline_site_width_and_strict_clause_mutations_are_rehashed() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, _, _) = resources();
    let dir = files(3, 2, false);
    let receipt = receipt(dir.path(), &budget);
    let path = export(dir.path(), &receipt, "identity-result");
    let base = records(&path);
    for case in 0..3 {
        let mut changed = base.clone();
        let site = changed.iter_mut().find(|r| r["kind"] == "sites").unwrap();
        if case == 0 {
            site["tensors"][0]["name"] = json!("matmul_pre");
        }
        if case == 1 {
            site["tensors"][0]["size"] = json!(3);
        }
        if case == 2 {
            let clause = changed.iter_mut().find(|r| r["kind"] == "clauses").unwrap();
            clause["list"][1]["atoms"][0] = json!("Y0:-1,Y2:1<=0");
        }
        let altered = mutated(dir.path(), &format!("bad-identity-{case}"), changed);
        assert_ne!(records(&altered), base);
    }
}
#[test]
fn prepared_replay_cancelled_source_refuses_before_export_storage_or_final_artifact() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let dir = files(3, 2, true);
    let receipt = receipt(dir.path(), &budget);
    let before = live(&memory);
    let writer = writer(dir.path(), "cancel-result");
    let final_path = writer.final_path.clone();
    cancel.cancel();
    assert!(retained_bundle_owner(&writer, &receipt).is_err());
    assert_eq!(live(&memory), before);
    assert!(matches!(
        writer.finalise("artifact_refused", None, FINALISE_WAIT),
        Artifact::None(_)
    ));
    assert!(!final_path.exists());
    drop(receipt);
    assert_eq!(live(&memory), 0);
}
#[test]
fn prepared_replay_complete_export_overlap_refuses_occupied_original_cap_before_buffers() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let dir = files(3, 2, true);
    let receipt = receipt(dir.path(), &budget);
    let baseline = live(&memory);
    let mut lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: CAP - baseline - 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    lease.mark_materialized().unwrap();
    let blocker = lease.commit_persistent().unwrap();
    let writer = writer(dir.path(), "cap-result");
    let path = writer.final_path.clone();
    assert!(retained_bundle_owner(&writer, &receipt).is_err());
    assert_eq!(live(&memory), CAP - 1);
    assert!(matches!(
        writer.finalise("artifact_refused", None, FINALISE_WAIT),
        Artifact::None(_)
    ));
    assert!(!path.exists());
    drop(blocker);
    assert_eq!(live(&memory), baseline);
    drop(receipt);
    assert_eq!(live(&memory), 0);
}
#[test]
fn prepared_replay_queue_retains_last_original_owner_and_export_claim_until_drop() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let dir = files(3, 2, true);
    let receipt = receipt(dir.path(), &budget);
    let writer = writer(dir.path(), "queue-result");
    let owner = retained_bundle_owner(&writer, &receipt).unwrap();
    let payload = prepared_replay::build(owner.clone()).unwrap();
    let (tx, rx) = mpsc::channel();
    tx.send(Line::PreparedReplay(payload)).unwrap();
    drop(owner);
    drop(receipt);
    assert!(live(&memory) > 0);
    cancel.cancel();
    let Line::PreparedReplay(held) = rx.recv().unwrap() else {
        panic!("real queued prepared batch")
    };
    assert!(held.owner.receipt.checkpoint_owner(None).is_err());
    assert!(live(&memory) > 0);
    drop(held);
    assert_eq!(live(&memory), 0);
    assert!(matches!(
        writer.finalise("artifact_refused", None, FINALISE_WAIT),
        Artifact::None(_)
    ));
}
#[test]
fn prepared_replay_foreign_actual_header_cannot_rename_genuine_owned_batch() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let dir = files(3, 2, false);
    let receipt = receipt(dir.path(), &budget);
    let foreign = files(3, 2, true);
    let writer = writer(foreign.path(), "foreign-result");
    let path = writer.final_path.clone();
    let owner = retained_bundle_owner(&writer, &receipt).unwrap();
    finish_publication(
        writer,
        "unsat",
        Some(receipt.published_summary()),
        Some(owner),
    );
    assert!(!path.exists());
    drop(receipt);
    assert_eq!(live(&memory), 0);
}
#[test]
fn prepared_replay_expired_earlier_handoff_never_renews_through_export_aliases() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources();
    let dir = files(3, 2, false);
    let original = receipt(dir.path(), &budget);
    let expired = UnsatReceipt::from_original_prepared(
        original.original_prepared_authority().unwrap(),
        Some(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test clock supports an expired deadline"),
        ),
    );
    assert!(expired.is_err());
    original.checkpoint_owner(None).unwrap();
    drop(original);
    assert_eq!(live(&memory), 0);
}

/// Test-only fixture transport to the permitted offline command's tests.
/// It keeps the actual source and receipt owners until checking has finished.
/// Production code cannot obtain this object or consume any checker report.
pub(crate) struct IndependentCase {
    pub(crate) paths: Vec<PathBuf>,
    _receipt: UnsatReceipt,
    _dir: tempfile::TempDir,
}
pub(crate) fn independent_case(kind: usize) -> IndependentCase {
    let (budget, _, _) = resources();
    let (outputs, target, tree) = match kind {
        0 | 4 | 5 => (3, 2, false),
        2 => (66, 17, true),
        _ => (3, 2, true),
    };
    let dir = files(outputs, target, tree);
    let receipt = receipt(dir.path(), &budget);
    let original = export(dir.path(), &receipt, "independent-result");
    let base = records(&original);
    let mut paths = Vec::new();
    match kind {
        0..=2 => paths.push(original),
        3 => {
            for mutation in 0..3 {
                let mut changed = base.clone();
                let positions: Vec<_> = changed
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r["kind"] == "node")
                    .map(|(i, _)| i)
                    .collect();
                match mutation {
                    0 => {
                        changed.remove(positions[1]);
                        for r in &mut changed {
                            if r["kind"] == "verdict" {
                                r["nodes"] = json!(1);
                            }
                        }
                    }
                    1 => changed[positions[1]]["lits"] = changed[positions[0]]["lits"].clone(),
                    2 => {
                        for &i in &positions {
                            changed[i]["dis"] = json!([1]);
                        }
                    }
                    _ => unreachable!(),
                }
                paths.push(mutated(
                    dir.path(),
                    &format!("incomplete-{mutation}"),
                    changed,
                ));
            }
        }
        4 => {
            // Hostile offline input changes both original bytes and their
            // hash declaration. Only fresh original arithmetic can reject it.
            let negative = model(3, 2, false, -0.75).encode_to_vec();
            fs::write(dir.path().join("original.onnx"), &negative).unwrap();
            let mut changed = base;
            changed[0]["onnx"]["sha256"] = json!(hex(&Sha256::digest(&negative)));
            changed[0]["onnx"]["bytes"] = json!(negative.len());
            for r in &mut changed {
                if r["kind"] == "verdict" {
                    r["source"] = json!("authorized");
                    r["min_margin"] = json!(1e300);
                }
                if r["kind"] == "node" {
                    r["hint"] = json!({"margin":1e300});
                }
            }
            paths.push(mutated(dir.path(), "hostile-negative-original", changed));
        }
        5 => {
            for mutation in 0..3 {
                let mut changed = base.clone();
                if mutation < 2 {
                    let site = changed.iter_mut().find(|r| r["kind"] == "sites").unwrap();
                    if mutation == 0 {
                        site["tensors"][0]["name"] = json!("matmul_pre");
                    } else {
                        site["tensors"][0]["size"] = json!(3);
                    }
                } else {
                    let clause = changed.iter_mut().find(|r| r["kind"] == "clauses").unwrap();
                    clause["list"][1]["atoms"][0] = json!("Y0:-1,Y2:1<=0");
                }
                paths.push(mutated(dir.path(), &format!("foreign-{mutation}"), changed));
            }
        }
        _ => panic!("unknown independent prepared fixture request"),
    }
    IndependentCase {
        paths,
        _receipt: receipt,
        _dir: dir,
    }
}
