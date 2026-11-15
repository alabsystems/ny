// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Real original-file issuer and actual retained bundle writer controls.
//! The serialized source is truthful; these controls do not claim that a
//! zero-node bundle contains a separately replayable numerical proof.
//! Process-global WIRED and installed proof sinks are never modified.

use super::*;
use crate::commands::{
    prepared_root_publication::OriginalPreparedRootPublication,
    unsat_receipt::{PublishedUnsatSource, UnsatReceipt},
};
use ny_core::proof_profile::{EntryPoint, ProofProfile, ResolvedProofProfile};
use ny_onnx::onnx_proto::{
    tensor_shape_proto::{dimension::Value, Dimension},
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::{
    certify::{
        residency::{CancellationToken, MemoryAuthority},
        work::{LeaseClass, WorkBudget, WorkLimits, WorkRequest, WorkReserve},
    },
    margin_row::{PreparedRootProofOptions, RootPrecisionRuntimePolicy},
};
use prost::Message;
use std::{
    fs,
    time::{Duration, Instant},
};

const CAP: usize = 32 << 20;

fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(Instant::now() + Duration::from_secs(60)),
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

fn tensor(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: dims.to_vec(),
        data_type: 1,
        float_data: values.to_vec(),
        ..Default::default()
    }
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
                        .map(|&v| Dimension {
                            value: Some(Value::DimValue(v)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input: inputs.iter().map(|s| (*s).into()).collect(),
        output: vec![output.into()],
        ..Default::default()
    }
}

fn original_model(inputs: usize) -> ModelProto {
    let mut first = node("Gemm", &["flat", "w1", "b1"], "head_pre");
    let mut last = node("Gemm", &["head", "w2", "b2"], "output");
    for gemm in [&mut first, &mut last] {
        gemm.attribute.push(AttributeProto {
            name: "transB".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
    }
    let mut weights = vec![0.0; inputs * 2];
    weights[0] = 1.0;
    weights[inputs] = -1.0;
    let n = i64::try_from(inputs).unwrap();
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            input: vec![info("input", &[1, 1, 1, n])],
            output: vec![info("output", &[1, 3])],
            initializer: vec![
                tensor("wc", &[1, 1, 1, 1], &[1.0]),
                tensor("bc", &[1], &[0.0]),
                tensor("w1", &[2, n], &weights),
                tensor("b1", &[2], &[0.0, 0.0]),
                tensor("w2", &[3, 2], &[0.0; 6]),
                tensor("b2", &[3], &[0.0, 2.0, 0.0]),
            ],
            node: vec![
                node("Conv", &["input", "wc", "bc"], "conv"),
                node("Relu", &["conv"], "trunk"),
                node("Flatten", &["trunk"], "flat"),
                first,
                node("Relu", &["head_pre"], "head"),
                last,
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn original_property(inputs: usize) -> String {
    use std::fmt::Write;
    let mut source = String::new();
    for i in 0..inputs {
        writeln!(source, "(declare-const X_{i} Real)").unwrap();
    }
    for i in 0..3 {
        writeln!(source, "(declare-const Y_{i} Real)").unwrap();
    }
    for i in 0..inputs {
        writeln!(source, "(assert (>= X_{i} -1))").unwrap();
        writeln!(source, "(assert (<= X_{i} 1))").unwrap();
    }
    source.push_str("(assert (or (and (>= Y_2 Y_1)) (< Y_1 Y_0)))\n");
    source
}

fn issue(budget: &WorkBudget, inputs: usize) -> (tempfile::TempDir, UnsatReceipt) {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("original.onnx");
    let property = dir.path().join("original.vnnlib");
    fs::write(&model, original_model(inputs).encode_to_vec()).unwrap();
    fs::write(&property, original_property(inputs)).unwrap();
    let owner = OriginalPreparedRootPublication::prove_original_files(
        &model,
        &property,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        budget,
        Instant::now() + Duration::from_secs(60),
    )
    .expect("real original model/property issuer and complete producer");
    let receipt = UnsatReceipt::from_original_prepared(owner, None).unwrap();
    (dir, receipt)
}

fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}

fn header(dir: &Path) -> Header {
    Header {
        profile: ResolvedProofProfile::resolve(EntryPoint::VnncompV1, Some(ProofProfile::Proof)),
        onnx: dir.join("original.onnx"),
        vnnlib: dir.join("original.vnnlib"),
    }
}

fn verdict(receipt: &UnsatReceipt) -> VerdictInfo {
    let (_classes, min_margin, source) = receipt.published_summary();
    VerdictInfo {
        source,
        min_margin,
        completion: Completion::OwnStream,
    }
}

fn actual_lines(file: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(file)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn retained_bundle_actual_original_header_summary_and_end_survive_final_rename() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let result = dir.path().join("result.txt");
    // The proof artifact is postpublication; it cannot write this verdict.
    fs::write(&result, "unsat\n").unwrap();
    let (writer, sink) = BundleWriter::create(&result, header(dir.path())).unwrap();
    let original_live = live(&memory);
    let before = budget.receipt().unwrap();
    let owner = retained_bundle_owner(&writer, &receipt).unwrap();
    assert!(owner._storage.shares_ledger(&budget));
    assert_eq!(live(&memory), original_live + owner._storage.bytes());
    let info = verdict(&receipt);
    let source = receipt.original_prepared_authority().unwrap();
    let model_hash = *source.model_sha256();
    let property_hash = *source.property_sha256();
    drop(source);
    drop(receipt);
    drop(sink);
    let artifact = writer.finalise_owned("unsat", Some(info), Some(owner), Duration::from_secs(2));
    // Nothing was streamed (no replay payload was sent), so the merged
    // writer's completion claims each of the instance's two canonical clauses
    // (ids by position) with one whole-region `unstreamed:` node: an
    // `unconfirmed` bundle, never a certificate.
    assert!(
        matches!(
            artifact,
            Artifact::Written {
                nodes: 2,
                claims: 2,
                ..
            }
        ),
        "{artifact:?}"
    );
    assert!(!partial_path(&result).exists());
    assert_eq!(fs::read_to_string(&result).unwrap(), "unsat\n");
    let lines = actual_lines(&bundle_path(&result));
    assert_eq!(header_asset_digest(&lines[0], "onnx").unwrap(), model_hash);
    assert_eq!(
        header_asset_digest(&lines[0], "vnnlib").unwrap(),
        property_hash
    );
    let summary = lines.iter().find(|line| line["kind"] == "verdict").unwrap();
    assert_eq!(
        summary["source"],
        PublishedUnsatSource::OriginalPrepared.token()
    );
    // `required` is every clause id of the bundle's `clauses` (the merged
    // writer), here the instance's canonical clauses by position.
    assert_eq!(summary["required"], serde_json::json!([0, 1]));
    let claims: Vec<&serde_json::Value> =
        lines.iter().filter(|line| line["kind"] == "node").collect();
    assert_eq!(claims.len(), 2);
    assert!(claims
        .iter()
        .all(|node| node["src"] == "unstreamed:original_prepared_root"
            && node["close"] == "opaque"
            && node["lits"] == serde_json::json!([])));
    assert!(summary["min_margin"].as_f64().unwrap() > 1.99);
    assert_eq!(lines.last().unwrap()["kind"], "end");
    assert!(budget.receipt().unwrap().admitted > before.admitted);
    assert_eq!(live(&memory), 0);
}

#[test]
fn retained_bundle_actual_queued_cancelled_owner_never_creates_final_artifact() {
    let (budget, memory, cancel) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let result = dir.path().join("result.txt");
    fs::write(&result, "unsat\n").unwrap();
    let (writer, sink) = BundleWriter::create(&result, header(dir.path())).unwrap();
    let owner = retained_bundle_owner(&writer, &receipt).unwrap();
    let info = verdict(&receipt);
    drop(receipt);
    drop(sink);
    cancel.cancel();
    let artifact = writer.finalise_owned("unsat", Some(info), Some(owner), Duration::from_secs(2));
    assert!(matches!(artifact, Artifact::None(_)));
    assert!(!bundle_path(&result).exists());
    assert_eq!(fs::read_to_string(&result).unwrap(), "unsat\n");
    assert_eq!(live(&memory), 0);
}

#[test]
fn retained_bundle_successful_earlier_clock_survives_real_owner_queue_handoff() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let source = receipt.original_prepared_authority().unwrap();
    drop(receipt);
    let earlier = Instant::now() + Duration::from_millis(200);
    let receipt = UnsatReceipt::from_original_prepared(source, Some(earlier)).unwrap();
    let result = dir.path().join("result.txt");
    fs::write(&result, "unsat\n").unwrap();
    let (writer, sink) = BundleWriter::create(&result, header(dir.path())).unwrap();
    let owner = retained_bundle_owner(&writer, &receipt).unwrap();
    let info = verdict(&receipt);
    drop(receipt);
    drop(sink);
    std::thread::sleep(
        earlier.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
    );
    let artifact = writer.finalise_owned("unsat", Some(info), Some(owner), Duration::from_secs(2));
    assert!(matches!(artifact, Artifact::None(_)));
    assert!(!bundle_path(&result).exists());
    assert_eq!(fs::read_to_string(&result).unwrap(), "unsat\n");
    assert_eq!(live(&memory), 0);
}

#[test]
fn retained_bundle_equal_geometry_foreign_original_file_cannot_rebind_the_header() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let mut changed = original_model(1);
    changed
        .graph
        .as_mut()
        .unwrap()
        .initializer
        .iter_mut()
        .find(|t| t.name == "b2")
        .unwrap()
        .float_data[1] = 3.0;
    // Change parameter bits, leaving the actual graph and all dimensions equal.
    fs::write(dir.path().join("original.onnx"), changed.encode_to_vec()).unwrap();
    let result = dir.path().join("result.txt");
    fs::write(&result, "unsat\n").unwrap();
    let (writer, sink) = BundleWriter::create(&result, header(dir.path())).unwrap();
    let owner = retained_bundle_owner(&writer, &receipt).unwrap();
    let info = verdict(&receipt);
    drop(receipt);
    drop(sink);
    let artifact = writer.finalise_owned("unsat", Some(info), Some(owner), Duration::from_secs(2));
    assert!(matches!(artifact, Artifact::None(_)));
    assert!(!bundle_path(&result).exists());
    assert_eq!(fs::read_to_string(&result).unwrap(), "unsat\n");
    assert_eq!(live(&memory), 0);
}

#[test]
fn retained_bundle_same_ledger_cap_refuses_before_summary_and_preserves_prior_source() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let result = dir.path().join("result.txt");
    fs::write(&result, "unsat\n").unwrap();
    let (writer, sink) = BundleWriter::create(&result, header(dir.path())).unwrap();
    let occupied = budget.receipt().unwrap().live_persistent_bytes;
    let mut blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: CAP - occupied - 1024,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    blocker.mark_materialized().unwrap();
    let blocker = blocker.commit_persistent().unwrap();
    let before = live(&memory);
    assert!(retained_bundle_owner(&writer, &receipt).is_err());
    assert_eq!(live(&memory), before);
    drop(sink);
    assert!(matches!(
        writer.finalise("artifact_refused", None, Duration::from_secs(2)),
        Artifact::None(_)
    ));
    assert!(!bundle_path(&result).exists());
    receipt.checkpoint_owner(None).unwrap();
    assert_eq!(fs::read_to_string(&result).unwrap(), "unsat\n");
    drop(blocker);
    drop(receipt);
    assert_eq!(live(&memory), 0);
}

#[test]
fn retained_bundle_real_channel_payload_retains_last_source_until_cancelled_consumer_drop() {
    let (budget, memory, cancel) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let result = dir.path().join("result.txt");
    let (writer, sink) = BundleWriter::create(&result, header(dir.path())).unwrap();
    let owner = retained_bundle_owner(&writer, &receipt).unwrap();
    let info = verdict(&receipt);
    let weak = Arc::downgrade(&owner);
    let expected = live(&memory);
    // Use the production Line/VerdictPayload and real mpsc queue, independent
    // of process-global proof sink registration.
    let (tx, rx) = mpsc::channel();
    tx.send(Line::Finish(
        Some(VerdictPayload {
            info,
            owner: Some(owner),
        }),
        "unsat".to_string(),
    ))
    .unwrap();
    drop(receipt);
    assert_eq!(live(&memory), expected);
    assert!(weak.upgrade().is_some());
    drop(sink);
    let _ = writer.finalise("not_published", None, Duration::from_secs(2));
    cancel.cancel();
    let payload = match rx.recv().unwrap() {
        Line::Finish(Some(payload), _) => payload,
        _ => panic!("actual verdict payload"),
    };
    assert!(payload
        .owner
        .as_ref()
        .unwrap()
        .receipt
        .checkpoint_owner(None)
        .is_err());
    drop(payload);
    assert!(weak.upgrade().is_none());
    assert_eq!(live(&memory), 0);
}

struct BufferDropProbe {
    memory: MemoryAuthority,
    expected: usize,
}
impl Write for BufferDropProbe {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        assert_eq!(live(&self.memory), self.expected);
        Err(std::io::Error::other("intentional output failure"))
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl Drop for BufferDropProbe {
    fn drop(&mut self) {
        assert_eq!(
            live(&self.memory),
            self.expected,
            "actual buffer must drop before last retained claim"
        );
    }
}

#[test]
fn retained_bundle_digesting_output_failure_drops_actual_buffer_before_last_owner_claim() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let result = dir.path().join("result.txt");
    let (writer, sink) = BundleWriter::create(&result, header(dir.path())).unwrap();
    let owner = retained_bundle_owner(&writer, &receipt).unwrap();
    drop(sink);
    let _ = writer.finalise("not_published", None, Duration::from_secs(2));
    let expected = live(&memory);
    let mut buffer = Digesting {
        w: BufferDropProbe {
            memory: memory.clone(),
            expected,
        },
        sha: Sha256::new(),
        lines: 0,
        retained_owner: Some(owner),
    };
    drop(receipt);
    assert!(buffer.line("end-of-paid-stream").is_err());
    drop(buffer);
    assert_eq!(live(&memory), 0);
}
