// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Actual original ONNX/property issuer -> genuine receipt -> real writer.
//! No imported token, verdict JSON or fabricated successful watchdog ACK is
//! used. Short-grace controls exercise actual staging, queue ownership and
//! refusal before rename; positive external-helper ACK qualification remains
//! a separate complete-driver obligation.

use super::*;
use crate::commands::{
    prepared_root_publication::OriginalPreparedRootPublication, unsat_receipt::PublishedUnsatSource,
};
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
    sync::{atomic::AtomicUsize, mpsc},
    thread::JoinHandle,
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
    issue_with_arithmetic(
        budget,
        inputs,
        ny_propagate::margin_row::PreparedReferenceArithmetic::F64,
    )
}
fn issue_with_arithmetic(
    budget: &WorkBudget,
    inputs: usize,
    arithmetic: ny_propagate::margin_row::PreparedReferenceArithmetic,
) -> (tempfile::TempDir, UnsatReceipt) {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("original.onnx");
    let property = dir.path().join("original.vnnlib");
    fs::write(&model, original_model(inputs).encode_to_vec()).unwrap();
    fs::write(&property, original_property(inputs)).unwrap();
    let owner = OriginalPreparedRootPublication::prove_original_files(
        &model,
        &property,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            reference_arithmetic: arithmetic,
            ..Default::default()
        },
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

fn queue(
    receipt: UnsatReceipt,
    destination: &Path,
) -> (
    ShortGracePublicationRequest,
    JoinHandle<std::io::Result<bool>>,
) {
    let (sender, receiver) = mpsc::channel();
    let publisher = ShortGraceResultPublisher {
        results_file: destination.to_owned(),
        sender,
    };
    let writer =
        std::thread::spawn(move || publisher.publish_terminal(VnncompResult::Unsat(receipt)));
    let request = receiver.recv_timeout(Duration::from_secs(2)).unwrap();
    (request, writer)
}

fn deliver(request: ShortGracePublicationRequest) -> mpsc::Receiver<ShortGracePublicationRequest> {
    let (sender, receiver) = mpsc::channel();
    sender.send(request).unwrap();
    drop(sender);
    receiver
}

#[test]
fn prepared_writer_actual_atomic_result_preserves_genuine_source_then_releases_last_owner() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let file = dir.path().join("results.txt");
    fs::write(&file, "unknown\n").unwrap();
    let retained = receipt.clone();
    let original_live = live(&memory);
    let before = budget.receipt().unwrap();
    assert!(write_terminal_result(&file, None, VnncompResult::Unsat(receipt)).unwrap());
    assert_eq!(fs::read_to_string(&file).unwrap(), "unsat\n");
    assert!(!file.with_extension("result.tmp").exists());
    assert_eq!(retained.original_prepared_target(), Some(1));
    assert_eq!(retained.published_summary().0, [2, 0]);
    assert_eq!(
        retained.published_summary().2,
        PublishedUnsatSource::OriginalPrepared
    );
    retained.checkpoint_owner(None).unwrap();
    let after = budget.receipt().unwrap();
    assert!(after.admitted > before.admitted);
    assert_eq!(after.proposals_reserved, before.proposals_reserved);
    assert_eq!(
        live(&memory),
        original_live,
        "temporary writer claim was released"
    );
    drop(retained);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_writer_binary32_reference_reaches_actual_original_writer_and_cancellation_refuses() {
    use ny_propagate::margin_row::PreparedReferenceArithmetic;
    for cancelled in [false, true] {
        let (budget, memory, cancel) = resources();
        let (dir, receipt) =
            issue_with_arithmetic(&budget, 1, PreparedReferenceArithmetic::Binary32Certified);
        let owner = receipt.original_prepared_authority().unwrap();
        let reference = owner.attempt_receipt().reference;
        assert_eq!(
            reference.arithmetic,
            PreparedReferenceArithmetic::Binary32Certified
        );
        assert_eq!(
            (reference.expected_conv_ops, reference.completed_conv_ops),
            (1, 1)
        );
        assert!(reference.multiply_add_pairs > 0);
        let file = dir.path().join("binary32-results.txt");
        fs::write(&file, "unknown\n").unwrap();
        if cancelled {
            cancel.cancel();
        }
        let result = write_terminal_result(&file, None, VnncompResult::Unsat(receipt));
        if cancelled {
            assert!(!matches!(result, Ok(true)));
            assert_eq!(fs::read_to_string(&file).unwrap(), "unknown\n");
        } else {
            assert!(result.unwrap());
            assert_eq!(fs::read_to_string(&file).unwrap(), "unsat\n");
        }
        assert!(!file.with_extension("result.tmp").exists());
        assert!(
            live(&memory) > 0,
            "the separately retained original owner remains paid"
        );
        drop(owner);
        assert_eq!(live(&memory), 0);
    }
}

#[test]
fn prepared_writer_binary32_original_open_root_tree_cover_reaches_final_writer() {
    use crate::commands::prepared_root_route::{
        admission, try_prepared_root_route, PreparedRouteOutcome, PreparedRouteStatus,
    };
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("tree.onnx");
    let property = dir.path().join("tree.vnnlib");
    let mut original = original_model(1);
    let graph = original.graph.as_mut().unwrap();
    graph.output[0] = info("output", &[1, 66]);
    for tensor in &mut graph.initializer {
        match tensor.name.as_str() {
            "w1" => tensor.float_data = vec![1.0, 1.0],
            "b1" => tensor.float_data = vec![-1.0, -1.0],
            "w2" => {
                tensor.dims = vec![66, 2];
                tensor.float_data = vec![0.0; 132];
                tensor.float_data[34] = 1.0;
                tensor.float_data[35] = -1.0;
            }
            "b2" => {
                tensor.dims = vec![66];
                tensor.float_data = vec![0.0; 66];
                tensor.float_data[17] = 0.75;
            }
            _ => {}
        }
    }
    fs::write(&model, original.encode_to_vec()).unwrap();
    let mut objective = String::from("(declare-const X_0 Real)\n");
    for c in 0..66 {
        objective.push_str(&format!("(declare-const Y_{c} Real)\n"));
    }
    objective.push_str("(assert (>= X_0 0))\n(assert (<= X_0 3))\n(assert (or");
    for c in (0usize..66).rev().filter(|&c| c != 17) {
        objective.push_str(&format!(" (>= Y_{c} Y_17)"));
    }
    objective.push_str("))\n");
    fs::write(&property, objective).unwrap();
    let levers = ny_levers::LeverSet::freeze_with(ny_levers::all(), |_| None);
    let policy = ny_propagate::margin_row::MarginRowRuntimePolicy::from_levers(&levers).unwrap();
    let outcome = try_prepared_root_route(
        &model,
        &property,
        &policy,
        admission(false, &policy, &levers),
        Some(Instant::now() + Duration::from_secs(60)),
        None,
    );
    let PreparedRouteOutcome::Proven { receipt, report } = outcome else {
        panic!("genuine binary32 original tree proof");
    };
    assert_eq!(report.status, PreparedRouteStatus::TreeProven);
    let root = report.root.unwrap();
    assert!(!root.terminal_closed && root.ordinary_completed);
    assert!(root.reference.multiply_add_pairs > 0);
    assert_eq!(report.tree.unwrap().bound_leaves, 2);
    let file = dir.path().join("tree-results.txt");
    fs::write(&file, "unknown\n").unwrap();
    assert!(write_terminal_result(&file, None, VnncompResult::Unsat(receipt)).unwrap());
    assert_eq!(fs::read_to_string(file).unwrap(), "unsat\n");
}

#[test]
fn prepared_writer_cancelled_original_owner_cannot_replace_prior_placeholder() {
    let (budget, memory, cancel) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let file = dir.path().join("results.txt");
    fs::write(&file, "unknown\n").unwrap();
    cancel.cancel();
    let before = budget.receipt().unwrap();
    assert!(write_results(&file, &VnncompResult::Unsat(receipt.clone())).is_err());
    assert_eq!(fs::read_to_string(&file).unwrap(), "unknown\n");
    assert!(!file.with_extension("result.tmp").exists());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    drop(receipt);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_writer_successful_earlier_owner_clock_cannot_be_renewed_at_final_write() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let owner = receipt.original_prepared_authority().unwrap();
    drop(receipt);
    let earlier = Instant::now() + Duration::from_millis(200);
    let receipt = UnsatReceipt::from_original_prepared(owner, Some(earlier)).unwrap();
    let file = dir.path().join("results.txt");
    fs::write(&file, "unknown\n").unwrap();
    std::thread::sleep(
        earlier.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
    );
    let before = budget.receipt().unwrap();
    assert!(write_results(&file, &VnncompResult::Unsat(receipt.clone())).is_err());
    assert_eq!(fs::read_to_string(&file).unwrap(), "unknown\n");
    assert!(!file.with_extension("result.tmp").exists());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    drop(receipt);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_writer_same_original_cap_refuses_before_temp_allocation_and_keeps_prior_proof() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let file = dir.path().join("results.txt");
    fs::write(&file, "unknown\n").unwrap();
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
    assert!(write_results(&file, &VnncompResult::Unsat(receipt.clone())).is_err());
    assert_eq!(live(&memory), before);
    assert_eq!(fs::read_to_string(&file).unwrap(), "unknown\n");
    assert!(!file.with_extension("result.tmp").exists());
    receipt.checkpoint_owner(None).unwrap();
    drop(blocker);
    write_results(&file, &VnncompResult::Unsat(receipt.clone())).unwrap();
    assert_eq!(fs::read_to_string(&file).unwrap(), "unsat\n");
    drop(receipt);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_writer_actual_queued_request_owns_last_source_and_exact_storage_claim() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let original_live = live(&memory);
    let file = dir.path().join("results.txt");
    fs::write(&file, "unknown\n").unwrap();
    let (request, writer) = queue(receipt, &file);
    let staged = request.staged_path.clone();
    assert_eq!(fs::read_to_string(&staged).unwrap(), "unsat\n");
    let claim = request._publication_claim.as_ref().unwrap();
    assert_eq!(claim.bytes(), retained_publication_bytes(&file).unwrap());
    assert!(claim.shares_ledger(&budget));
    assert_eq!(live(&memory), original_live + claim.bytes());
    assert!(request.receipt.as_ref().unwrap().has_retained_owner());
    // A genuine rejection ends the sender without simulating publication.
    // The actual received request now owns the last source/proof and storage.
    request
        .acknowledgement
        .send(Err(
            "retained ownership inspection rejected publication".into()
        ))
        .unwrap();
    assert!(writer.join().unwrap().is_err());
    request
        .receipt
        .as_ref()
        .unwrap()
        .checkpoint_owner(None)
        .unwrap();
    assert_eq!(live(&memory), original_live + claim.bytes());
    assert_eq!(fs::read_to_string(&file).unwrap(), "unknown\n");
    drop(request);
    assert!(!staged.exists());
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_writer_actual_queued_cancellation_declines_before_any_rename() {
    let (budget, memory, cancel) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let file = dir.path().join("results.txt");
    fs::write(&file, "unknown\n").unwrap();
    let (request, writer) = queue(receipt, &file);
    let staged = request.staged_path.clone();
    cancel.cancel();
    let renames = AtomicUsize::new(0);
    let outcome = await_short_grace_terminal_result_with_rename_and_record(
        &file,
        Instant::now() + Duration::from_secs(2),
        deliver(request),
        None,
        |src, dst| {
            renames.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            fs::rename(src, dst)
        },
        |_| panic!("cancelled original owner cannot reach a publication ACK"),
    );
    assert_eq!(outcome, ShortGraceCoordinatorOutcome::PublicationFailed);
    assert_eq!(renames.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(writer.join().unwrap().is_err());
    assert_eq!(fs::read_to_string(&file).unwrap(), "unknown\n");
    assert!(!staged.exists());
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_writer_actual_queued_source_expiry_declines_and_releases_before_rename() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let source = receipt.original_prepared_authority().unwrap();
    drop(receipt);
    let earlier = Instant::now() + Duration::from_millis(200);
    let receipt = UnsatReceipt::from_original_prepared(source, Some(earlier)).unwrap();
    let file = dir.path().join("results.txt");
    fs::write(&file, "unknown\n").unwrap();
    let (request, writer) = queue(receipt, &file);
    let staged = request.staged_path.clone();
    std::thread::sleep(
        earlier.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
    );
    let outcome = await_short_grace_terminal_result_with_rename_and_record(
        &file,
        Instant::now() + Duration::from_secs(2),
        deliver(request),
        None,
        |_, _| panic!("expired original owner must not rename"),
        |_| panic!("expired original owner must not reach a publication ACK"),
    );
    assert_eq!(outcome, ShortGraceCoordinatorOutcome::PublicationFailed);
    assert!(writer.join().unwrap().is_err());
    assert_eq!(fs::read_to_string(&file).unwrap(), "unknown\n");
    assert!(!staged.exists());
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_writer_coordinator_cutoff_includes_actual_original_owner_validation_cost() {
    let (budget, memory, _) = resources();
    // This genuine 256-input declaration makes the retained exact-input
    // replay nontrivial, without raising the original 32 MiB allowance.
    let (dir, receipt) = issue(&budget, 256);
    let file = dir.path().join("results.txt");
    fs::write(&file, "unknown\n").unwrap();
    let mut observed_validation_crossing = false;
    // A bounded sequence of measured sub-windows covers the concrete cutoff
    // race despite timer granularity. No successful ACK is invented: if the
    // real rename arrives before the cutoff, absence of a watchdog must still
    // restore unknown through the actual production refusal path.
    for attempt in 0..30_u32 {
        let started = Instant::now();
        receipt.checkpoint_owner(None).unwrap();
        let cost = started.elapsed();
        let window = (cost * (attempt % 15 + 1) / 16).max(Duration::from_nanos(1));
        let (request, writer) = queue(receipt.clone(), &file);
        let staged = request.staged_path.clone();
        let receiver = deliver(request);
        let before = budget.receipt().unwrap().admitted;
        let cutoff = Instant::now() + window;
        let renames = AtomicUsize::new(0);
        let outcome = await_short_grace_terminal_result_with_rename_and_record(
            &file,
            cutoff,
            receiver,
            None,
            |src, dst| {
                renames.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                fs::rename(src, dst)
            },
            |_| panic!("no real external helper is present, so no ACK can be fabricated"),
        );
        let entered_paid_validation = budget.receipt().unwrap().admitted > before;
        if outcome == ShortGraceCoordinatorOutcome::PublicationFailed {
            assert_eq!(renames.load(std::sync::atomic::Ordering::SeqCst), 0);
            assert!(writer.join().unwrap().is_err());
            if entered_paid_validation {
                observed_validation_crossing = true;
            }
        } else {
            assert!(matches!(
                outcome,
                ShortGraceCoordinatorOutcome::Deadline
                    | ShortGraceCoordinatorOutcome::AcknowledgementFailed
            ));
            assert!(!writer.join().unwrap().unwrap());
        }
        assert_eq!(fs::read_to_string(&file).unwrap(), "unknown\n");
        assert!(!staged.exists());
        if observed_validation_crossing {
            break;
        }
    }
    assert!(observed_validation_crossing, "must actually enter paid original-owner validation and hit its requested publication cutoff; an already expired channel check is not sufficient");
    drop(receipt);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_writer_closed_actual_queue_drops_staged_verdict_and_all_retained_claims() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let file = dir.path().join("results.txt");
    fs::write(&file, "unknown\n").unwrap();
    let (sender, receiver) = mpsc::channel();
    drop(receiver);
    let publisher = ShortGraceResultPublisher {
        results_file: file.clone(),
        sender,
    };
    assert!(!publisher
        .publish_terminal(VnncompResult::Unsat(receipt))
        .unwrap());
    assert_eq!(fs::read_to_string(&file).unwrap(), "unknown\n");
    assert!(!fs::read_dir(dir.path()).unwrap().any(|e| e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .contains(".staged")));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_writer_actual_partial_staging_failure_removes_unpublished_bytes() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let file = dir.path().join("results.txt");
    fs::write(&file, "unknown\n").unwrap();
    let original_live = live(&memory);
    let claim = receipt
        .reserve_owner_publication(retained_publication_bytes(&file).unwrap(), None)
        .unwrap();
    let mut partial_path = None;
    let result = VnncompResult::Unsat(receipt);
    let failure = stage_short_grace_result_with_writer(&file, &result, |path, rendered| {
        assert_eq!(rendered, "unsat\n");
        fs::write(path, &rendered.as_bytes()[..2])?;
        assert_eq!(fs::read(path)?, b"un");
        partial_path = Some(path.to_owned());
        Err(std::io::Error::other(
            "injected I/O error after an actual partial file write",
        ))
    });
    assert!(failure.is_err());
    assert!(!partial_path.unwrap().exists());
    assert_eq!(fs::read_to_string(&file).unwrap(), "unknown\n");
    drop(claim);
    assert_eq!(live(&memory), original_live);
    drop(result);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_writer_actual_rename_failure_removes_temporary_verdict() {
    let (budget, memory, _) = resources();
    let (dir, receipt) = issue(&budget, 1);
    let file = dir.path().join("results.txt");
    fs::create_dir(&file).unwrap();
    let marker = file.join("existing.txt");
    fs::write(&marker, "existing destination\n").unwrap();
    let original_live = live(&memory);
    let result = VnncompResult::Unsat(receipt);
    assert!(write_results(&file, &result).is_err());
    assert!(!file.with_extension("result.tmp").exists());
    assert_eq!(
        fs::read_to_string(&marker).unwrap(),
        "existing destination\n"
    );
    assert_eq!(live(&memory), original_live);
    drop(result);
    assert_eq!(live(&memory), 0);
}
