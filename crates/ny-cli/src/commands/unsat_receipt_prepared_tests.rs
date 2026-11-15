// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! These controls construct actual original model/property files and invoke
//! the real source issuer. No supplied token, proof arrays or verdict JSON can
//! create the prepared receipt authority. Writer integration is tested by its
//! own production controls; this module checks the retained receipt seam.

use super::*;
use ny_onnx::onnx_proto::{
    tensor_shape_proto::{dimension::Value, Dimension},
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::{
    certify::{
        residency::{CancellationToken, MemoryAuthority},
        work::{WorkBudget, WorkLimits, WorkReserve},
    },
    margin_row::{PreparedRootProofOptions, RootPrecisionRuntimePolicy},
};
use prost::Message;
use std::{path::PathBuf, time::Duration};

const CAP: usize = 32 << 20;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(deadline()),
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

fn original_model(positive_target: usize) -> ModelProto {
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
    let mut bias = [0.0_f32; 3];
    bias[positive_target] = 2.0;
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            input: vec![info("input", &[1, 1, 1, 1])],
            output: vec![info("output", &[1, 3])],
            initializer: vec![
                tensor("wc", &[1, 1, 1, 1], &[1.0]),
                tensor("bc", &[1], &[0.0]),
                tensor("w1", &[2, 1], &[1.0, -1.0]),
                tensor("b1", &[2], &[0.0, 0.0]),
                tensor("w2", &[3, 2], &[0.0; 6]),
                tensor("b2", &[3], &bias),
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

fn original_property(target: usize) -> String {
    let classes: Vec<_> = (0..3).filter(|&c| c != target).rev().collect();
    format!(
        "(declare-const X_0 Real)\n\
         (declare-const Y_0 Real)\n\
         (declare-const Y_1 Real)\n\
         (declare-const Y_2 Real)\n\
         (assert (>= X_0 -1))\n\
         (assert (<= X_0 1))\n\
         (assert (or (and (>= Y_{} Y_{target})) (< Y_{target} Y_{})))",
        classes[0], classes[1]
    )
}

fn files(target: usize) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("original.onnx");
    let property = dir.path().join("original.vnnlib");
    std::fs::write(&model, original_model(1).encode_to_vec()).unwrap();
    std::fs::write(&property, original_property(target)).unwrap();
    (dir, model, property)
}

fn issue(
    budget: &WorkBudget,
) -> (
    tempfile::TempDir,
    PathBuf,
    PathBuf,
    OriginalPreparedRootPublication,
) {
    let (dir, model, property) = files(1);
    let authority = OriginalPreparedRootPublication::prove_original_files(
        &model,
        &property,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        budget,
        deadline(),
    )
    .expect("actual original source and complete prepared producer");
    (dir, model, property, authority)
}

fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}

#[test]
fn receipt_original_prepared_summary_binds_complete_order_target_and_distinct_source() {
    let (budget, memory, _) = resources();
    let (_dir, _, _, authority) = issue(&budget);
    let model = *authority.model_sha256();
    let property = *authority.property_sha256();
    let receipt = UnsatReceipt::from_original_prepared(authority, Some(deadline())).unwrap();
    receipt.checkpoint_owner(None).unwrap();
    let (classes, margin, source) = receipt.published_summary();
    assert_eq!(classes, [2, 0]);
    assert_eq!(classes.len(), 3 - 1, "every non-target original class");
    assert_eq!(receipt.original_prepared_target(), Some(1));
    assert!(margin.is_finite() && margin > 1.99);
    assert_eq!(source, PublishedUnsatSource::OriginalPrepared);
    assert_eq!(source.token(), "original_prepared_root");
    assert_eq!(source.legacy(), None);
    assert!(receipt.has_retained_owner());
    receipt
        .validate_original_prepared_asset_hashes(model, property, None)
        .unwrap();
    #[cfg(feature = "mip")]
    assert!(receipt.hull_authority().is_none());
    drop(receipt);
    assert_eq!(live(&memory), 0);
}

#[test]
fn receipt_original_prepared_source_snapshots_cannot_be_rebound_by_same_shape_files() {
    use sha2::{Digest, Sha256};

    let (budget, memory, _) = resources();
    let (_dir, model_path, property_path, authority) = issue(&budget);
    let model = *authority.model_sha256();
    let property = *authority.property_sha256();
    let receipt = UnsatReceipt::from_original_prepared(authority, None).unwrap();
    std::fs::write(&model_path, original_model(0).encode_to_vec()).unwrap();
    std::fs::write(&property_path, original_property(0)).unwrap();
    let foreign_model: [u8; 32] = Sha256::digest(std::fs::read(model_path).unwrap()).into();
    let foreign_property: [u8; 32] = Sha256::digest(std::fs::read(property_path).unwrap()).into();
    assert_ne!(foreign_model, model);
    assert_ne!(foreign_property, property);
    for (m, p) in [(foreign_model, property), (model, foreign_property)] {
        assert!(receipt
            .validate_original_prepared_asset_hashes(m, p, None)
            .is_err());
    }
    receipt
        .validate_original_prepared_asset_hashes(model, property, None)
        .unwrap();
    drop(receipt);
    assert_eq!(live(&memory), 0);
}

#[test]
fn receipt_original_prepared_clones_and_writer_claims_release_on_their_actual_last_owners() {
    let (budget, memory, _) = resources();
    let (_dir, _, _, authority) = issue(&budget);
    let receipt = UnsatReceipt::from_original_prepared(authority, None).unwrap();
    let shared = receipt.clone();
    let source = shared.original_prepared_authority().unwrap();
    let before = budget.receipt().unwrap();
    let claim = receipt
        .reserve_owner_publication(4096, None)
        .unwrap()
        .unwrap();
    assert!(claim.shares_ledger(&budget));
    assert_eq!(claim.bytes(), 4096);
    let after = budget.receipt().unwrap();
    assert_eq!(
        after.live_persistent_bytes,
        before.live_persistent_bytes + 4096
    );
    assert_eq!(after.proposals_reserved, before.proposals_reserved);
    drop(receipt);
    shared.checkpoint_owner(None).unwrap();
    drop(shared);
    source.checkpoint(None).unwrap();
    assert!(live(&memory) > claim.bytes());
    drop(source);
    assert_eq!(live(&memory), claim.bytes());
    drop(claim);
    assert_eq!(live(&memory), 0);
}

#[test]
fn receipt_original_prepared_cancellation_before_receipt_cannot_mint_authority() {
    let (budget, memory, cancel) = resources();
    let (_dir, _, _, authority) = issue(&budget);
    let retained = authority.clone();
    let before = live(&memory);
    cancel.cancel();
    assert!(UnsatReceipt::from_original_prepared(authority, None).is_err());
    assert_eq!(live(&memory), before);
    assert!(retained.checkpoint(None).is_err());
    drop(retained);
    assert_eq!(live(&memory), 0);
}

#[test]
fn receipt_original_prepared_cancelled_writer_refuses_new_storage_without_refresh() {
    let (budget, memory, cancel) = resources();
    let (_dir, _, _, authority) = issue(&budget);
    let receipt = UnsatReceipt::from_original_prepared(authority, None).unwrap();
    let before = budget.receipt().unwrap();
    cancel.cancel();
    assert!(receipt.checkpoint_owner(None).is_err());
    assert!(receipt.reserve_owner_publication(4096, None).is_err());
    let after = budget.receipt().unwrap();
    assert_eq!(after.admitted, before.admitted);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.proposals_reserved, before.proposals_reserved);
    drop(receipt);
    assert_eq!(live(&memory), 0);
}

#[test]
fn receipt_original_prepared_earlier_requested_clock_refuses_before_writer_admission() {
    let (budget, memory, _) = resources();
    let (_dir, _, _, authority) = issue(&budget);
    let retained = authority.clone();
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("test clock supports an expired deadline");
    assert!(UnsatReceipt::from_original_prepared(authority, Some(expired)).is_err());
    let receipt = UnsatReceipt::from_original_prepared(retained, None).unwrap();
    let before = budget.receipt().unwrap();
    assert!(receipt.checkpoint_owner(Some(expired)).is_err());
    assert!(receipt
        .reserve_owner_publication(4096, Some(expired))
        .is_err());
    let after = budget.receipt().unwrap();
    assert_eq!(after.admitted, before.admitted);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    receipt.checkpoint_owner(None).unwrap();
    drop(receipt);
    assert_eq!(live(&memory), 0);
}

#[test]
fn receipt_original_prepared_successful_earlier_handoff_clock_survives_clone_and_source_accessor() {
    let (budget, memory, _) = resources();
    let (_dir, _, _, authority) = issue(&budget);
    let earlier = Instant::now() + Duration::from_millis(200);
    let receipt = UnsatReceipt::from_original_prepared(authority, Some(earlier)).unwrap();
    let clone = receipt.clone();
    let source = clone.original_prepared_authority().unwrap();
    let handed_again = UnsatReceipt::from_original_prepared(source.clone(), None).unwrap();
    // The successful writer admission itself must still use the same original
    // ledger and receipt's stored earlier handoff clock.
    let claim = receipt
        .reserve_owner_publication(4096, None)
        .unwrap()
        .unwrap();
    assert!(claim.shares_ledger(&budget));
    std::thread::sleep(
        earlier.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
    );
    for retained in [&receipt, &clone, &handed_again] {
        assert!(retained.checkpoint_owner(None).is_err());
        assert!(retained.checkpoint_owner(Some(deadline())).is_err());
        assert!(retained.reserve_owner_publication(1, None).is_err());
    }
    assert!(source.checkpoint(None).is_err());
    assert!(UnsatReceipt::from_original_prepared(source, None).is_err());
    drop(claim);
    drop(receipt);
    drop(clone);
    drop(handed_again);
    assert_eq!(live(&memory), 0);
}

#[test]
fn receipt_original_prepared_publication_cap_and_overflow_preserve_the_prior_proof() {
    let (budget, memory, _) = resources();
    let (_dir, _, _, authority) = issue(&budget);
    let receipt = UnsatReceipt::from_original_prepared(authority, None).unwrap();
    let original_live = budget.receipt().unwrap().live_persistent_bytes;
    let mut blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: CAP - original_live - 1024,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    blocker.mark_materialized().unwrap();
    let blocker = blocker.commit_persistent().unwrap();
    let before = live(&memory);
    assert!(receipt.reserve_owner_publication(4096, None).is_err());
    assert!(receipt.reserve_owner_publication(usize::MAX, None).is_err());
    assert_eq!(live(&memory), before);
    receipt.checkpoint_owner(None).unwrap();
    drop(blocker);
    let claim = receipt
        .reserve_owner_publication(4096, None)
        .unwrap()
        .unwrap();
    assert!(claim.shares_ledger(&budget));
    drop(claim);
    receipt.checkpoint_owner(None).unwrap();
    drop(receipt);
    assert_eq!(live(&memory), 0);
}

#[test]
fn receipt_original_prepared_unsafe_original_objective_cannot_issue_a_receipt() {
    let (budget, memory, _) = resources();
    let (_dir, model, property) = files(0);
    // Actual final output is exactly [0,2,0] for every input, so target zero's
    // complete unsafe objective is satisfiable. No diagnostic target can
    // replace the property before the private original receipt constructor.
    let issued = OriginalPreparedRootPublication::prove_original_files(
        &model,
        &property,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        &budget,
        deadline(),
    );
    assert!(issued.is_err());
    assert_eq!(live(&memory), 0);
}

#[test]
fn receipt_original_prepared_token_variant_has_no_source_upgrade_or_new_owner_claim() {
    let receipt = UnsatReceipt::for_test();
    assert!(!receipt.has_retained_owner());
    assert!(receipt.original_prepared_authority().is_none());
    assert_eq!(receipt.original_prepared_target(), None);
    assert!(receipt
        .reserve_owner_publication(4096, None)
        .unwrap()
        .is_none());
    receipt.checkpoint_owner(None).unwrap();
    assert!(receipt
        .validate_original_prepared_asset_hashes([0; 32], [0; 32], None)
        .is_err());
    assert!(matches!(
        receipt.published_summary().2,
        PublishedUnsatSource::Spine(_)
    ));
}

#[cfg(feature = "mip")]
#[test]
fn receipt_original_prepared_generic_owner_api_preserves_genuine_hull_authority() {
    let (captured, resources) =
        crate::commands::beta_crown::hull_publication::tests::ordinary_dispatch(0.75, false, false);
    let (_, authority) = captured.into_parts();
    let receipt = UnsatReceipt::from_original_hull(authority.unwrap()).unwrap();
    assert!(receipt.has_retained_owner());
    assert!(receipt.original_prepared_authority().is_none());
    assert_eq!(receipt.original_prepared_target(), None);
    assert_eq!(
        receipt.published_summary().2,
        PublishedUnsatSource::OriginalHull
    );
    receipt.checkpoint_owner(None).unwrap();
    let claim = receipt
        .reserve_owner_publication(4096, None)
        .unwrap()
        .unwrap();
    assert!(claim.shares_ledger(resources.budget()));
    assert_eq!(claim.bytes(), 4096);
    assert!(receipt.hull_authority().is_some());
    resources.cancel();
    assert!(receipt.checkpoint_owner(None).is_err());
    assert!(receipt.reserve_owner_publication(4096, None).is_err());
    drop(claim);
    drop(receipt);
    assert_eq!(
        resources.budget().receipt().unwrap().live_persistent_bytes,
        4096
    );
}
