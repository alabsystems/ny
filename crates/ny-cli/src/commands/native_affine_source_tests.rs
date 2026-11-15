// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Actual source bytes, native affine extraction, and original-file authority.
//! The single-row domain specializes the original batch-separable graph; no
//! test rewrites a model, changes its threat radii, or imports a numerical proof.

use super::super::margin_row_bab::{
    extract_original_twin_spec_named_bytes, extract_twin_spec_named_bytes,
};
use super::super::original_source_census::compile_staged_original_source;
use super::super::unsat_receipt::UnsatReceipt;
use super::*;
use ny_onnx::onnx_proto::{
    tensor_shape_proto::{dimension::Value, Dimension},
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto,
    TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::{
    certify::{
        residency::{CancellationToken, MemoryAuthority},
        work::{WorkLimits, WorkReserve},
    },
    margin_row::TwinOpSpec,
};
use prost::Message;
use std::time::Duration;

const CAP: usize = 32 << 20;
fn clock() -> Instant {
    Instant::now() + Duration::from_secs(20)
}
fn resources(cap: usize) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(cap);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(clock()),
            scratch_bytes: cap,
            persistent_bytes: cap,
            max_work: 1 << 30,
            max_proposals: 1,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(cap).unwrap();
    (budget, memory, cancel)
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}
fn info(name: &str, width: i64, symbolic: bool) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        Dimension {
                            value: Some(if symbolic {
                                Value::DimParam(format!("{name}_batch"))
                            } else {
                                Value::DimValue(1)
                            }),
                        },
                        Dimension {
                            value: Some(Value::DimValue(width)),
                        },
                    ],
                }),
            }),
        }),
    }
}
fn tensor(name: &str, dims: &[i64], values: &[f32], raw: bool) -> TensorProto {
    let mut t = TensorProto {
        name: name.into(),
        dims: dims.to_vec(),
        data_type: 1,
        ..Default::default()
    };
    if raw {
        t.raw_data = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
    } else {
        t.float_data = values.to_vec();
    }
    t
}
fn node(op: &str, input: &[&str], output: &str) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input: input.iter().map(|s| (*s).into()).collect(),
        output: vec![output.into()],
        ..Default::default()
    }
}
fn model(positive: bool, raw: bool, symbolic: bool) -> ModelProto {
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
            input: vec![info("embeddings", 2, symbolic)],
            output: vec![info("predictions", 3, symbolic)],
            initializer: vec![
                tensor("w1", &[2, 3], &[1.0, -2.0, 0.5, 0.0, 0.0, 1.0], raw),
                tensor("b1", &[3], &[0.5, 0.5, 0.25], raw),
                tensor(
                    "w2",
                    &[3, 3],
                    &[0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    raw,
                ),
                tensor(
                    "b2",
                    &[3],
                    &[0.0, if positive { 4.0 } else { -4.0 }, 0.0],
                    raw,
                ),
            ],
            node: vec![
                node("MatMul", &["embeddings", "w1"], "m1"),
                node("Add", &["m1", "b1"], "pre"),
                node("Relu", &["pre"], "hidden"),
                node("MatMul", &["hidden", "w2"], "m2"),
                node("Add", &["m2", "b2"], "predictions"),
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn property() -> String {
    "(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n(declare-const Y_2 Real)\n(assert (>= X_0 -1))\n(assert (<= X_0 1))\n(assert (>= X_1 -1))\n(assert (<= X_1 1))\n(assert (or (and (>= Y_2 Y_1)) (< Y_1 Y_0)))".into()
}
fn files(
    model: &ModelProto,
    text: &str,
) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let onnx = dir.path().join("original.onnx");
    let vnn = dir.path().join("original.vnnlib");
    std::fs::write(&onnx, model.encode_to_vec()).unwrap();
    std::fs::write(&vnn, text).unwrap();
    (dir, onnx, vnn)
}
fn prepare(onnx: &Path, vnn: &Path, budget: &WorkBudget) -> Result<OriginalPreparedRootSession> {
    OriginalPreparedRootSession::prepare_original_files_staged(
        onnx,
        vnn,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        budget,
        clock(),
    )
}
fn proven(outcome: OriginalPreparedRootOutcome) -> OriginalPreparedRootPublication {
    match outcome {
        OriginalPreparedRootOutcome::Proven(p) => p,
        OriginalPreparedRootOutcome::Open(_) => {
            panic!("actual native positive original must close")
        }
    }
}
fn exact_target_margin_quarters(x0: i64, positive: bool) -> i64 {
    // Independent original MatMul/Add/ReLU arithmetic on quarter-grid inputs.
    (if positive { 16 } else { -16 }) + (x0 + 2).max(0) - (-2 * x0 + 2).max(0)
}

#[test]
fn native_affine_actual_original_positive_single_row_reaches_g2_g3_and_private_receipt() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources(CAP);
    let (_dir, onnx, vnn) = files(&model(true, true, true), &property());
    let session = prepare(&onnx, &vnn, &budget).unwrap();
    assert_eq!(session.0.property.input.len(), 2);
    assert_eq!(session.classes(), [2, 0]);
    let owner = proven(session.try_prove_root(None).unwrap());
    assert_eq!(owner.target(), 1);
    assert!(owner.min_margin() > 0.0);
    for x0 in -4..=4 {
        for _x1 in -4..=4 {
            assert!(owner.min_margin() <= exact_target_margin_quarters(x0, true) as f64 / 4.0);
        }
    }
    assert!(owner.0.proof.tree_receipt().is_none());
    let receipt = UnsatReceipt::from_original_prepared(owner, None).unwrap();
    receipt.checkpoint_owner(None).unwrap();
    drop(session);
    assert!(live(&memory) > 0);
    let last = receipt.original_prepared_authority().unwrap();
    drop(receipt);
    last.checkpoint(None).unwrap();
    drop(last);
    assert_eq!(live(&memory), 0);
}

#[test]
fn native_affine_real_negative_original_keeps_open_session_without_verdict() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources(CAP);
    let (_dir, onnx, vnn) = files(&model(false, false, true), &property());
    let session = prepare(&onnx, &vnn, &budget).unwrap();
    assert!(exact_target_margin_quarters(0, false) < 0);
    let OriginalPreparedRootOutcome::Open(open) = session.try_prove_root(None).unwrap() else {
        panic!("original unsafe point cannot certify")
    };
    assert!(open.attempt_receipt().ordinary_completed);
    drop(session);
    open.checkpoint(None).unwrap();
    assert!(live(&memory) > 0);
    drop(open);
    assert_eq!(live(&memory), 0);
}

#[test]
fn native_affine_original_orientation_signed_zero_subnormal_and_raw_typed_bits_are_exact() {
    let _env = ny_test_utils::env::lock_env_shared();
    for raw in [false, true] {
        let (budget, memory, _) = resources(CAP);
        let mut m = model(true, raw, false);
        let g = m.graph.as_mut().unwrap();
        g.initializer[0] = tensor(
            "w1",
            &[2, 3],
            &[1.0, f32::from_bits(1), -0.0, 0.5, -f32::from_bits(1), 2.0],
            raw,
        );
        g.initializer[1] = tensor("b1", &[1, 3], &[-0.0, f32::from_bits(1), 0.25], raw);
        let bytes = m.encode_to_vec();
        let stage = compile_staged_original_source(&bytes, &budget, clock()).unwrap();
        // A separately prepaid diagnostic oracle inspects the exact extracted
        // arrays; it carries no proof authority and is dropped before its lease.
        let oracle = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    scratch_bytes: 1 << 20,
                    work: 100_000_000,
                    ..Default::default()
                },
            )
            .unwrap();
        let (spec, names) =
            extract_original_twin_spec_named_bytes(&bytes, &budget, clock()).unwrap();
        assert!(extract_twin_spec_named_bytes(&bytes).is_none());
        let TwinOpSpec::Gemm {
            weight,
            bias,
            bias_err,
            weight_rel_err,
            shape,
            input,
        } = &spec.ops[0]
        else {
            panic!("actual affine source is dense")
        };
        assert_eq!((*input, *shape), (0, (3, 2)));
        assert_eq!(
            weight.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![
                1.0f64.to_bits(),
                0.5f64.to_bits(),
                0x36a0_0000_0000_0000,
                0xb6a0_0000_0000_0000,
                0x8000_0000_0000_0000,
                2.0f64.to_bits()
            ]
        );
        assert_eq!(
            bias.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![
                0x8000_0000_0000_0000,
                0x36a0_0000_0000_0000,
                0.25f64.to_bits()
            ]
        );
        assert_eq!(bias_err.len(), 0);
        assert_eq!(*weight_rel_err, 0.0);
        assert_eq!(names, vec![(1, "pre".into(), 3)]);
        assert_eq!(spec.ops.len(), 3);
        drop(spec);
        drop(names);
        drop(oracle);
        drop(stage);
        assert_eq!(live(&memory), 0);
    }
}

#[test]
fn native_affine_commuted_actual_row_bias_keeps_original_graph_semantics() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources(CAP);
    let mut m = model(true, true, true);
    for i in [1, 4] {
        m.graph.as_mut().unwrap().node[i].input.swap(0, 1);
    }
    let (_dir, onnx, vnn) = files(&m, &property());
    let session = prepare(&onnx, &vnn, &budget).unwrap();
    let owner = proven(session.try_prove_root(None).unwrap());
    assert!(owner.min_margin() > 0.0);
    drop(owner);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn native_affine_multirow_feature_symbols_broadcast_and_dynamic_or_transposed_weights_refuse() {
    let _env = ny_test_utils::env::lock_env_shared();
    for case in 0..7 {
        let (budget, memory, _) = resources(CAP);
        let mut m = model(true, false, true);
        let g = m.graph.as_mut().unwrap();
        match case {
            0 => {
                g.input[0]
                    .r#type
                    .as_mut()
                    .unwrap()
                    .tensor_type
                    .as_mut()
                    .unwrap()
                    .shape
                    .as_mut()
                    .unwrap()
                    .dim[0]
                    .value = Some(Value::DimValue(2))
            }
            1 => {
                g.input[0]
                    .r#type
                    .as_mut()
                    .unwrap()
                    .tensor_type
                    .as_mut()
                    .unwrap()
                    .shape
                    .as_mut()
                    .unwrap()
                    .dim[1]
                    .value = Some(Value::DimParam("features".into()))
            }
            2 => g.initializer[1] = tensor("b1", &[1], &[0.5], false),
            3 => g.initializer[1] = tensor("b1", &[1, 1, 3], &[0.5, 0.5, 0.25], false),
            4 => g.node[0].input[1] = "embeddings".into(),
            5 => g.node[0].input.swap(0, 1),
            6 => g.initializer[0] = tensor("w1", &[3, 2], &[1.0, -2.0, 0.5, 0.0, 0.0, 1.0], false),
            _ => unreachable!(),
        }
        assert!(compile_staged_original_source(&m.encode_to_vec(), &budget, clock()).is_err());
        assert_eq!(live(&memory), 0);
    }
}

#[test]
fn native_affine_duplicate_standard_opsets_and_executed_custom_domain_refuse() {
    let _env = ny_test_utils::env::lock_env_shared();
    for case in 0..3 {
        let (budget, memory, _) = resources(CAP);
        let mut m = model(true, true, true);
        match case {
            0 => m.opset_import.push(OperatorSetIdProto {
                domain: "ai.onnx".into(),
                version: 13,
            }),
            1 => m.opset_import.push(OperatorSetIdProto {
                domain: "ai.onnx.ml".into(),
                version: 2,
            }),
            2 => m.graph.as_mut().unwrap().node[0].domain = "ai.onnx.ml".into(),
            _ => unreachable!(),
        }
        assert!(compile_staged_original_source(&m.encode_to_vec(), &budget, clock()).is_err());
        assert_eq!(live(&memory), 0);
    }
}

#[test]
fn native_affine_earlier_exported_tensor_unused_tail_and_multiple_matmul_uses_refuse() {
    let _env = ny_test_utils::env::lock_env_shared();
    for case in 0..3 {
        let (budget, memory, _) = resources(CAP);
        let mut m = model(true, false, true);
        let g = m.graph.as_mut().unwrap();
        match case {
            0 => g.output[0].name = "m1".into(),
            1 => g.output[0].name = "pre".into(),
            2 => g.node.insert(2, node("Relu", &["m1"], "unused")),
            _ => unreachable!(),
        }
        assert!(compile_staged_original_source(&m.encode_to_vec(), &budget, clock()).is_err());
        assert_eq!(live(&memory), 0);
    }
}

#[test]
fn native_affine_complete_original_feature_count_binds_single_row_property_domain() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources(CAP);
    let text = property()
        .replace(
            "(declare-const Y_0",
            "(declare-const X_2 Real)\n(declare-const Y_0",
        )
        .replace(
            "(assert (or",
            "(assert (>= X_2 -1))\n(assert (<= X_2 1))\n(assert (or",
        );
    let (_dir, onnx, vnn) = files(&model(true, true, true), &text);
    assert!(prepare(&onnx, &vnn, &budget).is_err());
    assert_eq!(live(&memory), 0);
}

#[test]
fn native_affine_original_cancel_expiry_cap_and_last_successful_clock_remain_binding() {
    let _env = ny_test_utils::env::lock_env_shared();
    let m = model(true, true, true);
    let bytes = m.encode_to_vec();
    let (small, memory, _) = resources(16 << 10);
    assert!(compile_staged_original_source(&bytes, &small, clock()).is_err());
    assert_eq!(live(&memory), 0);
    let (budget, memory, cancel) = resources(CAP);
    cancel.cancel();
    let before = budget.receipt().unwrap();
    assert!(compile_staged_original_source(&bytes, &budget, clock()).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(live(&memory), 0);
    let (budget, memory, _) = resources(CAP);
    assert!(compile_staged_original_source(
        &bytes,
        &budget,
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
    )
    .is_err());
    assert_eq!(live(&memory), 0);
    let (_dir, onnx, vnn) = files(&m, &property());
    let session = prepare(&onnx, &vnn, &budget).unwrap();
    let earlier = Instant::now() + Duration::from_secs(5);
    let session = session.restricted(Some(earlier)).unwrap();
    let owner = proven(session.try_prove_root(Some(clock())).unwrap());
    assert!(owner.deadline() <= earlier);
    assert!(owner.clone().restricted(None).unwrap().deadline() <= earlier);
    drop(owner);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn native_affine_original_files_deleted_after_preparation_keep_exact_source_for_receipt() {
    let _env = ny_test_utils::env::lock_env_shared();
    let (budget, memory, _) = resources(CAP);
    let (_dir, onnx, vnn) = files(&model(true, true, true), &property());
    let session = prepare(&onnx, &vnn, &budget).unwrap();
    let model_hash = *session.model_sha256();
    let property_hash = *session.property_sha256();
    std::fs::remove_file(onnx).unwrap();
    std::fs::remove_file(vnn).unwrap();
    let owner = proven(session.try_prove_root(None).unwrap());
    owner
        .validate_asset_hashes(model_hash, property_hash, None)
        .unwrap();
    drop(session);
    let receipt = UnsatReceipt::from_original_prepared(owner, None).unwrap();
    receipt.checkpoint_owner(None).unwrap();
    drop(receipt);
    assert_eq!(live(&memory), 0);
}

#[test]
#[cfg(feature = "external-models")]
fn native_affine_all26_authentic_safenlp_originals_reach_owned_positive_or_open_source_outcomes() {
    let _env = ny_test_utils::env::lock_env_shared();
    let archive =
        Path::new("<home>/ny-experiments/high-impact/2026-10-01/original-hull-full-cli-v6");
    let mut requests = std::fs::read_dir(archive)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().ends_with("-root"))
        .collect::<Vec<_>>();
    requests.sort();
    assert_eq!(requests.len(), 26);
    let mut covered = 0usize;
    for request in requests {
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(request.join("request.json")).unwrap()).unwrap();
        let row = &data["row"];
        let model = Path::new(row["model"]["path"].as_str().unwrap());
        let property = Path::new(row["property"]["path"].as_str().unwrap());
        assert_eq!(row["official_timeout_seconds"].as_u64(), Some(20));
        let (budget, memory, _) = resources(CAP);
        let session = prepare(model, property, &budget).unwrap();
        assert_eq!(session.0.property.input.len(), 30);
        assert_eq!(session.0.session.n_out(), 2);
        let model_hex =
            session
                .model_sha256()
                .iter()
                .fold(String::with_capacity(64), |mut text, b| {
                    use std::fmt::Write as _;
                    write!(&mut text, "{b:02x}").expect("writing to String is infallible");
                    text
                });
        let property_hex =
            session
                .property_sha256()
                .iter()
                .fold(String::with_capacity(64), |mut text, b| {
                    use std::fmt::Write as _;
                    write!(&mut text, "{b:02x}").expect("writing to String is infallible");
                    text
                });
        assert_eq!(model_hex, row["model"]["sha256"].as_str().unwrap());
        assert_eq!(property_hex, row["property"]["sha256"].as_str().unwrap());
        match session.try_prove_root(None).unwrap() {
            OriginalPreparedRootOutcome::Proven(owner) => {
                let receipt = UnsatReceipt::from_original_prepared(owner, None).unwrap();
                receipt.checkpoint_owner(None).unwrap();
                drop(receipt);
            }
            OriginalPreparedRootOutcome::Open(open) => {
                assert_eq!(open.classes().len(), 1);
                open.checkpoint(None).unwrap();
                drop(open);
            }
        }
        drop(session);
        assert_eq!(live(&memory), 0);
        covered += 1;
    }
    assert_eq!(covered, 26); // Source/functionality coverage; no extra solve or timing claim.
}
