// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Genuine original bytes -> actual extractor -> paid prepared producer.
//! Property negatives cannot attach numeric arrays or a fabricated token.

use super::*;
use ny_onnx::onnx_proto::{
    tensor_shape_proto::{dimension::Value, Dimension},
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::certify::{
    residency::{CancellationToken, MemoryAuthority},
    work::{WorkLimits, WorkReserve},
};
use prost::Message;
use std::time::Duration;

fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources(cap: usize) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(cap);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(at()),
            scratch_bytes: cap,
            persistent_bytes: cap,
            max_work: 100_000_000_000,
            max_proposals: 1,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(cap).unwrap();
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
                        .map(|&d| Dimension {
                            value: Some(Value::DimValue(d)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}
fn attr_i(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        r#type: 2,
        i: Some(value),
        ..Default::default()
    }
}
fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: vec![output.into()],
        ..Default::default()
    }
}
fn model(target: usize, positive: bool) -> ModelProto {
    let mut first = node("Gemm", &["flat", "w1", "b1"], "headpre");
    first.attribute.push(attr_i("transB", 1));
    let mut final_node = node("Gemm", &["head", "w2", "b2"], "output");
    final_node.attribute.push(attr_i("transB", 1));
    let mut bias = [0f32; 3];
    bias[target] = if positive { 2.0 } else { -2.0 };
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
                node("Relu", &["headpre"], "head"),
                final_node,
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn property(target: usize, lower: &str, upper: &str) -> String {
    let classes: Vec<_> = (0..3).filter(|&c| c != target).rev().collect();
    format!("(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n(declare-const Y_2 Real)\n(assert (>= X_0 {lower}))\n(assert (<= X_0 {upper}))\n(assert (or (and (>= Y_{} Y_{target})) (<= Y_{target} Y_{})))",classes[0],classes[1])
}
fn files(
    model: &ModelProto,
    property: &str,
) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let onnx = dir.path().join("original.onnx");
    let vnn = dir.path().join("original.vnnlib");
    std::fs::write(&onnx, model.encode_to_vec()).unwrap();
    std::fs::write(&vnn, property).unwrap();
    (dir, onnx, vnn)
}
fn prove(onnx: &Path, vnn: &Path, budget: &WorkBudget) -> Result<OriginalPreparedRootPublication> {
    OriginalPreparedRootPublication::prove_original_files(
        onnx,
        vnn,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        budget,
        at(),
    )
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}

// Independent integer cross multiplication for the small positive decimal
// endpoints in this control. No production rounding helper is used.
fn compare_decimal(value: f64, numerator: i128, denominator: i128) -> std::cmp::Ordering {
    assert!(value.is_finite() && value > 0.0);
    let bits = value.to_bits();
    let mantissa = i128::from((1u64 << 52) | (bits & ((1u64 << 52) - 1)));
    let exponent = i32::try_from((bits >> 52) & 0x7ff).unwrap() - 1023 - 52;
    assert!((-100..=0).contains(&exponent));
    (mantissa * denominator).cmp(&(numerator * (1i128 << (-exponent))))
}

#[test]
fn prepared_original_files_actual_producer_binds_nonzero_target_order_and_exact_decimal_box() {
    let (budget, memory, _) = resources(32 << 20);
    let (_dir, onnx, vnn) = files(&model(1, true), &property(1, "0.1", "0.3"));
    let owner = prove(&onnx, &vnn, &budget)
        .expect("actual source producer must authorize the positive original model");
    assert_eq!(owner.target(), 1);
    assert_eq!(owner.classes(), [2, 0]);
    assert!(owner.min_margin() > 1.99);
    assert!(owner.budget().shares_ledger(&budget));
    assert_ne!(
        compare_decimal(owner.0.proof.declared_lower()[0], 1, 10),
        std::cmp::Ordering::Greater
    );
    assert_ne!(
        compare_decimal(owner.0.proof.declared_upper()[0], 3, 10),
        std::cmp::Ordering::Less
    );
    assert_eq!(
        owner.0.proof.declared_lower()[0].to_bits(),
        owner.0.property.input.lower()[0].to_bits()
    );
    owner.checkpoint(None).unwrap();
    assert!(live(&memory) > 0);
    drop(owner);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_original_files_source_objective_cannot_be_replaced_by_expected_target() {
    let (budget, memory, _) = resources(32 << 20);
    let (_dir, onnx, vnn) = files(&model(1, true), &property(0, "-1", "1"));
    assert!(
        prove(&onnx, &vnn, &budget).is_err(),
        "actual property target zero is unsafe despite positive class-one declaration"
    );
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_original_files_retain_exact_snapshots_and_refuse_foreign_capture_hashes() {
    let (budget, _, _) = resources(32 << 20);
    let (_dir, onnx, vnn) = files(&model(1, true), &property(1, "-1", "1"));
    let owner = prove(&onnx, &vnn, &budget).unwrap();
    let old_model = *owner.model_sha256();
    let old_property = *owner.property_sha256();
    std::fs::write(&onnx, model(0, true).encode_to_vec()).unwrap();
    std::fs::write(&vnn, property(0, "-1", "1")).unwrap();
    owner
        .validate_asset_hashes(old_model, old_property, None)
        .unwrap();
    let now_model = hash(&std::fs::read(&onnx).unwrap(), &budget, at()).unwrap();
    let now_property = hash(&std::fs::read(&vnn).unwrap(), &budget, at()).unwrap();
    assert!(owner
        .validate_asset_hashes(now_model, old_property, None)
        .is_err());
    assert!(owner
        .validate_asset_hashes(old_model, now_property, None)
        .is_err());
    assert_eq!(owner.0.model_bytes, model(1, true).encode_to_vec());
}

#[test]
fn prepared_original_files_cancel_deadline_and_independent_last_owner_remain_binding() {
    let (budget, memory, cancel) = resources(32 << 20);
    let (_dir, onnx, vnn) = files(&model(2, true), &property(2, "-1", "1"));
    let owner = prove(&onnx, &vnn, &budget).unwrap();
    let staged = owner.clone();
    let held = live(&memory);
    drop(owner);
    assert_eq!(live(&memory), held);
    staged.checkpoint(None).unwrap();
    assert!(staged.checkpoint(Some(Instant::now())).is_err());
    cancel.cancel();
    assert!(staged.checkpoint(None).is_err());
    drop(staged);
    assert_eq!(live(&memory), 0);
    assert!(prove(&onnx, &vnn, &budget).is_err());
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_original_files_source_admission_refuses_before_producer_without_refresh() {
    let (budget, memory, _) = resources(1024);
    let (_dir, onnx, vnn) = files(&model(1, true), &property(1, "-1", "1"));
    let before = budget.receipt().unwrap();
    assert!(prove(&onnx, &vnn, &budget).is_err());
    let after = budget.receipt().unwrap();
    assert_eq!(after.admitted, before.admitted);
    assert!(after.refused > before.refused);
    assert_eq!(after.proposals_reserved, 0);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_original_property_refuses_extra_semantics_coupled_rows_and_missing_classes() {
    let (budget, _, _) = resources(32 << 20);
    let good = property(1, "-1", "1");
    let (_spec, _input) = parse_vnnlib_with_certified_input_box(&good).unwrap();
    assert!(authenticate_raw_objective(&good, &budget, at()).is_ok());
    for bad in [
        good.replace("(and (>= Y_2 Y_1))", "(and (>= Y_2 Y_1) (>= Y_0 Y_1))"),
        good.replace("(<= Y_1 Y_0)", "(>= (+ Y_0 0) Y_1)"),
        good.replace("(<= Y_1 Y_0)", "true"),
        good.replace("(<= Y_1 Y_0)", "(>= Y_2 Y_1)"),
        good.replace("(<= Y_1 Y_0)", "(<= Y_0 Y_1)"),
        format!("{good}\n(assert (>= Y_1 100))"),
        good.replace("(>= X_0 -1)", "(>= (+ X_0 Y_1) -1)"),
        good.replace("(declare-const Y_1 Real)", "(declare-const Y_01 Real)"),
    ] {
        assert!(
            authenticate_raw_objective(&bad, &budget, at()).is_err(),
            "accepted unsupported original formula: {bad}"
        );
    }
}

#[test]
fn prepared_original_property_signed_strict_or_atoms_cross_check_production_order() {
    let (budget, _, _) = resources(32 << 20);
    let text = property(1, "-0.1", "0.3")
        .replace("(>= Y_2 Y_1)", "(> Y_2 Y_1)")
        .replace("(<= Y_1 Y_0)", "(< Y_1 Y_0)");
    let objective = authenticate_raw_objective(&text, &budget, at()).unwrap();
    let (spec, input) = parse_vnnlib_with_certified_input_box(&text).unwrap();
    cross_check(&objective, &spec, &input).unwrap();
    assert_eq!(objective.target, 1);
    assert_eq!(objective.classes, [2, 0]);
    assert_eq!(objective.atoms[0].relation, Relation::Gt);
    assert_eq!(objective.atoms[1].relation, Relation::Lt);
}

#[test]
fn prepared_original_source_refuses_same_width_earlier_output_and_unused_positive_tail() {
    let (budget, memory, _) = resources(32 << 20);
    let mut earlier = model(1, true);
    let graph = earlier.graph.as_mut().unwrap();
    // Both original exported head and appended positive head have width three.
    // The actual extraction accepts its ordinary twin declaration, so this
    // control reaches the concrete source-output authority gap.
    graph
        .initializer
        .iter_mut()
        .find(|t| t.name == "w1")
        .unwrap()
        .dims = vec![3, 1];
    graph
        .initializer
        .iter_mut()
        .find(|t| t.name == "w1")
        .unwrap()
        .float_data = vec![1.0, -1.0, 0.0];
    graph
        .initializer
        .iter_mut()
        .find(|t| t.name == "b1")
        .unwrap()
        .dims = vec![3];
    graph
        .initializer
        .iter_mut()
        .find(|t| t.name == "b1")
        .unwrap()
        .float_data = vec![0.0; 3];
    graph
        .initializer
        .iter_mut()
        .find(|t| t.name == "w2")
        .unwrap()
        .dims = vec![3, 3];
    graph
        .initializer
        .iter_mut()
        .find(|t| t.name == "w2")
        .unwrap()
        .float_data = vec![0.0; 9];
    graph.output[0].name = "headpre".into();
    // Independent exact original witness: x=1 -> Conv=1 -> trunk=1 ->
    // exported first head [1,-1,0]. The unused final head is [0,2,0].
    let original_property = ny_onnx::vnnlib::parse_vnnlib(&property(1, "-1", "1")).unwrap();
    assert!(
        super::super::vnncomp::property_violated_f64(&original_property, &[1.0], &[1.0, -1.0, 0.0]),
        "actual original exported objective is unsafe at an exact dyadic witness"
    );
    let bytes = earlier.encode_to_vec();
    let extracted = super::super::margin_row_bab::extract_twin_spec_named_bytes(&bytes)
        .expect("unused positive twin tail still extracts through unchanged legacy entry");
    assert_eq!(
        extracted.0.ops.last().and_then(|op| match op {
            ny_propagate::margin_row::TwinOpSpec::Gemm { shape, .. } => Some(shape.0),
            _ => None,
        }),
        Some(3)
    );
    assert!(
        super::super::prepared_root_source::authenticate_original_twin_surface(
            &bytes,
            &budget,
            at()
        )
        .is_err()
    );
    let (_dir, onnx, vnn) = files(&earlier, &property(1, "-1", "1"));
    assert!(prove(&onnx, &vnn, &budget).is_err());
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_original_source_refuses_batch_axis_flatten_axis_training_mode_and_custom_domain() {
    let (budget, _, _) = resources(32 << 20);
    let mut batch = model(1, true);
    batch.graph.as_mut().unwrap().input[0]
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
        .value = Some(Value::DimValue(2));
    let mut flatten = model(1, true);
    flatten.graph.as_mut().unwrap().node[2]
        .attribute
        .push(attr_i("axis", 0));
    let mut custom = model(1, true);
    custom.graph.as_mut().unwrap().node[0].domain = "foreign".into();
    let mut omitted = model(1, true);
    omitted.graph.as_mut().unwrap().input[0].name.clear();
    omitted.graph.as_mut().unwrap().node[0].input[0].clear();
    for model in [batch, flatten, custom, omitted] {
        assert!(
            super::super::prepared_root_source::authenticate_original_twin_surface(
                &model.encode_to_vec(),
                &budget,
                at()
            )
            .is_err()
        );
    }
    let mut bn = model(1, true);
    bn.opset_import[0].version = 14;
    let graph = bn.graph.as_mut().unwrap();
    graph.initializer.extend([
        tensor("scale", &[1], &[1.0]),
        tensor("shift", &[1], &[0.0]),
        tensor("mean", &[1], &[0.0]),
        tensor("var", &[1], &[1.0]),
    ]);
    let mut normalization = node(
        "BatchNormalization",
        &["conv", "scale", "shift", "mean", "var"],
        "bn",
    );
    normalization.attribute.push(attr_i("training_mode", 1));
    graph.node.insert(1, normalization);
    graph.node[2].input[0] = "bn".into();
    assert!(
        super::super::prepared_root_source::authenticate_original_twin_surface(
            &bn.encode_to_vec(),
            &budget,
            at()
        )
        .is_err()
    );
}

#[test]
#[cfg(feature = "external-vnncomp")]
fn prepared_original_source_held_bytes_and_path_extraction_match_actual_cifar100_asset() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../benchmarks/vnncomp2025/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx",
    );
    let bytes = std::fs::read(&path).expect(
        "qualification requires the actual original CIFAR100 asset; no synthetic fallback or skip",
    );
    let digest = Sha256::digest(&bytes);
    assert_eq!(
        format!("{digest:x}"),
        "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4",
        "qualification requires the pinned original CIFAR100 source bytes"
    );
    let path_result = super::super::margin_row_bab::extract_twin_spec_named(&path)
        .expect("original CIFAR100 asset must extract");
    let held_result = super::super::margin_row_bab::extract_twin_spec_named_bytes(&bytes)
        .expect("same held original asset must extract");
    assert_eq!(
        format!("{:?}", path_result.0),
        format!("{:?}", held_result.0)
    );
    assert_eq!(path_result.1, held_result.1);
}

#[test]
fn prepared_original_source_exact_add_layout_and_inference_fold_are_authenticated() {
    let (budget, _, _) = resources(32 << 20);
    let mut residual = model(1, true);
    let graph = residual.graph.as_mut().unwrap();
    graph
        .node
        .insert(2, node("Add", &["conv", "trunk"], "residual"));
    graph.node[3].input[0] = "residual".into();
    super::super::prepared_root_source::authenticate_original_twin_surface(
        &residual.encode_to_vec(),
        &budget,
        at(),
    )
    .unwrap();
    let mut different = model(1, true);
    let graph = different.graph.as_mut().unwrap();
    graph.input[0] = info("input", &[1, 1, 1, 2]);
    graph.initializer.push(TensorProto {
        name: "shape".into(),
        data_type: 7,
        dims: vec![4],
        int64_data: vec![1, 2, 1, 1],
        ..Default::default()
    });
    graph
        .node
        .insert(2, node("Reshape", &["trunk", "shape"], "other_layout"));
    graph
        .node
        .insert(3, node("Add", &["trunk", "other_layout"], "added"));
    graph.node[4].input[0] = "added".into();
    assert!(
        super::super::prepared_root_source::authenticate_original_twin_surface(
            &different.encode_to_vec(),
            &budget,
            at()
        )
        .is_err(),
        "same volume does not authenticate original Add broadcast/layout"
    );
    let mut normalized = model(1, true);
    let graph = normalized.graph.as_mut().unwrap();
    graph.initializer.extend([
        tensor("scale", &[1], &[1.25]),
        tensor("shift", &[1], &[-0.25]),
        tensor("mean", &[1], &[0.5]),
        tensor("var", &[1], &[1.0]),
    ]);
    let mut bn = node(
        "BatchNormalization",
        &["conv", "scale", "shift", "mean", "var"],
        "bn",
    );
    bn.attribute.push(AttributeProto {
        name: "epsilon".into(),
        r#type: 1,
        f: Some(0.0),
        ..Default::default()
    });
    graph.node.insert(1, bn);
    graph.node[2].input[0] = "bn".into();
    let (_dir, onnx, vnn) = files(&normalized, &property(1, "-1", "1"));
    let owner = prove(&onnx, &vnn, &budget)
        .expect("actual original inference fold must reach the real publisher owner");
    assert!(owner.min_margin() > 1.99);
}

#[test]
fn prepared_original_files_precision_requests_cannot_be_relabelled_as_f64_delivery() {
    let (budget, memory, _) = resources(32 << 20);
    let (_dir, onnx, vnn) = files(&model(1, true), &property(1, "-1", "1"));
    for reference_arithmetic in [
        ny_propagate::margin_row::PreparedReferenceArithmetic::F64,
        ny_propagate::margin_row::PreparedReferenceArithmetic::Binary32Certified,
    ] {
        for policy in [
            RootPrecisionRuntimePolicy {
                root_f32: true,
                ..RootPrecisionRuntimePolicy::default()
            },
            RootPrecisionRuntimePolicy {
                tableau_f32: true,
                ..RootPrecisionRuntimePolicy::default()
            },
        ] {
            assert!(OriginalPreparedRootPublication::prove_original_files(
                &onnx,
                &vnn,
                policy,
                PreparedRootProofOptions {
                    reference_arithmetic,
                    ..Default::default()
                },
                &budget,
                at()
            )
            .is_err());
            assert_eq!(live(&memory), 0);
        }
    }
    let stored = OriginalPreparedRootPublication::prove_original_files(
        &onnx,
        &vnn,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            keep_input_rows: true,
            row_f32: true,
            ..PreparedRootProofOptions::default()
        },
        &budget,
        at(),
    )
    .unwrap();
    assert!(!stored.0.proof.policy().root_f32 && !stored.0.proof.policy().tableau_f32);
    stored.checkpoint(None).unwrap();
    drop(stored);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_original_source_explicit_selected_zero_wire_payload_keeps_original_meaning() {
    let (budget, _, _) = resources(32 << 20);
    let mut original = model(1, true);
    original.opset_import[0].version = 14;
    let graph = original.graph.as_mut().unwrap();
    graph.initializer.extend([
        tensor("scale", &[1], &[1.0]),
        tensor("shift", &[1], &[0.0]),
        tensor("mean", &[1], &[0.0]),
        tensor("var", &[1], &[1.0]),
    ]);
    let mut bn = node(
        "BatchNormalization",
        &["conv", "scale", "shift", "mean", "var"],
        "bn",
    );
    // Selected proto3 FLOAT/INT may omit the payload wire tag for zero.
    bn.attribute.push(AttributeProto {
        name: "epsilon".into(),
        r#type: 1,
        f: None,
        ..Default::default()
    });
    bn.attribute.push(AttributeProto {
        name: "training_mode".into(),
        r#type: 2,
        i: None,
        ..Default::default()
    });
    graph.node.insert(1, bn);
    graph.node[2].input[0] = "bn".into();
    let (_dir, onnx, vnn) = files(&original, &property(1, "-1", "1"));
    let owner = prove(&onnx, &vnn, &budget)
        .expect("actual zero schema payload must preserve exact inference source");
    assert!(owner.min_margin() > 1.99);
    owner.checkpoint(None).unwrap();
}
