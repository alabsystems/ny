// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use ny_onnx::onnx_proto::{
    GraphProto, NodeProto, OperatorSetIdProto, TensorShapeProto, TensorTypeProto, TypeProto,
};

fn info(name: &str, n: usize) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: [1, n]
                        .into_iter()
                        .map(|v| ny_onnx::onnx_proto::tensor_shape_proto::Dimension {
                            value: Some(
                                ny_onnx::onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                                    v as i64,
                                ),
                            ),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}
fn tensor(name: &str, dims: &[i64], data: &[f32]) -> TensorProto {
    TensorProto {
        name: name.into(),
        data_type: 1,
        dims: dims.to_vec(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}
fn model() -> ModelProto {
    let mut g = GraphProto {
        name: "safenlp_dense_fixture".into(),
        input: vec![info("input", 2)],
        output: vec![info("output", 2)],
        initializer: vec![
            tensor("w1", &[2, 1], &[1.0, 0.0]),
            tensor("b1", &[1], &[0.0]),
            tensor("w2", &[1, 2], &[0.0, 0.0]),
            tensor("b2", &[2], &[1.0, 0.0]),
        ],
        ..Default::default()
    };
    for (op, inputs, out) in [
        ("MatMul", vec!["input", "w1"], "mm1"),
        ("Add", vec!["mm1", "b1"], "y"),
        ("Relu", vec!["y"], "h"),
        ("MatMul", vec!["h", "w2"], "mm2"),
        ("Add", vec!["mm2", "b2"], "output"),
    ] {
        g.node.push(NodeProto {
            op_type: op.into(),
            input: inputs.into_iter().map(str::to_owned).collect(),
            output: vec![out.into()],
            ..Default::default()
        });
    }
    ModelProto {
        ir_version: 8,
        graph: Some(g),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        ..Default::default()
    }
}
const PROPERTY:&str="(vnnlib-version <2.0>)\n(declare-network N (declare-input X float32 [1, 2]) (declare-output Y float32 [1, 2]))\n(assert (>= X[0,0] 0.1))\n(assert (<= X[0,0] 0.1))\n(assert (>= X[0,1] -1.0))\n(assert (<= X[0,1] 1.0))\n(assert (<= Y[0,0] Y[0,1]))\n";

#[test]
fn authored_static_matmul_layout_and_complete_source_hash_are_preserved() {
    let m = model();
    let bytes = m.encode_to_vec();
    let s = extract(&bytes).unwrap();
    assert_eq!(s.inputs(), 2);
    assert_eq!(s.heads(), 1);
    assert_eq!(
        s.authored_digest(),
        <[u8; 32]>::from(Sha256::digest(&bytes))
    );
    for fault in 0..7 {
        let mut b = model();
        let g = b.graph.as_mut().unwrap();
        match fault {
            0 => g.node[0].input.swap(0, 1),
            1 => g.initializer[0].dims = vec![1, 2],
            2 => g.node[1].input[1] = "input".into(),
            3 => g.node[2].op_type = "Identity".into(),
            4 => g.initializer[1].dims = vec![1, 1],
            5 => g.initializer[0].data_type = 10,
            _ => g.node[0].output[0] = g.input[0].name.clone(),
        }
        assert!(extract(&b.encode_to_vec()).is_none(), "fault{fault}");
    }
}

#[test]
fn original_singleton_exact_decimal_box_and_inclusive_margin_mapping_are_owned() {
    let p = property_from_source(PROPERTY, 2).unwrap();
    assert_eq!((p.target, p.adversary), (0, 1));
    assert!(!p.spec.is_disjunction && p.spec.output_constraint_clauses.len() == 1);
    assert!(p.lower[0] < 0.1 && p.upper[0] == 0.1); // nearest0.1 lies above the exactdecimal
    let (guard, fragment) =
        super::super::margin_row_bab::margin_row_spec_agreement_owned(&p.spec, 0, &[1]);
    assert!(
        guard.is_clear() && fragment.starts_with("g3=ran "),
        "{fragment}"
    );
    for wrong in [
        PROPERTY.replace("(<= Y[0,0] Y[0,1])", "(< Y[0,0] Y[0,1])"),
        PROPERTY.replace("X[0,0]", "X[1,0]"),
        PROPERTY.replace("[1, 2]", "[2, 1]"),
        format!("{PROPERTY}(assert (>= Y[0,0] Y[0,1]))\n"),
        format!(
            "{}\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)",
            PROPERTY.replace("Y[0,0]", "Y_0").replace("Y[0,1]", "Y_1")
        ),
        PROPERTY.replace(
            "(declare-output Y float32 [1, 2]))",
            "(declare-output Y float32 [1, 2]) (isomorphic-to OTHER))",
        ),
        PROPERTY.replace(
            "(declare-output Y float32 [1, 2])",
            "(declare-output Y float32 [1, 2] (equal-to OTHER))",
        ),
    ] {
        assert!(property_from_source(&wrong, 2).is_none());
    }
}

#[test]
fn segmented_and_conflicting_inline_tensor_storage_are_refused() {
    let mut segmented = model();
    segmented.graph.as_mut().unwrap().initializer[0].segment =
        Some(ny_onnx::onnx_proto::TensorSegmentProto { begin: 0, end: 2 });
    assert!(extract(&segmented.encode_to_vec()).is_none());

    // Each representation alone encodes the complete same matrix. Combining
    // them is ambiguous original storage and may not become a certificate.
    let mut raw = model();
    let first = &mut raw.graph.as_mut().unwrap().initializer[0];
    first.raw_data = first
        .float_data
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert!(extract(&raw.encode_to_vec()).is_none());
    raw.graph.as_mut().unwrap().initializer[0]
        .float_data
        .clear();
    assert!(extract(&raw.encode_to_vec()).is_some());
}

fn wire_bytes(tag: u32, payload: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    for mut n in [u64::from(tag) * 8 + 2, payload.len() as u64] {
        while n >= 128 {
            result.push((n as u8) | 0x80);
            n >>= 7;
        }
        result.push(n as u8);
    }
    result.extend_from_slice(payload);
    result
}

#[test]
fn semantic_fields_dropped_by_the_shared_decoder_are_refused_from_original_wire() {
    let m = model();
    let mut functions = m.encode_to_vec();
    functions.extend(wire_bytes(25, &[]));
    assert!(extract(&functions).is_none());
    let mut g = m.graph.clone().unwrap();
    let nodes = std::mem::take(&mut g.node);
    let mut graph = g.encode_to_vec();
    for (i, n) in nodes.iter().enumerate() {
        let mut raw = n.encode_to_vec();
        if i == 0 {
            raw.extend(wire_bytes(8, b"alternate"));
        }
        graph.extend(wire_bytes(1, &raw));
    }
    let mut shell = m.clone();
    shell.graph = None;
    let mut overloaded = shell.encode_to_vec();
    overloaded.extend(wire_bytes(7, &graph));
    assert!(extract(&overloaded).is_none());
    for ir in [0, 3, 11] {
        let mut bad = m.clone();
        bad.ir_version = ir;
        assert!(extract(&bad.encode_to_vec()).is_none());
    }
    let mut bad = m;
    bad.graph.as_mut().unwrap().name.clear();
    assert!(extract(&bad.encode_to_vec()).is_none());
}

#[test]
fn competing_nondense_type_and_duplicate_annotations_are_preserved_and_refused() {
    let m = model();
    let mut g = m.graph.clone().unwrap();
    let x = g.input.remove(0);
    let mut raw_x = x.encode_to_vec();
    raw_x.extend(wire_bytes(2, &wire_bytes(4, &[])));
    let mut graph = g.encode_to_vec();
    graph.extend(wire_bytes(11, &raw_x));
    let mut shell = m.clone();
    shell.graph = None;
    let mut bytes = shell.encode_to_vec();
    bytes.extend(wire_bytes(7, &graph));
    assert!(extract(&bytes).is_none());
    let mut graph = m.graph.as_ref().unwrap().encode_to_vec();
    let annotation = info("mm1", 1).encode_to_vec();
    graph.extend(wire_bytes(13, &annotation));
    graph.extend(wire_bytes(13, &annotation));
    let mut bytes = shell.encode_to_vec();
    bytes.extend(wire_bytes(7, &graph));
    assert!(extract(&bytes).is_none());
}

#[test]
fn flipped_singleton_g3_cannot_ratify_the_original_margin_mapping() {
    let p = property_from_source(PROPERTY, 2).unwrap();
    let (guard, fragment) =
        super::super::margin_row_bab::margin_row_spec_agreement_owned(&p.spec, 1, &[0]);
    assert!(!guard.is_clear(), "{fragment}");
}

#[test]
fn source_mutation_is_refused_after_a_complete_original_parse() {
    let dir = tempfile::tempdir().unwrap();
    let onnx = dir.path().join("net.onnx");
    let prop = dir.path().join("spec.vnnlib");
    let raw = model().encode_to_vec();
    std::fs::write(&onnx, &raw).unwrap();
    std::fs::write(&prop, PROPERTY).unwrap();
    let m = extract(&raw).unwrap();
    let p = property(&prop, 2).unwrap();
    assert!(unchanged_sources(&onnx, &prop, &m, &p));
    std::fs::write(&prop, PROPERTY.replace("-1.0", "-2.0")).unwrap();
    assert!(!unchanged_sources(&onnx, &prop, &m, &p));
    std::fs::write(&prop, PROPERTY).unwrap();
    let mut changed = model();
    changed.graph.as_mut().unwrap().initializer[3].float_data[0] = 0.0;
    std::fs::write(&onnx, changed.encode_to_vec()).unwrap();
    assert!(!unchanged_sources(&onnx, &prop, &m, &p));
}

#[test]
fn dark_and_expired_routes_do_not_read_or_rewrite_original_artifacts() {
    assert!(try_input_head(
        Path::new("missing.onnx"),
        Path::new("missing.vnnlib"),
        Some(Instant::now() + Duration::from_secs(1)),
        &MarginRowRuntimePolicy::default()
    )
    .is_none());
    let p = MarginRowRuntimePolicy {
        head_groups: ny_propagate::margin_row::HeadGroupPolicy::new(1.0, 1, 1),
        ..Default::default()
    };
    assert!(try_input_head(
        Path::new("missing.onnx"),
        Path::new("missing.vnnlib"),
        Some(Instant::now()),
        &p
    )
    .is_none());
}

#[test]
fn real_singleton_route_enters_instance_g2_g3_without_an_extra_output_or_disjunct() {
    let dir = tempfile::tempdir().unwrap();
    let onnx = dir.path().join("net.onnx");
    let prop = dir.path().join("spec.vnnlib");
    std::fs::write(&onnx, model().encode_to_vec()).unwrap();
    std::fs::write(&prop, PROPERTY).unwrap();
    let p = MarginRowRuntimePolicy {
        head_groups: ny_propagate::margin_row::HeadGroupPolicy::new(1.0, 1, 1),
        ..Default::default()
    };
    assert!(matches!(
        try_input_head(
            &onnx,
            &prop,
            Some(Instant::now() + Duration::from_secs(2)),
            &p
        ),
        Some(VnncompResult::Unsat(_))
    ));
    assert_eq!(std::fs::read_to_string(&prop).unwrap(), PROPERTY);
}

#[test]
fn frozen_phase_cover_policy_reaches_the_original_singleton_route() {
    let dir = tempfile::tempdir().unwrap();
    let onnx = dir.path().join("net.onnx");
    let prop = dir.path().join("spec.vnnlib");
    std::fs::write(&onnx, model().encode_to_vec()).unwrap();
    std::fs::write(&prop, PROPERTY).unwrap();
    let levers = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| match name {
        "NY_MARGIN_ROW_HEAD_GROUPS" => Some("1".to_owned()),
        "NY_MARGIN_ROW_HEAD_PHASE_COVER" => Some("7".to_owned()),
        _ => None,
    });
    let policy = MarginRowRuntimePolicy::from_levers(&levers).unwrap();
    assert_eq!(
        policy
            .head_groups
            .unwrap()
            .phase_cover()
            .unwrap()
            .max_heads(),
        7
    );
    assert!(matches!(
        try_input_head(
            &onnx,
            &prop,
            Some(Instant::now() + Duration::from_secs(2)),
            &policy
        ),
        Some(VnncompResult::Unsat(_))
    ));
    assert_eq!(std::fs::read_to_string(prop).unwrap(), PROPERTY);
}

// These dyadic constants reproduce the actual original export's admission
// metadata (IR7, standard13 plus unused ai.onnx.ml2, symbolic batch, raw
// [30,128]/[128]/[128,2]/[2] FLOAT32 storage). They do not claim a verdict on
// the trained original weights, which remain a separate matched CLI control.
fn symbolic_info(name: &str, width: usize, batch: &str) -> ValueInfoProto {
    let mut value = info(name, width);
    value
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
        .value =
        Some(ny_onnx::onnx_proto::tensor_shape_proto::dimension::Value::DimParam(batch.into()));
    value
}

fn original_export_metadata_model() -> ModelProto {
    let mut m = model();
    m.ir_version = 7;
    m.opset_import.push(OperatorSetIdProto {
        domain: "ai.onnx.ml".into(),
        version: 2,
    });
    let g = m.graph.as_mut().unwrap();
    g.name = "tf2onnx".into();
    g.input = vec![symbolic_info("embeddings", 30, "unk__6")];
    g.output = vec![symbolic_info("predictions", 2, "unk__7")];
    g.node[0].input[0] = "embeddings".into();
    g.node[4].output[0] = "predictions".into();
    let mut first = vec![0.0; 30 * 128];
    first[0] = 1.0;
    first[29 * 128 + 127] = 0.25; // full original last input is genuine
    let mut last = vec![0.0; 128 * 2];
    last[0] = 0.125;
    last[127 * 2] = 0.25;
    g.initializer = vec![
        tensor("w1", &[30, 128], &first),
        tensor("b1", &[128], &[0.0; 128]),
        tensor("w2", &[128, 2], &last),
        tensor("b2", &[2], &[1.0, 0.0]),
    ];
    for t in &mut g.initializer {
        t.raw_data = t.float_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        t.float_data.clear();
    }
    m
}

fn original_singleton_property() -> String {
    let mut text = "(vnnlib-version <2.0>)\n(declare-network N (declare-input X float32 [1, 30]) (declare-output Y float32 [1, 2]))\n".to_string();
    for k in 0..30 {
        text.push_str(&format!(
            "(assert (>= X[0,{k}] -1))\n(assert (<= X[0,{k}] 1))\n"
        ));
    }
    text.push_str("(assert (<= Y[0,0] Y[0,1]))\n");
    text
}

fn with_original_annotations(m: &ModelProto, values: &[ValueInfoProto]) -> Vec<u8> {
    let mut shell = m.clone();
    let mut graph = shell.graph.take().unwrap().encode_to_vec();
    for value in values {
        graph.extend(wire_bytes(13, &value.encode_to_vec()));
    }
    let mut bytes = shell.encode_to_vec();
    bytes.extend(wire_bytes(7, &graph));
    bytes
}

#[test]
fn original_export_metadata_preserves_raw_constants_under_owned_singleton_binding() {
    let m = original_export_metadata_model();
    let bytes = m.encode_to_vec();
    let p = property_from_source(&original_singleton_property(), 30).unwrap();
    assert!(extract(&bytes).is_none()); // symbolic axis has no independent authority
    assert_eq!(input_width_hint(&bytes), Some(30));
    let s = extract_bound(&bytes, Some(&p.singleton_layout)).unwrap();
    assert_eq!((s.inputs(), s.heads()), (30, 128));
    assert_eq!(
        s.authored_digest(),
        <[u8; 32]>::from(Sha256::digest(&bytes))
    );
    let g = m.graph.as_ref().unwrap();
    assert_eq!(g.node.len(), 5);
    assert_eq!(g.initializer.len(), 4);
    assert_eq!(
        parameters(&g.initializer[0], &[30, 128]).unwrap()[29 * 128 + 127],
        0.25
    );
    assert_eq!(
        parameters(&g.initializer[2], &[128, 2]).unwrap()[127 * 2],
        0.25
    );
    assert_eq!((p.spec.num_inputs, p.spec.num_outputs), (30, 2));
    assert!(!p.spec.is_disjunction && p.spec.output_constraint_clauses.len() == 1);
}

#[test]
fn unused_unique_imports_never_authorize_used_unknown_or_overridden_domains() {
    let mut m = model();
    m.opset_import.push(OperatorSetIdProto {
        domain: "ai.onnx.ml".into(),
        version: 2,
    });
    assert!(extract(&m.encode_to_vec()).is_some()); // inert import alone
    for (domain, version) in [
        ("", 13),
        ("", 14),
        ("ai.onnx.ml", 2),
        ("ai.onnx.ml", 3),
        ("ai.onnx", 13),
        ("ai.onnx", 14),
    ] {
        let mut bad = m.clone();
        bad.opset_import.push(OperatorSetIdProto {
            domain: domain.into(),
            version,
        });
        assert!(
            extract(&bad.encode_to_vec()).is_none(),
            "duplicate {domain}/{version}"
        );
    }
    for domain in ["ai.onnx.ml", "unimported.custom", "ai.onnx"] {
        let mut bad = m.clone();
        bad.graph.as_mut().unwrap().node[0].domain = domain.into();
        assert!(extract(&bad.encode_to_vec()).is_none(), "used {domain}");
    }
    let mut missing = m.clone();
    missing.opset_import.remove(0);
    assert!(extract(&missing.encode_to_vec()).is_none());
    let mut invalid = m.clone();
    invalid.opset_import[0].version = 22;
    assert!(extract(&invalid.encode_to_vec()).is_none());
    let mut invalid = m.clone();
    invalid.opset_import[1].version = 0;
    assert!(extract(&invalid.encode_to_vec()).is_none());
    let mut overridden = m.encode_to_vec();
    overridden.extend(wire_bytes(25, &[]));
    assert!(extract(&overridden).is_none()); // local-function override still refused
}

#[test]
fn symbolic_batch_refuses_missing_or_non_singleton_typed_property_authority() {
    let m = original_export_metadata_model();
    let bytes = m.encode_to_vec();
    assert!(extract_bound(&bytes, None).is_none());
    let foreign = property_from_source(PROPERTY, 2).unwrap();
    assert!(extract_bound(&bytes, Some(&foreign.singleton_layout)).is_none());
    let original = original_singleton_property();
    for wrong in [
        original.replace(
            "declare-input X float32 [1, 30]",
            "declare-input X float32 [2, 30]",
        ),
        original.replace(
            "declare-output Y float32 [1, 2]",
            "declare-output Y float32 [2, 2]",
        ),
        original.replace("float32", "float64"),
        original.replace("X[0,0]", "X[1,0]"),
        original.replace("[1, 30]", "[30]"),
    ] {
        assert!(property_from_source(&wrong, 30).is_none());
    }
}

#[test]
fn singleton_specialization_proves_all_chain_axes_and_refuses_conflicts() {
    use ny_onnx::onnx_proto::tensor_shape_proto::dimension::Value;
    let p = property_from_source(&original_singleton_property(), 30).unwrap();
    let layout = Some(&p.singleton_layout);
    for fault in 0..10 {
        let mut m = original_export_metadata_model();
        let g = m.graph.as_mut().unwrap();
        match fault {
            0 => g.input[0] = info("embeddings", 30), // static1 also valid
            1 => g.output[0] = info("predictions", 2), // static1 also valid
            2 => {
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
            3 => {
                g.output[0]
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
            4 => {
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
            5 => {
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
                    .value = None
            }
            6 => g.input[0]
                .r#type
                .as_mut()
                .unwrap()
                .tensor_type
                .as_mut()
                .unwrap()
                .shape
                .as_mut()
                .unwrap()
                .dim
                .push(ny_onnx::onnx_proto::tensor_shape_proto::Dimension {
                    value: Some(Value::DimValue(1)),
                }),
            7 => g.output[0] = symbolic_info("predictions", 3, "unk__7"),
            8 => g.initializer[2].dims = vec![2, 128],
            _ => g.input[0] = symbolic_info("embeddings", 30, ""),
        }
        assert_eq!(
            extract_bound(&m.encode_to_vec(), layout).is_some(),
            fault < 2,
            "fault{fault}"
        );
    }
    let mut conflicting = original_export_metadata_model();
    conflicting.graph.as_mut().unwrap().initializer[1].dims = vec![1, 128];
    assert!(extract_bound(&conflicting.encode_to_vec(), layout).is_none());
}

#[test]
fn symbolic_hidden_annotations_are_checked_even_with_minimal_decoder() {
    let m = original_export_metadata_model();
    let p = property_from_source(&original_singleton_property(), 30).unwrap();
    let layout = Some(&p.singleton_layout);
    let valid = [
        symbolic_info("mm1", 128, "hidden"),
        symbolic_info("y", 128, "hidden"),
        symbolic_info("mm2", 2, "output_batch"),
        symbolic_info("predictions", 2, "unk__7"),
    ];
    assert!(extract_bound(&with_original_annotations(&m, &valid), layout).is_some());
    for bad in [
        symbolic_info("mm1", 129, "hidden"),
        symbolic_info("mm1", 128, ""),
        symbolic_info("foreign", 128, "hidden"),
        symbolic_info("mm2", 128, "output_batch"),
    ] {
        assert!(extract_bound(&with_original_annotations(&m, &[bad]), layout).is_none());
    }
    assert!(extract_bound(
        &with_original_annotations(&m, &[valid[0].clone(), valid[0].clone()]),
        layout
    )
    .is_none());
    let mut bad = valid[0].clone();
    bad.r#type
        .as_mut()
        .unwrap()
        .tensor_type
        .as_mut()
        .unwrap()
        .elem_type = 10;
    assert!(extract_bound(&with_original_annotations(&m, &[bad]), layout).is_none());
}

#[test]
fn real_original_export_layout_enters_one_class_instance_g2_g3_without_graph_rewrite() {
    let dir = tempfile::tempdir().unwrap();
    let onnx = dir.path().join("original.onnx");
    let prop = dir.path().join("original.vnnlib");
    let raw = original_export_metadata_model().encode_to_vec();
    let original = original_singleton_property();
    std::fs::write(&onnx, &raw).unwrap();
    std::fs::write(&prop, &original).unwrap();
    let p = property(&prop, 30).unwrap();
    let model = extract_bound(&raw, Some(&p.singleton_layout)).unwrap();
    assert!(unchanged_sources(&onnx, &prop, &model, &p));
    let policy = MarginRowRuntimePolicy {
        head_groups: ny_propagate::margin_row::HeadGroupPolicy::new(1.0, 1, 1),
        ..Default::default()
    };
    assert!(matches!(
        try_input_head(
            &onnx,
            &prop,
            Some(Instant::now() + Duration::from_secs(2)),
            &policy
        ),
        Some(VnncompResult::Unsat(_))
    ));
    assert_eq!(std::fs::read(&onnx).unwrap(), raw);
    assert_eq!(std::fs::read_to_string(&prop).unwrap(), original);
    assert_eq!(p.spec.num_outputs, 2);
    assert!(!p.spec.is_disjunction && p.spec.output_constraint_clauses.len() == 1);
    std::fs::write(&prop, original.replace("-1)", "-2)")).unwrap();
    assert!(!unchanged_sources(&onnx, &prop, &model, &p));
}

/// B2 (review(eng14)): the route closes its one class OUTSIDE every
/// margin-row frame and returns before the lane proof capture is armed, so it
/// must claim that class itself. With capture on, a PUBLISHED input-head
/// `unsat` streams exactly ONE `opaque` `input_head` node over the whole
/// region whose classes are the receipt's own (the verdict record's
/// `required`); a refused attempt (expired original deadline) streams
/// nothing. Mutation target: the `stream_opaque_claim` call (removing it
/// leaves a published bundle with `nodes=0`, which the checker reads as a
/// cover gap).
#[test]
fn published_input_head_unsat_claims_exactly_its_receipt_classes_as_one_opaque_node() {
    use ny_propagate::certify::{
        with_thread_sink, AlphaRecord, FdStateRecord, LeafRecord, NodeClose, NodeRecord, ProofSink,
        SitesRecord, VerdictRecord,
    };
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<NodeRecord>>>);
    impl ProofSink for Shared {
        fn leaf(&mut self, _leaf: &LeafRecord) {}
        fn node(&mut self, node: &NodeRecord) {
            self.0.lock().unwrap().push(node.clone());
        }
        fn alpha(&mut self, _alpha: &AlphaRecord) {}
        fn sites(&mut self, _sites: &SitesRecord) {}
        fn fd_state(&mut self, _state: &FdStateRecord) {}
        fn finish(&mut self, _verdict: &VerdictRecord) {}
    }

    // The Proof arm's process flag (other proof-bundle tests of this binary
    // switch it on the same way); the thread sink isolates this test's nodes.
    crate::commands::proof_bundle::enable_hint_capture_for_test();
    let dir = tempfile::tempdir().unwrap();
    let onnx = dir.path().join("net.onnx");
    let prop = dir.path().join("spec.vnnlib");
    std::fs::write(&onnx, model().encode_to_vec()).unwrap();
    std::fs::write(&prop, PROPERTY).unwrap();
    let policy = MarginRowRuntimePolicy {
        head_groups: ny_propagate::margin_row::HeadGroupPolicy::new(1.0, 1, 1),
        ..Default::default()
    };
    let run = |deadline: Instant| {
        let sink = Shared::default();
        let (out, _) = with_thread_sink(Box::new(sink.clone()), || {
            try_input_head(&onnx, &prop, Some(deadline), &policy)
        });
        let nodes = sink.0.lock().unwrap().clone();
        (out, nodes)
    };
    let (out, nodes) = run(Instant::now() + Duration::from_secs(2));
    let Some(VnncompResult::Unsat(receipt)) = out else {
        panic!("the fixture must publish unsat through the route (else VOID)");
    };
    let (classes, _, _) = receipt.published_summary();
    assert_eq!(classes, vec![1], "PROPERTY's one adversary (Y0 <= Y1)");
    assert_eq!(nodes.len(), 1, "{nodes:?}");
    let n = &nodes[0];
    assert_eq!(n.close, NodeClose::Opaque, "{n:?}");
    assert_eq!(n.src, "input_head", "{n:?}");
    assert_eq!(n.dis.as_deref(), Some(classes.as_slice()), "{n:?}");
    assert!(
        n.lits.is_empty() && n.xlits.is_empty(),
        "whole region: {n:?}"
    );
    let (refused, refused_nodes) = run(Instant::now());
    assert!(
        refused.is_none(),
        "an expired original deadline publishes nothing"
    );
    assert!(
        refused_nodes.is_empty(),
        "a refused attempt claims nothing: {refused_nodes:?}"
    );
}
