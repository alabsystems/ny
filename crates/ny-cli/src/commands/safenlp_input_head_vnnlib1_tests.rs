// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use ny_onnx::onnx_proto::{
    tensor_shape_proto, GraphProto, NodeProto, OperatorSetIdProto, TensorShapeProto,
    TensorTypeProto, TypeProto,
};

// Literal original VNN2026 1.0 property355: no synthesized typed wrapper.
// The controlled model below has the actual export metadata and a known
// authored constant margin. It makes no trained-model closure assertion.
const ORIGINAL_PROPERTY: &str = r#"; medical perturbations property.

(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const X_3 Real)
(declare-const X_4 Real)
(declare-const X_5 Real)
(declare-const X_6 Real)
(declare-const X_7 Real)
(declare-const X_8 Real)
(declare-const X_9 Real)
(declare-const X_10 Real)
(declare-const X_11 Real)
(declare-const X_12 Real)
(declare-const X_13 Real)
(declare-const X_14 Real)
(declare-const X_15 Real)
(declare-const X_16 Real)
(declare-const X_17 Real)
(declare-const X_18 Real)
(declare-const X_19 Real)
(declare-const X_20 Real)
(declare-const X_21 Real)
(declare-const X_22 Real)
(declare-const X_23 Real)
(declare-const X_24 Real)
(declare-const X_25 Real)
(declare-const X_26 Real)
(declare-const X_27 Real)
(declare-const X_28 Real)
(declare-const X_29 Real)

(declare-const Y_0 Real)
(declare-const Y_1 Real)

; Input constraints:
(assert (>= X_0 0.21293231017343062))
(assert (<= X_0 0.279565079995821))

(assert (>= X_1 -0.27497669230549265))
(assert (<= X_1 -0.2361803784405555))

(assert (>= X_2 0.1277749534300999))
(assert (<= X_2 0.19705884316941868))

(assert (>= X_3 0.21671806097390123))
(assert (<= X_3 0.3114666845928889))

(assert (>= X_4 -0.14576905597438156))
(assert (<= X_4 -0.09950878271114302))

(assert (>= X_5 0.06735409589134791))
(assert (<= X_5 0.0830055594471975))

(assert (>= X_6 0.1913660831083479))
(assert (<= X_6 0.26758713008295504))

(assert (>= X_7 -0.0771878544735144))
(assert (<= X_7 0.014045217646075778))

(assert (>= X_8 -0.010069590147151882))
(assert (<= X_8 0.04386079280225089))

(assert (>= X_9 -0.0047139962220969535))
(assert (<= X_9 0.08899460439910024))

(assert (>= X_10 -0.06994104771981272))
(assert (<= X_10 -0.049239325377698574))

(assert (>= X_11 -0.0734553265833889))
(assert (<= X_11 -0.027578616580472854))

(assert (>= X_12 -0.03568377228971603))
(assert (<= X_12 0.027770392475443894))

(assert (>= X_13 -0.18412637176156277))
(assert (<= X_13 -0.09294478873379536))

(assert (>= X_14 0.21622786805268135))
(assert (<= X_14 0.24458864343593917))

(assert (>= X_15 -0.0222583751343581))
(assert (<= X_15 0.10724549056147581))

(assert (>= X_16 -0.061029653976464975))
(assert (<= X_16 0.009392494138993564))

(assert (>= X_17 0.09792545252643912))
(assert (<= X_17 0.14601604560878295))

(assert (>= X_18 -0.0011057302489327337))
(assert (<= X_18 0.026158582611631974))

(assert (>= X_19 0.09468703339202503))
(assert (<= X_19 0.14030921371484045))

(assert (>= X_20 -0.018766678297836993))
(assert (<= X_20 0.03197747743967959))

(assert (>= X_21 -0.04083962836913556))
(assert (<= X_21 0.01984331506907734))

(assert (>= X_22 0.11093959727747099))
(assert (<= X_22 0.1456457589880925))

(assert (>= X_23 0.10704488071731467))
(assert (<= X_23 0.178531986082806))

(assert (>= X_24 -0.008244155803777601))
(assert (<= X_24 0.058751601199969763))

(assert (>= X_25 -0.07977972064233592))
(assert (<= X_25 -0.03281278280003197))

(assert (>= X_26 0.05121536115532042))
(assert (<= X_26 0.12058674407291756))

(assert (>= X_27 0.0717061897948455))
(assert (<= X_27 0.10217551502821033))

(assert (>= X_28 -0.16267098874223035))
(assert (<= X_28 -0.08975712477296197))

(assert (>= X_29 -0.058825522696346284))
(assert (<= X_29 -0.02668901652421179))

; Output constraints:
(assert (<= Y_0 Y_1))"#;

fn declaration_property() -> String {
    let mut p = String::new();
    for i in 0..30 {
        p.push_str(&format!("(declare-const X_{i} Real)\n"));
    }
    p.push_str("(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n");
    for i in 0..30 {
        p.push_str(&format!("(assert (>= X_{i} -1))\n(assert (<= X_{i} 1))\n"));
    }
    p.push_str("(assert (<= Y_0 Y_1))\n");
    p
}
fn typed_property() -> String {
    let mut p = "(vnnlib-version <2.0>)\n(declare-network N (declare-input X float32 [1,30]) (declare-output Y float32 [1,2]))\n".to_owned();
    for i in 0..30 {
        p.push_str(&format!(
            "(assert (>= X[0,{i}] -1))\n(assert (<= X[0,{i}] 1))\n"
        ));
    }
    p.push_str("(assert (<= Y[0,0] Y[0,1]))\n");
    p
}
fn info(name: &str, width: usize, batch: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                                batch.into(),
                            )),
                        },
                        tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(
                                width as i64,
                            )),
                        },
                    ],
                }),
            }),
        }),
    }
}
fn tensor(name: &str, dims: &[i64], data: Vec<f32>) -> TensorProto {
    TensorProto {
        name: name.into(),
        data_type: 1,
        dims: dims.to_vec(),
        raw_data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ..Default::default()
    }
}
#[expect(
    clippy::field_reassign_with_default,
    reason = "Retain staged fixture construction and constructor/overwrite order."
)]
fn model(margin: f32) -> ModelProto {
    let mut g = GraphProto::default();
    g.name = "tf2onnx".into();
    g.input = vec![info("embeddings", 30, "unk__6")];
    g.output = vec![info("predictions", 2, "unk__7")];
    g.initializer = vec![
        tensor("w1", &[30, 128], vec![0.; 30 * 128]),
        tensor("b1", &[128], vec![0.; 128]),
        tensor("w2", &[128, 2], vec![0.; 256]),
        tensor("b2", &[2], vec![margin, 0.]),
    ];
    for (op, inputs, out) in [
        ("MatMul", vec!["embeddings", "w1"], "mm1"),
        ("Add", vec!["mm1", "b1"], "y"),
        ("Relu", vec!["y"], "h"),
        ("MatMul", vec!["h", "w2"], "mm2"),
        ("Add", vec!["mm2", "b2"], "predictions"),
    ] {
        g.node.push(NodeProto {
            op_type: op.into(),
            input: inputs.into_iter().map(str::to_owned).collect(),
            output: vec![out.into()],
            ..Default::default()
        });
    }
    ModelProto {
        ir_version: 7,
        graph: Some(g),
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
        ..Default::default()
    }
}
fn bound_model(margin: f32, p: &Property) -> InputHeadSpec {
    extract_bound(&model(margin).encode_to_vec(), Some(&p.singleton_layout)).unwrap()
}
#[expect(
    clippy::field_reassign_with_default,
    reason = "Snapshot runtime defaults before applying this test policy override."
)]
fn policy() -> MarginRowRuntimePolicy {
    let mut p = MarginRowRuntimePolicy::default();
    p.head_groups = ny_propagate::margin_row::HeadGroupPolicy::new(1., 1, 1);
    p
}

#[test]
fn original_legacy_real_property_binds_distinct_symbolic_batches_only_through_complete_chain() {
    assert_eq!(
        format!("{:x}", Sha256::digest(ORIGINAL_PROPERTY.as_bytes())),
        "e6285ee94f332c9cea0ed3cbeb965943572c264587fb074a86345914a64d8432"
    );
    let p = property_from_source(ORIGINAL_PROPERTY, 30).unwrap();
    let raw = model(1.).encode_to_vec();
    assert!(extract(&raw).is_none());
    assert_eq!(input_width_hint(&raw), Some(30));
    let s = extract_bound(&raw, Some(&p.singleton_layout)).unwrap();
    assert_eq!((s.inputs(), s.heads()), (30, 128));
    assert_eq!(s.authored_digest(), <[u8; 32]>::from(Sha256::digest(&raw)));
    assert_eq!(
        (p.spec.num_inputs, p.spec.num_outputs, p.target, p.adversary),
        (30, 2, 0, 1)
    );
    assert!(p.spec.version.is_none() && p.spec.dual_network.is_none());
    assert_eq!(p.format, OriginalPropertyFormat::ScalarRealV1);
    assert_eq!(
        (ny_onnx::vnnlib::parse_vnnlib_assignment_declarations(ORIGINAL_PROPERTY).unwrap()).len(),
        0
    );
}

#[test]
fn scalar_namespace_holes_duplicates_aliases_types_and_extra_declarations_refuse() {
    let good = declaration_property();
    for wrong in [
        good.replace("(declare-const X_7 Real)\n", ""),
        good.replace("(declare-const Y_0 Real)\n", ""),
        format!("{good}(declare-const X_29 Real)\n"),
        format!("{good}(declare-const Y_1 Real)\n"),
        format!("{good}(declare-const Z_0 Real)\n"),
        good.replace("X_7", "X_07"),
        good.replace("Y_1", "Y_01"),
        good.replace("(declare-const X_0 Real)", "(declare-const X_0 Int)"),
        good.replace("(declare-const Y_0 Real)", "(declare-const Y_0 float32)"),
        good.replace("X_29", "X_30"),
    ] {
        assert!(property_from_source(&wrong, 30).is_none(), "{wrong}");
    }
    assert!(property_from_source(&good, 29).is_none());
    assert!(property_from_source(&good, 31).is_none());
}

#[test]
fn scalar_semantic_extras_mixed_syntax_and_changed_objectives_refuse() {
    let good = declaration_property();
    for wrong in [format!("{good}(define-fun X () Real 0)\n"),
        format!("{good}(declare-const Z Real)\n"),
        format!("{good}(assert (>= Y_0 Y_1))\n"),
        format!("{good}(declare-network N (declare-input X float32 [1,30]) (declare-output Y float32 [1,2]))\n"),
        good.replace("(<= Y_0 Y_1)","(< Y_0 Y_1)"),
        good.replace("(<= Y_0 Y_1)","(<= Y_0 Y_0)"),
        good.replace("(<= Y_0 Y_1)","(or (<= Y_0 Y_1) (>= Y_0 Y_1))"),
        good.replace("(declare-const X_0 Real)","(declare-const X_0 Real (equal-to Z))"),
        good.replace("(>= X_0 -1)","(>= X_0 Y_0)"),
        good.replace("X_0","X[0,0]"),
        format!("(vnnlib-version <2.0>)\n{good}")] {
        assert!(property_from_source(&wrong,30).is_none(),"{wrong}");
    }
}

#[test]
fn original_decimal_real_box_and_scalar_mapping_are_preserved_without_casting_or_rewrite() {
    let raw = declaration_property()
        .replace("(>= X_0 -1)", "(>= X_0 0.1)")
        .replace("(<= X_0 1)", "(<= X_0 0.1)");
    let p = property_from_source(&raw, 30).unwrap();
    assert!(p.lower[0] < 0.1 && p.upper[0] == 0.1);
    assert_ne!(p.upper[0], f64::from(0.1f32));
    assert_eq!(
        p.source_digest,
        <[u8; 32]>::from(Sha256::digest(raw.as_bytes()))
    );
    let (g3, fragment) = super::super::margin_row_bab::margin_row_spec_agreement_owned(
        &p.spec,
        p.target,
        &[p.adversary],
    );
    assert!(
        g3.is_clear() && fragment.starts_with("g3=ran "),
        "{fragment}"
    );
    let reverse = property_from_source(&raw.replace("(<= Y_0 Y_1)", "(>= Y_0 Y_1)"), 30).unwrap();
    assert_eq!((reverse.target, reverse.adversary), (1, 0));
}

#[test]
fn scalar_layout_declines_multi_batch_unknown_feature_wrong_rank_and_chain_faults() {
    let p = property_from_source(&declaration_property(), 30).unwrap();
    for fault in 0..15 {
        let mut m = model(1.);
        let g = m.graph.as_mut().unwrap();
        match fault {
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
                    .value = Some(tensor_shape_proto::dimension::Value::DimValue(2))
            }
            1 => {
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
                    .value = Some(tensor_shape_proto::dimension::Value::DimValue(2))
            }
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
                    .dim
                    .remove(0);
            }
            3 => {
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
                    .value = Some(tensor_shape_proto::dimension::Value::DimParam(
                    "features".into(),
                ))
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
                    .dim[0]
                    .value = None
            }
            5 => g.node[0].input.swap(0, 1),
            6 => g.node[2].op_type = "Transpose".into(),
            7 => g.initializer[1].dims = vec![1, 128],
            8 => g.node[4].input[0] = "y".into(),
            9 => g.output.push(info("extra", 2, "B")),
            10 => {
                g.input[0]
                    .r#type
                    .as_mut()
                    .unwrap()
                    .tensor_type
                    .as_mut()
                    .unwrap()
                    .elem_type = 11
            }
            11 => {
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
                    .dim[1]
                    .value = Some(tensor_shape_proto::dimension::Value::DimValue(3))
            }
            12 => {
                g.output[0]
                    .r#type
                    .as_mut()
                    .unwrap()
                    .tensor_type
                    .as_mut()
                    .unwrap()
                    .shape = None
            }
            13 => {
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
                    .value = Some(tensor_shape_proto::dimension::Value::DimParam(String::new()))
            }
            _ => {
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
                    .value = Some(tensor_shape_proto::dimension::Value::DimValue(0))
            }
        }
        assert!(
            extract_bound(&m.encode_to_vec(), Some(&p.singleton_layout)).is_none(),
            "fault{fault}"
        );
    }
    let typed = property_from_source(&typed_property(), 30).unwrap();
    assert!(extract_bound(&model(1.).encode_to_vec(), Some(&typed.singleton_layout)).is_some());
    for fixed_side in [0, 1] {
        let mut m = model(1.);
        let g = m.graph.as_mut().unwrap();
        let side = if fixed_side == 0 {
            &mut g.input[0]
        } else {
            &mut g.output[0]
        };
        side.r#type
            .as_mut()
            .unwrap()
            .tensor_type
            .as_mut()
            .unwrap()
            .shape
            .as_mut()
            .unwrap()
            .dim[0]
            .value = Some(tensor_shape_proto::dimension::Value::DimValue(1));
        assert!(extract_bound(&m.encode_to_vec(), Some(&p.singleton_layout)).is_some());
    }
}

#[test]
fn legacy_original_singleton_reaches_instance_g2_and_original_g3_with_no_property_rewrite() {
    let d = tempfile::tempdir().unwrap();
    let onnx = d.path().join("original.onnx");
    let vnn = d.path().join("original.vnnlib");
    let raw = model(1.).encode_to_vec();
    std::fs::write(&onnx, &raw).unwrap();
    std::fs::write(&vnn, ORIGINAL_PROPERTY).unwrap();
    let p = property(&vnn, 30).unwrap();
    let token = match ny_propagate::margin_row::verify_input_head(
        &bound_model(1., &p),
        &p.lower,
        &p.upper,
        p.target,
        p.adversary,
        policy().head_groups.unwrap(),
        Instant::now() + Duration::from_secs(2),
    ) {
        InputHeadOutcome::Unsat(_, token) => token,
        InputHeadOutcome::Unknown(_) => panic!("authored constant positive margin must close"),
    };
    assert_eq!(token.scope(), ny_propagate::certify::TokenScope::Instance);
    assert_eq!(token.classes(), &[1]);
    assert!(token.publishable());
    let (g3, fragment) =
        super::super::margin_row_bab::margin_row_spec_agreement_owned(&p.spec, 0, &[1]);
    assert!(g3.is_clear() && fragment.starts_with("g3=ran "));
    assert!(ny_propagate::certify::ratify(token, &g3).is_ok());
    assert!(matches!(
        try_input_head(
            &onnx,
            &vnn,
            Some(Instant::now() + Duration::from_secs(2)),
            &policy()
        ),
        Some(VnncompResult::Unsat(_))
    ));
    assert_eq!(std::fs::read(&onnx).unwrap(), raw);
    assert_eq!(std::fs::read_to_string(&vnn).unwrap(), ORIGINAL_PROPERTY);
}

#[test]
fn legacy_zero_tie_and_unsafe_margin_cannot_publish_instance() {
    let p = property_from_source(&declaration_property(), 30).unwrap();
    for margin in [0., -1.] {
        assert!(matches!(
            ny_propagate::margin_row::verify_input_head(
                &bound_model(margin, &p),
                &p.lower,
                &p.upper,
                p.target,
                p.adversary,
                policy().head_groups.unwrap(),
                Instant::now() + Duration::from_secs(2)
            ),
            InputHeadOutcome::Unknown(_)
        ));
    }
    let reverse = property_from_source(
        &declaration_property().replace("(<= Y_0 Y_1)", "(>= Y_0 Y_1)"),
        30,
    )
    .unwrap();
    assert!(matches!(
        ny_propagate::margin_row::verify_input_head(
            &bound_model(1., &reverse),
            &reverse.lower,
            &reverse.upper,
            reverse.target,
            reverse.adversary,
            policy().head_groups.unwrap(),
            Instant::now() + Duration::from_secs(2)
        ),
        InputHeadOutcome::Unknown(_)
    ));
}

#[test]
fn source_changed_original_scalar_atom_or_graph_refuses_binding() {
    let d = tempfile::tempdir().unwrap();
    let onnx = d.path().join("original.onnx");
    let vnn = d.path().join("original.vnnlib");
    let raw = model(1.).encode_to_vec();
    let property = declaration_property();
    std::fs::write(&onnx, &raw).unwrap();
    std::fs::write(&vnn, &property).unwrap();
    let p = super::property(&vnn, 30).unwrap();
    let spec = extract_bound(&raw, Some(&p.singleton_layout)).unwrap();
    assert!(unchanged_sources(&onnx, &vnn, &spec, &p));
    std::fs::write(&vnn, property.replace("(<= Y_0 Y_1)", "(>= Y_0 Y_1)")).unwrap();
    assert!(!unchanged_sources(&onnx, &vnn, &spec, &p));
    std::fs::write(&vnn, &property).unwrap();
    std::fs::write(&onnx, model(-1.).encode_to_vec()).unwrap();
    assert!(!unchanged_sources(&onnx, &vnn, &spec, &p));
}

#[test]
fn original_scalar_and_typed_tensor_surfaces_use_their_own_raw_contracts() {
    let scalar = declaration_property();
    let typed = typed_property();
    let a = property_from_source(&scalar, 30).unwrap();
    let b = property_from_source(&typed, 30).unwrap();
    assert_eq!(a.lower, b.lower);
    assert_eq!(a.upper, b.upper);
    assert_eq!((a.target, a.adversary), (b.target, b.adversary));
    assert!(
        ny_onnx::vnnlib::parse_vnnlib_with_certified_singleton_batch_property(&scalar).is_err()
    );
    assert!(
        ny_onnx::vnnlib::parse_vnnlib_with_certified_scalar_singleton_property(&typed, 30).is_err()
    );
    assert_ne!(a.source_digest, b.source_digest);
    assert_ne!(a.spec.version, b.spec.version);
    assert_eq!(a.format, OriginalPropertyFormat::ScalarRealV1);
    assert_eq!(b.format, OriginalPropertyFormat::TensorFloat32V2);
}

#[test]
fn legacy_flattened_coordinate_mapping_preserves_the_original_last_feature_and_classes() {
    // The actual authored mathematical margin is X_29. Restricting X_0
    // instead must not create the same positive certificate.
    let mut m = model(-2.);
    let g = m.graph.as_mut().unwrap();
    let mut w1 = vec![0.; 30 * 128];
    w1[29 * 128] = 1.;
    let mut b1 = vec![0.; 128];
    b1[0] = 2.;
    let mut w2 = vec![0.; 128 * 2];
    w2[0] = 1.;
    g.initializer[0] = tensor("w1", &[30, 128], w1);
    g.initializer[1] = tensor("b1", &[128], b1);
    g.initializer[2] = tensor("w2", &[128, 2], w2);
    let raw = m.encode_to_vec();
    let broad = declaration_property();
    for (coordinate, lo, hi, should_close) in [
        (29, "0.5", "0.75", true),
        (0, "0.5", "0.75", false),
        (29, "-0.75", "-0.5", false),
    ] {
        let content = broad
            .replace(
                &format!("(>= X_{coordinate} -1)"),
                &format!("(>= X_{coordinate} {lo})"),
            )
            .replace(
                &format!("(<= X_{coordinate} 1)"),
                &format!("(<= X_{coordinate} {hi})"),
            );
        let p = property_from_source(&content, 30).unwrap();
        let spec = extract_bound(&raw, Some(&p.singleton_layout)).unwrap();
        let outcome = ny_propagate::margin_row::verify_input_head(
            &spec,
            &p.lower,
            &p.upper,
            0,
            1,
            policy().head_groups.unwrap(),
            Instant::now() + Duration::from_secs(2),
        );
        match outcome {
            InputHeadOutcome::Unsat(_, token) => {
                assert!(should_close);
                assert_eq!(token.scope(), ny_propagate::certify::TokenScope::Instance);
                assert_eq!(token.classes(), &[1]);
                let (g3, fragment) =
                    super::super::margin_row_bab::margin_row_spec_agreement_owned(&p.spec, 0, &[1]);
                assert!(fragment.starts_with("g3=ran ") && g3.is_clear());
                assert!(ny_propagate::certify::ratify(token, &g3).is_ok());
            }
            InputHeadOutcome::Unknown(_) => assert!(!should_close),
        }
    }
}
