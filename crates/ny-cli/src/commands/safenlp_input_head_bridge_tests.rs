// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{arr1, arr2};
use ny_onnx::onnx_proto::{
    GraphProto, NodeProto, OperatorSetIdProto, TensorShapeProto, TensorTypeProto, TypeProto,
};
use ny_propagate::layers::{LinearLayer, MaxPool2dLayer, ReLULayer, ReshapeLayer};
use ny_propagate::{Layer, Network};

const PROPERTY: &str = "(vnnlib-version <2.0>)\n(declare-network N (declare-input X float32 [1, 1]) (declare-output Y float32 [1, 2]))\n(assert (>= X[0,0] -1))\n(assert (<= X[0,0] 1))\n(assert (<= Y[0,0] Y[0,1]))\n";

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
#[expect(
    clippy::field_reassign_with_default,
    reason = "Retain staged fixture construction and constructor/overwrite order."
)]
fn model_bytes() -> Vec<u8> {
    let mut g = GraphProto::default();
    g.name = "actual_singleton_bridge".into();
    g.input = vec![info("input", 1)];
    g.output = vec![info("output", 2)];
    for (name, dims, data) in [
        ("w1", vec![1, 1], vec![1.0]),
        ("b1", vec![1], vec![0.0]),
        ("w2", vec![1, 2], vec![-1.0, 0.0]),
        ("b2", vec![2], vec![0.5, 0.0]),
    ] {
        g.initializer.push(TensorProto {
            name: name.into(),
            dims,
            data_type: 1,
            float_data: data,
            ..Default::default()
        });
    }
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
    .encode_to_vec()
}
fn carry() -> (
    tempfile::TempDir,
    BoundInputHeadContinuation,
    ny_tensor::BoundedTensor,
) {
    carry_from(model_bytes(), PROPERTY)
}
fn carry_from(
    raw: Vec<u8>,
    property_text: &str,
) -> (
    tempfile::TempDir,
    BoundInputHeadContinuation,
    ny_tensor::BoundedTensor,
) {
    let dir = tempfile::tempdir().unwrap();
    let onnx = dir.path().join("net.onnx");
    let vnnlib = dir.path().join("spec.vnnlib");
    std::fs::write(&onnx, &raw).unwrap();
    std::fs::write(&vnnlib, property_text).unwrap();
    let p = property(&vnnlib, 1).unwrap();
    let model = extract_bound(&raw, Some(&p.singleton_layout)).unwrap();
    let (lo, hi) = evaluation_box(&p).unwrap();
    let authority = Instant::now() + Duration::from_secs(3);
    let (outcome, source) = ny_propagate::margin_row::verify_input_head_retained(
        &model,
        &p.lower,
        &p.upper,
        p.target,
        p.adversary,
        ny_propagate::margin_row::HeadGroupPolicy::new(1.0, 1, 1).unwrap(),
        authority,
        &lo,
        &hi,
        authority,
    );
    assert!(
        matches!(outcome, InputHeadOutcome::Unknown(_)),
        "the complete unoptimized root portfolio must be open"
    );
    let (f32_lo, f32_hi) = p.spec.split_input_bounds_f32();
    let input =
        ny_tensor::BoundedTensor::new(arr1(&f32_lo).into_dyn(), arr1(&f32_hi).into_dyn()).unwrap();
    (
        dir,
        BoundInputHeadContinuation {
            source: source.expect("remaining native allowance can retain the open source"),
            model,
            property: p,
            onnx,
            vnnlib,
            authority,
            independent_full_root_mip: Mutex::new(None),
        },
        input,
    )
}
fn scalar() -> Network {
    let mut n = Network::new();
    n.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    n.add_layer(Layer::ReLU(ReLULayer::new()));
    n.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[-1.0], [0.0]]), Some(arr1(&[0.5, 0.0]))).unwrap(),
    ));
    n.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, -1.0]]), None).unwrap(),
    ));
    n.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 1, 1])));
    n.add_layer(Layer::MaxPool2d(MaxPool2dLayer::new(
        (1, 1),
        (1, 1),
        (0, 0),
    )));
    n.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1])));
    n
}

#[test]
fn retained_cli_binding_requires_exact_property_and_actual_widened_root() {
    let (_dir, c, input) = carry();
    assert!(c.admitted_source(&c.property.spec, &input).is_some());
    let exact =
        ny_tensor::BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    assert!(
        c.admitted_source(&c.property.spec, &exact).is_none(),
        "a different root is not the consumer evaluation source"
    );
    let mut wrong = c.property.spec.clone();
    wrong.output_constraints = vec![ny_onnx::vnnlib::OutputConstraint::GreaterEq(0, 1)];
    assert!(c.admitted_source(&wrong, &input).is_none());
    let mut wrong = c.property.spec.clone();
    wrong.input_bounds[0].0 = -2.0;
    assert!(c.admitted_source(&wrong, &input).is_none());
}

#[test]
fn retained_cli_source_change_and_original_expiry_refuse_handoff_and_publication() {
    let (_dir, mut c, input) = carry();
    std::fs::write(&c.vnnlib, PROPERTY.replace("-1)", "-2)")).unwrap();
    assert!(c.admitted_source(&c.property.spec, &input).is_none());
    assert!(c.finish_original().is_none());
    std::fs::write(&c.vnnlib, PROPERTY).unwrap();
    c.authority = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(c.admitted_source(&c.property.spec, &input).is_none());
    assert!(c.finish_original().is_none());
}

#[test]
fn dependent_legacy_verified_cannot_replace_a_missing_original_source_sibling() {
    let (_dir, c, _input) = carry();
    let network = scalar();
    assert!(c.source.matches_scalar_network(&network));
    assert!(
        c.source
            .try_support_box(&[-1.0], &[-0.25], c.authority)
            .unwrap()
            > 0.0
    );
    assert!(!c.source.has_complete_source_cover(c.authority));
    c.source.mark_dependent_use();
    let legacy = super::super::unsat_receipt::UnsatReceipt::for_test();
    assert!(matches!(
        c.settle_result(VnncompResult::Unsat(legacy), None),
        VnncompResult::Unknown
    ));
    assert!(c.finish_original().is_none());
    // The right sibling contains a genuine unsafe point: M(1)=1/2-1=-1/2.
    assert!(
        c.source
            .try_support_box(&[0.0], &[1.0], c.authority)
            .unwrap()
            <= -0.5
    );
    assert!(!c.source.has_complete_source_cover(c.authority));
}

#[test]
fn independent_generic_verdict_and_trusted_sat_are_preserved_without_source_authority() {
    let (_dir, c, _input) = carry();
    assert!(!c.source.dependent_use());
    assert!(matches!(
        c.settle_result(
            VnncompResult::Unsat(super::super::unsat_receipt::UnsatReceipt::for_test()),
            None
        ),
        VnncompResult::Unsat(_)
    ));
    c.source.mark_dependent_use();
    let witness = "ordinary trusted-oracle candidate".to_owned();
    match c.settle_result(
        VnncompResult::Sat {
            witness: Some(witness.clone()),
        },
        None,
    ) {
        VnncompResult::Sat {
            witness: Some(actual),
        } => assert_eq!(actual, witness),
        _ => panic!("the source bridge cannot rewrite the ordinary SAT consumer"),
    }
}

fn safe_carry() -> (
    tempfile::TempDir,
    BoundInputHeadContinuation,
    ny_tensor::BoundedTensor,
) {
    let mut m = ModelProto::decode(model_bytes().as_slice()).unwrap();
    let g = m.graph.as_mut().unwrap();
    g.initializer[0].dims = vec![1, 2];
    g.initializer[0].float_data = vec![1.0, -1.0];
    g.initializer[1].dims = vec![2];
    g.initializer[1].float_data = vec![0.0, 1.0];
    g.initializer[2].dims = vec![2, 2];
    g.initializer[2].float_data = vec![1.0, 0.0, 0.5, 0.0];
    g.initializer[3].float_data = vec![-0.25, 0.0];
    let text = PROPERTY.replace("(<= X[0,0] 1)", "(<= X[0,0] 2)");
    carry_from(m.encode_to_vec(), &text)
}

#[test]
fn complete_source_children_publish_only_original_singleton_instance_and_actually_run_g3() {
    let (_dir, c, input) = safe_carry();
    let lo = input.lower().as_slice().unwrap()[0];
    let hi = input.upper().as_slice().unwrap()[0];
    assert!(
        c.source
            .try_support_box(&[lo], &[0.0], c.authority)
            .unwrap()
            > 0.125
    );
    c.source.mark_dependent_use();
    assert!(
        c.finish_original().is_none(),
        "one child cannot satisfy the original instance door"
    );
    assert!(
        c.source
            .try_support_box(&[0.0], &[hi], c.authority)
            .unwrap()
            > 0.125
    );
    let VnncompResult::Unsat(receipt) = c.settle_result(VnncompResult::Unknown, None) else {
        panic!("complete Source cover + actual original singleton G3 must publish")
    };
    let (classes, minimum, _source) = receipt.published_summary();
    assert_eq!(classes, vec![1]);
    assert!(minimum > 0.125 && minimum <= 0.25);
}

/// B2 (eng14 landing review): the bridge publishes an Instance `unsat` from
/// its private original-input Source partition (`finish_original`, reached
/// through `settle_result`). That class closes OUTSIDE every margin-row frame,
/// and the only consumer of the partition — the sequential input-split lane —
/// records no lane tape, so nothing else streams it: under the Proof profile
/// the bundle would carry `nodes=0` under a published `unsat`, which the
/// checker reads as a cover gap. With capture on, a refused finish (one child
/// only) claims nothing; the complete partition publishes and streams exactly
/// ONE `opaque` `input_head_bridge` node over the whole region whose classes
/// are the receipt's own. Mutation target: the `stream_opaque_claim` call in
/// `finish_original` (removing it leaves zero nodes here).
#[test]
fn bridge_published_unsat_claims_exactly_its_receipt_classes_as_one_opaque_node() {
    use ny_propagate::certify::{
        with_thread_sink, AlphaRecord, FdStateRecord, LeafRecord, NodeClose, NodeRecord, ProofSink,
        SitesRecord, VerdictRecord,
    };

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
    let (_dir, c, input) = safe_carry();
    let lo = input.lower().as_slice().unwrap()[0];
    let hi = input.upper().as_slice().unwrap()[0];
    let sink = Shared::default();
    let (published, _) = with_thread_sink(Box::new(sink.clone()), || {
        assert!(
            c.source
                .try_support_box(&[lo], &[0.0], c.authority)
                .unwrap()
                > 0.125
        );
        c.source.mark_dependent_use();
        assert!(
            c.finish_original().is_none(),
            "one child cannot satisfy the original instance door"
        );
        let refused = sink.0.lock().unwrap().clone();
        assert!(
            refused.is_empty(),
            "a refused finish claims nothing: {refused:?}"
        );
        assert!(
            c.source
                .try_support_box(&[0.0], &[hi], c.authority)
                .unwrap()
                > 0.125
        );
        c.settle_result(VnncompResult::Unknown, None)
    });
    let VnncompResult::Unsat(receipt) = published else {
        panic!("the complete Source partition must publish through the bridge (else VOID)")
    };
    let (classes, _, _) = receipt.published_summary();
    assert_eq!(classes, vec![1], "PROPERTY's one adversary (Y0 <= Y1)");
    let nodes = sink.0.lock().unwrap().clone();
    assert_eq!(nodes.len(), 1, "{nodes:?}");
    let n = &nodes[0];
    assert_eq!(n.close, NodeClose::Opaque, "{n:?}");
    assert_eq!(n.src, "input_head_bridge", "{n:?}");
    assert_eq!(n.dis.as_deref(), Some(classes.as_slice()), "{n:?}");
    assert!(
        n.lits.is_empty() && n.xlits.is_empty(),
        "whole region: {n:?}"
    );
}

#[cfg(feature = "mip")]
#[test]
fn independent_whole_root_mip_publication_requires_exact_capture_network_and_domain() {
    let (_dir, c, input) = safe_carry();
    c.source.mark_dependent_use();
    let mut original = Network::new();
    original.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0], [-1.0]]), Some(arr1(&[0.0, 1.0]))).unwrap(),
    ));
    original.add_layer(Layer::ReLU(ReLULayer::new()));
    original.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.5], [0.0, 0.0]]), Some(arr1(&[-0.25, 0.0]))).unwrap(),
    ));
    let text = "typed MIP publication identity for route-control test";
    let receipt = || super::super::beta_crown::test_mip_publication_receipt(text);
    assert!(
        !c.note_independent_full_root_mip_publication(
            &scalar(),
            &c.property.spec,
            &input,
            receipt()
        ),
        "an augmented objective has no independent whole-root MIP provenance"
    );
    let child =
        ny_tensor::BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[0.0]).into_dyn()).unwrap();
    assert!(!c.note_independent_full_root_mip_publication(
        &original,
        &c.property.spec,
        &child,
        receipt()
    ));
    assert!(!c.independent_mip_capture_matches(Some(text)));
    // Exercises routing with explicit test-only receipt construction; this
    // test does not execute MIP. The independently authored original margin
    // is ReLU(x)+1/2 ReLU(1-x)-1/4, whose true minimum is1/4.
    assert!(c.note_independent_full_root_mip_publication(
        &original,
        &c.property.spec,
        &input,
        receipt()
    ));
    assert!(c.independent_mip_capture_matches(Some(text)));
    assert!(!c.independent_mip_capture_matches(None));
    assert!(!c.independent_mip_capture_matches(Some("overwritten generic capture")));
    let verdict = || VnncompResult::Unsat(super::super::unsat_receipt::UnsatReceipt::for_test());
    assert!(matches!(
        c.settle_result(verdict(), None),
        VnncompResult::Unknown
    ));
    assert!(matches!(
        c.settle_result(verdict(), Some("overwritten generic capture")),
        VnncompResult::Unknown
    ));
    assert!(matches!(c.settle_result(verdict(),Some(text)),VnncompResult::Unsat(_)),"authentic independent whole-root publication retains its existing authority without a Source cover upgrade");
    assert!(c.finish_original().is_none());
    std::fs::write(&c.vnnlib, PROPERTY.replace("-1)", "-2)")).unwrap();
    assert!(matches!(
        c.settle_result(verdict(), Some(text)),
        VnncompResult::Unknown
    ));
}
