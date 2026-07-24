// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{onnx_proto::NodeProto, CustomOpHandler};
use ndarray::{arr1, ArrayD, IxDyn};
use ny_core::{LayerType, NyError};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

fn make_node(op_type: &str) -> NodeProto {
    NodeProto {
        input: vec!["input".to_string()],
        output: vec!["output".to_string()],
        name: "node".to_string(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    }
}

struct CustomFooHandler;

impl CustomOpHandler for CustomFooHandler {
    fn try_convert(&self, node: &NodeProto) -> Option<LayerSpec> {
        if node.op_type == "CustomFoo" {
            return Some(LayerSpec {
                name: "custom_foo".to_string(),
                layer_type: LayerType::ReLU,
                inputs: node.input.clone(),
                outputs: node.output.clone(),
                weights: None,
                attributes: HashMap::new(),
            });
        }
        None
    }

    fn supports(&self, op_type: &str) -> bool {
        op_type == "CustomFoo"
    }
}

struct CustomFooSecondHandler;

impl CustomOpHandler for CustomFooSecondHandler {
    fn try_convert(&self, node: &NodeProto) -> Option<LayerSpec> {
        if node.op_type == "CustomFoo" {
            return Some(LayerSpec {
                name: "custom_foo_second".to_string(),
                layer_type: LayerType::GELU,
                inputs: node.input.clone(),
                outputs: node.output.clone(),
                weights: None,
                attributes: HashMap::new(),
            });
        }
        None
    }
}

struct CapturingHandler {
    seen_version: Arc<Mutex<Option<i64>>>,
}

impl CustomOpHandler for CapturingHandler {
    fn try_convert(&self, node: &NodeProto) -> Option<LayerSpec> {
        self.try_convert_with_context(node, None)
    }

    fn try_convert_with_context(
        &self,
        _node: &NodeProto,
        opset_version: Option<i64>,
    ) -> Option<LayerSpec> {
        *self.seen_version.lock().expect("lock opset version") = opset_version;
        Some(LayerSpec {
            name: "captured".to_string(),
            layer_type: LayerType::ReLU,
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        })
    }

    fn supports(&self, op_type: &str) -> bool {
        op_type == "CustomFoo"
    }
}

#[test]
fn custom_op_handler_overrides_op_map() {
    let registry = CustomOpRegistry::from_handlers(vec![Arc::new(CustomFooHandler)]);
    let node = make_node("CustomFoo");
    let layer = convert_node_to_layer(&node, &registry, &HashMap::new())
        .expect("convert should succeed")
        .expect("custom handler should return a layer");
    assert_eq!(layer.name, "custom_foo");
    assert_eq!(layer.layer_type, LayerType::ReLU);
}

#[test]
fn custom_op_registry_order_is_deterministic() {
    let registry = CustomOpRegistry::from_handlers(vec![
        Arc::new(CustomFooHandler),
        Arc::new(CustomFooSecondHandler),
    ]);
    let node = make_node("CustomFoo");
    let layer = convert_node_to_layer(&node, &registry, &HashMap::new())
        .expect("convert should succeed")
        .expect("custom handler should return a layer");
    assert_eq!(layer.name, "custom_foo");
}

#[test]
fn custom_handler_sees_default_domain_opset_alias() {
    let seen_version = Arc::new(Mutex::new(None));
    let handler = CapturingHandler {
        seen_version: Arc::clone(&seen_version),
    };
    let registry = CustomOpRegistry::from_handlers(vec![Arc::new(handler)]);
    let node = make_node("CustomFoo");
    let mut opset_imports = HashMap::new();
    opset_imports.insert("ai.onnx".to_string(), 17);

    let layer = convert_node_to_layer(&node, &registry, &opset_imports)
        .expect("conversion should succeed")
        .expect("handler should return a layer");

    assert_eq!(layer.name, "captured");
    assert_eq!(
        seen_version.lock().expect("lock opset version").clone(),
        Some(17)
    );
}

#[test]
fn resize_is_mapped_to_layer_spec() {
    let registry = CustomOpRegistry::default();
    let node = NodeProto {
        input: vec!["x".to_string(), "roi".to_string(), "scales".to_string()],
        output: vec!["y".to_string()],
        name: "resize".to_string(),
        op_type: "Resize".to_string(),
        domain: String::new(),
        attribute: vec![],
    };
    let layer = convert_node_to_layer(&node, &registry, &HashMap::new())
        .expect("Resize conversion should succeed")
        .expect("Resize should produce a layer");
    assert_eq!(layer.layer_type, LayerType::Resize);
}

#[test]
fn upsample_is_mapped_to_layer_spec() {
    let registry = CustomOpRegistry::default();
    let node = NodeProto {
        input: vec!["x".to_string(), "scales".to_string()],
        output: vec!["y".to_string()],
        name: "upsample".to_string(),
        op_type: "Upsample".to_string(),
        domain: String::new(),
        attribute: vec![],
    };
    let layer = convert_node_to_layer(&node, &registry, &HashMap::new())
        .expect("Upsample conversion should succeed")
        .expect("Upsample should produce a layer");
    assert_eq!(layer.layer_type, LayerType::Resize);
}

#[test]
fn compare_nodes_receive_compare_op_attribute_4269() {
    let registry = CustomOpRegistry::default();
    let mut node = make_node("Greater");
    node.input = vec!["lhs".to_string(), "rhs".to_string()];

    let layer = convert_node_to_layer(&node, &registry, &HashMap::new())
        .expect("Greater conversion should succeed")
        .expect("Greater should produce a layer");

    assert_eq!(layer.layer_type, LayerType::Compare);
    assert_eq!(
        layer.attributes.get("compare_op"),
        Some(&AttributeValue::String("Gt".to_string()))
    );
}

#[test]
fn custom_domain_missing_registration_is_error() {
    let registry = CustomOpRegistry::default();
    let node = NodeProto {
        input: vec!["input".to_string()],
        output: vec!["output".to_string()],
        name: "node".to_string(),
        op_type: "MyCustomOp".to_string(),
        domain: "custom".to_string(),
        attribute: Vec::new(),
    };
    let mut opset_imports = HashMap::new();
    opset_imports.insert("custom".to_string(), 7);
    let err = convert_node_to_layer(&node, &registry, &opset_imports).unwrap_err();
    match err {
        NyError::UnsupportedConfiguration(message) => {
            assert!(message.contains("domain=\"custom\""));
            assert!(message.contains("op_type=\"MyCustomOp\""));
            assert!(message.contains("opset_version=7"));
            assert!(message.contains("CustomOpHandler"));
        }
        _ => {
            unreachable!("expected UnsupportedConfiguration for missing custom op registration")
        }
    }
}

#[test]
fn custom_domain_missing_opset_reports_unknown() {
    let registry = CustomOpRegistry::default();
    let node = NodeProto {
        input: vec!["input".to_string()],
        output: vec!["output".to_string()],
        name: "node".to_string(),
        op_type: "MyCustomOp".to_string(),
        domain: "custom".to_string(),
        attribute: Vec::new(),
    };
    let err = convert_node_to_layer(&node, &registry, &HashMap::new()).unwrap_err();
    match err {
        NyError::UnsupportedConfiguration(message) => {
            assert!(message.contains("opset import"));
            assert!(message.contains("domain=\"custom\""));
            assert!(message.contains("op_type=\"MyCustomOp\""));
            assert!(message.contains("opset_version=unknown"));
            assert!(message.contains("CustomOpHandler"));
        }
        other => unreachable!("expected UnsupportedConfiguration, got {other:?}"),
    }
}

/// Slice maps to data-path, unknown ops error, skip ops return None. (#2931)
#[test]
fn op_map_categories_2931() {
    let reg = CustomOpRegistry::default();
    let s = convert_node_to_layer(&make_node("Slice"), &reg, &HashMap::new())
        .unwrap()
        .unwrap();
    assert_eq!(s.layer_type, LayerType::Slice);
    let e = convert_node_to_layer(&make_node("NotARealOp"), &reg, &HashMap::new());
    assert!(matches!(e, Err(NyError::UnsupportedOp(_))));
    let shape = convert_node_to_layer(&make_node("Shape"), &reg, &HashMap::new())
        .unwrap()
        .expect("Shape should stay in the graph for const folding");
    assert_eq!(shape.layer_type, LayerType::Shape);
    for op in &["Constant", "Identity", "Cast", "Dropout"] {
        let r = convert_node_to_layer(&make_node(op), &reg, &HashMap::new()).unwrap();
        assert!(r.is_none(), "{op} should be skipped");
    }
    let expand = convert_node_to_layer(&make_node("Expand"), &reg, &HashMap::new())
        .unwrap()
        .expect("Expand should stay in the graph");
    assert_eq!(expand.layer_type, LayerType::Expand);
    let tile = convert_node_to_layer(&make_node("Tile"), &reg, &HashMap::new())
        .unwrap()
        .expect("Tile should stay in the graph");
    assert_eq!(tile.layer_type, LayerType::Tile);
}

#[test]
fn tile_repeats_input_normalizes_to_single_unbatched_axis() {
    let mut nodes = vec![NodeProto {
        input: vec!["data".to_string(), "repeats".to_string()],
        output: vec!["out".to_string()],
        name: "tile".to_string(),
        op_type: "Tile".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    }];
    let mut weights = WeightStore::new();
    weights.insert_integers("repeats".to_string(), arr1(&[1_i64, 1, 2, 1, 1]).into_dyn());
    let tensor_shapes = HashMap::from([("data".to_string(), vec![1, 2, 1, 4, 64])]);

    let layers = convert_graph_to_layers(
        &mut nodes,
        &mut weights,
        &CustomOpRegistry::default(),
        &HashMap::new(),
        &tensor_shapes,
        &std::collections::HashSet::new(),
        false,
    )
    .expect("Tile with a single non-unit repeat axis should normalize");

    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].layer_type, LayerType::Tile);
    assert_eq!(layers[0].attributes["axis"], AttributeValue::Int(1));
    assert_eq!(layers[0].attributes["reps"], AttributeValue::Int(2));
}

#[test]
fn conv_with_rank3_kernel_normalizes_to_conv1d_3500() {
    let mut nodes = vec![NodeProto {
        input: vec!["data".to_string(), "kernel".to_string()],
        output: vec!["out".to_string()],
        name: "conv".to_string(),
        op_type: "Conv".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    }];
    let mut weights = WeightStore::new();
    weights.insert("kernel".to_string(), ArrayD::zeros(IxDyn(&[4, 2, 7])));

    let layers = convert_graph_to_layers(
        &mut nodes,
        &mut weights,
        &CustomOpRegistry::default(),
        &HashMap::new(),
        &HashMap::new(),
        &std::collections::HashSet::new(),
        false,
    )
    .expect("Conv with a rank-3 kernel should normalize");

    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].layer_type, LayerType::Conv1d);
}

#[test]
fn conv_transpose_with_rank3_kernel_normalizes_to_conv_transpose1d_3500() {
    let mut nodes = vec![NodeProto {
        input: vec!["data".to_string(), "kernel".to_string()],
        output: vec!["out".to_string()],
        name: "deconv".to_string(),
        op_type: "ConvTranspose".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    }];
    let mut weights = WeightStore::new();
    weights.insert("kernel".to_string(), ArrayD::zeros(IxDyn(&[2, 4, 7])));

    let layers = convert_graph_to_layers(
        &mut nodes,
        &mut weights,
        &CustomOpRegistry::default(),
        &HashMap::new(),
        &HashMap::new(),
        &std::collections::HashSet::new(),
        false,
    )
    .expect("ConvTranspose with a rank-3 kernel should normalize");

    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].layer_type, LayerType::ConvTranspose1d);
}

struct SupportsButReturnsNone;

impl CustomOpHandler for SupportsButReturnsNone {
    fn try_convert(&self, _node: &NodeProto) -> Option<LayerSpec> {
        None
    }

    fn supports_with_context(
        &self,
        op_type: &str,
        _domain: &str,
        _opset_version: Option<i64>,
    ) -> bool {
        op_type == "MyCustomOp"
    }
}

#[test]
fn custom_handler_supports_but_returns_none_is_error() {
    let handler = Arc::new(SupportsButReturnsNone);
    let registry = CustomOpRegistry::from_handlers(vec![handler]);
    let node = NodeProto {
        input: vec!["input".to_string()],
        output: vec!["output".to_string()],
        name: "node".to_string(),
        op_type: "MyCustomOp".to_string(),
        domain: "custom".to_string(),
        attribute: Vec::new(),
    };
    let err = convert_node_to_layer(&node, &registry, &HashMap::new()).unwrap_err();
    match err {
        NyError::UnsupportedConfiguration(message) => {
            assert!(message.contains("claimed support"));
            assert!(message.contains("domain=\"custom\""));
            assert!(message.contains("op_type=\"MyCustomOp\""));
        }
        _ => unreachable!("expected UnsupportedConfiguration for handler returning None"),
    }
}

// ---------------------------------------------------------------------------
// Cast lowering (#cctsdb B1): integer targets -> Trunc; f32/f64 targets
// dropped; f16/bf16 targets refused (fail closed)
// ---------------------------------------------------------------------------

fn cast_node(to: i64) -> NodeProto {
    let mut node = make_node("Cast");
    node.attribute = vec![onnx_proto::AttributeProto {
        name: "to".to_string(),
        i: to,
        r#type: onnx_proto::attribute_proto::AttributeType::Int as i32,
        ..Default::default()
    }];
    node
}

/// Cast with an integer target dtype on the activation path lowers to a
/// Trunc layer spec: float->int casts truncate toward zero, so the previous
/// identity drop was unsound for fractional intervals (trunc(0.5)=0 is not
/// in [0.5, 62]).
#[test]
fn cast_to_int_lowers_to_trunc() {
    let registry = CustomOpRegistry::from_handlers(vec![]);
    for to in [2_i64, 3, 4, 5, 6, 7, 12, 13] {
        let layer = convert_node_to_layer(&cast_node(to), &registry, &HashMap::new())
            .expect("convert should succeed")
            .unwrap_or_else(|| panic!("Cast to={to} must produce a Trunc layer, not be dropped"));
        assert_eq!(layer.layer_type, LayerType::Trunc, "Cast to={to}");
        assert_eq!(layer.inputs, vec!["input".to_string()]);
        assert_eq!(layer.outputs, vec!["output".to_string()]);
    }
}

/// Cast with a full-precision float target stays an identity drop (all bound
/// math is f32, so a cast to f32/f64 preserves values exactly).
#[test]
fn cast_to_float_stays_identity_drop() {
    let registry = CustomOpRegistry::from_handlers(vec![]);
    // FLOAT=1, DOUBLE=11
    for to in [1_i64, 11] {
        let layer = convert_node_to_layer(&cast_node(to), &registry, &HashMap::new())
            .expect("convert should succeed");
        assert!(layer.is_none(), "Cast to={to} (float) must stay dropped");
    }
}

/// Cast with a reduced-precision float target (FLOAT16=10, BFLOAT16=16) must
/// NOT be dropped as identity: f16/bf16 rounding is up to 2^-11 relative and
/// is not modeled. The node lowers to a Cast layer that ny-build's
/// `convert_layer` refuses with `UnsupportedOp`, so the permissive graph
/// build degrades it to a sound OpaqueSkip [-inf, +inf] (fail closed).
#[test]
fn cast_to_f16_bf16_is_refused_not_identity() {
    let registry = CustomOpRegistry::from_handlers(vec![]);
    for to in [10_i64, 16] {
        let layer = convert_node_to_layer(&cast_node(to), &registry, &HashMap::new())
            .expect("convert should succeed")
            .unwrap_or_else(|| panic!("Cast to={to} (f16/bf16) must not be identity-dropped"));
        assert_eq!(layer.layer_type, LayerType::Cast, "Cast to={to}");
        assert_eq!(layer.inputs, vec!["input".to_string()]);
        assert_eq!(layer.outputs, vec!["output".to_string()]);
    }
}

/// Cast to BOOL keeps the legacy identity drop (cast-to-bool is x != 0, not
/// truncation; changing it is a separate mask-propagation feature).
#[test]
fn cast_to_bool_stays_identity_drop() {
    let registry = CustomOpRegistry::from_handlers(vec![]);
    let layer = convert_node_to_layer(&cast_node(9), &registry, &HashMap::new())
        .expect("convert should succeed");
    assert!(layer.is_none(), "Cast to BOOL must keep the legacy drop");
}

/// End-to-end pin for the f16 fail-closed path: an ACTIVATION-path f32->f16->
/// f32 Cast round-trip (the quantization pattern the identity drop used to
/// erase) must degrade to conservative [-inf, +inf] bounds via the graph
/// build's OpaqueSkip catch-all — NOT propagate the finite bounds an
/// exact-identity treatment would produce.
#[test]
fn f16_cast_round_trip_degrades_to_conservative_bounds() {
    use crate::onnx_proto;
    use prost::Message;

    let dim = |v: i64| onnx_proto::tensor_shape_proto::Dimension {
        value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
            v,
        )),
    };
    let vinfo = |name: &str, elem: i32| onnx_proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type: elem,
                shape: Some(onnx_proto::TensorShapeProto {
                    dim: vec![dim(1), dim(2)],
                }),
            }),
        }),
    };
    let cast = |name: &str, input: &str, output: &str, to: i64| {
        let mut node = cast_node(to);
        node.name = name.to_string();
        node.input = vec![input.to_string()];
        node.output = vec![output.to_string()];
        node
    };
    let model = onnx_proto::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(onnx_proto::GraphProto {
            node: vec![
                cast("to_f16", "X", "h", 10), // FLOAT16
                cast("to_f32", "h", "Y", 1),  // FLOAT
            ],
            name: "f16_round_trip".to_string(),
            input: vec![vinfo("X", 1)],
            output: vec![vinfo("Y", 1)],
            ..Default::default()
        }),
        ..Default::default()
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("f16_round_trip.onnx");
    std::fs::write(&path, model.encode_to_vec()).expect("write model");

    let loaded = crate::load_onnx(&path).expect("load");
    let graph = loaded.to_graph_network().expect("graph build");
    let input = ny_tensor::BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();
    let out = graph.propagate_ibp(&input).expect("ibp");
    assert!(
        out.upper().iter().all(|v| *v == f32::INFINITY)
            && out.lower().iter().all(|v| *v == f32::NEG_INFINITY),
        "f16 round-trip Cast must degrade to conservative [-inf, +inf] bounds, got {:?}..{:?}",
        out.lower(),
        out.upper()
    );
}
