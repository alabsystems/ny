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

fn main_opset(version: i64) -> HashMap<String, i64> {
    HashMap::from([(String::new(), version), ("ai.onnx".to_string(), version)])
}

fn normalization_node(op_type: &str) -> NodeProto {
    let mut node = make_node(op_type);
    node.input = vec!["x".to_string(), "scale".to_string()];
    node.output = vec!["y".to_string()];
    node
}

fn int_attribute(name: &str, value: i64) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        i: Some(value),
        r#type: onnx_proto::attribute_type::INT,
        ..Default::default()
    }
}

fn float_attribute(name: &str, value: f32) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        f: Some(value),
        r#type: onnx_proto::attribute_type::FLOAT,
        ..Default::default()
    }
}

fn assert_normalization_rejected(node: &NodeProto, opset: i64) {
    let error = convert_node_to_layer(node, &CustomOpRegistry::default(), &main_opset(opset), None)
        .expect_err("unsupported normalization semantics must fail closed");
    assert!(matches!(error, NyError::UnsupportedOp(_)), "{error}");
}

#[test]
fn live_expand_authentication_requires_complete_direct_shape_reference() {
    let mut shape = make_node("Shape");
    shape.name = "shape".to_string();
    shape.input = vec!["reference".to_string()];
    shape.output = vec!["target_shape".to_string()];
    let mut expand = make_node("Expand");
    expand.name = "expand".to_string();
    expand.input = vec!["source".to_string(), "target_shape".to_string()];
    expand.output = vec!["expanded".to_string()];
    let nodes = vec![shape.clone(), expand.clone()];
    let producers = HashMap::from([("target_shape", 0_usize)]);
    let shapes = HashMap::from([
        ("source".to_string(), vec![1, 8, 1]),
        ("reference".to_string(), vec![1, 8, -1]),
    ]);
    let weights = WeightStore::new();

    assert_eq!(
        authenticate_live_shape_expand(&expand, &nodes, &producers, &weights, &shapes).unwrap(),
        Some("reference".to_string())
    );

    let mut sliced_shape = shape;
    sliced_shape.op_type = "Gather".to_string();
    let nodes = vec![sliced_shape, expand.clone()];
    assert_eq!(
        authenticate_live_shape_expand(&expand, &nodes, &producers, &weights, &shapes).unwrap(),
        None,
        "first-input provenance through shape arithmetic is not a target-value proof"
    );

    let mismatched_shapes = HashMap::from([
        ("source".to_string(), vec![1, 7, 1]),
        ("reference".to_string(), vec![1, 8, -1]),
    ]);
    let nodes = vec![
        {
            let mut direct = make_node("Shape");
            direct.input = vec!["reference".to_string()];
            direct.output = vec!["target_shape".to_string()];
            direct
        },
        expand.clone(),
    ];
    assert_eq!(
        authenticate_live_shape_expand(&expand, &nodes, &producers, &weights, &mismatched_shapes,)
            .unwrap(),
        None,
        "the narrow runtime layer cannot represent prefix broadcasting"
    );
}

fn convert_and_authenticate_softmax(
    node: &NodeProto,
    opset: i64,
    input_shape: Option<Vec<i64>>,
) -> Result<LayerSpec> {
    let opsets = main_opset(opset);
    let mut layer = convert_node_to_layer(node, &CustomOpRegistry::default(), &opsets, None)?
        .expect("Softmax-family node should produce a layer");
    let shapes = input_shape
        .map(|shape| HashMap::from([("input".to_string(), shape)]))
        .unwrap_or_default();
    authenticate_standard_softmax_semantics(node, &mut layer, &opsets, &shapes)?;
    Ok(layer)
}

#[test]
fn versioned_softmax_defaults_and_legacy_flattening_fail_closed() {
    for op_type in ["Softmax", "LogSoftmax"] {
        let node = make_node(op_type);

        let legacy = convert_and_authenticate_softmax(&node, 11, Some(vec![2, 10]))
            .expect("legacy rank-two default axis is exactly representable");
        assert_eq!(legacy.attributes["axis"], AttributeValue::Int(1));

        let error = convert_and_authenticate_softmax(&node, 12, Some(vec![2, 3, 4]))
            .expect_err("legacy default axis flattens two suffix dimensions");
        assert!(error.to_string().contains("flattens multiple"), "{error}");

        let mut trailing = node.clone();
        trailing.attribute.push(int_attribute("axis", -1));
        let legacy_trailing = convert_and_authenticate_softmax(&trailing, 11, None)
            .expect("legacy axis -1 is single-axis for every positive rank");
        assert_eq!(legacy_trailing.attributes["axis"], AttributeValue::Int(-1));

        let modern = convert_and_authenticate_softmax(&node, 13, Some(vec![2, 3, 4]))
            .expect("modern default acts on the final axis");
        assert_eq!(modern.attributes["axis"], AttributeValue::Int(-1));
    }
}

#[test]
fn softmax_schema_rejects_unknown_malformed_and_duplicate_attributes() {
    for attributes in [
        vec![float_attribute("axis", -1.0)],
        vec![int_attribute("axis", -1), int_attribute("axis", -1)],
        vec![int_attribute("unknown", 0)],
    ] {
        let mut node = make_node("Softmax");
        node.attribute = attributes;
        assert!(convert_and_authenticate_softmax(&node, 13, Some(vec![2, 3])).is_err());
    }
}

#[test]
fn direct_layer_norm_canonicalizes_supported_optional_placeholders() {
    let mut node = normalization_node("LayerNormalization");
    node.input.push(String::new());
    node.output.extend([String::new(), String::new()]);
    node.attribute = vec![
        int_attribute("axis", -1),
        float_attribute("epsilon", 1e-5),
        int_attribute("stash_type", 1),
    ];

    let layer = convert_node_to_layer(&node, &CustomOpRegistry::default(), &main_opset(17), None)
        .expect("supported LayerNormalization should convert")
        .expect("LayerNormalization should produce a layer");
    assert_eq!(layer.layer_type, LayerType::LayerNorm);
    assert_eq!(layer.inputs, ["x", "scale"]);
    assert_eq!(layer.outputs, ["y"]);
}

#[test]
fn direct_normalization_rejects_unsupported_axes_and_attribute_encodings() {
    let mut node = normalization_node("LayerNormalization");
    node.attribute.push(int_attribute("axis", 1));
    assert_normalization_rejected(&node, 17);

    node.attribute = vec![float_attribute("axis", -1.0)];
    assert_normalization_rejected(&node, 17);

    node.attribute = vec![int_attribute("axis", -1), int_attribute("axis", -1)];
    assert_normalization_rejected(&node, 17);

    node.attribute = vec![int_attribute("unknown", 1)];
    assert_normalization_rejected(&node, 17);
}

#[test]
fn direct_normalization_rejects_unrepresented_precision_and_epsilon() {
    let mut node = normalization_node("LayerNormalization");
    node.attribute = vec![int_attribute("stash_type", 10)];
    assert_normalization_rejected(&node, 17);

    node.attribute = vec![float_attribute("stash_type", 1.0)];
    assert_normalization_rejected(&node, 17);

    for epsilon in [
        0.0,
        f32::from_bits(NORMALIZATION_MIN_EPS.to_bits() - 1),
        f32::INFINITY,
        f32::NAN,
    ] {
        node.attribute = vec![float_attribute("epsilon", epsilon)];
        assert_normalization_rejected(&node, 17);
    }

    node.attribute = vec![int_attribute("epsilon", 1)];
    assert_normalization_rejected(&node, 17);
}

#[test]
fn direct_normalization_rejects_unsupported_signatures_and_statistic_outputs() {
    let mut layer_norm = normalization_node("LayerNormalization");
    layer_norm.input = vec!["x".to_string()];
    assert_normalization_rejected(&layer_norm, 17);

    layer_norm = normalization_node("LayerNormalization");
    layer_norm.input[1].clear();
    assert_normalization_rejected(&layer_norm, 17);

    layer_norm = normalization_node("LayerNormalization");
    layer_norm.output.push("mean".to_string());
    assert_normalization_rejected(&layer_norm, 17);

    let mut rms = normalization_node("RMSNormalization");
    rms.input.push("extra".to_string());
    assert_normalization_rejected(&rms, 23);

    rms = normalization_node("RMSNormalization");
    rms.output.push(String::new());
    assert_normalization_rejected(&rms, 23);
}

#[test]
fn direct_normalization_enforces_operator_versions_and_main_domain() {
    assert_normalization_rejected(&normalization_node("LayerNormalization"), 16);
    assert_normalization_rejected(&normalization_node("RMSNormalization"), 22);

    let simplified = normalization_node("SimplifiedLayerNormalization");
    let layer = convert_node_to_layer(
        &simplified,
        &CustomOpRegistry::default(),
        &main_opset(1),
        None,
    )
    .expect("legacy SimplifiedLayerNormalization v1 should convert")
    .expect("SimplifiedLayerNormalization should produce a layer");
    assert_eq!(layer.layer_type, LayerType::RMSNorm);

    let mut wrong_domain = normalization_node("LayerNormalization");
    wrong_domain.domain = "ai.onnx.ml".to_string();
    let opsets = HashMap::from([("ai.onnx.ml".to_string(), 17)]);
    let error = convert_node_to_layer(&wrong_domain, &CustomOpRegistry::default(), &opsets, None)
        .expect_err("ai.onnx.ml lookalike must not use main-domain semantics");
    assert!(matches!(error, NyError::UnsupportedConfiguration(_)));
}

#[test]
fn core_operator_lookalikes_in_ai_onnx_ml_require_explicit_registration() {
    let mut erf = make_node("Erf");
    erf.input = vec!["x".to_string()];
    erf.output = vec!["y".to_string()];
    erf.domain = "ai.onnx.ml".to_string();
    let opsets = HashMap::from([("ai.onnx.ml".to_string(), 3)]);

    let error = convert_node_to_layer(&erf, &CustomOpRegistry::default(), &opsets, None)
        .expect_err("ai.onnx.ml Erf lookalike must not use core-domain semantics");
    assert!(matches!(error, NyError::UnsupportedConfiguration(_)));
}

#[test]
fn direct_simplified_layer_norm_canonicalizes_empty_statistic_output() {
    let mut node = normalization_node("SimplifiedLayerNormalization");
    node.output.push(String::new());
    let layer = convert_node_to_layer(&node, &CustomOpRegistry::default(), &main_opset(1), None)
        .expect("empty optional inv_std_var should be accepted")
        .expect("SimplifiedLayerNormalization should produce a layer");
    assert_eq!(layer.outputs, ["y"]);

    node.output[1] = "inv_std_var".to_string();
    assert_normalization_rejected(&node, 1);
}

#[test]
fn adjacent_direct_normalization_schemas_accept_supported_subset() {
    let mut instance = normalization_node("InstanceNormalization");
    instance.input.push("bias".to_string());
    let layer = convert_node_to_layer(
        &instance,
        &CustomOpRegistry::default(),
        &main_opset(13),
        None,
    )
    .expect("valid InstanceNormalization should convert")
    .expect("InstanceNormalization should produce a layer");
    assert_eq!(layer.layer_type, LayerType::InstanceNorm);

    let mut group = normalization_node("GroupNormalization");
    group.input.push("bias".to_string());
    group.attribute = vec![
        int_attribute("num_groups", 2),
        int_attribute("stash_type", 1),
    ];
    let layer = convert_node_to_layer(&group, &CustomOpRegistry::default(), &main_opset(21), None)
        .expect("valid GroupNormalization should convert")
        .expect("GroupNormalization should produce a layer");
    assert_eq!(layer.layer_type, LayerType::GroupNorm);
}

#[test]
fn batch_norm_schema_is_opset_exact_and_canonicalizes_empty_outputs() {
    let mut node = normalization_node("BatchNormalization");
    node.input
        .extend(["bias".to_string(), "mean".to_string(), "var".to_string()]);
    node.output
        .extend([String::new(), String::new(), String::new(), String::new()]);
    let layer = convert_node_to_layer(&node, &CustomOpRegistry::default(), &main_opset(9), None)
        .expect("opset-9 inference placeholders should convert")
        .expect("BatchNormalization should produce a layer");
    assert_eq!(layer.layer_type, LayerType::BatchNorm);
    assert_eq!(layer.outputs, ["y"]);

    assert_normalization_rejected(&node, 8);

    let mut opset14 = node.clone();
    opset14.output.truncate(3);
    opset14.attribute.push(int_attribute("training_mode", 0));
    convert_node_to_layer(
        &opset14,
        &CustomOpRegistry::default(),
        &main_opset(14),
        None,
    )
    .expect("opset-14 inference placeholders and training_mode=0 should convert")
    .expect("BatchNormalization should produce a layer");

    let mut training_attr_on_old_schema = node.clone();
    training_attr_on_old_schema
        .attribute
        .push(int_attribute("training_mode", 0));
    assert_normalization_rejected(&training_attr_on_old_schema, 13);

    let mut too_many_new_outputs = opset14;
    too_many_new_outputs.output.push(String::new());
    assert_normalization_rejected(&too_many_new_outputs, 14);

    let mut requested_statistic = node;
    requested_statistic.output[1] = "running_mean".to_string();
    assert_normalization_rejected(&requested_statistic, 9);
}

#[test]
fn adjacent_direct_normalization_rejects_malformed_semantics() {
    let mut instance = normalization_node("InstanceNormalization");
    instance.input.push("bias".to_string());
    instance.attribute.push(int_attribute("stash_type", 1));
    assert_normalization_rejected(&instance, 13);

    let mut group = normalization_node("GroupNormalization");
    group.input.push("bias".to_string());
    assert_normalization_rejected(&group, 21);
    group.attribute = vec![int_attribute("num_groups", 0)];
    assert_normalization_rejected(&group, 21);
    group.attribute = vec![
        int_attribute("num_groups", 2),
        int_attribute("stash_type", 10),
    ];
    assert_normalization_rejected(&group, 21);
}

#[test]
fn channel_normalizations_require_constant_channel_affines() {
    let shapes = HashMap::from([("x".to_string(), vec![1, 4, 3])]);
    let mut weights = WeightStore::new();
    for (name, value) in [("scale", 1.0), ("bias", 0.0)] {
        weights.insert(name.to_string(), ArrayD::from_elem(IxDyn(&[4]), value));
    }

    let mut instance = normalization_node("InstanceNormalization");
    instance.input.push("bias".to_string());
    authenticate_direct_normalization_parameters(&instance, &weights, &shapes)
        .expect("InstanceNormalization channel parameters should authenticate");

    let mut group = normalization_node("GroupNormalization");
    group.input.push("bias".to_string());
    group.attribute.push(int_attribute("num_groups", 2));
    authenticate_direct_normalization_parameters(&group, &weights, &shapes)
        .expect("divisible GroupNormalization parameters should authenticate");
    group.attribute[0].i = Some(3);
    assert!(authenticate_direct_normalization_parameters(&group, &weights, &shapes).is_err());

    let ambiguous_image_shapes = HashMap::from([("x".to_string(), vec![1, 4, 4, 2])]);
    assert!(
        authenticate_direct_normalization_parameters(&group, &weights, &ambiguous_image_shapes)
            .is_err(),
        "GroupNorm's [C,T] implementation must reject raw [N,C,H,W] even when C == H"
    );

    let mut wrong_scale = WeightStore::new();
    wrong_scale.insert(
        "scale".to_string(),
        ArrayD::from_elem(IxDyn(&[1, 4, 1]), 1.0),
    );
    wrong_scale.insert("bias".to_string(), ArrayD::zeros(IxDyn(&[4])));
    assert!(
        authenticate_direct_normalization_parameters(&instance, &wrong_scale, &shapes).is_err()
    );
}

#[test]
fn normalization_parameter_authentication_is_standard_domain_only_and_fail_closed() {
    let mut custom = make_node("InstanceNormalization");
    custom.domain = "vendor.example".to_string();
    custom.input.clear();
    authenticate_direct_normalization_parameters(&custom, &WeightStore::new(), &HashMap::new())
        .expect("the registered custom-domain handler owns its schema");

    custom.domain.clear();
    assert!(
        authenticate_direct_normalization_parameters(
            &custom,
            &WeightStore::new(),
            &HashMap::new(),
        )
        .is_err(),
        "a malformed core ONNX normalization must fail without indexing past its inputs"
    );
}

#[test]
fn direct_normalization_requires_constant_last_axis_affines() {
    let mut node = normalization_node("LayerNormalization");
    node.input.push("bias".to_string());
    let shapes = HashMap::from([("x".to_string(), vec![1, 2, 3])]);

    let mut weights = WeightStore::new();
    weights.insert("scale".to_string(), arr1(&[1.0, 1.0, 1.0]).into_dyn());
    weights.insert("bias".to_string(), arr1(&[0.0, 0.0, 0.0]).into_dyn());
    authenticate_direct_normalization_parameters(&node, &weights, &shapes)
        .expect("matching constant one-dimensional affines should authenticate");

    let missing_bias = {
        let mut weights = WeightStore::new();
        weights.insert("scale".to_string(), arr1(&[1.0, 1.0, 1.0]).into_dyn());
        weights
    };
    assert!(authenticate_direct_normalization_parameters(&node, &missing_bias, &shapes).is_err());

    let mut channel_broadcast = WeightStore::new();
    channel_broadcast.insert(
        "scale".to_string(),
        ArrayD::from_elem(IxDyn(&[1, 2, 1]), 1.0),
    );
    channel_broadcast.insert(
        "bias".to_string(),
        ArrayD::from_elem(IxDyn(&[1, 2, 1]), 0.0),
    );
    assert!(
        authenticate_direct_normalization_parameters(&node, &channel_broadcast, &shapes).is_err()
    );

    let dynamic_shapes = HashMap::from([("x".to_string(), vec![1, 2, -1])]);
    assert!(
        authenticate_direct_normalization_parameters(&node, &weights, &dynamic_shapes).is_err()
    );
}

#[test]
fn batch_norm_requires_four_matching_channel_vectors() {
    let mut node = normalization_node("BatchNormalization");
    node.input
        .extend(["bias".to_string(), "mean".to_string(), "var".to_string()]);
    let shapes = HashMap::from([("x".to_string(), vec![1, 2, 2])]);
    let mut weights = WeightStore::new();
    for name in ["scale", "bias", "mean", "var"] {
        weights.insert(name.to_string(), ArrayD::ones(IxDyn(&[2])));
    }
    authenticate_direct_normalization_parameters(&node, &weights, &shapes)
        .expect("matching BatchNormalization channel vectors should authenticate");

    weights.insert("var".to_string(), ArrayD::ones(IxDyn(&[2, 1])));
    assert!(authenticate_direct_normalization_parameters(&node, &weights, &shapes).is_err());
}

#[test]
fn unauthenticated_semantic_rewrites_remain_dark() {
    const { assert!(!MERGE_LINEAR_EXACT_COMPOSITION_AUTHENTICATED) };
    const { assert!(!CAUSAL_SOFTMAX_MASK_AUTHENTICATED) };
    const { assert!(!DECOMPOSED_ERF_GELU_SOURCE_AUTHENTICATED) };
    const { assert!(!DECOMPOSED_TANH_GELU_SOURCE_AUTHENTICATED) };
    const { assert!(!DECOMPOSED_INSTANCE_NORM_SOURCE_AUTHENTICATED) };
    const { assert!(!QDQ_PERTURBATION_SOURCE_AUTHENTICATED) };
    // BatchNorm folding is deliberately NOT in this list: it is governed by
    // `BatchNormFoldingPolicy` (default LegacyEnvironment = folds enabled), and
    // the hard const that used to pin it dark silently cost 15 field-confirmed
    // `unsat` rows plus the CROWN-IBP collector lane on both cifar100 resnets
    // (41 -> 61 nodes crossed the 50-node threshold). See #bn-fold-restore in
    // convert.rs. `bn_fold_policy_default_folds` below pins the default.
}

#[test]
fn bn_fold_policy_default_folds() {
    // The default policy must keep Conv/Gemm+BN folding ENABLED. If this ever
    // flips, cifar100/tinyimagenet resnets grow past the per-node CROWN-IBP
    // threshold and a verdict lane disappears without a log line.
    assert_eq!(
        BatchNormFoldingPolicy::default(),
        BatchNormFoldingPolicy::LegacyEnvironment,
    );
}

#[test]
fn qdq_fusion_preserves_quantized_graph_output_and_attributes() {
    let quant = NodeProto {
        input: vec!["x".to_string(), "scale".to_string()],
        output: vec!["quantized".to_string()],
        op_type: "QuantizeLinear".to_string(),
        ..Default::default()
    };
    let dequant = NodeProto {
        input: vec!["quantized".to_string(), "scale".to_string()],
        output: vec!["out".to_string()],
        op_type: "DequantizeLinear".to_string(),
        ..Default::default()
    };
    let nodes = vec![quant, dequant];
    let consumers = HashMap::from([
        ("x", vec![0]),
        ("scale", vec![0, 1]),
        ("quantized", vec![1]),
    ]);

    assert!(
        try_fuse_qdq_relaxation(&nodes, 0, &consumers, &std::collections::HashSet::new(),)
            .is_some()
    );
    assert!(try_fuse_qdq_relaxation(
        &nodes,
        0,
        &consumers,
        &std::collections::HashSet::from(["quantized".to_string()]),
    )
    .is_none());

    let mut attributed = nodes;
    attributed[0].attribute.push(onnx_proto::AttributeProto {
        name: "output_dtype".to_string(),
        i: Some(3),
        r#type: onnx_proto::attribute_type::INT,
        ..Default::default()
    });
    assert!(try_fuse_qdq_relaxation(
        &attributed,
        0,
        &consumers,
        &std::collections::HashSet::new(),
    )
    .is_none());
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

struct CastOverrideHandler;

impl CustomOpHandler for CastOverrideHandler {
    fn try_convert(&self, node: &NodeProto) -> Option<LayerSpec> {
        (node.op_type == "Cast").then(|| LayerSpec {
            name: "unsafe_cast_override".to_string(),
            layer_type: LayerType::ReLU,
            inputs: node.input.clone(),
            outputs: node.output.clone(),
            weights: None,
            attributes: HashMap::new(),
        })
    }

    fn supports(&self, op_type: &str) -> bool {
        op_type == "Cast"
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
    let layer = convert_node_to_layer(&node, &registry, &HashMap::new(), None)
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
    let layer = convert_node_to_layer(&node, &registry, &HashMap::new(), None)
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

    let layer = convert_node_to_layer(&node, &registry, &opset_imports, None)
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
    let layer = convert_node_to_layer(&node, &registry, &HashMap::new(), None)
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
    let layer = convert_node_to_layer(&node, &registry, &HashMap::new(), None)
        .expect("Upsample conversion should succeed")
        .expect("Upsample should produce a layer");
    assert_eq!(layer.layer_type, LayerType::Resize);
}

#[test]
fn compare_nodes_receive_compare_op_attribute_4269() {
    let registry = CustomOpRegistry::default();
    let mut node = make_node("Greater");
    node.input = vec!["lhs".to_string(), "rhs".to_string()];

    let layer = convert_node_to_layer(&node, &registry, &HashMap::new(), None)
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
    let err = convert_node_to_layer(&node, &registry, &opset_imports, None).unwrap_err();
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
    let err = convert_node_to_layer(&node, &registry, &HashMap::new(), None).unwrap_err();
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
    let s = convert_node_to_layer(&make_node("Slice"), &reg, &HashMap::new(), None)
        .unwrap()
        .unwrap();
    assert_eq!(s.layer_type, LayerType::Slice);
    let e = convert_node_to_layer(&make_node("NotARealOp"), &reg, &HashMap::new(), None);
    assert!(matches!(e, Err(NyError::UnsupportedOp(_))));
    let shape = convert_node_to_layer(&make_node("Shape"), &reg, &HashMap::new(), None)
        .unwrap()
        .expect("Shape should stay in the graph for const folding");
    assert_eq!(shape.layer_type, LayerType::Shape);
    for op in &["Constant", "Identity", "Dropout"] {
        let r = convert_node_to_layer(&make_node(op), &reg, &HashMap::new(), None).unwrap();
        assert!(r.is_none(), "{op} should be skipped");
    }
    let expand = convert_node_to_layer(&make_node("Expand"), &reg, &HashMap::new(), None)
        .unwrap()
        .expect("Expand should stay in the graph");
    assert_eq!(expand.layer_type, LayerType::Expand);
    let tile = convert_node_to_layer(&make_node("Tile"), &reg, &HashMap::new(), None)
        .unwrap()
        .expect("Tile should stay in the graph");
    assert_eq!(tile.layer_type, LayerType::Tile);

    for op in ["Attention", "MultiHeadAttention"] {
        let error = convert_node_to_layer(&make_node(op), &reg, &HashMap::new(), None)
            .expect_err("standard attention aliases must not bypass raw schema authentication");
        assert!(matches!(error, NyError::UnsupportedOp(_)), "{error}");
    }
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
        &std::collections::HashSet::new(),
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
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
        &std::collections::HashSet::new(),
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
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
        &std::collections::HashSet::new(),
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
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
    let err = convert_node_to_layer(&node, &registry, &HashMap::new(), None).unwrap_err();
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
// Cast lowering: FLOAT32 is an exact identity, integer targets lower to Trunc
// (exact trunc-toward-zero), BOOL is an identity only on a provably
// {0,1}-valued operand, and every other target fails closed as LayerType::Cast
// because ny does not retain a non-f32 arithmetic dtype.
// ---------------------------------------------------------------------------

fn cast_node(to: i64) -> NodeProto {
    let mut node = make_node("Cast");
    node.attribute = vec![onnx_proto::AttributeProto {
        name: "to".to_string(),
        i: Some(to),
        r#type: onnx_proto::attribute_proto::AttributeType::Int as i32,
        ..Default::default()
    }];
    node
}

/// A float->int Cast truncates toward zero, so it must lower to a Trunc layer,
/// NOT be dropped as an identity (trunc(0.5) = 0 is not in [0.5, 62]).
/// cctsdb_yolo_2023's position masks and the cell-enumeration driver both
/// depend on these nodes surviving as Trunc.
#[test]
fn cast_to_int_lowers_to_trunc() {
    let registry = CustomOpRegistry::from_handlers(vec![]);
    for to in [6_i64, 7] {
        let layer = convert_node_to_layer(&cast_node(to), &registry, &HashMap::new(), None)
            .unwrap_or_else(|error| panic!("to={to} must convert: {error}"))
            .unwrap_or_else(|| panic!("to={to} must not be dropped"));
        assert_eq!(
            layer.layer_type,
            LayerType::Trunc,
            "to={to} must lower to Trunc"
        );
        assert!(
            matches!(layer.attributes.get("to"), Some(AttributeValue::Int(value)) if *value == to),
            "to={to} must survive lowering so ny-build can enforce the destination domain: {:?}",
            layer.attributes
        );
    }
}

/// The narrow and unsigned integer targets are NOT admitted. `trunc` is the
/// ONNX float->int reading only in range; out-of-range is undefined and ONNX
/// Runtime wraps, and these dtypes are exactly the ones a real graph overflows.
/// No shipped VNN-COMP model casts to them, so they stay fail-closed.
#[test]
fn cast_to_narrow_or_unsigned_int_fails_closed() {
    let registry = CustomOpRegistry::from_handlers(vec![]);
    for to in [2_i64, 3, 4, 5, 12, 13] {
        let layer = convert_node_to_layer(&cast_node(to), &registry, &HashMap::new(), None)
            .unwrap_or_else(|error| panic!("to={to}: {error}"))
            .unwrap_or_else(|| panic!("to={to} must not be dropped"));
        assert_eq!(
            layer.layer_type,
            LayerType::Cast,
            "to={to} must fail closed as LayerType::Cast"
        );
    }
}

#[test]
fn graph_conversion_cannot_bypass_int_cast_rejection_with_a_materialized_output() {
    let registry = CustomOpRegistry::from_handlers(vec![Arc::new(CastOverrideHandler)]);
    let mut nodes = vec![cast_node(7)];
    let mut weights = WeightStore::new();
    weights.insert("output".to_string(), arr1(&[3.0_f32]).into_dyn());
    weights.insert_integers("output".to_string(), arr1(&[3_i64]).into_dyn());

    let error = convert_graph_to_layers(
        &mut nodes,
        &mut weights,
        &registry,
        &HashMap::new(),
        &HashMap::new(),
        &std::collections::HashSet::new(),
        &std::collections::HashSet::new(),
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
    )
    .expect_err("generic graph conversion must reject INT64 Cast regardless of staged weights");
    assert!(matches!(error, NyError::UnsupportedOp(_)), "{error}");
}

#[test]
fn graph_conversion_rejects_cast_with_empty_required_input_before_custom_override() {
    let registry = CustomOpRegistry::from_handlers(vec![Arc::new(CastOverrideHandler)]);
    let mut node = cast_node(1);
    node.input = vec![String::new()];
    let mut nodes = vec![node];

    let error = convert_graph_to_layers(
        &mut nodes,
        &mut WeightStore::new(),
        &registry,
        &HashMap::new(),
        &HashMap::new(),
        &std::collections::HashSet::new(),
        &std::collections::HashSet::new(),
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
    )
    .expect_err("Cast with an empty required input must fail before a custom override");
    assert!(matches!(error, NyError::UnsupportedOp(_)), "{error}");
}

/// Cast with a FLOAT32 target stays an identity drop.
#[test]
fn cast_to_float_stays_identity_drop() {
    let registry = CustomOpRegistry::from_handlers(vec![]);
    let layer = convert_node_to_layer(&cast_node(1), &registry, &HashMap::new(), None)
        .expect("convert should succeed");
    assert!(layer.is_none(), "Cast to FLOAT32 must stay dropped");
}

fn insert_integer_constant(weights: &mut WeightStore, name: &str, values: &[i64]) {
    weights.insert(
        name.to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[values.len()]),
            values.iter().map(|&value| value as f32).collect(),
        )
        .unwrap(),
    );
    weights.insert_integers(
        name.to_string(),
        ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).unwrap(),
    );
}

fn static_int64_cast_reshape_nodes() -> Vec<NodeProto> {
    let mut first_cast = cast_node(7);
    first_cast.name = "cast_first".to_string();
    first_cast.input = vec!["head_count".to_string()];
    first_cast.output = vec!["head_count_i64".to_string()];

    let mut second_cast = cast_node(7);
    second_cast.name = "cast_second".to_string();
    second_cast.input = vec!["head_count_i64".to_string()];
    second_cast.output = vec!["head_count_i64_again".to_string()];

    vec![
        first_cast,
        second_cast,
        NodeProto {
            input: vec!["head_count_i64_again".to_string()],
            output: vec!["head_count_vec".to_string()],
            name: "unsqueeze_head_count".to_string(),
            op_type: "Unsqueeze".to_string(),
            attribute: vec![onnx_proto::AttributeProto {
                name: "axes".to_string(),
                ints: vec![0],
                r#type: onnx_proto::attribute_type::INTS,
                ..Default::default()
            }],
            ..Default::default()
        },
        NodeProto {
            input: vec!["prefix".to_string(), "head_count_vec".to_string()],
            output: vec!["reshape_shape".to_string()],
            name: "concat_shape".to_string(),
            op_type: "Concat".to_string(),
            attribute: vec![onnx_proto::AttributeProto {
                name: "axis".to_string(),
                i: Some(0),
                r#type: onnx_proto::attribute_type::INT,
                ..Default::default()
            }],
            ..Default::default()
        },
        NodeProto {
            input: vec!["data".to_string(), "reshape_shape".to_string()],
            output: vec!["reshaped".to_string()],
            name: "reshape".to_string(),
            op_type: "Reshape".to_string(),
            ..Default::default()
        },
    ]
}

fn static_int64_cast_weights() -> WeightStore {
    let mut weights = WeightStore::new();
    for (name, values) in [
        ("head_count", &[8_i64][..]),
        ("head_count_i64", &[8_i64][..]),
        ("head_count_i64_again", &[8_i64][..]),
        ("head_count_vec", &[8_i64][..]),
        ("prefix", &[1_i64, 64][..]),
        ("reshape_shape", &[1_i64, 64, 8][..]),
    ] {
        insert_integer_constant(&mut weights, name, values);
    }
    weights
}

fn static_int64_raw_values() -> std::collections::HashSet<String> {
    [
        "head_count",
        "head_count_i64",
        "head_count_i64_again",
        "head_count_vec",
        "prefix",
        "reshape_shape",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

#[test]
fn exactly_materialized_int64_casts_are_skipped_only_on_static_reshape_shape_cone() {
    let mut nodes = static_int64_cast_reshape_nodes();
    let mut weights = static_int64_cast_weights();
    let layers = convert_graph_to_layers(
        &mut nodes,
        &mut weights,
        &CustomOpRegistry::default(),
        &HashMap::new(),
        &HashMap::new(),
        &std::collections::HashSet::from(["reshaped".to_string()]),
        &static_int64_raw_values(),
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
    )
    .expect("the exact constant Cast cone should reach only Reshape's shape port");

    assert!(
        layers
            .iter()
            .all(|layer| layer.layer_type != LayerType::Cast),
        "no non-FLOAT Cast may survive the proven constant shape cone"
    );
}

#[test]
fn materialized_int64_cast_rejects_graph_output_and_unproven_constant_folds() {
    let mut graph_output_nodes = static_int64_cast_reshape_nodes();
    let mut graph_output_weights = static_int64_cast_weights();
    let error = convert_graph_to_layers(
        &mut graph_output_nodes,
        &mut graph_output_weights,
        &CustomOpRegistry::default(),
        &HashMap::new(),
        &HashMap::new(),
        &std::collections::HashSet::from(["head_count_i64".to_string()]),
        &static_int64_raw_values(),
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
    )
    .expect_err("an authored INT64 graph output must remain unsupported");
    assert!(error.to_string().contains("targets dtype 7"), "{error}");
    assert!(
        error.to_string().contains("authored graph output"),
        "{error}"
    );

    // An INT64 Cast whose output the constant folder materialized WITHOUT an
    // exact i64 payload + f32 mirror still fails closed: the rounded constant
    // is baked in before conversion, so lowering to Trunc cannot undo it.
    let mut first_cast = cast_node(7);
    first_cast.input = vec!["source".to_string()];
    first_cast.output = vec!["integer".to_string()];
    let mut to_float = cast_node(1);
    to_float.name = "to_float".to_string();
    to_float.input = vec!["integer".to_string()];
    to_float.output = vec!["float".to_string()];
    let mut nodes = vec![
        first_cast,
        to_float,
        NodeProto {
            input: vec!["data".to_string(), "float".to_string()],
            output: vec!["sum".to_string()],
            name: "add".to_string(),
            op_type: "Add".to_string(),
            ..Default::default()
        },
    ];
    let mut weights = WeightStore::new();
    insert_integer_constant(&mut weights, "integer", &[8]);
    weights.insert("float".to_string(), arr1(&[8.0]).into_dyn());
    let error = convert_graph_to_layers(
        &mut nodes,
        &mut weights,
        &CustomOpRegistry::default(),
        &HashMap::new(),
        &HashMap::new(),
        &std::collections::HashSet::new(),
        &std::collections::HashSet::new(),
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
    )
    .expect_err("an unproven materialized INT64 Cast must fail closed");
    assert!(error.to_string().contains("targets dtype 7"), "{error}");
    assert!(
        error
            .to_string()
            .contains("materialized by constant folding"),
        "{error}"
    );
}

/// The regression this whole change exists for: an INT64 Cast on a RUNTIME
/// operand (no materialized output, not a graph output) must lower to `Trunc`
/// rather than fail closed. That is cctsdb_yolo_2023's patch-position gate.
#[test]
fn runtime_int64_cast_lowers_to_trunc_in_graph_conversion() {
    let mut first_cast = cast_node(7);
    first_cast.name = "position_gate".to_string();
    first_cast.input = vec!["source".to_string()];
    first_cast.output = vec!["integer".to_string()];
    let mut to_float = cast_node(1);
    to_float.name = "to_float".to_string();
    to_float.input = vec!["integer".to_string()];
    to_float.output = vec!["float".to_string()];
    let mut nodes = vec![
        first_cast,
        to_float,
        NodeProto {
            input: vec!["data".to_string(), "float".to_string()],
            output: vec!["sum".to_string()],
            name: "add".to_string(),
            op_type: "Add".to_string(),
            ..Default::default()
        },
    ];
    let mut weights = WeightStore::new();
    let layers = convert_graph_to_layers(
        &mut nodes,
        &mut weights,
        &CustomOpRegistry::default(),
        &HashMap::new(),
        &HashMap::new(),
        &std::collections::HashSet::new(),
        &std::collections::HashSet::new(),
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
    )
    .expect("a runtime INT64 Cast must convert, not fail closed");
    assert_eq!(
        layers
            .iter()
            .filter(|layer| layer.layer_type == LayerType::Trunc)
            .count(),
        1,
        "the runtime INT64 Cast must survive as exactly one Trunc: {layers:?}"
    );
    assert!(
        layers
            .iter()
            .all(|layer| layer.layer_type != LayerType::Cast),
        "no fail-closed Cast may survive: {layers:?}"
    );
}

#[test]
fn int64_cast_skip_requires_integer_provenance_and_exact_f32_mirror() {
    for (integer_payload, float_payload, publish_integer) in [
        (0_i64, 0.0_f32, false), // fractional FLOAT->INT64 fold has no integer side
        (16_777_217_i64, 16_777_217_i64 as f32, true),
        (i64::MAX, i64::MAX as f32, true),
    ] {
        let mut nodes = static_int64_cast_reshape_nodes();
        let mut weights = static_int64_cast_weights();
        weights.insert(
            "head_count_i64".to_string(),
            arr1(&[float_payload]).into_dyn(),
        );
        if publish_integer {
            weights.insert_integers(
                "head_count_i64".to_string(),
                arr1(&[integer_payload]).into_dyn(),
            );
        }

        let error = convert_graph_to_layers(
            &mut nodes,
            &mut weights,
            &CustomOpRegistry::default(),
            &HashMap::new(),
            &HashMap::new(),
            &std::collections::HashSet::new(),
            &static_int64_raw_values(),
            false,
            BatchNormFoldingPolicy::LegacyEnvironment,
        )
        .expect_err("an unproven materialized INT64 Cast must reach the fail-closed handler");
        assert!(error.to_string().contains("targets dtype 7"), "{error}");
        assert!(
            error
                .to_string()
                .contains("materialized by constant folding"),
            "{error}"
        );
    }
}

/// Non-FLOAT32 floating targets must fail closed: DOUBLE changes subsequent
/// arithmetic semantics; f16/bf16 introduce unmodeled rounding. `LayerType::Cast`
/// is the fail-closed marker — ny-build refuses it.
#[test]
fn cast_to_non_f32_float_is_rejected() {
    let registry = CustomOpRegistry::from_handlers(vec![]);
    for to in [10_i64, 11, 16] {
        let layer = convert_node_to_layer(&cast_node(to), &registry, &HashMap::new(), None)
            .unwrap_or_else(|error| panic!("to={to}: {error}"))
            .unwrap_or_else(|| panic!("to={to} must not be dropped"));
        assert_eq!(
            layer.layer_type,
            LayerType::Cast,
            "to={to} must fail closed as LayerType::Cast"
        );
    }
}

/// BOOL is `x != 0`. Without a producer that proves the operand is already
/// {0,1}-valued, the identity drop is unsound and must fail closed.
#[test]
fn cast_to_bool_without_boolean_producer_is_rejected() {
    let registry = CustomOpRegistry::from_handlers(vec![]);
    let layer = convert_node_to_layer(&cast_node(9), &registry, &HashMap::new(), None)
        .expect("convert should succeed")
        .expect("Cast to BOOL must not be dropped without a boolean producer");
    assert_eq!(layer.layer_type, LayerType::Cast);

    // A producer with an arbitrary real range is equally unprovable.
    let producer = make_node("Relu");
    let layer = convert_node_to_layer(&cast_node(9), &registry, &HashMap::new(), Some(&producer))
        .expect("convert should succeed")
        .expect("Cast to BOOL after Relu must not be dropped");
    assert_eq!(layer.layer_type, LayerType::Cast);
}

/// `x != 0` IS the identity on a value that is already 0 or 1, so a BOOL cast
/// fed by a comparison/logical op is exact and stays dropped. cctsdb_yolo_2023
/// masks class matches through exactly this `Equal -> Cast(BOOL)` pair.
#[test]
fn cast_to_bool_after_comparison_stays_identity_drop() {
    let registry = CustomOpRegistry::from_handlers(vec![]);
    for producer_op in [
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "And",
        "Or",
        "Xor",
        "Not",
        "IsNaN",
        "IsInf",
    ] {
        let producer = make_node(producer_op);
        let layer =
            convert_node_to_layer(&cast_node(9), &registry, &HashMap::new(), Some(&producer))
                .unwrap_or_else(|error| panic!("{producer_op}: {error}"));
        assert!(
            layer.is_none(),
            "Cast to BOOL after {producer_op} must stay dropped"
        );
    }

    // Idempotent: BOOL cast of a BOOL cast.
    let layer = convert_node_to_layer(
        &cast_node(9),
        &registry,
        &HashMap::new(),
        Some(&cast_node(9)),
    )
    .expect("convert should succeed");
    assert!(
        layer.is_none(),
        "BOOL cast of a BOOL cast must stay dropped"
    );
}

/// The `{0,1}` guarantee is ONNX's, so it only holds in the standard domain. A
/// custom-domain op is free to reuse the name `Equal` with any semantics, and
/// dropping the BOOL cast after one would be an unproven identity.
#[test]
fn cast_to_bool_after_a_custom_domain_comparison_is_rejected() {
    let registry = CustomOpRegistry::from_handlers(vec![]);
    let mut producer = make_node("Equal");
    producer.domain = "com.example".to_string();
    let layer = convert_node_to_layer(&cast_node(9), &registry, &HashMap::new(), Some(&producer))
        .expect("convert should succeed")
        .expect("Cast to BOOL after a custom-domain Equal must not be dropped");
    assert_eq!(layer.layer_type, LayerType::Cast);
}

/// End-to-end pin: an activation-path f32->f16->f32 round trip is rejected
/// before graph construction, never erased into an exact identity.
#[test]
fn f16_cast_round_trip_is_rejected() {
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

    let error = crate::load_onnx(&path).expect_err("f16 Cast must fail closed");
    assert!(
        error.to_string().contains("unsupported dtype 10"),
        "unexpected error: {error}"
    );
}
