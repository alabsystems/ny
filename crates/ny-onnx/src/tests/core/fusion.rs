// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX fusion pass end-to-end tests (load + verify fused graph).

use super::*;
use crate::{
    load_onnx_bytes, load_onnx_bytes_with_config, load_onnx_with_config, BatchNormFoldingPolicy,
    OnnxLoadConfig, ShapeInferencePolicy,
};
use approx::assert_relative_eq;
use ny_core::LayerType;
use ny_propagate::layers::{BatchNormLayer, Conv2dLayer, ReLULayer};
use ny_propagate::Layer as PropLayer;
use ny_propagate::Network as PropNetwork;
use ny_tensor::BoundedTensor;
use prost::Message;

fn tensor_value_info(name: &str, shape: &[i64], elem_type: i32) -> onnx_proto::ValueInfoProto {
    let dims = shape
        .iter()
        .map(|dim| onnx_proto::tensor_shape_proto::Dimension {
            value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                *dim,
            )),
        })
        .collect();
    onnx_proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type,
                shape: Some(onnx_proto::TensorShapeProto { dim: dims }),
            }),
        }),
    }
}

fn tensor_f32(name: &str, shape: &[i64], data: &[f32]) -> onnx_proto::TensorProto {
    assert_eq!(shape.iter().product::<i64>() as usize, data.len());
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: 1,
        name: name.to_string(),
        raw_data: Vec::new(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}

fn float_attr(name: &str, value: f32) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        f: Some(value),
        r#type: onnx_proto::attribute_type::FLOAT,
        ..Default::default()
    }
}

fn conv_bn_relu_model_bytes_with_params(
    input_shape: &[i64],
    output_shape: &[i64],
    conv_weight_shape: &[i64],
    conv_weight_data: &[f32],
    bn_scale: &[f32],
    bn_bias: &[f32],
    bn_mean: &[f32],
    bn_var: &[f32],
) -> Vec<u8> {
    assert_eq!(bn_scale.len(), bn_bias.len());
    assert_eq!(bn_scale.len(), bn_mean.len());
    assert_eq!(bn_scale.len(), bn_var.len());

    let conv = onnx_proto::NodeProto {
        input: vec!["input".to_string(), "conv_w".to_string()],
        output: vec!["conv_out".to_string()],
        name: "conv".to_string(),
        op_type: "Conv".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    };
    let batch_norm = onnx_proto::NodeProto {
        input: vec![
            "conv_out".to_string(),
            "bn_scale".to_string(),
            "bn_bias".to_string(),
            "bn_mean".to_string(),
            "bn_var".to_string(),
        ],
        output: vec!["bn_out".to_string()],
        name: "bn".to_string(),
        op_type: "BatchNormalization".to_string(),
        domain: String::new(),
        attribute: vec![float_attr("epsilon", 1.0e-3)],
    };
    let relu = onnx_proto::NodeProto {
        input: vec!["bn_out".to_string()],
        output: vec!["out".to_string()],
        name: "relu".to_string(),
        op_type: "Relu".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    };

    let graph = onnx_proto::GraphProto {
        node: vec![conv, batch_norm, relu],
        name: "conv_bn_relu".to_string(),
        initializer: vec![
            tensor_f32("conv_w", conv_weight_shape, conv_weight_data),
            tensor_f32("bn_scale", &[bn_scale.len() as i64], bn_scale),
            tensor_f32("bn_bias", &[bn_bias.len() as i64], bn_bias),
            tensor_f32("bn_mean", &[bn_mean.len() as i64], bn_mean),
            tensor_f32("bn_var", &[bn_var.len() as i64], bn_var),
        ],
        sparse_initializer: Vec::new(),
        input: vec![tensor_value_info("input", input_shape, 1)],
        output: vec![tensor_value_info("out", output_shape, 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model_proto = onnx_proto::ModelProto {
        ir_version: 9,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        producer_name: "ny-onnx-fixture".to_string(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
    };

    model_proto.encode_to_vec()
}

/// Build a Conv → BatchNormalization → ReLU ONNX model as bytes.
fn conv_bn_relu_model_bytes() -> Vec<u8> {
    conv_bn_relu_model_bytes_with_params(
        &[1, 1, 2, 2],
        &[1, 1, 2, 2],
        &[1, 1, 1, 1],
        &[2.0],
        &[1.5],
        &[0.25],
        &[0.5],
        &[4.0],
    )
}

/// Build an unfused Conv → BN → ReLU propagation network with fixed weights.
fn unfused_conv_bn_relu_network_with_params(
    conv_weight_shape: &[usize],
    conv_weight_data: &[f32],
    bn_scale: &[f32],
    bn_bias: &[f32],
    bn_mean: &[f32],
    bn_var: &[f32],
) -> PropNetwork {
    let mut network = PropNetwork::new();
    let conv_kernel = ArrayD::from_shape_vec(IxDyn(conv_weight_shape), conv_weight_data.to_vec())
        .expect("valid conv kernel");
    let conv_layer = Conv2dLayer::new(conv_kernel, None, (1, 1), (0, 0)).expect("valid conv");
    let ny = ndarray::Array1::from_vec(bn_scale.to_vec()).into_dyn();
    let beta = ndarray::Array1::from_vec(bn_bias.to_vec()).into_dyn();
    let mean = ndarray::Array1::from_vec(bn_mean.to_vec()).into_dyn();
    let var = ndarray::Array1::from_vec(bn_var.to_vec()).into_dyn();
    let bn_layer = BatchNormLayer::new(&ny, &beta, &mean, &var, 1.0e-3).unwrap();
    network.add_layer(PropLayer::Conv2d(conv_layer));
    network.add_layer(PropLayer::BatchNorm(bn_layer));
    network.add_layer(PropLayer::ReLU(ReLULayer::new()));
    network
}

fn unfused_conv_bn_relu_network() -> PropNetwork {
    unfused_conv_bn_relu_network_with_params(&[1, 1, 1, 1], &[2.0], &[1.5], &[0.25], &[0.5], &[4.0])
}

fn conv_bn_relu_multichannel_weight_data() -> Vec<f32> {
    (0..(4 * 3 * 3 * 3))
        .map(|i| ((i as f32) - 18.0) / 11.0)
        .collect()
}

fn conv_bn_relu_multichannel_model_bytes() -> Vec<u8> {
    let conv_weight_data = conv_bn_relu_multichannel_weight_data();
    conv_bn_relu_model_bytes_with_params(
        &[1, 3, 5, 5],
        &[1, 4, 3, 3],
        &[4, 3, 3, 3],
        &conv_weight_data,
        &[2.0, 0.5, 1.0, -1.0],
        &[0.25, -0.5, 1.0, 0.75],
        &[0.5, -1.0, 0.25, 1.5],
        &[4.0, 1.25, 2.5, 3.5],
    )
}

fn unfused_conv_bn_relu_multichannel_network() -> PropNetwork {
    let conv_weight_data = conv_bn_relu_multichannel_weight_data();
    unfused_conv_bn_relu_network_with_params(
        &[4, 3, 3, 3],
        &conv_weight_data,
        &[2.0, 0.5, 1.0, -1.0],
        &[0.25, -0.5, 1.0, 0.75],
        &[0.5, -1.0, 0.25, 1.5],
        &[4.0, 1.25, 2.5, 3.5],
    )
}

/// Assert that two BoundedTensor outputs match element-wise.
fn assert_ibp_bounds_match_with_epsilon(a: &BoundedTensor, b: &BoundedTensor, epsilon: f32) {
    assert_eq!(a.lower().shape(), b.lower().shape());
    assert_eq!(a.upper().shape(), b.upper().shape());
    for (va, vb) in a.lower().iter().zip(b.lower().iter()) {
        assert_relative_eq!(*va, *vb, epsilon = epsilon);
    }
    for (va, vb) in a.upper().iter().zip(b.upper().iter()) {
        assert_relative_eq!(*va, *vb, epsilon = epsilon);
    }
}

fn assert_ibp_bounds_match(a: &BoundedTensor, b: &BoundedTensor) {
    assert_ibp_bounds_match_with_epsilon(a, b, 1e-6);
}

/// The unauthenticated affine composition must remain dark: preserve the raw
/// Conv → BN → ReLU topology and every authored coefficient.
#[ntest::timeout(10000)]
#[test]
fn test_conv_bn_relu_fusion_structure() {
    let bytes = conv_bn_relu_model_bytes();
    let model =
        load_onnx_bytes("conv_bn_relu_fusion.onnx", &bytes).expect("conv+bn+relu model loads");

    assert_eq!(model.network.layers.len(), 2);
    assert_eq!(model.network.layers[0].layer_type, LayerType::Conv2d);
    assert_eq!(model.network.layers[1].layer_type, LayerType::ReLU);
    assert!(
        !model
            .network
            .layers
            .iter()
            .any(|l| l.layer_type == LayerType::BatchNorm),
        "BatchNorm node should be removed by fusion"
    );

    let conv_spec = &model.network.layers[0];
    assert_eq!(conv_spec.outputs[0], "bn_out");
    assert_eq!(conv_spec.inputs.len(), 3);

    let expected_scale = 1.5 / (4.0_f32 + 1.0e-3).sqrt();
    let expected_shift = 0.25 - 1.5 * 0.5 / (4.0_f32 + 1.0e-3).sqrt();
    let fused_weight = model.weights.get("conv_w").expect("fused conv weight");
    let fused_bias = model
        .weights
        .get(conv_spec.inputs[2].as_str())
        .expect("fused bias");
    assert_relative_eq!(
        fused_weight[[0, 0, 0, 0]],
        2.0 * expected_scale,
        epsilon = 1e-6
    );
    assert_relative_eq!(fused_bias[[0]], expected_shift, epsilon = 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn default_batch_norm_policy_matches_convenience_loader() {
    let bytes = conv_bn_relu_model_bytes();
    let convenience = load_onnx_bytes("bn_default_convenience.onnx", &bytes).expect("default load");
    let configured = load_onnx_bytes_with_config(
        "bn_default_configured.onnx",
        &bytes,
        &OnnxLoadConfig::default(),
    )
    .expect("explicit default load");

    let signature = |model: &OnnxModel| {
        model
            .network
            .layers
            .iter()
            .map(|layer| {
                (
                    layer.layer_type.clone(),
                    layer.inputs.clone(),
                    layer.outputs.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(signature(&convenience), signature(&configured));
    assert_eq!(
        convenience.weights.get("conv_w"),
        configured.weights.get("conv_w")
    );
}

#[ntest::timeout(10000)]
#[test]
fn call_local_policy_preserves_raw_batch_norm_and_authored_weights() {
    let bytes = conv_bn_relu_model_bytes();
    let config = OnnxLoadConfig::default()
        .with_shape_inference_policy(ShapeInferencePolicy::Skip)
        .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw);
    let model =
        load_onnx_bytes_with_config("bn_preserve_raw.onnx", &bytes, &config).expect("raw-BN load");

    let layer_types = model
        .network
        .layers
        .iter()
        .map(|layer| layer.layer_type.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        layer_types,
        vec![LayerType::Conv2d, LayerType::BatchNorm, LayerType::ReLU]
    );
    assert_eq!(model.network.layers[0].outputs, ["conv_out"]);
    assert_eq!(
        model.network.layers[1].inputs,
        ["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"]
    );
    assert_eq!(model.network.layers[1].outputs, ["bn_out"]);
    assert_eq!(
        model
            .weights
            .get("conv_w")
            .expect("authored convolution weight")[[0, 0, 0, 0]],
        2.0
    );
    assert_eq!(
        model.network.layers[0].inputs,
        ["input".to_string(), "conv_w".to_string()]
    );
}

#[test]
fn bn_policy_selects_raw_or_folded() {
    // #bn-fold-restore. PreserveRaw keeps the authored BatchNorm layer and every
    // authored FLOAT initializer bit-identical; the DEFAULT policy
    // (LegacyEnvironment) folds BN into the preceding Conv/Gemm — the same
    // preprocessing the published alpha-beta-CROWN reference applies
    // (complete_verifier/onnx_opt.py). The quarantine-era version asserted
    // raw-for-every-policy; that hard gate silently cost 15 field-confirmed
    // unsat rows on cifar100 (see convert.rs #bn-fold-restore).
    let bytes = conv_bn_relu_model_bytes();
    let raw_config = OnnxLoadConfig::default()
        .with_shape_inference_policy(ShapeInferencePolicy::Skip)
        .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
        .with_require_authored_float32_initializers(true);
    let raw = load_onnx_bytes_with_config("bn_authored_admitted.onnx", &bytes, &raw_config)
        .expect("raw BatchNorm graph must preserve every authored FLOAT initializer");
    assert_eq!(
        raw.authored_float32_initializers_match_current(),
        Some(true)
    );
    assert!(raw
        .network
        .layers
        .iter()
        .any(|layer| layer.layer_type == LayerType::BatchNorm));

    // Default policy: the BN layer is folded away. The composed conv weight is
    // a deliberate FLOAT rewrite, so the authored-float admission flag is not
    // requested here — fold attribution is covered by the fidelity fold tests.
    let default_config =
        OnnxLoadConfig::default().with_shape_inference_policy(ShapeInferencePolicy::Skip);
    let folded = load_onnx_bytes_with_config("bn_folded_default.onnx", &bytes, &default_config)
        .expect("default-policy load folds BN");
    assert!(
        folded
            .network
            .layers
            .iter()
            .all(|layer| layer.layer_type != LayerType::BatchNorm),
        "default policy must fold Conv+BN: {:?}",
        folded
            .network
            .layers
            .iter()
            .map(|layer| &layer.layer_type)
            .collect::<Vec<_>>()
    );
}

#[ntest::timeout(10000)]
#[test]
fn call_local_policy_reaches_file_loader_without_ambient_state() {
    let directory = tempfile::tempdir().expect("temporary fixture directory");
    let path = directory.path().join("conv_bn_relu.onnx");
    std::fs::write(&path, conv_bn_relu_model_bytes()).expect("write miniature ONNX fixture");
    let config = OnnxLoadConfig::default()
        .with_shape_inference_policy(ShapeInferencePolicy::Skip)
        .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw);

    let model = load_onnx_with_config(&path, &config).expect("call-local file load");
    assert_eq!(model.network.layers.len(), 3);
    assert_eq!(model.network.layers[1].layer_type, LayerType::BatchNorm);

    let missing = directory.path().join("missing.onnx");
    let error =
        load_onnx_with_config(&missing, &config).expect_err("missing model path must fail closed");
    assert!(matches!(error, NyError::ModelLoad(_)));
}

#[ntest::timeout(10000)]
#[test]
fn batch_norm_policy_is_call_local_under_parallel_loads() {
    let bytes = std::sync::Arc::new(conv_bn_relu_model_bytes());
    let workers = (0..8)
        .map(|worker| {
            let bytes = std::sync::Arc::clone(&bytes);
            std::thread::spawn(move || {
                let policy = if worker % 2 == 0 {
                    BatchNormFoldingPolicy::PreserveRaw
                } else {
                    BatchNormFoldingPolicy::LegacyEnvironment
                };
                let expected_layers = if policy == BatchNormFoldingPolicy::PreserveRaw {
                    3
                } else {
                    2
                };
                for iteration in 0..32 {
                    let config = OnnxLoadConfig::default()
                        .with_shape_inference_policy(ShapeInferencePolicy::Skip)
                        .with_batch_norm_folding_policy(policy);
                    let model = load_onnx_bytes_with_config(
                        &format!("bn_parallel_{worker}_{iteration}.onnx"),
                        &bytes,
                        &config,
                    )
                    .expect("parallel call-local load");
                    assert_eq!(model.network.layers.len(), expected_layers);
                }
            })
        })
        .collect::<Vec<_>>();

    for worker in workers {
        worker.join().expect("parallel loader worker");
    }
}

#[ntest::timeout(10000)]
#[test]
fn malformed_batch_norm_signature_is_rejected_before_any_rewrite() {
    let mut proto =
        onnx_proto::ModelProto::decode(conv_bn_relu_model_bytes().as_slice()).expect("fixture");
    let graph = proto.graph.as_mut().expect("fixture graph");
    graph.node[1].input.pop();
    let bytes = proto.encode_to_vec();

    for policy in [
        BatchNormFoldingPolicy::LegacyEnvironment,
        BatchNormFoldingPolicy::PreserveRaw,
    ] {
        let config = OnnxLoadConfig::default()
            .with_shape_inference_policy(ShapeInferencePolicy::Skip)
            .with_batch_norm_folding_policy(policy);
        let error = load_onnx_bytes_with_config("bn_malformed_path.onnx", &bytes, &config)
            .expect_err("a four-input standard BatchNormalization is malformed");
        assert!(matches!(error, NyError::ModelLoad(message) if message.contains("exactly five")));
    }
}

#[ntest::timeout(10000)]
#[test]
fn preserve_raw_policy_does_not_weaken_malformed_onnx_rejection() {
    let config = OnnxLoadConfig::default()
        .with_shape_inference_policy(ShapeInferencePolicy::Skip)
        .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw);
    let error = load_onnx_bytes_with_config("malformed.onnx", &[0xff, 0x80], &config)
        .expect_err("malformed protobuf must fail closed");
    assert!(
        matches!(error, NyError::ModelLoad(_)),
        "unexpected malformed-model error: {error}"
    );
}

/// Loaded raw IBP bounds must match the independently constructed raw graph.
#[ntest::timeout(10000)]
#[test]
fn test_conv_bn_relu_raw_ibp_equivalence() {
    let bytes = conv_bn_relu_model_bytes();
    let model =
        load_onnx_bytes("conv_bn_relu_fusion.onnx", &bytes).expect("conv+bn+relu model loads");
    let loaded_network = model.to_propagate_network().expect("loaded network");
    let unfused_network = unfused_conv_bn_relu_network();

    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![-1.0, 0.0, 1.0, 2.0])
        .expect("valid lower");
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![-0.5, 0.5, 1.5, 2.5])
        .expect("valid upper");
    let input = BoundedTensor::new(lower, upper).expect("valid input");

    let loaded_output = loaded_network.propagate_ibp(&input).expect("loaded IBP");
    let unfused_output = unfused_network.propagate_ibp(&input).expect("unfused IBP");
    assert_ibp_bounds_match(&loaded_output, &unfused_output);
}

/// #2318: use distinct output/input channel counts so BN scaling-axis mistakes
/// cannot hide behind the old degenerate 1-channel fixture.
#[ntest::timeout(10000)]
#[test]
fn test_conv_bn_relu_fusion_ibp_equivalence_multichannel_2318() {
    let bytes = conv_bn_relu_multichannel_model_bytes();
    let model = load_onnx_bytes("conv_bn_relu_fusion_multichannel.onnx", &bytes)
        .expect("multi-channel conv+bn+relu model loads");
    let conv_weight_data = conv_bn_relu_multichannel_weight_data();
    let bn_scale: [f32; 4] = [2.0, 0.5, 1.0, -1.0];
    let bn_bias: [f32; 4] = [0.25, -0.5, 1.0, 0.75];
    let bn_mean: [f32; 4] = [0.5, -1.0, 0.25, 1.5];
    let bn_var: [f32; 4] = [4.0, 1.25, 2.5, 3.5];
    let conv_spec = &model.network.layers[0];
    let fused_weight = model.weights.get("conv_w").expect("fused conv weight");
    let fused_bias = model
        .weights
        .get(conv_spec.inputs[2].as_str())
        .expect("fused bias");
    for output_channel in 0..4 {
        let scale = bn_scale[output_channel] / (bn_var[output_channel] + 1.0e-3).sqrt();
        let expected_shift = bn_bias[output_channel] - scale * bn_mean[output_channel];
        let original_idx = output_channel * 3 * 3 * 3;
        assert_relative_eq!(
            fused_weight[[output_channel, 0, 0, 0]],
            conv_weight_data[original_idx] * scale,
            epsilon = 1e-6
        );
        assert_relative_eq!(fused_bias[[output_channel]], expected_shift, epsilon = 1e-6);
    }
    let fused_network = model.to_propagate_network().expect("fused network");
    let unfused_network = unfused_conv_bn_relu_multichannel_network();

    let lower_data: Vec<f32> = (0..(3 * 5 * 5))
        .map(|i| ((i % 11) as f32 - 5.0) / 9.0)
        .collect();
    let upper_data: Vec<f32> = lower_data
        .iter()
        .enumerate()
        .map(|(i, &lower)| lower + 0.2 + (i % 3) as f32 * 0.05)
        .collect();
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3, 5, 5]), lower_data).expect("valid lower");
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 3, 5, 5]), upper_data).expect("valid upper");
    let input = BoundedTensor::new(lower, upper).expect("valid input");

    let fused_output = fused_network.propagate_ibp(&input).expect("fused IBP");
    let unfused_output = unfused_network.propagate_ibp(&input).expect("unfused IBP");
    assert_ibp_bounds_match_with_epsilon(&fused_output, &unfused_output, 1e-5);
}

#[test]
fn resnet_shaped_graph_folds_all_batch_norms_through_the_production_path() {
    // #bn-fold-restore regression guard. The BN fold's own unit tests kept
    // passing for the entire life of the `BATCH_NORM_AFFINE_COMPOSITION_
    // AUTHENTICATED = false` quarantine, because the `#[cfg(test)]` wrapper in
    // batch_norm_fold.rs enters the fold directly and never crosses the
    // convert.rs call-site gate that was dead. This test goes through the REAL
    // loader entry on a cifar100-resnet-shaped graph, so ANY future silent
    // unfold — a const gate, a policy-default flip, an adjacency-map break, a
    // provenance reclassification — fails loudly here.
    //
    // Two properties, both load-bearing:
    //   1. every BatchNormalization folds under the default policy;
    //   2. the post-fold layer count stays at or under 50, because crossing
    //      CROWN_IBP_PER_NODE_THRESHOLD (ny-propagate graph/mod.rs) silently
    //      deletes the per-node CROWN-IBP collector lane. The measured cost of
    //      crossing it was 15 field-confirmed `unsat` rows on cifar100.
    let mut nodes = Vec::new();
    let mut initializers = vec![tensor_f32("fc_w", &[4, 2], &[0.1; 8])];
    let mut prev = "input".to_string();
    for block in 0..10 {
        let names = [
            format!("conv_a_{block}"),
            format!("bn_a_{block}"),
            format!("conv_b_{block}"),
            format!("bn_b_{block}"),
            format!("add_{block}"),
            format!("relu_{block}"),
        ];
        for (conv, bn) in [(&names[0], &names[1]), (&names[2], &names[3])] {
            initializers.push(tensor_f32(&format!("{conv}_w"), &[1, 1, 1, 1], &[1.0]));
            for (suffix, value) in [
                ("scale", 1.5f32),
                ("bias", 0.25),
                ("mean", 0.1),
                ("var", 0.9),
            ] {
                initializers.push(tensor_f32(&format!("{bn}_{suffix}"), &[1], &[value]));
            }
        }
        let block_input = prev.clone();
        nodes.push(onnx_proto::NodeProto {
            input: vec![block_input.clone(), format!("{}_w", names[0])],
            output: vec![format!("{}_out", names[0])],
            name: names[0].clone(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        });
        nodes.push(onnx_proto::NodeProto {
            input: vec![
                format!("{}_out", names[0]),
                format!("{}_scale", names[1]),
                format!("{}_bias", names[1]),
                format!("{}_mean", names[1]),
                format!("{}_var", names[1]),
            ],
            output: vec![format!("{}_out", names[1])],
            name: names[1].clone(),
            op_type: "BatchNormalization".to_string(),
            domain: String::new(),
            attribute: vec![float_attr("epsilon", 1.0e-3)],
        });
        nodes.push(onnx_proto::NodeProto {
            input: vec![format!("{}_out", names[1]), format!("{}_w", names[2])],
            output: vec![format!("{}_out", names[2])],
            name: names[2].clone(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        });
        nodes.push(onnx_proto::NodeProto {
            input: vec![
                format!("{}_out", names[2]),
                format!("{}_scale", names[3]),
                format!("{}_bias", names[3]),
                format!("{}_mean", names[3]),
                format!("{}_var", names[3]),
            ],
            output: vec![format!("{}_out", names[3])],
            name: names[3].clone(),
            op_type: "BatchNormalization".to_string(),
            domain: String::new(),
            attribute: vec![float_attr("epsilon", 1.0e-3)],
        });
        nodes.push(onnx_proto::NodeProto {
            input: vec![format!("{}_out", names[3]), block_input],
            output: vec![format!("{}_out", names[4])],
            name: names[4].clone(),
            op_type: "Add".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        });
        nodes.push(onnx_proto::NodeProto {
            input: vec![format!("{}_out", names[4])],
            output: vec![format!("{}_out", names[5])],
            name: names[5].clone(),
            op_type: "Relu".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        });
        prev = format!("{}_out", names[5]);
    }
    nodes.push(onnx_proto::NodeProto {
        input: vec![prev],
        output: vec!["flat_out".to_string()],
        name: "flatten".to_string(),
        op_type: "Flatten".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    });
    nodes.push(onnx_proto::NodeProto {
        input: vec!["flat_out".to_string(), "fc_w".to_string()],
        output: vec!["out".to_string()],
        name: "fc".to_string(),
        op_type: "Gemm".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    });
    let authored_nodes = nodes.len();
    assert!(
        authored_nodes > 50,
        "the skeleton must START past the collector cliff to prove folding \
         brings it back under: {authored_nodes}"
    );

    let graph = onnx_proto::GraphProto {
        node: nodes,
        name: "resnet_skeleton".to_string(),
        initializer: initializers,
        sparse_initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[1, 1, 2, 2], 1)],
        output: vec![tensor_value_info("out", &[1, 2], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model_proto = onnx_proto::ModelProto {
        ir_version: 9,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        producer_name: "ny-onnx-fixture".to_string(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
    };
    let bytes = model_proto.encode_to_vec();

    let config = OnnxLoadConfig::default().with_shape_inference_policy(ShapeInferencePolicy::Skip);
    let model = load_onnx_bytes_with_config("resnet_skeleton.onnx", &bytes, &config)
        .expect("resnet-shaped graph must load through the production path");
    let bn_layers = model
        .network
        .layers
        .iter()
        .filter(|layer| layer.layer_type == LayerType::BatchNorm)
        .count();
    assert_eq!(
        bn_layers, 0,
        "every Conv+BN must fold under the default policy; {bn_layers} survived"
    );
    assert_eq!(
        model.network.layers.len(),
        authored_nodes - 20,
        "exactly the 20 BatchNorm nodes fold away"
    );
    assert!(
        model.network.layers.len() <= 50,
        "post-fold layer count {} crossed the per-node CROWN-IBP threshold (50); \
         that silently deletes a verdict lane",
        model.network.layers.len()
    );
}
