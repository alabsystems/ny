// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX fusion pass end-to-end tests (load + verify fused graph).

use super::*;
use crate::load_onnx_bytes;
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
        f: value,
        i: 0,
        s: Vec::new(),
        t: None,
        r#type: onnx_proto::attribute_type::FLOAT,
        floats: Vec::new(),
        ints: Vec::new(),
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

/// Conv+BN+ReLU fusion: BN should be folded into Conv at load time and the
/// fused IBP bounds should match the unfused (Conv → BN → ReLU) network.
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

/// Fused IBP bounds must match unfused Conv → BN → ReLU bounds.
#[ntest::timeout(10000)]
#[test]
fn test_conv_bn_relu_fusion_ibp_equivalence() {
    let bytes = conv_bn_relu_model_bytes();
    let model =
        load_onnx_bytes("conv_bn_relu_fusion.onnx", &bytes).expect("conv+bn+relu model loads");
    let fused_network = model.to_propagate_network().expect("fused network");
    let unfused_network = unfused_conv_bn_relu_network();

    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![-1.0, 0.0, 1.0, 2.0])
        .expect("valid lower");
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![-0.5, 0.5, 1.5, 2.5])
        .expect("valid upper");
    let input = BoundedTensor::new(lower, upper).expect("valid input");

    let fused_output = fused_network.propagate_ibp(&input).expect("fused IBP");
    let unfused_output = unfused_network.propagate_ibp(&input).expect("unfused IBP");
    assert_ibp_bounds_match(&fused_output, &unfused_output);
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
