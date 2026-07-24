// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for the first Conv3d in the official VNN-COMP 2026
//! Smart Turn multimodal model.

use ny_build::{build_graph_network, GraphBuildInputs};
use ny_onnx::{load_onnx, GraphNetworkOptions, TensorSpec};
use ny_propagate::Layer;
use ny_test_utils::workspace_root;

const FIRST_VIDEO_CONV: &str = "/video_backbone/stem/stem.0/Conv";
const FIRST_VIDEO_CONV_OUTPUT: &str = "/video_backbone/stem/stem.0/Conv_output_0";
const FIRST_VIDEO_CONV_FALLBACK: &str = "/video_backbone/stem/stem.0/Conv__skip";

#[test]
#[ignore = "requires the local VNN-COMP 2026 Smart Turn corpus"]
fn official_smart_turn_first_conv3d_uses_conservative_opaque_fallback() {
    let model_path = workspace_root().join(
        "benchmarks/vnncomp2026/benchmarks/smart_turn_multimodal_2026/2.0/onnx/\
         smart-turn-multimodal-cpu.onnx",
    );
    assert!(
        model_path.is_file(),
        "official Smart Turn model is missing: {}",
        model_path.display()
    );

    let model = load_onnx(&model_path).expect("official Smart Turn ONNX must decode");
    let conv_index = model
        .network
        .layers
        .iter()
        .position(|layer| layer.name == FIRST_VIDEO_CONV)
        .expect("official model must contain its first video Conv3d");
    let layers = &model.network.layers[..=conv_index];
    let outputs = [TensorSpec {
        name: FIRST_VIDEO_CONV_OUTPUT.to_string(),
        shape: vec![1, 64, 32, 56, 56],
        dtype: ny_onnx::DataType::Float32,
    }];
    let inputs = GraphBuildInputs {
        layers,
        inputs: &model.network.inputs,
        outputs: &outputs,
        weights: &model.weights,
        tensor_producer: model.tensor_producer(),
        constant_tensors: model.constant_tensors(),
        tensor_shapes: model.tensor_shapes(),
    };

    let graph = build_graph_network(&inputs, GraphNetworkOptions::default())
        .expect("unsupported Conv3d must degrade to an unbounded graph node, not hard-fail");
    let conv = graph
        .node(FIRST_VIDEO_CONV_FALLBACK)
        .expect("the conservative Conv3d fallback node must be present");
    assert!(
        matches!(conv.layer(), Layer::OpaqueSkip(_)),
        "official Conv3d must remain fail-closed, got {}",
        conv.layer().layer_type()
    );
    assert_eq!(
        graph.output_name(),
        FIRST_VIDEO_CONV_FALLBACK,
        "the official Conv3d output must resolve to the conservative fallback"
    );
}

#[test]
#[ignore = "requires the local VNN-COMP 2026 Smart Turn corpus"]
fn official_smart_turn_full_graph_converts_with_conservative_conv3d_fallbacks() {
    let model_path = workspace_root().join(
        "benchmarks/vnncomp2026/benchmarks/smart_turn_multimodal_2026/2.0/onnx/\
         smart-turn-multimodal-cpu.onnx",
    );
    let model = load_onnx(&model_path).expect("official Smart Turn ONNX must decode");
    let graph = model
        .to_graph_network()
        .expect("full Smart Turn graph must convert past its unsupported Conv3d layers");
    assert!(
        matches!(
            graph
                .node(FIRST_VIDEO_CONV_FALLBACK)
                .map(|node| node.layer()),
            Some(Layer::OpaqueSkip(_))
        ),
        "full graph must keep the first Conv3d conservative"
    );
}
