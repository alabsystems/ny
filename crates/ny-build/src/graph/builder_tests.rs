// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    AttributeValue, CompoundNodePolicy, DataType, TensorSpec, WeightStore,
    EXPAND_LIVE_SHAPE_REFERENCE_ATTR,
};
use ndarray::{arr1, ArrayD, IxDyn};
use ny_core::LayerType;
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

fn tensor_spec(name: &str, shape: &[i64]) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        shape: shape.to_vec(),
        dtype: DataType::Float32,
    }
}

fn layer_spec(name: &str, layer_type: LayerType, inputs: &[&str], outputs: &[&str]) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type,
        inputs: inputs.iter().copied().map(str::to_owned).collect(),
        outputs: outputs.iter().copied().map(str::to_owned).collect(),
        weights: None,
        attributes: HashMap::new(),
    }
}

fn standard_layernorm_spec(attributes: HashMap<String, AttributeValue>) -> LayerSpec {
    LayerSpec {
        name: "layernorm".to_string(),
        layer_type: LayerType::LayerNorm,
        inputs: vec!["input".to_string(), "ny".to_string(), "beta".to_string()],
        outputs: vec!["layernorm_out".to_string()],
        weights: None,
        attributes,
    }
}

fn layernorm_weights(include_gamma: bool, include_beta: bool) -> WeightStore {
    let mut weights = WeightStore::new();
    if include_gamma {
        weights.insert("ny".to_string(), arr1(&[1.0, 1.5, 0.5, 2.0]).into_dyn());
    }
    if include_beta {
        weights.insert(
            "beta".to_string(),
            arr1(&[0.0, 0.25, -0.5, 0.75]).into_dyn(),
        );
    }
    weights
}

fn layernorm_tensor_shapes() -> HashMap<String, Vec<i64>> {
    HashMap::from([
        ("input".to_string(), vec![1, 2, 4]),
        ("layernorm_out".to_string(), vec![1, 2, 4]),
    ])
}

fn build_layernorm_graph(
    options: GraphNetworkOptions,
    layer: LayerSpec,
    weights: WeightStore,
) -> GraphNetwork {
    let layers = vec![layer];
    let inputs = vec![tensor_spec("input", &[1, 2, 4])];
    let outputs = vec![tensor_spec("layernorm_out", &[1, 2, 4])];
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let tensor_shapes = layernorm_tensor_shapes();

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    build_graph_network(&data, options).expect("LayerNorm graph should build")
}

#[test]
fn decompose_layernorm_policy_rewrites_standard_layernorm_4172() {
    let graph = build_layernorm_graph(
        GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        },
        standard_layernorm_spec(HashMap::new()),
        layernorm_weights(true, true),
    );

    let output = graph
        .node("layernorm")
        .expect("final LayerNorm node should exist");
    assert!(
        matches!(output.layer(), Layer::AddConstant(_)),
        "expected decomposed LayerNorm output to become AddConstant, got {:?}",
        output.layer()
    );
    assert!(
        matches!(
            graph.node("layernorm__mean").expect("mean node").layer(),
            Layer::ReduceMean(_)
        ),
        "expected decomposed LayerNorm to emit a ReduceMean node"
    );
    assert!(
        matches!(
            graph
                .node("layernorm__normalized")
                .expect("normalized node")
                .layer(),
            Layer::MulBinary(_)
        ),
        "expected centered * inv_std to stay a binary multiply"
    );
    assert!(
        matches!(
            graph
                .node("layernorm__scaled")
                .expect("scaled node")
                .layer(),
            Layer::MulConstant(_)
        ),
        "expected ny application to become MulConstant"
    );
    assert!(
        matches!(
            graph
                .node("layernorm__inv_std")
                .expect("inv_std node")
                .layer(),
            Layer::Reciprocal(_)
        ),
        "expected decomposed LayerNorm to emit a Reciprocal node"
    );
    assert_eq!(graph.output_name(), "layernorm");
}

#[test]
fn decompose_layernorm_policy_preserve_keeps_monolithic_layernorm_4172() {
    let graph = build_layernorm_graph(
        GraphNetworkOptions::default(),
        standard_layernorm_spec(HashMap::new()),
        layernorm_weights(true, true),
    );

    assert!(
        matches!(
            graph
                .node("layernorm")
                .expect("LayerNorm node should exist")
                .layer(),
            Layer::LayerNorm(_)
        ),
        "default graph build should preserve monolithic LayerNorm"
    );
    assert!(
        graph.node("layernorm__mean").is_none(),
        "default policy must not emit decomposed LayerNorm fragments"
    );
    assert_eq!(graph.num_nodes(), 1);
}

#[test]
fn decompose_layernorm_policy_skips_deept_alias_layernorm_4176() {
    let mut attributes = HashMap::new();
    attributes.insert(
        "layernorm_mode".to_string(),
        AttributeValue::String("deept".to_string()),
    );
    let graph = build_layernorm_graph(
        GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        },
        standard_layernorm_spec(attributes),
        layernorm_weights(true, true),
    );

    assert!(
        matches!(
            graph
                .node("layernorm")
                .expect("LayerNorm node should exist")
                .layer(),
            Layer::LayerNorm(_)
        ),
        "deept LayerNorm alias must stay on the monolithic preserve path"
    );
    assert!(
        graph.node("layernorm__mean").is_none(),
        "deept LayerNorm alias must not emit decomposed fragments"
    );
}

#[test]
fn decompose_layernorm_policy_skips_missing_affine_layernorm_4172() {
    let graph = build_layernorm_graph(
        GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        },
        standard_layernorm_spec(HashMap::new()),
        layernorm_weights(true, false),
    );

    assert!(
        matches!(
            graph
                .node("layernorm")
                .expect("LayerNorm node should exist")
                .layer(),
            Layer::LayerNorm(_)
        ),
        "missing affine weights must stay on the monolithic preserve path"
    );
    assert!(
        graph.node("layernorm__mean").is_none(),
        "missing affine weights must not emit decomposed fragments"
    );
}

#[test]
fn decompose_instance_norm_rank4_broadcasts_affines_by_channel() {
    let layers = vec![LayerSpec {
        name: "instancenorm".to_string(),
        layer_type: LayerType::InstanceNorm,
        inputs: vec!["input".to_string(), "scale".to_string(), "bias".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    }];
    let inputs = vec![tensor_spec("input", &[1, 2, 3, 2])];
    let outputs = vec![tensor_spec("out", &[1, 2, 3, 2])];
    let mut weights = WeightStore::new();
    weights.insert("scale".to_string(), arr1(&[2.0, 3.0]).into_dyn());
    weights.insert("bias".to_string(), arr1(&[10.0, -10.0]).into_dyn());
    let tensor_shapes = HashMap::from([
        ("input".to_string(), vec![1, 2, 3, 2]),
        ("out".to_string(), vec![1, 2, 3, 2]),
        ("scale".to_string(), vec![2]),
        ("bias".to_string(), vec![2]),
    ]);
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };
    let graph = build_graph_network(
        &data,
        GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        },
    )
    .expect("rank-4 InstanceNormalization should decompose");

    let values = vec![
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    let point = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), values.clone()).unwrap();
    let output = graph
        .propagate_ibp(&BoundedTensor::new(point.clone(), point).unwrap())
        .expect("decomposed rank-4 InstanceNormalization IBP should succeed");
    assert_eq!(output.shape(), &[2, 3, 2]);

    let eps = 1e-5_f32 as f64;
    for channel in 0..2 {
        let start = channel * 6;
        let channel_values = &values[start..start + 6];
        let mean = channel_values
            .iter()
            .map(|&value| value as f64)
            .sum::<f64>()
            / 6.0;
        let variance = channel_values
            .iter()
            .map(|&value| {
                let centered = value as f64 - mean;
                centered * centered
            })
            .sum::<f64>()
            / 6.0;
        let scale = [2.0_f64, 3.0][channel];
        let bias = [10.0_f64, -10.0][channel];
        for spatial in 0..6 {
            let h = spatial / 2;
            let w = spatial % 2;
            let expected =
                scale * (channel_values[spatial] as f64 - mean) / (variance + eps).sqrt() + bias;
            let lower = output.lower()[[channel, h, w]] as f64;
            let upper = output.upper()[[channel, h, w]] as f64;
            assert!(
                lower <= expected && expected <= upper,
                "channel {channel}, ({h},{w}): expected {expected} outside [{lower}, {upper}]"
            );
        }
    }
}

/// #2685: Unresolvable tensor references must produce an error, not silently
/// fall back to the _input node.
#[test]
fn test_dangling_tensor_reference_returns_error() {
    // A ReLU layer references tensor "nonexistent" which is not a network input
    // and not produced by any previous layer. Before the fix for #2685, this
    // silently connected to _input. Now it should return an error.
    let weights = WeightStore::new();
    let layers = vec![layer_spec(
        "relu_0",
        LayerType::ReLU,
        &["nonexistent"],
        &["out_0"],
    )];
    let inputs = vec![tensor_spec("real_input", &[1, 3])];
    let outputs = vec![tensor_spec("out_0", &[1, 3])];
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let tensor_shapes = HashMap::new();

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    let result = build_graph_network(&data, GraphNetworkOptions::default());
    assert!(
        result.is_err(),
        "Expected error for dangling tensor reference, got Ok"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("unresolvable tensor"),
        "Error should mention unresolvable tensor, got: {err_msg}"
    );
}

/// #2685: Legitimate _input references (first layer referencing a declared
/// network input) must still work correctly.
#[test]
fn test_legitimate_input_reference_succeeds() {
    let weights = WeightStore::new();
    let layers = vec![layer_spec(
        "relu_0",
        LayerType::ReLU,
        &["real_input"],
        &["out_0"],
    )];
    let inputs = vec![tensor_spec("real_input", &[1, 3])];
    let outputs = vec![tensor_spec("out_0", &[1, 3])];
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let tensor_shapes = HashMap::new();

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    let result = build_graph_network(&data, GraphNetworkOptions::default());
    assert!(
        result.is_ok(),
        "Expected Ok for legitimate input reference, got: {:?}",
        result.err()
    );
}

#[test]
fn test_instance_norm_missing_affine_inputs_use_embedded_defaults_3500() {
    let weights = WeightStore::new();
    let layers = vec![layer_spec(
        "instancenorm",
        LayerType::InstanceNorm,
        &["input", "missing_scale", "missing_bias"],
        &["out"],
    )];
    let inputs = vec![tensor_spec("input", &[1, 3, 4])];
    let outputs = vec![tensor_spec("out", &[1, 3, 4])];
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let tensor_shapes = HashMap::from([
        ("input".to_string(), vec![1, 3, 4]),
        ("out".to_string(), vec![1, 3, 4]),
    ]);

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    let graph = build_graph_network(&data, GraphNetworkOptions::default())
        .expect("InstanceNorm should embed missing affine defaults");
    let node = graph
        .node("instancenorm")
        .expect("InstanceNorm node should exist");
    assert_eq!(node.inputs(), &["_input".to_string()]);
    let Layer::InstanceNorm1d(layer) = node.layer() else {
        panic!("expected InstanceNorm1d, got {:?}", node.layer());
    };
    assert_eq!(layer.num_channels(), 3);
}

#[test]
fn test_scatter_nd_with_constant_data_uses_graph_inputs_for_indices_and_updates() {
    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), ArrayD::zeros(IxDyn(&[4])));

    let layers = vec![
        layer_spec("indices", LayerType::ReLU, &["input"], &["indices_out"]),
        layer_spec("updates", LayerType::ReLU, &["input"], &["updates_out"]),
        layer_spec(
            "scatter",
            LayerType::ScatterND,
            &["data", "indices_out", "updates_out"],
            &["out"],
        ),
    ];
    let inputs = vec![tensor_spec("input", &[1, 4])];
    let outputs = vec![tensor_spec("out", &[1, 4])];
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let tensor_shapes = HashMap::new();

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    let graph = build_graph_network(&data, GraphNetworkOptions::default()).unwrap();
    let scatter = graph.node("scatter").expect("scatter node should exist");
    assert_eq!(
        scatter.inputs(),
        &["indices".to_string(), "updates".to_string()]
    );

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.0, 1.0, 2.0]).into_dyn(),
        arr1(&[-1.0_f32, 0.0, 1.0, 2.0]).into_dyn(),
    )
    .unwrap();
    let output = graph.propagate_ibp(&input).unwrap();

    assert_eq!(output.lower().as_slice().unwrap(), &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn dynamic_expand_rejects_unauthenticated_first_producer_shape_trace() {
    let weights = WeightStore::new();
    let layers = vec![
        layer_spec("summary", LayerType::ReLU, &["input"], &["summary_out"]),
        layer_spec("reference", LayerType::ReLU, &["input"], &["reference_out"]),
        layer_spec(
            "expand",
            LayerType::Expand,
            &["summary_out", "shape_cast_out"],
            &["expanded_out"],
        ),
    ];
    let inputs = vec![tensor_spec("input", &[1, 4])];
    let outputs = vec![tensor_spec("expanded_out", &[1, 4])];
    let tensor_producer = HashMap::from([
        ("shape_cast_out".to_string(), "shape_gather_out".to_string()),
        (
            "shape_gather_out".to_string(),
            "shape_of_reference".to_string(),
        ),
        (
            "shape_of_reference".to_string(),
            "reference_out".to_string(),
        ),
    ]);
    let constant_tensors = HashSet::from([
        "shape_of_reference".to_string(),
        "shape_gather_out".to_string(),
        "shape_cast_out".to_string(),
    ]);
    let tensor_shapes = HashMap::from([
        ("summary_out".to_string(), vec![1, 1]),
        ("reference_out".to_string(), vec![1, 4]),
    ]);

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    let error = build_graph_network(&data, GraphNetworkOptions::default())
        .expect_err("a generic first-producer chain must not authenticate Expand target values");
    assert!(
        error
            .to_string()
            .contains("lacks an authenticated full Shape(reference) source"),
        "unexpected error: {error}"
    );
}

#[test]
fn authenticated_dynamic_expand_uses_exact_live_reference_input() {
    let weights = WeightStore::new();
    let layers = vec![
        LayerSpec {
            name: "summary".to_string(),
            layer_type: LayerType::ReduceMean,
            inputs: vec!["input".to_string()],
            outputs: vec!["summary_out".to_string()],
            weights: None,
            attributes: HashMap::from([
                ("axes".to_string(), AttributeValue::Ints(vec![1])),
                ("keepdims".to_string(), AttributeValue::Int(1)),
            ]),
        },
        layer_spec("reference", LayerType::ReLU, &["input"], &["reference_out"]),
        LayerSpec {
            name: "expand".to_string(),
            layer_type: LayerType::Expand,
            inputs: vec!["summary_out".to_string(), "reference_out".to_string()],
            outputs: vec!["expanded_out".to_string()],
            weights: None,
            attributes: HashMap::from([(
                EXPAND_LIVE_SHAPE_REFERENCE_ATTR.to_string(),
                AttributeValue::String("reference_out".to_string()),
            )]),
        },
    ];
    let inputs = vec![tensor_spec("input", &[1, 4])];
    let outputs = vec![tensor_spec("expanded_out", &[1, 4])];
    let tensor_producer = HashMap::from([
        ("summary_out".to_string(), "input".to_string()),
        ("reference_out".to_string(), "input".to_string()),
    ]);
    let constant_tensors = HashSet::new();
    let tensor_shapes = HashMap::from([
        ("input".to_string(), vec![1, 4]),
        ("summary_out".to_string(), vec![1, 1]),
        ("reference_out".to_string(), vec![1, 4]),
        ("expanded_out".to_string(), vec![1, 4]),
    ]);

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    let graph = build_graph_network(&data, GraphNetworkOptions::default()).unwrap();
    let expand = graph.node("expand").expect("expand node should exist");
    assert!(
        matches!(expand.layer(), Layer::ExpandLikeLastAxis(_)),
        "authenticated Shape(reference) Expand should use runtime ExpandLikeLastAxis"
    );
    assert_eq!(
        expand.inputs(),
        &["summary".to_string(), "reference".to_string()],
        "the authenticated normalized reference should route directly to its live node"
    );

    let input = BoundedTensor::new(
        arr1(&[1.0, 2.0, 3.0, 4.0]).into_dyn(),
        arr1(&[1.0, 2.0, 3.0, 4.0]).into_dyn(),
    )
    .unwrap();
    let output = graph.propagate_ibp(&input).unwrap();
    assert!(output.lower().iter().all(|&value| value <= 2.5));
    assert!(output.upper().iter().all(|&value| value >= 2.5));
}

fn build_split_graph_with_constant_sizes() -> GraphNetwork {
    let layers = vec![LayerSpec {
        name: "split".to_string(),
        layer_type: LayerType::Slice,
        inputs: vec!["input".to_string(), "split_sizes".to_string()],
        outputs: vec!["out0".to_string(), "out1".to_string()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(1))]),
    }];
    let inputs = vec![tensor_spec("input", &[1, 6, 2])];
    let outputs = vec![
        tensor_spec("out0", &[1, 2, 2]),
        tensor_spec("out1", &[1, 4, 2]),
    ];
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let tensor_shapes = HashMap::from([
        ("input".to_string(), vec![1, 6, 2]),
        ("split_sizes".to_string(), vec![2]),
    ]);

    let mut split_weights = WeightStore::new();
    split_weights.insert(
        "split_sizes".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 4.0]).unwrap(),
    );

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &split_weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    build_graph_network(&data, GraphNetworkOptions::default()).unwrap()
}

fn split_graph_constant_size_input() -> BoundedTensor {
    BoundedTensor::concrete(
        ArrayD::from_shape_vec(IxDyn(&[6, 2]), (0..12).map(|value| value as f32).collect())
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn test_split_uses_constant_second_input_sizes() {
    let graph = build_split_graph_with_constant_sizes();
    let slice0 = graph.node("split_slice_0").expect("first split slice");
    let slice1 = graph.node("split_slice_1").expect("second split slice");

    assert_eq!(slice0.inputs(), &["_input".to_string()]);
    assert_eq!(slice1.inputs(), &["_input".to_string()]);

    let input = split_graph_constant_size_input();
    let node_bounds = graph.collect_node_bounds(&input).unwrap();

    assert_eq!(
        node_bounds["split_slice_0"].shape(),
        &[2, 2],
        "first split output should use constant size 2 on the channel axis"
    );
    assert_eq!(
        node_bounds["split_slice_1"].shape(),
        &[4, 2],
        "second split output should use constant size 4 on the channel axis"
    );
}

#[test]
fn test_constant_linear_reshape_split_chain_feeds_add_constant() {
    let graph = build_style_gate_constant_graph();
    assert_style_gate_constant_graph_structure(&graph);
    assert_style_gate_constant_graph_bounds(&graph);
}

#[test]
fn test_constant_conv_with_evaluated_kernel_feeds_add_constant_3500() {
    let graph = build_constant_conv_with_evaluated_kernel_graph();

    assert!(
        graph.node("kernel_scaled").is_none(),
        "constant kernel prelude should be folded out of the graph"
    );
    assert!(
        graph.node("const_conv").is_none(),
        "constant conv fed by evaluated constants should be folded out of the graph"
    );

    let add = graph
        .node("mixed_add")
        .expect("mixed add node should exist");
    assert_eq!(add.inputs(), &["_input".to_string()]);

    let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1, 2]))).unwrap();
    let output = graph.propagate_ibp(&input).unwrap();
    let expected = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![28.0, 40.0]).unwrap();

    assert_eq!(output.lower(), &expected);
    assert_eq!(output.upper(), &expected);
}

fn style_gate_constant_weights() -> WeightStore {
    let mut weights = WeightStore::new();
    weights.insert(
        "style".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0]).unwrap(),
    );
    weights.insert(
        "linear_weight".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[4, 2]),
            vec![
                1.0, 0.0, //
                0.0, 1.0, //
                1.0, 1.0, //
                2.0, 0.0,
            ],
        )
        .unwrap(),
    );
    weights.insert(
        "linear_bias".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
    );
    weights.insert(
        "reshape_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 2.0]).unwrap(),
    );
    weights.insert(
        "split_sizes".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    );
    weights
}

fn style_gate_constant_layers() -> Vec<LayerSpec> {
    vec![
        LayerSpec {
            name: "style_linear".to_string(),
            layer_type: LayerType::Linear,
            inputs: vec![
                "style".to_string(),
                "linear_weight".to_string(),
                "linear_bias".to_string(),
            ],
            outputs: vec!["linear_out".to_string()],
            weights: None,
            attributes: HashMap::from([("transB".to_string(), AttributeValue::Int(1))]),
        },
        layer_spec(
            "style_reshape",
            LayerType::Reshape,
            &["linear_out", "reshape_shape"],
            &["reshaped"],
        ),
        LayerSpec {
            name: "style_split".to_string(),
            layer_type: LayerType::Slice,
            inputs: vec!["reshaped".to_string(), "split_sizes".to_string()],
            outputs: vec!["style_gate".to_string(), "style_residual".to_string()],
            weights: None,
            attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(1))]),
        },
        layer_spec(
            "mixed_add",
            LayerType::Add,
            &["activation", "style_gate"],
            &["out"],
        ),
    ]
}

fn style_gate_constant_tensor_shapes() -> HashMap<String, Vec<i64>> {
    HashMap::from([
        ("activation".to_string(), vec![1, 1, 2]),
        ("style".to_string(), vec![1, 2]),
        ("linear_out".to_string(), vec![1, 4]),
        ("reshaped".to_string(), vec![1, 2, 2]),
        ("style_gate".to_string(), vec![1, 1, 2]),
        ("style_residual".to_string(), vec![1, 1, 2]),
    ])
}

fn build_style_gate_constant_graph() -> GraphNetwork {
    let weights = style_gate_constant_weights();
    let layers = style_gate_constant_layers();
    let inputs = vec![tensor_spec("activation", &[1, 1, 2])];
    let outputs = vec![tensor_spec("out", &[1, 1, 2])];
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let tensor_shapes = style_gate_constant_tensor_shapes();

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    build_graph_network(&data, GraphNetworkOptions::default()).unwrap()
}

fn build_constant_conv_with_evaluated_kernel_graph() -> GraphNetwork {
    let mut weights = WeightStore::new();
    weights.insert(
        "const_signal".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    );
    weights.insert(
        "kernel_base".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
    );
    weights.insert(
        "kernel_scale".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
    );
    weights.insert(
        "conv_bias".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
    );

    let layers = vec![
        layer_spec(
            "kernel_scaled",
            LayerType::Mul,
            &["kernel_base", "kernel_scale"],
            &["scaled_kernel"],
        ),
        LayerSpec {
            name: "const_conv".to_string(),
            layer_type: LayerType::Conv2d,
            inputs: vec![
                "const_signal".to_string(),
                "scaled_kernel".to_string(),
                "conv_bias".to_string(),
            ],
            outputs: vec!["conv_out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        },
        layer_spec(
            "mixed_add",
            LayerType::Add,
            &["activation", "conv_out"],
            &["out"],
        ),
    ];
    let inputs = vec![tensor_spec("activation", &[1, 1, 2])];
    let outputs = vec![tensor_spec("out", &[1, 1, 2])];
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let tensor_shapes = HashMap::from([
        ("activation".to_string(), vec![1, 1, 2]),
        ("scaled_kernel".to_string(), vec![1, 1, 3]),
        ("conv_out".to_string(), vec![1, 1, 2]),
    ]);

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    build_graph_network(&data, GraphNetworkOptions::default()).unwrap()
}

fn assert_style_gate_constant_graph_structure(graph: &GraphNetwork) {
    assert!(
        graph.node("style_linear").is_none(),
        "constant linear prelude should be folded out of the graph"
    );
    assert!(
        graph.node("style_split_slice_0").is_none(),
        "constant split outputs should be folded out of the graph"
    );

    let add = graph
        .node("mixed_add")
        .expect("mixed add node should exist");
    assert_eq!(add.inputs(), &["_input".to_string()]);
}

fn assert_style_gate_constant_graph_bounds(graph: &GraphNetwork) {
    let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1, 2]))).unwrap();
    let output = graph.propagate_ibp(&input).unwrap();
    let expected = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![10.0, 20.0]).unwrap();

    assert_eq!(output.lower(), &expected);
    assert_eq!(output.upper(), &expected);
}

/// vit_2023 regression: a Shape node whose input is an ACTIVATION but whose
/// output was already const-folded at load time (ny-onnx const_fold resolves
/// Shape from static shape inference with batch pinned to 1) must be skipped
/// as a constant — NOT routed to convert_layer_spec, which rejects Shape and
/// (permissive mode) would insert a dangling OpaqueSkipLayer with [-inf, +inf]
/// bounds that poison downstream bounds and fabricate false counterexamples.
#[test]
fn test_shape_with_prefolded_output_skips_instead_of_opaque_skip() {
    let mut weights = WeightStore::new();
    // The load-time const-folder stored Shape's value under its output name.
    weights.insert_integers(
        "shape_out".to_string(),
        ndarray::ArrayD::from_shape_vec(IxDyn(&[2]), vec![1, 4]).unwrap(),
    );
    let layers = vec![
        layer_spec("act", LayerType::ReLU, &["input"], &["act_out"]),
        layer_spec("shape", LayerType::Shape, &["act_out"], &["shape_out"]),
        layer_spec("post", LayerType::ReLU, &["act_out"], &["post_out"]),
    ];
    let inputs = vec![tensor_spec("input", &[1, 4])];
    let outputs = vec![tensor_spec("post_out", &[1, 4])];
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let tensor_shapes = HashMap::from([("act_out".to_string(), vec![1, 4])]);

    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };

    let graph = build_graph_network(&data, GraphNetworkOptions::default()).unwrap();
    assert!(
        graph.node("shape").is_none() && graph.node("shape__skip").is_none(),
        "pre-folded Shape must vanish from the graph (no node, no OpaqueSkip)"
    );
    assert!(graph.node("post").is_some(), "live path must survive");
}
