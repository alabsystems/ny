// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Executable handoff examples for traced producers populating `GraphModel`.
//!
//! `#3288` needs a concrete contract for non-ONNX integrations that record a
//! computation graph after tracing imperative code. These tests exercise the
//! curated `ny_api::model` surface the same way an external producer would:
//! assemble a neutral graph contract, attach tracing metadata, then call
//! `GraphModel::build_graph_network(...)`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_api::graph::{GraphNetwork, GraphNode, NETWORK_INPUT};
use ny_api::layers::{CumsumLayer, Layer, LinearLayer, ReLULayer};
use ny_api::model::{
    AttributeValue, DataType, GraphModel, GraphModelBuilder, GraphNetworkOptions, LayerSpec,
    LayerType, NetworkSpec, TensorSpec, WeightStore, EXPAND_LIVE_SHAPE_REFERENCE_ATTR,
};
use ny_api::parallel::{
    verify_parallel, verify_parallel_with_engine, verify_parallel_with_method,
    verify_parallel_with_method_and_engine,
};
use ny_api::verify::{PropagationConfig, PropagationMethod, Verifier};
use ny_api::{Bound, BoundedTensor, VerificationResult, VerificationSpec};
use ny_test_utils::CountingGemmEngine;

fn attributed_layer_spec(
    name: &str,
    layer_type: LayerType,
    inputs: &[&str],
    outputs: &[&str],
    attributes: HashMap<String, AttributeValue>,
) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type,
        inputs: inputs.iter().map(|value| value.to_string()).collect(),
        outputs: outputs.iter().map(|value| value.to_string()).collect(),
        weights: None,
        attributes,
    }
}

fn layer_spec(name: &str, layer_type: LayerType, inputs: &[&str], outputs: &[&str]) -> LayerSpec {
    attributed_layer_spec(name, layer_type, inputs, outputs, HashMap::new())
}

#[test]
fn traced_shape_path_routes_expand_to_live_reference_node() {
    let graph_model = traced_expand_graph_model(&[1, 4]);

    let graph = graph_model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("traced producer handoff should build through ny_api::model");
    let expand = graph.node("expand").expect("expand node should exist");
    assert_eq!(
        expand.inputs(),
        &["summary".to_string(), "reference".to_string()],
        "tensor_producer should trace structural shape ops back to the live activation node"
    );
}

fn traced_expand_graph_model(shape: &[i64]) -> GraphModel {
    assert!(shape.len() >= 2 && shape.last().copied().is_some_and(|width| width > 0));
    let mut summary_shape = shape.to_vec();
    *summary_shape.last_mut().expect("non-empty shape") = 1;
    GraphModelBuilder::new("expand-like")
        .input("input", shape, DataType::Float32)
        .output("expanded_out", shape, DataType::Float32)
        .layer(attributed_layer_spec(
            "summary",
            LayerType::ReduceMean,
            &["input"],
            &["summary_out"],
            HashMap::from([
                (
                    "axes".to_string(),
                    AttributeValue::Ints(vec![
                        i64::try_from(shape.len() - 1).expect("rank fits i64")
                    ]),
                ),
                ("keepdims".to_string(), AttributeValue::Int(1)),
            ]),
        ))
        .layer(layer_spec(
            "reference",
            LayerType::ReLU,
            &["input"],
            &["reference_out"],
        ))
        .layer(attributed_layer_spec(
            "expand",
            LayerType::Expand,
            &["summary_out", "reference_out"],
            &["expanded_out"],
            HashMap::from([(
                EXPAND_LIVE_SHAPE_REFERENCE_ATTR.to_string(),
                AttributeValue::String("reference_out".to_string()),
            )]),
        ))
        .tensor_producer("summary_out", "input")
        .tensor_producer("reference_out", "input")
        .tensor_shape("input", shape)
        .tensor_shape("summary_out", &summary_shape)
        .tensor_shape("reference_out", shape)
        .tensor_shape("expanded_out", shape)
        .build()
}

fn style_gate_linear_layer() -> LayerSpec {
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
    }
}

fn style_gate_split_layer() -> LayerSpec {
    LayerSpec {
        name: "style_split".to_string(),
        layer_type: LayerType::Slice,
        inputs: vec!["reshaped".to_string(), "split_sizes".to_string()],
        outputs: vec!["style_gate".to_string(), "style_residual".to_string()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(1))]),
    }
}

fn style_gate_builder_with_weights(builder: GraphModelBuilder) -> GraphModelBuilder {
    builder
        .weight(
            "linear_weight",
            ArrayD::from_shape_vec(
                IxDyn(&[4, 2]),
                vec![
                    1.0, 0.0, //
                    0.0, 1.0, //
                    1.0, 1.0, //
                    2.0, 0.0,
                ],
            )
            .expect("valid linear weight tensor"),
        )
        .weight(
            "linear_bias",
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 0.0, 0.0, 0.0])
                .expect("valid bias tensor"),
        )
        .weight(
            "reshape_shape",
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 2.0]).expect("valid reshape tensor"),
        )
        .weight(
            "split_sizes",
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).expect("valid split sizes tensor"),
        )
}

fn style_gate_builder_with_layers(builder: GraphModelBuilder) -> GraphModelBuilder {
    builder
        .layer(style_gate_linear_layer())
        .layer(layer_spec(
            "style_reshape",
            LayerType::Reshape,
            &["linear_out", "reshape_shape"],
            &["reshaped"],
        ))
        .layer(style_gate_split_layer())
        .layer(layer_spec(
            "mixed_add",
            LayerType::Add,
            &["activation", "style_gate"],
            &["out"],
        ))
}

fn style_gate_graph_model_builder(name: &str) -> GraphModelBuilder {
    let builder = GraphModelBuilder::new(name)
        .input("activation", &[1, 1, 2], DataType::Float32)
        .output("out", &[1, 1, 2], DataType::Float32)
        .frozen_input(
            "style",
            &[1, 2],
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0]).expect("valid style tensor"),
        )
        .tensor_shape("activation", &[1, 1, 2])
        .tensor_shape("linear_out", &[1, 4])
        .tensor_shape("reshaped", &[1, 2, 2])
        .tensor_shape("style_gate", &[1, 1, 2])
        .tensor_shape("style_residual", &[1, 1, 2]);
    style_gate_builder_with_layers(style_gate_builder_with_weights(builder))
}

fn style_gate_constant_graph_model() -> GraphModel {
    style_gate_graph_model_builder("style-gate").build()
}

fn talker_like_mask_slice_layer() -> LayerSpec {
    // Axis 2 in producer-declared [1, 4, 4] space becomes runtime axis 1 after
    // batch stripping, yielding the intended [4, 2] mask gate.
    attributed_layer_spec(
        "mask_gate",
        LayerType::Slice,
        &["mask"],
        &["mask_gate_out"],
        HashMap::from([
            ("axis".to_string(), AttributeValue::Int(2)),
            ("start".to_string(), AttributeValue::Int(0)),
            ("end".to_string(), AttributeValue::Int(2)),
        ]),
    )
}

fn talker_like_rotary_bias_graph_model_builder(name: &str) -> GraphModelBuilder {
    GraphModelBuilder::new(name)
        .input("hidden_states", &[1, 4, 2], DataType::Float32)
        .output("out", &[1, 4, 2], DataType::Float32)
        .frozen_input(
            "cos",
            &[1, 4, 2],
            ArrayD::from_elem(IxDyn(&[4, 2]), 1.0_f32),
        )
        .frozen_input(
            "sin",
            &[1, 4, 2],
            ArrayD::from_elem(IxDyn(&[4, 2]), 2.0_f32),
        )
        .frozen_input(
            "mask",
            &[1, 4, 4],
            ArrayD::from_shape_vec(
                IxDyn(&[4, 4]),
                vec![
                    1.0_f32, 1.0, 9.0, 9.0, //
                    1.0, 1.0, 9.0, 9.0, //
                    1.0, 1.0, 9.0, 9.0, //
                    1.0, 1.0, 9.0, 9.0,
                ],
            )
            .expect("valid mask tensor"),
        )
        .layer(layer_spec(
            "cos_merge",
            LayerType::Add,
            &["hidden_states", "cos"],
            &["hidden_plus_cos"],
        ))
        .layer(layer_spec(
            "pre_mask_merge",
            LayerType::Add,
            &["hidden_plus_cos", "sin"],
            &["pre_mask_out"],
        ))
        .layer(talker_like_mask_slice_layer())
        .layer(layer_spec(
            "output_merge",
            LayerType::Add,
            &["pre_mask_out", "mask_gate_out"],
            &["out"],
        ))
        .tensor_shape("hidden_states", &[1, 4, 2])
        .tensor_shape("hidden_plus_cos", &[1, 4, 2])
        .tensor_shape("pre_mask_out", &[1, 4, 2])
        .tensor_shape("mask_gate_out", &[1, 4, 2])
        .tensor_shape("out", &[1, 4, 2])
}

fn talker_like_rotary_bias_graph_model() -> GraphModel {
    talker_like_rotary_bias_graph_model_builder("talker-like-rotary-bias").build()
}

fn bounds(pairs: &[(f32, f32)]) -> Vec<Bound> {
    pairs
        .iter()
        .map(|&(lower, upper)| Bound::new(lower, upper))
        .collect()
}

fn exact_unbatched_input(shape: &[usize], lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), lower),
        ArrayD::from_elem(IxDyn(shape), upper),
    )
    .expect("exact unbatched input bounds should be valid")
}

fn assert_talker_like_graph_model_metadata(graph_model: &GraphModel) {
    assert_eq!(
        graph_model
            .network
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        vec!["hidden_states"],
        "the direct talker-like packet should keep hidden_states as the only live network input"
    );
    assert_eq!(
        graph_model
            .constant_tensors
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>(),
        HashSet::from(["cos", "sin", "mask"]),
        "the direct talker-like packet should record exactly the frozen auxiliary tensor set"
    );
    assert_eq!(
        graph_model.tensor_shapes.get("cos"),
        Some(&vec![1, 4, 2]),
        "cos should retain its producer-declared shape metadata"
    );
    assert_eq!(
        graph_model.tensor_shapes.get("sin"),
        Some(&vec![1, 4, 2]),
        "sin should retain its producer-declared shape metadata"
    );
    assert_eq!(
        graph_model.tensor_shapes.get("mask"),
        Some(&vec![1, 4, 4]),
        "mask should retain its producer-declared shape metadata"
    );
    assert_eq!(
        graph_model
            .weights
            .get("cos")
            .expect("cos should be stored as a frozen weight")
            .shape(),
        &[4_usize, 2_usize],
        "cos should be stored unbatched"
    );
    assert_eq!(
        graph_model
            .weights
            .get("sin")
            .expect("sin should be stored as a frozen weight")
            .shape(),
        &[4_usize, 2_usize],
        "sin should be stored unbatched"
    );
    assert_eq!(
        graph_model
            .weights
            .get("mask")
            .expect("mask should be stored as a frozen weight")
            .shape(),
        &[4_usize, 4_usize],
        "mask should be stored unbatched"
    );
}

fn assert_talker_like_runtime_path(graph: &GraphNetwork) {
    assert_eq!(
        graph.output_name(),
        "output_merge",
        "the direct talker-like packet should keep the final add as the graph output"
    );
    let cos_merge = graph
        .node("cos_merge")
        .expect("the first frozen-input merge should remain as a runtime node");
    assert_eq!(
        cos_merge.inputs(),
        &[NETWORK_INPUT.to_string()],
        "the first merge should depend only on the live bounded activation input"
    );
    let pre_mask_merge = graph
        .node("pre_mask_merge")
        .expect("the second frozen-input merge should remain as a runtime node");
    assert_eq!(
        pre_mask_merge.inputs(),
        &["cos_merge".to_string()],
        "the second merge should extend the single live-input chain"
    );
    let output_merge = graph
        .node("output_merge")
        .expect("the final talker-like merge should remain as the runtime output node");
    assert_eq!(
        output_merge.inputs(),
        &["pre_mask_merge".to_string()],
        "the final runtime node should only depend on the live activation chain after frozen-branch folding"
    );
}

fn assert_tensor_bounds(tensor: &BoundedTensor, lower: &[f32], upper: &[f32]) {
    assert_eq!(
        tensor.lower().iter().copied().collect::<Vec<_>>(),
        lower,
        "lower bounds should match the traced-producer contract"
    );
    assert_eq!(
        tensor.upper().iter().copied().collect::<Vec<_>>(),
        upper,
        "upper bounds should match the traced-producer contract"
    );
}

fn small_parallel_graph(hidden_dim: usize) -> GraphNetwork {
    let linear1 = LinearLayer::new(
        Array2::from_shape_fn((hidden_dim, hidden_dim), |(i, j)| {
            if i == j {
                0.5_f32
            } else {
                0.01
            }
        }),
        Some(Array1::zeros(hidden_dim)),
    )
    .expect("first Linear layer should be valid");

    let linear2 = LinearLayer::new(
        Array2::from_shape_fn((hidden_dim, hidden_dim), |(i, j)| {
            if i == j {
                0.3_f32
            } else {
                -0.01
            }
        }),
        Some(Array1::zeros(hidden_dim)),
    )
    .expect("second Linear layer should be valid");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

fn small_parallel_input(seq_len: usize, hidden_dim: usize) -> BoundedTensor {
    let values = ArrayD::from_elem(IxDyn(&[seq_len, hidden_dim]), 0.5_f32);
    BoundedTensor::from_epsilon(values, 0.1).expect("bounded input should be valid")
}

#[test]
fn traced_tensor_shapes_fold_constant_prelude_before_add() {
    let graph_model = style_gate_constant_graph_model();

    let graph = graph_model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("tensor_shapes should be enough to build the traced constant prelude");

    assert!(
        graph.node("style_linear").is_none(),
        "tensor_shapes should let the traced constant linear prelude fold away"
    );
    assert!(
        graph.node("style_split_slice_0").is_none(),
        "tensor_shapes should let the constant split outputs fold away before graph construction"
    );

    let add = graph
        .node("mixed_add")
        .expect("mixed add node should exist after constant folding");
    assert_eq!(
        add.inputs(),
        &[NETWORK_INPUT.to_string()],
        "only the live activation input should remain after folding the traced constant branch"
    );
}

#[test]
fn traced_tensor_producer_path_supports_parallel_verification_through_curated_api() {
    let graph = traced_expand_graph_model(&[1, 2, 1])
        .build_graph_network(GraphNetworkOptions::default())
        .expect("tensor_producer metadata should build a traced graph for parallel verification");
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), vec![0.1, 0.3]).expect("valid lower bounds");
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), vec![0.2, 0.4]).expect("valid upper bounds");
    let input = BoundedTensor::new(lower, upper).expect("valid bounded tensor");

    let output = verify_parallel_with_method(&graph, &input, 1, PropagationMethod::Ibp)
        .expect("traced producer handoff should remain usable through parallel verification");

    assert_eq!(output.shape(), &[1, 2, 1]);
    assert_tensor_bounds(&output, &[0.1, 0.3], &[0.2, 0.4]);
}

#[test]
fn curated_parallel_helpers_match_engine_aware_bounds_and_hit_gemm() {
    let graph = small_parallel_graph(4);
    let input = small_parallel_input(2, 4);

    let baseline = verify_parallel(&graph, &input, 0)
        .expect("CPU-default curated parallel helper should verify a small graph");
    let default_engine = CountingGemmEngine::default();
    // `verify_parallel*` defaults to IBP, so this path checks facade parity
    // while the explicit CROWN helper below proves the engine route is used.
    let with_engine = verify_parallel_with_engine(&graph, &input, 0, Arc::new(default_engine))
        .expect("engine-aware curated parallel helper should verify a small graph");

    assert_eq!(
        baseline.lower().iter().copied().collect::<Vec<_>>(),
        with_engine.lower().iter().copied().collect::<Vec<_>>(),
        "curated engine-aware lower bounds should match the CPU-default helper"
    );
    assert_eq!(
        baseline.upper().iter().copied().collect::<Vec<_>>(),
        with_engine.upper().iter().copied().collect::<Vec<_>>(),
        "curated engine-aware upper bounds should match the CPU-default helper"
    );

    let crown_baseline = verify_parallel_with_method(&graph, &input, 0, PropagationMethod::Crown)
        .expect("curated CROWN helper should verify a small graph");
    let crown_engine = CountingGemmEngine::default();
    let crown_with_engine = verify_parallel_with_method_and_engine(
        &graph,
        &input,
        0,
        PropagationMethod::Crown,
        Arc::new(crown_engine.clone()),
    )
    .expect("engine-aware curated CROWN helper should verify a small graph");

    // The CROWN path multiplies through the GemmEngine, so the default helper
    // and the explicit-engine helper accumulate the same dot products in a
    // possibly different order (faer's reduction order is not bit-stable across
    // versions). That makes the two routes agree only up to f32 rounding — a few
    // ULPs — not bit-for-bit. A genuine engine-routing regression would diverge
    // by orders of magnitude, which this tolerance still catches.
    let assert_bounds_close = |label: &str, a: Vec<f32>, b: Vec<f32>| {
        assert_eq!(a.len(), b.len(), "{label}: bound length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let tol = 1e-5_f32 * (1.0 + x.abs().max(y.abs()));
            assert!(
                (x - y).abs() <= tol,
                "{label}: element {i} diverged beyond f32 rounding: {x} vs {y}"
            );
        }
    };
    assert_bounds_close(
        "engine-aware curated CROWN lower bounds should match the CPU-default helper",
        crown_baseline.lower().iter().copied().collect(),
        crown_with_engine.lower().iter().copied().collect(),
    );
    assert_bounds_close(
        "engine-aware curated CROWN upper bounds should match the CPU-default helper",
        crown_baseline.upper().iter().copied().collect(),
        crown_with_engine.upper().iter().copied().collect(),
    );
    assert!(
        crown_engine.gemm_calls() > 0,
        "engine-aware curated CROWN helper should route through GemmEngine"
    );
}

#[test]
fn traced_tensor_shapes_path_verifies_folded_graph_through_curated_verifier() {
    let graph = style_gate_constant_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("tensor_shapes metadata should build a traced constant-prelude graph");
    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    });
    let spec = VerificationSpec::from_parts(
        bounds(&[(0.0, 1.0), (0.0, 1.0)]),
        bounds(&[(10.0, 11.0), (20.0, 21.0)]),
        Some(5_000),
        Some(vec![1, 1, 2]),
    )
    .expect("valid folded-graph verification spec");

    let result = verifier
        .verify_graph(&graph, &spec)
        .expect("folded traced graph should remain verifiable through ny_api::verify");

    match result {
        VerificationResult::Verified {
            output_bounds,
            actual_method,
            ..
        } => {
            assert_eq!(output_bounds, bounds(&[(10.0, 11.0), (20.0, 21.0)]));
            assert_eq!(actual_method.as_deref(), Some("Ibp"));
        }
        other => panic!("expected folded traced graph to verify, got {other:?}"),
    }
}

#[test]
fn test_talker_like_graph_model_multi_frozen_inputs_match_exact_ibp_bounds_3924() {
    let graph_model = talker_like_rotary_bias_graph_model();
    assert_talker_like_graph_model_metadata(&graph_model);

    let graph = graph_model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("the direct talker-like packet should build through the curated GraphModel API");
    assert_talker_like_runtime_path(&graph);

    let output = graph
        .propagate_ibp(&exact_unbatched_input(&[4, 2], 0.0_f32, 1.0_f32))
        .expect("the direct talker-like packet should propagate exact IBP bounds");

    assert_eq!(
        output.lower().shape(),
        &[4_usize, 2_usize],
        "the talker-like packet should preserve the unbatched [4, 2] runtime output shape"
    );
    assert_tensor_bounds(&output, &[4.0_f32; 8], &[5.0_f32; 8]);
}

fn cumsum_graph_model() -> GraphModel {
    GraphModelBuilder::new("cumsum-handoff")
        .input("data", &[1, 3], DataType::Float32)
        .output("out", &[1, 3], DataType::Float32)
        // Axis tensor: ONNX axis=1 as a scalar constant weight.
        // The converter remaps positive ONNX axes to the trailing-relative
        // (negative) internal encoding, so ONNX axis=1 of the rank-2 [1, 3]
        // input becomes adjusted_axis=-1 in CumsumLayer; the layer resolves
        // it against the actual runtime rank at propagation time.
        .weight(
            "axis_tensor",
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0_f32]).expect("valid axis scalar tensor"),
        )
        .constant_tensor("axis_tensor")
        .tensor_shape("axis_tensor", &[1])
        .layer(attributed_layer_spec(
            "cumsum",
            LayerType::CumSum,
            &["data", "axis_tensor"],
            &["out"],
            HashMap::from([
                ("exclusive".to_string(), AttributeValue::Int(0)),
                ("reverse".to_string(), AttributeValue::Int(0)),
            ]),
        ))
        .tensor_shape("data", &[1, 3])
        .tensor_shape("out", &[1, 3])
        .build()
}

#[test]
fn test_cumsum_handoff_layer_type_builds_to_layer_cumsum_3949() {
    let graph_model = cumsum_graph_model();
    let graph = graph_model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("CumSum handoff should build through the curated GraphModel API");

    let cumsum_node = graph.node("cumsum").expect("cumsum node should exist");
    assert_eq!(
        cumsum_node.inputs(),
        &[NETWORK_INPUT.to_string()],
        "the CumSum node should route only the live activation input through NETWORK_INPUT"
    );

    // Explicit type annotation proves CumsumLayer is importable from the facade.
    let layer: &CumsumLayer = match cumsum_node.layer() {
        Layer::CumSum(layer) => layer,
        other => panic!("expected Layer::CumSum(CumsumLayer), got {other:?}"),
    };
    assert_eq!(
        layer.axis, -1,
        "ONNX axis=1 of the rank-2 [1, 3] input should become the \
         trailing-relative axis -1 (batch-layout-independent encoding)"
    );
    assert!(!layer.exclusive, "exclusive=0 should map to false");
    assert!(!layer.reverse, "reverse=0 should map to false");
}

// ---------------------------------------------------------------------------
// Owned `GraphModel::new(...)` traced-producer contract (#3958)
//
// Mirrors the builder-based talker-like rotary-bias packet above, but
// constructs the `NetworkSpec` and `WeightStore` directly — the shape an
// external translator naturally has in hand after layer-spec
// translation.
// ---------------------------------------------------------------------------

fn talker_like_rotary_bias_owned_graph_model() -> GraphModel {
    let network_spec = NetworkSpec {
        name: "talker-like-rotary-bias-owned".to_string(),
        inputs: vec![TensorSpec {
            name: "hidden_states".to_string(),
            shape: vec![1, 4, 2],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "out".to_string(),
            shape: vec![1, 4, 2],
            dtype: DataType::Float32,
        }],
        layers: vec![
            layer_spec(
                "cos_merge",
                LayerType::Add,
                &["hidden_states", "cos"],
                &["hidden_plus_cos"],
            ),
            layer_spec(
                "pre_mask_merge",
                LayerType::Add,
                &["hidden_plus_cos", "sin"],
                &["pre_mask_out"],
            ),
            talker_like_mask_slice_layer(),
            layer_spec(
                "output_merge",
                LayerType::Add,
                &["pre_mask_out", "mask_gate_out"],
                &["out"],
            ),
        ],
        param_count: 0,
    };

    let mut weights = WeightStore::default();
    weights.insert(
        "cos".to_string(),
        ArrayD::from_elem(IxDyn(&[4, 2]), 1.0_f32),
    );
    weights.insert(
        "sin".to_string(),
        ArrayD::from_elem(IxDyn(&[4, 2]), 2.0_f32),
    );
    weights.insert(
        "mask".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[4, 4]),
            vec![
                1.0_f32, 1.0, 9.0, 9.0, //
                1.0, 1.0, 9.0, 9.0, //
                1.0, 1.0, 9.0, 9.0, //
                1.0, 1.0, 9.0, 9.0,
            ],
        )
        .expect("valid mask tensor"),
    );

    GraphModel::new(network_spec, weights)
        .with_constant_tensors(HashSet::from([
            "cos".to_string(),
            "sin".to_string(),
            "mask".to_string(),
        ]))
        .with_tensor_shapes(HashMap::from([
            ("hidden_states".to_string(), vec![1, 4, 2]),
            ("hidden_plus_cos".to_string(), vec![1, 4, 2]),
            ("pre_mask_out".to_string(), vec![1, 4, 2]),
            ("mask_gate_out".to_string(), vec![1, 4, 2]),
            ("out".to_string(), vec![1, 4, 2]),
            ("cos".to_string(), vec![1, 4, 2]),
            ("sin".to_string(), vec![1, 4, 2]),
            ("mask".to_string(), vec![1, 4, 4]),
        ]))
}

#[test]
fn traced_owned_graph_model_preserves_talker_like_frozen_aux_metadata_3958() {
    let graph_model = talker_like_rotary_bias_owned_graph_model();
    assert_talker_like_graph_model_metadata(&graph_model);
}

#[test]
fn traced_owned_graph_model_builds_talker_like_runtime_path_3958() {
    let graph_model = talker_like_rotary_bias_owned_graph_model();
    let graph = graph_model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("owned GraphModel::new should build the same verification graph as the builder");
    assert_talker_like_runtime_path(&graph);
}

#[test]
fn traced_owned_graph_model_matches_builder_talker_like_bounds_3958() {
    let owned = talker_like_rotary_bias_owned_graph_model();
    let builder = talker_like_rotary_bias_graph_model();

    let input = exact_unbatched_input(&[4, 2], 0.0_f32, 1.0_f32);

    let owned_graph = owned
        .build_graph_network(GraphNetworkOptions::default())
        .expect("owned contract should build");
    let builder_graph = builder
        .build_graph_network(GraphNetworkOptions::default())
        .expect("builder contract should build");

    let owned_output = owned_graph
        .propagate_ibp(&input)
        .expect("owned contract should propagate IBP");
    let builder_output = builder_graph
        .propagate_ibp(&input)
        .expect("builder contract should propagate IBP");

    assert_eq!(
        owned_output.lower().iter().copied().collect::<Vec<_>>(),
        builder_output.lower().iter().copied().collect::<Vec<_>>(),
        "owned lower bounds should match builder lower bounds"
    );
    assert_eq!(
        owned_output.upper().iter().copied().collect::<Vec<_>>(),
        builder_output.upper().iter().copied().collect::<Vec<_>>(),
        "owned upper bounds should match builder upper bounds"
    );
    // Also verify the absolute values match expectations
    assert_tensor_bounds(&owned_output, &[4.0_f32; 8], &[5.0_f32; 8]);
}
