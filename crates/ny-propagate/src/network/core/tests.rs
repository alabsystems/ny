// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::domain_clip::DomainClipper;
use crate::layers::{
    AddLayer, ArctanLayer, AttentionMask, CausalSoftmaxLayer, CosLayer, GELULayer,
    GeluApproximation, Layer, LayerNormLayer, LayerNormMode, LinearLayer, LogSoftmaxLayer,
    OpaqueSkipLayer, ReLULayer, SelfAttentionLayer, SinLayer, SkipMergeLayer, SoftmaxLayer,
    TanLayer,
};
use ndarray::{arr1, arr2, array};
use ny_tensor::BoundedTensor;

// broadcast_shapes→shape/mod.rs, relu_ibp→layers/activations/relu/ibp.rs, relu_crown→relu_relax.rs
// AttentionGraphBuilder tests moved to graph_builder.rs (Part of #170)

#[ntest::timeout(10000)]
#[test]
fn test_propagate_crown_to_node_add_splits_bias() {
    // Regression test: in backward CROWN-to-node, Add must split the bias term;
    // otherwise downstream constants get double-counted across both branches.

    // Two-branch linear -> add -> linear(with bias) -> output
    let wa = arr2(&[[1.0, 2.0], [-3.0, 0.5]]);
    let ba = arr1(&[0.1, -0.2]);
    let wb = arr2(&[[0.3, -0.7], [1.2, -1.0]]);
    let bb = arr1(&[0.0, 0.05]);
    let wout = arr2(&[[2.0, -1.0]]);
    let bout = arr1(&[0.7]);
    let lin_a = LinearLayer::new(wa.clone(), Some(ba.clone())).unwrap();
    let lin_b = LinearLayer::new(wb.clone(), Some(bb.clone())).unwrap();
    let lin_out = LinearLayer::new(wout.clone(), Some(bout.clone())).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "lin_a",
        Layer::Linear(lin_a),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin_b",
        Layer::Linear(lin_b),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "add",
        Layer::Add(AddLayer),
        "lin_a",
        "lin_b",
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(lin_out),
        vec!["add".to_string()],
    ));
    graph.set_output("out");
    let input =
        BoundedTensor::new(array![-1.0, -2.0].into_dyn(), array![3.0, 4.0].into_dyn()).unwrap();
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let crown = graph
        .propagate_crown_to_node(
            &input,
            "out",
            &std::collections::HashMap::new(),
            &ibp_bounds,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // Expected exact bounds since the full graph is linear.
    let wsum = &wa + &wb; // 2x2
    let bsum = &ba + &bb; // 2
    let combined_w = wout.dot(&wsum); // 1x2
    let combined_bias = wout.row(0).dot(&bsum) + bout[0];

    let l = [-1.0_f32, -2.0_f32];
    let u = [3.0_f32, 4.0_f32];
    let w0 = combined_w[[0, 0]];
    let w1 = combined_w[[0, 1]];

    let mut expected_lower = combined_bias;
    let mut expected_upper = combined_bias;
    for (w, (li, ui)) in [(w0, (l[0], u[0])), (w1, (l[1], u[1]))] {
        if w >= 0.0 {
            expected_lower += w * li;
            expected_upper += w * ui;
        } else {
            expected_lower += w * ui;
            expected_upper += w * li;
        }
    }

    let got_lower = crown.lower()[[0]];
    let got_upper = crown.upper()[[0]];
    assert!((got_lower - expected_lower).abs() < 1e-4);
    assert!((got_upper - expected_upper).abs() < 1e-4);
}

// ==================== Network tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_network_new_empty() {
    let net = Network::new();
    assert_eq!(net.layers.len(), 0);
    assert_eq!(net.num_layers(), 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_network_add_layer() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));
    assert_eq!(net.layers.len(), 1);
    assert_eq!(net.num_layers(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_network_crown_skipmerge_identity() {
    let w1 = arr2(&[[1.0, -2.0], [0.5, 3.0]]);
    let b1 = arr1(&[0.1, -0.4]);
    let w2 = arr2(&[[2.0, -1.5]]);
    let b2 = arr1(&[0.25]);

    let lin1 = LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap();
    let lin2 = LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap();

    let mut net = Network::new();
    net.add_layer(Layer::Linear(lin1));
    net.add_layer(Layer::SkipMerge(SkipMergeLayer::new()));
    net.add_layer(Layer::Linear(lin2));

    let input =
        BoundedTensor::new(array![-1.0, 0.5].into_dyn(), array![2.0, 3.0].into_dyn()).unwrap();
    let output = net.propagate_crown(&input).unwrap();

    let combined_w = w2.dot(&w1);
    let combined_bias = w2.row(0).dot(&b1) + b2[0];

    let l = [-1.0_f32, 0.5_f32];
    let u = [2.0_f32, 3.0_f32];
    let w0 = combined_w[[0, 0]];
    let w1 = combined_w[[0, 1]];

    let mut expected_lower = combined_bias;
    let mut expected_upper = combined_bias;
    for (w, (li, ui)) in [(w0, (l[0], u[0])), (w1, (l[1], u[1]))] {
        if w >= 0.0 {
            expected_lower += w * li;
            expected_upper += w * ui;
        } else {
            expected_lower += w * ui;
            expected_upper += w * li;
        }
    }

    let got_lower = output.lower()[[0]];
    let got_upper = output.upper()[[0]];
    assert!((got_lower - expected_lower).abs() < 1e-4);
    assert!((got_upper - expected_upper).abs() < 1e-4);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_opaque_skip_multi_input_unbounded() {
    let wa = arr2(&[[1.0, -2.0], [0.5, 3.0]]);
    let ba = arr1(&[0.1, -0.4]);
    let wb = arr2(&[[2.0, -1.5], [0.2, 0.3]]);
    let bb = arr1(&[0.25, -0.1]);

    let lin_a = LinearLayer::new(wa, Some(ba)).unwrap();
    let lin_b = LinearLayer::new(wb, Some(bb)).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "lin_a",
        Layer::Linear(lin_a),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin_b",
        Layer::Linear(lin_b),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "skip",
        Layer::OpaqueSkip(OpaqueSkipLayer::new()),
        vec!["lin_a".to_string(), "lin_b".to_string()],
    ));
    graph.set_output("skip");

    let input =
        BoundedTensor::new(array![-1.0, 0.5].into_dyn(), array![2.0, 3.0].into_dyn()).unwrap();
    let output = graph.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2]);
    assert!(output
        .lower()
        .iter()
        .all(|v| v.is_infinite() && v.is_sign_negative()));
    assert!(output
        .upper()
        .iter()
        .all(|v| v.is_infinite() && v.is_sign_positive()));
}

#[ntest::timeout(10000)]
#[test]
fn test_network_set_layernorm_forward_mode() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));

    // No LayerNorm layers
    let count = net.set_layernorm_forward_mode(true);
    assert_eq!(count, 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_network_set_layernorm_forward_mode_updates() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::LayerNorm(
        LayerNormLayer::new_default(2, 1e-5).unwrap(),
    ));

    let count = net.set_layernorm_forward_mode(true);
    assert_eq!(count, 1);

    match &net.layers[1] {
        Layer::LayerNorm(ln) => assert!(ln.forward_mode),
        _ => panic!("Expected LayerNorm layer at index 1"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_set_layernorm_norm_mode() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::LayerNorm(
        LayerNormLayer::new_default(2, 1e-5).unwrap(),
    ));

    let count = net.set_layernorm_norm_mode(LayerNormMode::MeanOnly);
    assert_eq!(count, 1);

    match &net.layers[1] {
        Layer::LayerNorm(ln) => assert_eq!(ln.mode, LayerNormMode::MeanOnly),
        _ => panic!("Expected LayerNorm layer at index 1"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_set_layernorm_norm_mode() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "layernorm",
        Layer::LayerNorm(LayerNormLayer::new_default(2, 1e-5).unwrap()),
        vec!["_input".to_string()],
    ));
    graph.set_output("layernorm");

    let original_scope = graph.cut_fold_scope();
    assert_eq!(graph.set_layernorm_norm_mode(LayerNormMode::Standard), 1);
    assert_eq!(
        graph.cut_fold_scope(),
        original_scope,
        "setting the already-active point semantics preserves graph identity"
    );

    let count = graph.set_layernorm_norm_mode(LayerNormMode::MeanOnly);
    assert_eq!(count, 1);
    assert_ne!(
        graph.cut_fold_scope(),
        original_scope,
        "changing the modeled point function must mint a new graph identity"
    );

    let node = graph.node("layernorm").expect("layernorm node missing");
    match &node.layer {
        Layer::LayerNorm(ln) => assert_eq!(ln.mode, LayerNormMode::MeanOnly),
        _ => panic!("Expected LayerNorm layer"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_set_layernorm_forward_mode_updates() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "layernorm",
        Layer::LayerNorm(LayerNormLayer::new_default(2, 1e-5).unwrap()),
        vec!["_input".to_string()],
    ));
    graph.set_output("layernorm");

    let count = graph.set_layernorm_forward_mode(true);
    assert_eq!(count, 1);

    let node = graph.node("layernorm").expect("layernorm node missing");
    match &node.layer {
        Layer::LayerNorm(ln) => assert!(ln.forward_mode),
        _ => panic!("Expected LayerNorm layer"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_set_gelu_sound_mode() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::GELU(GELULayer::new(GeluApproximation::Tanh)));

    let count = net.set_gelu_sound_mode(true);
    assert_eq!(count, 1);

    match &net.layers[1] {
        Layer::GELU(gelu) => assert!(gelu.sound),
        _ => panic!("Expected GELU layer at index 1"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_set_gelu_sound_mode() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::new(GeluApproximation::Erf)),
        vec!["_input".to_string()],
    ));
    graph.set_output("gelu");

    let count = graph.set_gelu_sound_mode(true);
    assert_eq!(count, 1);

    let node = graph.node("gelu").expect("gelu node missing");
    match &node.layer {
        Layer::GELU(gelu) => assert!(gelu.sound),
        _ => panic!("Expected GELU layer"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_set_logsoftmax_sound_mode() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::LogSoftmax(LogSoftmaxLayer::new(-1)));

    let count = net.set_logsoftmax_sound_mode(true);
    assert_eq!(count, 1);

    match &net.layers[1] {
        Layer::LogSoftmax(logsoftmax) => assert!(logsoftmax.sound),
        _ => panic!("Expected LogSoftmax layer at index 1"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_set_logsoftmax_sound_mode() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "logsoftmax",
        Layer::LogSoftmax(LogSoftmaxLayer::new(-1)),
        vec!["_input".to_string()],
    ));
    graph.set_output("logsoftmax");

    let count = graph.set_logsoftmax_sound_mode(true);
    assert_eq!(count, 1);

    let node = graph.node("logsoftmax").expect("logsoftmax node missing");
    match &node.layer {
        Layer::LogSoftmax(logsoftmax) => assert!(logsoftmax.sound),
        _ => panic!("Expected LogSoftmax layer"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_ibp_detailed_empty_graph() {
    let graph = GraphNetwork::new();
    let input =
        BoundedTensor::new(array![-1.0, 0.0].into_dyn(), array![2.0, 3.0].into_dyn()).unwrap();

    let result = graph.propagate_ibp_detailed(&input, 0.1).unwrap();

    assert_eq!(result.nodes.len(), 0);
    assert_eq!(result.total_nodes, 0);
    assert!(result.degraded_at_node.is_none());
    assert!((result.final_width - input.max_width()).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_activation_statistics_empty_graph() {
    let graph = GraphNetwork::new();
    let input =
        BoundedTensor::new(array![-1.0, 0.0].into_dyn(), array![2.0, 3.0].into_dyn()).unwrap();
    let mut clipper = DomainClipper::default_config();

    graph
        .collect_activation_statistics(&input, &mut clipper)
        .unwrap();

    assert!(clipper.statistics.is_empty());
    assert_eq!(clipper.clip_count, 0);
    assert_eq!(clipper.total_width_reduction, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_activation_statistics_single_node() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["_input".to_string()],
    ));
    graph.set_output("relu");

    let input =
        BoundedTensor::new(array![-1.0, 0.0].into_dyn(), array![2.0, 3.0].into_dyn()).unwrap();
    let mut clipper = DomainClipper::default_config();

    graph
        .collect_activation_statistics(&input, &mut clipper)
        .unwrap();

    let stats = clipper.statistics("relu").expect("relu stats missing");
    assert_eq!(stats.num_samples, 1);
    assert_eq!(stats.shape, vec![2]);
}

#[ntest::timeout(10000)]
#[test]
fn test_network_set_softmax_sound_mode() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::Softmax(SoftmaxLayer::new(-1)));

    let count = net.set_softmax_sound_mode(true);
    assert_eq!(count, 1);

    match &net.layers[1] {
        Layer::Softmax(softmax) => assert!(softmax.sound),
        _ => panic!("Expected Softmax layer at index 1"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_crown_fast_tan_arctan_roundtrip() {
    let mut net = Network::new();
    net.add_layer(Layer::Tan(TanLayer::new()));
    net.add_layer(Layer::Arctan(ArctanLayer::new()));

    let input = BoundedTensor::new(array![-0.5].into_dyn(), array![0.5].into_dyn()).unwrap();
    let bounds = net.propagate_crown_fast(&input).unwrap();
    let ibp_bounds = net.propagate_ibp(&input).unwrap();

    assert!(
        bounds.lower()[0].is_finite(),
        "crown-fast lower non-finite: lower={}, upper={}, ibp=[{}, {}]",
        bounds.lower()[0],
        bounds.upper()[0],
        ibp_bounds.lower()[0],
        ibp_bounds.upper()[0]
    );
    assert!(
        bounds.upper()[0].is_finite(),
        "crown-fast upper non-finite: lower={}, upper={}, ibp=[{}, {}]",
        bounds.lower()[0],
        bounds.upper()[0],
        ibp_bounds.lower()[0],
        ibp_bounds.upper()[0]
    );

    for x in [-0.45_f32, -0.2, 0.0, 0.1, 0.4] {
        let y = x;
        assert!(
            y >= bounds.lower()[0] - 1e-4 && y <= bounds.upper()[0] + 1e-4,
            "atan(tan({}))={} not in [{}, {}]",
            x,
            y,
            bounds.lower()[0],
            bounds.upper()[0]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_set_softmax_sound_mode() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "softmax",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        vec!["_input".to_string()],
    ));
    graph.set_output("softmax");

    let count = graph.set_softmax_sound_mode(true);
    assert_eq!(count, 1);

    let node = graph.node("softmax").expect("softmax node missing");
    match &node.layer {
        Layer::Softmax(softmax) => assert!(softmax.sound),
        _ => panic!("Expected Softmax layer"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_set_sin_sound_mode() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::Sin(SinLayer::new()));

    let count = net.set_sin_sound_mode(true);
    assert_eq!(count, 1);

    match &net.layers[1] {
        Layer::Sin(sin) => assert!(sin.sound),
        _ => panic!("Expected Sin layer at index 1"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_set_sin_sound_mode() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "sin",
        Layer::Sin(SinLayer::new()),
        vec!["_input".to_string()],
    ));
    graph.set_output("sin");

    let count = graph.set_sin_sound_mode(true);
    assert_eq!(count, 1);

    let node = graph.node("sin").expect("sin node missing");
    match &node.layer {
        Layer::Sin(sin) => assert!(sin.sound),
        _ => panic!("Expected Sin layer"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_set_cos_sound_mode() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::Cos(CosLayer::new()));

    let count = net.set_cos_sound_mode(true);
    assert_eq!(count, 1);

    match &net.layers[1] {
        Layer::Cos(cos) => assert!(cos.sound),
        _ => panic!("Expected Cos layer at index 1"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_set_cos_sound_mode() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "cos",
        Layer::Cos(CosLayer::new()),
        vec!["_input".to_string()],
    ));
    graph.set_output("cos");

    let count = graph.set_cos_sound_mode(true);
    assert_eq!(count, 1);

    let node = graph.node("cos").expect("cos node missing");
    match &node.layer {
        Layer::Cos(cos) => assert!(cos.sound),
        _ => panic!("Expected Cos layer"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_set_causal_softmax_sound_mode() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)));

    let count = net.set_causal_softmax_sound_mode(true);
    assert_eq!(count, 1);

    match &net.layers[1] {
        Layer::CausalSoftmax(softmax) => assert!(softmax.sound),
        _ => panic!("Expected CausalSoftmax layer at index 1"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_set_causal_softmax_sound_mode() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "causal_softmax",
        Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)),
        vec!["_input".to_string()],
    ));
    graph.set_output("causal_softmax");

    let count = graph.set_causal_softmax_sound_mode(true);
    assert_eq!(count, 1);

    let node = graph
        .node("causal_softmax")
        .expect("causal_softmax node missing");
    match &node.layer {
        Layer::CausalSoftmax(softmax) => assert!(softmax.sound),
        _ => panic!("Expected CausalSoftmax layer"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_propagate_ibp_empty() {
    let net = Network::new();
    let input =
        BoundedTensor::new(array![1.0, 2.0].into_dyn(), array![2.0, 3.0].into_dyn()).unwrap();
    let output = net.propagate_ibp(&input).unwrap();
    // Empty network returns input unchanged
    assert_eq!(output.lower(), input.lower());
    assert_eq!(output.upper(), input.upper());
}

#[ntest::timeout(10000)]
#[test]
fn test_network_propagate_ibp_relu() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(array![-1.0, 1.0].into_dyn(), array![1.0, 2.0].into_dyn()).unwrap();
    let output = net.propagate_ibp(&input).unwrap();
    assert_eq!(output.lower(), array![0.0, 1.0].into_dyn());
    assert_eq!(output.upper(), array![1.0, 2.0].into_dyn());
}

#[ntest::timeout(10000)]
#[test]
fn test_network_collect_ibp_bounds_empty() {
    let net = Network::new();
    let input = BoundedTensor::new(array![1.0].into_dyn(), array![2.0].into_dyn()).unwrap();
    let bounds = net.collect_ibp_bounds(&input).unwrap();
    assert_eq!(bounds.len(), 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_network_collect_ibp_bounds_single() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(array![-1.0, 2.0].into_dyn(), array![1.0, 3.0].into_dyn()).unwrap();
    let bounds = net.collect_ibp_bounds(&input).unwrap();
    assert_eq!(bounds.len(), 1);
    assert_eq!(bounds[0].lower(), array![0.0, 2.0].into_dyn());
    assert_eq!(bounds[0].upper(), array![1.0, 3.0].into_dyn());
}

#[ntest::timeout(10000)]
#[test]
fn test_network_collect_ibp_bounds_chain() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(array![-1.0, 2.0].into_dyn(), array![1.0, 3.0].into_dyn()).unwrap();
    let bounds = net.collect_ibp_bounds(&input).unwrap();
    assert_eq!(bounds.len(), 2);
    // First ReLU
    assert_eq!(bounds[0].lower(), array![0.0, 2.0].into_dyn());
    assert_eq!(bounds[0].upper(), array![1.0, 3.0].into_dyn());
    // Second ReLU (no change since already non-negative lower)
    assert_eq!(bounds[1].lower(), array![0.0, 2.0].into_dyn());
    assert_eq!(bounds[1].upper(), array![1.0, 3.0].into_dyn());
}

// ==================== GraphNetwork tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_new() {
    let graph = GraphNetwork::new();
    assert_eq!(graph.num_nodes(), 0);
    assert_eq!(graph.node_names().len(), 0);
    assert_eq!(graph.output_name(), "");
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_add_node() {
    let mut graph = GraphNetwork::new();
    let node = GraphNode::from_input("input", Layer::ReLU(ReLULayer));
    graph.add_node(node);

    assert_eq!(graph.num_nodes(), 1);
    assert!(graph.node("input").is_some());
}

#[allow(deprecated)]
#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_get_node_alias_matches_node() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("input", Layer::ReLU(ReLULayer)));

    let node = graph.node("input").expect("node lookup");
    let alias = graph.get_node("input").expect("get_node lookup");

    assert_eq!(alias.name(), node.name());
    assert_eq!(alias.layer().layer_type(), node.layer().layer_type());
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_try_add_node_duplicate_error() {
    // Part of #172: Validate try_add_node returns error on duplicate names
    let mut graph = GraphNetwork::new();
    let node1 = GraphNode::from_input("duplicate", Layer::ReLU(ReLULayer));
    graph.try_add_node(node1).unwrap();

    // Second add with same name should fail
    let node2 = GraphNode::from_input("duplicate", Layer::ReLU(ReLULayer));
    let result = graph.try_add_node(node2);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));

    // Graph should still have only 1 node
    assert_eq!(graph.num_nodes(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_set_output() {
    let mut graph = GraphNetwork::new();
    graph.set_output("output");
    assert_eq!(graph.output_name(), "output");
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_topological_sort_linear() {
    // Linear chain: A -> B -> C
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("A", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "B",
        Layer::ReLU(ReLULayer),
        vec!["A".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "C",
        Layer::ReLU(ReLULayer),
        vec!["B".to_string()],
    ));
    graph.set_output("C");

    let sorted = graph.topological_sort().unwrap();
    assert_eq!(sorted, vec!["A", "B", "C"]);
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_topological_sort_diamond() {
    // Diamond: A -> B, A -> C, B -> D, C -> D
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("A", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "B",
        Layer::ReLU(ReLULayer),
        vec!["A".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "C",
        Layer::ReLU(ReLULayer),
        vec!["A".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "D",
        Layer::ReLU(ReLULayer),
        vec!["B".to_string(), "C".to_string()],
    ));
    graph.set_output("D");

    let sorted = graph.topological_sort().unwrap();
    // A must come first, D must come last, B and C can be in either order
    assert_eq!(sorted[0], "A");
    assert_eq!(sorted[3], "D");
    assert!(sorted[1] == "B" || sorted[1] == "C");
    assert!(sorted[2] == "B" || sorted[2] == "C");
    assert_ne!(sorted[1], sorted[2]);
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_exec_order_reuses_cached_slice_4295() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("A", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "B",
        Layer::ReLU(ReLULayer),
        vec!["A".to_string()],
    ));

    let first_order = graph.exec_order().unwrap();
    assert_eq!(
        first_order.iter().map(String::as_str).collect::<Vec<_>>(),
        vec!["A", "B"]
    );
    let first_ptr = first_order.as_ptr();

    let second_order = graph.exec_order().unwrap();
    assert_eq!(
        second_order.iter().map(String::as_str).collect::<Vec<_>>(),
        vec!["A", "B"]
    );
    assert_eq!(first_ptr, second_order.as_ptr());
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_exec_order_invalidates_after_add_and_try_add_4295() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("A", Layer::ReLU(ReLULayer)));

    assert_eq!(
        graph
            .exec_order()
            .unwrap()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["A"]
    );

    graph.add_node(GraphNode::new(
        "B",
        Layer::ReLU(ReLULayer),
        vec!["A".to_string()],
    ));
    assert_eq!(
        graph
            .exec_order()
            .unwrap()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["A", "B"]
    );

    graph
        .try_add_node(GraphNode::new(
            "C",
            Layer::ReLU(ReLULayer),
            vec!["B".to_string()],
        ))
        .unwrap();
    assert_eq!(graph.topological_sort().unwrap(), vec!["A", "B", "C"]);
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_exec_order_invalidates_on_self_attention_decompose_4295() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("q", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("k", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::from_input("v", Layer::ReLU(ReLULayer)));

    assert_eq!(
        graph
            .exec_order()
            .unwrap()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["k", "q", "v"]
    );

    graph
        .try_add_node(GraphNode::new(
            "attn",
            Layer::SelfAttention(SelfAttentionLayer::new(AttentionMask::Standard, Some(1.0))),
            vec!["q".to_string(), "k".to_string(), "v".to_string()],
        ))
        .unwrap();

    let exec_order = graph
        .exec_order()
        .unwrap()
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let position = |name: &str| {
        exec_order
            .iter()
            .position(|candidate| *candidate == name)
            .unwrap()
    };

    assert_eq!(exec_order.len(), 6);
    assert!(position("q") < position("attn/qk"));
    assert!(position("k") < position("attn/qk"));
    assert!(position("attn/qk") < position("attn/softmax"));
    assert!(position("attn/softmax") < position("attn"));
    assert!(position("v") < position("attn"));
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_propagate_ibp_single_node() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input =
        BoundedTensor::new(array![-1.0, 1.0].into_dyn(), array![1.0, 2.0].into_dyn()).unwrap();
    let output = graph.propagate_ibp(&input).unwrap();
    assert_eq!(output.lower(), array![0.0, 1.0].into_dyn());
    assert_eq!(output.upper(), array![1.0, 2.0].into_dyn());
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_propagate_ibp_chain() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu1", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["relu1".to_string()],
    ));
    graph.set_output("relu2");

    let input =
        BoundedTensor::new(array![-2.0, 1.0].into_dyn(), array![1.0, 3.0].into_dyn()).unwrap();
    let output = graph.propagate_ibp(&input).unwrap();
    // After two ReLUs, result should be same as one ReLU on this input
    assert_eq!(output.lower(), array![0.0, 1.0].into_dyn());
    assert_eq!(output.upper(), array![1.0, 3.0].into_dyn());
}

// ==================== GraphNode tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_graphnode_from_input() {
    let node = GraphNode::from_input("input_node", Layer::ReLU(ReLULayer));
    assert_eq!(node.name, "input_node");
    // from_input sets "_input" as the input source
    assert_eq!(node.inputs, vec!["_input"]);
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnode_new_with_inputs() {
    let node = GraphNode::new(
        "compute",
        Layer::ReLU(ReLULayer),
        vec!["a".to_string(), "b".to_string()],
    );
    assert_eq!(node.name, "compute");
    assert_eq!(node.inputs, vec!["a", "b"]);
}

/// Regression test for #2686: duplicate node names must return Err, not panic.
#[ntest::timeout(10000)]
#[test]
fn test_try_add_node_duplicate_returns_error_2686() {
    let mut graph = GraphNetwork::new();
    let node1 = GraphNode::new("relu_0", Layer::ReLU(ReLULayer), vec!["input".to_string()]);
    let node2 = GraphNode::new("relu_0", Layer::ReLU(ReLULayer), vec!["input".to_string()]);

    graph
        .try_add_node(node1)
        .expect("invariant: first add succeeds");
    let err = graph
        .try_add_node(node2)
        .expect_err("duplicate node 'relu_0' must return Err");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(_)),
        "expected InvalidSpec for duplicate node, got {err:?}"
    );
}

// AttentionGraphBuilder tests moved to graph_builder.rs (Part of #170)
