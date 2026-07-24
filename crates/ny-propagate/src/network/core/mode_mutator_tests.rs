// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Parity matrix tests for layer-mode mutators.
//!
//! Verifies that all 9 mode mutators produce identical effects on equivalent
//! `Network` and `GraphNetwork` fixtures. Part of #2803.

use super::{GraphNetwork, GraphNode, Network};
use crate::layers::{
    CausalSoftmaxLayer, CosLayer, GELULayer, GeluApproximation, GroupNormLayer,
    InstanceNorm1dLayer, Layer, LayerNormCrownMode, LayerNormLayer, LayerNormMode, LogSoftmaxLayer,
    ReLULayer, SinLayer, SoftmaxLayer,
};

fn build_network() -> Network {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(Layer::LayerNorm(
        LayerNormLayer::new_default(2, 1e-5).expect("invariant: valid test eps"),
    ));
    net.add_layer(Layer::GroupNorm(
        GroupNormLayer::new_default(2, 1, 1e-5).expect("invariant: valid test eps"),
    ));
    net.add_layer(Layer::GELU(GELULayer::new(GeluApproximation::Tanh)));
    net.add_layer(Layer::LogSoftmax(LogSoftmaxLayer::new(-1)));
    net.add_layer(Layer::Softmax(SoftmaxLayer::new(-1)));
    net.add_layer(Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)));
    net.add_layer(Layer::Sin(SinLayer::new()));
    net.add_layer(Layer::Cos(CosLayer::new()));
    net
}

fn build_graph() -> GraphNetwork {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    g.add_node(GraphNode::new(
        "ln",
        Layer::LayerNorm(LayerNormLayer::new_default(2, 1e-5).expect("invariant: valid test eps")),
        vec!["relu".into()],
    ));
    g.add_node(GraphNode::new(
        "gn",
        Layer::GroupNorm(
            GroupNormLayer::new_default(2, 1, 1e-5).expect("invariant: valid test eps"),
        ),
        vec!["ln".into()],
    ));
    g.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::new(GeluApproximation::Tanh)),
        vec!["gn".into()],
    ));
    g.add_node(GraphNode::new(
        "lsm",
        Layer::LogSoftmax(LogSoftmaxLayer::new(-1)),
        vec!["gelu".into()],
    ));
    g.add_node(GraphNode::new(
        "sm",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        vec!["lsm".into()],
    ));
    g.add_node(GraphNode::new(
        "csm",
        Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)),
        vec!["sm".into()],
    ));
    g.add_node(GraphNode::new(
        "sin",
        Layer::Sin(SinLayer::new()),
        vec!["csm".into()],
    ));
    g.add_node(GraphNode::new(
        "cos",
        Layer::Cos(CosLayer::new()),
        vec!["sin".into()],
    ));
    g.set_output("cos");
    g
}

fn build_relu_chain_graph(num_nodes: usize) -> GraphNetwork {
    assert!(num_nodes > 0, "test graph should contain at least one node");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    for idx in 1..num_nodes {
        graph.add_node(GraphNode::new(
            format!("relu{idx}"),
            Layer::ReLU(ReLULayer),
            vec![format!("relu{}", idx - 1)],
        ));
    }
    graph.set_output(format!("relu{}", num_nodes - 1));
    graph
}

/// Assert one mutator produces the same count on both surfaces.
fn assert_parity(label: &str, net_count: usize, graph_count: usize, expected: usize) {
    assert_eq!(net_count, graph_count, "{label} count mismatch");
    assert_eq!(net_count, expected, "{label} expected {expected} match(es)");
}
#[ntest::timeout(10000)]
#[test]
fn test_mode_mutators_parity_layernorm_forward() {
    let mut n = build_network();
    let mut g = build_graph();
    // 2 = LayerNorm + GroupNorm (both respond to set_layernorm_forward_mode)
    assert_parity(
        "ln_fwd",
        n.set_layernorm_forward_mode(true),
        g.set_layernorm_forward_mode(true),
        2,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mode_mutators_parity_layernorm_crown() {
    let mut n = build_network();
    let mut g = build_graph();
    let mode = LayerNormCrownMode::Cut;
    // 2 = LayerNorm + GroupNorm (both respond to set_layernorm_crown_mode)
    assert_parity(
        "ln_crown",
        n.set_layernorm_crown_mode(mode),
        g.set_layernorm_crown_mode(mode),
        2,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mode_mutators_parity_layernorm_norm() {
    let mut n = build_network();
    let mut g = build_graph();
    let mode = LayerNormMode::MeanOnly;
    assert_parity(
        "ln_norm",
        n.set_layernorm_norm_mode(mode),
        g.set_layernorm_norm_mode(mode),
        1,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mode_mutators_parity_gelu() {
    let mut n = build_network();
    let mut g = build_graph();
    assert_parity(
        "gelu",
        n.set_gelu_sound_mode(true),
        g.set_gelu_sound_mode(true),
        1,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mode_mutators_parity_logsoftmax() {
    let mut n = build_network();
    let mut g = build_graph();
    assert_parity(
        "lsm",
        n.set_logsoftmax_sound_mode(true),
        g.set_logsoftmax_sound_mode(true),
        1,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mode_mutators_parity_softmax() {
    let mut n = build_network();
    let mut g = build_graph();
    assert_parity(
        "sm",
        n.set_softmax_sound_mode(true),
        g.set_softmax_sound_mode(true),
        1,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mode_mutators_parity_causal_softmax() {
    let mut n = build_network();
    let mut g = build_graph();
    assert_parity(
        "csm",
        n.set_causal_softmax_sound_mode(true),
        g.set_causal_softmax_sound_mode(true),
        1,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mode_mutators_parity_sin() {
    let mut n = build_network();
    let mut g = build_graph();
    assert_parity(
        "sin",
        n.set_sin_sound_mode(true),
        g.set_sin_sound_mode(true),
        1,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mode_mutators_parity_cos() {
    let mut n = build_network();
    let mut g = build_graph();
    assert_parity(
        "cos",
        n.set_cos_sound_mode(true),
        g.set_cos_sound_mode(true),
        1,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mode_mutators_all_setters() {
    let mut net = build_network();
    net.set_layernorm_forward_mode(true);
    net.set_layernorm_norm_mode(LayerNormMode::MeanOnly);
    net.set_gelu_sound_mode(true);
    net.set_logsoftmax_sound_mode(true);
    net.set_softmax_sound_mode(true);
    net.set_causal_softmax_sound_mode(true);
    net.set_sin_sound_mode(true);
    net.set_cos_sound_mode(true);
    net.set_layernorm_crown_mode(LayerNormCrownMode::Cut);

    let mut graph = build_graph();
    graph.set_layernorm_forward_mode(true);
    graph.set_layernorm_norm_mode(LayerNormMode::MeanOnly);
    graph.set_gelu_sound_mode(true);
    graph.set_logsoftmax_sound_mode(true);
    graph.set_softmax_sound_mode(true);
    graph.set_causal_softmax_sound_mode(true);
    graph.set_sin_sound_mode(true);
    graph.set_cos_sound_mode(true);
    graph.set_layernorm_crown_mode(LayerNormCrownMode::Cut);

    // Spot-check: LayerNorm on Network
    let Layer::LayerNorm(ln) = &net.layers[1] else {
        unreachable!("expected LayerNorm at index 1");
    };
    assert!(ln.forward_mode);
    assert_eq!(ln.mode, LayerNormMode::MeanOnly);
    assert_eq!(ln.crown_mode, LayerNormCrownMode::Cut);

    // Spot-check: GroupNorm on Network (Part of #3391)
    let Layer::GroupNorm(gn) = &net.layers[2] else {
        unreachable!("expected GroupNorm at index 2");
    };
    assert!(gn.forward_mode);
    assert_eq!(gn.crown_mode, LayerNormCrownMode::Cut);

    // Spot-check: LayerNorm on GraphNetwork
    let ln_node = graph.node("ln").expect("ln node missing");
    let Layer::LayerNorm(ln_g) = &ln_node.layer else {
        unreachable!("expected LayerNorm node");
    };
    assert!(ln_g.forward_mode);
    assert_eq!(ln_g.mode, LayerNormMode::MeanOnly);
    assert_eq!(ln_g.crown_mode, LayerNormCrownMode::Cut);

    // Spot-check: GroupNorm on GraphNetwork (Part of #3391)
    let gn_node = graph.node("gn").expect("gn node missing");
    let Layer::GroupNorm(gn_g) = &gn_node.layer else {
        unreachable!("expected GroupNorm node");
    };
    assert!(gn_g.forward_mode);
    assert_eq!(gn_g.crown_mode, LayerNormCrownMode::Cut);

    // Spot-check: GELU on both
    let Layer::GELU(gelu) = &net.layers[3] else {
        unreachable!("expected GELU at index 3");
    };
    assert!(gelu.sound);

    let gelu_node = graph.node("gelu").expect("gelu node missing");
    let Layer::GELU(gelu_g) = &gelu_node.layer else {
        unreachable!("expected GELU node");
    };
    assert!(gelu_g.sound);
}

/// Unary transformer ops (Softmax, CausalSoftmax, LayerNorm, RmsNorm, SiLU,
/// GELU) now allow CROWN-IBP intermediates because they have direct CROWN
/// backward support and the demand-driven selection limits tightening to
/// needed producers (#3775). Binary relaxation ops and GroupNorm/AdaIN1d
/// remain blocked.
#[ntest::timeout(10000)]
#[test]
fn test_should_use_crown_ibp_intermediates_unary_transformer_allowed_3775() {
    use crate::layers::{RmsNormLayer, SiLULayer};

    // Unary transformer surfaces: now ALLOWED.
    let unary_cases: Vec<(&str, Layer)> = vec![
        ("softmax", Layer::Softmax(SoftmaxLayer::new(-1))),
        (
            "causal_softmax",
            Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)),
        ),
        (
            "layernorm",
            Layer::LayerNorm(
                LayerNormLayer::new_default(2, 1e-5).expect("invariant: valid test eps"),
            ),
        ),
        (
            "rmsnorm",
            Layer::RmsNorm(RmsNormLayer::new_default(2, 1e-5).expect("invariant: valid test eps")),
        ),
        ("silu", Layer::SiLU(SiLULayer)),
        ("gelu", Layer::GELU(GELULayer::new(GeluApproximation::Tanh))),
    ];

    for (label, layer) in unary_cases {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(label, layer));
        g.set_output(label);
        assert!(
            g.should_use_crown_ibp_intermediates(),
            "{label}-only graph should NOW use CROWN-IBP intermediates (#3775)"
        );
    }

    // ReLU-only: still allowed.
    let mut simple = GraphNetwork::new();
    simple.add_node(GraphNode::from_input("r1", Layer::ReLU(ReLULayer)));
    simple.set_output("r1");
    assert!(
        simple.should_use_crown_ibp_intermediates(),
        "ReLU-only graph should use CROWN-IBP intermediates"
    );
}

/// GroupNorm, MatMul, and MulBinary remain blocked from CROWN-IBP
/// intermediates. These are the conservative surfaces kept out of the first
/// demand-driven packet (#3775).
#[ntest::timeout(10000)]
#[test]
fn test_should_use_crown_ibp_intermediates_binary_and_groupnorm_blocked_3775() {
    use crate::layers::binary_ops::{BilinearCrownLayer, MatMulLayer, MulBinaryLayer};

    // GroupNorm: still blocked.
    let mut gn_only = GraphNetwork::new();
    gn_only.add_node(GraphNode::from_input(
        "gn",
        Layer::GroupNorm(
            GroupNormLayer::new_default(2, 1, 1e-5).expect("invariant: valid test eps"),
        ),
    ));
    gn_only.set_output("gn");
    assert!(
        !gn_only.should_use_crown_ibp_intermediates(),
        "GroupNorm-only graph should NOT use CROWN-IBP intermediates (#3775)"
    );

    // MatMul: still blocked (binary relaxation).
    let mut matmul_graph = GraphNetwork::new();
    matmul_graph.add_node(GraphNode::from_input("r1", Layer::ReLU(ReLULayer)));
    matmul_graph.add_node(GraphNode::from_input("r2", Layer::ReLU(ReLULayer)));
    matmul_graph.add_node(GraphNode::new(
        "mm",
        Layer::MatMul(MatMulLayer::new(false, None)),
        vec!["r1".into(), "r2".into()],
    ));
    matmul_graph.set_output("mm");
    assert!(
        !matmul_graph.should_use_crown_ibp_intermediates(),
        "MatMul graph should NOT use CROWN-IBP intermediates (#3775)"
    );

    // BilinearCrown: still blocked (binary relaxation, variant of MatMul).
    let mut bilinear_graph = GraphNetwork::new();
    bilinear_graph.add_node(GraphNode::from_input("r1", Layer::ReLU(ReLULayer)));
    bilinear_graph.add_node(GraphNode::from_input("r2", Layer::ReLU(ReLULayer)));
    bilinear_graph.add_node(GraphNode::new(
        "bc",
        Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        vec!["r1".into(), "r2".into()],
    ));
    bilinear_graph.set_output("bc");
    assert!(
        !bilinear_graph.should_use_crown_ibp_intermediates(),
        "BilinearCrown graph should NOT use CROWN-IBP intermediates (#3775)"
    );

    // MulBinary: still blocked.
    let mut mul_graph = GraphNetwork::new();
    mul_graph.add_node(GraphNode::from_input("r1", Layer::ReLU(ReLULayer)));
    mul_graph.add_node(GraphNode::from_input("r2", Layer::ReLU(ReLULayer)));
    mul_graph.add_node(GraphNode::new(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        vec!["r1".into(), "r2".into()],
    ));
    mul_graph.set_output("mul");
    assert!(
        !mul_graph.should_use_crown_ibp_intermediates(),
        "MulBinary graph should NOT use CROWN-IBP intermediates (#3775)"
    );

    // build_graph() fixture still blocked (has GroupNorm).
    let graph = build_graph();
    assert!(
        !graph.should_use_crown_ibp_intermediates(),
        "graph with GroupNorm should NOT use CROWN-IBP intermediates"
    );
}

/// InstanceNorm1d is a CNN-style normalization layer compatible with CROWN-IBP
/// intermediate tightening. Unlike transformer-style norms (LayerNorm, RmsNorm),
/// InstanceNorm operates per-channel on CNN feature maps and the CROWN backward
/// path is fully supported via `propagate_linear_with_bounds`.  Part of #3596.
#[ntest::timeout(10000)]
#[test]
fn test_should_use_crown_ibp_intermediates_instance_norm() {
    // InstanceNorm1d-only graph should use CROWN-IBP intermediates.
    let mut in_only = GraphNetwork::new();
    in_only.add_node(GraphNode::from_input(
        "in1d",
        Layer::InstanceNorm1d(
            InstanceNorm1dLayer::new_default(2, 1e-5).expect("invariant: valid test eps"),
        ),
    ));
    in_only.set_output("in1d");
    assert!(
        in_only.should_use_crown_ibp_intermediates(),
        "InstanceNorm1d-only graph SHOULD use CROWN-IBP intermediates (CNN-style, #3596)"
    );

    // ReLU + InstanceNorm1d: still CNN-style, should use CROWN-IBP.
    let mut relu_in = GraphNetwork::new();
    relu_in.add_node(GraphNode::from_input("r1", Layer::ReLU(ReLULayer)));
    relu_in.add_node(GraphNode::new(
        "in1d",
        Layer::InstanceNorm1d(
            InstanceNorm1dLayer::new_default(2, 1e-5).expect("invariant: valid test eps"),
        ),
        vec!["r1".into()],
    ));
    relu_in.set_output("in1d");
    assert!(
        relu_in.should_use_crown_ibp_intermediates(),
        "ReLU+InstanceNorm1d graph SHOULD use CROWN-IBP intermediates (CNN-style, #3596)"
    );
}

/// Deep CNN DAGs should skip the O(N²) per-node CROWN-IBP collection and use
/// the reference-style IBP intermediates + final CROWN pass once they exceed
/// the small-graph threshold. Part of #3596.
#[ntest::timeout(10000)]
#[test]
fn test_should_collect_per_node_crown_ibp_intermediates_threshold_3596() {
    let shallow = build_relu_chain_graph(crate::network::core::graph::CROWN_IBP_PER_NODE_THRESHOLD);
    assert!(
        shallow.should_collect_per_node_crown_ibp_intermediates(),
        "CNN-style graphs at the threshold should still use per-node CROWN-IBP intermediates"
    );

    let deep =
        build_relu_chain_graph(crate::network::core::graph::CROWN_IBP_PER_NODE_THRESHOLD + 1);
    assert!(
        !deep.should_collect_per_node_crown_ibp_intermediates(),
        "deep CNN-style graphs above the threshold should use IBP intermediates for the final CROWN pass"
    );
}

/// Verify mode mutations do not affect `should_use_crown_ibp_intermediates`
/// (it checks layer presence, not mode settings). Part of #3391.
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_intermediates_invariant_under_mode_mutation() {
    let mut graph = build_graph();
    assert!(!graph.should_use_crown_ibp_intermediates());

    // Mutate all modes
    graph.set_layernorm_forward_mode(true);
    graph.set_layernorm_crown_mode(LayerNormCrownMode::Cut);
    graph.set_gelu_sound_mode(true);
    graph.set_softmax_sound_mode(true);

    // Presence check is mode-independent
    assert!(
        !graph.should_use_crown_ibp_intermediates(),
        "should_use_crown_ibp_intermediates must be mode-invariant"
    );
}
