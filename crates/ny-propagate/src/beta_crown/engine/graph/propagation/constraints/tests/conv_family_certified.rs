// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #graph-constrained-forward-sound: the per-domain constrained forward must
//! publish CERTIFIED conv-family node boxes.
//!
//! Before the fix `compute_constrained_forward_bounds_view_inner` routed only
//! Conv2d and Linear to certified arms and every other unary kind -- including
//! ConvTranspose2d, the cGAN generator's every node -- to the PLAIN
//! round-to-nearest forward, then intersected the result tighter-wins into the
//! inherited sound enclosure. Cancellation kernel `W=[2^24, -1, -2^24]` on the
//! point input `[1,1,1]`: plain f32 gives `2^24 + (-1) + (-2^24) = 0` (the -1 is
//! below half an ULP at 2^24 and is lost) while the true value is `-1`. A node
//! box `[0, 0]` that excludes `-1` classifies a downstream ReLU stable-active
//! and can mint a FALSE VERIFIED domain.

use ndarray::{arr1, ArrayD, IxDyn};

use crate::beta_crown::{GraphNeuronConstraint, GraphSplitHistory};
use crate::layers::{Conv1dLayer, ConvTranspose2dLayer};
use crate::{
    BetaCrownConfig, BetaCrownVerifier, BoundPropagation, GraphNetwork, GraphNode, Layer, ReLULayer,
};
use ny_tensor::BoundedTensor;

const TWO24: f32 = 16_777_216.0; // 2^24
const W: [f32; 3] = [TWO24, -1.0, -TWO24];
/// `2^24 - 1 - 2^24`, exact in f64.
const TRUE_LOWER: f64 = -1.0;

fn point_3ch(shape: &[usize]) -> BoundedTensor {
    let pt = ArrayD::from_shape_vec(IxDyn(shape), vec![1.0_f32, 1.0, 1.0]).unwrap();
    BoundedTensor::new(pt.clone(), pt).unwrap()
}

/// input -> relu0 -> <conv node> ; the split on `relu0` puts the conv node in
/// the recompute cone, so the constrained forward re-propagates it.
fn graph_with_cone_conv(conv: Layer) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new("conv", conv, vec!["relu0".to_string()]));
    graph.set_output("conv");
    graph
}

fn history_on_relu0() -> GraphSplitHistory {
    // The point input is 1 > 0 on every channel, so "active" is consistent
    // with the box; the split only serves to define the cone.
    GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu0".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    })
}

fn assert_conv_node_encloses(graph: &GraphNetwork, input: &BoundedTensor, label: &str) {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let history = history_on_relu0();
    // NOTE the second element: `compute_constrained_forward_bounds` returns
    // `(bounds_cache, constrained_INPUT)` -- the input box after the split
    // constraints, NOT a network output. The node box in the cache is the
    // object under test.
    let (cache, _constrained_input) = verifier
        .compute_constrained_forward_bounds(graph, input, &history, None, None)
        .expect("constrained forward");
    let node_box = cache.get("conv").expect("conv node box in the cache");
    let node_lower = node_box.lower().iter().copied().next().unwrap();
    // Sanity on the fixture: the PLAIN forward of the same layer really does
    // exclude the true value (this is what the old `_` arm published).
    let plain = graph
        .node("conv")
        .expect("conv node")
        .layer
        .propagate_ibp(input)
        .expect("plain forward");
    let plain_lower = plain.lower().iter().copied().next().unwrap();
    assert!(
        f64::from(plain_lower) > TRUE_LOWER,
        "{label}: the plain forward should EXCLUDE the true {TRUE_LOWER} under cancellation, \
         got {plain_lower}; the fixture no longer discriminates"
    );
    assert!(
        f64::from(node_lower) <= TRUE_LOWER + 1e-4,
        "{label}: the constrained forward's node box lower {node_lower} must enclose the true \
         {TRUE_LOWER}; a plain-forward value here is the #graph-constrained-forward-sound defect"
    );
    // The node box's UPPER endpoint must enclose too: the cancellation makes
    // the plain forward a point box at 0 while the true value is -1, so a
    // certified box has to straddle it.
    let node_upper = node_box.upper().iter().copied().next().unwrap();
    assert!(
        f64::from(node_upper) >= TRUE_LOWER - 1e-4,
        "{label}: the node box upper {node_upper} must sit above the true {TRUE_LOWER}"
    );
}

#[test]
fn constrained_forward_conv_transpose2d_box_encloses_true_value_under_cancellation() {
    // transpose kernel (in_c=3, out_c=1, kh=1, kw=1); input (in_c=3, H=1, W=1).
    let kernel = ArrayD::from_shape_vec(IxDyn(&[3, 1, 1, 1]), W.to_vec()).unwrap();
    let layer = ConvTranspose2dLayer::new(kernel, Some(arr1(&[0.0_f32])), (1, 1), (0, 0)).unwrap();
    let graph = graph_with_cone_conv(Layer::ConvTranspose2d(layer));
    assert_conv_node_encloses(&graph, &point_3ch(&[3, 1, 1]), "ConvTranspose2d");
}

#[test]
fn constrained_forward_conv1d_box_encloses_true_value_under_cancellation() {
    // kernel (out_c=1, in_c=3, kw=1); input (in_c=3, length=1).
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 3, 1]), W.to_vec()).unwrap();
    let layer = Conv1dLayer::new(kernel, Some(arr1(&[0.0_f32])), 1, 0).unwrap();
    let graph = graph_with_cone_conv(Layer::Conv1d(layer));
    assert_conv_node_encloses(&graph, &point_3ch(&[3, 1]), "Conv1d");
}
