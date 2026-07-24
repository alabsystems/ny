// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use crate::batched_domain::{DomainMetadata, PickedDomains};
use crate::layers::{Layer, ReLULayer};
use crate::{GraphNetwork, GraphNode};

pub(super) fn make_simple_picked(
    input_lower: &[f32],
    input_upper: &[f32],
    layer_lower: &[f32],
    layer_upper: &[f32],
    layer_name: &str,
    lower_bound: f32,
    upper_bound: f32,
) -> PickedDomains {
    let input_len = input_lower.len();
    let layer_len = layer_lower.len();

    let mut layer_lowers = HashMap::new();
    let mut layer_uppers = HashMap::new();
    layer_lowers.insert(
        layer_name.to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, layer_len]), layer_lower.to_vec()).unwrap(),
    );
    layer_uppers.insert(
        layer_name.to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, layer_len]), layer_upper.to_vec()).unwrap(),
    );

    let metadata = vec![DomainMetadata::root(lower_bound, upper_bound).unwrap()];

    PickedDomains {
        batch_size: 1,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, input_len]), input_lower.to_vec()).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, input_len]), input_upper.to_vec()).unwrap(),
        global_lbs: vec![lower_bound],
        global_ubs: vec![upper_bound],
        metadata,
    }
}

pub(super) fn make_relu_graph(relu_name: &str) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(relu_name, Layer::ReLU(ReLULayer)));
    graph.set_output(relu_name);
    graph
}
