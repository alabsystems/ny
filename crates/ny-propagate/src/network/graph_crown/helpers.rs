// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::Layer;

use super::super::core::{GraphNetwork, GraphNode};

pub(super) fn is_softmax_decomposition_mul(graph: &GraphNetwork, node: &GraphNode) -> bool {
    if node.inputs.len() != 2 {
        return false;
    }
    let (input_a, input_b) = match node.require_binary_inputs() {
        Ok(inputs) => inputs,
        Err(_) => return false,
    };
    let input_a_node = graph.nodes.get(input_a);
    let input_b_node = graph.nodes.get(input_b);

    let (exp_name, recip_name) = match (input_a_node, input_b_node) {
        (Some(exp_node), Some(recip_node))
            if matches!(exp_node.layer, Layer::Exp(_))
                && matches!(recip_node.layer, Layer::Reciprocal(_)) =>
        {
            (input_a, input_b)
        }
        (Some(recip_node), Some(exp_node))
            if matches!(recip_node.layer, Layer::Reciprocal(_))
                && matches!(exp_node.layer, Layer::Exp(_)) =>
        {
            (input_b, input_a)
        }
        _ => return false,
    };

    let recip_node = match graph.nodes.get(recip_name) {
        Some(node) => node,
        None => return false,
    };
    if recip_node.inputs.len() != 1 {
        return false;
    }
    let reduce_name = match recip_node.require_unary_input() {
        Ok(input_name) => input_name,
        Err(_) => return false,
    };
    let reduce_node = match graph.nodes.get(reduce_name) {
        Some(node) => node,
        None => return false,
    };
    if !matches!(reduce_node.layer, Layer::ReduceSum(_)) {
        return false;
    }
    if reduce_node.inputs.len() != 1 {
        return false;
    }
    let exp_reduce_name = match reduce_node.require_unary_input() {
        Ok(input_name) => input_name,
        Err(_) => return false,
    };
    if exp_reduce_name != exp_name {
        return false;
    }
    let exp_reduce_node = match graph.nodes.get(exp_reduce_name) {
        Some(node) => node,
        None => return false,
    };
    if !matches!(exp_reduce_node.layer, Layer::Exp(_)) {
        return false;
    }

    true
}
