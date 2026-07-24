// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::warn;

use crate::bounds::{AdamParams, GraphAlphaState, Optimizer, SqrtGradients};
use crate::layers::Layer;
use crate::network::core::{GraphNetwork, NETWORK_INPUT};

pub(super) fn initialize_sqrt_alpha_nodes(
    graph: &GraphNetwork,
    exec_order: &[String],
    input: &BoundedTensor,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: &mut GraphAlphaState,
) -> Result<Vec<String>> {
    let sqrt_nodes: Vec<String> = exec_order
        .iter()
        .filter(|name| {
            graph
                .nodes
                .get(*name)
                .map(|node| matches!(node.layer, Layer::Sqrt(_)))
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    for node_name in &sqrt_nodes {
        let node = graph
            .nodes
            .get(node_name)
            .ok_or_else(|| NyError::InvalidSpec(format!("Sqrt node {} not found", node_name)))?;
        let input_name = node.require_unary_input()?;
        let pre_activation = if input_name == NETWORK_INPUT {
            input
        } else {
            ibp_bounds.get(input_name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Pre-activation bounds for sqrt node '{}' not found",
                    node_name
                ))
            })?
        };
        if pre_activation
            .lower()
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        {
            alpha_state.add_sqrt_node(node_name, pre_activation)?;
        }
    }
    Ok(sqrt_nodes)
}

#[allow(clippy::too_many_arguments)] // The DAG alpha loop owns the node list, gradient map, optimizer choice, and step parameters separately; threading them here keeps alpha.rs below the size limit without changing control flow.
pub(super) fn update_sqrt_alpha_gradients(
    sqrt_nodes: &[String],
    gradients: &BTreeMap<String, SqrtGradients>,
    alpha_state: &mut GraphAlphaState,
    optimizer: Optimizer,
    adam_params: &AdamParams,
    learning_rate: f32,
    momentum: f32,
    iter: usize,
) {
    for node_name in sqrt_nodes {
        let Some(grad) = gradients.get(node_name) else {
            continue;
        };
        if grad.any_non_finite() {
            warn!(
                iter = iter,
                node = node_name.as_str(),
                "α-CROWN: non-finite sqrt gradient for {node_name}, skipping update (#3773)"
            );
            continue;
        }
        let neg_grad = grad.negate();
        if let Some(alpha) = alpha_state.sqrt_alpha_mut(node_name) {
            match optimizer {
                Optimizer::Adam => alpha.update_adam(&neg_grad, adam_params),
                Optimizer::Sgd => alpha.update_sgd(&neg_grad, learning_rate, momentum),
            }
        }
    }
}
