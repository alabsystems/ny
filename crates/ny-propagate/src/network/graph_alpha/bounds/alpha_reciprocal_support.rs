// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::warn;

use crate::bounds::alpha_reciprocal::ReciprocalGradients;
use crate::bounds::{AdamParams, GraphAlphaState, Optimizer};
use crate::layers::Layer;
use crate::network::core::{GraphNetwork, NETWORK_INPUT};

pub(super) fn initialize_reciprocal_alpha_nodes(
    graph: &GraphNetwork,
    exec_order: &[String],
    input: &BoundedTensor,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: &mut GraphAlphaState,
) -> Result<Vec<String>> {
    let reciprocal_nodes: Vec<String> = exec_order
        .iter()
        .filter(|name| {
            graph
                .nodes
                .get(*name)
                .map(|node| matches!(node.layer, Layer::Reciprocal(_)))
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    for node_name in &reciprocal_nodes {
        let node = graph.nodes.get(node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!("Reciprocal node {} not found", node_name))
        })?;
        let input_name = node.require_unary_input()?;
        let pre_activation = if input_name == NETWORK_INPUT {
            input
        } else {
            ibp_bounds.get(input_name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Pre-activation bounds for reciprocal node '{}' not found",
                    node_name
                ))
            })?
        };
        // Reciprocal requires strictly positive or strictly negative domain (no zero crossing).
        let lower = pre_activation.lower();
        let all_positive = lower.iter().all(|v| v.is_finite() && *v > 0.0);
        let all_negative = pre_activation
            .upper()
            .iter()
            .all(|v| v.is_finite() && *v < 0.0);
        if all_positive || all_negative {
            alpha_state.add_reciprocal_node(node_name, pre_activation)?;
        }
    }
    Ok(reciprocal_nodes)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update_reciprocal_alpha_gradients(
    reciprocal_nodes: &[String],
    gradients: &BTreeMap<String, ReciprocalGradients>,
    alpha_state: &mut GraphAlphaState,
    optimizer: Optimizer,
    adam_params: &AdamParams,
    learning_rate: f32,
    momentum: f32,
    iter: usize,
) {
    for node_name in reciprocal_nodes {
        let Some(grad) = gradients.get(node_name) else {
            continue;
        };
        if grad.any_non_finite() {
            warn!(
                iter = iter,
                node = node_name.as_str(),
                "α-CROWN: non-finite reciprocal gradient for {node_name}, skipping update"
            );
            continue;
        }
        let neg_grad = grad.negate();
        if let Some(alpha) = alpha_state.reciprocal_alpha_mut(node_name) {
            match optimizer {
                Optimizer::Adam => alpha.update_adam(&neg_grad, adam_params),
                Optimizer::Sgd => alpha.update_sgd(&neg_grad, learning_rate, momentum),
            }
        }
    }
}
