// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use ny_tensor::BoundedTensor;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::{GraphNetwork, Layer, NETWORK_INPUT};

use super::BetaCrownVerifier;

impl BetaCrownVerifier {
    /// Compute gradients for graph cut lambda parameters.
    ///
    /// For each cut: d(lb)/d(λ_c) = constraint_min_c - bias_c
    pub(super) fn compute_graph_cut_gradients(
        &self,
        graph: &GraphNetwork,
        cut_pool: &mut GraphCutPool,
        node_bounds: &std::collections::HashMap<String, Arc<BoundedTensor>>,
        input_bounds: &BoundedTensor,
    ) {
        for cut in cut_pool.cuts_mut() {
            // Skip cuts with zero lambda (won't contribute to gradient)
            if cut.lambda().abs() < 1e-10 && cut.lambda_grad().abs() < 1e-10 {
                // Still compute gradient for initialization
            }

            // Compute minimum value of the constraint
            let constraint_min: f32 = cut
                .terms
                .iter()
                .filter_map(|term| {
                    let relu_node = graph.nodes.get(&term.node_name)?;
                    if !matches!(relu_node.layer, Layer::ReLU(_)) {
                        return None;
                    }
                    let pre_name = relu_node
                        .inputs
                        .first()
                        .map(|s| s.as_str())
                        // #2098: Return None for nodes with empty inputs.
                        .or_else(|| {
                            tracing::warn!(node = %term.node_name, "ReLU node has empty inputs");
                            None
                        })?;
                    let bounds: &BoundedTensor = if pre_name == NETWORK_INPUT {
                        input_bounds
                    } else {
                        node_bounds.get(pre_name)?.as_ref()
                    };
                    let flat = bounds.flatten();

                    if term.neuron_idx >= flat.len() {
                        return None;
                    }

                    let l = flat.lower()[[term.neuron_idx]];
                    let u = flat.upper()[[term.neuron_idx]];

                    let (z_min, z_max) = if l >= 0.0 {
                        (1.0, 1.0)
                    } else if u <= 0.0 {
                        (0.0, 0.0)
                    } else {
                        (0.0, 1.0)
                    };

                    let value = if term.coefficient > 0.0 {
                        term.coefficient * z_min
                    } else {
                        term.coefficient * z_max
                    };
                    Some(value)
                })
                .sum();

            // Gradient: d(lb)/d(λ) = bias - constraint_min (for maximization)
            cut.set_lambda_grad(cut.bias() - constraint_min);
        }
    }
}
