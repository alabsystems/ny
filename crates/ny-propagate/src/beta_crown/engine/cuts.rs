// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use ny_tensor::BoundedTensor;

use crate::beta_crown::bab_cuts::{CuttingPlane, GraphCutPool, GraphCuttingPlane};
use crate::{GraphNetwork, Layer, NETWORK_INPUT};

use super::BetaCrownVerifier;

impl BetaCrownVerifier {
    /// Compute the total contribution of cuts to the lower bound.
    ///
    /// For each cut: contribution = lambda * (min_constraint_value - bias)
    /// where min_constraint_value is computed using ReLU indicator bounds.
    ///
    /// BICCOS cuts use ReLU activation indicators z_i ∈ {0,1}:
    /// - z_i = 0 if neuron is inactive (x_i < 0)
    /// - z_i = 1 if neuron is active (x_i >= 0)
    /// - z_i ∈ [0, 1] for unstable neurons
    pub(super) fn compute_cut_contribution(
        &self,
        cuts: &[&CuttingPlane],
        layer_bounds: &[Arc<BoundedTensor>],
    ) -> f32 {
        let mut total = 0.0f32;

        for cut in cuts {
            // Skip cuts with zero lambda (no contribution)
            if cut.lambda().abs() < 1e-10 {
                continue;
            }

            // Compute minimum value of the constraint: sum_i(coeff_i * z_i)
            // where z_i is the ReLU indicator (0 if inactive, 1 if active)
            let constraint_min: f32 = cut
                .terms
                .iter()
                .filter_map(|term| {
                    // Get pre-activation bounds for this layer
                    // term.layer_idx is the ReLU layer, pre-activation is from layer_idx - 1
                    if term.layer_idx == 0 || term.layer_idx > layer_bounds.len() {
                        return None;
                    }
                    let pre_bounds = &layer_bounds[term.layer_idx - 1];
                    let flat = pre_bounds.flatten();

                    if term.neuron_idx >= flat.len() {
                        return None;
                    }

                    let l = flat.lower()[[term.neuron_idx]];
                    let u = flat.upper()[[term.neuron_idx]];

                    // Determine ReLU indicator bounds [z_min, z_max]
                    let (z_min, z_max) = if l >= 0.0 {
                        // Stable active: z = 1
                        (1.0, 1.0)
                    } else if u <= 0.0 {
                        // Stable inactive: z = 0
                        (0.0, 0.0)
                    } else {
                        // Unstable: z ∈ [0, 1]
                        (0.0, 1.0)
                    };

                    // Worst-case (minimum) value of coeff * z
                    let value = if term.coefficient > 0.0 {
                        term.coefficient * z_min // Use lower bound of z for positive coeff
                    } else {
                        term.coefficient * z_max // Use upper bound of z for negative coeff
                    };
                    Some(value)
                })
                .sum();

            // Lagrangian contribution: lambda * (constraint_value - bias)
            // The cut constraint is: sum(coeff_i * z_i) <= bias
            // Lagrangian dual adds: -lambda * (sum(coeff_i * z_i) - bias) to lower bound
            // For minimizing violation: lambda * (bias - constraint_min) >= 0
            let contribution = cut.lambda() * (cut.bias() - constraint_min);
            total += contribution;
            cut.metadata.note_contribution(contribution);
        }

        total
    }

    /// Compute the total contribution of graph cuts to the lower bound.
    ///
    /// For each graph cut: contribution = lambda * (min_constraint_value - bias)
    /// where min_constraint_value is computed using ReLU indicator bounds from
    /// `node_bounds`. (The historical non-Arc wrapper was removed by #cone-delta
    /// increment 2: every caller's cache is `Arc`-shared now.)
    pub(super) fn compute_graph_cut_contribution_arc(
        &self,
        graph: &GraphNetwork,
        cuts: &[&GraphCuttingPlane],
        node_bounds: &std::collections::HashMap<String, Arc<BoundedTensor>>,
        input_bounds: &BoundedTensor,
    ) -> f32 {
        let mut total = 0.0f32;

        for cut in cuts {
            // Skip cuts with zero lambda (no contribution)
            if cut.lambda().abs() < 1e-10 {
                continue;
            }

            // Compute minimum value of the constraint: sum_i(coeff_i * z_i)
            let constraint_min: f32 = cut
                .terms
                .iter()
                .filter_map(|term| {
                    // Graph cut terms are keyed by ReLU node name, but the indicator variable
                    // is determined from the *pre-activation* bounds (the ReLU input node).
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

                    // Determine ReLU indicator bounds [z_min, z_max] from pre-activation bounds.
                    let (z_min, z_max) = if l >= 0.0 {
                        // Stable active: z = 1
                        (1.0, 1.0)
                    } else if u <= 0.0 {
                        // Stable inactive: z = 0
                        (0.0, 0.0)
                    } else {
                        // Unstable: z ∈ [0, 1]
                        (0.0, 1.0)
                    };

                    // Worst-case (minimum) value of coeff * z
                    let value = if term.coefficient > 0.0 {
                        term.coefficient * z_min
                    } else {
                        term.coefficient * z_max
                    };
                    Some(value)
                })
                .sum();

            // Lagrangian contribution: lambda * (bias - constraint_min)
            let contribution = cut.lambda() * (cut.bias() - constraint_min);
            total += contribution;
            cut.metadata.note_contribution(contribution);
        }

        total
    }

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
