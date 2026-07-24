// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential alpha-CROWN collection helpers for warm-startable graph alpha state.

use crate::bounds::{AlphaCrownConfig, AlphaState, GraphAlphaState};
use crate::layers::{BoundPropagation, Layer};
use crate::network::core::GraphNetwork;
use crate::network::graph_alpha::alpha_projection::graph_alpha_state_from_sequential;
use crate::network::graph_alpha::reference_bounds::GraphAlphaReferenceBounds;
use crate::NETWORK_INPUT;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;

use super::SequentialAlphaOptimizationContext;

impl GraphNetwork {
    pub(in crate::network::graph_alpha) fn maybe_collect_sequential_alpha_crown_bounds_with_engine(
        &self,
        exec_order: &[String],
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Option<Result<(HashMap<String, BoundedTensor>, GraphAlphaState)>> {
        let has_non_relu_alpha_nodes = exec_order.iter().any(|name| {
            self.nodes.get(name).is_some_and(|node| {
                matches!(
                    &node.layer,
                    Layer::Sigmoid(_) | Layer::Tanh(_) | Layer::Sqrt(_)
                )
            })
        });

        (self.is_sequential_graph(exec_order)
            && !has_non_relu_alpha_nodes
            && self.can_reuse_sequential_alpha_collection(exec_order))
        .then(|| self.collect_alpha_crown_bounds_sequential_with_engine(input, config, engine))
    }

    /// Match the public sequential alpha-CROWN route so warm-start collection
    /// only reuses the sequential optimizer on the same ReLU-only graph family.
    pub(in crate::network::graph_alpha) fn can_reuse_sequential_alpha_collection(
        &self,
        exec_order: &[String],
    ) -> bool {
        exec_order.iter().all(|node_name| {
            self.nodes.get(node_name).is_some_and(|node| {
                matches!(
                    &node.layer,
                    Layer::Linear(_)
                        | Layer::ReLU(_)
                        | Layer::Transpose(_)
                        | Layer::GELU(_)
                        | Layer::Tile(_)
                )
            })
        })
    }

    /// Reuse the sequential alpha-CROWN optimizer, then project the learned
    /// alpha state back into the graph-keyed format used by BaB warm starts.
    ///
    /// This keeps root alpha collection on the same gradient-method dispatch as
    /// `propagate_alpha_crown_with_config*` for eligible sequential ReLU graphs.
    pub(in crate::network::graph_alpha) fn collect_alpha_crown_bounds_sequential_with_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(HashMap<String, BoundedTensor>, GraphAlphaState)> {
        let exec_order = self.exec_order()?;
        let output_node = if self.output_node.is_empty() {
            exec_order
                .last()
                .cloned()
                .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            self.output_node.clone()
        };

        let activation_count = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .is_some_and(|node| node.layer.requires_pre_activation_bounds())
            })
            .count();
        let deep_override = config.fix_interm_bounds
            && activation_count >= 3
            && self.should_use_crown_ibp_intermediates();
        let use_crown_ibp = !config.fix_interm_bounds || deep_override;

        let mut initial_bounds = if use_crown_ibp {
            self.collect_crown_ibp_bounds_dag_with_deadline_and_engine(
                input,
                config.deadline,
                engine,
            )?
        } else {
            self.collect_node_bounds(input)?
        };
        initial_bounds.insert(NETWORK_INPUT.to_string(), input.clone());

        let relu_nodes: Vec<(String, usize)> = exec_order
            .iter()
            .enumerate()
            .filter(|(_, name)| {
                self.nodes
                    .get(*name)
                    .is_some_and(|node| matches!(&node.layer, Layer::ReLU(_)))
            })
            .map(|(idx, name)| (name.clone(), idx))
            .collect();

        let pre_activation_bounds: Vec<BoundedTensor> = relu_nodes
            .iter()
            .map(|(name, _)| {
                self.relu_preactivation_bounds(
                    name,
                    input,
                    &initial_bounds,
                    "sequential-alpha-collection-init",
                )
                .cloned()
            })
            .collect::<Result<Vec<_>>>()?;

        let mut alpha_state = AlphaState::from_preactivation_bounds(
            &pre_activation_bounds,
            &(0..relu_nodes.len()).collect::<Vec<_>>(),
        )?;
        let relu_name_to_idx: HashMap<String, usize> = relu_nodes
            .iter()
            .enumerate()
            .map(|(idx, (name, _))| (name.clone(), idx))
            .collect();

        let mut reference_bounds = GraphAlphaReferenceBounds::new(
            initial_bounds,
            self.graph_alpha_reference_bound_targets()?,
        )?;
        let output_dim = reference_bounds
            .current()
            .get(&output_node)
            .ok_or_else(|| NyError::InvalidSpec(format!("Output node {output_node} not found")))?
            .len();

        let best_output = if alpha_state.num_unstable() > 0 {
            Some(self.optimize_sequential_alpha_crown_with_reference_bounds(
                SequentialAlphaOptimizationContext {
                    input,
                    config,
                    engine,
                    reference_bounds: &mut reference_bounds,
                    alpha_state: &mut alpha_state,
                    exec_order,
                    output_dim,
                    relu_name_to_idx: &relu_name_to_idx,
                    invprop_enabled: false,
                    carry_forward_reference_bounds: true,
                },
            )?)
        } else {
            None
        };

        let graph_alpha_state = graph_alpha_state_from_sequential(&alpha_state, &relu_name_to_idx)
            .ok_or_else(|| {
                NyError::InternalError(
                    "Sequential alpha collection could not project alpha state into graph order"
                        .to_string(),
                )
            })?;

        let mut final_bounds = if best_output.is_some() && !config.fix_interm_bounds {
            let baseline_bounds = reference_bounds.current();
            let mut alpha_bounds = self.collect_crown_bounds_with_alpha(
                input,
                baseline_bounds,
                &graph_alpha_state,
                engine,
                config.deadline,
            )?;

            for (name, baseline_bound) in baseline_bounds {
                match alpha_bounds.get(name) {
                    Some(alpha_bound) if alpha_bound.shape() == baseline_bound.shape() => {
                        let (tightened, _disjoint) = baseline_bound
                            .intersection_per_element(alpha_bound)
                            .unwrap_or_else(|| (baseline_bound.clone(), 0));
                        alpha_bounds.insert(name.clone(), tightened);
                    }
                    Some(_) => {
                        alpha_bounds.insert(name.clone(), baseline_bound.clone());
                    }
                    None => {
                        alpha_bounds.insert(name.clone(), baseline_bound.clone());
                    }
                }
            }

            alpha_bounds
        } else {
            reference_bounds.current().clone()
        };

        if let Some(best_output) = best_output {
            merge_output_bound(&mut final_bounds, &output_node, &best_output);
        }

        final_bounds.remove(NETWORK_INPUT);
        Ok((final_bounds, graph_alpha_state))
    }
}

fn merge_output_bound(
    node_bounds: &mut HashMap<String, BoundedTensor>,
    output_node: &str,
    best_output: &BoundedTensor,
) {
    let merged = match node_bounds.get(output_node) {
        Some(existing) if existing.shape() == best_output.shape() => existing
            .intersection_per_element(best_output)
            .map(|(tightened, _disjoint)| tightened)
            .unwrap_or_else(|| existing.clone()),
        _ => best_output.clone(),
    };
    node_bounds.insert(output_node.to_string(), merged);
}
