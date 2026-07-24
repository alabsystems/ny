// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Main α-CROWN orchestration: graph validation, IBP collection, and dispatch.
//!
//! Contains `propagate_alpha_crown_with_config_and_engine_impl` — the core entry
//! point that validates the graph structure, collects intermediate bounds, initializes
//! alpha state, and dispatches to the optimization loop.

use crate::bounds::{AlphaCrownConfig, AlphaState};
use crate::layers::{BoundPropagation, Layer};
use crate::network::core::GraphNetwork;
use crate::network::graph_alpha::reference_bounds::GraphAlphaReferenceBounds;
use crate::NETWORK_INPUT;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info};

use super::{SequentialAlphaOptimizationContext, SequentialAlphaOptimizationResult};

#[cfg(test)]
use super::{
    SEQUENTIAL_REFERENCE_REFRESH_ATTEMPTS, SEQUENTIAL_REFERENCE_TIGHTENED_TARGETS_TOTAL,
    SEQUENTIAL_ROOT_COLLECTION_EPISODES,
};

impl GraphNetwork {
    pub(super) fn propagate_alpha_crown_with_config_and_engine_impl(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        carry_forward_reference_bounds: bool,
    ) -> Result<SequentialAlphaOptimizationResult> {
        // Disable the L2/Cauchy–Schwarz lever for the entire alpha-CROWN scope.
        // This is the chokepoint for both sequential and DAG graph alpha-CROWN
        // (the DAG branch below delegates from here), so the many CROWN-internal
        // IBP forward passes (reference bounds, per-block IBP, intermediate
        // recomputation) skip the per-pass center allocation and O(out·in)
        // Cauchy–Schwarz nominal. Sound: the lever only tightens. Restored on
        // drop (panic-safe). See `crate::l2_lever_gate`.
        let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
        if self.nodes.is_empty() {
            return Ok(SequentialAlphaOptimizationResult::from_bounds(
                input.clone(),
            ));
        }

        // Get execution order
        let exec_order = self.exec_order()?;

        // Check if graph is sequential
        let is_sequential = self.is_sequential_graph(exec_order);
        if !is_sequential {
            // Use DAG α-CROWN for non-sequential graphs
            debug!("GraphNetwork α-CROWN: non-sequential graph, using DAG α-CROWN");
            return self
                .propagate_dag_alpha_crown_with_config_and_engine(input, config, engine)
                .map(SequentialAlphaOptimizationResult::from_bounds);
        }

        // Step 1: Routing (#dedup-root-collections Fix A). Every routing
        // decision below depends ONLY on layer types (cheap scans over
        // exec_order), never on collected bounds — so it runs BEFORE the
        // expensive intermediate-bound collection in Step 2. Previously the
        // deep-sequential CROWN-IBP collection ran first and was dropped
        // unread on every arm that routes away (measured ~73 s of dead work
        // per root episode on vggnet16_2022 spec1). Arm order is preserved
        // exactly: (a) no-ReLU + Sigmoid/Tanh/Sqrt → DAG; (b) no-ReLU →
        // fixed-slope CROWN; (c) per-layer DAG routing scan. Arms that
        // consume the collection (num_unstable==0 fixed-slope fallback) stay
        // below the collection.
        //
        // Identify ReLU nodes (pre-activation bounds are fetched in Step 2,
        // after collection, for graphs that do not route away).
        let relu_nodes: Vec<(String, usize)> = exec_order
            .iter()
            .enumerate()
            .filter(|(_, name)| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::ReLU(_)))
                    .unwrap_or(false)
            })
            .map(|(idx, name)| (name.clone(), idx))
            .collect();

        // Check for non-ReLU activation families that only the DAG alpha path
        // currently optimizes (Sigmoid, Tanh, Sqrt). Without this, a pure
        // sequential model in one of those families falls back to fixed-slope
        // CROWN with no alpha optimization.
        let has_non_relu_alpha_nodes = exec_order.iter().any(|name| {
            self.nodes
                .get(name)
                .map(|n| matches!(n.layer, Layer::Sigmoid(_) | Layer::Tanh(_) | Layer::Sqrt(_)))
                .unwrap_or(false)
        });

        if relu_nodes.is_empty() && has_non_relu_alpha_nodes {
            debug!(
                "GraphNetwork α-CROWN: no ReLU but has Sigmoid/Tanh/Sqrt, using DAG α-CROWN for non-ReLU alpha optimization (#3619, #3773)"
            );
            return self
                .propagate_dag_alpha_crown_with_config_and_engine(input, config, engine)
                .map(SequentialAlphaOptimizationResult::from_bounds);
        }

        if relu_nodes.is_empty() {
            // No optimizable alpha state — use fixed-slope CROWN.
            return self
                .propagate_crown_with_engine_and_deadline(input, engine, config.deadline)
                .map(|result| SequentialAlphaOptimizationResult::from_bounds(result.bounds));
        }

        // Check for operations that need DAG α-CROWN path (has better layer support)
        for node_name in exec_order {
            if let Some(node) = self.nodes.get(node_name) {
                match &node.layer {
                    // Ops that need DAG α-CROWN: Conv2d, MaxPool2d, BatchNorm, Concat
                    // need specialized graph-level handling (patches mode, binary splits).
                    // Normalization and softmax ops are handled by DAG backward dispatch
                    // via propagate_crown_backward trait — previously these fell back to
                    // plain CROWN with no alpha optimization, which was unnecessarily
                    // conservative. The DAG path supports these ops and preserves alpha
                    // optimization for any co-located ReLU nodes.
                    // MatMul and Add are binary ops that make is_sequential_graph() return
                    // false, so they never reach this match arm — included for completeness.
                    Layer::Conv2d(_)
                    | Layer::ConvTranspose2d(_)
                    | Layer::MaxPool2d(_)
                    | Layer::BatchNorm(_)
                    | Layer::Concat(_)
                    | Layer::Softmax(_)
                    | Layer::LayerNorm(_)
                    | Layer::RmsNorm(_)
                    | Layer::InstanceNorm1d(_)
                    | Layer::AdaIN1d(_)
                    | Layer::GroupNorm(_)
                    | Layer::MatMul(_)
                    | Layer::Add(_) => {
                        debug!(
                            "GraphNetwork α-CROWN: {} detected, using DAG α-CROWN (better layer support)",
                            node.layer.layer_type()
                        );
                        return self
                            .propagate_dag_alpha_crown_with_config_and_engine(input, config, engine)
                            .map(SequentialAlphaOptimizationResult::from_bounds);
                    }
                    // Explicitly supported sequential ops: no routing needed.
                    Layer::Linear(_)
                    | Layer::ReLU(_)
                    | Layer::Transpose(_)
                    | Layer::GELU(_)
                    | Layer::Tile(_) => {}
                    // Unknown/unrecognized layer: fall back to DAG α-CROWN (#2235).
                    // The backward pass catch-all can handle arbitrary layers via
                    // propagate_crown_backward, but DAG α-CROWN has better layer
                    // support than sequential for non-standard layer types.
                    _ => {
                        debug!(
                            "GraphNetwork α-CROWN: unrecognized op {} in sequential path, using DAG α-CROWN",
                            node.layer.layer_type()
                        );
                        return self
                            .propagate_dag_alpha_crown_with_config_and_engine(input, config, engine)
                            .map(SequentialAlphaOptimizationResult::from_bounds);
                    }
                }
            }
        }

        // Step 2: Collect bounds at each node for ReLU relaxation.
        // With fix_interm_bounds=true (default), use cheap IBP bounds (O(N)).
        // With fix_interm_bounds=false, use expensive CROWN-IBP (O(N^2)) for
        // tighter pre-activation bounds. This matches collect_alpha_crown_bounds_dag
        // in bounds/alpha.rs which also respects this flag. #3218
        //
        // Deep-sequential auto-override (#3628): when fix_interm_bounds=true but
        // the graph has 3+ non-linear activation layers, IBP intermediate bounds
        // compound relaxation error at each layer. After 3+ activations the CROWN
        // backward pass produces bounds wider than forward IBP on every element,
        // causing the IBP intersection to collapse alpha-CROWN to IBP quality.
        // Upgrade to CROWN-IBP intermediates when the graph passes the existing
        // safety heuristic (no transformer-style blocklisted ops).
        const DEEP_SEQUENTIAL_ACTIVATION_THRESHOLD: usize = 3;

        let activation_count = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .map(|n| n.layer.requires_pre_activation_bounds())
                    .unwrap_or(false)
            })
            .count();

        let deep_override = config.fix_interm_bounds
            && activation_count >= DEEP_SEQUENTIAL_ACTIVATION_THRESHOLD
            && self.should_use_crown_ibp_intermediates();

        if deep_override {
            debug!(
                "GraphNetwork α-CROWN: deep sequential override (#3628): \
                 {} activations >= threshold {}, upgrading to CROWN-IBP intermediates",
                activation_count, DEEP_SEQUENTIAL_ACTIVATION_THRESHOLD,
            );
        }

        let use_crown_ibp = !config.fix_interm_bounds || deep_override;
        #[cfg(test)]
        SEQUENTIAL_ROOT_COLLECTION_EPISODES.with(|slot| slot.set(slot.get() + 1));
        let mut node_bounds = if use_crown_ibp {
            // #3795: thread deadline into CROWN-IBP intermediate collection so
            // expensive Conv2d backward passes can bail early during initial bounds.
            self.collect_crown_ibp_bounds_dag_with_deadline_and_engine(
                input,
                config.deadline,
                engine,
            )?
        } else {
            self.collect_node_bounds(input)?
        };
        node_bounds.insert(NETWORK_INPUT.to_string(), input.clone());

        let mut reference_bounds = GraphAlphaReferenceBounds::new(
            node_bounds,
            self.graph_alpha_reference_bound_targets()?,
        )?;

        // Determine output dimension
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        let output_bounds = reference_bounds
            .current()
            .get(output_node_name)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!("Output node {} not found", output_node_name))
            })?;
        let output_dim = output_bounds.len();

        // Get pre-activation bounds for each ReLU node
        let pre_activation_bounds: Vec<BoundedTensor> = relu_nodes
            .iter()
            .map(|(name, _)| {
                self.relu_preactivation_bounds(
                    name,
                    input,
                    reference_bounds.current(),
                    "sequential-alpha-init",
                )
                .cloned()
            })
            .collect::<Result<Vec<_>>>()?;

        // Initialize alpha state
        let mut alpha_state = AlphaState::from_preactivation_bounds(
            &pre_activation_bounds,
            &(0..relu_nodes.len()).collect::<Vec<_>>(),
        )?;

        // Initialize INVPROP state if enabled and constraints are provided
        let invprop_enabled = config.invprop.enabled && config.output_constraints.is_some();
        if config.invprop.enabled {
            if let Some(ref constraints) = config.output_constraints {
                alpha_state.init_invprop_state(constraints.clone(), 1)?;

                // Allocate per-layer gammas for nodes matching apply_output_constraints_to
                if let Some(ref mut state) = alpha_state.invprop_state {
                    let num_constraints = constraints.num_constraints();
                    for node_name in exec_order {
                        if let Some(node) = self.nodes.get(node_name) {
                            let layer_type = format!("Bound{}", node.layer.layer_type());
                            if config.invprop.should_apply_to(node_name, &layer_type) {
                                if let Some(bounds) =
                                    reference_bounds.current().get(node_name.as_str())
                                {
                                    let num_neurons = bounds.len();
                                    let gammas = crate::invprop::LayerGammas::new(
                                        num_constraints,
                                        num_neurons,
                                        config.invprop.share_gammas,
                                    );
                                    state.add_layer_gammas(node_name.clone(), gammas);
                                }
                            }
                        }
                    }
                    if config.invprop.should_apply_to_input()
                        && state.layer_gammas(NETWORK_INPUT).is_none()
                    {
                        let num_neurons = input.len();
                        let gammas = crate::invprop::LayerGammas::new(
                            num_constraints,
                            num_neurons,
                            config.invprop.share_gammas,
                        );
                        state.add_layer_gammas(NETWORK_INPUT.to_string(), gammas);
                    }
                }

                info!(
                    "GraphNetwork α-CROWN: INVPROP enabled with {} constraints, {} layers with gammas",
                    constraints.num_constraints(),
                    alpha_state.invprop_state.as_ref().map(|s| s.layer_gammas.len()).unwrap_or(0)
                );
            } else {
                tracing::warn!(
                    "GraphNetwork α-CROWN: INVPROP enabled in config but no output_constraints provided"
                );
            }
        }

        let num_unstable = alpha_state.num_unstable();
        if num_unstable == 0 {
            debug!("GraphNetwork α-CROWN: No unstable neurons, using CROWN");
            return self
                .propagate_crown_with_engine_and_deadline(input, engine, config.deadline)
                .map(|result| SequentialAlphaOptimizationResult::from_bounds(result.bounds));
        }

        debug!(
            "GraphNetwork α-CROWN: Starting optimization with {} unstable neurons across {} ReLU nodes{}",
            num_unstable,
            relu_nodes.len(),
            if invprop_enabled { " (INVPROP enabled)" } else { "" }
        );

        // Map from node name to ReLU index
        let relu_name_to_idx: std::collections::HashMap<String, usize> = relu_nodes
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i))
            .collect();

        let bounds = self.optimize_sequential_alpha_crown_with_reference_bounds(
            SequentialAlphaOptimizationContext {
                input,
                config,
                engine,
                reference_bounds: &mut reference_bounds,
                alpha_state: &mut alpha_state,
                exec_order,
                output_dim,
                relu_name_to_idx: &relu_name_to_idx,
                invprop_enabled,
                carry_forward_reference_bounds,
            },
        )?;

        #[cfg(test)]
        let reference_refresh_attempts = if carry_forward_reference_bounds {
            SEQUENTIAL_REFERENCE_REFRESH_ATTEMPTS.with(std::cell::Cell::get)
        } else {
            0
        };
        #[cfg(not(test))]
        let reference_refresh_attempts = 0usize;

        #[cfg(test)]
        let reference_tightened_targets_total = if carry_forward_reference_bounds {
            SEQUENTIAL_REFERENCE_TIGHTENED_TARGETS_TOTAL.with(std::cell::Cell::get)
        } else {
            0
        };
        #[cfg(not(test))]
        let reference_tightened_targets_total = 0usize;

        Ok(SequentialAlphaOptimizationResult::with_reference_bounds(
            bounds,
            reference_bounds.current().clone(),
            reference_bounds.targets().to_vec(),
            reference_refresh_attempts,
            reference_tightened_targets_total,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::SEQUENTIAL_ROOT_COLLECTION_EPISODES;
    use crate::bounds::AlphaCrownConfig;
    use crate::layers::{Conv2dLayer, Layer, LinearLayer, ReLULayer};
    use crate::network::{GraphNetwork, GraphNode};
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    fn reset_collection_episode_counter() {
        SEQUENTIAL_ROOT_COLLECTION_EPISODES.with(|slot| slot.set(0));
    }

    fn collection_episodes() -> usize {
        SEQUENTIAL_ROOT_COLLECTION_EPISODES.with(std::cell::Cell::get)
    }

    /// #dedup-root-collections Fix A: a sequential graph containing Conv2d
    /// routes to the DAG α-CROWN delegate; the routing scan must run BEFORE
    /// the orchestration's intermediate-bound collection, so no sequential
    /// collection episode may start (previously one full — for deep nets
    /// CROWN-IBP — collection was performed and then dropped unread).
    #[ntest::timeout(60000)]
    #[test]
    fn conv2d_graph_routes_to_dag_without_sequential_collection_episode() {
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3, 3]),
            vec![1.0, 0.0, -1.0, 0.5, 0.0, -0.5, 0.25, 0.0, -0.25],
        )
        .unwrap();
        let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["conv".to_string()],
        ));
        graph.set_output("relu");

        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 3, 3]), -0.1_f32),
            ArrayD::from_elem(IxDyn(&[1, 3, 3]), 0.1_f32),
        )
        .unwrap();

        let config = AlphaCrownConfig {
            iterations: 1,
            ..Default::default()
        };

        reset_collection_episode_counter();
        let bounds = graph
            .propagate_alpha_crown_with_config(&input, &config)
            .expect("conv graph alpha-CROWN should succeed via DAG delegate");
        assert!(bounds.lower().iter().all(|v| v.is_finite()));
        assert!(bounds.upper().iter().all(|v| v.is_finite()));
        assert_eq!(
            collection_episodes(),
            0,
            "Conv2d graph must reach the DAG delegate without starting a \
             sequential root collection episode"
        );
    }

    /// Control for the counter: a pure Linear/ReLU sequential graph stays on
    /// the sequential path and must start exactly ONE collection episode.
    #[ntest::timeout(60000)]
    #[test]
    fn pure_sequential_relu_graph_starts_exactly_one_collection_episode() {
        let mut graph = GraphNetwork::new();
        let lin1 = LinearLayer::new(
            arr2(&[[1.0_f32, -0.5], [0.25, 0.75]]),
            Some(arr1(&[0.1_f32, -0.2])),
        )
        .unwrap();
        let lin2 = LinearLayer::new(arr2(&[[0.5_f32, -1.0]]), Some(arr1(&[0.0_f32]))).unwrap();
        graph.add_node(GraphNode::from_input("lin1", Layer::Linear(lin1)));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "lin2",
            Layer::Linear(lin2),
            vec!["relu1".to_string()],
        ));
        graph.set_output("lin2");

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -0.5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5_f32, 1.0]).unwrap(),
        )
        .unwrap();

        let config = AlphaCrownConfig {
            iterations: 1,
            ..Default::default()
        };

        reset_collection_episode_counter();
        graph
            .propagate_alpha_crown_with_config(&input, &config)
            .expect("sequential graph alpha-CROWN should succeed");
        assert_eq!(
            collection_episodes(),
            1,
            "pure sequential graph must collect intermediate bounds exactly once"
        );
    }
}
