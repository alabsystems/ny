// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha state initialization for DAG α-CROWN propagation.
//!
//! Handles CROWN-IBP forward pass, node discovery, alpha state initialization
//! for all nonlinear node types (ReLU, S-shaped, Sqrt, BilinearCrown, MulBinary),
//! and INVPROP state setup.

use crate::bounds::{AlphaCrownConfig, GraphAlphaState};
use crate::invprop::InvpropState;
use crate::layers::Layer;
use crate::NETWORK_INPUT;

use ndarray::{Array2, Array4};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use tracing::{debug, info};

use super::super::runtime_state::DagAlphaRuntimeState;
use crate::network::core::GraphNetwork;

/// Result of DAG alpha state initialization.
///
/// Either an early return (no optimizable nodes) or the full initialized state
/// needed by the optimization loop.
#[allow(clippy::large_enum_variant)] // EarlyReturn (224B) vs Ready (8B boxed) — boxing both is unnecessary overhead
pub(super) enum DagAlphaInitResult {
    /// No optimizable activations found — return these bounds directly. A
    /// typed cGAN route also carries the already-paid-for collection artifact
    /// so collection callers do not interpret the handled exit as "not
    /// delegated" and start the typed transaction again.
    EarlyReturn {
        bounds: BoundedTensor,
        collection_artifact: Option<super::DagAlphaCollectionArtifact>,
    },
    /// Full initialized state for the optimization loop.
    Ready(Box<DagAlphaInitState>),
}

/// Initialized state for the DAG alpha optimization loop.
///
/// All fields are `pub(super)` to allow destructuring in `mod.rs`.
pub(super) struct DagAlphaInitState {
    pub(super) node_bounds: HashMap<String, BoundedTensor>,
    /// Which collector produced `node_bounds` (#dedup-root-collections Fix B):
    /// lets the optimization loop reuse the map for the pre-loop initial CROWN
    /// bound instead of re-collecting the identical map, but only when it is
    /// the same grade (or tighter) as the internal Step-1 collection would be.
    pub(super) node_bounds_source: crate::network::graph_alpha::bounds::AlphaReferenceBoundsSource,
    pub(super) exec_order: Vec<String>,
    pub(super) output_dim: usize,
    pub(super) input_dim: usize,
    pub(super) relu_nodes: Vec<(String, usize)>,
    pub(super) runtime: DagAlphaRuntimeState,
    pub(super) bilinear_alphas: HashMap<String, Array4<f32>>,
    pub(super) bilinear_adam_m: HashMap<String, Array4<f32>>,
    pub(super) bilinear_adam_v: HashMap<String, Array4<f32>>,
    pub(super) mul_binary_alphas: HashMap<String, Array2<f32>>,
    pub(super) mul_binary_adam_m: HashMap<String, Array2<f32>>,
    pub(super) mul_binary_adam_v: HashMap<String, Array2<f32>>,
    pub(super) has_bilinear: bool,
    pub(super) has_mul_binary: bool,
    pub(super) has_s_shaped: bool,
    pub(super) has_sqrt: bool,
    pub(super) has_reciprocal: bool,
    pub(super) invprop_enabled: bool,
}

impl GraphNetwork {
    /// Initialize all alpha state for DAG α-CROWN optimization.
    ///
    /// Performs:
    /// 1. CROWN-IBP forward to collect intermediate bounds
    /// 2. Topological sort and output dimension determination
    /// 3. Nonlinear node discovery (ReLU, S-shaped, Sqrt)
    /// 4. BilinearCrown and MulBinary alpha initialization
    /// 5. INVPROP state setup
    /// 6. DagAlphaRuntimeState construction
    ///
    /// Returns `EarlyReturn` if no optimizable activations exist, otherwise `Ready`
    /// with the full state for the optimization loop.
    pub(super) fn init_dag_alpha_state(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        precomputed_reference: Option<crate::network::graph_alpha::PrecomputedAlphaReferenceBounds>,
    ) -> Result<DagAlphaInitResult> {
        // Step 1: Collect the shared alpha reference bounds. With
        // `fix_interm_bounds=true`, DAG warmup now follows the same IBP contract
        // as the root bootstrap instead of forcing CROWN-IBP (#4404).
        let exec_order = self.exec_order()?;
        let (mut node_bounds, node_bounds_source) = match precomputed_reference {
            Some(reference) => (reference.bounds, reference.source),
            None => self.collect_alpha_reference_bounds_with_engine_and_source(
                input, config, engine, exec_order,
            )?,
        };
        node_bounds.insert(NETWORK_INPUT.to_string(), input.clone());

        // Determine output dimension
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;
        let output_dim = output_bounds.len();
        let input_dim = input.len();

        // Step 2: Identify optimizable nonlinear nodes and their pre-activation bounds.
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
        let s_shaped_nodes: Vec<String> = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::Sigmoid(_) | Layer::Tanh(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        let sqrt_nodes: Vec<String> = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::Sqrt(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        let gamma_only_invprop = config.invprop.optimize_gammas
            && super::invprop_output_seed_treatment_eligible(config, output_dim)
            && !super::uses_patches_output_seed(self, output_bounds);

        if relu_nodes.is_empty()
            && s_shaped_nodes.is_empty()
            && sqrt_nodes.is_empty()
            && !gamma_only_invprop
        {
            // No optimizable activation nodes — optimize BilinearCrown alphas if present,
            // else fall back to the existing CROWN/Batched alpha dispatch.
            // Pure attention graphs (no ReLUs) benefit from McCormick face selection
            // optimization via the batched alpha-CROWN path.
            // #3588: thread the caller's engine into the batched alpha optimizer
            // instead of dropping it.
            debug!(
                "DAG α-CROWN: No optimizable activation nodes, trying BilinearCrown alpha optimization"
            );
            let has_bilinear = self
                .nodes
                .values()
                .any(|node| matches!(node.layer, Layer::BilinearCrown(_)));
            if node_bounds_source.is_typed_cgan() && !has_bilinear {
                // The typed transaction already published a certified output
                // enclosure. With no alpha-bearing activation or bilinear
                // state there is nothing for the fallback optimizer to change;
                // carrying this handled artifact prevents the outer collector
                // from entering the typed transaction a second time.
                let bounds = output_bounds.clone();
                let alpha_state = GraphAlphaState::new();
                return Ok(DagAlphaInitResult::EarlyReturn {
                    bounds: bounds.clone(),
                    collection_artifact: Some(super::DagAlphaCollectionArtifact {
                        output_bounds: bounds,
                        alpha_state,
                        reference_bounds: node_bounds,
                        reference_bounds_source: node_bounds_source,
                        completed_iterations: 0,
                        optimizer_updates_completed: 0,
                    }),
                });
            }
            return Ok(DagAlphaInitResult::EarlyReturn {
                bounds: self.propagate_alpha_crown_batched(input, config, engine)?,
                collection_artifact: None,
            });
        }

        let mut graph_alpha_state = GraphAlphaState::new();
        for (relu_name, _) in &relu_nodes {
            let pre_activation =
                self.relu_preactivation_bounds(relu_name, input, &node_bounds, "dag-alpha-init")?;
            graph_alpha_state.add_relu_node(relu_name, pre_activation, !config.full_conv_alpha)?;
        }
        for node_name in &s_shaped_nodes {
            let node = self.nodes.get(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("S-shaped node {} not found", node_name))
            })?;
            let input_name = node.require_unary_input()?;
            let pre_activation = if input_name == NETWORK_INPUT {
                input
            } else {
                node_bounds.get(input_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        input_name
                    ))
                })?
            };
            match &node.layer {
                Layer::Sigmoid(_) => {
                    graph_alpha_state.add_sigmoid_node(node_name, pre_activation)?
                }
                Layer::Tanh(_) => graph_alpha_state.add_tanh_node(node_name, pre_activation)?,
                _ => {}
            }
        }
        let has_s_shaped = graph_alpha_state.monotone_alpha_names().next().is_some();
        for node_name in &sqrt_nodes {
            let node = self.nodes.get(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("Sqrt node {} not found", node_name))
            })?;
            let input_name = node.require_unary_input()?;
            let pre_activation = if input_name == NETWORK_INPUT {
                input
            } else {
                node_bounds.get(input_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        input_name
                    ))
                })?
            };
            if pre_activation
                .lower()
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
            {
                graph_alpha_state.add_sqrt_node(node_name, pre_activation)?;
            }
        }
        let has_sqrt = graph_alpha_state.sqrt_alpha_names().next().is_some();
        let reciprocal_nodes: Vec<String> = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::Reciprocal(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        for node_name in &reciprocal_nodes {
            let node = self.nodes.get(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("Reciprocal node {} not found", node_name))
            })?;
            let input_name = node.require_unary_input()?;
            let pre_activation = if input_name == NETWORK_INPUT {
                input
            } else {
                node_bounds.get(input_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        input_name
                    ))
                })?
            };
            let all_positive = pre_activation
                .lower()
                .iter()
                .all(|v| v.is_finite() && *v > 0.0);
            let all_negative = pre_activation
                .upper()
                .iter()
                .all(|v| v.is_finite() && *v < 0.0);
            if all_positive || all_negative {
                graph_alpha_state.add_reciprocal_node(node_name, pre_activation)?;
            }
        }
        let has_reciprocal = graph_alpha_state.reciprocal_alpha_names().next().is_some();

        // Step 2b: Collect BilinearCrown nodes and initialize alpha parameters (#3287).
        // This enables joint ReLU + bilinear alpha optimization in graphs with both
        // activation nodes and attention matmul operations (e.g., full transformers).
        // Reference: designs/2026-03-04-286-attention-bilinear-alternative.md Approach B.
        let mut bilinear_alphas: HashMap<String, Array4<f32>> = HashMap::new();
        let mut bilinear_adam_m: HashMap<String, Array4<f32>> = HashMap::new();
        let mut bilinear_adam_v: HashMap<String, Array4<f32>> = HashMap::new();
        for (name, node) in &self.nodes {
            if let Layer::BilinearCrown(bilinear) = &node.layer {
                if let Ok((input_a_name, input_b_name)) = node.require_binary_inputs() {
                    let input_a_bounds = node_bounds.get(input_a_name);
                    let input_b_bounds = node_bounds.get(input_b_name);
                    if let (Some(a_bounds), Some(b_bounds)) = (input_a_bounds, input_b_bounds) {
                        let (m, n, k) = bilinear.alpha_shape(a_bounds.shape(), b_bounds.shape())?;
                        // Initialize to ones (r=1.0), matching auto_LiRPA convention
                        // (starts from L1/U1 planes). Shape: [4, m, n, k].
                        bilinear_alphas.insert(name.clone(), Array4::ones((4, m, n, k)));
                        bilinear_adam_m.insert(name.clone(), Array4::zeros((4, m, n, k)));
                        bilinear_adam_v.insert(name.clone(), Array4::zeros((4, m, n, k)));
                    }
                }
            }
        }
        let has_bilinear = !bilinear_alphas.is_empty();
        if has_bilinear {
            debug!(
                "DAG α-CROWN: {} BilinearCrown nodes registered for alpha optimization",
                bilinear_alphas.len()
            );
        }

        // Step 2c: Collect MulBinary nodes and initialize alpha parameters (#3439 Phase 3).
        // Element-wise mul z = x*y: shape [2, n] (r_l, r_u per element).
        // Reference: designs/2026-03-06-mulbinary-crown-backward-dispatch.md Phase 3.
        let mut mul_binary_alphas: HashMap<String, Array2<f32>> = HashMap::new();
        let mut mul_binary_adam_m: HashMap<String, Array2<f32>> = HashMap::new();
        let mut mul_binary_adam_v: HashMap<String, Array2<f32>> = HashMap::new();
        for (name, node) in &self.nodes {
            if let Layer::MulBinary(_) = &node.layer {
                if let Some(bounds) = node_bounds.get(name) {
                    let n = bounds.lower().len();
                    // Initialize to 0.5 (Middle mode default).
                    mul_binary_alphas.insert(name.clone(), Array2::from_elem((2, n), 0.5));
                    mul_binary_adam_m.insert(name.clone(), Array2::zeros((2, n)));
                    mul_binary_adam_v.insert(name.clone(), Array2::zeros((2, n)));
                }
            }
        }
        let has_mul_binary = !mul_binary_alphas.is_empty();
        if has_mul_binary {
            debug!(
                "DAG α-CROWN: {} MulBinary nodes registered for alpha optimization",
                mul_binary_alphas.len()
            );
        }

        // Initialize INVPROP state if enabled and constraints are provided
        let invprop_enabled = config.invprop.enabled && config.output_constraints.is_some();
        let mut invprop_state: Option<InvpropState> = None;
        if config.invprop.enabled {
            if let Some(ref constraints) = config.output_constraints {
                let mut state = InvpropState::new(constraints.clone(), 1);
                let num_constraints = constraints.num_constraints();

                // Output-seed duals (the shipped, output-node-only assume-violation
                // channel). Keyed by the OUTPUT NODE name: apply_invprop_constraints
                // fires the augment on that node's incoming identity seed, where
                // folding raw C is dimensionally valid. neuron dim = output_dim.
                let output_node_name: String = if self.output_node.is_empty() {
                    exec_order.last().cloned().unwrap_or_default()
                } else {
                    self.output_node.clone()
                };
                if !output_node_name.is_empty() {
                    let seed_gammas = crate::invprop::LayerGammas::new(
                        num_constraints,
                        constraints.output_dim(),
                        config.invprop.share_gammas,
                    );
                    state.add_layer_gammas(output_node_name.clone(), seed_gammas);
                }

                // Per-layer intermediate-bound gammas: research channel only. Default
                // output-node-only leaves these unallocated (per-node augment no-ops).
                if config.invprop.per_layer_gammas {
                    for node_name in exec_order {
                        if node_name == &output_node_name {
                            continue; // already the seed
                        }
                        if let Some(node) = self.nodes.get(node_name) {
                            let layer_type = format!("Bound{}", node.layer.layer_type());
                            if config.invprop.should_apply_to(node_name, &layer_type) {
                                if let Some(bounds) = node_bounds.get(node_name) {
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
                    "DAG α-CROWN: INVPROP enabled, {} constraints, {} gamma group(s) \
                     (output-seed assume-violation dual)",
                    constraints.num_constraints(),
                    state.layer_gammas.len()
                );
                invprop_state = Some(state);
                crate::execution_telemetry::record_invprop_alpha_initialization();
            } else {
                tracing::warn!(
                    "DAG α-CROWN: INVPROP enabled in config but no output_constraints provided"
                );
            }
        }

        let runtime = DagAlphaRuntimeState::new(
            graph_alpha_state,
            invprop_state,
            relu_nodes.iter().map(|(name, _)| name.clone()).collect(),
        );

        if runtime.graph().num_unstable() == 0
            && !has_s_shaped
            && !has_sqrt
            && !has_reciprocal
            && !gamma_only_invprop
        {
            debug!("DAG α-CROWN: No optimizable activation state, using CROWN");
            if node_bounds_source.is_typed_cgan() && !has_bilinear && !has_mul_binary {
                let bounds = self
                    .propagate_crown_with_engine_and_deadline_and_node_bounds(
                        input,
                        engine,
                        config.deadline,
                        Some(&node_bounds),
                    )?
                    .bounds;
                let alpha_state = runtime.into_graph_alpha_state();
                return Ok(DagAlphaInitResult::EarlyReturn {
                    bounds: bounds.clone(),
                    collection_artifact: Some(super::DagAlphaCollectionArtifact {
                        output_bounds: bounds,
                        alpha_state,
                        reference_bounds: node_bounds,
                        reference_bounds_source: node_bounds_source,
                        completed_iterations: 0,
                        optimizer_updates_completed: 0,
                    }),
                });
            }
            return Ok(DagAlphaInitResult::EarlyReturn {
                bounds: self
                    .propagate_crown_with_engine_and_deadline(input, engine, config.deadline)?
                    .bounds,
                collection_artifact: None,
            });
        }

        Ok(DagAlphaInitResult::Ready(Box::new(DagAlphaInitState {
            node_bounds,
            node_bounds_source,
            exec_order: exec_order.to_vec(),
            output_dim,
            input_dim,
            relu_nodes,
            runtime,
            bilinear_alphas,
            bilinear_adam_m,
            bilinear_adam_v,
            mul_binary_alphas,
            mul_binary_adam_m,
            mul_binary_adam_v,
            has_bilinear,
            has_mul_binary,
            has_s_shaped,
            has_sqrt,
            has_reciprocal,
            invprop_enabled,
        })))
    }
}
