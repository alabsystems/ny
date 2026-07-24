// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha-CROWN optimization for per-block GELU slope parameters.
//!
//! Uses per-neuron coordinate descent to optimize GELU alpha parameters
//! within each transformer block for tighter lower bounds.
//!
//! Part of #3221 Phase 4, #3447.

use std::collections::HashMap;
use std::time::Instant;

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info};

use crate::layers::Layer;

use crate::network::core::graph::GraphNetwork;

use super::{BlockAlphaState, BlockCrownComparison, BlockWiseCrownResult};

impl GraphNetwork {
    /// Per-block alpha-CROWN: optimize GELU slopes within each block.
    ///
    /// Uses per-neuron coordinate descent: for each GELU neuron, searches over
    /// alpha candidates while keeping other neurons fixed. This is a standard
    /// alpha-CROWN optimization strategy that captures per-neuron optimal tangent
    /// points, unlike uniform alpha which forces all neurons to use the same value.
    ///
    /// The objective is `sum_of_widths` (sum of upper - lower across all output
    /// dimensions), which is differentiable and decomposable across neurons.
    ///
    /// Part of #3221 Phase 4.
    pub fn propagate_alpha_crown_block_wise(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
    ) -> Result<BlockWiseCrownResult> {
        self.propagate_alpha_crown_block_wise_internal(input, epsilon, None, None)
    }

    /// Engine-aware variant of `propagate_alpha_crown_block_wise` (#3597).
    pub fn propagate_alpha_crown_block_wise_with_engine(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BlockWiseCrownResult> {
        self.propagate_alpha_crown_block_wise_internal(input, epsilon, engine, None)
    }

    fn propagate_alpha_crown_block_wise_internal(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BlockWiseCrownResult> {
        let exec_order = self.exec_order()?;
        let block_nodes = Self::collect_block_nodes(exec_order);

        if block_nodes.is_empty() {
            return Ok(BlockWiseCrownResult {
                blocks: vec![],
                total_blocks: 0,
                block_epsilon: epsilon,
            });
        }

        let mut comparisons = Vec::with_capacity(block_nodes.len());

        for (&block_idx, nodes_in_block) in &block_nodes {
            let block_name = format!("layer{}", block_idx);
            let block_input_shape = input.shape().to_vec();
            let block_input =
                BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&block_input_shape)), epsilon)?;

            let block_node_bounds = self.collect_block_ibp_bounds(nodes_in_block, &block_input)?;

            let last_node_name = nodes_in_block
                .last()
                .ok_or_else(|| NyError::InvalidSpec("Empty block".to_string()))?;
            let ibp_output = block_node_bounds.get(last_node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "IBP bounds not found for block output node '{}'",
                    last_node_name
                ))
            })?;
            let ibp_max_width = ibp_output.max_width();

            let crown_result = self.crown_backward_within_block_with_engine(
                nodes_in_block,
                &block_node_bounds,
                &block_input,
                engine,
                None,
                deadline,
            );
            let (crown_max_width, crown_successful) = match crown_result {
                Ok((bounds, _stats, provenance)) => {
                    // #4256: derive success from provenance, not from Ok(_).
                    (bounds.max_width(), !provenance.is_fallback())
                }
                Err(e) => {
                    debug!("Per-block CROWN failed for {}: {}", block_name, e);
                    (ibp_max_width, false)
                }
            };

            let gelu_nodes = self.collect_gelu_nodes_in_block(nodes_in_block, &block_node_bounds);

            let alpha_crown_max_width = if !gelu_nodes.is_empty() && crown_successful {
                self.optimize_block_alphas_coordinate_descent(
                    nodes_in_block,
                    &block_node_bounds,
                    &block_input,
                    &gelu_nodes,
                    &block_name,
                    engine,
                )?
            } else {
                None
            };

            let crown_ibp_ratio = if ibp_max_width > f32::EPSILON {
                crown_max_width / ibp_max_width
            } else {
                1.0
            };
            let alpha_crown_ibp_ratio = alpha_crown_max_width.map(|w| {
                if ibp_max_width > f32::EPSILON {
                    w / ibp_max_width
                } else {
                    1.0
                }
            });

            comparisons.push(BlockCrownComparison {
                block_index: block_idx,
                block_name,
                ibp_max_width,
                crown_max_width,
                crown_ibp_ratio,
                crown_successful,
                alpha_crown_max_width,
                alpha_crown_ibp_ratio,
            });
        }

        Ok(BlockWiseCrownResult {
            total_blocks: comparisons.len(),
            blocks: comparisons,
            block_epsilon: epsilon,
        })
    }

    #[cfg(test)]
    pub(crate) fn propagate_alpha_crown_block_wise_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BlockWiseCrownResult> {
        self.propagate_alpha_crown_block_wise_internal(input, epsilon, engine, deadline)
    }

    /// Per-neuron coordinate descent optimization of GELU alpha parameters.
    ///
    /// Uses sum-of-widths as the optimization objective (standard alpha-CROWN
    /// approach — more sensitive to per-neuron improvements than max-width).
    /// Reports max-width for final result since that's the verification metric.
    ///
    /// For each GELU neuron, tries alpha candidates on a 10-point grid [0.0, 0.1,
    /// ..., 1.0] while keeping other neurons at their current best. Runs multiple
    /// sweeps until convergence (no improvement) or max iterations reached.
    ///
    /// Returns `Some(best_max_width)` — the max-width achieved at the best
    /// sum-of-widths alpha configuration.
    fn optimize_block_alphas_coordinate_descent(
        &self,
        block_nodes: &[String],
        block_node_bounds: &HashMap<String, BoundedTensor>,
        block_input: &BoundedTensor,
        gelu_nodes: &[(String, usize)],
        block_name: &str,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Option<f32>> {
        let alpha_candidates: &[f32] = &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let max_sweeps = 3;

        let mut alpha_state = BlockAlphaState::new_midpoint(gelu_nodes);

        // Compute baseline sum-of-widths for the midpoint (alpha=0.5) configuration.
        let (baseline_result, _, _provenance) = self.crown_backward_within_block_with_engine(
            block_nodes,
            block_node_bounds,
            block_input,
            engine,
            Some(&alpha_state),
            None,
        )?;
        let baseline_width = baseline_result.max_width();
        let mut best_sum_width: f64 = baseline_result.width().iter().map(|&w| w as f64).sum();

        for sweep in 0..max_sweeps {
            let mut any_improved = false;

            for (gelu_name, dim) in gelu_nodes {
                let alphas = alpha_state
                    .gelu_alphas
                    .get(gelu_name)
                    .cloned()
                    .unwrap_or_else(|| Array1::from_elem(*dim, 0.5_f32));

                let mut neuron_best_alphas = alphas.clone();
                for neuron_idx in 0..*dim {
                    let current_alpha = neuron_best_alphas[neuron_idx];
                    let mut best_neuron_sum = best_sum_width;
                    let mut best_alpha = current_alpha;

                    for &candidate in alpha_candidates {
                        if (candidate - current_alpha).abs() < 1e-8 {
                            continue;
                        }
                        let mut trial_alphas = neuron_best_alphas.clone();
                        trial_alphas[neuron_idx] = candidate;

                        let mut trial_state = alpha_state.clone();
                        trial_state
                            .gelu_alphas
                            .insert(gelu_name.clone(), trial_alphas);

                        if let Ok((bounds, _, _)) = self.crown_backward_within_block_with_engine(
                            block_nodes,
                            block_node_bounds,
                            block_input,
                            engine,
                            Some(&trial_state),
                            None,
                        ) {
                            let mw = bounds.max_width();
                            if !mw.is_finite() {
                                continue;
                            }
                            let sw: f64 = bounds.width().iter().map(|&w| w as f64).sum();
                            if sw < best_neuron_sum {
                                best_neuron_sum = sw;
                                best_alpha = candidate;
                            }
                        }
                    }

                    if (best_alpha - current_alpha).abs() > 1e-8 {
                        neuron_best_alphas[neuron_idx] = best_alpha;
                        if best_neuron_sum < best_sum_width {
                            best_sum_width = best_neuron_sum;
                            any_improved = true;
                        }
                    }
                }

                alpha_state
                    .gelu_alphas
                    .insert(gelu_name.clone(), neuron_best_alphas);
            }

            debug!(
                "Block {}: alpha sweep {}/{} sum_width={:.6} improved={}",
                block_name,
                sweep + 1,
                max_sweeps,
                best_sum_width,
                any_improved,
            );

            if !any_improved {
                break;
            }
        }

        // Evaluate the final optimized alpha configuration for max_width.
        let (final_result, _, _provenance) = self.crown_backward_within_block_with_engine(
            block_nodes,
            block_node_bounds,
            block_input,
            engine,
            Some(&alpha_state),
            None,
        )?;
        let best_max_width = final_result.max_width();

        if best_max_width < baseline_width {
            info!(
                "Block {}: alpha-CROWN {:.6} -> {:.6} ({:.1}% tighter)",
                block_name,
                baseline_width,
                best_max_width,
                (1.0 - best_max_width / baseline_width) * 100.0
            );
        }
        Ok(Some(best_max_width))
    }

    /// Collect GELU node names and their neuron dimensions within a block.
    fn collect_gelu_nodes_in_block(
        &self,
        block_nodes: &[String],
        block_node_bounds: &HashMap<String, BoundedTensor>,
    ) -> Vec<(String, usize)> {
        block_nodes
            .iter()
            .filter_map(|name| {
                let node = self.nodes.get(name)?;
                if !matches!(&node.layer, Layer::GELU(_)) {
                    return None;
                }
                let dim = block_node_bounds
                    .get(name)
                    .map(|b| *b.shape().last().unwrap_or(&0))
                    .unwrap_or(0);
                (dim > 0).then(|| (name.clone(), dim))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use ndarray::{Array2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    use crate::layers::linear::LinearLayer;
    use crate::tests::crown::helpers::CountingGemmEngine;
    use crate::{GELULayer, GraphNetwork, GraphNode, Layer, NETWORK_INPUT};

    fn build_single_block_gelu_graph(hidden: usize, expansion: usize) -> GraphNetwork {
        let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
        let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();

        let mut graph = GraphNetwork::new();
        let up = LinearLayer::new(
            Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
                let phase = (i * 17 + j * 31) as f32;
                scale1 * phase.sin() * 0.15
            }),
            None,
        )
        .expect("up projection should construct");
        graph.add_node(GraphNode::new(
            "layer0_ffn_up",
            Layer::Linear(up),
            vec![NETWORK_INPUT.to_string()],
        ));
        graph.add_node(GraphNode::new(
            "layer0_ffn_act",
            Layer::GELU(GELULayer::default()),
            vec!["layer0_ffn_up".to_string()],
        ));
        let down = LinearLayer::new(
            Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
                let phase = (i * 23 + j * 37) as f32;
                scale2 * phase.cos() * 0.15
            }),
            None,
        )
        .expect("down projection should construct");
        graph.add_node(GraphNode::new(
            "layer0_ffn_down",
            Layer::Linear(down),
            vec!["layer0_ffn_act".to_string()],
        ));
        graph.set_output("layer0_ffn_down");
        graph
    }

    // This test takes the process-wide environment lock. In the full parallel
    // suite it may wait behind long-running budget oracles before its tiny body
    // starts, so the timeout must include shared-host lock contention.
    #[ntest::timeout(300000)]
    #[test]
    fn test_alpha_block_wise_deadline_fallback_skips_optimization_4256() {
        crate::tests::with_crown_dense_budget_mb("2048", || {
            let graph = build_single_block_gelu_graph(4, 2);
            let epsilon = 0.05_f32;
            let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[4])), epsilon).unwrap();
            let expired = Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("system uptime exceeds 1s");
            let engine = CountingGemmEngine::new();

            let result = graph
                .propagate_alpha_crown_block_wise_with_engine_and_deadline(
                    &input,
                    epsilon,
                    Some(&engine),
                    Some(expired),
                )
                .expect("expired deadline should fall back, not error");

            assert_eq!(result.total_blocks, 1, "expected one block");
            let block = &result.blocks[0];
            assert!(
                !block.crown_successful,
                "expired deadline fallback should mark the fixed baseline as unsuccessful"
            );
            assert!(
                block.alpha_crown_max_width.is_none(),
                "alpha optimization should be skipped when the fixed baseline already fell back"
            );
            assert!(
                block.alpha_crown_ibp_ratio.is_none(),
                "alpha ratio should be absent when optimization is skipped on fallback"
            );
            assert_eq!(
                engine.gemm_calls(),
                0,
                "expired deadline fallback should short-circuit before any alpha optimization GEMM work"
            );
        });
    }
}
