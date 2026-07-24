// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use ndarray::{ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::network::core::graph::GraphNetwork;
use crate::types::CrownBackwardResult;

use super::{
    BlockCrownComparison, BlockSpec, BlockSpecEntry, BlockWiseCrownResult, LayerNormValidationStats,
};

impl GraphNetwork {
    /// Per-block CROWN verification: CROWN within blocks, IBP at boundaries.
    ///
    /// Block boundaries are discovered from `layer{N}` node-name conventions.
    /// For graphs with non-standard node names, use `propagate_crown_with_blocks`
    /// to supply explicit block boundaries via `BlockSpec`.
    ///
    /// For each detected transformer block:
    /// 1. Creates a fresh epsilon-ball as the block input (boundary reset)
    /// 2. Runs IBP forward through the block to get intermediate node bounds
    /// 3. Runs CROWN backward from block output to block input
    /// 4. Concretizes at the block input
    /// 5. Compares CROWN result with IBP result
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (used for shape information)
    /// * `epsilon` - Perturbation radius for each block's fresh input bounds
    pub fn propagate_crown_block_wise(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
    ) -> Result<BlockWiseCrownResult> {
        self.propagate_crown_block_wise_with_deadline(input, epsilon, None)
    }

    /// Deadline-aware variant of `propagate_crown_block_wise` (#4242).
    pub fn propagate_crown_block_wise_with_deadline(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        deadline: Option<Instant>,
    ) -> Result<BlockWiseCrownResult> {
        self.propagate_crown_block_wise_with_engine_and_deadline(input, epsilon, None, deadline)
    }

    /// Engine-aware variant of `propagate_crown_block_wise` (#3597).
    ///
    /// Builds a `BlockSpec` from the legacy `layer{N}` name-based discovery
    /// and delegates to `propagate_crown_with_blocks_and_engine`.
    pub fn propagate_crown_block_wise_with_engine(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BlockWiseCrownResult> {
        self.propagate_crown_block_wise_with_engine_and_deadline(input, epsilon, engine, None)
    }

    /// Deadline-aware engine variant of `propagate_crown_block_wise` (#4242).
    pub fn propagate_crown_block_wise_with_engine_and_deadline(
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

        // Build a BlockSpec from the legacy layer{N} name-based discovery,
        // preserving the same metadata the old path produced (#4024).
        let spec = BlockSpec {
            blocks: block_nodes
                .iter()
                .map(|(&block_idx, node_names)| BlockSpecEntry {
                    block_index: block_idx,
                    block_name: format!("layer{}", block_idx),
                    node_names: node_names.clone(),
                })
                .collect(),
        };

        self.propagate_crown_with_blocks_and_engine_and_deadline(
            input, epsilon, &spec, engine, deadline,
        )
    }

    /// Per-block CROWN verification with explicit consumer-supplied block boundaries.
    ///
    /// Same per-block CROWN/IBP math as `propagate_crown_block_wise`, but block
    /// boundaries and metadata come from the caller's `BlockSpec` instead of
    /// being parsed from `layer{N}` node names. This allows traced or imported
    /// graphs with arbitrary node names to use block-wise CROWN. Part of #4024.
    pub fn propagate_crown_with_blocks(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        block_spec: &BlockSpec,
    ) -> Result<BlockWiseCrownResult> {
        self.propagate_crown_with_blocks_with_deadline(input, epsilon, block_spec, None)
    }

    /// Deadline-aware variant of `propagate_crown_with_blocks` (#4242).
    pub fn propagate_crown_with_blocks_with_deadline(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        block_spec: &BlockSpec,
        deadline: Option<Instant>,
    ) -> Result<BlockWiseCrownResult> {
        self.propagate_crown_with_blocks_and_engine_and_deadline(
            input, epsilon, block_spec, None, deadline,
        )
    }

    /// Engine-aware variant of `propagate_crown_with_blocks` (#4024).
    pub fn propagate_crown_with_blocks_and_engine(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        block_spec: &BlockSpec,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BlockWiseCrownResult> {
        self.propagate_crown_with_blocks_and_engine_and_deadline(
            input, epsilon, block_spec, engine, None,
        )
    }

    /// Deadline-aware engine variant of `propagate_crown_with_blocks` (#4242).
    pub fn propagate_crown_with_blocks_and_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        block_spec: &BlockSpec,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BlockWiseCrownResult> {
        // Disable the L2/Cauchy–Schwarz lever for block-wise CROWN: the per-block
        // loop below runs `collect_block_ibp_bounds` (a lever-firing IBP forward
        // pass) for every block. Chokepoint for all block-wise CROWN entry points.
        // Sound; restored on drop. See `crate::l2_lever_gate`.
        let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
        let exec_order = self.exec_order()?;
        block_spec.validate(self, exec_order)?;

        let mut comparisons = Vec::with_capacity(block_spec.blocks.len());

        for entry in &block_spec.blocks {
            let block_input_shape = input.shape().to_vec();
            let block_input =
                BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&block_input_shape)), epsilon)?;
            let block_node_bounds =
                self.collect_block_ibp_bounds(&entry.node_names, &block_input)?;

            let last_node_name = entry.node_names.last().ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Empty block '{}' (index {})",
                    entry.block_name, entry.block_index
                ))
            })?;
            let ibp_output = block_node_bounds.get(last_node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "IBP bounds not found for block output node '{}'",
                    last_node_name
                ))
            })?;
            let ibp_max_width = ibp_output.max_width();

            let crown_result = self.crown_backward_within_block_with_engine(
                &entry.node_names,
                &block_node_bounds,
                &block_input,
                engine,
                None,
                deadline,
            );

            let (crown_max_width, crown_successful) = match crown_result {
                Ok((crown_bounds, _stats, provenance)) => {
                    // #4256: derive success from provenance, not from Ok(_).
                    // Fallback provenance means we returned IBP bounds, not CROWN.
                    (crown_bounds.max_width(), !provenance.is_fallback())
                }
                Err(e) => {
                    tracing::debug!(
                        "Per-block CROWN failed for block {}: {}, using IBP width",
                        entry.block_name,
                        e
                    );
                    (ibp_max_width, false)
                }
            };

            let crown_ibp_ratio = if ibp_max_width > f32::EPSILON {
                crown_max_width / ibp_max_width
            } else {
                1.0
            };

            comparisons.push(BlockCrownComparison {
                block_index: entry.block_index,
                block_name: entry.block_name.clone(),
                ibp_max_width,
                crown_max_width,
                crown_ibp_ratio,
                crown_successful,
                alpha_crown_max_width: None,
                alpha_crown_ibp_ratio: None,
            });
        }

        Ok(BlockWiseCrownResult {
            total_blocks: comparisons.len(),
            blocks: comparisons,
            block_epsilon: epsilon,
        })
    }

    /// Run CROWN backward through all nodes in the graph as a single block.
    ///
    /// Treats the entire graph as one transformer block. Useful when the graph
    /// already represents a single block (e.g., from Whisper `encoder_layer_graph`).
    ///
    /// Runs IBP forward to get intermediate bounds, then CROWN backward from
    /// output to input. Normalization and bilinear layers trigger partial fallback
    /// (concretized to IBP-quality), while linear and activation layers get full
    /// CROWN tightening.
    ///
    /// Part of #318.
    pub fn propagate_crown_within_graph(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_crown_within_graph_with_deadline(input, None)
    }

    /// Run single-block graph CROWN with fallback provenance.
    pub fn propagate_crown_within_graph_with_provenance(
        &self,
        input: &BoundedTensor,
    ) -> Result<CrownBackwardResult> {
        self.propagate_crown_within_graph_with_provenance_and_deadline(input, None)
    }

    /// Deadline-aware single-block graph CROWN with fallback provenance (#4256).
    pub fn propagate_crown_within_graph_with_provenance_and_deadline(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<CrownBackwardResult> {
        self.propagate_crown_within_graph_with_provenance_and_engine_and_deadline(
            input, None, deadline,
        )
    }

    /// Engine-aware single-block graph CROWN with fallback provenance (#4256).
    pub fn propagate_crown_within_graph_with_provenance_and_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<CrownBackwardResult> {
        self.propagate_crown_within_graph_with_provenance_and_engine_and_deadline(
            input, engine, None,
        )
    }

    /// Engine + deadline single-block graph CROWN with fallback provenance (#4256).
    pub fn propagate_crown_within_graph_with_provenance_and_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<CrownBackwardResult> {
        let (result, _stats) = self
            .propagate_crown_within_graph_with_provenance_and_stats_and_engine_and_deadline(
                input, engine, deadline,
            )?;
        Ok(result)
    }

    /// Deadline-aware variant of `propagate_crown_within_graph` (#4242).
    pub fn propagate_crown_within_graph_with_deadline(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_within_graph_with_engine_and_deadline(input, None, deadline)
    }

    /// Engine-aware variant of `propagate_crown_within_graph` (#3772).
    pub fn propagate_crown_within_graph_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_within_graph_with_engine_and_deadline(input, engine, None)
    }

    /// Deadline-aware engine variant of `propagate_crown_within_graph` (#4242).
    pub fn propagate_crown_within_graph_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_within_graph_with_provenance_and_engine_and_deadline(
            input, engine, deadline,
        )
        .map(|result| result.bounds)
    }

    /// Same as `propagate_crown_within_graph` but also returns row-collapse counts
    /// from each decomposed normalization site. Part of #318.
    pub fn propagate_crown_within_graph_with_stats(
        &self,
        input: &BoundedTensor,
    ) -> Result<(BoundedTensor, Vec<LayerNormValidationStats>)> {
        self.propagate_crown_within_graph_with_stats_and_deadline(input, None)
    }

    /// Deadline-aware variant of `propagate_crown_within_graph_with_stats` (#4242).
    pub fn propagate_crown_within_graph_with_stats_and_deadline(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<(BoundedTensor, Vec<LayerNormValidationStats>)> {
        self.propagate_crown_within_graph_with_stats_and_engine_and_deadline(input, None, deadline)
    }

    /// Engine-aware variant of `propagate_crown_within_graph_with_stats` (#3772).
    pub fn propagate_crown_within_graph_with_stats_and_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(BoundedTensor, Vec<LayerNormValidationStats>)> {
        self.propagate_crown_within_graph_with_stats_and_engine_and_deadline(input, engine, None)
    }

    /// Deadline-aware engine variant of `propagate_crown_within_graph_with_stats` (#4242).
    pub fn propagate_crown_within_graph_with_stats_and_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<(BoundedTensor, Vec<LayerNormValidationStats>)> {
        let (result, stats) = self
            .propagate_crown_within_graph_with_provenance_and_stats_and_engine_and_deadline(
                input, engine, deadline,
            )?;
        Ok((result.bounds, stats))
    }

    fn propagate_crown_within_graph_with_provenance_and_stats_and_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<(CrownBackwardResult, Vec<LayerNormValidationStats>)> {
        let exec_order = self.exec_order()?;
        if exec_order.is_empty() {
            return Err(NyError::InvalidSpec(
                "propagate_crown_within_graph: empty graph".to_string(),
            ));
        }

        let block_node_bounds = self.collect_block_ibp_bounds(exec_order, input)?;

        let (bounds, stats, provenance) = self.crown_backward_within_block_with_engine(
            exec_order,
            &block_node_bounds,
            input,
            engine,
            None,
            deadline,
        )?;
        Ok((CrownBackwardResult { bounds, provenance }, stats))
    }
}
