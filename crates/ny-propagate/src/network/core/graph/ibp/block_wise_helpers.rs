// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Block-wise IBP helper utilities.

use crate::types::{BlockBoundsInfo, BlockWiseResult};

use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::super::{GraphNetwork, GraphNode, NETWORK_INPUT};

impl GraphNetwork {
    /// Parse block index from node name (e.g., "layer0_attn_norm" -> Some(0)).
    pub(crate) fn parse_block_index(node_name: &str) -> Option<usize> {
        let after_layer = node_name.strip_prefix("layer")?;
        let digits_end = after_layer
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after_layer.len());
        if digits_end == 0 {
            return None;
        }
        let num_str = &after_layer[..digits_end];
        num_str.parse().ok()
    }

    pub(crate) fn collect_block_nodes(
        exec_order: &[String],
    ) -> std::collections::BTreeMap<usize, Vec<String>> {
        let mut block_nodes: std::collections::BTreeMap<usize, Vec<String>> =
            std::collections::BTreeMap::new();
        for node_name in exec_order {
            if let Some(block_idx) = Self::parse_block_index(node_name) {
                block_nodes
                    .entry(block_idx)
                    .or_default()
                    .push(node_name.clone());
            }
        }
        block_nodes
    }

    pub(crate) fn block_wise_fallback_result(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
    ) -> Result<BlockWiseResult> {
        let layer_result = self.propagate_ibp_detailed(input, epsilon)?;
        let input_width = epsilon * 2.0;
        let degraded = layer_result.degraded_at_node.is_some();
        let sensitivity = if input_width > 0.0 {
            layer_result.final_width / input_width
        } else {
            f32::INFINITY
        };
        Ok(BlockWiseResult {
            blocks: vec![BlockBoundsInfo {
                block_index: 0,
                block_name: "all_layers".to_string(),
                nodes: layer_result.nodes,
                input_width,
                output_width: layer_result.final_width,
                sensitivity,
                qk_matmul_width: None,
                swiglu_width: None,
                degraded,
            }],
            block_epsilon: epsilon,
            total_blocks: 1,
            max_sensitivity: sensitivity,
            degraded_blocks: if degraded { 1 } else { 0 },
        })
    }

    /// Bounds from cache or create fresh for block-wise verification.
    pub(crate) fn bounds_for_block(
        &self,
        input_name: &str,
        block_input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<BoundedTensor> {
        if input_name == NETWORK_INPUT {
            Ok(block_input.clone())
        } else if let Some(cached) = bounds_cache.get(input_name) {
            Ok(cached.clone())
        } else {
            // #2812 Finding 51: Missing cache entry for non-NETWORK_INPUT node.
            // This may be legitimate (input from outside this block) or a
            // dependency ordering error within the block. Log a warning so
            // callers can detect potential topology bugs.
            tracing::warn!(
                input_name = input_name,
                "get_bounds_for_block: cache miss for non-NETWORK_INPUT node; \
                 substituting block_input (may indicate dependency ordering error)"
            );
            Ok(block_input.clone())
        }
    }

    /// Try to apply zonotope tightening for Q@K^T in block-wise mode.
    pub(crate) fn try_attention_matmul_bounds_zonotope_block(
        &self,
        matmul_node: &GraphNode,
        block_input: &BoundedTensor,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<Option<BoundedTensor>> {
        // Delegate to the main zonotope function with fresh block input.
        self.try_attention_matmul_bounds_zonotope(matmul_node, block_input, bounds_cache)
    }
}
