// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Block-wise IBP propagation and checkpoint helpers.

use crate::bounds::nan_propagating_max;
use crate::layers::Layer;
use crate::types::{
    BlockBoundsInfo, BlockProgress, BlockWiseResult, NodeBoundsInfo, VerificationCheckpoint,
};

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::super::{GraphNetwork, NETWORK_INPUT};
use super::dispatch::{
    check_nan_firewall, dispatch_ibp_resolved, resolve_node_inputs, ResolvedInputs,
};

impl GraphNetwork {
    /// Propagate bounds through the graph block-wise with zonotope reset per block.
    ///
    /// This method processes transformer blocks independently, resetting bounds
    /// at each block boundary. This prevents bound explosion from propagating
    /// through the entire network and allows zonotope tightening to be effective
    /// for each block's Q@K^T attention.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (used for shape information)
    /// * `epsilon` - Perturbation epsilon to use for each block's fresh input
    ///
    /// # Block Detection
    /// Blocks are detected by node name prefixes (e.g., "layer0_", "layer1_").
    /// Each block runs from attn_norm through add2 (the second residual add).
    ///
    /// # Returns
    /// A `BlockWiseResult` containing per-block sensitivity analysis.
    pub fn propagate_ibp_block_wise(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
    ) -> Result<BlockWiseResult> {
        self.propagate_ibp_block_wise_with_progress(input, epsilon, None::<fn(BlockProgress)>)
    }

    /// Block-wise IBP propagation with progress reporting callback.
    ///
    /// Same as `propagate_ibp_block_wise`, but calls the provided callback after each
    /// block is processed. This is useful for long-running verifications on large models.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (used for shape information)
    /// * `epsilon` - Perturbation epsilon to use for each block's fresh input
    /// * `progress_callback` - Optional callback called after each block with progress info
    ///
    /// # Example
    /// ```rust,no_run
    /// // Block-wise IBP with progress callback:
    /// // graph.propagate_ibp_block_wise_with_progress(&input, epsilon, Some(|p| {
    /// //     eprintln!("Block {}/{}: {}", p.block_index + 1, p.total_blocks, p.block_name);
    /// // })).unwrap()
    /// ```
    pub fn propagate_ibp_block_wise_with_progress<F>(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        progress_callback: Option<F>,
    ) -> Result<BlockWiseResult>
    where
        F: Fn(BlockProgress),
    {
        self.propagate_ibp_block_wise_with_options(input, epsilon, progress_callback, 0)
    }

    /// Block-wise IBP propagation with progress and max_blocks limit.
    ///
    /// Same as `propagate_ibp_block_wise_with_progress`, but can limit verification to
    /// the first `max_blocks` blocks (0 = all blocks).
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (used for shape information)
    /// * `epsilon` - Perturbation epsilon to use for each block's fresh input
    /// * `progress_callback` - Optional callback called after each block with progress info
    /// * `max_blocks` - Maximum number of blocks to verify (0 = all blocks)
    pub fn propagate_ibp_block_wise_with_options<F>(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        progress_callback: Option<F>,
        max_blocks: usize,
    ) -> Result<BlockWiseResult>
    where
        F: Fn(BlockProgress),
    {
        self.propagate_ibp_block_wise_internal(
            input,
            epsilon,
            progress_callback.as_ref(),
            None::<&fn(&BlockBoundsInfo, u64, usize)>,
            max_blocks,
            None,
        )
    }

    fn propagate_ibp_block_wise_internal<F, G>(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        progress_callback: Option<&F>,
        checkpoint_callback: Option<&G>,
        max_blocks: usize,
        resume_from: Option<&VerificationCheckpoint>,
    ) -> Result<BlockWiseResult>
    where
        F: Fn(BlockProgress),
        G: Fn(&BlockBoundsInfo, u64, usize),
    {
        let start_time = std::time::Instant::now();
        let prior_elapsed_ms = resume_from.map(|c| c.elapsed_ms).unwrap_or(0);

        if self.nodes.is_empty() {
            return Ok(BlockWiseResult {
                blocks: vec![],
                block_epsilon: epsilon,
                total_blocks: 0,
                max_sensitivity: 1.0,
                degraded_blocks: 0,
            });
        }

        let exec_order = self.exec_order()?;
        let block_nodes = Self::collect_block_nodes(exec_order);
        if block_nodes.is_empty() {
            debug!("No transformer blocks detected in graph, falling back to layer-by-layer");
            return self.block_wise_fallback_result(input, epsilon);
        }

        let (mut blocks, mut max_sensitivity, mut degraded_blocks, skip_blocks) =
            if let Some(checkpoint) = resume_from {
                (
                    checkpoint.completed_blocks.clone(),
                    checkpoint.max_sensitivity,
                    checkpoint.degraded_blocks,
                    checkpoint.next_block_index,
                )
            } else {
                (Vec::new(), 0.0_f32, 0_usize, 0_usize)
            };

        let actual_total_blocks = block_nodes.len();
        let total_blocks = if max_blocks > 0 && max_blocks < actual_total_blocks {
            max_blocks
        } else {
            actual_total_blocks
        };

        for (&block_idx, nodes_in_block) in block_nodes.iter().take(total_blocks) {
            if block_idx < skip_blocks {
                continue;
            }

            let block_name = format!("layer{}", block_idx);
            let first_node_name = &nodes_in_block[0];
            let first_node = self.nodes.get(first_node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("Node not found: {}", first_node_name))
            })?;
            let block_input_shape =
                self.resolve_block_input_shape(&block_name, first_node_name, first_node, input)?;

            let block_input_center = ArrayD::zeros(IxDyn(&block_input_shape));
            let block_input = BoundedTensor::from_epsilon(block_input_center, epsilon)?;
            let block_input_width = epsilon * 2.0;

            let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
                std::collections::HashMap::with_capacity(nodes_in_block.len());
            let mut node_infos: Vec<NodeBoundsInfo> = Vec::with_capacity(nodes_in_block.len());
            let mut qk_matmul_width: Option<f32> = None;
            let mut swiglu_width: Option<f32> = None;
            let mut block_degraded = false;
            let mut block_output_width = block_input_width;

            for node_name in nodes_in_block {
                let node = self.nodes.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!("Node not found: {}", node_name))
                })?;
                let (output_bounds, node_info, qk_width, swiglu_node_width) = self
                    .propagate_ibp_block_node(
                        node_name,
                        node,
                        &block_input,
                        block_input_width,
                        &bounds_cache,
                    )?;

                if let Some(width) = qk_width {
                    qk_matmul_width = Some(width);
                }
                if let Some(width) = swiglu_node_width {
                    swiglu_width = Some(width);
                }
                if node_info.has_degraded() {
                    block_degraded = true;
                }

                block_output_width = node_info.output_width;
                node_infos.push(node_info);
                bounds_cache.insert(node_name.clone(), output_bounds);
            }

            let block_sensitivity = if block_input_width > 0.0 {
                block_output_width / block_input_width
            } else {
                f32::INFINITY
            };
            max_sensitivity = nan_propagating_max(max_sensitivity, block_sensitivity);
            if block_degraded {
                degraded_blocks += 1;
            }

            let block_info = BlockBoundsInfo {
                block_index: block_idx,
                block_name: block_name.clone(),
                nodes: node_infos,
                input_width: block_input_width,
                output_width: block_output_width,
                sensitivity: block_sensitivity,
                qk_matmul_width,
                swiglu_width,
                degraded: block_degraded,
            };

            if let Some(callback) = progress_callback {
                callback(BlockProgress {
                    block_index: block_idx,
                    total_blocks,
                    block_name: block_name.clone(),
                    elapsed: start_time.elapsed(),
                    current_max_sensitivity: max_sensitivity,
                    degraded_so_far: degraded_blocks,
                });
            }

            if let Some(callback) = checkpoint_callback {
                let elapsed_ms = prior_elapsed_ms + start_time.elapsed().as_millis() as u64;
                callback(&block_info, elapsed_ms, total_blocks);
            }

            blocks.push(block_info);
        }

        Ok(BlockWiseResult {
            total_blocks: blocks.len(),
            blocks,
            block_epsilon: epsilon,
            max_sensitivity,
            degraded_blocks,
        })
    }

    fn propagate_ibp_block_node(
        &self,
        node_name: &str,
        node: &crate::GraphNode,
        block_input: &BoundedTensor,
        block_input_width: f32,
        bounds_cache: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<(BoundedTensor, NodeBoundsInfo, Option<f32>, Option<f32>)> {
        const MAX_BOUND: f32 = f32::MAX / 2.0;

        let input_width = if node.inputs.is_empty() {
            block_input_width
        } else {
            node.inputs
                .iter()
                .map(|inp| {
                    if inp == NETWORK_INPUT {
                        block_input_width
                    } else if let Some(cached) = bounds_cache.get(inp) {
                        cached.max_width()
                    } else {
                        block_input_width
                    }
                })
                .fold(0.0_f32, nan_propagating_max)
        };

        // Unified dispatch (#2405). Uses resolve_node_inputs for all layer
        // types, with custom Binary/Unary handling for zonotope tightening.
        let resolved = resolve_node_inputs(node, node_name, &mut |name| {
            self.bounds_for_block(name, block_input, bounds_cache)
        })?;

        // OpaqueSkip shape mismatch logging (unique to block-wise and graph_ibp).
        if matches!(&node.layer, Layer::OpaqueSkip(_)) && node.inputs.len() > 1 {
            // Only runs for multi-input OpaqueSkip, where `require_unary_input`
            // (which rejects >1 input) would always error. Use the first declared
            // input directly for the diagnostic shape comparison (#2666 regression).
            let first_input_name = node.inputs[0].as_str();
            let first_shape = self
                .bounds_for_block(first_input_name, block_input, bounds_cache)?
                .shape()
                .to_vec();
            let has_mismatch = node.inputs[1..].iter().any(|input_name| {
                self.bounds_for_block(input_name, block_input, bounds_cache)
                    .map(|b| b.shape() != first_shape.as_slice())
                    .unwrap_or(false)
            });
            if has_mismatch {
                debug!(
                    "Block-wise IBP: OpaqueSkip {} has mismatched input shapes; using first input shape",
                    node_name
                );
            }
        }

        let (output_bounds, qk_width, swiglu_width) = match resolved {
            ResolvedInputs::Binary(ref a, ref b) => {
                // Zonotope tightening for MatMul (Q@K^T), MulBinary (SwiGLU).
                match &node.layer {
                    Layer::MatMul(matmul) if matmul.transpose_b => {
                        if let Some(tighter) = self.try_attention_matmul_bounds_zonotope_block(
                            node,
                            block_input,
                            bounds_cache,
                        )? {
                            let width = tighter.max_width();
                            (tighter, Some(width), None)
                        } else {
                            (node.layer.propagate_ibp_binary(a, b)?, None, None)
                        }
                    }
                    Layer::MulBinary(_) => {
                        let ibp = node.layer.propagate_ibp_binary(a, b)?;
                        match self.try_ffn_swiglu_bounds_zonotope_block(
                            node,
                            block_input,
                            bounds_cache,
                        )? {
                            // Intersect zonotope with plain IBP: both sound, keep
                            // the tighter per element (no regression where the
                            // zonotope is looser). See dispatch::intersect_zonotope_ibp.
                            Some(zono) => {
                                let tighter = super::dispatch::intersect_zonotope_ibp(zono, ibp);
                                let width = tighter.max_width();
                                (tighter, None, Some(width))
                            }
                            None => (ibp, None, None),
                        }
                    }
                    _ => (node.layer.propagate_ibp_binary(a, b)?, None, None),
                }
            }
            ResolvedInputs::Unary(_) if matches!(&node.layer, Layer::Linear(_)) => {
                // FFN down-projection zonotope tightening (block-wise only).
                if let Some(tighter) =
                    self.try_ffn_down_zonotope_block(node, block_input, bounds_cache)?
                {
                    debug!("FFN down zonotope applied for {}", node_name);
                    (tighter, None, None)
                } else {
                    (
                        dispatch_ibp_resolved(node, node_name, resolved)?,
                        None,
                        None,
                    )
                }
            }
            other => (dispatch_ibp_resolved(node, node_name, other)?, None, None),
        };

        // DFL / expectation-decode tightening (see `ibp::dfl_envelope`):
        // intersect a constant-weighted Softmax contraction with its convex-
        // combination envelope. Intersection only tightens; no-op otherwise.
        let output_bounds = if matches!(&node.layer, Layer::Linear(_) | Layer::MatMul(_)) {
            match self.try_dfl_simplex_envelope(node, &output_bounds, block_input, bounds_cache)? {
                Some(tightened) => tightened,
                None => output_bounds,
            }
        } else {
            output_bounds
        };

        let output_width = output_bounds.max_width();
        let mut min_bound = f32::INFINITY;
        let mut max_bound = f32::NEG_INFINITY;
        let mut saturated = false;
        let mut has_nan = false;
        let mut has_infinite = false;

        for (&l, &u) in output_bounds
            .lower()
            .iter()
            .zip(output_bounds.upper().iter())
        {
            if l.is_nan() || u.is_nan() {
                has_nan = true;
            }
            if !l.is_finite() || !u.is_finite() {
                has_infinite = true;
            }
            min_bound = min_bound.min(l);
            max_bound = max_bound.max(u);
            if l <= -0.999 * MAX_BOUND || u >= 0.999 * MAX_BOUND {
                saturated = true;
            }
        }

        let sensitivity = if input_width > 0.0 && input_width.is_finite() {
            output_width / input_width
        } else if output_width == 0.0 {
            1.0
        } else {
            f32::INFINITY
        };

        let node_info = NodeBoundsInfo {
            name: node_name.to_string(),
            layer_type: node.layer.layer_type().to_string(),
            input_width,
            output_width,
            sensitivity,
            output_shape: output_bounds.shape().to_vec(),
            min_bound,
            max_bound,
            saturated,
            has_nan,
            has_infinite,
        };

        // NaN firewall (#2812 Slice 3, #2706). Metadata recorded above.
        check_nan_firewall(
            &output_bounds,
            "IBP block-wise",
            node_name,
            node.layer.layer_type(),
        )?;

        Ok((output_bounds, node_info, qk_width, swiglu_width))
    }

    /// Derive the block input shape from the first node and reject malformed blocks.
    ///
    /// Empty-input nodes are invalid in this context: they previously fell through the
    /// `_input` branch and fabricated the network input shape.
    fn resolve_block_input_shape(
        &self,
        block_name: &str,
        first_node_name: &str,
        first_node: &crate::GraphNode,
        input: &BoundedTensor,
    ) -> Result<Vec<usize>> {
        let first_input = first_node.inputs.first().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Block '{}' first node '{}' ({}) has no inputs",
                block_name,
                first_node_name,
                first_node.layer.layer_type()
            ))
        })?;

        if first_input == NETWORK_INPUT {
            Ok(input.shape().to_vec())
        } else {
            // For now block-wise mode approximates with the global input shape, matching
            // prior behavior for non-_input predecessors.
            Ok(input.shape().to_vec())
        }
    }

    /// Block-wise IBP propagation with checkpoint support for resumable verification.
    ///
    /// Supports resuming from a previous checkpoint and saving progress after each block.
    /// Useful for multi-hour verification of very large models (32B+).
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (used for shape information)
    /// * `epsilon` - Perturbation epsilon to use for each block's fresh input
    /// * `progress_callback` - Optional callback called after each block with progress info
    /// * `checkpoint_callback` - Optional callback called after each block to save checkpoint
    /// * `max_blocks` - Maximum number of blocks to verify (0 = all blocks)
    /// * `resume_from` - Optional checkpoint to resume from (skips completed blocks)
    pub fn propagate_ibp_block_wise_with_checkpoint<F, G>(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        progress_callback: Option<F>,
        checkpoint_callback: Option<G>,
        max_blocks: usize,
        resume_from: Option<&VerificationCheckpoint>,
    ) -> Result<BlockWiseResult>
    where
        F: Fn(BlockProgress),
        G: Fn(&BlockBoundsInfo, u64, usize), // (block_info, elapsed_ms, total_blocks)
    {
        self.propagate_ibp_block_wise_internal(
            input,
            epsilon,
            progress_callback.as_ref(),
            checkpoint_callback.as_ref(),
            max_blocks,
            resume_from,
        )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::GraphNetwork;
    use crate::{BoundedTensor, GraphNode, Layer, ReLULayer};

    /// Construct a GraphNode with empty inputs, bypassing the debug_assert in
    /// GraphNode::new() which validates min_inputs. This is intentional — these
    /// tests exercise the downstream error path in resolve_block_input_shape(),
    /// not the constructor guard (#2481).
    fn make_empty_input_node(name: &str, layer: Layer) -> GraphNode {
        GraphNode {
            name: name.to_string(),
            layer,
            inputs: vec![],
        }
    }

    #[test]
    fn test_block_wise_rejects_empty_first_node_inputs_2112() {
        let mut graph = GraphNetwork::new();
        graph.add_node(make_empty_input_node("layer0_relu", Layer::ReLU(ReLULayer)));

        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("valid input bounds");

        let err = graph
            .propagate_ibp_block_wise_with_options(
                &input,
                0.1,
                None::<fn(crate::types::BlockProgress)>,
                0,
            )
            .expect_err("empty first-node inputs must return InvalidSpec");
        let msg = err.to_string();
        assert!(
            msg.contains("layer0") && msg.contains("has no inputs"),
            "expected block + empty-input diagnostic, got: {msg}"
        );
    }

    #[test]
    fn test_block_wise_checkpoint_rejects_empty_first_node_inputs_2112() {
        let mut graph = GraphNetwork::new();
        graph.add_node(make_empty_input_node("layer0_relu", Layer::ReLU(ReLULayer)));

        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("valid input bounds");

        let err = graph
            .propagate_ibp_block_wise_with_checkpoint(
                &input,
                0.1,
                None::<fn(crate::types::BlockProgress)>,
                None::<fn(&crate::types::BlockBoundsInfo, u64, usize)>,
                0,
                None,
            )
            .expect_err("empty first-node inputs must return InvalidSpec");
        let msg = err.to_string();
        assert!(
            msg.contains("layer0") && msg.contains("has no inputs"),
            "expected block + empty-input diagnostic, got: {msg}"
        );
    }
}
