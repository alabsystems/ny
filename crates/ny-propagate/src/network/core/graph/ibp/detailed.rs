// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Detailed layer-by-layer IBP diagnostics.

use crate::bounds::nan_propagating_max;
use crate::layers::Layer;
use crate::types::{LayerByLayerResult, LayerProgress, NodeBoundsInfo};

use super::super::{GraphNetwork, NETWORK_INPUT};
use super::dispatch::{
    check_nan_firewall, dispatch_ibp_resolved, resolve_node_inputs, ResolvedInputs,
};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

impl GraphNetwork {
    /// Propagate bounds through the graph using IBP, returning detailed per-node information.
    ///
    /// This is useful for layer-by-layer verification to track bound growth through the
    /// network and identify where bounds saturate or degrade.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor
    /// * `epsilon` - Input perturbation epsilon (for reporting)
    ///
    /// # Returns
    /// A `LayerByLayerResult` containing detailed bounds information for each node.
    pub fn propagate_ibp_detailed(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
    ) -> Result<LayerByLayerResult> {
        self.propagate_ibp_detailed_with_progress(input, epsilon, None::<fn(LayerProgress)>)
    }

    /// Propagate bounds through the graph using IBP, returning detailed per-node information,
    /// with optional progress callback.
    ///
    /// Same as `propagate_ibp_detailed`, but calls the provided callback after each node is
    /// processed. This is useful for long-running layer-by-layer runs on large graphs.
    pub fn propagate_ibp_detailed_with_progress<F>(
        &self,
        input: &BoundedTensor,
        epsilon: f32,
        progress_callback: Option<F>,
    ) -> Result<LayerByLayerResult>
    where
        F: Fn(LayerProgress),
    {
        let start_time = std::time::Instant::now();

        if self.nodes.is_empty() {
            return Ok(LayerByLayerResult {
                nodes: vec![],
                input_epsilon: epsilon,
                final_width: input.max_width(),
                degraded_at_node: None,
                total_nodes: 0,
            });
        }

        const MAX_BOUND: f32 = f32::MAX / 2.0;

        // Get execution order
        let exec_order = self.exec_order()?;
        let total_nodes = exec_order.len();

        // Store bounds for each node's output
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::with_capacity(total_nodes);

        // Track per-node information
        let mut node_infos: Vec<NodeBoundsInfo> = Vec::with_capacity(total_nodes);
        let mut degraded_at_node: Option<usize> = None;
        let mut degraded_so_far: usize = 0;
        let mut max_sensitivity_so_far: f32 = 1.0;

        // Process nodes in topological order
        for (node_index, node_name) in exec_order.iter().enumerate() {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Get input bounds width.
            //
            // For DAG nodes (especially binary ops like Add/MatMul), use the max width across
            // all inputs to avoid underestimating the node's effective input uncertainty.
            let input_width = if node.inputs.is_empty() {
                input.max_width()
            } else {
                node.inputs
                    .iter()
                    .map(|inp| {
                        if inp == NETWORK_INPUT {
                            input.max_width()
                        } else {
                            bounds_cache
                                .get(inp)
                                .map(|b| b.max_width())
                                .unwrap_or(input.max_width())
                        }
                    })
                    .fold(0.0_f32, nan_propagating_max)
            };

            // Unified dispatch (#2405). Uses resolve_node_inputs for all layer
            // types, with custom Binary handling for zonotope tightening.
            let resolved = resolve_node_inputs(node, node_name, &mut |name| {
                Ok(self.bounds_ref(name, input, &bounds_cache)?.clone())
            })?;

            let output_bounds = match resolved {
                ResolvedInputs::Binary(ref a, ref b) => {
                    // Zonotope tightening for attention MatMul and SwiGLU MulBinary.
                    match &node.layer {
                        Layer::MatMul(matmul) if matmul.transpose_b => {
                            if let Some(tighter) = self.try_attention_matmul_bounds_zonotope(
                                node,
                                input,
                                &bounds_cache,
                            )? {
                                tighter
                            } else {
                                node.layer.propagate_ibp_binary(a, b)?
                            }
                        }
                        Layer::MulBinary(_) => {
                            let ibp = node.layer.propagate_ibp_binary(a, b)?;
                            // Intersect zonotope with plain IBP: both sound, keep
                            // the tighter per element (no regression where the
                            // zonotope is looser). See dispatch::intersect_zonotope_ibp.
                            match self.try_ffn_swiglu_bounds_zonotope(node, input, &bounds_cache)? {
                                Some(zono) => crate::network::core::graph::ibp::dispatch::intersect_zonotope_ibp(zono, ibp),
                                None => ibp,
                            }
                        }
                        _ => node.layer.propagate_ibp_binary(a, b)?,
                    }
                }
                other => dispatch_ibp_resolved(node, node_name, other)?,
            };

            // DFL / expectation-decode tightening (see `ibp::dfl_envelope`):
            // intersect a constant-weighted Softmax contraction with its convex-
            // combination envelope. Intersection only tightens; no-op otherwise.
            let output_bounds = if matches!(&node.layer, Layer::Linear(_) | Layer::MatMul(_)) {
                match self.try_dfl_simplex_envelope(node, &output_bounds, input, &bounds_cache)? {
                    Some(tightened) => tightened,
                    None => output_bounds,
                }
            } else {
                output_bounds
            };

            // Collect statistics
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
                name: node_name.clone(),
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
                "IBP detailed",
                node_name,
                node.layer.layer_type(),
            )?;

            // Track first degraded node
            if degraded_at_node.is_none() && node_info.has_degraded() {
                degraded_at_node = Some(node_infos.len());
            }

            if node_info.has_degraded() {
                degraded_so_far += 1;
            }
            if sensitivity.is_finite() {
                max_sensitivity_so_far = max_sensitivity_so_far.max(sensitivity);
            } else {
                max_sensitivity_so_far = f32::INFINITY;
            }

            if let Some(ref callback) = progress_callback {
                callback(LayerProgress {
                    node_index,
                    total_nodes,
                    node_name: node_info.name.clone(),
                    layer_type: node_info.layer_type.clone(),
                    elapsed: start_time.elapsed(),
                    current_max_sensitivity: max_sensitivity_so_far,
                    degraded_so_far,
                });
            }

            node_infos.push(node_info);
            bounds_cache.insert(node_name.clone(), output_bounds);
        }

        // Get final output width
        let output_node_name = if self.output_node.is_empty() {
            exec_order.last().map(|s| s.as_str()).unwrap_or("")
        } else {
            &self.output_node
        };
        let final_width = bounds_cache
            .get(output_node_name)
            .map(|b| b.max_width())
            .unwrap_or(f32::INFINITY);

        Ok(LayerByLayerResult {
            nodes: node_infos,
            input_epsilon: epsilon,
            final_width,
            degraded_at_node,
            total_nodes,
        })
    }
}
