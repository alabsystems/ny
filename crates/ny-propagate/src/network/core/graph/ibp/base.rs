// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Base IBP non-delegating methods and domain clipping helpers.
//!
//! Delegation methods (`propagate_ibp`, `propagate_ibp_sound`) moved to
//! `network/dispatch/graph_ibp.rs` (#2380). This file retains only methods
//! that are self-contained (no sibling imports).

use crate::domain_clip::DomainClipper;

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::super::{GraphNetwork, NETWORK_INPUT};
use super::dispatch::{check_nan_firewall, dispatch_ibp_for_node};

impl GraphNetwork {
    /// Collect activation statistics from a concrete forward pass.
    ///
    /// Runs a concrete forward pass through the network using the center values
    /// of the input bounds, collecting per-layer activation statistics for use
    /// with domain clipping.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor (center values will be used)
    /// * `clipper` - Domain clipper to store statistics in
    ///
    /// # Example
    /// ```rust,no_run
    /// // Collect activation statistics for domain clipping:
    /// // let mut clipper = DomainClipper::default();
    /// // graph.collect_activation_statistics(&input, &mut clipper).unwrap();
    /// // let bounds = graph.propagate_ibp_with_clipper(&input, &mut clipper).unwrap();
    /// ```
    pub fn collect_activation_statistics(
        &self,
        input: &BoundedTensor,
        clipper: &mut DomainClipper,
    ) -> Result<()> {
        if self.nodes.is_empty() {
            return Ok(());
        }

        // Use center of input bounds as concrete value
        let center = (input.lower() + input.upper()) / 2.0;

        // Get execution order
        let exec_order = self.exec_order()?;

        // Store concrete values for each node
        let mut value_cache: std::collections::HashMap<String, ArrayD<f32>> =
            std::collections::HashMap::new();

        // Process nodes in topological order
        for node_name in exec_order {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Resolve concrete values via unified dispatch (#2405).
            // Wraps get_concrete_value → BoundedTensor::concrete, then
            // extracts lower bound (center value) from the propagation result.
            let output_bounds = dispatch_ibp_for_node(node, node_name, &mut |name| {
                let val = self.concrete_value(name, &center, &value_cache)?;
                BoundedTensor::concrete(val)
            })?;
            let output_value = output_bounds.lower().clone();

            // Record statistics for this layer
            clipper.observe(node_name, &output_value)?;

            // Store for downstream nodes
            value_cache.insert(node_name.clone(), output_value);
        }

        Ok(())
    }

    /// Concrete value from cache or input.
    pub(crate) fn concrete_value(
        &self,
        input_name: &str,
        center: &ArrayD<f32>,
        cache: &std::collections::HashMap<String, ArrayD<f32>>,
    ) -> Result<ArrayD<f32>> {
        if input_name == NETWORK_INPUT {
            Ok(center.clone())
        } else {
            cache.get(input_name).cloned().ok_or_else(|| {
                NyError::InvalidSpec(format!("Concrete value not found for {}", input_name))
            })
        }
    }

    /// Propagate bounds through the graph using IBP with domain clipping.
    ///
    /// Similar to `propagate_ibp`, but applies domain clipping after each layer
    /// using statistics collected by the clipper. This can significantly tighten
    /// bounds for deep networks where bound explosion is a problem.
    ///
    /// # Arguments
    /// * `input` - Input bounded tensor
    /// * `clipper` - Domain clipper with pre-collected statistics
    ///
    /// # Returns
    /// Output bounds after propagation with clipping applied.
    ///
    /// # Example
    /// ```rust,no_run
    /// // Propagate with domain clipping for tighter bounds:
    /// // let mut clipper = DomainClipper::default();
    /// // graph.collect_activation_statistics(&sample, &mut clipper).unwrap();
    /// // let bounds = graph.propagate_ibp_with_clipper(&input, &mut clipper).unwrap();
    /// ```
    pub fn propagate_ibp_with_clipper(
        &self,
        input: &BoundedTensor,
        clipper: &mut DomainClipper,
    ) -> Result<BoundedTensor> {
        if self.nodes.is_empty() {
            return Ok(input.clone());
        }

        // Get execution order
        let exec_order = self.exec_order()?;

        // Store bounds for each node's output
        let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        // Track total clipping effect
        let mut total_reduction = 0.0_f32;

        // Process nodes in topological order
        for node_name in exec_order {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            // Compute output bounds using unified dispatch (#2405).
            let output_bounds = dispatch_ibp_for_node(node, node_name, &mut |name| {
                Ok(self.bounds_ref(name, input, &bounds_cache)?.clone())
            })?;

            // NaN firewall (#2812 Slice 2, #2706).
            check_nan_firewall(
                &output_bounds,
                "IBP with clipper",
                node_name,
                node.layer.layer_type(),
            )?;

            // Apply domain clipping
            let (clipped_bounds, reduction) = clipper.clip_bounds(node_name, &output_bounds)?;
            total_reduction += reduction;

            bounds_cache.insert(node_name.clone(), clipped_bounds);
        }

        if total_reduction > 0.0 {
            debug!(
                "Domain clipping reduced total bound width by {:.2e}",
                total_reduction
            );
        }

        // Return the output node's bounds
        if self.output_node.is_empty() {
            let last_name = exec_order
                .last()
                .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?;
            bounds_cache.remove(last_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("Output bounds not found for {}", last_name))
            })
        } else {
            bounds_cache.remove(&self.output_node).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Output node {} not found in results",
                    self.output_node
                ))
            })
        }
    }
}
