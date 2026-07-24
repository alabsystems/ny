// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ReLU-splitting branch-and-bound for graph networks.
//!
//! Implements the full DAG β-CROWN BaB loop that branches on unstable ReLU neurons
//! rather than input dimensions. For each domain, unstable neurons (where lower < 0 < upper)
//! are split into active (x ≥ 0) and inactive (x ≤ 0) constraints, creating two child
//! domains with tighter bounds. More precise than input splitting but requires constraint
//! tracking through the graph topology.
//!
//! Entry point: `BetaCrownVerifier::verify_graph_relu_split`.

mod aggregate;
mod bab_loop;
mod child_eval;
mod domain_filter;
mod split;
mod status;

#[cfg(test)]
mod tests;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::instrument;

use crate::beta_crown::engine::graph::shared::init::{
    compute_graph_bab_bootstrap, compute_graph_root_output_bounds,
};
use crate::beta_crown::result::BetaCrownResult;
use crate::GraphNetwork;

use super::super::BetaCrownVerifier;

impl BetaCrownVerifier {
    /// GraphNetwork verification with ReLU-splitting branch-and-bound.
    ///
    /// This is the full DAG β-CROWN implementation that branches on unstable
    /// ReLU neurons rather than input dimensions. It is more precise than
    /// input splitting but requires more complex constraint tracking.
    ///
    /// # REQUIRES
    /// - `graph` must be a valid GraphNetwork with compatible node dimensions
    /// - `input` shape must match graph's input node dimension
    /// - `input.lower()[i] <= input.upper()[i]` for all elements (well-formed bounds)
    /// - `objective` length must match graph's output dimension
    ///
    /// # ENSURES
    /// - If `Verified`: `objective · output > threshold` for all inputs in region (sound)
    /// - If `PotentialViolation`: counterexample region found
    /// - Sound: no false positives (Verified implies property holds)
    /// - Bounds tighter than input-split method (ReLU constraints are more precise)
    ///
    /// # Algorithm
    /// 1. Collect initial node bounds via the configured bootstrap mode
    /// 2. Run CROWN with constraint-aware ReLU relaxation
    /// 3. Find unstable neurons (l < 0 < u) that aren't constrained
    /// 4. Select neuron to split using branching heuristic
    /// 5. Create two child domains (active/inactive constraints)
    /// 6. Repeat until all domains verified or limits reached
    ///
    /// When `use_alpha_crown` is enabled, the initial bounds are computed using
    /// α-CROWN optimization which provides ~10x tighter bounds than IBP.
    /// When `use_forward_bounds` is enabled, the root output reuses the shared
    /// forward-linear node-bound bootstrap instead of dropping back to plain
    /// DAG-CROWN intermediates.
    ///
    /// # Limitations
    /// - Branching heuristic is currently "widest bound"
    #[instrument(skip(self, graph, input, objective), fields(threshold, input_shape = ?input.shape(), num_nodes = graph.nodes.len()))]
    pub fn verify_graph_relu_split(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
    ) -> Result<BetaCrownResult> {
        self.verify_graph_relu_split_impl(graph, input, objective, threshold, self.engine(), None)
    }

    /// Verify GraphNetwork with ReLU splitting and optional GPU acceleration.
    ///
    /// `deadline`: If `Some`, the BaB engine derives its phase budgets from
    /// remaining wall-clock time instead of `self.config.timeout` (#4321).
    pub fn verify_graph_relu_split_with_engine_gpu(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<BetaCrownResult> {
        let engine = self.resolve_engine(engine);
        self.verify_graph_relu_split_impl(graph, input, objective, threshold, engine, deadline)
    }

    /// Pre-compute initial bounds (intermediate + output) using the configured
    /// graph root mode (`alpha-crown`, `forward+crown`, or the plain CROWN path).
    ///
    /// Call this once before verifying multiple constraints on the same graph/input.
    /// The returned tuple contains (intermediate node bounds, output bounds).
    ///
    /// This optimization provides ~9x speedup for CIFAR-10 classification
    /// (9 constraints that would each re-compute bounds).
    /// `deadline`: If set, alpha-CROWN optimization will bail early when this
    /// wall-clock deadline is exceeded (#2698).
    pub fn compute_initial_graph_bounds(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        deadline: Option<std::time::Instant>,
    ) -> Result<(
        std::collections::HashMap<String, BoundedTensor>,
        BoundedTensor,
    )> {
        let graph = self.configured_graph_for_crown(graph);
        let graph = &graph;
        let engine = self.engine();
        let bootstrap = compute_graph_bab_bootstrap(graph, input, &self.config, engine, deadline)?;
        // Warmup cap (#4095) must survive the non-alpha root-output path:
        // reuse the bootstrap deadline so the caller's budget is honored.
        let output_bounds = compute_graph_root_output_bounds(
            graph,
            input,
            &self.config,
            engine,
            &bootstrap,
            bootstrap.alpha_config.deadline,
        )?;
        Ok((bootstrap.initial_node_bounds, output_bounds))
    }
}
