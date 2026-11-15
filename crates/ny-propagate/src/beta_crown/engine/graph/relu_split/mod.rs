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
pub(crate) mod depth_drop;
mod domain_filter;
pub(super) mod queue_budget;
mod split;
mod status;

#[cfg(test)]
mod tests;

use std::time::Instant;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::{info, instrument};

use crate::beta_crown::domain::SharedGraphBootstrap;
use crate::beta_crown::engine::graph::shared::init::{
    compute_graph_bab_bootstrap, compute_graph_root_output_bounds,
    graph_root_output_bounds_warmup_fallback,
};
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
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
    /// remaining wall-clock time instead of `self.config.timeout`.
    pub fn verify_graph_relu_split_with_engine_gpu(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
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
    /// wall-clock deadline is exceeded.
    pub fn compute_initial_graph_bounds(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<(
        std::collections::HashMap<String, BoundedTensor>,
        BoundedTensor,
    )> {
        graph.ensure_proof_eligible()?;
        self.config.validate()?;
        let graph = self.configured_graph_for_crown(graph);
        let graph = &graph;
        let engine = self.engine();
        let bootstrap = compute_graph_bab_bootstrap(graph, input, &self.config, engine, deadline)?;
        // Warmup cap must survive the non-alpha root-output path:
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

    /// #disjunctive-shared-bootstrap: the bootstrap the serial disjunctive
    /// loop lends to every clause, computed ONCE for the property's input box.
    ///
    /// Unlike [`Self::compute_initial_graph_bounds`] this KEEPS the optimised
    /// root alpha state (that function drops it, so its lane starts every BaB
    /// child from heuristic slopes) and publishes the margin-subset seed for
    /// the UNION of the clauses' referenced output indices, the way the
    /// multi-objective root does -- a superset of rows tightened is sound and
    /// leaves no clause's rows looser than its own bootstrap would have.
    ///
    /// Deadlines mirror the per-objective lane (`bab_loop.rs`): the mandatory
    /// node sweep gets `bootstrap_deadline` (the caller's BaB boundary) and the
    /// iterative root alpha pass gets `iterative_deadline` (the caller's
    /// `initial_bounds_fraction` slice of it). A PHASE-only expiry of the root
    /// pass is recovered the #budget-abandonment way when that key is armed;
    /// otherwise, and on any other deadline expiry, this returns `Ok(None)` --
    /// "not shared" -- and the caller runs the per-clause bootstrap exactly as
    /// before. It never mints a verdict, so a decline can never change one.
    pub fn compute_shared_graph_bootstrap(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        referenced_outputs: Vec<usize>,
        bootstrap_deadline: Option<Instant>,
        iterative_deadline: Option<Instant>,
    ) -> Result<Option<SharedGraphBootstrap>> {
        graph.ensure_proof_eligible()?;
        self.config.validate()?;
        let graph = self.configured_graph_for_crown(graph);
        let graph = &graph;
        let engine = self.engine();
        let start = Instant::now();
        // Same RAII scope as `compute_relu_split_initial_bounds`: the guard
        // restores the empty publication on every exit path below.
        let _margin_seed_guard =
            crate::output_margin_seed::MarginOutputSeedGuard::publish(referenced_outputs);
        info!(
            "#disjunctive-shared-bootstrap: computing one graph bootstrap for the whole \
             disjunction (alpha={}, iterative_cap_s={:.1})",
            self.config.use_alpha_crown,
            iterative_deadline.map_or(0.0, |dl| dl.saturating_duration_since(start).as_secs_f64()),
        );
        let mut bootstrap = match compute_graph_bab_bootstrap(
            graph,
            input,
            &self.config,
            engine,
            bootstrap_deadline,
        ) {
            Ok(bootstrap) => bootstrap,
            Err(error) if error.is_deadline_exceeded() => {
                info!("#disjunctive-shared-bootstrap: node sweep expired; not shared");
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        // The iterative root pass runs on the warmup slice, as in bab_loop.rs.
        bootstrap.alpha_config.deadline = iterative_deadline;
        let (output_bounds, warmup_fallback_fired) = match compute_graph_root_output_bounds(
            graph,
            input,
            &self.config,
            engine,
            &bootstrap,
            iterative_deadline,
        ) {
            Ok(output) => (output, false),
            Err(error) if error.is_deadline_exceeded() => {
                let now = Instant::now();
                let phase_only = matches!(
                    ny_core::phase_yield::classify_expiry(
                        now,
                        iterative_deadline,
                        bootstrap_deadline
                    ),
                    ny_core::phase_yield::Expiry::PhaseOnly
                );
                if !(self.config.initial_bounds_warmup_fallback && phase_only) {
                    info!("#disjunctive-shared-bootstrap: root pass expired; not shared");
                    return Ok(None);
                }
                // FIX A's retry slice: a fraction of the REMAINING BaB budget,
                // never past the global deadline.
                let frac = self
                    .config
                    .phase_budget
                    .initial_bounds_fraction
                    .clamp(0.0, 1.0);
                let retry_deadline = bootstrap_deadline.map(|global| {
                    let slice = global.saturating_duration_since(now).mul_f32(frac);
                    GraphBabLifecycle::fail_closed_deadline(now, slice).min(global)
                });
                match graph_root_output_bounds_warmup_fallback(
                    graph,
                    input,
                    &self.config,
                    engine,
                    &bootstrap,
                    retry_deadline,
                ) {
                    Some((output, tier)) => {
                        info!(
                            "#disjunctive-shared-bootstrap: root pass expired; using the \
                             warmup-fallback enclosure (tier={})",
                            tier.label()
                        );
                        (output, true)
                    }
                    None => return Ok(None),
                }
            }
            Err(error) => return Err(error),
        };
        Ok(Some(SharedGraphBootstrap::new(
            bootstrap.initial_node_bounds,
            output_bounds,
            bootstrap.root_alpha_state,
            start.elapsed().as_secs_f64(),
            warmup_fallback_fired,
        )))
    }
}
