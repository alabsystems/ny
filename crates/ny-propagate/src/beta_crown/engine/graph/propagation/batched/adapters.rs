// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::batched_domain::BatchedDomains;
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::graph::DomainCrownResult;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::faer_parallelism::RayonTaskGuard;
use crate::GraphNetwork;

use super::{BatchedBackwardContext, BatchedBackwardMode, BatchedBackwardResult};

impl BetaCrownVerifier {
    /// Batched CROWN backward propagation using BatchedBackwardContext.
    ///
    /// This is the preferred API for GPU-accelerated backward propagation. It takes
    /// a `BatchedBackwardContext` which provides direct access to pre-batched tensors,
    /// enabling efficient GPU transfer without tuple conversions.
    ///
    /// # Performance
    /// For N domains processing through L Linear layers:
    /// - Sequential: N × L GPU kernel launches (each small)
    /// - Batched: L GPU kernel launches (each large, good GPU utilization)
    ///
    /// # Arguments
    /// * `graph` - The network graph
    /// * `ctx` - Batched context with pre-packed domain data
    /// * `objective` - Objective coefficients (same for all domains)
    /// * `engine` - GPU compute engine
    ///
    /// # Returns
    /// Vec of (output_bounds, node_bounds_cache) per domain
    ///
    /// # Reference
    /// alpha-beta-CROWN: `complete_verifier/branching_domains.py:270-356`
    // Production callers use `propagate_crown_with_batched_domains_full_timed`.
    // Retained for test use (tests.rs, tests_soundness.rs).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_batched_with_context(
        &self,
        graph: &GraphNetwork,
        ctx: &BatchedBackwardContext,
        objective: &[f32],
        engine: &dyn GemmEngine,
    ) -> Result<Vec<DomainCrownResult>> {
        let result = self.batched_forward_then_backward(
            graph,
            ctx,
            objective,
            engine,
            BatchedBackwardMode::Standard,
        )?;
        Ok(result.results)
    }

    /// Batched CROWN backward propagation with lA capture.
    ///
    /// Like `propagate_crown_batched_with_context` but also captures intermediate
    /// `LinearBounds` (lA matrices) at each node during the backward pass. These
    /// can be cached in child domains to avoid recomputation.
    ///
    /// # Arguments
    /// * `graph` - The network graph
    /// * `ctx` - Batched context with pre-packed domain data
    /// * `objective` - Objective coefficients (same for all domains)
    /// * `engine` - GPU compute engine
    ///
    /// # Returns
    /// `BatchedBackwardResult` containing:
    /// - `results`: Vec of (output_bounds, node_bounds_cache) per domain
    /// - `intermediate_la`: Map from node name to LinearBounds per domain
    ///
    /// # Reference
    /// Issue: #1564 (lA matrix caching)
    /// alpha-beta-CROWN: `complete_verifier/tensor_storage.py` (all_lAs storage)
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_batched_with_context_capture_la(
        &self,
        graph: &GraphNetwork,
        ctx: &BatchedBackwardContext,
        objective: &[f32],
        engine: &dyn GemmEngine,
    ) -> Result<BatchedBackwardResult> {
        // Capture intermediate lA only when warm-start is enabled — no point
        // storing lA that will never be consumed. (#1669)
        let capture_intermediate = self.config.enable_la_warm_start;
        self.batched_forward_then_backward(
            graph,
            ctx,
            objective,
            engine,
            BatchedBackwardMode::WithLaCapture {
                histories: &ctx.histories,
                cached_la: &ctx.cached_la,
                capture_intermediate,
            },
        )
    }

    /// Core batched CROWN backward implementation shared by standard and lA-capture modes.
    ///
    /// Runs the forward pass (parallel per-domain intermediate bounds computation),
    /// then delegates to `propagate_crown_batched_backward_core` for the backward pass.
    fn batched_forward_then_backward(
        &self,
        graph: &GraphNetwork,
        ctx: &BatchedBackwardContext,
        objective: &[f32],
        engine: &dyn GemmEngine,
        mode: BatchedBackwardMode<'_>,
    ) -> Result<BatchedBackwardResult> {
        if ctx.is_empty() {
            return Ok(BatchedBackwardResult {
                results: Vec::new(),
                intermediate_la: if matches!(mode, BatchedBackwardMode::WithLaCapture { .. }) {
                    Some(Vec::new())
                } else {
                    None
                },
                stage_timing: None,
            });
        }

        let n_domains = ctx.len();
        let plan = graph.dispatch_plan()?;

        tracing::debug!(
            n_domains = n_domains,
            n_layers = plan.exec_order.len(),
            "Starting batched CROWN backward pass"
        );

        // ===== FORWARD PASS: Compute intermediate bounds for each domain =====
        // Uses batched input tensors sliced per-domain for constraint application.
        // Future optimization: batch the entire forward pass on GPU.
        let forward_start = Instant::now();
        let forward_results: Vec<_> = (0..n_domains)
            .into_par_iter()
            .map(|idx| {
                let _rayon_task_guard = RayonTaskGuard::new();
                let input = ctx.batched.input_bounds_at(idx)?;
                self.compute_constrained_forward_bounds(
                    graph,
                    &input,
                    ctx.histories[idx],
                    ctx.base_bounds[idx],
                    ctx.delta_seeds[idx], // #cone-delta: dark, NY_CONE_REFRESH-gated
                )
            })
            .collect();
        let forward_elapsed_s = forward_start.elapsed().as_secs_f64();

        // Check for errors and extract bounds_caches
        let mut bounds_caches: Vec<std::collections::HashMap<String, Arc<BoundedTensor>>> =
            Vec::with_capacity(n_domains);
        let mut constrained_inputs: Vec<BoundedTensor> = Vec::with_capacity(n_domains);

        for (i, result) in forward_results.into_iter().enumerate() {
            match result {
                Ok((cache, input)) => {
                    bounds_caches.push(cache);
                    constrained_inputs.push(input);
                }
                Err(e) if e.is_infeasible_domain() => {
                    // #2926: Propagate InfeasibleDomain without type erasure.
                    // The caller's fallback path handles this correctly.
                    return Err(e);
                }
                Err(e) => {
                    return Err(NyError::InvalidSpec(format!(
                        "Forward pass failed for domain {}: {}",
                        i, e
                    )));
                }
            }
        }

        // Delegate to unified backward pass
        let backward_start = Instant::now();
        let mut result = self.propagate_crown_batched_backward_core(
            graph,
            n_domains,
            plan,
            &bounds_caches,
            &constrained_inputs,
            &ctx.beta_states,
            &ctx.alpha_states,
            objective,
            engine,
            mode,
            ctx.mul_binary_alphas, // #4284: thread shared MulBinary alphas
        )?;
        let backward_elapsed_s = backward_start.elapsed().as_secs_f64();

        tracing::info!(
            n_domains,
            forward_s = forward_elapsed_s,
            backward_s = backward_elapsed_s,
            forward_pct = format!(
                "{:.1}",
                forward_elapsed_s / (forward_elapsed_s + backward_elapsed_s).max(1e-9) * 100.0
            ),
            "batched CROWN stage timing (#4398)"
        );

        result.stage_timing = Some(super::BatchedStageTiming {
            forward_elapsed_s,
            backward_elapsed_s,
        });

        Ok(result)
    }

    /// Propagate CROWN bounds for batched domains, returning full results including node caches.
    ///
    /// This is a variant of `propagate_crown_with_batched_domains` that returns the full
    /// output bounds and node caches needed by the BaB loop to update child domain bounds.
    ///
    /// # Arguments
    /// * `graph` - GraphNetwork to verify
    /// * `domains` - Slice of domain references (must match batched.batch_size())
    /// * `batched` - Pre-batched domain representation for GPU transfer
    /// * `objective` - Objective coefficients (same for all domains)
    /// * `engine` - GPU compute engine
    ///
    /// # Returns
    /// Vec of (output_bounds, node_cache) tuples per domain, or error.
    /// Returns None for domains that fail propagation.
    pub fn propagate_crown_with_batched_domains_full(
        &self,
        graph: &GraphNetwork,
        domains: &[&GraphBabDomain],
        batched: &BatchedDomains,
        objective: &[f32],
        engine: &dyn GemmEngine,
    ) -> Result<Vec<Option<DomainCrownResult>>> {
        let (results, _timing) = self.propagate_crown_with_batched_domains_full_timed(
            graph, domains, batched, objective, engine,
        )?;
        Ok(results)
    }

    /// Like `propagate_crown_with_batched_domains_full` but also returns
    /// forward/backward stage timing for executor-level observability.
    /// Part of #4398 Packet B.
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_with_batched_domains_full_timed(
        &self,
        graph: &GraphNetwork,
        domains: &[&GraphBabDomain],
        batched: &BatchedDomains,
        objective: &[f32],
        engine: &dyn GemmEngine,
    ) -> Result<(
        Vec<Option<DomainCrownResult>>,
        Option<super::BatchedStageTiming>,
    )> {
        if domains.is_empty() {
            return Ok((Vec::new(), None));
        }

        let ctx = BatchedBackwardContext::from_domains(domains, batched)?;
        let result = self.batched_forward_then_backward(
            graph,
            &ctx,
            objective,
            engine,
            BatchedBackwardMode::Standard,
        )?;

        let timing = result.stage_timing;
        let results = result.results.into_iter().map(Some).collect();
        Ok((results, timing))
    }
}
