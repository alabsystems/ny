// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::batched_domain::{BatchedDomains, DomainUpdate};
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::graph::DomainCrownResult;
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::GraphNetwork;

use super::BetaCrownVerifier;

impl BetaCrownVerifier {
    /// Batched CROWN backward propagation for multiple domains.
    ///
    /// This function processes N domains through the backward pass with true tensor-level
    /// batching at Linear layers. Instead of calling `propagate_crown_with_graph_constraints`
    /// N times (each launching GPU kernels), this batches all Linear layer GEMMs into
    /// single kernel launches, dramatically improving GPU utilization.
    ///
    /// # Performance
    /// For N domains processing through L Linear layers:
    /// - Sequential (old): N × L GPU kernel launches (each small)
    /// - Batched (this): L GPU kernel launches (each large, good GPU utilization)
    ///
    /// For cifar10_resnet with ~8 Linear layers and batch_size=64:
    /// - Old: 512 small GPU kernel launches per batch
    /// - New: 8 large GPU kernel launches per batch
    ///
    /// # Arguments
    /// * `graph` - The network graph
    /// * `domain_data` - Vec of (input_bounds, history, beta_state, base_bounds) per domain
    /// * `objective` - Objective coefficients (same for all domains)
    /// * `engine` - GPU compute engine
    ///
    /// # Returns
    /// Vec of (output_bounds, node_bounds_cache) per domain
    ///
    /// # Deprecated
    /// Prefer `propagate_crown_batched_with_context` which uses `BatchedBackwardContext`
    /// for direct tensor access.
    // Justification: Deprecated test-only API. The complex domain_data tuple type packs
    // per-domain verification state (input bounds, split history, beta state, base bounds)
    // for batched CROWN. The replacement API uses BatchedBackwardContext struct instead.
    #[allow(clippy::type_complexity)]
    pub(in crate::beta_crown::engine::graph) fn batched_backward_legacy(
        &self,
        graph: &GraphNetwork,
        domain_data: &[(
            &BoundedTensor,                                                 // input_bounds
            &GraphSplitHistory,                                             // history
            Option<&GraphBetaState>,                                        // beta_state
            Option<&std::collections::HashMap<String, Arc<BoundedTensor>>>, // base_bounds
        )],
        objective: &[f32],
        engine: &dyn GemmEngine,
    ) -> Result<Vec<DomainCrownResult>> {
        if domain_data.is_empty() {
            return Ok(Vec::new());
        }

        let n_domains = domain_data.len();
        let plan = graph.dispatch_plan()?;

        // ===== FORWARD PASS: Compute intermediate bounds for each domain =====
        // This is done in parallel but not GPU-batched (IBP is cheap)
        let forward_results: Vec<_> = domain_data
            .par_iter()
            .map(|(input, history, _beta_state, base_bounds)| {
                // Deprecated tuple API carries no delta — full-history seeds.
                self.compute_constrained_forward_bounds(graph, input, history, *base_bounds, None)
            })
            .collect();

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
                Err(e) => {
                    return Err(NyError::InvalidSpec(format!(
                        "Forward pass failed for domain {}: {}",
                        i, e
                    )));
                }
            }
        }

        // Extract beta_states for the internal backward pass
        let beta_states: Vec<Option<&GraphBetaState>> =
            domain_data.iter().map(|(_, _, bs, _)| *bs).collect();

        // No alpha states in deprecated tuple API — use empty
        let alpha_states: Vec<Option<&GraphDomainAlphaState>> = vec![None; n_domains];

        // Delegate to unified backward pass (standard mode: no lA capture or warm-start)
        let result = self.propagate_crown_batched_backward_core(
            graph,
            n_domains,
            plan,
            &bounds_caches,
            &constrained_inputs,
            &beta_states,
            &alpha_states,
            objective,
            engine,
            super::BatchedBackwardMode::Standard,
            None, // mul_binary_alphas: deprecated tuple API, no MulBinary support
        )?;
        Ok(result.results)
    }

    /// Propagate CROWN bounds using BatchedDomains representation.
    ///
    /// This is the high-level API for GPU-batched bound propagation. It takes a
    /// `BatchedDomains` struct (from `BatchedDomains::from_graph_domains()`) and
    /// returns `DomainUpdate` structs that can be applied back to the original domains.
    ///
    /// # Performance
    /// Using BatchedDomains provides:
    /// - Pre-stacked tensors ready for GPU transfer
    /// - Cleaner API than manual tuple construction
    /// - Foundation for future GPU-side domain management
    ///
    /// # Arguments
    /// * `graph` - The network graph
    /// * `domains` - Slice of GraphBabDomain references to process
    /// * `batched` - Pre-constructed BatchedDomains from `from_graph_domains()`
    /// * `objective` - Objective coefficients (same for all domains)
    /// * `engine` - GPU compute engine
    ///
    /// # Returns
    /// Vec of DomainUpdate with new bounds per domain, or error.
    /// Currently unused by the BaB loop but kept for GPU batching follow-ups.
    ///
    /// Note: This stays crate-visible so engine tests can exercise the batched path.
    pub(crate) fn propagate_crown_with_batched_domains(
        &self,
        graph: &GraphNetwork,
        domains: &[&GraphBabDomain],
        batched: &BatchedDomains,
        objective: &[f32],
        engine: &dyn GemmEngine,
    ) -> Result<Vec<DomainUpdate>> {
        if domains.is_empty() || batched.is_empty() {
            return Ok(Vec::new());
        }

        if domains.len() != batched.batch_size() {
            return Err(NyError::InvalidSpec(format!(
                "BatchedDomains size mismatch: domains={}, batch_size={}",
                domains.len(),
                batched.batch_size()
            )));
        }

        let mut inputs: Vec<BoundedTensor> = Vec::with_capacity(domains.len());
        let mut histories: Vec<&GraphSplitHistory> = Vec::with_capacity(domains.len());
        let mut beta_states: Vec<&GraphBetaState> = Vec::with_capacity(domains.len());
        let mut base_bounds: Vec<&std::collections::HashMap<String, Arc<BoundedTensor>>> =
            Vec::with_capacity(domains.len());

        for (idx, domain) in domains.iter().enumerate() {
            let input = batched.input_bounds_at(idx)?;
            inputs.push(input);
            histories.push(&domain.history);
            beta_states.push(&domain.beta_state);
            base_bounds.push(&domain.node_bounds);
        }

        let domain_data: Vec<_> = (0..domains.len())
            .map(|idx| {
                (
                    &inputs[idx],
                    histories[idx],
                    Some(beta_states[idx]),
                    Some(base_bounds[idx]),
                )
            })
            .collect();

        // Run batched CROWN backward pass
        let results = self.batched_backward_legacy(graph, &domain_data, objective, engine)?;

        // Convert results to DomainUpdate format
        let mut new_lower_bounds = Vec::with_capacity(results.len());
        let mut new_upper_bounds = Vec::with_capacity(results.len());

        for (output, _node_cache) in &results {
            // Apply objective to output bounds to get scalar objective bounds
            let (lb, ub) = self.apply_objective_to_output(output, objective)?;
            new_lower_bounds.push(lb);
            new_upper_bounds.push(ub);
            // Note: Layer bounds are handled directly by the BaB loop for now.
            // Future optimization: batch node_cache into layer bounds arrays and use
            // batched.extract_updates_with_layer_bounds() for full layer bound extraction.
        }

        // Create updates using the objective bounds
        // Layer bounds are handled directly by the BaB loop for now
        batched.extract_updates(&new_lower_bounds, &new_upper_bounds)
    }

    /// Apply objective coefficients to output bounds to get scalar bounds.
    ///
    /// Given output bounds and objective vector c, computes:
    /// - lower = sum_i (c_i * output_lower_i if c_i >= 0 else c_i * output_upper_i)
    /// - upper = sum_i (c_i * output_upper_i if c_i >= 0 else c_i * output_lower_i)
    fn apply_objective_to_output(
        &self,
        output: &BoundedTensor,
        objective: &[f32],
    ) -> Result<(f32, f32)> {
        let flat = output.flatten();
        if flat.len() != objective.len() {
            return Err(NyError::shape_mismatch(
                vec![objective.len()],
                vec![flat.len()],
            ));
        }

        let mut lower = 0.0f32;
        let mut upper = 0.0f32;
        for (idx, &c) in objective.iter().enumerate() {
            let l = flat.lower()[[idx]];
            let u = flat.upper()[[idx]];
            if c >= 0.0 {
                lower += c * l;
                upper += c * u;
            } else {
                lower += c * u;
                upper += c * l;
            }
        }
        Ok((lower, upper))
    }
}
