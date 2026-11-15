// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Borrow one intermediate target from the original graph's cached dispatch plan.
//!
//! The graph output, topology and proof eligibility are never rewritten. A future
//! intermediate collector can prepare these identities once and reuse them over
//! domain batches, keeping each domain's own bound map. This primitive does not
//! collect or certify those maps, choose demanded rows, tighten intervals, or
//! enable finite traversal: the existing shared SoA deadline refusal still applies.

use super::*;

/// Target identity tied to an immutable borrow of its original graph and plan.
#[allow(dead_code)] // Collector integration and finite traversal remain unqualified.
pub(crate) struct PreparedSoaTarget<'a> {
    graph: &'a GraphNetwork,
    plan: &'a CrownDispatchPlan,
    node_idx: usize,
}

#[allow(dead_code)]
impl<'a> PreparedSoaTarget<'a> {
    pub(crate) fn new(graph: &'a GraphNetwork, node_name: &str) -> Result<Self> {
        // Check the entire original graph, including nodes outside this target's
        // ancestry. A target walk cannot bypass point-only proof ineligibility.
        graph.ensure_proof_eligible()?;
        let plan = graph.dispatch_plan()?;
        let node_idx = plan.index_of(node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!("SoA intermediate target '{node_name}' is unknown"))
        })?;
        if plan.is_network_input(node_idx) {
            return Err(NyError::InvalidSpec(
                "SoA intermediate target must be a graph node, not NETWORK_INPUT".into(),
            ));
        }
        Ok(Self {
            graph,
            plan,
            node_idx,
        })
    }

    pub(crate) fn name(&self) -> &str {
        self.plan.name_of(self.node_idx)
    }

    /// Return relations in each supplied domain's own NETWORK_INPUT coordinates.
    ///
    /// The caller owns the certified-map/input correspondence and seed meaning.
    /// This validates layout only. A finite deadline always takes the existing
    /// typed-expiry/clean-decline path before any domain storage is inspected.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_backward(
        &self,
        n_domains: usize,
        bounds_caches: &[&HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        seed: &LinearBounds,
        engine: &dyn GemmEngine,
        deadline: Option<std::time::Instant>,
        mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    ) -> Result<SoaBackwardOutcome> {
        if deadline.is_none() {
            if n_domains == 0 {
                if bounds_caches.is_empty()
                    && constrained_inputs.is_empty()
                    && alpha_states.is_empty()
                {
                    return Ok(None);
                }
                return Err(NyError::InvalidSpec(
                    "SoA intermediate target: nonempty arrays for empty batch".into(),
                ));
            }
            if bounds_caches.len() != n_domains
                || constrained_inputs.len() != n_domains
                || alpha_states.len() != n_domains
            {
                return Err(NyError::InvalidSpec(
                    "SoA intermediate target: parallel array length mismatch".into(),
                ));
            }
            let first = bounds_caches[0].get(self.name()).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "SoA intermediate target '{}' has no domain0 bound",
                    self.name()
                ))
            })?;
            if seed.num_outputs() == 0
                || seed.num_inputs() == 0
                || seed.num_inputs() != first.len()
                || constrained_inputs[0].is_empty()
            {
                return Err(NyError::InvalidSpec(
                    "SoA intermediate target: seed or input shape mismatch".into(),
                ));
            }
            for (domain, (cache, input)) in bounds_caches.iter().zip(constrained_inputs).enumerate()
            {
                let bounds = cache.get(self.name()).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "SoA intermediate target '{}' has no domain{domain} bound",
                        self.name()
                    ))
                })?;
                if bounds.shape() != first.shape() || input.shape() != constrained_inputs[0].shape()
                {
                    return Err(NyError::InvalidSpec(
                        "SoA intermediate target: cross-domain shape mismatch".into(),
                    ));
                }
            }
        }
        try_backward_soa_seeded(
            self.graph,
            self.plan,
            self.node_idx,
            n_domains,
            bounds_caches,
            constrained_inputs,
            alpha_states,
            seed,
            engine,
            deadline,
            mul_binary_alphas,
        )
    }
}

#[cfg(test)]
#[path = "target_tests.rs"]
mod tests;
