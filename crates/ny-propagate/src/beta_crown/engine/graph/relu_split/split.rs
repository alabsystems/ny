// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential domain-processing helpers for ReLU-split branch-and-bound.

use ny_core::{GemmEngine, Result};
use tracing::debug;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::branching::{BranchingHeuristic, GraphNeuronConstraint};
use crate::beta_crown::domain::{GraphBabDomain, GraphCrownContext};
use crate::GraphNetwork;

use super::super::super::domain_results::GraphDomainResult;
use super::super::super::tensor_ext::BoundedTensorExt;
use super::super::super::BetaCrownVerifier;
use super::child_eval::ChildOutcome;

impl BetaCrownVerifier {
    /// Process domains sequentially with cut support and full SPSA β optimization.
    ///
    /// `split_depth` controls multi-depth ReLU splitting (#2767): when > 1,
    /// selects top-k unstable neurons and creates 2^k child domains per parent.
    /// Reference: alpha-beta-CROWN depth mode (`domain_updater.py:63-304`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn process_sequential_domains(
        &self,
        graph: &GraphNetwork,
        domains: &[GraphBabDomain],
        relu_nodes: &[String],
        objective: &[f32],
        threshold: f32,
        cut_pool: &mut GraphCutPool,
        engine: Option<&dyn GemmEngine>,
        split_depth: usize,
    ) -> Result<Vec<GraphDomainResult>> {
        let mut seq_results = Vec::with_capacity(domains.len());
        for domain in domains {
            if matches!(
                self.config.branching_heuristic,
                BranchingHeuristic::GenBaB(_)
            ) {
                seq_results.push(self.process_graph_domain_parallel(
                    graph,
                    domain,
                    relu_nodes,
                    objective,
                    threshold,
                    engine,
                    split_depth,
                ));
                continue;
            }

            let unstable = self.find_unstable_graph_neurons(graph, domain, relu_nodes);
            if unstable.is_empty() {
                seq_results.push(self.process_no_unstable_domain(
                    graph, domain, cut_pool, objective, threshold, engine,
                ));
            } else if split_depth > 1 && unstable.len() > 1 {
                seq_results.extend(self.process_multi_depth_split(
                    graph,
                    domain,
                    &unstable,
                    objective,
                    threshold,
                    cut_pool,
                    engine,
                    split_depth,
                )?);
            } else {
                seq_results.extend(self.process_single_depth_split(
                    graph, domain, &unstable, objective, threshold, cut_pool, engine,
                )?);
            }
        }
        Ok(seq_results)
    }

    /// Process a domain using the NoUnstable fallback: CROWN with β, α, and cuts.
    pub(super) fn process_no_unstable_domain(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        cut_pool: &GraphCutPool,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> GraphDomainResult {
        let context = GraphCrownContext::new(
            &domain.history,
            Some(cut_pool),
            Some(&domain.node_bounds),
            engine,
        )
        .with_alpha(&domain.alpha_state);
        match self.propagate_crown_with_graph_beta(
            graph,
            domain.input_bounds.as_ref(),
            &context,
            &domain.beta_state,
            Some(objective),
        ) {
            Ok((output, _node_cache)) => {
                let l = output.lower_scalar();
                let u = output.upper_scalar();
                let verified = self.config.domain_is_verified(l, u, threshold);
                GraphDomainResult::NoUnstable {
                    lower: l,
                    upper: u,
                    verified,
                }
            }
            Err(ref e) if e.is_infeasible_domain() => {
                debug!(error = %e, depth = domain.depth, "NoUnstable domain infeasible (empty)");
                GraphDomainResult::AlreadyVerified
            }
            Err(e) => {
                debug!(error = %e, depth = domain.depth, "NoUnstable CROWN propagation failed — returning PropagationFailure (#1978)");
                GraphDomainResult::PropagationFailure
            }
        }
    }

    /// Process a domain with multi-depth splitting: select top-k neurons, create 2^k children.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn process_multi_depth_split(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
        objective: &[f32],
        threshold: f32,
        cut_pool: &GraphCutPool,
        engine: Option<&dyn GemmEngine>,
        split_depth: usize,
    ) -> Result<Vec<GraphDomainResult>> {
        let k = super::super::cap_relu_split_depth_for_parent(
            split_depth,
            unstable.len(),
            domain.depth,
            self.config.max_depth,
        );
        if k == 0 {
            return Ok(vec![GraphDomainResult::PropagationFailure]);
        }
        let branches = match self.select_graph_branches(graph, domain, unstable, k) {
            Ok(branches) => branches,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    depth = domain.depth,
                    k,
                    "select_graph_branches failed in multi-depth split"
                );
                return Ok(vec![GraphDomainResult::PropagationFailure]);
            }
        };
        let child_domains =
            domain.with_multi_constraints(graph, &branches, self.config.verify_upper_bound)?;
        debug!(
            depth = domain.depth,
            split_k = branches.len(),
            num_children = child_domains.len(),
            "Multi-depth ReLU split (sequential)"
        );

        let mut children: Vec<(GraphBabDomain, bool)> = Vec::with_capacity(child_domains.len());
        let mut any_child_failed = false;
        for child in child_domains {
            match self.evaluate_existing_child(
                graph,
                child,
                &domain.node_bounds,
                objective,
                threshold,
                Some(cut_pool),
                engine,
            ) {
                ChildOutcome::Evaluated(child, verified) => children.push((*child, verified)),
                ChildOutcome::Infeasible => {
                    debug!("Multi-depth child infeasible (empty), pruning");
                }
                ChildOutcome::Failed => any_child_failed = true,
                ChildOutcome::NoChild => {}
            }
        }

        let mut results = vec![GraphDomainResult::Children(children)];
        if any_child_failed {
            results.push(GraphDomainResult::PropagationFailure);
        }
        Ok(results)
    }

    /// Process a domain with single-depth splitting: one neuron -> active + inactive.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn process_single_depth_split(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
        objective: &[f32],
        threshold: f32,
        cut_pool: &GraphCutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Vec<GraphDomainResult>> {
        if domain.depth >= self.config.max_depth {
            return Ok(vec![GraphDomainResult::PropagationFailure]);
        }
        let (node_name, neuron_idx, score) = match self
            .select_graph_branch_or_propagation_failure_in_relu_split(graph, domain, unstable)
        {
            Ok(selection) => selection,
            Err(result) => return Ok(vec![result]),
        };

        let mut children: Vec<(GraphBabDomain, bool)> = Vec::with_capacity(2);
        let mut any_child_failed = false;

        for is_active in [true, false] {
            let constraint = GraphNeuronConstraint {
                node_name: node_name.clone(),
                neuron_idx,
                is_active,
                score,
            };
            match self.evaluate_and_classify_child(
                graph,
                domain,
                constraint,
                objective,
                threshold,
                Some(cut_pool),
                engine,
            )? {
                ChildOutcome::Evaluated(child, verified) => children.push((*child, verified)),
                ChildOutcome::Infeasible => {
                    let child_kind = if is_active { "Active" } else { "Inactive" };
                    debug!("{child_kind} child infeasible (empty domain), pruning");
                }
                ChildOutcome::Failed => any_child_failed = true,
                ChildOutcome::NoChild => {}
            }
        }

        let mut results = vec![GraphDomainResult::Children(children)];
        if any_child_failed {
            results.push(GraphDomainResult::PropagationFailure);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sequential_fallback_caps_parent_at_max_depth_minus_one() {
        assert_eq!(
            super::super::super::cap_relu_split_depth_for_parent(4, 8, 3, 4),
            1,
            "sequential fallback must reduce a depth-four request to the last legal level"
        );
        assert_eq!(
            super::super::super::cap_relu_split_depth_for_parent(4, 8, 4, 4),
            0,
            "a max-depth parent must not expand"
        );
    }
}
