// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-batched domain processing for single-objective graph BaB verification.

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use rayon::prelude::*;
use tracing::debug;

use crate::batched_domain::{BatchedDomainOptions, BatchedDomains};
use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::branching::GraphNeuronConstraint;
use crate::beta_crown::domain::{DomainWithUnstable, GraphBabDomain, GraphCrownContext};
use crate::beta_crown::engine::graph::adaptive_microbatch::MicrobatchRefusalReason;
use crate::GraphNetwork;

use super::super::super::super::domain_results::GraphDomainResult;
use super::super::super::super::tensor_ext::BoundedTensorExt;
use super::super::super::super::BetaCrownVerifier;
use super::super::super::DomainCrownResult;

impl BetaCrownVerifier {
    /// Process multiple graph BaB domains with GPU-batched CROWN computation.
    ///
    /// Implements GPU-batched processing for single-objective verification:
    /// pre-filter verified/violated, find unstable neurons, branch, batch
    /// child CROWN backward passes into L kernel launches (not N×L), then
    /// optionally run per-child beta optimization for unverified shallow children.
    /// Returns results for each domain in the same order as input.
    ///
    /// Part of #3813: `cut_pool` is a read-only view of the current cutting
    /// planes, applied during CROWN backward propagation. Cut generation and
    /// merging remain in the outer BaB loop.
    // Justification: batched single-objective processing needs graph, domains,
    // relu nodes, objective/threshold, engine, and the read-only cut pool as
    // one unit of verification context; adding an ad hoc context wrapper here
    // would only hide the same inputs behind another internal struct.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn process_graph_domains_batched_gpu(
        &self,
        graph: &GraphNetwork,
        domains: &[&GraphBabDomain],
        relu_nodes: &[String],
        objective: &[f32],
        threshold: f32,
        engine: &dyn GemmEngine,
        cut_pool: Option<&GraphCutPool>,
        split_depth: usize,
        retry_refusals: bool,
    ) -> Result<Vec<GraphDomainResult>, MicrobatchRefusalReason> {
        if domains.is_empty() {
            return Ok(Vec::new());
        }

        // Pre-filter: separate already-verified, violation, and to-process domains
        // Use a HashMap for sparse results storage (avoids Clone requirement)
        let mut quick_results: std::collections::HashMap<usize, GraphDomainResult> =
            std::collections::HashMap::new();
        let mut domains_to_process: Vec<(usize, &GraphBabDomain)> = Vec::new();

        for (idx, domain) in domains.iter().enumerate() {
            // Quick verification check
            let already_verified =
                self.config
                    .domain_is_verified(domain.lower_bound, domain.upper_bound, threshold);
            if already_verified {
                quick_results.insert(idx, GraphDomainResult::AlreadyVerified);
                continue;
            }

            // Quick violation check
            let violation =
                self.config
                    .domain_is_violation(domain.lower_bound, domain.upper_bound, threshold);
            if violation {
                quick_results.insert(idx, GraphDomainResult::Violation);
                continue;
            }

            domains_to_process.push((idx, domain));
        }

        if domains_to_process.is_empty() {
            // All domains were quick-filtered — every idx should be in quick_results.
            return Ok((0..domains.len())
                .map(|idx| {
                    quick_results.remove(&idx).unwrap_or_else(|| {
                        tracing::warn!(
                            "process_graph_domains_batched_gpu: missing quick_result for idx {} (#1993)",
                            idx
                        );
                        GraphDomainResult::PropagationFailure
                    })
                })
                .collect());
        }

        // Find unstable neurons for all domains in parallel (cheap operation)
        let unstable_per_domain: Vec<(usize, Vec<(String, usize)>)> = domains_to_process
            .par_iter()
            .map(|(idx, domain)| {
                let unstable = self.find_unstable_graph_neurons(graph, domain, relu_nodes);
                (*idx, unstable)
            })
            .collect();

        // Separate domains with/without unstable neurons
        let mut domains_with_unstable: Vec<DomainWithUnstable<'_>> = Vec::new();

        // O(1) index from domain idx → domain ref, replacing a per-iteration
        // linear `.find()` over `domains_to_process` (was O(D²) for batch size D,
        // up to thousands of domains). `idx` is the unique `.enumerate()` index
        // assigned when `domains_to_process` was built, so each key maps to
        // exactly one domain — identical to the first-match `.find()` semantics.
        let domain_by_idx: std::collections::HashMap<usize, &GraphBabDomain> =
            domains_to_process.iter().map(|(i, d)| (*i, *d)).collect();

        for (idx, unstable) in unstable_per_domain {
            let Some(domain) = domain_by_idx.get(&idx).copied() else {
                tracing::warn!(
                    "process_graph_domains_batched_gpu: missing domain at idx {} while resolving unstable set (#1993)",
                    idx
                );
                quick_results.insert(idx, GraphDomainResult::PropagationFailure);
                continue;
            };

            if unstable.is_empty() {
                // No unstable neurons - compute final bounds
                let context = GraphCrownContext::new(
                    &domain.history,
                    cut_pool, // Part of #3813: apply existing cuts
                    Some(&domain.node_bounds),
                    Some(engine),
                )
                .with_alpha(&domain.alpha_state);
                match self.propagate_crown_with_graph_constraints(
                    graph,
                    domain.input_bounds.as_ref(),
                    &context,
                    None,
                    Some(objective),
                ) {
                    Ok((output, _node_cache)) => {
                        let l = output.lower_scalar();
                        let u = output.upper_scalar();
                        let verified = self.config.domain_is_verified(l, u, threshold);
                        quick_results.insert(
                            idx,
                            GraphDomainResult::NoUnstable {
                                lower: l,
                                upper: u,
                                verified,
                            },
                        );
                    }
                    Err(ref e) if e.is_infeasible_domain() => {
                        // #2926: empty = verified
                        quick_results.insert(idx, GraphDomainResult::AlreadyVerified);
                    }
                    Err(e) => {
                        debug!(error = %e, depth = domain.depth, "Batched NoUnstable CROWN propagation failed — returning PropagationFailure (#1978)");
                        quick_results.insert(idx, GraphDomainResult::PropagationFailure);
                    }
                }
            } else {
                domains_with_unstable.push((idx, domain, unstable));
            }
        }

        if domains_with_unstable.is_empty() {
            return Ok((0..domains.len())
                .map(|idx| {
                    quick_results.remove(&idx).unwrap_or_else(|| {
                        tracing::warn!(
                            "process_graph_domains_batched_gpu: missing result for idx {} after unstable scan (#1993)",
                            idx
                        );
                        GraphDomainResult::PropagationFailure
                    })
                })
                .collect());
        }

        // For domains with unstable neurons, process children in batched manner
        // First, create all child domains (this is relatively cheap)
        //
        // Returns Ok(children_info) on success, Err(idx) on branch selection failure.
        // This makes failures explicit instead of relying on the empty-children
        // catch-all to detect them (#2143).
        let child_creation_results: Vec<_> = domains_with_unstable
            .par_iter()
            .map(|(idx, domain, unstable)| {
                let remaining_depth = self.config.max_depth.saturating_sub(domain.depth);
                let effective_split_depth =
                    split_depth.max(1).min(unstable.len()).min(remaining_depth);
                if effective_split_depth == 0 {
                    tracing::warn!(
                        parent_idx = idx,
                        parent_depth = domain.depth,
                        max_depth = self.config.max_depth,
                        "shared single-objective executor refused a parent with no depth budget"
                    );
                    return Err(*idx);
                }

                if effective_split_depth > 1 {
                    let branches = match self.select_graph_branches(
                        graph,
                        domain,
                        unstable,
                        effective_split_depth,
                    ) {
                        Ok(branches) => branches,
                        Err(error) => {
                            tracing::warn!("select_graph_branches failed for idx {}: {error}", idx);
                            return Err(*idx);
                        }
                    };
                    let children = match domain.with_multi_constraints(
                        graph,
                        &branches,
                        self.config.verify_upper_bound,
                    ) {
                        Ok(children) => children,
                        Err(error) if error.is_infeasible_domain() => Vec::new(),
                        Err(error) => {
                            tracing::warn!(
                                "with_multi_constraints failed for idx {}: {error}",
                                idx
                            );
                            return Err(*idx);
                        }
                    };
                    let children_info = children
                        .into_iter()
                        // The phase bit is legacy metadata and is ignored by
                        // every downstream bound/result path.
                        .map(|child| (*idx, child, true))
                        .collect();
                    return Ok((*idx, children_info));
                }

                let (node_name, neuron_idx, score) = match self
                    .select_graph_branch(graph, domain, unstable)
                {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!("select_graph_branch failed for idx {}: {e} (#1915)", idx);
                        return Err(*idx);
                    }
                };

                // Create active and inactive child domains
                let mut children_info = Vec::with_capacity(2);

                // Active child
                let active_constraint = GraphNeuronConstraint {
                    node_name: node_name.clone(),
                    neuron_idx,
                    is_active: true,
                    score,
                };
                match domain.with_constraint(
                    graph,
                    active_constraint,
                    self.config.verify_upper_bound,
                ) {
                    Ok(Some(child)) => children_info.push((*idx, child, true)),
                    Ok(None) => {}
                    Err(ref e) if e.is_infeasible_domain() => {
                        // #2926: Infeasible constraint = empty child, skip.
                    }
                    Err(e) => {
                        tracing::warn!("with_constraint (active) failed for idx {}: {e}", idx);
                        return Err(*idx);
                    }
                }

                // Inactive child
                let inactive_constraint = GraphNeuronConstraint {
                    node_name,
                    neuron_idx,
                    is_active: false,
                    score,
                };
                match domain.with_constraint(
                    graph,
                    inactive_constraint,
                    self.config.verify_upper_bound,
                ) {
                    Ok(Some(child)) => children_info.push((*idx, child, false)),
                    Ok(None) => {}
                    Err(ref e) if e.is_infeasible_domain() => {
                        // #2926: Infeasible constraint = empty child, skip.
                    }
                    Err(e) => {
                        tracing::warn!("with_constraint (inactive) failed for idx {}: {e}", idx);
                        return Err(*idx);
                    }
                }

                Ok((*idx, children_info))
            })
            .collect();

        // Collect all children that need CROWN bounds computation.
        // Handle branch selection failures explicitly (#2143) instead of letting
        // empty children flow through to the catch-all.
        let mut all_children: Vec<(usize, GraphBabDomain, bool)> = Vec::new();
        let mut parent_domain_lookup: std::collections::HashMap<usize, &GraphBabDomain> =
            std::collections::HashMap::new();
        let mut branch_selection_failures: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        // O(1) index from parent idx → parent domain ref, replacing a per-iteration
        // linear `.find()` over `domains_with_unstable` (was O(D²) for batch size D,
        // up to thousands of domains). Each `idx` is the unique `.enumerate()` index
        // carried from `domains_to_process`, so a key maps to exactly one domain —
        // identical to the first-match `.find()` semantics.
        let domain_by_idx: std::collections::HashMap<usize, &GraphBabDomain> =
            domains_with_unstable
                .iter()
                .map(|(i, d, _)| (*i, *d))
                .collect();

        for result in &child_creation_results {
            let (parent_idx, children_info) = match result {
                Ok(v) => v,
                Err(failed_idx) => {
                    // Branch selection failed — record explicitly (#2143).
                    branch_selection_failures.insert(*failed_idx);
                    quick_results.insert(*failed_idx, GraphDomainResult::PropagationFailure);
                    continue;
                }
            };
            let Some(parent_domain) = domain_by_idx.get(parent_idx).copied() else {
                tracing::warn!(
                    "process_graph_domains_batched_gpu: missing parent domain for idx {} (#1993)",
                    parent_idx
                );
                quick_results.insert(*parent_idx, GraphDomainResult::PropagationFailure);
                continue;
            };
            parent_domain_lookup.insert(*parent_idx, parent_domain);

            for (_, child, is_active) in children_info {
                all_children.push((*parent_idx, child.clone(), *is_active));
            }
        }

        // Now compute CROWN bounds for all children with TRUE tensor-level GPU batching.
        let child_domain_refs: Vec<&GraphBabDomain> =
            all_children.iter().map(|(_, child, _)| child).collect();

        // Create BatchedDomains for GPU transfer optimization
        let batched_options = BatchedDomainOptions {
            enable_interm_transfer: self.config.enable_interm_transfer,
        };
        let batched = match BatchedDomains::from_graph_domains_with_options(
            &child_domain_refs,
            relu_nodes,
            batched_options,
        ) {
            Ok(b) => b,
            Err(e) => {
                if retry_refusals {
                    if let Some(reason) = MicrobatchRefusalReason::from_error(&e) {
                        return Err(reason);
                    }
                }
                tracing::warn!(
                    "BatchedDomains creation failed ({}), returning PropagationFailure (#1993)",
                    e
                );
                return Ok((0..domains.len())
                    .map(|idx| {
                        quick_results
                            .remove(&idx)
                            .unwrap_or(GraphDomainResult::PropagationFailure)
                    })
                    .collect());
            }
        };

        let child_bounds: Vec<Option<DomainCrownResult>> = match self
            .propagate_crown_with_batched_domains_full(
                graph,
                &child_domain_refs,
                &batched,
                objective,
                engine,
            ) {
            Ok(results) => results,
            Err(e) => {
                if retry_refusals {
                    if let Some(reason) = MicrobatchRefusalReason::from_error(&e) {
                        return Err(reason);
                    }
                }
                // If batched processing fails, fall back to sequential with beta
                // optimization. Part of #1484.
                tracing::warn!(
                    "Batched CROWN failed ({}), falling back to sequential with beta optimization",
                    e
                );
                all_children
                    .iter_mut()
                    .map(|(parent_idx, child, _is_active)| {
                        let Some(parent) = parent_domain_lookup.get(parent_idx) else {
                            tracing::warn!(
                                "process_graph_domains_batched_gpu: missing parent lookup for child of idx {}",
                                parent_idx
                            );
                            return None;
                        };
                        match self.evaluate_graph_child_bounds(
                            graph,
                            child,
                            &parent.node_bounds,
                            objective,
                            cut_pool, // Part of #3813: apply existing cuts
                            Some(engine),
                        ) {
                            Ok(true) => {
                                // Guard: NaN/Inf in BaB child scalar bounds →
                                // drop child (unsound to propagate).
                                if !child.lower_bound.is_finite()
                                    || !child.upper_bound.is_finite()
                                {
                                    tracing::warn!(
                                        parent_idx,
                                        lower = child.lower_bound,
                                        upper = child.upper_bound,
                                        "Multi-objective batched fallback child dropped: non-finite bounds"
                                    );
                                    return None;
                                }
                                let output = BoundedTensor::new(
                                    ndarray::arr1(&[child.lower_bound]).into_dyn(),
                                    ndarray::arr1(&[child.upper_bound]).into_dyn(),
                                );
                                match output {
                                    Ok(bt) => {
                                        // #cone-delta increment 2: Arc-clone the
                                        // domain's map (no tensor copies).
                                        Some((bt, child.node_bounds.clone()))
                                    }
                                    Err(e) => {
                                        tracing::warn!("Batched single child CROWN failed: {e}");
                                        None
                                    }
                                }
                            }
                            _ => None,
                        }
                    })
                    .collect()
            }
        };

        // Build results from child bounds
        let mut children_by_parent: std::collections::HashMap<usize, Vec<(GraphBabDomain, bool)>> =
            std::collections::HashMap::new();
        // #1861: Track parents that had at least one child propagation failure.
        let mut parents_with_failure: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        // Part of #1484: check if beta optimization is configured
        let should_optimize_any = self.config.beta_iterations > 0;

        for ((parent_idx, mut child, _is_active), bounds_result) in
            all_children.into_iter().zip(child_bounds)
        {
            if let Some((output, node_cache)) = bounds_result {
                let l = output.lower_scalar();
                let u = output.upper_scalar();
                // #cone-delta increment 2: the result map is already Arc-shared
                // (out-of-cone entries alias the parent's) — install by move.
                child.node_bounds = node_cache;
                // #cone-delta: post-bounding replacement — delta restarts empty.
                child.delta_pre_nodes.clear();
                child.lower_bound = l;
                child.upper_bound = u;
                child.priority = match self.config.domain_priority(l, u) {
                    Ok(p) => p,
                    Err(e) => {
                        // NaN in domain bounds → treat as propagation failure (#2982)
                        tracing::warn!("Batched single domain_priority failed (NaN bounds): {e}");
                        parents_with_failure.insert(parent_idx);
                        continue;
                    }
                };

                let mut verified = self.config.domain_is_verified(l, u, threshold);

                // Part of #1484: Post-batched beta optimization refinement.
                if !verified && should_optimize_any && child.depth <= self.config.beta_max_depth {
                    let parent_bounds = child.node_bounds.clone();
                    match self.evaluate_graph_child_bounds(
                        graph,
                        &mut child,
                        &parent_bounds,
                        objective,
                        cut_pool, // Part of #3813: apply existing cuts
                        Some(engine),
                    ) {
                        Ok(true) => {
                            verified = self.config.domain_is_verified(
                                child.lower_bound,
                                child.upper_bound,
                                threshold,
                            );
                        }
                        Ok(false) => {} // Bounds trivially empty — keeps pre-refinement bounds
                        Err(ref e) if e.is_infeasible_domain() => {
                            verified = true;
                        } // #2926
                        Err(e) => {
                            debug!(error = %e, depth = child.depth, "Beta refinement failed for child (#1978)");
                        }
                    }
                }

                children_by_parent
                    .entry(parent_idx)
                    .or_default()
                    .push((child, verified));
            } else {
                // #1861: child bounds computation failed — sub-region unexplored.
                parents_with_failure.insert(parent_idx);
            }
        }

        // Assemble final results
        for (parent_idx, _, _) in &domains_with_unstable {
            // Branch selection failures are already in quick_results (#2143).
            if branch_selection_failures.contains(parent_idx) {
                continue;
            }
            if parents_with_failure.contains(parent_idx) {
                // #1861: At least one child failed — signal propagation failure.
                if let Some(children) = children_by_parent.remove(parent_idx) {
                    if !children.is_empty() {
                        quick_results.insert(*parent_idx, GraphDomainResult::Children(children));
                    }
                }
                // Override with PropagationFailure — soundness over performance (#1861).
                quick_results.insert(*parent_idx, GraphDomainResult::PropagationFailure);
            } else if let Some(children) = children_by_parent.remove(parent_idx) {
                quick_results.insert(*parent_idx, GraphDomainResult::Children(children));
            } else {
                // Both children infeasible (with_constraint returned None for both).
                // This is a legitimate outcome, not an internal failure (#2143).
                tracing::debug!(
                    "process_graph_domains_batched_gpu: both children infeasible for parent idx {} (#2143)",
                    parent_idx
                );
                quick_results.insert(*parent_idx, GraphDomainResult::PropagationFailure);
            }
        }

        // Return results in order
        Ok((0..domains.len())
            .map(|idx| {
                quick_results.remove(&idx).unwrap_or_else(|| {
                    tracing::warn!(
                        "process_graph_domains_batched_gpu: missing final result for idx {} (#1993)",
                        idx
                    );
                    GraphDomainResult::PropagationFailure
                })
            })
            .collect())
    }
}
