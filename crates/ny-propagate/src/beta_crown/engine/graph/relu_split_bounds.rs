// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ReLU-splitting BaB with pre-computed initial bounds.
//!
//! Variant of [`super::relu_split`] that accepts pre-computed IBP/α-CROWN bounds
//! via [`GraphPrecomputedBounds`], avoiding redundant forward passes when verifying
//! multiple constraints on the same graph and input. Useful for multi-property
//! verification where initial bounds are shared across objectives.
//!
//! Entry point: `BetaCrownVerifier::verify_graph_relu_split_with_bounds`.

use std::collections::BinaryHeap;
use std::time::Instant;

use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};
use tracing::{debug, info, instrument, warn};

use crate::beta_crown::branching::{BranchingHeuristic, GraphNeuronConstraint};
use crate::beta_crown::conflict_clauses_graph::GraphClauseStore;
use crate::beta_crown::domain::{GraphBabDomain, GraphCrownContext, GraphPrecomputedBounds};
use crate::beta_crown::engine::graph::shared::setup::{
    build_graph_bab_setup, build_graph_cut_pool, build_root_alpha_state,
};
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::network::SpecCrownRequest;
use crate::GraphNetwork;

use super::super::domain_results::GraphDomainResult;
use super::super::tensor_ext::BoundedTensorExt;
use super::super::BetaCrownVerifier;
use super::objectives::objective_bounds;
use super::relu_split::queue_budget::{enforce_graph_queue_budget, GraphBabQueueBudget};

/// The ledger lane name of the precomputed-bounds ReLU-split driver.
const RELU_SPLIT_BOUNDS_LANE: &str = "graph_relu_split_with_bounds";

fn drop_non_finite_domain_in_relu_split_bounds(
    domain: &GraphBabDomain,
    unresolved_due_to_propagation_failure: &mut bool,
) -> bool {
    // Match the sibling ReLU-split guards in `relu_split/bab_loop.rs` and
    // `gpu_bab/prefilter.rs`: NaN/Inf bounds mean upstream propagation
    // failed, so this domain must not flow into verified/violation checks.
    if !domain.lower_bound.is_finite() || !domain.upper_bound.is_finite() {
        warn!(
            depth = domain.depth,
            lower = domain.lower_bound,
            upper = domain.upper_bound,
            "relu_split_with_bounds: domain dropped — non-finite bounds"
        );
        *unresolved_due_to_propagation_failure = true;
        return true;
    }

    false
}

#[cfg(test)]
pub(crate) fn test_non_finite_domain_result_in_relu_split_bounds(
    domain: &GraphBabDomain,
) -> BetaCrownResult {
    let mut unresolved_due_to_propagation_failure = false;
    let dropped = drop_non_finite_domain_in_relu_split_bounds(
        domain,
        &mut unresolved_due_to_propagation_failure,
    );
    assert!(dropped, "test hook expects a non-finite domain");
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    lifecycle.domains_explored = 1;
    lifecycle.max_depth_reached = domain.depth;
    lifecycle.unresolved_due_to_propagation_failure = unresolved_due_to_propagation_failure;
    lifecycle.build_final_result()
}

impl BetaCrownVerifier {
    /// Verify a graph using ReLU splitting with pre-computed initial bounds.
    ///
    /// Use this when verifying multiple constraints on the same graph/input.
    /// Pre-compute bounds once with `compute_initial_graph_bounds`, then
    /// call this method for each constraint.
    ///
    /// Arguments:
    /// - `precomputed_bounds`: Pre-computed node and output bounds
    #[instrument(skip(self, graph, input, objective, precomputed_bounds), fields(threshold, input_shape = ?input.shape(), num_nodes = graph.nodes.len()))]
    pub fn verify_graph_relu_split_with_bounds(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        precomputed_bounds: &GraphPrecomputedBounds<'_>,
    ) -> Result<BetaCrownResult> {
        self.verify_graph_relu_split_with_bounds_impl(
            graph,
            input,
            objective,
            threshold,
            precomputed_bounds,
            self.engine(),
            None,
        )
    }

    /// Verify GraphNetwork with ReLU splitting using pre-computed bounds, with optional GPU acceleration.
    ///
    /// `deadline`: If `Some`, the BaB engine derives its phase budgets from
    /// remaining wall-clock time instead of `self.config.timeout` (#4321).
    pub fn verify_graph_relu_split_with_bounds_with_engine(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        precomputed_bounds: &GraphPrecomputedBounds<'_>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        let engine = self.resolve_engine(engine);
        self.verify_graph_relu_split_with_bounds_impl(
            graph,
            input,
            objective,
            threshold,
            precomputed_bounds,
            engine,
            deadline,
        )
    }

    /// #graph-bab-cover-ledger: like the direct ReLU-split driver, the search
    /// runs under an armed completeness ledger (split-literal paths below
    /// `input`); a `Verified` it returns is kept only if the ledger's
    /// certified leaves cover the root (see `graph::cover_ledger`). This is the
    /// lane the serial multi-clause disjunctive loop hands one clause at a
    /// time; the instance door aggregates the clauses.
    pub(crate) fn verify_graph_relu_split_with_bounds_impl(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        precomputed_bounds: &GraphPrecomputedBounds<'_>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        graph.ensure_proof_eligible()?;
        let scope = self.graph_cover.arm_paths(
            RELU_SPLIT_BOUNDS_LANE,
            graph,
            input,
            objective,
            threshold,
            self.config.verify_upper_bound,
        );
        let result = self.verify_graph_relu_split_with_bounds_unledgered(
            graph,
            input,
            objective,
            threshold,
            precomputed_bounds,
            engine,
            deadline,
        )?;
        Ok(self.settle_graph_cover(RELU_SPLIT_BOUNDS_LANE, scope, result, deadline))
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_graph_relu_split_with_bounds_unledgered(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        precomputed_bounds: &GraphPrecomputedBounds<'_>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        // This path accepts caller-supplied root bounds and can return a verdict
        // before the ordinary graph bootstrap.  Validate at the shared ingress
        // so neither public wrapper can bypass proof-authority quarantine.
        self.config.validate()?;
        let graph = self.configured_graph_for_crown(graph);
        let graph = &graph;
        let now = Instant::now();
        let mut lifecycle = GraphBabLifecycle::new(now);
        // #budget-abandonment FIX C, this lane: the depth cap drops domains
        // here too, so the counter must be armed or `dropped=0` would be read
        // as "the cap is irrelevant" when it is only unobserved.
        let _depth_drop = super::relu_split::depth_drop::DepthDropScope::new(self.config.max_depth);
        // Use pre-computed bounds (clone to owned HashMap)
        let initial_node_bounds: std::collections::HashMap<String, BoundedTensor> =
            precomputed_bounds.node_bounds.clone();

        // Use pre-computed output bounds - apply objective to get initial bounds
        let (projected_lower, projected_upper) =
            objective_bounds(precomputed_bounds.output_bounds, objective)?;
        // #disjunctive-shared-bootstrap: when the caller kept the bootstrap's
        // optimised root alpha, run ONE spec-guided CROWN pass for this
        // objective with those slopes (the DomainList lane's root, gpu_bab/
        // init.rs) and INTERSECT it with the projection. Both are certified
        // enclosures of the same scalar over the same box, so their
        // intersection is one too; the projection alone is the historical
        // lane. Any failure of the extra pass -- expiry, empty, non-finite,
        // a reversed pair -- keeps the projection, so this can only tighten.
        let (root_lower, root_upper, root_src) = match precomputed_bounds.root_alpha_state {
            Some(alpha_state) => {
                let spec_pass = Array2::from_shape_vec((1, objective.len()), objective.to_vec())
                    .map_err(|error| {
                        ny_core::NyError::InternalError(format!(
                            "objective row matrix for the shared-alpha root pass: {error}"
                        ))
                    })
                    .and_then(|spec_matrix| {
                        SpecCrownRequest::new(graph, input, &spec_matrix, engine)
                            .node_bounds(precomputed_bounds.node_bounds)
                            .alpha_state_opt(Some(alpha_state))
                            .deadline_opt(deadline)
                            .truncate_after_opt(self.config.crown_backward_layers)
                            .run()
                    });
                match spec_pass {
                    Ok(output) if output.len() == 1 => {
                        let spec_lower = output.lower()[[0]];
                        let spec_upper = output.upper()[[0]];
                        let lower = spec_lower.max(projected_lower);
                        let upper = spec_upper.min(projected_upper);
                        if spec_lower.is_finite() && spec_upper.is_finite() && lower <= upper {
                            (lower, upper, "spec")
                        } else {
                            (projected_lower, projected_upper, "proj-nonfinite")
                        }
                    }
                    Ok(_) => (projected_lower, projected_upper, "proj-shape"),
                    Err(error) => {
                        debug!(%error, "shared-alpha spec root pass declined; using projection");
                        (projected_lower, projected_upper, "proj-err")
                    }
                }
            }
            None => (projected_lower, projected_upper, "proj"),
        };

        // #budget-abandonment FIX A does NOT apply on this lane -- the root
        // bounds arrive ALREADY COMPUTED from the caller, so there is no
        // warmup phase here to expire and nothing to fall back on. The line is
        // emitted anyway, naming the lane, because an ABSENT line is
        // indistinguishable from "the fallback was armed and not needed", and
        // this is the lane the multi-constraint conjunctive rows take.
        // `root_src=` names where the root came from: `proj` is the historical
        // projection of the caller's output box, `spec` the shared-alpha
        // spec-guided pass intersected with it (#disjunctive-shared-bootstrap).
        eprintln!(
            "[warmup-fallback] armed={} fired=0 tier=none lane=precomputed \
cap_s=0.0 elapsed_s=0.0 remaining_s={:.1} root=[{root_lower},{root_upper}] thr={threshold} \
root_src={root_src}",
            u8::from(self.config.initial_bounds_warmup_fallback),
            deadline.map_or(0.0, |dl| dl.saturating_duration_since(now).as_secs_f64()),
        );

        info!(
            "Graph β-CROWN (ReLU split, pre-computed bounds) initial objective: [{}, {}], threshold: {}, verify_upper={}",
            root_lower, root_upper, threshold, self.config.verify_upper_bound
        );

        // Quick verification/violation check.
        let root_status = if self
            .config
            .domain_is_verified(root_lower, root_upper, threshold)
        {
            Some((BabVerificationStatus::Verified, 1))
        } else if self
            .config
            .domain_is_violation(root_lower, root_upper, threshold)
        {
            Some((BabVerificationStatus::potential_violation(), 0))
        } else {
            None
        };
        if let Some((status, root_verified_count)) = root_status {
            if matches!(status, BabVerificationStatus::Verified) {
                self.cover_note_bound(
                    objective,
                    threshold,
                    &crate::beta_crown::branching::GraphSplitHistory::new(),
                    None,
                    root_lower,
                    root_upper,
                );
            }
            lifecycle.domains_explored = 1;
            lifecycle.domains_verified = root_verified_count;
            // Widen repair: a non-finite root bound (±Inf, e.g. an unbounded
            // output via a degenerate BatchNorm channel) is sound — it just means
            // the domain status is reported with conservative bounds rather than
            // aborting graph beta-CROWN at strict construction.
            return Ok(lifecycle.build_result_with_bounds(
                status,
                BoundedTensor::new_repaired(
                    Array1::from_vec(vec![root_lower]).into_dyn(),
                    Array1::from_vec(vec![root_upper]).into_dyn(),
                    RepairStrategy::Widen,
                )?,
            ));
        }

        // Shared setup: Arc-wrapped bounds and sorted ReLU nodes (#1860 Packet B)
        let setup = build_graph_bab_setup(graph, &initial_node_bounds);

        // Create root domain and initialize alpha state from initial bounds (#1841).
        let mut root_domain = GraphBabDomain::root(
            initial_node_bounds,
            root_lower,
            root_upper,
            input,
            self.config.verify_upper_bound,
        )?;
        root_domain.alpha_state = build_root_alpha_state(
            graph,
            input,
            &root_domain.history,
            &setup.initial_node_bounds_arc,
            // Historically `None` (no root alpha on this lane); the shared
            // bootstrap supplies its optimised slopes here so children
            // warm-start exactly as the per-objective lane's do.
            precomputed_bounds.root_alpha_state,
            self.config.beta_iterations > 0,
        );
        if self.config.enable_clip_interm_domain {
            self.complete_clip_root_bounds_cache.store_finalized(
                graph,
                input,
                &setup.initial_node_bounds_arc,
            );
        }

        // Branch-and-bound queue. The byte budget is shared with the ordinary
        // ReLU-split heap; zero preserves the historical unlimited route.
        let queue_budget = GraphBabQueueBudget::from_config(&self.config);
        let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
        queue.push(root_domain);

        info!("Found {} ReLU nodes for branching", setup.relu_nodes.len());

        // #genbab-precomputed-lane arming observable. Printed unconditionally
        // and BEFORE the loop, so the three states are distinguishable without
        // inference: `armed=0` is dark, `armed=1 genbab=0` is armed on a
        // benchmark whose heuristic is not GenBaB (inert by design), and
        // `armed=1 genbab=1` is the route actually available. What it DID is
        // then read off `[bab-stop]`: a lane that used to return
        // `explored=1 deepest=0 unresolved=no_branch` should now split.
        let genbab_route = self.config.genbab_on_precomputed_lane
            && matches!(
                self.config.branching_heuristic,
                BranchingHeuristic::GenBaB(_)
            );
        eprintln!(
            "[genbab-lane] armed={} genbab={} route={} relu_nodes={}",
            u8::from(self.config.genbab_on_precomputed_lane),
            u8::from(matches!(
                self.config.branching_heuristic,
                BranchingHeuristic::GenBaB(_)
            )),
            if genbab_route { "genbab" } else { "relu_only" },
            setup.relu_nodes.len(),
        );

        // Shared cut pool initialization (#1860 Packet B)
        let mut cut_pool = build_graph_cut_pool(
            graph,
            &setup.initial_node_bounds_arc,
            &setup.relu_nodes,
            &self.config,
        )?;

        // Conflict-clause learning, graph port (win-plan arc C, v2): per-run
        // store, gated NY_BAB_CLAUSE_LEARN=1 (default OFF => disabled store =>
        // byte-identical loop). Scope of THIS store: one graph, one root input
        // box, one objective vector, one threshold, one sense — the
        // region-inclusion argument holds for every recorded clause BY
        // CONSTRUCTION; the store's purity guard fails closed for any history
        // carrying GenBaB/norm splits. See `conflict_clauses_graph`.
        let mut clause_store = GraphClauseStore::from_env();
        if clause_store.is_enabled() {
            info!("Graph BaB conflict-clause learning enabled (NY_BAB_CLAUSE_LEARN=1)");
        }

        // Lambda optimization state — read from config (#2761)
        let mut lambda_opt_iter = 0usize;
        let lambda_opt_interval = self.config.lambda_opt_interval.max(1);
        let lambda_lr = self.config.lambda_lr;
        let lambda_beta1 = self.config.adaptive_config.beta1;
        let lambda_beta2 = self.config.adaptive_config.beta2;
        let lambda_epsilon = self.config.adaptive_config.epsilon;

        // BaB timeout with post-BaB PGD reservation (#4095).
        // When a wall-clock deadline is provided (#4321), derive the effective
        // timeout from remaining time instead of the configured timeout.
        let pgd_frac = self
            .config
            .phase_budget
            .post_bab_pgd_fraction
            .clamp(0.0, 0.5);
        let bab_timeout = match deadline {
            // An explicit deadline is already the caller ledger's BaB slice.
            Some(dl) => dl.saturating_duration_since(now),
            None => self.config.timeout.mul_f32(1.0 - pgd_frac),
        };
        let _complete_clip_deadline = self.complete_clip_deadline_overrides.scoped(Some(
            GraphBabLifecycle::fail_closed_deadline(now, bab_timeout),
        ));
        let mut bab_iteration = 0usize;

        while let Some(domain) = queue.pop() {
            // Check timeout and domain limit (#1860 Packet A shared lifecycle, #4095)
            lifecycle.cuts_generated = cut_pool.total_generated;
            if let Some(result) = lifecycle.check_termination(bab_timeout, self.config.max_domains)
            {
                // #bab-frontier graph lane: the just-popped domain is still
                // part of the surviving frontier (highest priority, best
                // seed), so chain it back in front of the queue for the
                // export (env-gated, guidance only — see bab_frontier_export).
                crate::beta_crown::bab_frontier_export::record_graph_bab_frontier_if_enabled(
                    std::iter::once(&domain).chain(queue.iter()),
                    input,
                );
                return Ok(result);
            }

            lifecycle.domains_explored += 1;
            lifecycle.max_depth_reached = lifecycle.max_depth_reached.max(domain.depth);

            if drop_non_finite_domain_in_relu_split_bounds(
                &domain,
                &mut lifecycle.unresolved_due_to_propagation_failure,
            ) {
                continue;
            }

            // Conflict-clause prune at the pop (NY_BAB_CLAUSE_LEARN=1, default
            // off): if a recorded clause is a subset of this domain's pure
            // ReLU-at-0 literal set, its region is a subregion of an
            // already-certified one (same run, same root box, same objective
            // and threshold by construction) — close it as verified WITHOUT
            // bound work. Fails closed for impure (GenBaB/norm) histories.
            // Deliberately not fed to cut generation: its history is a
            // superset of a stored clause whose domain the cut machinery
            // already saw. Ordering mirrors the v1 sequential prefilter:
            // prune-check precedes the domain's own verified/violation checks.
            if clause_store.should_prune(&domain.history) {
                lifecycle.domains_verified += 1;
                continue;
            }

            // Lambda optimization: periodically optimize cut lambdas
            if self.config.enable_cuts
                && !cut_pool.is_empty()
                && lifecycle
                    .domains_explored
                    .is_multiple_of(lambda_opt_interval)
            {
                lambda_opt_iter += 1;

                // Compute gradients using current domain's node bounds
                self.compute_graph_cut_gradients(
                    graph,
                    &mut cut_pool,
                    &domain.node_bounds,
                    domain.input_bounds.as_ref(),
                );

                // Update all cut lambdas
                for cut in cut_pool.cuts_mut() {
                    cut.update_lambda_adam(
                        lambda_lr,
                        lambda_beta1,
                        lambda_beta2,
                        lambda_epsilon,
                        lambda_opt_iter,
                    );
                }
                cut_pool.zero_grad();

                debug!(
                    "Lambda optimization iter {}: total_lambda = {:.4}",
                    lambda_opt_iter,
                    cut_pool.total_lambda()
                );
            }

            // Check if domain is already verified
            if self
                .config
                .domain_is_verified(domain.lower_bound, domain.upper_bound, threshold)
            {
                self.cover_note_bound(
                    objective,
                    threshold,
                    &domain.history,
                    Some(&domain.beta_state),
                    domain.lower_bound,
                    domain.upper_bound,
                );
                lifecycle.domains_verified += 1;
                // Verified close: record the literal set as a conflict clause
                // (no-op unless gated on AND the history is pure ReLU-at-0).
                clause_store.record_verified_close(&domain.history);
                // Generate cut from verified domain
                if self.config.enable_cuts
                    && domain.depth >= self.config.min_cut_depth
                    && cut_pool.add_from_verified_domain(&domain.history)?
                {
                    debug!(
                        "Generated cut from verified domain (depth={}, total cuts={})",
                        domain.depth,
                        cut_pool.len()
                    );
                    let merged_len = cut_pool.merge_cuts();
                    debug!(
                        "Merged verified-domain graph cuts (pool_len={})",
                        merged_len
                    );
                }
                continue;
            }

            // Check for conclusive violation
            if self
                .config
                .domain_is_violation(domain.lower_bound, domain.upper_bound, threshold)
            {
                lifecycle.cuts_generated = cut_pool.total_generated;
                return Ok(lifecycle.build_result(BabVerificationStatus::potential_violation()));
            }

            // Near-miss cut generation: generate cuts from domains close to verification
            // This can help prune similar regions in the search space
            if self.config.enable_cuts
                && self.config.enable_near_miss_cuts
                && domain.depth >= self.config.min_cut_depth
            {
                let bound_for_check = self
                    .config
                    .relevant_bound(domain.lower_bound, domain.upper_bound);
                if cut_pool.add_from_near_miss_domain(
                    &domain.history,
                    bound_for_check,
                    threshold,
                    self.config.near_miss_margin,
                )? {
                    debug!(
                        "Generated near-miss cut (depth={}, lb={:.4}, threshold={:.4}, total cuts={})",
                        domain.depth,
                        bound_for_check,
                        threshold,
                        cut_pool.len()
                    );
                    let merged_len = cut_pool.merge_cuts();
                    debug!("Merged near-miss graph cuts (pool_len={})", merged_len);
                }
            }

            // Check depth limit
            if domain.depth >= self.config.max_depth {
                lifecycle.unresolved_due_to_depth = true;
                continue;
            }

            // #genbab-precomputed-lane: honour a configured GenBaB nonlinear
            // heuristic, exactly as `process_sequential_domains` does, BEFORE
            // falling through to the ReLU-only search below. Without this the
            // lane scores ReLU nodes only, so a benchmark that branches
            // `MulBinary` products finds nothing to split and returns holding
            // its whole budget. The enqueue/verified/violation fold is NOT
            // duplicated: this calls the same `aggregate_bab_results` the
            // sequential lane uses, so the two lanes cannot disagree about
            // what a domain result means.
            if genbab_route {
                let outcome = self.process_graph_domain_parallel(
                    graph,
                    &domain,
                    &setup.relu_nodes,
                    objective,
                    threshold,
                    engine,
                    // This lane is single-depth (`lane=relu_split_single`): it
                    // has never produced 2^k children per parent, and 1 keeps
                    // that shape.
                    1,
                );
                if let Some(result) = self.aggregate_bab_results(
                    vec![outcome],
                    threshold,
                    &|lower: f32, upper: f32| self.config.domain_priority(lower, upper),
                    &mut queue,
                    &mut lifecycle,
                    &mut cut_pool,
                )? {
                    return Ok(result);
                }
                enforce_graph_queue_budget(
                    queue_budget,
                    &mut queue,
                    &mut lifecycle,
                    "relu-split-precomputed",
                );
                continue;
            }

            // Find unstable neurons to branch on
            let unstable = self.find_unstable_graph_neurons(graph, &domain, &setup.relu_nodes);
            if unstable.is_empty() {
                // No unstable ReLU/Sign neurons left to branch on. If bounds are still inconclusive,
                // we cannot refine this domain further.
                // Use β and α for tightest possible bounds on fully-constrained domains (#1851)
                let context = GraphCrownContext::new(
                    &domain.history,
                    Some(&cut_pool),
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
                        if self.config.domain_is_verified(l, u, threshold) {
                            self.cover_note_bound(
                                objective,
                                threshold,
                                &domain.history,
                                Some(&domain.beta_state),
                                l,
                                u,
                            );
                            lifecycle.domains_verified += 1;
                            clause_store.record_verified_close(&domain.history);
                            continue;
                        }

                        if self.config.domain_is_violation(l, u, threshold) {
                            lifecycle.cuts_generated = cut_pool.total_generated;
                            return Ok(lifecycle
                                .build_result(BabVerificationStatus::potential_violation()));
                        }
                    }
                    Err(ref e) if e.is_infeasible_domain() => {
                        // #2926: Infeasible domain = empty = trivially verified.
                        // Record: emptiness is a property of the region itself
                        // (pure literal set over the root box), and every
                        // superset literal set yields a subset of the empty
                        // region — also empty, also trivially verified.
                        debug!("[#2926] no-unstable domain infeasible (empty), pruning: {e}");
                        self.cover_note_infeasible(objective, threshold, &domain.history, None);
                        lifecycle.domains_verified += 1;
                        clause_store.record_verified_close(&domain.history);
                    }
                    Err(e) => {
                        // #1867/#1861: propagation failed on fully-constrained domain.
                        // This sub-region is unexplored — must not claim Verified.
                        debug!("[#1867] no-unstable propagation failed: {e}");
                        lifecycle.unresolved_due_to_propagation_failure = true;
                    }
                }
                lifecycle.unresolved_due_to_no_branch = true;
                continue;
            }

            // This sequential entry point processes one splittable parent per
            // outer BaB round. One stamp covers branch scoring and both child
            // propagations; terminal/no-work pops do not consume a round.
            if self.config.enable_clip_interm_domain {
                bab_iteration = bab_iteration.saturating_add(1);
                let _ = self.complete_clip_root_bounds_cache.set_bab_iteration(
                    graph,
                    input,
                    bab_iteration,
                );
            }

            // Select neuron to split using branching heuristic.
            // #2097/#2038: per-domain branch-selection failure must not abort the whole BaB loop.
            let (node_name, neuron_idx, score) = match self
                .select_graph_branch_or_propagation_failure_in_relu_split_bounds(
                    graph, &domain, &unstable,
                ) {
                Ok(selection) => selection,
                Err(GraphDomainResult::PropagationFailure) => {
                    lifecycle.unresolved_due_to_propagation_failure = true;
                    continue;
                }
                Err(other) => {
                    lifecycle.unresolved_due_to_propagation_failure = true;
                    warn!(
                        depth = domain.depth,
                        result = ?other,
                        "Unexpected domain result from branch-selection fallback helper"
                    );
                    continue;
                }
            };

            // Create active child (x >= 0)
            let active_constraint = GraphNeuronConstraint {
                node_name: node_name.clone(),
                neuron_idx,
                is_active: true,
                score,
            };
            // #graph-bab-cover-ledger: `Ok(None)` is a genuinely empty
            // half-space; the `else` below puts that close on the record.
            if let Some(mut active_child) = domain.with_constraint(
                graph,
                active_constraint.clone(),
                self.config.verify_upper_bound,
            )? {
                // Recompute bounds with constraint and apply cuts
                // Use parent domain's node_bounds as base for efficiency
                // β-CROWN adds Lagrangian contribution from split constraints
                // Alpha state enables optimized ReLU relaxation slopes (#1851)
                // #cone-delta: the child inherited `node_bounds` verbatim from
                // `domain`, so its delta describes exactly the base map here
                // (dark, NY_CONE_REFRESH-gated).
                let context = GraphCrownContext::new(
                    &active_child.history,
                    Some(&cut_pool),
                    Some(&domain.node_bounds),
                    engine,
                )
                .with_alpha(&active_child.alpha_state)
                .with_delta_seeds(&active_child.delta_pre_nodes);
                match self.propagate_crown_with_graph_beta(
                    graph,
                    active_child.input_bounds.as_ref(),
                    &context,
                    &active_child.beta_state,
                    Some(objective),
                ) {
                    Ok((output, node_cache)) => {
                        // #cone-delta increment 2: already Arc-shared — move.
                        active_child.node_bounds = node_cache;
                        // #cone-delta: post-bounding replacement — delta
                        // restarts empty.
                        active_child.delta_pre_nodes.clear();
                        let l = output.lower_scalar();
                        let u = output.upper_scalar();
                        // #bab-warm-start trace, graph-lane twin of the
                        // sequential [child-trace] (NY_BAB_CHILD_TRACE=1).
                        {
                            static TRACE_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                            static COUNT: std::sync::atomic::AtomicU64 =
                                std::sync::atomic::AtomicU64::new(0);
                            if *TRACE_ON.get_or_init(|| {
                                std::env::var("NY_BAB_CHILD_TRACE").ok().as_deref() == Some("1")
                            }) {
                                let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if n.is_multiple_of(97) {
                                    eprintln!(
                                        "[gchild-trace] depth={} lower={l:.6} upper={u:.6} threshold_pending",
                                        active_child.depth,
                                    );
                                }
                            }
                        }
                        active_child.lower_bound = l;
                        active_child.upper_bound = u;

                        // #3707: guard NaN/Inf child bounds — treat as propagation
                        // failure instead of aborting the entire verification via `?`.
                        if !l.is_finite() || !u.is_finite() {
                            warn!(
                                depth = active_child.depth,
                                lower = l,
                                upper = u,
                                "relu_split_with_bounds: active child dropped — non-finite bounds"
                            );
                            lifecycle.unresolved_due_to_propagation_failure = true;
                        } else {
                            active_child.priority = self.config.domain_priority(l, u)?;

                            let child_verified = self.config.domain_is_verified(l, u, threshold);
                            if child_verified {
                                self.cover_note_bound(
                                    objective,
                                    threshold,
                                    &active_child.history,
                                    Some(&active_child.beta_state),
                                    l,
                                    u,
                                );
                                lifecycle.domains_verified += 1;
                                clause_store.record_verified_close(&active_child.history);
                                // Generate cut from verified child
                                if self.config.enable_cuts
                                    && active_child.depth >= self.config.min_cut_depth
                                    && cut_pool.add_from_verified_domain(&active_child.history)?
                                {
                                    debug!(
                                        "Generated cut from verified child (depth={}, total cuts={})",
                                        active_child.depth,
                                        cut_pool.len()
                                    );
                                    let merged_len = cut_pool.merge_cuts();
                                    debug!(
                                        "Merged verified-child graph cuts (pool_len={})",
                                        merged_len
                                    );
                                }
                            } else {
                                queue.push(active_child);
                            }
                        }
                    }
                    Err(ref e) if e.is_infeasible_domain() => {
                        // #2926: Infeasible domain = empty = trivially verified.
                        // Recording is sound: see the no-unstable infeasible arm.
                        debug!("[#2926] active child infeasible (empty), pruning: {e}");
                        self.cover_note_infeasible(
                            objective,
                            threshold,
                            &active_child.history,
                            None,
                        );
                        lifecycle.domains_verified += 1;
                        clause_store.record_verified_close(&active_child.history);
                    }
                    Err(e) => {
                        // #1867/#1861: active child propagation failed — sub-region unexplored.
                        debug!("[#1867] active child propagation failed: {e}");
                        lifecycle.unresolved_due_to_propagation_failure = true;
                    }
                }
            } else {
                self.cover_note_infeasible(
                    objective,
                    threshold,
                    &domain.history,
                    Some(&active_constraint),
                );
            }

            // Create inactive child (x < 0)
            let inactive_constraint = GraphNeuronConstraint {
                node_name,
                neuron_idx,
                is_active: false,
                score,
            };
            if let Some(mut inactive_child) = domain.with_constraint(
                graph,
                inactive_constraint.clone(),
                self.config.verify_upper_bound,
            )? {
                // Recompute bounds with constraint, β, and α (#1851)
                // #cone-delta: same base/delta pairing as the active child.
                let context = GraphCrownContext::new(
                    &inactive_child.history,
                    Some(&cut_pool),
                    Some(&domain.node_bounds),
                    engine,
                )
                .with_alpha(&inactive_child.alpha_state)
                .with_delta_seeds(&inactive_child.delta_pre_nodes);
                match self.propagate_crown_with_graph_beta(
                    graph,
                    inactive_child.input_bounds.as_ref(),
                    &context,
                    &inactive_child.beta_state,
                    Some(objective),
                ) {
                    Ok((output, node_cache)) => {
                        // #cone-delta increment 2: already Arc-shared — move.
                        inactive_child.node_bounds = node_cache;
                        // #cone-delta: post-bounding replacement — delta
                        // restarts empty.
                        inactive_child.delta_pre_nodes.clear();
                        let l = output.lower_scalar();
                        let u = output.upper_scalar();
                        inactive_child.lower_bound = l;
                        inactive_child.upper_bound = u;

                        // #3707: guard NaN/Inf child bounds — treat as propagation
                        // failure instead of aborting the entire verification via `?`.
                        if !l.is_finite() || !u.is_finite() {
                            warn!(
                                depth = inactive_child.depth,
                                lower = l,
                                upper = u,
                                "relu_split_with_bounds: inactive child dropped — non-finite bounds"
                            );
                            lifecycle.unresolved_due_to_propagation_failure = true;
                        } else {
                            inactive_child.priority = self.config.domain_priority(l, u)?;

                            let child_verified = self.config.domain_is_verified(l, u, threshold);
                            if child_verified {
                                self.cover_note_bound(
                                    objective,
                                    threshold,
                                    &inactive_child.history,
                                    Some(&inactive_child.beta_state),
                                    l,
                                    u,
                                );
                                lifecycle.domains_verified += 1;
                                clause_store.record_verified_close(&inactive_child.history);
                                // Generate cut from verified child
                                if self.config.enable_cuts
                                    && inactive_child.depth >= self.config.min_cut_depth
                                    && cut_pool.add_from_verified_domain(&inactive_child.history)?
                                {
                                    debug!(
                                        "Generated cut from verified child (depth={}, total cuts={})",
                                        inactive_child.depth,
                                        cut_pool.len()
                                    );
                                    let merged_len = cut_pool.merge_cuts();
                                    debug!(
                                        "Merged verified-child graph cuts (pool_len={})",
                                        merged_len
                                    );
                                }
                            } else {
                                queue.push(inactive_child);
                            }
                        }
                    }
                    Err(ref e) if e.is_infeasible_domain() => {
                        // #2926: Infeasible domain = empty = trivially verified.
                        // Recording is sound: see the no-unstable infeasible arm.
                        debug!("[#2926] inactive child infeasible (empty), pruning: {e}");
                        self.cover_note_infeasible(
                            objective,
                            threshold,
                            &inactive_child.history,
                            None,
                        );
                        lifecycle.domains_verified += 1;
                        clause_store.record_verified_close(&inactive_child.history);
                    }
                    Err(e) => {
                        // #1867/#1861: inactive child propagation failed — sub-region unexplored.
                        debug!("[#1867] inactive child propagation failed: {e}");
                        lifecycle.unresolved_due_to_propagation_failure = true;
                    }
                }
            } else {
                self.cover_note_infeasible(
                    objective,
                    threshold,
                    &domain.history,
                    Some(&inactive_constraint),
                );
            }
            enforce_graph_queue_budget(
                queue_budget,
                &mut queue,
                &mut lifecycle,
                "relu-split-precomputed",
            );
        }

        // Queue exhaustion: shared lifecycle handles unresolved vs verified logic
        lifecycle.cuts_generated = cut_pool.total_generated;
        if clause_store.is_enabled() {
            debug!(
                clause_pruned = clause_store.pruned_count(),
                "Graph BaB conflict-clause learning stats"
            );
        }
        let final_result = lifecycle.build_final_result();

        if matches!(final_result.result, BabVerificationStatus::Verified) {
            info!(
                "Graph β-CROWN (ReLU split, pre-computed bounds) verified after {} domains, {} verified, {} cuts",
                lifecycle.domains_explored, lifecycle.domains_verified, cut_pool.total_generated
            );
        }

        Ok(final_result)
    }

    fn select_graph_branch_or_propagation_failure_in_relu_split_bounds(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
    ) -> std::result::Result<(String, usize, f32), GraphDomainResult> {
        match self.select_graph_branch(graph, domain, unstable) {
            Ok(selection) => Ok(selection),
            Err(e) => {
                warn!(
                    error = %e,
                    depth = domain.depth,
                    "select_graph_branch failed in ReLU split with bounds loop; marking domain as PropagationFailure (#2097, #2038)"
                );
                Err(GraphDomainResult::PropagationFailure)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for #2097: branch-selection failures in ReLU-split-with-bounds
    /// must map to per-domain PropagationFailure, not abort the whole verification.
    #[ntest::timeout(5000)]
    #[test]
    fn test_select_graph_branch_failure_maps_to_propagation_failure_2097() {
        let verifier = BetaCrownVerifier::new(crate::beta_crown::BetaCrownConfig::default());

        let mut graph = GraphNetwork::new();
        graph.add_node(crate::GraphNode::from_input(
            "relu",
            crate::Layer::ReLU(crate::ReLULayer),
        ));
        graph.add_node(crate::GraphNode::new(
            "linear1",
            crate::Layer::Linear(
                crate::LinearLayer::new(ndarray::arr2(&[[1.0, 1.0]]), None).unwrap(),
            ),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear1");

        let input = BoundedTensor::new(
            ndarray::arr1(&[-1.0f32, -1.0]).into_dyn(),
            ndarray::arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .unwrap();
        let domain =
            GraphBabDomain::root(std::collections::HashMap::new(), -1.0, 1.0, &input, false)
                .unwrap();

        let empty_unstable: Vec<(String, usize)> = vec![];
        let result = verifier.select_graph_branch_or_propagation_failure_in_relu_split_bounds(
            &graph,
            &domain,
            &empty_unstable,
        );

        assert!(
            matches!(result, Err(GraphDomainResult::PropagationFailure)),
            "select_graph_branch failure must map to PropagationFailure, got {result:?}"
        );
    }
}
