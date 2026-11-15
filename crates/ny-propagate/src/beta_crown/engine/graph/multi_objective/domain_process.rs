// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Single-domain processing for graph BaB verification.
//!
//! Contains `process_graph_domain_parallel` (ReLU branching) and
//! `process_graph_domain_genbab` (general nonlinearity branching).
//!
//! Extracted from `multi_objective.rs`.

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::beta_crown::branching::{BranchingHeuristic, GraphNeuronConstraint};
use crate::beta_crown::domain::{GraphBabDomain, GraphCrownContext};
use crate::beta_crown::nonlinear_branching::NonlinearBranching;
use crate::GraphNetwork;

use super::super::super::domain_results::GraphDomainResult;
use super::super::super::tensor_ext::BoundedTensorExt;
use super::super::super::BetaCrownVerifier;

impl BetaCrownVerifier {
    /// Process a single graph domain: split and compute child bounds.
    ///
    /// This is the parallel-safe workhorse for batched Graph BaB. It processes
    /// one domain and returns its children without touching mutable shared state.
    ///
    /// `split_depth` controls multi-depth ReLU splitting: when > 1,
    /// selects top-k unstable neurons and creates 2^k child domains per parent.
    ///
    /// Note: Does NOT use cuts (cut_pool is not passed) because this function
    /// is used in the CPU-parallel path where cuts require mutable access.
    /// The GPU-batched path (`process_graph_domains_batched_gpu`) accepts a
    /// read-only cut pool. Cut generation happens after parallel
    /// processing in both paths.
    // Justification: domain processing needs graph, domain, relu nodes, objective,
    // threshold, engine, and split depth — full verification context.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn process_graph_domain_parallel(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        relu_nodes: &[String],
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        split_depth: usize,
    ) -> GraphDomainResult {
        // Quick verification check (already done in caller, but re-check for safety)
        let already_verified =
            self.config
                .domain_is_verified(domain.lower_bound, domain.upper_bound, threshold);
        if already_verified {
            self.cover_note_bound(
                objective,
                threshold,
                &domain.history,
                Some(&domain.beta_state),
                domain.lower_bound,
                domain.upper_bound,
            );
            return GraphDomainResult::AlreadyVerified;
        }

        // Quick violation check
        if self
            .config
            .domain_is_violation(domain.lower_bound, domain.upper_bound, threshold)
        {
            return GraphDomainResult::Violation;
        }

        let split_depth = super::super::cap_relu_split_depth_for_parent(
            split_depth,
            usize::MAX,
            domain.depth,
            self.config.max_depth,
        );
        if split_depth == 0 {
            tracing::warn!(
                parent_depth = domain.depth,
                max_depth = self.config.max_depth,
                "parallel domain processor refused a parent with no remaining depth budget"
            );
            return GraphDomainResult::PropagationFailure;
        }
        // Check for GenBaB heuristic FIRST — GenBaB handles all nonlinearities
        // (GELU, Sigmoid, BilinearCrown, etc.), not just ReLU. The GenBaB path
        // has its own split_nodes discovery that includes BilinearCrown.
        // If checked after find_unstable_graph_neurons, graphs with no ReLU nodes
        // would short-circuit to NoUnstable before GenBaB gets a chance to split.
        if let BranchingHeuristic::GenBaB(ref genbab_config) = self.config.branching_heuristic {
            return self.process_graph_domain_genbab(
                graph,
                domain,
                genbab_config,
                objective,
                threshold,
                engine,
            );
        }

        // ReLU-specific branching (non-GenBaB heuristics only)
        let unstable = self.find_unstable_graph_neurons(graph, domain, relu_nodes);
        if unstable.is_empty() {
            // No unstable neurons left — this is a fully-decided (affine) leaf
            // subproblem. With every ReLU fixed, the network is affine over the
            // (split-constrained) input box and the exact bound is computable.
            // A single CROWN pass with *inherited* β multipliers leaves the
            // split constraints loosely enforced and can return Undecided on a
            // leaf that is in fact verifiable. Optimize the β (and α)
            // multipliers here so the split constraints are tightened to their
            // exact contribution, then verify against the threshold.
            return self
                .process_graph_leaf_no_unstable(graph, domain, objective, threshold, engine);
        }

        // Multi-depth splitting: when split_depth > 1, select top-k
        // neurons and create 2^k child domains per parent to fill GPU batches.
        // Falls back to single-neuron split when depth=1 or only 1 unstable.
        // Reference: alpha-beta-CROWN depth mode (domain_updater.py:63-304).
        if split_depth > 1 && unstable.len() > 1 {
            let k = split_depth.min(unstable.len());
            let branches = match self.select_graph_branches(graph, domain, &unstable, k) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        depth = domain.depth,
                        k = k,
                        "select_graph_branches failed in multi-depth parallel split"
                    );
                    return GraphDomainResult::PropagationFailure;
                }
            };
            let child_domains = match domain.with_multi_constraints_observed(
                graph,
                &branches,
                self.config.verify_upper_bound,
                &mut |history, constraint| {
                    self.cover_note_infeasible(objective, threshold, history, Some(constraint));
                },
            ) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(error = %e, "with_multi_constraints failed");
                    return GraphDomainResult::PropagationFailure;
                }
            };
            debug!(
                depth = domain.depth,
                split_k = branches.len(),
                num_children = child_domains.len(),
                "Multi-depth ReLU split (parallel)"
            );

            let mut children: Vec<(GraphBabDomain, bool)> = Vec::with_capacity(child_domains.len());
            let mut any_child_failed = false;

            for mut child in child_domains {
                match self.evaluate_graph_child_bounds(
                    graph,
                    &mut child,
                    &domain.node_bounds,
                    objective,
                    None, // No cuts in parallel path
                    engine,
                ) {
                    Ok(true) => {
                        let verified = self.verify_child_inline_if_leaf(
                            graph, &mut child, relu_nodes, objective, threshold, engine,
                        );
                        children.push((child, verified));
                    }
                    Err(ref e) if e.is_infeasible_domain() => {
                        debug!("Multi-depth child infeasible (empty): {e}");
                        self.cover_note_infeasible(objective, threshold, &child.history, None);
                    }
                    _ => {
                        any_child_failed = true;
                    }
                }
            }

            if any_child_failed {
                return GraphDomainResult::PropagationFailure;
            }
            return GraphDomainResult::Children(children);
        }

        // Single-depth path: select one neuron, create active + inactive children
        let (node_name, neuron_idx, score) = match self
            .select_graph_branch_or_propagation_failure_parallel(graph, domain, &unstable)
        {
            Ok(v) => v,
            Err(result) => return result,
        };

        let mut children: Vec<(GraphBabDomain, bool)> = Vec::with_capacity(2);
        let mut any_child_failed = false;

        // Create active (x >= 0) and inactive (x <= 0) children
        for is_active in [true, false] {
            let constraint = GraphNeuronConstraint {
                node_name: node_name.clone(),
                neuron_idx,
                is_active,
                score,
            };
            match domain.with_constraint(graph, constraint.clone(), self.config.verify_upper_bound)
            {
                Ok(Some(mut child)) => {
                    match self.evaluate_graph_child_bounds(
                        graph,
                        &mut child,
                        &domain.node_bounds,
                        objective,
                        None, // No cuts in parallel path
                        engine,
                    ) {
                        Ok(true) => {
                            let verified = self.verify_child_inline_if_leaf(
                                graph, &mut child, relu_nodes, objective, threshold, engine,
                            );
                            children.push((child, verified));
                        }
                        Err(ref e) if e.is_infeasible_domain() => {
                            tracing::debug!("Child infeasible (empty domain): {e}");
                            self.cover_note_infeasible(objective, threshold, &child.history, None);
                        }
                        _ => {
                            any_child_failed = true;
                        }
                    }
                }
                Ok(None) => {
                    self.cover_note_infeasible(
                        objective,
                        threshold,
                        &domain.history,
                        Some(&constraint),
                    );
                }
                Err(ref e) if e.is_infeasible_domain() => {
                    tracing::debug!("with_constraint infeasible: {e}");
                    self.cover_note_infeasible(
                        objective,
                        threshold,
                        &domain.history,
                        Some(&constraint),
                    );
                }
                Err(e) => {
                    tracing::warn!("with_constraint failed: {e}");
                    any_child_failed = true;
                }
            }
        }

        if any_child_failed {
            return GraphDomainResult::PropagationFailure;
        }

        GraphDomainResult::Children(children)
    }

    /// Compute the verification result for a fully-decided (no-unstable-neuron)
    /// leaf domain.
    ///
    /// When `find_unstable_graph_neurons` returns empty, every ReLU on every path
    /// to the output is fixed (active or inactive) by the split history, so the
    /// network is *affine* over the split-constrained input box and its exact
    /// bound is computable. A single CROWN pass with the leaf's *inherited* β
    /// multipliers leaves the split constraints loosely enforced and can yield a
    /// bound that fails the threshold on a leaf that is in fact verifiable.
    /// We therefore optimize the β (and α) multipliers here, exactly as
    /// the child-bound path does, so the split constraints contribute their
    /// tightest sound value.
    ///
    /// Soundness: every bound combined below is an independently sound CROWN(-βα)
    /// bound for the same subproblem. Combining two sound upper bounds by taking
    /// the smaller (and two sound lower bounds by taking the larger) is sound —
    /// the tighter side of a valid interval is still valid. The optimizer
    /// (`optimize_graph_beta_analytical`) only ever returns a sound CROWN-β bound
    /// it actually observed, so `verified` rests on a sound bound. We never widen
    /// past, or replace, a sound bound with an unverified guess.
    pub(in crate::beta_crown::engine::graph) fn process_graph_leaf_no_unstable(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> GraphDomainResult {
        let context = GraphCrownContext::new(
            &domain.history,
            None, // No cuts in parallel path
            Some(&domain.node_bounds),
            engine,
        );

        // Baseline: single CROWN pass with inherited β and α (no leaf
        // optimization). This is always a sound bound.
        let baseline = {
            let ctx = context.with_alpha(&domain.alpha_state);
            self.propagate_crown_with_graph_beta(
                graph,
                domain.input_bounds.as_ref(),
                &ctx,
                &domain.beta_state,
                Some(objective),
            )
        };

        let (mut best_lower, mut best_upper) = match baseline {
            Ok((output, _node_cache)) => (output.lower_scalar(), output.upper_scalar()),
            Err(ref e) if e.is_infeasible_domain() => {
                // Infeasible domain = empty = trivially verified.
                debug!(error = %e, depth = domain.depth, "NoUnstable domain infeasible (empty)");
                self.cover_note_infeasible(objective, threshold, &domain.history, None);
                return GraphDomainResult::AlreadyVerified;
            }
            Err(e) => {
                debug!(error = %e, depth = domain.depth, "NoUnstable CROWN propagation failed — returning PropagationFailure (#1978)");
                return GraphDomainResult::PropagationFailure;
            }
        };

        // Optimize β (and α) on the leaf so the split constraints are tightened
        // to their exact contribution for this affine subproblem. Clone
        // the states because `domain` is shared/immutable in the parallel path.
        //
        // The leaf must be optimized regardless of `beta_iterations`/`beta_max_depth`:
        // those throughput knobs gate *interior* per-domain optimization, but a
        // no-unstable-neuron leaf is the final, exactly-determined affine
        // subproblem — its bound directly decides Verified vs Undecided. Per-domain
        // `beta_iterations` defaults to 0, which is exactly why this leaf path
        // previously never optimized and returned Undecided on verifiable leaves.
        // Use `root_beta_iterations` as the leaf budget.
        //
        // `optimize_graph_leaf_beta_directional` keeps the tightest bound in the
        // *verification direction* (min upper bound when `verify_upper_bound`),
        // which the generic lower-bound-maximizing optimizer would discard.
        let leaf_iterations = self
            .config
            .beta_iterations
            .max(self.config.root_beta_iterations);
        if leaf_iterations > 0 && !domain.beta_state.is_empty() {
            let mut beta_state = domain.beta_state.clone();
            let mut alpha_state = domain.alpha_state.clone();
            match self.optimize_graph_leaf_beta_directional(
                graph,
                domain.input_bounds.as_ref(),
                &context,
                &mut beta_state,
                &mut alpha_state,
                objective,
                leaf_iterations,
                self.config.verify_upper_bound,
            ) {
                Ok((l, u)) => {
                    // Combine soundly: tighten each side only with a finite,
                    // independently-sound bound. Never loosen.
                    if l.is_finite() && l > best_lower {
                        best_lower = l;
                    }
                    if u.is_finite() && u < best_upper {
                        best_upper = u;
                    }
                }
                Err(e) => {
                    // Optimization failed — keep the sound baseline bound.
                    debug!(error = %e, depth = domain.depth, "NoUnstable β optimization failed; using baseline bound");
                }
            }
        }

        let verified = self
            .config
            .domain_is_verified(best_lower, best_upper, threshold);

        // Soundness sanity check: a "Verified" leaf must actually satisfy the
        // threshold via the bound we are reporting (leaf optimization must only
        // verify MORE TRUE things, never a false property).
        if verified {
            self.cover_note_bound(
                objective,
                threshold,
                &domain.history,
                Some(&domain.beta_state),
                best_lower,
                best_upper,
            );
            debug_assert!(
                self.config
                    .domain_is_verified(best_lower, best_upper, threshold),
                "leaf reported Verified but bound [{best_lower}, {best_upper}] does not satisfy threshold {threshold}"
            );
        } else {
            debug!(
                lower = best_lower,
                upper = best_upper,
                threshold = threshold,
                depth = domain.depth,
                splits = domain.history.depth(),
                "[#1817/#1896] NoUnstable leaf not verified after β optimization"
            );
        }

        GraphDomainResult::NoUnstable {
            lower: best_lower,
            upper: best_upper,
            verified,
        }
    }

    /// #beta-ascent-sat engagement observable.
    ///
    /// `tightened` = the child's own bound was strictly tighter on the violation
    /// side and was kept; `passthrough` = the leaf bound was adopted unchanged
    /// (the pre-fix behavior, and still the common case).
    ///
    /// Recording happens strictly AFTER the merge decision and never touches a
    /// bound value or a route, so it cannot affect soundness or the verdict — it
    /// only makes the decision observable. A null measured through this fix with
    /// `tightened=0` is VACUOUS, not negative; these counts are how the two are
    /// told apart. Rate-limited by construction: a line is printed only when the
    /// just-bumped counter crosses a power of two.
    ///
    /// Mirrors `DeadlinePreserveCounters` (network/core/graph/backward_helpers.rs).
    fn record_leaf_merge(tightened: bool) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TIGHTENED: AtomicU64 = AtomicU64::new(0);
        static PASSTHROUGH: AtomicU64 = AtomicU64::new(0);

        let bumped = if tightened { &TIGHTENED } else { &PASSTHROUGH };
        let after = bumped.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        if after.is_power_of_two() {
            eprintln!(
                "[beta-ascent-sat] leaf-merge: tightened={} passthrough={}",
                TIGHTENED.load(Ordering::Relaxed),
                PASSTHROUGH.load(Ordering::Relaxed),
            );
        }
    }

    /// Combine a child's existing bound with the leaf-optimized bound, keeping the
    /// tighter value on the **violation** side only.
    ///
    /// # Why this exists (#beta-ascent-sat)
    ///
    /// Verification and violation read OPPOSITE ends of the same interval:
    ///
    /// | `verify_upper_bound` | verified when      | violation when      |
    /// |----------------------|--------------------|---------------------|
    /// | `false`              | `lower > threshold`| `upper < threshold` |
    /// | `true`               | `upper < threshold`| `lower >= threshold`|
    ///
    /// Every β/α optimizer in this engine ascends the **lower** bound, and both
    /// `optimize_graph_beta_analytical` and `optimize_graph_leaf_beta_directional`
    /// select their returned iterate by the **verification** direction. The
    /// off-direction value is a passenger: it is whatever accompanied the chosen
    /// iterate, never the tightest one observed. So the violation side degrades as
    /// soon as ascent is enabled, and BaB reports `Unknown` on domains where a
    /// violation witness was already in hand.
    ///
    /// # Soundness
    ///
    /// Both inputs are independently sound CROWN(-βα) bounds for the *same*
    /// subproblem (β-CROWN is a valid Lagrangian dual for any β ≥ 0, and any α in
    /// range is a valid relaxation slope), and shipped code already treats each of
    /// them as verdict-authoritative on its own. Taking the smaller of two sound
    /// upper bounds — or the larger of two sound lower bounds — is therefore sound:
    /// this is CONTAINMENT, and it can only ever TIGHTEN. It never widens a bound
    /// and never manufactures finiteness (a non-finite candidate is ignored).
    ///
    /// # Why this cannot move an unsat verdict
    ///
    /// The value on the **verification** side is passed through from `leaf`
    /// unchanged in both modes, so `domain_is_verified` returns bit-identical
    /// results before and after this merge. This function is structurally
    /// incapable of producing a new `Verified` domain; it can only make
    /// `domain_is_violation` fire on a domain that already carried the witness.
    /// A `PotentialViolation` is a search signal — a sat verdict must still clear
    /// the unchanged trusted-oracle gate — so the 0-wrong moat is untouched.
    ///
    /// An inverted result (`lower > upper`) would mean the two bound pairs are
    /// disjoint, i.e. at least one is unsound; rather than trust either, the merge
    /// declines and returns `leaf` unchanged (today's behavior).
    pub(in crate::beta_crown::engine::graph) fn merge_leaf_bound_violation_side(
        verify_upper_bound: bool,
        existing: (f32, f32),
        leaf: (f32, f32),
    ) -> (f32, f32) {
        let (existing_lower, existing_upper) = existing;
        let (leaf_lower, leaf_upper) = leaf;

        // Fail closed on ANY non-finite component, matching
        // `domain_is_violation_for_mode`. A NaN/Inf in either half of a bound pair
        // is evidence that the propagation which produced it was corrupted, so the
        // finite half is not independently trustworthy either.
        if !existing_lower.is_finite()
            || !existing_upper.is_finite()
            || !leaf_lower.is_finite()
            || !leaf_upper.is_finite()
        {
            Self::record_leaf_merge(false);
            return leaf;
        }

        if verify_upper_bound {
            // Violation side is the LOWER bound: keep the larger sound value.
            if existing_lower > leaf_lower && existing_lower <= leaf_upper {
                Self::record_leaf_merge(true);
                return (existing_lower, leaf_upper);
            }
        } else {
            // Violation side is the UPPER bound: keep the smaller sound value.
            if existing_upper < leaf_upper && existing_upper >= leaf_lower {
                Self::record_leaf_merge(true);
                return (leaf_lower, existing_upper);
            }
        }
        Self::record_leaf_merge(false);
        leaf
    }

    /// Decide whether a freshly-evaluated child domain is verified, applying the
    /// fully-decided-leaf β/α optimization inline when the child has no unstable
    /// neurons left.
    ///
    /// Children created by a split inherit their parent's (loose) intermediate
    /// bounds, so a child whose split made it fully decided (every ReLU fixed)
    /// still gets a loose CROWN bound from `evaluate_graph_child_bounds` and would
    /// be re-queued, only to verify later at the no-unstable pop path. Verifying
    /// it here — via the same sound directional leaf optimization — resolves it
    /// inline, avoiding a redundant BaB iteration and matching the
    /// branch-time verification semantics.
    ///
    /// The "loose child bound" premise above holds only while β ascent is OFF.
    /// With `beta_iterations > 0` the child bound can be strictly TIGHTER than the
    /// leaf recompute on both sides (see `merge_leaf_bound_violation_side`), so the
    /// leaf bound is adopted on the verification side but merged — never blindly
    /// substituted — on the violation side.
    ///
    /// Soundness: returns `true` only when `domain_is_verified` holds for a sound
    /// CROWN(-βα) bound (either the standard child bound or a tighter sound bound
    /// from `process_graph_leaf_no_unstable`). Never weakens a soundness check.
    fn verify_child_inline_if_leaf(
        &self,
        graph: &GraphNetwork,
        child: &mut GraphBabDomain,
        relu_nodes: &[String],
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> bool {
        if self
            .config
            .domain_is_verified(child.lower_bound, child.upper_bound, threshold)
        {
            self.cover_note_bound(
                objective,
                threshold,
                &child.history,
                Some(&child.beta_state),
                child.lower_bound,
                child.upper_bound,
            );
            return true;
        }

        // Only the fully-decided (no-unstable-neuron) case benefits from the leaf
        // optimization; a child with remaining unstable neurons is handled by
        // further branching.
        if !self
            .find_unstable_graph_neurons(graph, child, relu_nodes)
            .is_empty()
        {
            return false;
        }

        match self.process_graph_leaf_no_unstable(graph, child, objective, threshold, engine) {
            GraphDomainResult::AlreadyVerified => true,
            GraphDomainResult::NoUnstable {
                lower,
                upper,
                verified,
            } => {
                // #beta-ascent-sat: adopt the leaf bound, but do NOT discard the
                // bound the child already carries on the VIOLATION side.
                //
                // The doc comment above assumes the incoming `child.{lower,upper}_bound`
                // is the loose one. That holds only when ascent is off. With
                // `beta_iterations > 0`, `evaluate_graph_child_bounds` runs
                // `optimize_graph_beta_analytical`, which returns the bound from its
                // BEST-seen iterate while leaving `beta_state`/`alpha_state` at the
                // FINAL iterate. `process_graph_leaf_no_unstable` then re-propagates
                // from that final-iterate state, so its bound can be strictly LOOSER
                // on both sides than the one the child already holds — and the
                // unconditional overwrite threw the tighter one away.
                let (merged_lower, merged_upper) = Self::merge_leaf_bound_violation_side(
                    self.config.verify_upper_bound,
                    (child.lower_bound, child.upper_bound),
                    (lower, upper),
                );
                child.lower_bound = merged_lower;
                child.upper_bound = merged_upper;
                verified
            }
            // Violation/PropagationFailure: leave unverified; the child stays in
            // the queue (or is handled by the caller's failure tracking).
            _ => false,
        }
    }

    fn select_graph_branch_or_propagation_failure_parallel(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
    ) -> Result<(String, usize, f32), GraphDomainResult> {
        match self.select_graph_branch(graph, domain, unstable) {
            Ok(selection) => Ok(selection),
            Err(e) => {
                tracing::warn!("select_graph_branch failed: {e} (#1915)");
                Err(GraphDomainResult::PropagationFailure)
            }
        }
    }

    /// Process a graph domain using GenBaB (general nonlinearity branching).
    ///
    /// This method uses NonlinearBranching to select neurons and branching points
    /// for general nonlinearities (GeLU, Sigmoid, Tanh, etc.) instead of the
    /// ReLU-specific branching at 0.
    fn process_graph_domain_genbab(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        genbab_config: &crate::beta_crown::nonlinear_branching::NonlinearBranchingConfig,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> GraphDomainResult {
        // Get split nodes: all nonlinear activation nodes, BilinearCrown, and
        // MulBinary nodes. BilinearCrown (attention Q@K^T) and MulBinary (element-wise
        // x·y, e.g. ml4acopf power flow) are McCormick-relaxed: splitting an input
        // interval reduces the envelope gap (ux−lx)(uy−ly)/4 that frozen root facets
        // cannot close.
        // Reference: auto_LiRPA BoundMatMul.splittable (linear.py:948).
        // Matches the GPU BaB path filter in gpu_bab/init.rs.
        let split_nodes: Vec<String> = graph
            .nodes
            .iter()
            .filter(|(_, node)| {
                node.layer.is_elementwise_activation()
                    || matches!(
                        node.layer,
                        crate::layers::Layer::BilinearCrown(_)
                            | crate::layers::Layer::MulBinary(_)
                            // #norm-genbab: RmsNorm is branchable on its internal
                            // inv_rms scalar (see selector::score_rms_norm_inv_rms).
                            | crate::layers::Layer::RmsNorm(_)
                    )
            })
            .map(|(name, _)| name.clone())
            .collect();

        if split_nodes.is_empty() {
            return GraphDomainResult::NoUnstable {
                lower: domain.lower_bound,
                upper: domain.upper_bound,
                verified: false,
            };
        }

        // Build node bounds map for NonlinearBranching. Include the network
        // input under the NETWORK_INPUT key so RmsNorm branching (#norm-genbab)
        // can read its input bounds when the norm's input IS the network input
        // (as in the swiglu-residual kernel `RmsNorm(x)` with `x` perturbed).
        let mut node_bounds: std::collections::HashMap<String, BoundedTensor> = domain
            .node_bounds
            .iter()
            .map(|(k, v)| (k.clone(), v.as_ref().clone()))
            .collect();
        node_bounds.insert(
            crate::NETWORK_INPUT.to_string(),
            domain.input_bounds.as_ref().clone(),
        );

        // Create NonlinearBranching instance and get decisions. Thread the
        // domain's applied norm inv_rms windows so RmsNorm branching converges
        // (#norm-genbab).
        let branching = NonlinearBranching::new(genbab_config.clone());
        let norm_windows = domain.history().norm_inv_rms_overrides();
        let decisions = match branching.decisions_with_norm_windows(
            graph,
            &node_bounds,
            &split_nodes,
            norm_windows.as_ref(),
        ) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("NonlinearBranching::get_decisions failed: {e}");
                return GraphDomainResult::PropagationFailure;
            }
        };

        if decisions.is_empty() {
            // No splittable neurons found
            return GraphDomainResult::NoUnstable {
                lower: domain.lower_bound,
                upper: domain.upper_bound,
                verified: false,
            };
        }

        // Use the best decision (first one, sorted by score)
        let decision = &decisions[0];

        // Create child domains using the branching decision
        let splits = match decision.to_splits() {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("decision.to_splits() failed: {e}");
                return GraphDomainResult::PropagationFailure;
            }
        };
        let mut children: Vec<(GraphBabDomain, bool)> = Vec::with_capacity(splits.len());

        let mut any_child_failed = false;
        for split in splits {
            match domain.with_general_split(graph, split, self.config.verify_upper_bound) {
                Ok(Some(mut child)) => {
                    // Compute bounds for the child
                    let context = GraphCrownContext::new(
                        &child.history,
                        None, // No cuts in parallel path
                        Some(&domain.node_bounds),
                        engine,
                    )
                    .with_alpha(&child.alpha_state);
                    match self.propagate_crown_with_graph_constraints(
                        graph,
                        child.input_bounds.as_ref(),
                        &context,
                        Some(&child.beta_state),
                        Some(objective),
                    ) {
                        Ok((output, node_cache)) => {
                            let l = output.lower_scalar();
                            let u = output.upper_scalar();
                            // #cone-delta increment 2: the result map is already
                            // Arc-shared — install by move.
                            child.node_bounds = node_cache;
                            // #cone-delta: post-bounding replacement — delta
                            // restarts empty.
                            child.delta_pre_nodes.clear();
                            // Use validated mutator to enforce NaN/Inf rejection.
                            let priority = match self.config.domain_priority(l, u) {
                                Ok(p) => p,
                                Err(e) => {
                                    // NaN in domain bounds → treat as propagation failure
                                    tracing::warn!(
                                        "GenBaB domain_priority failed (NaN bounds): {e}"
                                    );
                                    any_child_failed = true;
                                    continue;
                                }
                            };
                            if child.update_bounds(l, u, priority).is_err() {
                                // NaN/Inf in updated bounds → propagation failure
                                any_child_failed = true;
                                continue;
                            }

                            let verified = self.config.domain_is_verified(l, u, threshold);
                            if verified {
                                self.cover_note_bound(
                                    objective,
                                    threshold,
                                    &child.history,
                                    Some(&child.beta_state),
                                    l,
                                    u,
                                );
                            }
                            children.push((child, verified));
                        }
                        Err(ref e) if e.is_infeasible_domain() => {
                            // Infeasible domain = empty = trivially verified.
                            tracing::debug!("GenBaB child infeasible (empty): {e}");
                            self.cover_note_infeasible(objective, threshold, &child.history, None);
                        }
                        Err(e) => {
                            // Child propagation failed — sub-region unexplored.
                            tracing::warn!("GenBaB child propagation failed: {e}");
                            any_child_failed = true;
                        }
                    }
                }
                // An EMPTY child: the region really is empty, so dropping it is
                // sound, but its sibling is the unchanged parent — a no-op split.
                // Counted so `[genbab-select] empty_children=` exposes any
                // selector/`with_general_split` convention mismatch
                // (#genbab-selector-convention; must stay 0).
                Ok(None) => {
                    crate::beta_crown::nonlinear_branching::record_genbab_empty_child();
                    // The ledger does not rebuild a general split's half-space
                    // from the decision: leave the frame UNBUILT (legacy door).
                    self.cover_note_unbuildable("genbab_empty_child");
                }
                Err(ref e) if e.is_infeasible_domain() => {
                    // Infeasible split child = empty = skip.
                    tracing::debug!("with_general_split infeasible: {e}");
                    self.cover_note_unbuildable("genbab_infeasible_split");
                }
                Err(e) => {
                    tracing::warn!("with_general_split failed: {e}");
                    any_child_failed = true;
                }
            }
        }

        if any_child_failed {
            return GraphDomainResult::PropagationFailure;
        }

        GraphDomainResult::Children(children)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    use ny_tensor::BoundedTensor;

    use super::*;
    use crate::beta_crown::BetaCrownConfig;
    use crate::{GraphNode, Layer, LinearLayer, ReLULayer};

    // ---- #beta-ascent-sat: violation-side bound merge -------------------------
    //
    // These pin the exact quantity the pre-fix code discarded. Each one is
    // written so that the OLD behavior (`leaf` adopted unconditionally) reads
    // RED: the expected value differs from `leaf` on the violation side.

    /// The regression witness, in the numbers actually measured on
    /// `y = relu(x), x in [-1,1] |= y > 0.5` with `beta_iterations = 15`.
    ///
    /// The ascent returned its best-iterate pair `(-1.1754944e-38, 1.1754944e-38)`
    /// for the inactive child while leaving β/α at the final iterate; the leaf
    /// recompute then re-propagated from that state and produced the strictly
    /// looser `(-0.7499974, 0.7499974)`, which the unconditional overwrite
    /// adopted. `upper = 0.75` fails `upper < 0.5`, so the violation witness was
    /// lost and BaB ended `Unknown`.
    #[test]
    fn merge_leaf_bound_keeps_violation_side_upper_under_ascent() {
        let existing = (-1.1754944e-38_f32, 1.1754944e-38_f32);
        let leaf = (-0.7499974_f32, 0.7499974_f32);

        let (lower, upper) =
            BetaCrownVerifier::merge_leaf_bound_violation_side(false, existing, leaf);

        assert_eq!(
            upper, existing.1,
            "violation side (upper, verify_upper_bound=false) must keep the tighter \
             already-computed bound; adopting the leaf's {} loses the witness",
            leaf.1
        );
        assert_eq!(
            lower, leaf.0,
            "verification side (lower) must pass through from the leaf unchanged"
        );
        // The property the whole fix exists to restore.
        assert!(
            upper < 0.5,
            "merged upper must certify the violation of `y > 0.5`"
        );
    }

    /// Mirror case: in upper-bound mode the violation side is the LOWER bound.
    #[test]
    fn merge_leaf_bound_keeps_violation_side_lower_in_upper_bound_mode() {
        let existing = (0.90_f32, 2.0_f32);
        let leaf = (0.10_f32, 1.5_f32);

        let (lower, upper) =
            BetaCrownVerifier::merge_leaf_bound_violation_side(true, existing, leaf);

        assert_eq!(
            lower, existing.0,
            "violation side (lower, verify_upper_bound=true) must keep the larger \
             already-computed bound"
        );
        assert_eq!(
            upper, leaf.1,
            "verification side (upper) must pass through from the leaf unchanged"
        );
    }

    /// The merge must never move the VERIFICATION side, in either mode. This is
    /// what makes the change structurally incapable of creating a new `Verified`
    /// domain — and therefore incapable of moving an unsat verdict.
    #[test]
    fn merge_leaf_bound_never_moves_the_verification_side() {
        let cases: [((f32, f32), (f32, f32)); 6] = [
            ((-1e-30, 1e-30), (-0.75, 0.75)),
            ((0.9, 2.0), (0.1, 1.5)),
            ((-5.0, -4.0), (-1.0, 1.0)),
            ((f32::NAN, 0.0), (-1.0, 1.0)),
            ((0.0, f32::INFINITY), (-1.0, 1.0)),
            ((-1.0, 1.0), (-1.0, 1.0)),
        ];
        for (existing, leaf) in cases {
            let (lo_f, up_f) =
                BetaCrownVerifier::merge_leaf_bound_violation_side(false, existing, leaf);
            assert_eq!(
                lo_f, leaf.0,
                "verify_upper_bound=false must pass the lower bound through: {existing:?} {leaf:?}"
            );
            assert!(
                up_f <= leaf.1,
                "verify_upper_bound=false may only tighten the upper bound: {existing:?} {leaf:?}"
            );

            let (lo_t, up_t) =
                BetaCrownVerifier::merge_leaf_bound_violation_side(true, existing, leaf);
            assert_eq!(
                up_t, leaf.1,
                "verify_upper_bound=true must pass the upper bound through: {existing:?} {leaf:?}"
            );
            assert!(
                lo_t >= leaf.0,
                "verify_upper_bound=true may only tighten the lower bound: {existing:?} {leaf:?}"
            );
        }
    }

    /// Non-finite candidates must never win: the merge tightens, it never
    /// manufactures finiteness, and NaN must not poison a sound leaf bound.
    #[test]
    fn merge_leaf_bound_ignores_non_finite_candidates() {
        let leaf = (-1.0_f32, 1.0_f32);
        for existing in [
            (f32::NAN, f32::NAN),
            (0.0, f32::NAN),
            (f32::NEG_INFINITY, f32::NEG_INFINITY),
            (0.0, f32::NEG_INFINITY),
        ] {
            assert_eq!(
                BetaCrownVerifier::merge_leaf_bound_violation_side(false, existing, leaf),
                leaf,
                "non-finite existing bound must leave the leaf pair untouched: {existing:?}"
            );
            assert_eq!(
                BetaCrownVerifier::merge_leaf_bound_violation_side(true, existing, leaf),
                leaf,
                "non-finite existing bound must leave the leaf pair untouched: {existing:?}"
            );
        }
    }

    /// Disjoint bound pairs mean at least one input is unsound. The merge must
    /// decline rather than build an inverted interval, which `domain_is_violation`
    /// rejects anyway (#violdrop) and `domain_priority` treats as an error.
    #[test]
    fn merge_leaf_bound_declines_when_it_would_invert_the_interval() {
        // existing upper (-5.0) sits below the leaf lower (-1.0): disjoint.
        let leaf = (-1.0_f32, 1.0_f32);
        assert_eq!(
            BetaCrownVerifier::merge_leaf_bound_violation_side(false, (-6.0, -5.0), leaf),
            leaf,
            "must not produce lower > upper"
        );
        // existing lower (5.0) sits above the leaf upper (1.0): disjoint.
        assert_eq!(
            BetaCrownVerifier::merge_leaf_bound_violation_side(true, (5.0, 6.0), leaf),
            leaf,
            "must not produce lower > upper"
        );
        // And in general the result is never inverted.
        for existing in [(-6.0_f32, -5.0_f32), (5.0, 6.0), (0.0, 0.0)] {
            for mode in [false, true] {
                let (lo, up) =
                    BetaCrownVerifier::merge_leaf_bound_violation_side(mode, existing, leaf);
                assert!(lo <= up, "merged interval inverted: {lo} > {up}");
            }
        }
    }

    /// A looser existing bound must not be adopted — the merge only ever tightens.
    #[test]
    fn merge_leaf_bound_ignores_a_looser_existing_bound() {
        let leaf = (-1.0_f32, 1.0_f32);
        assert_eq!(
            BetaCrownVerifier::merge_leaf_bound_violation_side(false, (-2.0, 2.0), leaf),
            leaf,
            "a looser existing upper must not replace the leaf's"
        );
        assert_eq!(
            BetaCrownVerifier::merge_leaf_bound_violation_side(true, (-2.0, 2.0), leaf),
            leaf,
            "a looser existing lower must not replace the leaf's"
        );
    }

    #[test]
    fn test_select_graph_branch_failure_maps_to_propagation_failure_1915() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "linear1",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0, 1.0]]), None).unwrap()),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear1");

        let input = BoundedTensor::new(
            arr1(&[-1.0f32, -1.0]).into_dyn(),
            arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .unwrap();
        let domain =
            GraphBabDomain::root(std::collections::HashMap::new(), -1.0, 1.0, &input, false)
                .unwrap();

        let empty_unstable: Vec<(String, usize)> = vec![];
        let result = verifier.select_graph_branch_or_propagation_failure_parallel(
            &graph,
            &domain,
            &empty_unstable,
        );

        assert!(
            matches!(result, Err(GraphDomainResult::PropagationFailure)),
            "select_graph_branch failure must map to PropagationFailure, got {result:?}"
        );
    }

    /// Direct repro: a GenBaB split on MulBinary input_index=1 must tighten the
    /// SECOND input node, not misindex into the first. Inputs have different
    /// lengths so a misindex either errors (InvalidSpec) or applies the wrong
    /// neuron's split point.
    #[test]
    fn test_genbab_mul_binary_input1_split_tightens_second_input() {
        use crate::beta_crown::branching::{LayerRef, NeuronSplit};
        use crate::layers::binary_ops::MulBinaryLayer;

        // gate (input 0): length 2; up (input 1): length 2. Element-wise mul.
        let w_gate = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let w_up = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "gate",
            Layer::Linear(LinearLayer::new(w_gate, None).unwrap()),
        ));
        graph.add_node(GraphNode::from_input(
            "up",
            Layer::Linear(LinearLayer::new(w_up, None).unwrap()),
        ));
        graph.add_node(GraphNode::binary(
            "mul",
            Layer::MulBinary(MulBinaryLayer),
            "gate",
            "up",
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32, 1.0]]), None).unwrap()),
            vec!["mul".to_string()],
        ));
        graph.set_output("out");

        let input = BoundedTensor::new(
            arr1(&[-2.0_f32, -2.0]).into_dyn(),
            arr1(&[2.0_f32, 2.0]).into_dyn(),
        )
        .unwrap();
        let node_bounds = graph.collect_node_bounds(&input).unwrap();
        let root = GraphBabDomain::root(node_bounds, -10.0, 10.0, &input, false).unwrap();

        // Split MulBinary's input 1 ("up") neuron 0: upper branch up[0] >= 0.5.
        // The child's "up" node bounds for neuron 0 should become [0.5, 2].
        let split = NeuronSplit::new(LayerRef::Name("mul".to_string()), 0, Some(0.5), None, 1.0)
            .unwrap()
            .with_input_index(1);

        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let child = root
            .with_general_split(&graph, split, false)
            .expect("split should not hard-error")
            .expect("split should be feasible");

        let context = GraphCrownContext::new(&child.history, None, Some(&root.node_bounds), None);
        let (_out, node_cache) = verifier
            .propagate_crown_with_graph_constraints(
                &graph,
                child.input_bounds.as_ref(),
                &context,
                Some(&child.beta_state),
                Some(&[1.0_f32]),
            )
            .expect("child propagation should succeed for input_index=1 split");

        // The constraint up[0] >= 0.5 must tighten the "up" node bound, NOT "gate".
        let up = node_cache.get("up").expect("up bounds present");
        let up_lo0 = up.flatten().lower()[[0]];
        assert!(
            up_lo0 >= 0.5 - 1e-5,
            "input_index=1 split must tighten 'up'[0] lower to >= 0.5, got {up_lo0}"
        );
        // And it must NOT have wrongly tightened "gate"[0].
        let gate = node_cache.get("gate").expect("gate bounds present");
        let gate_lo0 = gate.flatten().lower()[[0]];
        assert!(
            gate_lo0 <= -2.0 + 1e-5,
            "input_index=1 split must NOT tighten 'gate'[0] (got lower={gate_lo0})"
        );
    }
}
