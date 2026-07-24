// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{trace, warn};

use crate::beta_crown::config::InputClipType;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::GraphNetwork;

use super::super::grouped_semantics::{disjunctive_domain_priority, disjunctive_domain_verified};
use super::super::shared::{
    extract_obj_bounds, try_graph_spec_ibp_prescreen_bounds, MultiObjBounds, MultiObjInputDomain,
};

/// Per-sub-domain warm-start result for the grouped disjunctive loop.
pub(super) type WarmDisjunctiveBoundsResult =
    (Vec<(f32, f32)>, Option<LinearBounds>, GraphAlphaState);

/// Warm-start bound closure for an eager grouped-disjunctive child.
pub(super) type WarmDisjunctiveComputeBoundsFn<'a> = dyn Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
        &GraphAlphaState,
    ) -> Result<WarmDisjunctiveBoundsResult>
    + 'a;

/// Per-verifier-run telemetry for the eager grouped-disjunctive warm-alpha path.
///
/// The structured progress/final records make a campaign self-invalidating when
/// the gate was enabled but no child warm call was actually reached. This is
/// deliberately local rather than process-global so deterministic restart runs
/// each report their own activation count.
pub(super) struct WarmAlphaTelemetry {
    enabled: bool,
    attempts: Cell<usize>,
    warm_bound_completions: Cell<usize>,
    warm_failures: Cell<usize>,
    frozen_fallback_completions: Cell<usize>,
}

impl WarmAlphaTelemetry {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            attempts: Cell::new(0),
            warm_bound_completions: Cell::new(0),
            warm_failures: Cell::new(0),
            frozen_fallback_completions: Cell::new(0),
        }
    }

    fn record_warm_bound_completion(&self) {
        self.attempts.set(self.attempts.get() + 1);
        self.warm_bound_completions
            .set(self.warm_bound_completions.get() + 1);
        self.maybe_emit_progress();
    }

    /// Returns whether the detailed warm-failure diagnostic should be emitted.
    fn record_warm_failure(&self) -> bool {
        self.attempts.set(self.attempts.get() + 1);
        self.warm_failures.set(self.warm_failures.get() + 1);
        self.maybe_emit_progress();
        let failures = self.warm_failures.get();
        failures == 1 || failures.is_power_of_two()
    }

    fn record_frozen_fallback_completion(&self) {
        self.frozen_fallback_completions
            .set(self.frozen_fallback_completions.get() + 1);
    }

    fn marker(&self, status: &str) -> String {
        format!(
            "NY_WARM_ALPHA route=grouped-disjunctive-eager status={} attempts={} warm_bound_completions={} warm_failures={} frozen_fallback_completions={}",
            status,
            self.attempts.get(),
            self.warm_bound_completions.get(),
            self.warm_failures.get(),
            self.frozen_fallback_completions.get()
        )
    }

    fn maybe_emit_progress(&self) {
        let attempts = self.attempts.get();
        if attempts == 1 || attempts.is_power_of_two() {
            eprintln!("{}", self.marker("progress"));
        }
    }

    #[cfg(test)]
    fn snapshot(&self) -> (usize, usize, usize, usize) {
        (
            self.attempts.get(),
            self.warm_bound_completions.get(),
            self.warm_failures.get(),
            self.frozen_fallback_completions.get(),
        )
    }
}

impl Drop for WarmAlphaTelemetry {
    fn drop(&mut self) {
        if self.enabled {
            eprintln!("{}", self.marker("final"));
        }
    }
}

fn compute_child_bounds(
    child_input: &BoundedTensor,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    compute_bounds: &impl Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
    ) -> Result<MultiObjBounds>,
    warm_compute_bounds: Option<&WarmDisjunctiveComputeBoundsFn<'_>>,
    prior_alpha: Option<&Arc<GraphAlphaState>>,
    warm_alpha_telemetry: &WarmAlphaTelemetry,
) -> Result<(
    Vec<(f32, f32)>,
    Option<LinearBounds>,
    Option<Arc<GraphAlphaState>>,
)> {
    let (Some(warm), Some(prior_alpha)) = (warm_compute_bounds, prior_alpha) else {
        let (obj_bounds, linear) = compute_bounds(child_input, node_bounds)?;
        return Ok((obj_bounds, linear, prior_alpha.cloned()));
    };

    match warm(child_input, node_bounds, prior_alpha.as_ref()) {
        Ok((obj_bounds, linear, refined_alpha)) => {
            // Completion proves that the warm route executed and returned a
            // sound state. It does not imply that an optimizer step changed a
            // slope (the deadline may already have expired, for example).
            warm_alpha_telemetry.record_warm_bound_completion();
            Ok((
                obj_bounds,
                linear,
                Some(Arc::new(refined_alpha.into_warm_start_seed())),
            ))
        }
        Err(err) => {
            // A failed optimization is not a failed proof attempt: deliberately
            // retry the historical frozen-root-alpha bound and retain the prior
            // Arc as the next generation's seed. If that sound fallback also
            // fails, propagate its error (fail closed).
            let emit_diagnostic = warm_alpha_telemetry.record_warm_failure();
            if emit_diagnostic {
                eprintln!("{}", warm_alpha_telemetry.marker("warm-failure"));
                warn!(
                    error = %err,
                    warm_alpha_failures = warm_alpha_telemetry.warm_failures.get(),
                    warm_alpha_route = "grouped-disjunctive-eager",
                    "grouped-disjunctive warm-alpha bound failed; attempting frozen bound fallback"
                );
            }
            let (obj_bounds, linear) = compute_bounds(child_input, node_bounds)?;
            warm_alpha_telemetry.record_frozen_fallback_completion();
            if emit_diagnostic {
                eprintln!(
                    "{}",
                    warm_alpha_telemetry.marker("frozen-fallback-complete")
                );
            }
            Ok((obj_bounds, linear, Some(prior_alpha.clone())))
        }
    }
}

/// Apply all sound lower bounds observed for this child before grouped
/// verification. A recompute can regress an individual row while improving a
/// different clause, so checking only the newest vector can miss a proof that
/// is already present in the component-wise maximum.
fn apply_running_lower_floor(obj_bounds: &mut [(f32, f32)], running_floor: &mut [f32]) {
    debug_assert_eq!(obj_bounds.len(), running_floor.len());
    for ((lower, _), floor) in obj_bounds.iter_mut().zip(running_floor.iter_mut()) {
        *floor = (*floor).max(*lower);
        *lower = *floor;
    }
}

/// #disj-cross-clause-clip-unsat: CLAUSE-AWARE per-child relaxed clip for the
/// `InputClipType::Relaxed` per-child screen lane (`clip_with_precomputed_linear`
/// over the disjunctive spec).
///
/// The historical loop clipped the child box against EVERY threshold row in
/// sequence, carrying the shrunk box forward across ALL clauses — i.e. it
/// intersected the per-row still-possibly-violating half-spaces of DIFFERENT
/// clauses. For an OR-of-clauses spec a counterexample only has to satisfy ONE
/// clause, so that cross-clause intersection discards the sub-boxes that satisfy
/// a single clause (genuine counterexamples: the lsnc false-unsat).
///
/// This clips EACH clause independently from the ORIGINAL child box
/// (intersecting only that clause's rows) and returns the per-child UNION
/// bounding box of the clauses that remain feasible — which ENCLOSES every
/// clause's violating region, so no counterexample is discarded and the deferred
/// re-bound stays a true over-approximation. `all_refuted` is true iff EVERY
/// clause was refuted within its own rows (some row's certified lower bound
/// exceeds its threshold over the clause box, which — since the clause box
/// encloses the clause's violating region — proves that clause unsatisfiable
/// over the whole child box); the caller may then count the domain verified.
///
/// Single-clause specs take the whole-spec sequential clip verbatim (the
/// conjunctive lanes are unchanged bit-for-bit).
fn clause_aware_child_relaxed_clip(
    verifier: &BetaCrownVerifier,
    child_input: &BoundedTensor,
    shape: &[usize],
    linear_bounds: &LinearBounds,
    thresholds: &[f32],
    clause_sizes: &[usize],
) -> Result<(BoundedTensor, bool)> {
    // Single clause (or degenerate): the historical whole-spec sequential clip,
    // carrying the box forward across all rows (bit-identical).
    if clause_sizes.len() <= 1 {
        let mut cur = child_input.clone();
        for (i, &threshold) in thresholds.iter().enumerate() {
            let outcome =
                verifier.clip_with_precomputed_linear(&cur, shape, linear_bounds, i, threshold)?;
            cur = outcome.bounds;
        }
        return Ok((cur, false));
    }

    let x_dim = child_input.flatten().lower().len();
    let mut union_l = vec![f32::INFINITY; x_dim];
    let mut union_u = vec![f32::NEG_INFINITY; x_dim];
    let mut any_kept = false;

    let mut offset = 0usize;
    for &size in clause_sizes {
        let end = offset + size;
        // Intersect ONLY this clause's rows, from the ORIGINAL child box.
        let mut clause_box = child_input.clone();
        let mut refuted = false;
        for i in offset..end {
            let outcome = verifier.clip_with_precomputed_linear(
                &clause_box,
                shape,
                linear_bounds,
                i,
                thresholds[i],
            )?;
            clause_box = outcome.bounds;
            if outcome.verified {
                // Row i's certified lower bound exceeds its threshold over the
                // clause box (which encloses the clause's violating region), so
                // this clause is unsatisfiable over the whole child box.
                refuted = true;
                break;
            }
        }
        if refuted {
            offset = end;
            continue;
        }
        any_kept = true;
        let cf = clause_box.flatten();
        let cl = cf.lower();
        let cu = cf.upper();
        for d in 0..x_dim {
            let l = cl[[d]];
            let u = cu[[d]];
            if l < union_l[d] {
                union_l[d] = l;
            }
            if u > union_u[d] {
                union_u[d] = u;
            }
        }
        offset = end;
    }

    if !any_kept {
        // Every clause refuted -> the child is verified; box is unused.
        return Ok((child_input.clone(), true));
    }

    let union_lower = ArrayD::from_shape_vec(IxDyn(shape), union_l).map_err(|e| {
        NyError::InvalidSpec(format!("clause-aware clip: union lower reshape: {e}"))
    })?;
    let union_upper = ArrayD::from_shape_vec(IxDyn(shape), union_u).map_err(|e| {
        NyError::InvalidSpec(format!("clause-aware clip: union upper reshape: {e}"))
    })?;
    Ok((BoundedTensor::new(union_lower, union_upper)?, false))
}

/// Screen a disjunctive child domain through clipping and bounding.
/// Part of #3740 Packet B and #4267.
#[allow(clippy::too_many_arguments)]
pub(super) fn screen_disjunctive_child(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    mut child_input: BoundedTensor,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn GemmEngine>,
    compute_bounds: &impl Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
    ) -> Result<MultiObjBounds>,
    warm_compute_bounds: Option<&WarmDisjunctiveComputeBoundsFn<'_>>,
    warm_alpha_telemetry: &WarmAlphaTelemetry,
    parent_domain: &MultiObjInputDomain,
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_clip: &mut usize,
) -> Result<()> {
    let mut complete_clip_node_bounds: Option<HashMap<String, BoundedTensor>> = None;

    if verifier.config.reorder_bab {
        // IBP first: cheap early exit for the vast majority of child domains.
        // For small models (lsnc_relu), clipping is expensive relative to IBP/CROWN
        // because relaxed_iterations × num_thresholds clip evaluations dominate.
        // Moving IBP before clipping avoids that overhead for IBP-verified domains.
        // Part of #4283.
        if verifier.config.input_split_ibp_enhancement {
            if let Some(ibp_bounds) = try_graph_spec_ibp_prescreen_bounds(
                graph,
                &child_input,
                spec_matrix,
                engine,
                None,
                "disjunctive reorder child",
            )? {
                let ibp_obj_bounds = extract_obj_bounds(&ibp_bounds, thresholds.len())?;
                if disjunctive_domain_verified(&ibp_obj_bounds, thresholds, clause_sizes) {
                    lifecycle.domains_verified += 1;
                    return Ok(());
                }
            }
        }

        // Clip only for domains IBP couldn't verify — tightens bounds before
        // deferring to the CROWN rebound batch.
        if verifier.config.enable_relaxed_clip {
            if let Some(linear_bounds) = parent_domain.linear_bounds.as_ref() {
                let shape = child_input.lower().shape().to_vec();
                match verifier.config.input_clip_type {
                    InputClipType::Relaxed => {
                        // Clause-aware (#disj-cross-clause-clip-unsat): per-clause
                        // intersection from the original box + union bbox, instead
                        // of the cross-clause sequential intersection.
                        let (clipped, all_refuted) = clause_aware_child_relaxed_clip(
                            verifier,
                            &child_input,
                            &shape,
                            linear_bounds,
                            thresholds,
                            clause_sizes,
                        )?;
                        if all_refuted {
                            *domains_verified_by_clip += 1;
                            lifecycle.domains_verified += 1;
                            return Ok(());
                        }
                        child_input = clipped;
                    }
                    InputClipType::Complete => {
                        let clip_outcome = verifier.complete_clip_with_precomputed_specs(
                            &child_input,
                            &shape,
                            linear_bounds,
                            thresholds,
                        )?;
                        child_input = clip_outcome.bounds;
                        complete_clip_node_bounds =
                            match super::super::super::clip_complete::build_graph_complete_clip_node_bounds(
                                graph,
                                &child_input,
                                linear_bounds,
                                thresholds,
                                verifier.config.verify_upper_bound,
                                verifier.config.clip_neuron_selection_ratio,
                                engine,
                            ) {
                                Ok(node_bounds) => node_bounds,
                                Err(err) => {
                                    trace!(
                                        "graph complete clip: skipping hidden-layer tightening due to {}",
                                        err
                                    );
                                    None
                                }
                            };
                    }
                }
            }
        }

        queue.push(MultiObjInputDomain {
            input_bounds: Arc::new(child_input),
            obj_bounds: parent_domain.obj_bounds.clone(),
            linear_bounds: None,
            depth: parent_domain.depth + 1,
            priority: parent_domain.priority,
            needs_bounding: true,
            node_bounds_override: complete_clip_node_bounds.map(Arc::new),
            // Carry the parent's refined α slopes forward unchanged so the
            // deferred-rebound warm-α overlay can refine per-domain (cgan
            // step-2C/2D). `None` unless the root was seeded (gated on
            // `input_split_alpha_iteration > 0`).
            inherited_alpha_state: parent_domain.inherited_alpha_state.clone(),
        });
        return Ok(());
    }

    let mut running_lower_floor: Vec<f32> = parent_domain
        .obj_bounds
        .iter()
        .map(|(lower, _)| *lower)
        .collect();
    let (mut obj_bounds, mut linear, mut child_alpha) = compute_child_bounds(
        &child_input,
        None,
        compute_bounds,
        warm_compute_bounds,
        parent_domain.inherited_alpha_state.as_ref(),
        warm_alpha_telemetry,
    )?;
    apply_running_lower_floor(&mut obj_bounds, &mut running_lower_floor);
    if disjunctive_domain_verified(&obj_bounds, thresholds, clause_sizes) {
        lifecycle.domains_verified += 1;
        return Ok(());
    }

    if verifier.config.enable_relaxed_clip {
        if let Some(ref linear_bounds) = linear {
            let shape = child_input.lower().shape().to_vec();
            match verifier.config.input_clip_type {
                InputClipType::Relaxed => {
                    // Clause-aware (#disj-cross-clause-clip-unsat): per-clause
                    // intersection from the original box + union bbox.
                    let (clipped, all_refuted) = clause_aware_child_relaxed_clip(
                        verifier,
                        &child_input,
                        &shape,
                        linear_bounds,
                        thresholds,
                        clause_sizes,
                    )?;
                    if all_refuted {
                        *domains_verified_by_clip += 1;
                        lifecycle.domains_verified += 1;
                        return Ok(());
                    }
                    child_input = clipped;
                    (obj_bounds, linear, child_alpha) = compute_child_bounds(
                        &child_input,
                        None,
                        compute_bounds,
                        warm_compute_bounds,
                        child_alpha.as_ref(),
                        warm_alpha_telemetry,
                    )?;
                    apply_running_lower_floor(&mut obj_bounds, &mut running_lower_floor);
                    if disjunctive_domain_verified(&obj_bounds, thresholds, clause_sizes) {
                        *domains_verified_by_clip += 1;
                        lifecycle.domains_verified += 1;
                        return Ok(());
                    }
                }
                InputClipType::Complete => {
                    let clip_outcome = verifier.complete_clip_with_precomputed_specs(
                        &child_input,
                        &shape,
                        linear_bounds,
                        thresholds,
                    )?;
                    child_input = clip_outcome.bounds;

                    let complete_clip_node_bounds =
                        match super::super::super::clip_complete::build_graph_complete_clip_node_bounds(
                            graph,
                            &child_input,
                            linear_bounds,
                            thresholds,
                            verifier.config.verify_upper_bound,
                            verifier.config.clip_neuron_selection_ratio,
                            engine,
                        ) {
                            Ok(node_bounds) => node_bounds,
                            Err(err) => {
                                trace!(
                                    "graph complete clip: skipping hidden-layer tightening due to {}",
                                    err
                                );
                                None
                            }
                        };
                    (obj_bounds, linear, child_alpha) = compute_child_bounds(
                        &child_input,
                        complete_clip_node_bounds.as_ref(),
                        compute_bounds,
                        warm_compute_bounds,
                        child_alpha.as_ref(),
                        warm_alpha_telemetry,
                    )?;
                    apply_running_lower_floor(&mut obj_bounds, &mut running_lower_floor);
                    if disjunctive_domain_verified(&obj_bounds, thresholds, clause_sizes) {
                        *domains_verified_by_clip += 1;
                        lifecycle.domains_verified += 1;
                        return Ok(());
                    }
                }
            }
        }
    }

    // `obj_bounds` already contains the running parent/pre-clip/post-clip
    // monotonic lower floor applied before each grouped verification.
    let priority = disjunctive_domain_priority(&obj_bounds, thresholds, clause_sizes);
    queue.push(MultiObjInputDomain {
        input_bounds: Arc::new(child_input),
        obj_bounds,
        linear_bounds: linear,
        depth: parent_domain.depth + 1,
        priority,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: child_alpha,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use ndarray::{arr1, arr2};
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;

    use super::*;
    use crate::beta_crown::config::{BetaCrownConfig, InputClipType};
    use crate::bounds::LinearBounds;
    use crate::layers::ReLULayer;
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer};

    fn complete_reorder_test_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "hidden",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("hidden linear")),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["hidden".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("out linear")),
            vec!["relu".to_string()],
        ));
        graph.set_output("out");
        graph
    }

    fn grouped_parent_domain(
        child_input: &BoundedTensor,
        linear_bounds: LinearBounds,
    ) -> MultiObjInputDomain {
        MultiObjInputDomain {
            input_bounds: Arc::new(child_input.clone()),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(linear_bounds),
            depth: 0,
            priority: 1.0,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: None,
        }
    }

    fn default_grouped_linear_bounds() -> LinearBounds {
        LinearBounds {
            lower_a: arr2(&[[1.0_f32], [1.0_f32]]),
            lower_b: arr1(&[0.0_f32, 0.0_f32]),
            upper_a: arr2(&[[1.0_f32], [1.0_f32]]),
            upper_b: arr1(&[0.0_f32, 0.0_f32]),
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    fn three_clause_linear_bounds() -> LinearBounds {
        LinearBounds {
            lower_a: arr2(&[[1.0_f32], [1.0_f32], [1.0_f32]]),
            lower_b: arr1(&[0.0_f32, 0.0_f32, 0.0_f32]),
            upper_a: arr2(&[[1.0_f32], [1.0_f32], [1.0_f32]]),
            upper_b: arr1(&[0.0_f32, 0.0_f32, 0.0_f32]),
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    fn row_local_verified_linear_bounds() -> LinearBounds {
        LinearBounds {
            lower_a: arr2(&[[0.0_f32], [0.0_f32]]),
            lower_b: arr1(&[0.3_f32, -0.1_f32]),
            upper_a: arr2(&[[0.0_f32], [0.0_f32]]),
            upper_b: arr1(&[0.3_f32, -0.1_f32]),
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    fn run_reorder_child_screen(
        verifier: &BetaCrownVerifier,
        parent_domain: MultiObjInputDomain,
        compute_bounds: &impl Fn(
            &BoundedTensor,
            Option<&HashMap<String, BoundedTensor>>,
        ) -> Result<MultiObjBounds>,
    ) -> (Option<MultiObjInputDomain>, usize, usize) {
        let graph = complete_reorder_test_graph();
        let child_input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("finite child bounds");
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
        let mut domains_verified_by_clip = 0usize;
        let warm_alpha_telemetry = WarmAlphaTelemetry::new(false);

        screen_disjunctive_child(
            verifier,
            &graph,
            child_input,
            &arr2(&[[1.0_f32], [1.0_f32]]),
            &[0.2, 0.2],
            &[1, 1],
            None,
            compute_bounds,
            None,
            &warm_alpha_telemetry,
            &parent_domain,
            &mut queue,
            &mut lifecycle,
            &mut domains_verified_by_clip,
        )
        .expect("reorder grouped child screen should not error");

        (
            queue.pop(),
            lifecycle.domains_verified,
            domains_verified_by_clip,
        )
    }

    #[test]
    fn test_screen_disjunctive_child_reorder_complete_defers_with_node_bounds_override_4267() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip: true,
            input_clip_type: InputClipType::Complete,
            reorder_bab: true,
            input_split_ibp_enhancement: false,
            ..Default::default()
        });
        let child_input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("finite child bounds");
        let parent_domain = grouped_parent_domain(&child_input, default_grouped_linear_bounds());

        let (child, domains_verified, domains_verified_by_clip) =
            run_reorder_child_screen(&verifier, parent_domain, &|_, _| {
                panic!("reorder grouped screening must not eagerly re-run child CROWN")
            });

        let child = child.expect("child should remain unresolved");
        assert_eq!(domains_verified, 0);
        assert_eq!(domains_verified_by_clip, 0);
        assert!(
            child.needs_bounding,
            "reorder_bab must still defer grouped child bounding in complete mode"
        );
        assert!(
            child.node_bounds_override.is_some(),
            "complete clipping should carry node-bounds overrides into the deferred grouped child"
        );
        assert!(
            child.linear_bounds.is_none(),
            "deferred grouped children should clear linear bounds until the batch rebound pass"
        );
    }

    /// Verify that the IBP-first early exit in the reorder_bab path (#4283)
    /// actually fires. Both existing tests set `input_split_ibp_enhancement: false`,
    /// so lines 51-59 of screen_child.rs had zero test coverage.
    ///
    /// Constructs a narrow positive input domain where IBP trivially verifies,
    /// and asserts the domain is verified without calling `compute_bounds`.
    #[test]
    fn test_screen_disjunctive_child_reorder_ibp_first_early_exit_4283() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip: false,
            input_clip_type: InputClipType::Relaxed,
            reorder_bab: true,
            input_split_ibp_enhancement: true,
            ..Default::default()
        });

        // Narrow positive domain: after linear(W=1)→relu→linear(W=1), IBP gives [0.5, 1.0].
        // With thresholds [-0.1, -0.1], lower 0.5 > -0.1 so IBP verifies both clauses.
        let child_input =
            BoundedTensor::new(arr1(&[0.5_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("finite child bounds");
        let parent_domain = grouped_parent_domain(&child_input, default_grouped_linear_bounds());

        let graph = complete_reorder_test_graph();
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
        let mut domains_verified_by_clip = 0usize;
        let warm_alpha_telemetry = WarmAlphaTelemetry::new(false);

        screen_disjunctive_child(
            &verifier,
            &graph,
            child_input,
            &arr2(&[[1.0_f32], [1.0_f32]]),
            &[-0.1, -0.1],
            &[1, 1],
            None,
            &|_, _| panic!("IBP-first early exit must not call compute_bounds"),
            None,
            &warm_alpha_telemetry,
            &parent_domain,
            &mut queue,
            &mut lifecycle,
            &mut domains_verified_by_clip,
        )
        .expect("IBP-first reorder screen should not error");

        assert_eq!(
            lifecycle.domains_verified, 1,
            "IBP should verify the narrow positive domain without needing CROWN"
        );
        assert!(
            queue.is_empty(),
            "verified domain should not be enqueued for further processing"
        );
        assert_eq!(
            domains_verified_by_clip, 0,
            "IBP verification should not count as clip verification"
        );
    }

    /// #disj-cross-clause-clip-unsat regression at the per-child screen level:
    /// two single-row OR clauses whose feasible half-intervals are DISJOINT
    /// (row0 keeps x<=0.3, row1 keeps x>=0.7). The historical loop clipped the
    /// child box against both rows in sequence (cross-clause), collapsing it to
    /// x<=0.3 and DISCARDING the x>=0.7 half — a genuine counterexample region
    /// (the lsnc false-unsat). The clause-aware clip must keep the child (each
    /// clause is individually feasible) with a UNION box that still reaches the
    /// far edge (x up to ~1.0), so no counterexample is carved away.
    #[test]
    fn test_screen_disjunctive_child_reorder_clause_aware_union_keeps_disjoint_clause() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip: true,
            input_clip_type: InputClipType::Relaxed,
            reorder_bab: true,
            input_split_ibp_enhancement: false,
            relaxed_clip_iterations: 20,
            ..Default::default()
        });
        // Row 0 (clause 0): x <= 0.3 ; Row 1 (clause 1): -x <= -0.7 -> x >= 0.7.
        let linear_bounds = LinearBounds {
            lower_a: arr2(&[[1.0_f32], [-1.0_f32]]),
            lower_b: arr1(&[0.0_f32, 0.0_f32]),
            upper_a: arr2(&[[1.0_f32], [-1.0_f32]]),
            upper_b: arr1(&[0.0_f32, 0.0_f32]),
            lower_a_err: None,
            upper_a_err: None,
        };
        let child_input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("finite child bounds");
        let parent_domain = grouped_parent_domain(&child_input, linear_bounds);

        let graph = complete_reorder_test_graph();
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
        let mut domains_verified_by_clip = 0usize;
        let warm_alpha_telemetry = WarmAlphaTelemetry::new(false);

        screen_disjunctive_child(
            &verifier,
            &graph,
            child_input,
            &arr2(&[[1.0_f32], [-1.0_f32]]),
            &[0.3, -0.7],
            &[1, 1],
            None,
            &|_, _| panic!("reorder path must not call compute_bounds"),
            None,
            &warm_alpha_telemetry,
            &parent_domain,
            &mut queue,
            &mut lifecycle,
            &mut domains_verified_by_clip,
        )
        .expect("clause-aware screen should not error");

        assert_eq!(
            lifecycle.domains_verified, 0,
            "two individually-feasible OR clauses must not verify"
        );
        assert_eq!(domains_verified_by_clip, 0);
        let child = queue.pop().expect("survivor must reach the queue");
        // The union box must still reach the far (x>=0.7) clause's edge; the old
        // cross-clause clip would have collapsed the upper bound to ~0.3.
        assert!(
            child.input_bounds.upper()[[0]] >= 0.7,
            "union box must enclose the x>=0.7 clause (got upper {})",
            child.input_bounds.upper()[[0]]
        );
    }

    #[test]
    fn test_screen_disjunctive_child_reorder_ignores_single_row_clip_verification_4267() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip: true,
            input_clip_type: InputClipType::Relaxed,
            reorder_bab: true,
            input_split_ibp_enhancement: false,
            ..Default::default()
        });
        let compute_calls = Cell::new(0usize);
        let child_input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("finite child bounds");
        let parent_domain = grouped_parent_domain(&child_input, row_local_verified_linear_bounds());

        let (child, domains_verified, domains_verified_by_clip) =
            run_reorder_child_screen(&verifier, parent_domain, &|_, _| {
                compute_calls.set(compute_calls.get() + 1);
                Ok((
                    vec![(-1.0, 1.0), (-1.0, 1.0)],
                    Some(default_grouped_linear_bounds()),
                ))
            });

        let child = child.expect("one satisfied row must not discharge the whole grouped child");
        assert_eq!(
            compute_calls.get(),
            0,
            "reorder grouped path should reuse parent linear bounds instead of eagerly bounding the child"
        );
        assert_eq!(
            domains_verified,
            0,
            "grouped reorder path must not count a single satisfied row as full-domain verification"
        );
        assert_eq!(
            domains_verified_by_clip, 0,
            "grouped reorder clip must treat row-local verification as tightening only"
        );
        assert!(
            child.needs_bounding,
            "grouped reorder child must stay deferred after parent-linear clipping"
        );
        assert!(
            child.node_bounds_override.is_none(),
            "relaxed grouped clipping should not synthesize node-bounds overrides"
        );
    }

    fn eager_child_input() -> BoundedTensor {
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("finite child bounds")
    }

    fn eager_parent_domain(
        child_input: &BoundedTensor,
        linear_bounds: Option<LinearBounds>,
        parent_alpha: Arc<GraphAlphaState>,
    ) -> MultiObjInputDomain {
        MultiObjInputDomain {
            input_bounds: Arc::new(child_input.clone()),
            obj_bounds: vec![(0.3_f32, 1.0_f32), (0.35_f32, 1.0_f32)],
            linear_bounds,
            depth: 0,
            priority: 1.0,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: Some(parent_alpha),
        }
    }

    #[test]
    fn test_compute_child_bounds_default_is_frozen_and_state_free() {
        let child_input = eager_child_input();
        let frozen_calls = Cell::new(0usize);
        let telemetry = WarmAlphaTelemetry::new(false);
        let (obj_bounds, linear, child_alpha) = compute_child_bounds(
            &child_input,
            None,
            &|_, _| {
                frozen_calls.set(frozen_calls.get() + 1);
                Ok((vec![(-0.5_f32, 0.5_f32)], None))
            },
            None,
            None,
            &telemetry,
        )
        .expect("default frozen bound should succeed");

        assert_eq!(frozen_calls.get(), 1);
        assert_eq!(obj_bounds, vec![(-0.5_f32, 0.5_f32)]);
        assert!(linear.is_none());
        assert!(child_alpha.is_none());
        assert_eq!(telemetry.snapshot(), (0, 0, 0, 0));
    }

    #[test]
    fn test_screen_disjunctive_child_eager_warm_alpha_completes_and_stores_state() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip: false,
            reorder_bab: false,
            ..Default::default()
        });
        let graph = complete_reorder_test_graph();
        let child_input = eager_child_input();
        let parent_alpha = Arc::new(GraphAlphaState::default());
        let parent = eager_parent_domain(&child_input, None, parent_alpha.clone());
        let warm_calls = Cell::new(0usize);
        let warm = |_: &BoundedTensor,
                    _: Option<&HashMap<String, BoundedTensor>>,
                    seed: &GraphAlphaState|
         -> Result<WarmDisjunctiveBoundsResult> {
            warm_calls.set(warm_calls.get() + 1);
            assert!(std::ptr::eq(seed, parent_alpha.as_ref()));
            Ok((
                vec![(0.1_f32, 0.8_f32), (0.2_f32, 0.9_f32)],
                None,
                GraphAlphaState::default(),
            ))
        };
        let telemetry = WarmAlphaTelemetry::new(true);
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
        let mut domains_verified_by_clip = 0usize;

        screen_disjunctive_child(
            &verifier,
            &graph,
            child_input,
            &arr2(&[[1.0_f32], [-1.0_f32]]),
            &[0.4_f32, 0.4_f32],
            &[1, 1],
            None,
            &|_, _| -> Result<MultiObjBounds> {
                panic!("frozen bounds must not run after a completed warm call")
            },
            Some(&warm),
            &telemetry,
            &parent,
            &mut queue,
            &mut lifecycle,
            &mut domains_verified_by_clip,
        )
        .expect("eager grouped warm-alpha screen should succeed");

        assert_eq!(warm_calls.get(), 1);
        assert_eq!(telemetry.snapshot(), (1, 1, 0, 0));
        assert_eq!(
            telemetry.marker("final"),
            "NY_WARM_ALPHA route=grouped-disjunctive-eager status=final attempts=1 warm_bound_completions=1 warm_failures=0 frozen_fallback_completions=0"
        );
        let child = queue.pop().expect("unresolved child should be queued");
        assert_eq!(
            child.obj_bounds,
            vec![(0.3_f32, 0.8_f32), (0.35_f32, 0.9_f32)],
            "warm bounds must retain the grouped parent monotonicity floor"
        );
        let child_alpha = child
            .inherited_alpha_state
            .expect("returned alpha state must be stored on the child");
        assert!(!Arc::ptr_eq(&child_alpha, &parent_alpha));
    }

    #[test]
    fn test_screen_disjunctive_child_warm_error_uses_frozen_and_carries_prior_alpha() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip: false,
            reorder_bab: false,
            ..Default::default()
        });
        let graph = complete_reorder_test_graph();
        let child_input = eager_child_input();
        let parent_alpha = Arc::new(GraphAlphaState::default());
        let parent = eager_parent_domain(&child_input, None, parent_alpha.clone());
        let warm_calls = Cell::new(0usize);
        let frozen_calls = Cell::new(0usize);
        let warm = |_: &BoundedTensor,
                    _: Option<&HashMap<String, BoundedTensor>>,
                    _: &GraphAlphaState|
         -> Result<WarmDisjunctiveBoundsResult> {
            warm_calls.set(warm_calls.get() + 1);
            Err(NyError::InvalidSpec("forced warm failure".to_string()))
        };
        let telemetry = WarmAlphaTelemetry::new(true);
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
        let mut domains_verified_by_clip = 0usize;

        screen_disjunctive_child(
            &verifier,
            &graph,
            child_input,
            &arr2(&[[1.0_f32], [-1.0_f32]]),
            &[0.4_f32, 0.4_f32],
            &[1, 1],
            None,
            &|_, _| {
                frozen_calls.set(frozen_calls.get() + 1);
                Ok((vec![(0.1_f32, 0.8_f32), (0.2_f32, 0.9_f32)], None))
            },
            Some(&warm),
            &telemetry,
            &parent,
            &mut queue,
            &mut lifecycle,
            &mut domains_verified_by_clip,
        )
        .expect("warm failure must fall back to frozen bounds");

        assert_eq!(warm_calls.get(), 1);
        assert_eq!(frozen_calls.get(), 1);
        assert_eq!(telemetry.snapshot(), (1, 0, 1, 1));
        let child_alpha = queue
            .pop()
            .expect("fallback child should be queued")
            .inherited_alpha_state
            .expect("fallback child must retain its prior alpha state");
        assert!(
            Arc::ptr_eq(&child_alpha, &parent_alpha),
            "fallback must carry the exact prior alpha Arc"
        );
    }

    #[test]
    fn test_compute_child_bounds_warm_and_frozen_errors_do_not_claim_fallback_completion() {
        let child_input = eager_child_input();
        let parent_alpha = Arc::new(GraphAlphaState::default());
        let warm = |_: &BoundedTensor,
                    _: Option<&HashMap<String, BoundedTensor>>,
                    _: &GraphAlphaState|
         -> Result<WarmDisjunctiveBoundsResult> {
            Err(NyError::InvalidSpec("forced warm failure".to_string()))
        };
        let telemetry = WarmAlphaTelemetry::new(true);

        let result = compute_child_bounds(
            &child_input,
            None,
            &|_, _| {
                Err(NyError::InvalidSpec(
                    "forced frozen fallback failure".to_string(),
                ))
            },
            Some(&warm),
            Some(&parent_alpha),
            &telemetry,
        );

        assert!(result.is_err(), "frozen fallback failure must fail closed");
        assert_eq!(
            telemetry.snapshot(),
            (1, 0, 1, 0),
            "a failed frozen bound must not be reported as a completed fallback"
        );
    }

    fn assert_eager_clip_refreshes_to_newest_alpha(input_clip_type: InputClipType) {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip: true,
            input_clip_type,
            reorder_bab: false,
            ..Default::default()
        });
        let graph = complete_reorder_test_graph();
        let child_input = eager_child_input();
        let parent_alpha = Arc::new(GraphAlphaState::default());
        let parent = eager_parent_domain(
            &child_input,
            Some(default_grouped_linear_bounds()),
            parent_alpha.clone(),
        );
        let warm_calls = Cell::new(0usize);
        let second_seed = Cell::new(std::ptr::null::<GraphAlphaState>());
        let warm = |_: &BoundedTensor,
                    _: Option<&HashMap<String, BoundedTensor>>,
                    seed: &GraphAlphaState|
         -> Result<WarmDisjunctiveBoundsResult> {
            let call = warm_calls.get();
            warm_calls.set(call + 1);
            if call == 0 {
                assert!(std::ptr::eq(seed, parent_alpha.as_ref()));
            } else if call == 1 {
                assert!(!std::ptr::eq(seed, parent_alpha.as_ref()));
                second_seed.set(std::ptr::from_ref(seed));
            }
            Ok((
                vec![(-1.0_f32, 1.0_f32), (-1.0_f32, 1.0_f32)],
                Some(default_grouped_linear_bounds()),
                GraphAlphaState::default(),
            ))
        };
        let telemetry = WarmAlphaTelemetry::new(true);
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
        let mut domains_verified_by_clip = 0usize;

        screen_disjunctive_child(
            &verifier,
            &graph,
            child_input,
            &arr2(&[[1.0_f32], [1.0_f32]]),
            &[0.4_f32, 0.4_f32],
            &[1, 1],
            None,
            &|_, _| -> Result<MultiObjBounds> {
                panic!("completed warm calls must not use frozen bounds")
            },
            Some(&warm),
            &telemetry,
            &parent,
            &mut queue,
            &mut lifecycle,
            &mut domains_verified_by_clip,
        )
        .expect("eager grouped clip refresh should succeed");

        assert_eq!(warm_calls.get(), 2, "initial + post-clip warm bounds");
        assert_eq!(telemetry.snapshot(), (2, 2, 0, 0));
        let child_alpha = queue
            .pop()
            .expect("post-clip child should remain unresolved")
            .inherited_alpha_state
            .expect("post-clip child must retain returned alpha state");
        assert_ne!(second_seed.get(), std::ptr::null());
        assert_ne!(
            Arc::as_ptr(&child_alpha),
            second_seed.get(),
            "queued child must store the second returned state, not its first seed"
        );
    }

    #[test]
    fn test_screen_disjunctive_child_eager_clip_uses_newest_alpha_state() {
        assert_eager_clip_refreshes_to_newest_alpha(InputClipType::Relaxed);
        assert_eager_clip_refreshes_to_newest_alpha(InputClipType::Complete);
    }

    #[test]
    fn test_screen_disjunctive_child_combines_parent_pre_and_post_clip_clause_floors() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip: true,
            input_clip_type: InputClipType::Relaxed,
            reorder_bab: false,
            ..Default::default()
        });
        let graph = complete_reorder_test_graph();
        let child_input = eager_child_input();
        let parent = MultiObjInputDomain {
            input_bounds: Arc::new(child_input.clone()),
            // Only clause 0 is discharged by the parent.
            obj_bounds: vec![(0.3_f32, 1.0_f32), (-1.0_f32, 1.0_f32), (-1.0_f32, 1.0_f32)],
            linear_bounds: Some(three_clause_linear_bounds()),
            depth: 0,
            priority: 1.0,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: None,
        };
        let compute_calls = Cell::new(0usize);
        let compute = |_: &BoundedTensor,
                       _: Option<&HashMap<String, BoundedTensor>>|
         -> Result<MultiObjBounds> {
            let call = compute_calls.get();
            compute_calls.set(call + 1);
            match call {
                // The pre-clip pass discharges only clause 1.
                0 => Ok((
                    vec![(-1.0_f32, 1.0_f32), (0.3_f32, 1.0_f32), (-1.0_f32, 1.0_f32)],
                    Some(three_clause_linear_bounds()),
                )),
                // The post-clip pass discharges only clause 2. No single
                // vector verifies, but their running lower floor does.
                1 => Ok((
                    vec![(-1.0_f32, 1.0_f32), (-1.0_f32, 1.0_f32), (0.3_f32, 1.0_f32)],
                    Some(three_clause_linear_bounds()),
                )),
                _ => panic!("combined clause floor should finish after two passes"),
            }
        };
        let telemetry = WarmAlphaTelemetry::new(false);
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
        let mut domains_verified_by_clip = 0usize;

        screen_disjunctive_child(
            &verifier,
            &graph,
            child_input,
            &arr2(&[[1.0_f32], [1.0_f32], [1.0_f32]]),
            &[0.2_f32, 0.2_f32, 0.2_f32],
            &[1, 1, 1],
            None,
            &compute,
            None,
            &telemetry,
            &parent,
            &mut queue,
            &mut lifecycle,
            &mut domains_verified_by_clip,
        )
        .expect("running grouped clause floor should verify the child");

        assert_eq!(compute_calls.get(), 2, "pre-clip + post-clip bounds");
        assert!(
            queue.is_empty(),
            "jointly verified child must not be queued"
        );
        assert_eq!(lifecycle.domains_verified, 1);
        assert_eq!(domains_verified_by_clip, 1);
    }

    #[test]
    fn test_screen_disjunctive_child_reorder_defers_and_carries_parent_alpha_f8() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip: false,
            reorder_bab: true,
            ..Default::default()
        });
        let graph = complete_reorder_test_graph();
        let child_input = eager_child_input();
        let parent_alpha = Arc::new(GraphAlphaState::default());
        let parent = eager_parent_domain(&child_input, None, Arc::clone(&parent_alpha));
        let telemetry = WarmAlphaTelemetry::new(true);
        let warm_calls = Cell::new(0usize);
        let warm = |_: &BoundedTensor,
                    _: Option<&HashMap<String, BoundedTensor>>,
                    _: &GraphAlphaState|
         -> Result<WarmDisjunctiveBoundsResult> {
            warm_calls.set(warm_calls.get() + 1);
            panic!("reordered grouped-disjunctive screening must remain frozen")
        };
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
        let mut domains_verified_by_clip = 0usize;

        screen_disjunctive_child(
            &verifier,
            &graph,
            child_input,
            &arr2(&[[1.0_f32], [-1.0_f32]]),
            &[0.4_f32, 0.4_f32],
            &[1, 1],
            None,
            &|_, _| -> Result<MultiObjBounds> {
                panic!("reordered screening must defer frozen bounds")
            },
            Some(&warm),
            &telemetry,
            &parent,
            &mut queue,
            &mut lifecycle,
            &mut domains_verified_by_clip,
        )
        .expect("reordered grouped screen should remain deferred");

        assert_eq!(warm_calls.get(), 0);
        assert_eq!(telemetry.snapshot(), (0, 0, 0, 0));
        let child = queue.pop().expect("reordered child should be deferred");
        assert!(child.needs_bounding);
        let carried = child
            .inherited_alpha_state
            .as_ref()
            .expect("reordered child must carry parent alpha into deferred rebound");
        assert!(
            Arc::ptr_eq(carried, &parent_alpha),
            "reordered screening must preserve the exact parent alpha Arc"
        );
    }
}
