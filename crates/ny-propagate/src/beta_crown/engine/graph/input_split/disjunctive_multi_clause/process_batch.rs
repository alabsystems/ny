// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant as FreshClipInstant};

use ndarray::{Array2, ArrayD};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::trace;

use crate::beta_crown::config::InputClipType;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::bounds::LinearBounds;
use crate::GraphNetwork;

use super::super::fresh_domain_clip::{
    clip_with_fresh_exact_domain_planes, FreshDomainClipResult, FreshDomainClipStatus,
    FreshDomainClipTelemetry,
};
use super::super::grouped_semantics::{disjunctive_domain_verified, valid_disjunctive_layout};
use super::super::ibp_prescreen_flat::batched_ibp_prescreen_from_flat;
use super::super::shared::{build_child_input, MultiObjBounds, MultiObjInputDomain};
use super::push_survivors::{push_batched_relaxed_survivors, push_fallback_survivors};
use super::screen_child::{
    screen_disjunctive_child, WarmAlphaTelemetry, WarmDisjunctiveComputeBoundsFn,
};

pub(super) fn graph_ibp_prescreen_error_should_skip(err: &NyError) -> bool {
    match err {
        NyError::ShapeMismatch { .. }
        | NyError::UnsupportedOp(_)
        | NyError::UnsupportedConfiguration(_)
        | NyError::NumericalInstability(_)
        | NyError::DeadlineExceeded(_)
        | NyError::InfeasibleDomain(_) => true,
        NyError::InvalidSpec(message) => message.contains("empty after clamping"),
        NyError::LayerError { source, .. } => graph_ibp_prescreen_error_should_skip(source),
        _ => false,
    }
}

/// Pending child with flat 1D lower/upper arrays. BoundedTensor construction
/// is deferred until after IBP prescreen filtering. Part of #4366 Packet A.
pub(super) struct FlatPendingChild {
    pub(super) flat_lower: ArrayD<f32>,
    pub(super) flat_upper: ArrayD<f32>,
    pub(super) obj_bounds: Vec<(f32, f32)>,
    pub(super) linear_bounds: Option<LinearBounds>,
    pub(super) depth: usize,
    pub(super) priority: f32,
    /// Parent's refined α slopes for the deferred-rebound warm-α overlay
    /// (cgan step-2C/2D). `None` (the default gate off) keeps the frozen path.
    pub(super) inherited_alpha_state: Option<Arc<crate::bounds::GraphAlphaState>>,
}

/// Revoke a clip result completed at or after the BaB deadline.
///
/// The affine proof remains mathematically sound, but strict budget semantics
/// give it no dispatcher authority. Returning the exact source as `Skipped`
/// also ensures telemetry cannot report a late terminal refutation.
fn decline_fresh_clip_after_deadline(
    source: &BoundedTensor,
    outcome: FreshDomainClipResult,
    deadline: FreshClipInstant,
    completed_at: FreshClipInstant,
) -> FreshDomainClipResult {
    if completed_at >= deadline {
        FreshDomainClipResult {
            bounds: source.clone(),
            status: FreshDomainClipStatus::Skipped,
        }
    } else {
        outcome
    }
}

/// Apply the default-dark fresh exact-domain clip to THIS popped domain.
///
/// The callback is invoked inside `clip_with_fresh_exact_domain_planes` with the
/// exact source box. Its planes are private to that primitive and are dropped
/// before this helper returns. The caller has already discarded the domain's
/// pre-existing batch rebound plane: this batch-stack-unsafe route gives that
/// plane no proof or split-scoring authority. The reordered child constructors
/// below also explicitly store `None`, so no old or fresh planes reach a child
/// clip, heuristic, or verdict path.
fn fresh_clip_current_domain<F>(
    verifier: &BetaCrownVerifier,
    domain: &mut MultiObjInputDomain,
    thresholds: &[f32],
    clause_sizes: &[usize],
    compute_bounds: &F,
    telemetry: &FreshDomainClipTelemetry,
    deadline: FreshClipInstant,
) -> bool
where
    F: Fn(&BoundedTensor, Option<&HashMap<String, BoundedTensor>>) -> Result<MultiObjBounds>,
{
    if !telemetry.enabled() {
        return false;
    }
    debug_assert!(verifier.config.reorder_bab);
    debug_assert!(verifier.config.input_split_ibp_enhancement);
    debug_assert!(
        !verifier.config.enable_relaxed_clip,
        "fresh current-domain clip must coexist with the legacy child clip disabled"
    );

    let source = Arc::clone(&domain.input_bounds);
    let outcome = clip_with_fresh_exact_domain_planes(
        source.as_ref(),
        thresholds,
        clause_sizes,
        verifier.config.verify_upper_bound,
        verifier.config.relaxed_clip_iterations,
        |exact_domain| {
            if FreshClipInstant::now() >= deadline {
                return Err(NyError::DeadlineExceeded(
                    "fresh exact-domain full-spec CROWN deadline expired before bounding".into(),
                ));
            }
            let (_objective_bounds, linear) = compute_bounds(exact_domain, None)?;
            if FreshClipInstant::now() >= deadline {
                return Err(NyError::DeadlineExceeded(
                    "fresh exact-domain full-spec CROWN deadline expired after bounding".into(),
                ));
            }
            linear.ok_or_else(|| {
                NyError::UnsupportedConfiguration(
                    "fresh exact-domain full-spec CROWN returned no affine planes".into(),
                )
            })
        },
    );
    let outcome = decline_fresh_clip_after_deadline(
        source.as_ref(),
        outcome,
        deadline,
        FreshClipInstant::now(),
    );
    telemetry.record(source.as_ref(), &outcome);
    match outcome.status {
        FreshDomainClipStatus::Applied => {
            domain.input_bounds = Arc::new(outcome.bounds);
            false
        }
        FreshDomainClipStatus::AllClausesRefuted => true,
        FreshDomainClipStatus::Skipped => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_disjunctive_domain_batch<F>(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domains: Vec<MultiObjInputDomain>,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn GemmEngine>,
    compute_bounds: &F,
    warm_compute_bounds: Option<&WarmDisjunctiveComputeBoundsFn<'_>>,
    warm_alpha_telemetry: &WarmAlphaTelemetry,
    fresh_domain_clip_telemetry: &FreshDomainClipTelemetry,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    bab_timeout: Duration,
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_clip: &mut usize,
) -> Result<Option<BetaCrownResult>>
where
    F: Fn(&BoundedTensor, Option<&HashMap<String, BoundedTensor>>) -> Result<MultiObjBounds>,
{
    debug_assert!(
        !verifier.config.reorder_bab || warm_compute_bounds.is_none(),
        "reordered grouped-disjunctive BaB must keep eager warm alpha disabled"
    );
    if verifier.config.reorder_bab && verifier.config.input_split_ibp_enhancement {
        process_reorder_prescreen_batch(
            verifier,
            graph,
            domains,
            spec_matrix,
            thresholds,
            clause_sizes,
            engine,
            compute_bounds,
            fresh_domain_clip_telemetry,
            mul_binary_alphas,
            bab_timeout,
            queue,
            lifecycle,
            domains_verified_by_clip,
        )
    } else {
        process_per_child_batch(
            verifier,
            graph,
            domains,
            spec_matrix,
            thresholds,
            clause_sizes,
            engine,
            compute_bounds,
            warm_compute_bounds,
            warm_alpha_telemetry,
            mul_binary_alphas,
            bab_timeout,
            queue,
            lifecycle,
            domains_verified_by_clip,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn process_reorder_prescreen_batch<F>(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domains: Vec<MultiObjInputDomain>,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn GemmEngine>,
    compute_bounds: &F,
    fresh_domain_clip_telemetry: &FreshDomainClipTelemetry,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    bab_timeout: Duration,
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_clip: &mut usize,
) -> Result<Option<BetaCrownResult>>
where
    F: Fn(&BoundedTensor, Option<&HashMap<String, BoundedTensor>>) -> Result<MultiObjBounds>,
{
    debug_assert!(
        !fresh_domain_clip_telemetry.enabled() || !verifier.config.enable_relaxed_clip,
        "fresh current-domain clipping requires the legacy child clip to stay disabled"
    );
    // #lsnc-child-batch (S1): the consolidated child pipeline replaces the
    // per-child FlatPendingChild clone chain (split -> prescreen -> clip ->
    // push) for the batched-relaxed-clip lane. Bit-parity class; the body
    // below stays in-tree as the A/B + parity reference
    // (`test_child_batch_reorder_prescreen_parity_lsnc_s1`). Only the
    // relaxed-clip configuration is eligible; Complete/disabled clip takes
    // the unchanged reference path.
    let batched_relaxed_eligible = verifier.config.enable_relaxed_clip
        && matches!(verifier.config.input_clip_type, InputClipType::Relaxed);
    if batched_relaxed_eligible && super::child_batch::input_split_child_batch_enabled() {
        return super::child_batch::process_reorder_prescreen_child_batch(
            verifier,
            graph,
            domains,
            spec_matrix,
            thresholds,
            clause_sizes,
            engine,
            bab_timeout,
            queue,
            lifecycle,
            domains_verified_by_clip,
        );
    }

    let mut pending_children: Vec<FlatPendingChild> = Vec::new();
    // Track the child shape from the first domain that produces children.
    // All domains in a batch share the same input shape.
    let mut child_shape: Option<Vec<usize>> = None;
    // #relational-bab option B: per-wave α edge-pass budget.
    let mut edge_alpha_budget = verifier.config.input_split_edge_alpha_top;

    for mut domain in domains {
        if lifecycle.start_time.elapsed() > bab_timeout {
            return Ok(Some(lifecycle.timeout_result()));
        }
        if lifecycle.domains_explored >= verifier.config.max_domains {
            return Ok(Some(lifecycle.build_result(
                BabVerificationStatus::Unknown {
                    reason: format!(
                        "Domain limit {}: {}/{} verified",
                        verifier.config.max_domains,
                        lifecycle.domains_verified,
                        lifecycle.domains_explored
                    ),
                },
            )));
        }

        lifecycle.domains_explored += 1;
        lifecycle.max_depth_reached = lifecycle.max_depth_reached.max(domain.depth);

        if fresh_domain_clip_telemetry.enabled() {
            // This route exists precisely because the graph's domain-stacked
            // rebound planes are not trusted. Drop the old carrier before any
            // edge helper or branch heuristic can consume it. The fresh
            // non-domain-stacked full-spec planes remain private to the clip
            // primitive and are dropped there as well, so SB intentionally
            // takes its documented width-only fallback below.
            domain.linear_bounds = None;
        }

        if lifecycle.domains_explored.is_multiple_of(1000) || lifecycle.domains_explored <= 5 {
            trace!(
                "[disjunctive-multi-clause] explored={} verified={} clipped={} depth={} queue={} pri={:.4}",
                lifecycle.domains_explored,
                lifecycle.domains_verified,
                *domains_verified_by_clip,
                domain.depth,
                queue.len(),
                domain.priority,
            );
        }

        if disjunctive_domain_verified(&domain.obj_bounds, thresholds, clause_sizes) {
            lifecycle.domains_verified += 1;
            continue;
        }
        // #relational-bab option B (config-gated, default inert): the α-slope
        // pass first — optimized lower bounds over the exact sub-box, budget
        // -capped per wave (pops are worst-gap-first, so the cap keeps the
        // most negative gaps). Verified ⇒ done; still-short ⇒ the domain
        // continues with the α-improved, monotonicity-guarded bounds into
        // the MILP finisher / split path below.
        if verifier.config.input_split_edge_alpha && edge_alpha_budget > 0 {
            if let Some(row_indices) = edge_domain_rows(
                &domain,
                thresholds,
                clause_sizes,
                verifier.config.input_split_edge_milp_gap,
                verifier.config.input_split_edge_milp_depth,
                spec_matrix.nrows(),
            ) {
                edge_alpha_budget -= 1;
                if let Some(fresh) = try_edge_alpha_pass(
                    verifier,
                    graph,
                    &domain,
                    spec_matrix,
                    &row_indices,
                    engine,
                    lifecycle.deadline(bab_timeout),
                ) {
                    domain.obj_bounds =
                        super::super::batching::tighten_obj_lower_bounds(&domain.obj_bounds, fresh);
                    if disjunctive_domain_verified(&domain.obj_bounds, thresholds, clause_sizes) {
                        lifecycle.domains_verified += 1;
                        continue;
                    }
                }
            }
        }
        // #relational-bab EDGE-DOMAIN ESCALATION (config-gated + oracle
        // -attached, default inert): a near-verified deep domain is offered
        // to the exact Graph-MIP leaf oracle BEFORE further splitting (and
        // before the max_depth drop). `VerifiedAllRows` is certified-UNSAT
        // -only per the oracle contract; anything else falls through to the
        // unchanged split path.
        if try_edge_milp_escalation(
            verifier,
            graph,
            &domain,
            spec_matrix,
            thresholds,
            clause_sizes,
            engine,
            lifecycle.deadline(bab_timeout),
        ) {
            lifecycle.domains_verified += 1;
            continue;
        }
        if fresh_domain_clip_telemetry.enabled() {
            let fresh_clip_deadline = lifecycle.deadline(bab_timeout);
            if fresh_clip_current_domain(
                verifier,
                &mut domain,
                thresholds,
                clause_sizes,
                compute_bounds,
                fresh_domain_clip_telemetry,
                fresh_clip_deadline,
            ) {
                *domains_verified_by_clip += 1;
                lifecycle.domains_verified += 1;
                continue;
            }
            // A late full-spec bound is converted to `Skipped` inside the
            // callback. Return through the ordinary timeout result before
            // spending more time splitting or prescreening that unchanged
            // domain.
            if FreshClipInstant::now() >= fresh_clip_deadline {
                return Ok(Some(lifecycle.timeout_result()));
            }
        }
        if domain.depth >= verifier.config.max_depth {
            lifecycle.unresolved_due_to_depth = true;
            continue;
        }

        let domain_bounds: Vec<f32> = domain
            .obj_bounds
            .iter()
            .map(|(lower, upper)| {
                if verifier.config.verify_upper_bound {
                    *upper
                } else {
                    *lower
                }
            })
            .collect();
        let split_dim = verifier.select_input_dimension_sb(
            domain.input_bounds.as_ref(),
            domain.linear_bounds.as_ref(),
            Some(domain_bounds.as_slice()),
            Some(thresholds),
        );
        let flat = domain.input_bounds.as_ref().flatten();

        let unsplittable = split_dim >= flat.len() || {
            let l = flat.lower()[[split_dim]];
            let u = flat.upper()[[split_dim]];
            !l.is_finite() || !u.is_finite() || u <= l
        };
        if unsplittable {
            // #lsnc-f64-tail call site 2 (design §6.3): one certified f64
            // last chance BEFORE the unresolved drop — these are exactly the
            // queue-drain leaks of the precision-limited lsnc instances.
            // Gate `NY_F64_TAIL=1` (default OFF => `false` with no work).
            if super::super::f64_tail::f64_tail_last_chance(
                graph,
                &domain,
                spec_matrix,
                thresholds,
                clause_sizes,
                mul_binary_alphas,
                engine,
                lifecycle.deadline(bab_timeout),
            ) {
                lifecycle.domains_verified += 1;
                continue;
            }
            lifecycle.unresolved_due_to_unsplittable = true;
            continue;
        }
        let l = flat.lower()[[split_dim]];
        let u = flat.upper()[[split_dim]];

        let mid = l + (u - l) / 2.0;
        let shape = domain.input_bounds.as_ref().lower().shape().to_vec();
        if child_shape.is_none() {
            child_shape = Some(shape.clone());
        }
        let mut child_lower = flat.lower().clone();
        let mut child_upper = flat.upper().clone();

        // Left child: [l, mid] on split_dim. Store flat — no reshape yet.
        // Part of #4366 Packet A.
        child_lower[[split_dim]] = l;
        child_upper[[split_dim]] = mid;
        pending_children.push(FlatPendingChild {
            flat_lower: child_lower.clone(),
            flat_upper: child_upper.clone(),
            obj_bounds: domain.obj_bounds.clone(),
            // The fresh exact-domain planes have already been dropped. Also refuse to
            // carry the current rebound plane into a child while this route is
            // armed: every child must earn its own exact-domain plane when it is
            // later popped.
            linear_bounds: if fresh_domain_clip_telemetry.enabled() {
                None
            } else {
                domain.linear_bounds.clone()
            },
            depth: domain.depth + 1,
            priority: domain.priority,
            inherited_alpha_state: domain.inherited_alpha_state.clone(),
        });

        // Right child: [mid, u] on split_dim.
        child_lower[[split_dim]] = mid;
        child_upper[[split_dim]] = u;
        pending_children.push(FlatPendingChild {
            flat_lower: child_lower,
            flat_upper: child_upper,
            obj_bounds: domain.obj_bounds.clone(),
            linear_bounds: if fresh_domain_clip_telemetry.enabled() {
                None
            } else {
                domain.linear_bounds.clone()
            },
            depth: domain.depth + 1,
            priority: domain.priority,
            inherited_alpha_state: domain.inherited_alpha_state.clone(),
        });
    }

    if pending_children.is_empty() {
        return Ok(None);
    }

    // Batched IBP prescreen from flat rows — avoids N per-child BoundedTensor
    // allocations. Part of #4366 Packet A.
    let shape = child_shape.as_deref().ok_or_else(|| {
        NyError::InvalidSpec(
            "no child shape recorded despite non-empty pending_children".to_string(),
        )
    })?;
    let flat_lowers: Vec<ArrayD<f32>> = pending_children
        .iter()
        .map(|c| c.flat_lower.clone())
        .collect();
    let flat_uppers: Vec<ArrayD<f32>> = pending_children
        .iter()
        .map(|c| c.flat_upper.clone())
        .collect();
    let verified_mask = match batched_ibp_prescreen_from_flat(
        graph,
        &flat_lowers,
        &flat_uppers,
        shape,
        spec_matrix,
        thresholds,
        Some(clause_sizes),
        false,
        engine,
    ) {
        Ok(mask) => mask,
        Err(err) if graph_ibp_prescreen_error_should_skip(&err) => {
            tracing::debug!(
                "disjunctive reorder prescreen: skipping enhancement-only IBP pass for {} children after {}",
                pending_children.len(),
                err
            );
            vec![false; pending_children.len()]
        }
        Err(err) => return Err(err),
    };

    // Partition survivors: IBP-verified vs alive. Part of #4366 Packet C.
    let mut survivors: Vec<FlatPendingChild> = Vec::new();
    for (child, verified) in pending_children.into_iter().zip(verified_mask) {
        if verified {
            lifecycle.domains_verified += 1;
        } else {
            survivors.push(child);
        }
    }

    if survivors.is_empty() {
        return Ok(None);
    }

    // Batched relaxed clip for eligible survivors, per-child fallback otherwise.
    let use_batched_relaxed = verifier.config.enable_relaxed_clip
        && matches!(verifier.config.input_clip_type, InputClipType::Relaxed);

    if use_batched_relaxed {
        push_batched_relaxed_survivors(
            verifier,
            survivors,
            shape,
            thresholds,
            clause_sizes,
            queue,
            lifecycle,
            domains_verified_by_clip,
        )?;
    } else {
        push_fallback_survivors(
            verifier,
            graph,
            survivors,
            shape,
            thresholds,
            engine,
            queue,
            lifecycle,
            domains_verified_by_clip,
        )?;
    }

    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn process_per_child_batch<F>(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domains: Vec<MultiObjInputDomain>,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn GemmEngine>,
    compute_bounds: &F,
    warm_compute_bounds: Option<&WarmDisjunctiveComputeBoundsFn<'_>>,
    warm_alpha_telemetry: &WarmAlphaTelemetry,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    bab_timeout: Duration,
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_clip: &mut usize,
) -> Result<Option<BetaCrownResult>>
where
    F: Fn(&BoundedTensor, Option<&HashMap<String, BoundedTensor>>) -> Result<MultiObjBounds>,
{
    // #relational-bab option B: per-wave α edge-pass budget.
    let mut edge_alpha_budget = verifier.config.input_split_edge_alpha_top;
    for mut domain in domains {
        if lifecycle.start_time.elapsed() > bab_timeout {
            return Ok(Some(lifecycle.timeout_result()));
        }
        if lifecycle.domains_explored >= verifier.config.max_domains {
            return Ok(Some(lifecycle.build_result(
                BabVerificationStatus::Unknown {
                    reason: format!(
                        "Domain limit {}: {}/{} verified",
                        verifier.config.max_domains,
                        lifecycle.domains_verified,
                        lifecycle.domains_explored
                    ),
                },
            )));
        }

        lifecycle.domains_explored += 1;
        lifecycle.max_depth_reached = lifecycle.max_depth_reached.max(domain.depth);

        if lifecycle.domains_explored.is_multiple_of(1000) || lifecycle.domains_explored <= 5 {
            trace!(
                "[disjunctive-multi-clause] explored={} verified={} clipped={} depth={} queue={} pri={:.4}",
                lifecycle.domains_explored,
                lifecycle.domains_verified,
                *domains_verified_by_clip,
                domain.depth,
                queue.len(),
                domain.priority,
            );
        }

        if disjunctive_domain_verified(&domain.obj_bounds, thresholds, clause_sizes) {
            lifecycle.domains_verified += 1;
            continue;
        }
        // #relational-bab option B (config-gated, default inert): the α-slope
        // pass first — optimized lower bounds over the exact sub-box, budget
        // -capped per wave (pops are worst-gap-first, so the cap keeps the
        // most negative gaps). Verified ⇒ done; still-short ⇒ the domain
        // continues with the α-improved, monotonicity-guarded bounds into
        // the MILP finisher / split path below.
        if verifier.config.input_split_edge_alpha && edge_alpha_budget > 0 {
            if let Some(row_indices) = edge_domain_rows(
                &domain,
                thresholds,
                clause_sizes,
                verifier.config.input_split_edge_milp_gap,
                verifier.config.input_split_edge_milp_depth,
                spec_matrix.nrows(),
            ) {
                edge_alpha_budget -= 1;
                if let Some(fresh) = try_edge_alpha_pass(
                    verifier,
                    graph,
                    &domain,
                    spec_matrix,
                    &row_indices,
                    engine,
                    lifecycle.deadline(bab_timeout),
                ) {
                    domain.obj_bounds =
                        super::super::batching::tighten_obj_lower_bounds(&domain.obj_bounds, fresh);
                    if disjunctive_domain_verified(&domain.obj_bounds, thresholds, clause_sizes) {
                        lifecycle.domains_verified += 1;
                        continue;
                    }
                }
            }
        }
        // #relational-bab EDGE-DOMAIN ESCALATION (config-gated + oracle
        // -attached, default inert): see `try_edge_milp_escalation`.
        if try_edge_milp_escalation(
            verifier,
            graph,
            &domain,
            spec_matrix,
            thresholds,
            clause_sizes,
            engine,
            lifecycle.deadline(bab_timeout),
        ) {
            lifecycle.domains_verified += 1;
            continue;
        }
        if domain.depth >= verifier.config.max_depth {
            lifecycle.unresolved_due_to_depth = true;
            continue;
        }

        let domain_bounds: Vec<f32> = domain
            .obj_bounds
            .iter()
            .map(|(lower, upper)| {
                if verifier.config.verify_upper_bound {
                    *upper
                } else {
                    *lower
                }
            })
            .collect();
        // #relational-bab lever 2 (config-gated, default OFF = the historical
        // single-dim split byte-identically): honor `input_split_depth` —
        // split the top-k SB dims per pop into up to 2^k children exactly
        // covering the parent, mirroring the conjunctive lane
        // (multi_objective.rs). Deeper per pop = fewer rebound cycles to a
        // given box width on the band-limited relational difference nets.
        if verifier.config.input_split_disjunctive_multi_dim {
            let split_dims = verifier.select_input_dimensions_sb(
                domain.input_bounds.as_ref(),
                domain.linear_bounds.as_ref(),
                Some(domain_bounds.as_slice()),
                Some(thresholds),
            );
            let shape = domain.input_bounds.as_ref().lower().shape().to_vec();
            let (flat_lower, flat_upper) = {
                let flat = domain.input_bounds.as_ref().flatten();
                (flat.lower().clone(), flat.upper().clone())
            };
            let child_boxes =
                super::super::shared::multi_dim_split_boxes(flat_lower, flat_upper, &split_dims);
            if child_boxes.len() <= 1 {
                // #lsnc-f64-tail call site 2 (design §6.3): last chance
                // before the unresolved drop (gate off => no-op `false`).
                if super::super::f64_tail::f64_tail_last_chance(
                    graph,
                    &domain,
                    spec_matrix,
                    thresholds,
                    clause_sizes,
                    mul_binary_alphas,
                    engine,
                    lifecycle.deadline(bab_timeout),
                ) {
                    lifecycle.domains_verified += 1;
                    continue;
                }
                lifecycle.unresolved_due_to_unsplittable = true;
                continue;
            }
            for (child_lower, child_upper) in child_boxes {
                let child_input = build_child_input(&child_lower, &child_upper, &shape)?;
                screen_disjunctive_child(
                    verifier,
                    graph,
                    child_input,
                    spec_matrix,
                    thresholds,
                    clause_sizes,
                    engine,
                    compute_bounds,
                    warm_compute_bounds,
                    warm_alpha_telemetry,
                    &domain,
                    queue,
                    lifecycle,
                    domains_verified_by_clip,
                )?;
            }
            continue;
        }

        let split_dim = verifier.select_input_dimension_sb(
            domain.input_bounds.as_ref(),
            domain.linear_bounds.as_ref(),
            Some(domain_bounds.as_slice()),
            Some(thresholds),
        );
        let flat = domain.input_bounds.as_ref().flatten();

        let unsplittable = split_dim >= flat.len() || {
            let l = flat.lower()[[split_dim]];
            let u = flat.upper()[[split_dim]];
            !l.is_finite() || !u.is_finite() || u <= l
        };
        if unsplittable {
            // #lsnc-f64-tail call site 2 (design §6.3): one certified f64
            // last chance BEFORE the unresolved drop — these are exactly the
            // queue-drain leaks of the precision-limited lsnc instances.
            // Gate `NY_F64_TAIL=1` (default OFF => `false` with no work).
            if super::super::f64_tail::f64_tail_last_chance(
                graph,
                &domain,
                spec_matrix,
                thresholds,
                clause_sizes,
                mul_binary_alphas,
                engine,
                lifecycle.deadline(bab_timeout),
            ) {
                lifecycle.domains_verified += 1;
                continue;
            }
            lifecycle.unresolved_due_to_unsplittable = true;
            continue;
        }
        let l = flat.lower()[[split_dim]];
        let u = flat.upper()[[split_dim]];

        let mid = l + (u - l) / 2.0;
        let shape = domain.input_bounds.as_ref().lower().shape().to_vec();
        let mut child_lower = flat.lower().clone();
        let mut child_upper = flat.upper().clone();

        child_lower[[split_dim]] = l;
        child_upper[[split_dim]] = mid;
        let left_input = build_child_input(&child_lower, &child_upper, &shape)?;
        screen_disjunctive_child(
            verifier,
            graph,
            left_input,
            spec_matrix,
            thresholds,
            clause_sizes,
            engine,
            compute_bounds,
            warm_compute_bounds,
            warm_alpha_telemetry,
            &domain,
            queue,
            lifecycle,
            domains_verified_by_clip,
        )?;

        child_lower[[split_dim]] = mid;
        child_upper[[split_dim]] = u;
        let right_input = build_child_input(&child_lower, &child_upper, &shape)?;
        screen_disjunctive_child(
            verifier,
            graph,
            right_input,
            spec_matrix,
            thresholds,
            clause_sizes,
            engine,
            compute_bounds,
            warm_compute_bounds,
            warm_alpha_telemetry,
            &domain,
            queue,
            lifecycle,
            domains_verified_by_clip,
        )?;
    }

    Ok(None)
}

/// Shared EDGE-domain eligibility (#relational-bab): all-single-row clauses,
/// deep enough, and every unverified row within the configured gap of its
/// threshold. Returns the unverified row indices on success.
pub(super) fn edge_domain_rows(
    domain: &MultiObjInputDomain,
    thresholds: &[f32],
    clause_sizes: &[usize],
    gap: f32,
    min_depth: usize,
    spec_rows: usize,
) -> Option<Vec<usize>> {
    if !clause_sizes.iter().all(|&size| size == 1) {
        return None;
    }
    if domain.depth < min_depth {
        return None;
    }
    if domain.obj_bounds.len() != thresholds.len() || spec_rows != thresholds.len() {
        return None;
    }
    let mut rows = Vec::new();
    for (j, (lower, upper)) in domain.obj_bounds.iter().enumerate() {
        let threshold = thresholds[j];
        if super::super::grouped_semantics::objective_interval_verified(*lower, *upper, threshold) {
            continue;
        }
        let shortfall = threshold - *lower;
        if !shortfall.is_finite() || shortfall > gap {
            return None;
        }
        rows.push(j);
    }
    (!rows.is_empty()).then_some(rows)
}

/// #relational-bab OPTION B: per-domain α-CROWN pass on an EDGE domain.
/// One short α collection provides the state STRUCTURE + α-tightened
/// per-node bounds; each UNVERIFIED row then gets its slopes RE-TARGETED by
/// the spec-objective SPSA (the per-disjunct / IMB recipe — a state frozen
/// for another objective can be WORSE than default slopes, so per-row
/// re-targeting is what actually closes the floor). Returns fresh bounds for
/// exactly the optimized rows (`±inf` elsewhere; the caller's monotonic
/// merge keeps everything else), `None` on any miss. Purely a TIGHTENING
/// source — a bad pass can only fail to help, never loosen a bound.
pub(super) fn try_edge_alpha_pass(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domain: &MultiObjInputDomain,
    spec_matrix: &Array2<f32>,
    row_indices: &[usize],
    engine: Option<&dyn GemmEngine>,
    deadline: std::time::Instant,
) -> Option<Vec<(f32, f32)>> {
    let input = domain.input_bounds.as_ref();
    // Relaxation base: the SAME per-domain intermediates the rebound uses
    // (per-node CROWN-IBP on small graphs) — an α-collection's own node
    // bounds can be LOOSER and would sink the row bound below the plain
    // baseline regardless of slopes.
    let alpha_bounds =
        crate::network::collect_intermediate_bounds(graph, input, Some(deadline), engine).ok()?;
    // α state STRUCTURE at default slopes (0 optimization iterations): the
    // per-row SPSA below starts AT the plain-CROWN baseline and keeps its
    // best, so the pass can only meet-or-beat plain (a state pre-optimized
    // for another objective starts BELOW baseline and often stays there).
    let init_config = edge_alpha_child_config(&verifier.config.alpha_config, 0, deadline);
    let (_, init_alpha) = graph
        .collect_alpha_crown_bounds_dag_with_engine(input, &init_config, engine)
        .ok()?;

    let row_config = edge_alpha_child_config(
        &verifier.config.alpha_config,
        verifier.config.input_split_edge_alpha_iters.max(1),
        deadline,
    );

    let num_specs = spec_matrix.nrows();
    let mut fresh = vec![(f32::NEG_INFINITY, f32::INFINITY); num_specs];
    for &j in row_indices {
        if std::time::Instant::now() >= deadline {
            break;
        }
        let row = spec_matrix.row(j).to_vec();
        let optimized = graph
            .optimize_alpha_for_spec_objective(
                input,
                &alpha_bounds,
                &init_alpha,
                &row_config,
                &row,
                engine,
            )
            .ok()?;
        let row_spec = Array2::from_shape_vec((1, row.len()), row).ok()?;
        let (bounds, _linear) = super::super::shared::compute_crown_or_ibp_bounds_with_node_bounds(
            graph,
            input,
            &row_spec,
            engine,
            Some(&alpha_bounds),
            None,
            Some(&optimized),
            None,
            Some(deadline),
            None,
            false,
        )
        .ok()?;
        let row_bounds = super::super::shared::extract_obj_bounds(&bounds, 1).ok()?;
        fresh[j] = row_bounds[0];
    }
    Some(fresh)
}

/// Derive α settings for a child-domain edge pass without leaking root-only
/// collection authority.
///
/// The cGAN atomic collector is a root transaction. An edge-alpha experiment
/// can be enabled independently, so cloning the root config verbatim would let
/// that optional child path re-enter the root collector when both canaries are
/// combined. Preserve every ordinary edge setting, including the caller's
/// intermediate-bound policy, while clearing only the root-only flag.
fn edge_alpha_child_config(
    root: &crate::bounds::AlphaCrownConfig,
    iterations: usize,
    deadline: std::time::Instant,
) -> crate::bounds::AlphaCrownConfig {
    let mut child = root.clone();
    child.iterations = iterations;
    child.deadline = Some(deadline);
    child.cgan_sparse_target_complete_root = false;
    child.cgan_complete_crown_ibp_root = false;
    child
}

/// Offer one still-undecided domain to the default-dark input-box-only oracle.
///
/// This deliberately runs before the edge-MIP path collects intermediate
/// node bounds. Returns `true` only for a `VerifiedAllRows` result completed
/// strictly before `deadline`; every other outcome preserves the caller's
/// historical edge-MIP/split path.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn try_input_leaf_escalation(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domain: &MultiObjInputDomain,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    deadline: std::time::Instant,
) -> bool {
    try_input_leaf_escalation_with_clock(
        verifier,
        graph,
        domain,
        spec_matrix,
        thresholds,
        clause_sizes,
        deadline,
        FreshClipInstant::now,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_input_leaf_escalation_with_clock<N>(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domain: &MultiObjInputDomain,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    deadline: std::time::Instant,
    mut now: N,
) -> bool
where
    N: FnMut() -> std::time::Instant,
{
    use crate::beta_crown::graph_mip_leaf::{GraphInputLeafRequest, GraphMipLeafVerdict};

    if !verifier.config.input_split_input_leaf_oracle {
        return false;
    }
    let Some(oracle) = verifier.graph_mip_leaf_oracle() else {
        return false;
    };
    if !valid_disjunctive_layout(spec_matrix.nrows(), thresholds.len(), clause_sizes)
        || domain.obj_bounds.len() != spec_matrix.nrows()
        || spec_matrix
            .iter()
            .any(|coefficient| !coefficient.is_finite())
        || thresholds.iter().any(|threshold| !threshold.is_finite())
        || now() >= deadline
    {
        return false;
    }

    let request = GraphInputLeafRequest {
        graph,
        input_bounds: domain.input_bounds.as_ref(),
        objectives: spec_matrix,
        advisory_objective_bounds: &domain.obj_bounds,
        thresholds,
        clause_sizes,
        depth: domain.depth,
        deadline: Some(deadline),
    };
    let verdict = oracle.solve_input_leaf(&request);
    if now() >= deadline {
        trace!("input-leaf oracle completed at or after the BaB deadline; result declined");
        return false;
    }
    match verdict {
        GraphMipLeafVerdict::VerifiedAllRows => true,
        GraphMipLeafVerdict::Violated { .. } => {
            // Advisory per the shared oracle contract: sat reporting stays
            // with the attack lanes and this domain follows the old path.
            trace!("input-leaf oracle reported an advisory violation; domain requeued");
            false
        }
        GraphMipLeafVerdict::Undecided => false,
    }
}

/// Run the two independent leaf escalations in cheapest-first order.
///
/// The input-box-only oracle above is tried first whenever its separate gate
/// is armed. If it declines, #relational-bab offers a NEAR-VERIFIED deep
/// domain to the attached Graph-MIP leaf oracle with NO split premises (an
/// input-split subdomain is fully described by its input box). Returns `true`
/// ONLY on [`GraphMipLeafVerdict::VerifiedAllRows`] — the certified-UNSAT
/// contract — so the caller counts the domain verified instead of splitting
/// it. Every other outcome (gates unmet, collection failure, `Undecided`,
/// advisory `Violated`) returns `false` and the domain proceeds through the
/// unchanged split path.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_edge_milp_escalation(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domain: &MultiObjInputDomain,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn GemmEngine>,
    deadline: std::time::Instant,
) -> bool {
    try_edge_milp_escalation_with_clock(
        verifier,
        graph,
        domain,
        spec_matrix,
        thresholds,
        clause_sizes,
        engine,
        deadline,
        FreshClipInstant::now,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_edge_milp_escalation_with_clock<N>(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domain: &MultiObjInputDomain,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn GemmEngine>,
    deadline: std::time::Instant,
    mut now: N,
) -> bool
where
    N: FnMut() -> std::time::Instant,
{
    use crate::beta_crown::graph_mip_leaf::{GraphMipLeafRequest, GraphMipLeafVerdict};

    if try_input_leaf_escalation_with_clock(
        verifier,
        graph,
        domain,
        spec_matrix,
        thresholds,
        clause_sizes,
        deadline,
        &mut now,
    ) {
        return true;
    }
    if !verifier.config.input_split_edge_milp {
        return false;
    }
    if now() >= deadline {
        trace!("edge-MILP deadline expired before eligibility/collection; escalation declined");
        return false;
    }
    let Some(oracle) = verifier.graph_mip_leaf_oracle() else {
        return false;
    };
    let Some(row_indices) = edge_domain_rows(
        domain,
        thresholds,
        clause_sizes,
        verifier.config.input_split_edge_milp_gap,
        verifier.config.input_split_edge_milp_depth,
        spec_matrix.nrows(),
    ) else {
        return false;
    };
    let rows: Vec<(Vec<f32>, f32)> = row_indices
        .into_iter()
        .map(|j| (spec_matrix.row(j).to_vec(), thresholds[j]))
        .collect();
    // Sound per-node boxes over THIS domain's exact input box (the encoder's
    // big-M ranges). Failure declines (fail-closed).
    let Ok(collected) = crate::network::collect_intermediate_bounds(
        graph,
        domain.input_bounds.as_ref(),
        Some(deadline),
        engine,
    ) else {
        return false;
    };
    if now() >= deadline {
        trace!(
            "edge-MILP intermediate bounds completed at or after the BaB deadline; solve declined"
        );
        return false;
    }
    let node_bounds: HashMap<String, Arc<BoundedTensor>> = collected
        .into_iter()
        .map(|(name, bounds)| (name, Arc::new(bounds)))
        .collect();
    let request = GraphMipLeafRequest {
        graph,
        input_bounds: domain.input_bounds.as_ref(),
        node_bounds: &node_bounds,
        splits: Vec::new(),
        rows,
        depth: domain.depth,
        deadline: Some(deadline),
    };
    let verdict = oracle.solve_leaf(&request);
    if now() >= deadline {
        trace!("edge-MILP oracle completed at or after the BaB deadline; result declined");
        return false;
    }
    match verdict {
        GraphMipLeafVerdict::VerifiedAllRows => true,
        GraphMipLeafVerdict::Violated { .. } => {
            // Advisory per the oracle contract: sat reporting stays with the
            // attack lanes; the domain continues in BaB unchanged.
            trace!("edge-MILP oracle reported an advisory violation; domain requeued");
            false
        }
        GraphMipLeafVerdict::Undecided => false,
    }
}

#[cfg(test)]
mod tests;
