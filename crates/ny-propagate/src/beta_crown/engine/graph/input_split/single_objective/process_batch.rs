// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use ndarray::Array2;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::bounds::LinearBounds;
use crate::GraphNetwork;

use super::super::ibp_prescreen::batched_ibp_prescreen;
use super::super::parent_clip::clip_child_with_parent_linear;
use super::super::shared::{
    build_child_input_owned, graph_ibp_prescreen_error_should_skip, multi_dim_split_boxes,
    GraphInputDomain,
};
use super::screen_child::{screen_single_child, WarmComputeBoundsFn};

struct PendingChild {
    input_bounds: BoundedTensor,
    lower_bound: f32,
    upper_bound: f32,
    depth: usize,
    priority: f32,
    node_bounds_override: Option<Arc<HashMap<String, BoundedTensor>>>,
}

/// Pick the input-split dimensions for one domain: the Saturation-Escape
/// ranking when the SEB gate is armed (preset
/// `bab.branching.input_split.sat_escape_branch`, env `NY_SAT_ESCAPE_BRANCH`
/// override — `BetaCrownConfig::sat_escape_branch_armed`) AND the graph has a
/// saturated smooth activation (else `None`), otherwise the baseline SB
/// heuristic. Advisory only — either way the returned dims are midpoint-split
/// into a child set that exactly covers the parent box, so the choice never
/// affects soundness.
fn seb_or_sb_split_dims(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domain: &GraphInputDomain,
    domain_bounds: &[f32],
    thresholds: &[f32],
    engine: Option<&dyn GemmEngine>,
) -> Vec<usize> {
    if verifier.config.sat_escape_branch_armed() {
        if let Some(dims) = super::super::sat_escape::select_seb_dims(
            graph,
            domain.input_bounds.as_ref(),
            engine,
            verifier.config.input_split_depth,
        ) {
            return dims;
        }
    }
    verifier.select_input_dimensions_sb(
        domain.input_bounds.as_ref(),
        domain.linear_bounds.as_ref(),
        Some(domain_bounds),
        Some(thresholds),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_single_objective_domain_batch<F>(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domains: Vec<GraphInputDomain>,
    objective: &[f32],
    threshold: f32,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    compute_bounds: &F,
    warm_compute_bounds: Option<&WarmComputeBoundsFn<'_>>,
    bab_timeout: Duration,
    queue: &mut BinaryHeap<GraphInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_ibp: &mut usize,
    domains_screened_by_crown: &mut usize,
) -> Result<Option<BetaCrownResult>>
where
    F: Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
    ) -> Result<(f32, f32, Option<LinearBounds>)>,
{
    if verifier.config.reorder_bab && verifier.config.input_split_ibp_enhancement {
        // Reorder mode defers per-domain bounding into the batched dense-spec pass,
        // which uses the frozen root α state; per-sub-domain warm-start does not
        // apply here, so the warm closure is intentionally not threaded in.
        process_reorder_prescreen_batch(
            verifier,
            graph,
            domains,
            objective,
            threshold,
            spec_matrix,
            engine,
            compute_bounds,
            bab_timeout,
            queue,
            lifecycle,
            domains_verified_by_ibp,
            domains_screened_by_crown,
        )
    } else {
        process_per_child_batch(
            verifier,
            graph,
            domains,
            objective,
            threshold,
            spec_matrix,
            engine,
            compute_bounds,
            warm_compute_bounds,
            bab_timeout,
            queue,
            lifecycle,
            domains_verified_by_ibp,
            domains_screened_by_crown,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn process_reorder_prescreen_batch<F>(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domains: Vec<GraphInputDomain>,
    objective: &[f32],
    threshold: f32,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    compute_bounds: &F,
    bab_timeout: Duration,
    queue: &mut BinaryHeap<GraphInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_ibp: &mut usize,
    domains_screened_by_crown: &mut usize,
) -> Result<Option<BetaCrownResult>>
where
    F: Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
    ) -> Result<(f32, f32, Option<LinearBounds>)>,
{
    let mut pending_children = Vec::new();

    for domain in domains {
        if let Some(termination) =
            lifecycle.check_termination(bab_timeout, verifier.config.max_domains)
        {
            return Ok(Some(termination));
        }

        lifecycle.domains_explored += 1;
        lifecycle.max_depth_reached = lifecycle.max_depth_reached.max(domain.depth);

        if verifier
            .config
            .domain_is_verified(domain.lower_bound, domain.upper_bound, threshold)
        {
            lifecycle.domains_verified += 1;
            continue;
        }
        if verifier
            .config
            .domain_is_violation(domain.lower_bound, domain.upper_bound, threshold)
        {
            return Ok(Some(
                lifecycle.build_result(BabVerificationStatus::potential_violation()),
            ));
        }
        if domain.depth >= verifier.config.max_depth {
            lifecycle.unresolved_due_to_depth = true;
            continue;
        }

        let domain_bounds = [if verifier.config.verify_upper_bound {
            domain.upper_bound
        } else {
            domain.lower_bound
        }];
        let thresholds = [threshold];
        // Multi-dimensional input split: select the top `input_split_depth` dims by
        // SB score and midpoint-split each, producing up to 2^depth children that
        // EXACTLY COVER the parent (completeness preserved). At depth 1 this returns
        // exactly the original left/right pair, so behaviour is unchanged.
        //
        // Saturation-Escape Branching (advisory, preset
        // `bab.branching.input_split.sat_escape_branch` / env
        // `NY_SAT_ESCAPE_BRANCH` override, default OFF): when the binding
        // pre-activation is saturated the SB
        // objective coefficient vanishes and degenerates to width-only, so
        // reorder the split dims to the ones that de-saturate the logit. Purely
        // a reordering of `split_dims` — the box partition below is unchanged, so
        // the union cover stays exact.
        let split_dims = seb_or_sb_split_dims(
            verifier,
            graph,
            &domain,
            &domain_bounds,
            &thresholds,
            engine,
        );
        let shape = domain.input_bounds.as_ref().lower().shape().to_vec();
        let (flat_lower, flat_upper) = domain.input_bounds.as_ref().flatten().into_parts();
        let child_boxes = multi_dim_split_boxes(flat_lower, flat_upper, &split_dims);

        if child_boxes.len() <= 1 {
            lifecycle.unresolved_due_to_unsplittable = true;
            continue;
        }

        for (child_lower, child_upper) in child_boxes {
            let child_input = build_child_input_owned(child_lower, child_upper, &shape)?;
            collect_reorder_child(
                verifier,
                graph,
                child_input,
                &shape,
                objective,
                threshold,
                spec_matrix,
                engine,
                compute_bounds,
                &domain,
                queue,
                lifecycle,
                domains_verified_by_ibp,
                domains_screened_by_crown,
                &mut pending_children,
            )?;
        }
    }

    if pending_children.is_empty() {
        return Ok(None);
    }

    let child_inputs: Vec<&BoundedTensor> = pending_children
        .iter()
        .map(|child| &child.input_bounds)
        .collect();
    let verified_mask = match batched_ibp_prescreen(
        graph,
        &child_inputs,
        spec_matrix,
        &[threshold],
        None,
        verifier.config.verify_upper_bound,
        engine,
    ) {
        Ok(mask) => mask,
        Err(err) if graph_ibp_prescreen_error_should_skip(&err) => {
            tracing::debug!(
                "single-objective reorder prescreen: skipping enhancement-only IBP pass for {} children after {}",
                pending_children.len(),
                err
            );
            vec![false; pending_children.len()]
        }
        Err(err) => return Err(err),
    };

    for (child, verified) in pending_children.into_iter().zip(verified_mask) {
        if verified {
            *domains_verified_by_ibp += 1;
            lifecycle.domains_verified += 1;
            continue;
        }

        queue.push(GraphInputDomain {
            input_bounds: Arc::new(child.input_bounds),
            lower_bound: child.lower_bound,
            upper_bound: child.upper_bound,
            depth: child.depth,
            priority: child.priority,
            linear_bounds: None,
            needs_bounding: true,
            node_bounds_override: child.node_bounds_override,
            // Reorder mode defers bounding into the batched dense-spec pass, which
            // uses the frozen root α state — per-sub-domain warm-start refinement
            // does not apply on this path, so no parent α slopes are carried.
            inherited_alpha_state: None,
        });
    }

    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn collect_reorder_child<F>(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    child_input: BoundedTensor,
    shape: &[usize],
    objective: &[f32],
    threshold: f32,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    compute_bounds: &F,
    parent_domain: &GraphInputDomain,
    queue: &mut BinaryHeap<GraphInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_ibp: &mut usize,
    domains_screened_by_crown: &mut usize,
    pending_children: &mut Vec<PendingChild>,
) -> Result<()>
where
    F: Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
    ) -> Result<(f32, f32, Option<LinearBounds>)>,
{
    let mut child_input = child_input;
    let mut node_bounds_override = None;

    if verifier.config.enable_relaxed_clip {
        if parent_domain.linear_bounds.is_none() {
            let child = screen_single_child(
                verifier,
                graph,
                child_input,
                shape,
                objective,
                threshold,
                spec_matrix,
                engine,
                compute_bounds,
                // Reorder path: bounding is deferred, so no per-domain warm-start.
                None,
                parent_domain,
                lifecycle,
                domains_verified_by_ibp,
                domains_screened_by_crown,
            )?;
            if let Some(child) = child {
                push_single_objective_child(verifier, threshold, queue, lifecycle, child);
            }
            return Ok(());
        }

        let clip_result = clip_child_with_parent_linear(
            verifier,
            graph,
            &child_input,
            shape,
            objective,
            threshold,
            parent_domain.linear_bounds.as_ref(),
            engine,
        )?;
        if clip_result.verified {
            lifecycle.domains_verified += 1;
            return Ok(());
        }
        child_input = clip_result.bounds;
        node_bounds_override = clip_result.complete_clip_node_bounds.map(Arc::new);
    }

    pending_children.push(PendingChild {
        input_bounds: child_input,
        lower_bound: parent_domain.lower_bound,
        upper_bound: parent_domain.upper_bound,
        depth: parent_domain.depth + 1,
        priority: parent_domain.priority,
        node_bounds_override,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_per_child_batch<F>(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    domains: Vec<GraphInputDomain>,
    objective: &[f32],
    threshold: f32,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    compute_bounds: &F,
    warm_compute_bounds: Option<&WarmComputeBoundsFn<'_>>,
    bab_timeout: Duration,
    queue: &mut BinaryHeap<GraphInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_ibp: &mut usize,
    domains_screened_by_crown: &mut usize,
) -> Result<Option<BetaCrownResult>>
where
    F: Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
    ) -> Result<(f32, f32, Option<LinearBounds>)>,
{
    for domain in domains {
        if let Some(termination) =
            lifecycle.check_termination(bab_timeout, verifier.config.max_domains)
        {
            return Ok(Some(termination));
        }

        lifecycle.domains_explored += 1;
        lifecycle.max_depth_reached = lifecycle.max_depth_reached.max(domain.depth);

        if verifier
            .config
            .domain_is_verified(domain.lower_bound, domain.upper_bound, threshold)
        {
            lifecycle.domains_verified += 1;
            continue;
        }
        if verifier
            .config
            .domain_is_violation(domain.lower_bound, domain.upper_bound, threshold)
        {
            return Ok(Some(
                lifecycle.build_result(BabVerificationStatus::potential_violation()),
            ));
        }
        if domain.depth >= verifier.config.max_depth {
            lifecycle.unresolved_due_to_depth = true;
            continue;
        }

        let domain_bounds = [if verifier.config.verify_upper_bound {
            domain.upper_bound
        } else {
            domain.lower_bound
        }];
        let thresholds = [threshold];
        // Multi-dimensional input split: select the top `input_split_depth` dims by
        // SB score and midpoint-split each, producing up to 2^depth children that
        // EXACTLY COVER the parent (completeness preserved). At depth 1 this returns
        // exactly the original left/right pair, so behaviour is unchanged.
        //
        // Saturation-Escape Branching (advisory, preset
        // `bab.branching.input_split.sat_escape_branch` / env
        // `NY_SAT_ESCAPE_BRANCH` override, default OFF): reorder the split
        // dims toward the ones that de-saturate
        // the binding logit. Purely a reordering — the box partition below is
        // unchanged, so the union cover stays exact.
        let split_dims = seb_or_sb_split_dims(
            verifier,
            graph,
            &domain,
            &domain_bounds,
            &thresholds,
            engine,
        );
        let shape = domain.input_bounds.as_ref().lower().shape().to_vec();
        let (flat_lower, flat_upper) = domain.input_bounds.as_ref().flatten().into_parts();
        let child_boxes = multi_dim_split_boxes(flat_lower, flat_upper, &split_dims);

        if child_boxes.len() <= 1 {
            lifecycle.unresolved_due_to_unsplittable = true;
            continue;
        }

        let mut child_domains = Vec::with_capacity(child_boxes.len());
        for (child_lower, child_upper) in child_boxes {
            let child_input = build_child_input_owned(child_lower, child_upper, &shape)?;
            let child_domain = screen_single_child(
                verifier,
                graph,
                child_input,
                &shape,
                objective,
                threshold,
                spec_matrix,
                engine,
                compute_bounds,
                warm_compute_bounds,
                &domain,
                lifecycle,
                domains_verified_by_ibp,
                domains_screened_by_crown,
            )?;
            child_domains.push(child_domain);
        }

        for child in child_domains.into_iter().flatten() {
            push_single_objective_child(verifier, threshold, queue, lifecycle, child);
        }
    }

    Ok(None)
}

fn push_single_objective_child(
    verifier: &BetaCrownVerifier,
    threshold: f32,
    queue: &mut BinaryHeap<GraphInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    child: GraphInputDomain,
) {
    if child.needs_bounding
        || !verifier
            .config
            .domain_is_verified(child.lower_bound, child.upper_bound, threshold)
    {
        queue.push(child);
    } else {
        lifecycle.domains_verified += 1;
    }
}

#[cfg(test)]
mod tests;
