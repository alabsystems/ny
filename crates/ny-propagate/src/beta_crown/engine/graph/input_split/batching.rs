// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reordered-BaB batch helpers for graph input splitting.
//!
//! Phase 2 of #3870: `compute_crown_or_ibp_bounds_batched` now uses rayon to
//! parallelize individual CROWN backward passes across CPU cores. The batch size
//! cap limits how many domains are picked per DomainList iteration to control
//! termination-check granularity and scheduling overhead — not to throttle CROWN
//! throughput (which is now parallel).

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::graph::domain_batch::{
    DenseSpecBatchRequest, GraphDomainBatchEmitTiming, GraphDomainBatchExecutor,
    GraphDomainBatchMetricsSink, GraphDomainBatchPlan,
};
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::bounds::GraphAlphaState;
use crate::GraphNetwork;

use super::adv_check::{try_adv_check_on_input_bounds_batch, ADV_CHECK_INTERVAL};
use super::grouped_semantics::disjunctive_domain_priority;
pub(crate) use super::loop_batch_size::input_split_loop_batch_size;
use super::metrics::DenseSpecReboundTiming;
use super::shared::{
    compute_crown_or_ibp_bounds_with_node_bounds, extract_obj_bounds, multi_obj_domain_priority,
    GraphInputDomain, MultiObjInputDomain,
};

/// Monotonicity guard for multi-objective per-spec lower bounds: a child domain
/// (strict subset of parent) cannot have a worse lower bound than its parent.
/// Takes the element-wise max of parent lower bounds with new CROWN lower bounds.
/// Reference: alpha-beta-CROWN input_split/bounding.py:154.
pub(super) fn tighten_obj_lower_bounds(
    parent_obj_bounds: &[(f32, f32)],
    new_obj_bounds: Vec<(f32, f32)>,
) -> Vec<(f32, f32)> {
    new_obj_bounds
        .into_iter()
        .zip(parent_obj_bounds.iter())
        .map(|((new_l, new_u), &(old_l, _old_u))| (new_l.max(old_l), new_u))
        .collect()
}

/// Snapshot of one deferred domain's warm-α refinement inputs, captured BEFORE
/// the frozen rebound passes consume/clear `node_bounds_override`.
/// `(domain index, parent-refined α slopes, optional child-local node-bounds override)`.
type WarmRefineCandidate = (
    usize,
    Arc<GraphAlphaState>,
    Option<Arc<HashMap<String, BoundedTensor>>>,
);

/// #cgan-warm-par environment selector inside the preset-scoped activation
/// gate. A preset must first set `input_split_warm_parallel`; the environment
/// can then select `1`/`true` or force the serial reference with `0`/`false`.
/// Unset means use the preset value. Therefore a global environment setting can
/// never arm another category whose config keeps the default false.
static WARM_REFINE_PARALLEL_MODE: std::sync::atomic::AtomicI8 =
    std::sync::atomic::AtomicI8::new(-1);

fn warm_refine_parallel_enabled(config: &BetaCrownConfig) -> bool {
    if !config.input_split_warm_parallel {
        return false;
    }
    use std::sync::atomic::Ordering;
    match WARM_REFINE_PARALLEL_MODE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = !matches!(
                std::env::var("NY_INPUT_SPLIT_WARM_PARALLEL")
                    .ok()
                    .as_deref(),
                Some("0") | Some("false")
            );
            WARM_REFINE_PARALLEL_MODE.store(i8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// Evaluate each candidate independently, returning results in candidate order
/// so the domain mutations below remain deterministic. A failed refinement is
/// retained as an `Err` for the unchanged frozen-bound fallback at apply time.
fn execute_warm_refinements<T, F>(
    candidates: &[WarmRefineCandidate],
    parallel_enabled: bool,
    refine: F,
) -> Vec<(usize, Result<T>)>
where
    T: Send,
    F: Fn(&WarmRefineCandidate) -> Result<T> + Sync,
{
    let parallel = parallel_enabled && candidates.len() > 1;
    if parallel {
        use rayon::prelude::*;

        if !WARM_REFINE_PARALLEL_ANNOUNCED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "NY_WARM_ALPHA route=deferred-parallel status=engaged candidates={}",
                candidates.len()
            );
        }
        #[cfg(test)]
        WARM_REFINE_PARALLEL_BATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        candidates
            .par_iter()
            .map(|candidate| {
                // Match every other per-domain Rayon CROWN caller: keep nested
                // faer work sequential and inherit the driver's L2-lever state.
                let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                (candidate.0, refine(candidate))
            })
            .collect()
    } else {
        candidates
            .iter()
            .map(|candidate| (candidate.0, refine(candidate)))
            .collect()
    }
}

static WARM_REFINE_PARALLEL_ANNOUNCED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
pub(crate) static WARM_REFINE_PARALLEL_BATCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Test-only environment-selector override for [`WARM_REFINE_PARALLEL_MODE`]. Tests must
/// hold `propagation::batched::SPEC_GATE_TEST_LOCK` and restore `None`.
#[cfg(test)]
pub(crate) fn force_warm_refine_parallel(mode: Option<bool>) {
    use std::sync::atomic::Ordering;
    let value = match mode {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    WARM_REFINE_PARALLEL_MODE.store(value, Ordering::Relaxed);
}

/// Collect the deferred domains eligible for per-domain warm-α refinement.
///
/// Gate mirrors the eager (non-reorder) loops' `warm_alpha_enabled`: refinement
/// requires `input_split_alpha_iteration > 0` AND α-CROWN enabled AND the domain
/// carrying parent-refined slopes. With `input_split_alpha_iteration == 0`
/// (default) this returns an empty list, keeping the deferred rebound
/// byte-identical to the frozen-only path. Part of cgan step-2C.
fn warm_refine_candidates<'d, D>(
    config: &BetaCrownConfig,
    deferred_indices: &[usize],
    domains: &'d [D],
    inherited_alpha: impl Fn(&'d D) -> Option<&'d Arc<GraphAlphaState>>,
    node_bounds_override: impl Fn(&'d D) -> Option<&'d Arc<HashMap<String, BoundedTensor>>>,
) -> Vec<WarmRefineCandidate> {
    if config.input_split_alpha_iteration == 0 || !config.use_alpha_crown {
        return Vec::new();
    }
    deferred_indices
        .iter()
        .filter_map(|&idx| {
            let domain = &domains[idx];
            inherited_alpha(domain)
                .cloned()
                .map(|alpha| (idx, alpha, node_bounds_override(domain).cloned()))
        })
        .collect()
}

pub(super) fn pop_input_domain_batch(
    queue: &mut BinaryHeap<GraphInputDomain>,
    batch_size: usize,
) -> Vec<GraphInputDomain> {
    let mut domains = Vec::with_capacity(batch_size.min(queue.len()));
    while domains.len() < batch_size {
        let Some(domain) = queue.pop() else {
            break;
        };
        domains.push(domain);
    }
    domains
}

pub(super) fn pop_multi_obj_input_domain_batch(
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    batch_size: usize,
) -> Vec<MultiObjInputDomain> {
    let mut domains = Vec::with_capacity(batch_size.min(queue.len()));
    while domains.len() < batch_size {
        let Some(domain) = queue.pop() else {
            break;
        };
        domains.push(domain);
    }
    domains
}

#[inline]
pub(super) fn should_run_adv_check_on_batch(domains_explored: usize, adv_check: i32) -> bool {
    adv_check >= 0
        && domains_explored >= adv_check as usize
        && domains_explored.is_multiple_of(ADV_CHECK_INTERVAL)
}

pub(super) fn try_adv_check_on_batch(
    graph: &GraphNetwork,
    domains: &[GraphInputDomain],
    objective: &[f32],
    threshold: f32,
    verify_upper_bound: bool,
    deadline: Option<Instant>,
    seed_offset: u64,
    engine: Option<&dyn GemmEngine>,
) -> Result<bool> {
    try_adv_check_on_input_bounds_batch(
        graph,
        domains.iter().map(|domain| domain.input_bounds.as_ref()),
        objective,
        threshold,
        verify_upper_bound,
        deadline,
        seed_offset,
        engine,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn bound_deferred_domains_batch(
    domains: &mut [GraphInputDomain],
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    config: &BetaCrownConfig,
) -> Result<()> {
    bound_deferred_domains_batch_with_metrics(
        domains,
        graph,
        spec_matrix,
        engine,
        alpha_node_bounds,
        alpha_state,
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
        config,
        None,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bound_deferred_domains_batch_with_metrics(
    domains: &mut [GraphInputDomain],
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    config: &BetaCrownConfig,
    metrics_sink: Option<&dyn GraphDomainBatchMetricsSink>,
    batch_index: usize,
) -> Result<()> {
    let deferred_indices: Vec<usize> = domains
        .iter()
        .enumerate()
        .filter_map(|(idx, domain)| domain.needs_bounding.then_some(idx))
        .collect();
    if deferred_indices.is_empty() {
        return Ok(());
    }

    let deferred_count = deferred_indices.len();
    // cgan step-2C: snapshot warm-α candidates BEFORE the frozen passes below
    // consume/clear `node_bounds_override`. Empty unless
    // `input_split_alpha_iteration > 0` (byte-identical gate).
    let warm_candidates = warm_refine_candidates(
        config,
        &deferred_indices,
        domains,
        |domain| domain.inherited_alpha_state.as_ref(),
        |domain| domain.node_bounds_override.as_ref(),
    );
    let (override_indices, batched_indices): (Vec<usize>, Vec<usize>) = deferred_indices
        .into_iter()
        .partition(|&idx| domains[idx].node_bounds_override.is_some());
    let batched_count = batched_indices.len();
    let override_count = override_indices.len();
    let rebound_start = Instant::now();

    tracing::info!(
        deferred = deferred_count,
        batched = batched_count,
        override_count = override_count,
        warm_refine = warm_candidates.len(),
        warm_parallel = warm_refine_parallel_enabled(config) && warm_candidates.len() > 1,
        shared_alpha = alpha_state.is_some(),
        ibp_enhancement = config.input_split_ibp_enhancement,
        mul_binary_alphas = mul_binary_alphas.is_some(),
        crown_backward_layers = crown_backward_layers.is_some(),
        "input split deferred rebound"
    );

    let mut batched_timing = None;
    if !batched_indices.is_empty() {
        let deferred_inputs: Vec<&BoundedTensor> = batched_indices
            .iter()
            .map(|&idx| domains[idx].input_bounds.as_ref())
            .collect();
        let batched_bounds =
            GraphDomainBatchExecutor::execute_dense_specs(DenseSpecBatchRequest {
                graph,
                input_bounds_batch: &deferred_inputs,
                spec_matrix,
                engine,
                alpha_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
                ibp_enhancement: config.input_split_ibp_enhancement,
                stacked_rebound: config.input_split_stacked_rebound,
            })?;
        batched_timing = Some(batched_bounds.rebound_timing.clone());

        for (domain_idx, (bounds, linear_bounds)) in batched_indices.into_iter().zip(
            batched_bounds
                .bounds
                .into_iter()
                .zip(batched_bounds.linear_bounds),
        ) {
            let domain = &mut domains[domain_idx];
            let lower = bounds.lower_scalar();
            let upper = bounds.upper_scalar();
            // Monotonicity guard: a child domain (strict subset of parent) cannot
            // have a worse lower bound than its parent. CROWN relaxation on a
            // smaller domain can paradoxically produce a looser bound; clamp to
            // the parent's value (stored as placeholder in domain.lower_bound).
            // Reference: alpha-beta-CROWN input_split/bounding.py:154
            //   lb = torch.max(lb, dm_lb)
            domain.lower_bound = lower.max(domain.lower_bound);
            domain.upper_bound = upper;
            domain.linear_bounds = linear_bounds;
            domain.needs_bounding = false;
            domain.node_bounds_override = None;
            domain.priority = config.domain_priority(domain.lower_bound, upper)?;
        }
    }

    let mut override_results = Vec::with_capacity(override_indices.len());
    for &domain_idx in &override_indices {
        let domain = &domains[domain_idx];
        let (bounds, linear_bounds) = compute_crown_or_ibp_bounds_with_node_bounds(
            graph,
            domain.input_bounds.as_ref(),
            spec_matrix,
            engine,
            alpha_node_bounds,
            domain.node_bounds_override.as_deref(),
            alpha_state,
            mul_binary_alphas,
            deadline,
            crown_backward_layers,
            config.input_split_ibp_enhancement,
        )?;
        override_results.push((
            domain_idx,
            bounds.lower_scalar(),
            bounds.upper_scalar(),
            linear_bounds,
        ));
    }

    for (domain_idx, lower, upper, linear_bounds) in override_results {
        let domain = &mut domains[domain_idx];
        // Monotonicity guard (override path): same as batched path above.
        domain.lower_bound = lower.max(domain.lower_bound);
        domain.upper_bound = upper;
        domain.linear_bounds = linear_bounds;
        domain.needs_bounding = false;
        domain.node_bounds_override = None;
        domain.priority = config.domain_priority(domain.lower_bound, upper)?;
    }

    // cgan step-2C: per-domain warm-α refinement OVERLAY on the frozen batched
    // rebound. Every deferred domain above was already bounded with the frozen
    // root α (the sound baseline); domains carrying inherited parent slopes now
    // additionally run the per-sub-domain SPSA refinement and INTERSECT the
    // refined bounds with the frozen result, so the rebound is tighter-or-equal
    // to the frozen-only path BY CONSTRUCTION (contract of
    // `compute_warm_start_crown_bounds_with_refined_alpha`: refined α only
    // tightens; any failure falls back to the frozen result — fail-closed).
    // #cgan-warm-par optionally evaluates these independent candidates in
    // parallel, then applies their results below in the original order.
    let warm_results = execute_warm_refinements(
        &warm_candidates,
        warm_refine_parallel_enabled(config),
        |candidate| {
            let (domain_idx, parent_alpha, override_bounds) = candidate;
            super::shared::compute_warm_start_crown_bounds_with_refined_alpha(
                graph,
                domains[*domain_idx].input_bounds.as_ref(),
                spec_matrix,
                engine,
                override_bounds.as_deref(),
                parent_alpha.as_ref(),
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
                config,
            )
        },
    );
    for (domain_idx, warm_result) in warm_results {
        match warm_result {
            Ok((bounds, warm_linear, refined_alpha)) => {
                let domain = &mut domains[domain_idx];
                // Intersection of two sound enclosures is sound; `f32::max`/
                // `min` ignore NaN operands, so a NaN refined bound leaves the
                // frozen (sound) value in place.
                let new_lower = domain.lower_bound.max(bounds.lower_scalar());
                let new_upper = domain.upper_bound.min(bounds.upper_scalar());
                let new_priority = match config.domain_priority(new_lower, new_upper) {
                    Ok(priority) => priority,
                    Err(err) => {
                        tracing::debug!(
                            domain_idx,
                            "deferred rebound warm-α overlay produced unusable priority; \
                             keeping frozen batch bounds: {err}"
                        );
                        continue;
                    }
                };
                domain.lower_bound = new_lower;
                domain.upper_bound = new_upper;
                if warm_linear.is_some() {
                    domain.linear_bounds = warm_linear;
                }
                domain.priority = new_priority;
                // Save the refined slopes so children pushed from this domain
                // warm-start from them (mirrors the eager screen_child path),
                // without retaining optimizer maps that the next warm start
                // rebuilds rather than reads.
                domain.inherited_alpha_state = Some(Arc::new(refined_alpha.into_warm_start_seed()));
            }
            Err(err) => {
                tracing::debug!(
                    domain_idx,
                    "deferred rebound warm-α refinement failed; keeping frozen batch bounds: {err}"
                );
            }
        }
    }

    let rebound_timing = match batched_timing {
        Some(timing) => timing.with_total_elapsed(
            deferred_count,
            spec_matrix.nrows(),
            rebound_start.elapsed().as_secs_f64(),
        ),
        None => DenseSpecReboundTiming::override_only(
            deferred_count,
            spec_matrix.nrows(),
            rebound_start.elapsed().as_secs_f64(),
        ),
    };

    GraphDomainBatchPlan::for_dense_spec_rebound(
        batch_index,
        deferred_count,
        batched_count,
        override_count,
        &rebound_timing,
    )
    .emit_to_sink(
        metrics_sink,
        GraphDomainBatchEmitTiming::from_dense_spec(&rebound_timing),
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bound_deferred_dense_spec_domains_batch<F>(
    domains: &mut [MultiObjInputDomain],
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    config: &BetaCrownConfig,
    log_label: Option<&'static str>,
    priority_fn: F,
    metrics_sink: Option<&dyn GraphDomainBatchMetricsSink>,
    batch_index: usize,
) -> Result<DenseSpecReboundTiming>
where
    F: Fn(&[(f32, f32)]) -> f32 + Copy,
{
    let num_specs = spec_matrix.nrows();
    let deferred_indices: Vec<usize> = domains
        .iter()
        .enumerate()
        .filter_map(|(idx, domain)| domain.needs_bounding.then_some(idx))
        .collect();
    if deferred_indices.is_empty() {
        return Ok(DenseSpecReboundTiming::no_deferred_domains(num_specs));
    }

    let rebound_start = Instant::now();
    let deferred_count = deferred_indices.len();
    // cgan step-2C: snapshot warm-α candidates BEFORE the frozen passes below
    // consume/clear `node_bounds_override`. Empty unless
    // `input_split_alpha_iteration > 0` (byte-identical gate).
    let warm_candidates = warm_refine_candidates(
        config,
        &deferred_indices,
        domains,
        |domain| domain.inherited_alpha_state.as_ref(),
        |domain| domain.node_bounds_override.as_ref(),
    );
    let (override_indices, batched_indices): (Vec<usize>, Vec<usize>) = deferred_indices
        .into_iter()
        .partition(|&idx| domains[idx].node_bounds_override.is_some());
    let (batched_count, override_count) = (batched_indices.len(), override_indices.len());
    if let Some(log_label) = log_label {
        tracing::info!(
            deferred = deferred_count,
            batched = batched_count,
            override_count = override_count,
            warm_refine = warm_candidates.len(),
            warm_parallel = warm_refine_parallel_enabled(config) && warm_candidates.len() > 1,
            shared_alpha = alpha_state.is_some(),
            ibp_enhancement = config.input_split_ibp_enhancement,
            mul_binary_alphas = mul_binary_alphas.is_some(),
            crown_backward_layers = crown_backward_layers.is_some(),
            "{log_label}"
        );
    }

    let mut batched_timing = None;
    if !batched_indices.is_empty() {
        let deferred_inputs: Vec<&BoundedTensor> = batched_indices
            .iter()
            .map(|&idx| domains[idx].input_bounds.as_ref())
            .collect();
        let batched_bounds =
            GraphDomainBatchExecutor::execute_dense_specs(DenseSpecBatchRequest {
                graph,
                input_bounds_batch: &deferred_inputs,
                spec_matrix,
                engine,
                alpha_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
                ibp_enhancement: config.input_split_ibp_enhancement,
                stacked_rebound: config.input_split_stacked_rebound,
            })?;
        batched_timing = Some(batched_bounds.rebound_timing.clone());

        for (domain_idx, (bounds, linear_bounds)) in batched_indices.into_iter().zip(
            batched_bounds
                .bounds
                .into_iter()
                .zip(batched_bounds.linear_bounds),
        ) {
            let domain = &mut domains[domain_idx];
            let new_obj_bounds = extract_obj_bounds(&bounds, num_specs)?;
            // Monotonicity guard: per-spec lower bound cannot regress below parent.
            // Reference: alpha-beta-CROWN input_split/bounding.py:154
            domain.obj_bounds = tighten_obj_lower_bounds(&domain.obj_bounds, new_obj_bounds);
            domain.linear_bounds = linear_bounds;
            domain.needs_bounding = false;
            domain.node_bounds_override = None;
            domain.priority = priority_fn(&domain.obj_bounds);
        }
    }

    let mut override_results = Vec::with_capacity(override_indices.len());
    for &domain_idx in &override_indices {
        let domain = &domains[domain_idx];
        let (bounds, linear_bounds) = compute_crown_or_ibp_bounds_with_node_bounds(
            graph,
            domain.input_bounds.as_ref(),
            spec_matrix,
            engine,
            alpha_node_bounds,
            domain.node_bounds_override.as_deref(),
            alpha_state,
            mul_binary_alphas,
            deadline,
            crown_backward_layers,
            config.input_split_ibp_enhancement,
        )?;
        override_results.push((
            domain_idx,
            extract_obj_bounds(&bounds, num_specs)?,
            linear_bounds,
        ));
    }

    for (domain_idx, obj_bounds, linear_bounds) in override_results {
        let domain = &mut domains[domain_idx];
        // Monotonicity guard (multi-obj override path).
        domain.obj_bounds = tighten_obj_lower_bounds(&domain.obj_bounds, obj_bounds);
        domain.linear_bounds = linear_bounds;
        domain.needs_bounding = false;
        domain.node_bounds_override = None;
        domain.priority = priority_fn(&domain.obj_bounds);
    }

    // cgan step-2C: per-domain warm-α refinement OVERLAY on the frozen batched
    // rebound. Every deferred domain above was already bounded with the frozen
    // root α (the sound baseline); domains carrying inherited parent slopes now
    // additionally run the per-sub-domain SPSA refinement and INTERSECT the
    // refined per-spec bounds with the frozen result, so the rebound is
    // tighter-or-equal to the frozen-only path BY CONSTRUCTION (contract of
    // `compute_warm_start_crown_bounds_with_refined_alpha`: refined α only
    // tightens; any failure falls back to the frozen result — fail-closed).
    // #cgan-warm-par optionally evaluates these independent candidates in
    // parallel, then applies their results below in the original order.
    let warm_results = execute_warm_refinements(
        &warm_candidates,
        warm_refine_parallel_enabled(config),
        |candidate| {
            let (domain_idx, parent_alpha, override_bounds) = candidate;
            super::shared::compute_warm_start_crown_bounds_with_refined_alpha(
                graph,
                domains[*domain_idx].input_bounds.as_ref(),
                spec_matrix,
                engine,
                override_bounds.as_deref(),
                parent_alpha.as_ref(),
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
                config,
            )
            .and_then(|(bounds, linear, refined_alpha)| {
                Ok((
                    extract_obj_bounds(&bounds, num_specs)?,
                    linear,
                    refined_alpha,
                ))
            })
        },
    );
    for (domain_idx, warm_result) in warm_results {
        match warm_result {
            Ok((warm_obj, warm_linear, refined_alpha)) => {
                let domain = &mut domains[domain_idx];
                if domain.obj_bounds.len() != warm_obj.len() {
                    // Fail-closed: a shape drift would silently truncate the
                    // per-spec rows through `zip`; keep the frozen result.
                    tracing::debug!(
                        domain_idx,
                        frozen_rows = domain.obj_bounds.len(),
                        warm_rows = warm_obj.len(),
                        "deferred rebound warm-α overlay row-count mismatch; keeping frozen batch bounds"
                    );
                    continue;
                }
                // Intersection of two sound per-spec enclosures is sound;
                // `f32::max`/`min` ignore NaN operands, so a NaN refined bound
                // leaves the frozen (sound) value in place.
                domain.obj_bounds = domain
                    .obj_bounds
                    .iter()
                    .zip(warm_obj)
                    .map(|(&(frozen_l, frozen_u), (warm_l, warm_u))| {
                        (frozen_l.max(warm_l), frozen_u.min(warm_u))
                    })
                    .collect();
                if warm_linear.is_some() {
                    domain.linear_bounds = warm_linear;
                }
                domain.priority = priority_fn(&domain.obj_bounds);
                // Save the refined slopes so children pushed from this domain
                // warm-start from them (mirrors the eager screen_child path),
                // without retaining optimizer maps that the next warm start
                // rebuilds rather than reads.
                domain.inherited_alpha_state = Some(Arc::new(refined_alpha.into_warm_start_seed()));
            }
            Err(err) => {
                tracing::debug!(
                    domain_idx,
                    "deferred rebound warm-α refinement failed; keeping frozen batch bounds: {err}"
                );
            }
        }
    }

    let total_elapsed_s = rebound_start.elapsed().as_secs_f64();
    let rebound_timing = match batched_timing {
        Some(timing) => timing.with_total_elapsed(deferred_count, num_specs, total_elapsed_s),
        None => DenseSpecReboundTiming::override_only(deferred_count, num_specs, total_elapsed_s),
    };

    GraphDomainBatchPlan::for_dense_spec_rebound(
        batch_index,
        deferred_count,
        batched_count,
        override_count,
        &rebound_timing,
    )
    .emit_to_sink(
        metrics_sink,
        GraphDomainBatchEmitTiming::from_dense_spec(&rebound_timing),
    )?;

    Ok(rebound_timing)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bound_deferred_multi_obj_domains_batch(
    domains: &mut [MultiObjInputDomain],
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    config: &BetaCrownConfig,
    metrics_sink: Option<&dyn GraphDomainBatchMetricsSink>,
    batch_index: usize,
) -> Result<DenseSpecReboundTiming> {
    bound_deferred_dense_spec_domains_batch(
        domains,
        graph,
        spec_matrix,
        engine,
        alpha_node_bounds,
        alpha_state,
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
        config,
        Some("multi-obj input split deferred rebound"),
        |obj_bounds| multi_obj_domain_priority(obj_bounds, thresholds),
        metrics_sink,
        batch_index,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bound_deferred_disjunctive_domains_batch(
    domains: &mut [MultiObjInputDomain],
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    config: &BetaCrownConfig,
    metrics_sink: Option<&dyn GraphDomainBatchMetricsSink>,
    batch_index: usize,
) -> Result<DenseSpecReboundTiming> {
    // #relational-bab lever 1 (config-gated, default OFF => byte-identical):
    // the collection-verify shortcut applies only when the rebound would use
    // per-domain-collected intermediates anyway (no shared alpha reference,
    // no per-layer cap) — exactly the relational difference-net regime.
    let timing = if config.input_split_collection_verify_shortcut
        && alpha_node_bounds.is_none()
        && alpha_state.is_none()
        && mul_binary_alphas.is_none()
        && crown_backward_layers.is_none()
    {
        bound_deferred_disjunctive_shortcut_batch(
            domains,
            graph,
            spec_matrix,
            thresholds,
            clause_sizes,
            engine,
            deadline,
            config,
        )?
    } else {
        bound_deferred_dense_spec_domains_batch(
            domains,
            graph,
            spec_matrix,
            engine,
            alpha_node_bounds,
            alpha_state,
            mul_binary_alphas,
            deadline,
            crown_backward_layers,
            config,
            // Log label so warm-α overlay engagement (`warm_refine = N`) is
            // observable at `-v`; the step-2C composed-mode negative was
            // unfalsifiable without it (zero log surface in this lane).
            Some("disjunctive input split deferred rebound"),
            |obj_bounds| disjunctive_domain_priority(obj_bounds, thresholds, clause_sizes),
            metrics_sink,
            batch_index,
        )?
    };
    // #lsnc-f64-tail call site 1 (design §6.3, gate `NY_F64_TAIL=1`, default
    // OFF => byte-identical no-op): after the batched f32 rebound + monotonic
    // tighten, escalate still-unverified in-band domains through the certified
    // f64 backward. Additive: only certified `Verified` outcomes raise
    // obj_bounds (monotonically), so the untouched downstream f32 verdict
    // funnel decides as before.
    super::f64_tail::f64_tail_escalate_batch(
        domains,
        graph,
        spec_matrix,
        thresholds,
        clause_sizes,
        mul_binary_alphas,
        engine,
        deadline,
    );
    Ok(timing)
}

/// Per-spec-row interval bounds from a sound OUTPUT-node enclosure: row `c`
/// maps to `[Σ c_j·(c_j≥0 ? l_j : u_j), Σ c_j·(c_j≥0 ? u_j : l_j)]`. For a
/// `±e_i` row this is EXACTLY the output entry's own bound — bit-identical to
/// the spec backward when the entry came from per-node CROWN-IBP (measured on
/// the relational ACAS difference nets). NaN propagates (fail-closed: a NaN
/// bound never verifies).
fn interval_spec_obj_bounds(
    out_entry: &BoundedTensor,
    spec_matrix: &Array2<f32>,
) -> Option<Vec<(f32, f32)>> {
    let flat = out_entry.flatten();
    if flat.len() != spec_matrix.ncols() {
        return None;
    }
    let mut rows = Vec::with_capacity(spec_matrix.nrows());
    for row in spec_matrix.rows() {
        let (mut lo, mut hi) = (0.0f32, 0.0f32);
        for (j, &c) in row.iter().enumerate() {
            let l = flat.lower()[[j]];
            let u = flat.upper()[[j]];
            if c >= 0.0 {
                lo += c * l;
                hi += c * u;
            } else {
                lo += c * u;
                hi += c * l;
            }
        }
        rows.push((lo, hi));
    }
    Some(rows)
}

/// #relational-bab lever 1: rebound deferred disjunctive domains with the
/// COLLECTION-VERIFY SHORTCUT. Per domain (rayon):
///   1. collect per-domain intermediates once (per-node CROWN-IBP on small
///      graphs — the tight relaxation base);
///   2. derive spec-row bounds from the collection's OUTPUT entry by interval
///      arithmetic; a domain whose every clause is already refuted SKIPS the
///      spec backward entirely (it verifies at the process step and is never
///      split, so it needs no linear bounds);
///   3. survivors run the standard scalar spec backward over the SAME
///      collected intermediates (no recollection), keeping the linear bounds
///      the SB split scoring / relaxed clip consume.
///
/// Sound: both bound sources are sound enclosures; the monotonic
/// parent-tighten guard is applied identically to the generic path.
#[allow(clippy::too_many_arguments)]
fn bound_deferred_disjunctive_shortcut_batch(
    domains: &mut [MultiObjInputDomain],
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    config: &BetaCrownConfig,
) -> Result<DenseSpecReboundTiming> {
    use rayon::prelude::*;

    let num_specs = spec_matrix.nrows();
    let deferred_indices: Vec<usize> = domains
        .iter()
        .enumerate()
        .filter_map(|(idx, domain)| domain.needs_bounding.then_some(idx))
        .collect();
    if deferred_indices.is_empty() {
        return Ok(DenseSpecReboundTiming::no_deferred_domains(num_specs));
    }
    let rebound_start = Instant::now();
    let deferred_count = deferred_indices.len();
    let output_name = graph.output_name().to_string();

    type ShortcutResult = (
        usize,
        Vec<(f32, f32)>,
        Option<crate::bounds::LinearBounds>,
        bool,
    );
    let results: Result<Vec<ShortcutResult>> = deferred_indices
        .par_iter()
        .map(|&idx| {
            let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
            let domain = &domains[idx];
            let input = domain.input_bounds.as_ref();

            // Complete-clip override domains keep the generic override
            // semantics (child-local node bounds replace the collection).
            if domain.node_bounds_override.is_some() {
                let (bounds, linear) = compute_crown_or_ibp_bounds_with_node_bounds(
                    graph,
                    input,
                    spec_matrix,
                    engine,
                    None,
                    domain.node_bounds_override.as_deref(),
                    None,
                    None,
                    deadline,
                    None,
                    config.input_split_ibp_enhancement,
                )?;
                return Ok((idx, extract_obj_bounds(&bounds, num_specs)?, linear, false));
            }

            let node_bounds =
                crate::network::collect_intermediate_bounds(graph, input, deadline, engine)?;

            // The shortcut check: spec-row bounds off the output entry.
            if let Some(out_entry) = node_bounds.get(&output_name) {
                if let Some(quick) = interval_spec_obj_bounds(out_entry, spec_matrix) {
                    if super::grouped_semantics::disjunctive_domain_verified(
                        &quick,
                        thresholds,
                        clause_sizes,
                    ) {
                        return Ok((idx, quick, None, true));
                    }
                }
            }

            // Survivor: standard spec backward over the SAME intermediates.
            let (bounds, linear) = graph
                .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
                    input,
                    spec_matrix,
                    engine,
                    &node_bounds,
                    deadline,
                )?;
            Ok((idx, extract_obj_bounds(&bounds, num_specs)?, linear, false))
        })
        .collect();

    let mut shortcut_verified = 0usize;
    for (idx, obj_bounds, linear_bounds, took_shortcut) in results? {
        let domain = &mut domains[idx];
        // Monotonicity guard: identical to the generic path.
        domain.obj_bounds = tighten_obj_lower_bounds(&domain.obj_bounds, obj_bounds);
        domain.linear_bounds = linear_bounds;
        domain.needs_bounding = false;
        domain.node_bounds_override = None;
        domain.priority = disjunctive_domain_priority(&domain.obj_bounds, thresholds, clause_sizes);
        shortcut_verified += usize::from(took_shortcut);
    }

    tracing::debug!(
        deferred = deferred_count,
        shortcut_verified,
        "disjunctive rebound: collection-verify shortcut"
    );
    Ok(DenseSpecReboundTiming::override_only(
        deferred_count,
        num_specs,
        rebound_start.elapsed().as_secs_f64(),
    ))
}
