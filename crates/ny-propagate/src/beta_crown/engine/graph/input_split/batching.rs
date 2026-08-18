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
use crate::beta_crown::result::ViolationWitness;
use crate::bounds::GraphAlphaState;
use crate::GraphNetwork;

use super::adv_check::{try_adv_check_on_input_bounds_batch, ADV_CHECK_INTERVAL};
use super::grouped_semantics::disjunctive_domain_priority;
pub(crate) use super::loop_batch_size::input_split_loop_batch_size;
use super::metrics::{should_log_batch, DenseSpecReboundTiming};
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
    if parent_obj_bounds.len() != new_obj_bounds.len() {
        // A child is a subset of its parent, so retaining the complete parent
        // vector is sound. Never zip-truncate the objective layout: downstream
        // grouped verdicts require every row.
        return parent_obj_bounds.to_vec();
    }
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

/// Complete-clip overrides retain a child-local intermediate-bound map and
/// therefore cannot use the shared dense-spec batch executor. They are still
/// independent, but every CROWN call owns a full carrier. Keep the fan-out at a
/// fixed two workers so peak memory is bounded independently of host core count.
const OVERRIDE_REBOUND_PARALLEL_WIDTH: usize = 2;

/// Per-worker floor for allocator, graph traversal, and small-matrix scratch
/// not represented by the child-local activation map itself.
const OVERRIDE_REBOUND_MIN_WORKER_BYTES: usize = 64 << 20;

/// Conservative carrier/scratch allowance per activation cell and spec row.
/// One live backward can carry lower/upper f32 coefficients, error matrices,
/// f64 recomputation buffers, and layer-local scratch at the same time.
const OVERRIDE_REBOUND_BYTES_PER_CELL_SPEC: usize = 64;

/// The estimated extra worker may consume at most one eighth of live
/// kernel-enforced process headroom. This mirrors the dense-materialization
/// envelope discipline and leaves the other seven eighths for the serial
/// verifier working set, retained results, allocator slack, and shared state.
const OVERRIDE_REBOUND_HEADROOM_FACTOR: usize = 8;

/// Preset-scoped environment selector. The environment may disable an armed
/// category but cannot enable one whose typed config remains false.
static OVERRIDE_REBOUND_PARALLEL_MODE: std::sync::atomic::AtomicI8 =
    std::sync::atomic::AtomicI8::new(-1);

#[cfg(test)]
static OVERRIDE_REBOUND_ADMISSION_MODE: std::sync::atomic::AtomicI8 =
    std::sync::atomic::AtomicI8::new(-1);

fn override_rebound_parallel_enabled(config: &BetaCrownConfig) -> bool {
    if !config.input_split_override_parallel {
        return false;
    }
    use std::sync::atomic::Ordering;
    match OVERRIDE_REBOUND_PARALLEL_MODE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = !matches!(
                std::env::var("NY_INPUT_SPLIT_OVERRIDE_PARALLEL")
                    .ok()
                    .as_deref(),
                Some("0") | Some("false")
            );
            OVERRIDE_REBOUND_PARALLEL_MODE.store(i8::from(on), Ordering::Relaxed);
            on
        }
    }
}

#[inline]
fn collection_verify_shortcut_enabled(config: &BetaCrownConfig) -> bool {
    config.input_split_collection_verify_shortcut && !override_rebound_parallel_enabled(config)
}

/// Test-only selector for both the preset-scoped environment arm and the live
/// process-envelope decision. Tests must hold `SPEC_GATE_TEST_LOCK` and restore
/// `None` before releasing it.
#[cfg(test)]
pub(crate) fn force_override_rebound_parallel(mode: Option<bool>) {
    use std::sync::atomic::Ordering;
    let value = match mode {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    OVERRIDE_REBOUND_PARALLEL_MODE.store(value, Ordering::Relaxed);
    OVERRIDE_REBOUND_ADMISSION_MODE.store(value, Ordering::Relaxed);
}

#[inline]
fn override_rebound_worker_estimate_bytes(max_activation_cells: usize, spec_rows: usize) -> usize {
    max_activation_cells
        .saturating_mul(spec_rows.max(1))
        .saturating_mul(OVERRIDE_REBOUND_BYTES_PER_CELL_SPEC)
        .max(OVERRIDE_REBOUND_MIN_WORKER_BYTES)
}

#[inline]
fn override_rebound_has_process_headroom(
    worker_estimate_bytes: usize,
    envelope: crate::network::crown_memory::ProcessMemoryEnvelope,
) -> bool {
    let Some(required) = worker_estimate_bytes
        .checked_mul(OVERRIDE_REBOUND_HEADROOM_FACTOR)
        .and_then(|required| u64::try_from(required).ok())
    else {
        return false;
    };
    match envelope {
        crate::network::crown_memory::ProcessMemoryEnvelope::Bounded { headroom_bytes } => {
            required <= headroom_bytes
        }
        // This optional optimization requires an observable hard backstop.
        // Unbounded and unreadable envelopes keep the serial reference.
        crate::network::crown_memory::ProcessMemoryEnvelope::Unbounded
        | crate::network::crown_memory::ProcessMemoryEnvelope::Unavailable => false,
    }
}

fn override_rebound_parallel_admitted<'a>(
    config: &BetaCrownConfig,
    override_count: usize,
    spec_rows: usize,
    override_workloads: impl Iterator<Item = (&'a HashMap<String, BoundedTensor>, &'a BoundedTensor)>,
) -> bool {
    if override_count < 2 || !override_rebound_parallel_enabled(config) {
        return false;
    }
    let Some(max_working_cells) = override_workloads
        .map(|(map, input)| {
            map.values().fold(
                input.lower().len().max(input.upper().len()),
                |cells, bounds| {
                    cells.saturating_add(bounds.lower().len().max(bounds.upper().len()))
                },
            )
        })
        .max()
    else {
        return false;
    };
    let worker_estimate = override_rebound_worker_estimate_bytes(max_working_cells, spec_rows);
    #[cfg(test)]
    match OVERRIDE_REBOUND_ADMISSION_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        0 => return false,
        1 => return true,
        _ => {}
    }
    let envelope = crate::network::crown_memory::process_memory_envelope();
    let admitted = override_rebound_has_process_headroom(worker_estimate, envelope);
    if !admitted {
        tracing::debug!(
            worker_estimate,
            required_headroom = worker_estimate.saturating_mul(OVERRIDE_REBOUND_HEADROOM_FACTOR),
            ?envelope,
            "override rebound parallelism declined by process-envelope admission"
        );
    }
    admitted
}

static OVERRIDE_REBOUND_PARALLEL_ANNOUNCED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Serialize all override-rebound executors in this process. The scalar and
/// dense-spec entry points both hold this guard from their final live-headroom
/// sample through result collection, so independent verifier calls cannot each
/// launch a two-worker tranche against the same observation.
static OVERRIDE_REBOUND_EXECUTION_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Execute child-local override rebounds without exposing worker completion
/// order to the domain heap. Each two-item tranche finishes completely, then
/// errors and successful values are consumed in input order. Consequently the
/// first reported error is deterministic and no later tranche starts after it.
fn execute_override_rebounds<T, F>(
    domain_indices: &[usize],
    parallel_enabled: bool,
    rebound: F,
) -> Result<Vec<(usize, T)>>
where
    T: Send,
    F: Fn(usize) -> Result<T> + Sync,
{
    if !parallel_enabled || domain_indices.len() < 2 {
        return domain_indices
            .iter()
            .map(|&domain_idx| rebound(domain_idx).map(|value| (domain_idx, value)))
            .collect();
    }

    use rayon::prelude::*;

    if !OVERRIDE_REBOUND_PARALLEL_ANNOUNCED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "NY_INPUT_SPLIT route=override-rebound-parallel status=engaged width={OVERRIDE_REBOUND_PARALLEL_WIDTH}"
        );
    }

    let mut completed = Vec::with_capacity(domain_indices.len());
    for tranche in domain_indices.chunks(OVERRIDE_REBOUND_PARALLEL_WIDTH) {
        let tranche_results: Vec<Result<(usize, T)>> = tranche
            .par_iter()
            .map(|&domain_idx| {
                // Do not let each outer worker spawn nested faer work: two full
                // carriers are the complete concurrency and memory budget.
                let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                rebound(domain_idx).map(|value| (domain_idx, value))
            })
            .collect();
        completed.extend(tranche_results.into_iter().collect::<Result<Vec<_>>>()?);
    }
    Ok(completed)
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

/// Returns the concrete violating point when the probe finds one
/// (#advcheck-witness) so the caller can hand it to the confirmer instead of
/// discarding it.
pub(super) fn try_adv_check_on_batch(
    graph: &GraphNetwork,
    domains: &[GraphInputDomain],
    objective: &[f32],
    threshold: f32,
    verify_upper_bound: bool,
    deadline: Option<Instant>,
    seed_offset: u64,
    engine: Option<&dyn GemmEngine>,
) -> Result<Option<ViolationWitness>> {
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
        override_parallel_requested =
            override_rebound_parallel_enabled(config) && override_count > 1,
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

    let override_results = if override_indices.is_empty() {
        Vec::new()
    } else {
        let _execution_guard = OVERRIDE_REBOUND_EXECUTION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Admit at the last possible point, while holding the process-wide
        // execution lock: the ordinary batched rebound above may have retained
        // one LinearBounds carrier per domain and reduced live headroom.
        let override_parallel = override_rebound_parallel_admitted(
            config,
            override_count,
            spec_matrix.nrows(),
            override_indices.iter().filter_map(|&idx| {
                domains[idx]
                    .node_bounds_override
                    .as_deref()
                    .map(|map| (map, domains[idx].input_bounds.as_ref()))
            }),
        );
        tracing::info!(
            override_count,
            override_parallel_admitted = override_parallel,
            "input split override rebound admission"
        );
        execute_override_rebounds(&override_indices, override_parallel, |domain_idx| {
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
            Ok((bounds.lower_scalar(), bounds.upper_scalar(), linear_bounds))
        })?
    };

    for (domain_idx, (lower, upper, linear_bounds)) in override_results {
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
            override_parallel_requested =
                override_rebound_parallel_enabled(config) && override_count > 1,
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

    let override_results = if override_indices.is_empty() {
        Vec::new()
    } else {
        let _execution_guard = OVERRIDE_REBOUND_EXECUTION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Re-read the strict envelope under the shared executor lock after the
        // ordinary batched pass has retained its outputs.
        let override_parallel = override_rebound_parallel_admitted(
            config,
            override_count,
            num_specs,
            override_indices.iter().filter_map(|&idx| {
                domains[idx]
                    .node_bounds_override
                    .as_deref()
                    .map(|map| (map, domains[idx].input_bounds.as_ref()))
            }),
        );
        tracing::info!(
            override_count,
            override_parallel_admitted = override_parallel,
            "input split override rebound admission"
        );
        execute_override_rebounds(&override_indices, override_parallel, |domain_idx| {
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
            Ok((extract_obj_bounds(&bounds, num_specs)?, linear_bounds))
        })?
    };

    for (domain_idx, (obj_bounds, linear_bounds)) in override_results {
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
        should_log_batch(batch_index).then_some("multi-obj input split deferred rebound"),
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
    let timing = if collection_verify_shortcut_enabled(config)
        // The shortcut owns a separate global-pool fan-out. When bounded
        // override execution is requested, retain one concurrency contract by
        // selecting the generic dense-spec path below instead.
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
/// maps to `[Σ c_j·(c_j≥0 ? l_j : u_j), Σ c_j·(c_j≥0 ? u_j : l_j)]`.
///
/// The shared graph reducer accumulates in f64 and rounds both f32 endpoints
/// outward. This function feeds verdict predicates, so ordinary f32
/// accumulation is not sufficient: cancellation can round a lower endpoint
/// above a strict threshold and unsoundly prune a domain.
pub(super) fn interval_spec_obj_bounds(
    out_entry: &BoundedTensor,
    spec_matrix: &Array2<f32>,
) -> Option<Vec<(f32, f32)>> {
    let flat = out_entry.flatten();
    if flat.len() != spec_matrix.ncols() {
        return None;
    }
    let lower_values: Vec<f32> = flat.lower().iter().copied().collect();
    let upper_values: Vec<f32> = flat.upper().iter().copied().collect();
    Some(
        spec_matrix
            .rows()
            .into_iter()
            .map(|row| GraphNetwork::spec_row_interval_bounds(row, &lower_values, &upper_values))
            .collect(),
    )
}

/// Project the output entry in a collected root map through the packed spec.
/// Match graph propagation's established empty-output convention by resolving
/// the last executable node. An invalid/empty graph, missing output entry, or
/// shape mismatch declines the shortcut so callers retain their historical
/// fresh spec-CROWN path.
pub(super) fn root_map_spec_obj_bounds(
    graph: &GraphNetwork,
    root_node_bounds: &HashMap<String, BoundedTensor>,
    spec_matrix: &Array2<f32>,
) -> Option<Vec<(f32, f32)>> {
    let output_name = if graph.output_name().is_empty() {
        graph.exec_order().ok()?.last()?.as_str()
    } else {
        graph.output_name()
    };
    let out_entry = root_node_bounds.get(output_name)?;
    interval_spec_obj_bounds(out_entry, spec_matrix)
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
            if let Some(quick) = root_map_spec_obj_bounds(graph, &node_bounds, spec_matrix) {
                if super::grouped_semantics::disjunctive_domain_verified(
                    &quick,
                    thresholds,
                    clause_sizes,
                ) {
                    return Ok((idx, quick, None, true));
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

#[cfg(test)]
mod override_parallel_executor_tests {
    use super::{
        collection_verify_shortcut_enabled, execute_override_rebounds,
        force_override_rebound_parallel, override_rebound_has_process_headroom,
        override_rebound_parallel_enabled, override_rebound_worker_estimate_bytes,
        OVERRIDE_REBOUND_HEADROOM_FACTOR, OVERRIDE_REBOUND_MIN_WORKER_BYTES,
    };
    use crate::beta_crown::config::BetaCrownConfig;
    use crate::beta_crown::engine::graph::propagation::batched::SPEC_GATE_TEST_LOCK;
    use crate::network::crown_memory::ProcessMemoryEnvelope;
    use ny_core::{NyError, Result};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Barrier;

    #[test]
    fn result_order_matches_serial_reference() {
        let indices = [9usize, 3, 7, 1, 8];
        let serial = execute_override_rebounds(&indices, false, |idx| Ok(idx * 11)).unwrap();
        let parallel = execute_override_rebounds(&indices, true, |idx| Ok(idx * 11)).unwrap();
        assert_eq!(parallel, serial);
    }

    #[test]
    fn first_error_is_input_ordered_and_stops_later_tranches() {
        let touched_later_tranche = AtomicBool::new(false);
        let result: Result<Vec<(usize, usize)>> =
            execute_override_rebounds(&[0, 1, 2], true, |idx| match idx {
                0 => Err(NyError::InternalError("first-domain-error".to_string())),
                1 => Err(NyError::InternalError("second-domain-error".to_string())),
                _ => {
                    touched_later_tranche.store(true, Ordering::Relaxed);
                    Ok(idx)
                }
            });
        let error = result.unwrap_err();
        assert!(error.to_string().contains("first-domain-error"));
        assert!(!touched_later_tranche.load(Ordering::Relaxed));
    }

    #[test]
    fn process_envelope_admission_is_conservative_and_overflow_safe() {
        let estimate = override_rebound_worker_estimate_bytes(10, 2);
        assert_eq!(estimate, OVERRIDE_REBOUND_MIN_WORKER_BYTES);
        let required = estimate * OVERRIDE_REBOUND_HEADROOM_FACTOR;
        assert!(override_rebound_has_process_headroom(
            estimate,
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: required as u64,
            }
        ));
        assert!(!override_rebound_has_process_headroom(
            estimate,
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: required as u64 - 1,
            }
        ));
        assert!(!override_rebound_has_process_headroom(
            estimate,
            ProcessMemoryEnvelope::Unbounded
        ));
        assert!(!override_rebound_has_process_headroom(
            estimate,
            ProcessMemoryEnvelope::Unavailable
        ));
        assert!(!override_rebound_has_process_headroom(
            usize::MAX,
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: u64::MAX,
            }
        ));
    }

    #[test]
    fn typed_gate_runtime_kill_switch_and_shortcut_rollback_are_composed() {
        let _gate_guard = SPEC_GATE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        struct OverrideForceReset;
        impl Drop for OverrideForceReset {
            fn drop(&mut self) {
                force_override_rebound_parallel(None);
            }
        }
        let _force_reset = OverrideForceReset;

        let shortcut_only = BetaCrownConfig {
            input_split_collection_verify_shortcut: true,
            ..BetaCrownConfig::default()
        };
        force_override_rebound_parallel(Some(true));
        assert!(!override_rebound_parallel_enabled(&shortcut_only));
        assert!(collection_verify_shortcut_enabled(&shortcut_only));

        let armed = BetaCrownConfig {
            input_split_collection_verify_shortcut: true,
            input_split_override_parallel: true,
            ..BetaCrownConfig::default()
        };
        force_override_rebound_parallel(Some(false));
        assert!(!override_rebound_parallel_enabled(&armed));
        assert!(collection_verify_shortcut_enabled(&armed));

        force_override_rebound_parallel(Some(true));
        assert!(override_rebound_parallel_enabled(&armed));
        assert!(!collection_verify_shortcut_enabled(&armed));
    }

    #[test]
    fn active_workers_never_exceed_the_fixed_width() {
        let active = AtomicUsize::new(0);
        let max_active = AtomicUsize::new(0);
        let first_tranche_barrier = Barrier::new(2);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let results = pool
            .install(|| {
                execute_override_rebounds(&[0, 1, 2, 3], true, |idx| {
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active.fetch_max(now, Ordering::SeqCst);
                    if idx < 2 {
                        first_tranche_barrier.wait();
                    }
                    std::thread::yield_now();
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok(idx)
                })
            })
            .unwrap();
        assert_eq!(results, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
        assert_eq!(max_active.load(Ordering::SeqCst), 2);
    }
}
