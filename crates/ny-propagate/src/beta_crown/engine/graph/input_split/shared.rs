// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared domain types, priority ordering, and helper functions for
//! both single-objective and multi-objective input-split verifiers.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use super::bounds_eval::{
    build_input_split_reference_bounds, compute_plain_crown_or_ibp_bounds_with_node_bounds,
};
#[cfg(test)]
use crate::beta_crown::engine::{
    graph::domain_batch::{DenseSpecBatchRequest, GraphDomainBatchExecutor},
    tensor_ext::BoundedTensorExt,
};
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::GraphNetwork;

pub(crate) use super::bounds_eval::{
    graph_crown_error_should_fallback, graph_ibp_prescreen_error_should_skip,
    graph_output_bounds_are_finite, graph_spec_crown_with_mul_binary_and_truncation,
    graph_spec_ibp_fallback, graph_spec_ibp_root_screen_with_deadline,
    try_graph_spec_ibp_prescreen_bounds,
};

#[derive(Debug, Clone)]
pub(super) struct GraphInputDomain {
    pub(super) input_bounds: Arc<BoundedTensor>,
    pub(super) lower_bound: f32,
    pub(super) upper_bound: f32,
    pub(super) depth: usize,
    pub(super) priority: f32,
    pub(super) linear_bounds: Option<LinearBounds>,
    /// True if this domain needs CROWN bounding before it can be split.
    /// Used in reorder_bab mode: children are enqueued without bounds and
    /// bounded when popped. Reference: alpha-beta-CROWN reorder_bab.
    pub(super) needs_bounding: bool,
    /// Child-local node-bounds override carried by complete clipping until the
    /// deferred CROWN pass consumes it.
    pub(super) node_bounds_override: Option<Arc<HashMap<String, BoundedTensor>>>,
    /// Per-sub-domain refined α slopes inherited from the parent (warm-start).
    ///
    /// Only populated when `input_split_alpha_iteration > 0` (the lightweight
    /// per-sub-domain α-refinement path, mirroring alpha-beta-CROWN's input-split
    /// BaB). When set, a child warm-starts its α optimization from these slopes
    /// instead of the frozen root α state. `None` = ny's historical behavior
    /// (frozen root α threaded into every per-domain pass). Wrapped in `Arc` so
    /// attaching the same parent slopes to multiple children is cheap.
    ///
    /// SOUND: any α in [0,1] yields a valid CROWN over-approximation, and a
    /// parent's slopes are a feasible start for a child whose box is a SUBSET of
    /// the parent's; `warm_start` re-clamps to [0,1] regardless.
    pub(super) inherited_alpha_state: Option<Arc<GraphAlphaState>>,
}

pub(super) fn cmp_input_domain_priority(lhs: f32, rhs: f32) -> std::cmp::Ordering {
    match (lhs.is_nan(), rhs.is_nan()) {
        // Surface invalid domains immediately instead of treating them as arbitrary equals.
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (true, true) => std::cmp::Ordering::Equal,
        (false, false) => lhs.total_cmp(&rhs),
    }
}

impl PartialEq for GraphInputDomain {
    fn eq(&self, other: &Self) -> bool {
        cmp_input_domain_priority(self.priority, other.priority) == std::cmp::Ordering::Equal
    }
}

impl Eq for GraphInputDomain {}

impl PartialOrd for GraphInputDomain {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GraphInputDomain {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher priority = pop first
        cmp_input_domain_priority(self.priority, other.priority)
    }
}

/// Batched scalar objective bounds for single-objective input-split paths.
///
/// This carrier intentionally stores one scalar lower/upper pair per domain.
/// Multi-row spec matrices need a different result shape because collapsing a
/// dense spec-guided tensor through `lower_scalar()`/`upper_scalar()` loses the
/// per-objective semantics.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct BatchedScalarBounds {
    pub(crate) lower_bounds: Vec<f32>,
    pub(crate) upper_bounds: Vec<f32>,
    pub(crate) linear_bounds: Vec<Option<LinearBounds>>,
}

#[cfg(test)]
#[path = "shared_tests.rs"]
mod tests;

/// `#cgan-child-interm-refine`: give each BaB child its OWN intermediate-bound
/// map instead of the frozen ROOT one.
///
/// WHY HERE, AND NOT AT THE WARM-ALPHA SEAM. `warm_start_alpha_config` below
/// clears `cgan_complete_crown_ibp_root` on every child, which reads like the
/// natural place to re-enable it. It is not: the warm-alpha path is gated on
/// `use_alpha_crown` in BOTH its callers — `eager_warm_alpha_enabled`
/// (disjunctive_multi_clause.rs, which additionally requires `!reorder_bab`)
/// and `warm_refine_candidates` (batching.rs) — and the shipped cGAN preset
/// sets `solver.bound_prop_method: crown`, i.e. `use_alpha_crown = false`. So
/// `compute_warm_start_crown_bounds_with_refined_alpha` is NEVER CALLED on the
/// lane this lever exists for, and a switch flipped there is dead capability.
/// cGAN children are bounded by `compute_crown_or_ibp_bounds_with_node_bounds`
/// against the frozen root map — which is exactly the "children clear the
/// cascade" fact the lever means to change.
///
/// The refined map is therefore installed as the child's
/// `node_bounds_override` in the deferred-rebound prepass, a route both the
/// batched and override paths already consume. (`reorder_bab: true`, shipped
/// for cGAN, is what defers children in the first place, so this seam is
/// exactly the one that lane uses.)
///
/// SOUNDNESS. Enclosure SELECTION, never a looser relaxation. The map is
/// produced by the same collector the root lane ships, run on THIS child's
/// box: a certified forward-linear seed plus a shrink-only CROWN pass over it.
/// It is therefore a valid enclosure of the child box's reachable set — and so
/// is the frozen root map, since the root box CONTAINS the child box.
///
/// The downstream bounder does not choose between them. `child_node_bounds`
/// reaches `build_input_split_reference_bounds`, which merges through
/// `merge_tighter_bounds`: elementwise `max` of the lowers and `min` of the
/// uppers — the INTERSECTION — with NaN candidates discarded and disjoint
/// intervals falling back to the current value. The intersection of two sets
/// that each contain the reachable set contains the reachable set. So this
/// prepass cannot exclude a reachable point unless the shipped root cascade
/// already does, and it cannot drop a node either: the root map's entries
/// survive the merge whether or not this map has them.
///
/// Three fail-closed guards:
///   * a domain that already carries an override (complete clipping / fresh
///     domain clip — cgan2026 arms BOTH, so this is the common case, not the
///     rare one) keeps every entry of it; the refined map is intersected in,
///     never substituted for it. Skipping those domains instead would have
///     made the lever a near-total no-op on the only lane it exists for;
///   * a collector error leaves the domain on its historical frozen path;
///   * the map is installed ONLY when the collector REPORTS the cGAN cascade
///     as its source. A silently-refused cascade would otherwise install the
///     ordinary map and read as "the lever ran", which is the vacuous-null
///     shape this codebase keeps paying for.
pub(super) fn child_interm_refine_alpha_config(
    config: &crate::beta_crown::config::BetaCrownConfig,
    deadline: Option<Instant>,
) -> Option<crate::AlphaCrownConfig> {
    let max_secs = config
        .alpha_config
        .cgan_child_interm_refine_max_secs
        .filter(|secs| secs.is_finite() && *secs > 0.0)?;
    if !config.alpha_config.cgan_complete_crown_ibp_root {
        return None;
    }
    let mut alpha_config = config.alpha_config.clone();
    // The cascade's own eligibility predicate demands the fixed-intermediate
    // contract and refuses the other typed cGAN collector.
    alpha_config.fix_interm_bounds = true;
    alpha_config.cgan_sparse_target_complete_root = false;
    alpha_config.cgan_complete_crown_ibp_root = true;
    alpha_config.deadline = child_interm_refine_deadline(deadline, max_secs);
    child_interm_refine_meter::armed(max_secs);
    Some(alpha_config)
}

/// `min(caller ceiling, now + per-child budget)`. Like `child_bound_deadline`,
/// this can only ever bring the deadline NEARER, never extend it.
fn child_interm_refine_deadline(ceiling: Option<Instant>, max_secs: f64) -> Option<Instant> {
    let cap = Instant::now() + std::time::Duration::from_secs_f64(max_secs);
    Some(ceiling.map_or(cap, |ceiling| ceiling.min(cap)))
}

/// Outcome of one deferred batch's refinement prepass, for telemetry only.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct ChildIntermRefineBatch {
    pub(super) installed: usize,
    pub(super) merged_into_existing_override: usize,
    pub(super) declined: usize,
    // --- #cgan-child-seed (all zero on the historical serial, unseeded path) ---
    /// Children whose cascade ran from a SEEDED baseline.
    pub(super) seeded: usize,
    /// Children the seed was requested for but that had no inherited map at
    /// all (no root map and no override), so they took the unseeded cascade.
    pub(super) seed_unavailable: usize,
    /// Baseline nodes intersected with an inherited entry, summed over children.
    pub(super) seed_nodes: usize,
    /// Baseline nodes whose inherited entry was mismatched/NaN/disjoint and
    /// which kept their baseline entry, summed over children.
    pub(super) seed_declined_nodes: usize,
    /// Elements strictly tightened by the seed, summed over children.
    pub(super) seed_tightened_elems: usize,
    /// Sign-unstable ReLU pre-activation rows under the UNSEEDED child
    /// baseline, summed over seeded children.
    pub(super) unstable_rows_unseeded: usize,
    /// The same count under the SEEDED baseline the cascades received.
    pub(super) unstable_rows_seeded: usize,
    /// Widest tranche actually run (1 = serial).
    pub(super) max_width: usize,
    /// Tranches run (serial counts each child as its own tranche).
    pub(super) tranches: usize,
    /// Tranches whose admission fell back to serial on memory grounds.
    pub(super) memory_declined_tranches: usize,
    /// Sum of per-child cascade walls, nanoseconds.
    pub(super) child_nanos: u64,
}

/// `#cgan-child-seed`: how one deferred batch's refinement prepass is run.
///
/// `ChildRefineSchedule::serial_unseeded` is the historical prepass exactly;
/// the production schedule is built by `child_seed::child_refine_plan`.
#[derive(Debug, Clone, Copy)]
pub(super) struct ChildRefineSchedule {
    /// Seed each child cascade's forward-linear baseline with the domain's
    /// inherited map (root map, then the domain's own override).
    pub(super) seed_from_parent: bool,
    /// Concurrency ceiling (`0`/`1` = serial). Every tranche is re-admitted.
    pub(super) requested_width: usize,
    /// Conservative per-worker resident-set estimate, bytes.
    pub(super) worker_estimate_bytes: u64,
    /// Live memory-headroom probe consulted before every parallel tranche;
    /// `None` means unobservable, which runs that tranche serially.
    pub(super) headroom_probe: fn() -> Option<u64>,
}

impl ChildRefineSchedule {
    /// The historical prepass: serial, unseeded, no admission probe consulted.
    #[cfg(test)]
    pub(super) fn serial_unseeded() -> Self {
        Self {
            seed_from_parent: false,
            requested_width: 1,
            worker_estimate_bytes: 0,
            headroom_probe: || None,
        }
    }
}

/// One deferred child's inputs, snapshotted before any cascade runs so the
/// cascades can execute concurrently without touching the domain slice.
struct ChildRefineJob {
    idx: usize,
    child_box: Arc<BoundedTensor>,
    existing: Option<Arc<HashMap<String, BoundedTensor>>>,
}

/// `#cgan-child-seed` seed telemetry for one child, copied out of the
/// collector's report.
#[derive(Debug, Default, Clone, Copy)]
struct ChildSeedCounts {
    nodes_seeded: usize,
    nodes_declined: usize,
    elems_tightened: usize,
    unstable_rows_unseeded: usize,
    unstable_rows_seeded: usize,
}

/// What one child cascade produced. Only `Refined` maps came from the complete
/// cGAN cascade itself (asserted on the collector's own reported source).
enum ChildRefineResult {
    Refined {
        map: HashMap<String, BoundedTensor>,
        // Boxed: the five counters made `Refined` 97 bytes against
        // `Declined`'s 1 (clippy::large_enum_variant, workspace threshold 64).
        seed: Option<Box<ChildSeedCounts>>,
        seed_unavailable: bool,
    },
    Declined {
        seed_unavailable: bool,
    },
}

/// The two input-split domain carriers, seen through the only two fields this
/// prepass touches.
///
/// A trait rather than accessor closures on purpose: a closure returning
/// `&mut` out of `&mut D` needs a higher-ranked bound that inference will not
/// produce, and spelling `for<'a>` at two call sites is strictly worse than
/// naming the two fields once.
pub(super) trait ChildIntermRefineDomain {
    fn refine_input_bounds(&self) -> &Arc<BoundedTensor>;
    fn refine_override_slot(&mut self) -> &mut Option<Arc<HashMap<String, BoundedTensor>>>;
}

impl ChildIntermRefineDomain for GraphInputDomain {
    fn refine_input_bounds(&self) -> &Arc<BoundedTensor> {
        &self.input_bounds
    }
    fn refine_override_slot(&mut self) -> &mut Option<Arc<HashMap<String, BoundedTensor>>> {
        &mut self.node_bounds_override
    }
}

impl ChildIntermRefineDomain for MultiObjInputDomain {
    fn refine_input_bounds(&self) -> &Arc<BoundedTensor> {
        &self.input_bounds
    }
    fn refine_override_slot(&mut self) -> &mut Option<Arc<HashMap<String, BoundedTensor>>> {
        &mut self.node_bounds_override
    }
}

/// Intersect a freshly collected, child-box-specific intermediate map into each
/// deferred domain's `node_bounds_override`. Returns the per-batch tally; a
/// no-op — and no collector work at all — when the lever is dark, because the
/// caller only obtains an `alpha_config` for it from
/// [`child_interm_refine_alpha_config`], which returns `None` then.
///
/// This is the historical serial, unseeded prepass, kept as the reference the
/// `#cgan-child-seed` schedule is tested against; production calls
/// [`install_child_interm_refine_overrides_scheduled`] with the schedule from
/// `child_seed::child_refine_plan`, which is `serial_unseeded` while dark.
#[cfg(test)]
pub(super) fn install_child_interm_refine_overrides<D: ChildIntermRefineDomain>(
    alpha_config: &crate::AlphaCrownConfig,
    graph: &GraphNetwork,
    engine: Option<&dyn GemmEngine>,
    deferred_indices: &[usize],
    domains: &mut [D],
) -> ChildIntermRefineBatch {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::IntermRefresh);
    install_child_interm_refine_overrides_scheduled(
        alpha_config,
        graph,
        engine,
        None,
        ChildRefineSchedule::serial_unseeded(),
        deferred_indices,
        domains,
    )
}

/// `install_child_interm_refine_overrides` under a `#cgan-child-seed`
/// schedule: optionally SEEDED from `inherited_root` plus each domain's own
/// override, optionally run in admitted parallel tranches.
///
/// DETERMINISM. Every child's cascade reads only its own snapshotted box and
/// override, the shared read-only config and `inherited_root`; results are
/// applied to `domains` strictly in `deferred_indices` order after each
/// tranche completes. With `serial_unseeded()` this is the historical loop
/// call for call (same collector, same order, same merge).
///
/// SOUNDNESS. Unchanged from the historical prepass: a map is installed only
/// when the collector REPORTS the complete cGAN cascade as its source, and it
/// is intersected into (never substituted for) the existing override. The seed
/// only changes the cascade's starting baseline to an intersection of
/// certified enclosures of the same child box (see
/// `network/graph_alpha/bounds/alpha_child_seed.rs`).
pub(super) fn install_child_interm_refine_overrides_scheduled<D: ChildIntermRefineDomain>(
    alpha_config: &crate::AlphaCrownConfig,
    graph: &GraphNetwork,
    engine: Option<&dyn GemmEngine>,
    inherited_root: Option<&HashMap<String, BoundedTensor>>,
    schedule: ChildRefineSchedule,
    deferred_indices: &[usize],
    domains: &mut [D],
) -> ChildIntermRefineBatch {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::IntermRefresh);
    let mut tally = ChildIntermRefineBatch::default();
    let Ok(exec_order) = graph.exec_order() else {
        return tally;
    };
    let jobs: Vec<ChildRefineJob> = deferred_indices
        .iter()
        .filter_map(|&idx| {
            let domain = domains.get_mut(idx)?;
            let existing = domain.refine_override_slot().clone();
            let child_box = Arc::clone(domain.refine_input_bounds());
            Some(ChildRefineJob {
                idx,
                child_box,
                existing,
            })
        })
        .collect();
    let run = |job: &ChildRefineJob| -> (ChildRefineResult, u64) {
        let started = Instant::now();
        let result = run_child_refine_job(
            alpha_config,
            graph,
            engine,
            exec_order,
            inherited_root,
            schedule.seed_from_parent,
            job,
        );
        let elapsed = started.elapsed();
        child_interm_refine_meter::observe(elapsed);
        (
            result,
            u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        )
    };

    let width_ceiling = schedule.requested_width.max(1);
    if width_ceiling <= 1 || jobs.len() <= 1 {
        for job in &jobs {
            let (result, nanos) = run(job);
            tally.tranches += 1;
            tally.child_nanos = tally.child_nanos.saturating_add(nanos);
            apply_child_refine_result(domains, job, result, &mut tally);
        }
        tally.max_width = tally.max_width.max(usize::from(!jobs.is_empty()));
        return tally;
    }

    // One parallel prepass at a time per process: each tranche's admission
    // samples live headroom, and two concurrent prepasses admitting against
    // the same sample would each spend it.
    let _execution_guard = super::child_seed::CHILD_REFINE_EXECUTION_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut cursor = 0usize;
    while cursor < jobs.len() {
        let admission = super::child_seed::admitted_width(
            width_ceiling,
            schedule.worker_estimate_bytes,
            (schedule.headroom_probe)(),
        );
        if admission.memory_declined {
            tally.memory_declined_tranches += 1;
        }
        let width = admission.width.clamp(1, jobs.len() - cursor);
        let tranche = &jobs[cursor..cursor + width];
        let results: Vec<(ChildRefineResult, u64)> = if width > 1 {
            use rayon::prelude::*;
            tranche
                .par_iter()
                .map(|job| {
                    // Same contract as every per-domain Rayon CROWN caller:
                    // nested faer work stays sequential inside the worker.
                    let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                    run(job)
                })
                .collect()
        } else {
            tranche.iter().map(&run).collect()
        };
        for (job, (result, nanos)) in tranche.iter().zip(results) {
            tally.child_nanos = tally.child_nanos.saturating_add(nanos);
            apply_child_refine_result(domains, job, result, &mut tally);
        }
        tally.tranches += 1;
        tally.max_width = tally.max_width.max(width);
        cursor += width;
    }
    tally
}

/// Run ONE child's cascade. Pure in its inputs: it never touches the domain
/// slice, which is what lets tranches run concurrently.
fn run_child_refine_job(
    alpha_config: &crate::AlphaCrownConfig,
    graph: &GraphNetwork,
    engine: Option<&dyn GemmEngine>,
    exec_order: &[String],
    inherited_root: Option<&HashMap<String, BoundedTensor>>,
    seed_from_parent: bool,
    job: &ChildRefineJob,
) -> ChildRefineResult {
    let mut seed_unavailable = false;
    if seed_from_parent {
        // Root map first, then the domain's own override: each is a certified
        // enclosure of this child box's image (see `alpha_child_seed.rs`).
        let seeds: Vec<&HashMap<String, BoundedTensor>> = inherited_root
            .into_iter()
            .chain(job.existing.as_deref())
            .collect();
        if seeds.is_empty() {
            seed_unavailable = true;
        } else {
            return match graph.collect_cgan_child_reference_bounds_seeded(
                job.child_box.as_ref(),
                alpha_config,
                engine,
                exec_order,
                &seeds,
            ) {
                Ok(Some((map, source, report))) if source.is_cgan_complete_crown_ibp() => {
                    ChildRefineResult::Refined {
                        map,
                        seed: Some(Box::new(ChildSeedCounts {
                            nodes_seeded: report.nodes_seeded,
                            nodes_declined: report.nodes_declined,
                            elems_tightened: report.elems_tightened,
                            unstable_rows_unseeded: report.unstable_rows_unseeded,
                            unstable_rows_seeded: report.unstable_rows_seeded,
                        })),
                        seed_unavailable: false,
                    }
                }
                // Refused, errored, or not the cascade: historical path.
                _ => ChildRefineResult::Declined {
                    seed_unavailable: false,
                },
            };
        }
    }
    match graph.collect_alpha_reference_bounds_with_engine_and_source(
        job.child_box.as_ref(),
        alpha_config,
        engine,
        exec_order,
    ) {
        Ok((map, source)) if source.is_cgan_complete_crown_ibp() => ChildRefineResult::Refined {
            map,
            seed: None,
            seed_unavailable,
        },
        // Refused or errored: leave the domain on its historical frozen
        // path. Never install a map the cascade did not produce.
        _ => ChildRefineResult::Declined { seed_unavailable },
    }
}

/// Install one child's result exactly as the historical loop did: shrink-only
/// against whatever the domain already had; on a merge error the domain keeps
/// its existing override, i.e. its historical path.
fn apply_child_refine_result<D: ChildIntermRefineDomain>(
    domains: &mut [D],
    job: &ChildRefineJob,
    result: ChildRefineResult,
    tally: &mut ChildIntermRefineBatch,
) {
    match result {
        ChildRefineResult::Refined {
            map,
            seed,
            seed_unavailable,
        } => {
            tally.seed_unavailable += usize::from(seed_unavailable);
            if let Some(counts) = seed {
                tally.seeded += 1;
                tally.seed_nodes += counts.nodes_seeded;
                tally.seed_declined_nodes += counts.nodes_declined;
                tally.seed_tightened_elems += counts.elems_tightened;
                tally.unstable_rows_unseeded += counts.unstable_rows_unseeded;
                tally.unstable_rows_seeded += counts.unstable_rows_seeded;
            }
            let merged = match crate::network::merge_reference_bound_maps(
                job.existing.as_deref(),
                Some(&map),
            ) {
                Ok(Some(merged)) => merged,
                Ok(None) | Err(_) => {
                    tally.declined += 1;
                    return;
                }
            };
            let Some(domain) = domains.get_mut(job.idx) else {
                tally.declined += 1;
                return;
            };
            *domain.refine_override_slot() = Some(Arc::new(merged));
            tally.installed += 1;
            if job.existing.is_some() {
                tally.merged_into_existing_override += 1;
            }
        }
        ChildRefineResult::Declined { seed_unavailable } => {
            tally.seed_unavailable += usize::from(seed_unavailable);
            tally.declined += 1;
        }
    }
}

/// `#cgan-child-interm-refine` observable and per-domain cost meter.
///
/// The ARMED line is the lever's own proof of life: a run whose preset failed
/// to reach the cell is then a VOID rather than a silent negative. The periodic
/// line carries the number this direction lives or dies on — the per-domain
/// wall of a refined child — measured at the seam that pays it.
mod child_interm_refine_meter {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Once;
    use std::time::Duration;

    static ARMED: Once = Once::new();
    static CHILDREN: AtomicU64 = AtomicU64::new(0);
    static NANOS: AtomicU64 = AtomicU64::new(0);

    /// Report the first few children, then every 32nd: enough to separate
    /// warm-up cost from steady state without flooding a 900 s run's log.
    const REPORT_EVERY: u64 = 32;
    const REPORT_FIRST: u64 = 4;

    pub(super) fn armed(max_secs: f64) {
        ARMED.call_once(|| {
            eprintln!(
                "[cgan-child-interm-refine] armed max_secs={max_secs} \
                 route=input-split-deferred-rebound-prepass \
                 source=bab.alpha_crown.cgan_child_interm_refine_max_secs"
            );
        });
    }

    pub(super) fn observe(elapsed: Duration) {
        let n = CHILDREN.fetch_add(1, Ordering::Relaxed) + 1;
        let total = NANOS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed)
            + elapsed.as_nanos() as u64;
        if n <= REPORT_FIRST || n.is_multiple_of(REPORT_EVERY) {
            eprintln!(
                "[cgan-child-interm-refine] children={n} last={:.3}s mean={:.3}s",
                elapsed.as_secs_f64(),
                total as f64 / 1e9 / n as f64,
            );
        }
    }
}

/// Compute spec-guided CROWN bounds with IBP fallback for a sub-domain.
///
/// Tries CROWN with alpha/MulBinary alpha state; falls back to IBP on recoverable
/// errors. Used by both single-objective and multi-objective verifier loops.
///
/// When `ibp_enhancement` is true, runs a cheap IBP forward pass on the subdomain
/// first, builds a reference-bounds map from the IBP/inherited bounds, and then
/// still runs a fresh spec-guided CROWN pass that tightens against those
/// references. This matches alpha-beta-CROWN's `reference_bounds` contract
/// instead of freezing IBP as the active intermediate-bounds map.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_crown_or_ibp_bounds(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    ibp_enhancement: bool,
) -> Result<(BoundedTensor, Option<LinearBounds>)> {
    compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        input_bounds,
        spec_matrix,
        engine,
        alpha_node_bounds,
        None,
        alpha_state,
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
        ibp_enhancement,
    )
}

/// Like `compute_crown_or_ibp_bounds`, but allows a child-local node-bounds
/// override to replace the shared root alpha/IBP cache.
///
/// When `ibp_enhancement` is true, builds a reference-bounds map from the
/// inherited alpha/clip bounds plus current-domain IBP, then runs fresh CROWN
/// with that reference map so nonlinear relaxations see the tighter of
/// `{fresh CROWN, reference}` at each node.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_crown_or_ibp_bounds_with_node_bounds(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    child_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    ibp_enhancement: bool,
) -> Result<(BoundedTensor, Option<LinearBounds>)> {
    let fixed_node_bounds = child_node_bounds.or(alpha_node_bounds);
    if !ibp_enhancement {
        return compute_plain_crown_or_ibp_bounds_with_node_bounds(
            graph,
            input_bounds,
            spec_matrix,
            engine,
            fixed_node_bounds,
            alpha_state,
            mul_binary_alphas,
            deadline,
            crown_backward_layers,
        );
    }

    let cached_fallback_bounds = match build_input_split_reference_bounds(
        alpha_node_bounds,
        child_node_bounds,
        None,
    ) {
        Ok(bounds) => bounds,
        Err(err) => {
            tracing::debug!(
                    "IBP-enhanced reference-bounds merge failed before fresh IBP; retrying plain input-split bounds: {err}"
                );
            return compute_plain_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                fixed_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
            );
        }
    };
    let ibp_reference_bounds =
        // Keep the IBP reference-bounds pass under the caller deadline budget;
        // if it expires, avoid routing through an uncapped collection first.
        match graph.collect_node_bounds_with_engine_and_deadline(input_bounds, engine, deadline) {
            Ok(bounds) => Some(bounds),
            Err(NyError::DeadlineExceeded(_)) => {
                if let Some(bounds) = cached_fallback_bounds.as_ref() {
                    tracing::debug!(
                        "IBP-enhanced reference-bounds exceeded deadline, reusing cached node bounds for plain IBP fallback"
                    );
                    return graph_spec_ibp_fallback(
                        graph,
                        input_bounds,
                        spec_matrix,
                        engine,
                        Some(bounds),
                    );
                }
                tracing::debug!(
                    "IBP-enhanced reference-bounds exceeded deadline with no cached node bounds; returning conservative spec bounds"
                );
                return Ok((
                    BoundedTensor::new_conservative(&[spec_matrix.nrows()]),
                    None,
                ));
            }
            Err(err) => {
                tracing::debug!(
                    "IBP-enhanced reference-bounds setup failed; retrying plain input-split bounds: {err}"
                );
                return compute_plain_crown_or_ibp_bounds_with_node_bounds(
                    graph,
                    input_bounds,
                    spec_matrix,
                    engine,
                    fixed_node_bounds,
                    alpha_state,
                    mul_binary_alphas,
                    deadline,
                    crown_backward_layers,
                );
            }
        };
    let reference_node_bounds = match build_input_split_reference_bounds(
        alpha_node_bounds,
        child_node_bounds,
        ibp_reference_bounds.as_ref(),
    ) {
        Ok(bounds) => bounds,
        Err(err) => {
            tracing::debug!(
                "IBP-enhanced reference-bounds merge with fresh IBP failed; retrying plain input-split bounds: {err}"
            );
            return compute_plain_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                fixed_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
            );
        }
    };

    let crown_result = graph_spec_crown_with_mul_binary_and_truncation(
        graph,
        input_bounds,
        spec_matrix,
        engine,
        None,
        reference_node_bounds.as_ref(),
        alpha_state,
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
    );
    match crown_result {
        Ok(result) if graph_output_bounds_are_finite(&result.0) => Ok(result),
        Ok((_bounds, _linear)) => {
            tracing::debug!(
                "IBP-enhanced spec-guided CROWN produced non-finite bounds; retrying plain input-split bounds"
            );
            compute_plain_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                fixed_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
            )
        }
        Err(NyError::DeadlineExceeded(_)) => {
            tracing::debug!(
                "IBP-enhanced spec-guided CROWN exceeded deadline, falling back to IBP"
            );
            let ibp_result = graph_spec_ibp_fallback(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                reference_node_bounds.as_ref(),
            )?;
            if graph_output_bounds_are_finite(&ibp_result.0) {
                Ok(ibp_result)
            } else {
                tracing::debug!(
                    "IBP-enhanced deadline fallback produced non-finite bounds; retrying plain input-split bounds"
                );
                compute_plain_crown_or_ibp_bounds_with_node_bounds(
                    graph,
                    input_bounds,
                    spec_matrix,
                    engine,
                    fixed_node_bounds,
                    alpha_state,
                    mul_binary_alphas,
                    deadline,
                    crown_backward_layers,
                )
            }
        }
        Err(err) => {
            tracing::debug!(
                "IBP-enhanced spec-guided CROWN failed; retrying plain input-split bounds: {err}"
            );
            compute_plain_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds,
                spec_matrix,
                engine,
                fixed_node_bounds,
                alpha_state,
                mul_binary_alphas,
                deadline,
                crown_backward_layers,
            )
        }
    }
}

/// Lightweight per-sub-domain α refinement for the single-objective input-split
/// BaB loop, mirroring alpha-beta-CROWN's `input_split/bounding.py:90-179`.
///
/// Each sub-domain warm-starts from its parent's optimized alphas and re-optimizes
/// them for `config.input_split_alpha_iteration` SPSA iterations at
/// `lr = config.input_split_lr_alpha`, with `fix_interm_bounds = true` (skipping
/// the O(N²) intermediate CROWN pass). Only a handful of iterations are needed
/// because the parent slopes are already near-optimal for the child's tighter box.
///
/// Returns the spec-guided objective bounds + linear bounds (computed with the
/// REFINED alphas and refined intermediate bounds) AND the refined `GraphAlphaState`
/// so it can be saved onto BOTH child sub-domains as their warm-start seed,
/// AND the refined per-node bounds bundled with the exact `input_bounds` they
/// were computed for ([`super::f64_tail::F64TailAnchors`], design B step B3 +
/// FIX B4): the certified f64 tail must anchor on the SAME warm-refined boxes
/// the f32 pass used — falling back to a fresh `collect_intermediate_bounds`
/// gives looser anchors and a vacuous NotVerified that reads as "fp is not
/// the problem". Bundling here (the one site that has both the box and the
/// map from one invocation) makes cross-box anchor reuse unrepresentable.
///
/// SOUND: `collect_alpha_crown_bounds_dag_warm_with_engine` keeps every α in [0,1]
/// (so every per-domain bound is a valid CROWN over-approximation), and the final
/// spec-guided pass is the same sound CROWN-with-IBP-fallback used by the frozen
/// path — only the α values and the intermediate-bound map differ. This helper
/// propagates optimization or bound failures; each caller decides whether to
/// retry its historical frozen-alpha computation or fail closed. Callers must
/// only invoke this when `input_split_alpha_iteration > 0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_warm_start_crown_bounds_with_refined_alpha(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    child_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    parent_alpha: &GraphAlphaState,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    config: &crate::beta_crown::config::BetaCrownConfig,
) -> Result<(
    BoundedTensor,
    Option<LinearBounds>,
    GraphAlphaState,
    super::f64_tail::F64TailAnchors,
)> {
    let alpha_config = warm_start_alpha_config(config, deadline);

    // Step 1: warm-start + re-optimize alphas for this sub-domain's tighter box.
    let (refined_node_bounds, refined_alpha) = graph
        .collect_alpha_crown_bounds_dag_warm_with_engine(
            input_bounds,
            &alpha_config,
            parent_alpha,
            engine,
        )?;

    // Step 2: spec-guided objective pass with the REFINED alphas + refined
    // intermediate bounds. A child-local node-bounds override (e.g. from complete
    // clipping) still takes precedence over the refined map, exactly as in the
    // frozen path. Reuses the same sound CROWN-with-IBP-fallback dispatch.
    let (bounds, linear) = compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        input_bounds,
        spec_matrix,
        engine,
        Some(&refined_node_bounds),
        child_node_bounds,
        Some(&refined_alpha),
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
        config.input_split_ibp_enhancement,
    )?;

    // FIX B4: bundle the refined node bounds with the EXACT box they were
    // computed for, at the only site that has both from one invocation.
    let anchors = super::f64_tail::F64TailAnchors::bundle_for(input_bounds, refined_node_bounds);
    Ok((bounds, linear, refined_alpha, anchors))
}

/// Derive the deliberately lightweight child-domain alpha configuration.
///
/// Keeping this as one pure helper makes the root/child routing boundary
/// executable in tests: a root may opt into the cGAN target-complete collector,
/// but no inherited runtime flag may make a BaB child pay that root-only cost.
fn warm_start_alpha_config(
    config: &crate::beta_crown::config::BetaCrownConfig,
    deadline: Option<Instant>,
) -> crate::AlphaCrownConfig {
    let mut alpha_config = config.alpha_config.clone();
    alpha_config.iterations = config.input_split_alpha_iteration;
    alpha_config.learning_rate = config.input_split_lr_alpha;
    // Skip the O(N²) post-optimization intermediate CROWN pass: the warm-start
    // path returns the (sound) IBP intermediate bounds directly. Matches the
    // reference `fix_interm_bounds=True` for input-split refinement.
    alpha_config.fix_interm_bounds = true;
    alpha_config.cgan_sparse_target_complete_root = false;
    alpha_config.cgan_complete_crown_ibp_root = false;
    alpha_config.deadline = deadline;
    alpha_config
}

/// Compute CROWN/IBP bounds for a single-objective batch.
/// Enforces `spec_matrix.nrows() == 1` and extracts scalar lower/upper pairs.
/// Delegates to the dense-spec helper.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_crown_or_ibp_bounds_batched(
    graph: &GraphNetwork,
    input_bounds_batch: &[&BoundedTensor],
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
    ibp_enhancement: bool,
) -> Result<BatchedScalarBounds> {
    if spec_matrix.nrows() != 1 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "compute_crown_or_ibp_bounds_batched only supports single-row spec matrices; got {} rows",
            spec_matrix.nrows()
        )));
    }

    let spec_result = GraphDomainBatchExecutor::execute_dense_specs(DenseSpecBatchRequest {
        graph,
        input_bounds_batch,
        spec_matrix,
        engine,
        alpha_node_bounds,
        alpha_state,
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
        ibp_enhancement,
        stacked_rebound: false,
        nested_deadline: None,
    })?;

    let n = spec_result.bounds.len();
    let mut lower_bounds = Vec::with_capacity(n);
    let mut upper_bounds = Vec::with_capacity(n);

    for bounds in &spec_result.bounds {
        lower_bounds.push(bounds.lower_scalar());
        upper_bounds.push(bounds.upper_scalar());
    }

    Ok(BatchedScalarBounds {
        lower_bounds,
        upper_bounds,
        linear_bounds: spec_result.linear_bounds,
    })
}

/// Conjunctive domain check: verified if ANY objective has `lower > threshold`.
/// The proof-relevant lower endpoint and threshold must be finite, while a
/// `+inf` upper endpoint remains a valid one-sided enclosure. NaN and inverted
/// intervals never acquire proof authority.
pub(super) fn multi_obj_domain_verified(obj_bounds: &[(f32, f32)], thresholds: &[f32]) -> bool {
    if obj_bounds.is_empty() || obj_bounds.len() != thresholds.len() {
        return false;
    }
    obj_bounds
        .iter()
        .zip(thresholds.iter())
        .any(|(&(lower, upper), &threshold)| {
            super::grouped_semantics::objective_interval_verified(lower, upper, threshold)
        })
}

/// Conjunctive domain priority: use the closest-to-verified objective.
/// NaN bounds yield NEG_INFINITY priority.
pub(super) fn multi_obj_domain_priority(obj_bounds: &[(f32, f32)], thresholds: &[f32]) -> f32 {
    if obj_bounds.is_empty() || obj_bounds.len() != thresholds.len() {
        return f32::NEG_INFINITY;
    }
    obj_bounds
        .iter()
        .zip(thresholds.iter())
        .map(|(&(l, u), &t)| {
            if !l.is_finite() || !u.is_finite() || !t.is_finite() || l > u {
                return f32::NEG_INFINITY;
            }
            let gap = l - t;
            if gap.is_finite() {
                gap
            } else {
                f32::NEG_INFINITY
            }
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Extract per-objective bounds from a multi-row BoundedTensor.
///
/// Returns an error unless the tensor contains exactly one scalar interval per
/// objective row. Exact layout prevents trailing or non-contiguous storage
/// from being silently ignored by verdict code.
pub(crate) fn extract_obj_bounds(
    bounds: &BoundedTensor,
    num_specs: usize,
) -> Result<Vec<(f32, f32)>> {
    let lower = bounds.lower();
    let upper = bounds.upper();
    let n_lower = lower.len();
    let n_upper = upper.len();
    if num_specs == 0 || num_specs != n_lower || num_specs != n_upper {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_specs],
            got: vec![n_lower.max(n_upper)],
        });
    }
    let lower_flat = lower.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("objective lower bounds must be contiguous".to_string())
    })?;
    let upper_flat = upper.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("objective upper bounds must be contiguous".to_string())
    })?;
    let mut extracted = Vec::with_capacity(num_specs);
    for i in 0..num_specs {
        let (lower, upper) = (lower_flat[i], upper_flat[i]);
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(NyError::NumericalInstability(format!(
                "objective row {i} is malformed: lower={lower}, upper={upper}"
            )));
        }
        extracted.push((lower, upper));
    }
    Ok(extracted)
}

/// Build a child BoundedTensor from flat lower/upper arrays reshaped to `shape`.
pub(super) fn build_child_input(
    child_lower: &ndarray::ArrayD<f32>,
    child_upper: &ndarray::ArrayD<f32>,
    shape: &[usize],
) -> Result<BoundedTensor> {
    let lower_arr = child_lower
        .clone()
        .into_shape_clone(ndarray::IxDyn(shape))
        .map_err(|e| NyError::InvalidSpec(format!("reshape lower: {}", e)))?;
    let upper_arr = child_upper
        .clone()
        .into_shape_clone(ndarray::IxDyn(shape))
        .map_err(|e| NyError::InvalidSpec(format!("reshape upper: {}", e)))?;
    BoundedTensor::new(lower_arr, upper_arr)
}

/// Build a child BoundedTensor by consuming owned flat lower/upper arrays.
///
/// Identical result to `build_child_input`, but takes the arrays by value so a
/// caller that already owns the freshly-split boxes (and does not reuse them)
/// avoids two full-box clones per child. `into_shape_clone` only reallocates
/// when the array is not contiguous in the requested order; the input-split
/// boxes originate from `flatten()` (standard layout) so this is the same
/// bitwise reshape with no extra copy. The element values are untouched, so the
/// resulting bounds are bit-for-bit identical to the borrowing path.
pub(super) fn build_child_input_owned(
    child_lower: ndarray::ArrayD<f32>,
    child_upper: ndarray::ArrayD<f32>,
    shape: &[usize],
) -> Result<BoundedTensor> {
    let lower_arr = child_lower
        .into_shape_clone(ndarray::IxDyn(shape))
        .map_err(|e| NyError::InvalidSpec(format!("reshape lower: {}", e)))?;
    let upper_arr = child_upper
        .into_shape_clone(ndarray::IxDyn(shape))
        .map_err(|e| NyError::InvalidSpec(format!("reshape upper: {}", e)))?;
    BoundedTensor::new(lower_arr, upper_arr)
}

pub(super) fn scalar_output_bounds(lower: f32, upper: f32) -> Result<BoundedTensor> {
    BoundedTensor::new(
        Array1::from_vec(vec![lower]).into_dyn(),
        Array1::from_vec(vec![upper]).into_dyn(),
    )
}

/// Multi-dimensional midpoint input split (mirrors `create_input_split_children`):
/// midpoint-splits each dim in `split_dims`, producing up to 2^split_dims.len() child
/// boxes that EXACTLY COVER the parent box — every parent point lies in some child, so
/// BaB completeness (hence soundness of the verdict) is preserved. A dim that is out of
/// range, non-finite, or non-positive-width is left unsplit (the box passes through
/// whole). Boxes are flat, matching `build_child_input`'s expectation.
///
/// Shared by the single-objective and multi-objective conjunctive input-split loops
/// (moved from `single_objective/process_batch.rs` so both honor `input_split_depth`).
pub(super) fn multi_dim_split_boxes(
    flat_lower: ndarray::ArrayD<f32>,
    flat_upper: ndarray::ArrayD<f32>,
    split_dims: &[usize],
) -> Vec<(ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> {
    // The caller hands us the freshly-flattened root box by value, so it becomes
    // the initial box directly — no extra clone of the parent's lower/upper.
    let len = flat_lower.len();
    let mut boxes = vec![(flat_lower, flat_upper)];
    for &dim in split_dims {
        if dim >= len {
            continue;
        }
        let mut next = Vec::with_capacity(boxes.len() * 2);
        for (lo, hi) in boxes {
            let l = lo[[dim]];
            let u = hi[[dim]];
            if !l.is_finite() || !u.is_finite() || u <= l {
                next.push((lo, hi));
                continue;
            }
            let mid = l + (u - l) / 2.0;
            let mut left_hi = hi.clone();
            left_hi[[dim]] = mid;
            next.push((lo.clone(), left_hi));
            let mut right_lo = lo;
            right_lo[[dim]] = mid;
            next.push((right_lo, hi));
        }
        boxes = next;
    }
    boxes
}

pub(super) use super::multi_obj_domain::{MultiObjBounds, MultiObjInputDomain};
