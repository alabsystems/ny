// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core CROWN-IBP per-node tightening loop (#3596, #3499).

use crate::network::crown_quality::crown_gain_stat;
#[cfg(test)]
use crate::network::crown_quality::CrownGainStat;
use crate::types::{
    BoundsProvenance, CrownIbpFallbackEvent, CrownIbpFallbackReason, GraphCrownIbpBoundsResult,
};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use super::budget_policy;
use super::crown_repropagate;
use super::demand::{
    nodes_requiring_crown_tightening, relu_only_intermediate_target_is_eligible,
    relu_stability_profile, required_consumer_counts, sparse_interm_diag_enabled,
    sparse_relu_row_plan_for_target, sparse_relu_rows_enabled, SparseReluRowPlan,
};

/// Batteries-included gate for the adaptive hopeless-class collector skip
/// (#cifar100-collector-order): ON by default, opt out with
/// `NY_NO_HOPELESS_CLASS_SKIP=1` (disable-flag principle).
///
/// SOUND either way: the skip only ever substitutes IBP for CROWN, and IBP is a
/// valid enclosure. Disabling it restores the previous walk exactly.
fn hopeless_class_skip_from_raw(raw: Option<&str>) -> bool {
    raw != Some("1")
}

pub(super) fn hopeless_class_skip_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        hopeless_class_skip_from_raw(std::env::var("NY_NO_HOPELESS_CLASS_SKIP").ok().as_deref())
    })
}

fn is_structural_crown_skip(reason: CrownIbpFallbackReason) -> bool {
    matches!(
        reason,
        CrownIbpFallbackReason::DemandDrivenSkip
            | CrownIbpFallbackReason::StableReluRowsSkipped
            | CrownIbpFallbackReason::ObjectiveConeRowsSkipped
    )
}

fn sequential_fast_path_allowed(
    row_subset_engaged: bool,
    engine_present: bool,
    cpu_dense_chain: bool,
    width_threshold_present: bool,
    majority_skipped: bool,
    downstream_resweep: bool,
) -> bool {
    !row_subset_engaged
        // The sequential collector is also the bounded fast path for a pure
        // Linear/ReLU unary chain on CPU. In particular, Sequential -> Graph
        // conjunction upgrades (ACAS) otherwise enter the graph-native O(N^2)
        // target walker merely because no GEMM engine was requested.
        && (engine_present || cpu_dense_chain)
        && !width_threshold_present
        && !majority_skipped
        // The sequential remapper deliberately restores demand-skipped nodes
        // to their one-shot IBP boxes. Until that collector has an equivalent
        // composition pass, enabling the downstream resweep must route through
        // the graph-native loop where the feature is implemented.
        && !downstream_resweep
}

fn resolve_collection_output_node(graph: &GraphNetwork, exec_order: &[String]) -> Option<String> {
    if graph.output_name().is_empty() {
        exec_order.last().cloned()
    } else {
        Some(graph.output_name().to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CollectorBudgetRoute {
    FullWidth,
    PlannedSubset,
    RuntimeSubsetFailOpen,
}

fn collector_budget_route(subset_planned: bool, subset_succeeded: bool) -> CollectorBudgetRoute {
    match (subset_planned, subset_succeeded) {
        (true, true) => CollectorBudgetRoute::PlannedSubset,
        (true, false) => CollectorBudgetRoute::RuntimeSubsetFailOpen,
        (false, _) => CollectorBudgetRoute::FullWidth,
    }
}

/// Rows whose execution cost the collector budget must represent.
///
/// Only a still-planned subset may use its selected row count. A subset known
/// to be inadmissible before allocation and a subset that failed at runtime
/// both execute the full-width fallback and therefore use the raw target width.
fn collector_budget_rows(
    raw_rows: usize,
    selected_rows: usize,
    route: CollectorBudgetRoute,
) -> usize {
    match route {
        CollectorBudgetRoute::PlannedSubset => selected_rows,
        CollectorBudgetRoute::FullWidth | CollectorBudgetRoute::RuntimeSubsetFailOpen => raw_rows,
    }
}

/// Exact topology work proxy for a dense target-prefix CROWN walk.
///
/// Each dense objective row crosses every weighted ancestor. A Conv2d crossing
/// performs one contraction per kernel coefficient and output position; a
/// Linear crossing performs one contraction per weight. The factor four
/// represents lower/upper sign-split coefficient products. Its absolute value
/// is immaterial to the learned rate (the same factor is present in samples and
/// predictions), but retaining it makes the units describe the actual four
/// coefficient products rather than one forward MAC.
///
/// The estimator is deliberately narrower than the backward dispatcher. It
/// admits only the authenticated CIFAR/TinyImageNet ResNet subset and ignores
/// elementwise/pass-through costs, producing an optimistic estimate. Any
/// unsupported layer, missing shape, or checked-arithmetic overflow returns
/// `None`, which makes scheduling execute the walk unchanged.
fn dense_crown_prefix_work_units(
    graph: &GraphNetwork,
    target_node: &str,
    objective_rows: usize,
) -> Option<u128> {
    const COEFFICIENT_PRODUCTS_PER_MAC: u128 = 4;

    if objective_rows == 0 {
        return None;
    }
    let mut weighted_per_row = 0u128;
    let mut saw_weighted = false;
    for node_name in graph.ancestors(target_node).ok()? {
        let node = graph.nodes.get(&node_name)?;
        let work = match &node.layer {
            Layer::Conv2d(conv) => {
                let (input_h, input_w) = conv.input_shape?;
                let (output_h, output_w) = conv.output_size(input_h, input_w).ok()?;
                (conv.kernel.len() as u128)
                    .checked_mul(output_h as u128)?
                    .checked_mul(output_w as u128)?
            }
            Layer::Linear(linear) => {
                (linear.in_features() as u128).checked_mul(linear.out_features() as u128)?
            }
            // Exact ResNet inventory outside the weighted contractions. Their
            // omission makes the prediction faster/optimistic, never slower.
            Layer::BatchNorm(_)
            | Layer::ReLU(_)
            | Layer::Add(_)
            | Layer::Flatten(_)
            | Layer::Reshape(_) => continue,
            _ => return None,
        };
        saw_weighted = true;
        weighted_per_row = weighted_per_row.checked_add(work)?;
    }
    if !saw_weighted || weighted_per_row == 0 {
        return None;
    }
    weighted_per_row
        .checked_mul(objective_rows as u128)?
        .checked_mul(COEFFICIENT_PRODUCTS_PER_MAC)
}

/// Remaining target window for the learned prefix gate.
///
/// An internal Patches guard may produce a target deadline without an outer
/// collection deadline. That historical no-deadline route must remain outside
/// learned admission, so both deadlines are required. `now` is injected to
/// keep the boundary deterministic in tests.
fn prefix_cost_remaining_window(
    collection_deadline: Option<Instant>,
    target_deadline: Option<Instant>,
    now: Instant,
) -> Option<Duration> {
    collection_deadline?;
    target_deadline.map(|deadline| deadline.saturating_duration_since(now))
}

fn prefix_cost_instrumentation_active(
    explicitly_enabled: bool,
    collection_deadline: Option<Instant>,
    deadline_salvage_policy: PartialCrownDeadlineSalvagePolicy,
) -> bool {
    explicitly_enabled
        && collection_deadline.is_some()
        // Total-completion admission must not preempt an explicitly requested
        // partial-row salvage route. Until admission models time-to-first-useful
        // chunk, salvage owns this policy composition.
        && !deadline_salvage_policy.is_enabled()
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum M1CollectorTraceEvent {
    Plan {
        node: String,
        candidate: bool,
        weight: f64,
        raw_rows: usize,
        subset_rows: Option<usize>,
    },
    EffectiveRows {
        node: String,
        rows: usize,
        raw_rows: usize,
    },
    ObjectiveConeSkipBeforeBudget {
        node: String,
    },
    BudgetPhaseEntered {
        node: String,
    },
    RuntimeSubsetFailOpen {
        node: String,
        rows: usize,
        weight: f64,
        deadline_recomputed: bool,
        fallback_rejected: bool,
    },
    FixedWaveDispatch {
        node: String,
        drift_injected: bool,
    },
    HopelessLearned {
        node: String,
        rows: usize,
    },
    /// #cprime-admission: the walk-cost estimator refused this target before
    /// its walk started; the share rolls to later candidates.
    WalkRefused {
        node: String,
    },
    /// #walk-value-record: a completion or cooperative-abort full-walk record
    /// priced this target above its static share; the budgeter granted the
    /// recorded estimate bounded by the collection deadline.
    WalkMeasuredGrantApplied {
        node: String,
    },
    /// #cprime-admission: last demanded candidate admitted against the
    /// collection's full remaining time (accumulated rollover).
    WalkRolloverGranted {
        node: String,
    },
}

#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct M1CollectorTestControls {
    pub(crate) route_subset_budget_bytes: Option<usize>,
    pub(crate) fail_subset_node: Option<String>,
    pub(crate) reject_full_fallback_budget_node: Option<String>,
    pub(crate) drift_fixed_wave_node: Option<String>,
    /// #cprime-admission: force the estimator's est_secs for one node so the
    /// admission path is testable without a genuinely expensive graph.
    pub(crate) force_walk_estimate_secs: Option<(String, f64)>,
}

#[cfg(test)]
thread_local! {
    static M1_COLLECTOR_TEST_CONTROLS: std::cell::RefCell<M1CollectorTestControls> =
        std::cell::RefCell::new(M1CollectorTestControls::default());
    static M1_COLLECTOR_TEST_TRACE: std::cell::RefCell<Vec<M1CollectorTraceEvent>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
struct M1CollectorTestStateGuard {
    controls: M1CollectorTestControls,
    trace: Vec<M1CollectorTraceEvent>,
}

#[cfg(test)]
impl Drop for M1CollectorTestStateGuard {
    fn drop(&mut self) {
        M1_COLLECTOR_TEST_CONTROLS.with(|slot| {
            slot.replace(std::mem::take(&mut self.controls));
        });
        M1_COLLECTOR_TEST_TRACE.with(|slot| {
            slot.replace(std::mem::take(&mut self.trace));
        });
    }
}

#[cfg(test)]
pub(crate) fn run_with_m1_collector_test_controls<T>(
    controls: M1CollectorTestControls,
    f: impl FnOnce() -> T,
) -> (T, Vec<M1CollectorTraceEvent>) {
    let previous_controls = M1_COLLECTOR_TEST_CONTROLS.with(|slot| slot.replace(controls));
    let previous_trace = M1_COLLECTOR_TEST_TRACE.with(|slot| slot.replace(Vec::new()));
    let _guard = M1CollectorTestStateGuard {
        controls: previous_controls,
        trace: previous_trace,
    };
    let result = f();
    let trace = M1_COLLECTOR_TEST_TRACE.with(|slot| slot.borrow().clone());
    (result, trace)
}

#[cfg(test)]
fn record_m1_collector_trace(event: M1CollectorTraceEvent) {
    M1_COLLECTOR_TEST_TRACE.with(|slot| slot.borrow_mut().push(event));
}

fn collector_route_subset_budget_bytes(default: usize) -> usize {
    #[cfg(test)]
    {
        M1_COLLECTOR_TEST_CONTROLS
            .with(|slot| slot.borrow().route_subset_budget_bytes.unwrap_or(default))
    }
    #[cfg(not(test))]
    {
        default
    }
}

fn collector_force_subset_failure(node: &str) -> bool {
    #[cfg(test)]
    {
        M1_COLLECTOR_TEST_CONTROLS
            .with(|slot| slot.borrow().fail_subset_node.as_deref() == Some(node))
    }
    #[cfg(not(test))]
    {
        let _ = node;
        false
    }
}

fn collector_force_fixed_wave_drift(node: &str) -> bool {
    #[cfg(test)]
    {
        M1_COLLECTOR_TEST_CONTROLS
            .with(|slot| slot.borrow().drift_fixed_wave_node.as_deref() == Some(node))
    }
    #[cfg(not(test))]
    {
        let _ = node;
        false
    }
}

fn collector_force_full_fallback_budget_rejection(node: &str) -> bool {
    #[cfg(test)]
    {
        M1_COLLECTOR_TEST_CONTROLS
            .with(|slot| slot.borrow().reject_full_fallback_budget_node.as_deref() == Some(node))
    }
    #[cfg(not(test))]
    {
        let _ = node;
        false
    }
}

fn collector_forced_walk_estimate_secs(node: &str) -> Option<f64> {
    #[cfg(test)]
    {
        M1_COLLECTOR_TEST_CONTROLS.with(|slot| {
            slot.borrow()
                .force_walk_estimate_secs
                .as_ref()
                .filter(|(name, _)| name == node)
                .map(|(_, secs)| *secs)
        })
    }
    #[cfg(not(test))]
    {
        let _ = node;
        None
    }
}

use super::target_backward::{
    CrownCutContext, ObjectiveChunkRoutePlan, PartialCrownDeadlineSalvagePolicy,
};
// Used by the dense-prefix work estimator and this module's focused tests.
use crate::layers::Layer;
use crate::network::core::graph::CrownIbpRuntimePolicy;
use crate::network::core::{GraphNetwork, GraphTargetShapeContract};

/// Collection strategy snapshot supplied by the typed alpha configuration.
///
/// The ordinary strategy remains the default for every existing caller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum CrownIbpCollectionMode {
    #[default]
    Standard,
    /// Root-only cGAN strategy: run the ordinary all-demanded cascade from a
    /// certified forward-linear baseline. The distinct tag changes no bound
    /// algorithm; it supplies cache separation and duplicate-entry telemetry.
    CganComplete,
    /// Root-only cGAN strategy: one demanded ReLU-preactivation target receives
    /// the whole collection deadline and is published only when complete.
    CganSparseTargetComplete,
}

impl CrownIbpCollectionMode {
    const fn is_cgan_sparse_target_complete(self) -> bool {
        matches!(self, Self::CganSparseTargetComplete)
    }

    /// Typed cGAN transactions promise that their published map never widens
    /// the certified forward-linear baseline. The generic collector keeps its
    /// historical diagnostic-union behavior for disjoint sound enclosures.
    pub(crate) const fn requires_shrink_only_publication(self) -> bool {
        matches!(self, Self::CganComplete | Self::CganSparseTargetComplete)
    }

    /// Stable cache-identity tag (#cgan-restart-root-collection-cache).
    ///
    /// The collection mode decides WHICH targets the cascade walks and whether
    /// publication is shrink-only, so it is part of the CROWN-IBP collection
    /// cache fingerprint: entries are only ever served to a consumer running
    /// the SAME mode. This realises the "cache separation" the `CganComplete`
    /// variant doc promises — previously implemented as a total cache bypass,
    /// which made the typed cGAN root collection uncacheable and forced the
    /// disjunctive restart lane to re-collect a bit-identical root box.
    pub(crate) const fn cache_tag(self) -> u8 {
        match self {
            Self::Standard => 0,
            Self::CganComplete => 1,
            Self::CganSparseTargetComplete => 2,
        }
    }
}

/// Intersect one completed CROWN target with its forward baseline.
///
/// `BoundedTensor::intersection_per_element` returns a sound diagnostic union
/// for disjoint elements. That is the generic collector's historical contract,
/// but either typed cGAN transaction must reject the whole anomalous candidate
/// so its result remains elementwise no wider than the certified baseline.
fn intersect_completed_crown_target(
    baseline: &BoundedTensor,
    crown: &BoundedTensor,
    collection_mode: CrownIbpCollectionMode,
) -> Option<(BoundedTensor, usize)> {
    match baseline.intersection_per_element(crown) {
        Some((_diagnostic_union, disjoint))
            if collection_mode.requires_shrink_only_publication() && disjoint > 0 =>
        {
            None
        }
        other => other,
    }
}

/// Pick one deterministic cGAN proof target without hard-coding exported ONNX
/// node names.
///
/// The widest demanded ReLU-only producer is the dominant interval-poisoning
/// surface in the cGAN generator (BatchNormalization_11 in the measured
/// nCh=1 graph). Ties prefer the later node, which is closer to the root
/// objective. All-stable producers need no CROWN work and are excluded.
fn select_cgan_atomic_target(
    graph: &GraphNetwork,
    exec_order: &[String],
    demand_set: &HashSet<String>,
    output_name: Option<&str>,
    baseline_bounds: &HashMap<String, BoundedTensor>,
) -> Option<String> {
    exec_order
        .iter()
        .enumerate()
        .filter_map(|(index, node_name)| {
            let bounds = baseline_bounds.get(node_name)?;
            if !demand_set.contains(node_name)
                || !relu_only_intermediate_target_is_eligible(graph, node_name, output_name)
                || relu_stability_profile(bounds).retained() == 0
            {
                return None;
            }
            Some(((bounds.len(), index), node_name))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, node_name)| node_name.clone())
}

/// Default-dark downstream composition pass for CROWN-tightened intermediate
/// bounds. Exact `1` enables it; every other value preserves the historical
/// collector byte-for-byte.
///
/// The gate is intentionally dark until the deep-convolution benchmark cost is
/// measured. Soundness does not depend on the gate: an enabled step intersects
/// the original sound IBP box with the sound one-node IBP image of already
/// tightened predecessor boxes.
pub(super) const CROWN_IBP_DOWNSTREAM_RESWEEP_ENV: &str = "NY_CROWN_IBP_DOWNSTREAM_RESWEEP";

pub(super) fn crown_ibp_downstream_resweep_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

pub(super) fn crown_ibp_downstream_resweep_enabled() -> bool {
    crown_ibp_downstream_resweep_from_raw(
        std::env::var(CROWN_IBP_DOWNSTREAM_RESWEEP_ENV)
            .ok()
            .as_deref(),
    )
}

/// Shrink-only collector intersection for a deadline-truncated row assembly.
///
/// Both inputs should be valid enclosures. A disjoint element is therefore a
/// soundness/numerical anomaly: `BoundedTensor` can return a sound diagnostic
/// union, but that union may be looser than the certified IBP seed. Partial-row
/// salvage is never allowed to loosen the seed, so any malformed/disjoint
/// candidate rejects the whole target and retains IBP exactly.
fn retain_partial_crown_rows(
    ibp_bound: &BoundedTensor,
    partial_bound: Option<&BoundedTensor>,
) -> (BoundedTensor, usize) {
    match partial_bound.and_then(|bound| ibp_bound.intersection_per_element(bound)) {
        Some((tightened, 0)) => (tightened, 0),
        Some((_diagnostic_union, disjoint)) => (ibp_bound.clone(), disjoint),
        None => (ibp_bound.clone(), 0),
    }
}

impl GraphNetwork {
    /// Typed collection-mode variant used by the root alpha reference router.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::network::graph_alpha) fn collect_crown_ibp_bounds_core_inner_with_mode(
        &self,
        input: &BoundedTensor,
        ibp_bounds: HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
        deadline_is_hard: bool,
        engine: Option<&dyn ny_core::GemmEngine>,
        min_width_to_tighten: Option<f32>,
        cut_segment: usize,
        published_margin_indices: Option<std::sync::Arc<[usize]>>,
        downstream_resweep: bool,
        deadline_salvage_policy: PartialCrownDeadlineSalvagePolicy,
        collection_mode: CrownIbpCollectionMode,
        cgan_narrow_target_max_rows: Option<usize>,
        prefix_cost_admission_enabled: bool,
        runtime_policy: CrownIbpRuntimePolicy,
    ) -> Result<GraphCrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_core_inner_with_cut_segment_and_prefix_cost_policy(
            input,
            ibp_bounds,
            deadline,
            deadline_is_hard,
            engine,
            min_width_to_tighten,
            cut_segment,
            published_margin_indices,
            downstream_resweep,
            deadline_salvage_policy,
            collection_mode,
            cgan_narrow_target_max_rows,
            prefix_cost_admission_enabled,
            runtime_policy,
        )
    }

    /// Explicit-cut-segment variant of the core loop (#crown-cut-segment).
    /// `cut_segment = 0` disables cuts (byte-identical full-prefix backward).
    /// Production snapshots the segment before cache lookup; the soundness
    /// oracle injects it directly so parallel tests never mutate global state.
    #[cfg(test)]
    pub(crate) fn collect_crown_ibp_bounds_core_inner_with_cut_segment(
        &self,
        input: &BoundedTensor,
        ibp_bounds: HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
        engine: Option<&dyn ny_core::GemmEngine>,
        min_width_to_tighten: Option<f32>,
        cut_segment: usize,
    ) -> Result<GraphCrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_core_inner_with_cut_segment_and_policies(
            input,
            ibp_bounds,
            deadline,
            engine,
            min_width_to_tighten,
            cut_segment,
            false,
            PartialCrownDeadlineSalvagePolicy::Disabled,
            CrownIbpCollectionMode::Standard,
        )
    }

    /// Explicit cut/salvage variant used by policy and composition regressions.
    /// Production uses the fully snapshotted helper below so the admission
    /// route and its cache identity cannot observe different environment state.
    #[cfg(test)]
    pub(in crate::network::graph_alpha) fn collect_crown_ibp_bounds_core_inner_with_cut_segment_and_policies(
        &self,
        input: &BoundedTensor,
        ibp_bounds: HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
        engine: Option<&dyn ny_core::GemmEngine>,
        min_width_to_tighten: Option<f32>,
        cut_segment: usize,
        downstream_resweep: bool,
        deadline_salvage_policy: PartialCrownDeadlineSalvagePolicy,
        collection_mode: CrownIbpCollectionMode,
    ) -> Result<GraphCrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_core_inner_with_cut_segment_and_prefix_cost_policy(
            input,
            ibp_bounds,
            deadline,
            deadline.is_some(),
            engine,
            min_width_to_tighten,
            cut_segment,
            crate::output_margin_seed::published_margin_indices(),
            downstream_resweep,
            deadline_salvage_policy,
            collection_mode,
            None,
            budget_policy::prefix_cost_admission_enabled(),
            self.crown_ibp_runtime_policy(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_crown_ibp_bounds_core_inner_with_cut_segment_and_prefix_cost_policy(
        &self,
        input: &BoundedTensor,
        ibp_bounds: HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
        deadline_is_hard: bool,
        engine: Option<&dyn ny_core::GemmEngine>,
        min_width_to_tighten: Option<f32>,
        cut_segment: usize,
        published_margin_indices: Option<std::sync::Arc<[usize]>>,
        downstream_resweep: bool,
        deadline_salvage_policy: PartialCrownDeadlineSalvagePolicy,
        collection_mode: CrownIbpCollectionMode,
        cgan_narrow_target_max_rows: Option<usize>,
        prefix_cost_admission_enabled: bool,
        runtime_policy: CrownIbpRuntimePolicy,
    ) -> Result<GraphCrownIbpBoundsResult> {
        let honest_provenance_enabled = runtime_policy.honest_provenance_enabled;
        let exec_order = self.exec_order()?;
        let margin_subset_indices = |output_dim| {
            crate::output_margin_seed::margin_subset_indices_from_published(
                published_margin_indices.as_ref(),
                output_dim,
            )
        };

        // Linear-chain graphs can reuse the sequential `Network` CROWN-IBP
        // collector, which already contains the #3599 GPU partial-backward
        // fast path and deadline support. The sequential collector checks
        // deadlines between each layer, matching the graph-native loop.
        //
        // Gates:
        // 1. Engine presence: without GPU, the graph-native loop provides
        //    better per-node fallback for unsupported layers.
        // 2. Width-threshold mode remains graph-specific.
        // 3. Skip-fraction: the graph path skips nodes whose identity
        //    matrix exceeds the CPU dense budget. When the majority of
        //    nodes would be skipped (>50%), the graph path's selectivity
        //    saves more work than the sequential GPU path's single-pass
        //    efficiency. When most nodes are within budget, sequential GPU
        //    wins because one backward pass is cheaper than O(N²) per-node
        //    backward passes on CPU.
        //
        //    Measured heuristic (#3599):
        //    - soundnessbench: 37% skip → sequential GPU 30x faster
        //    - metaroom: 69% skip → graph path 4.8x faster
        let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let total_count = ibp_bounds.len().max(1);
        // Count every dense-overflow target for the fast-path gate. The
        // sequential collector cannot selectively skip large spatial targets,
        // so metaroom must stay on the graph-native collector when those nodes
        // dominate. The graph-native loop below uses a separate, patches-aware
        // budget helper because it can still tighten those targets via #3813.
        let exceed_count = ibp_bounds
            .iter()
            .filter(|(_node_name, bound)| {
                Self::counts_toward_sequential_skip_fraction(bound, budget)
            })
            .count();
        let skip_fraction = exceed_count as f64 / total_count as f64;
        let majority_skipped = skip_fraction > 0.5;
        debug!(
            "CROWN-IBP collection fast-path: {exceed_count}/{total_count} nodes exceed budget \
             (skip_fraction={skip_fraction:.2}), majority_skipped={majority_skipped}, \
             engine={}",
            engine.is_some(),
        );
        // Print-only telemetry must run before the sequential fast path can
        // return, otherwise precisely the collection route whose sparse-row
        // implementation already exists would be invisible to the audit.
        let sparse_interm_diag = sparse_interm_diag_enabled();
        let environment_sparse_relu_rows = sparse_relu_rows_enabled();
        let atomic_target_requested = collection_mode.is_cgan_sparse_target_complete();
        let resident_patches_root = super::resident_patches_root::enabled();
        let resident_patches_window = super::resident_patches_root::begin_planning_if_enabled(
            resident_patches_root,
            deadline,
        );
        if let Some(window) = resident_patches_window {
            super::resident_patches_root::observe_in_window(
                self,
                input,
                &ibp_bounds,
                exec_order,
                engine,
                window,
            );
        }
        let sparse_output_node = if environment_sparse_relu_rows || atomic_target_requested {
            resolve_collection_output_node(self, exec_order)
        } else {
            None
        };
        let early_demand_set =
            if environment_sparse_relu_rows || atomic_target_requested || sparse_interm_diag {
                Some(nodes_requiring_crown_tightening(
                    self,
                    exec_order,
                    &ibp_bounds,
                ))
            } else {
                None
            };
        let atomic_target = atomic_target_requested
            .then(|| {
                select_cgan_atomic_target(
                    self,
                    exec_order,
                    early_demand_set
                        .as_ref()
                        .expect("atomic target selection computes demand first"),
                    sparse_output_node.as_deref(),
                    &ibp_bounds,
                )
            })
            .flatten();
        let atomic_target_engaged = atomic_target.is_some();
        let downstream_resweep = downstream_resweep || atomic_target_engaged;
        // Partial-row deadline salvage is intentionally incompatible with the
        // target-complete authority boundary. The caller's environment policy
        // remains untouched for every ordinary collection.
        let deadline_salvage_policy = if atomic_target_engaged {
            PartialCrownDeadlineSalvagePolicy::Disabled
        } else {
            deadline_salvage_policy
        };
        if deadline.is_some() && deadline_salvage_policy.is_enabled() {
            info!(
                "[NY_CROWN_DEADLINE_CHUNK_SALVAGE_V1] stage=armed \
                 source=exact-environment deadline_bearing=true"
            );
        }

        let sparse_relu_rows = environment_sparse_relu_rows || atomic_target_engaged;
        let sparse_relu_row_plans = if atomic_target_engaged {
            atomic_target
                .iter()
                .filter_map(|node_name| {
                    let bounds = ibp_bounds.get(node_name)?;
                    let plan = sparse_relu_row_plan_for_target(
                        self,
                        node_name,
                        sparse_output_node.as_deref(),
                        bounds,
                    )?;
                    let patches_seed =
                        self.crown_ibp_target_can_start_in_patches(node_name, bounds);
                    if subset_seed_fits_dense_budget(
                        patches_seed,
                        plan.selected_len(),
                        bounds.len(),
                        budget,
                    ) {
                        Some((node_name.clone(), plan))
                    } else {
                        info!(
                            "#cgan-sparse-target-complete: selected node '{}' sparse selector \
                             {}/{} exceeds the {} byte dense budget; using dense/chunked target",
                            node_name,
                            plan.selected_len(),
                            bounds.len(),
                            budget,
                        );
                        None
                    }
                })
                .collect::<HashMap<_, _>>()
        } else if environment_sparse_relu_rows {
            early_demand_set
                .as_ref()
                .expect("sparse row gate computes demand before the sequential route")
                .iter()
                .filter_map(|node_name| {
                    let bounds = ibp_bounds.get(node_name)?;
                    let plan = sparse_relu_row_plan_for_target(
                        self,
                        node_name,
                        sparse_output_node.as_deref(),
                        bounds,
                    )?;
                    let patches_seed =
                        self.crown_ibp_target_can_start_in_patches(node_name, bounds);
                    if subset_seed_fits_dense_budget(
                        patches_seed,
                        plan.selected_len(),
                        bounds.len(),
                        budget,
                    ) {
                        Some((node_name.clone(), plan))
                    } else {
                        info!(
                            "#sparse-relu-rows: node '{}' selected {}/{} rows, but its dense \
                             selector pair exceeds the {} byte budget; retaining the existing \
                             full-width/chunked policy",
                            node_name,
                            plan.selected_len(),
                            bounds.len(),
                            budget,
                        );
                        None
                    }
                })
                .collect::<HashMap<_, _>>()
        } else {
            HashMap::new()
        };
        let sparse_plan_engaged = atomic_target_engaged || !sparse_relu_row_plans.is_empty();
        if let Some(target) = atomic_target.as_deref() {
            let bounds = ibp_bounds
                .get(target)
                .expect("selected target must have baseline bounds");
            let selected = sparse_relu_row_plans
                .get(target)
                .map(|plan| plan.selected_len());
            info!(
                "#cgan-sparse-target-complete: armed target='{}' dim={} route={} \
                 budget=uninterrupted publication=atomic baseline=forward-linear-compatible",
                target,
                bounds.len(),
                if selected.is_some() {
                    "sparse"
                } else {
                    "dense-threshold-or-memory-fallback"
                },
            );
        } else if atomic_target_requested {
            info!(
                "#cgan-sparse-target-complete: declined (no non-stable demanded ReLU-only target); \
                 returning the certified baseline without widening the authority surface"
            );
            let provenance = ibp_bounds
                .keys()
                .map(|node_name| {
                    (
                        node_name.clone(),
                        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DemandDrivenSkip),
                    )
                })
                .collect();
            return Ok(GraphCrownIbpBoundsResult {
                bounds: ibp_bounds,
                provenance,
                fallback_events: Vec::new(),
            });
        }
        if sparse_interm_diag {
            let diagnostic_demand = early_demand_set
                .as_ref()
                .expect("diagnostic gate computes demand before the sequential route");
            for node_name in exec_order
                .iter()
                .filter(|node_name| diagnostic_demand.contains(*node_name))
            {
                let Some(bounds) = ibp_bounds.get(node_name) else {
                    continue;
                };
                let profile = relu_stability_profile(bounds);
                let (relu_consumers, other_consumers) = required_consumer_counts(self, node_name);
                let total = profile.total();
                let stable_fraction = if total == 0 {
                    0.0
                } else {
                    profile.stable() as f64 / total as f64
                };
                eprintln!(
                    "[NY_SPARSE_INTERM_DIAG_V1] node='{}' dim={} active_stable={} \
                     inactive_stable={} unstable={} unresolved={} retained={} \
                     stable_fraction={stable_fraction:.6} relu_consumers={} \
                     other_bound_consumers={} relu_only_eligible={}",
                    node_name,
                    total,
                    profile.active_stable,
                    profile.inactive_stable,
                    profile.unstable,
                    profile.unresolved,
                    profile.retained(),
                    relu_consumers,
                    other_consumers,
                    relu_consumers > 0 && other_consumers == 0,
                );
            }
            if sparse_relu_rows {
                for node_name in exec_order {
                    let Some(plan) = sparse_relu_row_plans.get(node_name) else {
                        continue;
                    };
                    eprintln!(
                        "[NY_SPARSE_RELU_ROWS_V1] node='{}' selected={} total={} all_stable={}",
                        node_name,
                        plan.selected_len(),
                        plan.total_rows(),
                        plan.is_all_stable(),
                    );
                }
                eprintln!(
                    "[NY_SPARSE_RELU_ROWS_V1] eligible_targets={} selected_rows={} total_rows={}",
                    sparse_relu_row_plans.len(),
                    sparse_relu_row_plans
                        .values()
                        .map(|plan| plan.selected_len())
                        .sum::<usize>(),
                    sparse_relu_row_plans
                        .values()
                        .map(|plan| plan.total_rows())
                        .sum::<usize>(),
                );
            }
        }
        // The sequential collector does not implement objective-dependent row
        // seeding. Letting a structurally sequential graph take that shortcut
        // while a margin subset is published silently computes every output
        // row and defeats the subset's bounded-memory contract. Keep every
        // engaged row-subset strategy on the graph-native collector that owns
        // its scatter semantics.
        let margin_subset_engaged = resolve_collection_output_node(self, exec_order)
            .and_then(|output_node| ibp_bounds.get(&output_node))
            .is_some_and(|bound| margin_subset_indices(bound.len()).is_some());
        let row_subset_engaged = sparse_plan_engaged || margin_subset_engaged;
        let cpu_dense_chain = engine.is_none() && self.try_to_sequential_network().is_some();
        // The sequential collector does not implement intermediate cut
        // concretization.  An explicitly armed cut policy must therefore stay
        // on the graph-native route; otherwise every segment value silently
        // produces the gate-off result on CPU Linear/ReLU chains.
        // #cgan-narrow-target-budget: the sequential collector has no notion of
        // DECLINING a demanded target, so an armed row ceiling must stay on the
        // graph-native route where the decline is implemented — the same
        // contract that keeps `min_width_to_tighten` off this fast path
        // (both change WHICH nodes are tightened). `None` (the shipped
        // default) leaves the fast-path decision byte-identical.
        if cut_segment == 0
            && cgan_narrow_target_max_rows.is_none()
            && !runtime_policy.requires_graph_native_chunk_execution()
            && sequential_fast_path_allowed(
                row_subset_engaged,
                engine.is_some(),
                cpu_dense_chain,
                min_width_to_tighten.is_some(),
                majority_skipped,
                downstream_resweep,
            )
        {
            if cpu_dense_chain {
                info!(
                    "CROWN-IBP collection fast-path: CPU Linear/ReLU chain using sequential collector"
                );
            }
            if let Some(result) = self.try_collect_crown_ibp_bounds_via_sequential_network(
                exec_order,
                input,
                &ibp_bounds,
                engine,
                deadline,
                runtime_policy,
            )? {
                if sparse_interm_diag {
                    eprintln!("[NY_SPARSE_INTERM_DIAG_V1] collector_route=sequential");
                }
                return Ok(result);
            }
        }
        if sparse_interm_diag {
            eprintln!("[NY_SPARSE_INTERM_DIAG_V1] collector_route=graph_native");
        }

        // #crown-cut-segment (NY_CROWN_CUT_SEGMENT, default OFF = full-prefix
        // backward): designate every N-th node of the execution order as a
        // CUT. A per-target backward that reaches a cut node whose bounds this
        // sweep already finalized concretizes the accumulated linear relation
        // against that node's box (same directed-rounding concretization as
        // the input box; see CROWN_CUT_SEGMENT_ENV in target_backward.rs)
        // instead of expanding the node's prefix, dropping the sweep from
        // O(n²) to ~O(n·N) backward steps. Bounds can only get LOOSER (still
        // sound): the map under construction only ever holds valid enclosures,
        // so every cut box is a valid enclosure by construction. Topological
        // order finalizes every ancestor before its dependents; a cut node the
        // map does not (yet) cover is simply expanded as usual (fail-open).
        let cut_ctx: Option<CrownCutContext> = (cut_segment > 0).then(|| {
            CrownCutContext::new(
                exec_order
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| index % cut_segment == 0)
                    .map(|(_, name)| name.clone())
                    .collect(),
            )
        });

        // #margin-subset-seed: resolve the OUTPUT node once. When the
        // relu-split initial-bounds scope published spec-referenced margin
        // indices (crate::output_margin_seed), the OUTPUT-node tightening
        // below seeds only those k identity rows and scatters them over the
        // node's sound IBP bounds instead of running the full
        // `[output_dim x output_dim]` identity backward.
        let margin_subset_output_node =
            sparse_output_node.or_else(|| resolve_collection_output_node(self, exec_order));

        // #spec-influence-cone: build the backward influence boxes ONCE per
        // collection, from the published objective indices at the OUTPUT node.
        // `None` (no publication, no spatial output, or no objective box)
        // disengages the intermediate subset entirely and keeps every target on
        // its historical full-width seed.
        let influence_cones = margin_subset_output_node.as_deref().and_then(|out_node| {
            // The cone needs a SPATIAL origin. A network whose last op is a
            // Flatten/Reshape has a 1-D output node (TinyYOLO ends
            // `Conv_29 [125,13,13] -> Flatten_30 [21125]`), which would disarm
            // the cone entirely — measured: no `armed` line on yolo_2023, so the
            // cone was inert and the whole gain came from publishing the spec
            // indices instead.
            //
            // Walk back to the nearest 3-D ancestor with the SAME element count.
            // A pure reshape preserves the row-major flat ordering, so a flat
            // index at the output is the same flat index on that tensor and the
            // published indices carry over unchanged. Equal element count is the
            // guard: it rejects any transform that is not a pure reshape.
            let (out_node, out_shape) = {
                let direct = ibp_bounds.get(out_node).map(|b| b.shape().to_vec());
                match direct {
                    Some(s) if s.len() == 3 => (out_node.to_string(), s),
                    Some(s) => {
                        let numel: usize = s.iter().product();
                        let mut found = None;
                        for name in exec_order.iter().rev() {
                            if let Some(b) = ibp_bounds.get(name) {
                                let bs = b.shape();
                                if bs.len() == 3 && bs.iter().product::<usize>() == numel {
                                    found = Some((name.clone(), bs.to_vec()));
                                    break;
                                }
                            }
                        }
                        found?
                    }
                    None => return None,
                }
            };
            let out_node = out_node.as_str();
            let out_shape = (out_shape[0], out_shape[1], out_shape[2]);
            let indices = margin_subset_indices(out_shape.0 * out_shape.1 * out_shape.2)?;
            let objective = crate::spec_influence_cone::objective_box(&indices, out_shape)?;
            let shape_of = |name: &str| -> Option<(usize, usize, usize)> {
                let s = ibp_bounds.get(name)?.shape();
                (s.len() == 3).then(|| (s[0], s[1], s[2]))
            };
            info!(
                "#spec-influence-cone: armed from OUTPUT '{out_node}' {out_shape:?} with \
                 {} spec indices",
                indices.len()
            );
            Some(crate::spec_influence_cone::compute(
                self, exec_order, out_node, objective, &shape_of,
            ))
        });
        if influence_cones.is_none() {
            debug!(
                "#spec-influence-cone: DISARMED (no published spec indices, or the OUTPUT node \
                 is not a 3-D spatial tensor); every target keeps its full-width seed"
            );
        }

        let mut crown_ibp_bounds: HashMap<String, BoundedTensor> = HashMap::new();
        let mut provenance: HashMap<String, BoundsProvenance> = HashMap::new();
        let mut fallback_events = Vec::new();
        let demand_set = early_demand_set
            .unwrap_or_else(|| nodes_requiring_crown_tightening(self, exec_order, &ibp_bounds));
        debug!(
            "CROWN-IBP demand set: {}/{} nodes selected",
            demand_set.len(),
            ibp_bounds.len()
        );

        // #all-stable-skip: never spend a CROWN walk on a node with ZERO unstable
        // rows. Unconditional, and independent of the sparse-row subset gate below.
        //
        // CORRECTION 2026-08-03: 5f604b84 described this as "VALUE-IDENTICAL. No
        // computed bound changes." That is WRONG and was measured wrong on 8 of 20
        // A/B rows. The SKIPPED node's own box is unchanged — that part of the proof
        // below holds. But removing 14,336 of 32,868 row-weights from the demand set
        // makes `compute_weighted_per_node_budget_secs` (budget_policy.rs:429) hand
        // every SURVIVING target a ~1.8x larger slice, so targets that previously hit
        // PerNodeDeadlineExceeded now finish their walk and store `CROWN ∩ IBP`
        // instead of untouched IBP. On CIFAR100_resnet_medium that is 7/11 reverted
        // -> 0/7, i.e. three extra targets tightened per row.
        //
        // The change is therefore ALLOCATION-AFFECTING and TIGHTENING, not identical.
        // It remains SOUND in the safe direction: `CROWN ∩ IBP ⊆ IBP`, and a tighter
        // sound enclosure can only ever cost an UNSAT, never manufacture one.
        // Verified empirically: 20 banked cifar100/tinyimagenet rows through the
        // scored entry point at official budgets, arms interleaved — 20/20 identical
        // verdicts, 0 lost, 0 gained, 0 sat/unsat contradictions.
        //
        // Value-identity of the skipped node's box additionally assumes
        // NY_CROWN_CUT_SEGMENT unset/0 and NY_CROWN_REPROPAGATE unset — both read a
        // collected box by MAGNITUDE (target_backward.rs:511-701,
        // crown_tighten.rs:2208-2325). No preset, the vnncomp entry point, or
        // ny_measurement_provenance.py sets either. IF EITHER IS EVER PROMOTED TO
        // DEFAULT, RE-PRICE THIS.
        //
        // `sparse_relu_row_plan_for_target` admits a target only when it is not the
        // collection output and EVERY bound-demanding consumer is a ReLU
        // (`relu_consumers > 0 && other_consumers == 0`, demand.rs:172-175, counted
        // over consumers that actually list the target in
        // `required_input_bound_indices()`). For such a target the consumers read the
        // bound only to choose the `mask_pos` / `mask_neg` branch of the ReLU
        // relaxation — that is a decision on the SIGN, not the magnitude. When the
        // node additionally has zero unstable rows, IBP already resolves every sign,
        // and IBP is sound, so a CROWN walk cannot move any downstream coefficient or
        // bias. The bound it would produce is discarded into
        // `intersection_per_element` against a box that already decides everything
        // the consumer asks. alpha-beta-CROWN guards this identically
        // (`bound_general.py:1035-1036`: `elif unstable_size == 0: skip = True`).
        //
        // MEASURED on cifar100 idx_7704 at a 400s root cap: the dense collector spent
        // **79.6s of a 209.8s collection on Conv_23, which has 0/8192 unstable rows,
        // and reverted it to IBP anyway.** Conv_23/37/43/49 are all zero-unstable and
        // account for 14,336 of 32,868 row-weights. Removing them costs nothing and
        // frees ~38% of the collection wall for targets that can actually move the
        // root bound.
        //
        // This is deliberately ONLY the all-stable case. Seeding a strict SUBSET of
        // rows inside a partially-unstable node is a different change: the stable
        // remainder keeps IBP instead of receiving CROWN, so the stored box genuinely
        // differs, and `intersection_per_element` can WIDEN on disjoint elements via
        // its union fallback (numeric.rs:341-350). That stays behind
        // `NY_CROWN_IBP_SPARSE_RELU_ROWS` until a corpus sweep prices it.
        let all_stable_skip: HashSet<String> = {
            let output_node = resolve_collection_output_node(self, exec_order);
            demand_set
                .iter()
                .filter(|node_name| {
                    ibp_bounds.get(*node_name).is_some_and(|bounds| {
                        sparse_relu_row_plan_for_target(
                            self,
                            node_name,
                            output_node.as_deref(),
                            bounds,
                        )
                        .is_some_and(|plan| plan.is_all_stable())
                    })
                })
                .cloned()
                .collect()
        };
        if !all_stable_skip.is_empty() {
            debug!(
                "CROWN-IBP all-stable skip: {} of {} demanded targets have zero unstable \
                 rows and feed only ReLU sign decisions; their CROWN walks cannot change \
                 any bound and are skipped (#all-stable-skip)",
                all_stable_skip.len(),
                demand_set.len()
            );
        }
        if sparse_relu_rows {
            let selected: usize = sparse_relu_row_plans
                .values()
                .map(|plan| plan.selected_len())
                .sum();
            let total: usize = sparse_relu_row_plans
                .values()
                .map(|plan| plan.total_rows())
                .sum();
            info!(
                "#sparse-relu-rows: graph-native collector armed for {} eligible targets, \
                 selecting {selected}/{total} demanded rows",
                sparse_relu_row_plans.len(),
            );
        }
        let mut patches_budget =
            budget_policy::PatchesTighteningBudget::with_collection_deadline(deadline);
        // Preset-configurable floor/cap for the equal-share per-node budget
        // (#cgan-bn11-budget). Default (all-None) reproduces the #3499/#4413
        // constants exactly; only presets that set the knobs (cgan_2023's
        // 150 s cap for the 28,800-dim BN_11 chunked backward) change policy.
        let per_node_time_budget = self.crown_ibp_per_node_time_budget;
        let (per_node_floor_secs, _) =
            budget_policy::effective_per_node_time_budget(&per_node_time_budget);
        // Per-node time-budget candidates. This mask only feeds
        // `count_remaining_budget_candidates` (the equal-share split of the
        // remaining deadline). Nodes whose dense identity exceeds the memory
        // budget now COUNT as candidates (#cgan-bn11-chunk): they are no longer
        // skipped to IBP but rerouted through the objective-chunked backward,
        // so they consume a per-node time share like any other CROWN target.
        let mut global_budget_candidate_mask: Vec<bool> = exec_order
            .iter()
            .map(|node_name| {
                let Some(ibp_bound) = ibp_bounds.get(node_name) else {
                    return false;
                };
                if atomic_target_engaged
                    && atomic_target
                        .as_deref()
                        .is_some_and(|target| target != node_name)
                {
                    return false;
                }
                let width_eligible = min_width_to_tighten
                    .map(|threshold| ibp_bound.max_width() >= threshold)
                    .unwrap_or(true);
                let has_sparse_work = sparse_relu_row_plans
                    .get(node_name)
                    .is_none_or(|plan| !plan.is_all_stable());
                demand_set.contains(node_name) && width_eligible && has_sparse_work
            })
            .collect();

        // #cgan-narrow-target-budget: decline demanded targets wider than the
        // configured row ceiling.
        //
        // The per-node time slice below is proportional to objective-row count,
        // so on the official cGAN rows the generator targets (up to 28,800 and
        // 61,504 rows) draw ~86% of the collection budget. Measured offline
        // against an ORT-sampled envelope, those targets are already within
        // 1.02-1.13x of exact under the forward-linear seed the cascade starts
        // from, while the narrow discriminator targets they starve are where
        // the root over-approximation is actually manufactured.
        //
        // Declining a target is a SCHEDULING decision only: it keeps the
        // caller-supplied baseline enclosure for that node, which the collector
        // already holds and already publishes under the shrink-only contract.
        // No bound is widened and no bound is tightened by this branch, so
        // proof authority is unchanged whichever way the ceiling is set.
        if let Some(max_rows) = cgan_narrow_target_max_rows {
            let mut declined = 0usize;
            let mut retained = 0usize;
            for (index, node_name) in exec_order.iter().enumerate() {
                if !global_budget_candidate_mask[index] {
                    continue;
                }
                let rows = ibp_bounds.get(node_name).map_or(0, BoundedTensor::len);
                if rows > max_rows {
                    global_budget_candidate_mask[index] = false;
                    declined += 1;
                } else {
                    retained += 1;
                }
            }
            info!(
                "[cgan-narrow-target-budget] armed max_rows={max_rows} \
                 mode={collection_mode:?} declined={declined} retained={retained} \
                 targets; declined targets keep the caller-supplied baseline enclosure"
            );
        }

        let chunk_aware_budget = budget_policy::crown_chunk_aware_budget_enabled();

        // Resolve the subset seed exactly as the execution path below does.
        // M1 needs this BEFORE assigning a chunk-aware weight: a target that
        // takes a margin/cone/sparse subset seed never executes the full
        // objective-chunk route and therefore must retain its historical raw
        // demanded-row weight.
        let resolve_subset_indices =
            |node_name: &str,
             ibp_bound: &BoundedTensor,
             sparse_relu_plan: Option<&SparseReluRowPlan>,
             is_patches_target: bool| {
                let existing_subset_indices = margin_subset_output_node
                    .as_deref()
                    .filter(|output_node| *output_node == node_name)
                    .and_then(|_| margin_subset_indices(ibp_bound.len()))
                    .or_else(|| {
                        let cones = influence_cones.as_ref()?;
                        let shape = ibp_bound.shape();
                        if shape.len() != 3 {
                            return None;
                        }
                        let picked =
                            cones.subset_indices(node_name, (shape[0], shape[1], shape[2]))?;
                        info!(
                            "#spec-influence-cone: '{}' seeds {}/{} rows ({:.1}x fewer)",
                            node_name,
                            picked.len(),
                            ibp_bound.len(),
                            ibp_bound.len() as f64 / picked.len().max(1) as f64,
                        );
                        Some(std::sync::Arc::from(picked))
                    });
                let is_atomic_selected_target = atomic_target_engaged
                    && atomic_target
                        .as_deref()
                        .is_some_and(|target| target == node_name);
                // The typed target owns an exact authority boundary: sparse
                // means every selected unresolved ReLU row, and a dense
                // fallback must not be trimmed by a margin/cone subset.
                let subset_indices = compose_target_subset_indices(
                    existing_subset_indices,
                    sparse_relu_plan.map(|plan| plan.selected_rows()),
                    is_atomic_selected_target,
                );
                let subset_is_sparse = sparse_relu_plan.is_some();
                subset_indices.and_then(|indices| {
                    if subset_is_sparse
                        && !subset_seed_fits_dense_budget(
                            is_patches_target,
                            indices.len(),
                            ibp_bound.len(),
                            collector_route_subset_budget_bytes(budget),
                        )
                    {
                        info!(
                            "#sparse-relu-rows: node '{}' selected {}/{} rows, but its dense \
                             selector pair exceeds the {} byte budget; using the existing \
                             full-width chunked path",
                            node_name,
                            indices.len(),
                            ibp_bound.len(),
                            budget,
                        );
                        return None;
                    }
                    // #subset-forfeits-patches: a STRICT subset of a
                    // patches-eligible target is materialized as a dense
                    // `[k x dim]` seed and densifies the whole walk. Keep the
                    // virtual patches identity instead unless that dense seed
                    // is genuinely cheap.
                    //
                    // EXEMPT the atomic selected target. Its route is a typed
                    // strategy decision, not an opportunistic cone: the comment
                    // above already states that "the typed target owns an exact
                    // authority boundary", and `compose_target_subset_indices`
                    // deliberately hands it the sparse rows untouched by any
                    // margin/cone subset. Re-routing it here would silently
                    // withdraw the strategy the caller selected
                    // (`#cgan-sparse-target-complete`). yolo_2023 never engages
                    // an atomic target, so this exemption does not affect the
                    // measurement this guard was built from.
                    if subset_patches_guard_enabled()
                        && !is_atomic_selected_target
                        && subset_forfeits_patches_route(
                            is_patches_target,
                            indices.len(),
                            ibp_bound.len(),
                        )
                    {
                        info!(
                            "#subset-forfeits-patches: node '{}' selected {}/{} rows, but the \
                             strict-subset seed materializes a dense {} byte pair and forfeits \
                             the virtual patches identity this target is eligible for; taking \
                             the full-width PATCHES backward instead (set {}=0 to restore)",
                            node_name,
                            indices.len(),
                            ibp_bound.len(),
                            crate::network::crown_memory::dense_pair_bytes(
                                indices.len(),
                                ibp_bound.len()
                            )
                            .map_or_else(|| "overflow".to_string(), |b| b.to_string()),
                            SUBSET_PATCHES_GUARD_ENV,
                        );
                        return None;
                    }
                    Some(indices)
                })
            };

        #[derive(Clone)]
        struct ChunkAwareCollectorRoute {
            is_patches_target: bool,
            subset_indices: Option<std::sync::Arc<[usize]>>,
            chunk_plan: Option<ObjectiveChunkRoutePlan>,
            scheduling_plan: Option<budget_policy::ObjectiveChunkSchedulingPlan>,
        }

        // Disabled mode deliberately does not enter this closure: route and
        // subset discovery remain lazy at their historical positions below.
        // Armed mode resolves each candidate once and retains the exact plan
        // used by both scheduling and execution.
        let chunk_aware_candidate_routes: Option<Vec<Option<ChunkAwareCollectorRoute>>> =
            chunk_aware_budget.then(|| {
                exec_order
                    .iter()
                    .zip(global_budget_candidate_mask.iter())
                    .map(|(node_name, &is_candidate)| {
                        if !is_candidate {
                            return None;
                        }
                        let ibp_bound = ibp_bounds.get(node_name)?;
                        let sparse_relu_plan = sparse_relu_row_plans.get(node_name);
                        let is_patches_target =
                            self.crown_ibp_target_can_start_in_patches(node_name, ibp_bound);
                        let subset_indices = resolve_subset_indices(
                            node_name,
                            ibp_bound,
                            sparse_relu_plan,
                            is_patches_target,
                        );
                        let chunk_plan = budget_policy::auto_objective_chunk_route_plan(
                            self,
                            node_name,
                            ibp_bound,
                            input.len(),
                            budget,
                            deadline.is_some(),
                            // Preserve the historical auto-chunk route for the
                            // fail-open full-width fallback even when a subset
                            // seed is planned. `eligible_for_inflation` below
                            // separately excludes that subset from M1 weight.
                            true,
                        );
                        let scheduling_plan = chunk_plan.and_then(|execution| {
                            budget_policy::objective_chunk_scheduling_plan(
                                ibp_bound.len(),
                                execution,
                                deadline.is_some(),
                                cut_ctx.is_some(),
                                runtime_policy.chunk_wave_parallel_enabled,
                                runtime_policy.chunk_adaptive_growth_enabled,
                            )
                        });
                        Some(ChunkAwareCollectorRoute {
                            is_patches_target,
                            subset_indices,
                            chunk_plan,
                            scheduling_plan,
                        })
                    })
                    .collect()
            });
        // An armed sparse∩cone route with zero retained rows executes no
        // backward at all. Remove it before constructing the denominator so
        // every positive candidate weight names work the collector will
        // actually attempt. The loop consumes the same retained route and
        // records ObjectiveConeRowsSkipped before consulting either budget.
        if let Some(routes) = chunk_aware_candidate_routes.as_ref() {
            for (index, route) in routes.iter().enumerate() {
                let empty_sparse_cone = route.as_ref().is_some_and(|route| {
                    sparse_relu_row_plans.contains_key(&exec_order[index])
                        && route
                            .subset_indices
                            .as_deref()
                            .is_some_and(<[usize]>::is_empty)
                });
                if empty_sparse_cone {
                    global_budget_candidate_mask[index] = false;
                }
            }
        }
        // COST-WEIGHTED budget (#cgan-collection-cost-weight): each candidate's
        // per-node time slice is proportional to its objective-row count
        // (`ibp_bound.len()`) rather than an equal split, so a wide generator
        // target (BatchNorm_11 = 28,800 dims) gets enough time to COMPLETE on the
        // first pass instead of degrading to IBP and forcing redundant
        // re-collections. Non-candidates carry weight 0.0.
        //
        // #target-cost-weight: the row count is a COST proxy that is measurably
        // wrong. On cgan row 7 it hands the 28,800-row BatchNorm_11 54% of a
        // 405 s window while two discriminator convs share 2.9%, and measured
        // secs/row across those same targets spans 27x — because a shallow wide
        // target and a deep narrow one walk completely different prefixes.
        // `demanded_rows_by_index` is retained so the corrected model below can
        // price each candidate's ACTUAL walk.
        let mut demanded_rows_by_index: Vec<usize> = vec![0; exec_order.len()];
        let mut global_budget_candidate_weights: Vec<f64> = exec_order
            .iter()
            .zip(global_budget_candidate_mask.iter())
            .enumerate()
            .map(|(index, (node_name, &is_candidate))| {
                if !is_candidate {
                    return 0.0;
                }
                ibp_bounds
                    .get(node_name)
                    .map(|b| {
                        let historical_selected_rows = sparse_relu_row_plans
                            .get(node_name)
                            .map_or_else(|| b.len(), |plan| plan.selected_len());
                        let route = chunk_aware_candidate_routes
                            .as_ref()
                            .and_then(|routes| routes.get(index))
                            .and_then(Option::as_ref);
                        let demanded_rows = route.map_or(historical_selected_rows, |route| {
                            route
                                .subset_indices
                                .as_deref()
                                .map_or_else(|| b.len(), <[usize]>::len)
                        });
                        demanded_rows_by_index[index] = demanded_rows;
                        budget_policy::demanded_target_work_weight(
                            demanded_rows,
                            route.and_then(|route| route.scheduling_plan),
                            route.is_some_and(|route| route.subset_indices.is_none()),
                            chunk_aware_budget,
                        )
                    })
                    .unwrap_or(0.0)
            })
            .collect();
        // #target-cost-weight: correct the COST MODEL from `rows` to
        // `rows x prefix`, using the same MAC estimator the walk-admission gate
        // already trusts (`collector_walk_macs_with_shapes`, #cprime-admission)
        // — the one place in this file that knows what a backward walk from
        // THIS target actually costs. All-or-nothing: one admitted candidate the
        // estimator cannot price leaves every weight at today's value.
        //
        // SCHEDULING ONLY. Nothing here reaches a bound's value: it moves the
        // per-node DEADLINE and nothing else. A target that still completes
        // publishes the identical CROWN∩IBP box; a target that no longer
        // completes takes the unchanged overrun path and keeps its certified
        // IBP fallback.
        //
        // MUTUALLY EXCLUSIVE with M1's chunk-aware inflation, deliberately.
        // `objective_fixed_wave_work_weight` multiplies rows by `wave_count`
        // for exactly the reason this correction exists — the prefix is
        // REPLAYED and the row count cannot see it. Composing the two would
        // charge the same replay twice, and that composition has never been
        // measured. M1 is env-only (`NY_CROWN_CHUNK_AWARE_BUDGET=1`, which
        // `run_instance.sh` never exports), so this exclusion cannot fire in
        // competition; it only keeps a hand A/B from silently stacking two
        // cost models.
        let prefix_cost_weighted =
            budget_policy::prefix_cost_weighting_enabled(&per_node_time_budget)
                && !chunk_aware_budget;
        if budget_policy::prefix_cost_weighting_enabled(&per_node_time_budget)
            && chunk_aware_budget
        {
            eprintln!(
                "[target-cost-weight] declined: M1 chunk-aware weighting is armed and \
                 already inflates for prefix replay — refusing to charge it twice"
            );
        }
        let mut prefix_cost_multipliers: Option<Vec<f64>> = None;
        if prefix_cost_weighted {
            let row_weights_before = global_budget_candidate_weights.clone();
            let costs: Vec<Option<f64>> = exec_order
                .iter()
                .enumerate()
                .map(|(index, node_name)| {
                    let rows = demanded_rows_by_index[index];
                    self.collector_walk_macs_with_shapes(node_name, rows, Some(&ibp_bounds))
                        .and_then(|macs| budget_policy::prefix_cost_per_row(macs, rows))
                })
                .collect();
            let admitted: Vec<bool> = global_budget_candidate_weights
                .iter()
                .map(|weight| *weight > 0.0)
                .collect();
            match budget_policy::prefix_cost_multipliers(&costs, &admitted) {
                Some(multipliers) => {
                    let mut moved = 0usize;
                    let mut widest = 1.0f64;
                    for (weight, multiplier) in global_budget_candidate_weights
                        .iter_mut()
                        .zip(multipliers.iter())
                    {
                        if *weight > 0.0 && (*multiplier - 1.0).abs() > f64::EPSILON {
                            moved += 1;
                            widest = widest.max(*multiplier);
                        }
                        *weight *= *multiplier;
                    }
                    // THE LEVER'S OWN OBSERVABLE. `eprintln!` and not `info!`:
                    // the vnncomp tracing filter SWALLOWS info!, and a lever
                    // whose only evidence is swallowed reads as dark on exactly
                    // the runs that matter. Same convention as
                    // `[root-joint-interm-alpha]` / `[bounds-profile]`.
                    let admitted_count = admitted.iter().filter(|a| **a).count();
                    eprintln!(
                        "[target-cost-weight] armed: repriced {moved}/{admitted_count} \
                         admitted targets by rows x prefix (widest multiplier \
                         {widest:.2}x)"
                    );
                    // The REALLOCATION itself, per target, so an A/B can be read
                    // off the log instead of inferred: share_before is the row
                    // model's share of the window, share_after is this model's.
                    let before: f64 = row_weights_before.iter().sum();
                    let after: f64 = global_budget_candidate_weights.iter().sum();
                    if before > 0.0 && after > 0.0 {
                        for (index, node_name) in exec_order.iter().enumerate() {
                            let was = row_weights_before[index];
                            if was <= 0.0 {
                                continue;
                            }
                            eprintln!(
                                "[target-cost-weight] node={node_name} rows={} \
                                 share_before={:.4} share_after={:.4} mult={:.2}x",
                                demanded_rows_by_index[index],
                                was / before,
                                global_budget_candidate_weights[index] / after,
                                multipliers[index],
                            );
                        }
                    }
                    prefix_cost_multipliers = Some(multipliers);
                }
                None => {
                    eprintln!(
                        "[target-cost-weight] declined: at least one admitted target has \
                         no MAC estimate (unmodelled op or zero demanded rows) — keeping \
                         the historical row weights unchanged"
                    );
                }
            }
        }
        // #target-cost-weight: an inflated weight must never masquerade as node
        // WIDTH — `dim_scaled_cap_secs` is quadratic in that argument.
        let separate_cap_dims = chunk_aware_budget || prefix_cost_multipliers.is_some();
        #[cfg(test)]
        for (index, node_name) in exec_order.iter().enumerate() {
            let raw_rows = ibp_bounds.get(node_name).map_or(0, BoundedTensor::len);
            let subset_rows = chunk_aware_candidate_routes
                .as_ref()
                .and_then(|routes| routes.get(index))
                .and_then(Option::as_ref)
                .and_then(|route| route.subset_indices.as_deref().map(<[usize]>::len));
            record_m1_collector_trace(M1CollectorTraceEvent::Plan {
                node: node_name.clone(),
                candidate: global_budget_candidate_mask
                    .get(index)
                    .copied()
                    .unwrap_or(false),
                weight: global_budget_candidate_weights
                    .get(index)
                    .copied()
                    .unwrap_or(0.0),
                raw_rows,
                subset_rows,
            });
        }
        // #cgan-stacked-backward (NY_CGAN_STACKED_BACKWARD=1, default OFF =>
        // this whole block is skipped and the loop below is byte-identical).
        // ONE dense backward walk carries every admissible demanded trunk
        // target's identity rows, so the shared ConvTranspose generator
        // prefix — measured at 98.4-99.7% of EVERY demanded cgan pass
        // (docs/CGAN_COLLECTION_CACHE_DEFECTS_2026-08-03.md) — is walked once
        // instead of once per target. Results are served to the loop below
        // through the SAME Complete handling (shape restore + IBP
        // intersection + provenance) a solo walk uses.
        //
        // SOUND: each stacked row's arithmetic is identical to its solo pass
        // against the same (empty-so-far CROWN, IBP) maps — see
        // target_backward::StackedSeedInjectionPlan and the bit-identity test
        // — and the served bound then flows through the established
        // shrink-only intersection. The one QUALITY tradeoff vs the
        // sequential loop: solo walks late in the loop see earlier targets'
        // tightened boxes for their ReLU relaxations, while the stacked pass
        // sees only the pre-loop map. Sound either way (both maps are valid
        // enclosures); measured on cgan the per-walk cost is flat to +/-6%
        // across repeats ("nothing tightens"), so the cascade is worth ~0
        // there. Any planner/executor refusal leaves `stacked_results` empty
        // and every target on the historical per-target path.
        //
        // Excluded by construction: atomic-target collections (single-target
        // authority boundary), the CganSparseTargetComplete mode, sparse-row
        // planned targets, and the margin-subset OUTPUT node (their solo
        // routes use k-row seeds the stack does not model).
        let mut stacked_results: HashMap<String, BoundedTensor> = HashMap::new();
        if super::cgan_stacked::stacked_backward_enabled()
            && !atomic_target_engaged
            && !matches!(
                collection_mode,
                CrownIbpCollectionMode::CganSparseTargetComplete
            )
        {
            let stacked_candidates: Vec<super::cgan_stacked::StackedCandidate> = exec_order
                .iter()
                .enumerate()
                .filter(|(index, node_name)| {
                    global_budget_candidate_mask
                        .get(*index)
                        .copied()
                        .unwrap_or(false)
                        && !all_stable_skip.contains(*node_name)
                        && !sparse_relu_row_plans.contains_key(*node_name)
                        && margin_subset_output_node.as_deref() != Some(node_name.as_str())
                })
                .filter_map(|(index, node_name)| {
                    Some(super::cgan_stacked::StackedCandidate {
                        exec_index: index,
                        node_name: node_name.clone(),
                        rows: ibp_bounds.get(node_name)?.len(),
                    })
                })
                .collect();
            if let Some(plan) = super::cgan_stacked::plan_stacked_backward(
                self,
                exec_order,
                &stacked_candidates,
                &ibp_bounds,
            ) {
                // The stacked pass replaces its members' solo walks, so it
                // gets the SUM of their weighted shares — a single share
                // would starve a pass doing k targets' work
                // (#cgan-stacked-backward, budget_policy sum-of-shares).
                let mut below_floor = false;
                let stacked_deadline = deadline.and_then(|d| {
                    let now = Instant::now();
                    if now >= d {
                        below_floor = true;
                        return None;
                    }
                    let remaining_secs = d.duration_since(now).as_secs_f64();
                    let remaining_weight_sum = budget_policy::sum_remaining_budget_weights(
                        &global_budget_candidate_weights,
                        0,
                    );
                    let member_shares: Vec<(f64, f64)> = plan
                        .member_exec_indices
                        .iter()
                        .zip(plan.stack.iter())
                        .map(|(&exec_index, (_, rows))| {
                            let weight = global_budget_candidate_weights
                                .get(exec_index)
                                .copied()
                                .unwrap_or(0.0);
                            let cap_dims = budget_policy::weighted_budget_cap_dims(
                                weight,
                                *rows as f64,
                                separate_cap_dims,
                            );
                            (weight, cap_dims)
                        })
                        .collect();
                    let secs = budget_policy::compute_stacked_backward_budget_secs(
                        remaining_secs,
                        remaining_weight_sum,
                        &member_shares,
                        &per_node_time_budget,
                    );
                    match secs {
                        Some(secs) => Some(now + Duration::from_secs_f64(secs)),
                        None => {
                            below_floor = true;
                            None
                        }
                    }
                });
                if below_floor {
                    super::cgan_stacked::stacked_event(format_args!(
                        "stage=declined reason=sum-of-shares-below-floor"
                    ));
                } else {
                    let stacked_start = Instant::now();
                    match self.propagate_crown_to_node_stacked(
                        input,
                        &plan.stack,
                        &crown_ibp_bounds,
                        &ibp_bounds,
                        engine,
                        "CROWN-IBP stacked backward",
                        stacked_deadline,
                        deadline_is_hard,
                    ) {
                        Ok(results) => {
                            // Members' walks are done: zero their scheduling
                            // weights so later candidates' shares divide the
                            // remaining window correctly.
                            for &exec_index in &plan.member_exec_indices {
                                if let Some(weight) =
                                    global_budget_candidate_weights.get_mut(exec_index)
                                {
                                    *weight = 0.0;
                                }
                            }
                            super::cgan_stacked::stacked_event(format_args!(
                                "stage=executed members={} total_rows={} wall_secs={:.3}",
                                results.len(),
                                plan.total_rows,
                                stacked_start.elapsed().as_secs_f64(),
                            ));
                            stacked_results = results.into_iter().collect();
                        }
                        Err(error) => {
                            super::cgan_stacked::stacked_event(format_args!(
                                "stage=fallback error='{error}' wall_secs={:.3}",
                                stacked_start.elapsed().as_secs_f64(),
                            ));
                        }
                    }
                }
            }
        }
        let mut deadline_exceeded = false;
        // Smallest target dimension observed to burn its whole per-node share
        // and fail on time (#cifar100-collector-order). Later candidates at or
        // above it are routed straight to IBP so the remaining collection
        // budget can reach the cheap, objective-adjacent tail targets.
        let mut hopeless_min_dim: Option<usize> = None;
        // #cprime-admission: estimate-then-refuse walk admission. Engaged only
        // when a collection deadline exists — an unbounded collection admits
        // everything, byte-identical to the historical loop. The rate basis is
        // the per-process forward-linear probe (same GMAC unit as the walk
        // estimate; cached OnceLock, so the probe is paid at most once per
        // process, and lazily — only when a deadline-carrying non-patches
        // candidate actually reaches admission), corrected by the census
        // prior until the FIRST completed walk in THIS collection replaces it
        // (self-calibrating).
        let mut walk_cost_model: Option<budget_policy::WalkCostModel> = None;
        // #prefix-cost-admission: an explicit experimental same-collection
        // timing model. It learns only from complete dense target walks. Keep
        // the whole instrumentation path physically absent unless both the
        // opt-in and an outer collection deadline were snapshotted: disabled
        // and no-deadline collections retain the historical ancestor walks,
        // clock reads, and route behavior exactly.
        let prefix_cost_active = prefix_cost_instrumentation_active(
            prefix_cost_admission_enabled,
            deadline,
            deadline_salvage_policy,
        );
        let mut prefix_cost_model =
            budget_policy::PrefixCostAdmissionModel::new(prefix_cost_active);
        let total_nodes = exec_order.len();
        let collection_start = Instant::now();
        let mut crown_node_count = 0usize;
        let mut crown_total_secs = 0.0f64;
        let mut skip_count = 0usize;
        let mut demand_skip_count = 0usize;
        let mut stable_relu_skip_count = 0usize;
        let mut objective_cone_skip_count = 0usize;
        let mut downstream_resweep_merged = 0usize;
        let mut downstream_resweep_narrowed = 0usize;
        // Only descendants of a bound that actually narrowed can benefit.
        // This avoids duplicating the foundational forward pass before the
        // first useful CROWN result and avoids unrelated graph branches.
        let mut downstream_resweep_sources = downstream_resweep.then(HashSet::<String>::new);

        for (layer_index, node_name) in exec_order.iter().enumerate() {
            // Deadline check (#3109): if deadline exceeded, skip CROWN backward
            // for remaining nodes and use IBP bounds instead. This is sound
            // (IBP bounds are valid, just looser).
            if !deadline_exceeded {
                if let Some(d) = deadline {
                    if Instant::now() >= d {
                        info!(
                            "CROWN-IBP DAG: deadline exceeded at node {}/{}, remaining nodes use IBP",
                            layer_index, total_nodes
                        );
                        deadline_exceeded = true;
                    }
                }
            }

            let original_ibp_bound = match ibp_bounds.get(node_name) {
                Some(b) => b,
                None => continue,
            };
            // #crown-ibp-downstream-resweep: compose every already-finalized
            // predecessor tightening through this node before either the
            // demand-skip arm or this node's own backward CROWN runs.
            //
            // The old collector read `ibp_bounds` here, a one-shot map created
            // before the loop, so a CROWN-tight Add followed by a skipped ReLU
            // immediately widened back to the pre-tightening Add box. Reading
            // `crown_ibp_bounds` through the exact same one-node IBP dispatch as
            // the foundational forward pass lets tightening compound in
            // topological order.
            //
            // SOUND and fail-open: both the original box and the one-node image
            // are valid enclosures; their per-element intersection is valid.
            // Unsupported nodes, shape/NaN failures, and deadline expiry retain
            // the original box. The feature is default-dark pending measured
            // deep-convolution cost.
            let node = self.nodes.get(node_name);
            let has_narrowed_predecessor =
                downstream_resweep_sources.as_ref().is_some_and(|sources| {
                    node.is_some_and(|node| node.inputs().iter().any(|name| sources.contains(name)))
                });
            let composed_ibp_bound =
                if downstream_resweep && !deadline_exceeded && has_narrowed_predecessor {
                    let candidate = node
                        .and_then(|node| {
                            self.node_ibp_step(
                                node_name,
                                node,
                                input,
                                &crown_ibp_bounds,
                                false,
                                engine,
                                deadline,
                            )
                            .ok()
                        })
                        .and_then(|step| original_ibp_bound.intersection_per_element(&step))
                        // A disjoint element is impossible when both enclosures
                        // are valid. The tensor helper returns a sound union for
                        // diagnostics, but this composition pass is strictly
                        // shrink-only, so retain the original whole-node box.
                        .and_then(|(tightened, disjoint)| (disjoint == 0).then_some(tightened));
                    match candidate {
                        Some(tightened) => {
                            downstream_resweep_merged += 1;
                            if tightened.lower() != original_ibp_bound.lower()
                                || tightened.upper() != original_ibp_bound.upper()
                            {
                                downstream_resweep_narrowed += 1;
                                if let Some(sources) = downstream_resweep_sources.as_mut() {
                                    sources.insert(node_name.clone());
                                }
                            }
                            Some(tightened)
                        }
                        None => None,
                    }
                } else {
                    None
                };
            // Preserve the default-dark path exactly: when the gate is off (or
            // the fail-open recompute declines a node), borrow the original
            // map entry without allocating or copying its tensors.
            let ibp_bound = composed_ibp_bound.as_ref().unwrap_or(original_ibp_bound);
            let layer_type = node
                .map(|node| node.layer.layer_type().to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            // The typed cGAN lane has exactly one CROWN authority candidate.
            // Every other node still receives the baseline/downstream-reswept
            // enclosure assembled above, but it cannot consume target budget or
            // publish a CROWN claim.
            if atomic_target_engaged
                && atomic_target
                    .as_deref()
                    .is_some_and(|target| target != node_name)
            {
                demand_skip_count += 1;
                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                provenance.insert(
                    node_name.clone(),
                    BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DemandDrivenSkip),
                );
                continue;
            }

            // Demand-driven skip (#3775): no downstream nonlinear consumer needs this node.
            if !demand_set.contains(node_name) {
                demand_skip_count += 1;
                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                provenance.insert(
                    node_name.clone(),
                    BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DemandDrivenSkip),
                );
                continue;
            }

            let sparse_relu_plan = sparse_relu_row_plans.get(node_name);
            let is_atomic_selected_target = atomic_target_engaged
                && atomic_target
                    .as_deref()
                    .is_some_and(|target| target == node_name);
            // #all-stable-skip fires whether or not the subset gate is armed: a
            // zero-unstable node's CROWN walk is value-identical to its IBP bound.
            if sparse_relu_plan.is_some_and(|plan| plan.is_all_stable())
                || all_stable_skip.contains(node_name)
            {
                stable_relu_skip_count += 1;
                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                provenance.insert(
                    node_name.clone(),
                    BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::StableReluRowsSkipped,
                    ),
                );
                info!(
                    "#sparse-relu-rows: node '{}' is all stable (0/{} rows selected); \
                     retaining IBP without backward CROWN",
                    node_name,
                    ibp_bound.len(),
                );
                continue;
            }
            let chunk_aware_route = chunk_aware_candidate_routes
                .as_ref()
                .and_then(|routes| routes.get(layer_index))
                .and_then(Option::as_ref);
            if chunk_aware_route.is_some_and(|route| {
                sparse_relu_plan.is_some()
                    && route
                        .subset_indices
                        .as_deref()
                        .is_some_and(<[usize]>::is_empty)
            }) {
                #[cfg(test)]
                record_m1_collector_trace(M1CollectorTraceEvent::ObjectiveConeSkipBeforeBudget {
                    node: node_name.clone(),
                });
                objective_cone_skip_count += 1;
                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                provenance.insert(
                    node_name.clone(),
                    BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::ObjectiveConeRowsSkipped,
                    ),
                );
                info!(
                    "#sparse-relu-rows: node '{}' has no selected rows after objective-cone \
                     intersection; retaining IBP without backward CROWN",
                    node_name,
                );
                continue;
            }
            let historical_selected_rows =
                sparse_relu_plan.map_or_else(|| ibp_bound.len(), |plan| plan.selected_len());
            let mut effective_target_rows =
                chunk_aware_route.map_or(historical_selected_rows, |route| {
                    route
                        .subset_indices
                        .as_deref()
                        .map_or_else(|| ibp_bound.len(), <[usize]>::len)
                });
            #[cfg(test)]
            record_m1_collector_trace(M1CollectorTraceEvent::EffectiveRows {
                node: node_name.clone(),
                rows: effective_target_rows,
                raw_rows: ibp_bound.len(),
            });

            // When deadline exceeded, skip CROWN backward and use IBP directly (#3109).
            // A stacked result is already computed — serving it is ~free and
            // sound, so it bypasses every pre-walk skip below.
            if deadline_exceeded && !stacked_results.contains_key(node_name) {
                skip_count += 1;
                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                provenance.insert(
                    node_name.clone(),
                    BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded),
                );
                fallback_events.push(CrownIbpFallbackEvent {
                    layer_index,
                    layer_type,
                    reason: CrownIbpFallbackReason::DeadlineExceeded,
                    details: format!("node '{}' skipped CROWN (deadline exceeded)", node_name),
                });
                continue;
            }

            // Memory budget check: the graph-native loop may keep spatial unary
            // targets on the patches-start path from #3813, so only dense-only
            // targets trip this gate. This intentionally differs from the
            // sequential fast-path gate above, which must remain conservative
            // enough to keep metaroom off the wrong collector (#3839).
            //
            // Over-budget targets are no longer skipped to IBP: they reroute
            // through the bound-equivalent objective-chunked backward
            // (#cgan-bn11-chunk, `propagate_crown_to_node_chunked`) with an
            // auto chunk size that scales the identity pair down to the budget.
            // Memory is bounded by the chunk; a slow chunked node retains only
            // fully completed rows over certified IBP and stays explicitly
            // deadline-truncated. The under-budget path is unchanged
            // (`chunk_override = None` keeps the env-driven single-pass behavior
            // byte-for-byte).
            let (chunk_override, is_patches_target) = if let Some(route) = chunk_aware_route {
                if let Some(plan) = route.chunk_plan {
                    let node_dim = ibp_bound.len();
                    let required = crate::network::crown_memory::identity_pair_bytes(node_dim)
                        .unwrap_or(usize::MAX);
                    debug!(
                        "CROWN-IBP DAG: node '{}' dim={} identity requires {} bytes \
                             (budget {}), budget exceeded -> chunked backward \
                             requested C={}, effective initial C={}",
                        node_name,
                        node_dim,
                        required,
                        budget,
                        plan.requested_rows,
                        plan.effective_initial_rows,
                    );
                }
                (
                    route.chunk_plan.map(|plan| plan.requested_rows),
                    route.is_patches_target,
                )
            } else {
                // Historical lazy route resolution. Keeping this branch
                // intact is part of M1's default-dark compatibility.
                let chunk_override =
                    if self.graph_native_target_exceeds_budget(node_name, ibp_bound, budget) {
                        let node_dim = ibp_bound.len();
                        let required = crate::network::crown_memory::identity_pair_bytes(node_dim)
                            .unwrap_or(usize::MAX);
                        let chunk_rows =
                            budget_policy::auto_objective_chunk_rows(node_dim, input.len(), budget);
                        debug!(
                            "CROWN-IBP DAG: node '{}' dim={} identity requires {} bytes \
                             (budget {}), budget exceeded -> chunked backward C={}",
                            node_name, node_dim, required, budget, chunk_rows
                        );
                        Some(chunk_rows)
                    } else {
                        None
                    };
                (
                    chunk_override,
                    self.crown_ibp_target_can_start_in_patches(node_name, ibp_bound),
                )
            };
            #[cfg(test)]
            record_m1_collector_trace(M1CollectorTraceEvent::BudgetPhaseEntered {
                node: node_name.clone(),
            });

            if !is_atomic_selected_target
                && is_patches_target
                && !stacked_results.contains_key(node_name)
                && !patches_budget.can_start_node(budget_policy::MIN_PER_NODE_BUDGET_SECS)
            {
                let node_dim = ibp_bound.len();
                let patches_budget_used_secs = patches_budget.used_secs();
                debug!(
                    "CROWN-IBP DAG: node '{}' dim={} patches-eligible but aggregate \
                     patches budget exhausted/below {:.1}s floor ({patches_budget_used_secs:.3}s used), using IBP",
                    node_name,
                    node_dim,
                    budget_policy::MIN_PER_NODE_BUDGET_SECS,
                );
                skip_count += 1;
                budget_policy::record_patches_budget_fallback(
                    &mut crown_ibp_bounds,
                    &mut provenance,
                    &mut fallback_events,
                    node_name,
                    ibp_bound,
                    layer_index,
                    &layer_type,
                    node_dim,
                    patches_budget_used_secs,
                );
                continue;
            }

            // Width-based skip: when the IBP interval is already tight, CROWN
            // backward cannot meaningfully tighten further.  Skipping saves the
            // ~5-7s per-node cost for the budget to reach deeper, wider nodes
            // where tightening matters most (#3499).
            if !is_atomic_selected_target && !stacked_results.contains_key(node_name) {
                if let Some(threshold) = min_width_to_tighten {
                    let ibp_max_width = ibp_bound.max_width();
                    if ibp_max_width < threshold {
                        skip_count += 1;
                        debug!(
                        "CROWN-IBP DAG: node '{}' max_width={:.6} < threshold={:.6}, skipping CROWN",
                        node_name, ibp_max_width, threshold
                    );
                        crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                        provenance.insert(
                            node_name.clone(),
                            BoundsProvenance::ForwardFallback(
                                CrownIbpFallbackReason::WidthBelowThreshold,
                            ),
                        );
                        fallback_events.push(CrownIbpFallbackEvent {
                            layer_index,
                            layer_type,
                            reason: CrownIbpFallbackReason::WidthBelowThreshold,
                            details: format!(
                                "node '{}' max_width={:.6} < threshold={:.6}",
                                node_name, ibp_max_width, threshold
                            ),
                        });
                        continue;
                    }
                }
            }

            // Share the remaining deadline across the remaining globally
            // eligible tightening targets, then clamp with the #4413 cap.
            let patches_per_node = patches_budget
                .remaining_deadline(is_patches_target, budget_policy::MIN_PER_NODE_BUDGET_SECS);
            // #skip-log-names-the-deciding-term: the weighted share that the floor
            // test actually rejected, recorded so the skip log can print IT rather
            // than the equal share, which is a different number.
            let mut rejected_weighted_secs: Option<f64> = None;
            let mut per_node_deadline = if is_atomic_selected_target {
                // No equal-share/adaptive cap: this is the one target the typed
                // root strategy selected, so it owns the complete remaining
                // collection window.
                deadline
            } else {
                let global_per_node = deadline.and_then(|d| {
                    let now = Instant::now();
                    if now >= d {
                        return None;
                    }
                    let remaining = d.duration_since(now);
                    let remaining_secs = remaining.as_secs_f64();
                    // #cgan-collection-cost-weight: cost-proportional slice by
                    // objective-row count so the wide generator target completes on
                    // the first pass. Reduces to equal-share when all weights match.
                    let remaining_weight_sum = budget_policy::sum_remaining_budget_weights(
                        &global_budget_candidate_weights,
                        layer_index,
                    );
                    let this_weight = global_budget_candidate_weights
                        .get(layer_index)
                        .copied()
                        .unwrap_or(0.0);
                    let cap_dims = budget_policy::weighted_budget_cap_dims(
                        this_weight,
                        ibp_bound.len() as f64,
                        separate_cap_dims,
                    );
                    rejected_weighted_secs = budget_policy::weighted_per_node_budget_secs_unfloored(
                        remaining_secs,
                        remaining_weight_sum,
                        this_weight,
                        cap_dims,
                        &per_node_time_budget,
                    );
                    let per_node_secs = budget_policy::compute_weighted_per_node_budget_secs(
                        remaining_secs,
                        remaining_weight_sum,
                        this_weight,
                        cap_dims,
                        &per_node_time_budget,
                    )?;
                    Some(now + Duration::from_secs_f64(per_node_secs))
                });
                budget_policy::merge_per_node_deadlines(
                    global_per_node,
                    patches_per_node,
                    deadline.is_some(),
                )
            };

            // Skip this node when its per-node share falls below the minimum
            // floor even though the global deadline has not expired yet.
            if deadline.is_some()
                && per_node_deadline.is_none()
                && !stacked_results.contains_key(node_name)
            {
                let remaining_global_candidates = budget_policy::count_remaining_budget_candidates(
                    &global_budget_candidate_mask,
                    layer_index,
                );
                let remaining_secs = deadline
                    .map(|d| {
                        let now = Instant::now();
                        if now >= d {
                            0.0
                        } else {
                            d.duration_since(now).as_secs_f64()
                        }
                    })
                    .unwrap_or(0.0);
                debug!(
                    "CROWN-IBP DAG: node '{}' per-node budget {:.1}s < {:.1}s floor ({} tightening targets remain, {:.1}s left), using IBP",
                    node_name,
                    rejected_weighted_secs.unwrap_or(0.0),
                    per_node_floor_secs,
                    remaining_global_candidates,
                    remaining_secs,
                );
                skip_count += 1;
                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                // #per-node-floor-label: this target NEVER RAN. Recording it as
                // `PerNodeDeadlineExceeded` (what this site did for weeks) makes
                // a below-floor REFUSAL indistinguishable from a genuine mid-walk
                // overrun in every log line and every provenance histogram.
                provenance.insert(
                    node_name.clone(),
                    BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::PerNodeBudgetBelowFloor,
                    ),
                );
                fallback_events.push(CrownIbpFallbackEvent {
                    layer_index,
                    layer_type,
                    reason: CrownIbpFallbackReason::PerNodeBudgetBelowFloor,
                    details: format!(
                        "node '{}' REFUSED before any backward step: per-node share {:.3}s is \
                         below the {:.1}s floor ({} tightening targets remain, {:.1}s of \
                         collection window left) — the walk did not run and did not overrun",
                        node_name,
                        rejected_weighted_secs.unwrap_or(0.0),
                        per_node_floor_secs,
                        remaining_global_candidates,
                        remaining_secs,
                    ),
                });
                continue;
            }

            // Adaptive hopeless-class skip (#cifar100-collector-order): a target
            // at least as large as one that already burned its entire share and
            // failed on time cannot finish either. Route it to IBP (SOUND — IBP
            // is a valid enclosure) so the budget reaches cheaper later targets.
            if let Some(min_dim) =
                hopeless_min_dim.filter(|_| !stacked_results.contains_key(node_name))
            {
                if effective_target_rows >= min_dim {
                    info!(
                        "CROWN-IBP DAG: node '{}' (effective rows {}, dim {}) skipped as \
                         hopeless-class (>= {} rows which already exhausted its share), using IBP",
                        node_name,
                        effective_target_rows,
                        ibp_bound.len(),
                        min_dim
                    );
                    skip_count += 1;
                    crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                    provenance.insert(
                        node_name.clone(),
                        BoundsProvenance::ForwardFallback(
                            CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                        ),
                    );
                    fallback_events.push(CrownIbpFallbackEvent {
                        layer_index,
                        layer_type,
                        reason: CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                        details: format!(
                            "node '{}' effective rows {} (dim {}) >= hopeless-class rows {}",
                            node_name,
                            effective_target_rows,
                            ibp_bound.len(),
                            min_dim
                        ),
                    });
                    continue;
                }
            }

            // #cprime-admission: estimate-then-refuse BEFORE the walk starts.
            //
            // The equal/weighted-share policy starts walks it cannot finish:
            // measured on tinyimagenet (2026-08-02), Conv_17 burned 150.29 s
            // and row 7390 burned 185.06 s of share, both delivering the same
            // IBP bound a refusal delivers in ~0 s. Estimate the walk's MACs
            // from the graph (prefix depth x per-step GEMM dims), price it at
            // the measured rate, and refuse upfront when the padded estimate
            // exceeds the share. Refusal does not consume time, so later
            // candidates' shares divide the UNSPENT remainder — rollover is
            // structural, and the floor100 failure mode (granting time inside
            // a fixed window) cannot recur because refusal only ever FREES
            // budget. The LAST demanded candidate may claim the accumulated
            // rollover: with nobody after it, the monopolization cap is
            // meaningless and today's policy would burn the capped share for
            // nothing.
            //
            // SOUND either way (refusal keeps the valid IBP bound), and
            // fail-open by construction: patches-routed targets are exempt
            // (their composition is a different, cheaper cost class priced by
            // the aggregate patches budget), unmodeled ops under-estimate
            // toward admission, and an admitted under-prediction still hits
            // the existing per-node deadline backstop unchanged.
            let mut walk_estimated_macs: Option<u128> = None;
            if deadline.is_some()
                && per_node_deadline.is_some()
                && !is_patches_target
                // An explicitly armed partial-row policy prices usefulness at
                // the first completed chunk, not full-walk completion. Until
                // c-prime models that objective, it must not preempt salvage.
                && !deadline_salvage_policy.is_enabled()
                && walk_cost_model.is_none()
            {
                let rate = crate::network::forward_linear_measured_rate();
                walk_cost_model = Some(budget_policy::WalkCostModel::new(rate.macs_per_sec as f64));
            }
            if let (Some(model), Some(node_deadline), Some(collection_deadline)) =
                (walk_cost_model.as_ref(), per_node_deadline, deadline)
            {
                // A stacked-served member runs no walk: admission must not
                // refuse (or re-price) work that is already done.
                if !is_patches_target && !stacked_results.contains_key(node_name) {
                    let est_macs = self.collector_walk_macs_with_shapes(
                        node_name,
                        effective_target_rows,
                        Some(&ibp_bounds),
                    );
                    walk_estimated_macs = est_macs;
                    // #walk-value-record: a measured record for this exact
                    // (node, rows) — a completed walk's wall or a #chunk-abort
                    // projection from an earlier collection in this process —
                    // beats the forward-linear MAC proxy, which prices a
                    // DIFFERENT cost class (value-GEMM builds vs conv-transpose
                    // CROWN walks) and goes stale whenever the walk kernels
                    // change speed (b90a9fbf removed a measured 60x scalar
                    // tax). No record => the proxy path below, bit-identical.
                    let node_record =
                        budget_policy::node_walk_record(node_name, effective_target_rows);
                    // Review defect (cross-stream): the record must NEVER
                    // substitute for the proxy in the BASE decision — a slow
                    // recorded estimate (e.g. a contended completion) would then
                    // REVOKE an Admit the proxy grants today. The record's
                    // only role is the grant-rescue arm inside
                    // `admit_walk_with_record` (Refuse -> AdmitWithMeasuredGrant);
                    // the base estimate stays forced-override else proxy,
                    // bit-identical to the pre-record behavior.
                    let est_secs = collector_forced_walk_estimate_secs(node_name)
                        .or_else(|| est_macs.and_then(|macs| model.estimate_secs(macs)));
                    if let Some(est_secs) = est_secs {
                        let now = Instant::now();
                        let share_secs = node_deadline.saturating_duration_since(now).as_secs_f64();
                        let remaining_secs = collection_deadline
                            .saturating_duration_since(now)
                            .as_secs_f64();
                        let is_last_candidate = budget_policy::count_remaining_budget_candidates(
                            &global_budget_candidate_mask,
                            layer_index + 1,
                        ) == 0;
                        let recorded_secs = node_record.and_then(|record| record.estimate_secs());
                        let decision = budget_policy::admit_walk_with_record(
                            est_secs,
                            recorded_secs,
                            share_secs,
                            remaining_secs,
                            is_last_candidate,
                        );
                        budget_policy::note_walk_admission(model, node_name, decision);
                        match decision {
                            budget_policy::WalkAdmissionDecision::Admit => {}
                            budget_policy::WalkAdmissionDecision::AdmitWithMeasuredGrant {
                                grant_secs,
                            } => {
                                // #walk-value-record: a completion or a
                                // cooperative-abort full-walk projection prices
                                // this exact (node, rows) above its static
                                // weighted share. Grant the recorded estimate
                                // (with the standard margin), bounded by the
                                // collection deadline, instead of refusing at
                                // the static share. A walk that still fails
                                // falls back to IBP exactly as today.
                                let granted = collection_deadline
                                    .min(now + Duration::from_secs_f64(grant_secs.max(0.0)));
                                info!(
                                    "CROWN-IBP DAG: node '{}' (dim {}, rows {}) has a measured \
                                     record above its capped {share_secs:.1}s share that fits \
                                     the {remaining_secs:.1}s window — granting \
                                     {grant_secs:.1}s bounded by the collection deadline \
                                     (#walk-value-record)",
                                    node_name,
                                    ibp_bound.len(),
                                    effective_target_rows,
                                );
                                per_node_deadline = Some(granted);
                                #[cfg(test)]
                                record_m1_collector_trace(
                                    M1CollectorTraceEvent::WalkMeasuredGrantApplied {
                                        node: node_name.clone(),
                                    },
                                );
                            }
                            budget_policy::WalkAdmissionDecision::AdmitWithRollover => {
                                info!(
                                    "CROWN-IBP DAG: node '{}' (dim {}) is the LAST demanded \
                                     candidate; walk estimate {est_secs:.1}s exceeds its capped \
                                     {share_secs:.1}s share but fits the {remaining_secs:.1}s \
                                     rollover — granting the collection deadline \
                                     (#cprime-admission)",
                                    node_name,
                                    ibp_bound.len(),
                                );
                                per_node_deadline = Some(collection_deadline);
                                #[cfg(test)]
                                record_m1_collector_trace(
                                    M1CollectorTraceEvent::WalkRolloverGranted {
                                        node: node_name.clone(),
                                    },
                                );
                            }
                            budget_policy::WalkAdmissionDecision::Refuse {
                                estimated_secs,
                                budget_secs,
                            } => {
                                let reason = CrownIbpFallbackReason::WalkCostRefused;
                                info!(
                                    "CROWN-IBP DAG: node '{}' (dim {}, rows {}) walk REFUSED \
                                     upfront: estimate {estimated_secs:.1}s (macs {}, \
                                     correction {:.3}{}) x 1.25 margin > {budget_secs:.1}s \
                                     budget; share rolls to later candidates \
                                     (#cprime-admission)",
                                    node_name,
                                    ibp_bound.len(),
                                    effective_target_rows,
                                    walk_estimated_macs.unwrap_or(0),
                                    model.correction(),
                                    if model.is_calibrated() {
                                        " calibrated"
                                    } else {
                                        " prior"
                                    },
                                );
                                skip_count += 1;
                                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                                provenance.insert(
                                    node_name.clone(),
                                    BoundsProvenance::ForwardFallback(reason),
                                );
                                fallback_events.push(CrownIbpFallbackEvent {
                                    layer_index,
                                    layer_type,
                                    reason,
                                    details: format!(
                                        "node '{}' walk estimate {estimated_secs:.1}s exceeds \
                                         {budget_secs:.1}s budget (rows {}, refused upfront)",
                                        node_name, effective_target_rows,
                                    ),
                                });
                                #[cfg(test)]
                                record_m1_collector_trace(M1CollectorTraceEvent::WalkRefused {
                                    node: node_name.clone(),
                                });
                                continue;
                            }
                        }
                    }
                }
            }

            // Patches-eligible targets use the collector-specific entry point
            // that overrides matrix conv_mode for this cut-free path (#3813).
            let node_start = Instant::now();
            let layer_type_for_log = layer_type.clone();

            // #margin-subset-seed: OUTPUT-node margin-subset tightening. When
            // the initial-bounds scope published the spec-referenced OUTPUT
            // indices AND this is the OUTPUT node at/above the engagement
            // width (see `margin_subset_indices`), seed ONLY the k referenced
            // identity rows (each bit-identical in semantics to its full-width
            // counterpart by row-independence) and SCATTER them over the
            // node's sound IBP bounds; the scattered map then flows through
            // the SAME shape-restore + IBP-intersection path as a full map,
            // so every row remains a valid enclosure. Engaged proactively:
            // the k-row backward is cheaper than full-width even when the
            // conv memory cap would not trip. On ANY error the existing
            // full-width path runs unchanged (fail-open).
            //
            // Cache note (#cgan-collection-cache): a collection stored while
            // engaged carries IBP-loose unreferenced OUTPUT rows to later
            // same-box lookups — always sound (every row is a valid
            // enclosure); the verdict path derives root bounds from the
            // objective backward, never from the collected OUTPUT rows.
            // #spec-influence-cone: the same subset treatment, extended INWARD
            // to intermediate spatial targets.
            //
            // At the OUTPUT node the selected rows are the spec-referenced
            // indices. At an intermediate 3-D node they are the cells that can
            // REACH those indices — every other cell's relaxation contributes
            // nothing to the objective's backward, so tightening it is wasted
            // work. On TinyYOLO all five objectives sit at one cell of the
            // 13x13 grid, so Conv_25 needs 1,590 of its 10,816 rows (6.8x).
            //
            // Identical soundness to the OUTPUT case, and for the identical
            // reason: unselected rows keep the node's sound IBP bounds, so a
            // cone that is too small is merely looser. `None` anywhere (no
            // publication, unmodelled op, non-spatial node, or a cone covering
            // the whole grid) falls through to the historical full-width path.
            // Non-engagement diagnostic (#margin-subset-seed). When this node IS
            // the OUTPUT node, the choice between a k-row seed and a full
            // `[dim x dim]` identity is worth several GiB, so say which
            // precondition failed rather than silently taking the wide path.
            // Measured on TinyYOLO: the spec reads 5 of 21,125 outputs, but the
            // full identity pair is 8 * 21125^2 = 3.57 GB — the exact transient
            // the Conv2d scratch cap then refuses, degrading the node to IBP.
            let is_output_node = margin_subset_output_node
                .as_deref()
                .is_some_and(|output_node| output_node == node_name.as_str());
            if is_output_node && margin_subset_indices(ibp_bound.len()).is_none() {
                info!(
                    "#margin-subset-seed: OUTPUT node '{}' (dim={}) takes the FULL-WIDTH seed \
                     ({} identity pair bytes). Cause: {}.",
                    node_name,
                    ibp_bound.len(),
                    crate::network::crown_memory::identity_pair_bytes(ibp_bound.len())
                        .map_or_else(|| "overflow".to_string(), |b| b.to_string()),
                    if ibp_bound.len() < crate::output_margin_seed::MARGIN_SUBSET_MIN_OUTPUT_DIM {
                        "node is narrower than the subset-seed minimum"
                    } else {
                        "no spec indices published on this thread (the publishing scope did not \
                         reach this collection, or it ran on another thread)"
                    },
                );
            }
            let subset_indices = chunk_aware_route
                .map(|route| route.subset_indices.clone())
                .unwrap_or_else(|| {
                    // Historical lazy subset resolution. Armed M1 instead
                    // reuses the retained route that its scheduler modeled.
                    resolve_subset_indices(
                        node_name,
                        ibp_bound,
                        sparse_relu_plan,
                        is_patches_target,
                    )
                });
            if sparse_relu_plan.is_some()
                && subset_indices
                    .as_deref()
                    .is_some_and(|indices| indices.is_empty())
            {
                objective_cone_skip_count += 1;
                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                provenance.insert(
                    node_name.clone(),
                    BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::ObjectiveConeRowsSkipped,
                    ),
                );
                info!(
                    "#sparse-relu-rows: node '{}' has no selected rows after objective-cone \
                     intersection; retaining IBP without backward CROWN",
                    node_name,
                );
                continue;
            }

            // #prefix-cost-admission: price only the route that is actually
            // planned. A proper spatial subset is materialized as dense rows;
            // a full in-order spatial grid remains virtual Patches and is not
            // modeled. Likewise, a full-width non-patches target is dense. Any
            // other route has no authenticated proxy and runs unchanged.
            let mut prefix_work_units = None;
            let prefix_admission = if prefix_cost_active && !stacked_results.contains_key(node_name)
            {
                let prefix_objective_rows = match subset_indices.as_deref() {
                    Some(indices) => {
                        let virtual_full_grid = is_patches_target
                            && indices.len() == ibp_bound.len()
                            && indices.iter().enumerate().all(|(i, &row)| i == row);
                        (!virtual_full_grid).then_some(indices.len())
                    }
                    None => (!is_patches_target).then_some(ibp_bound.len()),
                };
                prefix_work_units = prefix_objective_rows
                    .and_then(|rows| dense_crown_prefix_work_units(self, node_name, rows));
                prefix_cost_model.admit(
                    prefix_work_units,
                    prefix_cost_remaining_window(deadline, per_node_deadline, Instant::now()),
                )
            } else {
                budget_policy::PrefixCostAdmission::RunWithoutEstimate
            };
            if let budget_policy::PrefixCostAdmission::RetainIbp {
                predicted_secs,
                remaining_secs,
                completed_samples,
            } = prefix_admission
            {
                let reason = CrownIbpFallbackReason::PerNodeDeadlineExceeded;
                skip_count += 1;
                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                provenance.insert(node_name.clone(), BoundsProvenance::ForwardFallback(reason));
                fallback_events.push(CrownIbpFallbackEvent {
                    layer_index,
                    layer_type: layer_type.clone(),
                    reason,
                    details: format!(
                        "node '{node_name}' prefix-cost admission retained IBP: optimistic dense \
                         prefix estimate {predicted_secs:.3}s exceeds remaining target window \
                         {remaining_secs:.3}s after {completed_samples} completed same-collection \
                         samples"
                    ),
                });
                info!(
                    "CROWN-IBP DAG: node '{}' prefix-cost admission predicts {:.3}s even at 2x \
                     the fastest completed same-collection rate, above its {:.3}s target window; \
                     retaining IBP before the backward walk (samples={})",
                    node_name, predicted_secs, remaining_secs, completed_samples,
                );
                continue;
            }
            let subset_is_sparse = sparse_relu_plan.is_some();
            let mut sparse_partial_succeeded = false;
            let mut atomic_subset_error = None;
            let subset_was_planned = subset_indices.is_some();
            // Time only the dense walk represented by `prefix_work_units`.
            // General node bookkeeping, subset discovery/scatter, result
            // intersection, and a failed first route are deliberately excluded
            // from a completed-rate sample.
            let mut completed_prefix_walk_elapsed = None;
            let margin_subset_bound = subset_indices.and_then(|indices| {
                let subset_walk_start =
                    (prefix_cost_active && prefix_work_units.is_some()).then(Instant::now);
                let subset_attempt = if collector_force_subset_failure(node_name) {
                    Err(NyError::UnsupportedConfiguration(format!(
                        "test-injected subset failure for '{node_name}'"
                    )))
                } else {
                    self.propagate_crown_to_node_subset(
                        input,
                        node_name,
                        &crown_ibp_bounds,
                        &ibp_bounds,
                        engine,
                        if subset_is_sparse {
                            "CROWN-IBP-sparse-relu-rows"
                        } else {
                            "CROWN-IBP-margin-subset"
                        },
                        per_node_deadline,
                        deadline_is_hard,
                        is_patches_target,
                        &indices,
                        cut_ctx.as_ref(),
                    )
                };
                let subset_walk_elapsed = subset_walk_start.map(|start| start.elapsed());
                match subset_attempt {
                    Ok((lower_rows, upper_rows)) => {
                        match scatter_margin_rows_over_bounds(
                            ibp_bound,
                            &indices,
                            &lower_rows,
                            &upper_rows,
                        ) {
                            Ok(bounds) => {
                                completed_prefix_walk_elapsed = subset_walk_elapsed;
                                sparse_partial_succeeded = subset_is_sparse;
                                info!(
                                    "CROWN-IBP DAG: node '{}' {} subset seed engaged \
                                         (k={} of {} rows; scattered over IBP)",
                                    node_name,
                                    if subset_is_sparse {
                                        "sparse-ReLU"
                                    } else {
                                        "margin"
                                    },
                                    indices.len(),
                                    ibp_bound.len()
                                );
                                Some(bounds)
                            }
                            Err(e) => {
                                if is_atomic_selected_target && subset_is_sparse {
                                    atomic_subset_error =
                                        Some(NyError::UnsupportedConfiguration(format!(
                                            "atomic sparse target '{}' scatter failed before \
                                             complete publication: {e}",
                                            node_name
                                        )));
                                } else {
                                    debug!(
                                        "CROWN-IBP DAG: node '{}' subset scatter \
                                             failed ({e}); falling back to full-width backward",
                                        node_name
                                    );
                                }
                                None
                            }
                        }
                    }
                    Err(e) => {
                        if is_atomic_selected_target && subset_is_sparse {
                            atomic_subset_error = Some(if e.is_deadline_exceeded() {
                                e
                            } else {
                                NyError::UnsupportedConfiguration(format!(
                                    "atomic sparse target '{}' did not complete: {e}",
                                    node_name
                                ))
                            });
                        } else {
                            debug!(
                                "CROWN-IBP DAG: node '{}' subset backward failed \
                                     ({e}); falling back to full-width backward",
                                node_name
                            );
                        }
                        None
                    }
                }
            });

            // A subset failure changes the route that will ACTUALLY execute.
            // Reweight the current candidate to the full target width and
            // recompute its live slice before entering the fail-open backward.
            // This uses the same fixed-wave scheduling plan as execution; a
            // sequential/adaptive route remains at raw full width.
            let runtime_subset_fail_open = subset_was_planned
                && margin_subset_bound.is_none()
                && atomic_subset_error.is_none();
            if runtime_subset_fail_open {
                // The planned subset's work estimate no longer describes the
                // route that will execute. Recompute only for a full dense
                // fallback; a Patches/chunk route remains deliberately
                // unmodeled and cannot train the admission model.
                if prefix_cost_active {
                    prefix_work_units = (!is_patches_target)
                        .then(|| dense_crown_prefix_work_units(self, node_name, ibp_bound.len()))
                        .flatten();
                }
            }
            let mut full_fallback_budget_rejected = false;
            if chunk_aware_budget && runtime_subset_fail_open {
                effective_target_rows = collector_budget_rows(
                    ibp_bound.len(),
                    effective_target_rows,
                    collector_budget_route(subset_was_planned, false),
                );
                // #target-cost-weight: this site REPLACES the planned weight with
                // the full-dense one, so it must re-apply the same prefix-cost
                // multiplier the planner applied — otherwise a runtime fail-open
                // silently reverts this one target to raw-row pricing while every
                // other candidate stays repriced, and the denominator they share
                // stops meaning one thing.
                let full_weight = budget_policy::demanded_target_work_weight(
                    effective_target_rows,
                    chunk_aware_route.and_then(|route| route.scheduling_plan),
                    true,
                    true,
                ) * prefix_cost_multipliers
                    .as_ref()
                    .and_then(|multipliers| multipliers.get(layer_index).copied())
                    .unwrap_or(1.0);
                if let Some(weight) = global_budget_candidate_weights.get_mut(layer_index) {
                    *weight = full_weight;
                }
                let full_global_per_node = deadline.and_then(|d| {
                    let now = Instant::now();
                    if now >= d {
                        return None;
                    }
                    let remaining_secs = d.duration_since(now).as_secs_f64();
                    let remaining_weight_sum = budget_policy::sum_remaining_budget_weights(
                        &global_budget_candidate_weights,
                        layer_index,
                    );
                    let cap_dims = budget_policy::weighted_budget_cap_dims(
                        full_weight,
                        ibp_bound.len() as f64,
                        true,
                    );
                    let secs = budget_policy::compute_weighted_per_node_budget_secs(
                        remaining_secs,
                        remaining_weight_sum,
                        full_weight,
                        cap_dims,
                        &per_node_time_budget,
                    )?;
                    Some(now + Duration::from_secs_f64(secs))
                });
                per_node_deadline = budget_policy::merge_per_node_deadlines(
                    full_global_per_node,
                    patches_per_node,
                    deadline.is_some(),
                );
                full_fallback_budget_rejected = deadline.is_some() && per_node_deadline.is_none();
                if collector_force_full_fallback_budget_rejection(node_name) {
                    per_node_deadline = None;
                    full_fallback_budget_rejected = true;
                }
                #[cfg(test)]
                record_m1_collector_trace(M1CollectorTraceEvent::RuntimeSubsetFailOpen {
                    node: node_name.clone(),
                    rows: effective_target_rows,
                    weight: full_weight,
                    deadline_recomputed: deadline.is_some(),
                    fallback_rejected: full_fallback_budget_rejected,
                });
            }

            // The subset attempt may have changed the necessary route from a
            // cheap dense subset to a full dense walk. Re-run admission against
            // the recomputed live share; the failed subset's elapsed time is
            // not learned, but it correctly reduces the remaining window.
            if prefix_cost_active && runtime_subset_fail_open && !full_fallback_budget_rejected {
                let remaining_full_window =
                    prefix_cost_remaining_window(deadline, per_node_deadline, Instant::now());
                if let budget_policy::PrefixCostAdmission::RetainIbp {
                    predicted_secs,
                    remaining_secs,
                    completed_samples,
                } = prefix_cost_model.admit(prefix_work_units, remaining_full_window)
                {
                    let reason = CrownIbpFallbackReason::PerNodeDeadlineExceeded;
                    skip_count += 1;
                    crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                    provenance.insert(node_name.clone(), BoundsProvenance::ForwardFallback(reason));
                    fallback_events.push(CrownIbpFallbackEvent {
                        layer_index,
                        layer_type: layer_type.clone(),
                        reason,
                        details: format!(
                            "node '{node_name}' full-fallback prefix-cost admission retained IBP: \
                             optimistic dense prefix estimate {predicted_secs:.3}s exceeds \
                             remaining target window {remaining_secs:.3}s after \
                             {completed_samples} completed same-collection samples"
                        ),
                    });
                    info!(
                        "CROWN-IBP DAG: node '{}' full-fallback prefix-cost admission predicts \
                         {:.3}s even at 2x the fastest completed same-collection rate, above its \
                         {:.3}s target window; retaining IBP before the full backward walk \
                         (samples={})",
                        node_name, predicted_secs, remaining_secs, completed_samples,
                    );
                    continue;
                }
            }

            let mut expected_fixed_waves = chunk_aware_route
                .and_then(|route| route.scheduling_plan)
                .map(|plan| plan.fixed_waves);
            let fixed_wave_drift_injected =
                expected_fixed_waves.is_some() && collector_force_fixed_wave_drift(node_name);
            if fixed_wave_drift_injected {
                let plan = expected_fixed_waves
                    .as_mut()
                    .expect("drift injection requires a fixed-wave plan");
                plan.wave_count = plan.wave_count.saturating_add(1);
            }
            let mut full_backward_attempted = false;
            let crown_result = if full_fallback_budget_rejected {
                Err(NyError::DeadlineExceeded(format!(
                    "CROWN-IBP: node '{node_name}' full-width subset fail-open share is below floor"
                )))
            } else if let Some(bounds) = margin_subset_bound {
                Ok(super::target_backward::TargetCrownCollectionResult::Complete(bounds))
            } else if let Some(error) = atomic_subset_error {
                // Atomic authority boundary: never turn an incomplete sparse
                // attempt into a second dense attempt, and never publish any
                // completed prefix. The match below retains the baseline.
                Err(error)
            } else if let Some(stacked_bound) = stacked_results.remove(node_name) {
                // #cgan-stacked-backward: this member's rows were computed by
                // the pre-loop stacked pass; serve them through the SAME
                // Complete handling (shape restore + IBP intersection +
                // provenance) a solo walk's result takes.
                super::cgan_stacked::stacked_event(format_args!("stage=serve node='{node_name}'"));
                Ok(super::target_backward::TargetCrownCollectionResult::Complete(stacked_bound))
            } else {
                full_backward_attempted = true;
                if expected_fixed_waves.is_some() {
                    #[cfg(test)]
                    record_m1_collector_trace(M1CollectorTraceEvent::FixedWaveDispatch {
                        node: node_name.clone(),
                        drift_injected: fixed_wave_drift_injected,
                    });
                }
                let full_walk_start =
                    (prefix_cost_active && prefix_work_units.is_some()).then(Instant::now);
                let result = self.propagate_crown_to_node_with_partial_for_collector(
                    input,
                    node_name,
                    &crown_ibp_bounds,
                    &ibp_bounds,
                    engine,
                    per_node_deadline,
                    deadline_is_hard,
                    chunk_override,
                    cut_ctx.as_ref(),
                    is_patches_target,
                    deadline_salvage_policy,
                    expected_fixed_waves,
                    runtime_policy,
                );
                if matches!(
                    &result,
                    Ok(super::target_backward::TargetCrownCollectionResult::Complete(_))
                ) {
                    completed_prefix_walk_elapsed = full_walk_start.map(|start| start.elapsed());
                }
                result
            };
            // #cprime-admission: only a COMPLETED walk calibrates the cost
            // model (an aborted one gives a lower bound, not a rate).
            let walk_completed_ok = matches!(
                &crown_result,
                Ok(super::target_backward::TargetCrownCollectionResult::Complete(_))
            );
            // #cprime-abort-calib (2026-08-03): ...unless the abort came from
            // #chunk-abort, which measures a real per-row rate before giving up
            // and publishes the full-walk projection. That is a genuine rate
            // sample, and on collections where NOTHING completes it is the only
            // one — the case that left cgan_2023 admitting 61/61 doomed walks.
            // Strictly weaker than a completed-walk calibration: raise-only,
            // never marks the model calibrated (see `observe_aborted_walk`).
            if !walk_completed_ok {
                if let (Some(model), Some(macs), Some(projected)) = (
                    walk_cost_model.as_mut(),
                    walk_estimated_macs,
                    budget_policy::take_walk_abort_projection(),
                ) {
                    let before = model.correction();
                    model.observe_aborted_walk(macs, projected);
                    // #walk-value-record: keep the projection per (node, rows)
                    // too, so a LATER collection prices this exact walk by its
                    // own measured projection instead of only the global
                    // correction. Skipped on a subset fail-open, whose rows no
                    // longer describe the walk that ran.
                    if !runtime_subset_fail_open {
                        budget_policy::record_node_walk_abort_projection(
                            node_name,
                            effective_target_rows,
                            projected,
                        );
                    }
                    if model.correction() > before {
                        debug!(
                            "CROWN-IBP DAG: walk cost correction raised {before:.3} -> {:.3} \
                             from '{node_name}' #chunk-abort projection {projected:.1}s \
                             ({macs} MACs) (#cprime-abort-calib)",
                            model.correction(),
                        );
                    }
                }
            }
            if let (
                Ok(super::target_backward::TargetCrownCollectionResult::Complete(_)),
                Some(elapsed),
            ) = (&crown_result, completed_prefix_walk_elapsed)
            {
                prefix_cost_model.observe_completed(prefix_work_units, elapsed);
            }
            match crown_result {
                Ok(super::target_backward::TargetCrownCollectionResult::Complete(crown_bound)) => {
                    let forward_contract =
                        GraphTargetShapeContract::from_bounds(node_name, ibp_bound);
                    let crown_bound = match forward_contract.reshape_for_forward_match(
                        crown_bound,
                        ibp_bound,
                        "CROWN-IBP forward-shape restore",
                    ) {
                        Ok(reshaped) => Some(reshaped),
                        Err(NyError::ShapeMismatch { expected, got }) => {
                            let reason = CrownIbpFallbackReason::ShapeMismatch;
                            debug!(
                                "CROWN-IBP DAG: {} shape mismatch IBP={:?} vs CROWN={:?}, using IBP",
                                node_name, expected, got
                            );
                            crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                            provenance.insert(
                                node_name.clone(),
                                BoundsProvenance::ForwardFallback(reason),
                            );
                            fallback_events.push(CrownIbpFallbackEvent {
                                layer_index,
                                layer_type: layer_type.clone(),
                                reason,
                                details: format!(
                                    "node '{}' crown shape {:?} does not match forward shape {:?}",
                                    node_name, got, expected
                                ),
                            });
                            None
                        }
                        Err(err) => return Err(err),
                    };
                    if let Some(crown_bound) = crown_bound {
                        // #crown-gain-probe (NY_CROWN_GAIN=1): report how much the
                        // CROWN bound actually tightens the IBP box for this
                        // target. A CROWN bound whose certified coefficient error
                        // has exploded concretizes no better than IBP, so the work
                        // is spent for nothing — and that is invisible in the
                        // provenance counters, which only say "Crown".
                        if std::env::var("NY_CROWN_GAIN").ok().as_deref() == Some("1") {
                            let width = |b: &BoundedTensor| -> f64 {
                                b.lower()
                                    .iter()
                                    .zip(b.upper().iter())
                                    .map(|(&l, &u)| f64::from(u) - f64::from(l))
                                    .filter(|d| d.is_finite())
                                    .sum::<f64>()
                            };
                            let (wi, wc) = (width(ibp_bound), width(&crown_bound));
                            info!(
                                "#crown-gain node='{}' ibp_width={:.6e} crown_width={:.6e} ratio={:.4}",
                                node_name,
                                wi,
                                wc,
                                if wi > 0.0 { wc / wi } else { f64::NAN }
                            );
                            // A SUM over every row says nothing about the rows the
                            // verdict actually reads. On yolo_2023 the output node
                            // is 21,125 wide and the spec reads 5 of them, so the
                            // aggregate above can be dominated by 21,120 rows no
                            // one looks at. Report the published spec rows
                            // individually so the collector's value at those rows
                            // can be compared against the number that reaches
                            // `check_spec` — which is what identifies WHICH pass
                            // produces the deciding bound.
                            if let Some(indices) = margin_subset_indices(ibp_bound.len()) {
                                for &idx in indices.iter() {
                                    let row = |b: &BoundedTensor| -> (f64, f64) {
                                        (
                                            f64::from(b.lower().as_slice().map_or(f32::NAN, |s| {
                                                s.get(idx).copied().unwrap_or(f32::NAN)
                                            })),
                                            f64::from(b.upper().as_slice().map_or(f32::NAN, |s| {
                                                s.get(idx).copied().unwrap_or(f32::NAN)
                                            })),
                                        )
                                    };
                                    let (il, iu) = row(ibp_bound);
                                    let (cl, cu) = row(&crown_bound);
                                    info!(
                                        "#crown-gain-row node='{}' idx={} ibp=[{:.4},{:.4}] \
                                         crown=[{:.4},{:.4}] ibp_w={:.4} crown_w={:.4}",
                                        node_name,
                                        idx,
                                        il,
                                        iu,
                                        cl,
                                        cu,
                                        iu - il,
                                        cu - cl
                                    );
                                }
                            }
                        }
                        // #crown-honest-provenance: MEASURE what this completed
                        // CROWN result actually bought, BEFORE recording a quality
                        // claim about it. Integer counts only -- no width sum, so
                        // no summation-order or GEMM-thread-count sensitivity.
                        let gain = crown_gain_stat(ibp_bound, &crown_bound);
                        let vacuous = honest_provenance_enabled && gain.is_vacuous();
                        let intersection = intersect_completed_crown_target(
                            ibp_bound,
                            &crown_bound,
                            collection_mode,
                        );
                        match intersection {
                            // Per-element intersection succeeded (#2935).
                            Some((tightened, disjoint)) => {
                                if disjoint > 0 {
                                    debug!(
                                        "CROWN-IBP DAG: {} per-element intersection: {} of {} elements disjoint, used union fallback",
                                        node_name, disjoint, tightened.len()
                                    );
                                }
                                if downstream_resweep
                                    && disjoint == 0
                                    && (tightened.lower() != original_ibp_bound.lower()
                                        || tightened.upper() != original_ibp_bound.upper())
                                {
                                    if let Some(sources) = downstream_resweep_sources.as_mut() {
                                        sources.insert(node_name.clone());
                                    }
                                }
                                // THE STORED BOUND IS IDENTICAL IN BOTH ARMS BELOW.
                                // Only the quality claim and the cut gate differ, so
                                // this decision point cannot change any bound.
                                crown_ibp_bounds.insert(node_name.clone(), tightened);
                                if vacuous {
                                    let reason = CrownIbpFallbackReason::CrownVacuousResult;
                                    warn!(
                                        "CROWN-IBP DAG: node '{}' CROWN result is VACUOUS -- \
                                         {}/{} elements the forward pass had bounded came back \
                                         non-finite, {}/{} improvable elements tightened \
                                         materially. Recording {:?}, not Crown: MORE BUDGET \
                                         CANNOT HELP THIS TARGET.",
                                        node_name,
                                        gain.crown_lost,
                                        gain.ibp_finite,
                                        gain.tightened_material,
                                        gain.improvable,
                                        reason,
                                    );
                                    provenance.insert(
                                        node_name.clone(),
                                        BoundsProvenance::ForwardFallback(reason),
                                    );
                                    fallback_events.push(CrownIbpFallbackEvent {
                                        layer_index,
                                        layer_type: layer_type.clone(),
                                        reason,
                                        details: format!(
                                            "node '{}' CROWN lost finiteness on {}/{} elements, \
                                             tightened {}/{} improvable materially ({} disjoint)",
                                            node_name,
                                            gain.crown_lost,
                                            gain.ibp_finite,
                                            gain.tightened_material,
                                            gain.improvable,
                                            disjoint,
                                        ),
                                    });
                                } else {
                                    // #crown-honest-provenance: FINITE but worse
                                    // than the IBP box it started from. Sound (the
                                    // intersection clamps it back), and still
                                    // tagged `Crown` because it did not fail --
                                    // but report it, because a node that spends a
                                    // full backward walk to lose ground is a
                                    // strategy signal, not a resource shortfall.
                                    if honest_provenance_enabled && gain.is_regressed() {
                                        info!(
                                            "CROWN-IBP DAG: node '{}' CROWN result REGRESSED -- \
                                             finite, but materially WIDER than IBP on {}/{} \
                                             improvable elements while tightening {}. The stored \
                                             bound is rescued by the intersection; the backward \
                                             walk bought nothing. MORE BUDGET CANNOT HELP THIS \
                                             TARGET.",
                                            node_name,
                                            gain.widened_material,
                                            gain.improvable,
                                            gain.tightened_material,
                                        );
                                    }
                                    provenance.insert(node_name.clone(), BoundsProvenance::Crown);
                                    // #cut-provenance-gate: this node's bound is
                                    // CROWN-tight, so it is now safe to concretize
                                    // a later walk against its box.
                                    //
                                    // #crown-honest-provenance -- AND ONLY NOW. A
                                    // vacuous CROWN intersects to EXACTLY the IBP
                                    // box, so `box_is_finite` accepts it and a later
                                    // walk cuts against IBP width. The `Crown` tag
                                    // was the last route by which an IBP-quality box
                                    // could still pass that gate.
                                    if !sparse_partial_succeeded {
                                        if let Some(ctx) = cut_ctx.as_ref() {
                                            ctx.mark_tight(node_name);
                                        }
                                    }
                                }
                                if is_atomic_selected_target {
                                    info!(
                                        "#cgan-sparse-target-complete: target='{}' complete=true \
                                         selected_rows={} total_rows={} publication=intersection",
                                        node_name,
                                        sparse_relu_plan.map_or_else(
                                            || ibp_bound.len(),
                                            |plan| plan.selected_len()
                                        ),
                                        ibp_bound.len(),
                                    );
                                }
                            }
                            // Malformed intersection, or a disjoint candidate
                            // rejected by a typed shrink-only transaction —
                            // retain the full forward baseline.
                            None => {
                                let reason = CrownIbpFallbackReason::EmptyIntersection;
                                debug!(
                                    "CROWN-IBP DAG: {} forward/CROWN intersection malformed or rejected as disjoint, using forward baseline",
                                    node_name,
                                );
                                crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                                provenance.insert(
                                    node_name.clone(),
                                    BoundsProvenance::ForwardFallback(reason),
                                );
                                fallback_events.push(CrownIbpFallbackEvent {
                                    layer_index,
                                    layer_type: layer_type.clone(),
                                    reason,
                                    details: format!(
                                        "node '{}' forward/CROWN intersection was malformed or a typed shrink-only transaction rejected a disjoint candidate for shape {:?}",
                                        node_name,
                                        ibp_bound.shape()
                                    ),
                                });
                            }
                        }
                    }
                }
                Ok(super::target_backward::TargetCrownCollectionResult::DeadlineTruncated {
                    bounds: partial_bound,
                    completed_rows,
                    total_rows,
                    details,
                }) => {
                    if is_atomic_selected_target {
                        let reason = CrownIbpFallbackReason::PerNodeDeadlineExceeded;
                        crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                        provenance
                            .insert(node_name.clone(), BoundsProvenance::ForwardFallback(reason));
                        fallback_events.push(CrownIbpFallbackEvent {
                            layer_index,
                            layer_type: layer_type.clone(),
                            reason,
                            details: format!(
                                "atomic target '{node_name}' completed only \
                                 {completed_rows}/{total_rows} rows ({details}); discarded the \
                                 whole candidate and retained the certified baseline"
                            ),
                        });
                        info!(
                            "#cgan-sparse-target-complete: target='{}' complete=false \
                             completed_rows={}/{} publication=baseline-only",
                            node_name, completed_rows, total_rows,
                        );
                        continue;
                    }
                    // Deadline-truncated objective-row salvage. `partial_bound`
                    // was seeded from the certified target IBP box and overwrote
                    // only fully completed, on-time chunks. Intersect once more
                    // through the collector's ordinary forward/CROWN seam so
                    // completed rows can tighten while unfinished rows remain no
                    // looser than the forward box.
                    let forward_contract =
                        GraphTargetShapeContract::from_bounds(node_name, ibp_bound);
                    let partial_bound = forward_contract
                        .reshape_for_forward_match(
                            partial_bound,
                            ibp_bound,
                            "CROWN-IBP partial-row forward-shape restore",
                        )
                        .ok();
                    let (retained, disjoint) =
                        retain_partial_crown_rows(ibp_bound, partial_bound.as_ref());

                    if downstream_resweep
                        && disjoint == 0
                        && (retained.lower() != original_ibp_bound.lower()
                            || retained.upper() != original_ibp_bound.upper())
                    {
                        if let Some(sources) = downstream_resweep_sources.as_mut() {
                            sources.insert(node_name.clone());
                        }
                    }

                    let reason = CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded;
                    crown_ibp_bounds.insert(node_name.clone(), retained);
                    provenance.insert(node_name.clone(), BoundsProvenance::ForwardFallback(reason));
                    fallback_events.push(CrownIbpFallbackEvent {
                        layer_index,
                        layer_type: layer_type.clone(),
                        reason,
                        details: format!(
                            "node '{node_name}' retained {completed_rows}/{total_rows} fully \
                             completed CROWN rows over certified IBP after deadline truncation \
                             ({details}); rejected the late in-flight chunk/wave; \
                             intersection disjoint rows={disjoint}"
                        ),
                    });
                    // A partially completed target exhausted its share just like
                    // the historical all-IBP deadline fallback. Preserve the
                    // scheduling signal without granting CROWN cut authority.
                    if hopeless_class_skip_enabled() {
                        hopeless_min_dim =
                            Some(hopeless_min_dim.map_or(effective_target_rows, |prev| {
                                prev.min(effective_target_rows)
                            }));
                    }
                    info!(
                        "CROWN-IBP DAG: node '{}' deadline-truncated after {}/{} objective rows; \
                         retained completed-row CROWN∩IBP, left unfinished rows at IBP, and \
                         recorded {:?} (complete=false)",
                        node_name, completed_rows, total_rows, reason,
                    );
                }
                // #3166, #3602, #3499: UnsupportedOp/Configuration, ShapeMismatch,
                // or NumericalInstability from CROWN backward — IBP fallback is
                // sound. NumericalInstability catches non-finite pre-activation
                // bounds (e.g., from Sqrt, Softmax) that prevent CROWN relaxation.
                // UnsupportedConfiguration now also covers per-node deadline
                // exceeded (#3499), which returns from propagate_crown_to_node_core.
                // CpuMemoryExceeded is the Conv2d backward memory-cap backstop
                // (#conv-crown-oom): the dense coefficient buffer would exceed the
                // per-buffer cap, so this target degrades to sound IBP instead of
                // OOMing. IBP bounds are a valid over-approximation.
                Err(
                    e @ (NyError::UnsupportedOp(_)
                    | NyError::UnsupportedConfiguration(_)
                    | NyError::ShapeMismatch { .. }
                    | NyError::NumericalInstability(_)
                    | NyError::CpuMemoryExceeded { .. }
                    | NyError::DeadlineExceeded(_)),
                ) => {
                    // #3795: structural match on DeadlineExceeded replaces string matching
                    let reason = if e.is_deadline_exceeded() {
                        CrownIbpFallbackReason::PerNodeDeadlineExceeded
                    } else if e.is_cpu_memory_exceeded() {
                        CrownIbpFallbackReason::MemoryBudgetExceeded
                    } else {
                        CrownIbpFallbackReason::CrownPropagationError
                    };
                    debug!(
                        "CROWN-IBP DAG: {} CROWN backward failed ({}), using IBP",
                        node_name, e,
                    );
                    // ADAPTIVE HOPELESS-CLASS SKIP (#cifar100-collector-order).
                    //
                    // This target consumed its ENTIRE per-node share and produced
                    // nothing. Measured on CIFAR100_resnet_medium: the four
                    // 128x4x4 dense-routed targets each need >120s against a 4.3s
                    // share (28x over) and always fail, while Gemm_56/Gemm_58 —
                    // dim 100, adjacent to the objective, and the two most
                    // objective-relevant targets in the graph — cost 5.88s each
                    // and NEVER START, because the collection deadline expires
                    // while the hopeless ones burn it. That trades 11.8s of work
                    // worth having for 17s that provably cannot finish.
                    //
                    // So: once a target of dimension D has burned its whole share
                    // and failed on time, treat every LATER candidate of dimension
                    // >= D as hopeless too and route it straight to IBP, letting
                    // the remaining budget reach the cheap tail. Cost is
                    // monotone-ish in target dimension along a walk, so this is a
                    // measured predictor rather than an assumed cost model.
                    //
                    // SOUND: skipping only ever substitutes IBP for CROWN, and
                    // IBP is a valid enclosure. It can lose tightness, never
                    // validity. Disable with NY_NO_HOPELESS_CLASS_SKIP=1.
                    if reason == CrownIbpFallbackReason::PerNodeDeadlineExceeded
                        && full_backward_attempted
                        && hopeless_class_skip_enabled()
                    {
                        hopeless_min_dim = Some(
                            hopeless_min_dim.map_or(effective_target_rows, |prev: usize| {
                                prev.min(effective_target_rows)
                            }),
                        );
                        #[cfg(test)]
                        record_m1_collector_trace(M1CollectorTraceEvent::HopelessLearned {
                            node: node_name.clone(),
                            rows: effective_target_rows,
                        });
                    }
                    crown_ibp_bounds.insert(node_name.clone(), ibp_bound.clone());
                    provenance.insert(node_name.clone(), BoundsProvenance::ForwardFallback(reason));
                    fallback_events.push(CrownIbpFallbackEvent {
                        layer_index,
                        layer_type,
                        reason,
                        details: format!(
                            "node '{}' CROWN backward failed ({e}), IBP fallback",
                            node_name
                        ),
                    });
                }
                Err(e) => return Err(e),
            }
            let node_secs = node_start.elapsed().as_secs_f64();
            // #cprime-admission self-calibration: the FIRST completed walk's
            // estimate-vs-actual ratio replaces the census prior for every
            // subsequent admission decision in this collection.
            // A runtime subset fail-open executed a DIFFERENT (full-width)
            // walk than the subset-rows estimate priced; calibrating from
            // that pair would inflate the correction and over-refuse later
            // targets, so it is excluded.
            if walk_completed_ok && !is_patches_target && !runtime_subset_fail_open {
                // #walk-value-record: remember the measured completion for this
                // exact (node, rows) so a later collection in this process
                // prices it by measurement instead of the proxy — and can GRANT
                // it this cost when the static share is smaller. Recorded even
                // when the collection is unbounded (a research pass's completion
                // is a valid price for the scored re-collection). Admission-only
                // state; bounds authority is untouched.
                budget_policy::record_node_walk_completed(
                    node_name,
                    effective_target_rows,
                    node_secs,
                );
                if let (Some(model), Some(macs)) = (walk_cost_model.as_mut(), walk_estimated_macs) {
                    let was_calibrated = model.is_calibrated();
                    model.observe_completed_walk(macs, node_secs);
                    if !was_calibrated && model.is_calibrated() {
                        debug!(
                            "CROWN-IBP DAG: walk cost model calibrated on '{}' — actual \
                             {node_secs:.3}s for {macs} MACs, correction {:.3} \
                             (#cprime-admission)",
                            node_name,
                            model.correction(),
                        );
                    }
                }
            }
            patches_budget.record_elapsed(is_patches_target, node_secs);
            crown_node_count += 1;
            crown_total_secs += node_secs;
            if node_secs > 0.5 {
                info!(
                    "CROWN-IBP DAG: node {}/{} '{}' ({}) took {node_secs:.3}s",
                    layer_index, total_nodes, node_name, layer_type_for_log,
                );
            }
        }

        let collection_secs = collection_start.elapsed().as_secs_f64();
        if downstream_resweep {
            info!(
                "#crown-ibp-downstream-resweep: merged {downstream_resweep_merged}/{total_nodes} \
                 node images; narrowed {downstream_resweep_narrowed}; \
                 gate={CROWN_IBP_DOWNSTREAM_RESWEEP_ENV}=1"
            );
        }
        if collection_secs > 0.1 {
            info!(
                "CROWN-IBP DAG collection: {collection_secs:.3}s total, \
                 {crown_node_count} crown nodes ({crown_total_secs:.3}s), \
                 {skip_count} skipped, {demand_skip_count} demand-skipped, \
                 {stable_relu_skip_count} stable-ReLU-skipped, \
                 {objective_cone_skip_count} objective-cone-skipped, {total_nodes} total",
            );
        }

        // #crown-degrade-visibility: report how many DEMANDED targets actually
        // completed with CROWN authority versus remaining fallback-grade
        // (including explicitly truncated partial-row hybrids), and why.
        //
        // WHY THIS IS DEFAULT-ON. `crown_node_count` above counts ATTEMPTS, not
        // successes: a node whose backward trips the per-node time budget or the
        // Conv2d dense memory cap is counted, then has its IBP bound written back
        // and continues. So a collection can report "8 crown nodes, 0 skipped"
        // while 7 of those 8 kept pure IBP. On a deep conv graph that is the
        // difference between a usable bound and a useless one, because IBP width
        // compounds multiplicatively per conv layer.
        //
        // Measured instance of exactly that (TinyYOLO / yolo_2023, 2026-07-27):
        // 8 demanded targets, provenance Crown=1, PerNodeDeadlineExceeded=6,
        // MemoryBudgetExceeded=1 -> the collected map was IBP everywhere that
        // mattered and the root objective bound came out byte-identical to the
        // pure-IBP path after 73 s of work. Nothing in the default log said so;
        // it took a dark `NY_CONV_PATCHES_DEBUG` dump to see it. This summary
        // makes that failure mode legible without a rebuild or a rerun.
        //
        // Sound either way: this only reports provenance already recorded by the
        // loop. Every fallback bound is a valid enclosure; the cost is tightness.
        let demanded_total = crown_node_count + skip_count;
        if demanded_total > 0 {
            // Count without allocating: every diagnostic-only string/map stays
            // behind the logarithmic admission receipt below.
            let degraded = provenance
                .values()
                .filter(|prov| {
                    matches!(
                        prov,
                        BoundsProvenance::ForwardFallback(reason)
                            if !is_structural_crown_skip(*reason)
                    )
                })
                .count();
            if degraded > 0 {
                // Losing most demanded targets to a RESOURCE limit (time/memory)
                // is a bound-quality cliff, not a tuning detail: say so loudly.
                let quality_cliff = degraded >= demanded_total.div_ceil(2);
                let receipt = if quality_cliff {
                    self.crown_degradation_warning_log_receipt()
                } else {
                    self.crown_degradation_info_log_receipt()
                };
                if let Some(receipt) = receipt {
                    let crown_kept = provenance
                        .values()
                        .filter(|p| matches!(p, BoundsProvenance::Crown))
                        .count();
                    let mut by_reason: BTreeMap<String, usize> = BTreeMap::new();
                    for prov in provenance.values() {
                        if let BoundsProvenance::ForwardFallback(reason) = prov {
                            if is_structural_crown_skip(*reason) {
                                continue;
                            }
                            *by_reason.entry(format!("{reason:?}")).or_default() += 1;
                        }
                    }
                    let detail = by_reason
                        .iter()
                        .map(|(reason, n)| format!("{reason}={n}"))
                        .collect::<Vec<_>>()
                        .join(" ");
                    // NAME the degraded targets, with the shape that decided
                    // their route. A reason histogram says a resource ran out;
                    // it does not say WHICH node, how wide it was, or whether it
                    // was even eligible for the memory-light patches path — and
                    // the patches start gate requires a 3-D spatial target, so a
                    // 1-D target (e.g. a post-Flatten output) can only ever
                    // densify. Recovering that on yolo_2023 previously took
                    // source edits and reruns. Build this relatively expensive
                    // detail only for retained logarithmic reports.
                    let mut named: Vec<String> = provenance
                        .iter()
                        .filter_map(|(name, prov)| match prov {
                            BoundsProvenance::ForwardFallback(reason)
                                if !is_structural_crown_skip(*reason) =>
                            {
                                let shape = ibp_bounds
                                    .get(name)
                                    .map(|b| format!("{:?}", b.shape()))
                                    .unwrap_or_else(|| "?".to_string());
                                let dim = ibp_bounds.get(name).map_or(0, |b| b.len());
                                let patches_eligible =
                                    ibp_bounds.get(name).is_some_and(|b| b.shape().len() == 3);
                                Some(format!(
                                    "{name}{shape} dim={dim} {reason:?}{}",
                                    if patches_eligible {
                                        ""
                                    } else {
                                        " NOT-3D=cannot-start-in-patches"
                                    }
                                ))
                            }
                            _ => None,
                        })
                        .collect();
                    named.sort();
                    info!("CROWN-IBP DAG degraded targets: {}", named.join(" | "));
                    let occurrence = receipt.occurrence;
                    let suppressed = receipt.suppressed_since_previous_checkpoint;
                    if quality_cliff {
                        warn!(
                            "CROWN-IBP DAG: {degraded}/{demanded_total} DEMANDED targets did not \
                         complete CROWN ({detail}); only {crown_kept} kept CROWN authority. \
                         Intermediate bounds remain fallback-grade (IBP plus any explicitly \
                         retained partial rows) — on a deep conv graph this compounds \
                         multiplicatively and the root bound may be far looser than the \
                         collection time suggests. Raise the per-node budget \
                         (crown_ibp_per_node_time_budget / alpha_crown.crown_ibp_intermediates) \
                         or the Conv2d dense cap (NY_CROWN_MEM_CAP_MB) to recover tightness. \
                         Rate-limited quality-cliff occurrence #{occurrence}; {suppressed} \
                         intervening occurrences were suppressed before this checkpoint \
                         (first and powers of two are retained)."
                        );
                    } else {
                        info!(
                        "CROWN-IBP DAG provenance: {crown_kept}/{demanded_total} demanded targets \
                         kept CROWN authority, {degraded} remained fallback-grade ({detail}); \
                         rate-limited occurrence #{occurrence}, with {suppressed} intervening \
                         occurrences suppressed before this checkpoint (first and powers of two \
                         are retained)"
                    );
                    }
                }
            }
        }
        // #crown-cut-segment: one-line sweep summary whenever the gate is on.
        if let Some(ctx) = cut_ctx.as_ref() {
            info!(
                "CROWN-IBP DAG cut-segment sweep: NY_CROWN_CUT_SEGMENT={cut_segment}, \
                 {crown_node_count} nodes swept, {} cuts used, {collection_secs:.3}s wall",
                ctx.cuts_used(),
            );
        }
        // #crown-repropagate (dark, `NY_CROWN_REPROPAGATE=1`, default OFF =>
        // byte-identical): push the tightening FORWARD so it compounds.
        //
        // WHY. `ibp_bounds` is the one-shot forward map computed BEFORE the loop
        // and is never written inside it; tightened results land in the separate
        // `crown_ibp_bounds`. A demand-skipped node therefore stores the
        // PRE-tightening value, so every CROWN result is discarded for everything
        // downstream of it that is not itself demanded.
        //
        // The sweep, its certified-transfer allowlist and its soundness argument
        // live in `crown_repropagate`; it is factored out so the admission rules
        // are unit-testable without a full collection.
        if crown_repropagate::enabled() {
            crown_repropagate::sweep(
                self,
                input,
                exec_order,
                engine,
                deadline,
                crown_repropagate::Options::from_env(),
                &mut crown_ibp_bounds,
            );
        }

        // #conv-patches-collect diagnostic (default-OFF): dump per-node provenance
        // for the spatial (3D) conv-graph nodes so a metaroom/cifar100 probe can
        // see exactly which deep conv targets tightened (Crown) vs fell back (and
        // why). stderr println so it survives the vnncomp log filter.
        if std::env::var_os("NY_CONV_PATCHES_DEBUG").is_some_and(|v| v != "0" && !v.is_empty()) {
            for node_name in exec_order.iter() {
                let Some(b) = crown_ibp_bounds.get(node_name) else {
                    continue;
                };
                if b.shape().len() != 3 {
                    continue;
                }
                let prov = provenance.get(node_name);
                eprintln!(
                    "[conv-patches-dbg] node={node_name} shape={:?} numel={} width={:.4} prov={:?}",
                    b.shape(),
                    b.len(),
                    b.max_width(),
                    prov,
                );
            }
        }

        Ok(GraphCrownIbpBoundsResult {
            bounds: crown_ibp_bounds,
            provenance,
            fallback_events,
        })
    }
}

/// Deterministic intersection for the two sorted row-index sets used by the
/// objective influence cone and sparse ReLU selection.
fn intersect_sorted_rows(left: &[usize], right: &[usize]) -> Vec<usize> {
    let (mut i, mut j) = (0usize, 0usize);
    let mut out = Vec::with_capacity(left.len().min(right.len()));
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                if out.last().copied() != Some(left[i]) {
                    out.push(left[i]);
                }
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Resolve the collector's target-row seed while keeping the typed atomic
/// strategy independent of objective-specific subset publications.
fn compose_target_subset_indices(
    existing: Option<std::sync::Arc<[usize]>>,
    sparse_rows: Option<&[usize]>,
    atomic_target: bool,
) -> Option<std::sync::Arc<[usize]>> {
    if atomic_target {
        return sparse_rows.map(|rows| std::sync::Arc::from(rows.to_vec()));
    }
    match (existing, sparse_rows) {
        (Some(existing), Some(rows)) => {
            Some(std::sync::Arc::from(intersect_sorted_rows(&existing, rows)))
        }
        (None, Some(rows)) => Some(std::sync::Arc::from(rows.to_vec())),
        (existing, None) => existing,
    }
}

/// Sparse Patches identity seeds are virtual. A non-spatial subset, however,
/// materializes two dense `[selected_rows, target_dim]` f32 matrices before
/// the backward walk. Refuse that path before allocation when it exceeds the
/// same adaptive Dense budget as the full collector; the caller then uses the
/// existing objective-chunked full-width path.
fn subset_seed_fits_dense_budget(
    patches_seed: bool,
    selected_rows: usize,
    target_dim: usize,
    budget_bytes: usize,
) -> bool {
    patches_seed
        || crate::network::crown_memory::dense_pair_bytes(selected_rows, target_dim)
            .is_some_and(|required| required <= budget_bytes)
}

/// `NY_SUBSET_PATCHES_GUARD=0` restores the pre-#subset-forfeits-patches
/// routing byte-for-byte (a patches-eligible target may again be hijacked into
/// a dense k-row subset seed of any size). Default-ON.
const SUBSET_PATCHES_GUARD_ENV: &str = "NY_SUBSET_PATCHES_GUARD";

fn subset_patches_guard_enabled() -> bool {
    match std::env::var_os(SUBSET_PATCHES_GUARD_ENV) {
        Some(value) => value != "0" && !value.is_empty(),
        None => true,
    }
}

/// #subset-forfeits-patches: would taking this row subset DESTROY the
/// memory-light patches backward the target would otherwise get?
///
/// `subset_seed_fits_dense_budget` short-circuits on `patches_seed`, on the
/// assumption that a patches-eligible target's subset seed is itself virtual
/// (free). That assumption holds for exactly ONE shape of subset:
/// [`super::target_backward::build_subset_seed`] emits
/// `PatchesLinearBounds::sparse_identity` only when the rows are the COMPLETE
/// output grid in flat order. For a STRICT subset it emits a dense
/// `[k x target_dim]` selection matrix instead — and the walk is then dense for
/// its whole length, because the 7-D `try_dense_spatial_patches_reentry` lift
/// costs `k * out_dim * in_c` (a further `in_c`x over the dense pair it is
/// meant to replace) and is refused.
///
/// MEASURED on yolo_2023 / TinyYOLO_prop_000005 (2026-08-07, this branch's
/// baseline), where the two routes ran side by side inside one collection:
///
/// - `Conv_1`, cone 40,000/43,264 rows: the dense seed (13.84 GB) EXCEEDED the
///   Dense budget, the subset was refused, the target took the full-width
///   patches backward and COMPLETED in 6.120 s — the only one of 8 demanded
///   targets to keep CROWN authority.
/// - `Conv_5`, cone 33,856/43,264 rows: the dense seed (11.72 GB) FIT under the
///   12.88 GB budget, so it was taken; the 7-D re-entry then asked for
///   187,487,485,952 B and was declined, and the dense walk did not finish a
///   single backward step in 28.465 s.
/// - `Conv_12` (6,400/10,816) and `Conv_20` (1,568/5,408) took the same dense
///   route and also exceeded their per-node deadline (22.509 s / 13.853 s),
///   after which `Add_8`, `Add_15`, `Conv_25` and `Flatten_30` were skipped as
///   hopeless-class because the collection budget was gone.
///
/// Net: 7/8 demanded targets fell back to IBP, and the ONLY difference between
/// the target that completed and the ones that did not was whether the dense
/// subset seed happened to fit the memory budget. Fitting a multi-GiB seed is
/// not a reason to prefer it.
///
/// So refuse the subset when all of:
///   * the full-width fallback WOULD start in patches (`is_patches_target`), so
///     refusing costs nothing but the extra rows — the seed is virtual;
///   * the subset is STRICT, i.e. `build_subset_seed` will materialize dense;
///   * that dense pair is not cheap, judged by the SAME crossover the patches
///     cost-admission already uses ([`super::patches_target::patches_cost_admission_threshold_bytes`],
///     128 MiB at the compiled default).
///
/// SOUND either way, and not a tightness trade in the risky direction: the
/// full-width backward computes the SAME per-row relaxation for a SUPERSET of
/// the subset's rows (row-independence is the premise the subset seed itself
/// rests on — see the `#margin-subset-seed` comment in the collection loop).
/// Rows the cone would have dropped keep CROWN instead of IBP, so the collected
/// map is tighter-or-equal, never tighter-than-true.
fn subset_forfeits_patches_route(
    is_patches_target: bool,
    selected_rows: usize,
    target_dim: usize,
) -> bool {
    if !is_patches_target || selected_rows >= target_dim {
        return false;
    }
    let threshold = super::patches_target::patches_cost_admission_threshold_bytes();
    crate::network::crown_memory::dense_pair_bytes(selected_rows, target_dim)
        .is_none_or(|required| required > threshold)
}

#[cfg(test)]
mod prefix_cost_work_tests {
    use super::{
        dense_crown_prefix_work_units, prefix_cost_instrumentation_active,
        prefix_cost_remaining_window,
    };
    use crate::layers::{Conv2dLayer, FlattenLayer, Layer, LinearLayer, ReLULayer, SigmoidLayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use crate::network::graph_alpha::bounds::target_backward::PartialCrownDeadlineSalvagePolicy;
    use ndarray::{Array1, Array2, ArrayD, IxDyn};
    use std::time::{Duration, Instant};

    fn resnet_subset_chain() -> GraphNetwork {
        let kernel = ArrayD::from_elem(IxDyn(&[2, 1, 3, 3]), 0.25f32);
        let conv =
            Conv2dLayer::with_input_shape(kernel, Some(Array1::zeros(2)), (1, 1), (1, 1), 5, 5)
                .expect("valid conv");
        let linear =
            LinearLayer::new(Array2::zeros((3, 50)), Some(Array1::zeros(3))).expect("valid linear");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["conv".into()],
        ));
        graph.add_node(GraphNode::new(
            "flatten",
            Layer::Flatten(FlattenLayer::flatten_all()),
            vec!["relu".into()],
        ));
        graph.add_node(GraphNode::new(
            "fc",
            Layer::Linear(linear),
            vec!["flatten".into()],
        ));
        graph.set_output("fc");
        graph
    }

    #[test]
    fn dense_prefix_work_is_exact_checked_and_scales_by_objective_rows() {
        let graph = resnet_subset_chain();
        // Conv: 18 kernel coefficients * 5 * 5 positions = 450 MAC/row.
        // Linear: 3 * 50 = 150 MAC/row. Lower/upper sign split = x4.
        assert_eq!(dense_crown_prefix_work_units(&graph, "fc", 1), Some(2_400));
        assert_eq!(dense_crown_prefix_work_units(&graph, "fc", 7), Some(16_800));
        assert_eq!(dense_crown_prefix_work_units(&graph, "fc", 0), None);
    }

    #[test]
    fn dense_prefix_work_fails_open_outside_authenticated_resnet_subset() {
        let mut graph = resnet_subset_chain();
        graph.add_node(GraphNode::new(
            "sigmoid",
            Layer::Sigmoid(SigmoidLayer),
            vec!["fc".into()],
        ));
        graph.set_output("sigmoid");
        assert_eq!(
            dense_crown_prefix_work_units(&graph, "sigmoid", 1),
            None,
            "an unsupported ancestor must decline the estimate rather than invent cost authority"
        );
        assert_eq!(dense_crown_prefix_work_units(&graph, "missing", 1), None);
    }

    #[test]
    fn prefix_window_requires_outer_deadline_and_has_exact_zero_boundary() {
        let now = Instant::now();
        let target = now + Duration::from_secs(7);
        assert_eq!(
            prefix_cost_remaining_window(None, Some(target), now),
            None,
            "an internal Patches guard must not change no-deadline behavior"
        );
        assert_eq!(prefix_cost_remaining_window(Some(target), None, now), None);
        assert_eq!(
            prefix_cost_remaining_window(Some(target), Some(target), now),
            Some(Duration::from_secs(7))
        );
        assert_eq!(
            prefix_cost_remaining_window(Some(now), Some(now), now),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn prefix_instrumentation_never_preempts_partial_deadline_salvage() {
        let deadline = Some(Instant::now() + Duration::from_secs(1));
        assert!(!prefix_cost_instrumentation_active(
            false,
            deadline,
            PartialCrownDeadlineSalvagePolicy::Disabled,
        ));
        assert!(!prefix_cost_instrumentation_active(
            true,
            None,
            PartialCrownDeadlineSalvagePolicy::Disabled,
        ));
        assert!(prefix_cost_instrumentation_active(
            true,
            deadline,
            PartialCrownDeadlineSalvagePolicy::Disabled,
        ));
        assert!(!prefix_cost_instrumentation_active(
            true,
            deadline,
            PartialCrownDeadlineSalvagePolicy::EnabledByExactEnvironment,
        ));
    }
}

#[cfg(test)]
mod partial_crown_row_intersection_tests {
    use super::{
        intersect_completed_crown_target, retain_partial_crown_rows, CrownIbpCollectionMode,
    };
    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    fn bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn())
            .expect("ordered finite bounds")
    }

    #[test]
    fn partial_rows_are_shrink_only_and_keep_ibp_unfinished_rows() {
        let ibp = bounds(&[-4.0, -3.0, -2.0], &[4.0, 3.0, 2.0]);
        let partial = bounds(&[-1.0, -3.0, -2.0], &[1.0, 3.0, 2.0]);
        let (retained, disjoint) = retain_partial_crown_rows(&ibp, Some(&partial));
        assert_eq!(disjoint, 0);
        assert_eq!(
            retained.lower().as_slice().expect("contiguous"),
            &[-1.0, -3.0, -2.0]
        );
        assert_eq!(
            retained.upper().as_slice().expect("contiguous"),
            &[1.0, 3.0, 2.0]
        );
        for ((&lower, &upper), (&ibp_lower, &ibp_upper)) in retained
            .lower()
            .iter()
            .zip(retained.upper())
            .zip(ibp.lower().iter().zip(ibp.upper()))
        {
            assert!(lower >= ibp_lower);
            assert!(upper <= ibp_upper);
        }
    }

    #[test]
    fn disjoint_partial_candidate_reverts_to_exact_ibp() {
        let ibp = bounds(&[-1.0, -1.0], &[1.0, 1.0]);
        let disjoint = bounds(&[5.0, -0.5], &[6.0, 0.5]);
        let (retained, disjoint_rows) = retain_partial_crown_rows(&ibp, Some(&disjoint));
        assert_eq!(disjoint_rows, 1);
        assert_eq!(retained.lower(), ibp.lower());
        assert_eq!(retained.upper(), ibp.upper());
    }

    #[test]
    fn complete_cgan_disjoint_candidate_is_rejected_instead_of_widening_baseline() {
        let baseline = bounds(&[-1.0, -1.0], &[1.0, 1.0]);
        let disjoint = bounds(&[5.0, -0.5], &[6.0, 0.5]);

        assert!(
            intersect_completed_crown_target(
                &baseline,
                &disjoint,
                CrownIbpCollectionMode::CganComplete,
            )
            .is_none(),
            "the complete typed transaction must retain its certified baseline"
        );

        let (generic, disjoint_rows) = intersect_completed_crown_target(
            &baseline,
            &disjoint,
            CrownIbpCollectionMode::Standard,
        )
        .expect("the generic collector keeps its diagnostic-union contract");
        assert_eq!(disjoint_rows, 1);
        assert_eq!(
            generic.upper().as_slice().expect("contiguous"),
            &[6.0, 0.5],
            "the control must force the historical widening branch"
        );
    }
}

#[cfg(test)]
mod downstream_resweep_tests {
    use super::{
        crown_ibp_downstream_resweep_from_raw, hopeless_class_skip_from_raw,
        CrownIbpCollectionMode, PartialCrownDeadlineSalvagePolicy,
    };
    use crate::layers::{AddLayer, Layer, LinearLayer, ReLULayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
    use ndarray::{arr1, arr2};
    use ny_core::NaiveCpuGemmEngine;
    use ny_tensor::BoundedTensor;

    #[test]
    fn hopeless_class_skip_parser_is_pure_and_exact() {
        assert!(hopeless_class_skip_from_raw(None));
        assert!(!hopeless_class_skip_from_raw(Some("1")));
        for raw in ["", "0", "true", "01"] {
            assert!(hopeless_class_skip_from_raw(Some(raw)), "raw={raw:?}");
        }
    }

    fn cancellation_dag() -> (GraphNetwork, BoundedTensor) {
        let positive = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("positive branch");
        let negative = LinearLayer::new(arr2(&[[-1.0_f32]]), None).expect("negative branch");
        let output = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("output");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("positive", Layer::Linear(positive)));
        graph.add_node(GraphNode::from_input("negative", Layer::Linear(negative)));
        graph.add_node(GraphNode::binary(
            "cancel",
            Layer::Add(AddLayer),
            "positive",
            "negative",
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["cancel".to_owned()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(output),
            vec!["relu".to_owned()],
        ));
        graph.set_output("out");

        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("unit box");
        (graph, input)
    }

    fn cancellation_linear_chain() -> (GraphNetwork, BoundedTensor) {
        // A unary chain that still contains correlation loss: the first layer
        // expands one scalar to [x, -x], and the second sums those coordinates.
        // CROWN proves the sum is zero while one-shot IBP gives [-2, 2].
        let expand = LinearLayer::new(arr2(&[[1.0_f32], [-1.0_f32]]), None).expect("expand branch");
        let cancel = LinearLayer::new(arr2(&[[1.0_f32, 1.0_f32]]), None).expect("cancel branch");
        let output = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("output");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("expand", Layer::Linear(expand)));
        graph.add_node(GraphNode::new(
            "cancel",
            Layer::Linear(cancel),
            vec!["expand".to_owned()],
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["cancel".to_owned()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(output),
            vec!["relu".to_owned()],
        ));
        graph.set_output("out");

        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("unit box");
        (graph, input)
    }

    #[test]
    fn gate_parser_is_exact_and_default_dark() {
        for raw in [None, Some(""), Some("0"), Some("true"), Some("01")] {
            assert!(!crown_ibp_downstream_resweep_from_raw(raw));
        }
        assert!(crown_ibp_downstream_resweep_from_raw(Some("1")));
    }

    /// The CROWN target `cancel` proves the two correlated branches sum to
    /// (approximately) zero. Historically the demand-skipped ReLU discarded
    /// that result and reused its one-shot IBP range `[0, 2]`; the resweep must
    /// compose the tightened predecessor through the ReLU.
    #[test]
    fn tightened_correlation_reaches_demand_skipped_descendant() {
        let (graph, input) = cancellation_dag();
        let ibp = graph.collect_node_bounds(&input).expect("IBP map");

        let historical = graph
            .collect_crown_ibp_bounds_core_inner_with_cut_segment_and_policies(
                &input,
                ibp.clone(),
                None,
                None,
                None,
                0,
                false,
                PartialCrownDeadlineSalvagePolicy::Disabled,
                CrownIbpCollectionMode::Standard,
            )
            .expect("historical collection");
        let reswept = graph
            .collect_crown_ibp_bounds_core_inner_with_cut_segment_and_policies(
                &input,
                ibp,
                None,
                None,
                None,
                0,
                true,
                PartialCrownDeadlineSalvagePolicy::Disabled,
                CrownIbpCollectionMode::Standard,
            )
            .expect("reswept collection");

        assert!(matches!(
            reswept.provenance.get("cancel"),
            Some(BoundsProvenance::Crown)
        ));
        assert!(matches!(
            reswept.provenance.get("relu"),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::DemandDrivenSkip
            ))
        ));

        let old_relu = historical.bounds.get("relu").expect("historical ReLU");
        let new_relu = reswept.bounds.get("relu").expect("reswept ReLU");
        assert!(
            new_relu.upper()[[0]] < old_relu.upper()[[0]] * 0.01,
            "the correlated cancellation must compound through the skipped ReLU: old={:?}, new={:?}",
            (old_relu.lower()[[0]], old_relu.upper()[[0]]),
            (new_relu.lower()[[0]], new_relu.upper()[[0]]),
        );
        assert!(
            new_relu.lower()[[0]] <= 0.0 && new_relu.upper()[[0]] >= 0.0,
            "the shrink-only result must still enclose the true ReLU value 0"
        );
    }

    #[test]
    fn enabled_resweep_bypasses_sequential_engine_route() {
        let (graph, input) = cancellation_linear_chain();
        let ibp = graph.collect_node_bounds(&input).expect("IBP map");

        let result = graph
            .collect_crown_ibp_bounds_core_inner_with_cut_segment_and_policies(
                &input,
                ibp,
                None,
                Some(&NaiveCpuGemmEngine),
                None,
                0,
                true,
                PartialCrownDeadlineSalvagePolicy::Disabled,
                CrownIbpCollectionMode::Standard,
            )
            .expect("engine-backed resweep collection");

        assert!(matches!(
            result.provenance.get("cancel"),
            Some(BoundsProvenance::Crown)
        ));
        assert!(matches!(
            result.provenance.get("relu"),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::DemandDrivenSkip
            ))
        ));
        let relu = result.bounds.get("relu").expect("reswept ReLU");
        assert!(
            relu.lower()[[0]] <= 0.0 && relu.upper()[[0]] >= 0.0,
            "the reswept bound must enclose the true ReLU value"
        );
        assert!(
            relu.upper()[[0]] < 0.02,
            "the engine-present route must not restore the one-shot [0, 2] IBP box: {relu:?}"
        );
    }

    #[test]
    fn cgan_atomic_dense_fallback_resweeps_to_the_dense_oracle() {
        let (graph, input) = cancellation_dag();
        let baseline = graph.collect_node_bounds(&input).expect("baseline");
        let dense = graph
            .collect_crown_ibp_bounds_core_inner_with_cut_segment_and_policies(
                &input,
                baseline.clone(),
                None,
                None,
                None,
                0,
                false,
                PartialCrownDeadlineSalvagePolicy::Disabled,
                CrownIbpCollectionMode::Standard,
            )
            .expect("dense oracle");
        let atomic = graph
            .collect_crown_ibp_bounds_core_inner_with_cut_segment_and_policies(
                &input,
                baseline,
                None,
                None,
                None,
                0,
                false,
                PartialCrownDeadlineSalvagePolicy::Disabled,
                CrownIbpCollectionMode::CganSparseTargetComplete,
            )
            .expect("atomic target-complete collection");

        // `cancel` has one unresolved row, so 1/1 is above the sparse 90%
        // threshold and the selected target must take the dense fallback.
        assert!(matches!(
            atomic.provenance.get("cancel"),
            Some(BoundsProvenance::Crown)
        ));
        assert!(matches!(
            atomic.provenance.get("out"),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::DemandDrivenSkip
            ))
        ));
        let dense_out = dense.bounds.get("out").expect("dense output");
        let atomic_out = atomic.bounds.get("out").expect("atomic output");
        for (&actual, &oracle) in atomic_out.lower().iter().zip(dense_out.lower()) {
            assert!((actual - oracle).abs() <= 1e-12);
        }
        for (&actual, &oracle) in atomic_out.upper().iter().zip(dense_out.upper()) {
            assert!((actual - oracle).abs() <= 1e-12);
        }
        assert_eq!(
            atomic_out.upper()[[0]] < 0.02,
            dense_out.upper()[[0]] < 0.02,
            "the root threshold verdict must agree with the dense oracle"
        );
        assert!(
            atomic_out.upper()[[0]] < 0.02,
            "the completed target must be reswept through ReLU/output before the root retest"
        );
    }
}

// #margin-subset-seed: `scatter_margin_rows_over_bounds` moved to
// `crate::output_margin_seed` so the root CROWN backward and the DAG alpha
// per-iteration backward share the exact same scatter (#margin-subset-alpha).
use crate::output_margin_seed::scatter_margin_rows_over_bounds;

#[cfg(test)]
mod sparse_relu_row_helpers_tests {
    use super::{
        collector_budget_route, collector_budget_rows, compose_target_subset_indices,
        intersect_sorted_rows, is_structural_crown_skip, sequential_fast_path_allowed,
        subset_forfeits_patches_route, subset_seed_fits_dense_budget, CollectorBudgetRoute,
    };
    use crate::types::CrownIbpFallbackReason;
    use std::sync::Arc;

    /// #subset-forfeits-patches. The three yolo_2023 / TinyYOLO targets whose
    /// strict cone subsets densified the whole backward walk in the 2026-08-07
    /// baseline must now be refused, and the one whose subset was already
    /// refused on memory grounds must stay refused. A NON-patches target and a
    /// full in-order grid (which seeds virtual sparse-identity Patches) must be
    /// untouched, and so must a genuinely cheap strict subset.
    #[test]
    fn strict_subset_of_a_patches_target_is_refused_when_its_dense_seed_is_not_cheap() {
        // Measured cone sizes; dense pair = rows * dim * 8 bytes.
        // Conv_5: 33_856/43_264 -> 11.72 GB.  Conv_12: 6_400/10_816 -> 553 MB.
        // Conv_20: 1_568/5_408 -> 67.8 MB (under the 128 MiB crossover).
        assert!(subset_forfeits_patches_route(true, 33_856, 43_264));
        assert!(subset_forfeits_patches_route(true, 6_400, 10_816));
        assert!(subset_forfeits_patches_route(true, 40_000, 43_264));
        assert!(
            !subset_forfeits_patches_route(true, 1_568, 5_408),
            "a cheap strict subset keeps the cone: 67.8 MB is under the 128 MiB crossover"
        );
        assert!(
            !subset_forfeits_patches_route(false, 33_856, 43_264),
            "a target that could not start in patches has no patches route to forfeit"
        );
        assert!(
            !subset_forfeits_patches_route(true, 43_264, 43_264),
            "the full in-order grid seeds a virtual sparse identity, not a dense pair"
        );
        assert!(
            !subset_forfeits_patches_route(true, usize::MAX, usize::MAX),
            "selected >= target_dim is never a strict subset, overflow or not"
        );
    }

    #[test]
    fn sparse_relu_rows_intersect_objective_cone_in_sorted_order() {
        assert_eq!(
            intersect_sorted_rows(&[1, 3, 4, 8, 11], &[0, 3, 8, 9, 11]),
            vec![3, 8, 11]
        );
        assert!(intersect_sorted_rows(&[1, 2], &[3, 4]).is_empty());
    }

    #[test]
    fn atomic_target_ignores_objective_subset_and_keeps_dense_fallback_dense() {
        let objective_subset: Arc<[usize]> = Arc::from(vec![3, 8]);
        let selected = compose_target_subset_indices(
            Some(Arc::clone(&objective_subset)),
            Some(&[1, 3, 8, 11]),
            true,
        )
        .expect("atomic sparse selection");
        assert_eq!(
            selected.as_ref(),
            &[1, 3, 8, 11],
            "the objective cone must not trim selected unresolved ReLU rows"
        );
        assert!(
            compose_target_subset_indices(Some(objective_subset), None, true).is_none(),
            "a rejected >90%/over-budget sparse plan must stay on dense fallback"
        );
    }

    #[test]
    fn sparse_dense_subset_refuses_over_budget_and_overflow_before_allocation() {
        assert!(subset_seed_fits_dense_budget(false, 2, 4, 64));
        assert!(!subset_seed_fits_dense_budget(false, 3, 4, 64));
        assert!(!subset_seed_fits_dense_budget(
            false,
            usize::MAX,
            usize::MAX,
            usize::MAX
        ));
        assert!(
            subset_seed_fits_dense_budget(true, usize::MAX, usize::MAX, 0),
            "a Patches sparse identity is virtual and does not allocate the dense pair"
        );
    }

    #[test]
    fn known_subset_rejection_budgets_full_width_execution() {
        let raw_rows = 28_800;
        let selected_rows = 9_000;
        let accepted = subset_seed_fits_dense_budget(false, selected_rows, raw_rows, 1_024);
        assert!(!accepted, "fixture must reject the dense subset seed");
        let route = collector_budget_route(accepted, false);
        assert_eq!(route, CollectorBudgetRoute::FullWidth);
        assert_eq!(
            collector_budget_rows(raw_rows, selected_rows, route),
            raw_rows
        );
    }

    #[test]
    fn runtime_subset_failure_reweights_fail_open_to_full_width() {
        let raw_rows = 28_800;
        let selected_rows = 9_000;
        let planned = collector_budget_route(true, true);
        assert_eq!(planned, CollectorBudgetRoute::PlannedSubset);
        assert_eq!(
            collector_budget_rows(raw_rows, selected_rows, planned),
            selected_rows
        );

        let simulated_runtime_attempt: Result<(), &str> = Err("subset backward failed");
        let fail_open = collector_budget_route(true, simulated_runtime_attempt.is_ok());
        assert_eq!(fail_open, CollectorBudgetRoute::RuntimeSubsetFailOpen);
        assert_eq!(
            collector_budget_rows(raw_rows, selected_rows, fail_open),
            raw_rows,
            "runtime fail-open must replace selected_len with full target width"
        );
    }

    #[test]
    fn exact_gate_without_an_engaged_sparse_plan_preserves_sequential_route() {
        assert!(sequential_fast_path_allowed(
            false, true, false, false, false, false
        ));
        assert!(
            sequential_fast_path_allowed(false, false, true, false, false, false),
            "a structurally typed Linear/ReLU CPU chain may use the sequential collector"
        );
        assert!(
            !sequential_fast_path_allowed(true, true, false, false, false, false),
            "an engaged sparse plan must use the graph-native subset collector"
        );
        assert!(!sequential_fast_path_allowed(
            false, false, false, false, false, false
        ));
        assert!(!sequential_fast_path_allowed(
            false, true, false, true, false, false
        ));
        assert!(!sequential_fast_path_allowed(
            false, true, false, false, true, false
        ));
        assert!(
            !sequential_fast_path_allowed(false, true, false, false, false, true),
            "the graph-native route owns downstream composition"
        );
        assert!(
            !sequential_fast_path_allowed(true, false, true, false, false, false),
            "the CPU-chain admission must not bypass the sparse-plan guard"
        );
    }

    #[test]
    fn sparse_row_omission_reasons_are_structural_not_degradations() {
        assert!(is_structural_crown_skip(
            CrownIbpFallbackReason::StableReluRowsSkipped
        ));
        assert!(is_structural_crown_skip(
            CrownIbpFallbackReason::ObjectiveConeRowsSkipped
        ));
        assert!(!is_structural_crown_skip(
            CrownIbpFallbackReason::DeadlineExceeded
        ));
    }
}

#[cfg(test)]
mod sparse_relu_row_collector_tests {
    use super::super::budget_policy::CROWN_CHUNK_AWARE_BUDGET_ENV;
    use super::super::demand::SPARSE_RELU_ROWS_ENV;
    use super::{
        run_with_m1_collector_test_controls, CrownIbpCollectionMode, M1CollectorTestControls,
        M1CollectorTraceEvent, PartialCrownDeadlineSalvagePolicy,
    };
    use crate::layers::{Conv2dLayer, ConvTranspose2dLayer, Layer, LinearLayer, ReLULayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use crate::output_margin_seed::MarginOutputSeedGuard;
    use crate::types::{BoundsProvenance, CrownIbpFallbackReason, GraphCrownIbpBoundsResult};
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;
    use std::time::{Duration, Instant};

    fn mixed_stability_net() -> (GraphNetwork, BoundedTensor) {
        let pre = LinearLayer::new(arr2(&[[1.0_f32], [-1.0]]), None).expect("pre");
        let producer = LinearLayer::new(
            arr2(&[[1.0_f32, -1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            Some(arr1(&[0.0_f32, -0.5, 1.0, -3.0])),
        )
        .expect("producer");
        let out = LinearLayer::new(
            arr2(&[[1.0_f32, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]),
            None,
        )
        .expect("out");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("pre", Layer::Linear(pre)));
        graph.add_node(GraphNode::new(
            "act1",
            Layer::ReLU(ReLULayer),
            vec!["pre".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "producer",
            Layer::Linear(producer),
            vec!["act1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "act2",
            Layer::ReLU(ReLULayer),
            vec!["producer".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(out),
            vec!["act2".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input");
        (graph, input)
    }

    fn collect(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        ibp: std::collections::HashMap<String, BoundedTensor>,
    ) -> GraphCrownIbpBoundsResult {
        collect_with_deadline(graph, input, ibp, None)
    }

    fn collect_with_deadline(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        ibp: std::collections::HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
    ) -> GraphCrownIbpBoundsResult {
        graph
            .collect_crown_ibp_bounds_core_inner_with_cut_segment(
                input, ibp, deadline, None, None, 0,
            )
            .expect("CROWN-IBP collection")
    }

    /// Exercise the M1 planner/executor contract without granting the deadline
    /// hard proof authority. Hard finite objective-chunk setup deliberately
    /// typed-closes before wave dispatch until its whole-target setup is fully
    /// cooperative; that boundary has dedicated refusal regressions in
    /// `target_backward`. This seam keeps the retained-plan and live-route
    /// drift checks non-vacuous on the legacy bounded scheduler.
    fn collect_with_soft_scheduling_deadline(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        ibp: std::collections::HashMap<String, BoundedTensor>,
        deadline: Instant,
    ) -> GraphCrownIbpBoundsResult {
        graph
            .collect_crown_ibp_bounds_core_inner_with_mode(
                input,
                ibp,
                Some(deadline),
                false,
                None,
                None,
                0,
                crate::output_margin_seed::published_margin_indices(),
                false,
                PartialCrownDeadlineSalvagePolicy::Disabled,
                CrownIbpCollectionMode::Standard,
                // #cgan-narrow-target-budget: no ceiling on this soft-deadline path.
                None,
                false,
                graph.crown_ibp_runtime_policy(),
            )
            .expect("soft-deadline CROWN-IBP collection")
    }

    fn collect_atomic(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        baseline: std::collections::HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
        salvage: PartialCrownDeadlineSalvagePolicy,
    ) -> GraphCrownIbpBoundsResult {
        graph
            .collect_crown_ibp_bounds_core_inner_with_cut_segment_and_policies(
                input,
                baseline,
                deadline,
                None,
                None,
                0,
                false,
                salvage,
                CrownIbpCollectionMode::CganSparseTargetComplete,
            )
            .expect("atomic target-complete collection")
    }

    #[test]
    fn atomic_sparse_target_matches_dense_rows_and_preserves_stable_baseline_rows() {
        let _env_lock = ny_test_utils::env::lock_env();
        let _sparse_gate = ny_test_utils::env::ScopedEnvVar::unset(SPARSE_RELU_ROWS_ENV);
        let (graph, input) = mixed_stability_net();
        let baseline = graph.collect_node_bounds(&input).expect("baseline");
        let dense = collect(&graph, &input, baseline.clone());
        let atomic = collect_atomic(
            &graph,
            &input,
            baseline.clone(),
            None,
            PartialCrownDeadlineSalvagePolicy::Disabled,
        );

        let dense_target = dense.bounds.get("producer").expect("dense target");
        let atomic_target = atomic.bounds.get("producer").expect("atomic target");
        let baseline_target = baseline.get("producer").expect("baseline target");
        for row in [0usize, 1] {
            assert_eq!(atomic_target.lower()[[row]], dense_target.lower()[[row]]);
            assert_eq!(atomic_target.upper()[[row]], dense_target.upper()[[row]]);
        }
        for row in [2usize, 3] {
            assert_eq!(atomic_target.lower()[[row]], baseline_target.lower()[[row]]);
            assert_eq!(atomic_target.upper()[[row]], baseline_target.upper()[[row]]);
        }
        assert!(matches!(
            atomic.provenance.get("producer"),
            Some(BoundsProvenance::Crown)
        ));
        assert!(
            atomic
                .provenance
                .iter()
                .filter(|(_, provenance)| matches!(provenance, BoundsProvenance::Crown))
                .count()
                <= 1,
            "the typed policy may publish at most its one selected target"
        );
    }

    #[test]
    fn atomic_incomplete_target_discards_every_row_even_if_salvage_is_requested() {
        let _env_lock = ny_test_utils::env::lock_env();
        let _sparse_gate = ny_test_utils::env::ScopedEnvVar::unset(SPARSE_RELU_ROWS_ENV);
        let (graph, input) = mixed_stability_net();
        let baseline = graph.collect_node_bounds(&input).expect("baseline");
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("past instant");
        let atomic = collect_atomic(
            &graph,
            &input,
            baseline.clone(),
            Some(expired),
            PartialCrownDeadlineSalvagePolicy::EnabledByExactEnvironment,
        );

        for (name, expected) in &baseline {
            let actual = atomic
                .bounds
                .get(name)
                .unwrap_or_else(|| panic!("missing baseline node '{name}'"));
            assert_eq!(actual.lower(), expected.lower(), "node '{name}' lower");
            assert_eq!(actual.upper(), expected.upper(), "node '{name}' upper");
        }
        assert!(matches!(
            atomic.provenance.get("producer"),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::DeadlineExceeded
                    | CrownIbpFallbackReason::PerNodeDeadlineExceeded
            ))
        ));
        assert!(atomic.fallback_events.iter().all(|event| {
            event.reason != CrownIbpFallbackReason::PartialCrownRowsDeadlineExceeded
        }));
    }

    #[test]
    fn atomic_policy_without_an_unresolved_target_returns_the_exact_baseline() {
        let _env_lock = ny_test_utils::env::lock_env();
        let _sparse_gate = ny_test_utils::env::ScopedEnvVar::unset(SPARSE_RELU_ROWS_ENV);
        let producer =
            LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[2.0_f32]))).expect("producer");
        let output = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("output");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("producer", Layer::Linear(producer)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["producer".to_owned()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(output),
            vec!["relu".to_owned()],
        ));
        graph.set_output("out");
        let fixed_input =
            BoundedTensor::new(arr1(&[-0.1_f32]).into_dyn(), arr1(&[0.1_f32]).into_dyn())
                .expect("stable input");
        let baseline = graph
            .collect_node_bounds(&fixed_input)
            .expect("fixed baseline");
        let atomic = collect_atomic(
            &graph,
            &fixed_input,
            baseline.clone(),
            None,
            PartialCrownDeadlineSalvagePolicy::Disabled,
        );

        for (name, expected) in &baseline {
            let actual = atomic
                .bounds
                .get(name)
                .unwrap_or_else(|| panic!("missing baseline node '{name}'"));
            assert_eq!(actual.lower(), expected.lower(), "node '{name}' lower");
            assert_eq!(actual.upper(), expected.upper(), "node '{name}' upper");
        }
        assert!(
            atomic
                .provenance
                .values()
                .all(|provenance| !matches!(provenance, BoundsProvenance::Crown)),
            "a target-complete policy with no selected target must not widen to ordinary CROWN"
        );
    }

    /// `conv1` has exactly one unstable ReLU row (flat row 0), while the
    /// published output objective can only influence flat row 511. The real
    /// sparse-plan/cone intersection is therefore empty.
    fn empty_sparse_cone_net() -> (GraphNetwork, BoundedTensor) {
        let identity = ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0_f32);
        let conv1 = Conv2dLayer::with_input_shape(identity.clone(), None, (1, 1), (0, 0), 1, 512)
            .expect("conv1");
        let conv2 =
            Conv2dLayer::with_input_shape(identity, None, (1, 1), (0, 0), 1, 512).expect("conv2");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["conv1".into()],
        ));
        graph.add_node(GraphNode::new(
            "conv2",
            Layer::Conv2d(conv2),
            vec!["relu".into()],
        ));
        graph.set_output("conv2");
        let mut lower = ArrayD::from_elem(IxDyn(&[1, 1, 512]), 1.0_f32);
        let mut upper = ArrayD::from_elem(IxDyn(&[1, 1, 512]), 2.0_f32);
        lower[[0, 0, 0]] = -1.0;
        upper[[0, 0, 0]] = 1.0;
        let input = BoundedTensor::new(lower, upper).expect("input");
        (graph, input)
    }

    fn direct_strided_conv_transpose_383() -> (GraphNetwork, BoundedTensor) {
        let conv = ConvTranspose2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), -0.75_f32),
            Some(arr1(&[0.125_f32])),
            (1, 2),
            (0, 0),
            1,
            192,
        )
        .expect("1x1 ConvTranspose2d");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("convt", Layer::ConvTranspose2d(conv)));
        graph.set_output("convt");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 192]), -0.4_f32),
            ArrayD::from_elem(IxDyn(&[1, 1, 192]), 0.6_f32),
        )
        .expect("input");
        (graph, input)
    }

    #[test]
    fn sparse_relu_collector_matches_selected_rows_and_keeps_other_rows_at_ibp() {
        let _env_lock = ny_test_utils::env::lock_env();
        let (graph, input) = mixed_stability_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");

        let full = {
            let _gate = ny_test_utils::env::ScopedEnvVar::unset(SPARSE_RELU_ROWS_ENV);
            collect(&graph, &input, ibp.clone())
        };
        let sparse = {
            let _gate = ny_test_utils::env::ScopedEnvVar::set(SPARSE_RELU_ROWS_ENV, "1");
            collect(&graph, &input, ibp.clone())
        };

        let producer_full = full.bounds.get("producer").expect("full producer");
        let producer_sparse = sparse.bounds.get("producer").expect("sparse producer");
        let producer_ibp = ibp.get("producer").expect("IBP producer");
        for row in [0usize, 1] {
            assert_eq!(producer_sparse.lower()[[row]], producer_full.lower()[[row]]);
            assert_eq!(producer_sparse.upper()[[row]], producer_full.upper()[[row]]);
        }
        for row in [2usize, 3] {
            assert_eq!(producer_sparse.lower()[[row]], producer_ibp.lower()[[row]]);
            assert_eq!(producer_sparse.upper()[[row]], producer_ibp.upper()[[row]]);
        }
        assert!(matches!(
            sparse.provenance.get("producer"),
            Some(BoundsProvenance::Crown)
        ));

        let out_full = full.bounds.get("out").expect("full output");
        let out_sparse = sparse.bounds.get("out").expect("sparse output");
        assert_eq!(out_sparse.lower(), out_full.lower());
        assert_eq!(out_sparse.upper(), out_full.upper());
    }

    #[test]
    fn sparse_relu_collector_skips_all_stable_intermediate_without_degradation_event() {
        let _env_lock = ny_test_utils::env::lock_env();
        let _gate = ny_test_utils::env::ScopedEnvVar::set(SPARSE_RELU_ROWS_ENV, "1");

        let producer = LinearLayer::new(arr2(&[[1.0_f32], [1.0]]), Some(arr1(&[2.0_f32, -2.0])))
            .expect("producer");
        let out = LinearLayer::new(arr2(&[[1.0_f32, -1.0]]), None).expect("output projection");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("producer", Layer::Linear(producer)));
        graph.add_node(GraphNode::new(
            "act",
            Layer::ReLU(ReLULayer),
            vec!["producer".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(out),
            vec!["act".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input");
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let result = collect(&graph, &input, ibp.clone());

        let producer_result = result.bounds.get("producer").expect("result producer");
        let producer_ibp = ibp.get("producer").expect("IBP producer");
        assert_eq!(producer_result.lower(), producer_ibp.lower());
        assert_eq!(producer_result.upper(), producer_ibp.upper());
        assert!(matches!(
            result.provenance.get("producer"),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::StableReluRowsSkipped
            ))
        ));
        assert!(
            result
                .fallback_events
                .iter()
                .all(|event| event.reason != CrownIbpFallbackReason::StableReluRowsSkipped),
            "a deterministic structural skip is not a degradation event"
        );
    }

    #[test]
    fn sparse_relu_gate_keeps_correlated_final_target_dense() {
        let _env_lock = ny_test_utils::env::lock_env();
        let _gate = ny_test_utils::env::ScopedEnvVar::set(SPARSE_RELU_ROWS_ENV, "1");

        let pre = LinearLayer::new(arr2(&[[1.0_f32], [-1.0]]), None).expect("pre");
        let out = LinearLayer::new(
            arr2(&[[1.0_f32, 1.0], [1.0, -1.0]]),
            Some(arr1(&[1.0_f32, 0.0])),
        )
        .expect("out");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("pre", Layer::Linear(pre)));
        graph.add_node(GraphNode::new(
            "act",
            Layer::ReLU(ReLULayer),
            vec!["pre".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(out),
            vec!["act".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input");
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let result = collect(&graph, &input, ibp.clone());

        let out_ibp = ibp.get("out").expect("IBP output");
        let out_crown = result.bounds.get("out").expect("CROWN output");
        assert!(
            out_crown.upper()[[0]] < out_ibp.upper()[[0]],
            "dense final CROWN must retain the h1+h2 correlation tightening"
        );
        assert!(matches!(
            result.provenance.get("out"),
            Some(BoundsProvenance::Crown)
        ));
    }

    #[test]
    fn m1_empty_sparse_cone_is_removed_before_budget_and_skipped_structurally() {
        ny_test_utils::env::with_env_edits(|env| {
            env.set(SPARSE_RELU_ROWS_ENV, "1");
            env.set(CROWN_CHUNK_AWARE_BUDGET_ENV, "1");
            env.set("NY_DENSE_BUDGET_MB", "4096");
            let (graph, input) = empty_sparse_cone_net();
            let ibp = graph.collect_node_bounds(&input).expect("IBP");
            let _objective = MarginOutputSeedGuard::publish(vec![511]);
            let (result, trace) =
                run_with_m1_collector_test_controls(M1CollectorTestControls::default(), || {
                    collect(&graph, &input, ibp)
                });

            assert!(trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::Plan {
                    node,
                    candidate: false,
                    weight,
                    raw_rows: 512,
                    subset_rows: Some(0),
                } if node == "conv1" && *weight == 0.0
            )));
            assert!(trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::ObjectiveConeSkipBeforeBudget { node }
                    if node == "conv1"
            )));
            assert!(!trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::BudgetPhaseEntered { node } if node == "conv1"
            )));
            assert_eq!(
                result.provenance.get("conv1").copied(),
                Some(BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::ObjectiveConeRowsSkipped,
                ))
            );
            assert!(result
                .fallback_events
                .iter()
                .all(|event| { event.reason != CrownIbpFallbackReason::ObjectiveConeRowsSkipped }));
        });
    }

    #[test]
    fn m1_known_subset_rejection_uses_full_width_before_budgeting() {
        ny_test_utils::env::with_env_edits(|env| {
            env.set(SPARSE_RELU_ROWS_ENV, "1");
            env.set(CROWN_CHUNK_AWARE_BUDGET_ENV, "1");
            env.set("NY_DENSE_BUDGET_MB", "4096");
            let (graph, input) = mixed_stability_net();
            let ibp = graph.collect_node_bounds(&input).expect("IBP");
            let controls = M1CollectorTestControls {
                route_subset_budget_bytes: Some(1),
                ..M1CollectorTestControls::default()
            };
            let (result, trace) = run_with_m1_collector_test_controls(controls, || {
                collect_with_deadline(
                    &graph,
                    &input,
                    ibp,
                    Some(Instant::now() + Duration::from_secs(30)),
                )
            });

            assert!(trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::Plan {
                    node,
                    candidate: true,
                    weight,
                    raw_rows: 4,
                    subset_rows: None,
                } if node == "producer" && *weight == 4.0
            )));
            assert!(trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::EffectiveRows {
                    node,
                    rows: 4,
                    raw_rows: 4,
                } if node == "producer"
            )));
            assert_eq!(
                result.provenance.get("producer").copied(),
                Some(BoundsProvenance::Crown)
            );
        });
    }

    #[test]
    fn m1_runtime_subset_failure_reweights_rebudgets_and_completes_full_width() {
        ny_test_utils::env::with_env_edits(|env| {
            env.set(SPARSE_RELU_ROWS_ENV, "1");
            env.set(CROWN_CHUNK_AWARE_BUDGET_ENV, "1");
            env.set("NY_DENSE_BUDGET_MB", "4096");
            let (graph, input) = mixed_stability_net();
            let ibp = graph.collect_node_bounds(&input).expect("IBP");
            let controls = M1CollectorTestControls {
                fail_subset_node: Some("producer".into()),
                ..M1CollectorTestControls::default()
            };
            let (result, trace) = run_with_m1_collector_test_controls(controls, || {
                collect_with_deadline(
                    &graph,
                    &input,
                    ibp,
                    Some(Instant::now() + Duration::from_secs(30)),
                )
            });

            assert!(trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::Plan {
                    node,
                    weight,
                    raw_rows: 4,
                    subset_rows: Some(2),
                    ..
                } if node == "producer" && *weight == 2.0
            )));
            assert!(trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::RuntimeSubsetFailOpen {
                    node,
                    rows: 4,
                    weight,
                    deadline_recomputed: true,
                    fallback_rejected: false,
                } if node == "producer" && *weight == 4.0
            )));
            assert_eq!(
                result.provenance.get("producer").copied(),
                Some(BoundsProvenance::Crown)
            );
            assert!(!trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::HopelessLearned { node, .. } if node == "producer"
            )));
        });
    }

    #[test]
    fn m1_unattempted_full_fallback_rejection_does_not_train_hopeless_class() {
        ny_test_utils::env::with_env_edits(|env| {
            env.set(SPARSE_RELU_ROWS_ENV, "1");
            env.set(CROWN_CHUNK_AWARE_BUDGET_ENV, "1");
            env.set("NY_DENSE_BUDGET_MB", "4096");
            let (graph, input) = mixed_stability_net();
            let ibp = graph.collect_node_bounds(&input).expect("IBP");
            let controls = M1CollectorTestControls {
                fail_subset_node: Some("producer".into()),
                reject_full_fallback_budget_node: Some("producer".into()),
                ..M1CollectorTestControls::default()
            };
            let (result, trace) = run_with_m1_collector_test_controls(controls, || {
                collect_with_deadline(
                    &graph,
                    &input,
                    ibp,
                    Some(Instant::now() + Duration::from_secs(30)),
                )
            });

            assert!(trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::RuntimeSubsetFailOpen {
                    node,
                    rows: 4,
                    weight,
                    deadline_recomputed: true,
                    fallback_rejected: true,
                } if node == "producer" && *weight == 4.0
            )));
            assert_eq!(
                result.provenance.get("producer").copied(),
                Some(BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::PerNodeDeadlineExceeded,
                ))
            );
            assert!(!trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::HopelessLearned { node, .. } if node == "producer"
            )));
        });
    }

    #[test]
    fn m1_retained_fixed_wave_plan_executes_and_drift_falls_back_with_provenance() {
        ny_test_utils::env::with_env_edits(|env| {
            env.set(CROWN_CHUNK_AWARE_BUDGET_ENV, "1");
            env.set("NY_DENSE_BUDGET_MB", "1");
            env.set("NY_CROWN_OBJ_CHUNK", "0");
            let (mut graph, input) = direct_strided_conv_transpose_383();
            graph.set_chunk_wave_parallel_enabled(true);
            let ibp = graph.collect_node_bounds(&input).expect("IBP");
            let (success, success_trace) =
                run_with_m1_collector_test_controls(M1CollectorTestControls::default(), || {
                    collect_with_soft_scheduling_deadline(
                        &graph,
                        &input,
                        ibp.clone(),
                        Instant::now() + Duration::from_secs(30),
                    )
                });
            assert!(success_trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::FixedWaveDispatch {
                    node,
                    drift_injected: false,
                } if node == "convt"
            )));
            assert_eq!(
                success.provenance.get("convt").copied(),
                Some(BoundsProvenance::Crown)
            );

            let controls = M1CollectorTestControls {
                drift_fixed_wave_node: Some("convt".into()),
                ..M1CollectorTestControls::default()
            };
            let (drifted, drift_trace) = run_with_m1_collector_test_controls(controls, || {
                collect_with_soft_scheduling_deadline(
                    &graph,
                    &input,
                    ibp.clone(),
                    Instant::now() + Duration::from_secs(30),
                )
            });
            assert!(drift_trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::FixedWaveDispatch {
                    node,
                    drift_injected: true,
                } if node == "convt"
            )));
            let drift_bounds = drifted.bounds.get("convt").expect("drift bounds");
            let ibp_bounds = ibp.get("convt").expect("IBP convt");
            assert_eq!(drift_bounds.lower(), ibp_bounds.lower());
            assert_eq!(drift_bounds.upper(), ibp_bounds.upper());
            assert_eq!(
                drifted.provenance.get("convt").copied(),
                Some(BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::CrownPropagationError,
                ))
            );
            assert!(drifted.fallback_events.iter().any(|event| {
                event.reason == CrownIbpFallbackReason::CrownPropagationError
                    && event.details.contains("convt")
                    && event.details.contains("changed after M1 admission")
            }));
            assert!(!drift_trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::HopelessLearned { node, .. } if node == "convt"
            )));
        });
    }

    #[test]
    fn m1_gate_unset_and_zero_are_production_collector_identical() {
        ny_test_utils::env::with_env_edits(|env| {
            env.set("NY_DENSE_BUDGET_MB", "1");
            env.set("NY_CROWN_OBJ_CHUNK", "0");
            env.remove(CROWN_CHUNK_AWARE_BUDGET_ENV);
            let (graph, input) = direct_strided_conv_transpose_383();
            let ibp = graph.collect_node_bounds(&input).expect("IBP");
            let unset = collect_with_deadline(
                &graph,
                &input,
                ibp.clone(),
                Some(Instant::now() + Duration::from_secs(30)),
            );
            env.set(CROWN_CHUNK_AWARE_BUDGET_ENV, "0");
            let zero = collect_with_deadline(
                &graph,
                &input,
                ibp,
                Some(Instant::now() + Duration::from_secs(30)),
            );

            assert_eq!(unset.bounds.len(), zero.bounds.len());
            for (node, left) in &unset.bounds {
                let right = zero.bounds.get(node).expect("same node under gate=0");
                assert_eq!(left.lower(), right.lower(), "{node} lower");
                assert_eq!(left.upper(), right.upper(), "{node} upper");
            }
            assert_eq!(unset.provenance, zero.provenance);
            assert_eq!(unset.fallback_events, zero.fallback_events);
        });
    }
}

#[cfg(test)]
mod margin_subset_collector_tests {
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use crate::output_margin_seed::MarginOutputSeedGuard;
    use crate::types::BoundsProvenance;
    use ndarray::{arr1, arr2, Array2};
    use ny_tensor::BoundedTensor;

    /// input(2) -> Linear(2->3) "pre" -> ReLU "act" -> Linear(3->600) "out".
    /// 600 outputs put the OUTPUT node at/above the margin-subset engagement
    /// width; the unstable ReLUs make CROWN strictly tighter than IBP.
    fn wide_output_net() -> (GraphNetwork, BoundedTensor) {
        let pre = LinearLayer::new(
            arr2(&[[1.0_f32, -0.5], [0.25, 0.75], [-0.6, 0.4]]),
            Some(arr1(&[0.05_f32, -0.1, 0.02])),
        )
        .expect("pre");
        // Deterministic mixed-sign weights so IBP loses correlations on
        // (essentially) every output row.
        let weights = Array2::from_shape_fn((600, 3), |(i, j)| {
            let v = ((i * 7 + j * 13) % 11) as f32 / 11.0 - 0.5;
            if v == 0.0 {
                0.3
            } else {
                v
            }
        });
        let out = LinearLayer::new(weights, None).expect("out");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("pre", Layer::Linear(pre)));
        graph.add_node(GraphNode::new(
            "act",
            Layer::ReLU(ReLULayer),
            vec!["pre".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(out),
            vec!["act".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5]).into_dyn(),
            arr1(&[1.0_f32, 0.75]).into_dyn(),
        )
        .expect("input");
        (graph, input)
    }

    /// #margin-subset-seed end-to-end through the collector: with published
    /// indices the OUTPUT node's referenced rows are BIT-IDENTICAL to the
    /// full-width collection's rows, every unreferenced row keeps the sound
    /// IBP enclosure, and provenance stays `Crown`. Without a publication the
    /// collection is byte-identical to the full-width behavior.
    #[test]
    fn collector_scatters_published_margin_rows_over_ibp() {
        let (graph, input) = wide_output_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP bounds");

        // Full-width reference collection (no publication on this thread).
        let full = graph
            .collect_crown_ibp_bounds_core_inner_with_cut_segment(
                &input,
                ibp.clone(),
                None,
                None,
                None,
                0,
            )
            .expect("full-width collection");

        // Published {5, 200}: the OUTPUT tighten seeds only those rows.
        let _guard = MarginOutputSeedGuard::publish(vec![200, 5]);
        let subset = graph
            .collect_crown_ibp_bounds_core_inner_with_cut_segment(
                &input,
                ibp.clone(),
                None,
                None,
                None,
                0,
            )
            .expect("margin-subset collection");

        let out_full = full.bounds.get("out").expect("full out");
        let out_subset = subset.bounds.get("out").expect("subset out");
        let out_ibp = ibp.get("out").expect("ibp out");
        assert_eq!(out_subset.shape(), out_full.shape());
        for i in 0..600 {
            if i == 5 || i == 200 {
                assert_eq!(
                    out_subset.lower()[[i]],
                    out_full.lower()[[i]],
                    "referenced lower row {i} must match the full-width collection"
                );
                assert_eq!(
                    out_subset.upper()[[i]],
                    out_full.upper()[[i]],
                    "referenced upper row {i} must match the full-width collection"
                );
            } else {
                assert_eq!(
                    out_subset.lower()[[i]],
                    out_ibp.lower()[[i]],
                    "unreferenced lower row {i} must keep the IBP enclosure"
                );
                assert_eq!(
                    out_subset.upper()[[i]],
                    out_ibp.upper()[[i]],
                    "unreferenced upper row {i} must keep the IBP enclosure"
                );
            }
        }
        assert!(matches!(
            subset.provenance.get("out"),
            Some(BoundsProvenance::Crown)
        ));
        // Meaningfulness guard: full-width CROWN actually tightens the
        // referenced rows past IBP (otherwise the equalities above are vacuous).
        assert!(
            [5_usize, 200].iter().any(|&i| {
                out_full.lower()[[i]] > out_ibp.lower()[[i]]
                    || out_full.upper()[[i]] < out_ibp.upper()[[i]]
            }),
            "CROWN must beat IBP on a referenced row"
        );
        // Non-output nodes are untouched by the publication.
        let pre_full = full.bounds.get("pre").expect("full pre");
        let pre_subset = subset.bounds.get("pre").expect("subset pre");
        assert_eq!(pre_subset.lower(), pre_full.lower());
        assert_eq!(pre_subset.upper(), pre_full.upper());
    }
}

#[cfg(test)]
mod margin_subset_scatter_tests {
    use super::scatter_margin_rows_over_bounds;
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    fn base_bounds() -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[6]), vec![-10.0f32; 6]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[6]), vec![10.0f32; 6]).unwrap(),
        )
        .unwrap()
    }

    /// Referenced rows take the CROWN values, unreferenced rows keep the base
    /// enclosure, and the (already-tighter) intersection with the base is a
    /// sound enclosure row-for-row.
    #[test]
    fn scatter_places_rows_and_intersection_stays_sound() {
        let base = base_bounds();
        let scattered =
            scatter_margin_rows_over_bounds(&base, &[1, 4], &[-1.5, 2.0], &[0.5, 3.25]).unwrap();
        assert_eq!(scattered.shape(), base.shape());
        let lo = scattered.lower();
        let up = scattered.upper();
        for i in 0..6 {
            match i {
                1 => assert_eq!((lo[[i]], up[[i]]), (-1.5, 0.5)),
                4 => assert_eq!((lo[[i]], up[[i]]), (2.0, 3.25)),
                _ => assert_eq!((lo[[i]], up[[i]]), (-10.0, 10.0)),
            }
        }
        // The collector's IBP intersection keeps every row inside the sound
        // IBP enclosure.
        let (tightened, disjoint) = base
            .intersection_per_element(&scattered)
            .expect("intersection succeeds");
        assert_eq!(disjoint, 0);
        for i in 0..6 {
            assert!(tightened.lower()[[i]] >= base.lower()[[i]]);
            assert!(tightened.upper()[[i]] <= base.upper()[[i]]);
            assert!(tightened.lower()[[i]] <= tightened.upper()[[i]]);
        }
    }

    /// Multi-dimensional base: the scatter runs over the FLAT index space and
    /// restores the base's shape.
    #[test]
    fn scatter_restores_multi_dim_shape() {
        let base = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0f32; 6]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32; 6]).unwrap(),
        )
        .unwrap();
        let scattered = scatter_margin_rows_over_bounds(&base, &[5], &[0.25], &[0.75]).unwrap();
        assert_eq!(scattered.shape(), &[2, 3]);
        assert_eq!(scattered.lower()[[1, 2]], 0.25);
        assert_eq!(scattered.upper()[[1, 2]], 0.75);
        assert_eq!(scattered.lower()[[0, 0]], 0.0);
    }

    /// Malformed requests fail (the consume site falls back to full-width).
    #[test]
    fn scatter_rejects_len_mismatch_and_out_of_range() {
        let base = base_bounds();
        assert!(scatter_margin_rows_over_bounds(&base, &[1, 4], &[0.0], &[0.0, 0.0]).is_err());
        assert!(scatter_margin_rows_over_bounds(&base, &[6], &[0.0], &[0.0]).is_err());
    }
}

// ── #crown-honest-provenance ──────────────────────────────────────────────
#[cfg(test)]
mod crown_honest_provenance_tests {
    use super::{crown_gain_stat, CrownGainStat};
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    /// Build a 1-D `BoundedTensor` that is allowed to carry infinities.
    fn bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).expect("shape"),
            ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).expect("shape"),
        )
        .expect("valid bounds")
    }

    fn rep(v: f32, n: usize) -> Vec<f32> {
        vec![v; n]
    }

    /// The measured yolo_2023 failure: CROWN returns `[-inf, inf]` where the
    /// forward IBP pass had a finite box. This is what used to be recorded as a
    /// `Crown` success.
    #[test]
    fn vacuous_crown_is_detected() {
        let ibp = bt(&rep(-1.0, 8), &rep(1.0, 8));
        let crown = bt(&rep(f32::NEG_INFINITY, 8), &rep(f32::INFINITY, 8));
        let stat = crown_gain_stat(&ibp, &crown);
        assert_eq!(stat.ibp_finite, 8);
        assert_eq!(stat.crown_lost, 8);
        assert_eq!(stat.tightened_material, 0);
        assert!(stat.is_vacuous());
    }

    /// A genuinely tightening CROWN (the `Conv_5` / `Add_8` case, 15x) must never
    /// be tagged vacuous.
    #[test]
    fn healthy_crown_is_not_vacuous() {
        let ibp = bt(&rep(-1.0, 8), &rep(1.0, 8));
        let crown = bt(&rep(-0.5, 8), &rep(0.5, 8));
        let stat = crown_gain_stat(&ibp, &crown);
        assert_eq!(stat.crown_lost, 0);
        assert_eq!(stat.tightened_material, 8);
        assert!(!stat.is_vacuous());
    }

    /// A ONE-SIDED explosion that still improves the other endpoint is NOT
    /// vacuous: the intersection keeps that improvement, so the work paid off.
    #[test]
    fn one_sided_explosion_that_still_tightens_is_not_vacuous() {
        let ibp = bt(&rep(-1.0, 8), &rep(1.0, 8));
        let crown = bt(&rep(f32::NEG_INFINITY, 8), &rep(0.2, 8));
        let stat = crown_gain_stat(&ibp, &crown);
        assert_eq!(stat.crown_lost, 8, "the lower endpoint did explode");
        assert_eq!(
            stat.tightened_material, 8,
            "but the upper endpoint improved"
        );
        assert!(!stat.is_vacuous());
    }

    /// On an unbounded sub-domain the IBP box is infinite too, so CROWN never had
    /// finiteness to destroy. Without the `ibp_finite > 0` conjunct this would
    /// fire on every node of an unbounded problem.
    #[test]
    fn unbounded_ibp_is_never_vacuous() {
        let ibp = bt(&rep(f32::NEG_INFINITY, 8), &rep(f32::INFINITY, 8));
        let crown = bt(&rep(f32::NEG_INFINITY, 8), &rep(f32::INFINITY, 8));
        let stat = crown_gain_stat(&ibp, &crown);
        assert_eq!(stat.ibp_finite, 0);
        assert!(!stat.is_vacuous());
    }

    /// A degenerate (point) IBP interval offers nothing to tighten.
    #[test]
    fn point_ibp_is_never_vacuous() {
        let ibp = bt(&rep(1.0, 8), &rep(1.0, 8));
        let crown = bt(&rep(1.0, 8), &rep(1.0, 8));
        let stat = crown_gain_stat(&ibp, &crown);
        assert_eq!(stat.improvable, 0);
        assert_eq!(stat.crown_lost, 0);
        assert!(!stat.is_vacuous());
    }

    /// ADVERSARIAL: one lucky row out of 1000 must not be able to hide 999
    /// exploded ones, and a 1-ULP shave is not a material tightening.
    #[test]
    fn one_ulp_shave_cannot_mask_vacuity() {
        let mut il = rep(-1.0, 1000);
        let mut iu = rep(1.0, 1000);
        let mut cl = rep(f32::NEG_INFINITY, 1000);
        let mut cu = rep(f32::INFINITY, 1000);
        il[999] = -1.0;
        iu[999] = 1.0;
        cl[999] = -1.0;
        cu[999] = 1.0 - f32::EPSILON;
        let stat = crown_gain_stat(&bt(&il, &iu), &bt(&cl, &cu));
        assert_eq!(stat.crown_lost, 999);
        assert_eq!(
            stat.tightened_material, 0,
            "a 1-ULP shave is noise, not a tightening"
        );
        assert!(stat.is_vacuous(), "999/1000 exploded is still vacuous");
    }

    /// ADVERSARIAL, other direction: a real minority gain must DEFEAT vacuity, so
    /// the detector cannot suppress a node that is genuinely contributing.
    #[test]
    fn minority_real_gain_defeats_vacuity() {
        let mut cl = rep(f32::NEG_INFINITY, 1000);
        let mut cu = rep(f32::INFINITY, 1000);
        for i in 900..1000 {
            cl[i] = -0.1;
            cu[i] = 0.1;
        }
        let stat = crown_gain_stat(&bt(&rep(-1.0, 1000), &rep(1.0, 1000)), &bt(&cl, &cu));
        assert_eq!(stat.crown_lost, 900);
        assert_eq!(stat.tightened_material, 100);
        assert!(!stat.is_vacuous(), "90% lost is below the 99% threshold");
    }

    /// A DISJOINT CROWN box means the stored bound got WIDER (union fallback), so
    /// it must never be counted as a tightening.
    #[test]
    fn disjoint_union_is_not_a_tightening() {
        let ibp = bt(&rep(-1.0, 8), &rep(1.0, 8));
        let crown = bt(&rep(5.0, 8), &rep(6.0, 8));
        let stat = crown_gain_stat(&ibp, &crown);
        assert_eq!(stat.tightened_material, 0);
    }

    /// The `Conv_20` case: CROWN comes back FINITE but 3.8x WIDER than the IBP
    /// box it was meant to improve. Not vacuous -- nothing went infinite -- and
    /// the intersection rescues the stored bound, so every pre-existing signal
    /// calls this a success. It must be reported as a regression.
    #[test]
    fn finite_but_wider_than_ibp_is_regressed_not_vacuous() {
        let ibp = bt(&rep(-68.47, 8), &rep(68.47, 8));
        let crown = bt(&rep(-260.30, 8), &rep(260.30, 8));
        let stat = crown_gain_stat(&ibp, &crown);
        assert_eq!(stat.crown_lost, 0, "nothing went non-finite");
        assert!(
            !stat.is_vacuous(),
            "vacuity must NOT fire -- this is the gap"
        );
        assert_eq!(stat.widened_material, 8);
        assert_eq!(stat.tightened_material, 0);
        assert!(stat.is_regressed(), "regression MUST fire");
    }

    /// A healthy tightening node must not be reported as regressed.
    #[test]
    fn healthy_crown_is_not_regressed() {
        let ibp = bt(&rep(-1.0, 8), &rep(1.0, 8));
        let crown = bt(&rep(-0.5, 8), &rep(0.5, 8));
        let stat = crown_gain_stat(&ibp, &crown);
        assert_eq!(stat.widened_material, 0);
        assert!(!stat.is_regressed());
    }

    /// A vacuous node is NOT counted as regressed: an infinite CROWN endpoint is
    /// excluded from the widened count, so the two predicates stay disjoint and
    /// each failure is reported exactly once.
    #[test]
    fn vacuous_is_not_also_regressed() {
        let ibp = bt(&rep(-1.0, 8), &rep(1.0, 8));
        let crown = bt(&rep(f32::NEG_INFINITY, 8), &rep(f32::INFINITY, 8));
        let stat = crown_gain_stat(&ibp, &crown);
        assert!(stat.is_vacuous());
        assert_eq!(
            stat.widened_material, 0,
            "infinite is not 'materially wider'"
        );
        assert!(!stat.is_regressed());
    }

    /// FAILS OPEN on a shape disagreement: the default stat is not vacuous, i.e.
    /// exactly the pre-change behavior.
    #[test]
    fn shape_mismatch_fails_open() {
        let ibp = bt(&rep(-1.0, 4), &rep(1.0, 4));
        let crown = BoundedTensor::new_allow_infinite(
            ArrayD::from_elem(IxDyn(&[2, 2]), -1.0),
            ArrayD::from_elem(IxDyn(&[2, 2]), 1.0),
        )
        .expect("valid bounds");
        let stat = crown_gain_stat(&ibp, &crown);
        assert_eq!(stat, CrownGainStat::default());
        assert!(!stat.is_vacuous());
    }
}

#[cfg(test)]
mod walk_admission_collector_tests {
    //! #cprime-admission integration: the collection on a DAG where one node
    //! is unaffordable refuses that node's walk upfront (typed, ~0 s instead
    //! of a share burn), the unspent share rolls forward so later demanded
    //! targets still tighten, and the refused node keeps its sound IBP bound.

    use super::{
        budget_policy, run_with_m1_collector_test_controls, M1CollectorTestControls,
        M1CollectorTraceEvent,
    };
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use crate::types::{BoundsProvenance, CrownIbpFallbackReason, GraphCrownIbpBoundsResult};
    use ndarray::{arr1, arr2};
    use ny_tensor::BoundedTensor;
    use std::time::{Duration, Instant};

    /// pre -> act1 -> producer -> act2 -> out: three demanded dense targets
    /// (pre, producer, out) in execution order.
    fn three_target_net() -> (GraphNetwork, BoundedTensor) {
        let pre = LinearLayer::new(arr2(&[[1.0_f32], [-1.0]]), None).expect("pre");
        let producer = LinearLayer::new(
            arr2(&[[1.0_f32, -1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            Some(arr1(&[0.0_f32, -0.5, 1.0, -3.0])),
        )
        .expect("producer");
        let out = LinearLayer::new(
            arr2(&[[1.0_f32, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]),
            None,
        )
        .expect("out");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("pre", Layer::Linear(pre)));
        graph.add_node(GraphNode::new(
            "act1",
            Layer::ReLU(ReLULayer),
            vec!["pre".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "producer",
            Layer::Linear(producer),
            vec!["act1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "act2",
            Layer::ReLU(ReLULayer),
            vec!["producer".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(out),
            vec!["act2".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input");
        (graph, input)
    }

    fn collect_with_controls(
        controls: M1CollectorTestControls,
    ) -> (
        GraphCrownIbpBoundsResult,
        Vec<M1CollectorTraceEvent>,
        std::collections::HashMap<String, BoundedTensor>,
    ) {
        let (graph, input) = three_target_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let ibp_for_result = ibp.clone();
        let (result, trace) = run_with_m1_collector_test_controls(controls, || {
            graph
                .collect_crown_ibp_bounds_core_inner_with_cut_segment(
                    &input,
                    ibp,
                    Some(Instant::now() + Duration::from_secs(30)),
                    None,
                    // This suite exercises graph-native c-prime admission.
                    // The merged audit branch routes a pure CPU Linear/ReLU
                    // chain through the sequential fast path, where per-target
                    // walk admission intentionally does not exist. A zero
                    // threshold leaves every nonnegative-width target eligible
                    // while explicitly selecting the graph-native collector.
                    Some(0.0),
                    0,
                )
                .expect("CROWN-IBP collection")
        });
        (result, trace, ibp_for_result)
    }

    /// The unaffordable node is refused UPFRONT (typed WalkCostRefused, IBP
    /// bound kept, degradation event emitted), while the share it did not
    /// burn reaches the later demanded target, which still tightens to a
    /// genuine CROWN bound. Nodes not downstream of the refusal are
    /// bit-identical to the unforced collection.
    #[test]
    fn walk_admission_refuses_unaffordable_node_and_rolls_share_cprime_admission() {
        // Baseline: no forcing; every walk is cheap and admitted.
        let (baseline, _, _) = collect_with_controls(M1CollectorTestControls::default());
        assert_eq!(
            baseline.provenance.get("producer").copied(),
            Some(BoundsProvenance::Crown),
            "baseline sanity: the cheap walk is admitted and completes"
        );

        // Forced: 'producer' estimates far past any share (the Conv_17
        // exhibit shape: estimate >> share).
        let controls = M1CollectorTestControls {
            force_walk_estimate_secs: Some(("producer".into(), 1.0e6)),
            ..M1CollectorTestControls::default()
        };
        let (result, trace, ibp) = collect_with_controls(controls);

        // Typed refusal, upfront: provenance + degradation event + trace.
        assert_eq!(
            result.provenance.get("producer").copied(),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::WalkCostRefused
            ))
        );
        assert!(result.fallback_events.iter().any(|event| {
            event.reason == CrownIbpFallbackReason::WalkCostRefused
                && event.details.contains("refused upfront")
        }));
        assert!(trace.iter().any(|event| matches!(
            event,
            M1CollectorTraceEvent::WalkRefused { node } if node == "producer"
        )));

        // Refusal keeps the sound IBP bound bit-for-bit.
        let producer_ibp = ibp.get("producer").expect("IBP producer");
        let producer_result = result.bounds.get("producer").expect("result producer");
        assert_eq!(producer_result.lower(), producer_ibp.lower());
        assert_eq!(producer_result.upper(), producer_ibp.upper());

        // The collection CONTINUES: the later demanded target still gets a
        // genuine CROWN bound (the rolled-over budget reached it).
        assert_eq!(
            result.provenance.get("out").copied(),
            Some(BoundsProvenance::Crown)
        );

        // A node not downstream of the refusal is bit-identical to baseline.
        let pre_baseline = baseline.bounds.get("pre").expect("baseline pre");
        let pre_forced = result.bounds.get("pre").expect("forced pre");
        assert_eq!(pre_forced.lower(), pre_baseline.lower());
        assert_eq!(pre_forced.upper(), pre_baseline.upper());
    }

    /// Determinism: the same forced estimate produces the same admission set
    /// and byte-identical bounds on a re-run.
    #[test]
    fn walk_admission_is_deterministic_across_reruns_cprime_admission() {
        let controls = || M1CollectorTestControls {
            force_walk_estimate_secs: Some(("producer".into(), 1.0e6)),
            ..M1CollectorTestControls::default()
        };
        let (first, _, _) = collect_with_controls(controls());
        let (second, _, _) = collect_with_controls(controls());
        assert_eq!(
            first.provenance.get("producer"),
            second.provenance.get("producer")
        );
        assert_eq!(first.provenance.get("out"), second.provenance.get("out"));
        for node in ["pre", "producer", "out"] {
            let a = first.bounds.get(node).expect("first bounds");
            let b = second.bounds.get(node).expect("second bounds");
            assert_eq!(a.lower(), b.lower(), "{node} lower must be identical");
            assert_eq!(a.upper(), b.upper(), "{node} upper must be identical");
        }
    }

    /// The LAST demanded target may claim the accumulated rollover: an
    /// estimate over its capped share but within the collection's remaining
    /// time is admitted with the collection deadline, completes, and keeps a
    /// genuine CROWN bound.
    #[test]
    fn walk_admission_last_target_rollover_grant_cprime_admission() {
        // 'out' is the last demanded candidate in execution order. Force an
        // estimate above any per-node share ever derived from a 30 s window
        // (cap = max(12, 0.25*remaining) = 12 s; 13 * 1.25 = 16.25 s <= ~30 s
        // remaining) so only the rollover can admit it.
        let controls = M1CollectorTestControls {
            force_walk_estimate_secs: Some(("out".into(), 13.0)),
            ..M1CollectorTestControls::default()
        };
        let (result, trace, _) = collect_with_controls(controls);
        assert!(trace.iter().any(|event| matches!(
            event,
            M1CollectorTraceEvent::WalkRolloverGranted { node } if node == "out"
        )));
        assert_eq!(
            result.provenance.get("out").copied(),
            Some(BoundsProvenance::Crown),
            "the rollover-granted walk must complete and keep CROWN"
        );
        assert!(
            !result
                .fallback_events
                .iter()
                .any(|event| event.reason == CrownIbpFallbackReason::WalkCostRefused),
            "a rollover grant is an admission, not a refusal"
        );
    }

    /// ADVERSARIAL VERIFY (#cprime-admission, attack 4 PARITY): when every
    /// walk fits its share, the admission layer is a no-op — a forced (but
    /// affordable) estimate and the default real-estimate path produce
    /// byte-identical bounds and provenance, no refusal or rollover events,
    /// and every demanded target keeps a genuine CROWN bound.
    #[test]
    fn verify_walk_admission_all_fit_parity_cprime() {
        // Pin the sparse-relu-rows gate for BOTH collections: sibling tests
        // in this file toggle NY_CROWN_IBP_SPARSE_RELU_ROWS under the env
        // lock, and an unpinned reader racing that toggle sees different row
        // plans between its two collections (observed: producer's stable
        // rows kept IBP in one of the two runs).
        let _env = ny_test_utils::env::lock_env();
        let _pin =
            ny_test_utils::env::ScopedEnvVar::unset(super::super::demand::SPARSE_RELU_ROWS_ENV);
        let (baseline, baseline_trace, _) =
            collect_with_controls(M1CollectorTestControls::default());
        let controls = M1CollectorTestControls {
            // Affordable by orders of magnitude: exercises the Admit arm
            // with the admission machinery fully engaged.
            force_walk_estimate_secs: Some(("producer".into(), 0.001)),
            ..M1CollectorTestControls::default()
        };
        let (forced, forced_trace, _) = collect_with_controls(controls);

        for trace in [&baseline_trace, &forced_trace] {
            assert!(
                !trace.iter().any(|event| matches!(
                    event,
                    M1CollectorTraceEvent::WalkRefused { .. }
                        | M1CollectorTraceEvent::WalkRolloverGranted { .. }
                )),
                "an all-fits collection must emit no refusal/rollover events"
            );
        }
        for node in ["pre", "producer", "out"] {
            assert_eq!(
                baseline.provenance.get(node),
                forced.provenance.get(node),
                "{node} provenance must be identical"
            );
            let a = baseline.bounds.get(node).expect("baseline bounds");
            let b = forced.bounds.get(node).expect("forced bounds");
            assert_eq!(a.lower(), b.lower(), "{node} lower must be identical");
            assert_eq!(a.upper(), b.upper(), "{node} upper must be identical");
        }
        for node in ["pre", "producer", "out"] {
            assert_eq!(
                baseline.provenance.get(node).copied(),
                Some(BoundsProvenance::Crown),
                "{node} must keep a genuine CROWN bound when everything fits"
            );
        }
    }

    /// ADVERSARIAL VERIFY (#cprime-admission, attacks 1+2 integration): the
    /// FIRST demanded candidate is refused PRE-calibration (no completed walk
    /// has taught the model anything yet — the refusal rides the census
    /// prior), it keeps its IBP bound bit-for-bit, and BOTH later demanded
    /// targets still tighten to genuine CROWN bounds — the refused share
    /// freed the window rather than starving what follows (the floor100
    /// death shape).
    #[test]
    fn verify_walk_admission_first_candidate_refused_precalibration_cprime() {
        // Pin the sparse-relu-rows gate: sibling tests toggle
        // NY_CROWN_IBP_SPARSE_RELU_ROWS under the env lock and an unpinned
        // reader races their row plans (see the parity test above).
        let _env = ny_test_utils::env::lock_env();
        let _pin =
            ny_test_utils::env::ScopedEnvVar::unset(super::super::demand::SPARSE_RELU_ROWS_ENV);
        let controls = M1CollectorTestControls {
            force_walk_estimate_secs: Some(("pre".into(), 1.0e6)),
            ..M1CollectorTestControls::default()
        };
        let (result, trace, ibp) = collect_with_controls(controls);

        assert_eq!(
            result.provenance.get("pre").copied(),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::WalkCostRefused
            )),
            "first candidate must be refused on the prior, before any calibration"
        );
        assert!(trace.iter().any(|event| matches!(
            event,
            M1CollectorTraceEvent::WalkRefused { node } if node == "pre"
        )));
        let pre_ibp = ibp.get("pre").expect("IBP pre");
        let pre_result = result.bounds.get("pre").expect("result pre");
        assert_eq!(pre_result.lower(), pre_ibp.lower());
        assert_eq!(pre_result.upper(), pre_ibp.upper());

        // The refusal freed its share: every later demanded target tightens.
        for node in ["producer", "out"] {
            assert_eq!(
                result.provenance.get(node).copied(),
                Some(BoundsProvenance::Crown),
                "{node} must still get a genuine CROWN bound after the refusal"
            );
        }
    }

    /// WIRING PIN (#walk-value-record): a measured completion record for the
    /// exact (node, rows) converts the forced-unaffordable refusal into a
    /// bounded measured grant, the walk runs inside the granted deadline, and
    /// the node keeps a genuine CROWN bound. Later demanded targets are
    /// untouched. The scenario mirrors the pre-calibration refusal test above
    /// with exactly one difference — the record — so the two tests together
    /// pin both faces: no record => refusal identical to today; record =>
    /// grant.
    #[test]
    fn verify_measured_record_converts_refusal_into_grant_walk_value_record() {
        let _env = ny_test_utils::env::lock_env();
        let _pin =
            ny_test_utils::env::ScopedEnvVar::unset(super::super::demand::SPARSE_RELU_ROWS_ENV);
        budget_policy::reset_node_walk_records();
        // A prior collection on this thread completed 'pre' (2 rows) in 0.5s.
        budget_policy::record_node_walk_completed("pre", 2, 0.5);
        // The proxy estimate says the walk is unaffordable (the stale-model
        // shape: forced 1e6s against a ~seconds share).
        let controls = M1CollectorTestControls {
            force_walk_estimate_secs: Some(("pre".into(), 1.0e6)),
            ..M1CollectorTestControls::default()
        };
        let (result, trace, _ibp) = collect_with_controls(controls);
        budget_policy::reset_node_walk_records();

        assert!(
            trace.iter().any(|event| matches!(
                event,
                M1CollectorTraceEvent::WalkMeasuredGrantApplied { node } if node == "pre"
            )),
            "the measured record must convert the refusal into a grant, trace: {trace:?}"
        );
        assert_eq!(
            result.provenance.get("pre").copied(),
            Some(BoundsProvenance::Crown),
            "the granted walk completes inside its measured deadline and keeps CROWN"
        );
        for node in ["producer", "out"] {
            assert_eq!(
                result.provenance.get(node).copied(),
                Some(BoundsProvenance::Crown),
                "{node} must still get a genuine CROWN bound after the grant"
            );
        }
    }
}

/// #target-cost-weight: the corrected cost model and the per-target share cap,
/// exercised through a real CROWN-IBP collection.
///
/// The defect these tests pin: the collector weights each candidate's slice of
/// the collection window by its objective-ROW COUNT. On cgan row 7 that hands
/// the 28,800-row BatchNorm_11 217 s of a 405 s window (54%) while Conv_19 and
/// Conv_22 share 2.9% — and measured secs/row across those same targets spans
/// 27x, so the row count is not what the walks cost. `rows x prefix` tracks
/// true cost to 1.5x; `rows` alone is off by 9.5x.
#[cfg(test)]
mod target_cost_weight_tests {
    use super::{
        run_with_m1_collector_test_controls, M1CollectorTestControls, M1CollectorTraceEvent,
    };
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use crate::types::{BoundsProvenance, CrownIbpPerNodeTimeBudget, GraphCrownIbpBoundsResult};
    use ndarray::{arr1, Array2};
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    const DARK: CrownIbpPerNodeTimeBudget = CrownIbpPerNodeTimeBudget {
        floor_secs: None,
        cap_secs: None,
        prefix_cost_weighted: false,
        max_target_share_frac: None,
    };

    /// The two axes that could reprice these targets behind the test's back,
    /// pinned unset through their EXISTING name constants.
    ///
    /// `CROWN_CHUNK_AWARE_BUDGET_ENV` in particular is the one mechanism this
    /// lever deliberately stands down for, so leaking it in from another test
    /// would turn every assertion below into a silent no-op.
    ///
    /// Spelled with the constants and not with `"NY_..."` string literals ON
    /// PURPOSE: `ny-levers/tests/source_frontier.rs` is an EXACT occurrence
    /// manifest over lever-name literals in which additions fail just as
    /// removals do, so six new literals here would have moved a
    /// parameter-governance ratchet that this change has no business touching.
    /// A `const` USE is not a literal site.
    struct EnvPins {
        _lock: std::sync::RwLockWriteGuard<'static, ()>,
        _pins: [ny_test_utils::env::ScopedEnvVar; 2],
    }

    fn pin_env() -> EnvPins {
        let lock = ny_test_utils::env::lock_env();
        EnvPins {
            _lock: lock,
            _pins: [
                ny_test_utils::env::ScopedEnvVar::unset(
                    super::super::budget_policy::CROWN_CHUNK_AWARE_BUDGET_ENV,
                ),
                ny_test_utils::env::ScopedEnvVar::unset(super::super::demand::SPARSE_RELU_ROWS_ENV),
            ],
        }
    }

    /// Deterministic dense weights; small magnitudes keep ReLUs genuinely
    /// unstable so every hidden node is a real tightening candidate.
    fn dense(out_features: usize, in_features: usize, seed: u64) -> LinearLayer {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((((state >> 32) as u32) as f64 / f64::from(u32::MAX)) as f32 - 0.5) * 0.6
        };
        let data: Vec<f32> = (0..out_features * in_features).map(|_| next()).collect();
        LinearLayer::new(
            Array2::from_shape_vec((out_features, in_features), data).expect("shape"),
            Some(arr1(&vec![0.05_f32; out_features])),
        )
        .expect("linear layer")
    }

    /// EQUAL demanded rows, WILDLY unequal walk cost — the exact shape the row
    /// weight cannot see.
    ///
    /// `in=2 -> l1(32) -> r1 -> l2(32) -> r2 -> l3(32) -> r3 -> out(2)`.
    /// Per-row prefix MACs: `l1` = 32·2 = 64, `l2` = 64 + 32·32 = 1088,
    /// `l3` = 1088 + 32·32 = 2112. So the three targets carry IDENTICAL row
    /// counts and their backward walks differ in cost by 33x.
    fn equal_rows_unequal_prefix_net() -> (GraphNetwork, BoundedTensor) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("l1", Layer::Linear(dense(32, 2, 11))));
        graph.add_node(GraphNode::new(
            "r1",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "l2",
            Layer::Linear(dense(32, 32, 22)),
            vec!["r1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "r2",
            Layer::ReLU(ReLULayer),
            vec!["l2".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "l3",
            Layer::Linear(dense(32, 32, 33)),
            vec!["r2".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "r3",
            Layer::ReLU(ReLULayer),
            vec!["l3".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(dense(2, 32, 44)),
            vec!["r3".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .expect("input box");
        (graph, input)
    }

    /// WRONG-LANE GUARD, and it is not optional.
    ///
    /// A pure CPU Linear/ReLU chain with no GEMM engine satisfies
    /// `cpu_dense_chain`, so `sequential_fast_path_allowed` sends this exact
    /// net to the SEQUENTIAL collector, which returns before the graph-native
    /// loop this lever lives in ever runs. The first version of these tests did
    /// that and reported a silent no-op: empty plan trace, zero refusals,
    /// nothing repriced. A no-op width threshold (`Some(0.0)` — every node's
    /// `max_width() >= 0.0`, so no candidate is filtered) sets
    /// `width_threshold_present` and forces the graph-native route, the same
    /// trick `test_crown_ibp_precomputed_short_deadline_falls_back_to_exact_ibp_3499`
    /// uses. `assert_graph_native` below then makes the lane a checked fact
    /// rather than an assumption.
    fn collect(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> (GraphCrownIbpBoundsResult, Vec<M1CollectorTraceEvent>) {
        let ibp = graph.collect_node_bounds(input).expect("IBP bounds");
        let (result, trace) =
            run_with_m1_collector_test_controls(M1CollectorTestControls::default(), || {
                graph
                    .collect_crown_ibp_bounds_core_inner_with_cut_segment(
                        input,
                        ibp,
                        deadline,
                        None,
                        Some(0.0),
                        0,
                    )
                    .expect("CROWN-IBP collection")
            });
        assert_graph_native(&trace);
        (result, trace)
    }

    /// The graph-native loop emits one `Plan` event per execution-order node.
    /// An EMPTY trace means the collection returned through the sequential fast
    /// path and none of the code under test executed — a VOID result, not a
    /// passing one.
    fn assert_graph_native(trace: &[M1CollectorTraceEvent]) {
        assert!(
            trace
                .iter()
                .any(|event| matches!(event, M1CollectorTraceEvent::Plan { .. })),
            "collection did NOT enter the graph-native collector (no Plan events): \
             the sequential fast path returned first and nothing under test ran"
        );
    }

    fn planned_weights(trace: &[M1CollectorTraceEvent]) -> HashMap<String, f64> {
        trace
            .iter()
            .filter_map(|event| match event {
                M1CollectorTraceEvent::Plan {
                    node,
                    candidate,
                    weight,
                    ..
                } if *candidate => Some((node.clone(), *weight)),
                _ => None,
            })
            .collect()
    }

    /// THE DEFECT, and the fix, on one graph.
    ///
    /// Dark, the three equal-row targets receive IDENTICAL weights — the
    /// allocator cannot tell a 64-MAC/row walk from a 2112-MAC/row walk. Armed,
    /// their weights become exactly `rows x prefix`, i.e. 1 : 17 : 33, so the
    /// deepest target's share of the collection window rises from 33% to 65%.
    #[test]
    fn prefix_cost_weighting_reprices_equal_row_targets_by_their_real_walk_cost() {
        let _env = pin_env();
        let (mut graph, input) = equal_rows_unequal_prefix_net();

        graph.set_crown_ibp_per_node_time_budget(DARK);
        let (_, dark_trace) = collect(&graph, &input, None);
        let dark = planned_weights(&dark_trace);

        graph.set_crown_ibp_per_node_time_budget(CrownIbpPerNodeTimeBudget {
            prefix_cost_weighted: true,
            ..DARK
        });
        let (_, armed_trace) = collect(&graph, &input, None);
        let armed = planned_weights(&armed_trace);

        for node in ["l1", "l2", "l3"] {
            assert!(
                dark.contains_key(node) && armed.contains_key(node),
                "{node} must be a budget candidate in both arms; \
                 dark={dark:?} armed={armed:?}"
            );
        }

        // Dark: row count only. Three 32-row targets, three identical weights.
        assert!(
            (dark["l1"] - dark["l2"]).abs() < 1e-9 && (dark["l2"] - dark["l3"]).abs() < 1e-9,
            "the shipped model prices these three targets IDENTICALLY even though \
             their walks differ 33x in cost: {dark:?}"
        );

        // Armed: rows x prefix. 64 : 1088 : 2112 per row, normalised to the
        // cheapest = 1 : 17 : 33.
        let base = armed["l1"];
        assert!(base > 0.0, "armed base weight must be positive: {armed:?}");
        assert!(
            (armed["l2"] / base - 17.0).abs() < 1e-6,
            "l2 walks 1088/64 = 17x the per-row prefix of l1: {armed:?}"
        );
        assert!(
            (armed["l3"] / base - 33.0).abs() < 1e-6,
            "l3 walks 2112/64 = 33x the per-row prefix of l1: {armed:?}"
        );

        // The number that actually matters: the deepest, most expensive target's
        // SHARE of the window.
        let dark_sum: f64 = ["l1", "l2", "l3"].iter().map(|n| dark[*n]).sum();
        let armed_sum: f64 = ["l1", "l2", "l3"].iter().map(|n| armed[*n]).sum();
        let dark_share = dark["l3"] / dark_sum;
        let armed_share = armed["l3"] / armed_sum;
        assert!(
            armed_share > dark_share * 1.5,
            "the expensive target's share must rise materially: \
             dark={dark_share:.3} armed={armed_share:.3}"
        );
    }

    /// Every sampled reachable activation must sit inside every published box,
    /// in BOTH arms and under a starved schedule.
    ///
    /// CONTAINMENT, not finiteness. A scheduling change that produced a tighter
    /// box by excluding reachable points would be a FALSE PROOF, not a win, so
    /// the envelope is computed independently of the CROWN backward path — a
    /// point forward through the graph's own IBP evaluator, which is exact on
    /// this affine+ReLU net.
    #[test]
    fn every_arm_still_contains_the_sampled_reachable_envelope() {
        let _env = pin_env();
        let (mut graph, input) = equal_rows_unequal_prefix_net();

        let mut samples: Vec<HashMap<String, Vec<f32>>> = Vec::new();
        for i in 0..17_u32 {
            for j in 0..17_u32 {
                let x = -1.0 + f32::from(i as u16) / 8.0;
                let y = -1.0 + f32::from(j as u16) / 8.0;
                let point = BoundedTensor::new(
                    arr1(&[x, y]).into_dyn(),
                    arr1(&[x, y]).into_dyn(),
                )
                .expect("degenerate point box");
                let values = graph.collect_node_bounds(&point).expect("point forward");
                samples.push(
                    values
                        .into_iter()
                        .map(|(node, bound)| {
                            (node, bound.lower().iter().copied().collect::<Vec<f32>>())
                        })
                        .collect(),
                );
            }
        }
        assert_eq!(samples.len(), 289, "the sampler itself must not be empty");

        let arms: [(&str, CrownIbpPerNodeTimeBudget, Option<Duration>); 3] = [
            ("dark", DARK, None),
            (
                "prefix-cost-weighted",
                CrownIbpPerNodeTimeBudget {
                    prefix_cost_weighted: true,
                    ..DARK
                },
                None,
            ),
            (
                // Starved by ARITHMETIC, not by wall clock: a 600 s window that
                // is never reached, against a 100 s floor, so most targets are
                // refused before any backward step runs.
                "starved+capped",
                CrownIbpPerNodeTimeBudget {
                    prefix_cost_weighted: true,
                    max_target_share_frac: Some(0.02),
                    floor_secs: Some(100.0),
                    ..DARK
                },
                Some(Duration::from_mins(10)),
            ),
        ];

        for (label, budget, window) in arms {
            graph.set_crown_ibp_per_node_time_budget(budget);
            let deadline = window.map(|d| Instant::now() + d);
            let (result, _) = collect(&graph, &input, deadline);
            for (node, bound) in &result.bounds {
                let lower = bound.lower();
                let upper = bound.upper();
                for sample in &samples {
                    let Some(values) = sample.get(node) else {
                        continue;
                    };
                    assert_eq!(values.len(), lower.len(), "{label}/{node} shape");
                    for (index, value) in values.iter().enumerate() {
                        let lo = lower.iter().nth(index).copied().expect("lower element");
                        let hi = upper.iter().nth(index).copied().expect("upper element");
                        assert!(
                            *value >= lo - 1e-4 && *value <= hi + 1e-4,
                            "{label}: node '{node}' row {index} EXCLUDES a reachable \
                             activation {value} from [{lo}, {hi}] — a tighter box that \
                             drops a reachable point is a false proof, not a win"
                        );
                    }
                }
            }
        }
    }

    /// SCHEDULING ONLY: this is what makes the lever safe to default-ON.
    ///
    /// Starving the schedule may only ever (a) leave a completed box
    /// BIT-IDENTICAL, or (b) decline a target, which republishes the exact IBP
    /// bound the collector was handed. It may never produce a box narrower than
    /// the fully-funded arm on any element — the only direction in which a
    /// scheduling change could become a soundness change.
    ///
    /// Deterministic by construction: the refusals come from the FLOOR, not
    /// from wall-clock expiry. The window is 600 s (never reached; this
    /// collection takes milliseconds) and the floor is 100 s, so which targets
    /// decline is decided purely by arithmetic on the weights.
    #[test]
    fn a_declined_target_publishes_exactly_what_it_publishes_today() {
        let _env = pin_env();
        use crate::types::CrownIbpFallbackReason;

        let (mut graph, input) = equal_rows_unequal_prefix_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP bounds");

        graph.set_crown_ibp_per_node_time_budget(DARK);
        let (full, _) = collect(&graph, &input, None);

        // Arithmetic, worked out in full so the expected split is auditable.
        // Weights: l1=32, l2=32x17=544, l3=32x33=1056, out=2x34=68 (sum 1700).
        // Window 600 s, share cap 0.35 (=210 s), adaptive per-node cap 150 s,
        // floor 100 s. Shares in exec order against the REMAINING weight sum:
        //   l1  600·32/1700   =  11.3  -> below the 100 s floor -> REFUSED
        //   l2  600·544/1668  = 195.7  -> capped to 150 -> granted
        //   l3  600·1056/1124 = 563.6  -> capped to 150 -> granted
        //   out sole claimant, share cap SKIPPED -> 150 -> granted
        // So the arm is guaranteed to produce both a refusal and completions,
        // which is what keeps the two assertions at the bottom non-vacuous.
        graph.set_crown_ibp_per_node_time_budget(CrownIbpPerNodeTimeBudget {
            prefix_cost_weighted: true,
            max_target_share_frac: Some(0.35),
            floor_secs: Some(100.0),
            ..DARK
        });
        let (starved, _) = collect(
            &graph,
            &input,
            Some(Instant::now() + Duration::from_mins(10)),
        );

        let mut refused = 0usize;
        let mut identical = 0usize;
        for (node, starved_bound) in &starved.bounds {
            let full_bound = full.bounds.get(node).expect("same node set");
            let starved_lo: Vec<f32> = starved_bound.lower().iter().copied().collect();
            let starved_hi: Vec<f32> = starved_bound.upper().iter().copied().collect();
            let full_lo: Vec<f32> = full_bound.lower().iter().copied().collect();
            let full_hi: Vec<f32> = full_bound.upper().iter().copied().collect();

            // CONTAINMENT: starving can only ever WIDEN. This is the assertion
            // that would fire if a scheduling change were secretly a bound
            // change, and it is checked on EVERY node, in every arm.
            for index in 0..starved_lo.len() {
                assert!(
                    starved_lo[index] <= full_lo[index] + 1e-5
                        && starved_hi[index] >= full_hi[index] - 1e-5,
                    "node '{node}' row {index}: the STARVED arm published [{}, {}], \
                     NARROWER than the fully-funded arm's [{}, {}] — a schedule must \
                     never tighten a box",
                    starved_lo[index],
                    starved_hi[index],
                    full_lo[index],
                    full_hi[index],
                );
            }

            match starved.provenance.get(node) {
                // A target REFUSED before any backward step must republish the
                // collector's own input, byte for byte. This is the precise
                // claim that makes the lever scheduling-only.
                Some(BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::PerNodeBudgetBelowFloor,
                )) => {
                    refused += 1;
                    let handed = ibp.get(node).expect("refused node had an IBP bound");
                    assert_eq!(
                        starved_bound.lower(),
                        handed.lower(),
                        "refused node '{node}' must publish EXACTLY the IBP lower bound \
                         it was handed"
                    );
                    assert_eq!(
                        starved_bound.upper(),
                        handed.upper(),
                        "refused node '{node}' must publish EXACTLY the IBP upper bound \
                         it was handed"
                    );
                }
                Some(BoundsProvenance::Crown) => {
                    if matches!(full.provenance.get(node), Some(BoundsProvenance::Crown)) {
                        identical += 1;
                        assert_eq!(
                            starved_lo, full_lo,
                            "node '{node}' completed in BOTH arms, so its box cannot \
                             depend on how much time it was granted"
                        );
                        assert_eq!(
                            starved_hi, full_hi,
                            "node '{node}' completed in BOTH arms, so its box cannot \
                             depend on how much time it was granted"
                        );
                    }
                }
                _ => {}
            }
        }
        assert!(
            refused > 0,
            "the starved arm must actually REFUSE a target below the floor, otherwise \
             the exact-IBP assertion above is vacuous: {:?}",
            starved.provenance
        );
        assert!(
            identical > 0,
            "at least one target must complete in BOTH arms, otherwise the \
             bit-identity assertion above is vacuous: {:?}",
            starved.provenance
        );
    }
}
