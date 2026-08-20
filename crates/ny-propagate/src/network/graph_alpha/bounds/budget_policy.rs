// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Budget policy helpers for graph CROWN-IBP collection (#3839).
//!
//! Two separate policies exist because the sequential fast-path gate and the
//! graph-native per-node guard serve different purposes:
//! - The fast-path gate decides which *collector* runs — it must be conservative.
//! - The graph-native guard decides per-node execution — it may consult
//!   `crown_ibp_target_can_start_in_patches()` from #3813.

use crate::layers::Layer;
use crate::network::core::GraphNetwork;
use crate::types::{
    BoundsProvenance, CrownIbpFallbackEvent, CrownIbpFallbackReason, CrownIbpPerNodeTimeBudget,
};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::target_backward::{
    live_objective_chunk_driver_route, objective_chunk_route_plan, ObjectiveChunkDriverRoute,
    ObjectiveChunkFixedWavePlan, ObjectiveChunkRoutePlan,
};

/// Minimum useful per-node time budget in seconds for DAG CROWN-IBP (#3499).
pub(crate) const MIN_PER_NODE_BUDGET_SECS: f64 = 2.0;

/// FLOOR for the global per-node time budget in seconds (#4413).
///
/// Historically this was a hard CAP: no target could ever exceed 12 s, whatever
/// the collection's actual budget. It now serves as the floor of the
/// budget-proportional cap derived by [`adaptive_per_node_cap_secs`], so a short
/// collection behaves exactly as before and a long one is no longer truncated
/// to a number that predates it.
pub(super) const ADAPTIVE_PER_NODE_CAP_FLOOR_SECS: f64 = 12.0;

/// Share of a collection's REMAINING time that any single node may claim
/// (#per-node-cap-from-budget).
///
/// WHY THIS IS A SHARE AND NOT A CONSTANT. The cap exists to stop one late
/// target monopolizing the collection — but "monopolizing" is inherently
/// relative to what is left, not an absolute number of seconds. 12 s is 60% of
/// a 20 s remainder (genuinely monopolizing) and 1.3% of a 900 s one (where it
/// merely truncates a node that could have finished). The same constant was
/// serving both cases because it was measured once, on one budget.
///
/// A quarter leaves at least three other nodes able to claim an equal share
/// before the collection is exhausted, which is the property the original hard
/// cap was reaching for.
///
/// This is the same principle `PatchesTighteningBudget::with_collection_deadline`
/// already applies to the aggregate patches budget: size the policy from the
/// live deadline, and keep the historical constant as a floor so nothing
/// regresses on short budgets.
const PER_NODE_CAP_BUDGET_SHARE: f64 = 0.25;

/// Explicitly enable the experimental learned target-prefix affordability gate.
///
/// The topology proxy has not yet been qualified across every target-specific
/// CPU/Patches/GPU-suffix route used by competition models, so it remains dark
/// by default. Exact `1` opts in; every other spelling preserves the historical
/// collector. No-deadline calls are independently excluded before any proxy
/// walk or timer is created.
pub(super) const PREFIX_COST_ADMISSION_ENV: &str = "NY_CROWN_PREFIX_COST_ADMISSION";

/// Require this many completed walks before a timing sample may refuse work.
///
/// One completion can be a first-touch/cache outlier. Two independent target
/// completions establish that the current graph/backend has actually executed
/// the work proxy before it becomes scheduling authority.
const PREFIX_COST_MIN_COMPLETIONS: usize = 2;

/// Assume a future walk may run this many times faster than the fastest
/// completed walk observed in the same collection.
///
/// This deliberately biases the gate toward false negatives (running work that
/// later times out) rather than false positives (skipping a walk that might
/// have completed). Only when the walk still cannot fit at this optimistic
/// speed does admission retain IBP up front.
const PREFIX_COST_OPTIMISTIC_SPEEDUP: u128 = 2;

pub(super) fn prefix_cost_admission_enabled_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

pub(super) fn prefix_cost_admission_enabled() -> bool {
    prefix_cost_admission_enabled_from_raw(std::env::var(PREFIX_COST_ADMISSION_ENV).ok().as_deref())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompletedPrefixRate {
    work_units: u128,
    elapsed_nanos: u128,
}

/// Pure scheduling result from [`PrefixCostAdmissionModel`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum PrefixCostAdmission {
    /// Gate disabled, no deadline, insufficient completed samples, malformed
    /// work, or overflow in the exact ratio comparison: execute the walk.
    RunWithoutEstimate,
    /// The optimistic predicted cost fits the live target window.
    RunEstimated {
        predicted_secs: f64,
        remaining_secs: f64,
        completed_samples: usize,
    },
    /// Even an optimistic 2x speedup over the fastest completed walk cannot fit.
    RetainIbp {
        predicted_secs: f64,
        remaining_secs: f64,
        completed_samples: usize,
    },
}

/// Same-collection target-prefix cost model.
///
/// The model never contributes to bound or verdict authority. It learns only
/// from fully completed target walks, keeps the FASTEST observed effective
/// work rate, and grants every prospective target an additional 2x optimistic
/// speedup. A refusal therefore means that the exact topology work proxy still
/// cannot fit the target's live scheduling window under deliberately favorable
/// assumptions. The fallback is the already-certified IBP target box.
#[derive(Clone, Copy, Debug)]
pub(super) struct PrefixCostAdmissionModel {
    enabled: bool,
    fastest: Option<CompletedPrefixRate>,
    completed_samples: usize,
}

impl PrefixCostAdmissionModel {
    pub(super) const fn new(enabled: bool) -> Self {
        Self {
            enabled,
            fastest: None,
            completed_samples: 0,
        }
    }

    /// Observe a FULLY completed walk. Deadline-truncated/error paths must not
    /// call this: elapsed/total-work for a partial walk is not a measured rate.
    pub(super) fn observe_completed(&mut self, work_units: Option<u128>, elapsed: Duration) {
        if !self.enabled {
            return;
        }
        let Some(work_units) = work_units.filter(|work| *work > 0) else {
            return;
        };
        let elapsed_nanos = elapsed.as_nanos();
        if elapsed_nanos == 0 {
            return;
        }
        let sample = CompletedPrefixRate {
            work_units,
            elapsed_nanos,
        };
        self.fastest = match self.fastest {
            None => {
                self.completed_samples = self.completed_samples.saturating_add(1);
                Some(sample)
            }
            Some(previous) => {
                // candidate work/time > previous work/time => candidate is
                // faster. If either exact cross-product overflows, disable the
                // model for this collection: ignoring an unrankable candidate
                // could retain a falsely slow rate and mint skip authority.
                let candidate_lhs = sample.work_units.checked_mul(previous.elapsed_nanos);
                let previous_lhs = previous.work_units.checked_mul(sample.elapsed_nanos);
                let Some((candidate, previous_rate)) = candidate_lhs.zip(previous_lhs) else {
                    self.enabled = false;
                    self.completed_samples = 0;
                    return;
                };
                self.completed_samples = self.completed_samples.saturating_add(1);
                if candidate > previous_rate {
                    Some(sample)
                } else {
                    Some(previous)
                }
            }
        };
    }

    /// Decide against an injected remaining target window.
    ///
    /// `remaining=None` is the no-deadline contract and always runs. Exact
    /// equality runs as well: refusal requires strict evidence that the
    /// optimistic prediction exceeds the window. Every arithmetic overflow
    /// runs rather than converting an imprecise estimate into authority.
    pub(super) fn admit(
        &self,
        work_units: Option<u128>,
        remaining: Option<Duration>,
    ) -> PrefixCostAdmission {
        if !self.enabled
            || remaining.is_none()
            || self.completed_samples < PREFIX_COST_MIN_COMPLETIONS
        {
            return PrefixCostAdmission::RunWithoutEstimate;
        }
        let Some(work_units) = work_units.filter(|work| *work > 0) else {
            return PrefixCostAdmission::RunWithoutEstimate;
        };
        let Some(sample) = self.fastest else {
            return PrefixCostAdmission::RunWithoutEstimate;
        };
        let remaining = remaining.expect("checked finite target window");
        let remaining_nanos = remaining.as_nanos();

        // predicted_optimistic = work * sample_time /
        //                        (sample_work * optimistic_speedup)
        // Refuse iff predicted_optimistic > remaining. Keep the comparison in
        // exact integers; checked overflow fails open.
        let Some(predicted_numerator) = work_units.checked_mul(sample.elapsed_nanos) else {
            return PrefixCostAdmission::RunWithoutEstimate;
        };
        let Some(predicted_denominator) = sample
            .work_units
            .checked_mul(PREFIX_COST_OPTIMISTIC_SPEEDUP)
        else {
            return PrefixCostAdmission::RunWithoutEstimate;
        };
        let Some(window_numerator) = remaining_nanos.checked_mul(predicted_denominator) else {
            return PrefixCostAdmission::RunWithoutEstimate;
        };
        let predicted_secs = predicted_numerator as f64 / predicted_denominator as f64 / 1e9;
        let common = (
            predicted_secs,
            remaining.as_secs_f64(),
            self.completed_samples,
        );
        if predicted_numerator > window_numerator {
            PrefixCostAdmission::RetainIbp {
                predicted_secs: common.0,
                remaining_secs: common.1,
                completed_samples: common.2,
            }
        } else {
            PrefixCostAdmission::RunEstimated {
                predicted_secs: common.0,
                remaining_secs: common.1,
                completed_samples: common.2,
            }
        }
    }
}

/// The per-node cap NY derives for itself from its own remaining budget.
///
/// Never below [`ADAPTIVE_PER_NODE_CAP_FLOOR_SECS`] (so short collections are
/// byte-identical to the historical behavior) and never above
/// [`DIM_SCALED_CAP_CEILING_SECS`] (so this stays a monopolization guard rather
/// than an open grant). Sound regardless: the cap only bounds how long a node's
/// CROWN backward may run, and a node that exceeds it degrades to its valid IBP
/// bound.
fn adaptive_per_node_cap_secs(remaining_secs: f64) -> f64 {
    if !remaining_secs.is_finite() || remaining_secs <= 0.0 {
        return ADAPTIVE_PER_NODE_CAP_FLOOR_SECS;
    }
    (remaining_secs * PER_NODE_CAP_BUDGET_SHARE).clamp(
        ADAPTIVE_PER_NODE_CAP_FLOOR_SECS,
        DIM_SCALED_CAP_CEILING_SECS,
    )
}

/// Sanitize a preset-supplied budget override: finite and > 0, else the default.
fn sanitize_budget_secs(override_secs: Option<f64>, default_secs: f64) -> f64 {
    override_secs
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default_secs)
}

/// Operator overrides for the per-node floor/cap.
///
/// These exist so the cap can be MEASURED rather than argued about: the
/// historical 12 s cap was never sweepable without editing source and
/// rebuilding, which is why it went unexamined while
/// `PerNodeDeadlineExceeded` fallbacks accumulated.
/// A value set here is treated exactly like a preset-supplied one (including
/// dim-aware scaling), and an unset/invalid value changes nothing.
const PER_NODE_CAP_ENV: &str = "NY_PER_NODE_CAP_SECS";
const PER_NODE_FLOOR_ENV: &str = "NY_PER_NODE_FLOOR_SECS";

fn budget_secs_from_raw(raw: Option<&str>) -> Option<f64> {
    raw.and_then(|raw| raw.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
}

/// Fully resolved per-node scheduling policy.
///
/// `cap_is_explicit` is part of the policy, not redundant metadata: an unset
/// cap uses [`adaptive_per_node_cap_secs`], whereas an explicit cap equal to
/// the 12 s built-in floor remains fixed at 12 s. Keeping the flag alongside
/// the resolved numbers also gives the truncated-map cache one canonical
/// identity for equivalent preset and environment overrides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ResolvedPerNodeTimeBudget {
    pub(super) floor_secs: f64,
    pub(super) cap_secs: f64,
    pub(super) cap_is_explicit: bool,
}

pub(super) fn resolve_per_node_time_budget(
    budget: &CrownIbpPerNodeTimeBudget,
) -> ResolvedPerNodeTimeBudget {
    let floor_raw = std::env::var(PER_NODE_FLOOR_ENV).ok();
    let cap_raw = std::env::var(PER_NODE_CAP_ENV).ok();
    resolve_per_node_time_budget_from_raw(budget, floor_raw.as_deref(), cap_raw.as_deref())
}

fn resolve_per_node_time_budget_from_raw(
    budget: &CrownIbpPerNodeTimeBudget,
    floor_raw: Option<&str>,
    cap_raw: Option<&str>,
) -> ResolvedPerNodeTimeBudget {
    let floor_override = budget_secs_from_raw(floor_raw).or(budget.floor_secs);
    let cap_override = budget_secs_from_raw(cap_raw)
        .or_else(|| budget.cap_secs.filter(|v| v.is_finite() && *v > 0.0));
    ResolvedPerNodeTimeBudget {
        floor_secs: sanitize_budget_secs(floor_override, MIN_PER_NODE_BUDGET_SECS),
        cap_secs: sanitize_budget_secs(cap_override, ADAPTIVE_PER_NODE_CAP_FLOOR_SECS),
        cap_is_explicit: cap_override.is_some(),
    }
}

/// Resolved floor and base cap after applying environment/preset overrides.
///
/// An unset cap reports the 12 s lower clamp here; the scheduling functions
/// recognize `cap_is_explicit == false` in [`resolve_per_node_time_budget`] and
/// derive the actual cap from one quarter of the live remaining collection
/// budget, clamped to 12..=600 s.
pub(super) fn effective_per_node_time_budget(
    budget: &CrownIbpPerNodeTimeBudget,
) -> (/* floor */ f64, /* cap */ f64) {
    let resolved = resolve_per_node_time_budget(budget);
    (resolved.floor_secs, resolved.cap_secs)
}

/// Default aggregate time budget in seconds for patches-startable nodes in the
/// graph-native CROWN-IBP collector (#3839).
///
/// This caps the total wall-clock time spent on nodes that pass the dense
/// memory guard via the #3813 patches-start path when no global deadline is
/// present.
pub(super) const DEFAULT_PATCHES_TIGHTENING_BUDGET_SECS: f64 = 5.0;

/// Raised aggregate patches budget when `NY_CONV_PATCHES_COLLECT` lifts the
/// large-conv gate (#conv-patches-collect). The default 5s only funds ~2 deep
/// conv-stage backwards; a metaroom-class net has 4+ stages that each want the
/// memory-light patches path. Still bounded by the effective per-node policy
/// and by the collection's own deadline.
pub(super) const CONV_PATCHES_COLLECT_TIGHTENING_BUDGET_SECS: f64 = 40.0;

const PATCHES_BUDGET_ENV: &str = "NY_PATCHES_BUDGET_SECS";

/// Fully resolved aggregate patches scheduling policy.
///
/// `is_explicit` is semantically relevant even when `floor_secs` is unchanged:
/// an explicit value is the fixed total, while the built-in value is only a
/// floor under the deadline-proportional policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ResolvedPatchesTighteningBudget {
    pub(super) floor_secs: f64,
    pub(super) is_explicit: bool,
    pub(super) conv_patches_collect_enabled: bool,
}

fn resolve_patches_tightening_budget_from_raw(
    raw: Option<&str>,
    conv_patches_collect_enabled: bool,
) -> ResolvedPatchesTighteningBudget {
    let explicit = budget_secs_from_raw(raw);
    let floor_secs = explicit.unwrap_or({
        if conv_patches_collect_enabled {
            CONV_PATCHES_COLLECT_TIGHTENING_BUDGET_SECS
        } else {
            DEFAULT_PATCHES_TIGHTENING_BUDGET_SECS
        }
    });
    ResolvedPatchesTighteningBudget {
        floor_secs,
        is_explicit: explicit.is_some(),
        conv_patches_collect_enabled,
    }
}

pub(super) fn resolve_patches_tightening_budget() -> ResolvedPatchesTighteningBudget {
    let raw = std::env::var(PATCHES_BUDGET_ENV).ok();
    resolve_patches_tightening_budget_from_raw(
        raw.as_deref(),
        crate::util::conv_patches_collect_enabled(),
    )
}

fn dense_identity_exceeds_budget(bounds: &BoundedTensor, budget: usize) -> bool {
    crate::network::crown_memory::identity_pair_bytes(bounds.len())
        .map_or(true, |required| required > budget)
}

/// Auto objective row-chunk size for a target whose dense `[dim x dim]`
/// identity pair exceeds the CPU dense budget (#cgan-bn11-chunk).
///
/// Solved against the SAME terms `LinearConcretizationAdmission::new` charges,
/// not against the coefficient pair alone. The pair is the dominant term but
/// not the only one, and the older model — `C = dim * budget /
/// identity_pair_bytes(dim)` — sized `C` so the pair ALONE exactly saturated
/// the budget (99.8% of it at dim=400/1 MiB). Every remaining term was then
/// guaranteed to overflow, so admission raised `CpuMemoryExceeded` and the
/// target degraded to the IBP this reroute exists to avoid. Measured: a
/// dim-400 target under a 1 MiB budget needed 1,056,864 bytes against
/// 1,048,576 — over by 0.8%, and over for EVERY modestly-over-budget target,
/// never just at one size.
///
/// Admission's charge, in the same order it accumulates them:
///
/// * `bounds.memory_bytes()` — the `[C x dim]` pair plus its bias pair:
///   `8*C*dim + 8*C`
/// * `endpoint_bytes` — `C * f64` plus `C * 2 * f32`: `16*C`
/// * `input_scratch_bytes` — both input endpoint arrays: `8*input_dim`
///
/// so the per-row cost is `8*dim + 24` and the fixed cost is `8*input_dim`,
/// giving `C = (budget - 8*input_dim) / (8*dim + 24)`. The estimate rounds
/// CONSERVATIVELY (it slightly over-charges relative to the measured nominal),
/// because under-charging is what produced the defect above. Computed in u128
/// so huge dims cannot overflow; clamped to `dim` (a full-size "chunk" is a
/// single pass) and floored at 1.
pub(super) fn auto_objective_chunk_rows(
    node_dim: usize,
    input_dim: usize,
    budget_bytes: usize,
) -> usize {
    let dim = node_dim.max(1) as u128;
    let f32_bytes = size_of::<f32>() as u128;
    let f64_bytes = size_of::<f64>() as u128;
    // Per row: the coefficient pair's row (2 * f32 * dim), its bias pair
    // (2 * f32), and the endpoint buffers (f64 + 2 * f32).
    let per_row_bytes = 2 * f32_bytes * dim + 2 * f32_bytes + f64_bytes + 2 * f32_bytes;
    // Fixed: both input endpoint arrays, charged whatever C turns out to be.
    let fixed_bytes = 2 * f32_bytes * (input_dim as u128);
    let available = (budget_bytes as u128).saturating_sub(fixed_bytes);
    let rows = available / per_row_bytes.max(1);
    rows.clamp(1, node_dim.max(1) as u128) as usize
}

/// Default-dark scheduling experiment for auto-chunked demanded targets.
///
/// The exact spelling `1` arms the experiment. Unset, empty, `0`, and every
/// other spelling preserve the existing raw-row scheduling weights.
pub(crate) const CROWN_CHUNK_AWARE_BUDGET_ENV: &str = "NY_CROWN_CHUNK_AWARE_BUDGET";

pub(crate) fn crown_chunk_aware_budget_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

pub(crate) fn crown_chunk_aware_budget_enabled() -> bool {
    crown_chunk_aware_budget_from_raw(std::env::var(CROWN_CHUNK_AWARE_BUDGET_ENV).ok().as_deref())
}

/// Exact fixed-wave scheduling proxy for a `rows`-wide target.
///
/// Chunks inside one fixed wave execute concurrently, so repeated-prefix cost
/// advances in `wave_count` groups rather than one group per chunk. The result
/// is `u128` so every native `usize` product is exact.
pub(crate) fn objective_fixed_wave_work_weight(rows: usize, wave_count: usize) -> u128 {
    assert!(
        wave_count > 0,
        "objective chunk wave count must be positive"
    );
    rows as u128 * wave_count as u128
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::network::graph_alpha) struct ObjectiveChunkSchedulingPlan {
    pub(in crate::network::graph_alpha) execution: ObjectiveChunkRoutePlan,
    pub(in crate::network::graph_alpha) fixed_waves: ObjectiveChunkFixedWavePlan,
}

/// Admit M1 only on the execution driver's invariant fixed-wave route.
///
/// A cut context or `NY_NO_CHUNK_WAVE_PAR=1` selects the sequential driver,
/// whose adaptive widths are measured at runtime and cannot be predicted
/// faithfully before execution. Those routes deliberately return `None` and
/// retain raw-row scheduling. Worker count and the memory-derived wave cap are
/// resolved by the same central helper execution uses.
pub(in crate::network::graph_alpha) fn objective_chunk_scheduling_plan(
    target_rows: usize,
    execution: ObjectiveChunkRoutePlan,
    has_deadline: bool,
    has_cut_context: bool,
) -> Option<ObjectiveChunkSchedulingPlan> {
    let chunk_rows = execution.effective_initial_rows;
    let chunk_ceiling = if execution.requested_rows > 0 {
        execution.requested_rows.max(chunk_rows)
    } else {
        target_rows
    };
    match live_objective_chunk_driver_route(
        target_rows,
        chunk_rows,
        chunk_ceiling,
        has_deadline,
        has_cut_context,
    ) {
        ObjectiveChunkDriverRoute::FixedWaves(fixed_waves) => Some(ObjectiveChunkSchedulingPlan {
            execution,
            fixed_waves,
        }),
        ObjectiveChunkDriverRoute::AnchorParallel
        | ObjectiveChunkDriverRoute::Sequential { .. } => None,
    }
}

/// Scheduling-only weight for one demanded target.
///
/// Gate-off returns the historical raw row count exactly. Gate-on inflates
/// only a target represented in the same budget denominator and guaranteed to
/// take the full-objective auto-chunk route. Collector subset seeds pass
/// `eligible_for_inflation=false` because they bypass that route entirely.
///
/// The proxy is admitted only after the central driver resolver proves that
/// execution will use fixed-width waves. Sequential/adaptive and cut-context
/// routes pass `None` and retain raw rows. The operator-only
/// `NY_CROWN_OBJ_CHUNK` knob remains orthogonal to M1.
pub(in crate::network::graph_alpha) fn demanded_target_work_weight(
    rows: usize,
    scheduling_plan: Option<ObjectiveChunkSchedulingPlan>,
    eligible_for_inflation: bool,
    chunk_aware: bool,
) -> f64 {
    match (chunk_aware && eligible_for_inflation, scheduling_plan) {
        (true, Some(plan)) if plan.fixed_waves.wave_count > 0 => {
            objective_fixed_wave_work_weight(rows, plan.fixed_waves.wave_count) as f64
        }
        _ => rows as f64,
    }
}

/// Preserve the parent's cap-dimension input when M1 is dark.
///
/// Before M1 the scheduling weight doubled as the dimension passed to
/// `dim_scaled_cap_secs`; sparse-row experiments therefore used their selected
/// row count. Armed M1 must separate those axes so an inflated work proxy does
/// not masquerade as node width, but dark execution keeps the parent's exact
/// allocation semantics.
pub(crate) fn weighted_budget_cap_dims(
    this_work_weight: f64,
    raw_node_dims: f64,
    chunk_aware: bool,
) -> f64 {
    if chunk_aware {
        raw_node_dims
    } else {
        this_work_weight
    }
}

/// Auto objective-chunk execution plan shared by M1 scheduling consumers.
///
/// `full_objective_route=false` excludes collector subset seeds before any
/// inflation. Otherwise this reproduces the established dense-budget auto
/// request, then resolves the SAME deadline/ConvTranspose initial width used
/// by `propagate_crown_to_node_core`. Callers retain the plan and pass its
/// `requested_rows` into execution. `NY_CROWN_OBJ_CHUNK` is intentionally not
/// read here.
pub(super) fn auto_objective_chunk_route_plan(
    graph: &GraphNetwork,
    node_name: &str,
    bounds: &BoundedTensor,
    input_dim: usize,
    budget_bytes: usize,
    has_deadline: bool,
    full_objective_route: bool,
) -> Option<ObjectiveChunkRoutePlan> {
    if !full_objective_route
        || !graph.graph_native_target_exceeds_budget(node_name, bounds, budget_bytes)
    {
        return None;
    }
    let requested_rows = auto_objective_chunk_rows(bounds.len(), input_dim, budget_bytes);
    let relevant_nodes = graph.ancestors(node_name).ok()?;
    let has_conv_transpose = relevant_nodes.iter().any(|name| {
        graph.nodes.get(name).is_some_and(|node| {
            matches!(
                node.layer,
                Layer::ConvTranspose1d(_) | Layer::ConvTranspose2d(_)
            )
        })
    });
    let plan = objective_chunk_route_plan(requested_rows, has_deadline, has_conv_transpose);
    (plan.effective_initial_rows > 0 && bounds.len() > plan.effective_initial_rows).then_some(plan)
}

pub(crate) fn count_remaining_budget_candidates(
    eligible_suffix_mask: &[bool],
    start_index: usize,
) -> usize {
    eligible_suffix_mask[start_index.min(eligible_suffix_mask.len())..]
        .iter()
        .filter(|eligible| **eligible)
        .count()
}

pub(crate) fn compute_global_per_node_budget_secs(
    remaining_secs: f64,
    remaining_candidates: usize,
    budget: &CrownIbpPerNodeTimeBudget,
) -> Option<f64> {
    if remaining_candidates == 0 || !remaining_secs.is_finite() || remaining_secs <= 0.0 {
        return None;
    }

    let resolved = resolve_per_node_time_budget(budget);
    // Same derivation as the weighted variant (#per-node-cap-from-budget): an
    // explicit preset/env cap wins, otherwise the cap follows the budget NY
    // actually has rather than a constant measured on a different one.
    let cap_secs = if resolved.cap_is_explicit {
        resolved.cap_secs
    } else {
        adaptive_per_node_cap_secs(remaining_secs)
    };
    let share = remaining_secs / remaining_candidates as f64;
    let capped = share.min(cap_secs);
    (capped >= resolved.floor_secs).then_some(capped)
}

/// Sum of the remaining candidate weights from `start_index` onward
/// (#cgan-collection-cost-weight). Zero-weight (non-candidate) nodes contribute
/// nothing, so this is the denominator for the cost-proportional per-node share.
pub(crate) fn sum_remaining_budget_weights(weights: &[f64], start_index: usize) -> f64 {
    weights[start_index.min(weights.len())..]
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w > 0.0)
        .sum()
}

/// Node width the preset per-node cap was measured against (#cgan-dim-cap).
///
/// The cgan_2023 preset cap (`crown_ibp_per_node_cap_secs: 150`) was sized for
/// the 28,800-dim imgSz32 generator target whose objective-chunked backward
/// measured ~95-125 s. Wider nodes scale that cost QUADRATICALLY: the chunked
/// backward's row budget is `~bytes/dim` per chunk, so chunk COUNT grows with
/// `dim` while per-chunk coefficient work also grows with `dim` — the imgSz64
/// generator's 61,504-dim node (2.14x dims) costs ~4.6x, far past the flat cap,
/// and a truncated node degrades the whole map back to IBP.
pub(crate) const PRESET_CAP_REFERENCE_DIMS: f64 = 28_800.0;

/// Ceiling for the dim-scaled preset cap (#cgan-dim-cap): 600 s keeps even the
/// widest single node from monopolizing a 900 s-budget collection slice while
/// covering the projected ~433-570 s need of the 61,504-dim imgSz64 target.
pub(crate) const DIM_SCALED_CAP_CEILING_SECS: f64 = 600.0;

/// Dim-aware preset cap scaling is default-ON; `NY_DIM_CAP_SCALE=0` restores
/// the flat preset cap.
pub(super) fn dim_cap_scale_enabled() -> bool {
    dim_cap_scale_from_raw(std::env::var("NY_DIM_CAP_SCALE").ok().as_deref())
}

fn dim_cap_scale_from_raw(raw: Option<&str>) -> bool {
    raw != Some("0")
}

/// Scale a PRESET-SUPPLIED per-node cap by the target's width (#cgan-dim-cap).
///
/// Only fires when the preset opted into a custom cap (`cap_secs` set) — the
/// derived cap for an unset preset is never width-scaled. Quadratic in
/// `dims / PRESET_CAP_REFERENCE_DIMS` (cost model above), clamped to
/// `DIM_SCALED_CAP_CEILING_SECS`, never below the preset cap itself. Sound
/// either way: the cap only bounds how long the chunked CROWN backward may
/// run; exceeding it degrades the node to IBP.
fn dim_scaled_cap_secs(cap_secs: f64, preset_cap_set: bool, node_dims: f64) -> f64 {
    if !preset_cap_set || !dim_cap_scale_enabled() || !node_dims.is_finite() {
        return cap_secs;
    }
    let ratio = (node_dims / PRESET_CAP_REFERENCE_DIMS).max(1.0);
    (cap_secs * ratio * ratio).min(DIM_SCALED_CAP_CEILING_SECS.max(cap_secs))
}

/// COST-WEIGHTED per-node time budget (#cgan-collection-cost-weight). The
/// equal-share variant above gives a 28,800-dim generator target (BatchNorm_11)
/// the SAME slice as a ~50-dim discriminator conv, so the expensive node's share
/// falls below its true backward cost and it degrades to IBP — truncating the map
/// and forcing 3–5 redundant re-collections. Weighting each candidate's slice by
/// its objective-row count (`ibp_bound.len()`) gives the wide node a
/// cost-proportional window so it COMPLETES on the first pass and the
/// complete-gated cache serves ONE map. Still clamped to `[floor, cap]` (the cap
/// bounds a runaway single node; an under-floor share still degrades to IBP —
/// sound either way). Reduces to the equal-share result when all weights match.
///
/// `this_work_weight` affects only the proportional scheduling share.
/// `raw_node_dims` separately controls the dim-scaled preset cap, so an
/// experimental work multiplier cannot pretend the node itself is wider.
pub(crate) fn compute_weighted_per_node_budget_secs(
    remaining_secs: f64,
    remaining_work_weight_sum: f64,
    this_work_weight: f64,
    raw_node_dims: f64,
    budget: &CrownIbpPerNodeTimeBudget,
) -> Option<f64> {
    if !remaining_secs.is_finite()
        || remaining_secs <= 0.0
        || !remaining_work_weight_sum.is_finite()
        || remaining_work_weight_sum <= 0.0
        || !this_work_weight.is_finite()
        || this_work_weight <= 0.0
        || !raw_node_dims.is_finite()
        || raw_node_dims <= 0.0
    {
        return None;
    }

    let resolved = resolve_per_node_time_budget(budget);
    // An EXPLICIT cap (preset or env) is authoritative and keeps its dim-aware
    // scaling. Absent one, NY derives the cap from the budget it actually has
    // (#per-node-cap-from-budget) instead of the flat 12 s constant, which was
    // measured on one budget and then applied to every other.
    let cap_secs = if resolved.cap_is_explicit {
        dim_scaled_cap_secs(resolved.cap_secs, true, raw_node_dims)
    } else {
        adaptive_per_node_cap_secs(remaining_secs)
    };
    let share = remaining_secs * (this_work_weight / remaining_work_weight_sum);
    let capped = share.min(cap_secs);
    (capped >= resolved.floor_secs).then_some(capped)
}

/// Sum-of-shares budget for ONE stacked multi-target backward
/// (#cgan-stacked-backward). The stacked pass replaces each member's solo
/// walk, so it must receive the SUM of the members' weighted per-node shares —
/// handing it a single share would starve a pass doing k targets' work (the
/// stack would abort where every solo walk would have completed).
///
/// Each member's share is computed EXACTLY as its solo walk would have
/// received it (`compute_weighted_per_node_budget_secs`, including the
/// dim-scaled cap and the floor), so the stacked pass can never be granted
/// more wall than the per-target path it replaces. Members whose solo share
/// would have been refused below-floor contribute nothing — they ride along
/// for free, which is the point of stacking. The total is clamped to the
/// remaining collection window. `None` (no member clears its floor) declines
/// the lane and the historical per-target path runs unchanged.
pub(crate) fn compute_stacked_backward_budget_secs(
    remaining_secs: f64,
    remaining_work_weight_sum: f64,
    member_weights_and_cap_dims: &[(f64, f64)],
    budget: &CrownIbpPerNodeTimeBudget,
) -> Option<f64> {
    let mut total = 0.0f64;
    let mut granted = 0usize;
    for &(weight, cap_dims) in member_weights_and_cap_dims {
        if let Some(share) = compute_weighted_per_node_budget_secs(
            remaining_secs,
            remaining_work_weight_sum,
            weight,
            cap_dims,
            budget,
        ) {
            total += share;
            granted += 1;
        }
    }
    (granted > 0 && total > 0.0).then(|| total.min(remaining_secs))
}

/// Allocation result for a node against an explicitly admitted denominator.
///
/// `NotAdmitted` and `BelowFloor` are both reference-bound fallbacks for the
/// armed alpha lane. Neither may be reinterpreted as "use the full remaining
/// envelope", which would let a node absent from (or too cheap for) the
/// denominator starve every admitted target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::network::graph_alpha) enum WeightedBudgetAdmission {
    NotAdmitted,
    BelowFloor,
    Allocate(f64),
}

#[allow(clippy::too_many_arguments)]
pub(in crate::network::graph_alpha) fn admitted_weighted_budget_secs(
    admitted: bool,
    remaining_secs: f64,
    admitted_weight_sum: f64,
    this_work_weight: f64,
    cap_dims: f64,
    budget: &CrownIbpPerNodeTimeBudget,
) -> WeightedBudgetAdmission {
    if !admitted {
        return WeightedBudgetAdmission::NotAdmitted;
    }
    compute_weighted_per_node_budget_secs(
        remaining_secs,
        admitted_weight_sum,
        this_work_weight,
        cap_dims,
        budget,
    )
    .map_or(
        WeightedBudgetAdmission::BelowFloor,
        WeightedBudgetAdmission::Allocate,
    )
}

/// Admission safety margin for collector walk estimates (#cprime-admission):
/// refuse a walk only when `estimate x 5/4` exceeds the budget it would run
/// under. Matches the forward-linear cold-build gate's margin
/// (`FORWARD_LINEAR_ADMISSION_MARGIN_*`), and for the same reason: the
/// expensive failure is the walk that runs to its deadline and returns
/// nothing, so a marginal admit must still be expected to FINISH.
const WALK_ADMISSION_MARGIN: f64 = 1.25;

/// Prior correction κ0 applied to `macs / fl_rate` before the first completed
/// walk calibrates it (#cprime-admission).
///
/// The rate basis is the forward-linear probe (`forward_linear_measured_rate`,
/// value-GEMM through the production dispatch chain, already derated 0.8).
/// The collector's walks run a DIFFERENT effective class — the chunked CPU
/// transpose-GEMM (`ops_transpose_gemm`, row chunk 256, faer inner) — which
/// the July-30 census measured at ~40 GFLOP/s ≈ 20 GMAC/s on the M5 class
/// while the FL probe reports ~11.6 GMAC/s on the same host
/// (`docs/FL_FIRST_MEASUREMENT_2026-08-02.md`), i.e. the collector runs
/// ~1.7-2x FASTER than the probe's rate. κ0 = 0.5 encodes that ratio on the
/// OPTIMISTIC side: before calibration the estimator under-predicts cost, so
/// it can only refuse walks that are far past their share (the 150 s/185 s
/// zero-yield exhibits are 7x+ over) and can never refuse a walk today's
/// policy would have completed — the "never worse than today" invariant.
const WALK_RATE_PRIOR_CORRECTION: f64 = 0.5;

/// Calibration floors: a completed walk teaches the model only when both the
/// raw estimate and the measured wall are large enough that per-walk overhead
/// (seed setup, tiny targets) does not dominate the ratio.
const WALK_CALIBRATION_MIN_SECS: f64 = 0.05;
/// κ is clamped so one degenerate measurement cannot poison every later
/// admission decision in the collection.
const WALK_CORRECTION_CLAMP: (f64, f64) = (0.05, 20.0);

/// Per-collection walk cost model (#cprime-admission): a MACs-based estimate
/// divided by the measured forward-linear rate, times a correction factor
/// that starts at the census-derived prior and is REPLACED by the
/// estimate-vs-actual ratio of the first completed walk in this collection
/// (self-calibrating; deterministic given the same graph, rate, and first
/// measurement).
#[derive(Debug, Clone)]
pub(crate) struct WalkCostModel {
    macs_per_sec: f64,
    correction: f64,
    calibrated: bool,
}

impl WalkCostModel {
    pub(crate) fn new(macs_per_sec: f64) -> Self {
        Self {
            macs_per_sec,
            correction: WALK_RATE_PRIOR_CORRECTION,
            calibrated: false,
        }
    }

    /// Whether the first completed walk already replaced the prior.
    pub(crate) fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// The correction factor currently in force (prior until calibrated).
    pub(crate) fn correction(&self) -> f64 {
        self.correction
    }

    /// Estimated wall seconds for a walk of `macs` MACs; `None` when the rate
    /// is unusable (a `None` estimate ADMITS — fail-open to today's policy).
    pub(crate) fn estimate_secs(&self, macs: u128) -> Option<f64> {
        if !self.macs_per_sec.is_finite() || self.macs_per_sec < 1.0 {
            return None;
        }
        let est = (macs as f64 / self.macs_per_sec) * self.correction;
        est.is_finite().then_some(est)
    }

    /// One-shot self-calibration from the FIRST completed walk: the measured
    /// wall replaces the prior for every subsequent target in this
    /// collection. Later completions are ignored (the first measurement keeps
    /// the admission set a function of one observation, not of ordering
    /// noise).
    pub(crate) fn observe_completed_walk(&mut self, macs: u128, actual_secs: f64) {
        if self.calibrated || !actual_secs.is_finite() || actual_secs < WALK_CALIBRATION_MIN_SECS {
            return;
        }
        if !self.macs_per_sec.is_finite() || self.macs_per_sec < 1.0 {
            return;
        }
        let raw = macs as f64 / self.macs_per_sec;
        if !raw.is_finite() || raw < WALK_CALIBRATION_MIN_SECS {
            return;
        }
        self.correction =
            (actual_secs / raw).clamp(WALK_CORRECTION_CLAMP.0, WALK_CORRECTION_CLAMP.1);
        self.calibrated = true;
    }

    /// Calibrate from an ABORTED walk's in-situ rate sample (#cprime-abort-calib,
    /// 2026-08-03).
    ///
    /// [`Self::observe_completed_walk`] is the only calibration path, so a
    /// collection where NO walk ever completes keeps the optimistic census
    /// prior forever and admits everything. That is not a corner case — it is
    /// cgan_2023's steady state: MEASURED `admitted=61 refused=0` while
    /// `#chunk-abort` killed every walk, because the four ConvTranspose/Conv
    /// targets each project 170-2,226 s against a 150 s cap and none can
    /// finish. The prior then under-prices them ~20x and c′ never refuses.
    ///
    /// `#chunk-abort` already measures exactly what is needed before it gives
    /// up — "576 of 28800 rows in 5.6s" — a direct rate sample on the very
    /// walk being priced. Scaling it to the full row count yields the walk's
    /// projected cost, which is a SOUNDER calibration input than a completed
    /// walk's total (it excludes the node's non-walk overhead).
    ///
    /// Kept strictly weaker than the completed-walk path: it never overwrites
    /// a real calibration, and it only ever moves the correction UPWARD (more
    /// pessimistic ⇒ more refusals). An abort sample can only be evidence that
    /// walks cost MORE than assumed, never less, so a downward move would be
    /// unsupported by the observation.
    pub(crate) fn observe_aborted_walk(&mut self, macs: u128, projected_secs: f64) {
        if self.calibrated || !projected_secs.is_finite() {
            return;
        }
        if !self.macs_per_sec.is_finite() || self.macs_per_sec < 1.0 {
            return;
        }
        let raw = macs as f64 / self.macs_per_sec;
        if !raw.is_finite() || raw < WALK_CALIBRATION_MIN_SECS {
            return;
        }
        let observed =
            (projected_secs / raw).clamp(WALK_CORRECTION_CLAMP.0, WALK_CORRECTION_CLAMP.1);
        if observed > self.correction {
            self.correction = observed;
        }
    }
}

/// Outcome of the estimate-then-refuse admission check (#cprime-admission).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum WalkAdmissionDecision {
    /// Estimate (with margin) fits the node's share: start the walk under the
    /// existing per-node deadline, unchanged.
    Admit,
    /// LAST demanded candidate whose estimate exceeds its capped share but
    /// fits the collection's full remaining time: start the walk with the
    /// accumulated rollover (deadline extended to the collection deadline).
    /// Monopolization is meaningless with no candidate after it, and today's
    /// policy would burn the capped share for nothing.
    AdmitWithRollover,
    /// Estimate (with margin) exceeds every budget this walk could run under:
    /// do not start it. The share is NOT consumed — it rolls forward
    /// structurally, because later shares divide the unspent remaining time.
    Refuse {
        estimated_secs: f64,
        budget_secs: f64,
    },
    /// #walk-value-record: a measured completion or cooperative-abort
    /// full-walk projection for this exact (node, rows) exceeds the static
    /// weighted share but fits the collection's remaining time. The budgeter
    /// grants the recorded estimate (with the standard margin), bounded by the
    /// collection deadline, instead of refusing at the static share. Later
    /// candidates lose only the granted time; a node that still fails to
    /// finish falls back to IBP exactly as today (widening-only on bounds
    /// authority).
    AdmitWithMeasuredGrant { grant_secs: f64 },
}

/// Pure admission arithmetic (#cprime-admission): compare the padded estimate
/// against the node's share, and — for the last demanded candidate — against
/// the collection's full remaining time. Deterministic: a pure function of
/// its inputs, so the same graph and the same rate produce the same admission
/// set.
pub(crate) fn admit_walk_with_estimate(
    estimated_secs: f64,
    share_secs: f64,
    remaining_secs: f64,
    is_last_candidate: bool,
) -> WalkAdmissionDecision {
    if !estimated_secs.is_finite() || estimated_secs <= 0.0 {
        return WalkAdmissionDecision::Admit; // fail-open: unusable estimate
    }
    let padded = estimated_secs * WALK_ADMISSION_MARGIN;
    if padded <= share_secs {
        return WalkAdmissionDecision::Admit;
    }
    if is_last_candidate && padded <= remaining_secs {
        return WalkAdmissionDecision::AdmitWithRollover;
    }
    WalkAdmissionDecision::Refuse {
        estimated_secs,
        budget_secs: if is_last_candidate {
            remaining_secs
        } else {
            share_secs
        },
    }
}

/// Record-aware admission (#walk-value-record): [`admit_walk_with_estimate`],
/// upgraded by a measured record when one exists for this exact (node, rows).
/// A completed wall is authoritative; when no completion exists, a full-walk
/// projection captured at a cooperative chunk abort is the conservative
/// estimate for an otherwise-unmodeled walk.
///
/// Layering keeps the fallback identity exact: with no measured record
/// this function IS `admit_walk_with_estimate` — bit-identical decisions, so
/// a collection with no records behaves exactly as today. A measurement can
/// only convert a `Refuse` into a bounded grant; it can never revoke an
/// `Admit` (a walk today's policy would start still starts — "never worse
/// than today", the same invariant the κ0 prior encodes).
///
/// Rationale for the grant: `PerNodeDeadlineExceeded` on a node whose record
/// shows a completed walk at ~2x its static share means the SHARE was wrong,
/// not the walk (the c9f04d1c duplicate-collection anatomy re-collects the
/// same nodes inside one row, so the second attempt has a real price for the
/// first attempt's work). Granting `measured x margin`, bounded by the
/// collection's remaining time, converts a guaranteed IBP fallback into a
/// likely CROWN completion. Pure arithmetic — deterministic and CPU-testable.
pub(crate) fn admit_walk_with_record(
    estimated_secs: f64,
    recorded_secs: Option<f64>,
    share_secs: f64,
    remaining_secs: f64,
    is_last_candidate: bool,
) -> WalkAdmissionDecision {
    let base = admit_walk_with_estimate(
        estimated_secs,
        share_secs,
        remaining_secs,
        is_last_candidate,
    );
    let Some(measured) = recorded_secs else {
        return base;
    };
    if !matches!(base, WalkAdmissionDecision::Refuse { .. }) {
        return base;
    }
    if !measured.is_finite() || measured <= 0.0 {
        return base;
    }
    let padded = measured * WALK_ADMISSION_MARGIN;
    if padded <= remaining_secs {
        return WalkAdmissionDecision::AdmitWithMeasuredGrant { grant_secs: padded };
    }
    base
}

/// Per-node measured walk cost record (#walk-value-record).
///
/// The b7c9d3c3 value model put each ROOT PHASE's measured cost beside its
/// realised yield; this carries the same cost/yield discipline one level down
/// to the per-node CROWN collector walks, where the b7c9d3c3 record cannot see
/// (a root phase runs once per instance; a collector node is re-priced on
/// every duplicate collection inside one row). It is the walk-granularity face
/// of the same record family, not a parallel mechanism: written at the exact
/// sites that already feed `WalkCostModel` calibration, consulted only by
/// admission arithmetic.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NodeWalkRecord {
    /// Seeded row count of the recorded walks. A record is consulted only when
    /// the current walk seeds the SAME row count, so a subset-rows walk can
    /// never be priced by its full-width twin (or vice versa).
    pub(crate) rows: usize,
    /// Wall seconds of the SLOWEST completed walk for this (node, rows) — the
    /// conservative price a grant must cover.
    pub(crate) completed_secs: Option<f64>,
    /// Largest #chunk-abort full-walk projection observed. Evidence that the
    /// walk costs at least this much; used to price otherwise-unmodeled
    /// targets (ConvTranspose class) that the MAC proxy admits blind.
    pub(crate) aborted_projection_secs: Option<f64>,
}

impl NodeWalkRecord {
    /// Measured estimate for admission: a real completion beats a projection,
    /// and either beats the forward-linear MAC proxy at the call site.
    pub(crate) fn estimate_secs(&self) -> Option<f64> {
        self.completed_secs.or(self.aborted_projection_secs)
    }
}

std::thread_local! {
    /// Thread-local on purpose: the collector runs on the verifier's thread,
    /// and duplicate collections within one row re-enter on that same thread.
    /// A record lost to a thread change degrades exactly to today's behavior
    /// (fail-open), and `cargo test`'s thread-per-test isolation keeps the
    /// pinned collection tests independent of execution order.
    static NODE_WALK_RECORDS: std::cell::RefCell<HashMap<String, NodeWalkRecord>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Exact opt-out: `NY_NO_WALK_RECORD_ADMISSION=1` stops CONSULTING records
/// (recording is passive and always on). Default engaged, mirroring the other
/// admission-arithmetic levers (#cprime-admission, hopeless-class skip).
const fn walk_record_admission_enabled_from_disabled(disabled: bool) -> bool {
    !disabled
}

fn walk_record_admission_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        walk_record_admission_enabled_from_disabled(
            ny_levers::read(&ny_levers::decls::collection::NO_WALK_RECORD_ADMISSION)
                .value
                .as_bool(),
        )
    })
}

/// Consult the record for `node`, gated on an exact row-count match.
pub(crate) fn node_walk_record(node: &str, rows: usize) -> Option<NodeWalkRecord> {
    if !walk_record_admission_enabled() {
        return None;
    }
    NODE_WALK_RECORDS.with(|slot| {
        slot.borrow()
            .get(node)
            .copied()
            .filter(|record| record.rows == rows)
    })
}

/// Record a COMPLETED walk's measured wall seconds for (node, rows).
///
/// A row-count change (route replan between collections) resets the record:
/// the newest geometry is the one later collections will execute. Walls below
/// [`WALK_CALIBRATION_MIN_SECS`] are not recorded — the same floor the
/// `WalkCostModel` uses, and for the same reason: per-walk overhead dominates
/// a sub-50ms measurement, and a degenerate record could grant a doomed
/// sub-share deadline where today's policy refuses cleanly.
pub(crate) fn record_node_walk_completed(node: &str, rows: usize, secs: f64) {
    if !secs.is_finite() || secs < WALK_CALIBRATION_MIN_SECS {
        return;
    }
    NODE_WALK_RECORDS.with(|slot| {
        let mut map = slot.borrow_mut();
        let record = map.entry(node.to_string()).or_default();
        if record.rows != rows {
            *record = NodeWalkRecord {
                rows,
                ..NodeWalkRecord::default()
            };
        }
        record.completed_secs = Some(record.completed_secs.map_or(secs, |prev| prev.max(secs)));
    });
}

/// Record a #chunk-abort full-walk projection for (node, rows). The same
/// [`WALK_CALIBRATION_MIN_SECS`] floor applies.
pub(crate) fn record_node_walk_abort_projection(node: &str, rows: usize, secs: f64) {
    if !secs.is_finite() || secs < WALK_CALIBRATION_MIN_SECS {
        return;
    }
    NODE_WALK_RECORDS.with(|slot| {
        let mut map = slot.borrow_mut();
        let record = map.entry(node.to_string()).or_default();
        if record.rows != rows {
            *record = NodeWalkRecord {
                rows,
                ..NodeWalkRecord::default()
            };
        }
        record.aborted_projection_secs = Some(
            record
                .aborted_projection_secs
                .map_or(secs, |prev| prev.max(secs)),
        );
    });
}

/// Test hook: clear this thread's records so a pinned scenario starts empty.
#[cfg(test)]
pub(crate) fn reset_node_walk_records() {
    NODE_WALK_RECORDS.with(|slot| slot.borrow_mut().clear());
}

/// Flight-recorder snapshot of collector walk-admission activity in this
/// process (#cprime-admission, I7): refusal/admission counts plus the numbers
/// of the most recent refusal, mirrored into the CLI sidecar the same way the
/// forward-linear admission record is.
#[derive(Debug, Clone, Default)]
pub struct CollectorWalkAdmissionRecord {
    /// Walks refused upfront (share would have been burned for nothing).
    pub refused: u32,
    /// Walks admitted after an estimate was computed.
    pub admitted: u32,
    /// Last-candidate rollover grants (deadline extended to the collection's).
    pub rollover_grants: u32,
    /// #walk-value-record grants (deadline extended to the recorded completion
    /// or full-walk projection, bounded by the collection deadline).
    pub measured_grants: u32,
    /// Estimated seconds saved by refusals: the sum of each refused node's
    /// share at refusal time — the time today's policy would have burned.
    pub reclaimed_share_secs: f64,
    /// Most recent refusal, for the sidecar line.
    pub last_refused_node: Option<String>,
    pub last_refused_estimate_secs: f64,
    pub last_refused_share_secs: f64,
    /// Rate basis and the correction in force at the last decision.
    pub macs_per_sec: u64,
    pub correction: f64,
    pub calibrated: bool,
}

static COLLECTOR_WALK_ADMISSION: std::sync::Mutex<Option<CollectorWalkAdmissionRecord>> =
    std::sync::Mutex::new(None);

/// Read the process-global walk-admission record (`None` until a
/// deadline-carrying collection consulted the estimator).
pub fn collector_walk_admission_record() -> Option<CollectorWalkAdmissionRecord> {
    COLLECTOR_WALK_ADMISSION
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
}

pub(crate) fn note_walk_admission(
    model: &WalkCostModel,
    node_name: &str,
    decision: WalkAdmissionDecision,
) {
    let Ok(mut guard) = COLLECTOR_WALK_ADMISSION.lock() else {
        return;
    };
    let record = guard.get_or_insert_with(CollectorWalkAdmissionRecord::default);
    record.macs_per_sec = model.macs_per_sec as u64;
    record.correction = model.correction;
    record.calibrated = model.calibrated;
    match decision {
        WalkAdmissionDecision::Admit => record.admitted += 1,
        WalkAdmissionDecision::AdmitWithRollover => {
            record.admitted += 1;
            record.rollover_grants += 1;
        }
        WalkAdmissionDecision::AdmitWithMeasuredGrant { .. } => {
            record.admitted += 1;
            record.measured_grants += 1;
        }
        WalkAdmissionDecision::Refuse {
            estimated_secs,
            budget_secs,
        } => {
            record.refused += 1;
            record.reclaimed_share_secs += budget_secs.max(0.0);
            record.last_refused_node = Some(node_name.to_string());
            record.last_refused_estimate_secs = estimated_secs;
            record.last_refused_share_secs = budget_secs;
        }
    }
}

impl GraphNetwork {
    /// MACs-based cost estimate of one collector backward walk
    /// (#cprime-admission): `2 · rows · Σ per_row(weighted prefix node)`.
    ///
    /// The walk seeds a `rows`-wide coefficient pair at the target and
    /// composes it backward through every ancestor (the target's own layer
    /// included — `ancestors()` contains the target). Each weighted node
    /// costs one GEMM per coefficient matrix: composing an `[rows x
    /// out_numel]` carrier through a Conv2d touches `contraction = in_c·kh·kw`
    /// inputs per output element, i.e. `rows · out_h·out_w·out_c·contraction`
    /// MACs — the SAME per-row unit `forward_linear_cold_build_macs` prices
    /// with, so the forward-linear measured rate is the right denominator.
    /// x2 for the lower/upper pair. Elementwise nodes (ReLU, Add, BatchNorm,
    /// …) are O(rows · width) and negligible against the GEMMs (census:
    /// per-step GEMMs are ~97% of pass cost).
    ///
    /// UNDER-estimates by construction on unmodeled weighted ops
    /// (ConvTranspose, MatMul, attention): those contribute nothing to the
    /// sum, so their targets are ADMITTED — fail-open to today's policy,
    /// never a new refusal class. `None` when no weighted ancestor exists or
    /// the graph lacks shape info (also admit).
    /// `node_shapes` supplies each ancestor's PROPAGATED input geometry.
    ///
    /// #cprime-shape-source (2026-08-03): this used to read
    /// `layer.input_shape`, which the pipeline never populates on the stored
    /// layer — `layers/layer_enum/dispatch.rs:356,370` sets the shape on a
    /// `c.clone()` and propagates through the clone, and the graph builder
    /// never sets it at all. Every ancestor therefore `continue`d, the
    /// function returned `None`, and that silently disabled walk admission,
    /// completed-walk calibration AND abort calibration together. MEASURED on
    /// cgan_2023: `admitted=61 refused=0` while `#chunk-abort` killed every
    /// walk. The bug is data-dependent — cifar100 refuses correctly
    /// (`admitted=2 refused=4`) because some of its convs carry ONNX-declared
    /// shapes — which is why it hid on exactly the highest-value block.
    ///
    /// Propagated bounds are the authoritative geometry the walk will actually
    /// see, so they are the right source. `layer.input_shape` remains the
    /// fallback for callers without a shape map (tests construct layers with
    /// it set explicitly).
    pub(crate) fn collector_walk_macs_with_shapes(
        &self,
        target: &str,
        rows: usize,
        node_shapes: Option<&HashMap<String, BoundedTensor>>,
    ) -> Option<u128> {
        self.collector_walk_macs_impl(target, rows, node_shapes)
    }

    #[cfg(test)]
    pub(crate) fn collector_walk_macs(&self, target: &str, rows: usize) -> Option<u128> {
        self.collector_walk_macs_impl(target, rows, None)
    }

    fn collector_walk_macs_impl(
        &self,
        target: &str,
        rows: usize,
        node_shapes: Option<&HashMap<String, BoundedTensor>>,
    ) -> Option<u128> {
        let rows = u128::try_from(rows).ok()?;
        if rows == 0 {
            return None;
        }
        let ancestors = self.ancestors(target).ok()?;
        let mut total: u128 = 0;
        let mut saw_weighted = false;
        // Propagated (H, W) for a node's own activation, when the caller
        // supplied the bounds map. Trailing two dims of a >=3D tensor.
        let shape_of = |n: &str| -> Option<(usize, usize)> {
            let bt = node_shapes?.get(n)?;
            let sh = bt.lower().shape();
            if sh.len() < 3 {
                return None;
            }
            Some((sh[sh.len() - 2], sh[sh.len() - 1]))
        };
        for name in &ancestors {
            let Some(node) = self.nodes.get(name) else {
                continue;
            };
            let per_row: u128 = match &node.layer {
                Layer::Conv2d(conv) => {
                    let Some((in_h, in_w)) = shape_of(name).or(conv.input_shape) else {
                        continue;
                    };
                    // Verify finding (#cprime-admission attack 1): the executed
                    // grouped transpose-GEMM contracts in_c/groups per group, and
                    // the layer's own output_size() accounts for dilation — the
                    // naive formula over-priced by x groups (x256 depthwise) and
                    // x1.78 on dilated convs, turning fail-open into wrongful
                    // refusals on such nets.
                    let Ok((out_h, out_w)) = conv.output_size(in_h, in_w) else {
                        continue;
                    };
                    let (kh, kw) = conv.kernel_size();
                    let in_c_per_group = if conv.kernel.ndim() == 4 {
                        conv.kernel.shape()[1]
                    } else {
                        continue;
                    };
                    let contraction = in_c_per_group.checked_mul(kh)?.checked_mul(kw)?;
                    (out_h as u128)
                        .checked_mul(out_w as u128)?
                        .checked_mul(contraction as u128)?
                        .checked_mul(conv.out_channels() as u128)?
                }
                // #cprime-convtranspose (2026-08-03): ConvTranspose was in the
                // documented "unmodeled -> contributes 0 -> admit" fail-open set,
                // and that hole is exactly where the cost lives on cgan_2023 —
                // MEASURED on cGAN_imgSz32_nCh_1 prop_1: the precheck funds
                // ConvTranspose_7 (12,544 rows, projected 452s) and
                // ConvTranspose_10 (28,800 rows, projected 2,226s) against 150s
                // per-node caps, and c′ refused ZERO walks there while
                // #chunk-abort paid 2.7-5.7s per walk to rediscover the same
                // verdict repeatedly. Pricing it lets the walk be refused before
                // it starts.
                //
                // Cost model: the transposed backward contracts over the kernel's
                // OUTPUT-channel axis per output element. Kernel layout is
                // (in_c, out_c/groups, kh, kw) — note the axis order is swapped
                // relative to Conv2d, so the per-group contraction reads
                // shape()[1], and the spatial map comes from the layer's own
                // output_size() (it accounts for stride/dilation/output_padding,
                // which a naive formula gets badly wrong on upsampling stacks).
                Layer::ConvTranspose2d(deconv) => {
                    let Some((in_h, in_w)) = shape_of(name).or(deconv.input_shape) else {
                        continue;
                    };
                    let Ok((out_h, out_w)) = deconv.output_size(in_h, in_w) else {
                        continue;
                    };
                    let (kh, kw) = deconv.kernel_size();
                    let out_c_per_group = if deconv.kernel.ndim() == 4 {
                        deconv.kernel.shape()[1]
                    } else {
                        continue;
                    };
                    let contraction = out_c_per_group.checked_mul(kh)?.checked_mul(kw)?;
                    (out_h as u128)
                        .checked_mul(out_w as u128)?
                        .checked_mul(contraction as u128)?
                        .checked_mul(deconv.in_channels() as u128)?
                }
                Layer::Linear(linear) => {
                    (linear.out_features() as u128).checked_mul(linear.in_features() as u128)?
                }
                _ => continue,
            };
            saw_weighted = true;
            total = total.checked_add(per_row.checked_mul(rows)?.checked_mul(2)?)?;
        }
        saw_weighted.then_some(total)
    }
}

pub(super) struct PatchesTighteningBudget {
    total_secs: f64,
    remaining_secs: f64,
}

/// Share of a collection's own remaining deadline the aggregate patches budget
/// may claim (#patches-budget-from-deadline).
///
/// Patches targets are the ones whose alternative is a LOOSE IBP bound, so on a
/// conv graph they are where the collection's time is best spent; the remaining
/// 30% covers the non-patches targets and per-node overheads.
const PATCHES_BUDGET_DEADLINE_SHARE: f64 = 0.70;

impl PatchesTighteningBudget {
    /// Aggregate patches budget sized from the collection's ACTUAL remaining
    /// deadline rather than a fixed constant (#patches-budget-from-deadline).
    ///
    /// WHY. The aggregate cap was a constant — 5 s normally, 40 s under
    /// `NY_CONV_PATCHES_COLLECT` — with no relation to how much time the
    /// collection actually had. Both numbers were derived from one measured
    /// network (metaroom's 4 conv stages) and then applied to every instance and
    /// every scored budget. Measured on TinyYOLO (yolo_2023) once the CPU GEMM
    /// engine made patches composition fast enough to be worth funding: the
    /// collection ran 52.7 s, the 40 s aggregate cap was exhausted, and two
    /// demanded targets reverted with `PatchesBudgetExceeded` while wall time
    /// remained.
    ///
    /// This is the same failure shape as every other fixed resource constant in
    /// this path: relieving one exposes the next. Sizing the cap as a share of
    /// the live deadline removes it from that chain — a long budget funds more
    /// patches work, a short one funds less, and neither needs a new constant.
    ///
    /// The constant is retained as a FLOOR so short-deadline and no-deadline
    /// callers behave exactly as before, and the explicit
    /// `NY_PATCHES_BUDGET_SECS` override still wins outright.
    ///
    /// Sound: this only schedules how much tightening is attempted. Every target
    /// that runs out of budget keeps its valid IBP bound.
    pub(super) fn with_collection_deadline(deadline: Option<Instant>) -> Self {
        let resolved = resolve_patches_tightening_budget();
        let total_secs = if resolved.is_explicit {
            resolved.floor_secs
        } else {
            let from_deadline = deadline
                .map(|d| {
                    d.saturating_duration_since(Instant::now()).as_secs_f64()
                        * PATCHES_BUDGET_DEADLINE_SHARE
                })
                .unwrap_or(0.0);
            from_deadline.max(resolved.floor_secs)
        };
        Self {
            total_secs,
            remaining_secs: total_secs,
        }
    }

    pub(super) fn can_start_node(&self, min_budget_secs: f64) -> bool {
        self.remaining_secs >= min_budget_secs
    }

    pub(super) fn used_secs(&self) -> f64 {
        (self.total_secs - self.remaining_secs).max(0.0)
    }

    pub(super) fn remaining_deadline(
        &self,
        is_patches_target: bool,
        min_budget_secs: f64,
    ) -> Option<Instant> {
        if is_patches_target && self.can_start_node(min_budget_secs) {
            Some(Instant::now() + Duration::from_secs_f64(self.remaining_secs))
        } else {
            None
        }
    }

    pub(super) fn record_elapsed(&mut self, is_patches_target: bool, elapsed_secs: f64) {
        if is_patches_target {
            self.remaining_secs -= elapsed_secs;
        }
    }
}

pub(super) fn merge_per_node_deadlines(
    global_deadline: Option<Instant>,
    patches_deadline: Option<Instant>,
    has_global_deadline: bool,
) -> Option<Instant> {
    match (global_deadline, patches_deadline) {
        (Some(global), Some(patches)) => Some(global.min(patches)),
        (Some(global), None) => Some(global),
        (None, Some(patches)) if !has_global_deadline => Some(patches),
        _ => None,
    }
}

impl GraphNetwork {
    /// The sequential collector cannot selectively skip large spatial targets,
    /// so the fast-path gate must conservatively count every dense-overflow
    /// target when deciding whether to reuse that collector. #3839
    pub(super) fn counts_toward_sequential_skip_fraction(
        bounds: &BoundedTensor,
        budget: usize,
    ) -> bool {
        dense_identity_exceeds_budget(bounds, budget)
    }

    /// The graph-native collector may keep spatial Conv2d targets on the
    /// patches-start path landed by #3813, so its per-target budget guard is
    /// intentionally looser than the sequential fast-path gate. #3839
    ///
    /// #patches-dense-peak — WHY THIS PREDICATE CANNOT SEE THE cifar_bias_field_46
    /// MISS, and why widening it is NOT the repair.
    ///
    /// Measured on `cifar_bias_field_46`, target `/layers.4/Relu`:
    ///
    /// ```text
    /// CPU memory exceeded at patches full dense materialization:
    /// requires 6,445,080,584 bytes but budget is 6,442,450,944
    /// ```
    ///
    /// Both terms of this predicate are FALSE there, and only one of them is the
    /// one a previous attempt tried to withdraw:
    ///
    /// * `dense_identity_exceeds_budget` charges the `[dim x dim]` f32 PAIR:
    ///   `16_384 * 16_384 * 4 * 2 = 2,147,483,648` = 2 GiB against a 6 GiB
    ///   budget. It is false by a factor of three, INDEPENDENTLY of the patches
    ///   exemption. Withdrawing `!crown_ibp_target_can_start_in_patches(..)`
    ///   therefore could not — and measurably did not — remove the refusal.
    /// * The site that actually aborts is not the identity pair at all. It is
    ///   `bounds/patches/to_dense.rs`'s full dense materialization, whose exact
    ///   peak (`dense_materialization_peak_bytes`) is
    ///   `resident + map + 6*M + bias_pair` with `M = rows * in_dim * 4`.
    ///   Here `rows = in_dim = 16_384`, so `M` is exactly 1 GiB, SIX live copies
    ///   are 6,442,450,944 = the budget exactly, and the 2,629,640-byte overflow
    ///   (map 270,344 + bias 131,072 + resident 2,228,224) is pure bookkeeping.
    ///   A 6D/6D carrier charges 6 matrices; a 7D explicit-rows carrier charges 8.
    ///
    /// So the honest cost model for a PATCHES-capable target is `6*rows*cols*f32`
    /// against the widest conv-ancestor column count the carrier can still be
    /// holding when it densifies — not `2*dim*dim*f32`. (The mid-walk pre-check
    /// `patches_densify_over_budget` in target_backward.rs has the same 3x
    /// under-charge: it consults `dense_pair_bytes`, i.e. two matrices, so it
    /// also waves the 2 GiB estimate through before the 6-matrix site fails.)
    ///
    /// Correcting the predicate here is necessary but NOT sufficient, and was
    /// deliberately not landed on its own — see the blocker recorded on
    /// `alpha_explicit::alpha_target_chunk_override`: the objective-chunked
    /// backward this predicate is supposed to steer over-budget targets into is
    /// itself refused at driver entry whenever a deadline is merely PRESENT, so
    /// a corrected predicate would only exchange one reference-bound fallback
    /// for another. Landing the cost model before that guard is expiry-decided
    /// would add chunk plans that never execute.
    pub(super) fn graph_native_target_exceeds_budget(
        &self,
        node_name: &str,
        bounds: &BoundedTensor,
        budget: usize,
    ) -> bool {
        dense_identity_exceeds_budget(bounds, budget)
            && !self.crown_ibp_target_can_start_in_patches(node_name, bounds)
    }
}

// Justification: fallback recording takes the target maps (bounds, provenance,
// events) plus the per-node context needed for a single provenance entry.
//
// NOTE: the collector's memory-budget IBP fallback recorder was removed with
// #cgan-bn11-chunk — over-budget targets now reroute through the objective-
// chunked backward instead of degrading to IBP. `MemoryBudgetExceeded` remains
// reachable via `CpuMemoryExceeded` errors surfaced by the backward itself.
#[allow(clippy::too_many_arguments)]
pub(super) fn record_patches_budget_fallback(
    crown_ibp_bounds: &mut HashMap<String, BoundedTensor>,
    provenance: &mut HashMap<String, BoundsProvenance>,
    fallback_events: &mut Vec<CrownIbpFallbackEvent>,
    node_name: &str,
    ibp_bound: &BoundedTensor,
    layer_index: usize,
    layer_type: &str,
    node_dim: usize,
    used_secs: f64,
) {
    crown_ibp_bounds.insert(node_name.to_string(), ibp_bound.clone());
    provenance.insert(
        node_name.to_string(),
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::PatchesBudgetExceeded),
    );
    fallback_events.push(CrownIbpFallbackEvent {
        layer_index,
        layer_type: layer_type.to_string(),
        reason: CrownIbpFallbackReason::PatchesBudgetExceeded,
        details: format!(
            "node '{}' dim={} patches budget exhausted after {:.3}s",
            node_name, node_dim, used_secs
        ),
    });
}

#[cfg(test)]
mod tests {
    use super::super::target_backward::{objective_chunk_route_plan, ObjectiveChunkFixedWavePlan};
    use super::{
        admitted_weighted_budget_secs, compute_global_per_node_budget_secs,
        compute_stacked_backward_budget_secs, compute_weighted_per_node_budget_secs,
        count_remaining_budget_candidates, crown_chunk_aware_budget_from_raw,
        demanded_target_work_weight, dim_cap_scale_from_raw, effective_per_node_time_budget,
        objective_fixed_wave_work_weight, prefix_cost_admission_enabled_from_raw,
        resolve_patches_tightening_budget_from_raw, resolve_per_node_time_budget_from_raw,
        sum_remaining_budget_weights, weighted_budget_cap_dims, CrownIbpPerNodeTimeBudget,
        ObjectiveChunkSchedulingPlan, PrefixCostAdmission, PrefixCostAdmissionModel,
        WeightedBudgetAdmission, ADAPTIVE_PER_NODE_CAP_FLOOR_SECS, MIN_PER_NODE_BUDGET_SECS,
        PRESET_CAP_REFERENCE_DIMS,
    };
    use std::time::Duration;

    const DEFAULT_BUDGET: CrownIbpPerNodeTimeBudget = CrownIbpPerNodeTimeBudget {
        floor_secs: None,
        cap_secs: None,
    };

    /// #cgan-stacked-backward: the stacked pass receives the SUM of its
    /// members' solo shares — never a single share, never more than the
    /// remaining window, and exactly zero grant when no member clears the
    /// floor.
    #[test]
    fn stacked_budget_is_the_sum_of_member_solo_shares() {
        let remaining = 300.0;
        let total_weight = 100.0;
        let members = [(30.0, 4_096.0), (20.0, 2_048.0), (10.0, 1_024.0)];
        let expected: f64 = members
            .iter()
            .filter_map(|&(weight, dims)| {
                compute_weighted_per_node_budget_secs(
                    remaining,
                    total_weight,
                    weight,
                    dims,
                    &DEFAULT_BUDGET,
                )
            })
            .sum();
        assert!(expected > 0.0, "fixture members must clear the floor");
        let granted = compute_stacked_backward_budget_secs(
            remaining,
            total_weight,
            &members,
            &DEFAULT_BUDGET,
        )
        .expect("members above floor must be granted");
        assert_eq!(granted, expected.min(remaining));
        // A single member's share must be strictly smaller than the stack's.
        let solo = compute_weighted_per_node_budget_secs(
            remaining,
            total_weight,
            members[0].0,
            members[0].1,
            &DEFAULT_BUDGET,
        )
        .expect("solo share");
        assert!(granted > solo, "the stack must get MORE than one share");
    }

    #[test]
    fn stacked_budget_clamps_to_the_remaining_window_and_floors_out() {
        // One member owning ~the whole weight: its solo share is ~remaining;
        // the sum must clamp at the window, never exceed it.
        let remaining = 50.0;
        let members = [(99.0, 4_096.0), (1.0, 512.0)];
        let granted =
            compute_stacked_backward_budget_secs(remaining, 100.0, &members, &DEFAULT_BUDGET)
                .expect("granted");
        assert!(granted <= remaining);
        // Every member below the 2 s floor: the lane must decline.
        let starved = [(0.001, 512.0), (0.001, 512.0)];
        assert_eq!(
            compute_stacked_backward_budget_secs(remaining, 100.0, &starved, &DEFAULT_BUDGET),
            None
        );
        // No members at all: decline.
        assert_eq!(
            compute_stacked_backward_budget_secs(remaining, 100.0, &[], &DEFAULT_BUDGET),
            None
        );
    }

    #[test]
    fn prefix_cost_admission_enable_parser_is_exact_and_default_dark() {
        assert!(!prefix_cost_admission_enabled_from_raw(None));
        assert!(prefix_cost_admission_enabled_from_raw(Some("1")));
        for raw in ["", "0", "true", "01", " 1", "2"] {
            assert!(
                !prefix_cost_admission_enabled_from_raw(Some(raw)),
                "raw={raw:?}"
            );
        }
    }

    #[test]
    fn prefix_cost_admission_requires_two_completed_samples_and_a_deadline() {
        let mut model = PrefixCostAdmissionModel::new(true);
        let expensive = Some(1_000u128);

        assert_eq!(
            model.admit(expensive, Some(Duration::from_secs(1))),
            PrefixCostAdmission::RunWithoutEstimate,
            "no observation must prefer a false negative (run) over a false-positive skip"
        );
        model.observe_completed(Some(100), Duration::from_secs(10));
        assert_eq!(
            model.admit(expensive, Some(Duration::from_secs(1))),
            PrefixCostAdmission::RunWithoutEstimate,
            "one potentially cold completion is not scheduling authority"
        );
        model.observe_completed(Some(200), Duration::from_secs(10));
        assert!(matches!(
            model.admit(expensive, None),
            PrefixCostAdmission::RunWithoutEstimate
        ));
    }

    #[test]
    fn prefix_cost_admission_exact_boundary_runs_and_one_nanosecond_short_refuses() {
        let mut model = PrefixCostAdmissionModel::new(true);
        // Fastest completed rate is 100 work/s. Admission gives a prospective
        // walk an extra 2x optimism, so 1,000 work predicts exactly 5 seconds.
        model.observe_completed(Some(100), Duration::from_secs(1));
        model.observe_completed(Some(200), Duration::from_secs(2));

        assert!(matches!(
            model.admit(Some(1_000), Some(Duration::from_secs(5))),
            PrefixCostAdmission::RunEstimated {
                predicted_secs: 5.0,
                remaining_secs: 5.0,
                completed_samples: 2,
            }
        ));
        assert!(matches!(
            model.admit(
                Some(1_000),
                Some(Duration::from_secs(5) - Duration::from_nanos(1)),
            ),
            PrefixCostAdmission::RetainIbp { .. }
        ));
    }

    #[test]
    fn prefix_cost_admission_uses_fastest_sample_and_invalid_inputs_fail_open() {
        let mut model = PrefixCostAdmissionModel::new(true);
        model.observe_completed(Some(100), Duration::from_secs(2)); // 50 work/s
        model.observe_completed(Some(100), Duration::from_secs(1)); // 100 work/s
        model.observe_completed(None, Duration::from_secs(1));
        model.observe_completed(Some(0), Duration::from_secs(1));
        model.observe_completed(Some(1), Duration::ZERO);

        // At the slower sample this would predict 10s after the 2x optimism
        // and refuse. The fastest authenticated sample predicts 5s and runs.
        assert!(matches!(
            model.admit(Some(1_000), Some(Duration::from_secs(6))),
            PrefixCostAdmission::RunEstimated {
                completed_samples: 2,
                ..
            }
        ));
        assert_eq!(
            model.admit(None, Some(Duration::from_secs(1))),
            PrefixCostAdmission::RunWithoutEstimate
        );
        assert_eq!(
            model.admit(Some(0), Some(Duration::from_secs(1))),
            PrefixCostAdmission::RunWithoutEstimate
        );

        let disabled = PrefixCostAdmissionModel::new(false);
        assert_eq!(
            disabled.admit(Some(u128::MAX), Some(Duration::ZERO)),
            PrefixCostAdmission::RunWithoutEstimate
        );

        let mut overflowing = PrefixCostAdmissionModel::new(true);
        overflowing.observe_completed(Some(1), Duration::from_nanos(2));
        overflowing.observe_completed(Some(1), Duration::from_nanos(2));
        assert_eq!(
            overflowing.admit(Some(u128::MAX), Some(Duration::from_secs(1))),
            PrefixCostAdmission::RunWithoutEstimate,
            "an unrepresentable exact ratio must run, never acquire skip authority"
        );

        let mut unrankable_sample = PrefixCostAdmissionModel::new(true);
        unrankable_sample.observe_completed(Some(u128::MAX), Duration::from_nanos(1));
        unrankable_sample.observe_completed(Some(u128::MAX), Duration::from_nanos(2));
        assert_eq!(
            unrankable_sample.admit(Some(1), Some(Duration::ZERO)),
            PrefixCostAdmission::RunWithoutEstimate,
            "overflow while ranking completed rates must disable admission for the collection"
        );
    }

    struct DefaultBudgetEnv {
        _guards: [ny_test_utils::env::ScopedEnvVar; 3],
        _lock: std::sync::RwLockWriteGuard<'static, ()>,
    }

    /// Production budget helpers intentionally read operator env overrides.
    /// Tests that assert the built-in/preset policy therefore participate in
    /// the shared env lock and pin those overrides absent.
    fn default_budget_env() -> DefaultBudgetEnv {
        let lock = ny_test_utils::env::lock_env();
        let guards = [
            ny_test_utils::env::ScopedEnvVar::unset("NY_PER_NODE_CAP_SECS"),
            ny_test_utils::env::ScopedEnvVar::unset("NY_PER_NODE_FLOOR_SECS"),
            ny_test_utils::env::ScopedEnvVar::unset("NY_DIM_CAP_SCALE"),
        ];
        DefaultBudgetEnv {
            _guards: guards,
            _lock: lock,
        }
    }

    /// Cache identity must use the resolved policy, including whether the cap
    /// is fixed or adaptive. An explicit 12 s cap and the adaptive default have
    /// equal base numbers but different scheduling behavior.
    #[test]
    fn resolved_per_node_policy_distinguishes_explicit_cap_without_mutating_env() {
        let adaptive = resolve_per_node_time_budget_from_raw(&DEFAULT_BUDGET, None, None);
        let fixed_same_value =
            resolve_per_node_time_budget_from_raw(&DEFAULT_BUDGET, None, Some("12"));
        assert_eq!(adaptive.floor_secs, fixed_same_value.floor_secs);
        assert_eq!(adaptive.cap_secs, fixed_same_value.cap_secs);
        assert!(!adaptive.cap_is_explicit);
        assert!(fixed_same_value.cap_is_explicit);
        assert_ne!(adaptive, fixed_same_value);

        let overridden =
            resolve_per_node_time_budget_from_raw(&DEFAULT_BUDGET, Some("3.5"), Some("19"));
        assert_eq!(overridden.floor_secs, 3.5);
        assert_eq!(overridden.cap_secs, 19.0);
        assert!(overridden.cap_is_explicit);

        let preset = CrownIbpPerNodeTimeBudget {
            floor_secs: Some(1.25),
            cap_secs: Some(150.0),
        };
        assert_eq!(
            resolve_per_node_time_budget_from_raw(&preset, Some("invalid"), Some("-1")),
            resolve_per_node_time_budget_from_raw(&preset, None, None),
            "invalid env text must resolve to the preset, not create a distinct policy"
        );
    }

    /// The aggregate patches budget has the same explicit-vs-adaptive
    /// distinction: an explicit value fixes the total while the built-in value
    /// is only a floor under the deadline-proportional policy.
    #[test]
    fn resolved_patches_policy_distinguishes_fixed_floor_without_mutating_env() {
        let adaptive = resolve_patches_tightening_budget_from_raw(None, true);
        let fixed_same_value = resolve_patches_tightening_budget_from_raw(Some("40"), true);
        assert_eq!(adaptive.floor_secs, fixed_same_value.floor_secs);
        assert!(!adaptive.is_explicit);
        assert!(fixed_same_value.is_explicit);
        assert_ne!(adaptive, fixed_same_value);

        assert_eq!(
            resolve_patches_tightening_budget_from_raw(Some("invalid"), false),
            resolve_patches_tightening_budget_from_raw(None, false)
        );
        assert_ne!(
            resolve_patches_tightening_budget_from_raw(None, true),
            resolve_patches_tightening_budget_from_raw(None, false),
            "the conv-patches gate changes both routing and the built-in floor"
        );
    }

    #[test]
    fn dim_cap_scale_parser_is_pure_and_exact() {
        assert!(dim_cap_scale_from_raw(None));
        assert!(!dim_cap_scale_from_raw(Some("0")));
        for raw in ["", "00", "false", "1"] {
            assert!(dim_cap_scale_from_raw(Some(raw)), "raw={raw:?}");
        }
    }

    fn fixed_wave_plan(
        requested_rows: usize,
        chunk_rows: usize,
        chunk_count: usize,
        wave_size: usize,
        wave_count: usize,
    ) -> ObjectiveChunkSchedulingPlan {
        ObjectiveChunkSchedulingPlan {
            execution: objective_chunk_route_plan(requested_rows, false, false),
            fixed_waves: ObjectiveChunkFixedWavePlan {
                chunk_rows,
                chunk_count,
                wave_size,
                wave_count,
            },
        }
    }

    /// #per-node-cap-from-budget CHANGED THIS EXPECTATION DELIBERATELY.
    ///
    /// This used to assert that a lone candidate with 210 s remaining got the
    /// flat 12 s cap. That was the defect: 12 s is 5.7% of the collection's own
    /// remaining time, so the single node it was protecting against did not
    /// exist, and the node that WOULD have finished was truncated anyway. The
    /// cap is now a share of what is actually left (210 * 0.25 = 52.5 s), which
    /// still bounds monopolization — three more nodes could each claim an equal
    /// share — without inheriting a number measured on a different budget.
    #[test]
    fn test_compute_global_per_node_budget_secs_caps_long_deadline_4413() {
        let _env = default_budget_env();
        assert_eq!(
            compute_global_per_node_budget_secs(210.0, 1, &DEFAULT_BUDGET),
            Some(52.5)
        );
        // Short budgets keep the historical constant exactly: 24 s remaining
        // yields a 12 s cap (24 * 0.25 = 6, floored at 12).
        assert_eq!(
            compute_global_per_node_budget_secs(24.0, 1, &DEFAULT_BUDGET),
            Some(ADAPTIVE_PER_NODE_CAP_FLOOR_SECS)
        );
    }

    /// The derived cap must bound monopolization, honor the historical floor on
    /// short budgets, and stay under the ceiling on very long ones.
    #[test]
    fn test_adaptive_per_node_cap_tracks_budget_per_node_cap_from_budget() {
        use super::{adaptive_per_node_cap_secs, DIM_SCALED_CAP_CEILING_SECS};

        // Short budget: the historical 12 s constant is the floor, so nothing
        // regresses where the old cap was already the binding one.
        assert_eq!(
            adaptive_per_node_cap_secs(20.0),
            ADAPTIVE_PER_NODE_CAP_FLOOR_SECS
        );
        assert_eq!(
            adaptive_per_node_cap_secs(48.0),
            ADAPTIVE_PER_NODE_CAP_FLOOR_SECS
        );

        // Long budget: a quarter share, so at least three more nodes can each
        // claim as much before the collection is exhausted.
        assert_eq!(adaptive_per_node_cap_secs(100.0), 25.0);
        assert_eq!(adaptive_per_node_cap_secs(900.0), 225.0);

        // Very long budget: clamped, so this stays a guard and not a grant.
        assert_eq!(
            adaptive_per_node_cap_secs(10_000.0),
            DIM_SCALED_CAP_CEILING_SECS
        );

        // Degenerate inputs fall back to the constant rather than 0 or NaN.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                adaptive_per_node_cap_secs(bad),
                ADAPTIVE_PER_NODE_CAP_FLOOR_SECS,
                "remaining={bad}"
            );
        }

        // The cap never exceeds its own share of the budget it came from.
        for remaining in [60.0_f64, 120.0, 300.0, 600.0] {
            let cap = adaptive_per_node_cap_secs(remaining);
            assert!(
                cap <= remaining,
                "cap {cap} exceeds the remaining budget {remaining}"
            );
        }
    }

    #[test]
    fn test_compute_global_per_node_budget_secs_uses_equal_share_4413() {
        let _env = default_budget_env();
        assert_eq!(
            compute_global_per_node_budget_secs(24.0, 4, &DEFAULT_BUDGET),
            Some(6.0)
        );
    }

    #[test]
    fn test_compute_global_per_node_budget_secs_respects_floor_4413() {
        let _env = default_budget_env();
        assert_eq!(
            compute_global_per_node_budget_secs(6.0, 4, &DEFAULT_BUDGET),
            None
        );
    }

    #[test]
    fn test_weighted_budget_gives_wide_node_a_bigger_slice_cgan_cost_weight() {
        let _env = default_budget_env();
        // 100s envelope, weights BN_11=28800 vs a 1200-dim node: the wide node's
        // cost-proportional slice (96s) saturates the MAX cap, the small one gets
        // a tiny slice — vs the equal-share 50s each that starved BN_11.
        let weights = [28800.0_f64, 1200.0];
        let sum = sum_remaining_budget_weights(&weights, 0);
        assert_eq!(sum, 30000.0);
        let wide = compute_weighted_per_node_budget_secs(
            100.0,
            sum,
            weights[0],
            weights[0],
            &DEFAULT_BUDGET,
        );
        let small = compute_weighted_per_node_budget_secs(
            100.0,
            sum,
            weights[1],
            weights[1],
            &DEFAULT_BUDGET,
        );
        // wide share = 100 * 28800/30000 = 96s, truncated by the cap.
        //
        // #per-node-cap-from-budget: that cap is now a quarter of the 100 s
        // remaining (25 s) rather than the flat 12 s. The property under test is
        // unchanged and is what this test is named for — the wide node still
        // gets a far bigger slice than the narrow one — but the wide node is no
        // longer cut to a constant that has nothing to do with this budget.
        assert_eq!(wide, Some(25.0));
        // small share = 100 * 1200/30000 = 4s (above the 2s floor, below the cap).
        assert_eq!(small, Some(4.0));
        assert!(
            wide > small,
            "cost weighting must still favour the wide node"
        );
    }

    #[test]
    fn test_weighted_budget_reduces_to_equal_share_when_weights_match() {
        let _env = default_budget_env();
        // Equal weights => each gets remaining/N, matching the equal-share fn.
        let weights = [10.0_f64, 10.0, 10.0, 10.0];
        let sum = sum_remaining_budget_weights(&weights, 0);
        assert_eq!(
            compute_weighted_per_node_budget_secs(
                24.0,
                sum,
                weights[0],
                weights[0],
                &DEFAULT_BUDGET,
            ),
            compute_global_per_node_budget_secs(24.0, 4, &DEFAULT_BUDGET),
        );
    }

    #[test]
    fn test_weighted_budget_respects_floor_and_zero_weight() {
        let _env = default_budget_env();
        // Below-floor weighted share => None (degrades to IBP, sound).
        assert_eq!(
            compute_weighted_per_node_budget_secs(6.0, 40.0, 10.0, 10.0, &DEFAULT_BUDGET),
            None
        );
        // Zero / non-finite guards => None.
        assert_eq!(
            compute_weighted_per_node_budget_secs(100.0, 30000.0, 0.0, 10.0, &DEFAULT_BUDGET,),
            None
        );
        assert_eq!(
            compute_weighted_per_node_budget_secs(100.0, 0.0, 10.0, 10.0, &DEFAULT_BUDGET),
            None
        );
        assert_eq!(
            compute_weighted_per_node_budget_secs(100.0, 10.0, 10.0, 0.0, &DEFAULT_BUDGET,),
            None
        );
    }

    #[test]
    fn test_resolved_default_reports_floor_and_adaptive_cap_floor() {
        let _env = default_budget_env();
        // Resolution reports the default floor and the lower clamp for the
        // adaptive cap. The scheduling functions derive the actual unset cap
        // from their live remaining budget.
        assert_eq!(
            effective_per_node_time_budget(&DEFAULT_BUDGET),
            (MIN_PER_NODE_BUDGET_SECS, ADAPTIVE_PER_NODE_CAP_FLOOR_SECS,)
        );
    }

    #[test]
    fn test_per_node_budget_cap_override_reaches_computation_cgan_bn11_budget() {
        let _env = default_budget_env();
        // cgan_2023 preset shape: cap raised to 150 s, floor left at default.
        let budget = CrownIbpPerNodeTimeBudget {
            floor_secs: None,
            cap_secs: Some(150.0),
        };
        // A single remaining candidate with a long deadline now gets the full
        // preset cap instead of the 12 s constant.
        assert_eq!(
            compute_global_per_node_budget_secs(210.0, 1, &budget),
            Some(150.0)
        );
        // Equal-share below the cap is unchanged.
        assert_eq!(
            compute_global_per_node_budget_secs(24.0, 4, &budget),
            Some(6.0)
        );
        // The default floor still applies.
        assert_eq!(compute_global_per_node_budget_secs(6.0, 4, &budget), None);
    }

    #[test]
    fn test_per_node_budget_floor_override_cgan_bn11_budget() {
        let _env = default_budget_env();
        let budget = CrownIbpPerNodeTimeBudget {
            floor_secs: Some(1.0),
            cap_secs: None,
        };
        // 6/4 = 1.5 s share clears a 1.0 s floor (would be skipped at 2.0 s).
        assert_eq!(
            compute_global_per_node_budget_secs(6.0, 4, &budget),
            Some(1.5)
        );
    }

    #[test]
    fn test_per_node_budget_invalid_overrides_fall_back_to_defaults() {
        let _env = default_budget_env();
        for bad in [Some(f64::NAN), Some(f64::INFINITY), Some(0.0), Some(-3.0)] {
            let budget = CrownIbpPerNodeTimeBudget {
                floor_secs: bad,
                cap_secs: bad,
            };
            assert_eq!(
                effective_per_node_time_budget(&budget),
                (MIN_PER_NODE_BUDGET_SECS, ADAPTIVE_PER_NODE_CAP_FLOOR_SECS,)
            );
        }
    }

    #[test]
    fn test_count_remaining_budget_candidates_ignores_noneligible_nodes_4413() {
        let mask = [true, false, false, true];
        assert_eq!(count_remaining_budget_candidates(&mask, 0), 2);
        assert_eq!(count_remaining_budget_candidates(&mask, 3), 1);
    }

    #[test]
    fn test_dim_scaled_cap_widens_preset_cap_for_imgsz64_node_cgan_dim_cap() {
        let _env = default_budget_env();
        // cgan_2023 preset shape: cap 150 s measured for a 28,800-dim node.
        let budget = CrownIbpPerNodeTimeBudget {
            floor_secs: None,
            cap_secs: Some(150.0),
        };
        // The 61,504-dim imgSz64 generator target: ratio 2.1355..^2 = 4.56;
        // 150 * 4.56 = 684 s clamps to the 600 s ceiling. With 765 s remaining
        // and this node holding ~55% of the weight, the share (~424 s) now
        // reaches the node instead of being truncated to 150 s.
        let share =
            compute_weighted_per_node_budget_secs(765.0, 111_456.0, 61_504.0, 61_504.0, &budget)
                .expect("share above floor");
        let expected_share = 765.0 * (61_504.0 / 111_456.0);
        assert!(
            (share - expected_share).abs() < 1e-9,
            "share {share} truncated"
        );
        assert!(share > 150.0, "flat preset cap must no longer truncate");

        // A hypothetical even-wider node saturates the 600 s ceiling.
        let saturated = compute_weighted_per_node_budget_secs(
            10_000.0, 130_000.0, 123_008.0, 123_008.0, &budget,
        )
        .expect("share above floor");
        assert_eq!(saturated, super::DIM_SCALED_CAP_CEILING_SECS);

        // The reference-width node itself keeps the flat preset cap (ratio 1).
        assert_eq!(
            compute_weighted_per_node_budget_secs(765.0, 30_000.0, 28_800.0, 28_800.0, &budget,),
            Some(150.0)
        );
    }

    /// The measured yolo_2023 policy is selective: its explicit 12-second base
    /// cap scales for the 43,264-row useful targets, while the later
    /// 10,816/5,408-row targets that timed out stay at 12 seconds.
    #[test]
    fn test_yolo_base_cap_scales_only_wide_targets() {
        let _env = default_budget_env();
        let budget = CrownIbpPerNodeTimeBudget {
            floor_secs: None,
            cap_secs: Some(12.0),
        };

        let wide =
            compute_weighted_per_node_budget_secs(300.0, 43_264.0, 43_264.0, 43_264.0, &budget)
                .expect("wide target has a budget");
        let expected_wide = 12.0 * (43_264.0_f64 / PRESET_CAP_REFERENCE_DIMS).powi(2);
        assert!(
            (wide - expected_wide).abs() < 1e-12,
            "wide target cap {wide} != {expected_wide}"
        );
        assert!((wide - 27.080_059_259_259_258).abs() < 1e-12);

        for rows in [10_816.0, 5_408.0] {
            assert_eq!(
                compute_weighted_per_node_budget_secs(300.0, rows, rows, rows, &budget),
                Some(12.0),
                "{rows}-row target must keep the measured base cap"
            );
        }
    }

    #[test]
    fn test_dim_scaled_cap_leaves_default_cap_alone_cgan_dim_cap() {
        let _env = default_budget_env();
        // No preset cap: the built-in cap must NOT scale with node WIDTH — the
        // dim-aware quadratic scaling stays reserved for preset-supplied caps,
        // which is what this test guards.
        //
        // #per-node-cap-from-budget: it does now scale with the remaining
        // BUDGET, which is a different axis. With 765 s left the cap is a
        // quarter of that (191.25 s), not the flat 12 s. The distinction that
        // matters here still holds: two nodes of DIFFERENT widths under the same
        // remaining budget receive the same cap, so width is not scaling it.
        let wide = compute_weighted_per_node_budget_secs(
            765.0,
            111_456.0,
            61_504.0,
            61_504.0,
            &DEFAULT_BUDGET,
        )
        .expect("share above floor");
        assert_eq!(wide, 191.25);
        let wider = compute_weighted_per_node_budget_secs(
            765.0,
            111_456.0,
            100_000.0,
            100_000.0,
            &DEFAULT_BUDGET,
        )
        .expect("share above floor");
        assert_eq!(
            wide, wider,
            "the built-in cap must not scale with node width"
        );
    }

    #[test]
    fn test_dim_scaled_cap_respects_preset_caps_above_ceiling_cgan_dim_cap() {
        let _env = default_budget_env();
        // A preset cap larger than the ceiling wins (the scaler never SHRINKS
        // a preset cap).
        let budget = CrownIbpPerNodeTimeBudget {
            floor_secs: None,
            cap_secs: Some(700.0),
        };
        assert_eq!(
            compute_weighted_per_node_budget_secs(10_000.0, 100_000.0, 61_504.0, 61_504.0, &budget,),
            Some(700.0)
        );
    }

    #[test]
    fn test_chunk_aware_budget_gate_is_exact_and_default_dark() {
        assert!(!crown_chunk_aware_budget_from_raw(None));
        for raw in ["", "0", "00", "true", " 1", "1 "] {
            assert!(
                !crown_chunk_aware_budget_from_raw(Some(raw)),
                "non-exact spelling {raw:?} must stay dark"
            );
        }
        assert!(crown_chunk_aware_budget_from_raw(Some("1")));
    }

    #[test]
    fn test_objective_fixed_wave_work_weight_counts_wave_groups() {
        assert_eq!(objective_fixed_wave_work_weight(0, 7), 0);
        assert_eq!(objective_fixed_wave_work_weight(16, 1), 16);
        assert_eq!(objective_fixed_wave_work_weight(16, 2), 32);
        assert_eq!(objective_fixed_wave_work_weight(17, 3), 51);
        assert_eq!(objective_fixed_wave_work_weight(61_504, 8), 492_032);
    }

    #[test]
    fn test_demanded_target_work_weight_is_gate_and_route_scoped() {
        let rows = 28_800;
        let chunk_rows = 9_320;
        let plan = fixed_wave_plan(chunk_rows, chunk_rows, 4, 1, 4);
        assert_eq!(
            demanded_target_work_weight(rows, Some(plan), true, false),
            rows as f64,
            "gate-off must retain the historical raw-row weight"
        );
        assert_eq!(
            demanded_target_work_weight(rows, None, true, true),
            rows as f64,
            "under-budget targets must retain the raw-row weight"
        );
        assert_eq!(
            demanded_target_work_weight(rows, Some(plan), true, true),
            objective_fixed_wave_work_weight(rows, 4) as f64,
        );
        assert_eq!(
            demanded_target_work_weight(rows, Some(plan), false, true),
            rows as f64,
            "subset-seeded targets bypass the full-objective chunk route"
        );
    }

    #[test]
    fn test_dark_cap_input_preserves_parent_allocation_identity() {
        let budget = CrownIbpPerNodeTimeBudget {
            floor_secs: None,
            cap_secs: Some(150.0),
        };
        let scheduling_weight = 9_000.0;
        let raw_node_dims = 61_504.0;
        let legacy = compute_weighted_per_node_budget_secs(
            1_000.0,
            scheduling_weight,
            scheduling_weight,
            scheduling_weight,
            &budget,
        );
        let dark = compute_weighted_per_node_budget_secs(
            1_000.0,
            scheduling_weight,
            scheduling_weight,
            weighted_budget_cap_dims(scheduling_weight, raw_node_dims, false),
            &budget,
        );
        assert_eq!(dark, legacy);
        assert_eq!(
            weighted_budget_cap_dims(scheduling_weight, raw_node_dims, false),
            scheduling_weight
        );
        assert_eq!(
            weighted_budget_cap_dims(scheduling_weight, raw_node_dims, true),
            raw_node_dims
        );
    }

    #[test]
    fn test_alpha_style_denominator_only_inflates_its_target_members() {
        let plan = ObjectiveChunkSchedulingPlan {
            execution: objective_chunk_route_plan(9_320, true, true),
            fixed_waves: ObjectiveChunkFixedWavePlan {
                chunk_rows: 32,
                chunk_count: 900,
                wave_size: 32,
                wave_count: 29,
            },
        };
        let wide_target = demanded_target_work_weight(28_800, Some(plan), true, true);
        let small_target = demanded_target_work_weight(100, None, true, true);
        let non_target = demanded_target_work_weight(28_800, Some(plan), false, true);
        let denominator = wide_target + small_target;

        assert_eq!(
            wide_target,
            objective_fixed_wave_work_weight(28_800, 29) as f64
        );
        assert_eq!(denominator, wide_target + 100.0);
        assert_eq!(
            non_target, 28_800.0,
            "a node absent from the denominator must not take an inflated numerator"
        );
    }

    #[test]
    fn test_alpha_non_target_below_floor_never_receives_full_envelope() {
        let inflated_target = objective_fixed_wave_work_weight(28_800, 29) as f64;
        let non_target_weight = 100.0;
        let remaining = 6.0;

        assert_eq!(
            compute_weighted_per_node_budget_secs(
                remaining,
                inflated_target,
                non_target_weight,
                non_target_weight,
                &DEFAULT_BUDGET,
            ),
            None,
            "the inflated target makes a hypothetical non-target slice genuinely below floor"
        );
        assert_eq!(
            admitted_weighted_budget_secs(
                false,
                remaining,
                inflated_target,
                non_target_weight,
                non_target_weight,
                &DEFAULT_BUDGET,
            ),
            WeightedBudgetAdmission::NotAdmitted,
            "a non-target must take the reference path, never the full envelope"
        );
        assert_eq!(
            admitted_weighted_budget_secs(
                true,
                remaining,
                inflated_target,
                inflated_target,
                28_800.0,
                &DEFAULT_BUDGET,
            ),
            WeightedBudgetAdmission::Allocate(6.0),
            "the one admitted target retains the denominator-owned envelope"
        );
    }

    #[test]
    fn test_chunk_work_weight_does_not_inflate_dim_scaled_cap() {
        let budget = CrownIbpPerNodeTimeBudget {
            floor_secs: None,
            cap_secs: Some(150.0),
        };
        let raw_dims = 28_800.0;
        let work_weight = objective_fixed_wave_work_weight(28_800, 4) as f64;
        assert_eq!(work_weight, 115_200.0);
        assert_eq!(
            compute_weighted_per_node_budget_secs(
                1_000.0,
                work_weight,
                work_weight,
                raw_dims,
                &budget,
            ),
            Some(150.0),
            "the preset cap must scale from raw node dims, not chunk work"
        );
    }

    #[test]
    fn test_auto_objective_chunk_rows_imgsz64_widest_node_cgan_dim_cap() {
        // imgSz64_nCh_3 widest generator node: 61,504 dims ([16, 62, 62]).
        // Full identity pair = 8 * 61504^2 ≈ 30.3 GB; the auto chunk must
        // scale the [C x dim] pair under a 2 GiB budget without overflow.
        let budget = 2usize * 1024 * 1024 * 1024;
        // `input_dim = 0` isolates the per-row term this case is about; the
        // fixed input-scratch term gets its own coverage in
        // `test_auto_objective_chunk_rows_fits_full_admission_charge`.
        let rows = super::auto_objective_chunk_rows(61_504, 0, budget);
        assert!(rows >= 1);
        assert!(2 * 4 * rows * 61_504 <= budget, "chunk pair exceeds budget");
        assert!(
            (rows + 1) * 2 * 4 * 61_504 > budget,
            "chunk under-fills budget"
        );
        assert!(rows < 61_504, "must be a genuine multi-pass chunk");
    }

    #[test]
    fn test_auto_objective_chunk_rows_cgan_bn11() {
        // cgan_2023 BatchNormalization_11: 28,800 dims, 2 GiB budget. The
        // full identity pair is 8 * 28800^2 = 6,635,520,000 bytes (~6.6 GB);
        // the auto chunk scales the pair down to fit the budget.
        let budget = 2usize * 1024 * 1024 * 1024;
        let rows = super::auto_objective_chunk_rows(28_800, 0, budget);
        // 9_319, not the historical 9_320: the row cost now also carries the
        // bias pair and the endpoint buffers admission charges (`8*dim + 24`
        // per row, not `8*dim`). One row narrower is the whole point — the old
        // count saturated the budget with the coefficient pair alone and left
        // nothing for the rest of the charge.
        assert_eq!(rows, 9_319);
        // The chunk's [C x dim] pair stays within budget.
        assert!(2 * 4 * rows * 28_800 <= budget);
        // And it is a genuine chunk (multiple passes required).
        assert!(rows < 28_800);
    }

    /// #cprime-admission: core fits/refuses arithmetic. A padded estimate at
    /// or under the share admits; over it refuses (non-last), with the share
    /// as the reported budget.
    #[test]
    fn test_walk_admission_fits_and_refuses_cprime_admission() {
        use super::{admit_walk_with_estimate, WalkAdmissionDecision};

        // 4s estimate * 1.25 = 5s == share -> admit.
        assert_eq!(
            admit_walk_with_estimate(4.0, 5.0, 40.0, false),
            WalkAdmissionDecision::Admit
        );
        // 4.1s estimate * 1.25 = 5.125s > 5s share -> refused, share reported.
        assert_eq!(
            admit_walk_with_estimate(4.1, 5.0, 40.0, false),
            WalkAdmissionDecision::Refuse {
                estimated_secs: 4.1,
                budget_secs: 5.0
            }
        );
        // Degenerate estimates fail open to admission (today's policy).
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                admit_walk_with_estimate(bad, 5.0, 40.0, false),
                WalkAdmissionDecision::Admit,
                "estimate={bad}"
            );
        }
    }

    /// #cprime-admission: the Conv_17 exhibit shape (tinyimagenet 2026-08-02).
    /// The walk that burned 150.29s of share and delivered nothing is refused
    /// upfront, and the UNSPENT share structurally enlarges the next
    /// candidate's slice (rollover): with the refusal consuming ~0s, the next
    /// node's weighted share divides the same remaining time, versus the
    /// burned-share world where 150s are gone.
    #[test]
    fn test_walk_admission_conv17_exhibit_refused_and_share_rolls_cprime_admission() {
        use super::{admit_walk_with_estimate, WalkAdmissionDecision};

        // Conv_17's true walk cost exceeded 150s (aborted mid-first-step).
        // Even the optimistic prior halves it: est ~200s vs a 150s share.
        let est_secs = 200.0;
        let share_secs = 150.0;
        assert_eq!(
            admit_walk_with_estimate(est_secs, share_secs, 600.0, false),
            WalkAdmissionDecision::Refuse {
                estimated_secs: est_secs,
                budget_secs: share_secs
            }
        );

        // Rollover: refusal leaves remaining time INTACT. Next candidate
        // (weight 100 of 200 remaining weight) gets 300s * 0.5 = 150s...
        let after_refusal =
            compute_weighted_per_node_budget_secs(600.0, 200.0, 100.0, 100.0, &DEFAULT_BUDGET)
                .expect("share above floor");
        // ...whereas a burned share leaves only 450s: 450 * 0.5 = 112.5s.
        let after_burn =
            compute_weighted_per_node_budget_secs(450.0, 200.0, 100.0, 100.0, &DEFAULT_BUDGET)
                .expect("share above floor");
        assert!(
            after_refusal > after_burn,
            "refusal must leave later candidates more budget than a burn \
             ({after_refusal} vs {after_burn})"
        );
        assert_eq!(after_refusal, 150.0);
        assert_eq!(after_burn, 112.5);
    }

    /// #cprime-admission: the LAST demanded candidate may claim the
    /// accumulated rollover — admitted against the collection's full
    /// remaining time even when its capped share is too small — and is still
    /// refused when even the rollover cannot fit it.
    #[test]
    fn test_walk_admission_last_target_gets_rollover_cprime_admission() {
        use super::{admit_walk_with_estimate, WalkAdmissionDecision};

        // est 100s * 1.25 = 125s: over the 60s capped share, within 300s
        // remaining -> rollover grant, but only for the LAST candidate.
        assert_eq!(
            admit_walk_with_estimate(100.0, 60.0, 300.0, true),
            WalkAdmissionDecision::AdmitWithRollover
        );
        assert_eq!(
            admit_walk_with_estimate(100.0, 60.0, 300.0, false),
            WalkAdmissionDecision::Refuse {
                estimated_secs: 100.0,
                budget_secs: 60.0
            }
        );
        // Even the rollover cannot host 400s * 1.25 -> refused, with the
        // FULL remaining time reported as the budget that was insufficient.
        assert_eq!(
            admit_walk_with_estimate(400.0, 60.0, 300.0, true),
            WalkAdmissionDecision::Refuse {
                estimated_secs: 400.0,
                budget_secs: 300.0
            }
        );
    }

    /// #cprime-admission: the first completed walk's actual time replaces the
    /// census prior; later observations are ignored (one-shot), and
    /// degenerate measurements never poison the correction.
    #[test]
    fn test_walk_cost_calibration_first_walk_replaces_prior_cprime_admission() {
        use super::{WalkCostModel, WALK_RATE_PRIOR_CORRECTION};

        let rate = 10_000_000_000.0; // 10 GMAC/s
        let mut model = WalkCostModel::new(rate);
        assert!(!model.is_calibrated());
        assert_eq!(model.correction(), WALK_RATE_PRIOR_CORRECTION);

        // 20 GMAC at 10 GMAC/s raw = 2.0s; prior 0.5 predicts 1.0s.
        let macs: u128 = 20_000_000_000;
        assert_eq!(model.estimate_secs(macs), Some(1.0));

        // First completed walk measured 3.0s -> correction = 3.0/2.0 = 1.5.
        model.observe_completed_walk(macs, 3.0);
        assert!(model.is_calibrated());
        assert!((model.correction() - 1.5).abs() < 1e-12);
        assert_eq!(model.estimate_secs(macs), Some(3.0));

        // One-shot: a second observation changes nothing.
        model.observe_completed_walk(macs, 30.0);
        assert!((model.correction() - 1.5).abs() < 1e-12);

        // Degenerate inputs never calibrate.
        for (m, s) in [(macs, f64::NAN), (macs, 0.0), (macs, 0.01), (0u128, 5.0)] {
            let mut fresh = WalkCostModel::new(rate);
            fresh.observe_completed_walk(m, s);
            assert!(!fresh.is_calibrated(), "macs={m} secs={s}");
        }
        // Unusable rate: estimates are None (fail-open admit) and never
        // calibrate.
        let mut broken = WalkCostModel::new(0.0);
        assert_eq!(broken.estimate_secs(macs), None);
        broken.observe_completed_walk(macs, 3.0);
        assert!(!broken.is_calibrated());
    }

    /// #cprime-admission determinism: the admission decision is a pure
    /// function of (estimate, share, remaining, last-flag) — the same graph
    /// and the same rate produce the same admission set, byte-for-byte.
    #[test]
    fn test_walk_admission_deterministic_cprime_admission() {
        use super::{admit_walk_with_estimate, WalkCostModel};

        let mut model_a = WalkCostModel::new(11_570_000_000.0);
        let mut model_b = WalkCostModel::new(11_570_000_000.0);
        let targets: [(u128, f64, f64, bool); 4] = [
            (5_000_000_000, 4.0, 40.0, false),
            (900_000_000_000, 20.0, 36.0, false),
            (30_000_000_000, 12.0, 16.0, false),
            (60_000_000_000, 4.0, 4.0, true),
        ];
        let run = |model: &mut WalkCostModel| {
            let mut decisions = Vec::new();
            for (i, &(macs, share, remaining, last)) in targets.iter().enumerate() {
                let est = model.estimate_secs(macs).expect("usable rate");
                let d = admit_walk_with_estimate(est, share, remaining, last);
                decisions.push(d);
                if i == 0 {
                    // Same first-walk measurement in both runs.
                    model.observe_completed_walk(macs, 0.6);
                }
            }
            decisions
        };
        assert_eq!(run(&mut model_a), run(&mut model_b));
        assert_eq!(model_a.correction(), model_b.correction());
    }

    #[test]
    fn test_auto_objective_chunk_rows_clamps_to_dim_and_floor() {
        // Under-budget target: clamp to dim (equivalent to a single pass).
        assert_eq!(super::auto_objective_chunk_rows(16, 0, usize::MAX >> 1), 16);
        // Tiny budget: floor at 1 row so the chunked loop always progresses.
        assert_eq!(super::auto_objective_chunk_rows(1_000_000, 0, 1), 1);
        // Degenerate dim.
        assert_eq!(super::auto_objective_chunk_rows(0, 0, 1024), 1);
        // Input scratch alone eats the whole budget: still floor at 1 row
        // rather than returning 0 and stalling the chunk loop.
        assert_eq!(super::auto_objective_chunk_rows(64, 1_000_000, 1024), 1);
    }

    /// REGRESSION (#cgan-bn11-chunk): the chunk size must fit the FULL charge
    /// `LinearConcretizationAdmission` levies, not just the coefficient pair.
    ///
    /// The exhibit is the case that was measured failing: a dim-400 target
    /// under a 1 MiB budget. The old model returned C=327, whose pair alone was
    /// 1,046,400 of 1,048,576 bytes (99.8%); admission then charged 1,056,864
    /// and raised `CpuMemoryExceeded`, so the reroute degraded to the very IBP
    /// it exists to avoid. Any future model that saturates the budget with one
    /// term fails here.
    #[test]
    fn test_auto_objective_chunk_rows_fits_full_admission_charge() {
        const F32: usize = size_of::<f32>();
        const F64: usize = size_of::<f64>();
        for &(dim, input_dim, budget) in &[
            (400usize, 400usize, 1024usize * 1024),
            (28_800, 100, 2 * 1024 * 1024 * 1024),
            (61_504, 12_288, 2 * 1024 * 1024 * 1024),
            (4096, 3072, 8 * 1024 * 1024),
        ] {
            let rows = super::auto_objective_chunk_rows(dim, input_dim, budget);
            assert!(rows >= 1, "dim={dim}: chunk loop must always progress");
            // Exactly the terms LinearConcretizationAdmission::new accumulates.
            let coefficient_pair = 2 * F32 * rows * dim;
            let bias_pair = 2 * F32 * rows;
            let endpoints = rows * F64 + rows * 2 * F32;
            let input_scratch = 2 * F32 * input_dim;
            let charged = coefficient_pair + bias_pair + endpoints + input_scratch;
            assert!(
                charged <= budget,
                "dim={dim} input_dim={input_dim}: C={rows} is charged {charged} \
                 bytes against a {budget} byte budget — admission would refuse \
                 and the reroute would degrade to IBP"
            );
        }
    }

    /// ADVERSARIAL VERIFY (#cprime-admission, attack 1 OVER-REFUSAL): the
    /// Conv_11-class exhibit — a walk with ~6.15 s true cost against a ~12 s
    /// share — must ADMIT both pre- and post-calibration.
    ///
    /// Numbers: FL probe rate ~11.57 GMAC/s on this host class; the census
    /// says the collector's transpose-GEMM runs ~2x faster (~20 GMAC/s), so a
    /// 6.15 s-true walk carries ~123 GMAC. raw = 123/11.57 = 10.63 s; prior
    /// 0.5 estimates 5.32 s; padded 6.65 s <= 12 s share -> Admit. After a
    /// perfect first-walk calibration (kappa = 0.578) the estimate is the
    /// true 6.15 s; padded 7.69 s <= 12 s -> still Admit.
    #[test]
    fn verify_walk_admission_conv11_class_boundary_admits_cprime() {
        use super::{admit_walk_with_estimate, WalkAdmissionDecision, WalkCostModel};

        let fl_rate = 11.57e9;
        let macs: u128 = 123_000_000_000; // ~6.15 s at the collector's ~20 GMAC/s
        let model = WalkCostModel::new(fl_rate);
        let est = model.estimate_secs(macs).expect("usable rate");
        assert!(
            est > 5.0 && est < 5.7,
            "prior estimate must sit near 5.3 s, got {est}"
        );
        assert_eq!(
            admit_walk_with_estimate(est, 12.0, 60.0, false),
            WalkAdmissionDecision::Admit,
            "the Conv_11-class walk must be admitted pre-calibration"
        );

        // Post-calibration on the exhibit itself.
        let mut cal = WalkCostModel::new(fl_rate);
        cal.observe_completed_walk(macs, 6.15);
        assert!(cal.is_calibrated());
        let est2 = cal.estimate_secs(macs).expect("usable rate");
        assert!(
            (est2 - 6.15).abs() < 1e-9,
            "calibrated estimate = true cost"
        );
        assert_eq!(
            admit_walk_with_estimate(est2, 12.0, 60.0, false),
            WalkAdmissionDecision::Admit,
            "the Conv_11-class walk must be admitted post-calibration"
        );
    }

    /// ADVERSARIAL VERIFY (#cprime-admission, attack 1 boundary FINDING):
    /// there IS a window of walks the old policy completed that the new
    /// policy refuses. With a perfectly calibrated model, refusal fires when
    /// padded > share, i.e. true cost > share/1.25 = 0.8 x share — a walk
    /// finishing with up to 20% headroom is refused by design (the margin
    /// trades it against walks that run to the deadline and return nothing).
    /// Pre-calibration the boundary is 1.6 x kappa_true x share; census
    /// kappa_true ~0.5-0.59 puts it at 0.80-0.94 x share.
    #[test]
    fn verify_walk_admission_margin_refuses_near_share_walks_cprime() {
        use super::{admit_walk_with_estimate, WalkAdmissionDecision};

        // Calibrated-model view: estimate == true cost.
        // 8.0 s true against a 10 s share (exact 0.8 boundary): admitted.
        assert_eq!(
            admit_walk_with_estimate(8.0, 10.0, 100.0, false),
            WalkAdmissionDecision::Admit
        );
        // 8.5 s true against a 10 s share: the old policy completed this walk
        // with 1.5 s headroom; the new policy refuses it (8.5 * 1.25 > 10).
        assert_eq!(
            admit_walk_with_estimate(8.5, 10.0, 100.0, false),
            WalkAdmissionDecision::Refuse {
                estimated_secs: 8.5,
                budget_secs: 10.0
            },
            "walks in (0.8, 1.0] x share are refused although they finished \
             under the old policy — deliberate margin trade, documented here"
        );
    }

    /// ADVERSARIAL VERIFY (#cprime-admission, attack 1 estimator FORM) —
    /// originally locked in an OVER-pricing defect (total-in_c contraction
    /// = x groups over-estimate; naive out-dims = x1.78 on dilated convs),
    /// which violated the fail-open direction by pushing grouped/dilated
    /// nets toward wrongful refusal. FIXED same-session: the estimator now
    /// contracts in_c/groups (kernel.shape()[1]) and uses the layer's own
    /// `output_size()`. This test now pins the CORRECT group- and
    /// dilation-aware pricing.
    #[test]
    fn verify_collector_walk_macs_overprices_grouped_and_dilated_convs_cprime() {
        use crate::layers::{Conv2dLayer, Layer};
        use crate::network::core::{GraphNetwork, GraphNode};
        use ndarray::{ArrayD, IxDyn};

        // Depthwise conv: 8 channels, groups=8, 3x3 kernel, 8x8 input, pad 1.
        let kernel = ArrayD::<f32>::zeros(IxDyn(&[8, 1, 3, 3]));
        let conv = Conv2dLayer::with_input_shape_full(kernel, None, (1, 1), (1, 1), 8, 8, 8)
            .expect("depthwise conv");
        assert_eq!(conv.in_channels(), 8);
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("dw", Layer::Conv2d(conv)));
        graph.set_output("dw");
        let rows = 4u128;
        let macs = graph
            .collector_walk_macs("dw", rows as usize)
            .expect("weighted ancestor");
        // Group-aware pricing: in_c/groups = 1 per-element contraction —
        // exactly what the executed grouped transpose-GEMM costs, so the
        // unit contraction drops out of the product below.
        let group_aware = 2 * rows * 8 * 8 * 8 * (3 * 3);
        assert_eq!(
            macs, group_aware,
            "estimator must price the per-group contraction the backward \
             actually executes (fail-open direction preserved for grouped convs)"
        );

        // Dilated conv: 1 channel, 3x3 kernel, dilation 2, 10x10 input, pad 0.
        let kernel = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 3, 3]));
        let conv = Conv2dLayer::new_dilated(kernel, None, (1, 1), (0, 0), (2, 2), 1)
            .expect("dilated conv");
        let mut conv = conv;
        conv.set_input_shape(10, 10);
        assert_eq!(
            conv.output_size(10, 10).expect("output size"),
            (6, 6),
            "layer's own formula accounts for the dilated span"
        );
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("dil", Layer::Conv2d(conv)));
        graph.set_output("dil");
        let macs = graph
            .collector_walk_macs("dil", rows as usize)
            .expect("weighted ancestor");
        // Dilation-aware pricing via the layer's own output_size(): 6x6.
        // out_c = 1 and in_c/groups = 1 are both unit factors here.
        assert_eq!(
            macs,
            2 * rows * 6 * 6 * (3 * 3),
            "estimator must price the dilated 6x6 output map, not the naive 8x8"
        );
    }

    /// ADVERSARIAL VERIFY (#cprime-admission, attack 2 ROLLOVER STARVATION,
    /// all-nodes-unaffordable edge): refusals consume no time and grant no
    /// time; even the last candidate's rollover is bounded by the window end,
    /// so the floor100 shape (a grant reaching past the fixed window and
    /// starving what follows) cannot recur.
    #[test]
    fn verify_walk_admission_all_unaffordable_frees_window_cprime() {
        use super::{admit_walk_with_estimate, WalkAdmissionDecision, WALK_ADMISSION_MARGIN};

        // Fixed 100 s window, 4 equal candidates, every walk estimates 100 s.
        // Refusals consume ~0 s, so the remaining window NEVER shrinks and
        // each successive share is computed against the full 100 s.
        let remaining = 100.0;
        let mut granted_secs = 0.0f64;
        for (idx, last) in [(0usize, false), (1, false), (2, false), (3, true)] {
            let candidates_left = 4 - idx;
            let share = remaining / candidates_left as f64;
            let decision = admit_walk_with_estimate(100.0, share, remaining, last);
            match decision {
                WalkAdmissionDecision::Refuse { budget_secs, .. } => {
                    // The reported budget never exceeds the window.
                    assert!(budget_secs <= remaining);
                }
                other => panic!("candidate {idx} must be refused, got {other:?}"),
            }
            // A refusal grants no execution time.
            granted_secs += 0.0;
        }
        assert_eq!(granted_secs, 0.0, "all-unaffordable consumes ~0 s");

        // A just-affordable LAST candidate is granted the rollover, and the
        // grant is bounded by the window: padded estimate <= remaining, and
        // the granted deadline (crown_tighten.rs:1339) IS the collection
        // deadline — never past it.
        let est = 79.0; // 79 * 1.25 = 98.75 <= 100
        assert_eq!(
            admit_walk_with_estimate(est, 25.0, remaining, true),
            WalkAdmissionDecision::AdmitWithRollover
        );
        assert!(est * WALK_ADMISSION_MARGIN <= remaining);
        // One epsilon more than the window refuses even the last candidate.
        assert_eq!(
            admit_walk_with_estimate(80.1, 25.0, remaining, true),
            WalkAdmissionDecision::Refuse {
                estimated_secs: 80.1,
                budget_secs: remaining
            }
        );
    }

    /// ADVERSARIAL VERIFY (#cprime-admission, attack 3 CALIBRATION POISON,
    /// fast lie): an anomalously fast first walk clamps kappa at 0.05 rather
    /// than following the lie all the way down, and the resulting mis-admits
    /// are bounded because Admit leaves the per-node deadline untouched
    /// (crown_tighten.rs:1328 is an empty match arm — the walk still runs
    /// under the unchanged share deadline and degrades to IBP on overrun).
    #[test]
    fn verify_walk_calibration_poison_fast_first_walk_clamped_cprime() {
        use super::{admit_walk_with_estimate, WalkAdmissionDecision, WalkCostModel};

        let rate = 1.0e10;
        let mut model = WalkCostModel::new(rate);
        // raw = 10 s walk "completes" in 0.05 s (the calibration floor):
        // kappa = 0.005 would follow the lie; the clamp holds it at 0.05.
        model.observe_completed_walk(100_000_000_000, 0.05);
        assert!(model.is_calibrated());
        assert_eq!(model.correction(), 0.05);

        // A raw-100 s walk now estimates 5 s and is ADMITTED against a 10 s
        // share — a mis-admit, but exactly today's behavior: the walk runs
        // under the unchanged per-node deadline and is cut at the share.
        let est = model.estimate_secs(1_000_000_000_000).expect("usable");
        assert_eq!(est, 5.0);
        assert_eq!(
            admit_walk_with_estimate(est, 10.0, 100.0, false),
            WalkAdmissionDecision::Admit
        );
    }

    /// ADVERSARIAL VERIFY (#cprime-admission, attack 3 CALIBRATION POISON,
    /// slow direction — LEFTOVER RISK, documented): calibration consumes
    /// `node_secs`, the WHOLE node wall time (walk + reshape + intersection +
    /// planning overhead), not the GEMM walk alone. A cheap first walk
    /// (raw 0.1 s) whose node spends 1.0 s in overhead passes both 0.05 s
    /// floors and calibrates kappa = 10; later genuinely cheap walks are
    /// then over-refused (raw 1.0 s, true ~0.5 s, estimates 10 s and is
    /// refused against a 12 s share the old policy met easily). The kappa
    /// clamp only bounds this at 20x.
    #[test]
    fn verify_walk_calibration_overhead_poison_slow_first_walk_cprime() {
        use super::{admit_walk_with_estimate, WalkAdmissionDecision, WalkCostModel};

        let rate = 1.0e10;
        let mut model = WalkCostModel::new(rate);
        // First walk: raw 0.1 s (1 GMAC), node wall 1.0 s (overhead-dominated
        // but above both floors) -> kappa = 10, calibrated.
        model.observe_completed_walk(1_000_000_000, 1.0);
        assert!(
            model.is_calibrated(),
            "the 0.05 s floors do NOT reject this overhead-dominated pair"
        );
        assert_eq!(model.correction(), 10.0);

        // A raw-1.0 s walk (true cost ~0.5 s at the census collector rate)
        // now estimates 10 s; padded 12.5 s > 12 s share -> REFUSED, though
        // the old policy completed it with >11 s headroom.
        let est = model.estimate_secs(10_000_000_000).expect("usable");
        assert_eq!(est, 10.0);
        assert_eq!(
            admit_walk_with_estimate(est, 12.0, 100.0, false),
            WalkAdmissionDecision::Refuse {
                estimated_secs: 10.0,
                budget_secs: 12.0
            },
            "over-refusal via overhead-poisoned calibration — leftover risk"
        );
    }

    /// ADVERSARIAL VERIFY (#cprime-admission, estimator FORM on a real
    /// chain): `collector_walk_macs` prices exactly
    /// `2 · rows · Σ per_row(weighted ancestors of target)` — the conv and
    /// linear per-row units match the documented formulas by hand, the sum
    /// scopes to ANCESTORS of the target only (an early target excludes its
    /// descendants' cost), rows scale linearly, and the fail-open `None`
    /// cases (rows = 0, no weighted ancestor) hold.
    #[test]
    fn verify_collector_walk_macs_chain_arithmetic_and_fail_open_cprime() {
        use crate::layers::{Conv2dLayer, Layer, LinearLayer, ReLULayer};
        use crate::network::core::{GraphNetwork, GraphNode};
        use ndarray::{Array2, ArrayD, IxDyn};

        // conv: out_c=4, in_c=3, 3x3, stride 1, pad 1, input 8x8 -> out 8x8.
        // per_row = 8*8*4*(3*3*3) = 6912.
        let kernel = ArrayD::<f32>::zeros(IxDyn(&[4, 3, 3, 3]));
        let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (1, 1), 8, 8).expect("conv");
        // fc: out=2, in=4*8*8=256. per_row = 2*256 = 512.
        let fc = LinearLayer::new(Array2::<f32>::zeros((2, 256)), None).expect("fc");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.add_node(GraphNode::new(
            "act",
            Layer::ReLU(ReLULayer),
            vec!["conv".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "fc",
            Layer::Linear(fc),
            vec!["act".to_string()],
        ));
        graph.set_output("fc");

        let conv_per_row: u128 = 8 * 8 * 4 * (3 * 3 * 3);
        assert_eq!(conv_per_row, 6912);
        let fc_per_row: u128 = 2 * 256;

        // Target = fc: both weighted ancestors priced, elementwise ReLU free.
        let rows: u128 = 5;
        assert_eq!(
            graph.collector_walk_macs("fc", rows as usize),
            Some(2 * rows * (conv_per_row + fc_per_row)),
            "chain sum: 2 * rows * (conv + fc per-row units)"
        );
        // Rows scale linearly (chunked execution processes the same total).
        assert_eq!(
            graph.collector_walk_macs("fc", 10),
            Some(2 * 10 * (conv_per_row + fc_per_row))
        );
        // Target = conv (early node): the DESCENDANT fc contributes nothing.
        assert_eq!(
            graph.collector_walk_macs("conv", rows as usize),
            Some(2 * rows * conv_per_row),
            "ancestors-only scoping: fc's cost must not be priced into conv"
        );
        // rows = 0 -> None -> admit (fail-open).
        assert_eq!(graph.collector_walk_macs("fc", 0), None);
        // No weighted ancestor -> None -> admit (fail-open).
        let mut relu_only = GraphNetwork::new();
        relu_only.add_node(GraphNode::from_input("act", Layer::ReLU(ReLULayer)));
        relu_only.set_output("act");
        assert_eq!(relu_only.collector_walk_macs("act", 4), None);
    }
}

#[cfg(test)]
mod abort_calibration_tests {
    use super::*;

    fn model() -> WalkCostModel {
        WalkCostModel::new(1_000_000_000.0)
    }

    /// #cprime-abort-calib: an abort sample raises the correction when it shows
    /// the walk is costlier than the prior assumed — the cgan_2023 case, where
    /// nothing ever completes so the completed-walk path never fires.
    #[test]
    fn aborted_walk_raises_the_correction_when_the_walk_is_costlier() {
        let mut m = model();
        let before = m.correction();
        // 10 G MACs at 1 GMAC/s => raw 10s; the abort projects 200s.
        m.observe_aborted_walk(10_000_000_000, 200.0);
        assert!(
            m.correction() > before,
            "abort evidence of a 20x-costlier walk must raise the correction \
             ({} -> {})",
            before,
            m.correction()
        );
        assert!(
            !m.is_calibrated(),
            "an abort sample is not a full calibration"
        );
    }

    /// It may only ever move UPWARD: an abort observation is evidence a walk
    /// costs MORE than priced, never less, so a cheap-looking sample must not
    /// make the model optimistic and start admitting doomed walks.
    #[test]
    fn aborted_walk_never_lowers_the_correction() {
        let mut m = model();
        let before = m.correction();
        m.observe_aborted_walk(10_000_000_000, 0.001);
        assert_eq!(
            m.correction(),
            before,
            "a cheap abort sample must not lower the correction"
        );
    }

    /// A real completed-walk calibration wins and is never overwritten.
    #[test]
    fn aborted_walk_does_not_overwrite_a_real_calibration() {
        let mut m = model();
        m.observe_completed_walk(10_000_000_000, 30.0);
        assert!(m.is_calibrated());
        let calibrated = m.correction();
        m.observe_aborted_walk(10_000_000_000, 900.0);
        assert_eq!(
            m.correction(),
            calibrated,
            "completed-walk calibration must not be displaced by an abort sample"
        );
    }
}

thread_local! {
    /// Last `#chunk-abort` projection, in seconds for the FULL walk
    /// (measured-so-far + projected-remainder), published by the objective-chunk
    /// driver and consumed by the CROWN-IBP collector (#cprime-abort-calib).
    ///
    /// Thread-local rather than global: the collector calls the walk
    /// synchronously on its own thread (rayon parallelism lives *inside* the
    /// GEMM, below this seam), so the value a collector takes is always the one
    /// its own aborted walk just published. A process-global would race across
    /// concurrent verifier instances.
    static LAST_WALK_ABORT_PROJECTION: std::cell::Cell<Option<f64>> =
        const { std::cell::Cell::new(None) };
}

/// Publish an aborted walk's full-walk cost projection (#cprime-abort-calib).
pub(crate) fn publish_walk_abort_projection(full_walk_secs: f64) {
    if full_walk_secs.is_finite() && full_walk_secs > 0.0 {
        LAST_WALK_ABORT_PROJECTION.with(|c| c.set(Some(full_walk_secs)));
    }
}

/// Consume the last aborted-walk projection, if any. Taking clears it so a
/// later walk cannot be calibrated from a stale sample.
pub(crate) fn take_walk_abort_projection() -> Option<f64> {
    LAST_WALK_ABORT_PROJECTION.with(std::cell::Cell::take)
}

#[cfg(test)]
mod abort_projection_channel_tests {
    /// Taking clears, so one abort sample cannot calibrate two walks.
    #[test]
    fn projection_is_taken_once() {
        super::publish_walk_abort_projection(123.0);
        assert_eq!(super::take_walk_abort_projection(), Some(123.0));
        assert_eq!(super::take_walk_abort_projection(), None);
    }

    /// Non-finite / non-positive samples are refused rather than published.
    #[test]
    fn nonsense_projections_are_not_published() {
        let _ = super::take_walk_abort_projection();
        super::publish_walk_abort_projection(f64::NAN);
        super::publish_walk_abort_projection(0.0);
        super::publish_walk_abort_projection(-5.0);
        assert_eq!(super::take_walk_abort_projection(), None);
    }
}

#[cfg(test)]
mod walk_value_record_tests {
    use super::*;

    /// FALLBACK IDENTITY: with no measured completion, the record-aware
    /// admission IS `admit_walk_with_estimate` — bit-identical decisions
    /// across the whole decision surface, so a collection with no records
    /// behaves exactly as today.
    #[test]
    fn no_record_is_bit_identical_to_the_proxy_admission() {
        for &(est, share, remaining, last) in &[
            (1.0_f64, 10.0_f64, 100.0_f64, false),
            (10.0, 10.0, 100.0, false), // padded 12.5 > share => Refuse
            (10.0, 10.0, 100.0, true),  // last candidate => rollover
            (200.0, 10.0, 100.0, true), // beyond even the rollover
            (f64::NAN, 10.0, 100.0, false),
            (-3.0, 10.0, 100.0, false),
        ] {
            assert_eq!(
                admit_walk_with_record(est, None, share, remaining, last),
                admit_walk_with_estimate(est, share, remaining, last),
                "est={est} share={share} remaining={remaining} last={last}"
            );
        }
    }

    /// The item-3 scenario: the value record shows this node COMPLETING at 2x
    /// its static share. The budgeter grants the measured cost (with margin)
    /// instead of refusing at the static share.
    #[test]
    fn measured_completion_above_share_within_window_becomes_a_bounded_grant() {
        let decision = admit_walk_with_record(20.0, Some(20.0), 10.0, 100.0, false);
        assert_eq!(
            decision,
            WalkAdmissionDecision::AdmitWithMeasuredGrant {
                grant_secs: 20.0 * WALK_ADMISSION_MARGIN,
            }
        );
    }

    /// A measurement can never revoke an Admit: when the padded estimate fits
    /// the share, the decision is Admit with or without a record.
    #[test]
    fn measured_completion_never_downgrades_an_admit() {
        assert_eq!(
            admit_walk_with_record(4.0, Some(4.0), 10.0, 100.0, false),
            WalkAdmissionDecision::Admit
        );
    }

    /// Beyond the collection's remaining time, even a measured completion is
    /// refused — the grant is bounded by the collection deadline.
    #[test]
    fn measured_completion_beyond_the_window_still_refuses() {
        let decision = admit_walk_with_record(200.0, Some(200.0), 10.0, 100.0, false);
        assert!(
            matches!(decision, WalkAdmissionDecision::Refuse { .. }),
            "got {decision:?}"
        );
    }

    /// Degenerate measurements fall back to the base decision.
    #[test]
    fn degenerate_measurements_fall_back_to_the_base_decision() {
        for measured in [f64::NAN, f64::INFINITY, 0.0, -7.0] {
            assert_eq!(
                admit_walk_with_record(20.0, Some(measured), 10.0, 100.0, false),
                admit_walk_with_estimate(20.0, 10.0, 100.0, false),
                "measured={measured}"
            );
        }
    }

    /// Record store: completed walls keep the MAX (conservative price), a
    /// consult requires an exact row match, and a row-count change resets the
    /// record so stale geometry cannot price the new route.
    #[test]
    fn record_store_maxes_matches_rows_and_resets_on_row_change() {
        reset_node_walk_records();
        record_node_walk_completed("Conv_11", 128, 30.0);
        record_node_walk_completed("Conv_11", 128, 20.0);
        let record = node_walk_record("Conv_11", 128).expect("record for matching rows");
        assert_eq!(record.completed_secs, Some(30.0), "max, not latest");
        assert_eq!(record.estimate_secs(), Some(30.0));
        assert!(
            node_walk_record("Conv_11", 64).is_none(),
            "row mismatch must not be priced by the full-width twin"
        );
        // Row-count change resets: the projection recorded for the OLD rows
        // must not survive into the new geometry.
        record_node_walk_abort_projection("Conv_11", 256, 500.0);
        let renewed = node_walk_record("Conv_11", 256).expect("record for the new rows");
        assert_eq!(renewed.completed_secs, None, "reset on row change");
        assert_eq!(renewed.aborted_projection_secs, Some(500.0));
        assert_eq!(renewed.estimate_secs(), Some(500.0));
        reset_node_walk_records();
        assert!(node_walk_record("Conv_11", 256).is_none());
    }

    /// A completion beats a projection inside one record: real completed wall
    /// is the estimate whenever both exist.
    #[test]
    fn completed_wall_beats_projection_within_a_record() {
        reset_node_walk_records();
        record_node_walk_abort_projection("ConvT_13", 512, 400.0);
        record_node_walk_completed("ConvT_13", 512, 55.0);
        let record = node_walk_record("ConvT_13", 512).expect("record");
        assert_eq!(record.estimate_secs(), Some(55.0));
        reset_node_walk_records();
    }

    /// An aborted full-walk projection is the only direct cost evidence for
    /// an unmodeled walk that has never completed. It must reach the same
    /// bounded grant arithmetic rather than being recorded and then ignored.
    #[test]
    fn aborted_projection_can_rescue_a_proxy_refusal() {
        reset_node_walk_records();
        record_node_walk_abort_projection("ConvT_13", 512, 20.0);
        let record = node_walk_record("ConvT_13", 512).expect("projection record");
        assert_eq!(record.completed_secs, None);
        assert_eq!(record.estimate_secs(), Some(20.0));
        assert_eq!(
            admit_walk_with_record(200.0, record.estimate_secs(), 10.0, 100.0, false),
            WalkAdmissionDecision::AdmitWithMeasuredGrant {
                grant_secs: 20.0 * WALK_ADMISSION_MARGIN,
            }
        );
        reset_node_walk_records();
    }

    /// Sub-floor walls carry no scheduling information and must not create a
    /// record — a degenerate 0.5ms "completion" could otherwise grant a
    /// doomed sub-share deadline where today's policy refuses cleanly, and it
    /// keeps the tiny walks of the pinned m1 collection tests out of the
    /// store entirely.
    #[test]
    fn sub_floor_measurements_are_not_recorded() {
        reset_node_walk_records();
        record_node_walk_completed("tiny", 2, WALK_CALIBRATION_MIN_SECS / 2.0);
        record_node_walk_abort_projection("tiny", 2, 0.001);
        assert!(node_walk_record("tiny", 2).is_none());
        record_node_walk_completed("tiny", 2, WALK_CALIBRATION_MIN_SECS);
        assert_eq!(
            node_walk_record("tiny", 2).and_then(|r| r.completed_secs),
            Some(WALK_CALIBRATION_MIN_SECS),
            "the floor itself is a valid measurement"
        );
        reset_node_walk_records();
    }

    /// The exact opt-out disables CONSULTING only.
    #[test]
    fn opt_out_parsing_is_exact() {
        for (raw, expected) in [
            (None, true),
            (Some("0"), true),
            (Some("true"), true),
            (Some("1"), false),
        ] {
            let resolved = ny_levers::read_with(
                &ny_levers::decls::collection::NO_WALK_RECORD_ADMISSION,
                |_| raw.map(str::to_owned),
            );
            assert_eq!(
                walk_record_admission_enabled_from_disabled(resolved.value.as_bool()),
                expected,
                "{raw:?}"
            );
        }
    }
}
