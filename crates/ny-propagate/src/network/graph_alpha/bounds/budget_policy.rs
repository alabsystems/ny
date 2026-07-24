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

use crate::network::core::GraphNetwork;
use crate::types::{
    BoundsProvenance, CrownIbpFallbackEvent, CrownIbpFallbackReason, CrownIbpPerNodeTimeBudget,
};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Minimum useful per-node time budget in seconds for DAG CROWN-IBP (#3499).
pub(crate) const MIN_PER_NODE_BUDGET_SECS: f64 = 2.0;

/// Hard cap on the global per-node time budget in seconds (#4413).
///
/// This keeps initial-bound collection broad by preventing one late target from
/// monopolizing the entire remaining warmup budget.
pub(super) const MAX_GLOBAL_PER_NODE_BUDGET_SECS: f64 = 12.0;

/// Sanitize a preset-supplied budget override: finite and > 0, else the default.
fn sanitize_budget_secs(override_secs: Option<f64>, default_secs: f64) -> f64 {
    override_secs
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default_secs)
}

/// Effective floor/cap for the equal-share per-node budget after applying
/// preset overrides (#cgan-bn11-budget). Unset/invalid fields keep the
/// built-in #3499/#4413 constants, so behavior is unchanged when no preset
/// sets the knobs.
pub(super) fn effective_per_node_time_budget(
    budget: &CrownIbpPerNodeTimeBudget,
) -> (/* floor */ f64, /* cap */ f64) {
    (
        sanitize_budget_secs(budget.floor_secs, MIN_PER_NODE_BUDGET_SECS),
        sanitize_budget_secs(budget.cap_secs, MAX_GLOBAL_PER_NODE_BUDGET_SECS),
    )
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
/// memory-light patches path. Still bounded per-node by
/// `MAX_GLOBAL_PER_NODE_BUDGET_SECS` and by the collection's own deadline.
pub(super) const CONV_PATCHES_COLLECT_TIGHTENING_BUDGET_SECS: f64 = 40.0;

const PATCHES_BUDGET_ENV: &str = "NY_PATCHES_BUDGET_SECS";

fn dense_identity_exceeds_budget(bounds: &BoundedTensor, budget: usize) -> bool {
    crate::network::crown_memory::identity_pair_bytes(bounds.len())
        .map_or(true, |required| required > budget)
}

/// Auto objective row-chunk size for a target whose dense `[dim x dim]`
/// identity pair exceeds the CPU dense budget (#cgan-bn11-chunk).
///
/// Mirrors the memory proxy of the budget gate
/// (`crown_memory::identity_pair_bytes`): the chunked backward's peak seed is
/// a `[C x dim]` coefficient pair, i.e. the identity pair scaled by `C / dim`.
/// Picking `C = max(1, dim * budget / identity_pair_bytes(dim))` keeps that
/// scaled pair at (or just under) `budget`. Computed in u128 so huge dims
/// cannot overflow; clamped to `dim` (a full-size "chunk" is a single pass).
pub(super) fn auto_objective_chunk_rows(node_dim: usize, budget_bytes: usize) -> usize {
    let dim = node_dim.max(1) as u128;
    // 2 (lower/upper) * f32 * dim * dim — same quantity identity_pair_bytes
    // estimates, but overflow-free.
    let pair_bytes = 2u128 * (size_of::<f32>() as u128) * dim * dim;
    let rows = (dim * budget_bytes as u128) / pair_bytes.max(1);
    rows.clamp(1, node_dim.max(1) as u128) as usize
}

pub(super) fn patches_tightening_budget_secs() -> f64 {
    // Explicit override always wins.
    if let Some(v) = std::env::var(PATCHES_BUDGET_ENV)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
    {
        return v;
    }
    // #conv-patches-collect: with the padded-conv patches composition enabled,
    // the deep spatial conv targets (e.g. metaroom's 4 conv stages) now tighten
    // via the memory-light patches path, but the 5s default aggregate budget only
    // funds ~2 of them before the rest degrade to loose IBP (measured: Conv3/Conv4
    // hit PatchesBudgetExceeded, keeping their 8.7 / 72.0 widths that dominate the
    // root margin). Raise the aggregate cap so every conv stage gets its patches
    // backward. The per-node global cap (12s) and the collection's own deadline
    // still bound wall time, so this cannot run away; it only lets the collection
    // spend more of the ALREADY-reserved warmup slice on the nodes that matter.
    if std::env::var_os("NY_CONV_PATCHES_COLLECT").is_some_and(|v| v != "0" && !v.is_empty()) {
        return CONV_PATCHES_COLLECT_TIGHTENING_BUDGET_SECS;
    }
    DEFAULT_PATCHES_TIGHTENING_BUDGET_SECS
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

    let (floor_secs, cap_secs) = effective_per_node_time_budget(budget);
    let share = remaining_secs / remaining_candidates as f64;
    let capped = share.min(cap_secs);
    (capped >= floor_secs).then_some(capped)
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
fn dim_cap_scale_enabled() -> bool {
    std::env::var("NY_DIM_CAP_SCALE").ok().as_deref() != Some("0")
}

/// Scale a PRESET-SUPPLIED per-node cap by the target's width (#cgan-dim-cap).
///
/// Only fires when the preset opted into a custom cap (`cap_secs` set) — the
/// built-in 12 s default cap is never scaled, so non-preset benchmarks are
/// untouched. Quadratic in `dims / PRESET_CAP_REFERENCE_DIMS` (cost model
/// above), clamped to `DIM_SCALED_CAP_CEILING_SECS`, never below the preset
/// cap itself. Sound either way: the cap only bounds how long the chunked
/// CROWN backward may run; exceeding it degrades the node to IBP.
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
/// #cgan-dim-cap: `this_weight` is the node's objective-row count, i.e. its
/// width — a preset-supplied cap is therefore scaled per-node via
/// `dim_scaled_cap_secs` so a >40k-dim target is not starved by a cap that was
/// measured for a 28,800-dim one.
pub(crate) fn compute_weighted_per_node_budget_secs(
    remaining_secs: f64,
    remaining_weight_sum: f64,
    this_weight: f64,
    budget: &CrownIbpPerNodeTimeBudget,
) -> Option<f64> {
    if !remaining_secs.is_finite()
        || remaining_secs <= 0.0
        || !remaining_weight_sum.is_finite()
        || remaining_weight_sum <= 0.0
        || !this_weight.is_finite()
        || this_weight <= 0.0
    {
        return None;
    }

    let (floor_secs, cap_secs) = effective_per_node_time_budget(budget);
    let preset_cap_set = budget.cap_secs.is_some_and(|v| v.is_finite() && v > 0.0);
    let cap_secs = dim_scaled_cap_secs(cap_secs, preset_cap_set, this_weight);
    let share = remaining_secs * (this_weight / remaining_weight_sum);
    let capped = share.min(cap_secs);
    (capped >= floor_secs).then_some(capped)
}

pub(super) struct PatchesTighteningBudget {
    total_secs: f64,
    remaining_secs: f64,
}

impl PatchesTighteningBudget {
    pub(super) fn new() -> Self {
        let total_secs = patches_tightening_budget_secs();
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
    use super::{
        compute_global_per_node_budget_secs, compute_weighted_per_node_budget_secs,
        count_remaining_budget_candidates, effective_per_node_time_budget,
        sum_remaining_budget_weights, CrownIbpPerNodeTimeBudget, MAX_GLOBAL_PER_NODE_BUDGET_SECS,
        MIN_PER_NODE_BUDGET_SECS,
    };

    const DEFAULT_BUDGET: CrownIbpPerNodeTimeBudget = CrownIbpPerNodeTimeBudget {
        floor_secs: None,
        cap_secs: None,
    };

    #[test]
    fn test_compute_global_per_node_budget_secs_caps_long_deadline_4413() {
        assert_eq!(
            compute_global_per_node_budget_secs(210.0, 1, &DEFAULT_BUDGET),
            Some(MAX_GLOBAL_PER_NODE_BUDGET_SECS)
        );
    }

    #[test]
    fn test_compute_global_per_node_budget_secs_uses_equal_share_4413() {
        assert_eq!(
            compute_global_per_node_budget_secs(24.0, 4, &DEFAULT_BUDGET),
            Some(6.0)
        );
    }

    #[test]
    fn test_compute_global_per_node_budget_secs_respects_floor_4413() {
        assert_eq!(
            compute_global_per_node_budget_secs(6.0, 4, &DEFAULT_BUDGET),
            None
        );
    }

    #[test]
    fn test_weighted_budget_gives_wide_node_a_bigger_slice_cgan_cost_weight() {
        // 100s envelope, weights BN_11=28800 vs a 1200-dim node: the wide node's
        // cost-proportional slice (96s) saturates the MAX cap, the small one gets
        // a tiny slice — vs the equal-share 50s each that starved BN_11.
        let weights = [28800.0_f64, 1200.0];
        let sum = sum_remaining_budget_weights(&weights, 0);
        assert_eq!(sum, 30000.0);
        let wide = compute_weighted_per_node_budget_secs(100.0, sum, weights[0], &DEFAULT_BUDGET);
        let small = compute_weighted_per_node_budget_secs(100.0, sum, weights[1], &DEFAULT_BUDGET);
        // wide share = 100 * 28800/30000 = 96s, saturates the MAX cap.
        assert_eq!(wide, Some(MAX_GLOBAL_PER_NODE_BUDGET_SECS));
        // small share = 100 * 1200/30000 = 4s (above the 2s floor, below the cap).
        assert_eq!(small, Some(4.0));
    }

    #[test]
    fn test_weighted_budget_reduces_to_equal_share_when_weights_match() {
        // Equal weights => each gets remaining/N, matching the equal-share fn.
        let weights = [10.0_f64, 10.0, 10.0, 10.0];
        let sum = sum_remaining_budget_weights(&weights, 0);
        assert_eq!(
            compute_weighted_per_node_budget_secs(24.0, sum, weights[0], &DEFAULT_BUDGET),
            compute_global_per_node_budget_secs(24.0, 4, &DEFAULT_BUDGET),
        );
    }

    #[test]
    fn test_weighted_budget_respects_floor_and_zero_weight() {
        // Below-floor weighted share => None (degrades to IBP, sound).
        assert_eq!(
            compute_weighted_per_node_budget_secs(6.0, 40.0, 10.0, &DEFAULT_BUDGET),
            None
        );
        // Zero / non-finite guards => None.
        assert_eq!(
            compute_weighted_per_node_budget_secs(100.0, 30000.0, 0.0, &DEFAULT_BUDGET),
            None
        );
        assert_eq!(
            compute_weighted_per_node_budget_secs(100.0, 0.0, 10.0, &DEFAULT_BUDGET),
            None
        );
    }

    #[test]
    fn test_per_node_budget_default_matches_old_constants_cgan_bn11_budget() {
        // Unset preset knobs == the historical #3499/#4413 constants.
        assert_eq!(
            effective_per_node_time_budget(&DEFAULT_BUDGET),
            (MIN_PER_NODE_BUDGET_SECS, MAX_GLOBAL_PER_NODE_BUDGET_SECS)
        );
    }

    #[test]
    fn test_per_node_budget_cap_override_reaches_computation_cgan_bn11_budget() {
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
        for bad in [Some(f64::NAN), Some(f64::INFINITY), Some(0.0), Some(-3.0)] {
            let budget = CrownIbpPerNodeTimeBudget {
                floor_secs: bad,
                cap_secs: bad,
            };
            assert_eq!(
                effective_per_node_time_budget(&budget),
                (MIN_PER_NODE_BUDGET_SECS, MAX_GLOBAL_PER_NODE_BUDGET_SECS)
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
        // cgan_2023 preset shape: cap 150 s measured for a 28,800-dim node.
        let budget = CrownIbpPerNodeTimeBudget {
            floor_secs: None,
            cap_secs: Some(150.0),
        };
        // The 61,504-dim imgSz64 generator target: ratio 2.1355..^2 = 4.56;
        // 150 * 4.56 = 684 s clamps to the 600 s ceiling. With 765 s remaining
        // and this node holding ~55% of the weight, the share (~424 s) now
        // reaches the node instead of being truncated to 150 s.
        let share = compute_weighted_per_node_budget_secs(765.0, 111_456.0, 61_504.0, &budget)
            .expect("share above floor");
        let expected_share = 765.0 * (61_504.0 / 111_456.0);
        assert!(
            (share - expected_share).abs() < 1e-9,
            "share {share} truncated"
        );
        assert!(share > 150.0, "flat preset cap must no longer truncate");

        // A hypothetical even-wider node saturates the 600 s ceiling.
        let saturated =
            compute_weighted_per_node_budget_secs(10_000.0, 130_000.0, 123_008.0, &budget)
                .expect("share above floor");
        assert_eq!(saturated, super::DIM_SCALED_CAP_CEILING_SECS);

        // The reference-width node itself keeps the flat preset cap (ratio 1).
        assert_eq!(
            compute_weighted_per_node_budget_secs(765.0, 30_000.0, 28_800.0, &budget),
            Some(150.0)
        );
    }

    #[test]
    fn test_dim_scaled_cap_leaves_default_cap_alone_cgan_dim_cap() {
        // No preset cap: the built-in 12 s cap must NOT scale with node width,
        // so non-preset benchmarks keep their historical budget shape.
        let wide =
            compute_weighted_per_node_budget_secs(765.0, 111_456.0, 61_504.0, &DEFAULT_BUDGET)
                .expect("share above floor");
        assert_eq!(wide, MAX_GLOBAL_PER_NODE_BUDGET_SECS);
    }

    #[test]
    fn test_dim_scaled_cap_respects_preset_caps_above_ceiling_cgan_dim_cap() {
        // A preset cap larger than the ceiling wins (the scaler never SHRINKS
        // a preset cap).
        let budget = CrownIbpPerNodeTimeBudget {
            floor_secs: None,
            cap_secs: Some(700.0),
        };
        assert_eq!(
            compute_weighted_per_node_budget_secs(10_000.0, 100_000.0, 61_504.0, &budget),
            Some(700.0)
        );
    }

    #[test]
    fn test_auto_objective_chunk_rows_imgsz64_widest_node_cgan_dim_cap() {
        // imgSz64_nCh_3 widest generator node: 61,504 dims ([16, 62, 62]).
        // Full identity pair = 8 * 61504^2 ≈ 30.3 GB; the auto chunk must
        // scale the [C x dim] pair under a 2 GiB budget without overflow.
        let budget = 2usize * 1024 * 1024 * 1024;
        let rows = super::auto_objective_chunk_rows(61_504, budget);
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
        let rows = super::auto_objective_chunk_rows(28_800, budget);
        assert_eq!(rows, 9_320);
        // The chunk's [C x dim] pair stays within budget.
        assert!(2 * 4 * rows * 28_800 <= budget);
        // And it is a genuine chunk (multiple passes required).
        assert!(rows < 28_800);
    }

    #[test]
    fn test_auto_objective_chunk_rows_clamps_to_dim_and_floor() {
        // Under-budget target: clamp to dim (equivalent to a single pass).
        assert_eq!(super::auto_objective_chunk_rows(16, usize::MAX >> 1), 16);
        // Tiny budget: floor at 1 row so the chunked loop always progresses.
        assert_eq!(super::auto_objective_chunk_rows(1_000_000, 1), 1);
        // Degenerate dim.
        assert_eq!(super::auto_objective_chunk_rows(0, 1024), 1);
    }
}
