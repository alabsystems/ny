// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN-IBP intermediate-bound collection results and fallback diagnostics.

use ny_tensor::BoundedTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reason why a layer fell back to forward bounds during CROWN-IBP collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrownIbpFallbackReason {
    /// Backward CROWN propagation failed for this layer.
    CrownPropagationError,
    /// CROWN and forward-bound tensor shapes were incompatible.
    ShapeMismatch,
    /// CROWN/forward intersection was empty, so forward bounds were kept.
    EmptyIntersection,
    /// Sequential CROWN Dense materialization exceeded the configured budget (#3515).
    MemoryBudgetExceeded,
    /// Aggregate time budget for patches-startable nodes exhausted (#3839).
    ///
    /// The graph-native collector limits total wall-clock time spent on
    /// patches-startable targets to prevent expensive Conv2d backward passes
    /// from dominating collection time. This node was patches-eligible, but
    /// earlier patches nodes already consumed the available budget.
    PatchesBudgetExceeded,
    /// Wall-clock deadline exceeded; used IBP bounds for this and remaining layers (#3328).
    DeadlineExceeded,
    /// Per-node time budget exceeded during backward CROWN pass (#3499).
    ///
    /// Unlike `DeadlineExceeded` (which skips all remaining nodes), this only
    /// affects the current target node. The tightening loop moves on to other
    /// nodes that may complete within their individual budgets.
    PerNodeDeadlineExceeded,
    /// IBP interval width below the skip threshold; CROWN tightening skipped (#3499).
    WidthBelowThreshold,
    /// Node not in the demand-driven selection set; no downstream nonlinear consumer
    /// requires tightened bounds at this producer (#3775).
    DemandDrivenSkip,
    /// Every demanded row consumed exclusively by ReLU successors is already
    /// stable under forward IBP, so backward CROWN has no unstable relaxation
    /// row to tighten.
    ///
    /// This is a structural, deterministic skip rather than a degradation or a
    /// time-budget truncation: the stored forward bound is the intended sound
    /// reference for the omitted rows. It must remain excluded from
    /// time-truncation completeness checks and degradation warning summaries.
    StableReluRowsSkipped,
    /// Sparse ReLU rows existed, but none lay in the current objective's
    /// backward influence cone. The collector retained sound IBP bounds because
    /// no selected row can affect the published objective.
    ///
    /// Like `StableReluRowsSkipped`, this is a deterministic structural skip,
    /// not a time/resource degradation. It is kept distinct so diagnostics do
    /// not falsely claim that every demanded ReLU row was phase-stable.
    ObjectiveConeRowsSkipped,
    /// CROWN COMPLETED for this node and returned a VACUOUS relation: it lost
    /// finiteness on ~every element the forward IBP pass had bounded, and
    /// materially tightened ~none of them (#crown-honest-provenance).
    ///
    /// This is what used to be recorded as `BoundsProvenance::Crown`. Measured on
    /// TinyYOLO / yolo_2023 (2026-07-29): four of eight demanded targets reported
    /// `Crown` while having concretized to `[-inf, inf]`, which is why a 20x
    /// per-node cap sweep and a 300 -> 900 s budget increase both left the root
    /// bound BYTE-IDENTICAL. More budget bought more of a computation that was
    /// returning nothing, and nothing in NY could see that.
    ///
    /// The STORED BOUND is unchanged by this tag (it is the same IBP-dominated
    /// intersection); only the quality claim is corrected.
    ///
    /// DETERMINISTIC, not time-based: finiteness is a pure function of
    /// (graph, input, options). This reason must therefore NOT be added to
    /// `crown_ibp_result_is_complete`'s time-reason list -- a collection carrying
    /// it is still complete and still cacheable.
    CrownVacuousResult,
    /// The collector REFUSED to start this target's backward walk because its
    /// MACs-based cost estimate exceeded the node's per-node time share
    /// (#cprime-admission).
    ///
    /// Estimate-then-refuse replaces start-then-burn: measured on
    /// tinyimagenet (2026-08-02), Conv_17 burned 150.29 s of its share and
    /// delivered zero completed backward steps before degrading to the same
    /// IBP bound a refusal produces in ~0 s. Refusal never consumes the
    /// share, so the unspent time rolls forward to later demanded targets
    /// that fit — the opposite of the floor-inflation failure (floor100
    /// smoke, −7905), which granted time instead of freeing it.
    ///
    /// TIME-CLASS, like `PerNodeDeadlineExceeded`: the decision depends on
    /// the live share, so a re-run with more budget can admit the walk. It
    /// therefore belongs in `crown_ibp_result_is_complete`'s time-reason
    /// list and is a degradation (not a structural skip) in summaries.
    WalkCostRefused,
    /// A deadline-bounded objective-row collector completed at least zero but
    /// fewer than all CROWN chunks. Fully completed rows were retained and
    /// intersected with the certified forward box; every unfinished (and every
    /// late in-flight) row remained exactly at that forward box.
    ///
    /// This is deliberately a fallback provenance, not `Crown`: the hybrid
    /// bound is sound and can be tighter on completed rows, but the target
    /// collection was truncated and must not acquire complete-cache or
    /// proof-authority status. The associated fallback event records the exact
    /// completed/total row counts.
    ///
    /// Kept at the end of this serialized enum so adding it does not renumber
    /// any pre-existing variant in non-self-describing serde formats.
    PartialCrownRowsDeadlineExceeded,
}

/// Preset-configurable overrides for CROWN-IBP collectors' per-node
/// time-budget policy (#4413, #cgan-bn11-budget).
///
/// The collector assigns a remaining-deadline share to each tightening target
/// and skips a node (sound IBP fallback) when its share falls below a floor.
/// An explicit cap clamps that share. Without one, the cap is adaptive: 25% of
/// the remaining collection budget, clamped to 12–600 seconds. Explicit caps
/// are dimension-scaled above the 28,800-row reference width.
///
/// The historical `2.0 s` floor remains the default. Values must be finite and
/// > 0; anything else is ignored in favor of the default policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct CrownIbpPerNodeTimeBudget {
    /// Minimum useful per-node share in seconds; below it the node is skipped
    /// to IBP. `None` = built-in `MIN_PER_NODE_BUDGET_SECS` (2.0).
    #[serde(default)]
    pub floor_secs: Option<f64>,
    /// Explicit base cap on a per-node budget in seconds. `None` selects the
    /// adaptive remaining-budget cap described above.
    #[serde(default)]
    pub cap_secs: Option<f64>,
}

/// Provenance tag for a bound returned by CROWN-IBP collection.
///
/// This makes fallback substitution explicit at each layer/node instead of
/// requiring callers to infer it from separate diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundsProvenance {
    /// Bound was produced by CROWN (or CROWN∩IBP) without fallback substitution.
    Crown,
    /// Bound remains fallback-grade due to a CROWN fallback condition.
    ///
    /// Usually this is the forward bound exactly. The
    /// `PartialCrownRowsDeadlineExceeded` case may retain fully completed CROWN
    /// rows after intersecting them with the forward bound, but it intentionally
    /// remains in this arm because the target collection did not complete.
    ForwardFallback(CrownIbpFallbackReason),
}

impl BoundsProvenance {
    /// Returns true if this provenance indicates a fallback substitution.
    pub fn is_fallback(self) -> bool {
        matches!(self, Self::ForwardFallback(_))
    }

    /// Returns the fallback reason when this provenance indicates substitution.
    pub fn fallback_reason(self) -> Option<CrownIbpFallbackReason> {
        match self {
            Self::ForwardFallback(reason) => Some(reason),
            Self::Crown => None,
        }
    }
}

/// One fallback event captured during CROWN-IBP intermediate-bound collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrownIbpFallbackEvent {
    /// Layer index where fallback occurred.
    pub layer_index: usize,
    /// Layer type for diagnostics.
    pub layer_type: String,
    /// Structured fallback reason.
    pub reason: CrownIbpFallbackReason,
    /// Human-readable details (error text, shape values, etc.).
    pub details: String,
}

/// Full result of CROWN-IBP intermediate-bound collection.
#[derive(Debug, Clone)]
pub struct CrownIbpBoundsResult {
    /// Per-layer intermediate bounds (same as `collect_crown_ibp_bounds`).
    pub bounds: Vec<BoundedTensor>,
    /// Per-layer provenance for each bound in `bounds`.
    pub provenance: Vec<BoundsProvenance>,
    /// Fallback events recorded while computing bounds.
    pub fallback_events: Vec<CrownIbpFallbackEvent>,
}

impl CrownIbpBoundsResult {
    /// Returns true when at least one layer used forward-bound fallback.
    pub fn has_fallbacks(&self) -> bool {
        !self.fallback_events.is_empty()
    }

    /// Number of fallback events recorded across all layers.
    pub fn fallback_count(&self) -> usize {
        self.fallback_events.len()
    }

    /// The first layer index where fallback occurred, if any.
    pub fn first_fallback_layer(&self) -> Option<usize> {
        self.fallback_events.first().map(|event| event.layer_index)
    }

    /// Provenance tag for a specific layer index.
    pub fn provenance_for_layer(&self, layer_index: usize) -> Option<BoundsProvenance> {
        self.provenance.get(layer_index).copied()
    }
}

/// Result of DAG-CROWN backward propagation with provenance metadata.
///
/// Wraps a `BoundedTensor` with information about whether the bounds came from
/// actual CROWN backward propagation or were silently replaced with forward bounds
/// due to invalid CROWN output (NaN/Inf or inverted intervals).
///
/// Used by `crown_backward_with_relaxation_and_provenance` to make the fallback
/// at `propagation.rs:488-496` observable to callers.
#[derive(Debug, Clone)]
pub struct CrownBackwardResult {
    /// The computed output bounds.
    pub bounds: BoundedTensor,
    /// Provenance indicating whether CROWN or forward-fallback was used.
    pub provenance: BoundsProvenance,
}

impl CrownBackwardResult {
    /// Returns true if the output bounds were produced by a forward-bound fallback.
    pub fn is_fallback(&self) -> bool {
        self.provenance.is_fallback()
    }
}

/// Full result of DAG CROWN-IBP intermediate-bound collection.
///
/// This mirrors [`CrownIbpBoundsResult`] for graph networks, where bounds are
/// keyed by node name instead of layer index.
#[derive(Debug, Clone)]
pub struct GraphCrownIbpBoundsResult {
    /// Per-node intermediate bounds.
    pub bounds: HashMap<String, BoundedTensor>,
    /// Per-node provenance.
    pub provenance: HashMap<String, BoundsProvenance>,
    /// Fallback events recorded while computing bounds.
    pub fallback_events: Vec<CrownIbpFallbackEvent>,
}

impl GraphCrownIbpBoundsResult {
    /// Returns true when at least one node used forward-bound fallback.
    pub fn has_fallbacks(&self) -> bool {
        !self.fallback_events.is_empty()
    }

    /// Number of fallback events recorded across all nodes.
    pub fn fallback_count(&self) -> usize {
        self.fallback_events.len()
    }

    /// The first topological index where fallback occurred, if any.
    pub fn first_fallback_layer(&self) -> Option<usize> {
        self.fallback_events.first().map(|event| event.layer_index)
    }

    /// Provenance tag for a specific node.
    pub fn provenance_for_node(&self, node_name: &str) -> Option<BoundsProvenance> {
        self.provenance.get(node_name).copied()
    }

    /// Whether every target requested by the collector completed its CROWN
    /// computation.
    ///
    /// Structural omissions are not requested work: demand-pruned nodes,
    /// already-stable ReLU rows, and rows outside the objective cone are
    /// intentionally left at their certified forward bounds. A vacuous CROWN
    /// result also completed (it simply failed to improve that bound). Every
    /// other fallback means some requested target was not completed and must
    /// keep an explicit partial status even though the returned map is sound.
    pub fn all_requested_crown_targets_completed(&self) -> bool {
        fn reason_is_completed_or_structural(reason: CrownIbpFallbackReason) -> bool {
            matches!(
                reason,
                CrownIbpFallbackReason::DemandDrivenSkip
                    | CrownIbpFallbackReason::StableReluRowsSkipped
                    | CrownIbpFallbackReason::ObjectiveConeRowsSkipped
                    | CrownIbpFallbackReason::CrownVacuousResult
            )
        }

        self.provenance.values().all(|provenance| match provenance {
            BoundsProvenance::Crown => true,
            BoundsProvenance::ForwardFallback(reason) => reason_is_completed_or_structural(*reason),
        }) && self
            .fallback_events
            .iter()
            .all(|event| reason_is_completed_or_structural(event.reason))
    }
}
