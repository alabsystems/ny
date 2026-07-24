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
}

/// Preset-configurable overrides for the graph CROWN-IBP collector's per-node
/// time-budget policy (#4413, #cgan-bn11-budget).
///
/// The collector splits the remaining warmup deadline equally across the
/// remaining tightening targets, clamps the share to a hard cap, and skips a
/// node (sound IBP fallback) when its share falls below a floor. Both knobs
/// were compile-time constants (`2.0 s` floor, `12.0 s` cap); benchmarks whose
/// dominant node needs a long chunked backward (cgan_2023's 28,800-dim
/// BatchNormalization_11: ~143 s measured for the full 7-node collection) can
/// now raise the cap through their preset.
///
/// `None` fields keep the built-in constants — semantics are byte-identical
/// when unset. Values must be finite and > 0; anything else is ignored in
/// favor of the default (fail-closed to the historical policy).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct CrownIbpPerNodeTimeBudget {
    /// Minimum useful per-node share in seconds; below it the node is skipped
    /// to IBP. `None` = built-in `MIN_PER_NODE_BUDGET_SECS` (2.0).
    #[serde(default)]
    pub floor_secs: Option<f64>,
    /// Hard cap on the equal-share per-node budget in seconds.
    /// `None` = built-in `MAX_GLOBAL_PER_NODE_BUDGET_SECS` (12.0).
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
    /// Bound was substituted with forward bounds due to a CROWN fallback condition.
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
}
