// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain types for β-CROWN branch-and-bound search.
//!
//! Contains:
//! - `GraphCrownContext`: Context for graph CROWN propagation
//! - `GraphPrecomputedBounds`: Pre-computed CROWN-IBP bounds
//! - `MultiObjectiveTargets`: Multi-objective verification targets
//! - `DomainProcessingConfig`: Domain processing configuration
//! - `GraphBabDomain`/`MultiObjectiveGraphBabDomain`: Graph network domains
//! - `BabDomain`: Sequential network domains
//! - `IntermediateLinearBounds`: Intermediate CROWN bounds storage

mod context;
mod graph;
mod multi_objective;
mod sequential;

/// NaN-aware priority comparison for BaB domain queue ordering.
///
/// NaN priorities are treated as highest priority (popped first) to surface
/// invalid domains immediately rather than letting them accumulate silently.
/// For finite values, uses `f32::total_cmp` for deterministic IEEE 754 ordering.
///
/// Used by `BabDomain`, `GraphBabDomain`, and `MultiObjectiveGraphBabDomain`
/// `Ord` implementations for `BinaryHeap` max-heap ordering.
pub(crate) fn cmp_domain_priority(lhs: f32, rhs: f32) -> std::cmp::Ordering {
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (true, true) => std::cmp::Ordering::Equal,
        (false, false) => lhs.total_cmp(&rhs),
    }
}

/// #cone-delta: conservative "delta unknown" marker for `delta_pre_nodes`.
///
/// Used wherever a domain's `node_bounds` was NOT installed by one of the
/// post-bounding replacement sites this invariant is defined against (root
/// construction, GPU `from_metadata` reconstruction, clip-replaced shim maps).
/// A `NETWORK_INPUT` entry is the natural conservative over-approximation:
/// "anything up to the network input may have changed since the map was
/// written". The delta-seed gate in
/// `compute_constrained_forward_bounds_inner` rejects any delta containing
/// `NETWORK_INPUT`, so these domains always take the full-history seed path
/// (today's behavior) — and even if a bug let the sentinel through,
/// `descendants_inclusive` treats a `NETWORK_INPUT` seed as reaching the whole
/// graph, i.e. a full recompute. Fail-closed twice over.
pub(crate) fn delta_pre_nodes_unknown() -> Vec<String> {
    vec![crate::NETWORK_INPUT.to_string()]
}

// Re-export all public types to preserve `crate::beta_crown::domain::*` paths.
pub use context::{
    DomainProcessingConfig, GraphCrownContext, GraphPrecomputedBounds, MultiObjectiveTargets,
};
pub use graph::GraphBabDomain;
pub use multi_objective::MultiObjectiveGraphBabDomain;
pub use sequential::{BabDomain, IntermediateLinearBounds};

/// Domain with unstable neurons for parallel processing.
///
/// Tuple of (domain_index, domain_ref, unstable_neurons).
/// Used in parallel domain verification to track which neurons need splitting.
pub type DomainWithUnstable<'a> = (usize, &'a GraphBabDomain, Vec<(String, usize)>);

/// Multi-objective domain with unstable neurons for parallel processing.
///
/// Tuple of (domain_index, domain_ref, unstable_neurons).
/// Used in parallel multi-objective verification.
pub type MultiObjDomainWithUnstable<'a> = (
    usize,
    &'a MultiObjectiveGraphBabDomain,
    Vec<(String, usize)>,
);
