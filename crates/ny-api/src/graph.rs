// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Low-level graph and network carrier types for already-built verification
//! models.
//!
//! Re-exports the types external consumers use to inspect, transport, and
//! manipulate realized network representations. [`SequentialNetwork`] is the
//! layer-based runtime verification model for sequential carriers. External
//! traced producers should build graphs through
//! `ny_api::model::{GraphModel, GraphModelBuilder}` and
//! `GraphModel::build_graph_network(...)`, not by treating this module as the
//! owned producer-construction contract. The curated public names in this
//! facade are [`GraphNetwork`], [`GraphNode`], [`SequentialNetwork`], and
//! [`NETWORK_INPUT`].

pub use ny_propagate::Network as SequentialNetwork;
pub use ny_propagate::{GraphNetwork, GraphNode, NETWORK_INPUT};

/// Soundness provenance and sqrt-domain guard inspection, for both carriers.
///
/// The `_graph` pair inspects a [`GraphNetwork`]; the `_network` pair inspects a
/// [`SequentialNetwork`]. Both are needed: a consumer that streams a sequential
/// carrier (see [`crate::streaming`]) has no graph to scan.
pub use ny_propagate::{
    count_sqrt_negative_domain_graph, count_sqrt_negative_domain_network,
    soundness_provenance_for_graph, soundness_provenance_for_network,
};

/// Per-node CROWN-IBP intermediate-bound collection results.
///
/// [`GraphNetwork::collect_crown_ibp_bounds_dag_with_status`] and its siblings
/// return a [`GraphCrownIbpBoundsResult`]: per-node bounds keyed by node name,
/// the [`BoundsProvenance`] that produced each one, and the
/// [`CrownIbpFallbackEvent`]s recorded along the way. Without these types the
/// collection API is callable but its result is unnameable, which forces a
/// consumer that wants per-layer bound reporting past the facade.
///
/// [`BoundsProvenance::ForwardFallback`] carries a [`CrownIbpFallbackReason`]:
/// some reasons (demand-pruned nodes, already-stable ReLU rows, rows outside the
/// objective cone, a vacuous CROWN result) are structural and mean the requested
/// work WAS completed, which is why
/// [`GraphCrownIbpBoundsResult::all_requested_crown_targets_completed`] is the
/// right question to ask rather than "were there any fallbacks".
pub use ny_propagate::types::{
    BoundsProvenance, CrownIbpFallbackEvent, CrownIbpFallbackReason, GraphCrownIbpBoundsResult,
};
