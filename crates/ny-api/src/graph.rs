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

/// Graph-level soundness provenance and sqrt-domain guard inspection.
pub use ny_propagate::{count_sqrt_negative_domain_graph, soundness_provenance_for_graph};
