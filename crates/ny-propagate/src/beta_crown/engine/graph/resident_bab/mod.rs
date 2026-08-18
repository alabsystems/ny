// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Default-dark retained-v2 bridge for the frozen multi-objective graph lane.
//!
//! The first implementation slice contains only the versioned graph-layout,
//! split-history, and six-family composers. Runtime custody and provider
//! selection are added only after these pure producers are reviewed.

// Checkpoint A is deliberately default-dark. These producers become live in
// the separately reviewed custody/runtime checkpoint.
#![allow(dead_code, unused_imports)]

mod budget;
mod history;
mod operands;
#[path = "static.rs"]
mod static_payload;

use ny_core::GpuBabBoundArenaRange;

/// One true ReLU site in the exact backward fold order consumed by the
/// retained-v2 provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::beta_crown::engine::graph) struct ResidentBabReluSiteV1 {
    pub topology_node_id: u32,
    pub node_name: String,
    pub preactivation_width: usize,
}

/// Bit-exact history and dense signed-beta materialization for one domain.
#[derive(Debug, PartialEq)]
pub(in crate::beta_crown::engine::graph) struct ResidentBabHistoryBetaV1 {
    pub history_words: Vec<u32>,
    pub beta: Vec<f32>,
    pub beta_rows: Vec<GpuBabBoundArenaRange>,
}

pub(in crate::beta_crown::engine::graph) use budget::{
    ResidentBabAdapterHostCapV1, ResidentBabComposeErrorV1,
};
pub(in crate::beta_crown::engine::graph) use history::validate_append_suffix_v1;
pub(in crate::beta_crown::engine::graph) use operands::{
    compose_domain_operands_v1, ResidentBabActivationSourceV1, ResidentBabDomainOperandsV1,
    ResidentBabDomainSourceV1, ResidentBabFrontierSourceV1,
};
pub(in crate::beta_crown::engine::graph) use static_payload::{
    compose_static_payload_v1, ResidentBabStaticPayloadV1, ResidentBabStaticSourceV1,
};
