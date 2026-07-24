// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-objective input domain carrier for conjunctive and disjunctive
//! input splitting. Extracted from `shared.rs` to stay under file size limit.

use std::collections::HashMap;
use std::sync::Arc;

use ny_tensor::BoundedTensor;

use super::shared::cmp_input_domain_priority;
use crate::bounds::{GraphAlphaState, LinearBounds};

/// Per-objective bounds with optional CROWN linear coefficients for split scoring.
pub(super) type MultiObjBounds = (Vec<(f32, f32)>, Option<LinearBounds>);

/// Shared queue carrier for conjunctive and disjunctive multi-objective
/// input splitting. Part of #4116 Packet B0.
#[derive(Debug, Clone)]
pub(super) struct MultiObjInputDomain {
    pub(super) input_bounds: Arc<BoundedTensor>,
    pub(super) obj_bounds: Vec<(f32, f32)>,
    /// CROWN linear bounds for sensitivity-based split dimension selection.
    /// Shape: lower_a/upper_a are (num_specs, input_dim). The split scorer
    /// applies the SB heuristic across spec rows, including per-spec margin
    /// weighting when objective bounds are available (#1074).
    pub(super) linear_bounds: Option<LinearBounds>,
    pub(super) depth: usize,
    pub(super) priority: f32,
    /// When true, this domain still needs CROWN/IBP bounding (deferred from
    /// batch creation). Part of #4116 Packet B0; consumed in Packet B1.
    pub(super) needs_bounding: bool,
    /// Optional child-local node bounds override for complete-clip recompute.
    /// Part of #4116 Packet B0; consumed in Packet B1.
    pub(super) node_bounds_override: Option<Arc<HashMap<String, BoundedTensor>>>,
    /// Parent-refined α slopes used to warm-start per-sub-domain α refinement
    /// (`input_split_alpha_iteration > 0`). `None` (the default) keeps the
    /// frozen root-alpha behavior. Mirrors `GraphInputDomain::inherited_alpha_state`.
    pub(super) inherited_alpha_state: Option<Arc<GraphAlphaState>>,
}

impl Eq for MultiObjInputDomain {}
impl PartialEq for MultiObjInputDomain {
    fn eq(&self, other: &Self) -> bool {
        cmp_input_domain_priority(self.priority, other.priority) == std::cmp::Ordering::Equal
    }
}
impl PartialOrd for MultiObjInputDomain {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MultiObjInputDomain {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // NaN-priority domains surface first via cmp_input_domain_priority.
        // Fixes #3442: was partial_cmp().unwrap_or(Equal), hiding NaN domains.
        cmp_input_domain_priority(self.priority, other.priority)
    }
}
