// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Context and configuration types for β-CROWN domain processing.
//!
//! - `GraphCrownContext`: Bundles parameters for graph CROWN propagation
//! - `GraphPrecomputedBounds`: Pre-computed CROWN-IBP bounds
//! - `MultiObjectiveTargets`: Multi-objective verification targets
//! - `DomainProcessingConfig`: Parallel domain processing configuration

use std::sync::Arc;
use std::time::Instant;

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::state::GraphDomainAlphaState;

use super::{NodeBoundsMap, NodeBoundsView};

///
/// Bundles common parameters for graph CROWN propagation to reduce function argument counts.
/// This struct holds references to propagation context that are typically passed together.
#[derive(Clone, Copy)]
pub struct GraphCrownContext<'a> {
    /// Split history for constraint tracking.
    pub history: &'a GraphSplitHistory,
    /// Optional cut pool for cutting planes.
    pub cut_pool: Option<&'a GraphCutPool>,
    /// Optional pre-computed bounds from CROWN-IBP, exposed only through an
    /// opaque read-only view over either supported owner. [`Self::new`] keeps
    /// accepting the legacy standard-map reference; direct struct literals
    /// must wrap one with [`NodeBoundsView::from_hash_map`].
    pub base_bounds: Option<NodeBoundsView<'a>>,
    /// Optional GPU/accelerated GEMM engine.
    pub engine: Option<&'a dyn GemmEngine>,
    /// Optional per-domain alpha state for optimized ReLU relaxation slopes.
    /// When present, the backward pass uses `propagate_linear_with_alpha` instead
    /// of the fixed heuristic, enabling joint α-β optimization.
    /// Issue: #1841
    pub alpha_state: Option<&'a GraphDomainAlphaState>,
    /// #cone-delta: pre-activation nodes of the constraints added since
    /// `base_bounds` was last fixpointed (`GraphBabDomain::delta_pre_nodes`).
    /// `None` = delta unknown ⇒ the constrained forward keeps its full-history
    /// seeds (fail-closed). Only consulted behind `NY_CONE_REFRESH=1`.
    pub delta_seeds: Option<&'a [String]>,
}

impl<'a> GraphCrownContext<'a> {
    /// Create a new graph CROWN context.
    pub fn new(
        history: &'a GraphSplitHistory,
        cut_pool: Option<&'a GraphCutPool>,
        base_bounds: Option<&'a std::collections::HashMap<String, Arc<BoundedTensor>>>,
        engine: Option<&'a dyn GemmEngine>,
    ) -> Self {
        Self {
            history,
            cut_pool,
            base_bounds: base_bounds.map(NodeBoundsView::from_hash_map),
            engine,
            alpha_state: None,
            delta_seeds: None,
        }
    }

    /// Create a context borrowing the provenance-tracked multi-objective
    /// node-bound carrier. Shared propagation receives only a read-only view.
    pub(crate) fn new_with_node_bounds_map(
        history: &'a GraphSplitHistory,
        cut_pool: Option<&'a GraphCutPool>,
        base_bounds: Option<&'a NodeBoundsMap>,
        engine: Option<&'a dyn GemmEngine>,
    ) -> Self {
        Self {
            history,
            cut_pool,
            base_bounds: base_bounds.map(NodeBoundsView::from_node_bounds_map),
            engine,
            alpha_state: None,
            delta_seeds: None,
        }
    }

    /// Create a minimal context with just history.
    pub fn for_history(history: &'a GraphSplitHistory) -> Self {
        Self {
            history,
            cut_pool: None,
            base_bounds: None,
            engine: None,
            alpha_state: None,
            delta_seeds: None,
        }
    }

    /// Create a minimal context with history and GPU engine (#3597).
    ///
    /// Like [`for_history`] but threads the engine for GPU-accelerated
    /// CROWN backward passes during BaB domain processing.
    pub fn for_history_and_engine(
        history: &'a GraphSplitHistory,
        engine: Option<&'a dyn GemmEngine>,
    ) -> Self {
        Self {
            history,
            cut_pool: None,
            base_bounds: None,
            engine,
            alpha_state: None,
            delta_seeds: None,
        }
    }

    /// Add a GPU engine to this context (#3597).
    ///
    /// Builder-style method for threading GPU acceleration into an
    /// existing context.
    pub fn with_engine(mut self, engine: Option<&'a dyn GemmEngine>) -> Self {
        self.engine = engine;
        self
    }

    /// Add alpha state for optimized ReLU relaxation.
    ///
    /// Returns a new context with the given alpha state attached.
    /// This enables `propagate_linear_with_alpha` in the backward pass.
    pub fn with_alpha(mut self, alpha_state: &'a GraphDomainAlphaState) -> Self {
        self.alpha_state = Some(alpha_state);
        self
    }

    /// #cone-delta: attach the domain's delta pre-nodes (builder-style).
    ///
    /// Pair with the SAME domain whose inherited `node_bounds` is passed as
    /// `base_bounds` — the delta is only meaningful relative to that map.
    pub fn with_delta_seeds(mut self, delta_seeds: &'a [String]) -> Self {
        self.delta_seeds = Some(delta_seeds);
        self
    }
}

/// Pre-computed bounds for graph verification.
///
/// Bundles pre-computed node and output bounds from CROWN-IBP to reduce argument counts.
/// These bounds are computed once and reused across multiple verification calls.
pub struct GraphPrecomputedBounds<'a> {
    /// Pre-computed intermediate node bounds.
    pub node_bounds: &'a std::collections::HashMap<String, BoundedTensor>,
    /// Pre-computed output bounds.
    pub output_bounds: &'a BoundedTensor,
}

impl<'a> GraphPrecomputedBounds<'a> {
    /// Create new pre-computed bounds.
    pub fn new(
        node_bounds: &'a std::collections::HashMap<String, BoundedTensor>,
        output_bounds: &'a BoundedTensor,
    ) -> Self {
        Self {
            node_bounds,
            output_bounds,
        }
    }
}

/// Multi-objective verification targets.
///
/// Bundles objective vectors, thresholds, and verification status for multi-objective SPSA.
pub struct MultiObjectiveTargets<'a> {
    /// Objective coefficient vectors for each property.
    pub objectives: &'a [Vec<f32>],
    /// Threshold values for each property.
    pub thresholds: &'a [f32],
    /// Mask indicating which properties are already verified.
    pub verified_mask: &'a [bool],
}

impl<'a> MultiObjectiveTargets<'a> {
    /// Create new multi-objective targets.
    ///
    /// Defense-in-depth: asserts that all three slices have the same length.
    /// The entry-point guard in `verify_graph_relu_split_multi_objective_core`
    /// validates objectives/thresholds before we get here, but this catches
    /// any future callers that bypass the entry point (#3383).
    pub fn new(
        objectives: &'a [Vec<f32>],
        thresholds: &'a [f32],
        verified_mask: &'a [bool],
    ) -> Self {
        debug_assert_eq!(
            objectives.len(),
            thresholds.len(),
            "MultiObjectiveTargets::new(): objectives/thresholds length mismatch ({} vs {}) (#3383)",
            objectives.len(),
            thresholds.len()
        );
        debug_assert_eq!(
            objectives.len(),
            verified_mask.len(),
            "MultiObjectiveTargets::new(): objectives/verified_mask length mismatch ({} vs {}) (#3383)",
            objectives.len(),
            verified_mask.len()
        );
        Self {
            objectives,
            thresholds,
            verified_mask,
        }
    }
}

/// Configuration for domain processing in the BaB loop.
///
/// Bundles processing options for the domain processing functions.
pub struct DomainProcessingConfig {
    /// Threshold for verification.
    pub threshold: f32,
    /// Whether to create children in parallel.
    pub use_parallel_children: bool,
    /// Wall-clock deadline for α-CROWN optimization inside input-split child
    /// creation (#2724). When set, per-domain alpha optimization bails early
    /// if the timeout budget is exhausted.
    pub deadline: Option<Instant>,
}

impl DomainProcessingConfig {
    /// Create new domain processing configuration.
    pub fn new(threshold: f32, use_parallel_children: bool) -> Self {
        Self {
            threshold,
            use_parallel_children,
            deadline: None,
        }
    }

    /// Create configuration with a deadline for α-CROWN early termination.
    pub fn for_deadline(
        threshold: f32,
        use_parallel_children: bool,
        deadline: Option<Instant>,
    ) -> Self {
        Self {
            threshold,
            use_parallel_children,
            deadline,
        }
    }
}

impl Default for DomainProcessingConfig {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            use_parallel_children: true,
            deadline: None,
        }
    }
}
