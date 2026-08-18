// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared graph domain-batch execution surface.
//!
//! Packet A for #4398 extracts one internal API that current graph/input-split
//! batch callers can route through without changing behavior. The underlying
//! single-objective, multi-objective, and dense-spec implementations stay in
//! their existing modules for now; later packets can move more mechanics behind
//! this seam without rewriting three separate call sites again.

mod metrics;
mod plan;

use std::collections::HashMap;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::domain::{GraphBabDomain, MultiObjectiveGraphBabDomain};
use crate::beta_crown::engine::domain_results::{
    GraphDomainResult, MultiObjectiveGraphDomainResult,
};
use crate::beta_crown::engine::graph::input_split::shared_specs::{
    compute_crown_or_ibp_bounds_batched_specs, BatchedSpecBounds,
};
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::GraphAlphaState;
use crate::GraphNetwork;

pub use self::metrics::{
    GraphDomainBatchCallerLane, GraphDomainBatchMetricsSink, GraphDomainBatchRecord,
};
pub(in crate::beta_crown::engine::graph) use self::plan::{
    GraphDomainBatchEmitTiming, GraphDomainBatchExecutionMode, GraphDomainBatchPlan,
    ReluSplitBatchContext,
};

/// Request for single-objective graph BaB batch execution.
pub(in crate::beta_crown::engine::graph) struct SingleObjectiveBatchRequest<'a> {
    pub(in crate::beta_crown::engine::graph) graph: &'a GraphNetwork,
    pub(in crate::beta_crown::engine::graph) domains: &'a [&'a GraphBabDomain],
    pub(in crate::beta_crown::engine::graph) relu_nodes: &'a [String],
    pub(in crate::beta_crown::engine::graph) objective: &'a [f32],
    pub(in crate::beta_crown::engine::graph) threshold: f32,
    pub(in crate::beta_crown::engine::graph) engine: &'a dyn GemmEngine,
    pub(in crate::beta_crown::engine::graph) cut_pool: Option<&'a GraphCutPool>,
    /// Requested adaptive ReLU split depth. The executor caps this separately
    /// for every parent by its remaining configured depth budget.
    pub(in crate::beta_crown::engine::graph) split_depth: usize,
    /// Surface retryable allocation/dispatch refusals to an adaptive caller.
    /// False preserves the historical internal sequential fallback.
    pub(in crate::beta_crown::engine::graph) retry_refusals: bool,
}

/// Request for multi-objective graph BaB batch execution.
pub(in crate::beta_crown::engine::graph) struct MultiObjectiveBatchRequest<'a> {
    /// Canonical zero-based outer BaB wave index.
    pub(in crate::beta_crown::engine::graph) bab_round: usize,
    pub(in crate::beta_crown::engine::graph) graph: &'a GraphNetwork,
    pub(in crate::beta_crown::engine::graph) domains: &'a [&'a MultiObjectiveGraphBabDomain],
    pub(in crate::beta_crown::engine::graph) relu_nodes: &'a [String],
    pub(in crate::beta_crown::engine::graph) objectives: &'a [Vec<f32>],
    pub(in crate::beta_crown::engine::graph) thresholds: &'a [f32],
    pub(in crate::beta_crown::engine::graph) engine: &'a dyn GemmEngine,
    pub(in crate::beta_crown::engine::graph) cut_pool: Option<&'a GraphCutPool>,
    /// Default-dark, first-wave-only expanded warmup W. W contributes an
    /// independently certified lower-bound certificate plus separate
    /// cache-invalidated continuation state; H's certified upper endpoint
    /// remains authoritative.
    pub(in crate::beta_crown::engine::graph) selective_root_alpha_candidate:
        Option<&'a crate::beta_crown::state::GraphDomainAlphaState>,
}

/// Request for dense-spec batch rebound over input-split domains.
pub(in crate::beta_crown::engine::graph) struct DenseSpecBatchRequest<'a> {
    pub(in crate::beta_crown::engine::graph) graph: &'a GraphNetwork,
    pub(in crate::beta_crown::engine::graph) input_bounds_batch: &'a [&'a BoundedTensor],
    pub(in crate::beta_crown::engine::graph) spec_matrix: &'a Array2<f32>,
    pub(in crate::beta_crown::engine::graph) engine: Option<&'a dyn GemmEngine>,
    pub(in crate::beta_crown::engine::graph) alpha_node_bounds:
        Option<&'a HashMap<String, BoundedTensor>>,
    pub(in crate::beta_crown::engine::graph) alpha_state: Option<&'a GraphAlphaState>,
    pub(in crate::beta_crown::engine::graph) mul_binary_alphas:
        Option<&'a HashMap<String, Array2<f32>>>,
    pub(in crate::beta_crown::engine::graph) deadline: Option<Instant>,
    pub(in crate::beta_crown::engine::graph) crown_backward_layers: Option<usize>,
    pub(in crate::beta_crown::engine::graph) ibp_enhancement: bool,
    /// #cgan-batched-stack: domain-stacked conv/BN backward + per-domain IBP
    /// refresh in the batched dense-spec kernel (preset-gated, default false).
    pub(in crate::beta_crown::engine::graph) stacked_rebound: bool,
}

/// Shared entry points for graph domain-batch execution.
pub(in crate::beta_crown::engine::graph) struct GraphDomainBatchExecutor;

impl GraphDomainBatchExecutor {
    pub(in crate::beta_crown::engine::graph) fn execute_single_objective(
        verifier: &BetaCrownVerifier,
        request: SingleObjectiveBatchRequest<'_>,
    ) -> std::result::Result<
        Vec<GraphDomainResult>,
        super::adaptive_microbatch::MicrobatchRefusalReason,
    > {
        verifier.process_graph_domains_batched_gpu(
            request.graph,
            request.domains,
            request.relu_nodes,
            request.objective,
            request.threshold,
            request.engine,
            request.cut_pool,
            request.split_depth,
            request.retry_refusals,
        )
    }

    pub(in crate::beta_crown::engine::graph) fn execute_multi_objective(
        verifier: &BetaCrownVerifier,
        request: MultiObjectiveBatchRequest<'_>,
    ) -> Vec<MultiObjectiveGraphDomainResult> {
        verifier.process_graph_domains_batched_gpu_multi_objective(
            request.bab_round,
            request.graph,
            request.domains,
            request.relu_nodes,
            request.objectives,
            request.thresholds,
            request.engine,
            request.cut_pool,
            request.selective_root_alpha_candidate,
        )
    }

    pub(in crate::beta_crown::engine::graph) fn execute_dense_specs(
        request: DenseSpecBatchRequest<'_>,
    ) -> Result<BatchedSpecBounds> {
        compute_crown_or_ibp_bounds_batched_specs(
            request.graph,
            request.input_bounds_batch,
            request.spec_matrix,
            request.engine,
            request.alpha_node_bounds,
            request.alpha_state,
            request.mul_binary_alphas,
            request.deadline,
            request.crown_backward_layers,
            request.ibp_enhancement,
            request.stacked_rebound,
        )
    }
}
