// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched backward context and result types for GPU-friendly CROWN propagation.

use std::sync::Arc;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::batched_domain::{BatchedDomains, CachedLinearBounds};
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::graph::{DomainCrownResult, DomainSpecCrownResult};
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::LinearBounds;

/// Batched context for GPU-friendly backward pass.
///
/// This struct provides direct access to pre-batched domain data, avoiding
/// tuple conversions that prevent efficient GPU transfer.
///
/// # Reference
/// alpha-beta-CROWN: `complete_verifier/branching_domains.py:270-356` (pick_out pattern)
#[derive(Debug)]
pub struct BatchedBackwardContext<'a> {
    /// Reference to the batched domains containing pre-packed tensors.
    pub batched: &'a BatchedDomains,
    /// Per-domain split histories (needed for constraint application).
    pub histories: Vec<&'a GraphSplitHistory>,
    /// Per-domain β states for Lagrangian optimization.
    pub beta_states: Vec<Option<&'a GraphBetaState>>,
    /// Per-domain base bounds for constraint transfer.
    pub base_bounds: Vec<Option<&'a std::collections::HashMap<String, Arc<BoundedTensor>>>>,
    /// #cone-delta: per-domain delta pre-nodes, parallel to `base_bounds`
    /// (`GraphBabDomain::delta_pre_nodes` — the pre-activation nodes of the
    /// constraints added since that domain's `node_bounds` was last
    /// fixpointed). `None` = delta unknown ⇒ the constrained forward keeps its
    /// full-history seeds (fail-closed). Consulted only behind
    /// `NY_CONE_REFRESH=1`.
    pub delta_seeds: Vec<Option<&'a [String]>>,
    /// Per-domain α states for ReLU lower-bound slope optimization.
    ///
    /// When present, the ReLU backward pass uses optimized per-neuron alpha values
    /// instead of the fixed heuristic. Enables joint α-β optimization.
    ///
    /// # Reference
    /// alpha-beta-CROWN: `auto_LiRPA/operators/relu.py` (optimizable slopes)
    /// Issue: #1841
    pub alpha_states: Vec<Option<&'a GraphDomainAlphaState>>,
    /// Per-domain cached lA coefficients from parent backward pass.
    ///
    /// When present for a domain, the backward pass can use these as initialization
    /// at intermediate nodes instead of recomputing from the output node. This enables
    /// the lA reuse optimization from alpha-beta-CROWN.
    ///
    /// # Reference
    /// alpha-beta-CROWN: `complete_verifier/tensor_storage.py` (all_lAs)
    /// Issue: #1564, #1669
    pub cached_la: Vec<Option<&'a CachedLinearBounds>>,
    /// Shared MulBinary alpha parameters for McCormick interpolation.
    ///
    /// Optimized once at the root domain via SPSA and frozen for all sub-domains.
    /// Maps node name → `[2, n]` array where row 0 = r_l (lower facet),
    /// row 1 = r_u (upper facet). When present, the backward dispatch uses
    /// `propagate_linear_binary_with_alpha` for tighter MulBinary bounds.
    ///
    /// # Reference
    /// - `input_split/mul_binary.rs` (SPSA optimizer)
    /// - Issue: #4284, #3439 Phase 4
    pub mul_binary_alphas: Option<&'a std::collections::HashMap<String, ndarray::Array2<f32>>>,
}

impl<'a> BatchedBackwardContext<'a> {
    /// Create a batched context from domains and pre-built BatchedDomains.
    ///
    /// This avoids re-extracting input bounds from BatchedDomains since
    /// the caller already has domain references.
    ///
    /// # Errors
    /// Returns `NyError::InvalidSpec` if `domains.len() != batched.batch_size()`.
    pub fn from_domains(
        domains: &'a [&'a GraphBabDomain],
        batched: &'a BatchedDomains,
    ) -> Result<Self> {
        if domains.len() != batched.batch_size() {
            return Err(NyError::InvalidSpec(format!(
                "BatchedBackwardContext size mismatch: domains={}, batch_size={}",
                domains.len(),
                batched.batch_size()
            )));
        }

        let histories: Vec<_> = domains.iter().map(|d| &d.history).collect();
        let beta_states: Vec<_> = domains.iter().map(|d| Some(&d.beta_state)).collect();
        let alpha_states: Vec<_> = domains.iter().map(|d| Some(&d.alpha_state)).collect();
        let base_bounds: Vec<_> = domains.iter().map(|d| Some(&d.node_bounds)).collect();
        // #cone-delta: each domain's delta is tracked against its OWN
        // `node_bounds` — exactly the map `base_bounds` carries above.
        let delta_seeds: Vec<_> = domains
            .iter()
            .map(|d| Some(d.delta_pre_nodes.as_slice()))
            .collect();
        let cached_la: Vec<_> = domains.iter().map(|d| d.cached_la.as_deref()).collect();

        Ok(Self {
            batched,
            histories,
            beta_states,
            alpha_states,
            base_bounds,
            delta_seeds,
            cached_la,
            mul_binary_alphas: None,
        })
    }

    /// Number of domains in this context.
    #[must_use]
    pub fn len(&self) -> usize {
        self.batched.batch_size()
    }

    /// Check if context is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.batched.is_empty()
    }
}

/// Result from batched backward pass with optional intermediate linear bounds.
///
/// This struct extends the standard backward pass result to optionally include
/// the intermediate `LinearBounds` at each node for each domain. These can be
/// cached in child domains to avoid recomputation in subsequent backward passes.
///
/// # Reference
/// alpha-beta-CROWN: `complete_verifier/tensor_storage.py` (all_lAs storage)
/// Issue: #1564 (lA matrix caching)
#[derive(Debug)]
pub struct BatchedBackwardResult {
    /// Output bounds per domain: (output_bounds, node_bounds_cache).
    pub results: Vec<DomainCrownResult>,
    /// Intermediate linear bounds per domain per node.
    ///
    /// `intermediate_la[domain_idx]` is a map from node name to `LinearBounds`
    /// representing the accumulated lA coefficients at that node after backward
    /// propagation from the output.
    ///
    /// Only populated when `capture_intermediate = true` is passed to the
    /// backward pass function.
    pub intermediate_la: Option<Vec<std::collections::HashMap<String, LinearBounds>>>,
    /// Optional stage timing for forward/backward observability.
    /// Populated by `batched_forward_then_backward`. Part of #4398 Packet B.
    pub stage_timing: Option<BatchedStageTiming>,
}

/// Fine-grained stage timing for standard (scalar-objective) batched CROWN.
///
/// Captures forward (constraint propagation) and backward (CROWN relaxation)
/// wall-clock time so callers can quantify the per-domain forward overhead vs
/// the batched backward win.  Part of #4398 Packet B observability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchedStageTiming {
    pub forward_elapsed_s: f64,
    pub backward_elapsed_s: f64,
}

/// Fine-grained stage timing for dense-spec batched rebound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseSpecStageTiming {
    pub forward_elapsed_s: f64,
    pub backward_elapsed_s: f64,
    pub materialize_elapsed_s: f64,
}

/// Result from dense-spec batched backward pass with per-domain input linear bounds.
///
/// Sibling of `BatchedBackwardResult` for the dense-spec (multi-row spec matrix) path.
/// Preserves per-domain input `LinearBounds` from the CROWN backward pass, which the
/// scalar path discards.
///
/// Part of #4116 Packet A: dense-spec result surface.
#[derive(Debug)]
pub struct BatchedSpecBackwardResult {
    /// Dense-spec output bounds per domain with input linear data.
    pub results: Vec<DomainSpecCrownResult>,
    /// Intermediate linear bounds per domain per node (same semantics as
    /// `BatchedBackwardResult::intermediate_la`).
    pub intermediate_la: Option<Vec<std::collections::HashMap<String, LinearBounds>>>,
    /// Optional stage timing for dense-spec rebound observability.
    pub stage_timing: Option<DenseSpecStageTiming>,
    /// Per-domain β states optimized by the GPU per-domain β loop
    /// (#w4-split-tightening). `None` (or a `None` slot) means the domain kept
    /// its inherited β (single-shot lane / opt not requested / not eligible).
    /// Only the GPU resnet fast-path fills this; callers use it to warm-start
    /// child inheritance — the BOUNDS are already β-optimized either way.
    pub optimized_betas: Option<Vec<Option<GraphBetaState>>>,
    /// #hard-six per-domain UNSHARED α (dark, `NY_WIDE_ALPHA_UNSHARED=1`):
    /// per-domain best-margin α snapshots from the wide α ascent. `None` (or a
    /// `None` slot) means the domain keeps its inherited α — the historical
    /// behavior, byte-identical when the gate is off. Callers use it to
    /// warm-start child α inheritance so the per-neuron ascent ACCUMULATES
    /// along the branch instead of restarting from root α every batch.
    pub optimized_alphas: Option<Vec<Option<GraphDomainAlphaState>>>,
    /// #interm-refine prune lane (dark, `NY_INTERM_REFINE_PRUNE=1`):
    /// `Some(flags)` marks domains whose split-premise constraint set the
    /// refinement pass PROVED empty (a sound refined enclosure strictly
    /// contradicts one of the domain's own split premises) — the subdomain is
    /// infeasible and verifies vacuously. `None` = lane off / no prune events.
    /// The per-domain BOUNDS for flagged domains are still valid (vacuously);
    /// callers that ignore this field stay sound, just without the prune.
    pub infeasible_domains: Option<Vec<bool>>,
}
