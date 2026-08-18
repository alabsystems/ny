// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Builder for spec-guided CROWN backward propagation requests.
//!
//! Replaces the 24-function combinatorial wrapper surface with a single
//! `SpecCrownRequest` builder that collects optional parameters and executes
//! the core backward loop. Part of #4220 / #2622.

use crate::batched_domain::CachedLinearBounds;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::network::graph_alpha::resnet_decompose::DeadlineBoundedResnetRowsResult;
use crate::network::ResnetSegmentSkeleton;
use crate::types::CrownBackwardResult;
use crate::MulBinaryRelaxationMode;

use ny_core::{GemmEngine, Result, DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::Instant;

use crate::network::core::GraphNetwork;

/// Builder for spec-guided CROWN backward propagation requests.
///
/// Collects the 7 optional parameters (deadline, node_bounds, alpha_state,
/// mul_binary_alphas, mul_binary_relaxation, crown_backward_layers,
/// reference_node_bounds) and the 2 output selectors (capture_linear,
/// capture_cache) into a typed struct, eliminating the 24-function
/// combinatorial wrapper surface.
///
/// # Example
///
/// ```text
/// let result = SpecCrownRequest::new(&graph, &input, &spec, engine)
///     .node_bounds(&node_bounds)
///     .deadline_opt(Some(deadline))
///     .mul_binary_alphas_opt(alphas.as_ref())
///     .run_with_cache()?;
/// ```
#[must_use]
pub(crate) struct SpecCrownRequest<'a> {
    // Required (set at construction)
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    spec_matrix: &'a ndarray::Array2<f32>,
    engine: Option<&'a dyn GemmEngine>,

    // Optional (builder methods)
    mul_binary_relaxation: MulBinaryRelaxationMode,
    precomputed_node_bounds: Option<&'a HashMap<String, BoundedTensor>>,
    reference_node_bounds: Option<&'a HashMap<String, BoundedTensor>>,
    alpha_state: Option<&'a GraphAlphaState>,
    deadline: Option<Instant>,
    mul_binary_alphas: Option<&'a HashMap<String, ndarray::Array2<f32>>>,
    crown_backward_layers: Option<usize>,

    // Output selectors
    capture_linear_cache: bool,
    /// Caller consumes the returned input `LinearBounds` (`run_with_linear`).
    /// The core skips the bounds-only root-candidate fast paths (GPU resnet
    /// root pass / forward-linear C-margin) for these callers, because those
    /// routes return `None` linear and would silently defeat the extraction
    /// (#w5-bab-throughput).
    wants_input_linear: bool,

    /// Multi-neuron (k-ReLU) group facets injected into the ReLU backward arm
    /// (`docs/MULTI_NEURON_RELAXATION_DESIGN.md` §2.2, increment 3, default-OFF).
    /// When present and non-empty the core is FORCED onto the CPU backward loop
    /// (the only arm where the §2.2 pre/post injection lives — the GPU/
    /// forward-linear root fast paths cannot carry a coupling facet), and each
    /// group's `β_c·(a·x+g·y−b)` Lagrangian term is injected as the sweep reaches
    /// its ReLU node. Every facet is a proven superset half-space with `β_c ≥ 0`,
    /// so the injected margin lower bound stays sound (Invariant MN).
    mn_pool: Option<&'a crate::multineuron::MultiNeuronPool>,
}

impl<'a> SpecCrownRequest<'a> {
    /// Create a new spec-guided CROWN request with the required parameters.
    pub(crate) fn new(
        graph: &'a GraphNetwork,
        input: &'a BoundedTensor,
        spec_matrix: &'a ndarray::Array2<f32>,
        engine: Option<&'a dyn GemmEngine>,
    ) -> Self {
        Self {
            graph,
            input,
            spec_matrix,
            engine,
            mul_binary_relaxation: MulBinaryRelaxationMode::default(),
            precomputed_node_bounds: None,
            reference_node_bounds: None,
            alpha_state: None,
            deadline: None,
            mul_binary_alphas: None,
            crown_backward_layers: None,
            capture_linear_cache: false,
            wants_input_linear: false,
            mn_pool: None,
        }
    }

    /// Attach a multi-neuron group-facet pool for §2.2 backward injection
    /// (increment 3). Presence forces the CPU backward loop. Sound: every group
    /// is a proven-superset facet with `β_c ≥ 0`.
    pub(crate) fn mn_pool_opt(
        mut self,
        pool: Option<&'a crate::multineuron::MultiNeuronPool>,
    ) -> Self {
        self.mn_pool = pool;
        self
    }

    /// Set pre-computed intermediate node bounds for tighter ReLU relaxation.
    ///
    /// When provided, uses these bounds for ReLU pre-activation instead of
    /// recomputing via CROWN-IBP or IBP. Mutually exclusive with
    /// `reference_bounds` — the core function returns an error if both are set.
    pub(crate) fn node_bounds(mut self, bounds: &'a HashMap<String, BoundedTensor>) -> Self {
        self.precomputed_node_bounds = Some(bounds);
        self
    }

    /// Set optional pre-computed node bounds. Convenience for callers with
    /// `Option<&HashMap<...>>`.
    pub(crate) fn node_bounds_opt(
        mut self,
        bounds: Option<&'a HashMap<String, BoundedTensor>>,
    ) -> Self {
        self.precomputed_node_bounds = bounds;
        self
    }

    /// Set optional reference node bounds. Convenience for callers with
    /// `Option<&HashMap<...>>`.
    pub(crate) fn reference_bounds_opt(
        mut self,
        bounds: Option<&'a HashMap<String, BoundedTensor>>,
    ) -> Self {
        self.reference_node_bounds = bounds;
        self
    }

    /// Set optional alpha state. Convenience for callers with `Option<&GraphAlphaState>`.
    pub(crate) fn alpha_state_opt(mut self, state: Option<&'a GraphAlphaState>) -> Self {
        self.alpha_state = state;
        self
    }

    /// Set an optional deadline. Convenience for callers that already have
    /// `Option<Instant>` from their own parameter signatures.
    pub(crate) fn deadline_opt(mut self, deadline: Option<Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    /// Set optional MulBinary alphas. Convenience for callers that already have
    /// `Option<&HashMap<...>>`.
    pub(crate) fn mul_binary_alphas_opt(
        mut self,
        alphas: Option<&'a HashMap<String, ndarray::Array2<f32>>>,
    ) -> Self {
        self.mul_binary_alphas = alphas;
        self
    }

    /// Set the MulBinary relaxation mode (default: McCormick).
    pub(crate) fn mul_binary_relaxation(mut self, mode: MulBinaryRelaxationMode) -> Self {
        self.mul_binary_relaxation = mode;
        self
    }

    /// Truncate CROWN backward after this many layers.
    #[cfg(test)]
    pub(crate) fn truncate_after(mut self, layers: usize) -> Self {
        self.crown_backward_layers = Some(layers);
        self
    }

    /// Set optional truncation limit. Convenience for callers with `Option<usize>`.
    pub(crate) fn truncate_after_opt(mut self, layers: Option<usize>) -> Self {
        self.crown_backward_layers = layers;
        self
    }

    /// Enable linear cache capture in the output.
    pub(crate) fn capture_cache(mut self) -> Self {
        self.capture_linear_cache = true;
        self
    }

    /// Execute and return only the output bounds (discards provenance/linear).
    pub(crate) fn run(self) -> Result<BoundedTensor> {
        let (result, _linear, _cache) = self.execute()?;
        Ok(result.bounds)
    }

    /// Execute and return the full `CrownBackwardResult` with provenance.
    pub(crate) fn run_with_provenance(self) -> Result<CrownBackwardResult> {
        let (result, _linear, _cache) = self.execute()?;
        Ok(result)
    }

    /// Execute and return bounds + optional `LinearBounds`.
    ///
    /// Marks the request as an input-linear consumer so the core takes the CPU
    /// backward loop (the only route that produces the linear map) instead of
    /// the bounds-only root-candidate fast paths (#w5-bab-throughput).
    pub(crate) fn run_with_linear(mut self) -> Result<(BoundedTensor, Option<LinearBounds>)> {
        self.wants_input_linear = true;
        let (result, linear, _cache) = self.execute()?;
        Ok((result.bounds, linear))
    }

    /// Execute and return bounds + optional `CachedLinearBounds`.
    pub(crate) fn run_with_cache(self) -> Result<(BoundedTensor, Option<CachedLinearBounds>)> {
        let (result, _linear, cache) = self.execute()?;
        Ok((result.bounds, cache))
    }

    /// Execute the full coefficient backward and return its cache.
    ///
    /// Setting `wants_input_linear` deliberately bypasses the bounds-only root
    /// candidates in the spec core. This is used after root intermediate boxes
    /// change: a warmup alpha is still sound on the new boxes, but can be badly
    /// stale, while the root GPU/forward candidates do not reproduce the
    /// adaptive scalar DAG backward. Capturing that backward gives the caller a
    /// certified adaptive candidate and the matching child-domain cache.
    pub(crate) fn run_with_backward_cache(
        mut self,
    ) -> Result<(BoundedTensor, Option<CachedLinearBounds>)> {
        self.wants_input_linear = true;
        self.capture_linear_cache = true;
        let (result, _linear, cache) = self.execute()?;
        Ok((result.bounds, cache))
    }

    /// Attempt exactly one fresh-slope, certified GPU-resident ResNet backward.
    ///
    /// This is deliberately narrower than [`Self::run`]: it never enters the
    /// forward-linear candidates or the CPU fallback, and its return type cannot
    /// publish an input-linear map or [`CachedLinearBounds`]. `Ok(None)` is the
    /// fail-closed result for an unavailable/unsupported sound GPU route.
    ///
    /// Callers must provide already-certified node bounds and a private deadline.
    /// Supplying any state that this narrow route would otherwise have to ignore
    /// (alpha, cache capture, truncation, reference bounds, MulBinary alpha, or a
    /// multi-neuron pool) declines the request instead. Passing `None` for alpha
    /// to the ResNet extractor selects its fresh adaptive ReLU slopes.
    pub(crate) fn run_fresh_slope_sound_gpu_bounds_only(self) -> Result<Option<BoundedTensor>> {
        let Some(node_bounds) = self.precomputed_node_bounds else {
            return Ok(None);
        };
        let Some(deadline) = self.deadline else {
            return Ok(None);
        };
        if self.spec_matrix.nrows() != 1
            || !super::core::spec_root_gpu_enabled()
            || Instant::now() >= deadline
            || self.reference_node_bounds.is_some()
            || self.alpha_state.is_some()
            || self.mul_binary_alphas.is_some()
            || self.crown_backward_layers.is_some()
            || self.capture_linear_cache
            || self.wants_input_linear
            || self.mn_pool.is_some_and(|pool| !pool.is_empty())
            || self.mul_binary_relaxation != MulBinaryRelaxationMode::default()
        {
            return Ok(None);
        }

        let exec_order = self.graph.exec_order()?;
        let output_node_name = super::setup::resolve_output_contract(
            self.graph,
            exec_order,
            node_bounds,
            self.spec_matrix.ncols(),
        )?;
        let seed = LinearBounds::from_spec_matrix(self.spec_matrix.clone())?;
        crate::network::graph_alpha::resnet_decompose::
            try_resnet_gpu_suffix_single_row_with_deadline(
                self.graph,
                self.input,
                output_node_name,
                node_bounds,
                node_bounds,
                self.engine,
                deadline,
                &seed,
            )
    }

    /// Attempt one alpha-bearing, certified GPU-resident ResNet backward.
    ///
    /// This is the state-paired twin of
    /// [`Self::run_fresh_slope_sound_gpu_bounds_only`].  It accepts exactly one
    /// explicit alpha state and otherwise keeps the same narrow contract: one
    /// row, certified node bounds, a mandatory private deadline, the dedicated
    /// call-local GPU API, and no cache/linear/fallback publication.
    pub(crate) fn run_alpha_sound_gpu_bounds_only(self) -> Result<Option<BoundedTensor>> {
        let Some(node_bounds) = self.precomputed_node_bounds else {
            return Ok(None);
        };
        let Some(alpha_state) = self.alpha_state else {
            return Ok(None);
        };
        let Some(deadline) = self.deadline else {
            return Ok(None);
        };
        if self.spec_matrix.nrows() != 1
            || !super::core::spec_root_gpu_enabled()
            || Instant::now() >= deadline
            || self.reference_node_bounds.is_some()
            || self.mul_binary_alphas.is_some()
            || self.crown_backward_layers.is_some()
            || self.capture_linear_cache
            || self.wants_input_linear
            || self.mn_pool.is_some_and(|pool| !pool.is_empty())
            || self.mul_binary_relaxation != MulBinaryRelaxationMode::default()
        {
            return Ok(None);
        }

        let exec_order = self.graph.exec_order()?;
        let output_node_name = super::setup::resolve_output_contract(
            self.graph,
            exec_order,
            node_bounds,
            self.spec_matrix.ncols(),
        )?;
        let seed = LinearBounds::from_spec_matrix(self.spec_matrix.clone())?;
        crate::network::graph_alpha::resnet_decompose::
            try_resnet_gpu_suffix_single_row_with_alpha_and_deadline(
                self.graph,
                self.input,
                output_node_name,
                node_bounds,
                node_bounds,
                alpha_state,
                self.engine,
                deadline,
                &seed,
            )
    }

    /// Attempt one alpha-bearing, certified GPU-resident ResNet backward for a
    /// bounded batch of `2..=8` specification rows.
    ///
    /// This is an additive sibling of
    /// [`Self::run_alpha_sound_gpu_bounds_only`], not a replacement for it.
    /// The one-row critical route therefore keeps its exact legacy entry and
    /// dispatch, while active-set callers can certify a complete small row
    /// vector in one call-local backend transaction.
    ///
    /// `skeleton` is mandatory and strict: it must match this graph and the
    /// resolved output suffix, and its per-state fold must succeed. A mismatch
    /// refuses instead of falling back to a fresh extraction, so the returned
    /// segments are exactly those used by the bounded backend call and can be
    /// reused by a caller's host replay.
    #[allow(dead_code)] // Phase-1 plumbing: the follow-up root integration consumes this entry.
    pub(crate) fn run_alpha_sound_gpu_bounded_rows_only(
        self,
        skeleton: &ResnetSegmentSkeleton,
    ) -> Result<Option<DeadlineBoundedResnetRowsResult>> {
        let Some(node_bounds) = self.precomputed_node_bounds else {
            return Ok(None);
        };
        let Some(alpha_state) = self.alpha_state else {
            return Ok(None);
        };
        let Some(deadline) = self.deadline else {
            return Ok(None);
        };
        if !(2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&self.spec_matrix.nrows())
            || !super::core::spec_root_gpu_enabled()
            || Instant::now() >= deadline
            || self.reference_node_bounds.is_some()
            || self.mul_binary_alphas.is_some()
            || self.crown_backward_layers.is_some()
            || self.capture_linear_cache
            || self.wants_input_linear
            || self.mn_pool.is_some_and(|pool| !pool.is_empty())
            || self.mul_binary_relaxation != MulBinaryRelaxationMode::default()
        {
            return Ok(None);
        }

        let exec_order = self.graph.exec_order()?;
        let output_node_name = super::setup::resolve_output_contract(
            self.graph,
            exec_order,
            node_bounds,
            self.spec_matrix.ncols(),
        )?;
        let seed = LinearBounds::from_spec_matrix(self.spec_matrix.clone())?;
        crate::network::graph_alpha::resnet_decompose::
            try_resnet_gpu_suffix_bounded_rows_with_alpha_and_deadline(
                self.graph,
                self.input,
                output_node_name,
                node_bounds,
                node_bounds,
                alpha_state,
                skeleton,
                self.engine,
                deadline,
                &seed,
            )
    }

    /// Execute and return the full triple for callers that need all outputs.
    #[cfg(test)]
    pub(crate) fn run_all(
        self,
    ) -> Result<(
        CrownBackwardResult,
        Option<LinearBounds>,
        Option<CachedLinearBounds>,
    )> {
        self.execute()
    }

    /// Delegates to the core backward loop in core.rs.
    fn execute(
        self,
    ) -> Result<(
        CrownBackwardResult,
        Option<LinearBounds>,
        Option<CachedLinearBounds>,
    )> {
        // Disable the L2/Cauchy–Schwarz lever for the spec-guided CROWN scope.
        // When `precomputed_node_bounds` is not supplied this path collects its
        // own CROWN-IBP / IBP intermediate bounds (setup.rs), each a lever-firing
        // IBP forward pass. Nests harmlessly under an outer CROWN guard. Sound;
        // restored on drop. See `crate::l2_lever_gate`.
        let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
        super::core::propagate_crown_with_specs_and_engine_with_linear_and_reference_bounds_and_deadline_and_truncation(
            self.graph,
            self.input,
            self.spec_matrix,
            self.engine,
            self.mul_binary_relaxation,
            self.precomputed_node_bounds,
            self.reference_node_bounds,
            self.alpha_state,
            self.deadline,
            self.mul_binary_alphas,
            self.capture_linear_cache,
            self.crown_backward_layers,
            self.wants_input_linear,
            self.mn_pool,
        )
    }
}
