// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN delegation entry points for GraphNetwork.
//!
//! Moved from `core/graph/crown.rs` to break bidirectional dependency (#2380).
//! Only the delegation methods (that import from sibling `graph_crown`) are here.
//! Non-delegating methods (`propagate_crown_per_position`) remain in
//! `core/graph/crown.rs` since they have no sibling imports.

use crate::bounds::{AlphaCrownConfig, LinearBounds};
use crate::network::graph_crown::spec_propagation::SpecCrownRequest;
use crate::network::graph_crown::GraphNetworkCrownExt;
use crate::network::GraphNetwork;
use crate::types::CrownBackwardResult;
use crate::MulBinaryRelaxationMode;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, instrument};

fn graph_crown_error_must_propagate(error: &NyError) -> bool {
    matches!(
        error,
        NyError::SoundnessRefusal(_) | NyError::InternalError(_)
    )
}

impl GraphNetwork {
    /// Propagate bounds through the graph using the strongest public CROWN path.
    ///
    /// This entrypoint first attempts alpha-CROWN using the default optimizer
    /// configuration, then falls back to fixed-slope CROWN if alpha
    /// optimization is unsupported or fails. Downstream consumers call this
    /// method directly and expect the best available incomplete verifier, not
    /// only the legacy single-pass baseline.
    pub fn propagate_crown(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_crown_with_engine(input, None)
    }

    #[inline]
    #[instrument(skip(self, input, engine), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        match self.propagate_alpha_crown_with_config_and_engine(
            input,
            &AlphaCrownConfig::default(),
            engine,
        ) {
            Ok(bounds) => Ok(bounds),
            Err(error) => {
                if graph_crown_error_must_propagate(&error) {
                    return Err(error);
                }
                debug!(
                    "Graph public CROWN path: alpha optimization failed ({}); \
                     falling back to fixed-slope CROWN",
                    error
                );
                self.propagate_crown_fixed_slope_with_engine(input, engine)
            }
        }
    }

    /// Propagate bounds through the graph using the legacy fixed-slope CROWN path.
    ///
    /// This preserves the pre-#3619 behavior for regression tests, profiling,
    /// and callers that explicitly need the original single-pass baseline.
    pub fn propagate_crown_fixed_slope(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_crown_fixed_slope_with_engine(input, None)
    }

    #[inline]
    #[instrument(skip(self, input, engine), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_fixed_slope_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_fixed_slope_with_engine_and_relaxation(
            input,
            engine,
            MulBinaryRelaxationMode::default(),
        )
    }

    #[inline]
    #[instrument(skip(self, input, engine), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_fixed_slope_with_engine_and_relaxation(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<BoundedTensor> {
        GraphNetworkCrownExt::crown_backward_with_relaxation(
            self,
            input,
            engine,
            mul_binary_relaxation,
        )
    }

    /// CROWN backward propagation with deadline enforcement (#3398).
    ///
    /// When `deadline` is `Some`, checks at each node in the backward loop.
    /// If exceeded, falls back to IBP (always sound). This prevents graph CROWN
    /// backward from exceeding the verification timeout for large models.
    #[inline]
    #[instrument(skip(self, input, engine, deadline), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<CrownBackwardResult> {
        GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline(
            self,
            input,
            engine,
            MulBinaryRelaxationMode::default(),
            deadline,
        )
    }

    /// CROWN backward propagation with deadline enforcement and
    /// caller-precollected intermediate node bounds (#dedup-root-collections
    /// Fix B).
    ///
    /// Like [`Self::propagate_crown_with_engine_and_deadline`] but when
    /// `node_bounds` is `Some`, the backward pass reuses the provided
    /// same-input-box enclosure map instead of re-running its internal
    /// CROWN-IBP/IBP intermediate collection (and skips the pre-collection
    /// deadline gate — the bounds are already paid for; per-node deadline
    /// checks in the backward loop remain in force). `None` is byte-for-byte
    /// the legacy behavior.
    #[inline]
    #[instrument(skip(self, input, engine, deadline, node_bounds), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_with_engine_and_deadline_and_node_bounds(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
        node_bounds: Option<&std::collections::HashMap<String, BoundedTensor>>,
    ) -> Result<CrownBackwardResult> {
        GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
            self,
            input,
            engine,
            MulBinaryRelaxationMode::default(),
            deadline,
            None,
            node_bounds,
            None,
        )
    }

    /// Like [`Self::propagate_crown_with_engine_and_deadline_and_node_bounds`],
    /// with an optional caller-local wall cap for Graph-CROWN Step 1's
    /// CROWN-IBP tightening sweep.
    ///
    /// This is deliberately crate-local: DAG alpha uses it to reserve time for
    /// its optimizer after a forward-linear reference refusal. Other CROWN and
    /// BaB callers retain the historical uncapped policy.
    #[inline]
    #[instrument(skip(self, input, engine, deadline, node_bounds), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub(crate) fn propagate_crown_with_engine_and_deadline_and_node_bounds_and_crown_ibp_cap(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
        node_bounds: Option<&std::collections::HashMap<String, BoundedTensor>>,
        crown_ibp_tightening_cap: Option<std::time::Duration>,
    ) -> Result<CrownBackwardResult> {
        GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
            self,
            input,
            engine,
            MulBinaryRelaxationMode::default(),
            deadline,
            None,
            node_bounds,
            crown_ibp_tightening_cap,
        )
    }

    /// CROWN backward propagation with both relaxation mode and deadline (#3398).
    ///
    /// Combines caller-controlled relaxation with deadline enforcement while
    /// preserving fallback provenance.
    ///
    /// Used by `crown_fallback_chain` in the verifier to thread the verification
    /// timeout into the flat CROWN backward loop.
    #[inline]
    #[instrument(skip(self, input, engine, deadline), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_with_engine_relaxation_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<std::time::Instant>,
    ) -> Result<CrownBackwardResult> {
        GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline(
            self,
            input,
            engine,
            mul_binary_relaxation,
            deadline,
        )
    }

    /// CROWN backward propagation with explicit provenance metadata.
    ///
    /// Returns a [`CrownBackwardResult`] that includes both the computed bounds
    /// and provenance indicating whether CROWN or a forward-bound fallback was used.
    pub fn propagate_crown_with_provenance(
        &self,
        input: &BoundedTensor,
    ) -> Result<CrownBackwardResult> {
        self.propagate_crown_with_provenance_and_engine(input, None)
    }

    /// CROWN backward propagation with provenance and optional GPU GEMM engine (#3597).
    ///
    /// Like [`propagate_crown_with_provenance`] but threads the engine through
    /// to per-layer CROWN backward passes for GPU acceleration.
    pub fn propagate_crown_with_provenance_and_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<CrownBackwardResult> {
        GraphNetworkCrownExt::crown_backward_with_relaxation_and_provenance(
            self,
            input,
            engine,
            MulBinaryRelaxationMode::default(),
        )
    }

    /// Specification-guided CROWN backward propagation.
    ///
    /// Instead of computing bounds on each output independently, this method computes
    /// bounds on linear combinations of outputs defined by a specification matrix `C`.
    /// This preserves correlation information and produces much tighter bounds for
    /// verification properties like "output_0 > output_1".
    ///
    /// # Arguments
    /// * `input` - Input bounds
    /// * `spec_matrix` - Specification matrix of shape [num_specs, output_dim]
    ///   Each row defines a linear combination of outputs to bound.
    /// * `engine` - Optional GPU GEMM engine
    ///
    /// # Returns
    /// BoundedTensor with shape \[num_specs\], where bounds\[i\] are bounds on spec_matrix\[i\] @ outputs.
    ///
    /// # Example
    /// For property "class_0 > class_1", use spec_matrix = [[1, -1, 0, ...]]
    /// to get bounds on output_0 - output_1 directly.
    #[instrument(skip(self, input, spec_matrix, engine), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_engine(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        GraphNetworkCrownExt::crown_backward_specs_with_relaxation(
            self,
            input,
            spec_matrix,
            engine,
            MulBinaryRelaxationMode::default(),
        )
    }

    /// Specification-guided CROWN with pre-computed intermediate node bounds.
    ///
    /// Like `propagate_crown_with_specs_and_engine` but uses caller-provided
    /// intermediate bounds (e.g., from alpha-CROWN) instead of recomputing
    /// them via CROWN-IBP. This produces much tighter output bounds when the
    /// pre-computed bounds come from an optimized method.
    ///
    /// #1817/#1848: GPU BaB computes alpha-CROWN intermediate bounds in Step 1.
    /// Passing them here avoids re-computing with the inferior CROWN-IBP method.
    #[instrument(skip(self, input, spec_matrix, engine, node_bounds), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_engine_with_node_bounds(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<BoundedTensor> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .node_bounds(node_bounds)
            .run()
    }

    /// Spec-guided CROWN with pre-computed node bounds and deadline enforcement.
    /// Falls back to IBP if deadline is exceeded during backward propagation.
    /// #3218/#3328
    #[instrument(skip(self, input, spec_matrix, engine, node_bounds), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_engine_with_node_bounds_and_deadline(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        deadline: Option<std::time::Instant>,
    ) -> Result<BoundedTensor> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .node_bounds(node_bounds)
            .deadline_opt(deadline)
            .run()
    }

    /// Spec-guided CROWN with node bounds, deadline, and provenance tracking.
    ///
    /// Returns `CrownBackwardResult` carrying `BoundsProvenance` that records
    /// whether the result came from CROWN backward or an IBP fallback (and why).
    /// Used by `training_signal` Packet C (#3520) to surface fallback metadata
    /// in weak-region mining reports.
    #[instrument(skip(self, input, spec_matrix, engine, node_bounds), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_provenance_and_engine_with_node_bounds_and_deadline(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        deadline: Option<std::time::Instant>,
    ) -> Result<CrownBackwardResult> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .node_bounds(node_bounds)
            .deadline_opt(deadline)
            .run_with_provenance()
    }

    #[instrument(skip(self, input, spec_matrix, engine), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_engine_with_linear(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)> {
        SpecCrownRequest::new(self, input, spec_matrix, engine).run_with_linear()
    }

    /// Spec-guided CROWN with pre-computed node bounds, returning both output
    /// bounds and linear coefficients (for input split dimension selection).
    ///
    /// Combines `_with_node_bounds` (tighter ReLU relaxation via pre-computed
    /// intermediates) with `_with_linear` (linear coefficient extraction).
    /// Used by input splitting with alpha-CROWN initial bounds (#3357).
    #[instrument(skip(self, input, spec_matrix, engine, node_bounds), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_node_bounds_and_linear(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .node_bounds(node_bounds)
            .run_with_linear()
    }

    /// Like `propagate_crown_with_specs_and_node_bounds_and_linear` but with
    /// an optional deadline. When the deadline expires, the backward pass
    /// terminates early and falls back to IBP provenance for remaining nodes.
    ///
    /// Part of #3499: direct-boundary LinearBounds certificates need a deadline
    /// on the identity-spec CROWN backward pass to prevent timeout on large
    /// prefix graphs.
    #[instrument(skip(self, input, spec_matrix, engine, node_bounds, deadline), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        deadline: Option<std::time::Instant>,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .node_bounds(node_bounds)
            .deadline_opt(deadline)
            .run_with_linear()
    }

    #[instrument(skip(self, input, spec_matrix, engine, node_bounds, deadline), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_node_bounds_and_cache_and_deadline(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        deadline: Option<std::time::Instant>,
    ) -> Result<(
        BoundedTensor,
        Option<crate::batched_domain::CachedLinearBounds>,
    )> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .node_bounds(node_bounds)
            .deadline_opt(deadline)
            .capture_cache()
            .run_with_cache()
    }

    /// Spec-guided CROWN with both pre-computed ReLU node bounds AND MulBinary
    /// alpha optimization. Combines tighter ReLU relaxation (from alpha-CROWN
    /// warm-start intermediates) with per-element McCormick facet selection.
    ///
    /// Part of #3453: Per-domain alpha-CROWN in graph input-split BaB.
    #[instrument(skip(self, input, spec_matrix, engine, node_bounds, mul_binary_alphas), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_node_bounds_and_mul_binary_alphas(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        mul_binary_alphas: Option<&std::collections::HashMap<String, ndarray::Array2<f32>>>,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .node_bounds(node_bounds)
            .mul_binary_alphas_opt(mul_binary_alphas)
            .run_with_linear()
    }

    /// Spec-guided CROWN with pre-computed ReLU node bounds, optimized
    /// MulBinary alphas, and deadline enforcement.
    ///
    /// Part of #3814: graph input-split must share one absolute verifier
    /// deadline across the root MulBinary prepass and all spec-guided CROWN
    /// calls, instead of hardcoding `deadline: None` for the MulBinary path.
    #[instrument(skip(self, input, spec_matrix, engine, node_bounds, mul_binary_alphas, deadline), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_node_bounds_and_mul_binary_alphas_and_deadline(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        mul_binary_alphas: Option<&std::collections::HashMap<String, ndarray::Array2<f32>>>,
        deadline: Option<std::time::Instant>,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .node_bounds(node_bounds)
            .mul_binary_alphas_opt(mul_binary_alphas)
            .deadline_opt(deadline)
            .run_with_linear()
    }

    /// Spec-guided CROWN with per-element MulBinary alpha optimization.
    ///
    /// Used by input-split BaB to thread root-optimized MulBinary alphas into
    /// per-domain CROWN backward passes. Returns both output bounds and linear
    /// coefficients. Part of #3439 Phase 4.
    #[instrument(skip(self, input, spec_matrix, engine, mul_binary_alphas), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_mul_binary_alphas(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        mul_binary_alphas: Option<&std::collections::HashMap<String, ndarray::Array2<f32>>>,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .mul_binary_alphas_opt(mul_binary_alphas)
            .run_with_linear()
    }

    /// Spec-guided CROWN with optimized MulBinary alphas and deadline
    /// enforcement.
    ///
    /// Part of #3814: graph input-split root and per-domain MulBinary CROWN
    /// calls must respect the verifier's absolute timeout budget.
    #[instrument(skip(self, input, spec_matrix, engine, mul_binary_alphas, deadline), fields(num_nodes = self.nodes.len(), num_specs = spec_matrix.nrows()))]
    pub fn propagate_crown_with_specs_and_mul_binary_alphas_and_deadline(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        mul_binary_alphas: Option<&std::collections::HashMap<String, ndarray::Array2<f32>>>,
        deadline: Option<std::time::Instant>,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .mul_binary_alphas_opt(mul_binary_alphas)
            .deadline_opt(deadline)
            .run_with_linear()
    }
}
