// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP delegation entry points for GraphNetwork.
//!
//! Moved from `core/graph/ibp/base.rs` to break bidirectional dependency (#2380).
//! Only the delegation methods (that import from sibling `graph_ibp`) are here.
//! Non-delegating methods (`collect_activation_statistics`, `propagate_ibp_with_clipper`)
//! remain in `core/graph/ibp/base.rs` since they have no sibling imports.

use crate::network::graph_ibp::GraphNetworkIbpExt;
use crate::network::GraphNetwork;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::instrument;

impl GraphNetwork {
    /// Propagate bounds through the graph using IBP.
    ///
    /// Executes nodes in topological order, storing intermediate bounds
    /// and using them for downstream operations.
    #[inline]
    #[instrument(skip(self, input), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        GraphNetworkIbpExt::propagate_ibp_impl(self, input)
    }

    /// Propagate bounds through the graph using IBP with an optional GEMM engine.
    ///
    /// Linear nodes use the supplied engine when present; all other node types
    /// keep their existing CPU IBP implementation.
    #[inline]
    pub fn propagate_ibp_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        GraphNetworkIbpExt::propagate_ibp_with_engine_impl(self, input, engine)
    }

    /// True concrete (point) forward through the graph at `input`.
    ///
    /// Returns a degenerate (lower == upper) tensor whose value is a faithful f32
    /// evaluation of the network — matching ONNX Runtime to ~1e-6 even on deep
    /// generators. Unlike `propagate_ibp_with_engine` (which propagates a BOX and,
    /// for a point input, returns a non-trivially-wide box because per-node
    /// soundness widening — esp. BatchNorm — is amplified by the deep DAG), this
    /// collapses each node output to its interval center. NON-soundness-critical:
    /// for sat-finding / witness evaluation, never to decide a Verified verdict.
    #[inline]
    pub fn propagate_concrete_point(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        GraphNetworkIbpExt::propagate_concrete_point_impl(self, input, engine, deadline)
    }

    /// Concrete point forward (see [`GraphNetwork::propagate_concrete_point`])
    /// preserving a prepended leading restart axis, for batched graph PGD.
    #[inline]
    pub fn propagate_concrete_point_preserve_leading_axis(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        GraphNetworkIbpExt::propagate_concrete_point_preserve_leading_axis_impl(self, input, engine)
    }

    /// Propagate bounds through the graph using IBP with an optional GEMM engine
    /// and a wall-clock deadline.
    ///
    /// Aborts with `DeadlineExceeded` between nodes once `deadline` passes, so a
    /// single forward pass over a deep conv DAG cannot overrun the verifier's own
    /// timeout (#4321). Passing `None` is equivalent to `propagate_ibp_with_engine`.
    #[inline]
    pub fn propagate_ibp_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        GraphNetworkIbpExt::propagate_ibp_with_engine_and_deadline_impl(
            self, input, engine, deadline,
        )
    }

    /// Propagate bounds through the graph using IBP while preserving a leading axis.
    ///
    /// This is a narrow execution mode for batched graph PGD, where the runtime
    /// tensor prepends a restart axis that should survive Reshape/Flatten nodes
    /// even though the stored graph metadata remains in unbatched convention.
    #[inline]
    pub fn propagate_ibp_with_engine_preserve_leading_axis(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        GraphNetworkIbpExt::propagate_ibp_with_engine_preserve_leading_axis_impl(
            self, input, engine,
        )
    }

    /// Propagate bounds through the graph using IBP with a certified per-node
    /// widening (lower bounds shifted toward -infinity, upper bounds toward
    /// +infinity), so accumulated floating-point rounding cannot make bounds
    /// unsound.
    ///
    /// Same bounds as `propagate_ibp` plus each node's certified error term:
    /// `in_features + 2` ULPs for Linear, the Higham window-sum term for the
    /// conv family, and 1 ULP for the remaining node types (see
    /// `GraphNetworkIbpExt::propagate_ibp_sound_impl` for the per-node rule).
    #[inline]
    #[instrument(skip(self, input), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_ibp_sound(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        GraphNetworkIbpExt::propagate_ibp_sound_impl(self, input)
    }

    /// Propagate SOUND (directed-rounding) bounds with an optional GEMM engine.
    ///
    /// Same certified per-node widening as [`GraphNetwork::propagate_ibp_sound`],
    /// but threads `engine` so a DAG-lowerable graph can take the GPU-resident SOUND
    /// DAG plan when the process soundness gate is engaged and the engine advertises
    /// one (`provides_sound_gpu_dag_ibp`). Every other case falls through to the
    /// proven-sound CPU graph loop, so the result is always a valid enclosure.
    #[inline]
    pub fn propagate_ibp_sound_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        GraphNetworkIbpExt::propagate_ibp_sound_with_engine_impl(self, input, engine)
    }
}
