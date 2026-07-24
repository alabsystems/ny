// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Public CROWN entry-point wrappers for sequential networks.
//!
//! Extracted from `crown.rs` as part of #4233 Packet B. These are thin
//! delegation wrappers around `propagate_crown_core` and
//! `propagate_crown_with_layer_bounds_*` which remain in `crown.rs`.

use crate::layers::Layer;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;

use super::super::Network;

impl Network {
    /// Check whether this network's layer types are all GPU CROWN fast-path eligible.
    ///
    /// Returns `true` if every layer is in the set supported by the GPU backward
    /// shader (Linear, Conv1d, Conv2d, ReLU, Sigmoid, Tanh, Exp, Log,
    /// AddConstant, SubConstant, MulConstant, DivConstant, Flatten, Reshape).
    ///
    /// This is a structural check only — it does not require pre-activation bounds
    /// or a GPU device. Use it in tests to assert that a model would take the GPU
    /// fast-path when an engine is provided.
    pub fn is_gpu_crown_eligible(&self) -> bool {
        self.layers.iter().all(|layer| {
            matches!(
                layer,
                Layer::Linear(_)
                    | Layer::Conv1d(_)
                    | Layer::Conv2d(_)
                    | Layer::ReLU(_)
                    | Layer::Sigmoid(_)
                    | Layer::Tanh(_)
                    | Layer::Exp(_)
                    | Layer::Log(_)
                    | Layer::AddConstant(_)
                    | Layer::SubConstant(_)
                    | Layer::MulConstant(_)
                    | Layer::DivConstant(_)
                    | Layer::Flatten(_)
                    | Layer::Reshape(_)
            )
        })
    }

    /// Propagate bounds through the entire network using CROWN.
    ///
    /// CROWN (Convex Relaxation based perturbation ON-the-fly Network) provides
    /// tighter bounds than IBP by representing bounds as linear functions of the input.
    /// This implementation matches Auto-LiRPA's "backward" method.
    ///
    /// # REQUIRES
    /// - `input` shape must match network's expected input dimension
    /// - `input.lower()[i] <= input.upper()[i]` for all elements (well-formed bounds)
    /// - Network must have at least one layer
    ///
    /// # ENSURES
    /// - Output bounds contain all possible network outputs for inputs in `input`
    /// - Bounds are at least as tight as `propagate_ibp()` (CROWN relaxation is optimal)
    /// - Soundness: for any `x` where `input.contains(x)`, `output.contains(network(x))`
    ///
    /// Algorithm:
    /// 1. Run CROWN-IBP to collect tighter pre-activation bounds
    ///    - For Linear layer outputs: CROWN backward from that layer to input
    ///    - For ReLU layer outputs: IBP from the tighter linear bounds
    /// 2. Initialize linear bounds at output: A = I, b = 0 (output = output)
    /// 3. Propagate backward through each layer using tighter intermediate bounds
    /// 4. Concretize final linear bounds using input bounds
    /// 5. Intersect with forward bounds to ensure output is at least as tight as forward pass (#2990)
    pub fn propagate_crown(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_crown_with_engine(input, None)
    }

    #[inline]
    pub fn propagate_crown_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_with_engine_and_deadline(input, engine, None)
    }

    /// CROWN backward propagation with optional deadline enforcement (#3328).
    ///
    /// When `deadline` is `Some`, checks at the start of each backward layer
    /// iteration. If exceeded, falls back to IBP (always sound and cheap).
    /// O(layer_count) granularity — each check costs ~ns vs ~5-10s per layer.
    ///
    /// GPU fast-path (#3397): when the engine supports [`GpuCrownBackward`] and all
    /// layers are GPU-supported (Linear + ReLU), the entire backward loop runs on GPU
    /// with A-matrices kept on device. Only the final concretized bounds are read back.
    pub fn propagate_crown_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_with_engine_and_deadline_and_limits(input, engine, deadline, None)
    }

    pub(crate) fn propagate_crown_with_engine_and_deadline_and_limits(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_core(input, None, engine, deadline, crown_backward_layers)
    }

    /// Propagate CROWN bounds using pre-computed per-layer IBP bounds (#3397).
    ///
    /// Skips the internal IBP forward pass inside CROWN-IBP collection, saving
    /// ~59s for soundnessbench-scale models. The `precomputed_ibp` must have
    /// exactly one entry per layer (from `collect_ibp_bounds`).
    ///
    /// This is the production-path optimization: any caller that already has
    /// per-layer IBP bounds (from a prior IBP stage, BaB iteration, etc.) can
    /// pass them here to avoid redundant computation.
    pub fn propagate_crown_with_precomputed_ibp(
        &self,
        input: &BoundedTensor,
        precomputed_ibp: Vec<BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_with_precomputed_ibp_and_limits(
            input,
            precomputed_ibp,
            engine,
            deadline,
            None,
        )
    }

    pub(crate) fn propagate_crown_with_precomputed_ibp_and_limits(
        &self,
        input: &BoundedTensor,
        precomputed_ibp: Vec<BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_core(
            input,
            Some(precomputed_ibp),
            engine,
            deadline,
            crown_backward_layers,
        )
    }

    /// CROWN-IBP: Run CROWN with IBP as the inner bound (#3043 dedup).
    ///
    /// Deduplicated in #3043 / #2535: the previous implementation was a copy of
    /// `propagate_crown_with_engine` differing only in the engine parameter
    /// and tracing labels. This duplication caused #2990 and #3037.
    #[inline]
    pub fn propagate_crown_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_crown_with_engine(input, None)
    }
}
