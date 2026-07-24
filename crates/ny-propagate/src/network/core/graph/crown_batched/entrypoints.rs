// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Public batched CROWN entrypoints for `GraphNetwork`.

use std::collections::HashMap;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, instrument};

use crate::types::{BoundsProvenance, CrownBackwardResult, CrownIbpFallbackReason};
use crate::MulBinaryRelaxationMode;

use super::binary_ops::AttentionCompositionRuntime;
use super::GraphNetwork;

impl GraphNetwork {
    /// Propagate bounds through the graph using N-D batched CROWN.
    ///
    /// This preserves tensor shape structure throughout propagation instead of
    /// flattening to 1D. Essential for transformer models where operations
    /// like attention have cross-position interactions.
    ///
    /// Supported layers:
    /// - Linear, ReLU, GELU, SiLU, Softmax, LayerNorm (full batched support)
    /// - All elementwise activations: Tanh, Sigmoid, Exp, Log, Sqrt, Reciprocal,
    ///   Softplus, HardSwish, Mish, Selu, Elu, Celu, Softsign, Arctan, Sin, Cos, Tan,
    ///   LeakyReLU, HardSigmoid, Clip, ThresholdedRelu, Abs, PowConstant,
    ///   Floor, Ceil, Round, Sign
    /// - Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d
    /// - Add, MatMul, MulBinary, BilinearCrown (binary)
    /// - Transpose, MulConstant, DivConstant, AddConstant, SubConstant
    /// - Flatten, Reshape, Squeeze, Unsqueeze (passthrough)
    #[inline]
    #[instrument(skip(self, input), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_crown_batched_with_provenance(input)
            .map(|result| result.bounds)
    }

    /// Propagate bounds through the graph using N-D batched CROWN with fallback provenance.
    ///
    /// When the batched path hits the shared CPU dense-materialization budget
    /// (`NyError::CpuMemoryExceeded`), reuse the existing unbatched DAG-CROWN
    /// provenance path instead of surfacing a raw error (#3550).
    #[instrument(skip(self, input), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched_with_provenance(
        &self,
        input: &BoundedTensor,
    ) -> Result<CrownBackwardResult> {
        self.propagate_crown_batched_with_provenance_and_engine(input, None)
    }

    /// Batched CROWN with fallback provenance and optional GPU GEMM engine (#3597).
    ///
    /// Threads `engine` through both the batched inner path and the DAG-CROWN
    /// fallback, ensuring GPU acceleration is available in both code paths.
    #[instrument(skip(self, input, engine), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched_with_provenance_and_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<CrownBackwardResult> {
        match self.propagate_crown_batched_inner(
            input,
            MulBinaryRelaxationMode::default(),
            None,
            None,
            engine,
            AttentionCompositionRuntime::production(),
        ) {
            Ok(result) => Ok(result),
            Err(err @ NyError::CpuMemoryExceeded { .. }) => {
                debug!(
                    "GraphNetwork batched CROWN: {}. Falling back to DAG-CROWN provenance path",
                    err
                );
                self.propagate_crown_with_provenance_and_engine(input, engine)
                    .map(|fallback| CrownBackwardResult {
                        bounds: fallback.bounds,
                        provenance: BoundsProvenance::ForwardFallback(
                            CrownIbpFallbackReason::MemoryBudgetExceeded,
                        ),
                    })
            }
            Err(err) => Err(err),
        }
    }

    /// Propagate bounds through the graph using N-D batched CROWN with bilinear alpha optimization.
    ///
    /// This variant passes direction-dependent McCormick alphas [4, m, n, k] to BilinearCrown
    /// nodes during backward propagation, enabling alpha-CROWN optimization of attention bounds.
    ///
    /// # Arguments
    /// * `bilinear_alphas` - Map from BilinearCrown node name to alpha [4, m, n, k].
    ///   Nodes not in the map use the fixed midpoint heuristic (no alpha optimization).
    ///
    /// Reference: auto_LiRPA bivariate.py:128-135 `_init_opt_parameters_impl`
    #[instrument(skip(self, input, bilinear_alphas), fields(num_nodes = self.nodes.len()))]
    pub fn propagate_crown_batched_with_bilinear_alphas(
        &self,
        input: &BoundedTensor,
        bilinear_alphas: &HashMap<String, ndarray::Array4<f32>>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_batched_with_bilinear_alphas_and_engine(input, bilinear_alphas, None)
    }

    /// Engine-aware variant of `propagate_crown_batched_with_bilinear_alphas` (#3588, #3772).
    ///
    /// Threads `engine` through the batched CROWN backward pass so that the
    /// pure-attention alpha-CROWN optimizer (no-ReLU DAG path) can benefit from
    /// GPU GEMM acceleration when the engine supports batched backward dispatch.
    pub fn propagate_crown_batched_with_bilinear_alphas_and_engine(
        &self,
        input: &BoundedTensor,
        bilinear_alphas: &HashMap<String, ndarray::Array4<f32>>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_batched_inner(
            input,
            MulBinaryRelaxationMode::default(),
            Some(bilinear_alphas),
            None,
            engine,
            AttentionCompositionRuntime::production(),
        )
        .map(|result| result.bounds)
    }

    /// Engine-aware variant of `propagate_crown_batched` (#3588).
    ///
    /// Threads `engine` into the batched CROWN backward pass for GPU acceleration.
    pub(crate) fn propagate_crown_batched_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_batched_inner(
            input,
            MulBinaryRelaxationMode::default(),
            None,
            None,
            engine,
            AttentionCompositionRuntime::production(),
        )
        .map(|result| result.bounds)
    }

    /// Experimental: propagate bounds using full attention-CROWN composition (#318).
    ///
    /// Unlike the production `propagate_crown_batched`, which concretizes with IBP
    /// at attention-shaped MatMul boundaries (partial CROWN), this method
    /// accumulates the attention-identity retry result and continues backward
    /// through the graph. This gives CROWN propagation through the attention
    /// MatMul at the cost of McCormick independence assumptions on Q and K.
    ///
    /// Use this only for diagnostic comparisons against IBP and zonotope seeds.
    /// On the current Whisper block-0 regression surface, the committed #318
    /// tests show this lane already collapses to the IBP seed at `context`, so
    /// it is intentionally kept out of the production verifier/config routing.
    ///
    /// Returns `CrownBackwardResult` with `BoundsProvenance::Crown`.
    /// If the attention-identity retry fails, falls back to partial CROWN
    /// identically to the production path.
    #[instrument(skip(self, input), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched_with_attention_full_composition(
        &self,
        input: &BoundedTensor,
    ) -> Result<CrownBackwardResult> {
        self.propagate_crown_batched_with_attention_full_composition_and_engine(input, None)
    }

    /// Engine-aware variant of `propagate_crown_batched_with_attention_full_composition` (#3772).
    pub fn propagate_crown_batched_with_attention_full_composition_and_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<CrownBackwardResult> {
        self.propagate_crown_batched_inner(
            input,
            MulBinaryRelaxationMode::default(),
            None,
            None,
            engine,
            AttentionCompositionRuntime::full_composition(),
        )
    }

    #[cfg(test)]
    pub(crate) fn propagate_crown_batched_with_attention_full_composition_diagnostic(
        &self,
        input: &BoundedTensor,
    ) -> Result<(CrownBackwardResult, bool)> {
        let mut used_attention_full_composition = false;
        let result = self.propagate_crown_batched_inner(
            input,
            MulBinaryRelaxationMode::default(),
            None,
            None,
            None,
            AttentionCompositionRuntime::full_composition_with_diagnostic(
                &mut used_attention_full_composition,
            ),
        )?;
        Ok((result, used_attention_full_composition))
    }

    /// Propagate bounds through the graph using N-D batched CROWN with configurable MulBinary relaxation.
    #[inline]
    #[instrument(skip(self, input), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched_with_relaxation(
        &self,
        input: &BoundedTensor,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_batched_inner(
            input,
            mul_binary_relaxation,
            None,
            None,
            None,
            AttentionCompositionRuntime::production(),
        )
        .map(|result| result.bounds)
    }

    /// Propagate bounds through the graph using N-D batched CROWN with deadline enforcement (#3398).
    ///
    /// When `deadline` is `Some`, checks at each node in the backward loop.
    /// If exceeded, falls back to IBP (always sound). This prevents batched CROWN
    /// from exceeding the verification timeout for large models.
    #[inline]
    #[instrument(skip(self, input, deadline), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched_with_relaxation_and_deadline(
        &self,
        input: &BoundedTensor,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
    ) -> Result<CrownBackwardResult> {
        self.propagate_crown_batched_with_engine_relaxation_and_deadline(
            input,
            None,
            mul_binary_relaxation,
            deadline,
        )
    }

    /// Propagate bounds through the graph using N-D batched CROWN with an
    /// explicit GEMM engine, configurable MulBinary relaxation, and deadline.
    #[inline]
    #[instrument(skip(self, input, engine, deadline), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched_with_engine_relaxation_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
    ) -> Result<CrownBackwardResult> {
        self.propagate_crown_batched_inner(
            input,
            mul_binary_relaxation,
            None,
            deadline,
            engine,
            AttentionCompositionRuntime::production(),
        )
    }
}
