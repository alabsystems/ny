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
    /// Run the ordinary production batched route and its existing memory
    /// fallback, returning observational decomposed-attention arrival counters.
    /// The receipt covers the batched attempt, including work completed before
    /// a fallback; it supplies no additional proof authority.
    pub fn propagate_crown_batched_with_attention_region_receipt(
        &self,
        input: &BoundedTensor,
    ) -> Result<(CrownBackwardResult, crate::network::AttentionRegionReceipt)> {
        self.propagate_crown_batched_with_attention_region_receipt_and_deadline(input, None)
    }

    /// The same production route with an original caller deadline. Expiry and
    /// memory refusal keep the ordinary complete forward/DAG fallback policies.
    pub fn propagate_crown_batched_with_attention_region_receipt_and_deadline(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<(CrownBackwardResult, crate::network::AttentionRegionReceipt)> {
        let mut receipt = Default::default();
        let result = self.propagate_crown_batched_with_provenance_runtime(
            input,
            None,
            deadline,
            AttentionCompositionRuntime::production_with_region_receipt(&mut receipt),
        );
        result.map(|result| (result, receipt))
    }

    /// Propagate bounds through the graph using N-D batched CROWN.
    ///
    /// This preserves tensor shape structure throughout propagation instead of
    /// flattening to 1D. Essential for transformer models where operations
    /// like attention have cross-position interactions.
    ///
    /// # Layer coverage
    ///
    /// Rewritten from the code it describes: the unary set below is exactly the
    /// supported arms of [`Layer::propagate_crown_backward_batched`]
    /// (`layers/layer_enum/dispatch.rs`), and the multi-input set is exactly the
    /// arms of `dispatch_special_batched_operator`
    /// (`crown_batched/dispatch.rs`). Both matches are exhaustive with no
    /// catch-all, so a new `Layer` variant is a compile error until it is placed.
    ///
    /// **Unary, full batched backward**
    /// - Linear, RoPE
    /// - Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d (Conv2d additionally
    ///   has a Patches fast path in the backward loop)
    /// - Every elementwise activation (the `for_each_elementwise_activation!`
    ///   list — ReLU, GELU, SiLU, Tanh, Sigmoid, Erf, Exp, Log, Sqrt,
    ///   Reciprocal, Softplus, HardSwish, Mish, Selu, Softsign, Arctan, Tan,
    ///   Sin, Cos, Elu, Celu, LeakyReLU, HardSigmoid, Clip, ThresholdedRelu,
    ///   Abs, PowConstant, Floor, Ceil, Round, Trunc, Sign, PRelu, Shrink,
    ///   Snake, Compare)
    /// - Softmax family: Softmax, CausalSoftmax, LogSoftmax, LogSumExp
    /// - Normalization: LayerNorm, RmsNorm, InstanceNorm1d, GroupNorm, BatchNorm,
    ///   AdaIN1d (fixed-style)
    /// - Constant arithmetic: AddConstant, SubConstant, MulConstant, DivConstant
    /// - Reductions: ReduceMean, ReduceSum, ReduceMax, ReduceMin, CumSum
    ///   (CumSum requires a last-axis grouped layout)
    /// - Shape: Flatten, Reshape, Squeeze, Unsqueeze (pass-through), Transpose,
    ///   Slice, Tile, Pad, Resize
    /// - QdqPerturbation, EpsilonExpand
    ///
    /// **Multi-input, handled by the special dispatcher**
    /// - Add, Sub, MatMul, MulBinary, BilinearCrown, ExpandLikeLastAxis
    /// - Concat — only when the join is a provable last-axis column split of two
    ///   dynamic operands (see the arm's guards); other joins keep the fallback
    /// - Where — embedded-constant select, sign-provable condition, and exact
    ///   constant-mask column split; a data-dependent condition keeps the
    ///   fallback
    ///
    /// **Not supported — these degrade to partial CROWN / IBP at the node (sound,
    /// looser)**
    /// - AveragePool, MaxPool2d
    /// - Gather, ScatterAdd, IndexAdd, ScatterND
    /// - Topk, ArgMax, ArgMin, ArgSort, NonZero
    /// - Div, Atan2, MinBinary, MaxBinary, CompareTensor
    /// - IntervalConst, SkipMerge, OpaqueSkip
    /// - SelfAttention — rejected up front with `UnsupportedOp` (see the fast-fail
    ///   note in `propagate_crown_batched_inner`), so the caller reaches the dense
    ///   backward that does handle it
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
    /// provenance path instead of surfacing a raw error.
    #[instrument(skip(self, input), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched_with_provenance(
        &self,
        input: &BoundedTensor,
    ) -> Result<CrownBackwardResult> {
        self.propagate_crown_batched_with_provenance_and_engine(input, None)
    }

    /// Batched CROWN with fallback provenance and optional GPU GEMM engine.
    ///
    /// Threads `engine` through both the batched inner path and the DAG-CROWN
    /// fallback, ensuring GPU acceleration is available in both code paths.
    #[instrument(skip(self, input, engine), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_batched_with_provenance_and_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<CrownBackwardResult> {
        self.propagate_crown_batched_with_provenance_runtime(
            input,
            engine,
            None,
            AttentionCompositionRuntime::production(),
        )
    }

    fn propagate_crown_batched_with_provenance_runtime(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        runtime: AttentionCompositionRuntime<'_>,
    ) -> Result<CrownBackwardResult> {
        // Keep the same incremental region allowance alive across the public
        // memory fallback into DAG-CROWN; the inner runtime alone ends first.
        let _region_scope =
            crate::network::graph_crown::attention_region::AttentionRegionScope::enter(deadline);
        match self.propagate_crown_batched_inner(
            input,
            MulBinaryRelaxationMode::default(),
            None,
            deadline,
            engine,
            runtime,
        ) {
            Ok(result) => Ok(result),
            Err(err @ NyError::CpuMemoryExceeded { .. }) => {
                debug!(
                    "GraphNetwork batched CROWN: {}. Falling back to DAG-CROWN provenance path",
                    err
                );
                let fallback = if deadline.is_some() {
                    self.propagate_crown_with_engine_and_deadline(input, engine, deadline)
                } else {
                    self.propagate_crown_with_provenance_and_engine(input, engine)
                };
                fallback.map(|fallback| CrownBackwardResult {
                    bounds: fallback.bounds,
                    provenance: BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::MemoryBudgetExceeded,
                    ),
                })
            }
            Err(error) if deadline.is_some() && error.is_deadline_exceeded() => {
                self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                    bounds,
                    provenance: BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::DeadlineExceeded,
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

    /// Engine-aware variant of `propagate_crown_batched_with_bilinear_alphas`.
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

    /// Engine-aware variant of `propagate_crown_batched`.
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

    /// Experimental: propagate bounds using full attention-CROWN composition.
    ///
    /// Unlike the production `propagate_crown_batched`, which concretizes with IBP
    /// at attention-shaped MatMul boundaries (partial CROWN), this method
    /// accumulates the attention-identity retry result and continues backward
    /// through the graph. This gives CROWN propagation through the attention
    /// MatMul at the cost of McCormick independence assumptions on Q and K.
    ///
    /// Use this only for diagnostic comparisons against IBP and zonotope seeds.
    /// On the current Whisper block-0 regression surface, the committed
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

    /// Engine-aware variant of `propagate_crown_batched_with_attention_full_composition`.
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

    /// Propagate bounds through the graph using N-D batched CROWN with deadline enforcement.
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
        let _region_scope =
            crate::network::graph_crown::attention_region::AttentionRegionScope::enter(deadline);
        match self.propagate_crown_batched_inner(
            input,
            mul_binary_relaxation,
            None,
            deadline,
            engine,
            AttentionCompositionRuntime::production(),
        ) {
            Err(error) if deadline.is_some() && error.is_deadline_exceeded() => {
                // Preserve the batched API's established terminal policy: a
                // mid-kernel deadline publishes the same sound forward fallback
                // as a loop-boundary expiry. It must not retry Dense or resume
                // any further CROWN work.
                self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                    bounds,
                    provenance: BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::DeadlineExceeded,
                    ),
                })
            }
            result => result,
        }
    }
}
