// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unified Layer enum for bound propagation.
//!
//! Split into submodules:
//! - `query`: Layer classification and type queries (layer_type, is_binary, supports_batched_crown)
//! - `dispatch`: BoundPropagation trait impl and batched CROWN dispatch

mod dispatch;
mod query;

// Import all layer types
use super::activations::CeluLayer;
use super::activations::ClipLayer;
use super::activations::EluLayer;
use super::activations::ExpLayer;
use super::activations::HardSigmoidLayer;
use super::activations::HardSwishLayer;
use super::activations::LeakyReLULayer;
use super::activations::LogLayer;
use super::activations::MishLayer;
use super::activations::PReluLayer;
use super::activations::ReLULayer;
use super::activations::SeluLayer;
use super::activations::ShrinkLayer;
use super::activations::SiLULayer;
use super::activations::SnakeLayer;
use super::activations::SoftsignLayer;
use super::activations::ThresholdedReluLayer;
use super::arithmetic::AbsLayer;
use super::arithmetic::AddConstantLayer;
use super::arithmetic::DivConstantLayer;
use super::arithmetic::MulConstantLayer;
use super::arithmetic::PowConstantLayer;
use super::arithmetic::SqrtLayer;
use super::arithmetic::SubConstantLayer;
use super::attention::SelfAttentionLayer;
use super::binary_ops::AddLayer;
use super::binary_ops::Atan2Layer;
use super::binary_ops::BilinearCrownLayer;
use super::binary_ops::CompareTensorLayer;
use super::binary_ops::ConcatLayer;
use super::binary_ops::DivLayer;
use super::binary_ops::MatMulLayer;
use super::binary_ops::MaxBinaryLayer;
use super::binary_ops::MinBinaryLayer;
use super::binary_ops::MulBinaryLayer;
use super::binary_ops::SubLayer;
use super::convolution::Conv1dLayer;
use super::convolution::Conv2dLayer;
use super::convolution::ConvTranspose1dLayer;
use super::convolution::ConvTranspose2dLayer;
use super::linear::LinearLayer;
use super::misc::CeilLayer;
use super::misc::CompareLayer;
use super::misc::FloorLayer;
use super::misc::NonZeroLayer;
use super::misc::QdqPerturbationLayer;
use super::misc::ReciprocalLayer;
use super::misc::RoundLayer;
use super::misc::SignLayer;
use super::misc::TruncLayer;
use super::misc::WhereLayer;
use super::misc::{OpaqueSkipLayer, SkipMergeLayer};
use super::normalization::AdaIN1dLayer;
use super::normalization::BatchNormLayer;
use super::normalization::GroupNormLayer;
use super::normalization::InstanceNorm1dLayer;
use super::normalization::LayerNormLayer;
use super::normalization::RmsNormLayer;
use super::pooling::AveragePoolLayer;
use super::pooling::MaxPool2dLayer;
use super::reduction::ArgMaxLayer;
use super::reduction::ArgMinLayer;
use super::reduction::ArgSortLayer;
use super::reduction::CumsumLayer;
use super::reduction::ReduceMaxLayer;
use super::reduction::ReduceMeanLayer;
use super::reduction::ReduceMinLayer;
use super::reduction::ReduceSumLayer;
use super::reduction::TopkLayer;
use super::rope::RopeLayer;
use super::softmax::CausalSoftmaxLayer;
use super::softmax::GELULayer;
use super::softmax::LogSoftmaxLayer;
use super::softmax::LogSumExpLayer;
use super::softmax::SoftmaxLayer;
use super::transform::ExpandLikeLastAxisLayer;
use super::transform::FlattenLayer;
use super::transform::GatherLayer;
use super::transform::IndexAddLayer;
use super::transform::PadLayer;
use super::transform::ReshapeLayer;
use super::transform::ResizeLayer;
use super::transform::ScatterAddLayer;
use super::transform::ScatterNdLayer;
use super::transform::SliceLayer;
use super::transform::SqueezeLayer;
use super::transform::TileLayer;
use super::transform::TransposeLayer;
use super::transform::UnsqueezeLayer;
use super::trigonometric::ArctanLayer;
use super::trigonometric::CosLayer;
use super::trigonometric::ErfLayer;
use super::trigonometric::SigmoidLayer;
use super::trigonometric::SinLayer;
use super::trigonometric::SoftplusLayer;
use super::trigonometric::TanLayer;
use super::trigonometric::TanhLayer;

/// Invoke a macro for each elementwise activation variant that has standard batched CROWN
/// signature: `propagate_linear_batched_with_bounds(&BatchedLinearBounds, &BoundedTensor)
/// -> Result<BatchedLinearBounds>`.
///
/// This is the single source of truth for which Layer variants are elementwise activations.
/// Used to generate match arms in `propagate_crown_backward_batched` (#1708) and
/// `is_elementwise_activation` (#1753), eliminating N-way duplication across dispatch sites.
macro_rules! for_each_elementwise_activation {
    ($macro_name:ident) => {
        $macro_name!(
            ReLU,
            GELU,
            SiLU,
            Tanh,
            Sigmoid,
            Erf,
            Exp,
            Log,
            Sqrt,
            Reciprocal,
            Softplus,
            HardSwish,
            Mish,
            Selu,
            Softsign,
            Arctan,
            Tan,
            Sin,
            Cos,
            Elu,
            Celu,
            LeakyReLU,
            HardSigmoid,
            Clip,
            ThresholdedRelu,
            Abs,
            PowConstant,
            Floor,
            Ceil,
            Round,
            Trunc,
            Sign,
            PRelu,
            Shrink,
            Snake,
            Compare
        )
    };
}

// Make the macro available to submodules (query.rs, dispatch.rs).
pub(crate) use for_each_elementwise_activation;

/// Invoke a macro for each elementwise activation that natively supports Patches CROWN backward.
///
/// These are activations with `propagate_patches_with_bounds(&PatchesLinearBounds, &BoundedTensor)
/// -> Result<CrownBounds>` — either from `impl_elementwise_activation!` macro generation
/// or custom `PatchesPropagation` trait impl (ReLU).
///
/// Activations NOT listed here fall back to Dense via `ensure_dense()` in the CROWN
/// backward engine. They can be added incrementally as they gain Patches support.
///
/// Part of #2613 Phase 2 step 11.
macro_rules! for_each_patches_capable_activation {
    ($macro_name:ident) => {
        $macro_name!(
            // From impl_elementwise_activation! macro (auto-generated patches method)
            ReLU,
            SiLU,
            Exp,
            Log,
            Selu,
            Elu,
            Celu,
            LeakyReLU,
            HardSigmoid,
            Clip,
            ThresholdedRelu,
            Snake,
            Shrink,
            // Manual-impl activations with patches method added in Phase 2
            GELU,
            Tanh,
            Sigmoid,
            Erf,
            Softplus,
            HardSwish,
            Mish,
            Softsign,
            Arctan,
            Tan,
            Sin,
            Cos,
            Sqrt,
            Abs,
            PowConstant,
            Reciprocal,
            // Piecewise constant layers (slope=0, constant bounds)
            Floor,
            Ceil,
            Round,
            Trunc,
            Sign,
            Compare
        )
    };
}
pub(crate) use for_each_patches_capable_activation;

/// Enum wrapper for different layer types with their parameters.
///
/// # Naming Conventions
///
/// ## Unary vs Binary Operations
///
/// Binary operations take two bounded inputs and have doc comments starting with "Binary:".
/// Unary operations have one bounded input; constant operations use the `*Constant` suffix.
///
/// | Pattern | Example | Description |
/// |---------|---------|-------------|
/// | `OpBinary` | `MulBinary`, `MinBinary`, `MaxBinary` | Two bounded inputs |
/// | `Op` | `Add`, `Sub`, `Div`, `MatMul` | Two bounded inputs (legacy) |
/// | `OpConstant` | `AddConstant`, `MulConstant` | One bounded input + constant |
///
/// Note: `Add`, `Sub`, `Div`, and `MatMul` are binary operations (two bounded inputs)
/// but lack the `Binary` suffix for historical reasons. Search for "Binary:" in doc
/// comments to find all binary operations.
///
/// ## Capitalization
///
/// Activation names follow their original paper capitalization where unambiguous:
/// - `ReLU`, `SiLU`, `GELU`: All-caps suffixes from literature
/// - `LeakyReLU`, `PRelu`, `Selu`, `Celu`: Mixed case variants
///
/// # Binary Operations
///
/// For binary operations, use `propagate_ibp_binary` method directly on the layer struct,
/// or use `GraphNetwork` for graph-based computation.
// Boxing LinearLayer would require updating 600+ pattern match sites across the codebase.
// The variant size (416 bytes) is acceptable for a core enum that's matched by reference.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum Layer {
    Linear(LinearLayer),
    Conv1d(Conv1dLayer),
    Conv2d(Conv2dLayer),
    ConvTranspose1d(ConvTranspose1dLayer),
    ConvTranspose2d(ConvTranspose2dLayer),
    /// Average pooling over spatial dimensions
    AveragePool(AveragePoolLayer),
    /// Max pooling over spatial dimensions
    MaxPool2d(MaxPool2dLayer),
    ReLU(ReLULayer),
    /// Leaky ReLU activation (allows small gradient for negative inputs)
    LeakyReLU(LeakyReLULayer),
    /// Clip: clamp values to [min, max] range
    Clip(ClipLayer),
    /// ELU (Exponential Linear Unit) activation
    Elu(EluLayer),
    /// SELU (Scaled ELU) activation with self-normalizing properties
    Selu(SeluLayer),
    /// PRelu (Parametric ReLU) with per-channel learned slopes
    PRelu(PReluLayer),
    /// HardSigmoid: max(0, min(1, alpha * x + beta))
    HardSigmoid(HardSigmoidLayer),
    /// HardSwish: x * HardSigmoid(x)
    HardSwish(HardSwishLayer),
    /// SiLU (Swish): x * sigmoid(x)
    SiLU(SiLULayer),
    /// Exp: element-wise exponential
    Exp(ExpLayer),
    /// Log: element-wise natural logarithm
    Log(LogLayer),
    /// Celu: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    Celu(CeluLayer),
    /// Mish: x * tanh(softplus(x))
    Mish(MishLayer),
    /// LogSoftmax: log(softmax(x)) = x - logsumexp(x)
    LogSoftmax(LogSoftmaxLayer),
    /// LogSumExp: log(sum(exp(x))) reduction
    LogSumExp(LogSumExpLayer),
    /// ThresholdedRelu: y = x if x > alpha, else 0
    ThresholdedRelu(ThresholdedReluLayer),
    /// Shrink: soft thresholding for sparse coding
    Shrink(ShrinkLayer),
    /// Softsign: x / (1 + |x|)
    Softsign(SoftsignLayer),
    /// Snake: x + (1/a) * sin²(a*x) — for neural audio synthesis
    Snake(SnakeLayer),
    Softmax(SoftmaxLayer),
    /// Causal softmax for decoder attention (masked)
    CausalSoftmax(CausalSoftmaxLayer),
    /// Ternary: self-attention (Q, K, V)
    SelfAttention(SelfAttentionLayer),
    GELU(GELULayer),
    LayerNorm(LayerNormLayer),
    /// Unary: RMSNorm — x / sqrt(mean(x^2) + eps), used in LLMs (LLaMA, Qwen, Mistral)
    RmsNorm(RmsNormLayer),
    /// Unary: InstanceNorm1d — per-channel normalization over time dimension, used in audio/style transfer
    InstanceNorm1d(InstanceNorm1dLayer),
    /// Unary: GroupNorm — per-group normalization across channels and time, used in Demucs DConv
    GroupNorm(GroupNormLayer),
    /// Unary: AdaIN1d — adaptive instance normalization (style_gamma * InstanceNorm(x) + style_beta)
    AdaIN1d(AdaIN1dLayer),
    /// Unary: batch normalization (for CNNs)
    BatchNorm(BatchNormLayer),
    /// Binary: bounded matrix multiplication (e.g., Q @ K^T)
    MatMul(MatMulLayer),
    /// Binary: element-wise multiplication (e.g., SwiGLU gating: up * silu(gate))
    MulBinary(MulBinaryLayer),
    /// Binary: element-wise addition (e.g., residual connections)
    Add(AddLayer),
    /// Binary: concatenation along axis (e.g., CLS token + patches in ViT)
    Concat(ConcatLayer),
    /// Binary: element-wise subtraction (e.g., x - mean(x) in LayerNorm)
    Sub(SubLayer),
    /// Binary: element-wise division (e.g., x / sqrt(var + eps) in LayerNorm)
    Div(DivLayer),
    /// Binary: two-argument arctangent atan2(y, x) for phase-style graph ops
    Atan2(Atan2Layer),
    /// Binary: bilinear CROWN matmul for attention Q@K^T with McCormick composition
    BilinearCrown(BilinearCrownLayer),
    /// Binary: element-wise minimum (e.g., clamp upper, residual min)
    MinBinary(MinBinaryLayer),
    /// Binary: element-wise maximum (e.g., clamp lower, residual max)
    MaxBinary(MaxBinaryLayer),
    /// Unary: add constant tensor (e.g., bias addition)
    AddConstant(AddConstantLayer),
    /// Tensor transpose (permute axes)
    Transpose(TransposeLayer),
    /// Tensor reshape (change shape, preserve total elements)
    Reshape(ReshapeLayer),
    /// Tensor flatten (flatten dimensions according to axis)
    Flatten(FlattenLayer),
    /// Unary: explicit pad with constant or reflected borders
    Pad(PadLayer),
    /// Unary: multiply by constant tensor (e.g., attention scaling)
    MulConstant(MulConstantLayer),
    /// Unary: element-wise absolute value
    Abs(AbsLayer),
    /// Unary: element-wise square root
    Sqrt(SqrtLayer),
    /// Unary: divide by constant tensor
    DivConstant(DivConstantLayer),
    /// Unary: subtract constant or subtract from constant
    SubConstant(SubConstantLayer),
    /// Unary: element-wise power (x^p)
    PowConstant(PowConstantLayer),
    /// Unary: reduce mean over axes
    ReduceMean(ReduceMeanLayer),
    /// Unary: reduce sum over axes
    ReduceSum(ReduceSumLayer),
    /// Unary: reduce max over axes (fixed_max_index CROWN assumption)
    ReduceMax(ReduceMaxLayer),
    /// Unary: reduce min over axes (fixed_min_index CROWN assumption)
    ReduceMin(ReduceMinLayer),
    /// Unary: top-k selection returning values or indices.
    Topk(TopkLayer),
    /// Unary: argmax reduction returning indices.
    ArgMax(ArgMaxLayer),
    /// Unary: argmin reduction returning indices.
    ArgMin(ArgMinLayer),
    /// Unary: argsort permutation indices.
    ArgSort(ArgSortLayer),
    /// Unary: cumulative sum (prefix sum) along axis
    CumSum(CumsumLayer),
    /// Unary: hyperbolic tangent activation
    Tanh(TanhLayer),
    /// Unary: sigmoid activation
    Sigmoid(SigmoidLayer),
    /// Unary: Gaussian error function
    Erf(ErfLayer),
    /// Unary: softplus activation (smooth ReLU)
    Softplus(SoftplusLayer),
    /// Unary: sine function (for positional encodings)
    Sin(SinLayer),
    /// Unary: cosine function (for positional encodings)
    Cos(CosLayer),
    /// Unary: tangent function
    Tan(TanLayer),
    /// Unary: arctangent function
    Arctan(ArctanLayer),
    /// Unary: Rotary Position Embedding — pair-wise rotation for positional encoding (avoice K6)
    RoPE(RopeLayer),
    /// Unary: tile/repeat along axis (for GQA KV head expansion)
    Tile(TileLayer),
    /// Binary: expand `[... ,1]` along the last axis to match a live reference tensor.
    ExpandLikeLastAxis(ExpandLikeLastAxisLayer),
    /// Unary: gather/index selection along axis (indices may be embedded).
    Gather(GatherLayer),
    /// Variable-arity additive scatter with optional embedded data/index/src.
    ScatterAdd(ScatterAddLayer),
    /// Variable-arity indexed add with optional embedded data/index/src.
    IndexAdd(IndexAddLayer),
    /// ScatterND: scatter updates into data using static or dynamic indices.
    ScatterNd(ScatterNdLayer),
    /// Unary: slice/extract contiguous range along axis (for Split op)
    Slice(SliceLayer),
    /// Unary: squeeze (remove dimension of size 1)
    Squeeze(SqueezeLayer),
    /// Unary: unsqueeze (insert dimension of size 1)
    Unsqueeze(UnsqueezeLayer),
    /// Unary: nearest-neighbor spatial resize (upsample last two dims).
    Resize(ResizeLayer),
    /// Unary: element-wise comparison (tensor vs scalar threshold), output in {0.0, 1.0}
    Compare(CompareLayer),
    /// Binary: element-wise comparison (tensor vs tensor), output in {0.0, 1.0}
    CompareTensor(CompareTensorLayer),
    /// Ternary: conditional selection Where(condition, x, y)
    Where(WhereLayer),
    /// Unary: NonZero - returns indices of non-zero elements (data-dependent output)
    NonZero(NonZeroLayer),
    /// Unary: floor(x) - round towards negative infinity
    Floor(FloorLayer),
    /// Unary: ceil(x) - round towards positive infinity
    Ceil(CeilLayer),
    /// Unary: round(x) - round to nearest integer
    Round(RoundLayer),
    /// Unary: trunc(x) - round toward zero (from ONNX Cast to integer dtype)
    Trunc(TruncLayer),
    /// Unary: sign(x) - returns -1, 0, or 1
    Sign(SignLayer),
    /// Unary: 1/x - reciprocal
    Reciprocal(ReciprocalLayer),
    /// Unary: dependency-preserving passthrough for skipped ops
    SkipMerge(SkipMergeLayer),
    /// Unary: opaque skip for multi-input skipped ops (conservative bounds)
    OpaqueSkip(OpaqueSkipLayer),
    /// Unary: sound QDQ fake-quantization relaxation y in x +/- epsilon
    QdqPerturbation(QdqPerturbationLayer),
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_batched;
