// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Layer implementations for bound propagation.
//!
//! This module contains all layer types that support IBP and CROWN bound propagation.
//! Each layer implements the `BoundPropagation` trait.
//!
//! ## Organization
//!
//! Layers are organized into submodules by category:
//! - `linear`: Fully-connected linear layers
//! - `activations`: ReLU family and element-wise activations
//! - `softmax`: Softmax, LogSoftmax, GELU, LogSumExp, CausalSoftmax layers
//! - `normalization`: LayerNorm, BatchNorm
//! - `convolution`: Conv1d, Conv2d
//! - `pooling`: AveragePool, MaxPool
//! - `binary_ops`: MatMul, Add, Mul, Concat, Sub, Div, Atan2
//! - `arithmetic`: Constant arithmetic operations
//! - `transform`: Reshape, Pad, Resize, Transpose, Tile, Expand, Slice, Flatten
//! - `reduction`: ReduceMean, ReduceSum
//! - `trigonometric`: Tanh, Sigmoid, Softplus, Sin, Cos, Tan, Arctan
//! - `misc`: SkipMerge, OpaqueSkip, Floor, Ceil, Round, Sign, Reciprocal, Where, NonZero

/// Common traits and utilities shared across all layer implementations.
pub(crate) mod common;

/// ReLU family and element-wise activations (ReLU, LeakyReLU, ELU, CELU, SELU, SiLU, etc.).
pub(crate) mod activations;
/// Constant arithmetic operations (AddConstant, MulConstant, Abs, Sqrt, Pow).
pub(crate) mod arithmetic;
/// Self-attention layers with causal masking for interval bound propagation.
pub(crate) mod attention;
/// Binary operations (MatMul, Add, Mul, Sub, Div, Atan2, Concat, Min, Max).
pub(crate) mod binary_ops;
/// Convolutional layers (Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d).
pub(crate) mod convolution;
/// Double-precision (f64) layer implementations for soundness-critical propagation.
pub(crate) mod float64;
/// Fully-connected linear layers (y = Wx + b) with faer-accelerated GEMM.
pub(crate) mod linear;
/// Miscellaneous layers (SkipMerge, OpaqueSkip, Floor, Ceil, Round, Sign, Reciprocal, Where, NonZero).
pub(crate) mod misc;
/// Normalization layers (LayerNorm, BatchNorm).
pub(crate) mod normalization;
/// Pooling layers (AveragePool, MaxPool2d).
pub(crate) mod pooling;
/// Reduction operations (ReduceMean, ReduceSum).
pub(crate) mod reduction;
/// Rotary Position Embedding (RoPE): pair-wise rotation for positional encoding in transformers.
pub(crate) mod rope;
#[cfg(test)]
mod rope_tests;
/// Softmax family (Softmax, LogSoftmax, LogSumExp, GELU, CausalSoftmax).
pub(crate) mod softmax;
/// Shape transformation layers (Reshape, Pad, Resize, Transpose, Tile, Expand, Slice, Flatten, Squeeze, Unsqueeze, Gather).
pub(crate) mod transform;
/// Legacy transformer-only IBP helper regressions retained for unit tests.
#[cfg(test)]
mod transformer;
/// Trigonometric and S-shaped activations (Tanh, Sigmoid, Softplus, Sin, Cos, Tan, Arctan).
pub(crate) mod trigonometric;

/// Unified `Layer` enum wrapping all layer types for dynamic dispatch.
pub(crate) mod layer_enum;

/// CROWN elementwise backward pass and the [`BoundPropagation`] trait implemented by all layers.
pub use common::BoundPropagation;

#[cfg(test)]
pub(crate) use convolution::conv1d::{conv1d_single, conv1d_transpose};
#[cfg(test)]
pub(crate) use convolution::conv2d::{conv2d_single, conv2d_transpose};

pub use softmax::{GeluApproximation, RelaxationMode};

/// Activation layer types (ReLU, ELU, CELU, SELU, SiLU, Mish, etc.).
pub use activations::{
    CeluLayer, ClipLayer, EluLayer, ExpLayer, HardSigmoidLayer, HardSwishLayer, LeakyReLULayer,
    LinearRelaxation, LogLayer, MishLayer, PReluLayer, ReLULayer, SeluLayer, ShrinkLayer,
    SiLULayer, SnakeLayer, SoftsignLayer, ThresholdedReluLayer,
};
/// Constant arithmetic layer types (AddConstant, Abs, Sqrt, Pow).
pub use arithmetic::{
    AbsLayer, AddConstantLayer, DivConstantLayer, MulConstantLayer, PowConstantLayer, SqrtLayer,
    SubConstantLayer,
};
/// Self-attention layer types with causal masking support.
pub use attention::{AttentionMask, SelfAttentionLayer};
/// Binary operation layer types (Add, Sub, Mul, Div, Atan2, MatMul, Concat, Min, Max, CompareTensor).
pub use binary_ops::{
    AddLayer, Atan2Layer, BilinearCrownLayer, CompareTensorLayer, ConcatLayer, DivLayer,
    MatMulIbpMode, MatMulLayer, MaxBinaryLayer, MinBinaryLayer, MulBinaryLayer, SubLayer,
};
/// Convolutional layer types (Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d).
pub use convolution::{Conv1dLayer, Conv2dLayer, ConvTranspose1dLayer, ConvTranspose2dLayer};
/// Fully-connected linear layer (y = Wx + b).
pub use linear::LinearLayer;
/// Miscellaneous layer types (Floor, Ceil, Round, Sign, Reciprocal, Compare, Where, NonZero, SkipMerge).
pub use misc::{
    CeilLayer, CompareLayer, CompareOp, FloorLayer, NonZeroLayer, OpaqueSkipLayer,
    QdqPerturbationLayer, ReciprocalLayer, RoundLayer, SignLayer, SkipMergeLayer, TruncLayer,
    WhereLayer,
};
/// Normalization layer types (BatchNorm, InstanceNorm1d, LayerNorm, RmsNorm) and CROWN mode configuration.
pub use normalization::{
    AdaIN1dLayer, BatchNormChannelAxisHint, BatchNormLayer, GroupNormLayer, InstanceNorm1dLayer,
    LayerNormCrownMode, LayerNormLayer, LayerNormMode, RmsNormLayer, NORMALIZATION_MIN_EPS,
};
/// Pooling layer types (AveragePool, MaxPool2d).
pub use pooling::{AveragePoolLayer, MaxPool2dLayer};
/// Reduction operation layer types (ReduceMean, ReduceSum, ReduceMax, ReduceMin, Topk, ArgMax, ArgMin, ArgSort).
pub use reduction::{
    ArgMaxLayer, ArgMinLayer, ArgSortLayer, CumsumLayer, ReduceMaxLayer, ReduceMeanLayer,
    ReduceMinLayer, ReduceSumLayer, TopkLayer, TopkOutputKind,
};
/// Rotary Position Embedding (RoPE) layer type.
pub use rope::RopeLayer;
/// Softmax family layer types (Softmax, LogSoftmax, LogSumExp, GELU, CausalSoftmax).
pub use softmax::{CausalSoftmaxLayer, GELULayer, LogSoftmaxLayer, LogSumExpLayer, SoftmaxLayer};
/// Shape transformation layer types (ExpandLikeLastAxis, Flatten, Gather, Pad, Reshape, Resize, ScatterAdd, IndexAdd, ScatterND, Slice, Squeeze, Tile, Transpose, Unsqueeze).
pub use transform::{
    normalize_transpose_perm_for_rank, ExpandLikeLastAxisLayer, FlattenLayer, GatherLayer,
    IndexAddLayer, PadLayer, PadMode, ReshapeLayer, ResizeLayer, ScatterAddLayer, ScatterNdLayer,
    SliceLayer, SqueezeLayer, TileLayer, TransposeLayer, UnsqueezeLayer,
};
/// Trigonometric and S-shaped activation layer types (Arctan, Cos, Erf, Sigmoid, Sin, Softplus, Tan, Tanh).
pub use trigonometric::{
    ArctanLayer, CosLayer, ErfLayer, SigmoidLayer, SinLayer, SoftplusLayer, TanLayer, TanhLayer,
};

/// Unified [`Layer`] enum for dynamic dispatch across all layer types.
pub use layer_enum::Layer;

// Re-exports for tests and Kani proofs: relaxation functions used by tests via
// `crate::layers::*` (#3240) and by external Kani proof harnesses (#2305).
pub use activations::{
    exp_linear_relaxation, log_linear_relaxation, silu_eval, silu_sound_linear_relaxation,
};
pub use arithmetic::{abs_linear_relaxation, pow2_linear_relaxation, sqrt_linear_relaxation};
pub use softmax::{
    adaptive_gelu_linear_relaxation, exp_interval_bounds, gelu_eval, gelu_linear_relaxation,
    gelu_sound_linear_relaxation, gelu_tanh_inflection_point, gelu_tanh_sound_linear_relaxation,
    logsoftmax_ibp_bounds, logsumexp_slice, softmax_ibp_element_bounds,
};
