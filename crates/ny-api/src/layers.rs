// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Curated semantic layer types for external consumer APIs.
//!
//! Re-exports the layer trait and the semantic model-layer subset that forms
//! the stable `ny_api` propagate facade. This is intentionally narrower than
//! the full internal `ny_propagate::layers` namespace.
//!
//! # What belongs here
//!
//! A layer type belongs on this surface when a downstream *graph translator*
//! has to name it to build a [`Layer`] — that is, when a consumer lowering its
//! own model IR into a ny graph would otherwise have to reach past the facade
//! into `ny_propagate::layers`. Reaching past the facade is what makes a
//! consumer depend on `ny-propagate` directly, which is exactly what this
//! module exists to prevent.
//!
//! Presence here says nothing about how tightly a layer's bounds propagate;
//! [`crate::conformance`] classifies that separately.

pub use ny_propagate::layers::{
    // Arithmetic
    AbsLayer,
    // Normalization
    AdaIN1dLayer,
    AddConstantLayer,
    // Binary ops
    AddLayer,
    // Trigonometric / inverse-trigonometric
    Atan2Layer,
    // Existing curated facade types
    AttentionMask,
    // Pooling
    AveragePoolLayer,
    BatchNormLayer,
    // Fused causal attention softmax
    CausalSoftmaxLayer,
    // Misc rounding / selection (elementwise, exactly enclosable)
    CeilLayer,
    // Activations
    ClipLayer,
    // Compare family: elementwise comparison against a constant threshold
    CompareLayer,
    CompareOp,
    // Transform / structural
    ConcatLayer,
    Conv1dLayer,
    Conv2dLayer,
    ConvTranspose1dLayer,
    ConvTranspose2dLayer,
    CosLayer,
    CumsumLayer,
    DivConstantLayer,
    DivLayer,
    // Robustness-region helper: widen bounds by a per-element epsilon
    EpsilonExpandLayer,
    ExpLayer,
    FloorLayer,
    GELULayer,
    GeluApproximation,
    InstanceNorm1dLayer,
    LayerNormCrownMode,
    LayerNormLayer,
    LinearLayer,
    LogSoftmaxLayer,
    // Linear
    MatMulLayer,
    MaxBinaryLayer,
    MaxPool2dLayer,
    MinBinaryLayer,
    MulBinaryLayer,
    MulConstantLayer,
    PowConstantLayer,
    ReLULayer,
    ReciprocalLayer,
    // Reductions
    ReduceMaxLayer,
    ReduceMeanLayer,
    ReduceMinLayer,
    ReduceSumLayer,
    ReshapeLayer,
    RmsNormLayer,
    // Position encoding
    RopeLayer,
    SelfAttentionLayer,
    SiLULayer,
    SigmoidLayer,
    SignLayer,
    SinLayer,
    SliceLayer,
    SnakeLayer,
    SoftmaxLayer,
    SoftplusLayer,
    SqrtLayer,
    SqueezeLayer,
    SubConstantLayer,
    SubLayer,
    TanhLayer,
    TileLayer,
    TransposeLayer,
    TruncLayer,
    UnsqueezeLayer,
    // Ternary select: elementwise `cond ? a : b` with sound branch-union bounds
    WhereLayer,
};
pub use ny_propagate::{BoundPropagation, Layer};
