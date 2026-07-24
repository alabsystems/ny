// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Curated semantic layer types for external consumer APIs.
//!
//! Re-exports the layer trait and the semantic model-layer subset that forms
//! the stable `ny_api` propagate facade. This is intentionally narrower than
//! the full internal `ny_propagate::layers` namespace.

pub use ny_propagate::layers::{
    // Arithmetic
    AbsLayer,
    // Normalization
    AdaIN1dLayer,
    AddConstantLayer,
    // Binary ops
    AddLayer,
    // Existing curated facade types
    AttentionMask,
    BatchNormLayer,
    // Activations
    ClipLayer,
    // Transform / structural
    ConcatLayer,
    Conv1dLayer,
    Conv2dLayer,
    ConvTranspose1dLayer,
    CumsumLayer,
    ExpLayer,
    GELULayer,
    GeluApproximation,
    InstanceNorm1dLayer,
    LayerNormCrownMode,
    LayerNormLayer,
    LinearLayer,
    // Linear
    MatMulLayer,
    MulBinaryLayer,
    MulConstantLayer,
    PowConstantLayer,
    ReLULayer,
    ReshapeLayer,
    RmsNormLayer,
    // Position encoding
    RopeLayer,
    SelfAttentionLayer,
    SiLULayer,
    SigmoidLayer,
    SliceLayer,
    SnakeLayer,
    SoftmaxLayer,
    SoftplusLayer,
    SqrtLayer,
    SubLayer,
    TanhLayer,
    TileLayer,
    TransposeLayer,
};
pub use ny_propagate::{BoundPropagation, Layer};
