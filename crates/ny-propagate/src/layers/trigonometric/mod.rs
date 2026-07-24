// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Trigonometric and S-shaped activation layers for bound propagation.
//!
//! This module provides implementations for:
//! - **S-shaped activations** (Tanh, Sigmoid, Arctan) using precomputed tangent tables
//! - **Softplus** - smooth ReLU approximation (convex function)
//! - **Periodic activations** (Sin, Cos, Tan) using tangent/secant relaxations

mod periodic;
mod s_shaped;
mod softplus;

// Re-export all public types at module root for backward compatibility
pub use periodic::{CosLayer, SinLayer, TanLayer};
pub(crate) use s_shaped::{
    sigmoid_crossing_default_tangents, sigmoid_linear_relaxation, tanh_crossing_default_tangents,
    tanh_linear_relaxation,
};
pub use s_shaped::{ArctanLayer, SigmoidLayer, TanhLayer};
pub use softplus::SoftplusLayer;
