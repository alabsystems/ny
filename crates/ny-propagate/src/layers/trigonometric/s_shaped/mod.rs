// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! S-shaped activation layers (Tanh, Sigmoid, Arctan) for bound propagation.
//!
//! These activations share a common structure:
//! - Monotonically increasing
//! - S-shaped curve (concave for x > 0, convex for x < 0, or vice versa)
//! - Use precomputed tangent point tables for tight linear relaxations

mod arctan;
mod erf;
mod shared;
mod sigmoid;
mod tanh;

#[cfg(test)]
mod tests;

pub use arctan::ArctanLayer;
pub use erf::ErfLayer;
pub use sigmoid::SigmoidLayer;
pub use tanh::TanhLayer;

pub(crate) use sigmoid::{sigmoid_crossing_default_tangents, sigmoid_linear_relaxation};
pub(crate) use tanh::{tanh_crossing_default_tangents, tanh_linear_relaxation};

// Re-export sigmoid_f64 for use by softplus within the trigonometric module.
pub(super) use sigmoid::sigmoid_f64;
