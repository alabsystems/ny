// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Periodic activation layers (Sin, Cos, Tan) for bound propagation.
//!
//! These activations share common infrastructure:
//! - Periodic functions requiring extrema detection for IBP
//! - Tangent/secant relaxations for intervals within fixed concavity regions
//! - Fallback to constant bounds when interval crosses inflection points

mod common;
mod cos;
mod sin;
mod tan;

pub use cos::CosLayer;
pub use sin::SinLayer;
pub use tan::TanLayer;

#[cfg(test)]
mod tests;
