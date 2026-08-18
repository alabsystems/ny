// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Zonotope tensor for correlation-aware bound propagation.
//!
//! Zonotopes represent values as center + Σᵢ (coeffᵢ · eᵢ) where eᵢ ∈ [-1, 1].
//! Unlike interval bounds, zonotopes track correlations between variables through
//! shared error symbols, giving tighter bounds for operations like Q@K^T in attention.
//!
//! # Key Insight
//!
//! For Q@K^T where Q = f(X) and K = g(X) depend on the same input X:
//! - IBP treats Q and K as independent: bounds explode by ~1600x per layer
//! - Zonotopes share error symbols: `e_i² ∈ [0,1]` not `[-1,1]`, giving tighter bounds
//!
//! # References
//!
//! - Bonaert et al. (2020): "Robustness Verification for Transformers" (arxiv:2002.06622)
//! - DeepT: research/repos/DeepT/.../Verifiers/Zonotope.py

mod bilinear;
mod bilinear_disjoint;
mod constructors;
mod core;
mod gelu;
mod graph;
mod linear;
mod nonlinear;
mod shape;
mod star;

/// Zonotope tensor type with shared error symbols for correlation-aware bounds.
pub use core::ZonotopeTensor;
/// Star-set affine form (zonotope + generator-constraint polytope) for reachability.
/// New, default-off, and unwired into any verdict path (S1-2 foundation).
pub use star::{Star, StarConv2dBlockLimits, StarConv2dBlockPlan, StarReluSplit};

#[cfg(test)]
mod tests;
