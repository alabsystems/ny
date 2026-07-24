// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Trait for external bound tightening backends.
//!
//! Bound tightening takes pre-activation bounds at a network layer and
//! returns tighter bounds by solving optimization problems. LP/MIP solvers
//! can produce tighter bounds than CROWN alone by encoding full network
//! constraints.
//!
//! Design: `designs/2026-03-04-highs-mip-solver-integration.md`
//! Part of #1763.

use crate::Bound;

/// Trait for external bound tightening backends (LP, MIP, etc.).
///
/// The BaB loop calls this (when configured) after CROWN propagation
/// to refine intermediate neuron bounds. Tighter bounds reduce the number
/// of unstable neurons and improve branching decisions.
///
/// ```text
/// BaB iteration:
///   1. CROWN propagation → intermediate bounds
///   2. (optional) bound tightening → tighter intermediate bounds
///   3. Branching with tighter bounds → fewer subdomains
/// ```
///
/// Reference: alpha-beta-CROWN `lp_solver()` in
/// `complete_verifier/lp_mip_solver/bounds_core.py:37-92`
pub trait BoundTightener {
    /// Error type returned by the tightener.
    type Error: std::fmt::Debug + std::fmt::Display;

    /// Tighten pre-activation bounds for a specific layer.
    ///
    /// The returned bounds must be at least as tight as (or equal to) the
    /// input `current_bounds` — they must never widen bounds. Implementations
    /// solve one optimization per unstable neuron (minimize for lower bound,
    /// maximize for upper bound).
    ///
    /// # Arguments
    ///
    /// * `layer_idx` — Index of the layer whose pre-activation bounds to tighten
    /// * `current_bounds` — Current pre-activation bounds for each neuron in the layer
    ///
    /// # Returns
    ///
    /// Tightened bounds for each neuron in the layer (same length as `current_bounds`).
    /// Stable neurons (lower >= 0 or upper <= 0) are returned unchanged.
    fn tighten(
        &self,
        layer_idx: usize,
        current_bounds: &[Bound],
    ) -> Result<Vec<Bound>, Self::Error>;
}
