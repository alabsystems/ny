// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `NormLayer` trait: shared interface for normalization layers.
//!
//! Implementors provide the norm-specific math (eval + jacobian).
//! The generic CROWN infrastructure (sampling, batching, mode gating)
//! is provided by free functions in [`super::crown_common`] that accept
//! any `NormLayer`.
//!
//! Reference: designs/2026-02-27-normalization-trait-dedup.md

use ndarray::{Array1, Array2};
use ny_core::Result;

use super::layer_norm::types::LayerNormCrownMode;

/// Trait for normalization layers that support CROWN linearization.
///
/// Each normalization variant (LayerNorm, RmsNorm, InstanceNorm1d, AdaIN1d)
/// implements this trait to provide its own `eval()` and `jacobian()`.
/// The shared CROWN backward propagation in [`super::crown_common`] is
/// generic over this trait, eliminating per-layer code duplication.
pub(crate) trait NormLayer {
    /// Layer name for error messages (e.g., "LayerNorm", "RMSNorm").
    fn layer_name(&self) -> &'static str;

    /// Evaluate the normalization: x → y.
    ///
    /// Input x is a flat 1D vector of all neurons. For InstanceNorm/AdaIN
    /// this is `C*T` elements (all channels concatenated).
    fn eval(&self, x: &Array1<f32>) -> Result<Array1<f32>>;

    /// Compute the Jacobian matrix dy/dx at point x.
    ///
    /// For LayerNorm/RmsNorm: full NxN Jacobian.
    /// For InstanceNorm/AdaIN: full NxN block-diagonal Jacobian (zeros in
    /// off-diagonal blocks — the block structure is encoded in the returned
    /// matrix, not in the iteration).
    fn jacobian(&self, x: &Array1<f32>) -> Result<Array2<f32>>;

    /// CROWN linearization mode.
    fn crown_mode(&self) -> LayerNormCrownMode;
}
