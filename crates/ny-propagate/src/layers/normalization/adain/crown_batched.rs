// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward propagation for AdaIN1d.
//!
//! Delegates to the shared batched CROWN infrastructure in
//! [`super::super::crown_batched_common`] via the `NormLayer` trait.
//!
//! Reference: designs/2026-02-27-normalization-trait-dedup.md

use ny_core::Result;
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::types::AdaIN1dLayer;
use crate::layers::normalization::crown_batched_common::{
    gate_crown_mode_batched, sampling_crown_batched,
};
use crate::layers::normalization::trait_norm::NormLayer;
use crate::layers::normalization::LayerNormCrownMode;
use crate::BatchedLinearBounds;

impl AdaIN1dLayer {
    /// Batched CROWN backward propagation through AdaIN1d with pre-activation bounds.
    ///
    /// Handles N-D inputs by processing each batch position independently using the 1D
    /// implementation. AdaIN1d operates on [C, T] with C*T flattened neurons per batch.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("AdaIN1d layer batched CROWN backward propagation");

        // Mode gating (Sound → error, Cut → identity, Sampling → proceed)
        if let Some(identity_bounds) = gate_crown_mode_batched(self, bounds)? {
            return Ok(identity_bounds);
        }

        if self.crown_mode() == LayerNormCrownMode::IbpValidated {
            return self
                .effective_instance_norm()?
                .propagate_linear_batched_with_bounds(bounds, pre_activation);
        }

        // Sampling stays on the legacy heuristic CROWN route.
        sampling_crown_batched("adain", bounds, pre_activation, |b, pa| {
            self.propagate_linear_with_bounds(b, pa)
        })
    }
}
