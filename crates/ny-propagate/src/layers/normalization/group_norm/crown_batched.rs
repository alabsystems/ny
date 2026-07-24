// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward propagation for GroupNorm.
//!
//! `IbpValidated` routes through the grouped centered decomposed helper, while
//! `Sampling` remains the explicit heuristic path via
//! [`super::super::crown_batched_common`].
//!
//! Reference: designs/2026-03-15-issue-3914-groupnorm-grouped-centered-crown.md

use ny_core::Result;
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::types::GroupNormLayer;
use crate::layers::normalization::crown_batched_common::{
    gate_crown_mode_batched, sampling_crown_batched,
};
use crate::layers::normalization::decomposed::decomposed_grouped_centered_crown_backward;
use crate::layers::normalization::LayerNormCrownMode;
use crate::BatchedLinearBounds;

impl GroupNormLayer {
    /// Batched CROWN backward propagation through GroupNorm with pre-activation bounds.
    ///
    /// Handles N-D inputs by processing each batch position independently using the 1D
    /// implementation. GroupNorm operates on [C, T] with C*T flattened neurons per batch.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("GroupNorm layer batched CROWN backward propagation");

        // Mode gating
        if let Some(identity_bounds) = gate_crown_mode_batched(self, bounds)? {
            return Ok(identity_bounds);
        }

        if self.crown_mode == LayerNormCrownMode::IbpValidated {
            return Ok(decomposed_grouped_centered_crown_backward(
                bounds,
                &self.ny,
                &self.beta,
                self.eps,
                pre_activation,
                self.forward_mode,
                self.num_channels(),
                self.num_groups,
            )?
            .bounds);
        }

        sampling_crown_batched("group_norm", bounds, pre_activation, |b, pa| {
            self.propagate_linear_with_bounds(b, pa)
        })
    }
}
