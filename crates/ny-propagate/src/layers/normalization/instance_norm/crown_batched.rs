// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward propagation for InstanceNorm1d.
//!
//! `IbpValidated` routes through the shared decomposed primitive-chain helper,
//! treating each channel as a LayerNorm over the time axis. `Sampling`
//! remains the explicit opt-in heuristic path via
//! [`super::super::crown_batched_common`].
//!
//! Reference: designs/2026-03-14-instance-norm-decomposed-crown-backward.md

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::types::InstanceNorm1dLayer;
use crate::layers::normalization::crown_batched_common::{
    gate_crown_mode_batched, sampling_crown_batched,
};
use crate::layers::normalization::decomposed::decomposed_instance_norm_crown_backward;
use crate::layers::normalization::LayerNormCrownMode;
use crate::BatchedLinearBounds;

impl InstanceNorm1dLayer {
    /// Batched CROWN backward propagation through InstanceNorm1d with pre-activation bounds.
    ///
    /// `IbpValidated` accepts either expanded `[...batch, C, T]` or flattened
    /// `[...batch, C*T]` pre-activation bounds; the decomposed helper reshapes
    /// to per-channel blocks internally. `Sampling` keeps the legacy flattened
    /// delegate path.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("InstanceNorm1d layer batched CROWN backward propagation");

        // Mode gating (Sound → error, Cut → identity, Sampling → proceed)
        if let Some(identity_bounds) = gate_crown_mode_batched(self, bounds)? {
            return Ok(identity_bounds);
        }

        if self.crown_mode == LayerNormCrownMode::IbpValidated {
            let num_channels = self.num_channels();
            if num_channels == 0 {
                return Err(NyError::InvalidSpec(
                    "InstanceNorm1d batched CROWN requires at least one channel".to_string(),
                ));
            }
            return Ok(decomposed_instance_norm_crown_backward(
                bounds,
                &self.ny,
                &self.beta,
                self.eps,
                pre_activation,
                self.forward_mode,
                num_channels,
            )?
            .bounds);
        }

        // Delegate to shared batched CROWN
        sampling_crown_batched("instance_norm", bounds, pre_activation, |b, pa| {
            self.propagate_linear_with_bounds(b, pa)
        })
    }
}
