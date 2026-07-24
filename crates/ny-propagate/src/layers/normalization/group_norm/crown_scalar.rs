// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Scalar CROWN backward propagation for GroupNorm.
//!
//! `IbpValidated` routes through the grouped centered decomposed helper, while
//! `Sampling` remains the explicit heuristic path via
//! [`super::super::crown_common`].
//!
//! Reference: designs/2026-03-15-issue-3914-groupnorm-grouped-centered-crown.md

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::types::GroupNormLayer;
use crate::layers::normalization::crown_common::{
    flatten_preactivation, gate_crown_mode, sampling_crown_scalar,
};
use crate::layers::normalization::decomposed::{
    batched_bounds_to_scalar, decomposed_grouped_centered_crown_backward, scalar_bounds_to_batched,
};
use crate::layers::normalization::layer_norm::types::LayerNormCrownMode;
use crate::LinearBounds;

impl GroupNormLayer {
    /// Compute CROWN linear bounds for GroupNorm with pre-activation bounds.
    ///
    /// Input is flattened to `[C*T]`. The Jacobian is block-diagonal with
    /// G blocks of size (C/G*T)×(C/G*T), one per group.
    ///
    /// Behavior depends on `crown_mode`:
    /// - `IbpValidated` (default): grouped centered decomposed CROWN with fused GroupNorm IBP validation
    /// - `Sound`: Returns error
    /// - `Cut`: Returns identity relaxation
    /// - `Sampling`: Heuristic sampling-based linearization
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        // Mode gating
        if let Some(identity_bounds) = gate_crown_mode(self, bounds)? {
            return Ok(identity_bounds);
        }

        // Flatten pre-activation bounds to 1D (C*T)
        let (pre_lower, pre_upper) = flatten_preactivation(pre_activation)?;
        let total_neurons = pre_lower.len();

        // Shape validation (GroupNorm-specific: channel divisibility)
        let num_channels = self.num_channels();
        if num_channels == 0 || total_neurons % num_channels != 0 {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_channels, total_neurons / num_channels.max(1)],
                got: vec![total_neurons],
            });
        }
        if bounds.num_inputs() != total_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        if self.crown_mode == LayerNormCrownMode::IbpValidated {
            let time_len = total_neurons / num_channels;
            let grouped_bounds = BoundedTensor::new(
                pre_lower
                    .into_shape_with_order((num_channels, time_len))
                    .map_err(|e| {
                        NyError::InvalidSpec(format!("reshape GroupNorm pre_lower to [C, T]: {e}"))
                    })?
                    .into_dyn(),
                pre_upper
                    .into_shape_with_order((num_channels, time_len))
                    .map_err(|e| {
                        NyError::InvalidSpec(format!("reshape GroupNorm pre_upper to [C, T]: {e}"))
                    })?
                    .into_dyn(),
            )?;
            let batched_bounds = scalar_bounds_to_batched(bounds)?;
            let decomposed = decomposed_grouped_centered_crown_backward(
                &batched_bounds,
                &self.ny,
                &self.beta,
                self.eps,
                &grouped_bounds,
                self.forward_mode,
                num_channels,
                self.num_groups,
            )?;
            return batched_bounds_to_scalar(&decomposed.bounds);
        }

        // Sampling mode: delegate to shared sampling CROWN (no IBP validation, #3775).
        sampling_crown_scalar(self, bounds, &pre_lower, &pre_upper)
    }
}
