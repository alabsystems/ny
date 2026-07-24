// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Scalar CROWN backward propagation for InstanceNorm1d.
//!
//! In `IbpValidated` mode, delegates to the decomposed per-channel CROWN
//! path (`decomposed_instance_norm_crown_backward`) via a scalar-to-batched
//! bridge. In `Sampling` mode, delegates to the shared sampling-based CROWN
//! linearization in [`super::super::crown_common`] via the `NormLayer` trait.
//!
//! Reference: designs/2026-03-14-instance-norm-decomposed-crown-backward.md

use ndarray::{Ix1, Ix2};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::types::InstanceNorm1dLayer;
use crate::bounds::BatchedLinearBounds;
use crate::layers::normalization::crown_common::{
    flatten_preactivation, gate_crown_mode, sampling_crown_scalar,
};
use crate::layers::normalization::decomposed::decomposed_instance_norm_crown_backward;
use crate::layers::normalization::layer_norm::types::LayerNormCrownMode;
use crate::LinearBounds;

impl InstanceNorm1dLayer {
    /// Compute CROWN linear bounds for InstanceNorm1d with pre-activation bounds.
    ///
    /// Input is flattened to `[C*T]`. The Jacobian is block-diagonal with
    /// C blocks of size T×T, one per channel.
    ///
    /// Behavior depends on `crown_mode`:
    /// - `IbpValidated` (default): Decomposed per-channel CROWN (sound). Part of #3830.
    /// - `Sound`: Returns error
    /// - `Cut`: Returns identity relaxation
    /// - `Sampling`: Heuristic sampling-based linearization
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        // Mode gating (Sound → error, Cut → identity, IbpValidated/Sampling → proceed)
        if let Some(identity_bounds) = gate_crown_mode(self, bounds)? {
            return Ok(identity_bounds);
        }

        // Flatten pre-activation bounds to 1D (C*T)
        let (pre_lower, pre_upper) = flatten_preactivation(pre_activation)?;
        let total_neurons = pre_lower.len();

        // Shape validation (InstanceNorm-specific: channel divisibility)
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

        // IbpValidated: use decomposed per-channel CROWN via scalar-to-batched bridge.
        // This reuses the LayerNorm decomposed infrastructure per channel. Part of #3830.
        if self.crown_mode == LayerNormCrownMode::IbpValidated {
            let time_len = total_neurons / num_channels;
            // Reshape to [C, T] — decomposed_instance_norm_crown_backward requires ≥2D input
            let channel_bounds = BoundedTensor::new(
                pre_lower
                    .into_shape_with_order((num_channels, time_len))
                    .map_err(|e| NyError::InvalidSpec(format!("reshape pre_lower to [C, T]: {e}")))?
                    .into_dyn(),
                pre_upper
                    .into_shape_with_order((num_channels, time_len))
                    .map_err(|e| NyError::InvalidSpec(format!("reshape pre_upper to [C, T]: {e}")))?
                    .into_dyn(),
            )?;
            let batched_bounds = scalar_bounds_to_batched(bounds)?;
            let decomposed = decomposed_instance_norm_crown_backward(
                &batched_bounds,
                &self.ny,
                &self.beta,
                self.eps,
                &channel_bounds,
                self.forward_mode,
                num_channels,
            )?;
            return batched_bounds_to_scalar(&decomposed.bounds);
        }

        // Sampling mode: delegate to shared sampling CROWN (no IBP validation, #3775).
        sampling_crown_scalar(self, bounds, &pre_lower, &pre_upper)
    }
}

/// Convert scalar `LinearBounds` (2D) to `BatchedLinearBounds` (dynamic).
///
/// Local bridge helper, same pattern as `layer_norm/crown_scalar.rs`.
/// If #3821 extracts these to `decomposed/common.rs`, import from there.
fn scalar_bounds_to_batched(bounds: &LinearBounds) -> Result<BatchedLinearBounds> {
    BatchedLinearBounds::new(
        bounds.lower_a().clone().into_dyn(),
        bounds.lower_b().clone().into_dyn(),
        bounds.upper_a().clone().into_dyn(),
        bounds.upper_b().clone().into_dyn(),
        vec![bounds.num_inputs()],
        vec![bounds.num_outputs()],
    )
}

/// Convert `BatchedLinearBounds` back to scalar `LinearBounds`.
///
/// Local bridge helper, same pattern as `layer_norm/crown_scalar.rs`.
fn batched_bounds_to_scalar(bounds: &BatchedLinearBounds) -> Result<LinearBounds> {
    let lower_a = bounds
        .lower_a()
        .clone()
        .into_dimensionality::<Ix2>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 2D lower_a from decomposed InstanceNorm, got {:?}",
                bounds.lower_a().shape()
            ))
        })?;
    let lower_b = bounds
        .lower_b()
        .clone()
        .into_dimensionality::<Ix1>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 1D lower_b from decomposed InstanceNorm, got {:?}",
                bounds.lower_b().shape()
            ))
        })?;
    let upper_a = bounds
        .upper_a()
        .clone()
        .into_dimensionality::<Ix2>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 2D upper_a from decomposed InstanceNorm, got {:?}",
                bounds.upper_a().shape()
            ))
        })?;
    let upper_b = bounds
        .upper_b()
        .clone()
        .into_dimensionality::<Ix1>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 1D upper_b from decomposed InstanceNorm, got {:?}",
                bounds.upper_b().shape()
            ))
        })?;

    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
}
