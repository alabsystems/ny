// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Scalar CROWN backward propagation for AdaIN1d.
//!
//! Delegates to the shared sampling-based CROWN linearization in
//! [`super::super::crown_common`] via the `NormLayer` trait.
//!
//! Previously 305 lines of duplicated code; now delegates to shared
//! infrastructure, keeping only shape validation (layer-specific).
//!
//! Reference: designs/2026-02-27-normalization-trait-dedup.md

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::types::AdaIN1dLayer;
use crate::layers::normalization::crown_common::{
    flatten_preactivation, gate_crown_mode, sampling_crown_scalar,
};
use crate::layers::normalization::layer_norm::types::LayerNormCrownMode;
use crate::layers::normalization::trait_norm::NormLayer;
use crate::LinearBounds;

impl AdaIN1dLayer {
    /// Compute CROWN linear bounds for AdaIN1d with pre-activation bounds.
    ///
    /// Follows the same pattern as InstanceNorm1d CROWN but uses the AdaIN
    /// Jacobian (= style_gamma * InstanceNorm Jacobian) and eval.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        // Mode gating (Sound → error, Cut → identity, IbpValidated/Sampling → proceed)
        if let Some(identity_bounds) = gate_crown_mode(self, bounds)? {
            return Ok(identity_bounds);
        }

        if self.crown_mode() == LayerNormCrownMode::IbpValidated {
            return self
                .effective_instance_norm()?
                .propagate_linear_with_bounds(bounds, pre_activation);
        }

        // Flatten pre-activation bounds to 1D (C*T)
        let (pre_lower, pre_upper) = flatten_preactivation(pre_activation)?;
        let total_neurons = pre_lower.len();

        // Shape validation (AdaIN-specific: channel divisibility)
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

        // Sampling stays on the legacy heuristic CROWN route.
        sampling_crown_scalar(self, bounds, &pre_lower, &pre_upper)
    }
}
