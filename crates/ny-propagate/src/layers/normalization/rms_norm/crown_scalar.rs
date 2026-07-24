// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Scalar CROWN backward propagation for RMSNorm.
//!
//! `IbpValidated` routes through the shared decomposed primitive-chain helper,
//! matching the alpha-beta-CROWN-style RmsNorm decomposition. `Sampling`
//! remains the explicit opt-in heuristic path and still delegates to the
//! shared sampling-based CROWN linearization in [`super::super::crown_common`].
//!
//! Reference: designs/2026-03-14-rmsnorm-shared-decomposed-crown.md

use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, warn};

use super::types::RmsNormLayer;
use crate::layers::normalization::crown_common::{flatten_preactivation, sampling_crown_scalar};
use crate::layers::normalization::decomposed::{
    batched_bounds_to_scalar, batched_bounds_to_scalar_multi_dim,
    decomposed_rms_norm_crown_backward, decomposed_rms_norm_crown_backward_with_override,
    scalar_bounds_to_batched, scalar_bounds_to_batched_multi_dim, InvRmsOverride,
};
use crate::layers::normalization::layer_norm::types::LayerNormCrownMode;
use crate::LinearBounds;

impl RmsNormLayer {
    fn maybe_propagate_multi_dim_via_batched_inv_rms(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        inv_rms_override: Option<&[Option<(f32, f32)>]>,
    ) -> Result<Option<LinearBounds>> {
        let pre_shape = pre_activation.shape();
        if pre_shape.len() <= 1 {
            return Ok(None);
        }

        let norm_size = *pre_shape.last().ok_or_else(|| {
            NyError::InvalidSpec("RMSNorm multi-dim dispatch: empty pre-activation shape".into())
        })?;
        if norm_size == 0 || self.ny.len() != norm_size {
            return Ok(None);
        }

        let batch_size =
            checked_shape_product(&pre_shape[..pre_shape.len() - 1]).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "RMSNorm multi-dim dispatch: batch shape {:?} overflows usize",
                    &pre_shape[..pre_shape.len() - 1]
                ))
            })?;
        if batch_size <= 1 {
            return Ok(None);
        }

        let expected_inputs = batch_size.checked_mul(norm_size).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "RMSNorm multi-dim dispatch overflow: batch_size={batch_size}, \
                 norm_size={norm_size}"
            ))
        })?;
        if bounds.num_inputs() != expected_inputs {
            return Err(NyError::shape_mismatch(
                vec![expected_inputs],
                vec![bounds.num_inputs()],
            ));
        }

        debug!(
            "RMSNorm scalar CROWN: reshaping {:?} into batched path with \
             batch_size={}, norm_size={}",
            pre_shape, batch_size, norm_size
        );
        let reshaped_pre = pre_activation.reshape(&[batch_size, norm_size])?;
        let batched_bounds = scalar_bounds_to_batched_multi_dim(bounds, batch_size, norm_size)?;
        let result = self.propagate_linear_batched_with_bounds_inv_rms(
            &batched_bounds,
            &reshaped_pre,
            inv_rms_override,
        )?;
        Ok(Some(batched_bounds_to_scalar_multi_dim(
            &result,
            bounds.lower_b(),
            bounds.upper_b(),
        )?))
    }

    /// Compute CROWN linear bounds for RMSNorm with pre-activation bounds.
    ///
    /// Behavior depends on `crown_mode`:
    /// - `IbpValidated` (default): shared decomposed primitive-chain CROWN
    ///   matching alpha-beta-CROWN decomposition (sound)
    /// - `Sound`: Returns error
    /// - `Cut`: Returns identity relaxation
    /// - `Sampling`: Heuristic sampling-based linearization (NOT provably sound)
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        self.propagate_linear_with_bounds_inv_rms(bounds, pre_activation, None)
    }

    /// [`Self::propagate_linear_with_bounds`] with an optional GenBaB `inv_rms`
    /// range override (#norm-genbab). `None` reproduces the un-narrowed path
    /// exactly; `Some((lo, hi))` narrows the `inv_rms` interval used by the
    /// `IbpValidated` decomposed CROWN for the requesting child subdomain.
    pub fn propagate_linear_with_bounds_inv_rms(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        inv_rms_override: Option<&[Option<(f32, f32)>]>,
    ) -> Result<LinearBounds> {
        match self.crown_mode {
            LayerNormCrownMode::Sound => {
                return Err(NyError::SoundnessRefusal(
                    "RMSNorm CROWN refused in Sound mode. \
                     `IbpValidated` uses decomposed primitive-chain CROWN; \
                     use `Sampling` for the heuristic Jacobian path."
                        .to_string(),
                ));
            }
            LayerNormCrownMode::Cut => return Ok(bounds.clone()),
            LayerNormCrownMode::Sampling => {
                warn!("RMSNorm using sampling-based CROWN linearization (not provably sound)")
            }
            LayerNormCrownMode::IbpValidated => {}
        }

        if let Some(result) = self.maybe_propagate_multi_dim_via_batched_inv_rms(
            bounds,
            pre_activation,
            inv_rms_override,
        )? {
            return Ok(result);
        }

        // Flatten pre-activation bounds to 1D
        let (pre_lower, pre_upper) = flatten_preactivation(pre_activation)?;
        let num_neurons = pre_lower.len();

        // Shape validation (RmsNorm-specific: ny only, no beta)
        if self.ny.len() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.ny.len()],
                got: vec![num_neurons],
            });
        }
        if bounds.num_inputs() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        // IbpValidated: route through decomposed primitive-chain CROWN backward
        // (same decomposed path as block-wise graph CROWN). Part of #3821.
        if self.crown_mode == LayerNormCrownMode::IbpValidated {
            let flattened_bounds = BoundedTensor::new(pre_lower.into_dyn(), pre_upper.into_dyn())?;
            let batched_bounds = scalar_bounds_to_batched(bounds)?;
            // Flattened scalar path => exactly one normalization group (b=0).
            let override_for_group = inv_rms_override
                .and_then(|w| w.first().copied().flatten())
                .map(|(lo, hi)| InvRmsOverride::single_group(0, lo, hi));
            let decomposed = match override_for_group {
                Some(ov) => decomposed_rms_norm_crown_backward_with_override(
                    &batched_bounds,
                    &self.ny,
                    self.eps,
                    &flattened_bounds,
                    Some(ov),
                )?,
                None => decomposed_rms_norm_crown_backward(
                    &batched_bounds,
                    &self.ny,
                    self.eps,
                    &flattened_bounds,
                )?,
            };
            return batched_bounds_to_scalar(&decomposed.bounds);
        }

        // Sampling: heuristic Jacobian-based linearization (no IBP validation)
        sampling_crown_scalar(self, bounds, &pre_lower, &pre_upper)
    }
}
