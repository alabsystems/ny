// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward propagation for RMSNorm.
//!
//! `IbpValidated` routes through the shared decomposed primitive-chain helper,
//! matching the alpha-beta-CROWN-style RmsNorm decomposition. `Sampling`
//! remains the explicit opt-in heuristic path and still delegates to the
//! shared batched CROWN infrastructure in [`super::super::crown_batched_common`].
//!
//! Reference: designs/2026-03-14-rmsnorm-shared-decomposed-crown.md

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, warn};

use super::types::RmsNormLayer;
use crate::layers::normalization::crown_batched_common::sampling_crown_batched;
use crate::layers::normalization::decomposed::{
    decomposed_rms_norm_crown_backward, decomposed_rms_norm_crown_backward_with_override,
    InvRmsOverride,
};
use crate::layers::normalization::layer_norm::types::LayerNormCrownMode;
use crate::BatchedLinearBounds;

impl RmsNormLayer {
    /// Batched CROWN backward propagation through RMSNorm with pre-activation bounds.
    ///
    /// Behavior depends on `crown_mode`:
    /// - `IbpValidated` (default): shared decomposed primitive-chain CROWN
    ///   matching alpha-beta-CROWN decomposition (sound)
    /// - `Sound`: Returns error
    /// - `Cut`: Returns identity relaxation (CROWN uses output interval bounds)
    /// - `Sampling`: Uses heuristic sampling-based linearization (NOT provably sound)
    ///
    /// Handles N-D inputs. RMSNorm operates on the last dimension (norm_size).
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        self.propagate_linear_batched_with_bounds_inv_rms(bounds, pre_activation, None)
    }

    /// [`Self::propagate_linear_batched_with_bounds`] with an optional GenBaB
    /// `inv_rms` range override (#norm-genbab). When `Some`, the decomposed
    /// `IbpValidated` path intersects its IBP-derived `inv_rms` interval with
    /// the window, tightening the reciprocal/sqrt relaxation for the requesting
    /// child subdomain. `None` reproduces the un-narrowed behavior exactly.
    pub fn propagate_linear_batched_with_bounds_inv_rms(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
        inv_rms_override: Option<&[Option<(f32, f32)>]>,
    ) -> Result<BatchedLinearBounds> {
        debug!("RMSNorm layer batched CROWN backward propagation");

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
                warn!(
                    "RMSNorm using sampling-based batched CROWN linearization (not provably sound)"
                )
            }
            LayerNormCrownMode::IbpValidated => {}
        }

        // RmsNorm-specific shape validation: ny must match norm_size
        let a_shape = bounds.lower_a.shape();
        let norm_size = a_shape[a_shape.len() - 1];
        if self.ny.len() != norm_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.ny.len()],
                got: vec![norm_size],
            });
        }

        // IbpValidated: route directly through decomposed primitive-chain CROWN backward.
        // The decomposed path operates on BatchedLinearBounds natively. Part of #3821.
        if self.crown_mode == LayerNormCrownMode::IbpValidated {
            // Per-group windows: batched `b` index == normalization group.
            let override_struct = inv_rms_override.and_then(|windows| {
                if windows.iter().any(|w| w.is_some()) {
                    Some(InvRmsOverride {
                        windows: windows.to_vec(),
                    })
                } else {
                    None
                }
            });
            let result = match override_struct {
                Some(ov) => decomposed_rms_norm_crown_backward_with_override(
                    bounds,
                    &self.ny,
                    self.eps,
                    pre_activation,
                    Some(ov),
                )?,
                None => {
                    decomposed_rms_norm_crown_backward(bounds, &self.ny, self.eps, pre_activation)?
                }
            };
            return Ok(result.bounds);
        }

        // Sampling: heuristic loop-over-batch-positions via scalar path
        sampling_crown_batched("rmsnorm", bounds, pre_activation, |b, pa| {
            self.propagate_linear_with_bounds(b, pa)
        })
    }
}
