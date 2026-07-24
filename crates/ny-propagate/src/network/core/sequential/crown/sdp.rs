// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SDP-CROWN ℓ2 ball propagation for Linear/ReLU networks.
//!
//! Extracted from `crown.rs` as part of #4233 Packet C.

use crate::bounds::LinearBounds;
use crate::layers::{BoundPropagation, Layer};
use crate::network::core::Network;
use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use tracing::debug;

use super::bounds_validation::has_degraded_bounds;

/// Validate that all layers are Linear or ReLU (SDP-CROWN requirement).
///
/// Returns an error naming the first unsupported layer.
fn validate_linear_relu_only(layers: &[Layer]) -> Result<()> {
    for (i, layer) in layers.iter().enumerate() {
        match layer {
            Layer::Linear(_) | Layer::ReLU(_) => {}
            other => {
                return Err(NyError::InvalidSpec(format!(
                    "SDP-CROWN currently supports Linear/ReLU networks only (saw {:?} at layer {})",
                    other, i
                )));
            }
        }
    }
    Ok(())
}

impl Network {
    /// Propagate bounds through a Linear/ReLU network using SDP-CROWN offsets for an ℓ2 input set.
    ///
    /// This implements the SDP-CROWN tightening for ReLU layers (arXiv:2506.06665) by:
    /// - Running CROWN-IBP on the ℓ∞ box `x_hat ± rho` (contains the ℓ2 ball) to obtain
    ///   elementwise pre-activation bounds for the usual CROWN slopes.
    /// - Replacing the box relaxation offset at each ReLU with the SDP-CROWN offset valid for
    ///   `||z - z_hat||_2 <= rho_z` at that layer.
    /// - Concretizing the final linear bounds over the input ℓ2 ball.
    ///
    /// Current limitations:
    /// - Only supports sequential networks consisting of `Linear` and `ReLU` layers.
    pub fn propagate_sdp_crown(
        &self,
        input: &BoundedTensor,
        x_hat: &Array1<f32>,
        rho: f32,
    ) -> Result<BoundedTensor> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }
        if !rho.is_finite() {
            return Err(NyError::InvalidSpec(format!(
                "SDP-CROWN: rho must be finite (got {rho})"
            )));
        }
        if rho < 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "SDP-CROWN: rho must be >= 0 (got {rho})"
            )));
        }

        if input.len() != x_hat.len() {
            return Err(NyError::shape_mismatch(
                vec![input.len()],
                vec![x_hat.len()],
            ));
        }
        let (lower, upper) = input.flatten_to_ix1("SDP-CROWN input")?;
        for i in 0..x_hat.len() {
            let l = lower[i];
            let u = upper[i];
            if !(l.is_finite() && u.is_finite()) {
                return Err(NyError::InvalidSpec(
                    "SDP-CROWN requires finite input bounds".to_string(),
                ));
            }
            let xh = x_hat[i];
            if !xh.is_finite() {
                return Err(NyError::InvalidSpec(format!(
                    "SDP-CROWN requires finite x_hat (got {xh} at index {i})"
                )));
            }
            if u < l {
                return Err(NyError::InvalidSpec(format!(
                    "Invalid input bounds at index {i}: [{l}, {u}]"
                )));
            }
            let min = xh - rho;
            let max = xh + rho;
            let tol = 1e-5f32 * min.abs().max(max.abs()).max(1.0);
            if l > min + tol || u < max - tol {
                return Err(NyError::InvalidSpec(
                    "SDP-CROWN requires input bounds to enclose x_hat +/- rho".to_string(),
                ));
            }
        }

        // Validate all layers are Linear/ReLU before proceeding.
        validate_linear_relu_only(&self.layers)?;

        // Step 1: CROWN-IBP forward on the box relaxation (needed for ReLU slopes).
        let layer_bounds = self.collect_crown_ibp_bounds(input)?;
        let output_bounds = layer_bounds
            .last()
            .ok_or_else(|| NyError::InvalidSpec("No layer bounds computed".to_string()))?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        // Step 2: Precompute ℓ2 ball centers/radii for each ReLU pre-activation.
        // Use Lipschitz propagation: ReLU is 1-Lipschitz, Linear scales by spectral norm.
        let mut relu_preactivation: Vec<Option<(Array1<f32>, f32)>> = vec![None; self.layers.len()];

        let mut center = x_hat.clone();
        let mut radius = rho;
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Linear(l) => {
                    let mut next = l.weight.dot(&center);
                    if let Some(b) = &l.bias {
                        next += b;
                    }
                    if next.iter().any(|v| !v.is_finite()) {
                        return Err(NyError::NumericalInstability(format!(
                            "SDP-CROWN center became non-finite at layer {i} (linear)"
                        )));
                    }
                    center = next;
                    radius *= l.spectral_norm();
                }
                Layer::ReLU(_) => {
                    relu_preactivation[i] = Some((center.clone(), radius));
                    // NaN-propagating ReLU: if v is NaN, result stays NaN (#2851).
                    center.mapv_inplace(|v| if v.is_nan() { v } else { v.max(0.0) });
                }
                _ => unreachable!("validated by validate_linear_relu_only"),
            }
        }

        debug!(
            "SDP-CROWN: Starting backward propagation from {} outputs",
            output_dim
        );

        // Step 3: Backward CROWN pass with SDP offsets at ReLUs.
        let mut linear_bounds = LinearBounds::identity(output_dim);
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };
            match layer {
                Layer::Linear(l) => {
                    let next = l.propagate_linear(&linear_bounds)?;
                    if let Cow::Owned(next) = next {
                        linear_bounds = next;
                    }
                }
                Layer::ReLU(r) => {
                    let (z_hat, z_rho) = relu_preactivation[i].as_ref().ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "SDP-CROWN: missing pre-activation ball for ReLU layer {i}"
                        ))
                    })?;
                    linear_bounds = r.propagate_linear_with_bounds_sdp(
                        &linear_bounds,
                        pre_activation,
                        z_hat,
                        *z_rho,
                    )?;
                }
                _ => unreachable!("validated by validate_linear_relu_only"),
            }
        }

        // Step 4: Concretize over input ℓ2 ball.
        let sdp_output = linear_bounds
            .concretize_l2_ball(x_hat, rho)?
            .reshape(&output_shape)?;

        // Guard: concretize_l2_ball repairs inversions to [-inf, +inf] (since W1-893),
        // but non-finite bounds still indicate degraded precision. Fall back to box-IBP.
        if has_degraded_bounds(&sdp_output) {
            debug!("SDP-CROWN: falling back to box-IBP — output contains non-finite bounds");
            Ok(output_bounds.clone())
        } else {
            Ok(sdp_output)
        }
    }
}
