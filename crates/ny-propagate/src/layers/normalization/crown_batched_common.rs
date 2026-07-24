// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared batched CROWN backward propagation for normalization layers.
//!
//! This module contains the reshape-loop-delegate pattern that was
//! duplicated 4x across LayerNorm, RmsNorm, InstanceNorm1d, and AdaIN1d
//! crown_batched.rs files (~750 lines of identical logic).
//!
//! The scalar CROWN counterpart lives in [`super::crown_common`].
//!
//! # Algorithm overview
//!
//! 1. Mode gating (Sound -> error, Cut -> identity, Sampling -> proceed)
//! 2. Reshape pre-activation and bounds to `[batch, out_dim, in_dim]`
//! 3. Loop over batch positions, delegating each to the 1D implementation
//! 4. Reshape results back to original batch dimensions
//!
//! Reference: designs/2026-02-27-normalization-trait-dedup.md

use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::warn;

use super::layer_norm::types::LayerNormCrownMode;
use super::trait_norm::NormLayer;
use crate::{BatchedLinearBounds, LinearBounds};

/// Gate the CROWN mode for batched bounds (same logic as scalar gating).
///
/// - `IbpValidated`: return `Ok(None)` (caller routes to its sound decomposed
///   primitive-chain CROWN, never to the shared batched sampling path)
/// - `Sound`: return `SoundnessRefusal` error
/// - `Cut`: return `Ok(Some(bounds.clone()))` (identity relaxation)
/// - `Sampling`: return `Ok(None)` (caller proceeds with sampling)
pub(crate) fn gate_crown_mode_batched<L: NormLayer>(
    layer: &L,
    bounds: &BatchedLinearBounds,
) -> Result<Option<BatchedLinearBounds>> {
    match layer.crown_mode() {
        LayerNormCrownMode::IbpValidated => {
            // Every caller intercepts IbpValidated and routes it to its sound
            // decomposed primitive-chain CROWN (#3775); only Sampling mode
            // reaches the shared batched sampling linearization below.
            Ok(None)
        }
        LayerNormCrownMode::Sound => Err(NyError::SoundnessRefusal(format!(
            "{} CROWN linearization uses heuristic sampling (not provably sound). \
             For sound verification, use IBP or cut CROWN at {} boundaries. \
             To proceed with sampling anyway, use the sampling mode.",
            layer.layer_name(),
            layer.layer_name(),
        ))),
        LayerNormCrownMode::Cut => Ok(Some(bounds.clone())),
        LayerNormCrownMode::Sampling => {
            warn!(
                "{} using sampling-based CROWN linearization (not provably sound)",
                layer.layer_name()
            );
            Ok(None)
        }
    }
}

/// Batched CROWN backward propagation, generic over normalization type.
///
/// Reshapes N-D inputs to `[batch, in_dim]`, loops over batch positions
/// delegating each to the 1D `propagate_1d` implementation, then reshapes
/// results back to the original batch dimensions.
///
/// # Parameters
///
/// - `layer_name`: layer name for error messages (e.g. "LayerNorm")
/// - `bounds`: incoming batched CROWN linear bounds
/// - `pre_activation`: pre-activation interval bounds
/// - `propagate_1d`: closure that applies the 1D CROWN backward pass
///   (typically `|b, pa| self.propagate_linear_with_bounds(b, pa)`)
pub(crate) fn sampling_crown_batched(
    layer_name: &str,
    bounds: &BatchedLinearBounds,
    pre_activation: &BoundedTensor,
    propagate_1d: impl Fn(&LinearBounds, &BoundedTensor) -> Result<LinearBounds>,
) -> Result<BatchedLinearBounds> {
    let pre_shape = pre_activation.shape();
    let a_shape = bounds.lower_a.shape();

    if a_shape.len() < 2 {
        return Err(NyError::InvalidSpec(
            "BatchedLinearBounds must have at least 2 dimensions".to_string(),
        ));
    }

    let out_dim = a_shape[a_shape.len() - 2];
    let in_dim = a_shape[a_shape.len() - 1];
    let batch_dims = &a_shape[..a_shape.len() - 2];
    let total_batch: usize = checked_shape_product(batch_dims)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "normalization batched CROWN: batch dimensions {batch_dims:?} overflow usize",
            ))
        })?
        .max(1);

    let pre_in_dim = *pre_shape.last().ok_or_else(|| NyError::ShapeMismatch {
        expected: vec![in_dim],
        got: vec![],
    })?;
    if pre_in_dim != in_dim {
        return Err(NyError::ShapeMismatch {
            expected: vec![in_dim],
            got: vec![pre_in_dim],
        });
    }

    // Reshape pre-activation to [batch, in_dim]
    let pre_lower_flat = pre_activation
        .lower()
        .view()
        .into_shape_with_order((total_batch, in_dim))
        .map_err(|_| NyError::InvalidSpec(format!("Cannot reshape pre_lower for {layer_name}")))?;
    let pre_upper_flat = pre_activation
        .upper()
        .view()
        .into_shape_with_order((total_batch, in_dim))
        .map_err(|_| NyError::InvalidSpec(format!("Cannot reshape pre_upper for {layer_name}")))?;

    // Reshape bounds to [batch, out_dim, in_dim]
    let lower_a_3d = bounds
        .lower_a
        .view()
        .into_shape_with_order((total_batch, out_dim, in_dim))
        .map_err(|_| NyError::InvalidSpec(format!("Cannot reshape lower_a for {layer_name}")))?;
    let upper_a_3d = bounds
        .upper_a
        .view()
        .into_shape_with_order((total_batch, out_dim, in_dim))
        .map_err(|_| NyError::InvalidSpec(format!("Cannot reshape upper_a for {layer_name}")))?;
    let lower_b_2d = bounds
        .lower_b
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|_| NyError::InvalidSpec(format!("Cannot reshape lower_b for {layer_name}")))?;
    let upper_b_2d = bounds
        .upper_b
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|_| NyError::InvalidSpec(format!("Cannot reshape upper_b for {layer_name}")))?;

    // Output arrays
    let mut new_lower_a = Array3::<f32>::zeros((total_batch, out_dim, in_dim));
    let mut new_upper_a = Array3::<f32>::zeros((total_batch, out_dim, in_dim));
    let mut new_lower_b = Array2::<f32>::zeros((total_batch, out_dim));
    let mut new_upper_b = Array2::<f32>::zeros((total_batch, out_dim));

    // Process each batch position independently
    for b in 0..total_batch {
        let pre_lower_1d: Array1<f32> = pre_lower_flat.row(b).to_owned();
        let pre_upper_1d: Array1<f32> = pre_upper_flat.row(b).to_owned();

        let lower_a_slice = lower_a_3d.slice(ndarray::s![b, .., ..]).to_owned();
        let upper_a_slice = upper_a_3d.slice(ndarray::s![b, .., ..]).to_owned();
        let lower_b_slice = lower_b_2d.row(b).to_owned();
        let upper_b_slice = upper_b_2d.row(b).to_owned();

        // CROWN backward NaN firewall (#2812): conservative fallback instead of hard error.
        let batch_bounds = LinearBounds::new_or_conservative(
            lower_a_slice,
            lower_b_slice,
            upper_a_slice,
            upper_b_slice,
        )?;

        let pre_bounds_1d = BoundedTensor::new(pre_lower_1d.into_dyn(), pre_upper_1d.into_dyn())?;

        let result = propagate_1d(&batch_bounds, &pre_bounds_1d)?;

        for j in 0..out_dim {
            for k in 0..in_dim {
                new_lower_a[[b, j, k]] = result.lower_a()[[j, k]];
                new_upper_a[[b, j, k]] = result.upper_a()[[j, k]];
            }
            new_lower_b[[b, j]] = result.lower_b()[j];
            new_upper_b[[b, j]] = result.upper_b()[j];
        }
    }

    // Reshape back to original batch dims
    let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
    let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
    let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
    let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

    let out_a_shape: Vec<usize> = batch_dims
        .iter()
        .copied()
        .chain([out_dim, in_dim])
        .collect();
    let out_b_shape: Vec<usize> = batch_dims.iter().copied().chain([out_dim]).collect();

    BatchedLinearBounds::new_or_conservative(
        ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
        ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
        ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
        ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
        bounds.input_shape.clone(),
        bounds.output_shape.clone(),
    )
}
