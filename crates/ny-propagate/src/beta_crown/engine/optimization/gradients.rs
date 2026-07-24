// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Gradient computation for β and joint α/β/λ optimization.
//!
//! Split into submodules by concern:
//! - `beta_only`: Beta-only gradient computation (test-only path)
//! - `joint`: Joint α/β/λ gradient computation (production path)

#[cfg(test)]
mod beta_only;
mod cut_gradients;
mod joint;

use ny_core::{checked_dim_product, Result};

use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

/// Propagate linear bounds through a layer, returning error on failure.
/// Replaces the `if let Ok(Cow::Owned(...))` pattern that silently swallowed errors.
fn propagate_linear_or_err(
    layer: &dyn BoundPropagation,
    lin_bounds: &mut LinearBounds,
) -> Result<()> {
    match layer.propagate_linear(lin_bounds) {
        Ok(std::borrow::Cow::Owned(new_bounds)) => {
            *lin_bounds = new_bounds;
            Ok(())
        }
        Ok(std::borrow::Cow::Borrowed(_)) => Ok(()), // identity
        Err(e) => Err(e),
    }
}

/// Infer 2D spatial dimensions from a BoundedTensor shape for Conv2d/ConvTranspose2d.
/// Returns `Err(UnsupportedConfiguration)` if shape inference fails.
fn infer_spatial_2d(
    shape: &[usize],
    in_channels: usize,
    layer_name: &str,
    layer_idx: usize,
) -> Result<(usize, usize)> {
    if in_channels == 0 {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "{layer_name} gradient: in_channels must be > 0 (shape={shape:?}) at layer {layer_idx}"
        )));
    }
    if shape.len() >= 3 {
        Ok((shape[shape.len() - 2], shape[shape.len() - 1]))
    } else if shape.len() >= 2 {
        let total: usize = checked_dim_product(
            shape,
            &format!("{layer_name} gradient at layer {layer_idx}"),
        )?;
        if !total.is_multiple_of(in_channels) {
            return Err(ny_core::NyError::UnsupportedConfiguration(format!(
                "{layer_name} gradient: total {total} not divisible by in_channels {in_channels} at layer {layer_idx}"
            )));
        }
        let spatial = total / in_channels;
        // Use integer sqrt to avoid float-to-usize cast (Part of #2983).
        // Manual integer sqrt for MSRV 1.75 compat (isqrt stabilized in 1.84).
        let side = {
            let mut s = (spatial as f64).sqrt() as usize;
            // Refine: f64 sqrt may be off by 1 for large values
            while s.checked_mul(s).map_or(true, |sq| sq > spatial) {
                s -= 1;
            }
            while (s + 1).checked_mul(s + 1).is_some_and(|sq| sq <= spatial) {
                s += 1;
            }
            s
        };
        if side.checked_mul(side) == Some(spatial) {
            Ok((side, side))
        } else {
            Err(ny_core::NyError::UnsupportedConfiguration(format!(
                "{layer_name} gradient: non-square spatial size {spatial} at layer {layer_idx}"
            )))
        }
    } else {
        Err(ny_core::NyError::UnsupportedConfiguration(format!(
            "{layer_name} gradient: shape {shape:?} has <2 dims at layer {layer_idx}"
        )))
    }
}

/// Infer 1D spatial length from a BoundedTensor shape for Conv1d/ConvTranspose1d.
fn infer_spatial_1d(
    shape: &[usize],
    in_channels: usize,
    layer_name: &str,
    layer_idx: usize,
) -> Result<usize> {
    if in_channels == 0 {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "{layer_name} gradient: in_channels must be > 0 (shape={shape:?}) at layer {layer_idx}"
        )));
    }
    if shape.len() >= 2 {
        Ok(shape[shape.len() - 1])
    } else if !shape.is_empty() {
        let total: usize = checked_dim_product(
            shape,
            &format!("{layer_name} gradient at layer {layer_idx}"),
        )?;
        if total.is_multiple_of(in_channels) {
            Ok(total / in_channels)
        } else {
            Err(ny_core::NyError::UnsupportedConfiguration(format!(
                "{layer_name} gradient: total {total} not divisible by in_channels {in_channels} at layer {layer_idx}"
            )))
        }
    } else {
        Err(ny_core::NyError::UnsupportedConfiguration(format!(
            "{layer_name} gradient: empty shape at layer {layer_idx}"
        )))
    }
}
