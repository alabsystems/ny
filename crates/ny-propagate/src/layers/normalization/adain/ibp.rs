// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Interval bound propagation for AdaIN1d.
//!
//! AdaIN(x) = style_gamma * InstanceNorm(x) + style_beta
//!
//! IBP strategy:
//! 1. Propagate through InstanceNorm to get [z_lower, z_upper] per element
//! 2. Apply the style affine transform per channel:
//!    - If style_gamma >= 0: y_lower = sg * z_lower + sb, y_upper = sg * z_upper + sb
//!    - If style_gamma < 0:  y_lower = sg * z_upper + sb, y_upper = sg * z_lower + sb
//!
//! This is sound because the style parameters are constants (not data-dependent).

use std::borrow::Cow;

use ny_core::{checked_dim_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::super::math_common::compute_batch_prefix;
use super::types::AdaIN1dLayer;
use crate::layers::common::BoundPropagation;
use crate::{contiguous_flat_slice, contiguous_flat_slice_mut, LinearBounds};

impl AdaIN1dLayer {
    /// Apply the fixed-style affine transform to InstanceNorm output bounds.
    ///
    /// For each channel c and each position:
    ///   y = style_gamma[c] * z + style_beta[c]
    ///
    /// Interval arithmetic with known sign of style_gamma.
    /// Only valid for fixed-style AdaIN.
    fn apply_style_affine(&self, instnorm_bounds: &BoundedTensor) -> Result<BoundedTensor> {
        let style_gamma = self.style_gamma()?;
        let style_beta = self.style_beta()?;
        let shape = instnorm_bounds.shape();
        let ndim = shape.len();

        if ndim < 2 {
            return Err(NyError::InvalidSpec(
                "AdaIN1d requires at least 2D input [C, T]".to_string(),
            ));
        }

        let num_channels = shape[ndim - 2];
        let time_len = shape[ndim - 1];
        let batch_size: usize =
            checked_dim_product(&shape[..ndim - 2], "AdaIN1d IBP batch dimensions")?;

        let mut out_lower = instnorm_bounds.lower().clone();
        let mut out_upper = instnorm_bounds.upper().clone();

        for batch_idx in 0..batch_size.max(1) {
            let batch_prefix = compute_batch_prefix(shape, ndim, batch_idx);

            for c in 0..num_channels {
                let sg = style_gamma[c];
                let sb = style_beta[c];

                for t in 0..time_len {
                    let mut idx = batch_prefix.clone();
                    idx.push(c);
                    idx.push(t);

                    let z_lo = instnorm_bounds.lower()[idx.as_slice()];
                    let z_hi = instnorm_bounds.upper()[idx.as_slice()];

                    // Affine transform with sign analysis.
                    // Directed rounding on final bounds: bare f32 mul+add.
                    // Part of #3344.
                    let (y_lo, y_hi) = if sg >= 0.0 {
                        (next_down_f32(sg * z_lo + sb), next_up_f32(sg * z_hi + sb))
                    } else {
                        (next_down_f32(sg * z_hi + sb), next_up_f32(sg * z_lo + sb))
                    };

                    out_lower[idx.as_slice()] = y_lo;
                    out_upper[idx.as_slice()] = y_hi;
                }
            }
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    /// Ternary IBP for variable-style AdaIN: `y = g * InstanceNorm(x) + b`.
    ///
    /// Requires all three inputs to have identical shapes (activation-shaped).
    /// Uses the 4-corner product hull for `g * z` plus additive `b`.
    ///
    /// Mathematical contract (designs/2026-03-18-issue-4142-packet-a-ny-local-ternary.md):
    /// ```text
    /// z ∈ [z_l, z_u], g ∈ [g_l, g_u], b ∈ [b_l, b_u]
    /// y = g * z + b
    /// ```
    pub fn propagate_ibp_ternary(
        &self,
        x: &BoundedTensor,
        style_gamma: &BoundedTensor,
        style_beta: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        if !self.requires_style_inputs() {
            return Err(NyError::UnsupportedOp(
                "propagate_ibp_ternary called on fixed-style AdaIN1d".to_string(),
            ));
        }

        // Require identical shapes for all three inputs.
        if x.shape() != style_gamma.shape() {
            return Err(NyError::ShapeMismatch {
                expected: x.shape().to_vec(),
                got: style_gamma.shape().to_vec(),
            });
        }
        if x.shape() != style_beta.shape() {
            return Err(NyError::ShapeMismatch {
                expected: x.shape().to_vec(),
                got: style_beta.shape().to_vec(),
            });
        }

        // Step 1: Propagate x through InstanceNorm to get z bounds.
        let z_bounds = self.instance_norm.propagate_ibp(x)?;

        // Step 2: Element-wise product hull between z_bounds and style_gamma,
        //         then add style_beta.
        let z_lo_arr = z_bounds.lower();
        let z_hi_arr = z_bounds.upper();
        let g_lo_arr = style_gamma.lower();
        let g_hi_arr = style_gamma.upper();
        let b_lo_arr = style_beta.lower();
        let b_hi_arr = style_beta.upper();

        let flat_len = z_lo_arr.len();
        let mut out_lower = z_lo_arr.clone();
        let mut out_upper = z_hi_arr.clone();

        let out_lo_slice = contiguous_flat_slice_mut(&mut out_lower)?;
        let out_hi_slice = contiguous_flat_slice_mut(&mut out_upper)?;

        let z_lo = contiguous_flat_slice(z_lo_arr);
        let z_hi = contiguous_flat_slice(z_hi_arr);
        let g_lo = contiguous_flat_slice(g_lo_arr);
        let g_hi = contiguous_flat_slice(g_hi_arr);
        let b_lo = contiguous_flat_slice(b_lo_arr);
        let b_hi = contiguous_flat_slice(b_hi_arr);

        for i in 0..flat_len {
            // 4-corner product hull for g * z.
            let products = [
                g_lo[i] * z_lo[i],
                g_lo[i] * z_hi[i],
                g_hi[i] * z_lo[i],
                g_hi[i] * z_hi[i],
            ];
            let prod_lo = products.iter().copied().fold(f32::INFINITY, f32::min);
            let prod_hi = products.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Add style_beta bounds. Directed rounding (#3344).
            out_lo_slice[i] = next_down_f32(prod_lo + b_lo[i]);
            out_hi_slice[i] = next_up_f32(prod_hi + b_hi[i]);
        }

        BoundedTensor::new(out_lower, out_upper)
    }
}

impl BoundPropagation for AdaIN1dLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Step 1: Propagate through InstanceNorm
        let instnorm_bounds = self.instance_norm.propagate_ibp(input)?;

        // Step 2: Apply style affine transform
        self.apply_style_affine(&instnorm_bounds)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "AdaIN1d is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        AdaIN1dLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}
