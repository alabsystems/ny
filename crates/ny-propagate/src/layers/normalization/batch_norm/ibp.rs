// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP and `BoundPropagation` trait wiring for BatchNorm.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};
use std::borrow::Cow;

use super::math::detect_input_layout;
use super::types::BatchNormLayer;
use crate::bounds::safe_mul_for_bounds;
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

impl BoundPropagation for BatchNormLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let input_shape = input.shape();
        let layout = detect_input_layout(input_shape, self.num_channels, None)?;

        // Create output arrays
        let mut out_lower = ArrayD::zeros(IxDyn(input_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(input_shape));

        // Apply per-channel affine transform
        // y = x * scale + bias
        // If scale > 0: y_l = x_l * scale + bias, y_u = x_u * scale + bias
        // If scale < 0: y_l = x_u * scale + bias, y_u = x_l * scale + bias

        for ((idx, &l), &u) in input.lower().indexed_iter().zip(input.upper().iter()) {
            // Channel index is at the determined position
            let channel_idx = idx[layout.channel_idx];
            let s = self.scale[[channel_idx]];
            let b = self.bias[[channel_idx]];

            // Outward widening for the f32 rounding error baked into the
            // precomputed `scale`/`bias`. The stored coefficients differ from the
            // exact real affine by up to `scale_err`/`bias_err`; over this input
            // element the worst-case effect on the value is
            // `max(|l|,|u|)·scale_err + bias_err`. Folding it outward keeps the
            // bound sound against the *real* batchnorm, not just the f32-rounded
            // affine — a single final next_down/next_up only covers the evaluation
            // rounding, not the ~ulp(scale)·|x| precompute error that can reach
            // several ulps at large magnitudes (#batchnorm-ibp-directed-rounding).
            // safe_mul keeps 0·∞ = 0 for degenerate channels (an ∞ scale_err with
            // a zero-magnitude input contributes nothing; an ∞ with nonzero |x|
            // genuinely widens to ±∞, which is sound).
            let se = self.scale_err[[channel_idx]];
            let be = self.bias_err[[channel_idx]];
            let xmag = l.abs().max(u.abs());
            let widen = safe_mul_for_bounds(xmag, se) + be;

            // Directed rounding on final bounds. safe_mul_for_bounds keeps 0*Inf=0
            // (not NaN) for degenerate var+eps==0 channels (Inf scale): without it,
            // a finite-zero input bound times the Inf scale poisons the bound with
            // NaN, which the downstream new_repaired must then catch. Folding it here
            // keeps NaN out of the arithmetic entirely (#3344, batchnorm-nan).
            if s >= 0.0 {
                out_lower[idx.clone()] = next_down_f32(safe_mul_for_bounds(l, s) + b - widen);
                out_upper[idx] = next_up_f32(safe_mul_for_bounds(u, s) + b + widen);
            } else {
                out_lower[idx.clone()] = next_down_f32(safe_mul_for_bounds(u, s) + b - widen);
                out_upper[idx] = next_up_f32(safe_mul_for_bounds(l, s) + b + widen);
            }
        }

        // Repair non-finite outputs for consistency with linear IBP (#3030).
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // BatchNorm propagate_linear lacks input shape needed for channel-axis heuristic.
        // Use propagate_linear_with_bounds instead - sequential/graph/streaming CROWN all
        // call the shape-aware method directly.
        Err(NyError::UnsupportedOp(
            "BatchNorm propagate_linear needs shape; use propagate_linear_with_bounds (CROWN paths already do)".to_string(),
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
        BatchNormLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}
