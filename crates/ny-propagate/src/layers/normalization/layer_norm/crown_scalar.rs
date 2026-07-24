// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Scalar CROWN backward propagation for LayerNorm.
//!
//! `IbpValidated` routes through the shared decomposed primitive-chain helper,
//! matching the alpha-beta-CROWN-style LayerNorm decomposition. `Sampling`
//! uses the LayerNorm-specific low-rank Jacobian helper in
//! [`super::sampling_low_rank`], exploiting the closed-form structure
//! `J[i,j] = ny_i/std * [δ_ij - 1/n - z_i*z_j/n]` for O(n) backward
//! instead of materializing the full n×n matrix.
//!
//! Keeps LayerNorm-specific logic: MeanOnly early-return and the analytical
//! MeanOnly backward pass (no sampling needed since mean subtraction is linear).
//!
//! Reference: designs/2026-03-14-issue-2077-layernorm-shared-decomposed-crown.md
//! Reference: designs/2026-03-17-issue-1957-layernorm-sampling-low-rank.md

use ndarray::Array2;
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::{debug, warn};

use super::sampling_low_rank::sampling_crown_scalar_low_rank;
use super::types::{LayerNormCrownMode, LayerNormLayer, LayerNormMode};
use crate::layers::normalization::crown_common::flatten_preactivation;
use crate::layers::normalization::decomposed::{
    batched_bounds_to_scalar, batched_bounds_to_scalar_multi_dim, decomposed_norm_crown_backward,
    scalar_bounds_to_batched, scalar_bounds_to_batched_multi_dim,
};
use crate::LinearBounds;

impl LayerNormLayer {
    fn maybe_propagate_multi_dim_via_batched(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<Option<LinearBounds>> {
        let pre_shape = pre_activation.shape();
        if pre_shape.len() <= 1 {
            return Ok(None);
        }

        let norm_size = *pre_shape.last().ok_or_else(|| {
            NyError::InvalidSpec("LayerNorm multi-dim dispatch: empty pre-activation shape".into())
        })?;
        if norm_size == 0 || self.ny.len() != norm_size {
            return Ok(None);
        }

        let batch_size =
            checked_shape_product(&pre_shape[..pre_shape.len() - 1]).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "LayerNorm multi-dim dispatch: batch shape {:?} overflows usize",
                    &pre_shape[..pre_shape.len() - 1]
                ))
            })?;
        if batch_size <= 1 {
            return Ok(None);
        }

        let expected_inputs = batch_size.checked_mul(norm_size).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "LayerNorm multi-dim dispatch overflow: batch_size={batch_size}, \
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
            "LayerNorm scalar CROWN: reshaping {:?} into batched path with \
             batch_size={}, norm_size={}",
            pre_shape, batch_size, norm_size
        );
        let reshaped_pre = pre_activation.reshape(&[batch_size, norm_size])?;
        let batched_bounds = scalar_bounds_to_batched_multi_dim(bounds, batch_size, norm_size)?;
        let result = self.propagate_linear_batched_with_bounds(&batched_bounds, &reshaped_pre)?;
        Ok(Some(batched_bounds_to_scalar_multi_dim(
            &result,
            bounds.lower_b(),
            bounds.upper_b(),
        )?))
    }

    /// Compute CROWN linear bounds for LayerNorm with pre-activation bounds.
    ///
    /// Behavior depends on `crown_mode`:
    /// - `IbpValidated` (default): shared decomposed primitive-chain CROWN with
    ///   fused LayerNorm IBP row fallback (sound)
    /// - `Sound`: Returns error and requires an explicit non-LayerNorm CROWN strategy
    /// - `Cut`: Returns identity relaxation (CROWN uses output interval bounds)
    /// - `Sampling`: Uses heuristic sampling-based linearization (NOT provably sound)
    ///
    /// Returns linear bounds: y_lower >= A_l @ x + b_l, y_upper <= A_u @ x + b_u
    ///
    /// Rank-1 inputs use the scalar path directly. Multi-dimensional inputs
    /// whose last dimension matches `ny.len()` are reshaped into the batched
    /// path so graph CROWN can reuse the existing per-slice normalization logic.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        // MeanOnly has an analytical CROWN backward pass (no sampling needed).
        if self.mode == LayerNormMode::MeanOnly {
            return self.propagate_linear_mean_only(bounds, pre_activation);
        }

        match self.crown_mode {
            LayerNormCrownMode::Sound => {
                return Err(NyError::SoundnessRefusal(
                    "LayerNorm CROWN refused in Sound mode. \
                     `IbpValidated` uses decomposed primitive-chain CROWN with fused-IBP row fallback; \
                     use `Sampling` for the heuristic Jacobian path."
                        .to_string(),
                ));
            }
            LayerNormCrownMode::Cut => return Ok(bounds.clone()),
            LayerNormCrownMode::Sampling => {
                warn!("LayerNorm using sampling-based CROWN linearization (not provably sound)")
            }
            LayerNormCrownMode::IbpValidated => {}
        }

        if let Some(result) = self.maybe_propagate_multi_dim_via_batched(bounds, pre_activation)? {
            return Ok(result);
        }

        let (pre_lower, pre_upper) = flatten_preactivation(pre_activation)?;
        let num_neurons = pre_lower.len();

        if self.ny.len() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.ny.len()],
                got: vec![num_neurons],
            });
        }
        if self.beta.len() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.beta.len()],
                got: vec![num_neurons],
            });
        }
        if bounds.num_inputs() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        if self.crown_mode == LayerNormCrownMode::IbpValidated {
            let flattened_bounds = BoundedTensor::new(pre_lower.into_dyn(), pre_upper.into_dyn())?;
            let batched_bounds = scalar_bounds_to_batched(bounds)?;
            let decomposed = decomposed_norm_crown_backward(
                &batched_bounds,
                &self.ny,
                &self.beta,
                self.eps,
                &flattened_bounds,
                self.forward_mode,
            )?;
            return batched_bounds_to_scalar(&decomposed.bounds);
        }

        sampling_crown_scalar_low_rank(self, bounds, &pre_lower, &pre_upper)
    }

    /// Analytical CROWN backward pass for MeanOnly LayerNorm.
    ///
    /// MeanOnly subtracts the mean but skips variance normalization, so
    /// `y_i = ny_i * (x_i - mean(x)) + beta_i`. This is a linear function
    /// and can be propagated exactly without sampling.
    fn propagate_linear_mean_only(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        if let Some(result) = self.maybe_propagate_multi_dim_via_batched(bounds, pre_activation)? {
            return Ok(result);
        }

        let pre_flat = pre_activation.flatten();
        let num_neurons = pre_flat.len();

        if self.ny.len() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.ny.len()],
                got: vec![num_neurons],
            });
        }
        if self.beta.len() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.beta.len()],
                got: vec![num_neurons],
            });
        }
        if bounds.num_inputs() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        if num_neurons > (1 << 24) {
            // f32 precision guard (#2136)
            return Err(NyError::InternalError(format!(
                "LayerNorm dimension {num_neurons} exceeds f32 exact integer range"
            )));
        }
        let inv_n = 1.0_f64 / num_neurons as f64;

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
        let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

        for out_row in 0..num_outputs {
            // Accumulate sums and bias in f64 to prevent catastrophic cancellation (#2175).
            let mut lower_sum = 0.0_f64;
            let mut upper_sum = 0.0_f64;
            let mut lower_bias = 0.0_f64;
            let mut upper_bias = 0.0_f64;

            for i in 0..num_neurons {
                let g = self.ny[i] as f64;
                lower_sum += bounds.lower_a()[[out_row, i]] as f64 * g;
                upper_sum += bounds.upper_a()[[out_row, i]] as f64 * g;
                lower_bias += bounds.lower_a()[[out_row, i]] as f64 * self.beta[i] as f64;
                upper_bias += bounds.upper_a()[[out_row, i]] as f64 * self.beta[i] as f64;
            }

            let lower_mean_term = lower_sum * inv_n;
            let upper_mean_term = upper_sum * inv_n;

            // Track whether any coefficient in this row becomes NaN from Inf-Inf
            // cancellation (#3027). If so, the entire row must become conservative.
            let mut row_has_nan = false;
            for j in 0..num_neurons {
                let g = self.ny[j] as f64;
                // Standard f64→f32 rounding for A-matrix (round-to-nearest), matching
                // alpha-beta-CROWN. Directed rounding on A is not unconditionally sound (#2208).
                let la = bounds.lower_a()[[out_row, j]] as f64 * g - lower_mean_term;
                let ua = bounds.upper_a()[[out_row, j]] as f64 * g - upper_mean_term;
                if la.is_nan() || ua.is_nan() {
                    row_has_nan = true;
                    break;
                }
                new_lower_a[[out_row, j]] = la as f32;
                new_upper_a[[out_row, j]] = ua as f32;
            }

            // NaN guard (#3027): when Inf coefficients from compose() NaN→Inf
            // fallback produce Inf*ny - Inf*ny/N = Inf - Inf = NaN, widen
            // the entire row to conservative bounds. LinearBounds::new() rejects
            // non-finite A-coefficients, so we zero out coefficients and push
            // the vacuousness into the bias (±Inf).
            if row_has_nan {
                for j in 0..num_neurons {
                    new_lower_a[[out_row, j]] = 0.0;
                    new_upper_a[[out_row, j]] = 0.0;
                }
                new_lower_b_f64[out_row] = f64::NEG_INFINITY;
                new_upper_b_f64[out_row] = f64::INFINITY;
            } else {
                new_lower_b_f64[out_row] += lower_bias;
                new_upper_b_f64[out_row] += upper_bias;
            }
        }

        LinearBounds::new_or_conservative(
            new_lower_a,
            new_lower_b_f64.mapv(|x| next_down_f32(x as f32)),
            new_upper_a,
            new_upper_b_f64.mapv(|x| next_up_f32(x as f32)),
        )
    }
}
