// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward propagation for LayerNorm.
//!
//! `IbpValidated` routes through the shared decomposed primitive-chain helper,
//! matching the alpha-beta-CROWN-style LayerNorm decomposition. `Sampling`
//! remains the explicit opt-in heuristic path and still delegates to the
//! shared batched CROWN infrastructure in [`super::super::crown_batched_common`].
//!
//! Keeps LayerNorm-specific logic: MeanOnly early-return and the analytical
//! MeanOnly batched backward pass (no sampling needed since mean subtraction
//! is linear).
//!
//! Reference: designs/2026-03-14-issue-2077-layernorm-shared-decomposed-crown.md

use ndarray::{Array2, Array3};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::{debug, warn};

use super::types::{LayerNormCrownMode, LayerNormLayer, LayerNormMode};
use crate::layers::normalization::crown_batched_common::sampling_crown_batched;
use crate::layers::normalization::decomposed::decomposed_norm_crown_backward;
use crate::BatchedLinearBounds;

impl LayerNormLayer {
    /// Batched CROWN backward propagation through LayerNorm with pre-activation bounds.
    ///
    /// Behavior depends on `crown_mode`:
    /// - `IbpValidated` (default): shared decomposed primitive-chain CROWN with
    ///   fused LayerNorm IBP row fallback (sound)
    /// - `Sound`: Returns error and requires an explicit non-LayerNorm CROWN strategy
    /// - `Cut`: Returns identity relaxation (CROWN uses output interval bounds)
    /// - `Sampling`: Uses heuristic sampling-based linearization (NOT provably sound)
    ///
    /// Handles N-D inputs by processing each batch position independently using the 1D
    /// implementation. LayerNorm operates on the last dimension (norm_size).
    ///
    /// Input shape: [...batch_dims, norm_size]
    /// Bounds shape: [...batch_dims, out_dim, norm_size]
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        // MeanOnly has an analytical CROWN backward pass (no sampling needed).
        if self.mode == LayerNormMode::MeanOnly {
            return self.propagate_linear_batched_mean_only(bounds, pre_activation);
        }

        debug!("LayerNorm layer batched CROWN backward propagation");

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
            LayerNormCrownMode::Sampling => warn!(
                "LayerNorm using sampling-based batched CROWN linearization (not provably sound)"
            ),
            LayerNormCrownMode::IbpValidated => {}
        }

        let a_shape = bounds.lower_a.shape();
        let norm_size = a_shape[a_shape.len() - 1];
        if self.ny.len() != norm_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.ny.len()],
                got: vec![norm_size],
            });
        }
        if self.beta.len() != norm_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.beta.len()],
                got: vec![norm_size],
            });
        }

        if self.crown_mode == LayerNormCrownMode::IbpValidated {
            return Ok(decomposed_norm_crown_backward(
                bounds,
                &self.ny,
                &self.beta,
                self.eps,
                pre_activation,
                self.forward_mode,
            )?
            .bounds);
        }

        sampling_crown_batched("layernorm", bounds, pre_activation, |b, pa| {
            self.propagate_linear_with_bounds(b, pa)
        })
    }

    /// Analytical batched CROWN backward pass for MeanOnly LayerNorm.
    ///
    /// MeanOnly subtracts the mean but skips variance normalization, so
    /// `y_i = ny_i * (x_i - mean(x)) + beta_i`. This is a linear function
    /// and can be propagated exactly without sampling.
    fn propagate_linear_batched_mean_only(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let pre_shape = pre_activation.shape();
        let a_shape = bounds.lower_a.shape();

        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let norm_size = a_shape[a_shape.len() - 1];
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = checked_shape_product(batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "LayerNorm batched CROWN: batch dimensions {batch_dims:?} overflow usize",
                ))
            })?
            .max(1);

        let pre_norm_size = *pre_shape.last().ok_or_else(|| NyError::ShapeMismatch {
            expected: vec![norm_size],
            got: vec![],
        })?;
        if pre_norm_size != norm_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![norm_size],
                got: vec![pre_norm_size],
            });
        }

        if self.ny.len() != norm_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![norm_size],
                got: vec![self.ny.len()],
            });
        }

        let lower_a = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, norm_size))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, norm_size))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;
        let lower_b = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
        let upper_b = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

        let mut new_lower_a = Array3::<f32>::zeros((total_batch, out_dim, norm_size));
        let mut new_upper_a = Array3::<f32>::zeros((total_batch, out_dim, norm_size));
        let mut new_lower_b = Array2::<f32>::zeros((total_batch, out_dim));
        let mut new_upper_b = Array2::<f32>::zeros((total_batch, out_dim));

        if norm_size > (1 << 24) {
            // f32 precision guard (#2136)
            return Err(NyError::InternalError(format!(
                "LayerNorm dimension {norm_size} exceeds f32 exact integer range"
            )));
        }
        let inv_n = 1.0_f64 / norm_size as f64;

        for b in 0..total_batch {
            for j in 0..out_dim {
                // Accumulate sums and bias in f64 to prevent catastrophic cancellation (#2169).
                // Consistent with the scalar path (crown_scalar.rs, fixed by #2164).
                let mut lower_sum = 0.0_f64;
                let mut upper_sum = 0.0_f64;
                let mut lower_bias = 0.0_f64;
                let mut upper_bias = 0.0_f64;

                for i in 0..norm_size {
                    let g = self.ny[i] as f64;
                    lower_sum += lower_a[[b, j, i]] as f64 * g;
                    upper_sum += upper_a[[b, j, i]] as f64 * g;
                    lower_bias += lower_a[[b, j, i]] as f64 * self.beta[i] as f64;
                    upper_bias += upper_a[[b, j, i]] as f64 * self.beta[i] as f64;
                }

                // Mean terms fully in f64 (both sum and inv_n are f64).
                let lower_mean_term = lower_sum * inv_n;
                let upper_mean_term = upper_sum * inv_n;

                // NaN guard (#3027, P1#759): use per-ROW strategy matching scalar path.
                // When Inf coefficients from compose() NaN→Inf fallback contaminate
                // mean_term, non-Inf elements get `finite - Inf = -Inf` which creates
                // lower_a > upper_a inversions. The per-element guard missed this because
                // the inversions aren't NaN — they're non-finite but valid f64 values.
                // Fix: if any coefficient in the row is NaN, zero the entire row and
                // push vacuousness into bias (±Inf).
                let mut row_has_nan = false;
                for i in 0..norm_size {
                    let g = self.ny[i] as f64;
                    // Standard f64→f32 rounding for A-matrix (round-to-nearest), matching
                    // alpha-beta-CROWN. Directed rounding on A is not unconditionally sound (#2208).
                    let la = lower_a[[b, j, i]] as f64 * g - lower_mean_term;
                    let ua = upper_a[[b, j, i]] as f64 * g - upper_mean_term;
                    if la.is_nan() || ua.is_nan() {
                        row_has_nan = true;
                        break;
                    }
                    new_lower_a[[b, j, i]] = la as f32;
                    new_upper_a[[b, j, i]] = ua as f32;
                }

                if row_has_nan {
                    // Zero all A-coefficients for this row and use conservative bias.
                    // BatchedLinearBounds::new() allows ±Inf in bias but coefficient
                    // inversions violate interval_mul_for_bounds preconditions.
                    for i in 0..norm_size {
                        new_lower_a[[b, j, i]] = 0.0;
                        new_upper_a[[b, j, i]] = 0.0;
                    }
                    new_lower_b[[b, j]] = f32::NEG_INFINITY;
                    new_upper_b[[b, j]] = f32::INFINITY;
                } else {
                    // Normal bias accumulation with directed rounding.
                    let lb = (lower_b[[b, j]] as f64 + lower_bias) as f32;
                    let ub = (upper_b[[b, j]] as f64 + upper_bias) as f32;
                    new_lower_b[[b, j]] = if lb.is_nan() {
                        f32::NEG_INFINITY
                    } else {
                        next_down_f32(lb)
                    };
                    new_upper_b[[b, j]] = if ub.is_nan() {
                        f32::INFINITY
                    } else {
                        next_up_f32(ub)
                    };
                }
            }
        }

        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

        let out_a_shape: Vec<usize> = batch_dims
            .iter()
            .copied()
            .chain([out_dim, norm_size])
            .collect();
        let out_b_shape: Vec<usize> = batch_dims.iter().copied().chain([out_dim]).collect();

        // Phase 4 audit: per-LayerNorm reassembly of per-batch validated results.
        BatchedLinearBounds::new_or_conservative(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_a_shape), new_lower_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_b_shape), new_lower_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_a_shape), new_upper_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_b_shape), new_upper_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )
    }
}
