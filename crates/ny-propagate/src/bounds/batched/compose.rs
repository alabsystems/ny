// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Composition of batched linear bounds for CROWN backward propagation.
//!
//! Extracted from `mod.rs` as part of #4212.

use super::BatchedLinearBounds;
use crate::bounds::safe_math::interval_mul_for_bounds;
use ndarray::{Array2, Array3, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};

impl BatchedLinearBounds {
    /// Compose two sets of linear bounds: result = other . self
    ///
    /// If self represents: y = A1 @ x + b1 (maps x -> y)
    /// And other represents: z = A2 @ y + b2 (maps y -> z)
    /// Then the composed result is: z = (A2 @ A1) @ x + (A2 @ b1 + b2)
    ///
    /// This is used for CROWN backward propagation to compose bounds across layers.
    ///
    /// # Arguments
    /// - `other`: The outer linear bounds (maps from self's output to new output)
    ///
    /// # Returns
    /// Composed bounds that map from self's input to other's output
    ///
    /// # Shape requirements
    /// - self.lower_a shape: [...batch, out_dim_1, in_dim_1]
    /// - other.lower_a shape: [...batch, out_dim_2, out_dim_1]
    /// - Result lower_a shape: [...batch, out_dim_2, in_dim_1]
    ///
    /// # SOUNDNESS — VERDICT-SAFE (certified coefficient error)
    ///
    /// Both compose paths are sound for verdict-path use. The BLAS fast path
    /// [`compose_blas`](Self::compose_blas) keeps the f32 SGEMM as the nominal
    /// coefficient and attaches a certified per-coefficient error matrix
    /// `err = |stored − D_f64| + γ_{k+1}^{f64}·S` (the EXACTLY-measured f32-SGEMM-vs-
    /// f64 divergence plus Higham's f64 dot bound; order-independent, so robust to
    /// the opaque BLAS accumulation order), which [`concretize`](Self::concretize)
    /// penalizes OUTWARD. The scalar fallback [`compose_scalar`](Self::compose_scalar)
    /// accumulates each exact f32→f64 product in f64 where the final 1-ULP directed
    /// cast suffices (error 0). Either way the concretized output is a sound
    /// enclosure of the exact real composition.
    pub fn compose(&self, other: &BatchedLinearBounds) -> Result<BatchedLinearBounds> {
        // Validate shape compatibility
        let self_shape = self.lower_a.shape();
        let other_shape = other.lower_a.shape();

        if self_shape.len() < 2 || other_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds::compose requires at least 2D coefficient matrices"
                    .to_string(),
            ));
        }

        let self_out_dim = self_shape[self_shape.len() - 2];
        let self_in_dim = self_shape[self_shape.len() - 1];
        let other_out_dim = other_shape[other_shape.len() - 2];
        let other_in_dim = other_shape[other_shape.len() - 1];

        // other's input dim should match self's output dim
        if other_in_dim != self_out_dim {
            return Err(NyError::shape_mismatch(
                vec![self_out_dim],
                vec![other_in_dim],
            ));
        }

        // Get batch dimensions
        let self_batch = &self_shape[..self_shape.len() - 2];
        let other_batch = &other_shape[..other_shape.len() - 2];

        // For simplicity, require matching batch dimensions
        if self_batch != other_batch {
            return Err(NyError::shape_mismatch(
                self_batch.to_vec(),
                other_batch.to_vec(),
            ));
        }

        let batch_dims = self_batch;
        let batch_size: usize = checked_shape_product(batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "compose: batch dimensions {batch_dims:?} overflow usize",
                ))
            })?
            .max(1);

        // Reshape for batched matrix multiplication
        // A1: [batch_size, out_dim_1, in_dim_1]
        // A2: [batch_size, out_dim_2, out_dim_1]
        // Result: [batch_size, out_dim_2, in_dim_1]
        let a1_lower = self
            .lower_a
            .view()
            .into_shape_with_order((batch_size, self_out_dim, self_in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape A1 for composition".to_string()))?;
        let a1_upper = self
            .upper_a
            .view()
            .into_shape_with_order((batch_size, self_out_dim, self_in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape A1 for composition".to_string()))?;
        let a2_lower = other
            .lower_a
            .view()
            .into_shape_with_order((batch_size, other_out_dim, other_in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape A2 for composition".to_string()))?;
        let a2_upper = other
            .upper_a
            .view()
            .into_shape_with_order((batch_size, other_out_dim, other_in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape A2 for composition".to_string()))?;

        // Bias vectors: [batch_size, out_dim]
        let b1_lower = self
            .lower_b
            .view()
            .into_shape_with_order((batch_size, self_out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape b1 for composition".to_string()))?;
        let b1_upper = self
            .upper_b
            .view()
            .into_shape_with_order((batch_size, self_out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape b1 for composition".to_string()))?;
        let b2_lower = other
            .lower_b
            .view()
            .into_shape_with_order((batch_size, other_out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape b2 for composition".to_string()))?;
        let b2_upper = other
            .upper_b
            .view()
            .into_shape_with_order((batch_size, other_out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape b2 for composition".to_string()))?;

        // Compute composed coefficient matrices: A_composed = A2 @ A1
        // Use interval arithmetic for interval coefficient matrices.
        //
        // Important: when bounds saturate, coefficients/biases may contain +/-inf.
        // Naive arithmetic can introduce NaNs via:
        // - 0 * inf
        // - inf + (-inf)
        //
        // For sound verification we must never propagate NaNs; in ambiguous cases we
        // conservatively widen (e.g., lower=-inf, upper=+inf).
        //
        // Accumulate in f64 to avoid O(n) rounding error in f32 dot products.
        // Each f32 product is exact when promoted to f64 (f32 ⊂ f64), so accumulated
        // error is only from f64 additions (eps_f64 ≈ 1.1e-16, negligible for any
        // practical n). The final f64→f32 cast uses directed rounding (next_down_f32
        // for lower, next_up_f32 for upper) so the result is a sound interval.
        //
        // Without this fix, f32 accumulation introduces O(n·eps_f32) ≈ O(n·6e-8) error,
        // which for n=768 (transformer hidden dim) can be ~5e-5 per element, compounding
        // across layers during backward propagation.
        //
        // Reference: batched_interval_matvec (interval.rs) uses the same f64 pattern
        // per #2214. alpha-beta-CROWN uses f64 intermediates (optimized_bounds.py:82).
        // Fix for #2269.
        //
        // Uses module-level functions: interval_mul_for_bounds (also verified by Kani proofs).
        //
        // Dispatch: BLAS SGEMM when all inputs are finite (#2220 Packet C),
        // scalar interval_mul_for_bounds fallback otherwise. The BLAS path
        // accumulates the coefficient dot in f32 and so returns a certified
        // per-coefficient error (`a_err = |stored − D_f64| + γ·S`); the scalar path
        // accumulates in f64 (exact products, sub-ULP sum) so carries no error.
        let (
            composed_lower_a,
            composed_upper_a,
            composed_lower_b,
            composed_upper_b,
            composed_a_err,
        ) = if Self::all_finite_for_compose(
            &a2_lower, &a2_upper, &a1_lower, &a1_upper, &b1_lower, &b1_upper, &b2_lower, &b2_upper,
        ) {
            let (la, ua, lb, ub, le, ue) = Self::compose_blas(
                &a2_lower,
                &a2_upper,
                &a1_lower,
                &a1_upper,
                &b1_lower,
                &b1_upper,
                &b2_lower,
                &b2_upper,
                batch_size,
                other_out_dim,
                self_in_dim,
            )?;
            (la, ua, lb, ub, Some((le, ue)))
        } else {
            let (la, ua, lb, ub) = Self::compose_scalar(
                &a2_lower,
                &a2_upper,
                &a1_lower,
                &a1_upper,
                &b1_lower,
                &b1_upper,
                &b2_lower,
                &b2_upper,
                batch_size,
                other_out_dim,
                other_in_dim,
                self_in_dim,
            )?;
            (la, ua, lb, ub, None)
        };

        // Reshape back to original batch structure
        let mut output_a_shape: Vec<usize> = batch_dims.to_vec();
        output_a_shape.push(other_out_dim);
        output_a_shape.push(self_in_dim);

        let mut output_b_shape: Vec<usize> = batch_dims.to_vec();
        output_b_shape.push(other_out_dim);

        let (composed_lower_a_vec, _) = composed_lower_a.into_raw_vec_and_offset();
        let (composed_upper_a_vec, _) = composed_upper_a.into_raw_vec_and_offset();
        let (composed_lower_b_vec, _) = composed_lower_b.into_raw_vec_and_offset();
        let (composed_upper_b_vec, _) = composed_upper_b.into_raw_vec_and_offset();

        // KEEP unchecked: compose() already widens any NaN to +/-Inf
        // conservative bounds before these vectors are reshaped.
        let mut result = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::from_shape_vec(IxDyn(&output_a_shape), composed_lower_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape composed lower_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&output_b_shape), composed_lower_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape composed lower_b".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&output_a_shape), composed_upper_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape composed upper_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&output_b_shape), composed_upper_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape composed upper_b".to_string()))?,
            self.input_shape.clone(),
            other.output_shape.clone(),
        );

        // Attach the certified coefficient error (BLAS path only). This is consumed
        // OUTWARD at concretize via `apply_coeff_err_penalty`, making the composed
        // bounds a sound enclosure of the exact real product. The scalar path is
        // f64-exact, so no error is attached.
        if let Some((lower_err, upper_err)) = composed_a_err {
            let (le_vec, _) = lower_err.into_raw_vec_and_offset();
            let (ue_vec, _) = upper_err.into_raw_vec_and_offset();
            let le = ArrayD::from_shape_vec(IxDyn(&output_a_shape), le_vec).map_err(|_| {
                NyError::InvalidSpec("Cannot reshape composed lower_a_err".to_string())
            })?;
            let ue = ArrayD::from_shape_vec(IxDyn(&output_a_shape), ue_vec).map_err(|_| {
                NyError::InvalidSpec("Cannot reshape composed upper_a_err".to_string())
            })?;
            result.set_coeff_err(le, ue);
        }

        Ok(result)
    }

    /// Scalar fallback for compose with full interval multiplication.
    ///
    /// Handles NaN/Inf in coefficients via `interval_mul_for_bounds`.
    /// Accumulates in f64 with directed rounding on final cast.
    fn compose_scalar(
        a2_lower: &ndarray::ArrayView3<f32>,
        a2_upper: &ndarray::ArrayView3<f32>,
        a1_lower: &ndarray::ArrayView3<f32>,
        a1_upper: &ndarray::ArrayView3<f32>,
        b1_lower: &ndarray::ArrayView2<f32>,
        b1_upper: &ndarray::ArrayView2<f32>,
        b2_lower: &ndarray::ArrayView2<f32>,
        b2_upper: &ndarray::ArrayView2<f32>,
        batch_size: usize,
        other_out_dim: usize,
        other_in_dim: usize,
        self_in_dim: usize,
    ) -> Result<(Array3<f32>, Array3<f32>, Array2<f32>, Array2<f32>)> {
        let mut composed_lower_a = Array3::<f32>::zeros((batch_size, other_out_dim, self_in_dim));
        let mut composed_upper_a = Array3::<f32>::zeros((batch_size, other_out_dim, self_in_dim));
        let mut composed_lower_b = Array2::<f32>::zeros((batch_size, other_out_dim));
        let mut composed_upper_b = Array2::<f32>::zeros((batch_size, other_out_dim));

        for b in 0..batch_size {
            for i in 0..other_out_dim {
                for j in 0..self_in_dim {
                    let mut lower_sum = 0.0_f64;
                    let mut upper_sum = 0.0_f64;

                    for k in 0..other_in_dim {
                        let (prod_lower, prod_upper) = interval_mul_for_bounds(
                            a2_lower[[b, i, k]],
                            a2_upper[[b, i, k]],
                            a1_lower[[b, k, j]],
                            a1_upper[[b, k, j]],
                        );
                        lower_sum += prod_lower as f64;
                        upper_sum += prod_upper as f64;
                    }

                    composed_lower_a[[b, i, j]] = if lower_sum.is_nan() {
                        f32::NEG_INFINITY
                    } else {
                        next_down_f32(lower_sum as f32)
                    };
                    composed_upper_a[[b, i, j]] = if upper_sum.is_nan() {
                        f32::INFINITY
                    } else {
                        next_up_f32(upper_sum as f32)
                    };
                }

                let mut bias_lower = b2_lower[[b, i]] as f64;
                let mut bias_upper = b2_upper[[b, i]] as f64;

                for k in 0..other_in_dim {
                    let (prod_lower, prod_upper) = interval_mul_for_bounds(
                        a2_lower[[b, i, k]],
                        a2_upper[[b, i, k]],
                        b1_lower[[b, k]],
                        b1_upper[[b, k]],
                    );
                    bias_lower += prod_lower as f64;
                    bias_upper += prod_upper as f64;
                }

                composed_lower_b[[b, i]] = if bias_lower.is_nan() {
                    f32::NEG_INFINITY
                } else {
                    next_down_f32(bias_lower as f32)
                };
                composed_upper_b[[b, i]] = if bias_upper.is_nan() {
                    f32::INFINITY
                } else {
                    next_up_f32(bias_upper as f32)
                };
            }
        }

        Ok((
            composed_lower_a,
            composed_upper_a,
            composed_lower_b,
            composed_upper_b,
        ))
    }
}
