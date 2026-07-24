// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concretization of batched linear bounds given input intervals.
//!
//! Extracted from `mod.rs` as part of #4212.
//!
//! # Concretization via positive/negative coefficient split (#2220 Packet B)
//!
//! For separate lower/upper bound functions f_L(x) = A_L @ x + b_L and
//! f_U(x) = A_U @ x + b_U, concretization computes:
//!
//!   min_{x ∈ [x_l, x_u]} f_L(x) = A_L_pos @ x_l + A_L_neg @ x_u + b_L
//!   max_{x ∈ [x_l, x_u]} f_U(x) = A_U_pos @ x_u + A_U_neg @ x_l + b_U
//!
//! This is tighter than full interval matvec (which treats [A_L, A_U] as an
//! interval on a single coefficient) and BLAS-acceleratable via ndarray dot.
//!
//! Reference: alpha-beta-CROWN `bound_general.py:1140-1160` uses the same
//! pos/neg split: `lA.clamp(min=0) * x_L + lA.clamp(max=0) * x_U`.

use super::BatchedLinearBounds;
use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};
use std::borrow::Cow;

impl BatchedLinearBounds {
    /// Concretize batched linear bounds given input bounds.
    ///
    /// For linear bounds A @ x + b, with x in [l, u]:
    /// - Lower bound: A_L_pos @ l + A_L_neg @ u + b_L (per position)
    /// - Upper bound: A_U_pos @ u + A_U_neg @ l + b_U (per position)
    ///
    /// Uses the positive/negative coefficient split for exact concretization,
    /// matching the reference (alpha-beta-CROWN `bound_general.py:1140-1160`).
    ///
    /// REQUIRES: `input_bounds.shape() == self.input_shape`, or the input/expected
    /// shape is vector-like (at most one non-1 dimension) with the same element
    /// count so it can be reshaped to `self.input_shape`.
    /// REQUIRES: `input_bounds.lower() <= input_bounds.upper()` element-wise (well-formed intervals).
    /// ENSURES: For all `x` such that `input_bounds.lower() <= x <= input_bounds.upper()`:
    ///   - `result.lower() <= lower_a @ x + lower_b` (sound lower bound),
    ///   - `result.upper() >= upper_a @ x + upper_b` (sound upper bound).
    ///     ENSURES: `result.shape() == self.output_shape`.
    ///
    /// # Errors
    /// - `NyError::ShapeMismatch` if input shape mismatches the expected bounds shape
    /// - `NyError::ShapeMismatch` if coefficients cannot broadcast to the input batch shape
    /// - `NyError::ShapeMismatch` if coefficient, input, or bias shapes are incompatible
    pub fn concretize(&self, input_bounds: &BoundedTensor) -> Result<BoundedTensor> {
        // input_bounds shape: [...batch, in_dim]
        // self.lower_a shape: [...batch, out_dim, in_dim]
        // self.lower_b shape: [...batch, out_dim]
        // output shape: [...batch, out_dim]

        let expected_shape = self.input_shape.as_slice();
        let got_shape = input_bounds.shape();
        let mut in_lower: Cow<ArrayD<f32>> = Cow::Borrowed(input_bounds.lower());
        let mut in_upper: Cow<ArrayD<f32>> = Cow::Borrowed(input_bounds.upper());

        if got_shape != expected_shape {
            let expected_elems = checked_shape_product(expected_shape).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "BatchedLinearBounds: expected shape product overflows: {:?}",
                    expected_shape
                ))
            })?;
            let got_elems = input_bounds.lower().len();
            let is_vector_like = |shape: &[usize]| shape.iter().filter(|&&d| d > 1).count() <= 1;
            let expected_vector_like = is_vector_like(expected_shape);
            let got_vector_like = is_vector_like(got_shape);

            // Allow reshape when either side is vector-like and element counts match.
            let can_reshape =
                expected_elems == got_elems && (expected_vector_like || got_vector_like);
            if can_reshape {
                let reshaped_lower = input_bounds
                    .lower()
                    .clone()
                    .into_shape_with_order(IxDyn(expected_shape))
                    .map_err(|_| {
                        NyError::shape_mismatch(self.input_shape.clone(), got_shape.to_vec())
                    })?;
                let reshaped_upper = input_bounds
                    .upper()
                    .clone()
                    .into_shape_with_order(IxDyn(expected_shape))
                    .map_err(|_| {
                        NyError::shape_mismatch(self.input_shape.clone(), got_shape.to_vec())
                    })?;
                in_lower = Cow::Owned(reshaped_lower);
                in_upper = Cow::Owned(reshaped_upper);
            } else {
                return Err(NyError::shape_mismatch(
                    self.input_shape.clone(),
                    got_shape.to_vec(),
                ));
            }
        }

        // Handle shape reconciliation between coefficient matrix and input.
        let a_shape = self.lower_a.shape();
        let x_shape = in_lower.shape();

        // Flat coefficients case: A is [out_dim, total_in] from attention graph
        // flatten_to_block_diagonal, and input is multi-dim [batch..., dim].
        // Flatten input to [total_in] for flat matvec instead of broadcasting A.
        let a_in_dim = *a_shape.last().unwrap_or(&0);
        let x_elems = in_lower.len();
        let x_last = *x_shape.last().unwrap_or(&0);
        let is_flat_attn =
            a_shape.len() == 2 && x_shape.len() > 1 && a_in_dim == x_elems && a_in_dim != x_last;

        if is_flat_attn {
            in_lower = Cow::Owned(
                in_lower
                    .as_ref()
                    .clone()
                    .into_shape_with_order(IxDyn(&[a_in_dim]))
                    .map_err(|_| {
                        NyError::InvalidSpec(format!(
                            "concretize flat: cannot reshape input lower to [{}]",
                            a_in_dim
                        ))
                    })?,
            );
            in_upper = Cow::Owned(
                in_upper
                    .as_ref()
                    .clone()
                    .into_shape_with_order(IxDyn(&[a_in_dim]))
                    .map_err(|_| {
                        NyError::InvalidSpec(format!(
                            "concretize flat: cannot reshape input upper to [{}]",
                            a_in_dim
                        ))
                    })?,
            );
        }

        let x_shape = in_lower.shape();
        let needs_broadcast = a_shape.len() == 2 && x_shape.len() > 1;

        // Justification: The 4-tuple represents linear bound coefficients (lower_A, upper_A,
        // lower_b, upper_b) that are either borrowed or owned depending on whether broadcasting
        // is needed. A named struct would add indirection for a local destructuring pattern.
        #[allow(clippy::type_complexity)]
        let (lower_a, upper_a, lower_b, upper_b): (
            Cow<ArrayD<f32>>,
            Cow<ArrayD<f32>>,
            Cow<ArrayD<f32>>,
            Cow<ArrayD<f32>>,
        ) = if needs_broadcast {
            // Coefficients are unbatched [out_dim, in_dim], input is batched [...batch, in_dim]
            // Broadcast by inserting leading batch dimensions
            let x_batch_dims = &x_shape[..x_shape.len() - 1];
            let mut new_a_shape: Vec<usize> = x_batch_dims.to_vec();
            new_a_shape.extend_from_slice(a_shape);
            let mut new_b_shape: Vec<usize> = x_batch_dims.to_vec();
            new_b_shape.push(a_shape[0]); // out_dim

            // Broadcast by repeating the unbatched matrices across batch dims
            let lower_a_bc = self
                .lower_a
                .broadcast(IxDyn(&new_a_shape))
                .ok_or_else(|| {
                    NyError::shape_mismatch(new_a_shape.clone(), self.lower_a.shape().to_vec())
                })?
                .to_owned();
            let upper_a_bc = self
                .upper_a
                .broadcast(IxDyn(&new_a_shape))
                .ok_or_else(|| {
                    NyError::shape_mismatch(new_a_shape.clone(), self.upper_a.shape().to_vec())
                })?
                .to_owned();
            let lower_b_bc = self
                .lower_b
                .broadcast(IxDyn(&new_b_shape))
                .ok_or_else(|| {
                    NyError::shape_mismatch(new_b_shape.clone(), self.lower_b.shape().to_vec())
                })?
                .to_owned();
            let upper_b_bc = self
                .upper_b
                .broadcast(IxDyn(&new_b_shape))
                .ok_or_else(|| {
                    NyError::shape_mismatch(new_b_shape.clone(), self.upper_b.shape().to_vec())
                })?
                .to_owned();
            (
                Cow::Owned(lower_a_bc),
                Cow::Owned(upper_a_bc),
                Cow::Owned(lower_b_bc),
                Cow::Owned(upper_b_bc),
            )
        } else {
            // No broadcasting needed - borrow original arrays (no clone!)
            (
                Cow::Borrowed(&self.lower_a),
                Cow::Borrowed(&self.upper_a),
                Cow::Borrowed(&self.lower_b),
                Cow::Borrowed(&self.upper_b),
            )
        };

        // Positive/negative coefficient split (#2220 Packet B).
        //
        // lower_a and upper_a are coefficients of SEPARATE linear bound functions,
        // not interval bounds on the same coefficient. The correct concretization is:
        //   lower = A_L_pos @ x_l + A_L_neg @ x_u + b_L
        //   upper = A_U_pos @ x_u + A_U_neg @ x_l + b_U
        //
        // This is both tighter and faster than the previous interval matvec approach.
        // Reference: alpha-beta-CROWN bound_general.py:1140-1160.
        let (concrete_lower, concrete_upper) = if Self::all_finite_for_blas(
            &lower_a, &upper_a, &lower_b, &upper_b, &in_lower, &in_upper,
        ) {
            Self::concretize_blas_posneg(
                &lower_a, &upper_a, &lower_b, &upper_b, &in_lower, &in_upper,
            )?
        } else {
            Self::concretize_scalar_posneg(
                &lower_a, &upper_a, &lower_b, &upper_b, &in_lower, &in_upper,
            )?
        };

        // Certified coefficient-error penalty (#vnncomp-aw-soundness). The batched
        // `A·W` is f64-accumulated, but that f64 accumulation still rounds; the
        // per-coefficient error `lower_a_err`/`upper_a_err` bounds `|stored - true|`.
        // Apply the SAME `max(|in_l|,|in_u|)`-scaled OUTWARD penalty the scalar
        // path uses at concretize, so the batched (β-CROWN/BaB) verdict path is
        // NOT 1-ULP optimistic. Skipped when there is no error (exact bounds).
        let (concrete_lower, concrete_upper) =
            self.apply_coeff_err_penalty(concrete_lower, concrete_upper, &in_lower, &in_upper)?;

        // Repair NaN/Inf at the type boundary (#3423). Widen strategy replaces NaN
        // with ±inf and fixes inversions.
        BoundedTensor::new_repaired(concrete_lower, concrete_upper, RepairStrategy::Widen)
    }

    /// Concretize batched linear bounds with directed rounding for soundness.
    ///
    /// Calls `concretize`, then applies a final 1-ULP directed rounding via
    /// `round_for_soundness_inplace`.
    ///
    /// Soundness of the underlying `concretize` is established inside each path,
    /// and rests on the SAME property in both: operands are cast f32→f64 so every
    /// product is exact, the dot product accumulates in f64, and the single
    /// f64→f32 cast is absorbed by a directed `next_down`/`next_up` — see
    /// `concretize_blas_posneg` (BLAS) and `concretize_scalar_posneg` (fallback).
    /// The 1 ULP of widening is measured at the RESULT magnitude, so it can only
    /// cover rounding that also happens at the result — which is why neither path
    /// may form its per-term products in f32, where round-to-nearest biases each
    /// term INWARD at the (possibly far larger) term magnitude. Both paths are
    /// therefore sound and equally tight. The extra 1-ULP applied here is strictly
    /// additive widening — it only makes bounds safer.
    ///
    /// NaN/Inf repair is centralized in `concretize` via `new_repaired(Widen)` (#3423).
    /// `round_for_soundness_inplace` (1-ULP widening) cannot introduce NaN or inversions.
    ///
    /// Reference: alpha-beta-CROWN `__double2float_rd`/`__double2float_ru`
    /// (`cuda_kernels.cu:8-21`).
    pub fn concretize_sound(&self, input_bounds: &BoundedTensor) -> Result<BoundedTensor> {
        let mut result = self.concretize(input_bounds)?;
        result.round_for_soundness_inplace();
        Ok(result)
    }
}
