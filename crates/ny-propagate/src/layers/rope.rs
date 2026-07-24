// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Rotary Position Embedding (RoPE) layer with IBP and CROWN propagation.
//!
//! RoPE applies a 2D rotation to each pair of consecutive elements:
//!
//! ```text
//! y[2i]   = x[2i] * cos(θ_i) - x[2i+1] * sin(θ_i)
//! y[2i+1] = x[2i] * sin(θ_i) + x[2i+1] * cos(θ_i)
//! ```
//!
//! where `θ_i = position / 10000^(2i/d_model)` are fixed frequencies determined
//! by the sequence position and embedding dimension.
//!
//! For fixed frequencies (the verification case), RoPE is a **linear** operation:
//! a block-diagonal matrix with 2×2 rotation blocks. This means:
//! - IBP is exact (no relaxation error)
//! - CROWN backward is exact linear composition (A_new = A @ R, b_new = b)
//!
//! # Mathematical Reference
//!
//! RoPE was introduced in:
//! - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
//! - Section 3.4.2: the rotation matrix formulation
//!
//! The block-diagonal rotation matrix for each pair i is:
//! ```text
//! R_i = [ cos(θ_i)  -sin(θ_i) ]
//!       [ sin(θ_i)   cos(θ_i) ]
//! ```
//!
//! # Integration
//!
//! Used by Qwen3-TTS attention layers (K6 kernel).
//! An external DSL decomposes single-tensor rotation into: Reshape → AxisSelect → Stack,
//! then applies RoPE element-wise. This layer handles the element-wise rotation.

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;

use crate::layers::common::BoundPropagation;
use crate::{contiguous_flat_slice, BatchedLinearBounds, LinearBounds};

// Loaders deserialize F16/BF16 RoPE weights into f32 before construction.
// Independent half-precision rounding can move cos^2 + sin^2 by ~5.6e-3 for a
// valid angle, so keep a small quantization envelope while still rejecting
// clearly malformed tables like (1.0, 1.0).
const UNIT_ROTATION_TOLERANCE: f64 = 1e-2;

/// Rotary Position Embedding (RoPE) layer.
///
/// Stores precomputed `cos(θ_i)` and `sin(θ_i)` values for each pair position.
/// The input tensor has shape `[..., head_dim]` where `head_dim` is even.
/// Each consecutive pair `(x[2i], x[2i+1])` is rotated by angle `θ_i`.
#[derive(Debug, Clone)]
pub struct RopeLayer {
    /// Precomputed `cos(θ_i)` for each pair position `i = 0..num_pairs`.
    pub(crate) cos_freqs: Vec<f32>,
    /// Precomputed `sin(θ_i)` for each pair position `i = 0..num_pairs`.
    pub(crate) sin_freqs: Vec<f32>,
}

impl RopeLayer {
    /// Create a RoPE layer from precomputed cos/sin frequency values.
    ///
    /// # Arguments
    /// * `cos_freqs` - `cos(θ_i)` for each pair index, length = head_dim/2
    /// * `sin_freqs` - `sin(θ_i)` for each pair index, length = head_dim/2
    ///
    /// # Errors
    /// Returns error if lengths don't match, values are non-finite, or any
    /// `(cos, sin)` pair violates the unit-rotation invariant.
    pub fn new(cos_freqs: Vec<f32>, sin_freqs: Vec<f32>) -> Result<Self> {
        if cos_freqs.len() != sin_freqs.len() {
            return Err(NyError::InvalidSpec(format!(
                "RoPE cos_freqs length {} != sin_freqs length {}",
                cos_freqs.len(),
                sin_freqs.len()
            )));
        }
        if cos_freqs.is_empty() {
            return Err(NyError::InvalidSpec(
                "RoPE requires at least one frequency pair".to_string(),
            ));
        }
        for (i, (&c, &s)) in cos_freqs.iter().zip(sin_freqs.iter()).enumerate() {
            if !c.is_finite() || !s.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "RoPE freq[{i}] non-finite: cos={c}, sin={s}"
                )));
            }
            // RoPE tables are only valid rotations when each stored `(cos, sin)` pair
            // stays on the unit circle. Reference:
            // designs/archive/2026-02-28-rope-layer-ibp-crown-propagation.md
            let norm_sq = f64::from(c).mul_add(f64::from(c), f64::from(s) * f64::from(s));
            if (norm_sq - 1.0).abs() > UNIT_ROTATION_TOLERANCE {
                return Err(NyError::InvalidSpec(format!(
                    "RoPE freq[{i}] violates unit-rotation invariant: cos={c}, sin={s}, norm_sq={norm_sq}"
                )));
            }
        }
        Ok(Self {
            cos_freqs,
            sin_freqs,
        })
    }

    /// Create a RoPE layer from raw frequency angles θ_i.
    ///
    /// Computes `cos(θ_i)` and `sin(θ_i)` for each angle.
    pub fn from_angles(angles: &[f32]) -> Result<Self> {
        let cos_freqs: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
        let sin_freqs: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
        Self::new(cos_freqs, sin_freqs)
    }

    /// Create a RoPE layer for a specific sequence position and model dimension.
    ///
    /// Computes `θ_i = position / 10000^(2i / d_model)` for `i = 0..head_dim/2`.
    ///
    /// # Arguments
    /// * `position` - Sequence position (0-indexed)
    /// * `head_dim` - Head dimension (must be even)
    /// * `base` - Base for frequency computation (default: 10000.0)
    pub fn from_position(position: usize, head_dim: usize, base: f32) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(NyError::InvalidSpec(format!(
                "RoPE head_dim must be positive and even, got {head_dim}"
            )));
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "RoPE base must be positive and finite, got {base}"
            )));
        }
        let num_pairs = head_dim / 2;
        let angles: Vec<f32> = (0..num_pairs)
            .map(|i| {
                let exponent = 2.0 * i as f32 / head_dim as f32;
                position as f32 / base.powf(exponent)
            })
            .collect();
        Self::from_angles(&angles)
    }

    /// Number of element pairs this layer operates on.
    pub fn num_pairs(&self) -> usize {
        self.cos_freqs.len()
    }

    /// Head dimension (= 2 * num_pairs).
    pub fn head_dim(&self) -> usize {
        self.cos_freqs.len() * 2
    }
}

/// Compute interval bounds for a*x + b*y where a, b are constants and x ∈ [xl, xu], y ∈ [yl, yu].
///
/// For a constant multiplier `a` and interval `[xl, xu]`:
/// - If a >= 0: a*[xl, xu] = [a*xl, a*xu]
/// - If a < 0:  a*[xl, xu] = [a*xu, a*xl]
///
/// Sum of two such terms gives the output interval.
///
/// Accumulates in f64 to avoid intermediate f32 product rounding compounding beyond
/// what a single ULP adjustment can recover. Directed rounding (`next_down_f32` /
/// `next_up_f32`) is applied after casting back to f32, ensuring the computed interval
/// always contains the true output.
///
/// Reference: designs/2026-02-28-rope-layer-ibp-crown-propagation.md
#[inline]
fn linear_combination_bounds(
    a: f32,
    x_lo: f32,
    x_hi: f32,
    b: f32,
    y_lo: f32,
    y_hi: f32,
) -> (f32, f32) {
    let (a, b) = (a as f64, b as f64);
    let (ax_lo, ax_hi) = if a >= 0.0 {
        (a * x_lo as f64, a * x_hi as f64)
    } else {
        (a * x_hi as f64, a * x_lo as f64)
    };
    let (by_lo, by_hi) = if b >= 0.0 {
        (b * y_lo as f64, b * y_hi as f64)
    } else {
        (b * y_hi as f64, b * y_lo as f64)
    };
    (
        next_down_f32((ax_lo + by_lo) as f32),
        next_up_f32((ax_hi + by_hi) as f32),
    )
}

impl BoundPropagation for RopeLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let input_shape = input.shape();
        let last_dim = *input_shape
            .last()
            .ok_or_else(|| NyError::InvalidSpec("RoPE IBP: empty input shape".to_string()))?;

        if last_dim != self.head_dim() {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.head_dim()],
                got: vec![last_dim],
            });
        }

        let total_elements: usize = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "RoPE CROWN: input shape product overflows usize: {:?}",
                input_shape,
            ))
        })?;
        let num_vectors = total_elements / last_dim;
        let hd = self.head_dim();

        let lower_flat = contiguous_flat_slice(input.lower());
        let upper_flat = contiguous_flat_slice(input.upper());

        let mut out_lower = vec![0.0f32; total_elements];
        let mut out_upper = vec![0.0f32; total_elements];

        for v in 0..num_vectors {
            let base = v * hd;
            for i in 0..self.num_pairs() {
                let c = self.cos_freqs[i];
                let s = self.sin_freqs[i];
                let idx_even = base + 2 * i;
                let idx_odd = base + 2 * i + 1;

                let x0_lo = lower_flat[idx_even];
                let x0_hi = upper_flat[idx_even];
                let x1_lo = lower_flat[idx_odd];
                let x1_hi = upper_flat[idx_odd];

                // y[2i]   = c * x[2i] - s * x[2i+1]  (= c * x0 + (-s) * x1)
                let (y_even_lo, y_even_hi) =
                    linear_combination_bounds(c, x0_lo, x0_hi, -s, x1_lo, x1_hi);

                // y[2i+1] = s * x[2i] + c * x[2i+1]  (= s * x0 + c * x1)
                let (y_odd_lo, y_odd_hi) =
                    linear_combination_bounds(s, x0_lo, x0_hi, c, x1_lo, x1_hi);

                out_lower[idx_even] = y_even_lo;
                out_upper[idx_even] = y_even_hi;
                out_lower[idx_odd] = y_odd_lo;
                out_upper[idx_odd] = y_odd_hi;
            }
        }

        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(input_shape), out_lower).map_err(|e| {
                NyError::InvalidSpec(format!("RoPE IBP: reshape lower failed: {e}"))
            })?,
            ArrayD::from_shape_vec(IxDyn(input_shape), out_upper).map_err(|e| {
                NyError::InvalidSpec(format!("RoPE IBP: reshape upper failed: {e}"))
            })?,
        )
    }

    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // RoPE is a linear layer: y = R @ x where R is a block-diagonal rotation matrix.
        // CROWN backward: new_A = A @ R, new_b = b (unchanged).
        //
        // For each pair i with cos(θ_i) = c, sin(θ_i) = s:
        //   new_A[:, 2i]   = A[:, 2i] * c + A[:, 2i+1] * s
        //   new_A[:, 2i+1] = A[:, 2i] * (-s) + A[:, 2i+1] * c
        //
        // This is the transpose of the forward rotation (R^T = R^{-1} for orthogonal R).

        let num_inputs = bounds.num_inputs();
        let num_outputs = bounds.num_outputs();

        if num_inputs != self.head_dim() {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.head_dim()],
                got: vec![num_inputs],
            });
        }

        let mut new_lower_a = bounds.lower_a().clone();
        let mut new_upper_a = bounds.upper_a().clone();

        for i in 0..self.num_pairs() {
            // Accumulate in f64 to prevent intermediate product rounding from
            // compounding beyond what concretization's directed rounding can recover.
            // Directed rounding on f64→f32 cast: lower_a rounds DOWN, upper_a rounds UP.
            // This is sound regardless of coefficient sign — see R1 proof on #3308.
            let c = self.cos_freqs[i] as f64;
            let s = self.sin_freqs[i] as f64;
            let col_even = 2 * i;
            let col_odd = 2 * i + 1;

            for j in 0..num_outputs {
                let la_even = bounds.lower_a()[[j, col_even]] as f64;
                let la_odd = bounds.lower_a()[[j, col_odd]] as f64;
                let ua_even = bounds.upper_a()[[j, col_even]] as f64;
                let ua_odd = bounds.upper_a()[[j, col_odd]] as f64;

                // new_A[:, 2i]   = A[:, 2i] * c + A[:, 2i+1] * s
                // new_A[:, 2i+1] = A[:, 2i] * (-s) + A[:, 2i+1] * c
                new_lower_a[[j, col_even]] = next_down_f32((la_even * c + la_odd * s) as f32);
                new_lower_a[[j, col_odd]] = next_down_f32((la_even * (-s) + la_odd * c) as f32);
                new_upper_a[[j, col_even]] = next_up_f32((ua_even * c + ua_odd * s) as f32);
                new_upper_a[[j, col_odd]] = next_up_f32((ua_even * (-s) + ua_odd * c) as f32);
            }
        }

        Ok(Cow::Owned(LinearBounds::new_or_conservative(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
        )?))
    }
}

impl RopeLayer {
    /// Batched CROWN backward propagation through RoPE.
    ///
    /// Same as `propagate_linear` but for batched coefficient matrices.
    /// RoPE is linear, so the backward pass is exact: `new_A = A @ R`, `new_b = b`.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        let a_shape = bounds.lower_a.shape();
        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let in_dim = a_shape[a_shape.len() - 1];
        if in_dim != self.head_dim() {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.head_dim()],
                got: vec![in_dim],
            });
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch: usize = checked_shape_product(batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "RoPE CROWN: batch dimensions {batch_dims:?} overflow usize",
                ))
            })?
            .max(1);

        // Reshape to 3D for processing
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;

        let mut new_lower_a = lower_a_3d.to_owned();
        let mut new_upper_a = upper_a_3d.to_owned();

        for b in 0..total_batch {
            for i in 0..self.num_pairs() {
                // Accumulate in f64 (same rationale as propagate_linear).
                // Directed rounding: lower_a DOWN, upper_a UP (#3308).
                let c = self.cos_freqs[i] as f64;
                let s = self.sin_freqs[i] as f64;
                let col_even = 2 * i;
                let col_odd = 2 * i + 1;

                for j in 0..out_dim {
                    let la_even = lower_a_3d[[b, j, col_even]] as f64;
                    let la_odd = lower_a_3d[[b, j, col_odd]] as f64;
                    let ua_even = upper_a_3d[[b, j, col_even]] as f64;
                    let ua_odd = upper_a_3d[[b, j, col_odd]] as f64;

                    new_lower_a[[b, j, col_even]] =
                        next_down_f32((la_even * c + la_odd * s) as f32);
                    new_lower_a[[b, j, col_odd]] =
                        next_down_f32((la_even * (-s) + la_odd * c) as f32);
                    new_upper_a[[b, j, col_even]] = next_up_f32((ua_even * c + ua_odd * s) as f32);
                    new_upper_a[[b, j, col_odd]] =
                        next_up_f32((ua_even * (-s) + ua_odd * c) as f32);
                }
            }
        }

        // Reshape back to original shape.
        // to_owned() always produces offset=0; assert defensively.
        let (new_lower_a_vec, offset_l) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, offset_u) = new_upper_a.into_raw_vec_and_offset();
        debug_assert_eq!(
            offset_l,
            Some(0),
            "unexpected non-zero offset in RoPE batched lower_a"
        );
        debug_assert_eq!(
            offset_u,
            Some(0),
            "unexpected non-zero offset in RoPE batched upper_a"
        );

        BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(a_shape), new_lower_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
            bounds.lower_b.clone(),
            ArrayD::from_shape_vec(IxDyn(a_shape), new_upper_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
            bounds.upper_b.clone(),
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )
    }
}
