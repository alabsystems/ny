// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Compressed bounds using f16 storage for memory efficiency.
//!
//! This module provides `CompressedBounds` which stores interval bounds using
//! 16-bit floats (f16/half-precision), reducing memory usage by 50% compared
//! to standard f32 storage.
//!
//! ## Trade-offs
//!
//! - **Memory**: 50% reduction (4 bytes -> 2 bytes per bound)
//! - **Precision**: f16 has ~3 decimal digits of precision (vs ~7 for f32)
//! - **Range**: f16 max is ~65504 (vs ~3.4e38 for f32)
//!
//! ## Use Cases
//!
//! - Checkpoint storage in streaming/gradient checkpointing
//! - Long-term bound storage when memory is constrained
//! - NOT for active computation (convert to f32 first)
//!
//! # Example
//!
//! ```
//! use ny_tensor::{BoundedTensor, CompressedBounds};
//! use ndarray::ArrayD;
//!
//! // Create bounds
//! let lower = ArrayD::from_elem(ndarray::IxDyn(&[100]), -1.0f32);
//! let upper = ArrayD::from_elem(ndarray::IxDyn(&[100]), 1.0f32);
//! let bounds = BoundedTensor::new(lower, upper).unwrap();
//!
//! // Compress for storage (50% memory reduction)
//! let compressed = CompressedBounds::from_bounded_tensor(&bounds);
//!
//! // Decompress when needed for computation
//! let restored = compressed.to_bounded_tensor().unwrap();
//! ```

use crate::{BoundedScalar, BoundedTensor};
use half::f16;
use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, nan_propagating_max, NyError, Result};
use serde::{Deserialize, Serialize};

/// Compressed bounds storage using f16 (half-precision) floats.
///
/// Provides 50% memory reduction compared to `BoundedTensor` at the cost
/// of reduced precision. Suitable for checkpoint storage, not for active
/// computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedBounds {
    /// Lower bounds in f16 format.
    lower: Vec<f16>,
    /// Upper bounds in f16 format.
    upper: Vec<f16>,
    /// Shape of the tensor.
    shape: Vec<usize>,
}

impl CompressedBounds {
    /// Create compressed bounds from raw f16 vectors.
    ///
    /// # Errors
    /// Returns error if lower and upper have different lengths.
    pub fn new(lower: Vec<f16>, upper: Vec<f16>, shape: Vec<usize>) -> Result<Self> {
        let expected_len = checked_shape_product(&shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "CompressedBounds::new: shape product overflows: {:?}",
                shape
            ))
        })?;
        if lower.len() != expected_len || upper.len() != expected_len {
            return Err(NyError::shape_mismatch(
                vec![expected_len],
                vec![lower.len(), upper.len()],
            ));
        }
        Ok(Self {
            lower,
            upper,
            shape,
        })
    }

    /// Create compressed bounds without validation.
    ///
    /// # Safety
    /// This is only for testing error paths. The shape must match the data length
    /// for `to_bounded_tensor()` to succeed.
    #[cfg(test)]
    pub(crate) fn new_unchecked(lower: Vec<f16>, upper: Vec<f16>, shape: Vec<usize>) -> Self {
        Self {
            lower,
            upper,
            shape,
        }
    }

    /// Create compressed bounds from a `BoundedTensor`.
    ///
    /// This is the primary way to create compressed bounds.
    /// Uses directed rounding for soundness: lower bounds round toward -∞,
    /// upper bounds round toward +∞. This guarantees the compressed bounds
    /// are a superset of the original bounds.
    ///
    /// Reference: `BoundedScalar::from_f32_down` / `from_f32_up` in `generic.rs`
    /// implement directed rounding via `next_down_f16` / `next_up_f16`.
    pub fn from_bounded_tensor(bounds: &BoundedTensor) -> Self {
        let shape = bounds.shape().to_vec();
        let (lower_bounds, upper_bounds) = bounds.lower_upper();

        // Directed rounding: lower bounds round DOWN (toward -∞),
        // upper bounds round UP (toward +∞) to preserve soundness.
        let lower: Vec<f16> = lower_bounds
            .iter()
            .map(|&v| <f16 as BoundedScalar>::from_f32_down(v))
            .collect();
        let upper: Vec<f16> = upper_bounds
            .iter()
            .map(|&v| <f16 as BoundedScalar>::from_f32_up(v))
            .collect();

        Self {
            lower,
            upper,
            shape,
        }
    }

    /// Convert back to `BoundedTensor` for computation.
    ///
    /// This restores the bounds to f32 format. Note that precision
    /// may be lost due to the f16 intermediate representation.
    ///
    /// Uses `new_allow_infinite` because f32 values exceeding f16::MAX (~65504)
    /// become ±Inf during compression. The directed rounding in `from_bounded_tensor`
    /// ensures only sound infinities appear: lower bounds get -Inf (widened toward -∞)
    /// and upper bounds get +Inf (widened toward +∞). Rejecting these would cause
    /// runtime failures for networks with large-valued bounds. (#2358)
    pub fn to_bounded_tensor(&self) -> Result<BoundedTensor> {
        let lower_f32: Vec<f32> = self.lower.iter().map(|&v| v.to_f32()).collect();
        let upper_f32: Vec<f32> = self.upper.iter().map(|&v| v.to_f32()).collect();

        let lower = ArrayD::from_shape_vec(IxDyn(&self.shape), lower_f32)
            .map_err(|e| NyError::InvalidSpec(format!("Failed to reshape lower: {}", e)))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&self.shape), upper_f32)
            .map_err(|e| NyError::InvalidSpec(format!("Failed to reshape upper: {}", e)))?;

        BoundedTensor::new_allow_infinite(lower, upper)
    }

    /// Shape of the compressed bounds.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    /// Memory usage in bytes.
    ///
    /// Returns the approximate memory footprint of this structure.
    pub fn memory_bytes(&self) -> usize {
        // Each f16 is 2 bytes, we have lower + upper
        let data_bytes = self.lower.len() * 2 * 2;
        // Plus shape overhead (8 bytes per dimension on 64-bit)
        let shape_bytes = self.shape.len() * 8;
        data_bytes + shape_bytes
    }

    /// Memory usage compared to equivalent f32 BoundedTensor.
    ///
    /// Returns (compressed_bytes, f32_bytes, compression_ratio).
    pub fn compression_stats(&self) -> (usize, usize, f32) {
        let compressed = self.memory_bytes();
        // f32 version: 4 bytes per element, lower + upper
        let f32_bytes = self.lower.len() * 4 * 2;
        let ratio = compressed as f32 / f32_bytes as f32;
        (compressed, f32_bytes, ratio)
    }

    /// Get raw lower bounds (f16).
    pub fn lower_raw(&self) -> &[f16] {
        &self.lower
    }

    /// Get raw upper bounds (f16).
    pub fn upper_raw(&self) -> &[f16] {
        &self.upper
    }

    /// Apply optional extra widening beyond the directed rounding in `from_bounded_tensor`.
    ///
    /// `from_bounded_tensor` already uses directed rounding (`from_f32_down`/`from_f32_up`)
    /// which guarantees compressed bounds are a superset of the originals. This method
    /// provides additional conservative widening by a relative epsilon, which may be
    /// useful when downstream consumers need extra margin beyond the minimum 1-ULP
    /// directed rounding guarantee.
    ///
    /// # Arguments
    /// * `relative_epsilon` - Relative widening factor (e.g., 0.001 for 0.1%)
    pub fn widen_for_soundness(&mut self, relative_epsilon: f32) {
        let eps = f16::from_f32(relative_epsilon);
        let min_delta = f16::from_f32(1e-6); // Minimum absolute widening

        for (l, u) in self.lower.iter_mut().zip(self.upper.iter_mut()) {
            // Widen lower bound down
            let l_abs = if *l < f16::ZERO { -*l } else { *l };
            let l_delta = l_abs * eps;
            let l_widen = if l_delta > min_delta {
                l_delta
            } else {
                min_delta
            };

            // Widen upper bound up
            let u_abs = if *u < f16::ZERO { -*u } else { *u };
            let u_delta = u_abs * eps;
            let u_widen = if u_delta > min_delta {
                u_delta
            } else {
                min_delta
            };

            // Apply widening (lower decreases, upper increases)
            *l -= l_widen;
            *u += u_widen;
        }
    }

    /// Check if any values are infinite or NaN after compression.
    ///
    /// Large f32 values (>65504) become infinity in f16.
    /// Returns true if bounds contain non-finite values.
    pub fn has_overflow(&self) -> bool {
        self.lower.iter().any(|v| !v.is_finite()) || self.upper.iter().any(|v| !v.is_finite())
    }

    /// Maximum precision loss from f32 -> f16 -> f32 round-trip.
    ///
    /// Computes the maximum absolute difference between original and
    /// round-tripped values. Returns (max_lower_error, max_upper_error).
    pub fn max_precision_loss(
        original: &BoundedTensor,
        compressed: &CompressedBounds,
    ) -> (f32, f32) {
        let expected_len = original.lower().len();
        if compressed.lower.len() != expected_len || compressed.upper.len() != expected_len {
            return (f32::INFINITY, f32::INFINITY);
        }

        let mut max_lower_error = 0.0f32;
        let mut max_upper_error = 0.0f32;

        // Use nan_propagating_max instead of f32::max — IEEE 754 maxNum absorbs
        // NaN, silently dropping NaN errors from corrupted data. (#3291 F2)
        for (orig, comp) in original.lower().iter().zip(compressed.lower.iter()) {
            let restored = comp.to_f32();
            let error = (orig - restored).abs();
            max_lower_error = nan_propagating_max(max_lower_error, error);
        }

        for (orig, comp) in original.upper().iter().zip(compressed.upper.iter()) {
            let restored = comp.to_f32();
            let error = (orig - restored).abs();
            max_upper_error = nan_propagating_max(max_upper_error, error);
        }

        (max_lower_error, max_upper_error)
    }
}

/// Statistics about compression quality.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Memory used by compressed representation (bytes).
    pub compressed_bytes: usize,
    /// Memory that would be used by f32 representation (bytes).
    pub original_bytes: usize,
    /// Compression ratio (compressed / original).
    pub compression_ratio: f32,
    /// Maximum precision loss in lower bounds.
    pub max_lower_error: f32,
    /// Maximum precision loss in upper bounds.
    pub max_upper_error: f32,
    /// Whether any values overflowed to infinity.
    pub has_overflow: bool,
}

impl CompressionStats {
    /// Compute statistics from original and compressed bounds.
    pub fn from_compression(original: &BoundedTensor, compressed: &CompressedBounds) -> Self {
        let (compressed_bytes, original_bytes, compression_ratio) = compressed.compression_stats();
        let (max_lower_error, max_upper_error) =
            CompressedBounds::max_precision_loss(original, compressed);
        let has_overflow = compressed.has_overflow();

        Self {
            compressed_bytes,
            original_bytes,
            compression_ratio,
            max_lower_error,
            max_upper_error,
            has_overflow,
        }
    }

    /// Memory savings as percentage.
    pub fn memory_savings_percent(&self) -> f32 {
        100.0 * (1.0 - self.compression_ratio)
    }
}

#[cfg(test)]
#[path = "compressed_tests.rs"]
mod tests;
