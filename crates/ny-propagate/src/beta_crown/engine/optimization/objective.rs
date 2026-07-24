// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Objective helper utilities for optimization bounds.

use ny_core::Result;
use ny_tensor::{next_down_f32, BoundedTensor};

/// Add a scalar offset to the lower bounds of a BoundedTensor.
///
/// Flattens the lower bound, adds the offset to each element, then reshapes
/// back to the original shape. The upper bound is preserved unchanged.
pub(super) fn offset_lower_bounds(bounds: &BoundedTensor, offset: f32) -> Result<BoundedTensor> {
    let lower_shape = bounds.lower().shape().to_vec();
    let upper_shape = bounds.upper().shape().to_vec();
    let lower_len = bounds.lower().len();

    let mut lower_flat = bounds
        .lower()
        .clone()
        .into_shape_clone(ndarray::IxDyn(&[lower_len]))
        .map_err(|_| ny_core::NyError::ShapeMismatch {
            expected: vec![lower_len],
            got: lower_shape.clone(),
        })?;

    for val in lower_flat.iter_mut() {
        // Directed rounding: lower bounds round toward -infinity.
        // Maintains sound overapproximation after GCP-CROWN offset addition.
        // Reference: concretize_sound uses next_down_f32 for lower bounds (#2239).
        *val = next_down_f32(*val + offset);
    }

    let lower_new = lower_flat
        .into_shape_clone(ndarray::IxDyn(&lower_shape))
        .map_err(|_| ny_core::NyError::ShapeMismatch {
            expected: lower_shape.clone(),
            got: vec![lower_len],
        })?;
    let upper_new = bounds
        .upper()
        .clone()
        .into_shape_clone(ndarray::IxDyn(&upper_shape))
        .map_err(|_| ny_core::NyError::ShapeMismatch {
            expected: upper_shape.clone(),
            got: vec![bounds.upper().len()],
        })?;

    BoundedTensor::new(lower_new, upper_new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    #[ntest::timeout(5000)]
    #[test]
    fn test_offset_lower_bounds_directed_rounding() -> Result<()> {
        // Verify that offset_lower_bounds uses directed rounding: every output
        // lower value must be <= (original + offset) computed in f64.
        // Upper bounds must be above lower + offset to avoid inversion.
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0f32, -1.0, 1e6, 1e-6])
            .expect("invariant: valid shape");
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0f32, 0.5, 1e6 + 2.0, 1.0])
            .expect("invariant: valid shape");
        let bounds = BoundedTensor::new(lower.clone(), upper.clone())?;

        let offset = 0.1f32;
        let result = offset_lower_bounds(&bounds, offset)?;

        // Each output lower must be <= (original_lower + offset) computed in f64
        for (i, (orig, out)) in lower.iter().zip(result.lower().iter()).enumerate() {
            let exact = *orig as f64 + offset as f64;
            assert!(
                (*out as f64) <= exact,
                "offset_lower_bounds[{}]: output {} > f64-exact {} (directed rounding violated)",
                i,
                out,
                exact
            );
        }

        // Upper bounds must be unchanged
        for (orig, out) in upper.iter().zip(result.upper().iter()) {
            assert_eq!(*orig, *out, "upper bounds must be preserved");
        }
        Ok(())
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_offset_lower_bounds_nonzero_offset() -> Result<()> {
        // Verify offset is actually applied (not just identity).
        let lower = ArrayD::from_elem(IxDyn(&[3]), 0.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[3]), 1.0f32);
        let bounds = BoundedTensor::new(lower, upper)?;

        let result = offset_lower_bounds(&bounds, 0.5)?;

        // All lower values should be near 0.5 (slightly below due to next_down_f32)
        for val in result.lower().iter() {
            assert!((*val - 0.5).abs() < 1e-6, "expected ~0.5, got {}", val);
            assert!(
                *val <= 0.5,
                "directed rounding: lower must be <= 0.5, got {}",
                val
            );
        }
        Ok(())
    }
}
