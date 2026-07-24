// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Double-precision bounded tensor for soundness-critical propagation.
//!
//! Used when `double_fp: true` is configured (required for VNN-COMP
//! soundnessbench and sat_relu benchmarks). Unlike [`BoundedTensor`] which
//! stores f32, this type stores and propagates entirely in f64, avoiding
//! the f64→f32 rounding errors that soundnessbench is designed to detect.
//!
//! Reference: alpha-beta-CROWN `double_fp` config (`abcrown.py:81-82`).

use ndarray::{Array1, ArrayD};
use ny_core::{NyError, Result};

use super::BoundedTensor;
use crate::rounding::{next_down_f32, next_up_f32};

/// Double-precision bounded tensor for soundness-critical propagation.
///
/// Only used when `double_fp: true` is configured. Supports the minimal
/// subset of operations needed for sequential FC+Conv2D+ReLU verification.
#[derive(Debug, Clone)]
pub struct BoundedTensor64 {
    lower: ArrayD<f64>,
    upper: ArrayD<f64>,
}

impl BoundedTensor64 {
    /// Create from f64 arrays with validation.
    ///
    /// # Errors
    /// Returns error if shapes don't match, bounds contain NaN, or lower > upper.
    pub fn new(lower: ArrayD<f64>, upper: ArrayD<f64>) -> Result<Self> {
        if lower.shape() != upper.shape() {
            return Err(NyError::InvalidSpec(format!(
                "BoundedTensor64 shape mismatch: lower {:?} != upper {:?}",
                lower.shape(),
                upper.shape()
            )));
        }
        if lower.iter().any(|v| v.is_nan()) || upper.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "BoundedTensor64 contains NaN".into(),
            ));
        }
        // Match f32 BoundedTensor::new() validation: reject inverted bounds.
        // Inverted bounds (lower > upper) indicate a propagation bug and must
        // not be silently accepted. See constructors.rs:120.
        let bounds_valid = ndarray::Zip::from(&lower).and(&upper).all(|&l, &u| l <= u);
        if !bounds_valid {
            return Err(NyError::InvalidSpec(
                "BoundedTensor64::new: found lower > upper (inverted bounds)".to_string(),
            ));
        }
        Ok(Self { lower, upper })
    }

    /// Create a "concrete" BoundedTensor64 with lower == upper == values.
    ///
    /// Used for point evaluation. Rejects NaN and Inf since concrete points
    /// must be finite.
    ///
    /// Needed by `ny_propagate::evaluate_network_f64` and downstream external
    /// verifier consumers.
    pub fn concrete(values: ArrayD<f64>) -> Result<Self> {
        if values.iter().any(|v| v.is_nan() || v.is_infinite()) {
            return Err(NyError::NumericalInstability(
                "BoundedTensor64::concrete: values contain NaN or Inf".to_string(),
            ));
        }
        Ok(Self {
            lower: values.clone(),
            upper: values,
        })
    }

    /// Convert from f32 BoundedTensor (exact widening, no rounding).
    ///
    /// f32→f64 conversion is always exact since f64 can represent all f32 values.
    pub fn from_f32(bt: &BoundedTensor) -> Self {
        Self {
            lower: bt.lower().mapv(|x| x as f64),
            upper: bt.upper().mapv(|x| x as f64),
        }
    }

    /// Convert back to f32 BoundedTensor with directed rounding for soundness.
    ///
    /// Lower bounds are rounded toward -∞ (`next_down_f32`), upper bounds
    /// toward +∞ (`next_up_f32`). This ensures the f32 result is a sound
    /// overapproximation of the f64 bounds.
    pub fn to_f32_sound(&self) -> BoundedTensor {
        let lower_f32 = self.lower.mapv(|x| {
            let cast = x as f32;
            if cast.is_finite() {
                next_down_f32(cast)
            } else {
                f32::NEG_INFINITY
            }
        });
        let upper_f32 = self.upper.mapv(|x| {
            let cast = x as f32;
            if cast.is_finite() {
                next_up_f32(cast)
            } else {
                f32::INFINITY
            }
        });
        // KEEP unchecked: directed rounding preserves the source shape and maps
        // each element to a finite value or +/-Inf, never NaN.
        BoundedTensor::from_parts_unchecked(lower_f32.into_dyn(), upper_f32.into_dyn())
    }

    /// Read-only view of the lower bounds.
    #[inline]
    pub fn lower(&self) -> &ArrayD<f64> {
        &self.lower
    }

    /// Read-only view of the upper bounds.
    #[inline]
    pub fn upper(&self) -> &ArrayD<f64> {
        &self.upper
    }

    /// Shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.lower.shape()
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Check if tensor is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    /// Flatten to 1D, returning (lower, upper) as Array1<f64>.
    pub fn flatten_to_1d(&self) -> (Array1<f64>, Array1<f64>) {
        let lower = Array1::from_iter(self.lower.iter().copied());
        let upper = Array1::from_iter(self.upper.iter().copied());
        (lower, upper)
    }

    /// Conservative (maximally loose) bounds for a given shape.
    pub fn conservative(shape: &[usize]) -> Self {
        Self {
            lower: ArrayD::from_elem(shape, f64::NEG_INFINITY),
            upper: ArrayD::from_elem(shape, f64::INFINITY),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_from_f32_roundtrip() {
        let lower = arr1(&[1.0f32, -2.5, 0.0]).into_dyn();
        let upper = arr1(&[3.0f32, -1.0, 1.0]).into_dyn();
        let bt = BoundedTensor::new(lower, upper).unwrap();
        let bt64 = BoundedTensor64::from_f32(&bt);

        // f32→f64 is exact
        assert_eq!(bt64.lower()[0], 1.0f64);
        assert_eq!(bt64.lower()[1], -2.5f64);
        assert_eq!(bt64.upper()[0], 3.0f64);

        // Roundtrip with directed rounding
        let back = bt64.to_f32_sound();
        // Sound: f32 lower <= f64 lower, f32 upper >= f64 upper
        for i in 0..3 {
            assert!(
                back.lower().iter().nth(i).unwrap() <= bt.lower().iter().nth(i).unwrap(),
                "lower bound at {i} not sound"
            );
            assert!(
                back.upper().iter().nth(i).unwrap() >= bt.upper().iter().nth(i).unwrap(),
                "upper bound at {i} not sound"
            );
        }
    }

    #[test]
    fn test_new_rejects_inverted_bounds() {
        let lower = arr1(&[1.0f64, 5.0]).into_dyn();
        let upper = arr1(&[2.0f64, 3.0]).into_dyn(); // upper[1] < lower[1]
        let err = BoundedTensor64::new(lower, upper)
            .expect_err("BoundedTensor64::new should reject inverted bounds");
        assert!(
            err.to_string().contains("inverted bounds"),
            "error should mention inverted bounds"
        );
    }

    #[test]
    fn test_new_rejects_nan() {
        let lower = arr1(&[1.0f64, f64::NAN]).into_dyn();
        let upper = arr1(&[2.0f64, 3.0]).into_dyn();
        let err =
            BoundedTensor64::new(lower, upper).expect_err("BoundedTensor64::new should reject NaN");
        assert!(err.to_string().contains("NaN"), "error should mention NaN");
    }

    #[test]
    fn test_new_rejects_shape_mismatch() {
        let lower = arr1(&[1.0f64, 2.0]).into_dyn();
        let upper = arr1(&[3.0f64]).into_dyn();
        let err = BoundedTensor64::new(lower, upper)
            .expect_err("BoundedTensor64::new should reject mismatched shapes");
        assert!(
            err.to_string().contains("shape mismatch"),
            "error should mention shape mismatch"
        );
    }

    #[test]
    fn test_flatten_to_1d() {
        let lower = ndarray::Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn();
        let upper = ndarray::Array2::from_shape_vec((2, 3), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .unwrap()
            .into_dyn();
        let bt = BoundedTensor64::new(lower, upper).unwrap();
        let (l, u) = bt.flatten_to_1d();
        assert_eq!(l.len(), 6);
        assert_eq!(u.len(), 6);
        assert_eq!(l[0], 1.0);
        assert_eq!(u[5], 12.0);
    }

    #[test]
    fn test_conservative() {
        let bt = BoundedTensor64::conservative(&[3, 4]);
        assert_eq!(bt.shape(), &[3, 4]);
        assert!(
            bt.lower().iter().all(|v| *v == f64::NEG_INFINITY),
            "conservative lower bounds should all be -inf, got {:?}",
            bt.lower()
        );
        assert!(
            bt.upper().iter().all(|v| *v == f64::INFINITY),
            "conservative upper bounds should all be +inf, got {:?}",
            bt.upper()
        );
    }
}
