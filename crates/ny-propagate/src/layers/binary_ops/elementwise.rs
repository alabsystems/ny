// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared implementation for element-wise binary operations (min, max).
//!
//! MinBinary and MaxBinary differ only in the combining function
//! (`nan_propagating_min` vs `nan_propagating_max`).
//! This module provides `elementwise_binary_ibp` to eliminate the duplication.
//!
//! **NaN safety (#3147):** All `op` calls are wrapped with a NaN guard so that
//! NaN inputs always produce NaN outputs, even if the caller accidentally passes
//! an IEEE 754 NaN-absorbing function like `f32::min`/`f32::max`.

use ndarray::IxDyn;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::shape::broadcast_shapes;

/// Propagate IBP bounds through an element-wise binary operation.
///
/// For C = op(A, B) where A ∈ [A_l, A_u] and B ∈ [B_l, B_u]:
/// - C_lower = op(A_l, B_l)
/// - C_upper = op(A_u, B_u)
///
/// This is sound for monotone operations (min, max) where
/// op(lower_a, lower_b) gives the lower bound of the output and
/// op(upper_a, upper_b) gives the upper bound.
///
/// **NaN safety (#3147):** `op` MUST be NaN-propagating (use
/// `nan_propagating_min`/`nan_propagating_max` from `ny_core`).
/// As defense-in-depth, this function wraps each `op` call so that
/// NaN in either operand always produces NaN in the output, even if
/// `op` itself absorbs NaN (as `f32::min`/`f32::max` do per IEEE 754).
pub fn elementwise_binary_ibp(
    input_a: &BoundedTensor,
    input_b: &BoundedTensor,
    op: fn(f32, f32) -> f32,
    op_name: &str,
) -> Result<BoundedTensor> {
    // Defense-in-depth: wrap op so NaN always propagates (#3147).
    // f32::min/max absorb NaN per IEEE 754 — this guard prevents silent
    // unsoundness if a caller accidentally passes the wrong function.
    let nan_safe_op = |a: f32, b: f32| -> f32 {
        if a.is_nan() || b.is_nan() {
            f32::NAN
        } else {
            op(a, b)
        }
    };

    if input_a.shape() == input_b.shape() {
        // Fast path: same shapes
        let out_lower = input_a
            .lower()
            .iter()
            .zip(input_b.lower().iter())
            .map(|(&a, &b)| nan_safe_op(a, b))
            .collect::<Vec<_>>();
        let out_upper = input_a
            .upper()
            .iter()
            .zip(input_b.upper().iter())
            .map(|(&a, &b)| nan_safe_op(a, b))
            .collect::<Vec<_>>();

        let shape = input_a.shape().to_vec();
        let lower = ndarray::ArrayD::from_shape_vec(IxDyn(&shape), out_lower)
            .map_err(|e| NyError::UnsupportedConfiguration(format!("{}: {}", op_name, e)))?;
        let upper = ndarray::ArrayD::from_shape_vec(IxDyn(&shape), out_upper)
            .map_err(|e| NyError::UnsupportedConfiguration(format!("{}: {}", op_name, e)))?;

        BoundedTensor::new(lower, upper)
    } else {
        // Broadcasting path
        let target_shape = broadcast_shapes(input_a.shape(), input_b.shape()).ok_or_else(|| {
            NyError::ShapeMismatch {
                expected: input_a.shape().to_vec(),
                got: input_b.shape().to_vec(),
            }
        })?;

        let a_lower = input_a
            .lower()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_a.shape().to_vec(),
            })?;
        let a_upper = input_a
            .upper()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_a.shape().to_vec(),
            })?;
        let b_lower = input_b
            .lower()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_b.shape().to_vec(),
            })?;
        let b_upper = input_b
            .upper()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_b.shape().to_vec(),
            })?;

        let out_lower = a_lower
            .iter()
            .zip(b_lower.iter())
            .map(|(&a, &b)| nan_safe_op(a, b))
            .collect::<Vec<_>>();
        let out_upper = a_upper
            .iter()
            .zip(b_upper.iter())
            .map(|(&a, &b)| nan_safe_op(a, b))
            .collect::<Vec<_>>();

        let lower = ndarray::ArrayD::from_shape_vec(IxDyn(&target_shape), out_lower)
            .map_err(|e| NyError::UnsupportedConfiguration(format!("{}: {}", op_name, e)))?;
        let upper = ndarray::ArrayD::from_shape_vec(IxDyn(&target_shape), out_upper)
            .map_err(|e| NyError::UnsupportedConfiguration(format!("{}: {}", op_name, e)))?;

        BoundedTensor::new(lower, upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};
    use ny_core::{nan_propagating_max, nan_propagating_min};

    fn make_bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        let n = lower.len();
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    fn make_bt_shape(lower: Vec<f32>, upper: Vec<f32>, shape: &[usize]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap(),
            ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn test_elementwise_binary_ibp_same_shape() {
        let a = make_bt(&[1.0, 5.0], &[3.0, 7.0]);
        let b = make_bt(&[2.0, 4.0], &[6.0, 8.0]);

        let min_result = elementwise_binary_ibp(&a, &b, nan_propagating_min, "Min").unwrap();
        assert!((min_result.lower()[0] - 1.0).abs() < 1e-5);
        assert!((min_result.lower()[1] - 4.0).abs() < 1e-5);
        assert!((min_result.upper()[0] - 3.0).abs() < 1e-5);
        assert!((min_result.upper()[1] - 7.0).abs() < 1e-5);

        let max_result = elementwise_binary_ibp(&a, &b, nan_propagating_max, "Max").unwrap();
        assert!((max_result.lower()[0] - 2.0).abs() < 1e-5);
        assert!((max_result.lower()[1] - 5.0).abs() < 1e-5);
        assert!((max_result.upper()[0] - 6.0).abs() < 1e-5);
        assert!((max_result.upper()[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_elementwise_binary_ibp_broadcast() {
        let a = make_bt_shape(vec![1.0, 5.0, 3.0], vec![4.0, 8.0, 6.0], &[1, 3]);
        let b = make_bt_shape(
            vec![2.0, 3.0, 1.0, 0.0, 6.0, 2.0],
            vec![3.0, 4.0, 7.0, 5.0, 9.0, 8.0],
            &[2, 3],
        );
        let result = elementwise_binary_ibp(&a, &b, nan_propagating_min, "Min").unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        // Row 0: min([1,5,3],[2,3,1]) = [1,3,1]
        assert!((result.lower()[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((result.lower()[[0, 1]] - 3.0).abs() < 1e-5);
        assert!((result.lower()[[0, 2]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_elementwise_binary_ibp_shape_mismatch() {
        let a = make_bt(&[1.0, 2.0], &[3.0, 4.0]);
        let b = make_bt(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let err = elementwise_binary_ibp(&a, &b, nan_propagating_min, "Min")
            .expect_err("incompatible shapes");
        assert!(matches!(err, NyError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_elementwise_binary_ibp_soundness_min() {
        let a = make_bt(&[-3.0, 0.0], &[2.0, 5.0]);
        let b = make_bt(&[-1.0, -2.0], &[4.0, 3.0]);
        let result = elementwise_binary_ibp(&a, &b, nan_propagating_min, "Min").unwrap();
        for k in 0..=20 {
            let ta = k as f32 / 20.0;
            for j in 0..=20 {
                let tb = j as f32 / 20.0;
                let a0 = -3.0 + 5.0 * ta;
                let b0 = -1.0 + 5.0 * tb;
                let m0 = a0.min(b0);
                assert!(m0 >= result.lower()[0] - 1e-5);
                assert!(m0 <= result.upper()[0] + 1e-5);
            }
        }
    }

    #[test]
    fn test_elementwise_binary_ibp_soundness_max() {
        let a = make_bt(&[-3.0, 0.0], &[2.0, 5.0]);
        let b = make_bt(&[-1.0, -2.0], &[4.0, 3.0]);
        let result = elementwise_binary_ibp(&a, &b, nan_propagating_max, "Max").unwrap();
        for k in 0..=20 {
            let ta = k as f32 / 20.0;
            for j in 0..=20 {
                let tb = j as f32 / 20.0;
                let a0 = -3.0 + 5.0 * ta;
                let b0 = -1.0 + 5.0 * tb;
                let m0 = a0.max(b0);
                assert!(m0 >= result.lower()[0] - 1e-5);
                assert!(m0 <= result.upper()[0] + 1e-5);
            }
        }
    }

    /// Helper to create a BoundedTensor that may contain NaN/Inf (bypasses validation).
    fn make_bt_unchecked(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        let n = lower.len();
        BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    fn make_bt_unchecked_shape(lower: Vec<f32>, upper: Vec<f32>, shape: &[usize]) -> BoundedTensor {
        BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap(),
            ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap(),
        )
        .unwrap()
    }

    /// Regression test for #3147: NaN in lower bounds of input A must propagate
    /// to the output (Err from BoundedTensor::new), not be silently absorbed
    /// into a finite Ok. Tests fast path (same shape).
    #[test]
    fn test_nan_propagation_lower_a_same_shape() {
        let a = make_bt_unchecked(&[f32::NAN, 1.0], &[2.0, 3.0]);
        let b = make_bt(&[0.0, 0.0], &[1.0, 1.0]);

        // NaN propagates through min → output has NaN → BoundedTensor::new rejects
        let min_err = elementwise_binary_ibp(&a, &b, nan_propagating_min, "Min");
        assert!(
            min_err.is_err(),
            "NaN in A lower must propagate through min → Err"
        );

        let max_err = elementwise_binary_ibp(&a, &b, nan_propagating_max, "Max");
        assert!(
            max_err.is_err(),
            "NaN in A lower must propagate through max → Err"
        );
    }

    /// Regression test for #3147: NaN in upper bounds of input B must propagate.
    #[test]
    fn test_nan_propagation_upper_b_same_shape() {
        let a = make_bt(&[0.0, 1.0], &[2.0, 3.0]);
        let b = make_bt_unchecked(&[0.0, 0.0], &[f32::NAN, 1.0]);

        let min_err = elementwise_binary_ibp(&a, &b, nan_propagating_min, "Min");
        assert!(
            min_err.is_err(),
            "NaN in B upper must propagate through min → Err"
        );

        let max_err = elementwise_binary_ibp(&a, &b, nan_propagating_max, "Max");
        assert!(
            max_err.is_err(),
            "NaN in B upper must propagate through max → Err"
        );
    }

    /// Regression test for #3147: defense-in-depth — even with the NaN-absorbing
    /// `f32::min`, the NaN guard inside `elementwise_binary_ibp` prevents silent
    /// absorption. Before the fix, `f32::min(NaN, 2.0) = 2.0` silently dropped
    /// NaN and produced `Ok(wrong_bounds)`. After: NaN propagates → `Err`.
    #[test]
    fn test_nan_defense_in_depth_absorbing_op() {
        let a = make_bt_unchecked(&[f32::NAN], &[2.0]);
        let b = make_bt(&[1.0], &[3.0]);

        // Even passing f32::min (NaN-absorbing), the guard must propagate NaN → Err.
        let result = elementwise_binary_ibp(&a, &b, f32::min, "Min");
        assert!(
            result.is_err(),
            "defense-in-depth: f32::min with NaN input must produce Err, not Ok(wrong_bounds)"
        );
    }

    /// Regression test for #3147: NaN propagation through broadcast path.
    #[test]
    fn test_nan_propagation_broadcast_path() {
        // A: [1, 2] with NaN in lower[0], B: [2, 2] — triggers broadcast path
        let a = make_bt_unchecked_shape(vec![f32::NAN, 1.0], vec![2.0, 3.0], &[1, 2]);
        let b = make_bt_shape(vec![0.0, 0.0, 1.0, 1.0], vec![1.0, 1.0, 2.0, 2.0], &[2, 2]);

        // NaN at A[0,0] broadcasts to [0,0] and [1,0] → output has NaN → Err
        let result = elementwise_binary_ibp(&a, &b, nan_propagating_min, "Min");
        assert!(
            result.is_err(),
            "NaN must propagate through broadcast path → Err"
        );
    }

    /// Negative control: confirm `f32::min` absorbs NaN (IEEE 754 behavior).
    /// This documents why the defense-in-depth guard in `elementwise_binary_ibp`
    /// is necessary — without it, NaN-absorbing ops silently produce wrong bounds.
    #[test]
    fn test_ieee754_nan_absorption_negative_control() {
        // This is the behavior we're defending against:
        assert_eq!(
            f32::min(f32::NAN, 2.0),
            2.0,
            "IEEE 754: f32::min absorbs NaN"
        );
        assert_eq!(
            f32::max(f32::NAN, 2.0),
            2.0,
            "IEEE 754: f32::max absorbs NaN"
        );
        // Our nan_propagating variants do NOT absorb:
        assert!(nan_propagating_min(f32::NAN, 2.0).is_nan());
        assert!(nan_propagating_max(f32::NAN, 2.0).is_nan());
    }
}
