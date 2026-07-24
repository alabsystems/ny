// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::elementwise::elementwise_binary_ibp;
use super::minmax_relax::{max_envelope, propagate_minmax_linear_binary};
use crate::bounds::nan_propagating_max;
use crate::LinearBounds;

/// Element-wise maximum layer: computes C = max(A, B) for two bounded inputs.
///
/// This is used for clamping (max(0, x) = ReLU), residual connections with max
/// operations, and attention masking patterns.
///
/// For A ∈ [A_l, A_u] and B ∈ [B_l, B_u]:
/// C ∈ [max(A_l, B_l), max(A_u, B_u)]
///
/// Reference: alpha-beta-CROWN `auto_LiRPA/operators/minmax.py:BoundMax`
#[derive(Debug, Clone)]
pub struct MaxBinaryLayer;

impl MaxBinaryLayer {
    /// Propagate IBP bounds through element-wise maximum.
    ///
    /// For C = max(A, B) where A ∈ [A_l, A_u] and B ∈ [B_l, B_u]:
    /// - C_lower = max(A_l, B_l)
    /// - C_upper = max(A_u, B_u)
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Use NaN-propagating max so NaN bounds poison rather than silently vanish (#2577).
        elementwise_binary_ibp(input_a, input_b, nan_propagating_max, "MaxBinary")
    }

    /// CROWN backward propagation for `z = max(x, y)` using the exact convex
    /// hull of `max` over the input box.
    ///
    /// `max` is convex and piecewise-linear, so it admits sound linear
    /// envelopes (see `minmax_relax`). Returns `(bounds_for_a, bounds_for_b)`
    /// following the `MulBinary` split convention: the relaxation constant is
    /// carried entirely in `bounds_a`'s bias channel, `bounds_b`'s bias is zero.
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(LinearBounds, LinearBounds)> {
        propagate_minmax_linear_binary(
            bounds,
            input_a_bounds,
            input_b_bounds,
            "MaxBinary",
            max_envelope,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};
    use ny_core::NyError;

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
    fn test_ibp_a_dominates() {
        // A bounds entirely above B bounds
        let layer = MaxBinaryLayer;
        let a = make_bt(&[5.0, 6.0], &[10.0, 12.0]);
        let b = make_bt(&[1.0, 2.0], &[3.0, 4.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // max(A_l, B_l) = A_l, max(A_u, B_u) = A_u
        assert!((result.lower()[0] - 5.0).abs() < 1e-5);
        assert!((result.lower()[1] - 6.0).abs() < 1e-5);
        assert!((result.upper()[0] - 10.0).abs() < 1e-5);
        assert!((result.upper()[1] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_b_dominates() {
        let layer = MaxBinaryLayer;
        let a = make_bt(&[1.0], &[3.0]);
        let b = make_bt(&[5.0], &[10.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert!((result.lower()[0] - 5.0).abs() < 1e-5);
        assert!((result.upper()[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_mixed() {
        // A lower > B lower, but B upper > A upper
        let layer = MaxBinaryLayer;
        let a = make_bt(&[3.0], &[5.0]);
        let b = make_bt(&[1.0], &[8.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // max(3,1) = 3, max(5,8) = 8
        assert!((result.lower()[0] - 3.0).abs() < 1e-5);
        assert!((result.upper()[0] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_negative_values() {
        let layer = MaxBinaryLayer;
        let a = make_bt(&[-5.0, -3.0], &[-1.0, 0.0]);
        let b = make_bt(&[-4.0, -6.0], &[-2.0, -1.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // max(-5,-4)=-4, max(-3,-6)=-3, max(-1,-2)=-1, max(0,-1)=0
        assert!((result.lower()[0] - (-4.0)).abs() < 1e-5);
        assert!((result.lower()[1] - (-3.0)).abs() < 1e-5);
        assert!((result.upper()[0] - (-1.0)).abs() < 1e-5);
        assert!((result.upper()[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_broadcast() {
        // A [1,3], B [2,3]
        let layer = MaxBinaryLayer;
        let a = make_bt_shape(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], &[1, 3]);
        let b = make_bt_shape(
            vec![0.0, 3.0, 1.0, 5.0, 1.0, 2.0],
            vec![2.0, 4.0, 7.0, 6.0, 3.0, 8.0],
            &[2, 3],
        );
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        // Row 0: max([1,2,3],[0,3,1]) = [1,3,3]; max([4,5,6],[2,4,7]) = [4,5,7]
        assert!((result.lower()[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((result.lower()[[0, 1]] - 3.0).abs() < 1e-5);
        assert!((result.upper()[[0, 2]] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_shape_mismatch() {
        let layer = MaxBinaryLayer;
        let a = make_bt(&[1.0, 2.0], &[3.0, 4.0]);
        let b = make_bt(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let err = layer
            .propagate_ibp_binary(&a, &b)
            .expect_err("incompatible shapes");
        assert!(matches!(err, NyError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_ibp_soundness_sampling() {
        let layer = MaxBinaryLayer;
        let a = make_bt(&[-3.0, 0.0], &[2.0, 5.0]);
        let b = make_bt(&[-1.0, -2.0], &[4.0, 3.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
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
}
