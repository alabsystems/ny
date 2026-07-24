// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::IxDyn;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use crate::shape::broadcast_shapes;
use crate::{BatchedLinearBounds, LinearBounds};

/// Element-wise addition layer for two bounded tensors (e.g., residual connections).
#[derive(Debug, Clone)]
pub struct AddLayer;

impl AddLayer {
    /// Propagate IBP bounds through element-wise addition.
    ///
    /// For C = A + B where A ∈ [A_l, A_u] and B ∈ [B_l, B_u]:
    /// C ∈ [A_l + B_l, A_u + B_u]
    ///
    /// Supports full NumPy/ONNX broadcasting (e.g., [1, 3] + [3, 1] → [3, 3],
    /// [5, 1, 48] + [5, 48] → [5, 1, 48]). Matches SubLayer behavior.
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let (out_lower, out_upper) = if input_a.shape() == input_b.shape() {
            // Fast path: shapes match exactly, no broadcasting needed.
            (
                input_a.lower() + input_b.lower(),
                input_a.upper() + input_b.upper(),
            )
        } else {
            // Full NumPy/ONNX broadcasting (matches SubLayer behavior).
            let target_shape =
                broadcast_shapes(input_a.shape(), input_b.shape()).ok_or_else(|| {
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

            (&a_lower + &b_lower, &a_upper + &b_upper)
        };

        // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #2549).
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    /// CROWN backward propagation for Add (C = A + B).
    ///
    /// For Add, the Jacobian w.r.t. both inputs is the identity:
    /// - ∂C/∂A = I (identity)
    /// - ∂C/∂B = I (identity)
    ///
    /// So incoming linear bounds on C pass through unchanged to both A and B.
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
    ) -> Result<(LinearBounds, LinearBounds)> {
        // C = A + B => W·C + b = W·A + W·B + b.
        // Split the bias so graph accumulation does not double-count constants.
        // Directed rounding on f32 halving to preserve soundness (#2173).
        let lower_b_half = bounds.lower_b().mapv(|v| next_down_f32(v * 0.5));
        let upper_b_half = bounds.upper_b().mapv(|v| next_up_f32(v * 0.5));

        let bounds_a = LinearBounds::new_or_conservative(
            bounds.lower_a().clone(),
            lower_b_half.clone(),
            bounds.upper_a().clone(),
            upper_b_half.clone(),
        )?;

        let bounds_b = LinearBounds::new_or_conservative(
            bounds.lower_a().clone(),
            lower_b_half,
            bounds.upper_a().clone(),
            upper_b_half,
        )?;

        Ok((bounds_a, bounds_b))
    }

    /// Batched CROWN backward propagation for Add (C = A + B).
    ///
    /// Same logic as `propagate_linear_binary` but for N-D batched bounds.
    /// For Add, the Jacobian w.r.t. both inputs is the identity:
    /// - ∂C/∂A = I (identity)
    /// - ∂C/∂B = I (identity)
    ///
    /// So incoming batched linear bounds on C pass through unchanged to both A and B.
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_batched_binary(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
        // C = A + B => W·C + b = W·A + W·B + b.
        // Split the bias so graph accumulation does not double-count constants.
        // Directed rounding on f32 halving to preserve soundness (#2173).
        let lower_b_half = bounds.lower_b.mapv(|v| next_down_f32(v * 0.5));
        let upper_b_half = bounds.upper_b.mapv(|v| next_up_f32(v * 0.5));

        // Phase 4 audit: per-layer passthrough + bias halving.
        let bounds_a = BatchedLinearBounds::new_or_conservative(
            bounds.lower_a.clone(),
            lower_b_half.clone(),
            bounds.upper_a.clone(),
            upper_b_half.clone(),
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )?;

        let bounds_b = BatchedLinearBounds::new_or_conservative(
            bounds.lower_a.clone(),
            lower_b_half,
            bounds.upper_a.clone(),
            upper_b_half,
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )?;

        Ok((bounds_a, bounds_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, ArrayD, IxDyn};

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

    // ── IBP tests ──────────────────────────────────────────────────────

    #[test]
    fn test_ibp_same_shape() {
        let layer = AddLayer;
        let a = make_bt(&[1.0, -2.0], &[3.0, 4.0]);
        let b = make_bt(&[0.5, 1.0], &[1.5, 2.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // C_l = A_l + B_l = [1.5, -1.0], C_u = A_u + B_u = [4.5, 6.0]
        assert!((result.lower()[0] - 1.5).abs() < 1e-5);
        assert!((result.lower()[1] - (-1.0)).abs() < 1e-5);
        assert!((result.upper()[0] - 4.5).abs() < 1e-5);
        assert!((result.upper()[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_negative_bounds() {
        let layer = AddLayer;
        let a = make_bt(&[-5.0, -3.0], &[-1.0, -0.5]);
        let b = make_bt(&[-2.0, -1.0], &[-0.5, 0.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // C_l = [-7.0, -4.0], C_u = [-1.5, -0.5]
        assert!((result.lower()[0] - (-7.0)).abs() < 1e-5);
        assert!((result.lower()[1] - (-4.0)).abs() < 1e-5);
        assert!((result.upper()[0] - (-1.5)).abs() < 1e-5);
        assert!((result.upper()[1] - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_scalar_broadcast_a() {
        // A is scalar [1], B is [3]
        let layer = AddLayer;
        let a = make_bt(&[2.0], &[3.0]);
        let b = make_bt(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // C_l = [3.0, 4.0, 5.0], C_u = [7.0, 8.0, 9.0]
        assert_eq!(result.shape(), &[3]);
        assert!((result.lower()[0] - 3.0).abs() < 1e-5);
        assert!((result.lower()[2] - 5.0).abs() < 1e-5);
        assert!((result.upper()[0] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_scalar_broadcast_b() {
        let layer = AddLayer;
        let a = make_bt(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let b = make_bt(&[10.0], &[20.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert!((result.lower()[0] - 11.0).abs() < 1e-5);
        assert!((result.upper()[2] - 26.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_broadcast() {
        // A shape [2, 1, 3], B shape [2, 3] — NumPy broadcasts to [2, 2, 3]
        // [2,1,3] pads to [2,1,3], [2,3] pads to [1,2,3], broadcast → [2,2,3]
        let layer = AddLayer;
        let a = make_bt_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            &[2, 1, 3],
        );
        let b = make_bt_shape(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            &[2, 3],
        );
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2, 3]);
        // a[0,0,:] = [1,2,3], b broadcast [0,:] = [0.1,0.2,0.3]
        // result[0,0,0] = 1.0 + 0.1 = 1.1
        assert!((result.lower()[[0, 0, 0]] - 1.1).abs() < 1e-4);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_ibp_broadcast_transpose_shapes() {
        // [1, 3] + [3, 1] broadcasts to [3, 3] — the lsnc model pattern (#411)
        let layer = AddLayer;
        let a = make_bt_shape(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], &[1, 3]);
        let b = make_bt_shape(vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0], &[3, 1]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        // result[0,0] = a[0,0] + b[0,0] = 1.0 + 10.0 = 11.0
        assert!((result.lower()[[0, 0]] - 11.0).abs() < 1e-4);
        // result[1,2] = a[0,2] + b[1,0] = 3.0 + 20.0 = 23.0
        assert!((result.lower()[[1, 2]] - 23.0).abs() < 1e-4);
        // upper: result[2,1] = a[0,1] + b[2,0] = 5.0 + 60.0 = 65.0
        assert!((result.upper()[[2, 1]] - 65.0).abs() < 1e-4);
    }

    #[test]
    fn test_ibp_shape_mismatch_error() {
        let layer = AddLayer;
        let a = make_bt(&[1.0, 2.0], &[3.0, 4.0]);
        let b = make_bt(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let err = layer
            .propagate_ibp_binary(&a, &b)
            .expect_err("should fail for incompatible shapes");
        assert!(matches!(err, NyError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_ibp_soundness_sampling() {
        // Verify at sampled points that A+B is within [C_l, C_u]
        let layer = AddLayer;
        let a = make_bt(&[-3.0, 0.0], &[2.0, 5.0]);
        let b = make_bt(&[-1.0, -4.0], &[4.0, 1.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        for k in 0..=20 {
            let t = k as f32 / 20.0;
            let a0 = -3.0 + 5.0 * t;
            let a1 = 0.0 + 5.0 * t;
            let b0 = -1.0 + 5.0 * t;
            let b1 = -4.0 + 5.0 * t;
            assert!(a0 + b0 >= result.lower()[0] - 1e-5);
            assert!(a0 + b0 <= result.upper()[0] + 1e-5);
            assert!(a1 + b1 >= result.lower()[1] - 1e-5);
            assert!(a1 + b1 <= result.upper()[1] + 1e-5);
        }
    }

    // ── CROWN backward tests ──────────────────────────────────────────

    #[test]
    fn test_crown_identity_bounds() {
        let layer = AddLayer;
        let bounds = LinearBounds::identity(2);
        let (ba, bb) = layer.propagate_linear_binary(&bounds).unwrap();
        // Coefficients should be same for both branches (identity)
        assert!((ba.lower_a[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((ba.lower_a[[1, 1]] - 1.0).abs() < 1e-5);
        assert!((bb.lower_a[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((bb.lower_a[[1, 1]] - 1.0).abs() < 1e-5);
        // Bias halved
        assert!(ba.lower_b[0].abs() < 1e-5);
        assert!(bb.lower_b[0].abs() < 1e-5);
    }

    #[test]
    fn test_crown_bias_splitting() {
        // Bias should be halved between branches to avoid double-counting
        let layer = AddLayer;
        let bounds = LinearBounds::new(
            Array2::eye(2),
            Array1::from_vec(vec![4.0, 6.0]),
            Array2::eye(2),
            Array1::from_vec(vec![10.0, 12.0]),
        )
        .unwrap();
        let (ba, bb) = layer.propagate_linear_binary(&bounds).unwrap();
        // Halved: lower_b = [2.0, 3.0], upper_b = [5.0, 6.0]
        assert!((ba.lower_b[0] - 2.0).abs() < 1e-5);
        assert!((ba.lower_b[1] - 3.0).abs() < 1e-5);
        assert!((bb.lower_b[0] - 2.0).abs() < 1e-5);
        assert!((bb.lower_b[1] - 3.0).abs() < 1e-5);
        assert!((ba.upper_b[0] - 5.0).abs() < 1e-5);
        assert!((bb.upper_b[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_crown_bias_splitting_directed_rounding_subnormal_underflow() {
        // Regression for #2173: when halving underflows to 0.0, directed rounding
        // must widen in the safe direction.
        let layer = AddLayer;
        let tiny = f32::from_bits(1); // smallest positive subnormal
        let bounds = LinearBounds::new(
            Array2::eye(1),
            Array1::from_vec(vec![tiny]),
            Array2::eye(1),
            Array1::from_vec(vec![tiny]),
        )
        .unwrap();

        let (ba, bb) = layer.propagate_linear_binary(&bounds).unwrap();
        let expected_lower = next_down_f32(tiny * 0.5);
        let expected_upper = next_up_f32(tiny * 0.5);

        assert_eq!(ba.lower_b[0], expected_lower);
        assert_eq!(bb.lower_b[0], expected_lower);
        assert_eq!(ba.upper_b[0], expected_upper);
        assert_eq!(bb.upper_b[0], expected_upper);
    }

    #[test]
    fn test_crown_non_identity_coefficients() {
        // Non-identity incoming: W = [[2, -1], [0, 3]]
        let layer = AddLayer;
        let w = Array2::from_shape_vec((2, 2), vec![2.0, -1.0, 0.0, 3.0]).unwrap();
        let bounds = LinearBounds::new(w.clone(), Array1::zeros(2), w, Array1::zeros(2)).unwrap();
        let (ba, bb) = layer.propagate_linear_binary(&bounds).unwrap();
        // Both branches get same W matrix
        assert!((ba.lower_a[[0, 0]] - 2.0).abs() < 1e-5);
        assert!((ba.lower_a[[0, 1]] - (-1.0)).abs() < 1e-5);
        assert!((bb.lower_a[[0, 0]] - 2.0).abs() < 1e-5);
        assert!((bb.lower_a[[1, 1]] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_crown_batched_bias_splitting() {
        let layer = AddLayer;
        let bounds = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::from_elem(IxDyn(&[2, 2]), 1.0_f32),
            ArrayD::from_elem(IxDyn(&[2]), 4.0_f32),
            ArrayD::from_elem(IxDyn(&[2, 2]), 1.0_f32),
            ArrayD::from_elem(IxDyn(&[2]), 8.0_f32),
            vec![2],
            vec![2],
        );
        let (ba, bb) = layer.propagate_linear_batched_binary(&bounds).unwrap();
        // Bias halved
        assert!((ba.lower_b[[0]] - 2.0).abs() < 1e-5);
        assert!((bb.lower_b[[0]] - 2.0).abs() < 1e-5);
        assert!((ba.upper_b[[0]] - 4.0).abs() < 1e-5);
        assert!((bb.upper_b[[0]] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_crown_soundness_concretization() {
        // Verify: for C = A + B, concretizing CROWN bounds over A and B
        // should contain the IBP result.
        let layer = AddLayer;
        let bounds = LinearBounds::identity(2);
        let (ba, bb) = layer.propagate_linear_binary(&bounds).unwrap();

        let a = make_bt(&[-1.0, 2.0], &[3.0, 5.0]);
        let b = make_bt(&[0.0, -1.0], &[2.0, 4.0]);

        // Concretize: lower = min(a_coeff * x_range) + min(b_coeff * x_range) + bias_a + bias_b
        // For identity coefficients, this simplifies to:
        // lower_from_a = A_l + bias_a, upper_from_a = A_u + bias_a
        // lower_from_b = B_l + bias_b, upper_from_b = B_u + bias_b
        // Combined lower = lower_from_a + lower_from_b, combined upper = upper_from_a + upper_from_b
        let combined_lower_0 =
            (ba.lower_a[[0, 0]] * a.lower()[0] + ba.lower_a[[0, 1]] * a.lower()[1] + ba.lower_b[0])
                + (bb.lower_a[[0, 0]] * b.lower()[0]
                    + bb.lower_a[[0, 1]] * b.lower()[1]
                    + bb.lower_b[0]);

        let ibp_result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert!(combined_lower_0 <= ibp_result.lower()[0] + 1e-4);
    }
}
