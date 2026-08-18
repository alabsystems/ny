// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::IxDyn;
use ny_core::{NyError, Result};
use ny_tensor::{
    next_down_f32, next_up_f32, sub_down_f32, sub_up_f32, BoundedTensor, RepairStrategy,
};

use crate::shape::broadcast_shapes;
use crate::LinearBounds;

/// Binary subtraction layer: computes C = A - B for two bounded inputs.
///
/// This is used when neither input is a constant (e.g., x - mean(x) in LayerNorm).
/// For A ∈ [A_l, A_u] and B ∈ [B_l, B_u]:
/// C ∈ [A_l - B_u, A_u - B_l]
#[derive(Debug, Clone)]
pub struct SubLayer;

impl SubLayer {
    /// Propagate IBP bounds through element-wise subtraction.
    ///
    /// Both endpoint subtractions are DIRECTED (`sub_down_f32` / `sub_up_f32`)
    /// for the same reason as [`super::add::AddLayer::propagate_ibp_binary`]:
    /// a plain f32 `-` is round-to-nearest and can move an endpoint INWARD by
    /// up to half an ULP, producing an interval that excludes the true
    /// difference. The directed forms are exact whenever the subtraction is.
    ///
    /// For C = A - B where A ∈ [A_l, A_u] and B ∈ [B_l, B_u]:
    /// - C_lower = A_l - B_u (minimize A, maximize B)
    /// - C_upper = A_u - B_l (maximize A, minimize B)
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Handle broadcasting for common cases (e.g., x - mean where mean is reduced)
        let (a_lower, a_upper, b_lower, b_upper) = if input_a.shape() == input_b.shape() {
            (
                input_a.lower().view(),
                input_a.upper().view(),
                input_b.lower().view(),
                input_b.upper().view(),
            )
        } else {
            // Try broadcasting
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

            let out_lower = ndarray::Zip::from(&a_lower)
                .and(&b_upper)
                .map_collect(|&x, &y| sub_down_f32(x, y));
            let out_upper = ndarray::Zip::from(&a_upper)
                .and(&b_lower)
                .map_collect(|&x, &y| sub_up_f32(x, y));
            // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #2742).
            return BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative);
        };

        let out_lower = ndarray::Zip::from(&a_lower)
            .and(&b_upper)
            .map_collect(|&x, &y| sub_down_f32(x, y));
        let out_upper = ndarray::Zip::from(&a_upper)
            .and(&b_lower)
            .map_collect(|&x, &y| sub_up_f32(x, y));
        // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #2742).
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    /// CROWN backward propagation for Sub (C = A - B).
    ///
    /// For Sub, the Jacobians are:
    /// - ∂C/∂A = I (identity)
    /// - ∂C/∂B = -I (negative identity)
    ///
    /// So linear bounds on A pass through unchanged, but bounds on B are negated.
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
    ) -> Result<(LinearBounds, LinearBounds)> {
        // C = A - B => W·C + b = W·A - W·B + b.
        // Split the bias for graph accumulation.
        // Directed rounding on f32 halving to preserve soundness (#2173).
        let lower_b_half = bounds.lower_b().mapv(|v| next_down_f32(v * 0.5));
        let upper_b_half = bounds.upper_b().mapv(|v| next_up_f32(v * 0.5));

        // Bounds for A pass through unchanged
        let bounds_a = LinearBounds::new_or_conservative(
            bounds.lower_a().clone(),
            lower_b_half.clone(),
            bounds.upper_a().clone(),
            upper_b_half.clone(),
        )?;

        // Bounds for B are NEGATED, with NO lower/upper swap.
        //
        // CROWN composes by substitution, so each relation keeps its own
        // coefficients: from `obj >= lower_a·C + lower_b` and `C = A - B`,
        //   obj >= lower_a·A + (-lower_a)·B + lower_b,
        // so B's LOWER coefficient is `-lower_a` — not `-upper_a`. The same for
        // the upper relation with `upper_a`. `sub_constant.rs:82` states this
        // rule explicitly ("negates A'; no lower/upper swap").
        //
        // Swapping is the rule for negating a bounded QUANTITY
        // (`l <= x <= u` gives `-u <= -x <= -l`), not for negating the
        // coefficients of a linear relation, and it is not conservative: with a
        // downstream relaxation making `lower_a != upper_a`, the swapped form
        // produced a FALSE bound whenever B could be negative. Reproduced on
        // `relu(u - v)` with `u = x`, `v = -x`, `x in [-1, 3]`: CROWN returned
        // an upper bound of 4.256 while `relu(2·2.2) = 4.4` is reachable.
        let bounds_b = LinearBounds::new_or_conservative(
            -bounds.lower_a(),
            lower_b_half,
            -bounds.upper_a(),
            upper_b_half,
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
        let layer = SubLayer;
        let a = make_bt(&[1.0, 2.0], &[5.0, 6.0]);
        let b = make_bt(&[0.5, 1.0], &[2.0, 3.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // C_l = A_l - B_u = [1-2, 2-3] = [-1, -1]
        // C_u = A_u - B_l = [5-0.5, 6-1] = [4.5, 5.0]
        assert!((result.lower()[0] - (-1.0)).abs() < 1e-5);
        assert!((result.lower()[1] - (-1.0)).abs() < 1e-5);
        assert!((result.upper()[0] - 4.5).abs() < 1e-5);
        assert!((result.upper()[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_negative_bounds() {
        let layer = SubLayer;
        let a = make_bt(&[-2.0], &[3.0]);
        let b = make_bt(&[-4.0], &[1.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // C_l = -2 - 1 = -3, C_u = 3 - (-4) = 7
        assert!((result.lower()[0] - (-3.0)).abs() < 1e-5);
        assert!((result.upper()[0] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_broadcast() {
        // A shape [1, 3], B shape [2, 3]
        let layer = SubLayer;
        let a = make_bt_shape(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], &[1, 3]);
        let b = make_bt_shape(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            &[2, 3],
        );
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        // C_l[0,0] = A_l[0,0] - B_u[0,0] = 1.0 - 0.5 = 0.5
        assert!((result.lower()[[0, 0]] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_ibp_shape_mismatch() {
        let layer = SubLayer;
        let a = make_bt(&[1.0, 2.0], &[3.0, 4.0]);
        let b = make_bt(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let err = layer
            .propagate_ibp_binary(&a, &b)
            .expect_err("incompatible shapes");
        assert!(matches!(err, NyError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_ibp_soundness_sampling() {
        let layer = SubLayer;
        let a = make_bt(&[-3.0, 1.0], &[2.0, 5.0]);
        let b = make_bt(&[-1.0, 0.0], &[4.0, 3.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        for k in 0..=20 {
            let t = k as f32 / 20.0;
            let a0 = -3.0 + 5.0 * t;
            let a1 = 1.0 + 4.0 * t;
            let b0 = -1.0 + 5.0 * t;
            let b1 = 0.0 + 3.0 * t;
            assert!(
                a0 - b0 >= result.lower()[0] - 1e-5,
                "a0-b0={} < lower={}",
                a0 - b0,
                result.lower()[0]
            );
            assert!(a0 - b0 <= result.upper()[0] + 1e-5);
            assert!(a1 - b1 >= result.lower()[1] - 1e-5);
            assert!(a1 - b1 <= result.upper()[1] + 1e-5);
        }
    }

    // ── CROWN backward tests ──────────────────────────────────────────

    #[test]
    fn test_crown_identity_bounds() {
        let layer = SubLayer;
        let bounds = LinearBounds::identity(2);
        let (ba, bb) = layer.propagate_linear_binary(&bounds).unwrap();
        // Bounds for A: coefficients unchanged
        assert!((ba.lower_a[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((ba.upper_a[[0, 0]] - 1.0).abs() < 1e-5);
        // Bounds for B: negated, NOT swapped (lower_a_b = -lower_a, upper_a_b = -upper_a)
        assert!((bb.lower_a[[0, 0]] - (-1.0)).abs() < 1e-5);
        assert!((bb.upper_a[[0, 0]] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_crown_negation_no_swap() {
        // Non-identity W to check negation WITHOUT a lower/upper swap: CROWN
        // composes `C = A - B` by substitution, so each relation keeps its own
        // coefficients negated (swapping produced a false bound, see
        // `propagate_linear_binary`).
        let layer = SubLayer;
        let w = Array2::from_shape_vec((2, 2), vec![2.0, -1.0, 0.0, 3.0]).unwrap();
        let bounds = LinearBounds::new(
            w,
            Array1::from_vec(vec![1.0, 2.0]),
            Array2::from_shape_vec((2, 2), vec![4.0, 0.0, 1.0, 5.0]).unwrap(),
            Array1::from_vec(vec![3.0, 4.0]),
        )
        .unwrap();
        let (_ba, bb) = layer.propagate_linear_binary(&bounds).unwrap();
        // bb.lower_a = -bounds.lower_a = [[-2, 1], [0, -3]]
        assert!((bb.lower_a[[0, 0]] - (-2.0)).abs() < 1e-5);
        assert!((bb.lower_a[[0, 1]] - 1.0).abs() < 1e-5);
        assert!((bb.lower_a[[1, 0]] - 0.0).abs() < 1e-5);
        assert!((bb.lower_a[[1, 1]] - (-3.0)).abs() < 1e-5);
        // bb.upper_a = -bounds.upper_a = [[-4, 0], [-1, -5]]
        assert!((bb.upper_a[[0, 0]] - (-4.0)).abs() < 1e-5);
        assert!((bb.upper_a[[0, 1]] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_crown_bias_splitting() {
        let layer = SubLayer;
        let bounds = LinearBounds::new(
            Array2::eye(2),
            Array1::from_vec(vec![6.0, 8.0]),
            Array2::eye(2),
            Array1::from_vec(vec![10.0, 12.0]),
        )
        .unwrap();
        let (ba, bb) = layer.propagate_linear_binary(&bounds).unwrap();
        // Bias halved
        assert!((ba.lower_b[0] - 3.0).abs() < 1e-5);
        assert!((bb.lower_b[0] - 3.0).abs() < 1e-5);
        assert!((ba.upper_b[0] - 5.0).abs() < 1e-5);
        assert!((bb.upper_b[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_crown_bias_splitting_directed_rounding_subnormal_underflow() {
        // Regression for #2173: halving smallest subnormal underflows to 0.0,
        // so directed rounding must widen to keep bounds sound.
        let layer = SubLayer;
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
    fn test_ibp_infinite_bounds_repaired() {
        // Regression for #2742: NaN subtraction results (inf - inf) must be
        // widened to ±inf, and ±inf endpoints pass through unchanged — a
        // non-finite endpoint proves nothing, so no finite substitute is sound
        // (#3423).
        let layer = SubLayer;
        // Element 0: conservative [-inf, +inf] — produces -inf and +inf (not NaN).
        // Element 1: finite arithmetic.
        // Element 2: degenerate [inf, inf] - [inf, inf] — produces inf-inf = NaN.
        let a = BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, 1.0, f32::INFINITY])
                .unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, 3.0, f32::INFINITY]).unwrap(),
        )
        .unwrap();
        let b = BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, -1.0, f32::INFINITY])
                .unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, 2.0, f32::INFINITY]).unwrap(),
        )
        .unwrap();

        let result = layer.propagate_ibp_binary(&a, &b).unwrap();

        // Element 0: (-inf) - (inf) = -inf → preserved (no proven lower bound).
        //            (inf) - (-inf) = inf → preserved (no proven upper bound).
        assert_eq!(result.lower()[0], f32::NEG_INFINITY);
        assert_eq!(result.upper()[0], f32::INFINITY);
        // Element 1: finite arithmetic is untouched (1.0 - 2.0 = -1.0, 3.0 - (-1.0) = 4.0).
        assert!((result.lower()[1] - (-1.0)).abs() < 1e-5);
        assert!((result.upper()[1] - 4.0).abs() < 1e-5);
        // Element 2: inf - inf = NaN → widened to ±inf.
        assert_eq!(result.lower()[2], f32::NEG_INFINITY);
        assert_eq!(result.upper()[2], f32::INFINITY);
    }

    #[test]
    fn test_ibp_infinite_bounds_broadcast_repaired() {
        // Regression for #2742: broadcast path also needs NaN repair (now via new_repaired, #3423).
        let layer = SubLayer;
        // A shape [1, 2], B shape [2, 2] — forces broadcast path.
        let a = BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![f32::NEG_INFINITY, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![f32::INFINITY, 3.0]).unwrap(),
        )
        .unwrap();
        let b = BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![f32::NEG_INFINITY, -1.0, 0.0, 0.5])
                .unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![f32::INFINITY, 2.0, 1.0, 1.5]).unwrap(),
        )
        .unwrap();

        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // [0,0]: (-inf) - (inf) = -inf, (inf) - (-inf) = inf → both preserved:
        // neither direction carries a proven bound.
        assert_eq!(result.lower()[[0, 0]], f32::NEG_INFINITY);
        assert_eq!(result.upper()[[0, 0]], f32::INFINITY);
        // [0,1]: finite: lower = 1.0 - 2.0 = -1.0, upper = 3.0 - (-1.0) = 4.0.
        assert!((result.lower()[[0, 1]] - (-1.0)).abs() < 1e-5);
        assert!((result.upper()[[0, 1]] - 4.0).abs() < 1e-5);
    }
}
