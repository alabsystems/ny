// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, ArrayViewD, IxDyn, Zip};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::shape::broadcast_shapes;
use crate::LinearBounds;

/// Binary division layer: computes C = A / B for two bounded inputs.
///
/// This is used when neither input is a constant (e.g., (x - mean(x)) / sqrt(var + eps) in LayerNorm).
/// Requires B > 0 (strictly positive divisor) for valid bounds.
///
/// For A ∈ [A_l, A_u] and B ∈ [B_l, B_u] where B_l > 0:
/// f(b) = a/b has sign-dependent monotonicity in b:
/// - a > 0: f decreasing in b → min at B_u, max at B_l
/// - a < 0: f increasing in b → min at B_l, max at B_u
///   Element-wise bounds by case:
/// - A_l >= 0: C_lower = A_l/B_u, C_upper = A_u/B_l
/// - A_u <= 0: C_lower = A_l/B_l, C_upper = A_u/B_u
/// - Mixed:    C_lower = A_l/B_l, C_upper = A_u/B_l
#[derive(Debug, Clone)]
pub struct DivLayer;

/// Compute element-wise sound division bounds for A/B where B > 0.
///
/// # Safety invariant
/// The caller must validate finite, well-formed divisor intervals
/// (`0 < b_lower <= b_upper`) before calling.
///
/// Monotonicity of f(b) = a/b reverses with sign of a, so we must
/// pick the correct B endpoint per element based on the sign of A.
fn div_bounds_elementwise(
    a_lower: &ArrayViewD<f32>,
    a_upper: &ArrayViewD<f32>,
    b_lower: &ArrayViewD<f32>,
    b_upper: &ArrayViewD<f32>,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut out_lower = ArrayD::zeros(IxDyn(a_lower.shape()));
    let mut out_upper = ArrayD::zeros(IxDyn(a_lower.shape()));

    Zip::from(&mut out_lower)
        .and(&mut out_upper)
        .and(a_lower)
        .and(a_upper)
        .and(b_lower)
        .and(b_upper)
        .for_each(|ol, ou, &al, &au, &bl, &bu| {
            if al >= 0.0 {
                // Both bounds non-negative: f(b) = a/b is decreasing in b
                // Directed rounding: lower bound rounds down, upper bound rounds up (#2855)
                *ol = next_down_f32(al / bu);
                *ou = next_up_f32(au / bl);
            } else if au <= 0.0 {
                // Both bounds non-positive: f(b) = a/b is increasing in b
                *ol = next_down_f32(al / bl);
                *ou = next_up_f32(au / bu);
            } else {
                // Mixed sign: al < 0 < au
                // Lower: most negative a / smallest b (largest magnitude negative)
                // Upper: most positive a / smallest b (largest magnitude positive)
                *ol = next_down_f32(al / bl);
                *ou = next_up_f32(au / bl);
            }
        });

    (out_lower, out_upper)
}

fn validate_positive_divisor_bounds(
    b_lower: &ArrayViewD<f32>,
    b_upper: &ArrayViewD<f32>,
) -> Result<()> {
    if let Some((flat_idx, (&bl, &bu))) =
        b_lower
            .iter()
            .zip(b_upper.iter())
            .enumerate()
            .find(|(_, (&bl, &bu))| {
                !bl.is_finite() || !bu.is_finite() || bl <= 0.0 || bu <= 0.0 || bl > bu
            })
    {
        return Err(NyError::InvalidSpec(format!(
            "DivLayer requires finite divisor bounds with 0 < lower <= upper; got lower={bl}, upper={bu} at flat index {flat_idx}"
        )));
    }

    Ok(())
}

impl DivLayer {
    /// Propagate IBP bounds through element-wise division.
    ///
    /// Assumes divisor is strictly positive (B_l > 0).
    /// For LayerNorm: divisor is sqrt(var + eps) which is always positive.
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Handle broadcasting
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

            validate_positive_divisor_bounds(&b_lower, &b_upper)?;

            let (out_lower, out_upper) =
                div_bounds_elementwise(&a_lower, &a_upper, &b_lower, &b_upper);
            return BoundedTensor::new(out_lower, out_upper);
        };

        validate_positive_divisor_bounds(&b_lower, &b_upper)?;

        let (out_lower, out_upper) = div_bounds_elementwise(&a_lower, &a_upper, &b_lower, &b_upper);
        BoundedTensor::new(out_lower, out_upper)
    }

    /// CROWN backward propagation for Div is not implemented.
    ///
    /// Division is a nonlinear operation that doesn't have a simple linear relaxation.
    pub fn propagate_linear_binary(
        &self,
        _bounds: &LinearBounds,
    ) -> Result<(LinearBounds, LinearBounds)> {
        Err(NyError::UnsupportedOp(
            "Div CROWN propagation not implemented - use IBP".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

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
    fn test_ibp_positive_a_positive_b() {
        let layer = DivLayer;
        let a = make_bt(&[2.0, 4.0], &[6.0, 8.0]);
        let b = make_bt(&[1.0, 2.0], &[2.0, 4.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // a >= 0: C_l = A_l/B_u, C_u = A_u/B_l
        // elem 0: C_l = 2/2 = 1, C_u = 6/1 = 6
        // elem 1: C_l = 4/4 = 1, C_u = 8/2 = 4
        assert!((result.lower()[0] - 1.0).abs() < 1e-5);
        assert!((result.upper()[0] - 6.0).abs() < 1e-5);
        assert!((result.lower()[1] - 1.0).abs() < 1e-5);
        assert!((result.upper()[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_negative_a_positive_b() {
        let layer = DivLayer;
        let a = make_bt(&[-6.0], &[-2.0]);
        let b = make_bt(&[1.0], &[3.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // a <= 0: C_l = A_l/B_l = -6/1 = -6, C_u = A_u/B_u = -2/3 ≈ -0.667
        assert!((result.lower()[0] - (-6.0)).abs() < 1e-5);
        assert!((result.upper()[0] - (-2.0 / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn test_ibp_mixed_sign_a() {
        let layer = DivLayer;
        let a = make_bt(&[-3.0], &[5.0]);
        let b = make_bt(&[2.0], &[4.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // Mixed: C_l = A_l/B_l = -3/2 = -1.5, C_u = A_u/B_l = 5/2 = 2.5
        assert!((result.lower()[0] - (-1.5)).abs() < 1e-5);
        assert!((result.upper()[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_non_positive_divisor_error() {
        let layer = DivLayer;
        let a = make_bt(&[1.0], &[2.0]);
        let b = make_bt(&[-1.0], &[1.0]); // B_l <= 0
        let err = layer
            .propagate_ibp_binary(&a, &b)
            .expect_err("divisor must be positive");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    #[test]
    fn test_ibp_zero_lower_divisor_error() {
        let layer = DivLayer;
        let a = make_bt(&[1.0], &[2.0]);
        let b = make_bt(&[0.0], &[1.0]); // B_l = 0
        let err = layer
            .propagate_ibp_binary(&a, &b)
            .expect_err("divisor lower must be > 0");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    #[test]
    fn test_ibp_broadcast() {
        // A [1,2], B [2,2] with broadcasting
        let layer = DivLayer;
        let a = make_bt_shape(vec![2.0, 4.0], vec![6.0, 8.0], &[1, 2]);
        let b = make_bt_shape(vec![1.0, 2.0, 0.5, 1.0], vec![2.0, 4.0, 1.0, 2.0], &[2, 2]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_ibp_soundness_sampling() {
        let layer = DivLayer;
        let a = make_bt(&[-2.0, 1.0], &[3.0, 5.0]);
        let b = make_bt(&[0.5, 1.0], &[2.0, 3.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        // With directed rounding (#2855), bounds are guaranteed to contain all true
        // values — no tolerance needed.
        for k in 0..=20 {
            let t = k as f32 / 20.0;
            for j in 0..=20 {
                let s = j as f32 / 20.0;
                let a0 = -2.0 + 5.0 * t;
                let b0 = 0.5 + 1.5 * s;
                let a1 = 1.0 + 4.0 * t;
                let b1 = 1.0 + 2.0 * s;
                assert!(
                    a0 / b0 >= result.lower()[0],
                    "a0/b0={} < lower={}",
                    a0 / b0,
                    result.lower()[0]
                );
                assert!(
                    a0 / b0 <= result.upper()[0],
                    "a0/b0={} > upper={}",
                    a0 / b0,
                    result.upper()[0]
                );
                assert!(
                    a1 / b1 >= result.lower()[1],
                    "a1/b1={} < lower={}",
                    a1 / b1,
                    result.lower()[1]
                );
                assert!(
                    a1 / b1 <= result.upper()[1],
                    "a1/b1={} > upper={}",
                    a1 / b1,
                    result.upper()[1]
                );
            }
        }
    }

    /// Directed rounding regression: lower bound must be <= exact, upper must be >= exact.
    /// Uses non-representable result (2/3) to confirm rounding direction. (#2855)
    #[test]
    fn test_ibp_directed_rounding() {
        let layer = DivLayer;
        // 2.0 / 3.0 is not exactly representable in f32.
        // For lower bound: next_down_f32(2.0/3.0) < 2.0/3.0 (exact)
        // For upper bound: next_up_f32(2.0/3.0) > 2.0/3.0 (exact)
        let a = make_bt(&[2.0], &[2.0]); // point interval
        let b = make_bt(&[3.0], &[3.0]); // point interval
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        let exact = 2.0_f64 / 3.0;
        assert!(
            (result.lower()[0] as f64) < exact,
            "lower {} should be < exact {exact}",
            result.lower()[0]
        );
        assert!(
            (result.upper()[0] as f64) > exact,
            "upper {} should be > exact {exact}",
            result.upper()[0]
        );
        // Bounds should be within 1 ULP of each other
        assert!(
            (result.upper()[0] - result.lower()[0]) < 2e-7,
            "gap {} should be ~1 ULP",
            result.upper()[0] - result.lower()[0]
        );
    }

    // ── CROWN tests ───────────────────────────────────────────────────

    #[test]
    fn test_crown_not_supported() {
        let layer = DivLayer;
        let bounds = LinearBounds::identity(2);
        let err = layer
            .propagate_linear_binary(&bounds)
            .expect_err("div CROWN not implemented");
        assert!(matches!(err, NyError::UnsupportedOp(_)));
    }
}
