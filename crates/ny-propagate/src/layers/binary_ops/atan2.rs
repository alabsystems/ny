// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, ArrayViewD, IxDyn, Zip};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::atan2_relax::atan2_envelope;
use super::minmax_relax::{propagate_minmax_linear_binary, Envelope, Plane};
use crate::shape::broadcast_shapes;
use crate::LinearBounds;

/// Binary atan2 layer: computes C = atan2(A, B) for two bounded inputs.
///
/// This is the standard two-argument arctangent: atan2(y, x) returns the
/// angle in radians between the positive x-axis and the point (x, y), with
/// output in (-pi, pi].
///
/// For the Kokoro forward-STFT pipeline, A = Im (imaginary DFT component)
/// and B = Re (real DFT component).
///
/// IBP uses quadrant-aware interval arithmetic. CROWN is not implemented
/// (returns UnsupportedOp, like DivLayer).
#[derive(Debug, Clone)]
pub struct Atan2Layer;

#[inline]
fn full_angle_range() -> (f32, f32) {
    (-std::f32::consts::PI, std::f32::consts::PI)
}

#[inline]
fn atan2_bounds_scalar(y_lower: f32, y_upper: f32, x_lower: f32, x_upper: f32) -> (f32, f32) {
    let contains_origin = y_lower <= 0.0 && y_upper >= 0.0 && x_lower <= 0.0 && x_upper >= 0.0;
    // Soundness note: the branch-cut/origin handling follows the quadrant analysis in
    // designs/2026-03-20-issue-4223-atan2-phase-kokoro-stft.md.
    // y_upper >= 0.0 (not strict >): when y_upper = 0 and x < 0 the rectangle
    // touches the branch cut; interior points y → 0⁻ drive atan2 toward -π,
    // which corners miss because atan2(0, x_neg) = +π (the other side of the cut).
    let crosses_negative_real_axis = x_lower < 0.0 && y_lower < 0.0 && y_upper >= 0.0;

    if contains_origin || crosses_negative_real_axis {
        return full_angle_range();
    }

    let corners = [
        (y_lower as f64).atan2(x_lower as f64),
        (y_lower as f64).atan2(x_upper as f64),
        (y_upper as f64).atan2(x_lower as f64),
        (y_upper as f64).atan2(x_upper as f64),
    ];

    let min_corner = corners.iter().copied().fold(f64::INFINITY, f64::min);
    let max_corner = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (
        next_down_f32(min_corner as f32),
        next_up_f32(max_corner as f32),
    )
}

fn atan2_bounds_elementwise(
    y_lower: &ArrayViewD<f32>,
    y_upper: &ArrayViewD<f32>,
    x_lower: &ArrayViewD<f32>,
    x_upper: &ArrayViewD<f32>,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut out_lower = ArrayD::zeros(IxDyn(y_lower.shape()));
    let mut out_upper = ArrayD::zeros(IxDyn(y_lower.shape()));

    Zip::from(&mut out_lower)
        .and(&mut out_upper)
        .and(y_lower)
        .and(y_upper)
        .and(x_lower)
        .and(x_upper)
        .for_each(|ol, ou, &yl, &yu, &xl, &xu| {
            let (lower, upper) = atan2_bounds_scalar(yl, yu, xl, xu);
            *ol = lower;
            *ou = upper;
        });

    (out_lower, out_upper)
}

impl Atan2Layer {
    /// Propagate IBP bounds through element-wise atan2(y, x).
    pub fn propagate_ibp_binary(
        &self,
        input_y: &BoundedTensor,
        input_x: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let (y_lower, y_upper, x_lower, x_upper) = if input_y.shape() == input_x.shape() {
            (
                input_y.lower().view(),
                input_y.upper().view(),
                input_x.lower().view(),
                input_x.upper().view(),
            )
        } else {
            let target_shape =
                broadcast_shapes(input_y.shape(), input_x.shape()).ok_or_else(|| {
                    NyError::ShapeMismatch {
                        expected: input_y.shape().to_vec(),
                        got: input_x.shape().to_vec(),
                    }
                })?;

            let y_lower = input_y
                .lower()
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| NyError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_y.shape().to_vec(),
                })?;
            let y_upper = input_y
                .upper()
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| NyError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_y.shape().to_vec(),
                })?;
            let x_lower = input_x
                .lower()
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| NyError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_x.shape().to_vec(),
                })?;
            let x_upper = input_x
                .upper()
                .broadcast(IxDyn(&target_shape))
                .ok_or_else(|| NyError::ShapeMismatch {
                    expected: target_shape.clone(),
                    got: input_x.shape().to_vec(),
                })?;

            let (out_lower, out_upper) =
                atan2_bounds_elementwise(&y_lower, &y_upper, &x_lower, &x_upper);
            return BoundedTensor::new(out_lower, out_upper);
        };

        let (out_lower, out_upper) =
            atan2_bounds_elementwise(&y_lower, &y_upper, &x_lower, &x_upper);
        BoundedTensor::new(out_lower, out_upper)
    }

    /// CROWN backward propagation for `z = atan2(y, x)` using a sound
    /// mean-value linear envelope over the input box (see `atan2_relax`).
    ///
    /// `input_a = y` and `input_b = x` (the layer's operand order). The
    /// per-element envelope is only built for well-conditioned boxes (strictly
    /// inside one open quadrant, or strictly in the open right half plane); for
    /// any element near the origin or straddling the branch cut the shared
    /// driver returns `UnsupportedOp` and the caller keeps the sound IBP
    /// fallback for the whole op.
    ///
    /// Returns `(bounds_for_a, bounds_for_b)` following the `MulBinary`/`Min`/
    /// `Max` split convention: the relaxation constant is carried entirely in
    /// `bounds_a`'s bias channel.
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(LinearBounds, LinearBounds)> {
        // The shared driver labels its first axis "x" (coeff_x, bound to
        // input_a) and second "y" (coeff_y, bound to input_b). For atan2 the
        // layer's input_a is the angle's `y` and input_b is `x`, so we call
        // `atan2_envelope(x_box, y_box)` and SWAP the returned coefficients so
        // coeff_x carries d/dy (the input_a axis) and coeff_y carries d/dx.
        propagate_minmax_linear_binary(
            bounds,
            input_a_bounds,
            input_b_bounds,
            "Atan2",
            |a_lo, a_hi, b_lo, b_hi| {
                // a = y (input_a), b = x (input_b).
                let env = atan2_envelope(b_lo, b_hi, a_lo, a_hi)?;
                Some(Envelope {
                    lower: Plane {
                        coeff_x: env.lower.coeff_y, // d/dy -> input_a axis
                        coeff_y: env.lower.coeff_x, // d/dx -> input_b axis
                        c: env.lower.c,
                    },
                    upper: Plane {
                        coeff_x: env.upper.coeff_y,
                        coeff_y: env.upper.coeff_x,
                        c: env.upper.c,
                    },
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn make_bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        let n = lower.len();
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).expect("test: valid lower shape"),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).expect("test: valid upper shape"),
        )
        .expect("test: valid bounded tensor")
    }

    #[test]
    fn test_ibp_first_quadrant() {
        let layer = Atan2Layer;
        let y = make_bt(&[1.0], &[2.0]);
        let x = make_bt(&[3.0], &[4.0]);
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("IBP atan2 should succeed");

        let expected_lower = (1.0_f64).atan2(4.0_f64) as f32;
        let expected_upper = (2.0_f64).atan2(3.0_f64) as f32;
        assert!(result.lower()[0] <= expected_lower);
        assert!(result.upper()[0] >= expected_upper);
    }

    #[test]
    fn test_ibp_exact_zero_x_uses_vertical_angles() {
        let layer = Atan2Layer;
        let y = make_bt(&[1.0, -2.0], &[2.0, -1.0]);
        let x = make_bt(&[0.0, 0.0], &[0.0, 0.0]);
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("IBP atan2 should succeed");

        let half_pi = std::f32::consts::FRAC_PI_2;
        assert!(result.lower()[0] <= half_pi);
        assert!(result.upper()[0] >= half_pi);
        assert!(result.lower()[1] <= -half_pi);
        assert!(result.upper()[1] >= -half_pi);
    }

    #[test]
    fn test_ibp_branch_cut_crossing_returns_full_range() {
        let layer = Atan2Layer;
        let y = make_bt(&[-0.5], &[0.5]);
        let x = make_bt(&[-2.0], &[-1.0]);
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("IBP atan2 should succeed");

        let (expected_lower, expected_upper) = full_angle_range();
        assert_eq!(result.lower()[0], expected_lower);
        assert_eq!(result.upper()[0], expected_upper);
    }

    #[test]
    fn test_ibp_origin_containment_returns_full_range() {
        let layer = Atan2Layer;
        let y = make_bt(&[-1.0], &[1.0]);
        let x = make_bt(&[-1.0], &[1.0]);
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("IBP atan2 should succeed");

        let (expected_lower, expected_upper) = full_angle_range();
        assert_eq!(result.lower()[0], expected_lower);
        assert_eq!(result.upper()[0], expected_upper);
    }

    #[test]
    fn test_ibp_broadcasting() {
        let layer = Atan2Layer;
        let y = make_bt(&[1.0, -1.0], &[2.0, 0.0]);
        let x = make_bt(&[2.0], &[4.0]);
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("broadcasted atan2 should succeed");

        assert_eq!(result.shape(), &[2]);
        let expected_upper0 = (2.0_f64).atan2(2.0_f64) as f32;
        let expected_lower1 = (-1.0_f64).atan2(2.0_f64) as f32;
        assert!(result.upper()[0] >= expected_upper0);
        assert!(result.lower()[1] <= expected_lower1);
    }

    #[test]
    fn test_ibp_soundness_sampling() {
        let layer = Atan2Layer;
        let y = make_bt(&[0.25], &[1.75]);
        let x = make_bt(&[-3.0], &[-1.0]);
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("IBP atan2 should succeed");

        for y_sample in [0.25_f64, 0.5, 1.0, 1.5, 1.75] {
            for x_sample in [-3.0_f64, -2.5, -2.0, -1.5, -1.0] {
                let angle = y_sample.atan2(x_sample) as f32;
                assert!(
                    angle >= result.lower()[0],
                    "sample {angle} fell below lower bound {}",
                    result.lower()[0]
                );
                assert!(
                    angle <= result.upper()[0],
                    "sample {angle} exceeded upper bound {}",
                    result.upper()[0]
                );
            }
        }
    }

    /// Quadrant II: x < 0, y > 0 → atan2 in (π/2, π)
    #[test]
    fn test_ibp_second_quadrant() {
        let layer = Atan2Layer;
        let y = make_bt(&[1.0], &[3.0]);
        let x = make_bt(&[-4.0], &[-2.0]);
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("IBP atan2 should succeed");

        // All four corners should be in (π/2, π)
        let corners = [
            (1.0_f64).atan2(-4.0),
            (1.0_f64).atan2(-2.0),
            (3.0_f64).atan2(-4.0),
            (3.0_f64).atan2(-2.0),
        ];
        let min_corner = corners.iter().copied().fold(f64::INFINITY, f64::min) as f32;
        let max_corner = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max) as f32;
        assert!(
            result.lower()[0] <= min_corner,
            "lower {} should be <= min corner {min_corner}",
            result.lower()[0]
        );
        assert!(
            result.upper()[0] >= max_corner,
            "upper {} should be >= max corner {max_corner}",
            result.upper()[0]
        );
        // Verify we're actually in Q2
        assert!(result.lower()[0] > std::f32::consts::FRAC_PI_4);
    }

    /// Quadrant III: x < 0, y < 0 → atan2 in (-π, -π/2)
    #[test]
    fn test_ibp_third_quadrant() {
        let layer = Atan2Layer;
        let y = make_bt(&[-3.0], &[-1.0]);
        let x = make_bt(&[-4.0], &[-2.0]);
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("IBP atan2 should succeed");

        let corners = [
            (-3.0_f64).atan2(-4.0),
            (-3.0_f64).atan2(-2.0),
            (-1.0_f64).atan2(-4.0),
            (-1.0_f64).atan2(-2.0),
        ];
        let min_corner = corners.iter().copied().fold(f64::INFINITY, f64::min) as f32;
        let max_corner = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max) as f32;
        assert!(
            result.lower()[0] <= min_corner,
            "lower {} should be <= min corner {min_corner}",
            result.lower()[0]
        );
        assert!(
            result.upper()[0] >= max_corner,
            "upper {} should be >= max corner {max_corner}",
            result.upper()[0]
        );
        // Verify we're actually in Q3
        assert!(result.upper()[0] < -std::f32::consts::FRAC_PI_4);
    }

    /// Quadrant IV: x > 0, y < 0 → atan2 in (-π/2, 0)
    #[test]
    fn test_ibp_fourth_quadrant() {
        let layer = Atan2Layer;
        let y = make_bt(&[-2.0], &[-1.0]);
        let x = make_bt(&[3.0], &[4.0]);
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("IBP atan2 should succeed");

        let corners = [
            (-2.0_f64).atan2(3.0),
            (-2.0_f64).atan2(4.0),
            (-1.0_f64).atan2(3.0),
            (-1.0_f64).atan2(4.0),
        ];
        let min_corner = corners.iter().copied().fold(f64::INFINITY, f64::min) as f32;
        let max_corner = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max) as f32;
        assert!(
            result.lower()[0] <= min_corner,
            "lower {} should be <= min corner {min_corner}",
            result.lower()[0]
        );
        assert!(
            result.upper()[0] >= max_corner,
            "upper {} should be >= max corner {max_corner}",
            result.upper()[0]
        );
        // Verify we're actually in Q4
        assert!(result.lower()[0] < 0.0);
        assert!(result.upper()[0] < 0.0);
    }

    /// x > 0, y spans zero → atan2 is continuous, matches atan(y/x) range
    #[test]
    fn test_ibp_positive_x_only() {
        let layer = Atan2Layer;
        let y = make_bt(&[-1.0], &[1.0]);
        let x = make_bt(&[2.0], &[3.0]);
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("IBP atan2 should succeed");

        // For x > 0, atan2(y, x) = atan(y/x). Monotone in y, so:
        // lower = atan2(-1, ?) and upper = atan2(1, ?)
        let expected_lower = (-1.0_f64).atan2(2.0_f64) as f32; // most negative angle at smallest x
        let expected_upper = (1.0_f64).atan2(2.0_f64) as f32; // most positive angle at smallest x
        assert!(
            result.lower()[0] <= expected_lower,
            "lower {} should be <= expected {expected_lower}",
            result.lower()[0]
        );
        assert!(
            result.upper()[0] >= expected_upper,
            "upper {} should be >= expected {expected_upper}",
            result.upper()[0]
        );
        // Should NOT expand to full range since x > 0 throughout
        assert!(result.lower()[0] > -std::f32::consts::FRAC_PI_2);
        assert!(result.upper()[0] < std::f32::consts::FRAC_PI_2);
    }

    /// Directed rounding: bounds must be strictly wider than exact for non-representable results.
    /// Uses atan2(1, 1) = π/4 which is not exactly representable in f32.
    #[test]
    fn test_ibp_directed_rounding() {
        let layer = Atan2Layer;
        let y = make_bt(&[1.0], &[1.0]); // point interval
        let x = make_bt(&[1.0], &[1.0]); // point interval
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("IBP atan2 should succeed");

        let exact = std::f64::consts::FRAC_PI_4;
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

    /// Regression: y_upper = 0 with x purely negative touches the branch cut.
    /// Before fix, `crosses_negative_real_axis` used `y_upper > 0.0` (strict)
    /// and missed this case, producing a lower bound of ~-2.897 when interior
    /// points approach -π (~-3.141). The gap was ~0.245 rad (14°) unsound.
    #[test]
    fn test_ibp_branch_cut_y_upper_zero_regression() {
        let layer = Atan2Layer;
        let y = make_bt(&[-1.0], &[0.0]); // y_upper = 0 exactly
        let x = make_bt(&[-4.0], &[-2.0]); // x purely negative
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("IBP atan2 should succeed");

        // Must return full range because interior approaches -π
        let (expected_lower, expected_upper) = full_angle_range();
        assert_eq!(
            result.lower()[0],
            expected_lower,
            "lower must be -π for y_upper=0 branch-cut touch"
        );
        assert_eq!(
            result.upper()[0],
            expected_upper,
            "upper must be π for y_upper=0 branch-cut touch"
        );

        // Verify the concrete counterexample is contained
        let near_cut = (-0.001_f64).atan2(-4.0_f64) as f32;
        assert!(
            near_cut >= result.lower()[0],
            "atan2(-0.001, -4) = {near_cut} must be >= lower {}",
            result.lower()[0]
        );
    }

    /// Ill-conditioned box (contains origin) must keep the IBP fallback: the
    /// CROWN driver returns `UnsupportedOp` so the dispatcher degrades safely.
    #[test]
    fn test_crown_falls_back_on_origin() {
        let layer = Atan2Layer;
        // input_a = y, input_b = x; box [-1,1]x[-1,1] contains the origin.
        let y = make_bt(&[-1.0], &[1.0]);
        let x = make_bt(&[-1.0], &[1.0]);
        let err = layer
            .propagate_linear_binary(&LinearBounds::identity(1), &y, &x)
            .expect_err("origin box must fall back to IBP");
        assert!(matches!(err, NyError::UnsupportedOp(_)));
    }

    /// Well-conditioned box (strict Q1) succeeds and the concretized affine
    /// lower/upper forms soundly enclose atan2 at every sampled point.
    #[test]
    fn test_crown_q1_box_is_sound() {
        let layer = Atan2Layer;
        // input_a = y in [1,3], input_b = x in [2,4]: strict Q1.
        let y = make_bt(&[1.0], &[3.0]);
        let x = make_bt(&[2.0], &[4.0]);
        let ident = LinearBounds::identity(1);
        let (lb_a, lb_b) = layer
            .propagate_linear_binary(&ident, &y, &x)
            .expect("Q1 box should produce a CROWN relaxation");

        // Reconstruct the affine forms: output 0 = a_coeff*y + b_coeff*x + bias.
        // Bias lives entirely on lb_a (lb_b bias is zeroed by the driver).
        let steps = 40;
        for i in 0..=steps {
            let yv = 1.0 + (3.0 - 1.0) * (i as f32 / steps as f32);
            for j in 0..=steps {
                let xv = 2.0 + (4.0 - 2.0) * (j as f32 / steps as f32);
                let z = (yv as f64).atan2(xv as f64) as f32;
                let lo =
                    lb_a.lower_a()[[0, 0]] * yv + lb_b.lower_a()[[0, 0]] * xv + lb_a.lower_b()[0];
                let hi =
                    lb_a.upper_a()[[0, 0]] * yv + lb_b.upper_a()[[0, 0]] * xv + lb_a.upper_b()[0];
                assert!(lo <= z, "lower {lo} > atan2 {z} at (y={yv}, x={xv})");
                assert!(hi >= z, "upper {hi} < atan2 {z} at (y={yv}, x={xv})");
            }
        }
    }

    /// Negated incoming spec exercises the `w < 0` plane-selection branch and
    /// must remain sound (lower form bounds -atan2 from below, upper from above).
    #[test]
    fn test_crown_q4_negated_spec_is_sound() {
        use ndarray::Array2;
        let layer = Atan2Layer;
        // Strict Q4: y in [-3,-1], x in [2,5].
        let y = make_bt(&[-3.0], &[-1.0]);
        let x = make_bt(&[2.0], &[5.0]);
        let neg_ident = {
            let mut la = Array2::<f32>::zeros((1, 1));
            let mut ua = Array2::<f32>::zeros((1, 1));
            la[[0, 0]] = -1.0;
            ua[[0, 0]] = -1.0;
            LinearBounds::new(la, ndarray::Array1::zeros(1), ua, ndarray::Array1::zeros(1)).unwrap()
        };
        let (lb_a, lb_b) = layer
            .propagate_linear_binary(&neg_ident, &y, &x)
            .expect("Q4 box should produce a CROWN relaxation");
        let steps = 40;
        for i in 0..=steps {
            let yv = -3.0 + (-1.0 - -3.0) * (i as f32 / steps as f32);
            for j in 0..=steps {
                let xv = 2.0 + (5.0 - 2.0) * (j as f32 / steps as f32);
                let z = -((yv as f64).atan2(xv as f64) as f32);
                let lo =
                    lb_a.lower_a()[[0, 0]] * yv + lb_b.lower_a()[[0, 0]] * xv + lb_a.lower_b()[0];
                let hi =
                    lb_a.upper_a()[[0, 0]] * yv + lb_b.upper_a()[[0, 0]] * xv + lb_a.upper_b()[0];
                assert!(lo <= z, "neg lower {lo} > {z} at (y={yv}, x={xv})");
                assert!(hi >= z, "neg upper {hi} < {z} at (y={yv}, x={xv})");
            }
        }
    }
}
