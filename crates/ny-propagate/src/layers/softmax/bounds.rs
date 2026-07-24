// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::{BatchedLinearBounds, LinearBounds};

pub(super) fn constant_bounds_from_output(
    bounds: &LinearBounds,
    output_bounds: &BoundedTensor,
) -> Result<LinearBounds> {
    let output_flat = output_bounds.flatten();
    let output_len = output_flat.len();
    if bounds.num_inputs() != output_len {
        return Err(NyError::ShapeMismatch {
            expected: vec![bounds.num_inputs()],
            got: vec![output_len],
        });
    }

    let concretized = bounds.concretize_sound(&output_flat); // #2236: directed rounding for soundness
    let lower_shape = concretized.lower().shape().to_vec();
    let lower = concretized
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![bounds.num_outputs()],
            got: lower_shape,
        })?;
    let upper_shape = concretized.upper().shape().to_vec();
    let upper = concretized
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![bounds.num_outputs()],
            got: upper_shape,
        })?;

    LinearBounds::new_or_conservative(
        Array2::zeros((bounds.num_outputs(), bounds.num_inputs())),
        lower,
        Array2::zeros((bounds.num_outputs(), bounds.num_inputs())),
        upper,
    )
}

pub(super) fn batched_constant_bounds_from_output(
    bounds: &BatchedLinearBounds,
    output_bounds: &BoundedTensor,
) -> Result<BatchedLinearBounds> {
    let expected_shape = bounds.output_shape.clone();
    let got_shape = output_bounds.shape().to_vec();
    if expected_shape != got_shape {
        return Err(NyError::ShapeMismatch {
            expected: expected_shape,
            got: got_shape,
        });
    }

    let a_shape = bounds.lower_a().shape().to_vec();
    let lower_a = ArrayD::zeros(IxDyn(&a_shape));
    let upper_a = ArrayD::zeros(IxDyn(&a_shape));

    // Apply directed rounding to bias terms for soundness (#2787).
    // The non-batched path uses concretize_sound which applies next_down/next_up.
    // Without this, rounding errors in IBP output bounds propagate unsoundly
    // through CROWN backward.
    let lower_b = output_bounds.lower().mapv(next_down_f32);
    let upper_b = output_bounds.upper().mapv(next_up_f32);

    // Phase 4 audit: IBP fallback — A=0 (trivially valid), biases from BoundedTensor.
    BatchedLinearBounds::new_or_conservative(
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        bounds.input_shape.clone(),
        bounds.output_shape.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn make_identity_bounds(n: usize) -> LinearBounds {
        LinearBounds::new(
            Array2::eye(n),
            Array1::zeros(n),
            Array2::eye(n),
            Array1::zeros(n),
        )
        .unwrap()
    }

    fn make_output_bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    // ========== constant_bounds_from_output tests ==========

    #[test]
    fn constant_bounds_zero_slopes() {
        let bounds = make_identity_bounds(3);
        let output = make_output_bt(&[0.1, 0.3, 0.2], &[0.4, 0.6, 0.5]);
        let result = constant_bounds_from_output(&bounds, &output).unwrap();

        for &v in result.lower_a.iter() {
            assert_eq!(v, 0.0, "lower_a should be all zeros");
        }
        for &v in result.upper_a.iter() {
            assert_eq!(v, 0.0, "upper_a should be all zeros");
        }
    }

    #[test]
    fn constant_bounds_bias_matches_concretized() {
        let bounds = make_identity_bounds(3);
        let lower = [0.1f32, 0.3, 0.2];
        let upper = [0.5f32, 0.7, 0.6];
        let output = make_output_bt(&lower, &upper);
        let result = constant_bounds_from_output(&bounds, &output).unwrap();

        for i in 0..3 {
            assert!(
                (result.lower_b[i] - lower[i]).abs() < 1e-4,
                "lower_b[{}] = {}, expected ~{}",
                i,
                result.lower_b[i],
                lower[i]
            );
            assert!(
                (result.upper_b[i] - upper[i]).abs() < 1e-4,
                "upper_b[{}] = {}, expected ~{}",
                i,
                result.upper_b[i],
                upper[i]
            );
        }
    }

    #[test]
    fn constant_bounds_shape_mismatch_errors() {
        let bounds = make_identity_bounds(3);
        let output = make_output_bt(&[0.1, 0.2], &[0.5, 0.6]);
        let result = constant_bounds_from_output(&bounds, &output);
        assert!(result.is_err(), "Should error on shape mismatch");
    }

    #[test]
    fn constant_bounds_single_element() {
        let bounds = make_identity_bounds(1);
        let output = make_output_bt(&[0.3], &[0.7]);
        let result = constant_bounds_from_output(&bounds, &output).unwrap();

        assert_eq!(result.lower_a.shape(), &[1, 1]);
        assert_eq!(result.lower_a[[0, 0]], 0.0);
        assert!((result.lower_b[0] - 0.3).abs() < 1e-4);
        assert!((result.upper_b[0] - 0.7).abs() < 1e-4);
    }

    #[test]
    fn constant_bounds_with_scaled_bounds() {
        // Non-identity bounds: lower_a = 2*I, so concretized lower = 2*lower_input
        let n = 2;
        let bounds = LinearBounds::new(
            Array2::eye(n) * 2.0,
            Array1::zeros(n),
            Array2::eye(n) * 2.0,
            Array1::zeros(n),
        )
        .unwrap();
        let output = make_output_bt(&[0.1, 0.2], &[0.5, 0.6]);
        let result = constant_bounds_from_output(&bounds, &output).unwrap();

        // Slopes should still be zero (constant bounds)
        for &v in result.lower_a.iter() {
            assert_eq!(v, 0.0);
        }
        for &v in result.upper_a.iter() {
            assert_eq!(v, 0.0);
        }
        // Shape should be preserved
        assert_eq!(result.lower_a.shape(), &[n, n]);
        assert_eq!(result.lower_b.len(), n);
    }

    /// Regression test for #2281: concretize_sound can produce inverted bounds
    /// (lower > upper) when asymmetric coefficients combine with directed rounding.
    /// The inversion guard should fall back to [-inf, +inf] for those elements.
    #[test]
    fn constant_bounds_inversion_guard() {
        // Craft asymmetric linear bounds where lower_a and upper_a diverge enough
        // to produce inverted concretized bounds.
        // lower = lower_a * input_lower + lower_b (for negative coeff: lower_a * input_upper)
        // upper = upper_a * input_upper + upper_b (for negative coeff: upper_a * input_lower)
        //
        // With lower_a = [-10], lower_b = [100]: lower ≈ -(-10)*input_upper + 100 = 10*5+100 = 150
        // Wait — concretize uses min/max over the coefficient sign, so let's just use
        // a case with very asymmetric coefficients that produce inversion after rounding.
        let n = 1;
        let bounds = LinearBounds::new(
            Array2::from_elem((n, n), -10.0),
            Array1::from_vec(vec![100.0]),
            Array2::from_elem((n, n), 10.0),
            Array1::from_vec(vec![-100.0]),
        )
        .unwrap();
        // With input = [5, 5]:
        // lower concretize: lower_a is negative, so lower = lower_a * input_upper + lower_b = -10*5 + 100 = 50
        // upper concretize: upper_a is positive, so upper = upper_a * input_lower + upper_b = 10*5 - 100 = -50
        // This gives lower > upper → inversion guard should activate.
        let output = make_output_bt(&[5.0], &[5.0]);
        let result = constant_bounds_from_output(&bounds, &output).unwrap();

        // After inversion guard, the element should be [-inf, +inf]
        // OR at minimum: lower <= upper (soundness invariant)
        assert!(
            result.lower_b[0] <= result.upper_b[0],
            "Inversion guard failed: lower_b={} > upper_b={}",
            result.lower_b[0],
            result.upper_b[0]
        );
    }

    // ========== batched_constant_bounds_from_output tests ==========

    #[test]
    fn batched_constant_bounds_zero_slopes() {
        let shape = vec![2, 3];
        let a_shape = vec![2, 3, 3];
        let bounds = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::from_elem(IxDyn(&a_shape), 1.0),
            ArrayD::zeros(IxDyn(&shape)),
            ArrayD::from_elem(IxDyn(&a_shape), 1.0),
            ArrayD::zeros(IxDyn(&shape)),
            shape.clone(),
            shape.clone(),
        );

        let lower_vals = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let upper_vals = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&shape), lower_vals).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&shape), upper_vals).unwrap(),
        )
        .unwrap();

        let result = batched_constant_bounds_from_output(&bounds, &output).unwrap();

        // Slopes should be zeroed
        for &v in result.lower_a.iter() {
            assert_eq!(v, 0.0, "batched lower_a should be all zeros");
        }
        for &v in result.upper_a.iter() {
            assert_eq!(v, 0.0, "batched upper_a should be all zeros");
        }
    }

    #[test]
    fn batched_constant_bounds_bias_matches_output() {
        let shape = vec![2, 3];
        let a_shape = vec![2, 3, 3];
        let bounds = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(&shape)),
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(&shape)),
            shape.clone(),
            shape.clone(),
        );

        let lower_vals = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let upper_vals = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&shape), lower_vals.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&shape), upper_vals.clone()).unwrap(),
        )
        .unwrap();

        let result = batched_constant_bounds_from_output(&bounds, &output).unwrap();

        // Bias should be within 1 ULP of output bounds, with directed rounding:
        // lower_b <= original lower (widened down), upper_b >= original upper (widened up).
        for (i, (&lb, &expected)) in result.lower_b.iter().zip(lower_vals.iter()).enumerate() {
            assert!(
                lb <= expected,
                "lower_b[{}] = {} must be <= {} (directed rounding down)",
                i,
                lb,
                expected
            );
            assert!(
                (lb - expected).abs() < 1e-6,
                "lower_b[{}] = {}, expected ~{}",
                i,
                lb,
                expected
            );
        }
        for (i, (&ub, &expected)) in result.upper_b.iter().zip(upper_vals.iter()).enumerate() {
            assert!(
                ub >= expected,
                "upper_b[{}] = {} must be >= {} (directed rounding up)",
                i,
                ub,
                expected
            );
            assert!(
                (ub - expected).abs() < 1e-6,
                "upper_b[{}] = {}, expected ~{}",
                i,
                ub,
                expected
            );
        }
    }

    #[test]
    fn batched_constant_bounds_shape_mismatch_errors() {
        let shape = vec![2, 3];
        let wrong_shape = vec![3, 3];
        let a_shape = vec![2, 3, 3];
        let bounds = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(&shape)),
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(&shape)),
            shape.clone(),
            shape,
        );

        let output = BoundedTensor::new(
            ArrayD::zeros(IxDyn(&wrong_shape)),
            ArrayD::ones(IxDyn(&wrong_shape)),
        )
        .unwrap();

        let result = batched_constant_bounds_from_output(&bounds, &output);
        assert!(result.is_err(), "Should error on shape mismatch");
    }

    #[test]
    fn batched_constant_bounds_preserves_shapes() {
        let shape = vec![2, 4];
        let a_shape = vec![2, 4, 4];
        let bounds = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(&shape)),
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(&shape)),
            shape.clone(),
            shape.clone(),
        );

        let output =
            BoundedTensor::new(ArrayD::zeros(IxDyn(&shape)), ArrayD::ones(IxDyn(&shape))).unwrap();

        let result = batched_constant_bounds_from_output(&bounds, &output).unwrap();

        assert_eq!(result.lower_a.shape(), &a_shape[..]);
        assert_eq!(result.upper_a.shape(), &a_shape[..]);
        assert_eq!(result.lower_b.shape(), &shape[..]);
        assert_eq!(result.upper_b.shape(), &shape[..]);
        assert_eq!(result.input_shape, shape);
        assert_eq!(result.output_shape, shape);
    }

    /// Regression test for #2787: batched path must apply directed rounding
    /// to lower_b/upper_b, matching the non-batched concretize_sound behavior.
    #[test]
    fn batched_constant_bounds_applies_directed_rounding_2787() {
        use ny_tensor::{next_down_f32, next_up_f32};

        let shape = vec![3];
        let a_shape = vec![3, 3];
        let bounds = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(&shape)),
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(&shape)),
            shape.clone(),
            shape.clone(),
        );

        let lower_vals = vec![0.25, 0.5, 1.0];
        let upper_vals = vec![0.75, 1.5, 2.0];
        let output = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&shape), lower_vals.clone()).expect("valid shape"),
            ArrayD::from_shape_vec(IxDyn(&shape), upper_vals.clone()).expect("valid shape"),
        )
        .expect("valid bounds");

        let result = batched_constant_bounds_from_output(&bounds, &output)
            .expect("batched constant bounds should succeed");

        // lower_b must be next_down of original lower (strictly less for finite nonzero)
        for (i, (&lb, &orig)) in result.lower_b.iter().zip(lower_vals.iter()).enumerate() {
            let expected = next_down_f32(orig);
            assert_eq!(
                lb, expected,
                "lower_b[{i}] = {lb} should be next_down_f32({orig}) = {expected}"
            );
            assert!(lb < orig, "lower_b must be strictly less than original");
        }

        // upper_b must be next_up of original upper (strictly greater for finite nonzero)
        for (i, (&ub, &orig)) in result.upper_b.iter().zip(upper_vals.iter()).enumerate() {
            let expected = next_up_f32(orig);
            assert_eq!(
                ub, expected,
                "upper_b[{i}] = {ub} should be next_up_f32({orig}) = {expected}"
            );
            assert!(ub > orig, "upper_b must be strictly greater than original");
        }
    }
}
