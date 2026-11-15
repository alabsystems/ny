// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, Array2, ArrayD, IxDyn};
use num_rational::BigRational;
use ny_tensor::BoundedTensor;

use super::RmsNormLayer;
use crate::LinearBounds;

fn boxed(shape: &[usize], lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).unwrap(),
    )
    .unwrap()
}
fn direction(c: &[f32]) -> LinearBounds {
    let a = Array2::from_shape_vec((1, c.len()), c.to_vec()).unwrap();
    LinearBounds::new(a.clone(), arr1(&[0.0]), a, arr1(&[0.0])).unwrap()
}
fn at(bounds: &LinearBounds, x: &[f32]) -> (f64, f64) {
    (
        bounds.lower_b()[0] as f64
            + bounds
                .lower_a()
                .row(0)
                .iter()
                .zip(x)
                .map(|(&a, &x)| a as f64 * x as f64)
                .sum::<f64>(),
        bounds.upper_b()[0] as f64
            + bounds
                .upper_a()
                .row(0)
                .iter()
                .zip(x)
                .map(|(&a, &x)| a as f64 * x as f64)
                .sum::<f64>(),
    )
}

#[test]
fn point_norm_and_cast_errors_enclose_exact_rational_cancellation() {
    // mean(x²)=.5, epsilon=.5 => denominator exactly one.
    let layer = RmsNormLayer::new(arr1(&[1.25, -0.5]), 0.5).unwrap();
    let c = [1e20_f32, -1e20_f32];
    let result = layer
        .propagate_radial_linear_with_bounds(
            &direction(&c),
            &boxed(&[2], &[1.0, 0.0], &[1.0, 0.0]),
            None,
        )
        .unwrap();
    let rational = |x: f32| BigRational::from_float(x as f64).unwrap();
    let truth = rational(c[0]) * rational(1.25);
    let lower = rational(result.lower_b()[0]) + rational(result.lower_a()[[0, 0]]);
    let upper = rational(result.upper_b()[0]) + rational(result.upper_a()[[0, 0]]);
    assert!(lower <= truth && truth <= upper);
}

#[test]
fn radial_signed_direction_encloses_vertices_and_zero_crossing() {
    for eps in [0.125_f32, 1e-12] {
        let layer = RmsNormLayer::new(arr1(&[2.0, -0.75, 0.5]), eps).unwrap();
        let c = [-1.5_f32, 0.25, 2.0];
        let lower = [-0.25_f32, -0.125, -0.375];
        let upper = [0.125_f32, 0.25, 0.375];
        let result = layer
            .propagate_radial_linear_with_bounds(&direction(&c), &boxed(&[3], &lower, &upper), None)
            .unwrap();
        for code in 0..125 {
            let x = std::array::from_fn::<_, 3, _>(|j| {
                let t = ((code / 5usize.pow(j as u32)) % 5) as f32 / 4.0;
                lower[j] * (1.0 - t) + upper[j] * t
            });
            let denominator =
                (x.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / 3.0 + eps as f64).sqrt();
            let truth = (0..3)
                .map(|j| c[j] as f64 * layer.ny[j] as f64 * x[j] as f64 / denominator)
                .sum::<f64>();
            let (lo, hi) = at(&result, &x);
            assert!(
                lo <= truth && truth <= hi,
                "eps={eps},x={x:?}: [{lo},{hi}] excludes {truth}"
            );
        }
    }
}

#[test]
fn default_retains_conditional_inv_rms_window_and_incoming_error() {
    let layer = RmsNormLayer::new_default(2, 0.5).unwrap();
    let pre = boxed(&[2], &[1.0, 0.0], &[1.0, 0.0]);
    let mut bounds = direction(&[1.0, 0.0]);
    bounds.lower_a_err = Some(Array2::from_elem((1, 2), 0.25));
    bounds.upper_a_err = bounds.lower_a_err.clone();
    let result = layer
        .propagate_linear_with_bounds_inv_rms(&bounds, &pre, Some(&[Some((1.0, 1.0))]))
        .unwrap();
    let (lo, hi) = at(&result, &[1.0, 0.0]);
    assert!(lo <= 0.75 && hi >= 1.25, "[{lo},{hi}]");
}

#[test]
fn zero_direction_is_exact_on_many_normalization_groups() {
    let layer = RmsNormLayer::new_default(3, 1e-5).unwrap();
    let pre = boxed(&[2, 3], &[-1.0; 6], &[1.0; 6]);
    let result = layer
        .propagate_radial_linear_with_bounds(&direction(&[0.0; 6]), &pre, None)
        .unwrap();
    assert!(result
        .lower_a()
        .iter()
        .chain(result.upper_a().iter())
        .all(|&x| x == 0.0));
    assert_eq!(result.lower_b()[0], 0.0);
    assert_eq!(result.upper_b()[0], 0.0);
}
