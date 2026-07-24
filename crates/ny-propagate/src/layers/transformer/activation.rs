// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::layers::softmax::{gelu_bound_interval, GeluApproximation};

/// Bound propagation for GELU activation (Tanh approximation).
///
/// GELU(x) = x * Phi(x) where Phi is the CDF of standard normal.
/// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// Delegates to `gelu_bound_interval` from the GELU CROWN module, which uses
/// a precise bisection-computed critical point (~-0.7517) rather than the
/// approximate -0.75 that was previously hardcoded here.
/// Ref: softmax/gelu/eval.rs:206-248 (gelu_critical_point)
fn gelu_bounds(input: &BoundedTensor) -> Result<BoundedTensor> {
    let mut out_lower = input.lower().clone();
    let mut out_upper = input.upper().clone();

    ndarray::Zip::from(&mut out_lower)
        .and(&mut out_upper)
        .and(input.lower())
        .and(input.upper())
        .for_each(|ol, ou, &il, &iu| {
            let (l, u) = gelu_bound_interval(il, iu, GeluApproximation::Tanh);
            *ol = l;
            *ou = u;
        });

    BoundedTensor::new(out_lower, out_upper)
}

#[cfg(test)]
mod tests {
    use super::gelu_bounds;
    use crate::layers::softmax::{gelu_eval, GeluApproximation};
    use ndarray::arr1;
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;

    #[test]
    fn gelu_bounds_rejects_non_finite_output_interval() {
        let invalid = BoundedTensor::new_unchecked(
            arr1(&[f32::NAN]).into_dyn(),
            arr1(&[f32::NAN]).into_dyn(),
        )
        .expect("shape should be valid");

        let err = gelu_bounds(&invalid).expect_err("NaN interval should be rejected");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    /// Regression test for #2262: the old code used an approximate critical point
    /// of -0.75 instead of the precise ~-0.7517. For intervals in the narrow
    /// window [-0.76, -0.75], the old code treated GELU as monotonically
    /// decreasing and returned (gelu(-0.75), gelu(-0.76)), but the true minimum
    /// at ~-0.7517 falls within this interval. The lower bound gelu(-0.75) was
    /// too high (unsound).
    #[test]
    fn gelu_bounds_narrow_window_near_critical_point() {
        // Interval [-0.76, -0.75] contains the true critical point ~-0.7517
        let l = -0.76_f32;
        let u = -0.75_f32;
        let input = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn())
            .expect("valid interval");

        let result = gelu_bounds(&input).expect("should succeed");

        // Sample 100 points in [l, u] and verify bounds contain all GELU values
        for i in 0..=100 {
            let x = l + (u - l) * (i as f32 / 100.0);
            let gx = gelu_eval(x, GeluApproximation::Tanh);
            assert!(
                result.lower()[[0]] <= gx + 1e-7,
                "lower bound {} exceeds gelu({}) = {} — unsound",
                result.lower()[[0]],
                x,
                gx
            );
            assert!(
                result.upper()[[0]] >= gx - 1e-7,
                "upper bound {} below gelu({}) = {} — unsound",
                result.upper()[[0]],
                x,
                gx
            );
        }
    }

    /// Verify bounds are sound across a range of intervals spanning the critical point.
    #[test]
    fn gelu_bounds_soundness_sampling() {
        let test_intervals = [
            (-2.0_f32, 2.0), // wide, contains critical point
            (-1.0, 0.0),     // contains critical point
            (-0.8, -0.7),    // narrow around critical point
            (0.0, 1.0),      // purely positive (monotone increasing)
            (-3.0, -1.0),    // purely negative, decreasing region
        ];

        for (l, u) in test_intervals {
            let input = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn())
                .expect("valid interval");

            let result = gelu_bounds(&input).expect("should succeed");

            // Sample and verify containment
            for i in 0..=50 {
                let x = l + (u - l) * (i as f32 / 50.0);
                let gx = gelu_eval(x, GeluApproximation::Tanh);
                assert!(
                    result.lower()[[0]] <= gx + 1e-7,
                    "[{l}, {u}]: lower bound {} exceeds gelu({x}) = {gx}",
                    result.lower()[[0]]
                );
                assert!(
                    result.upper()[[0]] >= gx - 1e-7,
                    "[{l}, {u}]: upper bound {} below gelu({x}) = {gx}",
                    result.upper()[[0]]
                );
            }
        }
    }
}
