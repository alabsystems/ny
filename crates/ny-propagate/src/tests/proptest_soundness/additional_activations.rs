// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::common::BoundPropagation;
use crate::layers::LeakyReLULayer;
use ndarray::arr1;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{sample_points, valid_interval, FP_TOLERANCE};

// =============================================================================
// ADDITIONAL ACTIVATION SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// LeakyReLU IBP soundness: for any x in [l, u], LeakyReLU(x) is in computed bounds.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_leaky_relu_ibp((l, u) in valid_interval(100.0), alpha in -3.0f32..3.0) {
        prop_assume!(alpha.abs() >= 0.001);
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let leaky_relu = LeakyReLULayer::new(alpha);
        let output = leaky_relu.propagate_ibp(&input).unwrap();

        // Verify soundness: LeakyReLU(x) in [output.lower(), output.upper()] for all x in [l, u]
        for x in sample_points(l, u, 20) {
            let leaky_relu_x = if x >= 0.0 { x } else { alpha * x };
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= leaky_relu_x && leaky_relu_x <= output.upper()[[0]] + FP_TOLERANCE,
                "LeakyReLU soundness violation: LeakyReLU({}, alpha={})={} not in [{}, {}]",
                x, alpha, leaky_relu_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// LeakyReLU IBP soundness with negative-only input (tests scaled region).
#[ntest::timeout(10000)]
    #[test]
    fn soundness_leaky_relu_negative_region(l in -100.0f32..0.0, u in -50.0f32..0.0, alpha in -3.0f32..3.0) {
        prop_assume!(alpha.abs() >= 0.001);
        // Ensure l <= u
        let (l, u) = (l.min(u), l.max(u));

        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let leaky_relu = LeakyReLULayer::new(alpha);
        let output = leaky_relu.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 10) {
            let leaky_relu_x = alpha * x;  // All negative, so scaled
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= leaky_relu_x && leaky_relu_x <= output.upper()[[0]] + FP_TOLERANCE,
                "LeakyReLU negative region soundness violation: {} not in [{}, {}]",
                leaky_relu_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn regression_leaky_relu_ibp_negative_alpha_negative_region_bounds() {
    let alpha = -0.5_f32;
    let l = -4.0_f32;
    let u = -1.0_f32;
    let input = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();

    let layer = LeakyReLULayer::new(alpha);
    let output = layer.propagate_ibp(&input).unwrap();

    // Decreasing branch on x<0: min at x=u, max at x=l.
    assert!((output.lower()[[0]] - alpha * u).abs() < 1e-6);
    assert!((output.upper()[[0]] - alpha * l).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn regression_leaky_relu_ibp_negative_alpha_crossing_zero_bounds() {
    let alpha = -0.5_f32;
    let l = -10.0_f32;
    let u = 1.0_f32;
    let input = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();

    let layer = LeakyReLULayer::new(alpha);
    let output = layer.propagate_ibp(&input).unwrap();

    // Crossing with alpha<0: minimum at x=0 (value 0), maximum at max(alpha*l, u).
    assert!(output.lower()[[0]].abs() < 1e-6);
    assert!((output.upper()[[0]] - (alpha * l).max(u)).abs() < 1e-6);
}
