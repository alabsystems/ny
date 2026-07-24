// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::common::BoundPropagation;
use crate::layers::{LayerNormLayer, SoftmaxLayer};
use ndarray::arr1;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{layernorm, softmax, valid_interval};

// =============================================================================
// TRANSFORMER LAYER SOUNDNESS TESTS (Softmax, LayerNorm)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Softmax IBP soundness: for any x in [l, u], softmax(x) is in computed bounds.
    ///
    /// Uses a smaller 3-element vector to keep test runtime reasonable.
    /// Softmax is challenging because outputs are coupled (sum to 1).
#[ntest::timeout(10000)]
    #[test]
    fn soundness_softmax_ibp_3d(
        (l0, u0) in valid_interval(3.0),
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn()
        ).unwrap();

        let softmax_layer = SoftmaxLayer::new(-1);
        let output = softmax_layer.propagate_ibp(&input).unwrap();

        // Test corner points and center - these are the extremal cases for softmax
        let test_points = vec![
            arr1(&[l0, l1, l2]),
            arr1(&[u0, u1, u2]),
            arr1(&[u0, l1, l2]),
            arr1(&[l0, u1, l2]),
            arr1(&[u0, l1, u2]),
            arr1(&[l0, l1, u2]),
            arr1(&[u0, u1, l2]),
            arr1(&[l0, u1, u2]),
            arr1(&[f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2)]),
        ];

        for x in test_points {
            let softmax_x = softmax(&x);

            for i in 0..3 {
                // Use larger tolerance for softmax due to exponential sensitivity
                let tol = 1e-4;
                prop_assert!(
                    output.lower()[[i]] - tol <= softmax_x[i] && softmax_x[i] <= output.upper()[[i]] + tol,
                    "Softmax soundness violation: softmax({:?})[{}]={} not in [{}, {}]",
                    x, i, softmax_x[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// Softmax IBP soundness with tighter bounds (smaller perturbation).
#[ntest::timeout(10000)]
    #[test]
    fn soundness_softmax_ibp_tight(
        center0 in -2.0f32..2.0,
        center1 in -2.0f32..2.0,
        center2 in -2.0f32..2.0,
        epsilon in 0.01f32..0.5,
    ) {
        let input = BoundedTensor::new(
            arr1(&[center0 - epsilon, center1 - epsilon, center2 - epsilon]).into_dyn(),
            arr1(&[center0 + epsilon, center1 + epsilon, center2 + epsilon]).into_dyn()
        ).unwrap();

        let softmax_layer = SoftmaxLayer::new(-1);
        let output = softmax_layer.propagate_ibp(&input).unwrap();

        // Sample within the epsilon-ball
        for i in 0..=20 {
            let t = (i as f32 / 20.0) * 2.0 - 1.0;
            let x = arr1(&[
                center0 + t * epsilon,
                center1 + (t * 0.7) * epsilon,
                center2 + (t * -0.4) * epsilon,
            ]);

            // Clamp to bounds
            let x_clamped = arr1(&[
                x[0].clamp(center0 - epsilon, center0 + epsilon),
                x[1].clamp(center1 - epsilon, center1 + epsilon),
                x[2].clamp(center2 - epsilon, center2 + epsilon),
            ]);

            let softmax_x = softmax(&x_clamped);

            for idx in 0..3 {
                let tol = 1e-4;
                prop_assert!(
                    output.lower()[[idx]] - tol <= softmax_x[idx] && softmax_x[idx] <= output.upper()[[idx]] + tol,
                    "Softmax tight soundness violation at {}: {} not in [{}, {}]",
                    idx, softmax_x[idx], output.lower()[[idx]], output.upper()[[idx]]
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// LayerNorm IBP soundness: for any x in [l, u], layernorm(x) is in computed bounds.
    ///
    /// Uses default ny=1, beta=0 for simplicity.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_layernorm_ibp_4d(
        (l0, u0) in valid_interval(2.0),
        (l1, u1) in valid_interval(2.0),
        (l2, u2) in valid_interval(2.0),
        (l3, u3) in valid_interval(2.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2, l3]).into_dyn(),
            arr1(&[u0, u1, u2, u3]).into_dyn()
        ).unwrap();

        let layernorm_layer = LayerNormLayer::new_default(4, 1e-5).unwrap();
        let output = layernorm_layer.propagate_ibp(&input).unwrap();

        let ny = ndarray::Array1::ones(4);
        let beta = ndarray::Array1::zeros(4);

        // Test corner points
        let corners = vec![
            arr1(&[l0, l1, l2, l3]),
            arr1(&[u0, u1, u2, u3]),
            arr1(&[u0, l1, l2, l3]),
            arr1(&[l0, u1, l2, l3]),
            arr1(&[l0, l1, u2, l3]),
            arr1(&[l0, l1, l2, u3]),
            arr1(&[f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2), f32::midpoint(l3, u3)]),
        ];

        for x in corners {
            let ln_x = layernorm(&x, &ny, &beta, 1e-5);

            for i in 0..4 {
                // LayerNorm can have larger errors due to division
                let tol = 1e-3;
                prop_assert!(
                    output.lower()[[i]] - tol <= ln_x[i] && ln_x[i] <= output.upper()[[i]] + tol,
                    "LayerNorm soundness violation: layernorm({:?})[{}]={} not in [{}, {}]",
                    x, i, ln_x[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// LayerNorm IBP soundness with custom ny/beta.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_layernorm_ibp_with_params(
        (l0, u0) in valid_interval(1.5),
        (l1, u1) in valid_interval(1.5),
        (l2, u2) in valid_interval(1.5),
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        b2 in -1.0f32..1.0,
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn()
        ).unwrap();

        let ny = ndarray::Array1::from_vec(vec![g0, g1, g2]);
        let beta = ndarray::Array1::from_vec(vec![b0, b1, b2]);

        let layernorm_layer = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5).unwrap();
        let output = layernorm_layer.propagate_ibp(&input).unwrap();

        // Test corner points
        for corner in 0..8 {
            let mut x_vec = vec![l0, l1, l2];
            if corner & 1 != 0 { x_vec[0] = u0; }
            if corner & 2 != 0 { x_vec[1] = u1; }
            if corner & 4 != 0 { x_vec[2] = u2; }

            let x = ndarray::Array1::from_vec(x_vec);
            let ln_x = layernorm(&x, &ny, &beta, 1e-5);

            for i in 0..3 {
                let tol = 1e-3;
                prop_assert!(
                    output.lower()[[i]] - tol <= ln_x[i] && ln_x[i] <= output.upper()[[i]] + tol,
                    "LayerNorm with params soundness violation at {}: {} not in [{}, {}]",
                    i, ln_x[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// LayerNorm forward mode IBP soundness test.
    /// Forward mode uses the midpoint for mean/std, so it's tighter but approximate.
    /// Tests ALL sampled points (not just center) for soundness.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_layernorm_forward_mode_ibp(
        center0 in -1.0f32..1.0,
        center1 in -1.0f32..1.0,
        center2 in -1.0f32..1.0,
        epsilon in 0.01f32..0.2,
    ) {
        let input = BoundedTensor::new(
            arr1(&[center0 - epsilon, center1 - epsilon, center2 - epsilon]).into_dyn(),
            arr1(&[center0 + epsilon, center1 + epsilon, center2 + epsilon]).into_dyn()
        ).unwrap();

        let ny = ndarray::Array1::ones(3);
        let beta = ndarray::Array1::zeros(3);

        let layernorm_layer = LayerNormLayer::new_forward_mode(ny.clone(), beta.clone(), 1e-5).unwrap();
        let output = layernorm_layer.propagate_ibp(&input).unwrap();

        // Test ALL 8 corners + center point (not just center).
        // The old test only checked center, which always trivially passes.
        let pts = [
            [center0, center1, center2],
            [center0 - epsilon, center1 - epsilon, center2 - epsilon],
            [center0 + epsilon, center1 - epsilon, center2 - epsilon],
            [center0 - epsilon, center1 + epsilon, center2 - epsilon],
            [center0 + epsilon, center1 + epsilon, center2 - epsilon],
            [center0 - epsilon, center1 - epsilon, center2 + epsilon],
            [center0 + epsilon, center1 - epsilon, center2 + epsilon],
            [center0 - epsilon, center1 + epsilon, center2 + epsilon],
            [center0 + epsilon, center1 + epsilon, center2 + epsilon],
        ];
        for pt in &pts {
            let x = arr1(pt);
            let ln_x = layernorm(&x, &ny, &beta, 1e-5);
            for i in 0..3 {
                let tol = 1e-3;
                prop_assert!(
                    output.lower()[[i]] - tol <= ln_x[i] && ln_x[i] <= output.upper()[[i]] + tol,
                    "LayerNorm forward mode soundness violation at {}: x={:?}, \
                     ln(x)[{i}]={} not in [{}, {}]",
                    i, pt, ln_x[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// LayerNorm forward mode: large perturbation where the box reaches zero
    /// variance (all elements equal). Regression for forward-mode second-order
    /// remainder using center-point std instead of minimum std over the box.
    ///
    /// Counterexample: c = [100, -100], r = [100, 100]. The box [0, 200] × [-200, 0]
    /// includes x = [0, 0] where var = 0, but center std = 100. Using center std
    /// gives second-order = 0.14, total radius = 0.14, bounds [0.86, 1.14].
    /// But y([0, 0]) = 0, which is outside [0.86, 1.14]. UNSOUND.
    ///
    /// Part of #3142.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_layernorm_forward_mode_large_perturbation(
        c0 in -100.0f32..100.0,
        c1 in -100.0f32..100.0,
        hw0 in 0.5f32..100.0,
        hw1 in 0.5f32..100.0,
    ) {
        let input = BoundedTensor::new(
            arr1(&[c0 - hw0, c1 - hw1]).into_dyn(),
            arr1(&[c0 + hw0, c1 + hw1]).into_dyn()
        ).unwrap();

        let ny = ndarray::Array1::ones(2);
        let beta = ndarray::Array1::zeros(2);

        let layernorm_layer = LayerNormLayer::new_forward_mode(ny.clone(), beta.clone(), 1e-5).unwrap();
        let output = layernorm_layer.propagate_ibp(&input).unwrap();

        // Sample 5 points per dimension = 25 total
        let s0 = super::sample_points(c0 - hw0, c0 + hw0, 5);
        let s1 = super::sample_points(c1 - hw1, c1 + hw1, 5);
        for &x0 in &s0 {
            for &x1 in &s1 {
                let x = arr1(&[x0, x1]);
                let ln_x = layernorm(&x, &ny, &beta, 1e-5);
                for i in 0..2 {
                    let tol = 1e-3;
                    prop_assert!(
                        output.lower()[[i]] - tol <= ln_x[i] && ln_x[i] <= output.upper()[[i]] + tol,
                        "LayerNorm forward mode large perturbation violation at dim {i}: \
                         x=[{x0}, {x1}], ln(x)[{i}]={} not in [{}, {}]",
                        ln_x[i], output.lower()[[i]], output.upper()[[i]]
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// LayerNorm forward mode with n=4: exercises second-order remainder
    /// formula with dimensions > 2.
    ///
    /// For n=2, LayerNorm output is effectively binary (z = ±1), masking
    /// Hessian formula errors. With n≥3, the z values vary continuously
    /// and σ can be large, testing the 1/σ^k denominators in the remainder.
    ///
    /// Uses large center spread to create σ >> 1 scenarios where the
    /// Hessian denominators (1/σ² vs 1/σ³) diverge significantly.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_layernorm_forward_mode_4d(
        c0 in -100.0f32..100.0,
        c1 in -100.0f32..100.0,
        c2 in -100.0f32..100.0,
        c3 in -100.0f32..100.0,
        hw0 in 0.5f32..50.0,
        hw1 in 0.5f32..50.0,
        hw2 in 0.5f32..50.0,
        hw3 in 0.5f32..50.0,
    ) {
        let input = BoundedTensor::new(
            arr1(&[c0 - hw0, c1 - hw1, c2 - hw2, c3 - hw3]).into_dyn(),
            arr1(&[c0 + hw0, c1 + hw1, c2 + hw2, c3 + hw3]).into_dyn()
        ).unwrap();

        let ny = ndarray::Array1::ones(4);
        let beta = ndarray::Array1::zeros(4);

        let layer = LayerNormLayer::new_forward_mode(ny.clone(), beta.clone(), 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        // Test all 16 corners
        for corner in 0..16u32 {
            let x = arr1(&[
                if corner & 1 != 0 { c0 + hw0 } else { c0 - hw0 },
                if corner & 2 != 0 { c1 + hw1 } else { c1 - hw1 },
                if corner & 4 != 0 { c2 + hw2 } else { c2 - hw2 },
                if corner & 8 != 0 { c3 + hw3 } else { c3 - hw3 },
            ]);
            let ln_x = layernorm(&x, &ny, &beta, 1e-5);
            for i in 0..4 {
                let tol = 1e-3;
                prop_assert!(
                    output.lower()[[i]] - tol <= ln_x[i] && ln_x[i] <= output.upper()[[i]] + tol,
                    "LayerNorm 4D forward mode violation at dim {i}: corner={corner}, \
                     x={:?}, ln(x)[{i}]={} not in [{}, {}]",
                    x, ln_x[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }

        // Also sample 3 points per dimension = 81 total
        let s0 = super::sample_points(c0 - hw0, c0 + hw0, 3);
        let s1 = super::sample_points(c1 - hw1, c1 + hw1, 3);
        let s2 = super::sample_points(c2 - hw2, c2 + hw2, 3);
        let s3 = super::sample_points(c3 - hw3, c3 + hw3, 3);
        for &x0 in &s0 {
            for &x1 in &s1 {
                for &x2 in &s2 {
                    for &x3 in &s3 {
                        let x = arr1(&[x0, x1, x2, x3]);
                        let ln_x = layernorm(&x, &ny, &beta, 1e-5);
                        for i in 0..4 {
                            let tol = 1e-3;
                            prop_assert!(
                                output.lower()[[i]] - tol <= ln_x[i]
                                    && ln_x[i] <= output.upper()[[i]] + tol,
                                "LayerNorm 4D forward mode sampling violation at dim {i}: \
                                 x=[{x0}, {x1}, {x2}, {x3}], ln(x)[{i}]={} not in [{}, {}]",
                                ln_x[i], output.lower()[[i]], output.upper()[[i]]
                            );
                        }
                    }
                }
            }
        }
    }

    /// LayerNorm forward mode with non-unit ny: exercises the |γ_i|
    /// scaling factor in the second-order remainder formula:
    ///   R₂(i) ≤ 7|γ_i|·||r||² / (2√n·σ_min²)
    ///
    /// All other forward-mode proptests use ny=1, so |γ_i|=1 and any
    /// misplacement of g_i.abs() (e.g. dividing instead of multiplying)
    /// would be masked. This test uses ny in [0.5, 3.0].
    ///
    /// Part of #3142.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_layernorm_forward_mode_4d_with_gamma(
        c0 in -50.0f32..50.0,
        c1 in -50.0f32..50.0,
        c2 in -50.0f32..50.0,
        c3 in -50.0f32..50.0,
        hw0 in 0.5f32..30.0,
        hw1 in 0.5f32..30.0,
        hw2 in 0.5f32..30.0,
        hw3 in 0.5f32..30.0,
        g0 in 0.5f32..3.0,
        g1 in -3.0f32..-0.5,
        g2 in 0.5f32..3.0,
        g3 in -3.0f32..-0.5,
    ) {
        let input = BoundedTensor::new(
            arr1(&[c0 - hw0, c1 - hw1, c2 - hw2, c3 - hw3]).into_dyn(),
            arr1(&[c0 + hw0, c1 + hw1, c2 + hw2, c3 + hw3]).into_dyn()
        ).unwrap();

        let ny = ndarray::Array1::from(vec![g0, g1, g2, g3]);
        let beta = ndarray::Array1::zeros(4);

        let layer = LayerNormLayer::new_forward_mode(ny.clone(), beta.clone(), 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        // Test all 16 corners
        for corner in 0..16u32 {
            let x = arr1(&[
                if corner & 1 != 0 { c0 + hw0 } else { c0 - hw0 },
                if corner & 2 != 0 { c1 + hw1 } else { c1 - hw1 },
                if corner & 4 != 0 { c2 + hw2 } else { c2 - hw2 },
                if corner & 8 != 0 { c3 + hw3 } else { c3 - hw3 },
            ]);
            let ln_x = layernorm(&x, &ny, &beta, 1e-5);
            for i in 0..4 {
                let tol = 1e-3;
                prop_assert!(
                    output.lower()[[i]] - tol <= ln_x[i] && ln_x[i] <= output.upper()[[i]] + tol,
                    "LayerNorm 4D+ny forward mode violation at dim {i}: \
                     corner={corner}, ny={:?}, x={:?}, ln(x)[{i}]={} not in [{}, {}]",
                    ny, x, ln_x[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }
}
