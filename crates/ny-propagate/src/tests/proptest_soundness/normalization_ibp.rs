// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP soundness proptests for LayerNorm, RmsNorm, InstanceNorm1d, and AdaIN1d.
//!
//! These tests verify that both conservative and forward-mode IBP
//! produce sound bounds: for any x in [l, u], f(x) lies within
//! the computed output bounds.
//!
//! Part of #3160, #3195, #3326.

use crate::layers::common::BoundPropagation;
use crate::layers::normalization::AdaIN1dLayer;
use crate::layers::{InstanceNorm1dLayer, LayerNormLayer, RmsNormLayer};
use ndarray::{arr1, Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{
    adain_eval_channel, instance_norm_channel, layernorm, rms_norm, sample_points, valid_interval,
};

// =============================================================================
// RMSNORM IBP SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// RmsNorm conservative IBP soundness: for any x in [l, u],
    /// rms_norm(x) is within computed bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rmsnorm_ibp_3d(
        (l0, u0) in valid_interval(3.0),
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn()
        ).unwrap();

        let layer = RmsNormLayer::new_default(3, 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        let ny = Array1::ones(3);

        // Test all 8 corners
        for corner in 0..8 {
            let x = arr1(&[
                if corner & 1 != 0 { u0 } else { l0 },
                if corner & 2 != 0 { u1 } else { l1 },
                if corner & 4 != 0 { u2 } else { l2 },
            ]);
            let y = rms_norm(&x, &ny, 1e-5);

            for i in 0..3 {
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[i]] - tol <= y[i] && y[i] <= output.upper()[[i]] + tol,
                    "RmsNorm soundness violation: rms_norm({:?})[{}]={} not in [{}, {}]",
                    x, i, y[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// RmsNorm conservative IBP soundness with custom ny.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rmsnorm_ibp_with_gamma(
        (l0, u0) in valid_interval(2.0),
        (l1, u1) in valid_interval(2.0),
        (l2, u2) in valid_interval(2.0),
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn()
        ).unwrap();

        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let layer = RmsNormLayer::new(ny.clone(), 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        for corner in 0..8 {
            let x = arr1(&[
                if corner & 1 != 0 { u0 } else { l0 },
                if corner & 2 != 0 { u1 } else { l1 },
                if corner & 4 != 0 { u2 } else { l2 },
            ]);
            let y = rms_norm(&x, &ny, 1e-5);

            for i in 0..3 {
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[i]] - tol <= y[i] && y[i] <= output.upper()[[i]] + tol,
                    "RmsNorm with ny violation at {}: {} not in [{}, {}]",
                    i, y[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// RmsNorm forward-mode IBP soundness with moderate perturbation.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rmsnorm_forward_mode_ibp(
        center0 in -1.0f32..1.0,
        center1 in -1.0f32..1.0,
        center2 in -1.0f32..1.0,
        epsilon in 0.01f32..0.2,
    ) {
        let input = BoundedTensor::new(
            arr1(&[center0 - epsilon, center1 - epsilon, center2 - epsilon]).into_dyn(),
            arr1(&[center0 + epsilon, center1 + epsilon, center2 + epsilon]).into_dyn()
        ).unwrap();

        let ny = Array1::ones(3);
        let layer = RmsNormLayer::new(ny.clone(), 1e-5).unwrap()
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        // Test all 8 corners + center
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
            let y = rms_norm(&x, &ny, 1e-5);
            for i in 0..3 {
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[i]] - tol <= y[i] && y[i] <= output.upper()[[i]] + tol,
                    "RmsNorm forward mode violation at {}: x={:?}, \
                     rms_norm(x)[{i}]={} not in [{}, {}]",
                    i, pt, y[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// RmsNorm forward-mode IBP soundness with large perturbation.
    ///
    /// Regression test for the σ_min fix (#3159): large boxes can include
    /// points near zero where RMS is very small, causing output to diverge
    /// from center-point predictions.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rmsnorm_forward_mode_large_perturbation(
        c0 in -100.0f32..100.0,
        c1 in -100.0f32..100.0,
        hw0 in 0.5f32..100.0,
        hw1 in 0.5f32..100.0,
    ) {
        let input = BoundedTensor::new(
            arr1(&[c0 - hw0, c1 - hw1]).into_dyn(),
            arr1(&[c0 + hw0, c1 + hw1]).into_dyn()
        ).unwrap();

        let ny = Array1::ones(2);
        let layer = RmsNormLayer::new(ny.clone(), 1e-5).unwrap()
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        let s0 = sample_points(c0 - hw0, c0 + hw0, 5);
        let s1 = sample_points(c1 - hw1, c1 + hw1, 5);
        for &x0 in &s0 {
            for &x1 in &s1 {
                let x = arr1(&[x0, x1]);
                let y = rms_norm(&x, &ny, 1e-5);
                for i in 0..2 {
                    let tol = 1e-6;
                    prop_assert!(
                        output.lower()[[i]] - tol <= y[i] && y[i] <= output.upper()[[i]] + tol,
                        "RmsNorm forward mode large perturbation violation at dim {i}: \
                         x=[{x0}, {x1}], rms_norm(x)[{i}]={} not in [{}, {}]",
                        y[i], output.lower()[[i]], output.upper()[[i]]
                    );
                }
            }
        }
    }
}

// =============================================================================
// INSTANCENORM1D IBP SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// InstanceNorm1d conservative IBP soundness for a single channel
    /// with time dimension T=4.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_instancenorm_ibp_1ch_4t(
        (l0, u0) in valid_interval(2.0),
        (l1, u1) in valid_interval(2.0),
        (l2, u2) in valid_interval(2.0),
        (l3, u3) in valid_interval(2.0),
    ) {
        // InstanceNorm requires [C, T] shape; use C=1, T=4
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 4]), vec![l0, l1, l2, l3]
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 4]), vec![u0, u1, u2, u3]
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = InstanceNorm1dLayer::new_default(1, 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        // Test corners of the 4D box
        for corner in 0..16 {
            let x = arr1(&[
                if corner & 1 != 0 { u0 } else { l0 },
                if corner & 2 != 0 { u1 } else { l1 },
                if corner & 4 != 0 { u2 } else { l2 },
                if corner & 8 != 0 { u3 } else { l3 },
            ]);
            let y = instance_norm_channel(&x, 1.0, 0.0, 1e-5);

            for t in 0..4 {
                let tol = 1e-6;
                let out_idx = [0usize, t];
                prop_assert!(
                    output.lower()[out_idx.as_slice()] - tol <= y[t]
                        && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                    "InstanceNorm soundness violation: channel 0, t={}: {} not in [{}, {}]",
                    t, y[t], output.lower()[out_idx.as_slice()], output.upper()[out_idx.as_slice()]
                );
            }
        }
    }

    /// InstanceNorm1d conservative IBP soundness with custom ny/beta, C=2, T=3.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_instancenorm_ibp_with_params(
        (l0, u0) in valid_interval(1.5),
        (l1, u1) in valid_interval(1.5),
        (l2, u2) in valid_interval(1.5),
        (l3, u3) in valid_interval(1.5),
        (l4, u4) in valid_interval(1.5),
        (l5, u5) in valid_interval(1.5),
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
    ) {
        // Shape: [C=2, T=3]
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![l0, l1, l2, l3, l4, l5]
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![u0, u1, u2, u3, u4, u5]
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![g0, g1]);
        let beta = Array1::from_vec(vec![b0, b1]);
        let layer = InstanceNorm1dLayer::new(ny, beta, 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        // Test 8 corners per channel (each channel has T=3 independent dims)
        let lowers = [[l0, l1, l2], [l3, l4, l5]];
        let uppers = [[u0, u1, u2], [u3, u4, u5]];
        let gammas = [g0, g1];
        let betas = [b0, b1];

        for c in 0..2 {
            for corner in 0..8 {
                let x = arr1(&[
                    if corner & 1 != 0 { uppers[c][0] } else { lowers[c][0] },
                    if corner & 2 != 0 { uppers[c][1] } else { lowers[c][1] },
                    if corner & 4 != 0 { uppers[c][2] } else { lowers[c][2] },
                ]);
                let y = instance_norm_channel(&x, gammas[c], betas[c], 1e-5);

                for t in 0..3 {
                    let tol = 1e-6;
                    let out_idx = [c, t];
                    prop_assert!(
                        output.lower()[out_idx.as_slice()] - tol <= y[t]
                            && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                        "InstanceNorm with params violation: ch={}, t={}: {} not in [{}, {}]",
                        c, t, y[t],
                        output.lower()[out_idx.as_slice()],
                        output.upper()[out_idx.as_slice()]
                    );
                }
            }
        }
    }

    /// InstanceNorm1d forward-mode IBP soundness, C=1, T=3.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_instancenorm_forward_mode_ibp(
        center0 in -1.0f32..1.0,
        center1 in -1.0f32..1.0,
        center2 in -1.0f32..1.0,
        epsilon in 0.01f32..0.2,
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 3]),
            vec![center0 - epsilon, center1 - epsilon, center2 - epsilon],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 3]),
            vec![center0 + epsilon, center1 + epsilon, center2 + epsilon],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = InstanceNorm1dLayer::new_default(1, 1e-5).unwrap()
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        // Test all 8 corners + center
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
            let y = instance_norm_channel(&x, 1.0, 0.0, 1e-5);
            for t in 0..3 {
                let tol = 1e-6;
                let out_idx = [0usize, t];
                prop_assert!(
                    output.lower()[out_idx.as_slice()] - tol <= y[t]
                        && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                    "InstanceNorm forward mode violation at t={}: x={:?}, \
                     y[{t}]={} not in [{}, {}]",
                    t, pt, y[t],
                    output.lower()[out_idx.as_slice()],
                    output.upper()[out_idx.as_slice()]
                );
            }
        }
    }

    /// InstanceNorm1d forward-mode with large perturbation.
    ///
    /// Regression test for σ_min fix (#3159): exercises the path where
    /// the input box reaches zero-variance configurations.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_instancenorm_forward_mode_large_perturbation(
        c0 in -100.0f32..100.0,
        c1 in -100.0f32..100.0,
        hw0 in 0.5f32..100.0,
        hw1 in 0.5f32..100.0,
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 2]),
            vec![c0 - hw0, c1 - hw1],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 2]),
            vec![c0 + hw0, c1 + hw1],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = InstanceNorm1dLayer::new_default(1, 1e-5).unwrap()
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        let s0 = sample_points(c0 - hw0, c0 + hw0, 5);
        let s1 = sample_points(c1 - hw1, c1 + hw1, 5);
        for &x0 in &s0 {
            for &x1 in &s1 {
                let x = arr1(&[x0, x1]);
                let y = instance_norm_channel(&x, 1.0, 0.0, 1e-5);
                for t in 0..2 {
                    let tol = 1e-6;
                    let out_idx = [0usize, t];
                    prop_assert!(
                        output.lower()[out_idx.as_slice()] - tol <= y[t]
                            && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                        "InstanceNorm forward mode large perturbation violation at t={t}: \
                         x=[{x0}, {x1}], y[{t}]={} not in [{}, {}]",
                        y[t],
                        output.lower()[out_idx.as_slice()],
                        output.upper()[out_idx.as_slice()]
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// InstanceNorm1d forward-mode with T=4 (time dim > 2).
    ///
    /// For T=2, the per-channel output is effectively binary (z = ±1),
    /// masking Hessian formula errors. T≥3 exposes the correct σ²
    /// denominator requirement in the second-order remainder.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_instancenorm_forward_mode_4t(
        c0 in -100.0f32..100.0,
        c1 in -100.0f32..100.0,
        c2 in -100.0f32..100.0,
        c3 in -100.0f32..100.0,
        hw0 in 0.5f32..50.0,
        hw1 in 0.5f32..50.0,
        hw2 in 0.5f32..50.0,
        hw3 in 0.5f32..50.0,
    ) {
        // Shape: [C=1, T=4]
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 4]),
            vec![c0 - hw0, c1 - hw1, c2 - hw2, c3 - hw3],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 4]),
            vec![c0 + hw0, c1 + hw1, c2 + hw2, c3 + hw3],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = InstanceNorm1dLayer::new_default(1, 1e-5).unwrap()
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        // Test all 16 corners
        for corner in 0..16u32 {
            let x = arr1(&[
                if corner & 1 != 0 { c0 + hw0 } else { c0 - hw0 },
                if corner & 2 != 0 { c1 + hw1 } else { c1 - hw1 },
                if corner & 4 != 0 { c2 + hw2 } else { c2 - hw2 },
                if corner & 8 != 0 { c3 + hw3 } else { c3 - hw3 },
            ]);
            let y = instance_norm_channel(&x, 1.0, 0.0, 1e-5);
            for t in 0..4 {
                let tol = 1e-6;
                let out_idx = [0usize, t];
                prop_assert!(
                    output.lower()[out_idx.as_slice()] - tol <= y[t]
                        && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                    "InstanceNorm 4T forward mode violation at t={t}: corner={corner}, \
                     x={:?}, y[{t}]={} not in [{}, {}]",
                    x, y[t],
                    output.lower()[out_idx.as_slice()],
                    output.upper()[out_idx.as_slice()]
                );
            }
        }
    }
}

// =============================================================================
// ADAIN1D IBP SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// AdaIN1d conservative IBP soundness with C=2, T=2.
    ///
    /// Tests that for any x in [l, u], AdaIN(x) = style_gamma * InstanceNorm(x) + style_beta
    /// lies within the computed output bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_adain_ibp_2ch_2t(
        (l0, u0) in valid_interval(1.5),
        (l1, u1) in valid_interval(1.5),
        (l2, u2) in valid_interval(1.5),
        (l3, u3) in valid_interval(1.5),
        sg0 in 0.5f32..2.0,
        sg1 in 0.5f32..2.0,
        sb0 in -1.0f32..1.0,
        sb1 in -1.0f32..1.0,
    ) {
        // Shape: [C=2, T=2]
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), vec![l0, l1, l2, l3]
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), vec![u0, u1, u2, u3]
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let inn = InstanceNorm1dLayer::new_default(2, 1e-5).unwrap();
        let style_gamma = Array1::from_vec(vec![sg0, sg1]);
        let style_beta = Array1::from_vec(vec![sb0, sb1]);
        let layer = AdaIN1dLayer::new(inn, style_gamma, style_beta).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        // Test all 4 corners per channel (each channel has T=2 independent dims)
        let lowers = [[l0, l1], [l2, l3]];
        let uppers = [[u0, u1], [u2, u3]];
        let sgs = [sg0, sg1];
        let sbs = [sb0, sb1];

        for c in 0..2 {
            for corner in 0..4 {
                let x = arr1(&[
                    if corner & 1 != 0 { uppers[c][0] } else { lowers[c][0] },
                    if corner & 2 != 0 { uppers[c][1] } else { lowers[c][1] },
                ]);
                // AdaIN = style_gamma * InstanceNorm(x, ny=1, beta=0) + style_beta
                let y = adain_eval_channel(&x, 1.0, 0.0, sgs[c], sbs[c], 1e-5);

                for t in 0..2 {
                    let tol = 1e-6;
                    let out_idx = [c, t];
                    prop_assert!(
                        output.lower()[out_idx.as_slice()] - tol <= y[t]
                            && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                        "AdaIN IBP soundness violation: ch={c}, t={t}: {} not in [{}, {}]",
                        y[t],
                        output.lower()[out_idx.as_slice()],
                        output.upper()[out_idx.as_slice()]
                    );
                }
            }
        }
    }

    /// AdaIN1d conservative IBP soundness with negative style_gamma.
    ///
    /// Exercises the sign-flip path in apply_style_affine where negative
    /// style_gamma swaps the lower/upper bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_adain_ibp_negative_style_gamma(
        (l0, u0) in valid_interval(1.5),
        (l1, u1) in valid_interval(1.5),
        (l2, u2) in valid_interval(1.5),
        sg0 in -2.0f32..-0.5,
        sb0 in -1.0f32..1.0,
    ) {
        // Shape: [C=1, T=3]
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 3]), vec![l0, l1, l2]
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 3]), vec![u0, u1, u2]
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let inn = InstanceNorm1dLayer::new_default(1, 1e-5).unwrap();
        let style_gamma = Array1::from_vec(vec![sg0]);
        let style_beta = Array1::from_vec(vec![sb0]);
        let layer = AdaIN1dLayer::new(inn, style_gamma, style_beta).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        // Test all 8 corners
        for corner in 0..8 {
            let x = arr1(&[
                if corner & 1 != 0 { u0 } else { l0 },
                if corner & 2 != 0 { u1 } else { l1 },
                if corner & 4 != 0 { u2 } else { l2 },
            ]);
            let y = adain_eval_channel(&x, 1.0, 0.0, sg0, sb0, 1e-5);

            for t in 0..3 {
                let tol = 1e-6;
                let out_idx = [0usize, t];
                prop_assert!(
                    output.lower()[out_idx.as_slice()] - tol <= y[t]
                        && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                    "AdaIN neg style_gamma violation: t={t}: {} not in [{}, {}]",
                    y[t],
                    output.lower()[out_idx.as_slice()],
                    output.upper()[out_idx.as_slice()]
                );
            }
        }
    }

    /// AdaIN1d IBP soundness with custom InstanceNorm ny/beta and style params.
    ///
    /// Tests the full parameter space: non-default inner InstanceNorm parameters
    /// combined with style transform. C=2, T=2.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_adain_ibp_with_params(
        (l0, u0) in valid_interval(1.5),
        (l1, u1) in valid_interval(1.5),
        (l2, u2) in valid_interval(1.5),
        (l3, u3) in valid_interval(1.5),
        ig0 in 0.5f32..2.0,
        ig1 in 0.5f32..2.0,
        ib0 in -0.5f32..0.5,
        ib1 in -0.5f32..0.5,
        sg0 in -1.5f32..1.5,
        sg1 in -1.5f32..1.5,
    ) {
        // Skip style_gamma near zero: when |sg| < 0.1, the output approaches
        // a constant (style_beta), making bounds trivially tight and the tolerance
        // comparison less meaningful for catching actual IBP bugs.
        prop_assume!(sg0.abs() > 0.1);
        prop_assume!(sg1.abs() > 0.1);

        // Shape: [C=2, T=2]
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), vec![l0, l1, l2, l3]
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), vec![u0, u1, u2, u3]
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![ig0, ig1]);
        let beta = Array1::from_vec(vec![ib0, ib1]);
        let inn = InstanceNorm1dLayer::new(ny, beta, 1e-5).unwrap();
        let style_gamma = Array1::from_vec(vec![sg0, sg1]);
        let style_beta = Array1::from_vec(vec![0.0, 0.0]);
        let layer = AdaIN1dLayer::new(inn, style_gamma, style_beta).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        let lowers = [[l0, l1], [l2, l3]];
        let uppers = [[u0, u1], [u2, u3]];
        let igs = [ig0, ig1];
        let ibs = [ib0, ib1];
        let sgs = [sg0, sg1];

        for c in 0..2 {
            for corner in 0..4 {
                let x = arr1(&[
                    if corner & 1 != 0 { uppers[c][0] } else { lowers[c][0] },
                    if corner & 2 != 0 { uppers[c][1] } else { lowers[c][1] },
                ]);
                let y = adain_eval_channel(&x, igs[c], ibs[c], sgs[c], 0.0, 1e-5);

                for t in 0..2 {
                    let tol = 1e-6;
                    let out_idx = [c, t];
                    prop_assert!(
                        output.lower()[out_idx.as_slice()] - tol <= y[t]
                            && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                        "AdaIN with params violation: ch={c}, t={t}: {} not in [{}, {}]",
                        y[t],
                        output.lower()[out_idx.as_slice()],
                        output.upper()[out_idx.as_slice()]
                    );
                }
            }
        }
    }
}

// =============================================================================
// LAYERNORM IBP SOUNDNESS TESTS
// =============================================================================
//
// LayerNorm: y_i = ny_i * (x_i - mean(x)) / sqrt(var(x) + eps) + beta_i
//
// These proptest soundness tests were missing despite being present for all
// other normalization layers (RmsNorm, InstanceNorm1d, AdaIN1d) since #3160.
// LayerNorm is the most commonly used normalization layer in transformers.
//
// Part of #3195.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// LayerNorm conservative IBP soundness: for any x in [l, u],
    /// layernorm(x) is within computed bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_layernorm_ibp_3d(
        (l0, u0) in valid_interval(3.0),
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn()
        ).unwrap();

        let ny = Array1::ones(3);
        let beta = Array1::zeros(3);
        let layer = LayerNormLayer::new_default(3, 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        // Test all 8 corners
        for corner in 0..8 {
            let x = arr1(&[
                if corner & 1 != 0 { u0 } else { l0 },
                if corner & 2 != 0 { u1 } else { l1 },
                if corner & 4 != 0 { u2 } else { l2 },
            ]);
            let y = layernorm(&x, &ny, &beta, 1e-5);

            for i in 0..3 {
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[i]] - tol <= y[i] && y[i] <= output.upper()[[i]] + tol,
                    "LayerNorm soundness violation: layernorm({:?})[{}]={} not in [{}, {}]",
                    x, i, y[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// LayerNorm conservative IBP soundness with custom ny/beta.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_layernorm_ibp_with_gamma(
        (l0, u0) in valid_interval(2.0),
        (l1, u1) in valid_interval(2.0),
        (l2, u2) in valid_interval(2.0),
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

        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let beta = Array1::from_vec(vec![b0, b1, b2]);
        let layer = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        for corner in 0..8 {
            let x = arr1(&[
                if corner & 1 != 0 { u0 } else { l0 },
                if corner & 2 != 0 { u1 } else { l1 },
                if corner & 4 != 0 { u2 } else { l2 },
            ]);
            let y = layernorm(&x, &ny, &beta, 1e-5);

            for i in 0..3 {
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[i]] - tol <= y[i] && y[i] <= output.upper()[[i]] + tol,
                    "LayerNorm with ny violation at {}: {} not in [{}, {}]",
                    i, y[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// LayerNorm forward-mode IBP soundness with moderate perturbation.
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

        let ny = Array1::ones(3);
        let beta = Array1::zeros(3);
        let layer = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5).unwrap()
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        // Test all 8 corners + center
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
            let y = layernorm(&x, &ny, &beta, 1e-5);
            for i in 0..3 {
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[i]] - tol <= y[i] && y[i] <= output.upper()[[i]] + tol,
                    "LayerNorm forward mode violation at {}: x={:?}, \
                     layernorm(x)[{i}]={} not in [{}, {}]",
                    i, pt, y[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// LayerNorm forward-mode IBP soundness with large perturbation.
    ///
    /// Regression test for the σ_min fix (#3098): large boxes can include
    /// points near constant-input configurations where variance is very small,
    /// causing output to diverge from center-point predictions.
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

        let ny = Array1::ones(2);
        let beta = Array1::zeros(2);
        let layer = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5).unwrap()
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        let s0 = sample_points(c0 - hw0, c0 + hw0, 5);
        let s1 = sample_points(c1 - hw1, c1 + hw1, 5);
        for &x0 in &s0 {
            for &x1 in &s1 {
                let x = arr1(&[x0, x1]);
                let y = layernorm(&x, &ny, &beta, 1e-5);
                for i in 0..2 {
                    let tol = 1e-6;
                    prop_assert!(
                        output.lower()[[i]] - tol <= y[i] && y[i] <= output.upper()[[i]] + tol,
                        "LayerNorm forward mode large perturbation violation at dim {i}: \
                         x=[{x0}, {x1}], layernorm(x)[{i}]={} not in [{}, {}]",
                        y[i], output.lower()[[i]], output.upper()[[i]]
                    );
                }
            }
        }
    }
}
