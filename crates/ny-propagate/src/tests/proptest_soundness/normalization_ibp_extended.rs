// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Extended IBP soundness proptests for normalization layers.
//!
//! Covers gaps not present in `normalization_ibp.rs`:
//! - AdaIN1d forward-mode IBP (delegates to InstanceNorm forward-mode + style affine)
//! - Negative ny for RmsNorm, InstanceNorm1d, LayerNorm (exercises g<0 bound-swap)
//! - AdaIN1d conservative IBP with large perturbation (σ_min regression)
//!
//! Part of #3326.

use crate::layers::common::BoundPropagation;
use crate::layers::normalization::AdaIN1dLayer;
use crate::layers::normalization::LayerNormMode;
use crate::layers::{InstanceNorm1dLayer, LayerNormLayer, RmsNormLayer};
use ndarray::{arr1, Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{
    adain_eval_channel, instance_norm_channel, layernorm, layernorm_mean_only, rms_norm,
    sample_points, valid_interval,
};

// =============================================================================
// ADAIN1D FORWARD-MODE IBP SOUNDNESS TESTS
// =============================================================================
//
// AdaIN1d delegates to InstanceNorm1d internally, then applies a style affine.
// The forward-mode path is exercised when with_forward_mode(true) is set.
// These tests verify the composition: forward-mode InstanceNorm + style affine.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// AdaIN1d forward-mode IBP soundness with moderate perturbation.
    ///
    /// Exercises the composition: forward-mode InstanceNorm (Jacobian-based)
    /// followed by style_gamma * z + style_beta affine transform.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_adain_forward_mode_ibp(
        center0 in -1.0f32..1.0,
        center1 in -1.0f32..1.0,
        center2 in -1.0f32..1.0,
        epsilon in 0.01f32..0.2,
        sg0 in 0.5f32..2.0,
        sb0 in -1.0f32..1.0,
    ) {
        // Shape: [C=1, T=3]
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 3]),
            vec![center0 - epsilon, center1 - epsilon, center2 - epsilon],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 3]),
            vec![center0 + epsilon, center1 + epsilon, center2 + epsilon],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let inn = InstanceNorm1dLayer::new_default(1, 1e-5).unwrap();
        let style_gamma = Array1::from_vec(vec![sg0]);
        let style_beta = Array1::from_vec(vec![sb0]);
        let layer = AdaIN1dLayer::new(inn, style_gamma, style_beta).unwrap()
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
            let y = adain_eval_channel(&x, 1.0, 0.0, sg0, sb0, 1e-5);
            for t in 0..3 {
                let tol = 1e-6;
                let out_idx = [0usize, t];
                prop_assert!(
                    output.lower()[out_idx.as_slice()] - tol <= y[t]
                        && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                    "AdaIN forward mode violation at t={t}: x={:?}, \
                     y[{t}]={} not in [{}, {}]",
                    pt, y[t],
                    output.lower()[out_idx.as_slice()],
                    output.upper()[out_idx.as_slice()]
                );
            }
        }
    }

    /// AdaIN1d forward-mode IBP with large perturbation.
    ///
    /// Regression test for σ_min fix (#3159): exercises the path where the
    /// inner InstanceNorm forward-mode encounters large input boxes that
    /// include near-zero-variance configurations.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_adain_forward_mode_large_perturbation(
        c0 in -100.0f32..100.0,
        c1 in -100.0f32..100.0,
        hw0 in 0.5f32..100.0,
        hw1 in 0.5f32..100.0,
        sg0 in -2.0f32..2.0,
        sb0 in -1.0f32..1.0,
    ) {
        // Skip style_gamma near zero: output approaches constant, making
        // tolerance checks less meaningful for detecting IBP bugs.
        prop_assume!(sg0.abs() > 0.1);

        // Shape: [C=1, T=2]
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 2]),
            vec![c0 - hw0, c1 - hw1],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 2]),
            vec![c0 + hw0, c1 + hw1],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let inn = InstanceNorm1dLayer::new_default(1, 1e-5).unwrap();
        let style_gamma = Array1::from_vec(vec![sg0]);
        let style_beta = Array1::from_vec(vec![sb0]);
        let layer = AdaIN1dLayer::new(inn, style_gamma, style_beta).unwrap()
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        let s0 = sample_points(c0 - hw0, c0 + hw0, 5);
        let s1 = sample_points(c1 - hw1, c1 + hw1, 5);
        for &x0 in &s0 {
            for &x1 in &s1 {
                let x = arr1(&[x0, x1]);
                let y = adain_eval_channel(&x, 1.0, 0.0, sg0, sb0, 1e-5);
                for t in 0..2 {
                    let tol = 1e-6;
                    let out_idx = [0usize, t];
                    prop_assert!(
                        output.lower()[out_idx.as_slice()] - tol <= y[t]
                            && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                        "AdaIN forward mode large perturbation violation at t={t}: \
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

// =============================================================================
// NEGATIVE NY IBP SOUNDNESS TESTS
// =============================================================================
//
// The proptests in `normalization_ibp.rs` use only positive ny (0.5..2.0).
// Negative ny exercises the `if g >= 0.0 ... else ...` branch that
// swaps lower/upper bounds. Without testing negative ny, this branch
// is only covered by unit tests, not property-based tests.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// RmsNorm conservative IBP soundness with negative ny.
    ///
    /// Exercises the `g < 0` branch in fallback and conservative paths
    /// where lower/upper bounds are swapped.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rmsnorm_ibp_negative_gamma(
        (l0, u0) in valid_interval(2.0),
        (l1, u1) in valid_interval(2.0),
        (l2, u2) in valid_interval(2.0),
        g0 in -2.0f32..-0.5,
        g1 in -2.0f32..-0.5,
        g2 in -2.0f32..-0.5,
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
                    "RmsNorm negative ny violation at {}: {} not in [{}, {}]",
                    i, y[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// InstanceNorm1d conservative IBP soundness with negative ny.
    ///
    /// Exercises the `g < 0` branch in ibp_conservative_channel where
    /// the ratio bounds are flipped before applying ny.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_instancenorm_ibp_negative_gamma(
        (l0, u0) in valid_interval(1.5),
        (l1, u1) in valid_interval(1.5),
        (l2, u2) in valid_interval(1.5),
        g0 in -2.0f32..-0.5,
        b0 in -1.0f32..1.0,
    ) {
        // Shape: [C=1, T=3]
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 3]), vec![l0, l1, l2]
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 3]), vec![u0, u1, u2]
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![g0]);
        let beta = Array1::from_vec(vec![b0]);
        let layer = InstanceNorm1dLayer::new(ny, beta, 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        for corner in 0..8 {
            let x = arr1(&[
                if corner & 1 != 0 { u0 } else { l0 },
                if corner & 2 != 0 { u1 } else { l1 },
                if corner & 4 != 0 { u2 } else { l2 },
            ]);
            let y = instance_norm_channel(&x, g0, b0, 1e-5);

            for t in 0..3 {
                let tol = 1e-6;
                let out_idx = [0usize, t];
                prop_assert!(
                    output.lower()[out_idx.as_slice()] - tol <= y[t]
                        && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                    "InstanceNorm negative ny violation: t={t}: {} not in [{}, {}]",
                    y[t],
                    output.lower()[out_idx.as_slice()],
                    output.upper()[out_idx.as_slice()]
                );
            }
        }
    }

    /// LayerNorm conservative IBP soundness with negative ny.
    ///
    /// Exercises the `g < 0` branch in both the fallback and conservative
    /// IBP paths where ny sign flips the lower/upper output bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_layernorm_ibp_negative_gamma(
        (l0, u0) in valid_interval(2.0),
        (l1, u1) in valid_interval(2.0),
        (l2, u2) in valid_interval(2.0),
        g0 in -2.0f32..-0.5,
        g1 in -2.0f32..-0.5,
        g2 in -2.0f32..-0.5,
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
                    "LayerNorm negative ny violation at {}: {} not in [{}, {}]",
                    i, y[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// LayerNorm MeanOnly forward-mode IBP soundness.
    ///
    /// MeanOnly: y_i = ny_i * (x_i - mean(X)) + beta_i is an affine function.
    /// The forward-mode IBP uses interval mean bounds for soundness.
    /// No IBP proptest existed for MeanOnly (only CROWN proptests).
    /// Part of #3333.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_layernorm_mean_only_forward_mode_ibp(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        epsilon in 0.01f32..0.3,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        b2 in -1.0f32..1.0,
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[3]),
            vec![c0 - epsilon, c1 - epsilon, c2 - epsilon],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[3]),
            vec![c0 + epsilon, c1 + epsilon, c2 + epsilon],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let beta = Array1::from_vec(vec![b0, b1, b2]);
        let layer = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5)
            .unwrap()
            .with_mode(LayerNormMode::MeanOnly)
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        for corner in 0..8_u32 {
            let x = arr1(&[
                if corner & 1 != 0 { c0 + epsilon } else { c0 - epsilon },
                if corner & 2 != 0 { c1 + epsilon } else { c1 - epsilon },
                if corner & 4 != 0 { c2 + epsilon } else { c2 - epsilon },
            ]);
            let y = layernorm_mean_only(&x, &ny, &beta);

            for i in 0..3 {
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[i]] - tol <= y[i] && y[i] <= output.upper()[[i]] + tol,
                    "LayerNorm MeanOnly forward-mode IBP violation at {i}: {} not in [{}, {}]",
                    y[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }

    /// LayerNorm MeanOnly forward-mode IBP with large perturbation.
    ///
    /// Exercises MeanOnly with wide input boxes. MeanOnly is affine, so
    /// there's no sigma_min concern, but large perturbation tests the
    /// interval arithmetic precision with wide ranges.
    /// Part of #3333.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_layernorm_mean_only_forward_mode_large_perturbation(
        c0 in -100.0f32..100.0,
        c1 in -100.0f32..100.0,
        c2 in -100.0f32..100.0,
        hw0 in 0.5f32..100.0,
        hw1 in 0.5f32..100.0,
        hw2 in 0.5f32..100.0,
        g0 in -2.0f32..2.0,
        g1 in -2.0f32..2.0,
        g2 in -2.0f32..2.0,
    ) {
        // Ensure at least one ny is non-trivial.
        prop_assume!(g0.abs() > 0.1 || g1.abs() > 0.1 || g2.abs() > 0.1);

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[3]),
            vec![c0 - hw0, c1 - hw1, c2 - hw2],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[3]),
            vec![c0 + hw0, c1 + hw1, c2 + hw2],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let beta = Array1::zeros(3);
        let layer = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5)
            .unwrap()
            .with_mode(LayerNormMode::MeanOnly)
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        let s0 = sample_points(c0 - hw0, c0 + hw0, 5);
        let s1 = sample_points(c1 - hw1, c1 + hw1, 5);
        let s2 = sample_points(c2 - hw2, c2 + hw2, 5);
        for &x0 in &s0 {
            for &x1 in &s1 {
                for &x2 in &s2 {
                    let x = arr1(&[x0, x1, x2]);
                    let y = layernorm_mean_only(&x, &ny, &beta);
                    for i in 0..3 {
                        // Directed rounding guarantees exact containment;
                        // 1e-6 is a small buffer for platform f32 behavior.
                        let tol = 1e-6;
                        prop_assert!(
                            output.lower()[[i]] - tol <= y[i]
                                && y[i] <= output.upper()[[i]] + tol,
                            "LayerNorm MeanOnly large perturbation violation at {i}: \
                             x=[{x0}, {x1}, {x2}], y={} not in [{}, {}]",
                            y[i], output.lower()[[i]], output.upper()[[i]]
                        );
                    }
                }
            }
        }
    }

    /// AdaIN1d conservative IBP soundness with large perturbation.
    ///
    /// Regression test for σ_min fix (#3159): exercises conservative IBP
    /// (not forward-mode) with wide input boxes where variance can be very
    /// small. All other normalization layers have this regression variant.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_adain_ibp_large_perturbation(
        c0 in -100.0f32..100.0,
        c1 in -100.0f32..100.0,
        hw0 in 0.5f32..100.0,
        hw1 in 0.5f32..100.0,
        sg0 in -2.0f32..2.0,
        sb0 in -1.0f32..1.0,
    ) {
        prop_assume!(sg0.abs() > 0.1);

        // Shape: [C=1, T=2]
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 2]),
            vec![c0 - hw0, c1 - hw1],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 2]),
            vec![c0 + hw0, c1 + hw1],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let inn = InstanceNorm1dLayer::new_default(1, 1e-5).unwrap();
        let style_gamma = Array1::from_vec(vec![sg0]);
        let style_beta = Array1::from_vec(vec![sb0]);
        let layer = AdaIN1dLayer::new(inn, style_gamma, style_beta).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        let s0 = sample_points(c0 - hw0, c0 + hw0, 5);
        let s1 = sample_points(c1 - hw1, c1 + hw1, 5);
        for &x0 in &s0 {
            for &x1 in &s1 {
                let x = arr1(&[x0, x1]);
                let y = adain_eval_channel(&x, 1.0, 0.0, sg0, sb0, 1e-5);
                for t in 0..2 {
                    // The f32 reference can compute z slightly above the
                    // theoretical max_norm=sqrt(T-1) due to variance rounding,
                    // yielding ~2e-6 gap. Use 1e-5 for this margin.
                    let tol = 1e-5;
                    let out_idx = [0usize, t];
                    prop_assert!(
                        output.lower()[out_idx.as_slice()] - tol <= y[t]
                            && y[t] <= output.upper()[out_idx.as_slice()] + tol,
                        "AdaIN large perturbation violation at t={t}: \
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
