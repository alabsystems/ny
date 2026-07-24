// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP soundness proptests for GroupNorm.
//!
//! GroupNorm: y[c, t] = ny[c] * (x[c, t] - mean_g) / sqrt(var_g + eps) + beta[c]
//! where mean_g, var_g are over all cpg * T elements in group g.
//!
//! GroupNorm generalizes InstanceNorm (num_groups = C) and approaches LayerNorm
//! (num_groups = 1). Tests cover conservative IBP and forward-mode IBP.
//!
//! Part of #3258.

use crate::layers::common::BoundPropagation;
use crate::layers::normalization::GroupNormLayer;
use ndarray::{Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{group_norm_group, sample_points, valid_interval};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// GroupNorm conservative IBP soundness: 2 groups, 2 channels/group, T=3.
    /// Each group has cpg=2 channels x 3 time = 6 elements.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_groupnorm_ibp_2g_2c_3t(
        (l0, u0) in valid_interval(2.0),
        (l1, u1) in valid_interval(2.0),
        (l2, u2) in valid_interval(2.0),
        (l3, u3) in valid_interval(2.0),
        (l4, u4) in valid_interval(2.0),
        (l5, u5) in valid_interval(2.0),
    ) {
        // Shape: [C=4, T=3], 2 groups of 2 channels each.
        // Only test group 0 (channels 0-1) to keep corner count manageable.
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[4, 3]),
            vec![l0, l1, l2, l3, l4, l5,
                 // Group 1: fixed values (don't affect group 0 bounds)
                 -0.5, 0.0, 0.5, -0.3, 0.1, 0.4],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[4, 3]),
            vec![u0, u1, u2, u3, u4, u5,
                 -0.5, 0.0, 0.5, -0.3, 0.1, 0.4],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = GroupNormLayer::new_default(4, 2, 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        // Test all 64 corners of the 6D box (group 0 elements)
        for corner in 0..64_u32 {
            let vals = [
                if corner & 1 != 0 { u0 } else { l0 },
                if corner & 2 != 0 { u1 } else { l1 },
                if corner & 4 != 0 { u2 } else { l2 },
                if corner & 8 != 0 { u3 } else { l3 },
                if corner & 16 != 0 { u4 } else { l4 },
                if corner & 32 != 0 { u5 } else { l5 },
            ];
            let y = group_norm_group(&vals, &[1.0, 1.0], &[0.0, 0.0], 2, 3, 1e-5);

            for (i, &y_val) in y.iter().enumerate().take(6) {
                let c = i / 3;
                let t = i % 3;
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[c, t]] - tol <= y_val
                        && y_val <= output.upper()[[c, t]] + tol,
                    "GroupNorm IBP soundness violation: corner={corner} elem {i}: \
                     val={:.6} not in [{:.6}, {:.6}]",
                    y_val, output.lower()[[c, t]], output.upper()[[c, t]]
                );
            }
        }
    }

    /// GroupNorm conservative IBP soundness with custom ny/beta.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_groupnorm_ibp_with_params(
        (l0, u0) in valid_interval(1.5),
        (l1, u1) in valid_interval(1.5),
        (l2, u2) in valid_interval(1.5),
        (l3, u3) in valid_interval(1.5),
        (l4, u4) in valid_interval(1.5),
        (l5, u5) in valid_interval(1.5),
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
    ) {
        // Shape: [C=2, T=3], 1 group (LayerNorm-like)
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![l0, l1, l2, l3, l4, l5]
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![u0, u1, u2, u3, u4, u5]
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![g0, g1]);
        let beta = Array1::from_vec(vec![0.0, 0.0]);
        let layer = GroupNormLayer::new(ny, beta, 1, 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        for corner in 0..64_u32 {
            let vals = [
                if corner & 1 != 0 { u0 } else { l0 },
                if corner & 2 != 0 { u1 } else { l1 },
                if corner & 4 != 0 { u2 } else { l2 },
                if corner & 8 != 0 { u3 } else { l3 },
                if corner & 16 != 0 { u4 } else { l4 },
                if corner & 32 != 0 { u5 } else { l5 },
            ];
            let y = group_norm_group(&vals, &[g0, g1], &[0.0, 0.0], 2, 3, 1e-5);

            for (i, &y_val) in y.iter().enumerate().take(6) {
                let c = i / 3;
                let t = i % 3;
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[c, t]] - tol <= y_val
                        && y_val <= output.upper()[[c, t]] + tol,
                    "GroupNorm with params violation: corner={corner} elem {i}: \
                     val={:.6} not in [{:.6}, {:.6}]",
                    y_val, output.lower()[[c, t]], output.upper()[[c, t]]
                );
            }
        }
    }

    /// GroupNorm forward-mode IBP soundness.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_groupnorm_forward_mode_ibp(
        c0 in -1.0f32..1.0,
        c1 in -1.0f32..1.0,
        c2 in -1.0f32..1.0,
        c3 in -1.0f32..1.0,
        c4 in -1.0f32..1.0,
        c5 in -1.0f32..1.0,
        epsilon in 0.01f32..0.2,
    ) {
        // Shape: [C=2, T=3], 1 group, forward-mode
        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - epsilon).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + epsilon).collect();

        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), lower_v.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), upper_v.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = GroupNormLayer::new_default(2, 1, 1e-5).unwrap()
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        // Test all 64 corners
        for corner in 0..64_u32 {
            let vals: Vec<f32> = (0..6)
                .map(|i| {
                    if corner & (1 << i) != 0 { upper_v[i] } else { lower_v[i] }
                })
                .collect();
            let y = group_norm_group(&vals, &[1.0, 1.0], &[0.0, 0.0], 2, 3, 1e-5);

            for (i, &y_val) in y.iter().enumerate().take(6) {
                let c = i / 3;
                let t = i % 3;
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[c, t]] - tol <= y_val
                        && y_val <= output.upper()[[c, t]] + tol,
                    "GroupNorm forward-mode IBP violation: corner={corner} elem {i}: \
                     val={:.6} not in [{:.6}, {:.6}]",
                    y_val, output.lower()[[c, t]], output.upper()[[c, t]]
                );
            }
        }
    }

    /// GroupNorm forward-mode IBP soundness with large perturbation.
    ///
    /// Regression test for sigma_min fix -- exercises large boxes where
    /// variance can approach zero within the input range.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_groupnorm_forward_mode_large_perturbation(
        c0 in -50.0f32..50.0,
        c1 in -50.0f32..50.0,
        c2 in -50.0f32..50.0,
        hw0 in 0.5f32..50.0,
        hw1 in 0.5f32..50.0,
        hw2 in 0.5f32..50.0,
    ) {
        // Shape: [C=1, T=3], 1 group (single channel, like LayerNorm)
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[1, 3]),
            vec![c0 - hw0, c1 - hw1, c2 - hw2],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[1, 3]),
            vec![c0 + hw0, c1 + hw1, c2 + hw2],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = GroupNormLayer::new_default(1, 1, 1e-5).unwrap()
            .with_forward_mode(true);
        let output = layer.propagate_ibp(&input).unwrap();

        let s0 = sample_points(c0 - hw0, c0 + hw0, 5);
        let s1 = sample_points(c1 - hw1, c1 + hw1, 5);
        let s2 = sample_points(c2 - hw2, c2 + hw2, 5);
        for &x0 in &s0 {
            for &x1 in &s1 {
                for &x2 in &s2 {
                    let vals = [x0, x1, x2];
                    let y = group_norm_group(&vals, &[1.0], &[0.0], 1, 3, 1e-5);
                    for (t, &y_val) in y.iter().enumerate().take(3) {
                        let tol = 1e-6;
                        let out_idx = [0usize, t];
                        prop_assert!(
                            output.lower()[out_idx.as_slice()] - tol <= y_val
                                && y_val <= output.upper()[out_idx.as_slice()] + tol,
                            "GroupNorm forward mode large perturbation violation at t={t}: \
                             x=[{x0}, {x1}, {x2}], y[{t}]={} not in [{}, {}]",
                            y_val,
                            output.lower()[out_idx.as_slice()],
                            output.upper()[out_idx.as_slice()]
                        );
                    }
                }
            }
        }
    }

    /// GroupNorm conservative IBP soundness with negative ny.
    ///
    /// Exercises the `g < 0` branch where ny sign flips lower/upper bounds.
    /// All other normalization layers have negative ny IBP proptests in
    /// `normalization_ibp_extended.rs`, but GroupNorm was missing.
    /// Part of #3333.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_groupnorm_ibp_negative_gamma(
        (l0, u0) in valid_interval(1.5),
        (l1, u1) in valid_interval(1.5),
        (l2, u2) in valid_interval(1.5),
        (l3, u3) in valid_interval(1.5),
        (l4, u4) in valid_interval(1.5),
        (l5, u5) in valid_interval(1.5),
        g0 in -2.0f32..-0.5,
        g1 in -2.0f32..-0.5,
    ) {
        // Shape: [C=2, T=3], 1 group (LayerNorm-like) with negative ny.
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![l0, l1, l2, l3, l4, l5]
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![u0, u1, u2, u3, u4, u5]
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![g0, g1]);
        let beta = Array1::from_vec(vec![0.0, 0.0]);
        let layer = GroupNormLayer::new(ny, beta, 1, 1e-5).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        for corner in 0..64_u32 {
            let vals = [
                if corner & 1 != 0 { u0 } else { l0 },
                if corner & 2 != 0 { u1 } else { l1 },
                if corner & 4 != 0 { u2 } else { l2 },
                if corner & 8 != 0 { u3 } else { l3 },
                if corner & 16 != 0 { u4 } else { l4 },
                if corner & 32 != 0 { u5 } else { l5 },
            ];
            let y = group_norm_group(&vals, &[g0, g1], &[0.0, 0.0], 2, 3, 1e-5);

            for (i, &y_val) in y.iter().enumerate().take(6) {
                let c = i / 3;
                let t = i % 3;
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[c, t]] - tol <= y_val
                        && y_val <= output.upper()[[c, t]] + tol,
                    "GroupNorm negative ny IBP violation: corner={corner} elem {i}: \
                     val={:.6} not in [{:.6}, {:.6}]",
                    y_val, output.lower()[[c, t]], output.upper()[[c, t]]
                );
            }
        }
    }
}
