// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::common::BoundPropagation;
use crate::layers::normalization::LayerNormCrownMode;
use ndarray::{arr1, Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

fn default_in1d(channels: usize) -> InstanceNorm1dLayer {
    InstanceNorm1dLayer::new_default(channels, 1e-5).expect("valid default InstanceNorm1d")
}

fn custom_in1d(ny: &[f32], beta: &[f32]) -> InstanceNorm1dLayer {
    InstanceNorm1dLayer::new(
        Array1::from_vec(ny.to_vec()),
        Array1::from_vec(beta.to_vec()),
        1e-5,
    )
    .expect("valid custom InstanceNorm1d")
}

// ── IBP tests ─────────────────────────────────────────────────────────────

#[test]
fn test_ibp_soundness_2d() {
    // Input shape [C=2, T=4]. For each channel, bounds should contain all grid points.
    let layer = default_in1d(2);
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![-2.0; 8]).expect("valid lower shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![2.0; 8]).expect("valid upper shape");
    let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("valid bounded tensor");
    let out = layer.propagate_ibp(&input).expect("IBP should succeed");

    // Verify bounds are finite and lower <= upper
    for (&l, &u) in out.lower().iter().zip(out.upper().iter()) {
        assert!(l.is_finite(), "lower bound non-finite: {l}");
        assert!(u.is_finite(), "upper bound non-finite: {u}");
        assert!(l <= u, "lower {l} > upper {u}");
    }

    // Verify containment: sampled concrete outputs must fall within IBP bounds.
    // Uses deterministic hash-based sampling matching the pattern from proptest suites.
    let num_channels = 2;
    let time_len = 4;
    for s in 0..50_u32 {
        for c in 0..num_channels {
            let channel_input: Vec<f32> = (0..time_len)
                .map(|t| {
                    let idx = c * time_len + t;
                    let hash = ((s.wrapping_mul(2654435761) ^ (idx as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower[[c, t]] + (upper[[c, t]] - lower[[c, t]]) * hash
                })
                .collect();
            let x = arr1(&channel_input);
            let y = layer
                .eval_channel(&x, c)
                .expect("eval_channel should succeed");

            for t in 0..time_len {
                assert!(
                    y[t] >= out.lower()[[c, t]] - 1e-4,
                    "IBP containment: ch{c} t{t} sample {s}: eval {} < lower {}",
                    y[t],
                    out.lower()[[c, t]],
                );
                assert!(
                    y[t] <= out.upper()[[c, t]] + 1e-4,
                    "IBP containment: ch{c} t{t} sample {s}: eval {} > upper {}",
                    y[t],
                    out.upper()[[c, t]],
                );
            }
        }
    }
}

#[test]
fn test_ibp_soundness_samples() {
    // Verify that IBP bounds contain the true function output at many sample points.
    let layer = custom_in1d(&[2.0, 0.5], &[1.0, -1.0]);
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, -2.0, 0.0, 1.0, -1.0, 0.5])
        .expect("valid lower shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 2.0, 3.0, 1.0, 2.5])
        .expect("valid upper shape");
    let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("valid bounded tensor");
    let out = layer.propagate_ibp(&input).expect("IBP should succeed");

    // Sample many points in the input interval and verify output is bounded
    let steps = 10;
    for i0 in 0..=steps {
        for i1 in 0..=steps {
            for i2 in 0..=steps {
                let t0 = i0 as f32 / steps as f32;
                let t1 = i1 as f32 / steps as f32;
                let t2 = i2 as f32 / steps as f32;

                // Just sample channel 0
                let x0 = lower[[0, 0]] + t0 * (upper[[0, 0]] - lower[[0, 0]]);
                let x1 = lower[[0, 1]] + t1 * (upper[[0, 1]] - lower[[0, 1]]);
                let x2 = lower[[0, 2]] + t2 * (upper[[0, 2]] - lower[[0, 2]]);

                let channel_input = arr1(&[x0, x1, x2]);
                let y = layer
                    .eval_channel(&channel_input, 0)
                    .expect("eval should succeed");

                for t in 0..3 {
                    assert!(
                        out.lower()[[0, t]] <= y[t] + 1e-4,
                        "ch0 lower {} > eval {} at t={t}",
                        out.lower()[[0, t]],
                        y[t]
                    );
                    assert!(
                        out.upper()[[0, t]] >= y[t] - 1e-4,
                        "ch0 upper {} < eval {} at t={t}",
                        out.upper()[[0, t]],
                        y[t]
                    );
                }
            }
        }
    }
}

#[test]
fn test_ibp_channels_independent() {
    // Verify that changing one channel's bounds doesn't affect the other channel's output.
    let layer = default_in1d(2);

    let lower1 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        .expect("valid lower1");
    let upper1 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
        .expect("valid upper1");
    let out1 = layer
        .propagate_ibp(&BoundedTensor::new(lower1, upper1).expect("valid bounded tensor"))
        .expect("IBP should succeed");

    let lower2 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-100.0, -50.0, 0.0, 0.0, 0.0, 0.0])
        .expect("valid lower2");
    let upper2 = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![100.0, 50.0, 200.0, 1.0, 1.0, 1.0])
        .expect("valid upper2");
    let out2 = layer
        .propagate_ibp(&BoundedTensor::new(lower2, upper2).expect("valid bounded tensor"))
        .expect("IBP should succeed");

    // Channel 1 bounds should be the same since channel 1 inputs are identical
    for t in 0..3 {
        assert!(
            (out1.lower()[[1, t]] - out2.lower()[[1, t]]).abs() < 1e-5,
            "Channel 1 lower bounds differ at t={t}: {} vs {}",
            out1.lower()[[1, t]],
            out2.lower()[[1, t]]
        );
        assert!(
            (out1.upper()[[1, t]] - out2.upper()[[1, t]]).abs() < 1e-5,
            "Channel 1 upper bounds differ at t={t}: {} vs {}",
            out1.upper()[[1, t]],
            out2.upper()[[1, t]]
        );
    }
}

#[test]
fn test_ibp_constant_input_gives_beta() {
    let layer = custom_in1d(&[2.0, 3.0], &[10.0, 20.0]);
    // All inputs in channel are identical → normalized to 0 → output = beta
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![5.0, 5.0, 5.0, 7.0, 7.0, 7.0])
        .expect("valid lower");
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![5.0, 5.0, 5.0, 7.0, 7.0, 7.0])
        .expect("valid upper");
    let input = BoundedTensor::new(lower, upper).expect("valid bounded tensor");
    let out = layer.propagate_ibp(&input).expect("IBP should succeed");

    // Output should be close to beta (constant input → variance≈0 → z≈0 → y≈beta).
    // Tolerance 1e-3: directed rounding on mean (next_down_f32/next_up_f32) creates
    // tiny centered intervals even for point inputs, which propagate through
    // variance/std to widen output bounds by ~ny * ULP_effect. This is expected
    // and correct — soundness requires this widening. Part of #3324.
    for t in 0..3 {
        assert!(
            (out.lower()[[0, t]] - 10.0).abs() < 1e-3,
            "ch0 t{t}: lower {} far from beta 10.0",
            out.lower()[[0, t]]
        );
        assert!(
            (out.upper()[[0, t]] - 10.0).abs() < 1e-3,
            "ch0 t{t}: upper {} far from beta 10.0",
            out.upper()[[0, t]]
        );
        assert!(
            (out.lower()[[1, t]] - 20.0).abs() < 1e-3,
            "ch1 t{t}: lower {} far from beta 20.0",
            out.lower()[[1, t]]
        );
        assert!(
            (out.upper()[[1, t]] - 20.0).abs() < 1e-3,
            "ch1 t{t}: upper {} far from beta 20.0",
            out.upper()[[1, t]]
        );
    }
}

#[test]
fn test_ibp_forward_mode_tighter() {
    // Forward mode is tighter than conservative when:
    // 1. Center-point variance is large (σ_min close to σ_center)
    // 2. Perturbation radius is small (second-order remainder < first-order)
    //
    // After #3159 fix (std_min via interval arithmetic instead of center-point std),
    // the second-order remainder is more conservative. With small T and moderate
    // perturbation (original T=4, r=0.3), second-order dominates and forward mode
    // can be wider. Use high-variance inputs with small perturbation where Jacobian-
    // based first-order term dominates.
    let conservative = default_in1d(2);
    let forward = default_in1d(2).with_forward_mode(true);

    // Channel 0: center at [0, 5, 10, 15], radius 0.05 → high variance, small perturbation
    // Channel 1: center at [-8, -3, 3, 8], radius 0.05 → same pattern
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![-0.05, 4.95, 9.95, 14.95, -8.05, -3.05, 2.95, 7.95],
    )
    .expect("valid lower");
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![0.05, 5.05, 10.05, 15.05, -7.95, -2.95, 3.05, 8.05],
    )
    .expect("valid upper");
    let input = BoundedTensor::new(lower, upper).expect("valid bounded tensor");

    let out_cons = conservative
        .propagate_ibp(&input)
        .expect("conservative IBP");
    let out_fwd = forward.propagate_ibp(&input).expect("forward IBP");

    // Forward mode bounds should be no wider than conservative (typically tighter)
    let cons_width: f32 = out_cons
        .upper()
        .iter()
        .zip(out_cons.lower().iter())
        .map(|(&u, &l)| u - l)
        .sum();
    let fwd_width: f32 = out_fwd
        .upper()
        .iter()
        .zip(out_fwd.lower().iter())
        .map(|(&u, &l)| u - l)
        .sum();

    assert!(
        fwd_width <= cons_width + 1e-3,
        "Forward mode width {fwd_width} > conservative width {cons_width}"
    );
}

// ── BoundPropagation trait tests ─────────────────────────────────────────

#[test]
fn test_propagate_linear_returns_error() {
    let layer = default_in1d(2);
    let bounds = crate::LinearBounds::new(
        ndarray::Array2::eye(6),
        Array1::zeros(6),
        ndarray::Array2::eye(6),
        Array1::zeros(6),
    )
    .expect("valid linear bounds");
    assert!(
        layer.propagate_linear(&bounds).is_err(),
        "propagate_linear should reject instance norm without pre-activation bounds"
    );
}

#[test]
fn test_requires_pre_activation_bounds() {
    let layer = default_in1d(1);
    assert!(
        layer.requires_pre_activation_bounds(),
        "instance norm requires pre-activation bounds"
    );
}

#[test]
fn test_ibp_rejects_1d_input() {
    let layer = default_in1d(3);
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).expect("valid lower");
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).expect("valid upper");
    let input = BoundedTensor::new(lower, upper).expect("valid bounded tensor");
    assert!(
        layer.propagate_ibp(&input).is_err(),
        "IBP should reject 1D input for instance norm"
    );
}

#[test]
fn test_ibp_shape_mismatch() {
    let layer = default_in1d(3);
    // Input has 2 channels but layer expects 3
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![0.0; 8]).expect("valid lower");
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0; 8]).expect("valid upper");
    let input = BoundedTensor::new(lower, upper).expect("valid bounded tensor");
    assert!(
        layer.propagate_ibp(&input).is_err(),
        "IBP should reject channel mismatch (2 vs 3)"
    );
}

// ── CROWN mode gating tests ─────────────────────────────────────────────

#[test]
fn test_crown_sound_mode_returns_error() {
    let layer = default_in1d(1).with_crown_mode(LayerNormCrownMode::Sound);
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![-1.0, 0.0, 1.0]).expect("valid lower"),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).expect("valid upper"),
    )
    .expect("valid bounded tensor");
    let bounds = crate::LinearBounds::new(
        ndarray::Array2::eye(3),
        Array1::zeros(3),
        ndarray::Array2::eye(3),
        Array1::zeros(3),
    )
    .expect("valid linear bounds");
    let result = layer.propagate_linear_with_bounds(&bounds, &pre);
    assert!(
        result.is_err(),
        "Sound mode should return error for instance norm CROWN"
    );
}

#[test]
fn test_crown_cut_mode_returns_identity() {
    let layer = default_in1d(1).with_crown_mode(LayerNormCrownMode::Cut);
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![-1.0, 0.0, 1.0]).expect("valid lower"),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).expect("valid upper"),
    )
    .expect("valid bounded tensor");
    let bounds = crate::LinearBounds::new(
        ndarray::Array2::eye(3),
        Array1::zeros(3),
        ndarray::Array2::eye(3),
        Array1::zeros(3),
    )
    .expect("valid linear bounds");
    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre)
        .expect("CROWN cut should succeed");
    // Should be identical to input bounds (identity relaxation)
    assert_eq!(result.lower_a(), bounds.lower_a());
}

// ── Forward-mode soundness regression tests (#3098) ──────────────────

/// Regression test for #3098: verifies that forward-mode IBP produces sound bounds.
///
/// The old formula `output_radius = sensitivity * (rv + max_radius / time_len)` was
/// unsound — this test generates random inputs within bounds and verifies that all
/// outputs fall within the computed output bounds.
#[test]
fn test_ibp_forward_mode_soundness_issue_3098_wide_perturbation() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let mut rng = StdRng::seed_from_u64(3098);

    let configs: Vec<(usize, usize)> = vec![
        (2, 8),  // 2 channels, 8 time steps
        (1, 16), // 1 channel, 16 time steps
        (3, 4),  // 3 channels, 4 time steps
    ];

    for (num_channels, time_len) in &configs {
        let nc = *num_channels;
        let tl = *time_len;
        let total = nc * tl;

        // Random ny and beta
        let ny: Vec<f32> = (0..nc).map(|_| rng.random_range(-3.0..3.0)).collect();
        let beta: Vec<f32> = (0..nc).map(|_| rng.random_range(-2.0..2.0)).collect();
        let layer = custom_in1d(&ny, &beta).with_forward_mode(true);

        // Random input bounds with wide perturbation
        let center: Vec<f32> = (0..total).map(|_| rng.random_range(-5.0..5.0)).collect();
        let half_width: Vec<f32> = (0..total).map(|_| rng.random_range(0.5..5.0)).collect();

        let lower_vals: Vec<f32> = center
            .iter()
            .zip(half_width.iter())
            .map(|(&c, &h)| c - h)
            .collect();
        let upper_vals: Vec<f32> = center
            .iter()
            .zip(half_width.iter())
            .map(|(&c, &h)| c + h)
            .collect();

        let lower =
            ArrayD::from_shape_vec(IxDyn(&[nc, tl]), lower_vals).expect("valid lower shape");
        let upper =
            ArrayD::from_shape_vec(IxDyn(&[nc, tl]), upper_vals).expect("valid upper shape");
        let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("valid bounded tensor");
        let out = layer
            .propagate_ibp(&input)
            .expect("forward-mode IBP should succeed");

        // Sample 500 random points and verify soundness
        let num_samples = 500;
        for sample in 0..num_samples {
            for c in 0..nc {
                let channel_input: Vec<f32> = (0..tl)
                    .map(|t| rng.random_range(lower[[c, t]]..=upper[[c, t]]))
                    .collect();
                let x = arr1(&channel_input);
                let y = layer.eval_channel(&x, c).expect("eval should succeed");

                for t in 0..tl {
                    assert!(
                        out.lower()[[c, t]] <= y[t] + 1e-3,
                        "Config C={nc} T={tl}: ch{c} t{t} sample {sample}: \
                         lower {} > eval {} (violation {:.6})",
                        out.lower()[[c, t]],
                        y[t],
                        out.lower()[[c, t]] - y[t]
                    );
                    assert!(
                        out.upper()[[c, t]] >= y[t] - 1e-3,
                        "Config C={nc} T={tl}: ch{c} t{t} sample {sample}: \
                         upper {} < eval {} (violation {:.6})",
                        out.upper()[[c, t]],
                        y[t],
                        y[t] - out.upper()[[c, t]]
                    );
                }
            }
        }
    }
}

/// Regression test for #3098: forward-mode soundness with custom ny/beta.
///
/// Uses large ny values and asymmetric perturbation to stress-test the
/// Jacobian-based propagation.
#[test]
fn test_ibp_forward_mode_soundness_custom_ny_3098() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let mut rng = StdRng::seed_from_u64(30981);

    let configs: Vec<(Vec<f32>, Vec<f32>, usize)> = vec![
        (vec![5.0, -3.0], vec![1.0, -2.0], 6), // large ny, 6 time steps
        (vec![0.1, 10.0], vec![-5.0, 0.0], 10), // mixed scales, 10 time steps
        (vec![-2.0, -2.0, 1.0], vec![0.0; 3], 5), // negative ny, 3 channels
    ];

    for (ny, beta, time_len) in &configs {
        let nc = ny.len();
        let tl = *time_len;
        let total = nc * tl;
        let layer = custom_in1d(ny, beta).with_forward_mode(true);

        // Random input bounds with asymmetric perturbation
        let center: Vec<f32> = (0..total).map(|_| rng.random_range(-3.0..3.0)).collect();
        let half_width: Vec<f32> = (0..total).map(|_| rng.random_range(0.2..3.0)).collect();

        let lower_vals: Vec<f32> = center
            .iter()
            .zip(half_width.iter())
            .map(|(&c, &h)| c - h)
            .collect();
        let upper_vals: Vec<f32> = center
            .iter()
            .zip(half_width.iter())
            .map(|(&c, &h)| c + h)
            .collect();

        let lower =
            ArrayD::from_shape_vec(IxDyn(&[nc, tl]), lower_vals).expect("valid lower shape");
        let upper =
            ArrayD::from_shape_vec(IxDyn(&[nc, tl]), upper_vals).expect("valid upper shape");
        let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("valid bounded tensor");
        let out = layer
            .propagate_ibp(&input)
            .expect("forward-mode IBP should succeed");

        // Sample 300 random points per config
        let num_samples = 300;
        for sample in 0..num_samples {
            for c in 0..nc {
                let channel_input: Vec<f32> = (0..tl)
                    .map(|t| rng.random_range(lower[[c, t]]..=upper[[c, t]]))
                    .collect();
                let x = arr1(&channel_input);
                let y = layer.eval_channel(&x, c).expect("eval should succeed");

                for t in 0..tl {
                    assert!(
                        out.lower()[[c, t]] <= y[t] + 1e-3,
                        "Custom ny {:?}: ch{c} t{t} sample {sample}: \
                         lower {} > eval {} (violation {:.6})",
                        ny,
                        out.lower()[[c, t]],
                        y[t],
                        out.lower()[[c, t]] - y[t]
                    );
                    assert!(
                        out.upper()[[c, t]] >= y[t] - 1e-3,
                        "Custom ny {:?}: ch{c} t{t} sample {sample}: \
                         upper {} < eval {} (violation {:.6})",
                        ny,
                        out.upper()[[c, t]],
                        y[t],
                        y[t] - out.upper()[[c, t]]
                    );
                }
            }
        }
    }
}

// ── f64 precision and directed rounding regression tests (#3324) ─────

/// Regression test for #3324: InstanceNorm conservative IBP with large T dimension.
///
/// With T=768 (typical transformer hidden dim), f32 accumulation of mean loses
/// ~log2(768)≈10 bits of precision. Values near the mean trigger catastrophic
/// cancellation in (xi - mean)^2. This test verifies that the f64 accumulation
/// and directed rounding hardening produces sound bounds in this regime.
#[test]
fn test_ibp_conservative_large_t_f64_precision_3324() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let mut rng = StdRng::seed_from_u64(3324);
    let time_len = 768;
    let num_channels = 2;

    let ny: Vec<f32> = (0..num_channels)
        .map(|_| rng.random_range(0.5..3.0))
        .collect();
    let beta: Vec<f32> = (0..num_channels)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let layer = custom_in1d(&ny, &beta);

    // Values clustered near a large mean → cancellation in (x - mean)^2 with f32.
    // The mean of 1000.0 + noise(0.01) is ~1000.0, so (xi - mean) ≈ small values
    // near the ULP boundary of 1000.0 in f32.
    let total = num_channels * time_len;
    let mut lower_vals = Vec::with_capacity(total);
    let mut upper_vals = Vec::with_capacity(total);
    for _c in 0..num_channels {
        for _t in 0..time_len {
            let center = 1000.0 + rng.random_range(-0.01..0.01_f32);
            let radius = rng.random_range(0.001..0.005_f32);
            lower_vals.push(center - radius);
            upper_vals.push(center + radius);
        }
    }

    let lower =
        ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), lower_vals).expect("valid lower");
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), upper_vals).expect("valid upper");
    let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("valid bounded tensor");
    let out = layer.propagate_ibp(&input).expect("IBP should succeed");

    // Verify bounds are finite and ordered
    for (&l, &u) in out.lower().iter().zip(out.upper().iter()) {
        assert!(l.is_finite(), "lower bound non-finite: {l}");
        assert!(u.is_finite(), "upper bound non-finite: {u}");
        assert!(l <= u, "lower {l} > upper {u}");
    }

    // Sample random points and verify soundness
    let mut rng2 = StdRng::seed_from_u64(33241);
    for _sample in 0..200 {
        for c in 0..num_channels {
            let channel_input: Vec<f32> = (0..time_len)
                .map(|t| rng2.random_range(lower[[c, t]]..=upper[[c, t]]))
                .collect();
            let x = arr1(&channel_input);
            let y = layer.eval_channel(&x, c).expect("eval should succeed");

            for t in 0..time_len {
                assert!(
                    out.lower()[[c, t]] <= y[t] + 1e-3,
                    "T={time_len} ch{c} t{t}: lower {} > eval {} (violation {:.6})",
                    out.lower()[[c, t]],
                    y[t],
                    out.lower()[[c, t]] - y[t]
                );
                assert!(
                    out.upper()[[c, t]] >= y[t] - 1e-3,
                    "T={time_len} ch{c} t{t}: upper {} < eval {} (violation {:.6})",
                    out.upper()[[c, t]],
                    y[t],
                    y[t] - out.upper()[[c, t]]
                );
            }
        }
    }
}

// ── NormLayer trait input validation tests (#3339) ───────────────────

/// Verify that eval() rejects input whose length is not divisible by num_channels.
/// Part of #3339.
#[test]
fn test_normlayer_eval_rejects_indivisible_input() {
    use super::super::trait_norm::NormLayer;
    let layer = default_in1d(3); // 3 channels
                                 // 7 elements is not divisible by 3
    let x = Array1::zeros(7);
    let result = NormLayer::eval(&layer, &x);
    assert!(
        result.is_err(),
        "eval should reject input of len 7 for 3 channels"
    );
}

/// Verify that jacobian() rejects input whose length is not divisible by num_channels.
/// Part of #3339.
#[test]
fn test_normlayer_jacobian_rejects_indivisible_input() {
    use super::super::trait_norm::NormLayer;
    let layer = default_in1d(3);
    let x = Array1::zeros(7);
    let result = NormLayer::jacobian(&layer, &x);
    assert!(
        result.is_err(),
        "jacobian should reject input of len 7 for 3 channels"
    );
}

/// Regression test for #3324: InstanceNorm forward-mode IBP with large T dimension.
///
/// Tests that the f64-hardened forward-mode path produces sound bounds with T=768.
/// The Jacobian computation (delta_st - 1/T - z_s*z_t/T) involves cancellation of
/// similarly-sized terms that requires f64 precision.
#[test]
fn test_ibp_forward_mode_large_t_f64_precision_3324() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let mut rng = StdRng::seed_from_u64(33242);
    let time_len = 768;
    let num_channels = 2;

    let ny: Vec<f32> = (0..num_channels)
        .map(|_| rng.random_range(0.5..3.0))
        .collect();
    let beta: Vec<f32> = (0..num_channels)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let layer = custom_in1d(&ny, &beta).with_forward_mode(true);

    // Use moderate perturbation so forward-mode is valid (second-order is small)
    let total = num_channels * time_len;
    let mut lower_vals = Vec::with_capacity(total);
    let mut upper_vals = Vec::with_capacity(total);
    for _c in 0..num_channels {
        for _t in 0..time_len {
            let center = rng.random_range(-5.0..5.0_f32);
            let radius = rng.random_range(0.001..0.01_f32);
            lower_vals.push(center - radius);
            upper_vals.push(center + radius);
        }
    }

    let lower =
        ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), lower_vals).expect("valid lower");
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), upper_vals).expect("valid upper");
    let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("valid bounded tensor");
    let out = layer
        .propagate_ibp(&input)
        .expect("forward-mode IBP should succeed");

    // Verify bounds are finite and ordered
    for (&l, &u) in out.lower().iter().zip(out.upper().iter()) {
        assert!(l.is_finite(), "lower bound non-finite: {l}");
        assert!(u.is_finite(), "upper bound non-finite: {u}");
        assert!(l <= u, "lower {l} > upper {u}");
    }

    // Sample random points and verify soundness
    let mut rng2 = StdRng::seed_from_u64(33243);
    for _sample in 0..200 {
        for c in 0..num_channels {
            let channel_input: Vec<f32> = (0..time_len)
                .map(|t| rng2.random_range(lower[[c, t]]..=upper[[c, t]]))
                .collect();
            let x = arr1(&channel_input);
            let y = layer.eval_channel(&x, c).expect("eval should succeed");

            for t in 0..time_len {
                assert!(
                    out.lower()[[c, t]] <= y[t] + 1e-3,
                    "T={time_len} fwd ch{c} t{t}: lower {} > eval {} (violation {:.6})",
                    out.lower()[[c, t]],
                    y[t],
                    out.lower()[[c, t]] - y[t]
                );
                assert!(
                    out.upper()[[c, t]] >= y[t] - 1e-3,
                    "T={time_len} fwd ch{c} t{t}: upper {} < eval {} (violation {:.6})",
                    out.upper()[[c, t]],
                    y[t],
                    y[t] - out.upper()[[c, t]]
                );
            }
        }
    }
}
