// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for GroupNorm IBP and CROWN propagation.
//! Part of #3205.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::types::GroupNormLayer;
use crate::layers::common::BoundPropagation;
use crate::layers::normalization::layer_norm::types::LayerNormCrownMode;
use crate::layers::normalization::trait_norm::NormLayer;
use crate::LinearBounds;

fn default_gn(c: usize, g: usize) -> GroupNormLayer {
    GroupNormLayer::new_default(c, g, 1e-5).unwrap()
}

// --- Construction tests ---

#[test]
fn test_construction_valid() {
    let layer = GroupNormLayer::new_default(8, 2, 1e-5).unwrap();
    assert_eq!(layer.num_channels(), 8);
    assert_eq!(layer.num_groups, 2);
    assert_eq!(layer.channels_per_group(), 4);
}

#[test]
fn test_construction_invalid_num_groups_zero() {
    let result = GroupNormLayer::new_default(8, 0, 1e-5);
    assert!(result.is_err(), "num_groups=0 should be rejected");
}

#[test]
fn test_construction_invalid_num_groups_not_divisible() {
    let result = GroupNormLayer::new_default(8, 3, 1e-5);
    assert!(
        result.is_err(),
        "num_groups=3 not divisible by channels=8 should be rejected"
    );
}

#[test]
fn test_construction_instance_norm_case() {
    // num_groups = C → InstanceNorm
    let layer = GroupNormLayer::new_default(4, 4, 1e-5).unwrap();
    assert_eq!(layer.channels_per_group(), 1);
}

#[test]
fn test_construction_layer_norm_case() {
    // num_groups = 1 → LayerNorm-like (all channels in one group)
    let layer = GroupNormLayer::new_default(4, 1, 1e-5).unwrap();
    assert_eq!(layer.channels_per_group(), 4);
}

// --- NormLayer trait tests ---

#[test]
fn test_eval_flat_constant_input() {
    let layer = default_gn(4, 2);
    // Input: 4 channels × 3 time = 12 elements, all constant
    let x = Array1::from_elem(12, 5.0_f32);
    let y = layer.eval(&x).unwrap();
    // All elements equal → (x - mean) = 0 → y = beta = 0
    for val in y.iter() {
        assert!(val.abs() < 1e-4, "Expected ~0, got {val}");
    }
}

#[test]
fn test_eval_flat_with_ny_beta() {
    let layer = GroupNormLayer::new(
        Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]),
        Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0]),
        2,
        1e-5,
    )
    .unwrap();

    // Constant input → y = beta
    let x = Array1::from_elem(8, 1.0_f32); // 4ch * 2t
    let y = layer.eval(&x).unwrap();
    // Channel 0: beta[0] = 10, channel 1: beta[1] = 20, etc.
    assert!(
        (y[0] - 10.0).abs() < 1e-3,
        "y[0] expected 10.0 (beta[0]), got {}",
        y[0]
    );
    assert!(
        (y[1] - 10.0).abs() < 1e-3,
        "y[1] expected 10.0 (beta[0]), got {}",
        y[1]
    );
    assert!(
        (y[2] - 20.0).abs() < 1e-3,
        "y[2] expected 20.0 (beta[1]), got {}",
        y[2]
    );
    assert!(
        (y[3] - 20.0).abs() < 1e-3,
        "y[3] expected 20.0 (beta[1]), got {}",
        y[3]
    );
    assert!(
        (y[4] - 30.0).abs() < 1e-3,
        "y[4] expected 30.0 (beta[2]), got {}",
        y[4]
    );
    assert!(
        (y[5] - 30.0).abs() < 1e-3,
        "y[5] expected 30.0 (beta[2]), got {}",
        y[5]
    );
    assert!(
        (y[6] - 40.0).abs() < 1e-3,
        "y[6] expected 40.0 (beta[3]), got {}",
        y[6]
    );
    assert!(
        (y[7] - 40.0).abs() < 1e-3,
        "y[7] expected 40.0 (beta[3]), got {}",
        y[7]
    );
}

#[test]
fn test_jacobian_flat_shape() {
    let layer = default_gn(4, 2);
    let x = Array1::from_vec((0..12).map(|i| i as f32 * 0.5).collect());
    let j = layer.jacobian(&x).unwrap();
    assert_eq!(j.shape(), &[12, 12]);
}

#[test]
fn test_jacobian_block_diagonal() {
    // With 2 groups of 2 channels, time_len=3:
    // Group 0: channels [0,1] → elements 0..6
    // Group 1: channels [2,3] → elements 6..12
    // Cross-group entries should be zero.
    let layer = default_gn(4, 2);
    let x = Array1::from_vec((0..12).map(|i| (i as f32 + 1.0) * 0.3).collect());
    let j = layer.jacobian(&x).unwrap();

    // Check cross-group entries are zero
    for i in 0..6 {
        for jj in 6..12 {
            assert!(
                j[[i, jj]].abs() < 1e-10,
                "Cross-group J[{i},{jj}] = {}",
                j[[i, jj]]
            );
            assert!(
                j[[jj, i]].abs() < 1e-10,
                "Cross-group J[{jj},{i}] = {}",
                j[[jj, i]]
            );
        }
    }
}

// --- IBP tests ---

#[ntest::timeout(10000)]
#[test]
fn test_ibp_conservative_basic() {
    let layer = default_gn(4, 2);
    // Input shape [4, 3] (4 channels, 3 time steps)
    let lower = ArrayD::from_elem(IxDyn(&[4, 3]), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[4, 3]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = layer.propagate_ibp(&input).unwrap();

    // Output should be finite and non-degenerate
    for val in output.lower().iter() {
        assert!(val.is_finite(), "Non-finite lower bound: {val}");
    }
    for val in output.upper().iter() {
        assert!(val.is_finite(), "Non-finite upper bound: {val}");
    }
    // Output shape should match input shape
    assert_eq!(output.shape(), &[4, 3]);

    // Verify containment: sampled concrete outputs must fall within IBP bounds.
    let c = 4;
    let t = 3;
    for s in 0..50_u32 {
        let point: Vec<f32> = (0..c * t)
            .map(|d| {
                let hash = ((s.wrapping_mul(2654435761) ^ (d as u32)).wrapping_mul(2654435761))
                    as f32
                    / u32::MAX as f32;
                // Map [0, 1] to [-1, 1]
                hash * 2.0 - 1.0
            })
            .collect();
        let x = Array1::from_vec(point);
        let y = layer.eval(&x).unwrap();

        for (i, &val) in y.iter().enumerate() {
            let ci = i / t;
            let ti = i % t;
            let lo = output.lower()[[ci, ti]];
            let hi = output.upper()[[ci, ti]];
            assert!(
                val >= lo - 1e-4 && val <= hi + 1e-4,
                "IBP containment: sample {s} elem {i}: val={val:.6} not in [{lo:.6}, {hi:.6}]"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_soundness_sampling() {
    // Soundness: IBP bounds must contain all concrete evaluations.
    let layer = default_gn(4, 2);
    let epsilon = 0.5_f32;
    let center = ArrayD::zeros(IxDyn(&[4, 3]));
    let lower = &center - epsilon;
    let upper = &center + epsilon;
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = layer.propagate_ibp(&input).unwrap();

    // Sample 50 random points and verify containment
    for s in 0..50 {
        let point: Vec<f32> = (0..12)
            .map(|d| {
                let hash = ((s * 7919 + d * 104729 + 31) % 10000) as f32 / 10000.0;
                (hash * 2.0 - 1.0) * epsilon
            })
            .collect();
        let x = Array1::from_vec(point);
        let y = layer.eval(&x).unwrap();

        for (i, &val) in y.iter().enumerate() {
            let c = i / 3;
            let t = i % 3;
            let lo = output.lower()[[c, t]];
            let hi = output.upper()[[c, t]];
            assert!(
                val >= lo - 1e-5 && val <= hi + 1e-5,
                "IBP soundness violation: sample {s} elem {i}: val={val:.6} not in [{lo:.6}, {hi:.6}]"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_forward_mode_soundness() {
    let layer = default_gn(4, 2).with_forward_mode(true);
    let epsilon = 0.3_f32;
    let center = ArrayD::from_shape_fn(IxDyn(&[4, 3]), |idx| {
        (idx[0] as f32 * 0.5 + idx[1] as f32 * 0.3) * 0.1
    });
    let lower = &center - epsilon;
    let upper = &center + epsilon;
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = layer.propagate_ibp(&input).unwrap();

    for s in 0..50 {
        let point: Vec<f32> = (0..12)
            .map(|d| {
                let c = d / 3;
                let t = d % 3;
                let center_val = (c as f32 * 0.5 + t as f32 * 0.3) * 0.1;
                let hash = ((s * 7919 + d * 104729 + 31) % 10000) as f32 / 10000.0;
                center_val + (hash * 2.0 - 1.0) * epsilon
            })
            .collect();
        let x = Array1::from_vec(point);
        let y = layer.eval(&x).unwrap();

        for (i, &val) in y.iter().enumerate() {
            let c = i / 3;
            let t = i % 3;
            let lo = output.lower()[[c, t]];
            let hi = output.upper()[[c, t]];
            assert!(
                val >= lo - 1e-5 && val <= hi + 1e-5,
                "Forward-mode IBP soundness violation: sample {s} elem {i}: \
                 val={val:.6} not in [{lo:.6}, {hi:.6}]"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_matches_instance_norm_when_groups_eq_channels() {
    // When num_groups = C, GroupNorm should produce the same bounds as InstanceNorm.
    use crate::layers::normalization::InstanceNorm1dLayer;

    let c = 3;
    let t = 4;
    let eps = 1e-5;

    let gn = GroupNormLayer::new_default(c, c, eps).unwrap();
    let inn = InstanceNorm1dLayer::new_default(c, eps).unwrap();

    let lower = ArrayD::from_shape_fn(IxDyn(&[c, t]), |idx| {
        -0.5 + (idx[0] * 7 + idx[1] * 13) as f32 * 0.01
    });
    let upper = &lower + 1.0;
    let input = BoundedTensor::new(lower, upper).unwrap();

    let gn_out = gn.propagate_ibp(&input).unwrap();
    let inn_out = inn.propagate_ibp(&input).unwrap();

    for c_idx in 0..c {
        for t_idx in 0..t {
            let gn_lo = gn_out.lower()[[c_idx, t_idx]];
            let inn_lo = inn_out.lower()[[c_idx, t_idx]];
            let gn_hi = gn_out.upper()[[c_idx, t_idx]];
            let inn_hi = inn_out.upper()[[c_idx, t_idx]];
            assert!(
                (gn_lo - inn_lo).abs() < 1e-4,
                "Lower mismatch at [{c_idx},{t_idx}]: GN={gn_lo:.6} IN={inn_lo:.6}"
            );
            assert!(
                (gn_hi - inn_hi).abs() < 1e-4,
                "Upper mismatch at [{c_idx},{t_idx}]: GN={gn_hi:.6} IN={inn_hi:.6}"
            );
        }
    }
}

// --- CROWN tests ---

#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_soundness() {
    // CROWN scalar backward must produce sound bounds.
    let layer = default_gn(4, 2);
    let epsilon = 0.3_f32;
    let c = 4;
    let t = 3;
    let total = c * t;

    let center = ArrayD::from_shape_fn(IxDyn(&[c, t]), |idx| {
        (idx[0] as f32 * 0.7 + idx[1] as f32 * 0.3) * 0.1
    });
    let lower = &center - epsilon;
    let upper = &center + epsilon;
    let pre_activation = BoundedTensor::new(lower, upper).unwrap();

    // Identity linear bounds (output = input)
    let identity_bounds = LinearBounds::identity(total);

    let result = layer
        .propagate_linear_with_bounds(&identity_bounds, &pre_activation)
        .unwrap();
    let crown_output = result.concretize(&pre_activation);

    // Sample and verify containment
    for s in 0..50 {
        let point: Vec<f32> = (0..total)
            .map(|d| {
                let ci = d / t;
                let ti = d % t;
                let center_val = (ci as f32 * 0.7 + ti as f32 * 0.3) * 0.1;
                let hash = ((s * 7919 + d * 104729 + 31) % 10000) as f32 / 10000.0;
                center_val + (hash * 2.0 - 1.0) * epsilon
            })
            .collect();
        let x = Array1::from_vec(point);
        let y = layer.eval(&x).unwrap();

        for (i, &val) in y.iter().enumerate() {
            let lo = crown_output.lower().iter().nth(i).copied().unwrap();
            let hi = crown_output.upper().iter().nth(i).copied().unwrap();
            assert!(
                val >= lo - 1e-4 && val <= hi + 1e-4,
                "CROWN soundness violation: sample {s} dim {i}: \
                 val={val:.6} not in [{lo:.6}, {hi:.6}]"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_batched_soundness() {
    // Batched CROWN backward must produce sound bounds.
    let layer = default_gn(4, 2);
    let epsilon = 0.3_f32;
    let c = 4;
    let t = 3;
    let total = c * t;

    // Pre-activation must be flat [total] to match the identity bounds [total, total].
    // The batched CROWN infrastructure checks last_dim(pre_activation) == in_dim.
    let center = ArrayD::from_shape_fn(IxDyn(&[total]), |idx| {
        let ci = idx[0] / t;
        let ti = idx[0] % t;
        (ci as f32 * 0.7 + ti as f32 * 0.3) * 0.1
    });
    let lower = &center - epsilon;
    let upper = &center + epsilon;
    let pre_activation = BoundedTensor::new(lower, upper).unwrap();

    let identity_batched = crate::BatchedLinearBounds::identity(&[total]).unwrap();

    let result = layer
        .propagate_linear_batched_with_bounds(&identity_batched, &pre_activation)
        .unwrap();
    let crown_output = result.concretize_sound(&pre_activation).unwrap();

    for s in 0..50 {
        let point: Vec<f32> = (0..total)
            .map(|d| {
                let ci = d / t;
                let ti = d % t;
                let center_val = (ci as f32 * 0.7 + ti as f32 * 0.3) * 0.1;
                let hash = ((s * 7919 + d * 104729 + 31) % 10000) as f32 / 10000.0;
                center_val + (hash * 2.0 - 1.0) * epsilon
            })
            .collect();
        let x = Array1::from_vec(point);
        let y = layer.eval(&x).unwrap();

        for (i, &val) in y.iter().enumerate() {
            let lo = crown_output.lower()[[i]];
            let hi = crown_output.upper()[[i]];
            assert!(
                val >= lo - 1e-4 && val <= hi + 1e-4,
                "Batched CROWN soundness violation: sample {s} dim {i}: \
                 val={val:.6} not in [{lo:.6}, {hi:.6}]"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_mode_sound_returns_error() {
    let layer = default_gn(4, 2).with_crown_mode(LayerNormCrownMode::Sound);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4, 3]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[4, 3]), 1.0_f32),
    )
    .unwrap();
    let identity = LinearBounds::identity(12);
    let result = layer.propagate_linear_with_bounds(&identity, &pre);
    assert!(result.is_err(), "Sound mode should return error");
}

/// Regression test for #3257: cpg*time_len exceeds f32 exact integer range
/// even when time_len alone does not.
/// Uses validate_ibp_input directly to avoid allocating huge computation buffers.
#[test]
fn test_validate_rejects_group_size_exceeding_f32_range() {
    // C=16384, num_groups=1 → cpg=16384. time_len=1025.
    // time_len=1025 < 2^24 → passes the time_len check.
    // cpg * time_len = 16384 * 1025 = 16,793,600 > 2^24 = 16,777,216 → should fail.
    let c = 16384;
    let g = 1;
    let layer = GroupNormLayer::new_default(c, g, 1e-5).unwrap();
    let time_len = 1025;
    let shape = IxDyn(&[c, time_len]);
    let lower = ArrayD::from_elem(shape.clone(), 0.0_f32);
    let upper = ArrayD::from_elem(shape, 0.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = layer.validate_ibp_input(&input);
    assert!(
        result.is_err(),
        "Should reject group_size={} > 2^24 but got Ok",
        c * time_len,
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("f32 exact integer range"),
        "Error should mention f32 range, got: {err_msg}"
    );
}

/// Verify that group_size exactly at 2^24 passes validation.
#[test]
fn test_validate_accepts_group_size_at_f32_boundary() {
    // C=16384, num_groups=1 → cpg=16384. time_len=1024.
    // cpg * time_len = 16384 * 1024 = 16,777,216 = exactly 2^24.
    // This is NOT > 2^24, so validation should pass.
    let c = 16384;
    let g = 1;
    let layer = GroupNormLayer::new_default(c, g, 1e-5).unwrap();
    let time_len = 1024;
    let shape = IxDyn(&[c, time_len]);
    let lower = ArrayD::from_elem(shape.clone(), 0.0_f32);
    let upper = ArrayD::from_elem(shape, 0.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = layer.validate_ibp_input(&input);
    assert!(
        result.is_ok(),
        "Should accept group_size={} == 2^24, got: {}",
        c * time_len,
        result.unwrap_err(),
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_mode_cut_returns_identity() {
    let layer = default_gn(4, 2).with_crown_mode(LayerNormCrownMode::Cut);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4, 3]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[4, 3]), 1.0_f32),
    )
    .unwrap();
    let identity = LinearBounds::identity(12);
    let result = layer.propagate_linear_with_bounds(&identity, &pre).unwrap();
    // Cut mode returns identity (unchanged bounds)
    assert_eq!(result.num_inputs(), 12);
}

// ── CROWN scalar NaN/Inf pre-activation guard tests (#3259) ────────────────
// GroupNorm delegates to crown_common::sampling_crown_scalar which has the
// non-finite guard. These tests verify GroupNorm reaches that guard correctly.

/// NaN in pre-activation lower bound triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_nan_pre_activation_lower_returns_constant_bounds() {
    // 4 channels, 2 groups, time_len=1 → 4 neurons total
    let layer = default_gn(4, 2).with_crown_mode(LayerNormCrownMode::Sampling);
    let bounds = LinearBounds::identity(4);
    let pre_act = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[4, 1]), vec![f32::NAN, 1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4, 1]), vec![2.0, 3.0, 4.0, 5.0]).unwrap(),
    )
    .unwrap();

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("NaN pre-activation should return constant bounds, not error");

    assert!(
        result.lower_a().iter().all(|&v| v == 0.0),
        "lower_a should be all zeros"
    );
    assert!(
        result.upper_a().iter().all(|&v| v == 0.0),
        "upper_a should be all zeros"
    );
    assert!(
        result.lower_b().iter().all(|&v| v == f32::NEG_INFINITY),
        "lower_b should be -inf"
    );
    assert!(
        result.upper_b().iter().all(|&v| v == f32::INFINITY),
        "upper_b should be +inf"
    );
}

/// NaN in pre-activation upper bound triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_nan_pre_activation_upper_returns_constant_bounds() {
    let layer = default_gn(4, 2).with_crown_mode(LayerNormCrownMode::Sampling);
    let bounds = LinearBounds::identity(4);
    let pre_act = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[4, 1]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4, 1]), vec![2.0, f32::NAN, 4.0, 5.0]).unwrap(),
    )
    .unwrap();

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("NaN pre-activation should return constant bounds, not error");

    assert!(
        result.lower_a().iter().all(|&v| v == 0.0),
        "lower_a should be all zeros"
    );
    assert!(
        result.upper_a().iter().all(|&v| v == 0.0),
        "upper_a should be all zeros"
    );
    assert!(
        result.lower_b().iter().all(|&v| v == f32::NEG_INFINITY),
        "lower_b should be -inf"
    );
    assert!(
        result.upper_b().iter().all(|&v| v == f32::INFINITY),
        "upper_b should be +inf"
    );
}

/// Inf in pre-activation bounds triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_inf_pre_activation_returns_constant_bounds() {
    let layer = default_gn(4, 2).with_crown_mode(LayerNormCrownMode::Sampling);
    let bounds = LinearBounds::identity(4);
    let pre_act = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[4, 1]), vec![f32::NEG_INFINITY, 1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4, 1]), vec![f32::INFINITY, 3.0, 4.0, 5.0]).unwrap(),
    )
    .unwrap();

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("Inf pre-activation should return constant bounds, not error");

    assert!(
        result.lower_a().iter().all(|&v| v == 0.0),
        "lower_a should be all zeros"
    );
    assert!(
        result.upper_a().iter().all(|&v| v == 0.0),
        "upper_a should be all zeros"
    );
    assert!(
        result.lower_b().iter().all(|&v| v == f32::NEG_INFINITY),
        "lower_b should be -inf"
    );
    assert!(
        result.upper_b().iter().all(|&v| v == f32::INFINITY),
        "upper_b should be +inf"
    );
}
