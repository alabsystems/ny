// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive soundness tests for Softmax bound propagation.
//!
//! Covers: IBP numerical edge cases (overflow, underflow, NaN, near-uniform,
//! extreme outlier, mixed sign), CROWN backward via `propagate_crown_backward`,
//! batched soundness-mode dispatch, and bounds.rs non-finite sanitization.
//!
//! Part of #1950.

use super::prelude::*;
use ny_core::VerificationSoundnessMode;

/// Verified bound-check tolerance for softmax CROWN vertex assertions.
const SOFTMAX_CROWN_VERTEX_TOLERANCE: f32 = 5e-4;
/// Minimum observed margin floor (si-lb and ub-si) on the 3-element audit case.
const SOFTMAX_CROWN_MARGIN_FLOOR: f32 = 1e-3;

// ============================================================================
// Helpers
// ============================================================================

/// Reference softmax in f64 for high-precision comparison.
fn reference_softmax(x: &[f32]) -> Vec<f32> {
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let max = x64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x64.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

fn make_bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let n = lower.len();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

fn make_bt_unchecked(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let n = lower.len();
    BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

/// Assert IBP bounds contain softmax(x) for all vertex corners + midpoint.
fn assert_ibp_contains(layer: &SoftmaxLayer, input: &BoundedTensor) {
    use crate::layers::common::BoundPropagation;
    let output = layer.propagate_ibp(input).unwrap();
    let n = input.len();
    let lower: Vec<f32> = input.lower().iter().copied().collect();
    let upper: Vec<f32> = input.upper().iter().copied().collect();

    // Vertex corners (up to 256)
    for mask in 0..(1u32 << n).min(256) {
        let x: Vec<f32> = (0..n)
            .map(|i| {
                if (mask >> i) & 1 == 0 {
                    lower[i]
                } else {
                    upper[i]
                }
            })
            .collect();
        let s = reference_softmax(&x);
        for (i, &si) in s.iter().enumerate() {
            assert!(
                output.lower()[[i]] <= si + 1e-4,
                "IBP lower[{}] = {} > softmax = {} for x={:?}",
                i,
                output.lower()[[i]],
                si,
                x
            );
            assert!(
                output.upper()[[i]] >= si - 1e-4,
                "IBP upper[{}] = {} < softmax = {} for x={:?}",
                i,
                output.upper()[[i]],
                si,
                x
            );
        }
    }

    // Midpoint
    let mid: Vec<f32> = (0..n).map(|i| f32::midpoint(lower[i], upper[i])).collect();
    let s_mid = reference_softmax(&mid);
    for (i, &si) in s_mid.iter().enumerate() {
        assert!(
            output.lower()[[i]] <= si + 1e-4,
            "IBP lower[{}] = {} > softmax = {} at midpoint",
            i,
            output.lower()[[i]],
            si,
        );
        assert!(
            output.upper()[[i]] >= si - 1e-4,
            "IBP upper[{}] = {} < softmax = {} at midpoint",
            i,
            output.upper()[[i]],
            si,
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_sound_mode_constant_bounds_contain_sampled_outputs() {
    let softmax = SoftmaxLayer::new(-1).with_sound_mode(true);

    // Force sound-mode fallback path to constant bounds by providing non-finite
    // pre-activation limits (the fallback emits [0,1] constants).
    let non_finite_input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY, f32::INFINITY, f32::INFINITY]).into_dyn(),
    )
    .unwrap();
    let linear_bounds = LinearBounds::identity(3);
    let result = softmax
        .propagate_linear_with_bounds(&linear_bounds, &non_finite_input, softmax.soundness_mode())
        .unwrap();

    assert!(
        result.lower_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero lower_a in constant fallback bounds"
    );
    assert!(
        result.upper_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero upper_a in constant fallback bounds"
    );

    // Concretize over a finite sampling domain and check concrete containment.
    let sample_domain = BoundedTensor::new(
        arr1(&[-3.0_f32, -1.0, 0.5]).into_dyn(),
        arr1(&[3.0_f32, 2.0, 2.5]).into_dyn(),
    )
    .unwrap();
    let concretized = result.concretize(&sample_domain);

    let samples_per_dim = 4usize; // 5^3 = 125 concrete points
    let sampled_points = (samples_per_dim + 1) * (samples_per_dim + 1) * (samples_per_dim + 1);
    assert!(
        sampled_points >= 100,
        "softmax soundness checks require at least 100 sampled points"
    );

    let lower = sample_domain.lower();
    let upper = sample_domain.upper();
    for i in 0..=samples_per_dim {
        for j in 0..=samples_per_dim {
            for k in 0..=samples_per_dim {
                let x0 =
                    lower[[0]] + (upper[[0]] - lower[[0]]) * (i as f32) / (samples_per_dim as f32);
                let x1 =
                    lower[[1]] + (upper[[1]] - lower[[1]]) * (j as f32) / (samples_per_dim as f32);
                let x2 =
                    lower[[2]] + (upper[[2]] - lower[[2]]) * (k as f32) / (samples_per_dim as f32);
                let point = arr1(&[x0, x1, x2]);
                let softmax_val = softmax.eval(&point);

                for dim in 0..3 {
                    assert!(
                        softmax_val[dim] >= concretized.lower()[[dim]] - 1e-5,
                        "lower violated at dim {}: {} < {} for point {:?}",
                        dim,
                        softmax_val[dim],
                        concretized.lower()[[dim]],
                        point
                    );
                    assert!(
                        softmax_val[dim] <= concretized.upper()[[dim]] + 1e-5,
                        "upper violated at dim {}: {} > {} for point {:?}",
                        dim,
                        softmax_val[dim],
                        concretized.upper()[[dim]],
                        point
                    );
                }
            }
        }
    }
}

// ============================================================================
// 1. Batched soundness-mode dispatch (existing test, preserved)
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_softmax_batched_explicit_soundness_matches_layer_sound_mode() {
    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let softmax_forced_sound = SoftmaxLayer::new(-1).with_sound_mode(true);

    let lower_vals = vec![-2.0_f32, 0.25, 1.5];
    let upper_vals = vec![-0.5_f32, 2.0, 3.0];

    let pre_bounds_batched = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), lower_vals).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), upper_vals).unwrap(),
    )
    .unwrap();

    let explicit_sound = softmax
        .propagate_linear_batched_with_bounds(
            &BatchedLinearBounds::identity(&[1, 3]).unwrap(),
            &pre_bounds_batched,
            VerificationSoundnessMode::Sound,
        )
        .unwrap()
        .concretize(&pre_bounds_batched)
        .unwrap();

    let forced_sound = softmax_forced_sound
        .propagate_linear_batched_with_bounds(
            &BatchedLinearBounds::identity(&[1, 3]).unwrap(),
            &pre_bounds_batched,
            VerificationSoundnessMode::Heuristic,
        )
        .unwrap()
        .concretize(&pre_bounds_batched)
        .unwrap();

    let tol = 1e-5_f32;
    for i in 0..3 {
        assert!(
            (explicit_sound.lower()[[0, i]] - forced_sound.lower()[[0, i]]).abs() <= tol,
            "lower mismatch at index {}: explicit_sound={} forced_sound={}",
            i,
            explicit_sound.lower()[[0, i]],
            forced_sound.lower()[[0, i]]
        );
        assert!(
            (explicit_sound.upper()[[0, i]] - forced_sound.upper()[[0, i]]).abs() <= tol,
            "upper mismatch at index {}: explicit_sound={} forced_sound={}",
            i,
            explicit_sound.upper()[[0, i]],
            forced_sound.upper()[[0, i]]
        );
    }

    let explicit_heuristic = softmax
        .propagate_linear_batched_with_bounds(
            &BatchedLinearBounds::identity(&[1, 3]).unwrap(),
            &pre_bounds_batched,
            VerificationSoundnessMode::Heuristic,
        )
        .unwrap()
        .concretize(&pre_bounds_batched)
        .unwrap();
    let mode_diff = (0..3).fold(0.0_f32, |acc, i| {
        acc + (explicit_sound.lower()[[0, i]] - explicit_heuristic.lower()[[0, i]]).abs()
            + (explicit_sound.upper()[[0, i]] - explicit_heuristic.upper()[[0, i]]).abs()
    });
    assert!(
        mode_diff > 1e-6,
        "soundness parameter had no effect in batched softmax path"
    );
}

// ============================================================================
// 2. Numerical edge cases for IBP — issue #1950 checklist
// ============================================================================

/// Near-uniform inputs: all elements close, bounds should be tight around 1/n.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_near_uniform_tight() {
    let layer = SoftmaxLayer::new(-1);
    let input = make_bt(&[0.99, 1.0, 1.01], &[1.01, 1.02, 1.03]);
    let output = layer.propagate_ibp(&input).unwrap();

    // For near-uniform inputs, all softmax outputs should be close to 1/3
    let expected = 1.0 / 3.0;
    for i in 0..3 {
        let width = output.upper()[[i]] - output.lower()[[i]];
        assert!(
            width < 0.1,
            "Near-uniform bounds should be tight: width[{}] = {} (lower={}, upper={})",
            i,
            width,
            output.lower()[[i]],
            output.upper()[[i]]
        );
        // Center of bounds should be near 1/3
        let center = f32::midpoint(output.lower()[[i]], output.upper()[[i]]);
        assert!(
            (center - expected).abs() < 0.1,
            "Near-uniform center[{}] = {} far from 1/3",
            i,
            center
        );
    }
    assert_ibp_contains(&layer, &input);
}

/// Extreme outlier: one element >> others → dominant element upper near 1, others' lower near 0.
/// Note: IBP bounds are conservative — the lower bound of the dominant element may be loose
/// because the IBP formula considers worst-case denominator combinations.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_extreme_outlier() {
    let layer = SoftmaxLayer::new(-1);
    // First element dominates (50-80), others are small (-1 to 1)
    let input = make_bt(&[50.0, -1.0, -1.0], &[80.0, 1.0, 1.0]);
    let output = layer.propagate_ibp(&input).unwrap();

    // Dominant element upper should be near 1.0
    assert!(
        output.upper()[[0]] > 0.9,
        "Dominant element upper should be near 1: got {}",
        output.upper()[[0]]
    );
    // Other elements lower should be near 0.0
    for i in 1..3 {
        assert!(
            output.lower()[[i]] < 0.01,
            "Non-dominant element lower[{}] should be near 0: got {}",
            i,
            output.lower()[[i]]
        );
    }
    // All bounds must be in [0, 1]
    for i in 0..3 {
        assert!(output.lower()[[i]] >= 0.0);
        assert!(output.upper()[[i]] <= 1.0);
    }
    assert_ibp_contains(&layer, &input);
}

/// Large negative inputs: underflow to zero, check for NaN propagation.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_large_negative_underflow() {
    let layer = SoftmaxLayer::new(-1);
    // exp(-100) ≈ 3.7e-44, very close to underflow
    let input = make_bt(&[-100.0, -100.0, -100.0], &[-80.0, -80.0, -80.0]);
    let output = layer.propagate_ibp(&input).unwrap();

    // All outputs should be finite and in [0, 1]
    for i in 0..3 {
        assert!(
            output.lower()[[i]].is_finite(),
            "lower[{}] should be finite: got {}",
            i,
            output.lower()[[i]]
        );
        assert!(
            output.upper()[[i]].is_finite(),
            "upper[{}] should be finite: got {}",
            i,
            output.upper()[[i]]
        );
        assert!(output.lower()[[i]] >= 0.0);
        assert!(output.upper()[[i]] <= 1.0);
    }
    // Near-uniform when all inputs are similar magnitude
    assert_ibp_contains(&layer, &input);
}

/// Mixed sign with large magnitude spread — the hardest case for bound tightness.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_mixed_sign_large_spread() {
    let layer = SoftmaxLayer::new(-1);
    let input = make_bt(&[-20.0, -5.0, 0.0, 5.0], &[0.0, 5.0, 10.0, 20.0]);
    let output = layer.propagate_ibp(&input).unwrap();

    // Must be in [0, 1] and sound
    for i in 0..4 {
        assert!(output.lower()[[i]] >= 0.0, "lower[{}] < 0", i);
        assert!(output.upper()[[i]] <= 1.0, "upper[{}] > 1", i);
        assert!(
            output.lower()[[i]] <= output.upper()[[i]] + 1e-6,
            "lower[{}] > upper[{}]",
            i,
            i
        );
    }
    assert_ibp_contains(&layer, &input);
}

/// Inputs near f32::MAX exponent range — overflow in exp().
/// IBP should handle this gracefully (fallback to [0, 1]).
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_near_overflow() {
    use crate::layers::common::BoundPropagation;
    let layer = SoftmaxLayer::new(-1);
    // exp(88) ≈ 1.65e38 (near f32::MAX ≈ 3.4e38)
    let input = make_bt(&[85.0, 86.0, 87.0], &[88.0, 89.0, 90.0]);
    let output = layer.propagate_ibp(&input).unwrap();

    // Should produce valid [0, 1] bounds even near overflow
    for i in 0..3 {
        assert!(
            output.lower()[[i]].is_finite(),
            "near-overflow lower[{}] not finite: {}",
            i,
            output.lower()[[i]]
        );
        assert!(
            output.upper()[[i]].is_finite(),
            "near-overflow upper[{}] not finite: {}",
            i,
            output.upper()[[i]]
        );
        assert!(output.lower()[[i]] >= 0.0);
        assert!(output.upper()[[i]] <= 1.0);
    }
}

/// NaN input: IBP should fall back to [0, 1] for non-finite inputs.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_nan_input_falls_back() {
    use crate::layers::common::BoundPropagation;
    let layer = SoftmaxLayer::new(-1);
    let input = make_bt_unchecked(&[f32::NAN, 0.0, 1.0], &[f32::NAN, 1.0, 2.0]);
    let output = layer.propagate_ibp(&input).unwrap();

    // Non-finite input → fallback to [0, 1]
    for i in 0..3 {
        assert!(
            output.lower()[[i]] >= 0.0,
            "NaN fallback lower[{}] = {} < 0",
            i,
            output.lower()[[i]]
        );
        assert!(
            output.upper()[[i]] <= 1.0,
            "NaN fallback upper[{}] = {} > 1",
            i,
            output.upper()[[i]]
        );
    }
}

/// Inf input: IBP should fall back to [0, 1] for Inf inputs.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_inf_input_falls_back() {
    use crate::layers::common::BoundPropagation;
    let layer = SoftmaxLayer::new(-1);
    let input = make_bt_unchecked(&[f32::NEG_INFINITY, 0.0, 1.0], &[f32::INFINITY, 1.0, 2.0]);
    let output = layer.propagate_ibp(&input).unwrap();

    for i in 0..3 {
        assert!(output.lower()[[i]] >= 0.0);
        assert!(output.upper()[[i]] <= 1.0);
    }
}

/// Two elements: simplest case, exhaustive vertex enumeration (4 vertices).
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_two_elements_exhaustive() {
    let layer = SoftmaxLayer::new(-1);
    let input = make_bt(&[-2.0, 3.0], &[1.0, 5.0]);
    assert_ibp_contains(&layer, &input);
}

/// Single element: softmax([x]) = [1.0] always.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_single_element() {
    use crate::layers::common::BoundPropagation;
    let layer = SoftmaxLayer::new(-1);
    let input = make_bt(&[-5.0], &[5.0]);
    let output = layer.propagate_ibp(&input).unwrap();

    // softmax of a single element is always 1.0
    assert!(
        (output.lower()[[0]] - 1.0).abs() < 1e-3,
        "Single-element softmax lower should be ~1.0: got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 1.0).abs() < 1e-3,
        "Single-element softmax upper should be ~1.0: got {}",
        output.upper()[[0]]
    );
}

/// Large dimension: softmax over 8 elements with mixed ranges.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_8_elements() {
    let layer = SoftmaxLayer::new(-1);
    let lower = vec![-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0];
    let upper = vec![-1.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0];
    let input = make_bt(&lower, &upper);
    assert_ibp_contains(&layer, &input);
}

// ============================================================================
// 3. CROWN backward via propagate_crown_backward (ibp.rs untested path)
// ============================================================================

/// Test that propagate_crown_backward delegates to propagate_linear_with_bounds
/// and produces sound CROWN bounds for a simple 3-element softmax.
/// Sound mode uses LSE-based affine bounds (non-zero slopes), not constant bounds.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_crown_backward_sound_3elem() {
    use crate::layers::common::BoundPropagation;
    let layer = SoftmaxLayer::new(-1); // sound by default

    let pre_activation = make_bt(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);
    let identity_bounds = LinearBounds::identity(3);

    let result = layer
        .propagate_crown_backward(&identity_bounds, Some(&pre_activation))
        .unwrap();

    // Sound CROWN uses LSE-based affine bounds: slopes should be finite
    for &v in result.lower_a.iter() {
        assert!(v.is_finite(), "Sound CROWN lower_a should be finite: {}", v);
    }
    for &v in result.upper_a.iter() {
        assert!(v.is_finite(), "Sound CROWN upper_a should be finite: {}", v);
    }
    for &v in result.lower_b.iter() {
        assert!(v.is_finite(), "Sound CROWN lower_b should be finite: {}", v);
    }
    for &v in result.upper_b.iter() {
        assert!(v.is_finite(), "Sound CROWN upper_b should be finite: {}", v);
    }

    // Verify soundness: CROWN linear bounds should contain softmax at vertices.
    // With identity incoming bounds: lower_bound = lower_a @ x + lower_b
    let lower = [0.0_f32, 1.0, 2.0];
    let upper = [1.0_f32, 2.0, 3.0];
    for mask in 0..8u32 {
        let x: Vec<f32> = (0..3)
            .map(|i| {
                if (mask >> i) & 1 == 0 {
                    lower[i]
                } else {
                    upper[i]
                }
            })
            .collect();
        let s = reference_softmax(&x);
        for (i, &si) in s.iter().enumerate() {
            // lower_bound_i = sum_j(lower_a[i,j] * x[j]) + lower_b[i]
            let lb: f32 =
                (0..3).map(|j| result.lower_a[[i, j]] * x[j]).sum::<f32>() + result.lower_b[i];
            let ub: f32 =
                (0..3).map(|j| result.upper_a[[i, j]] * x[j]).sum::<f32>() + result.upper_b[i];
            assert!(
                lb <= si + SOFTMAX_CROWN_VERTEX_TOLERANCE,
                "CROWN lower[{}] = {} > softmax = {} for x={:?}",
                i,
                lb,
                si,
                x
            );
            assert!(
                ub >= si - SOFTMAX_CROWN_VERTEX_TOLERANCE,
                "CROWN upper[{}] = {} < softmax = {} for x={:?}",
                i,
                ub,
                si,
                x
            );
        }
    }
}

/// Audit the minimum soundness margins on the 3-element CROWN backward case.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_crown_backward_margin_audit() {
    use crate::layers::common::BoundPropagation;
    let layer = SoftmaxLayer::new(-1);

    let pre_activation = make_bt(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);
    let identity_bounds = LinearBounds::identity(3);

    let result = layer
        .propagate_crown_backward(&identity_bounds, Some(&pre_activation))
        .unwrap();

    let lower = [0.0_f32, 1.0, 2.0];
    let upper = [1.0_f32, 2.0, 3.0];

    let mut min_lower_margin = f32::INFINITY;
    let mut min_upper_margin = f32::INFINITY;
    for mask in 0..8u32 {
        let x: Vec<f32> = (0..3)
            .map(|i| {
                if (mask >> i) & 1 == 0 {
                    lower[i]
                } else {
                    upper[i]
                }
            })
            .collect();
        let s = reference_softmax(&x);
        for (i, &si) in s.iter().enumerate() {
            let lb: f32 =
                (0..3).map(|j| result.lower_a[[i, j]] * x[j]).sum::<f32>() + result.lower_b[i];
            let ub: f32 =
                (0..3).map(|j| result.upper_a[[i, j]] * x[j]).sum::<f32>() + result.upper_b[i];
            min_lower_margin = min_lower_margin.min(si - lb);
            min_upper_margin = min_upper_margin.min(ub - si);
        }
    }

    assert!(
        min_lower_margin > SOFTMAX_CROWN_MARGIN_FLOOR,
        "Lower margin floor violated: min_lower_margin={} <= {}",
        min_lower_margin,
        SOFTMAX_CROWN_MARGIN_FLOOR
    );
    assert!(
        min_upper_margin > SOFTMAX_CROWN_MARGIN_FLOOR,
        "Upper margin floor violated: min_upper_margin={} <= {}",
        min_upper_margin,
        SOFTMAX_CROWN_MARGIN_FLOOR
    );
}

/// Test propagate_crown_backward returns error when no pre-activation bounds.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_crown_backward_requires_pre_activation() {
    use crate::layers::common::BoundPropagation;
    let layer = SoftmaxLayer::new(-1);
    let identity_bounds = LinearBounds::identity(3);

    let result = layer.propagate_crown_backward(&identity_bounds, None);
    assert!(
        result.is_err(),
        "propagate_crown_backward should require pre-activation bounds"
    );
}

/// CROWN backward with heuristic mode produces non-zero slopes (Jacobian-based).
#[ntest::timeout(10000)]
#[test]
fn test_softmax_crown_backward_heuristic_has_slopes() {
    let layer = SoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let pre_activation = make_bt(&[-1.0, 0.0, 1.0], &[0.0, 1.0, 2.0]);
    let identity_bounds = LinearBounds::identity(3);

    let result = layer
        .propagate_linear_with_bounds(
            &identity_bounds,
            &pre_activation,
            VerificationSoundnessMode::Heuristic,
        )
        .unwrap();

    // Heuristic mode should produce non-zero slopes (Jacobian linearization)
    let slope_sum: f32 = result.lower_a.iter().map(|v| v.abs()).sum::<f32>()
        + result.upper_a.iter().map(|v| v.abs()).sum::<f32>();
    assert!(
        slope_sum > 1e-6,
        "Heuristic CROWN should have non-zero slopes: total = {}",
        slope_sum
    );

    // Verify slopes are finite
    for &v in result.lower_a.iter() {
        assert!(v.is_finite(), "Heuristic lower_a has non-finite: {}", v);
    }
    for &v in result.upper_a.iter() {
        assert!(v.is_finite(), "Heuristic upper_a has non-finite: {}", v);
    }
}

// ============================================================================
// 4. CROWN backward with large A-matrix: exercises bounds.rs sanitization path
// ============================================================================

/// Verify that CROWN backward handles large incoming A-matrices gracefully.
/// Sound CROWN produces LSE-based affine bounds; with large A the output
/// coefficients should be scaled but remain finite.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_crown_backward_large_a_matrix_stable() {
    let layer = SoftmaxLayer::new(-1); // sound mode → LSE-based affine bounds

    let pre_activation = make_bt(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);

    // Large A-matrix: tests numerical stability of the CROWN composition
    let large_bounds = LinearBounds::new(
        Array2::eye(3) * 1000.0,
        Array1::zeros(3),
        Array2::eye(3) * 1000.0,
        Array1::zeros(3),
    )
    .unwrap();

    let result = layer
        .propagate_linear_with_bounds(
            &large_bounds,
            &pre_activation,
            VerificationSoundnessMode::Sound,
        )
        .unwrap();

    // All values should be finite (no overflow from large A-matrix)
    for &v in result.lower_a.iter() {
        assert!(v.is_finite(), "large A-matrix lower_a not finite: {}", v);
    }
    for &v in result.upper_a.iter() {
        assert!(v.is_finite(), "large A-matrix upper_a not finite: {}", v);
    }
    for i in 0..3 {
        assert!(
            result.lower_b[i].is_finite(),
            "large A-matrix lower_b[{}] not finite: {}",
            i,
            result.lower_b[i]
        );
        assert!(
            result.upper_b[i].is_finite(),
            "large A-matrix upper_b[{}] not finite: {}",
            i,
            result.upper_b[i]
        );
    }
}

// ============================================================================
// 5. IBP tightness: ratio of (IBP bound width) / (true range) ≤ 10x
// ============================================================================

/// Check IBP tightness for reasonable input ranges.
/// Issue #1950 requires: ratio of (IBP width) / (true range) ≤ 10x.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_tightness_ratio() {
    use crate::layers::common::BoundPropagation;
    let layer = SoftmaxLayer::new(-1);

    let test_cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
        // Near-uniform
        (vec![0.9, 0.9, 0.9], vec![1.1, 1.1, 1.1]),
        // Small perturbation
        (vec![1.0, 2.0, 3.0], vec![1.5, 2.5, 3.5]),
        // Moderate range
        (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 3.0]),
    ];

    for (lower, upper) in &test_cases {
        let input = make_bt(lower, upper);
        let output = layer.propagate_ibp(&input).unwrap();

        // Compute true range by sampling many vertices
        let n = lower.len();
        let mut true_lower = vec![f32::INFINITY; n];
        let mut true_upper = vec![f32::NEG_INFINITY; n];
        for mask in 0..(1u32 << n) {
            let x: Vec<f32> = (0..n)
                .map(|i| {
                    if (mask >> i) & 1 == 0 {
                        lower[i]
                    } else {
                        upper[i]
                    }
                })
                .collect();
            let s = reference_softmax(&x);
            for (i, &si) in s.iter().enumerate() {
                true_lower[i] = true_lower[i].min(si);
                true_upper[i] = true_upper[i].max(si);
            }
        }

        for i in 0..n {
            let ibp_width = output.upper()[[i]] - output.lower()[[i]];
            let true_width = true_upper[i] - true_lower[i];
            if true_width > 1e-6 {
                let ratio = ibp_width / true_width;
                assert!(
                    ratio <= 10.0,
                    "IBP tightness ratio too large for case {:?}: ratio[{}] = {} (ibp_width={}, true_width={})",
                    lower, i, ratio, ibp_width, true_width
                );
            }
        }
    }
}

// ============================================================================
// 6. Softmax → cross-module composition: bounds don't explode
// ============================================================================

/// Two consecutive softmax IBP passes: bounds should not explode.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_composition_stable() {
    use crate::layers::common::BoundPropagation;
    let layer = SoftmaxLayer::new(-1);
    let input = make_bt(&[-2.0, 0.0, 2.0], &[0.0, 2.0, 4.0]);

    // First pass
    let output1 = layer.propagate_ibp(&input).unwrap();
    // Second pass (softmax of softmax output)
    let output2 = layer.propagate_ibp(&output1).unwrap();

    // Bounds should still be in [0, 1] and finite
    for i in 0..3 {
        assert!(output2.lower()[[i]].is_finite());
        assert!(output2.upper()[[i]].is_finite());
        assert!(output2.lower()[[i]] >= 0.0);
        assert!(output2.upper()[[i]] <= 1.0);
        // Second softmax of values already in [0,1] should be tighter
        let width = output2.upper()[[i]] - output2.lower()[[i]];
        assert!(
            width <= 1.0,
            "Composed softmax width[{}] = {} should be <= 1.0",
            i,
            width
        );
    }
}

// ============================================================================
// 7. CROWN backward with non-finite pre-activation falls back to IBP
// ============================================================================

/// CROWN backward with Inf in pre-activation should fall back to IBP constant bounds.
#[ntest::timeout(10000)]
#[test]
fn test_softmax_crown_backward_nonfinite_preact_fallback() {
    let layer = SoftmaxLayer::new(-1); // sound mode

    let pre_activation =
        make_bt_unchecked(&[f32::NEG_INFINITY, 0.0, 1.0], &[f32::INFINITY, 1.0, 2.0]);
    let identity_bounds = LinearBounds::identity(3);

    let result = layer
        .propagate_linear_with_bounds(
            &identity_bounds,
            &pre_activation,
            VerificationSoundnessMode::Sound,
        )
        .unwrap();

    // Should fall back to constant bounds (zero slopes)
    for &v in result.lower_a.iter() {
        assert_eq!(v, 0.0, "Non-finite fallback should give zero slopes");
    }
    // Bounds should be conservative [0, 1]-ish
    for i in 0..3 {
        assert!(result.lower_b[i] >= -0.01); // might be slightly below 0 due to sanitization margin
        assert!(result.upper_b[i] <= 1.01);
    }
}
