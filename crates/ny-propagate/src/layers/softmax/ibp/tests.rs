// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{ArrayD, IxDyn};
use proptest::prelude::*;

fn make_bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let n = lower.len();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

/// Reference softmax in f64 for high-precision comparison.
fn reference_softmax(x: &[f32]) -> Vec<f32> {
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let max = x64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x64.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

/// Check IBP soundness: for grid of concrete points x in [lower, upper],
/// verify that softmax(x) is contained in IBP output bounds.
fn assert_ibp_soundness(lower: &[f32], upper: &[f32]) {
    let layer = SoftmaxLayer::new(-1);
    let input = make_bt(lower, upper);
    let output = layer.propagate_ibp(&input).unwrap();

    let n = lower.len();

    // Test vertex samples: set each element to its lower or upper extreme.
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
                output.lower()[[i]] <= si + 1e-5,
                "IBP lower[{}] = {} > softmax[{}] = {} for x={:?}",
                i,
                output.lower()[[i]],
                i,
                si,
                x
            );
            assert!(
                output.upper()[[i]] >= si - 1e-5,
                "IBP upper[{}] = {} < softmax[{}] = {} for x={:?}",
                i,
                output.upper()[[i]],
                i,
                si,
                x
            );
        }
    }

    // Also test midpoint
    let mid: Vec<f32> = (0..n).map(|i| f32::midpoint(lower[i], upper[i])).collect();
    let s_mid = reference_softmax(&mid);
    for (i, &si) in s_mid.iter().enumerate() {
        assert!(
            output.lower()[[i]] <= si + 1e-5,
            "IBP lower[{}] = {} > softmax[{}] = {} at midpoint",
            i,
            output.lower()[[i]],
            i,
            si,
        );
        assert!(
            output.upper()[[i]] >= si - 1e-5,
            "IBP upper[{}] = {} < softmax[{}] = {} at midpoint",
            i,
            output.upper()[[i]],
            i,
            si,
        );
    }
}

// ========== IBP soundness tests ==========

#[test]
fn softmax_ibp_soundness_basic() {
    assert_ibp_soundness(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
}

#[test]
fn softmax_ibp_soundness_near_uniform() {
    // All elements close: bounds should be tight around 1/n
    assert_ibp_soundness(&[0.9, 0.9, 0.9], &[1.1, 1.1, 1.1]);
}

#[test]
fn softmax_ibp_soundness_one_dominant() {
    // One element much larger than others
    assert_ibp_soundness(&[5.0, -1.0, -1.0], &[10.0, 1.0, 1.0]);
}

#[test]
fn softmax_ibp_soundness_negative_inputs() {
    assert_ibp_soundness(&[-5.0, -3.0, -1.0], &[-2.0, -1.0, 0.0]);
}

#[test]
fn softmax_ibp_soundness_near_f32_min() {
    // Extreme finite negative ranges near f32::MIN should stay numerically stable.
    // The max-subtraction trick keeps exp arguments finite and avoids NaN/Inf.
    assert_ibp_soundness(
        &[f32::MIN / 2.0, f32::MIN / 3.0, -1.0e30],
        &[f32::MIN / 4.0, f32::MIN / 5.0, -1.0e20],
    );
}

#[test]
fn softmax_ibp_soundness_wide_spread() {
    // Large magnitude spread: hardest for tightness
    assert_ibp_soundness(&[-10.0, -5.0, 0.0], &[0.0, 5.0, 10.0]);
}

#[test]
fn softmax_ibp_soundness_two_elements() {
    assert_ibp_soundness(&[0.0, 0.0], &[2.0, 2.0]);
    assert_ibp_soundness(&[-1.0, 1.0], &[1.0, 3.0]);
}

#[test]
fn softmax_ibp_soundness_four_elements() {
    assert_ibp_soundness(&[0.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn softmax_ibp_soundness_narrow_interval() {
    // Very narrow interval: bounds should be very tight
    assert_ibp_soundness(&[1.0, 2.0, 3.0], &[1.01, 2.01, 3.01]);
}

#[test]
fn softmax_ibp_soundness_asymmetric() {
    assert_ibp_soundness(&[-0.5, -0.5, -0.5], &[0.5, 10.0, 0.5]);
}

// ========== IBP output property tests ==========

#[test]
fn softmax_ibp_outputs_in_unit_interval() {
    let layer = SoftmaxLayer::new(-1);
    let input = make_bt(&[-5.0, 0.0, 5.0], &[0.0, 5.0, 10.0]);
    let output = layer.propagate_ibp(&input).unwrap();
    for i in 0..3 {
        assert!(
            output.lower()[[i]] >= 0.0,
            "IBP lower[{}] = {} < 0",
            i,
            output.lower()[[i]]
        );
        assert!(
            output.upper()[[i]] <= 1.0,
            "IBP upper[{}] = {} > 1",
            i,
            output.upper()[[i]]
        );
    }
}

#[test]
fn softmax_ibp_point_interval_matches_eval() {
    // When lower == upper, IBP should produce tight bounds around softmax(point)
    let layer = SoftmaxLayer::new(-1);
    let x = vec![1.0, 2.0, 3.0];
    let input = make_bt(&x, &x);
    let output = layer.propagate_ibp(&input).unwrap();
    let s = reference_softmax(&x);
    for (i, &si) in s.iter().enumerate() {
        assert!(
            output.lower()[[i]] <= si + 1e-4,
            "Point IBP lower[{}] = {} > softmax = {}",
            i,
            output.lower()[[i]],
            si,
        );
        assert!(
            output.upper()[[i]] >= si - 1e-4,
            "Point IBP upper[{}] = {} < softmax = {}",
            i,
            output.upper()[[i]],
            si,
        );
    }
}

// ========== Error handling tests ==========

#[test]
fn softmax_ibp_rejects_0d_input() {
    let layer = SoftmaxLayer::new(-1);
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap(),
    )
    .unwrap();
    assert!(layer.propagate_ibp(&input).is_err());
}

#[test]
fn softmax_ibp_non_finite_falls_back_to_unit() {
    let layer = SoftmaxLayer::new(-1);
    // Use new_unchecked for non-finite inputs
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, 0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, 1.0, 1.0]).unwrap(),
    )
    .unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    // Should fall back to [0, 1] for non-finite inputs
    for i in 0..3 {
        assert!(
            output.lower()[[i]] >= 0.0 && output.upper()[[i]] <= 1.0,
            "Non-finite fallback should be [0, 1]"
        );
    }
}

#[test]
fn softmax_ibp_negative_axis_resolves() {
    // axis=-1 should resolve to last axis and produce valid, sound bounds
    let layer = SoftmaxLayer::new(-1);
    let lower = [0.0, 1.0, 2.0];
    let upper = [1.0, 2.0, 3.0];
    let input = make_bt(&lower, &upper);
    let output = layer
        .propagate_ibp(&input)
        .expect("invariant: negative axis resolves for valid 1D input");

    // Output shape must match input
    assert_eq!(output.lower().shape(), input.lower().shape());

    // Softmax outputs are in [0, 1] and bounds must be ordered
    for i in 0..3 {
        assert!(output.lower()[[i]] >= 0.0, "lower[{}] < 0", i);
        assert!(output.upper()[[i]] <= 1.0, "upper[{}] > 1", i);
        assert!(
            output.lower()[[i]] <= output.upper()[[i]],
            "lower[{}] > upper[{}]",
            i,
            i
        );
    }

    // Soundness: verify bounds contain vertex softmax evaluations
    assert_ibp_soundness(&lower, &upper);
}

// ========== 2D / batched IBP tests ==========

fn make_bt_2d(lower: &[f32], upper: &[f32], shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

/// Reference softmax in f64 for high-precision comparison.
fn reference_softmax_64(x: &[f32]) -> Vec<f32> {
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let max = x64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x64.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

#[test]
fn softmax_ibp_2d_soundness_axis_last() {
    // Shape [2, 3], axis=-1: softmax along last dimension (each row independently)
    let layer = SoftmaxLayer::new(-1);
    let lower = vec![0.0, 1.0, 2.0, -1.0, 0.0, 1.0];
    let upper = vec![1.0, 2.0, 3.0, 0.0, 1.0, 2.0];
    let input = make_bt_2d(&lower, &upper, &[2, 3]);
    let output = layer.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 3]);

    // Verify soundness: test vertex corners for each row
    for row in 0..2 {
        let row_lower = &lower[row * 3..(row + 1) * 3];
        let row_upper = &upper[row * 3..(row + 1) * 3];
        for mask in 0..8u32 {
            let x: Vec<f32> = (0..3)
                .map(|j| {
                    if (mask >> j) & 1 == 0 {
                        row_lower[j]
                    } else {
                        row_upper[j]
                    }
                })
                .collect();
            let s = reference_softmax_64(&x);
            for (j, &sj) in s.iter().enumerate() {
                assert!(
                    output.lower()[[row, j]] <= sj + 1e-5,
                    "2D IBP lower[{},{}] = {} > softmax = {} for x={:?}",
                    row,
                    j,
                    output.lower()[[row, j]],
                    sj,
                    x
                );
                assert!(
                    output.upper()[[row, j]] >= sj - 1e-5,
                    "2D IBP upper[{},{}] = {} < softmax = {} for x={:?}",
                    row,
                    j,
                    output.upper()[[row, j]],
                    sj,
                    x
                );
            }
        }
    }
}

#[test]
fn softmax_ibp_2d_outputs_in_unit_interval() {
    let layer = SoftmaxLayer::new(-1);
    let lower = vec![-5.0, 0.0, 5.0, -10.0, 0.0, 10.0];
    let upper = vec![0.0, 5.0, 10.0, -5.0, 5.0, 15.0];
    let input = make_bt_2d(&lower, &upper, &[2, 3]);
    let output = layer.propagate_ibp(&input).unwrap();

    for &v in output.lower().iter() {
        assert!(v >= 0.0, "IBP lower = {} should be >= 0", v);
    }
    for &v in output.upper().iter() {
        assert!(v <= 1.0, "IBP upper = {} should be <= 1", v);
    }
}

#[test]
fn softmax_ibp_3d_soundness() {
    // Shape [2, 2, 2], axis=-1: softmax on dim 2
    let layer = SoftmaxLayer::new(-1);
    let lower = vec![0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0];
    let upper = vec![1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0];
    let input = make_bt_2d(&lower, &upper, &[2, 2, 2]);
    let output = layer.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2, 2]);

    // Check unit interval and basic soundness for all positions
    for &v in output.lower().iter() {
        assert!(v >= 0.0);
    }
    for &v in output.upper().iter() {
        assert!(v <= 1.0);
    }
    // Verify lower <= upper everywhere
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(*l <= u + 1e-6, "lower {} > upper {}", l, u);
    }
}

#[test]
fn softmax_ibp_2d_point_interval_tight() {
    // When lower == upper, 2D IBP should produce tight bounds
    let layer = SoftmaxLayer::new(-1);
    let vals = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
    let input = make_bt_2d(&vals, &vals, &[2, 3]);
    let output = layer.propagate_ibp(&input).unwrap();

    // Row 0: softmax([1,2,3])
    let s0 = reference_softmax_64(&[1.0, 2.0, 3.0]);
    for (j, &sj) in s0.iter().enumerate() {
        assert!(
            output.lower()[[0, j]] <= sj + 1e-4,
            "point lower[0,{}] = {} > actual {}",
            j,
            output.lower()[[0, j]],
            sj
        );
        assert!(
            output.upper()[[0, j]] >= sj - 1e-4,
            "point upper[0,{}] = {} < actual {}",
            j,
            output.upper()[[0, j]],
            sj
        );
    }
    // Row 1: softmax([0,0,0]) = [1/3, 1/3, 1/3]
    for j in 0..3 {
        assert!(
            output.lower()[[1, j]] <= 1.0 / 3.0 + 1e-4,
            "point lower[1,{}] too high",
            j
        );
        assert!(
            output.upper()[[1, j]] >= 1.0 / 3.0 - 1e-4,
            "point upper[1,{}] too low",
            j
        );
    }
}

// ========== propagate_linear error path test ==========

#[test]
fn softmax_propagate_linear_returns_not_supported() {
    use crate::LinearBounds;
    let layer = SoftmaxLayer::new(-1);
    let bounds = LinearBounds::identity(3);
    let result = layer.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "propagate_linear should return UnsupportedConfiguration"
    );
}

// ========== Axis out-of-range returns error ==========

#[test]
fn softmax_ibp_axis_out_of_range_returns_error() {
    // axis=5 for 1D input: out of range, should return error
    let layer = SoftmaxLayer::new(5);
    let input = make_bt(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);
    let result = layer.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "Out-of-range axis should return error, not silently fallback"
    );
}

// ========== -inf causal masking: tight bounds, not vacuous [0, 1] ==========

/// Assert IBP soundness for concrete points against pre-computed output bounds.
fn assert_ibp_soundness_against(output: &BoundedTensor, test_points: &[Vec<f32>]) {
    for x in test_points {
        let s = reference_softmax(x);
        for (i, &si) in s.iter().enumerate() {
            assert!(
                output.lower()[[i]] <= si + 1e-5,
                "IBP lower[{i}] = {} > softmax = {si} for x={x:?}",
                output.lower()[[i]],
            );
            assert!(
                output.upper()[[i]] >= si - 1e-5,
                "IBP upper[{i}] = {} < softmax = {si} for x={x:?}",
                output.upper()[[i]],
            );
        }
    }
}

/// Regression test for #2242: SoftmaxLayer handles -inf (exp(-inf)=0).
/// Masked positions get near-[0, 0] bounds, unmasked get tight bounds.
#[test]
fn softmax_ibp_neg_inf_causal_mask_produces_tight_bounds() {
    let layer = SoftmaxLayer::new(-1);

    let lower = ArrayD::from_shape_vec(
        IxDyn(&[4]),
        vec![1.0, 2.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[4]),
        vec![3.0, 4.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
    )
    .unwrap();
    let input = BoundedTensor::new_allow_infinite(lower, upper).unwrap();
    let output = layer.propagate_ibp(&input).unwrap();

    // Masked positions: near [0, 0] (sanitize margin adds ~1e-6 to upper)
    for i in 2..4 {
        assert!(
            output.upper()[[i]] < 1e-4,
            "masked {i}: {}",
            output.upper()[[i]]
        );
        assert!(output.lower()[[i]] >= 0.0);
    }
    // Unmasked positions: non-vacuous bounds
    for i in 0..2 {
        assert!(output.upper()[[i]] < 1.0, "unmasked {i}: vacuous upper");
        assert!(output.lower()[[i]] > 0.0, "unmasked {i}: zero lower");
    }
    // Soundness: bounds contain softmax(x) for concrete inputs
    assert_ibp_soundness_against(
        &output,
        &[
            vec![1.0, 2.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
            vec![3.0, 4.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
            vec![2.0, 3.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
            vec![1.0, 4.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
            vec![3.0, 2.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
        ],
    );
}

/// Test the 4D case with -inf masking, matching the shape used by
/// causal_attention_with_cache_ibp: [batch, heads, new_seq, total_seq].
#[test]
fn softmax_ibp_4d_neg_inf_causal_mask_tight() {
    let layer = SoftmaxLayer::new(-1);

    // Shape: [1, 1, 1, 4] — single batch, single head, 1 new token, 4 total positions.
    // The new token at position 2 (cache_seq=2, new_seq=1) can attend to 0..=2.
    // Position 3 is masked with -inf.
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 4]), vec![0.5, 1.0, 1.5, f32::NEG_INFINITY])
            .unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 4]), vec![1.5, 2.0, 2.5, f32::NEG_INFINITY])
            .unwrap();
    let input = BoundedTensor::new_allow_infinite(lower, upper).unwrap();

    let output = layer.propagate_ibp(&input).unwrap();

    // Masked position (index 3) should be near [0, 0], not [0, 1]
    assert!(
        output.upper()[[0, 0, 0, 3]] < 1e-4,
        "Masked position should have upper ~0, got {}",
        output.upper()[[0, 0, 0, 3]]
    );

    // Unmasked positions should have non-vacuous bounds
    for j in 0..3 {
        assert!(
            output.upper()[[0, 0, 0, j]] < 1.0,
            "Unmasked position {} should have upper < 1.0 (non-vacuous), got {}",
            j,
            output.upper()[[0, 0, 0, j]]
        );
        assert!(
            output.lower()[[0, 0, 0, j]] > 0.0,
            "Unmasked position {} should have lower > 0, got {}",
            j,
            output.lower()[[0, 0, 0, j]]
        );
    }
}

/// Mixed -inf lower / finite upper: position might or might not be masked.
/// exp(-inf)=0 for lower, exp(finite) for upper — soundness requires
/// output lower=0 and output upper reflecting the finite contribution.
#[test]
fn softmax_ibp_mixed_neg_inf_lower_finite_upper() {
    let layer = SoftmaxLayer::new(-1);
    // Position 2: lower=-inf (masked case), upper=2.0 (unmasked case)
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 1.0, f32::NEG_INFINITY]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 3.0, 2.0]).unwrap();
    let input = BoundedTensor::new_allow_infinite(lower, upper).unwrap();
    let output = layer.propagate_ibp(&input).unwrap();

    // Position 2 lower must be 0 (achieved when input=-inf, exp(-inf)=0)
    assert!(
        output.lower()[[2]] < 1e-5,
        "mixed lower={}",
        output.lower()[[2]]
    );
    // Position 2 upper must be positive (when input=2.0, it contributes)
    assert!(
        output.upper()[[2]] > 0.01,
        "mixed upper={}",
        output.upper()[[2]]
    );
    // Soundness: bounds contain softmax at several concrete points
    assert_ibp_soundness_against(
        &output,
        &[
            vec![1.0, 1.0, f32::NEG_INFINITY],
            vec![3.0, 3.0, 2.0],
            vec![2.0, 2.0, 1.0],
            vec![1.0, 3.0, 0.0],
        ],
    );
}

// ========== NaN integration tests (#2627) ==========
// Softmax IBP has NaN guards at lines 37-48 of ibp/mod.rs: if any input element
// is NaN, ok=false and fallback to [0, 1]. These tests verify the fold site
// integration, not just the nan_propagating_max primitive.

/// NaN in lower bound triggers [0, 1] fallback (#2627).
#[test]
fn softmax_ibp_nan_in_lower_falls_back_to_unit_2627() {
    let layer = SoftmaxLayer::new(-1);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    for i in 0..3 {
        assert!(
            output.lower()[[i]] >= 0.0 && output.upper()[[i]] <= 1.0,
            "NaN lower fallback should be [0, 1], got [{}, {}] at {i}",
            output.lower()[[i]],
            output.upper()[[i]]
        );
    }
}

/// NaN in upper bound triggers [0, 1] fallback (#2627).
#[test]
fn softmax_ibp_nan_in_upper_falls_back_to_unit_2627() {
    let layer = SoftmaxLayer::new(-1);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, f32::NAN, 1.0]).unwrap(),
    )
    .unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    for i in 0..3 {
        assert!(
            output.lower()[[i]] >= 0.0 && output.upper()[[i]] <= 1.0,
            "NaN upper fallback should be [0, 1] at {i}"
        );
    }
}

/// All-NaN input triggers [0, 1] fallback (#2627).
#[test]
fn softmax_ibp_all_nan_falls_back_to_unit_2627() {
    let layer = SoftmaxLayer::new(-1);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![f32::NAN; 4]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![f32::NAN; 4]).unwrap(),
    )
    .unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    for i in 0..4 {
        assert!(
            output.lower()[[i]] >= 0.0 && output.upper()[[i]] <= 1.0,
            "all-NaN fallback should be [0, 1] at {i}"
        );
    }
    // Output must not contain NaN
    assert!(
        output.lower().iter().all(|v| !v.is_nan()),
        "fallback lower must not contain NaN"
    );
    assert!(
        output.upper().iter().all(|v| !v.is_nan()),
        "fallback upper must not contain NaN"
    );
}

/// 2D batched input with NaN in one row: NaN row falls back, clean row computes normally (#2627).
#[test]
fn softmax_ibp_2d_nan_in_one_row_per_lane_fallback_2627() {
    let layer = SoftmaxLayer::new(-1);
    // Row 0: clean. Row 1: NaN in lower.
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, f32::NAN, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 1.0, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    assert_eq!(output.shape(), &[2, 3]);
    // Row 1 (NaN): should fall back to [0, 1]
    for j in 0..3 {
        assert!(
            output.lower()[[1, j]] >= 0.0 && output.upper()[[1, j]] <= 1.0,
            "NaN row [1,{j}] should be [0, 1]"
        );
    }
    // Row 0 (clean): should have non-vacuous bounds (lower > 0 for softmax)
    for j in 0..3 {
        assert!(
            output.lower()[[0, j]] >= 0.0 && output.upper()[[0, j]] <= 1.0,
            "clean row [0,{j}] should be in [0, 1]"
        );
    }
}

// ========== Monotonicity: tighter input → tighter output ==========

#[test]
fn softmax_ibp_monotonicity_tighter_input_tighter_output() {
    let layer = SoftmaxLayer::new(-1);

    // Wide interval
    let wide = make_bt(&[-5.0, -5.0, -5.0], &[5.0, 5.0, 5.0]);
    let out_wide = layer.propagate_ibp(&wide).unwrap();

    // Narrow interval (subset of wide)
    let narrow = make_bt(&[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0]);
    let out_narrow = layer.propagate_ibp(&narrow).unwrap();

    for i in 0..3 {
        let width_wide = out_wide.upper()[[i]] - out_wide.lower()[[i]];
        let width_narrow = out_narrow.upper()[[i]] - out_narrow.lower()[[i]];
        assert!(
            width_narrow <= width_wide + 1e-5,
            "Narrower input should produce narrower output bounds: \
             narrow width[{}] = {} > wide width = {}",
            i,
            width_narrow,
            width_wide
        );
    }
}

// ========== Preserve-leading-axis tests (#4096) ==========

/// Positive stored axis under preserve-leading-axis mode should shift right
/// by one, matching the sequential result. Part of #4096.
#[test]
fn softmax_preserve_leading_axis_restores_positive_axis_4096() {
    // Simulate: ONNX Softmax(axis=2) on [seq, hidden, vocab] stored as axis=1.
    // Sequential: shape [3, 4], axis=1 → softmax over dim 1 (4 elements).
    // Restart-batched: shape [2, 3, 4], axis=1 stored → restored to axis=2.
    let layer = SoftmaxLayer::new(1); // stored unbatched axis

    let sequential_lower = vec![0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0, 0.5, 1.5, 2.5, 3.5];
    let sequential_upper = vec![1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 4.5];
    let seq_input = make_bt_2d(&sequential_lower, &sequential_upper, &[3, 4]);
    let seq_output = layer.propagate_ibp(&seq_input).unwrap();

    // Restart-batched: prepend restart axis of size 1 to get [1, 3, 4].
    let batch_lower = ArrayD::from_shape_vec(IxDyn(&[1, 3, 4]), sequential_lower).unwrap();
    let batch_upper = ArrayD::from_shape_vec(IxDyn(&[1, 3, 4]), sequential_upper).unwrap();
    let batch_input = BoundedTensor::new(batch_lower, batch_upper).unwrap();
    let batch_output = layer
        .propagate_ibp_preserve_leading_axis(&batch_input)
        .unwrap();

    assert_eq!(batch_output.shape(), &[1, 3, 4]);

    // Batched result sliced at restart=0 should match sequential exactly.
    for i in 0..12 {
        let row = i / 4;
        let col = i % 4;
        let seq_l = seq_output.lower()[[row, col]];
        let bat_l = batch_output.lower()[[0, row, col]];
        assert!(
            (seq_l - bat_l).abs() < 1e-6,
            "lower[{row},{col}] seq={seq_l} vs batch={bat_l}",
        );
        let seq_u = seq_output.upper()[[row, col]];
        let bat_u = batch_output.upper()[[0, row, col]];
        assert!(
            (seq_u - bat_u).abs() < 1e-6,
            "upper[{row},{col}] seq={seq_u} vs batch={bat_u}",
        );
    }
}

/// Negative axis should behave identically with or without preserve-leading-axis
/// since negative axes are end-relative and unaffected by a prepended axis.
#[test]
fn softmax_preserve_leading_axis_negative_axis_unchanged_4096() {
    let layer = SoftmaxLayer::new(-1);

    let lower = vec![0.0, 1.0, 2.0, -1.0, 0.0, 1.0];
    let upper = vec![1.0, 2.0, 3.0, 0.0, 1.0, 2.0];

    // Sequential: [2, 3], axis=-1 → softmax over last dim.
    let seq_input = make_bt_2d(&lower, &upper, &[2, 3]);
    let seq_output = layer.propagate_ibp(&seq_input).unwrap();

    // Batched: [1, 2, 3], axis=-1 → still softmax over last dim.
    let batch_lower = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), lower).unwrap();
    let batch_upper = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), upper).unwrap();
    let batch_input = BoundedTensor::new(batch_lower, batch_upper).unwrap();
    let batch_output = layer
        .propagate_ibp_preserve_leading_axis(&batch_input)
        .unwrap();

    assert_eq!(batch_output.shape(), &[1, 2, 3]);

    for row in 0..2 {
        for col in 0..3 {
            assert!(
                (seq_output.lower()[[row, col]] - batch_output.lower()[[0, row, col]]).abs() < 1e-6,
                "negative axis: lower mismatch at [{row},{col}]"
            );
            assert!(
                (seq_output.upper()[[row, col]] - batch_output.upper()[[0, row, col]]).abs() < 1e-6,
                "negative axis: upper mismatch at [{row},{col}]"
            );
        }
    }
}

// ========== #4231: underflow / large-score-gap false-proof regression ==========
//
// A SHARED max-shift plus a fixed `+ SOFTMAX_EPSILON` in the denominator
// UNDER-approximated p_hi for a REACHABLE key whose own scores sat far below the
// row's max-upper: its numerator exp(u_i - M) underflowed toward 0 while epsilon
// swamped the surviving sub-1e-12 terms, so p_hi collapsed to ~0 and the IBP
// interval EXCLUDED the reachable true softmax (=1.0 at the dominating corner) —
// a FALSE certificate. The fix shifts each ratio by its own dominant term and
// drops the additive epsilon. These tests run the REAL propagate_ibp.

/// Exact per-coordinate monotone optimum, computed in f64 at the relevant corner.
/// p_hi[i]: coord i at upper, all others at lower. p_lo[i]: coord i at lower,
/// all others at upper. This is the bound the IBP must (outward-roundedly) match.
fn exact_corner_bounds(lower: &[f32], upper: &[f32]) -> Vec<(f64, f64)> {
    let n = lower.len();
    (0..n)
        .map(|i| {
            let hi_corner: Vec<f32> = (0..n)
                .map(|j| if j == i { upper[i] } else { lower[j] })
                .collect();
            let lo_corner: Vec<f32> = (0..n)
                .map(|j| if j == i { lower[i] } else { upper[j] })
                .collect();
            let p_hi = softmax_f64_at(&hi_corner, i);
            let p_lo = softmax_f64_at(&lo_corner, i);
            (p_lo, p_hi)
        })
        .collect()
}

fn softmax_f64_at(x: &[f32], i: usize) -> f64 {
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let m = x64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x64.iter().map(|&v| (v - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps[i] / sum
}

/// Witness from confirmation analyst #1: 3 keys, key index 1 reachable -> 1.0.
/// Pre-fix: ny returned p_hi[1] = 1e-6 (EXCLUDES true reachable softmax[1] = 1.0).
#[test]
fn softmax_ibp_underflow_witness_3key_encloses_true_4231() {
    // Exact f32 witness values (shortest decimals that round-trip to the exact
    // f32 bits of the original full-precision witness).
    let lower = [-142.971_16_f32, -433.949_86_f32, -171.118_55_f32];
    let upper = [464.317_93_f32, 20.294_498_f32, 510.837_46_f32];
    let layer = SoftmaxLayer::new(-1);
    let output = layer.propagate_ibp(&make_bt(&lower, &upper)).unwrap();

    // At the dominating corner x = (l0, u1, l2), key1 wins => true softmax[1] = 1.0.
    let corner = [lower[0], upper[1], lower[2]];
    let true_s1 = reference_softmax(&corner)[1];
    assert!(
        (true_s1 - 1.0).abs() < 1e-3,
        "sanity: true softmax[1] at dominating corner = {true_s1}, expected ~1.0"
    );

    let p_lo1 = output.lower()[[1]];
    let p_hi1 = output.upper()[[1]];
    assert!(
        p_lo1 <= true_s1 + 1e-5 && p_hi1 >= true_s1 - 1e-5,
        "UNSOUND #4231: ny p[1] = [{p_lo1}, {p_hi1}] EXCLUDES reachable softmax[1] = {true_s1}"
    );
    // The reachable upper must be ~1, not collapsed to the 1e-6 sanitize floor.
    assert!(
        p_hi1 > 0.99,
        "p_hi[1] collapsed to {p_hi1} (regression of the underflow false-proof)"
    );
}

/// Witness from confirmation analyst #2: 4 keys, key index 3 reachable -> 1.0.
/// Pre-fix: ny returned p_hi[3] = 1e-6 while softmax[3] reaches 1.0.
#[test]
fn softmax_ibp_underflow_witness_4key_encloses_true_4231() {
    let lower = [-175.8_f32, -476.9, -682.7, -43.6];
    let upper = [-173.9_f32, 172.5, 168.1, -16.1];
    let layer = SoftmaxLayer::new(-1);
    let output = layer.propagate_ibp(&make_bt(&lower, &upper)).unwrap();

    let corner = [lower[0], lower[1], lower[2], upper[3]];
    let true_s3 = reference_softmax(&corner)[3];
    assert!(
        (true_s3 - 1.0).abs() < 1e-3,
        "sanity: true softmax[3] at dominating corner = {true_s3}, expected ~1.0"
    );

    let p_lo3 = output.lower()[[3]];
    let p_hi3 = output.upper()[[3]];
    assert!(
        p_lo3 <= true_s3 + 1e-5 && p_hi3 >= true_s3 - 1e-5,
        "UNSOUND #4231: ny p[3] = [{p_lo3}, {p_hi3}] EXCLUDES reachable softmax[3] = {true_s3}"
    );
    assert!(
        p_hi3 > 0.99,
        "p_hi[3] collapsed to {p_hi3} (regression of the underflow false-proof)"
    );
}

/// A >745-gap row (the regime in the originally reported incident): one key's
/// upper sits ~750 below another's. The reachable key's p_hi must still be ~1.
#[test]
fn softmax_ibp_underflow_gap_745_encloses_true_4231() {
    // key0 can rise to 50; key1 sits ~750 below at upper; key2 keeps a finite,
    // very-negative lower so the row's lower-exp sum stays marginally > 0.
    let lower = [-900.0_f32, -50.0, -880.0];
    let upper = [50.0_f32, 700.0, -860.0];
    let layer = SoftmaxLayer::new(-1);
    let output = layer.propagate_ibp(&make_bt(&lower, &upper)).unwrap();

    // key0 reachable-winning corner: (u0=50, l1=-50, l2=-880) => softmax[0] ~ 1.
    let corner = [upper[0], lower[1], lower[2]];
    let true_s0 = reference_softmax(&corner)[0];
    let p_hi0 = output.upper()[[0]];
    assert!(
        p_hi0 >= true_s0 - 1e-5,
        "UNSOUND #4231: p_hi[0] = {p_hi0} EXCLUDES reachable softmax[0] = {true_s0} (gap ~750)"
    );
    assert!(
        p_hi0 > 0.99,
        "p_hi[0] collapsed to {p_hi0} in the >745-gap regime"
    );
}

/// Normal-regime exactness is PRESERVED: where the bound was already the exact
/// per-coordinate monotone optimum, the fix must keep it exact (no widening
/// beyond the outward 1-ULP + 1e-6 sanitize margin).
#[test]
fn softmax_ibp_normal_regime_matches_exact_corner_4231() {
    let cases: &[(&[f32], &[f32])] = &[
        (&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]),
        (&[-5.0, -3.0, -1.0], &[-2.0, -1.0, 0.0]),
        (&[5.0, -1.0, -1.0], &[10.0, 1.0, 1.0]),
        (&[0.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 3.0, 4.0]),
        (&[-10.0, -5.0, 0.0], &[0.0, 5.0, 10.0]),
    ];
    let layer = SoftmaxLayer::new(-1);
    for (lo, hi) in cases {
        let output = layer.propagate_ibp(&make_bt(lo, hi)).unwrap();
        let exact = exact_corner_bounds(lo, hi);
        for (i, &(ex_lo, ex_hi)) in exact.iter().enumerate() {
            let got_lo = output.lower()[[i]] as f64;
            let got_hi = output.upper()[[i]] as f64;
            // Sound (encloses exact corner) AND tight (within the sanitize margin).
            assert!(
                got_lo <= ex_lo + 1e-6 && got_lo >= ex_lo - 2.1e-6,
                "lower[{i}] = {got_lo} not exact vs corner {ex_lo} (lo={lo:?} hi={hi:?})"
            );
            assert!(
                got_hi >= ex_hi - 1e-6 && got_hi <= ex_hi + 2.1e-6,
                "upper[{i}] = {got_hi} not exact vs corner {ex_hi} (lo={lo:?} hi={hi:?})"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// SOUNDNESS over the FULL range including the >745-gap underflow regime:
    /// for every box, the IBP output must enclose true softmax at all 2^n
    /// vertices (the true optimum lives at a corner for each coordinate).
    #[test]
    fn softmax_ibp_sound_including_underflow_regime_4231(
        base in proptest::collection::vec(-900.0f32..900.0, 2..5),
        widths in proptest::collection::vec(0.0f32..850.0, 2..5),
    ) {
        let n = base.len().min(widths.len());
        let lower: Vec<f32> = base[..n].to_vec();
        let upper: Vec<f32> = (0..n).map(|i| lower[i] + widths[i]).collect();

        let layer = SoftmaxLayer::new(-1);
        let output = layer.propagate_ibp(&make_bt(&lower, &upper)).unwrap();

        // Output bounds must be valid: [0,1] and ordered.
        for i in 0..n {
            let lo = output.lower()[[i]];
            let hi = output.upper()[[i]];
            prop_assert!((0.0..=1.0).contains(&lo), "lower[{i}]={lo} out of [0,1]");
            prop_assert!((0.0..=1.0).contains(&hi), "upper[{i}]={hi} out of [0,1]");
            prop_assert!(lo <= hi + 1e-6, "lower[{i}]={lo} > upper[{i}]={hi}");
        }

        // Enclosure at every vertex (margin = outward ~1 ULP + 1e-6 sanitize).
        for mask in 0u32..(1u32 << n) {
            let x: Vec<f32> = (0..n)
                .map(|i| if (mask >> i) & 1 == 0 { lower[i] } else { upper[i] })
                .collect();
            let s = reference_softmax(&x);
            for i in 0..n {
                let lo = output.lower()[[i]];
                let hi = output.upper()[[i]];
                prop_assert!(
                    lo <= s[i] + 5e-5,
                    "UNSOUND lower[{i}]={lo} > softmax={} x={x:?}", s[i]
                );
                prop_assert!(
                    hi >= s[i] - 5e-5,
                    "UNSOUND upper[{i}]={hi} < softmax={} x={x:?}", s[i]
                );
            }
        }
    }

    /// NORMAL regime (no underflow): the bound stays the EXACT per-coordinate
    /// monotone optimum, within the outward 1-ULP + 1e-6 sanitize margin.
    #[test]
    fn softmax_ibp_normal_regime_exactness_preserved_4231(
        base in proptest::collection::vec(-15.0f32..15.0, 2..5),
        widths in proptest::collection::vec(0.0f32..8.0, 2..5),
    ) {
        let n = base.len().min(widths.len());
        let lower: Vec<f32> = base[..n].to_vec();
        let upper: Vec<f32> = (0..n).map(|i| lower[i] + widths[i]).collect();

        let layer = SoftmaxLayer::new(-1);
        let output = layer.propagate_ibp(&make_bt(&lower, &upper)).unwrap();
        let exact = exact_corner_bounds(&lower, &upper);

        for (i, &(ex_lo, ex_hi)) in exact.iter().enumerate() {
            let got_lo = output.lower()[[i]] as f64;
            let got_hi = output.upper()[[i]] as f64;
            // Outward (sound): encloses the exact corner.
            prop_assert!(got_lo <= ex_lo + 2e-6, "lower[{i}]={got_lo} above corner {ex_lo}");
            prop_assert!(got_hi >= ex_hi - 2e-6, "upper[{i}]={got_hi} below corner {ex_hi}");
            // Tight (exact): not widened beyond the sanitize margin.
            prop_assert!(got_lo >= ex_lo - 2.1e-6, "lower[{i}]={got_lo} too loose vs corner {ex_lo}");
            prop_assert!(got_hi <= ex_hi + 2.1e-6, "upper[{i}]={got_hi} too loose vs corner {ex_hi}");
        }
    }
}
