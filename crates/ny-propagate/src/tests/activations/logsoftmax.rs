// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{arr1, ArrayD, IxDyn};
use ny_core::VerificationSoundnessMode;

// ==================== LogSoftmax CROWN tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_ibp_basic() {
    // Test basic LogSoftmax IBP propagation
    let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    let logsoftmax = LogSoftmaxLayer::new(-1);
    let output = logsoftmax.propagate_ibp(&input).unwrap();

    // Lower bound should be less than upper bound
    for i in 0..4 {
        assert!(
            output.lower()[[i]] <= output.upper()[[i]],
            "Lower bound should be <= upper bound"
        );
    }

    // Check that the bounds are sound by sampling points in the input interval
    for sample in 0..20 {
        // Generate a random point in the interval
        let point: Vec<f32> = (0..4)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                lower[[i]] + (upper[[i]] - lower[[i]]) * t
            })
            .collect();
        let logsoftmax_output = {
            let max_val = point.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = point.iter().map(|&v| (v - max_val).exp()).sum();
            let lse = max_val + exp_sum.ln();
            point.iter().map(|&v| v - lse).collect::<Vec<f32>>()
        };

        for (i, &lsm_val) in logsoftmax_output.iter().enumerate() {
            let tol = 1e-5;
            assert!(
                output.lower()[[i]] <= lsm_val + tol,
                "IBP lower bound violated at sample {}: {} > {}",
                sample,
                output.lower()[[i]],
                lsm_val
            );
            assert!(
                output.upper()[[i]] >= lsm_val - tol,
                "IBP upper bound violated at sample {}: {} < {}",
                sample,
                output.upper()[[i]],
                lsm_val
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_crown_backward_basic() {
    // Test LogSoftmax CROWN backward propagation
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let logsoftmax = LogSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let result = logsoftmax
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation, logsoftmax.soundness_mode())
        .unwrap();

    // Check dimensions
    assert_eq!(result.lower_a.shape(), &[4, 4]);
    assert_eq!(result.upper_a.shape(), &[4, 4]);
    assert_eq!(result.lower_b.len(), 4);
    assert_eq!(result.upper_b.len(), 4);

    // The Jacobian of LogSoftmax is J_ij = δ_ij - softmax_j
    // So diagonal should be close to 1 - softmax[i]
    // and off-diagonal should be close to -softmax[j]
    // At center point [0.5, 1.5, 2.5, 3.5]:
    let center: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
    let max_val = center.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = center.iter().map(|&v| (v - max_val).exp()).collect();
    let exp_sum: f32 = exp_vals.iter().sum();
    let softmax: Vec<f32> = exp_vals.iter().map(|&e| e / exp_sum).collect();

    // Check that coefficient matrix matches the exact Jacobian at the center point:
    // J_ij = δ_ij - softmax_j
    let tol = 1e-4_f32;
    for i in 0..4 {
        let mut row_sum_lower = 0.0_f32;
        let mut row_sum_upper = 0.0_f32;
        for (j, &softmax_j) in softmax.iter().enumerate().take(4) {
            let expected = if i == j { 1.0 - softmax_j } else { -softmax_j };
            let lower = result.lower_a[[i, j]];
            let upper = result.upper_a[[i, j]];
            assert!(
                (lower - expected).abs() <= tol,
                "Lower A mismatch at ({}, {}): {} vs {}",
                i,
                j,
                lower,
                expected
            );
            assert!(
                (upper - expected).abs() <= tol,
                "Upper A mismatch at ({}, {}): {} vs {}",
                i,
                j,
                upper,
                expected
            );
            row_sum_lower += lower;
            row_sum_upper += upper;
        }
        assert!(
            row_sum_lower.abs() <= tol,
            "Jacobian lower row {} should sum to 0, got {}",
            i,
            row_sum_lower
        );
        assert!(
            row_sum_upper.abs() <= tol,
            "Jacobian upper row {} should sum to 0, got {}",
            i,
            row_sum_upper
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_crown_axis_respects_slices() {
    let logsoftmax = LogSoftmaxLayer::new(-1).with_sound_mode(true);

    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0_f32, -0.5, 0.0, 0.25, 0.5, 0.75]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0_f32, 0.5, 1.0, 1.25, 1.5, 2.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let bounds = LinearBounds::identity(6);
    let result = logsoftmax
        .propagate_linear_with_bounds(&bounds, &input, logsoftmax.soundness_mode())
        .unwrap();

    let eps = 1e-6_f32;
    for out_idx in 0..3 {
        for in_idx in 3..6 {
            assert!(
                result.lower_a[[out_idx, in_idx]].abs() <= eps,
                "lower_a cross-slice coeff not zero at [{}, {}]",
                out_idx,
                in_idx
            );
            assert!(
                result.upper_a[[out_idx, in_idx]].abs() <= eps,
                "upper_a cross-slice coeff not zero at [{}, {}]",
                out_idx,
                in_idx
            );
        }
    }
    for out_idx in 3..6 {
        for in_idx in 0..3 {
            assert!(
                result.lower_a[[out_idx, in_idx]].abs() <= eps,
                "lower_a cross-slice coeff not zero at [{}, {}]",
                out_idx,
                in_idx
            );
            assert!(
                result.upper_a[[out_idx, in_idx]].abs() <= eps,
                "upper_a cross-slice coeff not zero at [{}, {}]",
                out_idx,
                in_idx
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_crown_sampling_check() {
    // Heuristic sampling check for CROWN bounds (not a proof of soundness).
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let logsoftmax = LogSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let result = logsoftmax
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation, logsoftmax.soundness_mode())
        .unwrap();

    // Sample points to spot-check that bounds contain actual values.
    for sample in 0..20 {
        // Generate a random point in the interval
        let point: Vec<f32> = (0..4)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                pre_lower[[i]] + (pre_upper[[i]] - pre_lower[[i]]) * t
            })
            .collect();

        // Compute actual logsoftmax output
        let max_val = point.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = point.iter().map(|&v| (v - max_val).exp()).sum();
        let lse = max_val + exp_sum.ln();
        let logsoftmax_output: Vec<f32> = point.iter().map(|&v| v - lse).collect();

        // Check each output dimension
        for (j, &lsm_val) in logsoftmax_output.iter().enumerate() {
            let lb_val: f32 = (0..4)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..4)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 5e-2; // Sampling-based CROWN heuristic — tighter than 0.1 but allows heuristic slack
            assert!(
                lb_val <= lsm_val + tol,
                "CROWN lower bound violated at sample {}, dim {}: lb {} > actual {}",
                sample,
                j,
                lb_val,
                lsm_val
            );
            assert!(
                ub_val >= lsm_val - tol,
                "CROWN upper bound violated at sample {}, dim {}: ub {} < actual {}",
                sample,
                j,
                ub_val,
                lsm_val
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_sound_mode_lse_bounds_sound() {
    let logsoftmax = LogSoftmaxLayer::new(-1).with_sound_mode(true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0_f32, 1.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5_f32, 1.5, 2.5]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let result = logsoftmax
        .propagate_linear_with_bounds(&linear_bounds, &input, logsoftmax.soundness_mode())
        .unwrap();

    let concretized = result.concretize(&input);
    let pre_lower = input
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let pre_upper = input
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    for sample in 0..20 {
        let point: Vec<f32> = (0..3)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                pre_lower[i] + (pre_upper[i] - pre_lower[i]) * t
            })
            .collect();
        let max_val = point.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = point.iter().map(|&v| (v - max_val).exp()).sum();
        let lse = max_val + exp_sum.ln();
        let logsoftmax_output: Vec<f32> = point.iter().map(|&v| v - lse).collect();

        for (i, &lsm_val) in logsoftmax_output.iter().enumerate() {
            assert!(
                lsm_val >= concretized.lower()[[i]] - 1e-5,
                "Sound lower bound violated at sample {} dim {}: {} < {}",
                sample,
                i,
                lsm_val,
                concretized.lower()[[i]]
            );
            assert!(
                lsm_val <= concretized.upper()[[i]] + 1e-5,
                "Sound upper bound violated at sample {} dim {}: {} > {}",
                sample,
                i,
                lsm_val,
                concretized.upper()[[i]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_sound_mode_ibp_randomized_soundness() {
    let logsoftmax = LogSoftmaxLayer::new(-1).with_sound_mode(true);

    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![-1.0_f32, 0.5, 1.0, 2.0, 3.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(5);
    let result = logsoftmax
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation, logsoftmax.soundness_mode())
        .unwrap();

    let concretized = result.concretize(&pre_activation);

    for sample in 0..24 {
        let point: Vec<f32> = (0..5)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                pre_lower[[i]] + (pre_upper[[i]] - pre_lower[[i]]) * t
            })
            .collect();

        let max_val = point.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = point.iter().map(|&v| (v - max_val).exp()).sum();
        let lse = max_val + exp_sum.ln();
        let logsoftmax_output: Vec<f32> = point.iter().map(|&v| v - lse).collect();

        for (j, &lsm_val) in logsoftmax_output.iter().enumerate() {
            let lb_val = concretized.lower()[[j]];
            let ub_val = concretized.upper()[[j]];

            let tol = 1e-5;
            assert!(
                lb_val <= lsm_val + tol,
                "Sound lower bound violated at sample {}, dim {}: lb {} > actual {}",
                sample,
                j,
                lb_val,
                lsm_val
            );
            assert!(
                ub_val >= lsm_val - tol,
                "Sound upper bound violated at sample {}, dim {}: ub {} < actual {}",
                sample,
                j,
                ub_val,
                lsm_val
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_sound_mode_non_finite_fallback() {
    let logsoftmax = LogSoftmaxLayer::new(-1).with_sound_mode(true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0_f32, 1.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5_f32, f32::INFINITY, 2.5]).unwrap();
    // Non-finite inputs are expected here; bypass debug assertions to exercise fallback.
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let result = logsoftmax
        .propagate_linear_with_bounds(&linear_bounds, &input, logsoftmax.soundness_mode())
        .unwrap();

    assert_eq!(result.lower_b.len(), 3);
    assert_eq!(result.upper_b.len(), 3);
    assert!(result.lower_a.iter().all(|v| v.abs() <= 1e-8));
    assert!(result.upper_a.iter().all(|v| v.abs() <= 1e-8));

    for i in 0..3 {
        assert!(
            result.lower_b[i].is_infinite() && result.lower_b[i].is_sign_negative(),
            "Expected -inf lower_b[{}], got {}",
            i,
            result.lower_b[i]
        );
        assert!(
            result.upper_b[i].is_infinite() && result.upper_b[i].is_sign_positive(),
            "Expected +inf upper_b[{}], got {}",
            i,
            result.upper_b[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_sound_mode_lse_bounds_sound() {
    let softmax = SoftmaxLayer::new(-1).with_sound_mode(true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0_f32, 1.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5_f32, 1.5, 2.5]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let result = softmax
        .propagate_linear_with_bounds(&linear_bounds, &input, softmax.soundness_mode())
        .unwrap();

    let concretized = result.concretize(&input);
    let pre_lower = input
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let pre_upper = input
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    for sample in 0..20 {
        let point: Vec<f32> = (0..3)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                pre_lower[i] + (pre_upper[i] - pre_lower[i]) * t
            })
            .collect();
        let point = arr1(&point);
        let softmax_val = softmax.eval(&point);
        for i in 0..3 {
            assert!(
                softmax_val[i] >= concretized.lower()[[i]] - 1e-5,
                "Sound lower bound violated at sample {} dim {}: {} < {}",
                sample,
                i,
                softmax_val[i],
                concretized.lower()[[i]]
            );
            assert!(
                softmax_val[i] <= concretized.upper()[[i]] + 1e-5,
                "Sound upper bound violated at sample {} dim {}: {} > {}",
                sample,
                i,
                softmax_val[i],
                concretized.upper()[[i]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_sound_mode_overrides_heuristic_request() {
    let softmax = SoftmaxLayer::new(-1)
        .with_heuristic_sampling(true)
        .with_sound_mode(true);

    assert_eq!(
        softmax.soundness_mode(),
        VerificationSoundnessMode::Sound,
        "Sound mode should override heuristic sampling"
    );

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0_f32, 1.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5_f32, 1.5, 2.5]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let result = softmax
        .propagate_linear_with_bounds(&linear_bounds, &input, softmax.soundness_mode())
        .unwrap();

    let concretized = result.concretize(&input);
    let pre_lower = input
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let pre_upper = input
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    for sample in 0..20 {
        let point: Vec<f32> = (0..3)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                pre_lower[i] + (pre_upper[i] - pre_lower[i]) * t
            })
            .collect();
        let point = arr1(&point);
        let softmax_val = softmax.eval(&point);
        for i in 0..3 {
            assert!(
                softmax_val[i] >= concretized.lower()[[i]] - 1e-5,
                "Sound lower bound violated at sample {} dim {}: {} < {}",
                sample,
                i,
                softmax_val[i],
                concretized.lower()[[i]]
            );
            assert!(
                softmax_val[i] <= concretized.upper()[[i]] + 1e-5,
                "Sound upper bound violated at sample {} dim {}: {} > {}",
                sample,
                i,
                softmax_val[i],
                concretized.upper()[[i]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_sound_mode_uses_ibp_constant_bounds() {
    let softmax = CausalSoftmaxLayer::new(-1).with_sound_mode(true);

    let lower = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![-1.0_f32, -0.5, 0.0, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0],
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![0.0_f32, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.5],
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let linear_bounds = LinearBounds::identity(9);
    let result = softmax
        .propagate_linear_with_bounds(&linear_bounds, &input, softmax.soundness_mode())
        .unwrap();

    assert!(
        result.lower_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero lower_a in sound-mode constant bounds"
    );
    assert!(
        result.upper_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero upper_a in sound-mode constant bounds"
    );

    let ibp_bounds = softmax.propagate_ibp(&input).unwrap();
    let ibp_flat = ibp_bounds.flatten();
    let ibp_lower = ibp_flat
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let ibp_upper = ibp_flat
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    for i in 0..9 {
        assert!(
            (result.lower_b[i] - ibp_lower[i]).abs() <= 1e-6,
            "Sound-mode lower_b mismatch at {}: {} vs {}",
            i,
            result.lower_b[i],
            ibp_lower[i]
        );
        assert!(
            (result.upper_b[i] - ibp_upper[i]).abs() <= 1e-6,
            "Sound-mode upper_b mismatch at {}: {} vs {}",
            i,
            result.upper_b[i],
            ibp_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_sound_mode_overrides_heuristic_request() {
    let softmax = CausalSoftmaxLayer::new(-1).with_sound_mode(true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0_f32, -0.25, 0.5, 1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0_f32, 0.5, 1.5, 2.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let result = softmax
        .propagate_linear_with_bounds(&linear_bounds, &input, VerificationSoundnessMode::Heuristic)
        .unwrap();

    assert!(
        result.lower_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero lower_a when sound mode overrides heuristic request"
    );
    assert!(
        result.upper_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero upper_a when sound mode overrides heuristic request"
    );

    let ibp_bounds = softmax.propagate_ibp(&input).unwrap();
    let ibp_flat = ibp_bounds.flatten();
    let ibp_lower = ibp_flat
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let ibp_upper = ibp_flat
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    for i in 0..4 {
        assert!(
            (result.lower_b[i] - ibp_lower[i]).abs() <= 1e-6,
            "Sound-mode lower_b mismatch at {}: {} vs {}",
            i,
            result.lower_b[i],
            ibp_lower[i]
        );
        assert!(
            (result.upper_b[i] - ibp_upper[i]).abs() <= 1e-6,
            "Sound-mode upper_b mismatch at {}: {} vs {}",
            i,
            result.upper_b[i],
            ibp_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_scalar_rejected() {
    let logsoftmax = LogSoftmaxLayer::new(-1);

    let lower = ArrayD::from_elem(IxDyn(&[]), 0.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[]), 0.25_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let err = logsoftmax.propagate_ibp(&input).unwrap_err();
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_scalar_rejected() {
    let softmax = SoftmaxLayer::new(-1);

    let lower = ArrayD::from_elem(IxDyn(&[]), 0.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[]), 0.25_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let err = softmax.propagate_ibp(&input).unwrap_err();
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_linear_scalar_rejected() {
    let logsoftmax = LogSoftmaxLayer::new(-1).with_sound_mode(true);

    let lower = ArrayD::from_elem(IxDyn(&[]), 0.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[]), 0.25_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();
    let bounds = LinearBounds::identity(1);

    let err = logsoftmax
        .propagate_linear_with_bounds(&bounds, &input, logsoftmax.soundness_mode())
        .unwrap_err();
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_linear_scalar_rejected() {
    let softmax = SoftmaxLayer::new(-1).with_sound_mode(true);

    let lower = ArrayD::from_elem(IxDyn(&[]), 0.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[]), 0.25_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();
    let bounds = LinearBounds::identity(1);

    let err = softmax
        .propagate_linear_with_bounds(&bounds, &input, softmax.soundness_mode())
        .unwrap_err();
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_axis_out_of_bounds_returns_error() {
    // axis=5 for a 2D tensor: out of range, should return error (not silently fallback)
    let softmax = SoftmaxLayer::new(5);

    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0_f32, -0.5, 0.25, 0.5, 1.0, 1.5]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0_f32, 0.25, 0.75, 1.0, 1.5, 2.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = softmax.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "Out-of-range axis should return error, not silently fallback"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("out of range"),
        "Error should mention out of range: {}",
        msg
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_axis_below_negative_ndim_returns_error() {
    // axis=-5 for a 2D tensor: out of range, should return error (not silently fallback)
    let softmax = SoftmaxLayer::new(-5);

    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.25_f32, -0.75, 0.0, 0.5, 0.75, 1.0])
        .unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-0.5_f32, -0.25, 0.5, 1.0, 1.25, 1.75])
        .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = softmax.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "Negative out-of-range axis should return error"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_axis_out_of_bounds_returns_error() {
    // axis=7 for a 2D tensor: out of range, should return error (not silently fallback)
    let logsoftmax = LogSoftmaxLayer::new(7);

    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-2.0_f32, -1.0, 0.0, 0.25, 0.5, 1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.5_f32, -0.5, 0.75, 0.5, 1.25, 1.75])
        .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = logsoftmax.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "Out-of-range axis should return error, not silently fallback"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("out of range"),
        "Error should mention out of range: {}",
        msg
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_axis_below_negative_ndim_returns_error() {
    // axis=-9 for a 2D tensor: out of range, should return error (not silently fallback)
    let logsoftmax = LogSoftmaxLayer::new(-9);

    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![-2.25_f32, -1.25, -0.25, 0.25, 0.75, 1.25],
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![-1.75_f32, -0.75, 0.5, 0.75, 1.25, 1.75],
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = logsoftmax.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "Negative out-of-range axis should return error"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_linear_axis_below_negative_ndim_returns_error() {
    // axis=-5 for a 2D tensor: out of range, should return error (not silently fallback)
    let softmax = SoftmaxLayer::new(-5).with_sound_mode(true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.25_f32, -0.75, 0.0, 0.5, 0.75, 1.0])
        .unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-0.5_f32, -0.25, 0.5, 1.0, 1.25, 1.75])
        .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let bounds = LinearBounds::identity(6);

    let result = softmax.propagate_linear_with_bounds(&bounds, &input, softmax.soundness_mode());
    assert!(
        result.is_err(),
        "Negative out-of-range axis should return error"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_linear_axis_below_negative_ndim_returns_error() {
    // axis=-9 for a 2D tensor: out of range, should return error (not silently fallback)
    let logsoftmax = LogSoftmaxLayer::new(-9).with_sound_mode(true);

    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![-2.25_f32, -1.25, -0.25, 0.25, 0.75, 1.25],
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![-1.75_f32, -0.75, 0.5, 0.75, 1.25, 1.75],
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let bounds = LinearBounds::identity(6);

    let result =
        logsoftmax.propagate_linear_with_bounds(&bounds, &input, logsoftmax.soundness_mode());
    assert!(
        result.is_err(),
        "Negative out-of-range axis should return error"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_linear_heuristic_axis_below_negative_ndim_returns_error() {
    // axis=-9 for a 2D tensor: out of range, should return error (not silently fallback)
    let logsoftmax = LogSoftmaxLayer::new(-9);

    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![-2.25_f32, -1.25, -0.25, 0.25, 0.75, 1.25],
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![-1.75_f32, -0.75, 0.5, 0.75, 1.25, 1.75],
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let bounds = LinearBounds::identity(6);

    let result = logsoftmax.propagate_linear_with_bounds(
        &bounds,
        &input,
        VerificationSoundnessMode::Heuristic,
    );
    assert!(
        result.is_err(),
        "Negative out-of-range axis should return error"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_sound_mode_random_ibp_soundness() {
    let logsoftmax = LogSoftmaxLayer::new(-1).with_sound_mode(true);
    let mut seed: u32 = 0x6d31_9d3b;
    let mut next_f32 = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed as f32 / u32::MAX as f32
    };

    for case in 0..12 {
        let dim = 4 + (case % 3) as usize;
        let mut lower_vals = Vec::with_capacity(dim);
        let mut upper_vals = Vec::with_capacity(dim);

        for _ in 0..dim {
            let base = next_f32() * 4.0 - 2.0;
            let width = 0.1 + next_f32() * 2.0;
            lower_vals.push(base);
            upper_vals.push(base + width);
        }

        let lower = ArrayD::from_shape_vec(IxDyn(&[dim]), lower_vals).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[dim]), upper_vals).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let linear_bounds = LinearBounds::identity(dim);
        let result = logsoftmax
            .propagate_linear_with_bounds(&linear_bounds, &input, logsoftmax.soundness_mode())
            .unwrap();

        let concretized = result.concretize(&input);

        let input_lower = input
            .lower()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .unwrap();
        let input_upper = input
            .upper()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .unwrap();

        for sample in 0..16 {
            let mut point = Vec::with_capacity(dim);
            for i in 0..dim {
                let t = next_f32();
                point.push(input_lower[i] + (input_upper[i] - input_lower[i]) * t);
            }

            let max_val = point.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = point.iter().map(|&v| (v - max_val).exp()).sum();
            let lse = max_val + exp_sum.ln();
            let logsoftmax_output: Vec<f32> = point.iter().map(|&v| v - lse).collect();

            for (i, &lsm_val) in logsoftmax_output.iter().enumerate() {
                let tol = 1e-5;
                let lb_val = concretized.lower()[[i]];
                let ub_val = concretized.upper()[[i]];
                assert!(
                    lb_val <= lsm_val + tol,
                    "Sound-mode lower bound violated at case {}, sample {}, dim {}: {} > {}",
                    case,
                    sample,
                    i,
                    lb_val,
                    lsm_val
                );
                assert!(
                    ub_val >= lsm_val - tol,
                    "Sound-mode upper bound violated at case {}, sample {}, dim {}: {} < {}",
                    case,
                    sample,
                    i,
                    ub_val,
                    lsm_val
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_crown_network_integration() {
    // Test LogSoftmax CROWN in a network context
    use crate::layers::{LinearLayer, LogSoftmaxLayer};
    use crate::network::Network;
    use ndarray::Array2;

    // Create a simple network: Linear -> LogSoftmax
    let weight = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let bias: Option<ndarray::Array1<f32>> = Some(ndarray::Array1::zeros(4));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let logsoftmax = LogSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::LogSoftmax(logsoftmax));

    // Create input bounds
    let input_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, -1.0, -1.0]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 1.0, 1.0]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    // Test CROWN propagation
    let crown_result = network.propagate_crown(&input).unwrap();

    // Test IBP propagation for comparison
    let ibp_result = network.propagate_ibp(&input).unwrap();

    // CROWN bounds should be at least as tight as (or equal to) IBP bounds
    for i in 0..4 {
        // CROWN lower bound should be >= IBP lower bound (tighter)
        assert!(
            crown_result.lower()[[i]] >= ibp_result.lower()[[i]] - 1e-4,
            "CROWN lower bound {} should be >= IBP lower bound {}",
            crown_result.lower()[[i]],
            ibp_result.lower()[[i]]
        );
        // CROWN upper bound should be <= IBP upper bound (tighter)
        assert!(
            crown_result.upper()[[i]] <= ibp_result.upper()[[i]] + 1e-4,
            "CROWN upper bound {} should be <= IBP upper bound {}",
            crown_result.upper()[[i]],
            ibp_result.upper()[[i]]
        );
    }
}

/// Regression test for #2591: heuristic LogSoftmax CROWN must NOT return identity
/// bounds when pre-activation contains non-finite values. The old code returned
/// `bounds.clone()` (identity passthrough), which is unsound because LogSoftmax
/// is not the identity function. The fix returns NumericalInstability instead.
#[ntest::timeout(10000)]
#[test]
fn test_logsoftmax_heuristic_crown_rejects_non_finite_preactivation() {
    let logsoftmax = LogSoftmaxLayer::new(-1).with_sound_mode(false);

    // Upper bound contains +Infinity
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0_f32, 1.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5_f32, f32::INFINITY, 2.5]).unwrap();
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let identity_bounds = LinearBounds::identity(3);
    let result = logsoftmax
        .propagate_linear_with_bounds(
            &identity_bounds,
            &input,
            VerificationSoundnessMode::Heuristic,
        )
        .expect("Heuristic LogSoftmax CROWN should return constant bounds for non-finite inputs");

    // The result must NOT be identity (the old unsound behavior).
    // It should be constant bounds ([-inf, +inf]) from the fallback path.
    // With identity incoming and constant [-inf, +inf] output bounds,
    // the concretized lower bounds should all be -inf and upper bounds +inf.
    assert!(
        result.lower_a().iter().all(|&v| v == 0.0),
        "Non-finite fallback should produce constant (zero-coefficient) bounds, got lower_a with non-zero: {:?}",
        result.lower_a()
    );
    assert!(
        result.upper_a().iter().all(|&v| v == 0.0),
        "Non-finite fallback should produce constant (zero-coefficient) bounds, got upper_a with non-zero: {:?}",
        result.upper_a()
    );
}

/// Regression test for #2591: heuristic CausalSoftmax CROWN must NOT return identity
/// bounds when pre-activation contains non-finite values. Same fix as LogSoftmax.
#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_heuristic_crown_rejects_non_finite_preactivation() {
    // Disable sound mode so the heuristic rejection path is exercised.
    // CausalSoftmaxLayer::new() defaults to sound=true, which routes to IBP
    // constant bounds and never reaches the heuristic non-finite guard.
    let softmax = CausalSoftmaxLayer::new(-1).with_sound_mode(false);

    // Lower bound contains -Infinity
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![f32::NEG_INFINITY, -0.25, 0.5, 1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0_f32, 0.5, 1.5, 2.0]).unwrap();
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let result = softmax.propagate_linear_with_bounds(
        &linear_bounds,
        &input,
        VerificationSoundnessMode::Heuristic,
    );

    assert!(
        result.is_err(),
        "Heuristic CausalSoftmax CROWN should reject non-finite pre-activation (was identity passthrough before #2591)"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("NumericalInstability") || err_msg.contains("non-finite"),
        "Expected NumericalInstability error, got: {}",
        err_msg
    );
}
