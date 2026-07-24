// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{array, Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

/// Reference softmax for a 1D slice (for test verification).
fn reference_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// =========================================================================
// Construction and config
// =========================================================================

#[test]
fn new_defaults_to_sound() {
    let layer = CausalSoftmaxLayer::new(-1);
    assert_eq!(layer.axis, -1);
    assert!(layer.sound);
    assert!(layer.window_size.is_none());
    assert_eq!(layer.soundness_mode(), VerificationSoundnessMode::Sound);
}

#[test]
fn with_heuristic_sampling_toggles_sound() {
    let layer = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);
    assert!(!layer.sound);
    assert!(layer.window_size.is_none());
    assert_eq!(layer.soundness_mode(), VerificationSoundnessMode::Heuristic);
    let layer2 = layer.with_sound_mode(true);
    assert!(layer2.sound);
}

#[test]
fn with_window_size_sets_window() {
    let layer = CausalSoftmaxLayer::new(-1).with_window_size(2);
    assert_eq!(layer.window_size, Some(2));
}

// =========================================================================
// eval_row
// =========================================================================

#[test]
fn eval_row_sum_to_one_for_active_positions() {
    let layer = CausalSoftmaxLayer::new(-1);
    let x = array![1.0, 2.0, 3.0, 4.0];
    for row in 0..4 {
        let out = layer.eval_row(&x, row);
        let active_sum: f32 = out.iter().take(row + 1).sum();
        assert!(
            (active_sum - 1.0).abs() < 1e-5,
            "row {}: active sum = {} != 1",
            row,
            active_sum
        );
        // Masked positions should be 0
        for j in (row + 1)..4 {
            assert_eq!(out[j], 0.0, "row {}: masked pos {} != 0", row, j);
        }
    }
}

#[test]
fn eval_row_single_active_position() {
    let layer = CausalSoftmaxLayer::new(-1);
    let x = array![5.0, -3.0, 10.0];
    // Row 0: only position 0 is active → softmax of single element = 1.0
    let out = layer.eval_row(&x, 0);
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert_eq!(out[1], 0.0);
    assert_eq!(out[2], 0.0);
}

#[test]
fn eval_row_matches_reference_softmax() {
    let layer = CausalSoftmaxLayer::new(-1);
    let x = array![0.5, 1.0, -0.5, 2.0];
    // Row 3: all positions active
    let out = layer.eval_row(&x, 3);
    let ref_vals = reference_softmax(&[0.5, 1.0, -0.5, 2.0]);
    for i in 0..4 {
        assert!(
            (out[i] - ref_vals[i]).abs() < 1e-5,
            "row 3, pos {}: {} != {}",
            i,
            out[i],
            ref_vals[i]
        );
    }
    // Row 1: positions 0,1 active → softmax([0.5, 1.0])
    let out1 = layer.eval_row(&x, 1);
    let ref1 = reference_softmax(&[0.5, 1.0]);
    assert!((out1[0] - ref1[0]).abs() < 1e-5);
    assert!((out1[1] - ref1[1]).abs() < 1e-5);
    assert_eq!(out1[2], 0.0);
    assert_eq!(out1[3], 0.0);
}

#[test]
fn eval_row_large_input_stable() {
    let layer = CausalSoftmaxLayer::new(-1);
    let x = array![100.0, 200.0, 300.0];
    let out = layer.eval_row(&x, 2);
    assert!(out.iter().all(|&v| v.is_finite()), "should not overflow");
    assert!(
        (out[2] - 1.0).abs() < 1e-4,
        "dominant element should be ~1.0"
    );
}

#[test]
fn eval_row_sliding_window_masks_prefix() {
    let layer = CausalSoftmaxLayer::new(-1).with_window_size(1);
    let x = array![0.5, 1.0, -0.5, 2.0];
    let out = layer.eval_row(&x, 3);
    let ref_vals = reference_softmax(&[-0.5, 2.0]);

    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 0.0);
    assert!((out[2] - ref_vals[0]).abs() < 1e-5);
    assert!((out[3] - ref_vals[1]).abs() < 1e-5);
}

// =========================================================================
// jacobian_row
// =========================================================================

#[test]
fn jacobian_row_active_rows_sum_to_zero() {
    let layer = CausalSoftmaxLayer::new(-1);
    let x = array![1.0, 2.0, 3.0];
    let jac = layer.jacobian_row(&x, 2);
    // Each column of the active block should sum to 0 (since softmax sums to 1)
    for k in 0..3 {
        let col_sum: f32 = (0..3).map(|j| jac[[j, k]]).sum();
        assert!(col_sum.abs() < 1e-5, "column {} sum = {} != 0", k, col_sum);
    }
}

#[test]
fn jacobian_row_masked_entries_zero() {
    let layer = CausalSoftmaxLayer::new(-1);
    let x = array![1.0, 2.0, 3.0, 4.0];
    // Row 1: only positions 0,1 active
    let jac = layer.jacobian_row(&x, 1);
    // Rows 2,3 and cols 2,3 should be zero
    for j in 2..4 {
        for k in 0..4 {
            assert_eq!(jac[[j, k]], 0.0, "masked row {}, col {} should be 0", j, k);
        }
    }
    for j in 0..4 {
        for k in 2..4 {
            assert_eq!(jac[[j, k]], 0.0, "row {}, masked col {} should be 0", j, k);
        }
    }
}

#[test]
fn jacobian_row_diagonal_positive() {
    let layer = CausalSoftmaxLayer::new(-1);
    let x = array![0.5, 1.5, -0.5];
    let jac = layer.jacobian_row(&x, 2);
    let s = layer.eval_row(&x, 2);
    for i in 0..3 {
        // Diagonal: s_i * (1 - s_i) > 0 for 0 < s_i < 1
        let expected = s[i] * (1.0 - s[i]);
        assert!(
            (jac[[i, i]] - expected).abs() < 1e-5,
            "diagonal [{},{}]: {} != {}",
            i,
            i,
            jac[[i, i]],
            expected
        );
    }
}

// =========================================================================
// IBP propagation (propagate_ibp)
// =========================================================================

#[test]
fn ibp_2d_soundness_vertex_check() {
    let layer = CausalSoftmaxLayer::new(-1);
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![-1.0, -0.5, 0.0, -2.0, -1.0, 0.5, -0.5, 0.0, 1.0],
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![1.0, 0.5, 0.5, 0.0, 1.0, 2.0, 0.5, 1.0, 3.0],
    )
    .unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();

    // Check: for each row i, evaluate causal softmax at vertex points
    // and verify IBP bounds contain them.
    let seq_len = 3;
    for i in 0..seq_len {
        let active = i + 1;
        // Enumerate all 2^active vertices for the active positions
        for mask in 0..(1usize << active) {
            let mut row_input = Array1::<f32>::zeros(seq_len);
            for j in 0..active {
                row_input[j] = if (mask >> j) & 1 == 1 {
                    upper[[i, j]]
                } else {
                    lower[[i, j]]
                };
            }
            let row_output = layer.eval_row(&row_input, i);

            for j in 0..seq_len {
                let lb = result.lower()[[i, j]];
                let ub = result.upper()[[i, j]];
                assert!(
                    lb <= row_output[j] + 1e-4,
                    "IBP lower violated: row={}, col={}, lb={}, actual={}",
                    i,
                    j,
                    lb,
                    row_output[j]
                );
                assert!(
                    ub >= row_output[j] - 1e-4,
                    "IBP upper violated: row={}, col={}, ub={}, actual={}",
                    i,
                    j,
                    ub,
                    row_output[j]
                );
            }
        }
    }
}

#[test]
fn ibp_masked_positions_zero() {
    let layer = CausalSoftmaxLayer::new(-1);
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![-1.0; 9]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![1.0; 9]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();

    // Masked positions should be exactly 0
    for i in 0..3 {
        for j in (i + 1)..3 {
            assert_eq!(
                result.lower()[[i, j]],
                0.0,
                "masked lower [{},{}] should be 0",
                i,
                j
            );
            assert_eq!(
                result.upper()[[i, j]],
                0.0,
                "masked upper [{},{}] should be 0",
                i,
                j
            );
        }
    }
}

#[test]
fn ibp_sliding_window_masks_prefix_and_suffix() {
    let layer = CausalSoftmaxLayer::new(-1).with_window_size(1);
    let lower = ArrayD::from_shape_vec(IxDyn(&[4, 4]), vec![-1.0; 16]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[4, 4]), vec![1.0; 16]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();

    for j in 1..4 {
        assert_eq!(result.lower()[[0, j]], 0.0);
        assert_eq!(result.upper()[[0, j]], 0.0);
    }
    assert_eq!(result.lower()[[3, 0]], 0.0);
    assert_eq!(result.upper()[[3, 0]], 0.0);
    assert_eq!(result.lower()[[3, 1]], 0.0);
    assert_eq!(result.upper()[[3, 1]], 0.0);
}

#[test]
fn ibp_sliding_window_tighter_than_full_causal() {
    let full = CausalSoftmaxLayer::new(-1);
    let windowed = CausalSoftmaxLayer::new(-1).with_window_size(1);
    let lower = ArrayD::from_shape_vec(IxDyn(&[4, 4]), vec![-1.0; 16]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[4, 4]), vec![1.0; 16]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let full_bounds = full.propagate_ibp(&input).unwrap();
    let windowed_bounds = windowed.propagate_ibp(&input).unwrap();

    let full_total_width: f32 = full_bounds
        .lower()
        .iter()
        .zip(full_bounds.upper().iter())
        .map(|(l, u)| u - l)
        .sum();
    let windowed_total_width: f32 = windowed_bounds
        .lower()
        .iter()
        .zip(windowed_bounds.upper().iter())
        .map(|(l, u)| u - l)
        .sum();

    assert!(
        windowed_total_width < full_total_width,
        "sliding window should tighten total width: {windowed_total_width} !< {full_total_width}",
    );
}

#[test]
fn ibp_3d_batched() {
    let layer = CausalSoftmaxLayer::new(-1);
    // [batch=2, seq_q=2, seq_k=2]
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 2]),
        vec![-1.0, -0.5, -0.5, 0.0, 0.0, -1.0, -1.0, -0.5],
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 2]),
        vec![1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.5],
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    // Basic sanity: outputs should be in [0, 1]
    for &v in result.lower().iter() {
        assert!((0.0..=1.0).contains(&v), "lower {} not in [0,1]", v);
    }
    for &v in result.upper().iter() {
        assert!((0.0..=1.0).contains(&v), "upper {} not in [0,1]", v);
    }
    // lower <= upper
    for (&l, &u) in result.lower().iter().zip(result.upper().iter()) {
        assert!(l <= u + 1e-6, "lower {} > upper {}", l, u);
    }
}

#[test]
fn ibp_point_interval_matches_eval() {
    let layer = CausalSoftmaxLayer::new(-1);
    let vals = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![0.5, 1.0, -0.5, 2.0, -1.0, 0.0, 0.0, 0.5, 1.5],
    )
    .unwrap();
    let input = BoundedTensor::new(vals.clone(), vals.clone()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();

    // For point intervals, IBP bounds should be tight (≈ exact eval)
    for i in 0..3 {
        let row_input: Array1<f32> = (0..3).map(|j| vals[[i, j]]).collect();
        let expected = layer.eval_row(&row_input, i);
        for j in 0..3 {
            assert!(
                (result.lower()[[i, j]] - expected[j]).abs() < 0.01,
                "point interval lower [{},{}]: {} != {}",
                i,
                j,
                result.lower()[[i, j]],
                expected[j]
            );
            assert!(
                (result.upper()[[i, j]] - expected[j]).abs() < 0.01,
                "point interval upper [{},{}]: {} != {}",
                i,
                j,
                result.upper()[[i, j]],
                expected[j]
            );
        }
    }
}

// =========================================================================
// Error cases
// =========================================================================

#[test]
fn ibp_rejects_1d_input() {
    let layer = CausalSoftmaxLayer::new(-1);
    let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0; 4]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let err = layer.propagate_ibp(&input).unwrap_err();
    assert!(
        format!("{}", err).contains("at least 2D"),
        "expected 2D error, got: {}",
        err
    );
}

#[test]
fn ibp_rejects_seq_q_greater_than_seq_k() {
    let layer = CausalSoftmaxLayer::new(-1);
    // seq_q=3 > seq_k=2
    let lower = ArrayD::zeros(IxDyn(&[3, 2]));
    let upper = ArrayD::ones(IxDyn(&[3, 2]));
    let input = BoundedTensor::new(lower, upper).unwrap();
    let err = layer.propagate_ibp(&input).unwrap_err();
    assert!(
        format!("{}", err).contains("seq_q"),
        "expected seq_q error, got: {}",
        err
    );
}

#[test]
fn propagate_linear_rejects_without_preact() {
    let layer = CausalSoftmaxLayer::new(-1);
    let bounds = LinearBounds::identity(4);
    let err = layer.propagate_linear(&bounds).unwrap_err();
    assert!(
        format!("{}", err).contains("nonlinear"),
        "expected nonlinear error, got: {}",
        err
    );
}

#[test]
fn requires_pre_activation_bounds_is_true() {
    let layer = CausalSoftmaxLayer::new(-1);
    assert!(layer.requires_pre_activation_bounds());
}

// =========================================================================
// CROWN backward (propagate_crown_backward / propagate_linear_with_bounds)
// =========================================================================

#[test]
fn crown_sound_mode_returns_ibp_constant_bounds() {
    let layer = CausalSoftmaxLayer::new(-1); // sound=true by default
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, 0.0, -0.5, 0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 1.0, 0.5, 1.5]).unwrap();
    let pre = BoundedTensor::new(lower, upper).unwrap();
    let bounds = LinearBounds::identity(4); // 2x2 = 4 flattened

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap();

    // Sound mode uses IBP → constant bounds → slopes should be zero
    assert!(
        result.lower_a.iter().all(|&v| v.abs() < 1e-6),
        "sound mode: lower_a should be zero slopes"
    );
    assert!(
        result.upper_a.iter().all(|&v| v.abs() < 1e-6),
        "sound mode: upper_a should be zero slopes"
    );
    // Biases should contain IBP-derived bounds
    for i in 0..4 {
        assert!(
            result.lower_b[i] <= result.upper_b[i] + 1e-6,
            "lower_b[{}] > upper_b[{}]: {} > {}",
            i,
            i,
            result.lower_b[i],
            result.upper_b[i]
        );
    }
}

#[test]
fn crown_heuristic_bounds_contain_samples() {
    let layer = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5, 0.0, -1.0, 0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 1.0, 0.0, 1.5]).unwrap();
    let pre = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let bounds = LinearBounds::identity(4);

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Heuristic)
        .unwrap();

    // Concretize bounds and check they contain samples
    let pre_lower_flat: Vec<f32> = lower.iter().cloned().collect();
    let pre_upper_flat: Vec<f32> = upper.iter().cloned().collect();

    // Check center point
    let center: Vec<f32> = pre_lower_flat
        .iter()
        .zip(pre_upper_flat.iter())
        .map(|(l, u)| (l + u) / 2.0)
        .collect();
    let center_arr = Array1::from_vec(center);

    for i in 0..4 {
        let lb = result.lower_a.row(i).dot(&center_arr) + result.lower_b[i];
        let ub = result.upper_a.row(i).dot(&center_arr) + result.upper_b[i];
        assert!(
            lb <= ub + 1e-6,
            "center: lb[{}]={} > ub[{}]={}",
            i,
            lb,
            i,
            ub
        );
    }
}

#[test]
fn crown_backward_shape_mismatch_error() {
    let layer = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0; 6]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
    let pre = BoundedTensor::new(lower, upper).unwrap();
    // bounds expects 4 inputs but pre has 6
    let bounds = LinearBounds::identity(4);

    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Heuristic)
        .unwrap_err();
    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got: {:?}",
        err
    );
}

#[test]
fn crown_infinite_preact_returns_numerical_instability() {
    // Since commit 6fe9d09 (Part of #2423 / #2591): non-finite pre-activation
    // bounds correctly return NumericalInstability instead of unsound identity
    // passthrough. CausalSoftmax is NOT the identity function, so returning
    // identity bounds for non-finite inputs was unsound.
    let layer = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![f32::NEG_INFINITY, 0.0, -1.0, 0.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![f32::INFINITY, 1.0, 1.0, 1.0]).unwrap();
    let pre = BoundedTensor::new_unchecked(lower, upper).unwrap();
    let bounds = LinearBounds::identity(4);

    let result =
        layer.propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Heuristic);

    assert!(result.is_err(), "non-finite preact should return error");
    let err = result.unwrap_err();
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability, got: {:?}",
        err
    );
}

#[test]
fn ibp_4d_basic() {
    let layer = CausalSoftmaxLayer::new(-1);
    // [batch=1, heads=1, seq_q=2, seq_k=2]
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![-1.0, 0.0, -0.5, 0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 1.0, 0.5, 1.5]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();

    // All outputs should be in [0, 1]
    for &v in result.lower().iter() {
        assert!((0.0..=1.0).contains(&v), "lower {} not in [0,1]", v);
    }
    for &v in result.upper().iter() {
        assert!((0.0..=1.0).contains(&v), "upper {} not in [0,1]", v);
    }
    // Masked positions should be 0
    // Row 0 (seq_q=0): only position 0 active, position 1 masked
    assert_eq!(result.lower()[[0, 0, 0, 1]], 0.0);
    assert_eq!(result.upper()[[0, 0, 0, 1]], 0.0);
}

#[test]
fn ibp_5d_rejected() {
    let layer = CausalSoftmaxLayer::new(-1);
    let lower = ArrayD::zeros(IxDyn(&[1, 1, 1, 2, 2]));
    let upper = ArrayD::ones(IxDyn(&[1, 1, 1, 2, 2]));
    let input = BoundedTensor::new(lower, upper).unwrap();
    let err = layer.propagate_ibp(&input).unwrap_err();
    assert!(
        format!("{}", err).contains("5D"),
        "expected 5D error, got: {}",
        err
    );
}

#[test]
fn ibp_non_finite_input_falls_back_to_trivial() {
    let layer = CausalSoftmaxLayer::new(-1);
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![f32::NEG_INFINITY, 0.0, -1.0, 0.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, f32::INFINITY, 1.0, 1.0]).unwrap();
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();

    // Row 0: has non-finite → fallback [0, 1] for active positions
    assert_eq!(result.lower()[[0, 0]], 0.0);
    assert_eq!(result.upper()[[0, 0]], 1.0);
}

// =========================================================================
// CROWN sound — vertex soundness (IBP-backed constant bounds)
// =========================================================================

#[test]
fn crown_sound_bounds_contain_all_vertices() {
    // Sound mode falls back to IBP constant bounds. These MUST contain
    // true causal softmax output at all vertices.
    let layer = CausalSoftmaxLayer::new(-1); // sound=true by default
    let lower_vals = vec![-1.0, -0.5, -0.5, 0.0, -1.0, 0.5, -0.3, 0.0, 0.5];
    let upper_vals = vec![1.0, 0.5, 0.5, 0.5, 1.0, 2.0, 0.3, 1.0, 2.5];
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 3]), lower_vals.clone()).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 3]), upper_vals.clone()).unwrap();
    let pre = BoundedTensor::new(lower, upper).unwrap();
    let total = 9; // 3*3
    let bounds = LinearBounds::identity(total);

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap();

    // Sound mode produces constant bounds (zero slopes)
    assert!(
        result.lower_a.iter().all(|&v| v.abs() < 1e-6),
        "sound mode should have zero slopes"
    );

    // Check that the constant bounds contain causal softmax at all vertices
    // for each row independently.
    let seq_len = 3;
    for row_i in 0..seq_len {
        let active = row_i + 1;
        for mask in 0..(1usize << active) {
            let mut row_input = Array1::<f32>::zeros(seq_len);
            for j in 0..active {
                row_input[j] = if (mask >> j) & 1 == 1 {
                    upper_vals[row_i * seq_len + j]
                } else {
                    lower_vals[row_i * seq_len + j]
                };
            }
            let row_output = layer.eval_row(&row_input, row_i);

            for j in 0..seq_len {
                let flat_idx = row_i * seq_len + j;
                let lb = result.lower_b[flat_idx];
                let ub = result.upper_b[flat_idx];
                assert!(
                    lb <= row_output[j] + 1e-4,
                    "sound vertex: row={}, col={}, mask={}: lb={} > actual={}",
                    row_i,
                    j,
                    mask,
                    lb,
                    row_output[j]
                );
                assert!(
                    ub >= row_output[j] - 1e-4,
                    "sound vertex: row={}, col={}, mask={}: ub={} < actual={}",
                    row_i,
                    j,
                    mask,
                    ub,
                    row_output[j]
                );
            }
        }
    }
}

#[test]
fn crown_heuristic_bounds_contain_center_and_corners() {
    // Heuristic mode uses sampling-based linearization — not provably sound,
    // but should at least contain center and corner evaluations.
    let layer = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let lower_vals = vec![-0.5, 0.0, -1.0, 0.5];
    let upper_vals = vec![0.5, 1.0, 0.0, 1.5];
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), lower_vals.clone()).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), upper_vals.clone()).unwrap();
    let pre = BoundedTensor::new(lower, upper).unwrap();
    let total = 4;
    let bounds = LinearBounds::identity(total);

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Heuristic)
        .unwrap();

    // Test at center and all 2^4 = 16 vertices
    let pre_lower_flat: Vec<f32> = lower_vals;
    let pre_upper_flat: Vec<f32> = upper_vals;

    let mut test_points = Vec::new();
    // Center
    let center: Vec<f32> = pre_lower_flat
        .iter()
        .zip(pre_upper_flat.iter())
        .map(|(l, u)| (l + u) / 2.0)
        .collect();
    test_points.push(center);
    // All vertices
    for mask in 0..(1usize << total) {
        let point: Vec<f32> = (0..total)
            .map(|i| {
                if (mask >> i) & 1 == 1 {
                    pre_upper_flat[i]
                } else {
                    pre_lower_flat[i]
                }
            })
            .collect();
        test_points.push(point);
    }

    for point in &test_points {
        assert_heuristic_containment(&layer, &result, point, total, 2);
    }
}

/// Check that heuristic linear bounds both order correctly (lb <= ub)
/// and contain the true causal softmax output at a given test point.
fn assert_heuristic_containment(
    layer: &CausalSoftmaxLayer,
    result: &LinearBounds,
    point: &[f32],
    total: usize,
    seq_len: usize,
) {
    let sample = Array1::from_vec(point.to_vec());

    // Compute true causal softmax output at this point.
    let mut true_output = vec![0.0_f32; total];
    for row_i in 0..seq_len {
        let row_input = Array1::from_vec(point[row_i * seq_len..(row_i + 1) * seq_len].to_vec());
        let row_out = layer.eval_row(&row_input, row_i);
        for j in 0..seq_len {
            true_output[row_i * seq_len + j] = row_out[j];
        }
    }

    for (i, &true_value) in true_output.iter().enumerate().take(total) {
        let lb = result.lower_a.row(i).dot(&sample) + result.lower_b[i];
        let ub = result.upper_a.row(i).dot(&sample) + result.upper_b[i];
        assert!(
            lb <= ub + 1e-2,
            "heuristic ordering: point {:?}, dim {}: lb={} > ub={}",
            point,
            i,
            lb,
            ub
        );
        // Containment: heuristic bounds should contain the true output.
        // Generous tolerance since heuristic mode is sampling-based.
        let tol = 0.05;
        assert!(
            lb <= true_value + tol,
            "heuristic containment: point {:?}, dim {}: lb={} > true={}",
            point,
            i,
            lb,
            true_value,
        );
        assert!(
            ub >= true_value - tol,
            "heuristic containment: point {:?}, dim {}: ub={} < true={}",
            point,
            i,
            ub,
            true_value,
        );
    }
}

// =========================================================================
// IBP monotonicity: tighter input bounds → tighter output bounds
// =========================================================================

#[test]
fn ibp_monotonicity_tighter_input_gives_tighter_output() {
    let layer = CausalSoftmaxLayer::new(-1);

    // Wide bounds
    let lower_wide = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-2.0, -2.0, -2.0, -2.0]).unwrap();
    let upper_wide = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 2.0, 2.0, 2.0]).unwrap();
    let input_wide = BoundedTensor::new(lower_wide, upper_wide).unwrap();
    let result_wide = layer.propagate_ibp(&input_wide).unwrap();

    // Narrow bounds (subset of wide)
    let lower_narrow =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5, -0.5, -0.5, -0.5]).unwrap();
    let upper_narrow = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 0.5, 0.5, 0.5]).unwrap();
    let input_narrow = BoundedTensor::new(lower_narrow, upper_narrow).unwrap();
    let result_narrow = layer.propagate_ibp(&input_narrow).unwrap();

    // Narrow bounds should produce tighter or equal output bounds:
    // narrow.lower >= wide.lower AND narrow.upper <= wide.upper
    for (&nl, &wl) in result_narrow.lower().iter().zip(result_wide.lower().iter()) {
        assert!(
            nl >= wl - 1e-5,
            "monotonicity violated: narrow lower {} < wide lower {}",
            nl,
            wl
        );
    }
    for (&nu, &wu) in result_narrow.upper().iter().zip(result_wide.upper().iter()) {
        assert!(
            nu <= wu + 1e-5,
            "monotonicity violated: narrow upper {} > wide upper {}",
            nu,
            wu
        );
    }
}
