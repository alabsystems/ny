// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::LogSoftmaxLayer;
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;
use ndarray::{Array, ArrayD, IxDyn};
use ny_core::VerificationSoundnessMode;
use ny_tensor::BoundedTensor;

// ========== Helper functions ==========

fn make_bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let n = lower.len();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

/// Reference logsoftmax in f64 for high-precision comparison.
fn reference_logsoftmax(x: &[f32]) -> Vec<f32> {
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let max = x64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse: f64 = max + x64.iter().map(|&v| (v - max).exp()).sum::<f64>().ln();
    x64.iter().map(|&v| (v - lse) as f32).collect()
}

// ========== IBP soundness tests ==========

#[test]
fn logsoftmax_ibp_soundness_basic() {
    let layer = LogSoftmaxLayer::new(-1);
    let lower = vec![0.0, 1.0, 2.0];
    let upper = vec![1.0, 2.0, 3.0];
    let input = make_bt(&lower, &upper);
    let output = layer.propagate_ibp(&input).unwrap();

    // Test vertex corners
    for mask in 0..8u32 {
        let x: Vec<f32> = (0..3)
            .map(|j| {
                if (mask >> j) & 1 == 0 {
                    lower[j]
                } else {
                    upper[j]
                }
            })
            .collect();
        let ls = reference_logsoftmax(&x);
        for (i, &lsi) in ls.iter().enumerate() {
            assert!(
                output.lower()[[i]] <= lsi + 1e-4,
                "IBP lower[{}] = {} > logsoftmax = {} for x={:?}",
                i,
                output.lower()[[i]],
                lsi,
                x
            );
            assert!(
                output.upper()[[i]] >= lsi - 1e-4,
                "IBP upper[{}] = {} < logsoftmax = {} for x={:?}",
                i,
                output.upper()[[i]],
                lsi,
                x
            );
        }
    }
}

#[test]
fn logsoftmax_ibp_soundness_negative_inputs() {
    let layer = LogSoftmaxLayer::new(-1);
    let lower = vec![-5.0, -3.0, -1.0];
    let upper = vec![-2.0, -1.0, 0.0];
    let input = make_bt(&lower, &upper);
    let output = layer.propagate_ibp(&input).unwrap();

    for mask in 0..8u32 {
        let x: Vec<f32> = (0..3)
            .map(|j| {
                if (mask >> j) & 1 == 0 {
                    lower[j]
                } else {
                    upper[j]
                }
            })
            .collect();
        let ls = reference_logsoftmax(&x);
        for (i, &lsi) in ls.iter().enumerate() {
            assert!(
                output.lower()[[i]] <= lsi + 1e-4,
                "lower[{}]={} > actual={}",
                i,
                output.lower()[[i]],
                lsi
            );
            assert!(
                output.upper()[[i]] >= lsi - 1e-4,
                "upper[{}]={} < actual={}",
                i,
                output.upper()[[i]],
                lsi
            );
        }
    }
}

#[test]
fn logsoftmax_ibp_lower_bounds_always_nonpositive() {
    // logsoftmax_i = x_i - logsumexp(x). Since logsumexp(x) >= max(x) >= x_i,
    // logsoftmax_i <= 0 always. The IBP lower bound (lower_i - lse_upper) is
    // guaranteed <= 0 because lse_upper >= max(upper) >= upper_i >= lower_i.
    // The upper bound (upper_i - lse_lower) may exceed 0 due to over-approximation.
    let layer = LogSoftmaxLayer::new(-1);
    let input = make_bt(&[-5.0, 0.0, 5.0], &[0.0, 5.0, 10.0]);
    let output = layer.propagate_ibp(&input).unwrap();

    for &v in output.lower().iter() {
        assert!(v <= 1e-5, "logsoftmax lower {} should be <= 0", v);
    }
}

#[test]
fn logsoftmax_ibp_point_interval_tight() {
    let layer = LogSoftmaxLayer::new(-1);
    let x = vec![1.0, 2.0, 3.0];
    let input = make_bt(&x, &x);
    let output = layer.propagate_ibp(&input).unwrap();
    let ls = reference_logsoftmax(&x);

    for (i, &lsi) in ls.iter().enumerate() {
        assert!(
            output.lower()[[i]] <= lsi + 1e-3,
            "point lower[{}]={} > actual={}",
            i,
            output.lower()[[i]],
            lsi
        );
        assert!(
            output.upper()[[i]] >= lsi - 1e-3,
            "point upper[{}]={} < actual={}",
            i,
            output.upper()[[i]],
            lsi
        );
    }
}

#[test]
fn logsoftmax_ibp_rejects_0d() {
    let layer = LogSoftmaxLayer::new(-1);
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap(),
    )
    .unwrap();
    assert!(layer.propagate_ibp(&input).is_err());
}

#[test]
fn logsoftmax_ibp_nonfinite_fallback() {
    let layer = LogSoftmaxLayer::new(-1);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    assert_eq!(output.shape(), &[3]);
    // Verify lower <= upper (soundness: non-inverted intervals)
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(
            l <= u || (l.is_nan() && u.is_nan()),
            "non-finite fallback produced inverted interval: lower {l} > upper {u}"
        );
    }
    // Non-finite input triggers the conservative fallback: [-inf, +inf] per
    // element. new_repaired preserves the ±Inf endpoints — a non-finite
    // endpoint proves nothing, so any finite substitute would be an unsound
    // tightening (#3423).
    assert!(
        output.lower().iter().all(|&v| v == f32::NEG_INFINITY),
        "fallback lower should be -inf, got {:?}",
        output.lower()
    );
    assert!(
        output.upper().iter().all(|&v| v == f32::INFINITY),
        "fallback upper should be +inf, got {:?}",
        output.upper()
    );
}

/// Regression test for #2713: all-NEG_INFINITY inputs must NOT produce NaN.
/// The input guard returns conservative ±inf bounds (via new_repaired),
/// which prevents NaN from propagating through ln(sum_exp).
#[test]
fn logsoftmax_ibp_all_neg_infinity_no_nan_2713() {
    let layer = LogSoftmaxLayer::new(-1);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY; 3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY; 3]).unwrap(),
    )
    .unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    assert_eq!(output.shape(), &[3]);
    // Must return NaN-free bounds — the fallback widens to ±inf, which
    // new_repaired preserves (no finite substitute is sound).
    assert!(
        output.lower().iter().all(|&v| v == f32::NEG_INFINITY),
        "LogSoftmax IBP fallback lower for all-NEG_INFINITY should be -inf: {:?}",
        output.lower()
    );
    assert!(
        output.upper().iter().all(|&v| v == f32::INFINITY),
        "LogSoftmax IBP fallback upper for all-NEG_INFINITY should be +inf: {:?}",
        output.upper()
    );
    // Soundness: lower <= upper
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(l <= u, "inverted bounds: lower {l} > upper {u}");
    }
}

// ========== NaN integration tests (#2627) ==========
// LogSoftmax IBP has a non-finite input guard at logsoftmax/mod.rs:118-126:
// any NaN/Inf in input → conservative bounds via new_repaired. These tests
// verify the fold site (nan_propagating_max in logsumexp_directed) integrates
// correctly with the guard.

/// NaN in lower bound triggers conservative fallback (#2627).
#[test]
fn logsoftmax_ibp_nan_in_lower_fallback_2627() {
    let layer = LogSoftmaxLayer::new(-1);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let output = layer
        .propagate_ibp(&input)
        .expect("NaN lower should trigger fallback, not error");
    assert_eq!(output.shape(), &[3]);
    // Repaired bounds are the conservative ±inf fallback — never NaN, and
    // never a fabricated finite clamp.
    assert!(
        output.lower().iter().all(|&v| v == f32::NEG_INFINITY),
        "NaN fallback lower should be -inf, got {:?}",
        output.lower()
    );
    assert!(
        output.upper().iter().all(|&v| v == f32::INFINITY),
        "NaN fallback upper should be +inf, got {:?}",
        output.upper()
    );
    // Soundness: lower <= upper
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(l <= u, "inverted: lower {l} > upper {u}");
    }
}

/// NaN in upper bound triggers conservative fallback (#2627).
#[test]
fn logsoftmax_ibp_nan_in_upper_fallback_2627() {
    let layer = LogSoftmaxLayer::new(-1);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, f32::NAN, 3.0]).unwrap(),
    )
    .unwrap();
    let output = layer
        .propagate_ibp(&input)
        .expect("NaN upper should trigger fallback");
    assert!(
        output.lower().iter().all(|&v| v == f32::NEG_INFINITY),
        "NaN upper fallback lower should be -inf"
    );
    assert!(
        output.upper().iter().all(|&v| v == f32::INFINITY),
        "NaN upper fallback upper should be +inf"
    );
}

/// All-NaN input triggers conservative fallback with no NaN in output (#2627).
#[test]
fn logsoftmax_ibp_all_nan_fallback_2627() {
    let layer = LogSoftmaxLayer::new(-1);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN; 3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN; 3]).unwrap(),
    )
    .unwrap();
    let output = layer
        .propagate_ibp(&input)
        .expect("all-NaN should trigger fallback");
    assert!(
        output.lower().iter().all(|&v| v == f32::NEG_INFINITY),
        "all-NaN fallback lower should be -inf"
    );
    assert!(
        output.upper().iter().all(|&v| v == f32::INFINITY),
        "all-NaN fallback upper should be +inf"
    );
    // No NaN in output
    assert!(
        !output.lower().iter().any(|v| v.is_nan()),
        "all-NaN fallback should not produce NaN lower"
    );
    assert!(
        !output.upper().iter().any(|v| v.is_nan()),
        "all-NaN fallback should not produce NaN upper"
    );
}

#[test]
fn logsoftmax_ibp_2d_soundness() {
    let layer = LogSoftmaxLayer::new(-1);
    let lower_data = vec![0.0f32, 1.0, 2.0, -1.0, 0.0, 1.0];
    let upper_data = vec![1.0f32, 2.0, 3.0, 0.0, 1.0, 2.0];
    let lower = Array::from_shape_vec((2, 3), lower_data.clone())
        .unwrap()
        .into_dyn();
    let upper = Array::from_shape_vec((2, 3), upper_data.clone())
        .unwrap()
        .into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let output = layer.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    // Lower bounds should be <= 0 (logsoftmax is always <= 0)
    for &v in output.lower().iter() {
        assert!(v <= 1e-4, "2D logsoftmax lower {} should be <= 0", v);
    }
    // lower <= upper everywhere
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(*l <= u + 1e-6, "lower {} > upper {}", l, u);
    }
    // Vertex containment: check that concrete logsoftmax at each corner is within bounds.
    // logsoftmax applied along axis -1, so enumerate corners per row (2 rows, 3 cols each).
    for row in 0..2usize {
        let row_lower = &lower_data[row * 3..(row + 1) * 3];
        let row_upper = &upper_data[row * 3..(row + 1) * 3];
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
            let ls = reference_logsoftmax(&x);
            for (col, &lsi) in ls.iter().enumerate() {
                assert!(
                    output.lower()[[row, col]] <= lsi + 1e-4,
                    "row {row} col {col}: lower {} > actual {lsi}",
                    output.lower()[[row, col]]
                );
                assert!(
                    output.upper()[[row, col]] >= lsi - 1e-4,
                    "row {row} col {col}: upper {} < actual {lsi}",
                    output.upper()[[row, col]]
                );
            }
        }
    }
}

// ========== propagate_linear error path ==========

#[test]
fn logsoftmax_propagate_linear_returns_unsupported() {
    let layer = LogSoftmaxLayer::new(-1);
    let bounds = LinearBounds::identity(3);
    let result = layer.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "propagate_linear should return UnsupportedOp"
    );
}

// ========== CROWN backward tests ==========

#[test]
fn logsoftmax_crown_respects_axis_groups() {
    let layer = LogSoftmaxLayer::new(1);
    let lower = Array::from_shape_vec((2, 3), vec![-1.0, -0.5, 0.0, -2.0, 1.0, 2.0])
        .unwrap()
        .into_dyn();
    let upper = Array::from_shape_vec((2, 3), vec![0.5, 0.8, 1.2, -1.0, 1.5, 3.0])
        .unwrap()
        .into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let bounds = LinearBounds::identity(6);

    let linear = layer
        .propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Sound)
        .unwrap();

    for out_idx in 0..6 {
        for in_idx in 0..6 {
            if out_idx / 3 != in_idx / 3 {
                let lower_coeff = linear.lower_a[[out_idx, in_idx]];
                let upper_coeff = linear.upper_a[[out_idx, in_idx]];
                assert!(
                    lower_coeff.abs() < 1e-6,
                    "lower_a[{out_idx},{in_idx}]={lower_coeff}"
                );
                assert!(
                    upper_coeff.abs() < 1e-6,
                    "upper_a[{out_idx},{in_idx}]={upper_coeff}"
                );
            }
        }
    }
}

#[test]
fn logsoftmax_crown_respects_axis_groups_axis0() {
    let layer = LogSoftmaxLayer::new(0);
    let lower = Array::from_shape_vec((2, 3), vec![-1.0, -0.5, 0.0, -2.0, 1.0, 2.0])
        .unwrap()
        .into_dyn();
    let upper = Array::from_shape_vec((2, 3), vec![0.5, 0.8, 1.2, -1.0, 1.5, 3.0])
        .unwrap()
        .into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let bounds = LinearBounds::identity(6);

    let linear = layer
        .propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Sound)
        .unwrap();

    for out_idx in 0..6 {
        for in_idx in 0..6 {
            let out_col = out_idx % 3;
            let in_col = in_idx % 3;
            if out_col != in_col {
                let lower_coeff = linear.lower_a[[out_idx, in_idx]];
                let upper_coeff = linear.upper_a[[out_idx, in_idx]];
                assert!(
                    lower_coeff.abs() < 1e-6,
                    "lower_a[{out_idx},{in_idx}]={lower_coeff}"
                );
                assert!(
                    upper_coeff.abs() < 1e-6,
                    "upper_a[{out_idx},{in_idx}]={upper_coeff}"
                );
            }
        }
    }
}

#[test]
fn logsoftmax_crown_nd_empty_dim_preserves_bias_and_stays_finite() {
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 0, 3]), vec![]).expect("shape [2,0,3]"),
        ArrayD::from_shape_vec(IxDyn(&[2, 0, 3]), vec![]).expect("shape [2,0,3]"),
    )
    .expect("bounded tensor");
    let bounds = LinearBounds::new(
        Array::zeros((2, 0)),
        Array::from_vec(vec![-0.5, 1.75]),
        Array::zeros((2, 0)),
        Array::from_vec(vec![0.25, 2.0]),
    )
    .unwrap();

    let sound_layer = LogSoftmaxLayer::new(2);
    let sound_result = sound_layer
        .propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Sound)
        .expect("sound propagate N-D empty dim");
    assert_eq!(sound_result.lower_a.shape(), &[2, 0]);
    assert_eq!(sound_result.upper_a.shape(), &[2, 0]);
    assert_eq!(sound_result.lower_b, bounds.lower_b);
    assert_eq!(sound_result.upper_b, bounds.upper_b);
    assert!(sound_result.lower_b.iter().all(|v| v.is_finite()));
    assert!(sound_result.upper_b.iter().all(|v| v.is_finite()));

    let heuristic_layer = LogSoftmaxLayer::new(2).with_sound_mode(false);
    let heuristic_result = heuristic_layer
        .propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Heuristic)
        .expect("heuristic propagate N-D empty dim");
    assert_eq!(heuristic_result.lower_a.shape(), &[2, 0]);
    assert_eq!(heuristic_result.upper_a.shape(), &[2, 0]);
    assert_eq!(heuristic_result.lower_b, bounds.lower_b);
    assert_eq!(heuristic_result.upper_b, bounds.upper_b);
    assert!(heuristic_result.lower_b.iter().all(|v| v.is_finite()));
    assert!(heuristic_result.upper_b.iter().all(|v| v.is_finite()));
}

#[test]
fn logsoftmax_crown_sound_1d_bounds_contain_true_output() {
    // CROWN with sound mode: bounds should contain true logsoftmax output
    let layer = LogSoftmaxLayer::new(-1);
    let lower_v = vec![0.0, 1.0, 2.0];
    let upper_v = vec![1.0, 2.0, 3.0];
    let input = make_bt(&lower_v, &upper_v);
    let bounds = LinearBounds::identity(3);

    let linear = layer
        .propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Sound)
        .unwrap();

    // Concretize and check against reference at vertices
    for mask in 0..8u32 {
        let x: Vec<f32> = (0..3)
            .map(|j| {
                if (mask >> j) & 1 == 0 {
                    lower_v[j]
                } else {
                    upper_v[j]
                }
            })
            .collect();
        let ls = reference_logsoftmax(&x);

        // Concretize linear bounds at point x:
        // lower_bound[i] = sum_j(lower_a[i,j] * x[j]) + lower_b[i]  (if lower_a[i,j] >= 0)
        // but more precisely, concretize over interval using IBP on the linear form.
        // For vertex x, linear evaluation is exact:
        for (i, &lsi) in ls.iter().enumerate() {
            let lb: f32 =
                (0..3).map(|j| linear.lower_a[[i, j]] * x[j]).sum::<f32>() + linear.lower_b[i];
            let ub: f32 =
                (0..3).map(|j| linear.upper_a[[i, j]] * x[j]).sum::<f32>() + linear.upper_b[i];

            assert!(
                lb <= lsi + 1e-3,
                "CROWN lower[{}]={} > actual={} at x={:?}",
                i,
                lb,
                lsi,
                x
            );
            assert!(
                ub >= lsi - 1e-3,
                "CROWN upper[{}]={} < actual={} at x={:?}",
                i,
                ub,
                lsi,
                x
            );
        }
    }
}

#[test]
fn logsoftmax_crown_heuristic_bounds_contain_true_output() {
    let layer = LogSoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let lower_v = vec![0.0, 1.0, 2.0];
    let upper_v = vec![1.0, 2.0, 3.0];
    let input = make_bt(&lower_v, &upper_v);
    let bounds = LinearBounds::identity(3);

    let linear = layer
        .propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Heuristic)
        .unwrap();

    // Check at midpoint
    let mid: Vec<f32> = (0..3)
        .map(|j| f32::midpoint(lower_v[j], upper_v[j]))
        .collect();
    let ls = reference_logsoftmax(&mid);

    for (i, &lsi) in ls.iter().enumerate() {
        let lb: f32 =
            (0..3).map(|j| linear.lower_a[[i, j]] * mid[j]).sum::<f32>() + linear.lower_b[i];
        let ub: f32 =
            (0..3).map(|j| linear.upper_a[[i, j]] * mid[j]).sum::<f32>() + linear.upper_b[i];

        assert!(
            lb <= lsi + 1e-2,
            "Heuristic lower[{}]={} > actual={} at midpoint",
            i,
            lb,
            lsi
        );
        assert!(
            ub >= lsi - 1e-2,
            "Heuristic upper[{}]={} < actual={} at midpoint",
            i,
            ub,
            lsi
        );
    }
}

#[test]
fn logsoftmax_crown_shape_mismatch_errors() {
    let layer = LogSoftmaxLayer::new(-1);
    let input = make_bt(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);
    // Bounds for 4 inputs but pre-activation is 3
    let bounds = LinearBounds::identity(4);
    let result =
        layer.propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Sound);
    assert!(result.is_err(), "Should error on shape mismatch");
}

#[test]
fn logsoftmax_crown_nonfinite_preact_returns_constant() {
    let layer = LogSoftmaxLayer::new(-1);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(3);
    let result =
        layer.propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Sound);
    // Should not panic — falls back to constant bounds
    assert!(
        result.is_ok(),
        "Non-finite preact should produce constant fallback"
    );
    let linear = result.unwrap();
    // Slopes should be zero (constant)
    for &v in linear.lower_a.iter() {
        assert_eq!(v, 0.0, "fallback lower_a should be zero");
    }
}

/// Regression test for #2713: CROWN with all-NEG_INFINITY pre-activation
/// must not produce NaN. The sound path guard (crown.rs:127-134) detects
/// non-finite pre-activation bounds and returns constant fallback bounds.
#[test]
fn logsoftmax_crown_all_neg_infinity_no_nan_2713() {
    let layer = LogSoftmaxLayer::new(-1);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY; 3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY; 3]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(3);
    let result =
        layer.propagate_linear_with_bounds(&bounds, &input, VerificationSoundnessMode::Sound);
    assert!(
        result.is_ok(),
        "all-NEG_INFINITY should produce constant fallback, not panic"
    );
    let linear = result.unwrap();
    // Slopes must be zero (constant bounds)
    assert!(
        linear.lower_a.iter().all(|&v| v == 0.0),
        "fallback lower_a should be zero"
    );
    assert!(
        linear.upper_a.iter().all(|&v| v == 0.0),
        "fallback upper_a should be zero"
    );
    // Biases must not be NaN
    assert!(
        linear.lower_b.iter().all(|v| !v.is_nan()),
        "CROWN lower_b has NaN for all-NEG_INFINITY: {:?}",
        linear.lower_b
    );
    assert!(
        linear.upper_b.iter().all(|v| !v.is_nan()),
        "CROWN upper_b has NaN for all-NEG_INFINITY: {:?}",
        linear.upper_b
    );
}

// ========== Soundness mode tests ==========

#[test]
fn logsoftmax_default_is_sound() {
    let layer = LogSoftmaxLayer::new(-1);
    assert_eq!(layer.soundness_mode(), VerificationSoundnessMode::Sound);
    assert!(layer.sound);
}

#[test]
fn logsoftmax_heuristic_toggle() {
    let layer = LogSoftmaxLayer::new(-1).with_heuristic_sampling(true);
    assert_eq!(layer.soundness_mode(), VerificationSoundnessMode::Heuristic);
    let layer = layer.with_sound_mode(true);
    assert_eq!(layer.soundness_mode(), VerificationSoundnessMode::Sound);
}

#[test]
fn logsoftmax_default_trait() {
    let layer = LogSoftmaxLayer::default();
    assert_eq!(layer.axis, -1);
    assert!(layer.sound);
}

// ========== resolve_axis tests ==========

#[test]
fn logsoftmax_resolve_axis_negative() {
    let layer = LogSoftmaxLayer::new(-1);
    // ndim=3, axis=-1 → 2
    assert_eq!(layer.resolve_axis(3).unwrap(), 2);
}

#[test]
fn logsoftmax_resolve_axis_out_of_range() {
    let layer = LogSoftmaxLayer::new(10);
    // axis=10 for ndim=3 → now returns error instead of silent fallback
    assert!(layer.resolve_axis(3).is_err());
}

#[test]
fn logsoftmax_resolve_axis_negative_out_of_range() {
    let layer = LogSoftmaxLayer::new(-4);
    // axis=-4 for ndim=3 → out of range
    assert!(layer.resolve_axis(3).is_err());
}

#[test]
fn logsoftmax_resolve_axis_positive() {
    let layer = LogSoftmaxLayer::new(1);
    assert_eq!(layer.resolve_axis(3).unwrap(), 1);
}

// ========== eval and jacobian tests ==========

#[test]
fn logsoftmax_eval_matches_reference() {
    let layer = LogSoftmaxLayer::new(-1);
    let x = ndarray::array![1.0, 2.0, 3.0];
    let result = layer.eval(&x);
    let expected = reference_logsoftmax(&[1.0, 2.0, 3.0]);
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-5, "eval[{}]={}, expected={}", i, r, e);
    }
}

#[test]
fn logsoftmax_eval_outputs_nonpositive() {
    let layer = LogSoftmaxLayer::new(-1);
    let x = ndarray::array![-5.0, 0.0, 5.0, 10.0];
    let result = layer.eval(&x);
    for (i, &v) in result.iter().enumerate() {
        assert!(v <= 1e-6, "logsoftmax[{}]={} should be <= 0", i, v);
    }
}

#[test]
fn logsoftmax_jacobian_row_sums_to_zero() {
    // d/dx_j (sum_i logsoftmax_i) = sum_i (delta_ij - softmax_j) = 1 - n*softmax_j ≠ 0 in general
    // But each ROW i of J has: sum_j J[i,j] = sum_j (delta_ij - s_j) = 1 - 1 = 0
    let layer = LogSoftmaxLayer::new(-1);
    let x = ndarray::array![1.0, 2.0, 3.0];
    let j = layer.jacobian(&x);
    for i in 0..3 {
        let row_sum: f32 = j.row(i).sum();
        assert!(
            row_sum.abs() < 1e-5,
            "J row {} sum = {}, expected 0",
            i,
            row_sum
        );
    }
}

#[test]
fn logsoftmax_jacobian_diagonal_formula() {
    // J[i,i] = 1 - softmax[i]
    let layer = LogSoftmaxLayer::new(-1);
    let x = ndarray::array![1.0, 2.0, 3.0];
    let s = layer.softmax(&x);
    let j = layer.jacobian(&x);
    for i in 0..3 {
        let expected = 1.0 - s[i];
        assert!(
            (j[[i, i]] - expected).abs() < 1e-5,
            "J[{0},{0}]={1}, expected 1-s[{0}]={2}",
            i,
            j[[i, i]],
            expected
        );
    }
}

#[test]
fn logsoftmax_jacobian_off_diagonal_formula() {
    // J[i,j] = -softmax[j] for i ≠ j
    let layer = LogSoftmaxLayer::new(-1);
    let x = ndarray::array![0.5, 1.5, -0.5];
    let s = layer.softmax(&x);
    let j = layer.jacobian(&x);
    for i in 0..3 {
        for k in 0..3 {
            if i != k {
                let expected = -s[k];
                assert!(
                    (j[[i, k]] - expected).abs() < 1e-5,
                    "J[{},{}]={}, expected -s[{}]={}",
                    i,
                    k,
                    j[[i, k]],
                    k,
                    expected
                );
            }
        }
    }
}

// ========== propagate_crown_backward tests ==========

#[test]
fn logsoftmax_crown_backward_requires_preactivation() {
    let layer = LogSoftmaxLayer::new(-1);
    let bounds = LinearBounds::identity(3);
    let result = layer.propagate_crown_backward(&bounds, None);
    assert!(
        result.is_err(),
        "Should error without pre-activation bounds"
    );
}

#[test]
fn logsoftmax_crown_backward_with_preact_succeeds() {
    let layer = LogSoftmaxLayer::new(-1);
    let input = make_bt(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);
    let bounds = LinearBounds::identity(3);
    let linear = layer
        .propagate_crown_backward(&bounds, Some(&input))
        .expect("Should succeed with valid pre-activation bounds");
    assert_eq!(linear.lower_a.shape(), &[3, 3], "lower_a shape");
    assert_eq!(linear.upper_a.shape(), &[3, 3], "upper_a shape");
    assert_eq!(linear.lower_b.len(), 3, "lower_b length");
    assert_eq!(linear.upper_b.len(), 3, "upper_b length");
    assert!(
        linear.lower_a.iter().all(|v| v.is_finite()),
        "lower_a must be finite"
    );
    assert!(
        linear.upper_a.iter().all(|v| v.is_finite()),
        "upper_a must be finite"
    );
    assert!(
        linear.lower_b.iter().all(|v| v.is_finite()),
        "lower_b must be finite"
    );
    assert!(
        linear.upper_b.iter().all(|v| v.is_finite()),
        "upper_b must be finite"
    );
}

// ========== Preserve-leading-axis tests (#4096) ==========

fn make_bt_2d(lower: &[f32], upper: &[f32], shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

/// Positive stored axis under preserve-leading-axis mode should shift right
/// by one, matching the sequential result. Part of #4096.
#[test]
fn logsoftmax_preserve_leading_axis_restores_positive_axis_4096() {
    // Simulate: ONNX LogSoftmax(axis=2) on [seq, hidden, vocab] stored as axis=1.
    // Sequential: shape [3, 4], axis=1 → logsoftmax over dim 1.
    // Restart-batched: shape [1, 3, 4], axis=1 stored → restored to axis=2.
    let layer = LogSoftmaxLayer::new(1);

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

/// Negative axis should behave identically with or without preserve-leading-axis.
#[test]
fn logsoftmax_preserve_leading_axis_negative_axis_unchanged_4096() {
    let layer = LogSoftmaxLayer::new(-1);

    let lower = vec![0.0, 1.0, 2.0, -1.0, 0.0, 1.0];
    let upper = vec![1.0, 2.0, 3.0, 0.0, 1.0, 2.0];

    let seq_input = make_bt_2d(&lower, &upper, &[2, 3]);
    let seq_output = layer.propagate_ibp(&seq_input).unwrap();

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
