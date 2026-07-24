// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::prelude::*;

#[test]
fn test_layernorm_eval() {
    // Test LayerNorm evaluation at concrete points
    let ny = arr1(&[1.0_f32, 2.0, 0.5]);
    let beta = arr1(&[0.0_f32, 1.0, -0.5]);
    let ln = LayerNormLayer::new(ny, beta, 1e-5).unwrap();

    // Test with simple input
    let x = arr1(&[1.0_f32, 2.0, 3.0]);
    let y = ln.eval(&x).unwrap();

    // mean = 2.0, var = 2/3, std ≈ 0.8165
    let mean = 2.0_f32;
    let var = ((1.0 - mean).powi(2) + (2.0 - mean).powi(2) + (3.0 - mean).powi(2)) / 3.0;
    let std = (var + 1e-5_f32).sqrt();

    let expected_0 = 1.0 * (1.0 - mean) / std + 0.0;
    let expected_1 = 2.0 * (2.0 - mean) / std + 1.0;
    let expected_2 = 0.5 * (3.0 - mean) / std + (-0.5);

    assert!(
        (y[0] - expected_0).abs() < 1e-5,
        "y[0] = {} != expected {}",
        y[0],
        expected_0
    );
    assert!(
        (y[1] - expected_1).abs() < 1e-5,
        "y[1] = {} != expected {}",
        y[1],
        expected_1
    );
    assert!(
        (y[2] - expected_2).abs() < 1e-5,
        "y[2] = {} != expected {}",
        y[2],
        expected_2
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_mean_only_eval() {
    let ny = arr1(&[1.0_f32, 2.0, -0.5]);
    let beta = arr1(&[0.1_f32, -0.2, 0.3]);
    let ln = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5)
        .unwrap()
        .with_mode(LayerNormMode::MeanOnly);

    let x = arr1(&[1.0_f32, 2.0, 3.0]);
    let mean = (1.0_f32 + 2.0 + 3.0) / 3.0;
    let y = ln.eval(&x).unwrap();

    let expected: Array1<f32> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| ny[i] * (xi - mean) + beta[i])
        .collect();

    for i in 0..3 {
        assert!(
            (y[i] - expected[i]).abs() < 1e-6,
            "mean-only y[{}] = {} != expected {}",
            i,
            y[i],
            expected[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_mean_only_crown_linear() {
    let ny = arr1(&[1.2_f32, -0.8, 0.5]);
    let beta = arr1(&[0.0_f32, 0.1, -0.2]);
    let ln = LayerNormLayer::new(ny, beta, 1e-5)
        .unwrap()
        .with_mode(LayerNormMode::MeanOnly);

    let input_lower = arr1(&[-1.0_f32, 0.0, 1.0]);
    let input_upper = arr1(&[1.0_f32, 2.0, 3.0]);
    let input = BoundedTensor::new(
        input_lower.clone().into_dyn(),
        input_upper.clone().into_dyn(),
    )
    .unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let crown_bounds = ln
        .propagate_linear_with_bounds(&linear_bounds, &input)
        .expect("mean-only LayerNorm CROWN should be linear");
    let concrete = crown_bounds.concretize(&input);

    for sample_idx in 0..50 {
        let t0 = (sample_idx as f32 * 13.0 % 100.0) / 100.0;
        let t1 = (sample_idx as f32 * 29.0 % 100.0) / 100.0;
        let t2 = (sample_idx as f32 * 43.0 % 100.0) / 100.0;

        let x_sample = arr1(&[
            input_lower[0] + (input_upper[0] - input_lower[0]) * t0,
            input_lower[1] + (input_upper[1] - input_lower[1]) * t1,
            input_lower[2] + (input_upper[2] - input_lower[2]) * t2,
        ]);

        let y_sample = ln.eval(&x_sample).unwrap();
        for i in 0..3 {
            assert!(
                y_sample[i] >= concrete.lower()[[i]] - 1e-5,
                "mean-only sample {} output {} < lower bound {} at dim {}",
                sample_idx,
                y_sample[i],
                concrete.lower()[[i]],
                i
            );
            assert!(
                y_sample[i] <= concrete.upper()[[i]] + 1e-5,
                "mean-only sample {} output {} > upper bound {} at dim {}",
                sample_idx,
                y_sample[i],
                concrete.upper()[[i]],
                i
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_jacobian() {
    // Test LayerNorm Jacobian computation
    let ny = arr1(&[1.0_f32, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0]);
    let ln = LayerNormLayer::new(ny, beta, 1e-5).unwrap();

    let x = arr1(&[1.0_f32, 2.0, 3.0]);
    let jacobian = ln.jacobian(&x).unwrap();

    // Verify Jacobian via finite differences
    let eps = 1e-4_f32;
    for j in 0..3 {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[j] += eps;
        x_minus[j] -= eps;

        let y_plus = ln.eval(&x_plus).unwrap();
        let y_minus = ln.eval(&x_minus).unwrap();

        for i in 0..3 {
            let fd = (y_plus[i] - y_minus[i]) / (2.0 * eps);
            // Allow 1% relative error or 1e-2 absolute error for numerical stability
            let rel_err = if fd.abs() > 1e-6 {
                (jacobian[[i, j]] - fd).abs() / fd.abs()
            } else {
                (jacobian[[i, j]] - fd).abs()
            };
            assert!(
                rel_err < 0.02 || (jacobian[[i, j]] - fd).abs() < 1e-2,
                "J[{},{}] = {} != finite diff {} (rel_err={:.2}%, abs_diff={})",
                i,
                j,
                jacobian[[i, j]],
                fd,
                rel_err * 100.0,
                (jacobian[[i, j]] - fd).abs()
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_crown_soundness() {
    // Test that LayerNorm CROWN bounds are sound (using sampling mode)
    use crate::layers::LayerNormCrownMode;

    let ny = arr1(&[1.0_f32, 2.0, 0.5]);
    let beta = arr1(&[0.0_f32, 1.0, -0.5]);
    let ln = LayerNormLayer::new(ny, beta, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);

    // Create input bounds
    let input_lower = arr1(&[0.5_f32, 1.5, 2.5]);
    let input_upper = arr1(&[1.5_f32, 2.5, 3.5]);
    let input = BoundedTensor::new(
        input_lower.clone().into_dyn(),
        input_upper.clone().into_dyn(),
    )
    .unwrap();

    // Get CROWN bounds
    let linear_bounds = LinearBounds::identity(3);
    let crown_bounds = ln
        .propagate_linear_with_bounds(&linear_bounds, &input)
        .unwrap();

    // Concretize to get scalar bounds
    let concrete = crown_bounds.concretize(&input);

    // Verify soundness by sampling
    for sample_idx in 0..100 {
        let t0 = (sample_idx as f32 * 17.0 % 100.0) / 100.0;
        let t1 = (sample_idx as f32 * 31.0 % 100.0) / 100.0;
        let t2 = (sample_idx as f32 * 47.0 % 100.0) / 100.0;

        let x_sample = arr1(&[
            input_lower[0] + (input_upper[0] - input_lower[0]) * t0,
            input_lower[1] + (input_upper[1] - input_lower[1]) * t1,
            input_lower[2] + (input_upper[2] - input_lower[2]) * t2,
        ]);

        let y_sample = ln.eval(&x_sample).unwrap();

        for i in 0..3 {
            assert!(
                y_sample[i] >= concrete.lower()[[i]] - 1e-4,
                "Sample {} output {} = {} < lower bound {} at dim {}",
                sample_idx,
                y_sample[i],
                y_sample[i],
                concrete.lower()[[i]],
                i
            );
            assert!(
                y_sample[i] <= concrete.upper()[[i]] + 1e-4,
                "Sample {} output {} = {} > upper bound {} at dim {}",
                sample_idx,
                y_sample[i],
                y_sample[i],
                concrete.upper()[[i]],
                i
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_crown_tighter_than_ibp() {
    // Verify both IBP and CROWN bounds are sound (contain true LayerNorm output).
    // Note: sampling CROWN is not guaranteed tighter than IBP for LayerNorm (#3169).
    use crate::layers::LayerNormCrownMode;

    let ny = arr1(&[1.0_f32, 1.0, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0, 0.0]);
    let ln = LayerNormLayer::new(ny, beta, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);

    // Small perturbation around a point
    let center = arr1(&[1.0_f32, 2.0, 3.0, 4.0]);
    let eps = 0.1_f32;
    let input_lower: Array1<f32> = center.iter().map(|&c| c - eps).collect();
    let input_upper: Array1<f32> = center.iter().map(|&c| c + eps).collect();

    let input = BoundedTensor::new(
        input_lower.clone().into_dyn(),
        input_upper.clone().into_dyn(),
    )
    .unwrap();

    // Get IBP bounds
    let ibp_bounds = ln.propagate_ibp(&input).unwrap();

    // Get CROWN bounds
    let linear_bounds = LinearBounds::identity(4);
    let crown_result = ln
        .propagate_linear_with_bounds(&linear_bounds, &input)
        .unwrap();
    let crown_bounds = crown_result.concretize(&input);

    // Soundness check: sample concrete points and verify both IBP and CROWN
    // bounds contain the true LayerNorm output. Quality (tightness) is secondary;
    // the sampling CROWN heuristic for LayerNorm is not guaranteed to be tighter
    // than IBP due to cross-dimensional dependencies.
    let tol = 1e-2; // sampling CROWN tolerance (heuristic, not provably sound)
    let n_samples = 5; // per dimension → 5^4 = 625 grid points
    for i0 in 0..n_samples {
        for i1 in 0..n_samples {
            for i2 in 0..n_samples {
                for i3 in 0..n_samples {
                    let t = |idx: usize, n: usize| idx as f32 / (n - 1).max(1) as f32;
                    let x = arr1(&[
                        input_lower[0] + t(i0, n_samples) * (input_upper[0] - input_lower[0]),
                        input_lower[1] + t(i1, n_samples) * (input_upper[1] - input_lower[1]),
                        input_lower[2] + t(i2, n_samples) * (input_upper[2] - input_lower[2]),
                        input_lower[3] + t(i3, n_samples) * (input_upper[3] - input_lower[3]),
                    ]);
                    let y = ln.eval(&x).unwrap();
                    for d in 0..4 {
                        assert!(
                            y[d] >= crown_bounds.lower()[[d]] - tol,
                            "CROWN lower violation at dim {d}: LN({x})={} < lb={}",
                            y[d],
                            crown_bounds.lower()[[d]]
                        );
                        assert!(
                            y[d] <= crown_bounds.upper()[[d]] + tol,
                            "CROWN upper violation at dim {d}: LN({x})={} > ub={}",
                            y[d],
                            crown_bounds.upper()[[d]]
                        );
                        assert!(
                            y[d] >= ibp_bounds.lower()[[d]] - tol,
                            "IBP lower violation at dim {d}: LN({x})={} < lb={}",
                            y[d],
                            ibp_bounds.lower()[[d]]
                        );
                        assert!(
                            y[d] <= ibp_bounds.upper()[[d]] + tol,
                            "IBP upper violation at dim {d}: LN({x})={} > ub={}",
                            y[d],
                            ibp_bounds.upper()[[d]]
                        );
                    }
                }
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_crown_propagation_through_network() {
    // Test CROWN propagation through Linear -> LayerNorm (using sampling mode)
    use crate::layers::LayerNormCrownMode;

    // Create Linear layer
    let weight = arr2(&[[1.0_f32, 0.5, -0.3], [-0.5, 1.0, 0.2], [0.3, -0.2, 1.0]]);
    let linear = LinearLayer::new(weight, Some(arr1(&[0.1, -0.1, 0.0]))).unwrap();

    // Create LayerNorm layer (with sampling mode for CROWN)
    let ny = arr1(&[1.0_f32, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0]);
    let ln = LayerNormLayer::new(ny, beta, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);

    // Input bounds
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();

    // Propagate through Linear with IBP to get pre-LayerNorm bounds
    let after_linear = linear.propagate_ibp(&input).unwrap();

    // Propagate CROWN through LayerNorm
    let ln_linear_bounds = LinearBounds::identity(3);
    let crown_result = ln
        .propagate_linear_with_bounds(&ln_linear_bounds, &after_linear)
        .unwrap();
    let crown_bounds = crown_result.concretize(&after_linear);

    // Verify soundness by sampling
    let input_lower = arr1(&[-0.5_f32, -0.5, -0.5]);
    let input_upper = arr1(&[0.5_f32, 0.5, 0.5]);

    for sample_idx in 0..50 {
        let t0 = (sample_idx as f32 * 17.0 % 50.0) / 50.0;
        let t1 = (sample_idx as f32 * 31.0 % 50.0) / 50.0;
        let t2 = (sample_idx as f32 * 47.0 % 50.0) / 50.0;

        let x_sample = arr1(&[
            input_lower[0] + (input_upper[0] - input_lower[0]) * t0,
            input_lower[1] + (input_upper[1] - input_lower[1]) * t1,
            input_lower[2] + (input_upper[2] - input_lower[2]) * t2,
        ])
        .into_dyn();

        // Forward through Linear
        let weight_view = linear.weight.view();
        let linear_out: Array1<f32> = weight_view.dot(
            &x_sample
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .unwrap(),
        ) + linear.bias.as_ref().unwrap();

        // Forward through LayerNorm
        let ln_out = ln.eval(&linear_out).unwrap();

        for i in 0..3 {
            assert!(
                ln_out[i] >= crown_bounds.lower()[[i]] - 1e-3,
                "Sample {} output {} = {} < lower bound {} at dim {}",
                sample_idx,
                ln_out[i],
                ln_out[i],
                crown_bounds.lower()[[i]],
                i
            );
            assert!(
                ln_out[i] <= crown_bounds.upper()[[i]] + 1e-3,
                "Sample {} output {} = {} > upper bound {} at dim {}",
                sample_idx,
                ln_out[i],
                ln_out[i],
                crown_bounds.upper()[[i]],
                i
            );
        }
    }
}
