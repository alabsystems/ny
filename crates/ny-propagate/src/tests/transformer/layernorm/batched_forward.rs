// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_batched_linear_bounds() {
    // Test the LayerNormLayer::propagate_linear_batched_with_bounds directly (using sampling mode)
    use crate::layers::LayerNormCrownMode;

    let hidden = 4;
    let ln = LayerNormLayer::new_default(hidden, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);

    // Create 2D input shape [batch, hidden]
    let batch = 3;

    // Pre-activation bounds
    let mut pre_lower = ArrayD::zeros(IxDyn(&[batch, hidden]));
    let mut pre_upper = ArrayD::zeros(IxDyn(&[batch, hidden]));

    for b in 0..batch {
        for h in 0..hidden {
            let hash = ((b * 10 + h) as u32).wrapping_mul(2654435761_u32);
            let base = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0;
            pre_lower[[b, h]] = base - 0.2;
            pre_upper[[b, h]] = base + 0.2;
        }
    }

    let pre_bounds = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Create identity bounds: shape [batch, hidden, hidden]
    let identity = BatchedLinearBounds::identity(&[batch, hidden]).unwrap();

    // Propagate backward
    let result = ln
        .propagate_linear_batched_with_bounds(&identity, &pre_bounds)
        .unwrap();

    // Verify shape
    assert_eq!(result.lower_a.shape(), &[batch, hidden, hidden]);
    assert_eq!(result.lower_b.shape(), &[batch, hidden]);

    // Verify all values are finite
    let all_finite = result.lower_a.iter().all(|v| v.is_finite())
        && result.upper_a.iter().all(|v| v.is_finite())
        && result.lower_b.iter().all(|v| v.is_finite())
        && result.upper_b.iter().all(|v| v.is_finite());

    assert!(all_finite, "All batched layernorm bounds should be finite");

    // Verify soundness by sampling
    for sample_idx in 0..20 {
        // Sample a concrete input for each batch position
        let mut x_sample = ArrayD::<f32>::zeros(IxDyn(&[batch, hidden]));
        for b in 0..batch {
            for h in 0..hidden {
                let hash = ((sample_idx * 1000 + b * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let t = hash as f32 / u32::MAX as f32;
                x_sample[[b, h]] = pre_bounds.lower()[[b, h]]
                    + (pre_bounds.upper()[[b, h]] - pre_bounds.lower()[[b, h]]) * t;
            }
        }

        // Evaluate layernorm at this point for each batch
        for b in 0..batch {
            let x_1d: Array1<f32> = (0..hidden).map(|h| x_sample[[b, h]]).collect();
            let y_actual = ln.eval(&x_1d).unwrap();

            // Concretize the linear bounds at this sample point
            for j in 0..hidden {
                let mut lower_val = result.lower_b[[b, j]];
                let mut upper_val = result.upper_b[[b, j]];

                for k in 0..hidden {
                    let la = result.lower_a[[b, j, k]];
                    let ua = result.upper_a[[b, j, k]];

                    // For lower bound: if coeff positive, use lower of input; if negative, use upper
                    if la >= 0.0 {
                        lower_val += la * pre_bounds.lower()[[b, k]];
                    } else {
                        lower_val += la * pre_bounds.upper()[[b, k]];
                    }

                    // For upper bound: if coeff positive, use upper of input; if negative, use lower
                    if ua >= 0.0 {
                        upper_val += ua * pre_bounds.upper()[[b, k]];
                    } else {
                        upper_val += ua * pre_bounds.lower()[[b, k]];
                    }
                }

                // The actual output should be within the concretized bounds
                // Note: These are loose bounds due to sampling-based relaxation
                assert!(
                    y_actual[j] >= lower_val - 0.5,
                    "LayerNorm batch {} output {} violates lower: {} < {}",
                    b,
                    j,
                    y_actual[j],
                    lower_val
                );
                assert!(
                    y_actual[j] <= upper_val + 0.5,
                    "LayerNorm batch {} output {} violates upper: {} > {}",
                    b,
                    j,
                    y_actual[j],
                    upper_val
                );
            }
        }
    }

    println!(
        "LayerNorm batched bounds test passed with {} batch positions",
        batch
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_batched_mean_only_linear_bounds() {
    let hidden = 3;
    let batch = 2;

    let ny = arr1(&[1.2_f32, -0.7, 0.5]);
    let beta = arr1(&[0.05_f32, -0.1, 0.2]);
    let ln = LayerNormLayer::new(ny, beta, 1e-5)
        .unwrap()
        .with_mode(LayerNormMode::MeanOnly);

    let mut pre_lower = ArrayD::zeros(IxDyn(&[batch, hidden]));
    let mut pre_upper = ArrayD::zeros(IxDyn(&[batch, hidden]));

    for b in 0..batch {
        for h in 0..hidden {
            let base = (b as f32 * 0.3) + (h as f32 * 0.2) - 0.5;
            pre_lower[[b, h]] = base - 0.4;
            pre_upper[[b, h]] = base + 0.6;
        }
    }

    let pre_bounds = BoundedTensor::new(pre_lower, pre_upper).unwrap();
    let identity = BatchedLinearBounds::identity(&[batch, hidden]).unwrap();

    let result = ln
        .propagate_linear_batched_with_bounds(&identity, &pre_bounds)
        .unwrap();

    assert_eq!(result.lower_a.shape(), &[batch, hidden, hidden]);
    assert_eq!(result.lower_b.shape(), &[batch, hidden]);

    let concrete = result.concretize(&pre_bounds).unwrap();

    for sample_idx in 0..25 {
        let mut x_sample = ArrayD::<f32>::zeros(IxDyn(&[batch, hidden]));
        for b in 0..batch {
            for h in 0..hidden {
                let t = ((sample_idx + b * 7 + h * 11) as f32 % 17.0) / 17.0;
                x_sample[[b, h]] = pre_bounds.lower()[[b, h]]
                    + (pre_bounds.upper()[[b, h]] - pre_bounds.lower()[[b, h]]) * t;
            }
        }

        for b in 0..batch {
            let x_1d: Array1<f32> = (0..hidden).map(|h| x_sample[[b, h]]).collect();
            let y_actual = ln.eval(&x_1d).unwrap();

            for j in 0..hidden {
                assert!(
                    y_actual[j] >= concrete.lower()[[b, j]] - 1e-5,
                    "mean-only batch {} sample {} output {} < lower {} at dim {}",
                    b,
                    sample_idx,
                    y_actual[j],
                    concrete.lower()[[b, j]],
                    j
                );
                assert!(
                    y_actual[j] <= concrete.upper()[[b, j]] + 1e-5,
                    "mean-only batch {} sample {} output {} > upper {} at dim {}",
                    b,
                    sample_idx,
                    y_actual[j],
                    concrete.upper()[[b, j]],
                    j
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_batched_mean_only_counterexample_rounding_regression() {
    // Regression from #2175: large-magnitude ny/A values stress f32 rounding.
    // A-matrix uses round-to-nearest (#2208); soundness via bias directed rounding.
    let hidden = 3;
    let ny = arr1(&[44308008.0_f32, -54247556.0, 89054136.0]);
    let lower_a_row = [52456016.0_f32, -99578792.0, -10922561.0];

    let layer = LayerNormLayer::new(ny, Array1::zeros(hidden), 1e-5)
        .unwrap()
        .with_mode(LayerNormMode::MeanOnly);

    let pre_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, hidden]), vec![-1.0, -1.0, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, hidden]), vec![1.0, 1.0, 1.0]).unwrap(),
    )
    .unwrap();

    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, hidden]), lower_a_row.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, hidden]), lower_a_row.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0]).unwrap(),
        vec![1, hidden],
        vec![1, 1],
    );

    let result = layer
        .propagate_linear_batched_with_bounds(&bounds, &pre_bounds)
        .unwrap();

    // All values must be finite.
    assert!(
        result
            .lower_a
            .iter()
            .chain(result.upper_a.iter())
            .chain(result.lower_b.iter())
            .chain(result.upper_b.iter())
            .all(|v| v.is_finite()),
        "all bounds finite"
    );

    // Verify concretization soundness at sampled points.
    let concrete = result.concretize(&pre_bounds).unwrap();
    for s in 0..20 {
        let mut x = ArrayD::<f32>::zeros(IxDyn(&[1, hidden]));
        for h in 0..hidden {
            let t = ((s * 7 + h * 13) as f32 % 19.0) / 19.0;
            x[[0, h]] = pre_bounds.lower()[[0, h]]
                + (pre_bounds.upper()[[0, h]] - pre_bounds.lower()[[0, h]]) * t;
        }
        let x_1d: Array1<f32> = (0..hidden).map(|h| x[[0, h]]).collect();
        let y_actual = layer.eval(&x_1d).unwrap();
        let y_out: f64 = lower_a_row
            .iter()
            .zip(y_actual.iter())
            .map(|(a, y)| *a as f64 * *y as f64)
            .sum();
        let lb = concrete.lower()[[0, 0]] as f64;
        let ub = concrete.upper()[[0, 0]] as f64;
        assert!(y_out >= lb - 1.0, "sample {s}: {y_out} < lower {lb}");
        assert!(y_out <= ub + 1.0, "sample {s}: {y_out} > upper {ub}");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_forward_mode_vs_conservative() {
    // Compare forward-mode LayerNorm (tighter but approximate) vs
    // conservative mode (sound but may explode).
    //
    // Forward mode uses fixed mean/std from center point, dramatically
    // reducing bound explosion for small perturbations.

    let hidden = 8;
    let batch = 1;
    let seq = 4;

    // Create two LayerNorm layers: conservative and forward mode
    let ln_conservative = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
    let ln_forward = LayerNormLayer::new_default(hidden, 1e-5)
        .unwrap()
        .with_forward_mode(true);

    println!(
        "
=== Forward Mode vs Conservative LayerNorm IBP ==="
    );
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>12}",
        "Epsilon", "Conservative", "Forward", "Ratio", "Improvement"
    );
    println!("{}", "-".repeat(62));

    for epsilon in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5] {
        // Create input bounds
        let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
        let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

        for b in 0..batch {
            for s in 0..seq {
                for h in 0..hidden {
                    let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                    let base = (hash as f32 / u32::MAX as f32) * 0.5;
                    lower[[b, s, h]] = base - epsilon;
                    upper[[b, s, h]] = base + epsilon;
                }
            }
        }

        let input = BoundedTensor::new(lower, upper).unwrap();

        let avg_width = |bt: &BoundedTensor| -> f32 {
            bt.lower()
                .iter()
                .zip(bt.upper().iter())
                .map(|(l, u)| u - l)
                .sum::<f32>()
                / bt.len() as f32
        };

        // Flatten for LayerNorm
        let flat_input = input.reshape(&[batch * seq, hidden]).unwrap();

        // Conservative mode
        let cons_out = ln_conservative.propagate_ibp(&flat_input).unwrap();
        let cons_width = avg_width(&cons_out);

        // Forward mode
        let fwd_out = ln_forward.propagate_ibp(&flat_input).unwrap();
        let fwd_width = avg_width(&fwd_out);

        let ratio = cons_width / fwd_width;
        let improvement = (1.0 - fwd_width / cons_width) * 100.0;

        println!(
            "{:<10.3} {:>12.4} {:>12.4} {:>12.2}x {:>11.1}%",
            epsilon, cons_width, fwd_width, ratio, improvement
        );

        // Forward mode bounds must be valid (lower <= upper)
        for (l, u) in fwd_out.lower().iter().zip(fwd_out.upper().iter()) {
            assert!(
                l <= u,
                "Forward mode produced invalid bounds: lower {} > upper {}",
                l,
                u
            );
        }

        // Soundness check: sample concrete points and verify forward-mode
        // bounds contain the true output. Forward mode is not guaranteed
        // tighter than conservative for all cases (#3169).
        let tol = 1e-2; // sampling tolerance for heuristic bounds
        for sample_idx in 0..10 {
            let mut point = ArrayD::zeros(IxDyn(&[batch * seq, hidden]));
            for row in 0..(batch * seq) {
                for h in 0..hidden {
                    let hash =
                        ((sample_idx * 1000 + row * 100 + h) as u32).wrapping_mul(2654435761_u32);
                    let t = hash as f32 / u32::MAX as f32;
                    point[[row, h]] = flat_input.lower()[[row, h]]
                        + t * (flat_input.upper()[[row, h]] - flat_input.lower()[[row, h]]);
                }
            }
            let pt_bt = BoundedTensor::new(point.clone(), point).unwrap();

            // Check forward-mode bounds contain true output
            let true_out = ln_forward.propagate_ibp(&pt_bt).unwrap();
            for (idx, (y, (lb, ub))) in true_out
                .lower()
                .iter()
                .zip(fwd_out.lower().iter().zip(fwd_out.upper().iter()))
                .enumerate()
            {
                assert!(
                    *y >= *lb - tol,
                    "Forward mode soundness violation at eps={epsilon}, idx={idx}: \
                     y={y} < lb={lb}"
                );
                assert!(
                    *y <= *ub + tol,
                    "Forward mode soundness violation at eps={epsilon}, idx={idx}: \
                     y={y} > ub={ub}"
                );
            }

            // Check conservative-mode bounds also contain true output
            let cons_true = ln_conservative.propagate_ibp(&pt_bt).unwrap();
            for (idx, (y, (lb, ub))) in cons_true
                .lower()
                .iter()
                .zip(cons_out.lower().iter().zip(cons_out.upper().iter()))
                .enumerate()
            {
                assert!(
                    *y >= *lb - tol,
                    "Conservative soundness violation at eps={epsilon}, idx={idx}: \
                     y={y} < lb={lb}"
                );
                assert!(
                    *y <= *ub + tol,
                    "Conservative soundness violation at eps={epsilon}, idx={idx}: \
                     y={y} > ub={ub}"
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_forward_mode_low_variance_center() {
    // Test that LayerNorm forward-mode produces finite bounds when center has low variance.
    //
    // When bounds' midpoint (center) has near-zero variance, sensitivity = ny/std
    // is large (std ≈ sqrt(eps) ≈ 0.003 → ~356× amplification for ny=1).
    // Capping was removed (#2074) because it produced unsound bounds.
    // The bounds are still finite and valid — just loose for low-variance inputs.

    let hidden = 8;
    let ln = LayerNormLayer::new_default(hidden, 1e-5)
        .unwrap()
        .with_forward_mode(true);

    let epsilon = 0.001;
    let mut zero_lower = ArrayD::zeros(IxDyn(&[1, hidden]));
    let mut zero_upper = ArrayD::zeros(IxDyn(&[1, hidden]));
    for i in 0..hidden {
        zero_lower[[0, i]] = -epsilon;
        zero_upper[[0, i]] = epsilon;
    }
    let zero_input = BoundedTensor::new(zero_lower, zero_upper).unwrap();
    let zero_out = ln.propagate_ibp(&zero_input).unwrap();

    // All bounds must be finite (no NaN/Inf).
    assert!(
        zero_out.lower().iter().all(|v| v.is_finite()),
        "Forward-mode lower bounds must be finite for low-variance center"
    );
    assert!(
        zero_out.upper().iter().all(|v| v.is_finite()),
        "Forward-mode upper bounds must be finite for low-variance center"
    );

    // Bounds must be ordered (lower <= upper).
    for (l, u) in zero_out.lower().iter().zip(zero_out.upper().iter()) {
        assert!(l <= u, "Bounds must be ordered: lower={l} > upper={u}");
    }

    // Amplification is high without sensitivity capping (#2074 removed for soundness).
    // No arbitrary threshold — just verify soundness via sampling.
    let tol = 1e-2;
    for sample_idx in 0..20 {
        let mut point = ArrayD::zeros(IxDyn(&[1, hidden]));
        for h in 0..hidden {
            let hash = ((sample_idx * 100 + h) as u32).wrapping_mul(2654435761_u32);
            let t = hash as f32 / u32::MAX as f32;
            point[[0, h]] = zero_input.lower()[[0, h]]
                + t * (zero_input.upper()[[0, h]] - zero_input.lower()[[0, h]]);
        }
        let pt_bt = BoundedTensor::new(point.clone(), point).unwrap();
        let true_out = ln.propagate_ibp(&pt_bt).unwrap();
        for (idx, (y, (lb, ub))) in true_out
            .lower()
            .iter()
            .zip(zero_out.lower().iter().zip(zero_out.upper().iter()))
            .enumerate()
        {
            assert!(
                *y >= *lb - tol,
                "Low-variance soundness violation at idx={idx}: y={y} < lb={lb}"
            );
            assert!(
                *y <= *ub + tol,
                "Low-variance soundness violation at idx={idx}: y={y} > ub={ub}"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_ibp_rejects_nonfinite_input() {
    // Category B per domain validation policy: non-finite inputs → NumericalInstability.
    // Previously this test verified silent NaN-to-MAX_BOUND clamping; now it verifies
    // explicit error return per designs/2026-02-07-domain-validation-policy.md.
    let hidden = 4;
    let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();

    // Case 1: Infinity in bounds
    let lower = ArrayD::from_elem(IxDyn(&[2, hidden]), f32::NEG_INFINITY);
    let upper = ArrayD::from_elem(IxDyn(&[2, hidden]), f32::INFINITY);
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let err = ln
        .propagate_ibp(&input)
        .expect_err("LayerNorm IBP should reject non-finite inputs");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "Expected NumericalInstability, got: {err:?}"
    );

    // Case 2: NaN in lower bounds
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, hidden]), vec![f32::NAN, 1.0, 2.0, 3.0]).unwrap();
    let upper = ArrayD::from_elem(IxDyn(&[1, hidden]), 5.0f32);
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let err = ln
        .propagate_ibp(&input)
        .expect_err("LayerNorm IBP should reject NaN inputs");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "Expected NumericalInstability, got: {err:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_layernorm_forward_mode_large_finite_point_avoids_center_overflow() {
    let hidden = 4;
    let ln = LayerNormLayer::new_default(hidden, 1e-5)
        .unwrap()
        .with_forward_mode(true);

    let lower = ArrayD::from_elem(IxDyn(&[1, hidden]), f32::MAX);
    let upper = ArrayD::from_elem(IxDyn(&[1, hidden]), f32::MAX);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = ln.propagate_ibp(&input).unwrap();
    let evaluated = ln
        .eval(&Array1::from_elem(hidden, f32::MAX))
        .expect("large finite point evaluation");
    assert!(
        out.lower()
            .iter()
            .chain(out.upper().iter())
            .all(|&v| v.is_finite()),
        "output should be finite, got lower={:?} upper={:?}",
        out.lower(),
        out.upper()
    );
    for i in 0..hidden {
        assert!(
            out.lower()[[0, i]] <= evaluated[i] && evaluated[i] <= out.upper()[[0, i]],
            "dim {i}: [{}, {}] excludes {}",
            out.lower()[[0, i]],
            out.upper()[[0, i]],
            evaluated[i]
        );
    }
    assert!(
        out.max_width() < 1e-3,
        "a point interval should avoid the old overflow fallback: width={}",
        out.max_width()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sanitize_bounds_for_fallback_replaces_nan_and_inverted() {
    // Use new_unchecked to bypass debug_asserts - this test intentionally uses NaN and Inf
    let bounds = BoundedTensor::new_unchecked(
        arr1(&[f32::NAN, f32::INFINITY, 1.0]).into_dyn(),
        arr1(&[f32::NAN, f32::NEG_INFINITY, 0.0]).into_dyn(),
    )
    .unwrap();

    let sanitized = GraphNetwork::sanitize_bounds_for_fallback(&bounds);
    assert!(
        sanitized
            .lower()
            .iter()
            .chain(sanitized.upper().iter())
            .all(|&v| !v.is_nan()),
        "sanitized bounds should not contain NaN"
    );

    let pairs: Vec<(f32, f32)> = sanitized
        .lower()
        .iter()
        .cloned()
        .zip(sanitized.upper().iter().cloned())
        .collect();
    assert_eq!(pairs[0], (f32::NEG_INFINITY, f32::INFINITY));
    assert_eq!(pairs[1], (f32::NEG_INFINITY, f32::INFINITY));
    assert_eq!(pairs[2], (f32::NEG_INFINITY, f32::INFINITY));
}
