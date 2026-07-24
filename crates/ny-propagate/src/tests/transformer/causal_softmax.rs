// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CausalSoftmax tests.

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_layer_basic() {
    // Test CausalSoftmaxLayer IBP propagation
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    // 2D input: [seq_q=3, seq_k=3]
    // For causal: row i can only attend to columns 0..=i
    let values = arr2(&[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    let input = BoundedTensor::new(values.clone().into_dyn(), values.into_dyn()).unwrap();

    let output = causal_softmax.propagate_ibp(&input).unwrap();

    // Row 0: softmax([1.0]) = [1.0], masked positions are 0
    assert!(
        (output.lower()[[0, 0]] - 1.0).abs() < 1e-5,
        "Row 0, pos 0 should be 1.0"
    );
    assert!((output.upper()[[0, 0]] - 1.0).abs() < 1e-5);
    assert!(
        output.lower()[[0, 1]].abs() < 1e-5,
        "Row 0, pos 1 should be 0 (masked)"
    );
    assert!(output.upper()[[0, 1]].abs() < 1e-5);
    assert!(
        output.lower()[[0, 2]].abs() < 1e-5,
        "Row 0, pos 2 should be 0 (masked)"
    );
    assert!(output.upper()[[0, 2]].abs() < 1e-5);

    // Row 1: softmax([1.0, 2.0]) = [~0.27, ~0.73], position 2 masked
    let row1_sum: f32 = output.lower()[[1, 0]] + output.lower()[[1, 1]];
    assert!(
        (row1_sum - 1.0).abs() < 1e-4,
        "Row 1 unmasked sum should be 1.0, got {}",
        row1_sum
    );
    assert!(
        output.lower()[[1, 2]].abs() < 1e-5,
        "Row 1, pos 2 should be 0 (masked)"
    );
    assert!(output.upper()[[1, 2]].abs() < 1e-5);

    // Row 2: full softmax - all positions unmasked
    let row2_sum: f32 = output.lower()[[2, 0]] + output.lower()[[2, 1]] + output.lower()[[2, 2]];
    assert!(
        (row2_sum - 1.0).abs() < 1e-4,
        "Row 2 sum should be 1.0, got {}",
        row2_sum
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_layer_3d_basic() {
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    // 3D input: [batch=2, seq_q=3, seq_k=3]
    let batch = 2;
    let seq_q = 3;
    let seq_k = 3;
    let mut data = Vec::with_capacity(batch * seq_q * seq_k);
    for b in 0..batch {
        for i in 0..seq_q {
            for j in 0..seq_k {
                data.push((b as f32) * 0.1 + (i as f32) * 0.01 + (j as f32) * 0.001);
            }
        }
    }

    let values = ndarray::Array3::from_shape_vec((batch, seq_q, seq_k), data).unwrap();
    let input = BoundedTensor::new(values.clone().into_dyn(), values.into_dyn()).unwrap();
    let output = causal_softmax.propagate_ibp(&input).unwrap();

    for b in 0..batch {
        for i in 0..seq_q {
            let mut row_sum_lb = 0.0_f32;
            let mut row_sum_ub = 0.0_f32;
            for j in 0..seq_k {
                let lb = output.lower()[[b, i, j]];
                let ub = output.upper()[[b, i, j]];
                assert!(
                    lb <= ub + 1e-6,
                    "Batch {}, row {}, col {} invalid bounds: lb {} > ub {}",
                    b,
                    i,
                    j,
                    lb,
                    ub
                );
                if j > i {
                    assert!(
                        lb.abs() < 1e-6,
                        "Batch {}, row {}, col {} should be masked to 0 (lb {})",
                        b,
                        i,
                        j,
                        lb
                    );
                    assert!(
                        ub.abs() < 1e-6,
                        "Batch {}, row {}, col {} should be masked to 0 (ub {})",
                        b,
                        i,
                        j,
                        ub
                    );
                } else {
                    assert!(
                        lb >= -1e-6 && ub <= 1.0 + 1e-6,
                        "Batch {}, row {}, col {} bounds should be in [0,1], got [{}, {}]",
                        b,
                        i,
                        j,
                        lb,
                        ub
                    );
                    row_sum_lb += lb;
                    row_sum_ub += ub;
                }
            }
            assert!(
                row_sum_lb <= 1.0 + 1e-4,
                "Batch {}, row {} unmasked lower sum should be <= 1.0, got {}",
                b,
                i,
                row_sum_lb
            );
            assert!(
                row_sum_ub >= 1.0 - 1e-4,
                "Batch {}, row {} unmasked upper sum should be >= 1.0, got {}",
                b,
                i,
                row_sum_ub
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_layer_3d_nonsquare() {
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    // 3D input: [batch=1, seq_q=2, seq_k=4] (seq_q < seq_k)
    let batch = 1;
    let seq_q = 2;
    let seq_k = 4;
    let mut data = Vec::with_capacity(batch * seq_q * seq_k);
    for b in 0..batch {
        for i in 0..seq_q {
            for j in 0..seq_k {
                data.push((b as f32) * 0.2 + (i as f32) * 0.05 + (j as f32) * 0.01);
            }
        }
    }

    let values = ndarray::Array3::from_shape_vec((batch, seq_q, seq_k), data).unwrap();
    let input = BoundedTensor::new(values.clone().into_dyn(), values.into_dyn()).unwrap();
    let output = causal_softmax.propagate_ibp(&input).unwrap();

    for b in 0..batch {
        for i in 0..seq_q {
            let mut row_sum_lb = 0.0_f32;
            let mut row_sum_ub = 0.0_f32;
            for j in 0..seq_k {
                let lb = output.lower()[[b, i, j]];
                let ub = output.upper()[[b, i, j]];
                assert!(
                    lb <= ub + 1e-6,
                    "Batch {}, row {}, col {} invalid bounds: lb {} > ub {}",
                    b,
                    i,
                    j,
                    lb,
                    ub
                );
                if j > i {
                    assert!(
                        lb.abs() < 1e-6,
                        "Batch {}, row {}, col {} should be masked to 0 (lb {})",
                        b,
                        i,
                        j,
                        lb
                    );
                    assert!(
                        ub.abs() < 1e-6,
                        "Batch {}, row {}, col {} should be masked to 0 (ub {})",
                        b,
                        i,
                        j,
                        ub
                    );
                } else {
                    assert!(
                        lb >= -1e-6 && ub <= 1.0 + 1e-6,
                        "Batch {}, row {}, col {} bounds should be in [0,1], got [{}, {}]",
                        b,
                        i,
                        j,
                        lb,
                        ub
                    );
                    row_sum_lb += lb;
                    row_sum_ub += ub;
                }
            }
            assert!(
                row_sum_lb <= 1.0 + 1e-4,
                "Batch {}, row {} unmasked lower sum should be <= 1.0, got {}",
                b,
                i,
                row_sum_lb
            );
            assert!(
                row_sum_ub >= 1.0 - 1e-4,
                "Batch {}, row {} unmasked upper sum should be >= 1.0, got {}",
                b,
                i,
                row_sum_ub
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_layer_4d_basic() {
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    // 4D input: [batch=1, heads=2, seq_q=3, seq_k=3]
    let batch = 1;
    let heads = 2;
    let seq_q = 3;
    let seq_k = 3;
    let mut data = Vec::with_capacity(batch * heads * seq_q * seq_k);
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq_q {
                for j in 0..seq_k {
                    data.push(
                        (b as f32) * 0.1
                            + (h as f32) * 0.02
                            + (i as f32) * 0.01
                            + (j as f32) * 0.001,
                    );
                }
            }
        }
    }

    let values = ndarray::Array4::from_shape_vec((batch, heads, seq_q, seq_k), data).unwrap();
    let input = BoundedTensor::new(values.clone().into_dyn(), values.into_dyn()).unwrap();
    let output = causal_softmax.propagate_ibp(&input).unwrap();

    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq_q {
                let mut row_sum_lb = 0.0_f32;
                let mut row_sum_ub = 0.0_f32;
                for j in 0..seq_k {
                    let lb = output.lower()[[b, h, i, j]];
                    let ub = output.upper()[[b, h, i, j]];
                    assert!(
                        lb <= ub + 1e-6,
                        "Batch {}, head {}, row {}, col {} invalid bounds: lb {} > ub {}",
                        b,
                        h,
                        i,
                        j,
                        lb,
                        ub
                    );
                    if j > i {
                        assert!(
                            lb.abs() < 1e-6,
                            "Batch {}, head {}, row {}, col {} should be masked to 0 (lb {})",
                            b,
                            h,
                            i,
                            j,
                            lb
                        );
                        assert!(
                            ub.abs() < 1e-6,
                            "Batch {}, head {}, row {}, col {} should be masked to 0 (ub {})",
                            b,
                            h,
                            i,
                            j,
                            ub
                        );
                    } else {
                        assert!(
                            lb >= -1e-6 && ub <= 1.0 + 1e-6,
                            "Batch {}, head {}, row {}, col {} bounds should be in [0,1], got [{}, {}]",
                            b,
                            h,
                            i,
                            j,
                            lb,
                            ub
                        );
                        row_sum_lb += lb;
                        row_sum_ub += ub;
                    }
                }
                assert!(
                    row_sum_lb <= 1.0 + 1e-4,
                    "Batch {}, head {}, row {} unmasked lower sum should be <= 1.0, got {}",
                    b,
                    h,
                    i,
                    row_sum_lb
                );
                assert!(
                    row_sum_ub >= 1.0 - 1e-4,
                    "Batch {}, head {}, row {} unmasked upper sum should be >= 1.0, got {}",
                    b,
                    h,
                    i,
                    row_sum_ub
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_layer_3d_bounds_with_eps() {
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let batch = 2;
    let seq_q = 3;
    let seq_k = 3;
    let mut data = Vec::with_capacity(batch * seq_q * seq_k);
    for b in 0..batch {
        for i in 0..seq_q {
            for j in 0..seq_k {
                data.push((b as f32) * 0.15 + (i as f32) * 0.07 + (j as f32) * 0.03);
            }
        }
    }

    let center = ndarray::Array3::from_shape_vec((batch, seq_q, seq_k), data).unwrap();
    let eps = 0.05_f32;
    let lower = center.mapv(|v| v - eps);
    let upper = center.mapv(|v| v + eps);
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap();
    let output = causal_softmax.propagate_ibp(&input).unwrap();

    for b in 0..batch {
        for i in 0..seq_q {
            let mut row_sum_lb = 0.0_f32;
            let mut row_sum_ub = 0.0_f32;
            for j in 0..seq_k {
                let lb = output.lower()[[b, i, j]];
                let ub = output.upper()[[b, i, j]];
                assert!(
                    lb <= ub + 1e-6,
                    "Batch {}, row {}, col {} invalid bounds: lb {} > ub {}",
                    b,
                    i,
                    j,
                    lb,
                    ub
                );
                if j > i {
                    assert!(
                        lb.abs() < 1e-6,
                        "Batch {}, row {}, col {} should be masked to 0 (lb {})",
                        b,
                        i,
                        j,
                        lb
                    );
                    assert!(
                        ub.abs() < 1e-6,
                        "Batch {}, row {}, col {} should be masked to 0 (ub {})",
                        b,
                        i,
                        j,
                        ub
                    );
                } else {
                    assert!(
                        lb >= -1e-6 && ub <= 1.0 + 1e-6,
                        "Batch {}, row {}, col {} bounds should be in [0,1], got [{}, {}]",
                        b,
                        i,
                        j,
                        lb,
                        ub
                    );
                    row_sum_lb += lb;
                    row_sum_ub += ub;
                }
            }
            assert!(
                row_sum_lb <= 1.0 + 1e-4,
                "Batch {}, row {} unmasked lower sum should be <= 1.0, got {}",
                b,
                i,
                row_sum_lb
            );
            assert!(
                row_sum_ub >= 1.0 - 1e-4,
                "Batch {}, row {} unmasked upper sum should be >= 1.0, got {}",
                b,
                i,
                row_sum_ub
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_layer_4d_bounds_with_eps() {
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let batch = 1;
    let heads = 2;
    let seq_q = 3;
    let seq_k = 3;
    let mut data = Vec::with_capacity(batch * heads * seq_q * seq_k);
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq_q {
                for j in 0..seq_k {
                    data.push(
                        (b as f32) * 0.12
                            + (h as f32) * 0.06
                            + (i as f32) * 0.03
                            + (j as f32) * 0.01,
                    );
                }
            }
        }
    }

    let center = ndarray::Array4::from_shape_vec((batch, heads, seq_q, seq_k), data).unwrap();
    let eps = 0.04_f32;
    let lower = center.mapv(|v| v - eps);
    let upper = center.mapv(|v| v + eps);
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap();
    let output = causal_softmax.propagate_ibp(&input).unwrap();

    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq_q {
                let mut row_sum_lb = 0.0_f32;
                let mut row_sum_ub = 0.0_f32;
                for j in 0..seq_k {
                    let lb = output.lower()[[b, h, i, j]];
                    let ub = output.upper()[[b, h, i, j]];
                    assert!(
                        lb <= ub + 1e-6,
                        "Batch {}, head {}, row {}, col {} invalid bounds: lb {} > ub {}",
                        b,
                        h,
                        i,
                        j,
                        lb,
                        ub
                    );
                    if j > i {
                        assert!(
                            lb.abs() < 1e-6,
                            "Batch {}, head {}, row {}, col {} should be masked to 0 (lb {})",
                            b,
                            h,
                            i,
                            j,
                            lb
                        );
                        assert!(
                            ub.abs() < 1e-6,
                            "Batch {}, head {}, row {}, col {} should be masked to 0 (ub {})",
                            b,
                            h,
                            i,
                            j,
                            ub
                        );
                    } else {
                        assert!(
                            lb >= -1e-6 && ub <= 1.0 + 1e-6,
                            "Batch {}, head {}, row {}, col {} bounds should be in [0,1], got [{}, {}]",
                            b,
                            h,
                            i,
                            j,
                            lb,
                            ub
                        );
                        row_sum_lb += lb;
                        row_sum_ub += ub;
                    }
                }
                assert!(
                    row_sum_lb <= 1.0 + 1e-4,
                    "Batch {}, head {}, row {} unmasked lower sum should be <= 1.0, got {}",
                    b,
                    h,
                    i,
                    row_sum_lb
                );
                assert!(
                    row_sum_ub >= 1.0 - 1e-4,
                    "Batch {}, head {}, row {} unmasked upper sum should be >= 1.0, got {}",
                    b,
                    h,
                    i,
                    row_sum_ub
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_layer_invalid_seq_q_gt_seq_k() {
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let batch = 1;
    let seq_q = 4;
    let seq_k = 2;
    let mut data = Vec::with_capacity(batch * seq_q * seq_k);
    for b in 0..batch {
        for i in 0..seq_q {
            for j in 0..seq_k {
                data.push((b as f32) * 0.2 + (i as f32) * 0.05 + (j as f32) * 0.01);
            }
        }
    }

    let values = ndarray::Array3::from_shape_vec((batch, seq_q, seq_k), data).unwrap();
    let input = BoundedTensor::new(values.clone().into_dyn(), values.into_dyn()).unwrap();
    let result = causal_softmax.propagate_ibp(&input);

    assert!(
        matches!(result, Err(NyError::InvalidSpec(_))),
        "Expected InvalidSpec for seq_q > seq_k, got {:?}",
        result
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_sound_mode_uses_ibp_constant_bounds() {
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_sound_mode(true);

    let lower = arr2(&[[0.0_f32, 1.0, 2.0], [0.5, 1.5, 2.5], [1.0, 2.0, 3.0]]).into_dyn();
    let upper = arr2(&[[0.4_f32, 1.4, 2.4], [0.9, 1.9, 2.9], [1.4, 2.4, 3.4]]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let numel = input.lower().len();
    let tol = 1e-5_f32;

    let linear_bounds = LinearBounds::identity(numel);
    let result = causal_softmax
        .propagate_linear_with_bounds(&linear_bounds, &input, causal_softmax.soundness_mode())
        .unwrap();

    assert_eq!(result.lower_a.nrows(), numel);
    assert_eq!(result.lower_a.ncols(), numel);
    assert_eq!(result.upper_a.nrows(), numel);
    assert_eq!(result.upper_a.ncols(), numel);
    assert_eq!(result.lower_b.len(), numel);
    assert_eq!(result.upper_b.len(), numel);

    assert!(
        result.lower_a.iter().all(|v| v.abs() <= tol),
        "Expected zero lower_a in causal sound-mode constant bounds"
    );
    assert!(
        result.upper_a.iter().all(|v| v.abs() <= tol),
        "Expected zero upper_a in causal sound-mode constant bounds"
    );

    let ibp_bounds = causal_softmax.propagate_ibp(&input).unwrap();
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
    let concretized = result.concretize(&input);
    let concretized_flat = concretized.flatten();
    let concretized_lower = concretized_flat
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let concretized_upper = concretized_flat
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    assert_eq!(ibp_lower.len(), numel, "IBP lower length mismatch");
    assert_eq!(ibp_upper.len(), numel, "IBP upper length mismatch");
    assert_eq!(
        concretized_lower.len(),
        numel,
        "Concretized lower length mismatch"
    );
    assert_eq!(
        concretized_upper.len(),
        numel,
        "Concretized upper length mismatch"
    );

    for i in 0..numel {
        assert!(
            (result.lower_b[i] - ibp_lower[i]).abs() <= tol,
            "Causal sound-mode lower_b mismatch at {}: {} vs {}",
            i,
            result.lower_b[i],
            ibp_lower[i]
        );
        assert!(
            (result.upper_b[i] - ibp_upper[i]).abs() <= tol,
            "Causal sound-mode upper_b mismatch at {}: {} vs {}",
            i,
            result.upper_b[i],
            ibp_upper[i]
        );
        assert!(
            (concretized_lower[i] - ibp_lower[i]).abs() <= tol,
            "Causal sound-mode concretized lower mismatch at {}: {} vs {}",
            i,
            concretized_lower[i],
            ibp_lower[i]
        );
        assert!(
            (concretized_upper[i] - ibp_upper[i]).abs() <= tol,
            "Causal sound-mode concretized upper mismatch at {}: {} vs {}",
            i,
            concretized_upper[i],
            ibp_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_layer_soundness() {
    // Test that causal softmax bounds are sound under perturbation
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let eps = 0.1;

    let center = arr2(&[[0.0, 1.0, 2.0], [0.5, 1.5, 2.5], [1.0, 2.0, 3.0]]);
    let lower = center.mapv(|v| v - eps);
    let upper = center.mapv(|v| v + eps);
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap();

    let output = causal_softmax.propagate_ibp(&input).unwrap();

    // Verify bounds are valid (lower <= upper)
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                output.lower()[[i, j]] <= output.upper()[[i, j]] + 1e-6,
                "Invalid bounds at [{}, {}]: {} > {}",
                i,
                j,
                output.lower()[[i, j]],
                output.upper()[[i, j]]
            );
        }
    }

    // Verify masked positions are exactly 0
    assert!(
        output.upper()[[0, 1]].abs() < 1e-6,
        "Masked position [0,1] should be 0"
    );
    assert!(
        output.upper()[[0, 2]].abs() < 1e-6,
        "Masked position [0,2] should be 0"
    );
    assert!(
        output.upper()[[1, 2]].abs() < 1e-6,
        "Masked position [1,2] should be 0"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_layer_ibp_handles_nonfinite_bounds() {
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    // 2D input: [seq_q=3, seq_k=3]. Row 2 includes an infinite bound in an unmasked position.
    let lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let mut upper = lower.clone();
    upper[[2, 1]] = f32::INFINITY;

    // Use new_unchecked to bypass debug_asserts - this test intentionally uses Inf
    let input = BoundedTensor::new_unchecked(lower.into_dyn(), upper.into_dyn()).unwrap();
    let output = causal_softmax.propagate_ibp(&input).unwrap();

    // Masked positions remain exactly 0.
    assert!(output.lower()[[0, 1]].abs() < 1e-6);
    assert!(output.upper()[[0, 1]].abs() < 1e-6);
    assert!(output.lower()[[0, 2]].abs() < 1e-6);
    assert!(output.upper()[[0, 2]].abs() < 1e-6);
    assert!(output.lower()[[1, 2]].abs() < 1e-6);
    assert!(output.upper()[[1, 2]].abs() < 1e-6);

    // Row 2 unmasked positions are sanitized to [0, 1].
    for j in 0..3 {
        assert_eq!(output.lower()[[2, j]], 0.0);
        assert_eq!(output.upper()[[2, j]], 1.0);
    }

    // No NaNs are propagated.
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                !output.lower()[[i, j]].is_nan(),
                "NaN lower at [{},{}]",
                i,
                j
            );
            assert!(
                !output.upper()[[i, j]].is_nan(),
                "NaN upper at [{},{}]",
                i,
                j
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_sound_mode_handles_nonfinite_bounds() {
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_sound_mode(true);

    // 2D input: [seq_q=3, seq_k=3]. Row 2 includes an infinite bound in an unmasked position.
    let lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let mut upper = lower.clone();
    upper[[2, 0]] = f32::INFINITY;

    // Use new_unchecked to bypass debug_asserts - this test intentionally uses Inf
    let input = BoundedTensor::new_unchecked(lower.into_dyn(), upper.into_dyn()).unwrap();
    let numel = input.lower().len();
    let linear_bounds = LinearBounds::identity(numel);
    let sound_bounds = causal_softmax
        .propagate_linear_with_bounds(&linear_bounds, &input, causal_softmax.soundness_mode())
        .unwrap();

    assert!(
        sound_bounds.lower_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero lower_a for causal sound-mode nonfinite fallback"
    );
    assert!(
        sound_bounds.upper_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero upper_a for causal sound-mode nonfinite fallback"
    );

    let ibp_bounds = causal_softmax.propagate_ibp(&input).unwrap().flatten();
    let ibp_lower = ibp_bounds
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let ibp_upper = ibp_bounds
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    for i in 0..ibp_lower.len() {
        assert!(
            (sound_bounds.lower_b[i] - ibp_lower[i]).abs() <= 1e-6,
            "Causal sound-mode lower_b mismatch at {}: {} vs {}",
            i,
            sound_bounds.lower_b[i],
            ibp_lower[i]
        );
        assert!(
            (sound_bounds.upper_b[i] - ibp_upper[i]).abs() <= 1e-6,
            "Causal sound-mode upper_b mismatch at {}: {} vs {}",
            i,
            sound_bounds.upper_b[i],
            ibp_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_crown_backward_basic() {
    // Test CausalSoftmax CROWN backward propagation
    // 2D input: [seq_q=2, seq_k=3]
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 0.5, 1.5, 2.5]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity linear bounds for 6 elements (2*3)
    let linear_bounds = LinearBounds::identity(6);
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let result = causal_softmax
        .propagate_linear_with_bounds(
            &linear_bounds,
            &pre_activation,
            causal_softmax.soundness_mode(),
        )
        .unwrap();

    // Check dimensions
    assert_eq!(result.lower_a.shape(), &[6, 6]);
    assert_eq!(result.upper_a.shape(), &[6, 6]);
    assert_eq!(result.lower_b.len(), 6);
    assert_eq!(result.upper_b.len(), 6);

    // The Jacobian is block diagonal (each row is independent)
    // Row 0: softmax over position 0 only (masked: 1, 2)
    // Row 1: softmax over positions 0, 1 (masked: 2)

    // Check row 0 structure: only position [0,0] affects output [0,0]
    // Positions [0,1] and [0,2] are masked (output=0), so Jacobian is 0
    for k in 3..6 {
        // Row 0 outputs don't depend on row 1 inputs
        for j in 0..3 {
            assert!(
                result.lower_a[[j, k]].abs() < 1e-5,
                "Row 0 output {} should not depend on row 1 input {}",
                j,
                k
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_crown_sampling_check() {
    // Heuristic sampling check for CROWN bounds (not a proof of soundness).
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 1.0, -0.5, 0.5, 1.5]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 0.5, 1.5, 2.5]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let linear_bounds = LinearBounds::identity(6);
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let result = causal_softmax
        .propagate_linear_with_bounds(
            &linear_bounds,
            &pre_activation,
            causal_softmax.soundness_mode(),
        )
        .unwrap();

    // Sample points to spot-check that bounds contain actual values.
    for sample in 0..20 {
        // Generate a random point in the interval
        let point: Vec<f32> = (0..6)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                let pre_l = pre_lower.as_slice().unwrap()[i];
                let pre_u = pre_upper.as_slice().unwrap()[i];
                pre_l + (pre_u - pre_l) * t
            })
            .collect();

        // Compute actual causal softmax output
        // Row 0: softmax over position 0 only
        let row0_exp0 = point[0].exp();
        let row0_sum = row0_exp0 + 1e-8;
        let causal_output = [
            row0_exp0 / row0_sum, // [0,0]
            0.0,                  // [0,1] - masked
            0.0,                  // [0,2] - masked
            // Row 1: softmax over positions 0, 1
            {
                let max_val = point[3].max(point[4]);
                let exp0 = (point[3] - max_val).exp();
                let exp1 = (point[4] - max_val).exp();
                exp0 / (exp0 + exp1 + 1e-8)
            },
            {
                let max_val = point[3].max(point[4]);
                let exp0 = (point[3] - max_val).exp();
                let exp1 = (point[4] - max_val).exp();
                exp1 / (exp0 + exp1 + 1e-8)
            },
            0.0, // [1,2] - masked
        ];

        // Check each output dimension
        for (j, &causal_val) in causal_output.iter().enumerate() {
            let lb_val: f32 = (0..6)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..6)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 5e-2; // Sampling-based CROWN heuristic — tighter than 0.15 but allows heuristic slack
            assert!(
                lb_val <= causal_val + tol,
                "CROWN lower bound violated at sample {}, dim {}: lb {} > actual {}",
                sample,
                j,
                lb_val,
                causal_val
            );
            assert!(
                ub_val >= causal_val - tol,
                "CROWN upper bound violated at sample {}, dim {}: ub {} < actual {}",
                sample,
                j,
                ub_val,
                causal_val
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_crown_network_integration() {
    // Test CausalSoftmax CROWN in a network context
    use crate::layers::LinearLayer;
    use crate::network::Network;

    // Create a simple network: Linear -> CausalSoftmax
    // Input: 6 -> reshape as [2, 3] for causal softmax
    let weight = Array2::from_shape_vec(
        (6, 4),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let bias: Option<Array1<f32>> = Some(Array1::zeros(6));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    // Add reshape to [2, 3]
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![2, 3])));
    network.add_layer(Layer::CausalSoftmax(causal_softmax));

    // Create input bounds
    let input_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.5; 4]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.5; 4]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    // Test CROWN propagation
    let crown_result = network.propagate_crown(&input).unwrap();

    // Test IBP propagation for comparison
    let _ibp_result = network.propagate_ibp(&input).unwrap();

    // CROWN bounds should be at least as tight as (or equal to) IBP bounds
    // Allow some tolerance since both methods have approximation errors
    // Output shape is [2, 3] from the reshape
    for i in 0..2 {
        for j in 0..3 {
            // Both should produce valid bounds in [0, 1] range for softmax
            assert!(
                crown_result.lower()[[i, j]] >= -0.01,
                "CROWN lower bound [{}, {}] = {} should be >= 0",
                i,
                j,
                crown_result.lower()[[i, j]]
            );
            assert!(
                crown_result.upper()[[i, j]] <= 1.01,
                "CROWN upper bound [{}, {}] = {} should be <= 1",
                i,
                j,
                crown_result.upper()[[i, j]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_causal_softmax_crown_masked_positions() {
    // Verify that masked positions have bounds containing 0
    let pre_lower = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![
            0.0, 1.0, 2.0, // Row 0: only position 0 unmasked
            0.0, 1.0, 2.0, // Row 1: positions 0,1 unmasked
            0.0, 1.0, 2.0, // Row 2: all unmasked
        ],
    )
    .unwrap();
    let pre_upper = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    )
    .unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(9);
    let causal_softmax = CausalSoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let result = causal_softmax
        .propagate_linear_with_bounds(
            &linear_bounds,
            &pre_activation,
            causal_softmax.soundness_mode(),
        )
        .unwrap();

    // Masked positions should have bounds containing 0
    // Position [0,1] and [0,2] are masked (row 0)
    // Position [1,2] is masked (row 1)
    // All positions in row 2 are unmasked

    // For masked positions, verify the bounds can contain 0
    let masked_indices = vec![1, 2, 5]; // [0,1], [0,2], [1,2]
    for &idx in &masked_indices {
        // The actual output at masked positions is exactly 0
        // So bounds should contain 0 (lb <= 0 <= ub)
        let lb = result.lower_b[idx]; // With identity bounds and zero input center
        let ub = result.upper_b[idx];
        // At center point, the output is 0 for masked positions
        // The bounds should reflect this
        assert!(
            lb <= 0.1,
            "Lower bound at masked position {} should allow 0, got {}",
            idx,
            lb
        );
        assert!(
            ub >= -0.1,
            "Upper bound at masked position {} should allow 0, got {}",
            idx,
            ub
        );
    }
}
