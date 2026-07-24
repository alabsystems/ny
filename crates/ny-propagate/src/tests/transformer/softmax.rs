// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Softmax CROWN and batched integration tests.

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_softmax_eval() {
    // Test softmax evaluation at a concrete point
    let softmax = SoftmaxLayer::new(-1);
    let x = arr1(&[1.0_f32, 2.0, 3.0]);
    let s = softmax.eval(&x);

    // Softmax outputs should sum to 1
    let sum: f32 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Softmax sum {} != 1.0", sum);

    // Check expected softmax values (exp normalized)
    // exp([1,2,3]) = [e, e^2, e^3]
    // sum = e + e^2 + e^3 ≈ 30.19
    // softmax ≈ [0.090, 0.245, 0.665]
    assert!((s[0] - 0.090).abs() < 0.01, "s[0] = {}", s[0]);
    assert!((s[1] - 0.245).abs() < 0.01, "s[1] = {}", s[1]);
    assert!((s[2] - 0.665).abs() < 0.01, "s[2] = {}", s[2]);
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_jacobian() {
    // Test softmax Jacobian computation
    let softmax = SoftmaxLayer::new(-1);
    let x = arr1(&[0.0_f32, 1.0, 2.0]);
    let s = softmax.eval(&x);
    let j = softmax.jacobian(&x);

    // Check Jacobian dimensions
    assert_eq!(j.shape(), &[3, 3]);

    // Check diagonal elements: J[i,i] = s[i] * (1 - s[i])
    for i in 0..3 {
        let expected = s[i] * (1.0 - s[i]);
        assert!(
            (j[[i, i]] - expected).abs() < 1e-6,
            "J[{},{}] = {} != {}",
            i,
            i,
            j[[i, i]],
            expected
        );
    }

    // Check off-diagonal elements: J[i,j] = -s[i] * s[j]
    for i in 0..3 {
        for j_idx in 0..3 {
            if i != j_idx {
                let expected = -s[i] * s[j_idx];
                assert!(
                    (j[[i, j_idx]] - expected).abs() < 1e-6,
                    "J[{},{}] = {} != {}",
                    i,
                    j_idx,
                    j[[i, j_idx]],
                    expected
                );
            }
        }
    }

    // Check row sums are zero (property of softmax Jacobian)
    for i in 0..3 {
        let row_sum: f32 = (0..3).map(|jj| j[[i, jj]]).sum();
        assert!(row_sum.abs() < 1e-6, "Row {} sum = {} != 0", i, row_sum);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_crown_sampling_check() {
    // Heuristic sampling check for softmax CROWN bounds (not a proof of soundness).
    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);

    // Create bounded input
    let lower = arr1(&[0.0_f32, 1.0, 2.0]).into_dyn();
    let upper = arr1(&[0.5_f32, 1.5, 2.5]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Create identity linear bounds
    let linear_bounds = LinearBounds::identity(3);

    // Get CROWN bounds
    let crown_result = softmax
        .propagate_linear_with_bounds(&linear_bounds, &input, softmax.soundness_mode())
        .unwrap();
    let crown_bounds = crown_result.concretize(&input);

    // Sample points to spot-check that bounds contain actual values.
    for sample_idx in 0..50 {
        let t0 = (sample_idx as f32 * 17.0 % 50.0) / 50.0;
        let t1 = (sample_idx as f32 * 31.0 % 50.0) / 50.0;
        let t2 = (sample_idx as f32 * 47.0 % 50.0) / 50.0;

        let x_sample = arr1(&[0.0 + 0.5 * t0, 1.0 + 0.5 * t1, 2.0 + 0.5 * t2]);

        let s_sample = softmax.eval(&x_sample);

        for i in 0..3 {
            assert!(
                s_sample[i] >= crown_bounds.lower()[[i]] - 1e-5,
                "Sample {} softmax[{}] = {} < lower bound {}",
                sample_idx,
                i,
                s_sample[i],
                crown_bounds.lower()[[i]]
            );
            assert!(
                s_sample[i] <= crown_bounds.upper()[[i]] + 1e-5,
                "Sample {} softmax[{}] = {} > upper bound {}",
                sample_idx,
                i,
                s_sample[i],
                crown_bounds.upper()[[i]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_sound_mode_lse_bounds_sound() {
    let softmax = SoftmaxLayer::new(-1).with_sound_mode(true);

    let lower = arr1(&[0.0_f32, 1.0, 2.0]).into_dyn();
    let upper = arr1(&[0.5_f32, 1.5, 2.5]).into_dyn();
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
                "Sound bound lower violated at sample {} dim {}: {} < {}",
                sample,
                i,
                softmax_val[i],
                concretized.lower()[[i]]
            );
            assert!(
                softmax_val[i] <= concretized.upper()[[i]] + 1e-5,
                "Sound bound upper violated at sample {} dim {}: {} > {}",
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
fn test_softmax_sound_mode_lse_bounds_sampled() {
    let softmax = SoftmaxLayer::new(-1).with_sound_mode(true);

    let lower = arr1(&[-1.0_f32, 0.0, 1.0]).into_dyn();
    let upper = arr1(&[-0.5_f32, 0.5, 1.5]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let sound_bounds = softmax
        .propagate_linear_with_bounds(&linear_bounds, &input, softmax.soundness_mode())
        .unwrap();

    let concretized = sound_bounds.concretize(&input);
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
                "Sound bound lower violated at sample {} dim {}: {} < {}",
                sample,
                i,
                softmax_val[i],
                concretized.lower()[[i]]
            );
            assert!(
                softmax_val[i] <= concretized.upper()[[i]] + 1e-5,
                "Sound bound upper violated at sample {} dim {}: {} > {}",
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
fn test_softmax_crown_comparable_to_ibp() {
    // Test that softmax CROWN bounds are comparable to IBP
    // Note: Due to local linearization with safety margin, CROWN may sometimes
    // be slightly looser than IBP for very small perturbations, but should be
    // sound and within a reasonable factor.
    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);

    // Create bounded input with moderate perturbation
    let lower = arr1(&[0.0_f32, 1.0, 2.0]).into_dyn();
    let upper = arr1(&[0.5_f32, 1.5, 2.5]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Get IBP bounds
    let ibp_bounds = softmax.propagate_ibp(&input).unwrap();

    // Get CROWN bounds
    let linear_bounds = LinearBounds::identity(3);
    let crown_result = softmax
        .propagate_linear_with_bounds(&linear_bounds, &input, softmax.soundness_mode())
        .unwrap();
    let crown_bounds = crown_result.concretize(&input);

    // Both should be sound (verified by sampling test)
    // Check that CROWN is within 2x of IBP (reasonable for local linearization)
    let ibp_range: f32 = (0..3)
        .map(|i| ibp_bounds.upper()[[i]] - ibp_bounds.lower()[[i]])
        .sum();
    let crown_range: f32 = (0..3)
        .map(|i| crown_bounds.upper()[[i]] - crown_bounds.lower()[[i]])
        .sum();

    // CROWN should be within reasonable factor of IBP
    assert!(
        crown_range <= ibp_range * 2.0 + 1e-2,
        "CROWN range {} should be within 2x of IBP range {}",
        crown_range,
        ibp_range
    );

    // Both should give valid probability bounds [0, 1]
    for i in 0..3 {
        assert!(ibp_bounds.lower()[[i]] >= 0.0 - 1e-6);
        assert!(ibp_bounds.upper()[[i]] <= 1.0 + 1e-6);
        assert!(crown_bounds.lower()[[i]] >= 0.0 - 1e-2); // CROWN linear relaxation may slightly exceed [0,1]
        assert!(crown_bounds.upper()[[i]] <= 1.0 + 1e-2);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_with_softmax() {
    // Test that GraphNetwork CROWN works with Softmax
    let mut graph = GraphNetwork::new();

    // Linear -> Softmax network
    let linear = LinearLayer::new(Array2::eye(4), Some(arr1(&[0.0, 0.0, 0.0, 0.0]))).unwrap();
    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);

    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(linear),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "softmax",
        Layer::Softmax(softmax),
        vec!["linear".to_string()],
    ));
    graph.set_output("softmax");

    // Create bounded input
    let lower = arr1(&[0.0_f32, 1.0, 2.0, 3.0]).into_dyn();
    let upper = arr1(&[0.5_f32, 1.5, 2.5, 3.5]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Test IBP
    let ibp_result = graph.propagate_ibp(&input).unwrap();

    // Test CROWN
    let crown_result = graph.propagate_crown(&input).unwrap();

    // Both should give valid probability bounds
    for i in 0..4 {
        assert!(ibp_result.lower()[[i]] >= 0.0, "IBP lower[{}] < 0", i);
        assert!(ibp_result.upper()[[i]] <= 1.0, "IBP upper[{}] > 1", i);
        assert!(
            crown_result.lower()[[i]] >= 0.0 - 1e-5,
            "CROWN lower[{}] < 0",
            i
        );
        assert!(
            crown_result.upper()[[i]] <= 1.0 + 1e-5,
            "CROWN upper[{}] > 1",
            i
        );
    }

    // CROWN should be at least as tight as IBP on average
    let ibp_range: f32 = (0..4)
        .map(|i| ibp_result.upper()[[i]] - ibp_result.lower()[[i]])
        .sum();
    let crown_range: f32 = (0..4)
        .map(|i| crown_result.upper()[[i]] - crown_result.lower()[[i]])
        .sum();

    assert!(
        crown_range <= ibp_range + 1e-2, // Small tolerance for numerical errors
        "CROWN range {} should be <= IBP range {}",
        crown_range,
        ibp_range
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_network_crown_with_softmax() {
    // Test that Network CROWN works with Softmax (sequential)
    let linear = LinearLayer::new(Array2::eye(4), Some(arr1(&[0.0, 0.0, 0.0, 0.0]))).unwrap();
    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::Softmax(softmax));

    // Create bounded input
    let lower = arr1(&[0.0_f32, 1.0, 2.0, 3.0]).into_dyn();
    let upper = arr1(&[0.5_f32, 1.5, 2.5, 3.5]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Test CROWN
    let crown_result = network.propagate_crown(&input).unwrap();

    // Should give valid probability bounds
    for i in 0..4 {
        assert!(
            crown_result.lower()[[i]] >= 0.0 - 1e-5,
            "CROWN lower[{}] < 0: {}",
            i,
            crown_result.lower()[[i]]
        );
        assert!(
            crown_result.upper()[[i]] <= 1.0 + 1e-5,
            "CROWN upper[{}] > 1: {}",
            i,
            crown_result.upper()[[i]]
        );
    }

    // Sample points to spot-check that bounds contain actual values.
    let softmax_layer = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    for sample_idx in 0..20 {
        let t0 = (sample_idx as f32 * 13.0 % 20.0) / 20.0;
        let t1 = (sample_idx as f32 * 17.0 % 20.0) / 20.0;
        let t2 = (sample_idx as f32 * 23.0 % 20.0) / 20.0;
        let t3 = (sample_idx as f32 * 29.0 % 20.0) / 20.0;

        let x_sample = arr1(&[
            0.0 + 0.5 * t0,
            1.0 + 0.5 * t1,
            2.0 + 0.5 * t2,
            3.0 + 0.5 * t3,
        ]);

        let s_sample = softmax_layer.eval(&x_sample);

        for i in 0..4 {
            assert!(
                s_sample[i] >= crown_result.lower()[[i]] - 1e-3,
                "Sample {} softmax[{}] = {} < lower bound {}",
                sample_idx,
                i,
                s_sample[i],
                crown_result.lower()[[i]]
            );
            assert!(
                s_sample[i] <= crown_result.upper()[[i]] + 1e-3,
                "Sample {} softmax[{}] = {} > upper bound {}",
                sample_idx,
                i,
                s_sample[i],
                crown_result.upper()[[i]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_crown_batched_3d_sampling_check() {
    // Test Softmax CROWN with batched 3D inputs: [batch, seq, vocab]
    // Softmax along last axis (vocab)
    let batch = 2_usize;
    let seq = 2_usize;
    let vocab = 4_usize;

    let mut lower = ArrayD::zeros(vec![batch, seq, vocab]);
    let mut upper = ArrayD::zeros(vec![batch, seq, vocab]);

    // Initialize with some spread
    for idx in lower.indexed_iter_mut() {
        let hash = (idx.0[0] as u32 * 100 + idx.0[1] as u32 * 10 + idx.0[2] as u32)
            .wrapping_mul(2654435761_u32);
        let base = (hash as f32 / u32::MAX as f32) * 4.0 - 2.0;
        *idx.1 = base - 0.2;
    }
    for idx in upper.indexed_iter_mut() {
        let hash = (idx.0[0] as u32 * 100 + idx.0[1] as u32 * 10 + idx.0[2] as u32)
            .wrapping_mul(2654435761_u32);
        let base = (hash as f32 / u32::MAX as f32) * 4.0 - 2.0;
        *idx.1 = base + 0.2;
    }

    let pre_bounds = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // Identity bounds for full 3D tensor
    let total_size = batch * seq * vocab;
    let identity_bounds = LinearBounds::identity(total_size);

    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let crown_result = softmax
        .propagate_linear_with_bounds(&identity_bounds, &pre_bounds, softmax.soundness_mode())
        .unwrap();

    // Concretize bounds
    let crown_bounds = crown_result.concretize(&pre_bounds);

    // Sample points to spot-check that bounds contain actual values.
    for sample_idx in 0..20_usize {
        let mut x_sample = ArrayD::zeros(vec![batch, seq, vocab]);

        for idx in x_sample.indexed_iter_mut() {
            let l = lower[idx.0.clone()];
            let u = upper[idx.0.clone()];
            let hash = (sample_idx as u32)
                .wrapping_mul(2654435761_u32)
                .wrapping_add(idx.0[0] as u32 * 1000 + idx.0[1] as u32 * 100 + idx.0[2] as u32);
            let t = hash as f32 / u32::MAX as f32;
            *idx.1 = l + (u - l) * t;
        }

        // Compute softmax for each [batch, seq] position
        let mut s_sample = ArrayD::zeros(vec![batch, seq, vocab]);
        for b_idx in 0..batch {
            for s_idx in 0..seq {
                // Extract 1D slice
                let mut slice = vec![0.0_f32; vocab];
                for v in 0..vocab {
                    slice[v] = x_sample[[b_idx, s_idx, v]];
                }

                // Compute softmax
                let max_x = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_x: Vec<f32> = slice.iter().map(|&xi| (xi - max_x).exp()).collect();
                let sum_exp: f32 = exp_x.iter().sum();
                let softmax: Vec<f32> = exp_x.iter().map(|&ei| ei / sum_exp).collect();

                for v in 0..vocab {
                    s_sample[[b_idx, s_idx, v]] = softmax[v];
                }
            }
        }

        // Spot-check that bounds contain actual values.
        let lower_slice = crown_bounds.lower().as_slice().unwrap();
        let upper_slice = crown_bounds.upper().as_slice().unwrap();
        for (flat, &val) in s_sample.iter().enumerate() {
            assert!(
                val >= lower_slice[flat] - 1e-4,
                "Batched Softmax CROWN lower violation at flat {} sample {}: {} < {}",
                flat,
                sample_idx,
                val,
                lower_slice[flat]
            );
            assert!(
                val <= upper_slice[flat] + 1e-4,
                "Batched Softmax CROWN upper violation at flat {} sample {}: {} > {}",
                flat,
                sample_idx,
                val,
                upper_slice[flat]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_batched_linear_bounds() {
    // Test the new propagate_linear_batched_with_bounds for SoftmaxLayer
    // Input shape: [batch, seq, softmax_size] = [2, 3, 4]
    // This tests the batched CROWN backward through softmax
    let batch = 2_usize;
    let seq = 3_usize;
    let softmax_size = 4_usize;
    let _total_batch = batch * seq;

    // Initialize pre-activation bounds
    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, softmax_size]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, softmax_size]));

    for b in 0..batch {
        for s in 0..seq {
            for k in 0..softmax_size {
                let hash = ((b * 100 + s * 10 + k) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 4.0 - 2.0;
                lower[[b, s, k]] = base - 0.15;
                upper[[b, s, k]] = base + 0.15;
            }
        }
    }

    let pre_bounds = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // Create identity BatchedLinearBounds for the output
    let identity = BatchedLinearBounds::identity(&[batch, seq, softmax_size]).unwrap();

    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);

    // Test the batched method
    let result = softmax
        .propagate_linear_batched_with_bounds(&identity, &pre_bounds, softmax.soundness_mode())
        .unwrap();

    // Concretize to get final bounds
    let final_bounds = result.concretize(&pre_bounds).unwrap();

    // Verify shape
    assert_eq!(final_bounds.shape(), &[batch, seq, softmax_size]);

    // Sample points to spot-check that bounds contain actual values.
    for sample_idx in 0..15_usize {
        let mut x_sample = ArrayD::zeros(IxDyn(&[batch, seq, softmax_size]));

        for b in 0..batch {
            for s in 0..seq {
                for k in 0..softmax_size {
                    let l = lower[[b, s, k]];
                    let u = upper[[b, s, k]];
                    let hash = (sample_idx as u32)
                        .wrapping_mul(2654435761_u32)
                        .wrapping_add((b * 1000 + s * 100 + k) as u32);
                    let t = hash as f32 / u32::MAX as f32;
                    x_sample[[b, s, k]] = l + (u - l) * t;
                }
            }
        }

        // Compute softmax for each [batch, seq] position
        let mut s_sample = ArrayD::zeros(IxDyn(&[batch, seq, softmax_size]));
        for b in 0..batch {
            for s in 0..seq {
                // Extract 1D slice
                let mut slice = vec![0.0_f32; softmax_size];
                for k in 0..softmax_size {
                    slice[k] = x_sample[[b, s, k]];
                }

                // Compute softmax
                let max_x = slice.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
                let exp_x: Vec<f32> = slice.iter().map(|&xi| (xi - max_x).exp()).collect();
                let sum_exp: f32 = exp_x.iter().sum();
                let softmax_vals: Vec<f32> = exp_x.iter().map(|&ei| ei / sum_exp).collect();

                for k in 0..softmax_size {
                    s_sample[[b, s, k]] = softmax_vals[k];
                }
            }
        }

        // Spot-check that bounds contain actual values.
        for b in 0..batch {
            for s in 0..seq {
                for k in 0..softmax_size {
                    let val = s_sample[[b, s, k]];
                    let lb = final_bounds.lower()[[b, s, k]];
                    let ub = final_bounds.upper()[[b, s, k]];
                    assert!(
                        val >= lb - 1e-4,
                        "Batched Softmax CROWN lower violation at [{}, {}, {}] sample {}: {} < {}",
                        b,
                        s,
                        k,
                        sample_idx,
                        val,
                        lb
                    );
                    assert!(
                        val <= ub + 1e-4,
                        "Batched Softmax CROWN upper violation at [{}, {}, {}] sample {}: {} > {}",
                        b,
                        s,
                        k,
                        sample_idx,
                        val,
                        ub
                    );
                }
            }
        }
    }

    // Also print bound widths for diagnostics
    let mut total_width = 0.0_f64;
    let mut count = 0;
    for b in 0..batch {
        for s in 0..seq {
            for k in 0..softmax_size {
                let width =
                    (final_bounds.upper()[[b, s, k]] - final_bounds.lower()[[b, s, k]]) as f64;
                total_width += width;
                count += 1;
            }
        }
    }
    let avg_width = total_width / count as f64;
    println!(
        "Batched Softmax CROWN: shape [{}, {}, {}], avg bound width: {:.4}",
        batch, seq, softmax_size, avg_width
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_layer_ibp_handles_nonfinite_bounds() {
    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);

    // Two rows; second row contains non-finite bounds which should be sanitized to [0, 1].
    let lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let mut upper = lower.clone();
    upper[[1, 2]] = f32::INFINITY;

    // Use new_unchecked to bypass debug_asserts - this test intentionally uses Inf
    let input = BoundedTensor::new_unchecked(lower.into_dyn(), upper.into_dyn()).unwrap();
    let output = softmax.propagate_ibp(&input).unwrap();

    // Row 1: fallback to [0, 1].
    for j in 0..3 {
        assert_eq!(output.lower()[[1, j]], 0.0);
        assert_eq!(output.upper()[[1, j]], 1.0);
    }

    // Row 0: should remain finite and within [0, 1].
    for j in 0..3 {
        let lb = output.lower()[[0, j]];
        let ub = output.upper()[[0, j]];
        assert!(lb.is_finite(), "Row 0 lower should be finite");
        assert!(ub.is_finite(), "Row 0 upper should be finite");
        assert!(lb >= 0.0 - 1e-6, "Row 0 lower should be >= 0, got {}", lb);
        assert!(ub <= 1.0 + 1e-6, "Row 0 upper should be <= 1, got {}", ub);
        assert!(lb <= ub + 1e-6, "Row 0 bounds invalid: {} > {}", lb, ub);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_batched_sound_mode_lse_bounds_sound() {
    let batch = 2_usize;
    let seq = 2_usize;
    let softmax_size = 3_usize;

    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, softmax_size]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, softmax_size]));

    for b in 0..batch {
        for s in 0..seq {
            for k in 0..softmax_size {
                let hash = ((b * 100 + s * 10 + k) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 3.0 - 1.5;
                lower[[b, s, k]] = base - 0.2;
                upper[[b, s, k]] = base + 0.2;
            }
        }
    }

    let pre_bounds = BoundedTensor::new(lower, upper).unwrap();
    let identity = BatchedLinearBounds::identity(&[batch, seq, softmax_size]).unwrap();
    let softmax = SoftmaxLayer::new(-1).with_sound_mode(true);

    let result = softmax
        .propagate_linear_batched_with_bounds(&identity, &pre_bounds, softmax.soundness_mode())
        .unwrap();
    let concretized = result.concretize(&pre_bounds).unwrap();

    for sample in 0..8 {
        let mut x_sample = ArrayD::zeros(IxDyn(&[batch, seq, softmax_size]));
        for b in 0..batch {
            for s in 0..seq {
                for k in 0..softmax_size {
                    let t = ((sample as u32).wrapping_mul(2654435761) ^ (k as u32)) as f32
                        / u32::MAX as f32;
                    let l = pre_bounds.lower()[[b, s, k]];
                    let u = pre_bounds.upper()[[b, s, k]];
                    x_sample[[b, s, k]] = l + (u - l) * t;
                }
            }
        }

        for b in 0..batch {
            for s in 0..seq {
                let mut row = vec![0.0_f32; softmax_size];
                for k in 0..softmax_size {
                    row[k] = x_sample[[b, s, k]];
                }
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = row.iter().map(|&v| (v - max_val).exp()).sum();
                let softmax_vals: Vec<f32> =
                    row.iter().map(|&v| (v - max_val).exp() / exp_sum).collect();

                for (k, &softmax_k) in softmax_vals.iter().enumerate().take(softmax_size) {
                    let lb = concretized.lower()[[b, s, k]];
                    let ub = concretized.upper()[[b, s, k]];
                    assert!(
                        softmax_k >= lb - 1e-5,
                        "Sound lower bound violated at sample {} ({},{},{}): {} < {}",
                        sample,
                        b,
                        s,
                        k,
                        softmax_k,
                        lb
                    );
                    assert!(
                        softmax_k <= ub + 1e-5,
                        "Sound upper bound violated at sample {} ({},{},{}): {} > {}",
                        sample,
                        b,
                        s,
                        k,
                        softmax_k,
                        ub
                    );
                }
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_sound_mode_handles_nonfinite_bounds() {
    let softmax = SoftmaxLayer::new(-1).with_sound_mode(true);

    // Two rows; second row contains non-finite bounds which should fall back to [0, 1].
    let lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let mut upper = lower.clone();
    upper[[1, 1]] = f32::INFINITY;

    // Use new_unchecked to bypass debug_asserts - this test intentionally uses Inf
    let input = BoundedTensor::new_unchecked(lower.into_dyn(), upper.into_dyn()).unwrap();
    let linear_bounds = LinearBounds::identity(input.lower().len());
    let sound_bounds = softmax
        .propagate_linear_with_bounds(&linear_bounds, &input, softmax.soundness_mode())
        .unwrap();

    assert!(
        sound_bounds.lower_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero lower_a for sound-mode nonfinite fallback"
    );
    assert!(
        sound_bounds.upper_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero upper_a for sound-mode nonfinite fallback"
    );

    let ibp_bounds = softmax.propagate_ibp(&input).unwrap().flatten();
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
            "Sound-mode lower_b mismatch at {}: {} vs {}",
            i,
            sound_bounds.lower_b[i],
            ibp_lower[i]
        );
        assert!(
            (sound_bounds.upper_b[i] - ibp_upper[i]).abs() <= 1e-6,
            "Sound-mode upper_b mismatch at {}: {} vs {}",
            i,
            sound_bounds.upper_b[i],
            ibp_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_propagate_crown_batched_with_softmax() {
    // Test batched CROWN on a network with Linear -> Softmax
    // This verifies the Softmax integration in propagate_crown_batched

    let mut network = Network::new();

    // Linear: 4 -> 4
    let weight = Array2::from_shape_fn((4, 4), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        0.3 * phase.sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight, None).unwrap()));

    // Softmax
    network.add_layer(Layer::Softmax(
        SoftmaxLayer::new(-1).with_heuristic_sampling(true),
    ));

    // Input: [batch=2, seq=3, 4]
    let batch = 2;
    let seq = 3;
    let hidden = 4;
    let total_elements = batch * seq * hidden;

    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0;
                lower[[b, s, h]] = base - 0.1;
                upper[[b, s, h]] = base + 0.1;
            }
        }
    }

    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run batched CROWN
    let batched_result = network.propagate_crown_batched(&input).unwrap();

    // Verify output shape
    assert_eq!(batched_result.shape(), &[batch, seq, hidden]);

    // Verify all bounds are finite and valid
    let mut valid_count = 0;
    let mut finite_count = 0;
    for (l, u) in batched_result
        .lower()
        .iter()
        .zip(batched_result.upper().iter())
    {
        if l.is_finite() && u.is_finite() {
            finite_count += 1;
        }
        if *l <= *u + 1e-6 {
            valid_count += 1;
        }
    }

    assert_eq!(finite_count, total_elements, "All bounds should be finite");
    assert_eq!(valid_count, total_elements, "All bounds should be valid");

    // Softmax outputs should be in [0, 1]
    for (l, u) in batched_result
        .lower()
        .iter()
        .zip(batched_result.upper().iter())
    {
        assert!(*l >= -0.01, "Softmax lower bound should be >= 0, got {}", l);
        assert!(*u <= 1.01, "Softmax upper bound should be <= 1, got {}", u);
    }

    // Measure bound widths
    let avg_width: f32 = batched_result
        .lower()
        .iter()
        .zip(batched_result.upper().iter())
        .map(|(l, u)| u - l)
        .sum::<f32>()
        / total_elements as f32;

    println!(
        "Linear+Softmax batched CROWN: shape {:?}, avg bound width: {:.4}",
        batched_result.shape(),
        avg_width
    );

    // Bounds should not explode
    assert!(
        avg_width < 1.0,
        "Bound width should be reasonable (< 1 for softmax), got {}",
        avg_width
    );
}
