// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched backward propagation tests.

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_linear_layer_batched_backward() {
    // Test batched backward propagation through linear layer
    // Linear layer: y = Wx + b, W is [out_dim, in_dim]
    let weight = Array2::from_shape_vec(
        (3, 4),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    )
    .unwrap();
    let bias = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();

    // Create identity bounds at output: shape [batch=2, out_dim=3, out_dim=3]
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();

    let result = linear.propagate_linear_batched(&bounds).unwrap();

    // After backward through linear, we should have:
    // new_A = I @ W = W, shape [2, 3, 4]
    // new_b = I @ bias + 0 = bias, shape [2, 3]
    assert_eq!(result.lower_a.shape(), &[2, 3, 4]);
    assert_eq!(result.lower_b.shape(), &[2, 3]);

    // Check weight is propagated correctly
    // For batch 0, row 0: [1, 0, 0, 0]
    assert!((result.lower_a[[0, 0, 0]] - 1.0).abs() < 1e-5);
    assert!((result.lower_a[[0, 0, 1]] - 0.0).abs() < 1e-5);

    // For batch 0, row 2: [0, 0, 1, 1]
    assert!((result.lower_a[[0, 2, 2]] - 1.0).abs() < 1e-5);
    assert!((result.lower_a[[0, 2, 3]] - 1.0).abs() < 1e-5);

    // Check bias is propagated correctly
    assert!((result.lower_b[[0, 0]] - 0.1).abs() < 1e-5);
    assert!((result.lower_b[[0, 1]] - 0.2).abs() < 1e-5);
    assert!((result.lower_b[[0, 2]] - 0.3).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_layer_batched_backward_positive() {
    // Test batched ReLU backward when all inputs are positive (identity pass-through)
    let relu = ReLULayer;

    // Pre-activation bounds: all positive
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![2.0; 8]).unwrap(),
    )
    .unwrap();

    // Identity bounds at output
    let bounds = BatchedLinearBounds::identity(&[2, 4]).unwrap();

    let result = relu
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // With all positive pre-activation, ReLU is identity
    // So bounds should remain identity
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result.lower_a[[b, i, j]] - expected).abs() < 1e-5,
                    "lower_a[{}, {}, {}] = {}, expected {}",
                    b,
                    i,
                    j,
                    result.lower_a[[b, i, j]],
                    expected
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_layer_batched_backward_negative() {
    // Test batched ReLU backward when all inputs are negative (zero output)
    let relu = ReLULayer;

    // Pre-activation bounds: all negative
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![-2.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![-1.0; 8]).unwrap(),
    )
    .unwrap();

    // Identity bounds at output
    let bounds = BatchedLinearBounds::identity(&[2, 4]).unwrap();

    let result = relu
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // With all negative pre-activation, ReLU outputs zero
    // So all coefficients should be zero
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    result.lower_a[[b, i, j]].abs() < 1e-5,
                    "lower_a[{}, {}, {}] = {}, expected 0",
                    b,
                    i,
                    j,
                    result.lower_a[[b, i, j]]
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_linear_relu_chain() {
    // Test a simple Linear -> ReLU chain with batched bounds
    // This verifies that the batched backward propagation composes correctly

    // Linear layer: 4 -> 4 with identity weight
    let weight = Array2::eye(4);
    let linear = LinearLayer::new(weight, None).unwrap();

    let relu = ReLULayer;

    // Pre-activation bounds: mix of positive, negative, and crossing
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, -2.0, -0.5, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![2.0, -1.0, 0.5, 1.5]).unwrap(),
    )
    .unwrap();

    // Start with identity bounds at output
    let bounds = BatchedLinearBounds::identity(&[1, 4]).unwrap();

    // Backward through ReLU first
    let after_relu = relu
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // Then backward through Linear (identity weight, so should pass through)
    let final_bounds = linear.propagate_linear_batched(&after_relu).unwrap();

    // Verify shapes
    assert_eq!(final_bounds.lower_a.shape(), &[1, 4, 4]);

    // Position 0: always positive -> identity (slope 1)
    assert!((final_bounds.lower_a[[0, 0, 0]] - 1.0).abs() < 1e-5);

    // Position 1: always negative -> zero (slope 0)
    assert!(final_bounds.lower_a[[0, 1, 1]].abs() < 1e-5);

    // Position 2: crossing [-0.5, 0.5] -> linear relaxation
    // lambda = u/(u-l) = 0.5/1.0 = 0.5
    // For lower_a positive, uses alpha (heuristic: u > -l -> 0.5 > 0.5 -> false, alpha=0)
    // Actually u=0.5, -l=0.5, so u == -l, alpha = 0
    assert!(final_bounds.lower_a[[0, 2, 2]].abs() < 1e-5);

    // Position 3: crossing [0.5, 1.5] but positive lower bound -> identity for upper bound coeff >= 0
    // lambda = 1.5/1.0 = 1.5... wait, l=0.5 > 0, so this is always positive!
    // Actually l=0.5 >= 0, so this is identity
    assert!((final_bounds.lower_a[[0, 3, 3]] - 1.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_gelu_layer_batched_backward() {
    // Test batched GELU backward propagation
    let gelu = GELULayer::default();

    // Pre-activation bounds: mix of values
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![-1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0],
        )
        .unwrap(),
    )
    .unwrap();

    // Identity bounds at output
    let bounds = BatchedLinearBounds::identity(&[2, 4]).unwrap();

    let result = gelu
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // Verify shapes preserved
    assert_eq!(result.lower_a.shape(), &[2, 4, 4]);
    assert_eq!(result.lower_b.shape(), &[2, 4]);

    // Verify diagonal structure (off-diagonal should be zero since each output
    // depends only on corresponding input for identity bounds)
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        result.lower_a[[b, i, j]].abs() < 1e-5,
                        "lower_a[{}, {}, {}] should be 0 (off-diagonal), got {}",
                        b,
                        i,
                        j,
                        result.lower_a[[b, i, j]]
                    );
                }
            }
        }
    }

    // For very positive inputs (like x in [1, 2], [2, 3]), GELU slope is close to 1
    // Batch 0, position 2: input [1, 2] - GELU is approximately linear here
    // The diagonal should be close to 1 (but not exactly due to GELU curvature)
    assert!(
        result.lower_a[[0, 2, 2]] > 0.5,
        "GELU slope for positive input should be > 0.5, got {}",
        result.lower_a[[0, 2, 2]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_mlp_chain() {
    // Test a simple MLP: Linear -> GELU -> Linear with batched bounds
    // This represents a transformer MLP block (without the expansion factor)

    // First linear: 4 -> 4 (identity for simplicity)
    let weight1 = Array2::eye(4);
    let linear1 = LinearLayer::new(weight1, None).unwrap();

    // GELU activation
    let gelu = GELULayer::default();

    // Second linear: 4 -> 4 (identity for simplicity)
    let weight2 = Array2::eye(4);
    let linear2 = LinearLayer::new(weight2, None).unwrap();

    // Input bounds: [batch=1, seq=2, hidden=4]
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 4]), vec![-1.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 4]), vec![1.0; 8]).unwrap(),
    )
    .unwrap();

    // Forward pass through network to get pre-activation bounds
    let after_linear1 = linear1.propagate_ibp(&input).unwrap();
    let after_gelu = gelu.propagate_ibp(&after_linear1).unwrap();
    let _after_linear2 = linear2.propagate_ibp(&after_gelu).unwrap();

    // Now backward pass with batched CROWN
    // Start with identity bounds at output: [1, 2, 4, 4]
    let bounds = BatchedLinearBounds::identity(&[1, 2, 4]).unwrap();

    // Backward through linear2
    let after_l2_back = linear2.propagate_linear_batched(&bounds).unwrap();
    assert_eq!(after_l2_back.lower_a.shape(), &[1, 2, 4, 4]);

    // Backward through GELU with pre-activation bounds
    let after_gelu_back = gelu
        .propagate_linear_batched_with_bounds(&after_l2_back, &after_linear1)
        .unwrap();
    assert_eq!(after_gelu_back.lower_a.shape(), &[1, 2, 4, 4]);

    // Backward through linear1
    let final_bounds = linear1.propagate_linear_batched(&after_gelu_back).unwrap();
    assert_eq!(final_bounds.lower_a.shape(), &[1, 2, 4, 4]);

    // Concretize to get concrete bounds
    let concrete = final_bounds.concretize(&input).unwrap();
    assert_eq!(concrete.shape(), &[1, 2, 4]);

    // Verify soundness: concrete bounds should be valid
    // (lower <= actual output <= upper for all inputs in the input bounds)
    assert!(
        concrete.lower().iter().all(|&x| x.is_finite()),
        "All lower bounds should be finite"
    );
    assert!(
        concrete.upper().iter().all(|&x| x.is_finite()),
        "All upper bounds should be finite"
    );
    assert!(
        concrete
            .lower()
            .iter()
            .zip(concrete.upper().iter())
            .all(|(&l, &u)| l <= u),
        "Lower should be <= upper"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_gelu_batched_adaptive_relaxation_soundness() {
    // Test batched GELU backward propagation with adaptive relaxation mode.
    // Verifies that adaptive mode produces sound bounds across different input ranges.
    let gelu_adaptive = GELULayer::adaptive(GeluApproximation::Erf);

    // Pre-activation bounds: varied ranges to exercise different relaxation strategies
    // - Batch 0: includes critical region around GELU minimum (~-0.75)
    // - Batch 1: includes positive region where GELU is nearly linear
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![
                -1.5, -0.5, 0.0, 0.5, // Batch 0: negative, critical, zero, positive
                0.5, 1.0, 1.5, 2.0, // Batch 1: all positive (nearly linear)
            ],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![
                -0.5, 0.5, 1.0, 1.5, // Batch 0: expanded ranges
                1.5, 2.0, 2.5, 3.0, // Batch 1: positive region
            ],
        )
        .unwrap(),
    )
    .unwrap();

    // Identity bounds at output
    let bounds = BatchedLinearBounds::identity(&[2, 4]).unwrap();

    let result = gelu_adaptive
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // Verify shapes preserved
    assert_eq!(result.lower_a.shape(), &[2, 4, 4]);
    assert_eq!(result.lower_b.shape(), &[2, 4]);

    // Verify diagonal structure
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        result.lower_a[[b, i, j]].abs() < 1e-5,
                        "lower_a[{}, {}, {}] should be 0 (off-diagonal), got {}",
                        b,
                        i,
                        j,
                        result.lower_a[[b, i, j]]
                    );
                    assert!(
                        result.upper_a[[b, i, j]].abs() < 1e-5,
                        "upper_a[{}, {}, {}] should be 0 (off-diagonal), got {}",
                        b,
                        i,
                        j,
                        result.upper_a[[b, i, j]]
                    );
                }
            }
        }
    }

    // Verify soundness by sampling concrete inputs
    for b in 0..2 {
        for i in 0..4 {
            let l = pre_activation.lower()[[b, i]];
            let u = pre_activation.upper()[[b, i]];
            let slope = result.lower_a[[b, i, i]];
            let intercept = result.lower_b[[b, i]];
            let u_slope = result.upper_a[[b, i, i]];
            let u_intercept = result.upper_b[[b, i]];

            // Sample points in [l, u] and verify bounds
            for t in 0..=10 {
                let x = l + (u - l) * (t as f32 / 10.0);
                let gelu_x = gelu_eval(x, GeluApproximation::Erf);
                let lower_bound = slope * x + intercept;
                let upper_bound = u_slope * x + u_intercept;

                assert!(
                    gelu_x >= lower_bound - 1e-4,
                    "Batch {}, pos {}: GELU({}) = {} < lower bound {} at x={}",
                    b,
                    i,
                    x,
                    gelu_x,
                    lower_bound,
                    x
                );
                assert!(
                    gelu_x <= upper_bound + 1e-4,
                    "Batch {}, pos {}: GELU({}) = {} > upper bound {} at x={}",
                    b,
                    i,
                    x,
                    gelu_x,
                    upper_bound,
                    x
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_gelu_batched_adaptive_vs_chord_tightness() {
    // Verify that adaptive relaxation produces at least as tight bounds as the
    // heuristic chord within the same (non-sound) path. Both must use sound=false
    // so they share the same epsilon budget — comparing sound vs heuristic is
    // invalid because they have different safety margin strategies.
    let gelu_chord = GELULayer {
        sound: false,
        ..GELULayer::with_relaxation(GeluApproximation::Erf, RelaxationMode::Chord)
    };
    let gelu_adaptive = GELULayer::adaptive(GeluApproximation::Erf);

    // Pre-activation bounds with varied ranges to exercise different relaxation strategies
    // Use different intervals to ensure adaptive selection logic is tested
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![
                -2.0, -0.5, 0.0, 1.0, // Batch 0: wide negative, critical, zero, positive
                -1.0, -0.3, 0.5, 2.0, // Batch 1: mixed ranges
            ],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![
                0.0, 0.5, 1.0, 2.0, // Batch 0: expanded ranges
                0.0, 0.3, 1.5, 3.0, // Batch 1: varied widths
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let bounds = BatchedLinearBounds::identity(&[2, 4]).unwrap();

    let chord_result = gelu_chord
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();
    let adaptive_result = gelu_adaptive
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // Compare bound widths at the center of each input interval
    for b in 0..2 {
        for i in 0..4 {
            let l = pre_activation.lower()[[b, i]];
            let u = pre_activation.upper()[[b, i]];
            let c = f32::midpoint(l, u);

            let chord_lower = chord_result.lower_a[[b, i, i]] * c + chord_result.lower_b[[b, i]];
            let chord_upper = chord_result.upper_a[[b, i, i]] * c + chord_result.upper_b[[b, i]];
            let chord_width = chord_upper - chord_lower;

            let adaptive_lower =
                adaptive_result.lower_a[[b, i, i]] * c + adaptive_result.lower_b[[b, i]];
            let adaptive_upper =
                adaptive_result.upper_a[[b, i, i]] * c + adaptive_result.upper_b[[b, i]];
            let adaptive_width = adaptive_upper - adaptive_lower;

            assert!(
                adaptive_width <= chord_width + 1e-5,
                "Batch {}, pos {}: adaptive width {} > chord width {}",
                b,
                i,
                adaptive_width,
                chord_width
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_gelu_batched_tanh_approximation() {
    // Test batched GELU with Tanh approximation mode.
    // Tanh approximation is commonly used in transformers for efficiency.
    let gelu_tanh = GELULayer::new(GeluApproximation::Tanh);

    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![-2.0, -1.0, 0.0, 1.0, -1.5, -0.5, 0.5, 1.5],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![-1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 1.5, 2.5],
        )
        .unwrap(),
    )
    .unwrap();

    let bounds = BatchedLinearBounds::identity(&[2, 4]).unwrap();

    let result = gelu_tanh
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // Verify shapes
    assert_eq!(result.lower_a.shape(), &[2, 4, 4]);
    assert_eq!(result.lower_b.shape(), &[2, 4]);

    // Verify diagonal structure (off-diagonal elements should be zero)
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        result.lower_a[[b, i, j]].abs() < 1e-5,
                        "Tanh: lower_a[{}, {}, {}] should be 0, got {}",
                        b,
                        i,
                        j,
                        result.lower_a[[b, i, j]]
                    );
                    assert!(
                        result.upper_a[[b, i, j]].abs() < 1e-5,
                        "Tanh: upper_a[{}, {}, {}] should be 0, got {}",
                        b,
                        i,
                        j,
                        result.upper_a[[b, i, j]]
                    );
                }
            }
        }
    }

    // Verify soundness with Tanh approximation
    for b in 0..2 {
        for i in 0..4 {
            let l = pre_activation.lower()[[b, i]];
            let u = pre_activation.upper()[[b, i]];
            let slope = result.lower_a[[b, i, i]];
            let intercept = result.lower_b[[b, i]];
            let u_slope = result.upper_a[[b, i, i]];
            let u_intercept = result.upper_b[[b, i]];

            for t in 0..=10 {
                let x = l + (u - l) * (t as f32 / 10.0);
                let gelu_x = gelu_eval(x, GeluApproximation::Tanh);
                let lower_bound = slope * x + intercept;
                let upper_bound = u_slope * x + u_intercept;

                assert!(
                    gelu_x >= lower_bound - 1e-4,
                    "Tanh: Batch {}, pos {}: GELU({}) = {} < lower bound {}",
                    b,
                    i,
                    x,
                    gelu_x,
                    lower_bound
                );
                assert!(
                    gelu_x <= upper_bound + 1e-4,
                    "Tanh: Batch {}, pos {}: GELU({}) = {} > upper bound {}",
                    b,
                    i,
                    x,
                    gelu_x,
                    upper_bound
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_gelu_batched_3d_transformer_shape() {
    // Test batched GELU with 3D shape [batch, seq, hidden] matching actual transformer patterns.
    // Previous tests used 2D shapes; this verifies proper handling of the extra batch dimension.
    let gelu = GELULayer::adaptive(GeluApproximation::Erf);

    // Shape: [batch=2, seq=3, hidden=4] - realistic transformer dimensions
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 4]),
            vec![
                // Batch 0
                -1.0, -0.5, 0.0, 0.5, // seq 0
                -0.5, 0.0, 0.5, 1.0, // seq 1
                0.0, 0.5, 1.0, 1.5, // seq 2
                // Batch 1
                -1.5, -1.0, -0.5, 0.0, // seq 0
                -0.3, 0.2, 0.7, 1.2, // seq 1
                0.5, 1.0, 1.5, 2.0, // seq 2
            ],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 4]),
            vec![
                // Batch 0
                0.0, 0.5, 1.0, 1.5, // seq 0
                0.5, 1.0, 1.5, 2.0, // seq 1
                1.0, 1.5, 2.0, 2.5, // seq 2
                // Batch 1
                -0.5, 0.0, 0.5, 1.0, // seq 0
                0.7, 1.2, 1.7, 2.2, // seq 1
                1.5, 2.0, 2.5, 3.0, // seq 2
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let bounds = BatchedLinearBounds::identity(&[2, 3, 4]).unwrap();

    let result = gelu
        .propagate_linear_batched_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // Verify shapes: [batch, seq, hidden, hidden]
    assert_eq!(result.lower_a.shape(), &[2, 3, 4, 4]);
    assert_eq!(result.lower_b.shape(), &[2, 3, 4]);

    // Verify soundness by sampling for each batch/seq position
    for b in 0..2 {
        for s in 0..3 {
            for i in 0..4 {
                let l = pre_activation.lower()[[b, s, i]];
                let u = pre_activation.upper()[[b, s, i]];
                let slope = result.lower_a[[b, s, i, i]];
                let intercept = result.lower_b[[b, s, i]];
                let u_slope = result.upper_a[[b, s, i, i]];
                let u_intercept = result.upper_b[[b, s, i]];

                for t in 0..=5 {
                    let x = l + (u - l) * (t as f32 / 5.0);
                    let gelu_x = gelu_eval(x, GeluApproximation::Erf);
                    let lower_bound = slope * x + intercept;
                    let upper_bound = u_slope * x + u_intercept;

                    assert!(
                        gelu_x >= lower_bound - 1e-4,
                        "3D: [{}, {}, {}]: GELU({}) = {} < lower bound {}",
                        b,
                        s,
                        i,
                        x,
                        gelu_x,
                        lower_bound
                    );
                    assert!(
                        gelu_x <= upper_bound + 1e-4,
                        "3D: [{}, {}, {}]: GELU({}) = {} > upper bound {}",
                        b,
                        s,
                        i,
                        x,
                        gelu_x,
                        upper_bound
                    );
                }
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_propagate_crown_batched_with_layernorm() {
    // Test batched CROWN on a network with Linear -> LayerNorm (using sampling mode)
    // This verifies the LayerNorm integration in propagate_crown_batched
    use crate::layers::LayerNormCrownMode;

    let mut network = Network::new();

    let hidden = 4;

    // Linear: 4 -> 4
    let weight = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        0.3 * phase.sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight, None).unwrap()));

    // LayerNorm with default ny=1, beta=0 (will set mode after adding)
    let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
    network.add_layer(Layer::LayerNorm(ln));

    // Set all LayerNorm layers to sampling mode for this test
    network.set_layernorm_crown_mode(LayerNormCrownMode::Sampling);

    // Input: [batch=2, seq=3, 4]
    let batch = 2;
    let seq = 3;
    let total_elements = batch * seq * hidden;

    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let hash = ((b * 100 + s * 10 + h) as u32).wrapping_mul(2654435761_u32);
                let base = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0;
                lower[[b, s, h]] = base - 0.15;
                upper[[b, s, h]] = base + 0.15;
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

    // Soundness check via helper
    assert_batched_crown_soundness(&network, &input, &batched_result);
}

/// Sample concrete points within `input` bounds and verify `network`'s true
/// output at each point falls within `bounds`. Uses IBP on zero-width intervals
/// for exact forward evaluation (not CROWN, which may add sampling noise).
fn assert_batched_crown_soundness(
    network: &Network,
    input: &BoundedTensor,
    bounds: &BoundedTensor,
) {
    let tol = 1e-2; // sampling CROWN tolerance (heuristic, not provably sound)
    let shape = input.shape();
    for sample_idx in 0..10 {
        let mut point = ArrayD::zeros(IxDyn(shape));
        for (flat_idx, (lo, hi)) in input.lower().iter().zip(input.upper().iter()).enumerate() {
            let hash = ((sample_idx * 10000 + flat_idx) as u32).wrapping_mul(2654435761_u32);
            let t = hash as f32 / u32::MAX as f32;
            point.as_slice_mut().unwrap()[flat_idx] = lo + t * (hi - lo);
        }
        let pt_bt = BoundedTensor::new(point.clone(), point).unwrap();
        // IBP on zero-width interval = exact forward pass (no CROWN sampling noise)
        let true_out = network.propagate_ibp(&pt_bt).unwrap();
        for (idx, (y, (lb, ub))) in true_out
            .lower()
            .iter()
            .zip(bounds.lower().iter().zip(bounds.upper().iter()))
            .enumerate()
        {
            assert!(
                *y >= *lb - tol,
                "Soundness violation at sample={sample_idx}, idx={idx}: y={y} < lb={lb}"
            );
            assert!(
                *y <= *ub + tol,
                "Soundness violation at sample={sample_idx}, idx={idx}: y={y} > ub={ub}"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_propagate_crown_batched_with_constant_arithmetic_layers() {
    let mut network = Network::new();
    let hidden = 4;

    let weight1 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        let phase = (i * 19 + j * 23) as f32;
        0.2 * phase.cos()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));
    network.add_layer(Layer::AddConstant(AddConstantLayer::new(
        ArrayD::from_elem(IxDyn(&[]), 0.25),
    )));
    network.add_layer(Layer::SubConstant(SubConstantLayer::scalar(0.10)));
    network.add_layer(Layer::MulConstant(MulConstantLayer::scalar(1.25)));
    network.add_layer(Layer::DivConstant(DivConstantLayer::scalar(2.0)));

    let weight2 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        let phase = (i * 29 + j * 31) as f32;
        0.15 * phase.sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    let batch = 2;
    let seq = 3;
    let mut lower = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));
    let mut upper = ArrayD::zeros(IxDyn(&[batch, seq, hidden]));

    for b in 0..batch {
        for s in 0..seq {
            for h in 0..hidden {
                let idx = (b * 100 + s * 10 + h) as f32;
                let center = 0.4 * (0.13 * idx).sin();
                lower[[b, s, h]] = center - 0.1;
                upper[[b, s, h]] = center + 0.1;
            }
        }
    }
    let input = BoundedTensor::new(lower, upper).unwrap();

    let batched_result = network.propagate_crown_batched(&input).unwrap();
    let regular_result = network.propagate_crown(&input).unwrap();

    assert_eq!(batched_result.shape(), regular_result.shape());

    for (idx, (lhs, rhs)) in batched_result
        .lower()
        .iter()
        .zip(regular_result.lower().iter())
        .enumerate()
    {
        assert!(
            (lhs - rhs).abs() <= 1e-4,
            "lower mismatch at {}: batched={} regular={}",
            idx,
            lhs,
            rhs
        );
    }

    for (idx, (lhs, rhs)) in batched_result
        .upper()
        .iter()
        .zip(regular_result.upper().iter())
        .enumerate()
    {
        assert!(
            (lhs - rhs).abs() <= 1e-4,
            "upper mismatch at {}: batched={} regular={}",
            idx,
            lhs,
            rhs
        );
    }
}
