// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::{Bound, HeuristicUsed, VerificationSpec};

// ==================== Sqrt tests ====================

fn assert_sqrt_linear_bounds_sound_on_effective_domain(
    bounds: &LinearBounds,
    lower: f32,
    upper: f32,
) {
    for step in 0..=20 {
        let x = lower + (upper - lower) * (step as f32 / 20.0);
        let y = x.sqrt();
        let lower_val = bounds.lower_a[[0, 0]] * x + bounds.lower_b[0];
        let upper_val = bounds.upper_a[[0, 0]] * x + bounds.upper_b[0];
        let tol = (1e-4_f32).max(1e-3 * y);
        assert!(
            lower_val <= y + tol,
            "lower bound violated at x={x}: {lower_val} > {y}"
        );
        assert!(
            upper_val >= y - tol,
            "upper bound violated at x={x}: {upper_val} < {y}"
        );
    }
}

fn assert_sqrt_graph_crown_marks_negative_domain_heuristic(input: &BoundedTensor) {
    let mut network = Network::new();
    network.add_layer(Layer::Sqrt(SqrtLayer));
    let graph = GraphNetwork::from_sequential(&network).expect("sqrt graph should build");
    let sqrt_count = count_sqrt_negative_domain_graph(&graph, input)
        .expect("sqrt negative-domain scan should succeed");
    assert_eq!(sqrt_count, 1, "expected one negative-domain sqrt node");

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Crown,
        ..Default::default()
    });
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-0.5, 1.0)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        None,
        None,
    )
    .expect("sqrt verifier spec should construct");
    let verification = verifier
        .verify_graph(&graph, &spec)
        .expect("graph CROWN should keep working on clamped sqrt domains");
    assert_eq!(verification.actual_method(), Some("Crown"));
    assert!(
        verification
            .provenance()
            .heuristics_used()
            .iter()
            .any(|heuristic| matches!(
                heuristic,
                HeuristicUsed::SqrtNegativeDomain { num_nodes: 1 }
            )),
        "expected SqrtNegativeDomain heuristic, got {:?}",
        verification.provenance().heuristics_used()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sqrt_ibp_basic() {
    // Test sqrt on positive bounds
    let lower = ArrayD::from_elem(IxDyn(&[4]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sqrt = SqrtLayer;
    let output = sqrt.propagate_ibp(&input).unwrap();

    // sqrt([1, 4]) = [1, 2]
    for i in 0..4 {
        assert!(
            (output.lower()[[i]] - 1.0).abs() < 1e-6,
            "sqrt(1) should be 1, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 2.0).abs() < 1e-6,
            "sqrt(4) should be 2, got {}",
            output.upper()[[i]]
        );
    }
}

/// Design: #424 (gate negative sqrt via heuristic marking, not error).
/// Sqrt clamps negative inputs to 0; soundness provenance scanner detects and
/// flags SqrtNegativeDomain so the verifier reports heuristic mode.
#[ntest::timeout(10000)]
#[test]
fn test_sqrt_ibp_clamps_negative_1635() {
    // sqrt([-1, 9]) should clamp to sqrt([0, 9]) = [0, 3]
    let lower = ArrayD::from_elem(IxDyn(&[3]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 9.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sqrt = SqrtLayer;
    let output = sqrt.propagate_ibp(&input).unwrap();

    for i in 0..3 {
        assert!(
            output.lower()[[i]].abs() < 1e-6,
            "Clamped sqrt lower should be 0, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 3.0).abs() < 1e-6,
            "sqrt(9) should be 3, got {}",
            output.upper()[[i]]
        );
    }

    // SOUNDNESS CONTRACT (#424): When propagate_ibp clamps negative inputs,
    // the soundness provenance scanner must detect SqrtNegativeDomain so the
    // verifier marks the result as Heuristic. Verify the scanner catches this.
    let mut network = Network::new();
    network.add_layer(Layer::Sqrt(SqrtLayer));
    let sqrt_count = count_sqrt_negative_domain_network(&network, &input).unwrap();
    assert!(
        sqrt_count > 0,
        "Provenance scanner should detect negative sqrt domain, got count={}",
        sqrt_count
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sqrt_linear_requires_bounds() {
    // Sqrt CROWN propagation should reject missing pre-activation bounds.
    let bounds = LinearBounds::identity(4);
    let sqrt = SqrtLayer;
    let err = sqrt
        .propagate_linear(&bounds)
        .expect_err("Expected error without pre-activation bounds");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("pre-activation bounds"),
        "unexpected error message: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sqrt_crown_clamps_negative_bounds_and_marks_graph_heuristic_4118() {
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), -0.5f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let sqrt = SqrtLayer;

    let result = sqrt
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .expect("negative lower bounds should clamp locally instead of erroring");
    assert!(
        result.lower_a.iter().all(|value| value.is_finite())
            && result.upper_a.iter().all(|value| value.is_finite())
            && result.lower_b.iter().all(|value| value.is_finite())
            && result.upper_b.iter().all(|value| value.is_finite()),
        "sqrt CROWN clamp regression should keep linear bounds finite, got {result:?}"
    );
    assert_sqrt_linear_bounds_sound_on_effective_domain(&result, 0.0, 1.0);

    assert_sqrt_graph_crown_marks_negative_domain_heuristic(&pre_activation);
}

#[ntest::timeout(10000)]
#[test]
fn test_sqrt_crown_soundness() {
    // Test that CROWN bounds for sqrt are sound
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 4.0, 0.25, 9.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![4.0, 9.0, 1.0, 16.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let sqrt = SqrtLayer;

    let result = sqrt
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points in the input range and verify bounds hold
    let test_points: [Vec<f32>; 4] = [
        vec![1.0, 4.0, 0.25, 9.0],   // lower
        vec![4.0, 9.0, 1.0, 16.0],   // upper
        vec![2.5, 6.5, 0.625, 12.5], // midpoint
        vec![2.0, 5.0, 0.5, 10.0],   // random
    ];

    for point in &test_points {
        let sqrt_output: Vec<f32> = point.iter().map(|x| x.sqrt()).collect();

        // Check each output dimension
        for (j, &sqrt_val) in sqrt_output.iter().enumerate() {
            // Lower bound: lower_a * x + lower_b should be <= sqrt(point)
            let lb_val: f32 = (0..4)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            // Upper bound: upper_a * x + upper_b should be >= sqrt(point)
            let ub_val: f32 = (0..4)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-3; // Slightly relaxed for sqrt's curvature
            assert!(
                lb_val <= sqrt_val + tol,
                "Lower bound violated at point {:?}: lb {} > sqrt {}",
                point,
                lb_val,
                sqrt_val
            );
            assert!(
                ub_val >= sqrt_val - tol,
                "Upper bound violated at point {:?}: ub {} < sqrt {}",
                point,
                ub_val,
                sqrt_val
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sqrt_crown_concave_property() {
    // Sqrt is concave, so chord should be a lower bound
    // and tangent-based approximation should be an upper bound
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), 4.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let sqrt = SqrtLayer;

    let result = sqrt
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Chord from (1, 1) to (4, 2): slope = (2-1)/(4-1) = 1/3 ≈ 0.333
    let expected_chord_slope = (2.0 - 1.0) / (4.0 - 1.0);
    // Due to numerical sampling, the slope should be close to chord slope
    let slope = result.lower_a[[0, 0]];

    // For a concave function, CROWN lower bound uses the chord slope analytically.
    assert!(
        (slope - expected_chord_slope).abs() < 1e-3,
        "Lower slope {} should be close to chord slope {}",
        slope,
        expected_chord_slope
    );
}
