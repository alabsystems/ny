// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for RMSNorm layer.

use ndarray::{arr1, Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::types::RmsNormLayer;
use crate::layers::common::BoundPropagation;

fn default_rn(n: usize) -> RmsNormLayer {
    RmsNormLayer::new_default(n, 1e-5).expect("valid default RmsNorm")
}

fn custom_rn(ny: &[f32], eps: f32) -> RmsNormLayer {
    RmsNormLayer::new(Array1::from_vec(ny.to_vec()), eps).expect("valid custom RmsNorm")
}

// ── IBP soundness tests ─────────────────────────────────────────────────────

#[test]
fn test_ibp_contains_eval_1d() {
    let rn = default_rn(3);
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 1.0, 2.0]).expect("valid lower shape");
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.5, 3.0, 4.0]).expect("valid upper shape");
    let input = BoundedTensor::new(lower, upper).expect("valid bounded tensor");

    let output = rn.propagate_ibp(&input).expect("IBP should succeed");

    // Evaluate at multiple points within bounds and verify containment
    let test_points: Vec<[f32; 3]> = vec![
        [0.5, 1.0, 2.0], // lower corner
        [1.5, 3.0, 4.0], // upper corner
        [1.0, 2.0, 3.0], // midpoint
        [0.5, 3.0, 2.0], // mixed
        [1.5, 1.0, 4.0], // mixed
    ];

    for point in &test_points {
        let x = arr1(point);
        let y = rn.eval(&x).expect("eval should succeed");
        for i in 0..3 {
            assert!(
                y[i] >= output.lower()[[i]] - 1e-5,
                "y[{i}]={} below lower bound {} at point {:?}",
                y[i],
                output.lower()[[i]],
                point
            );
            assert!(
                y[i] <= output.upper()[[i]] + 1e-5,
                "y[{i}]={} above upper bound {} at point {:?}",
                y[i],
                output.upper()[[i]],
                point
            );
        }
    }
}

#[test]
fn test_ibp_contains_eval_with_negative_inputs() {
    let rn = default_rn(3);
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, -1.0, 0.5]).expect("valid lower shape");
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).expect("valid upper shape");
    let input = BoundedTensor::new(lower, upper).expect("valid bounded tensor");

    let output = rn.propagate_ibp(&input).expect("IBP should succeed");

    let test_points: Vec<[f32; 3]> = vec![
        [-2.0, -1.0, 0.5],
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 1.5],
        [-1.0, 1.0, 2.0],
    ];

    for point in &test_points {
        let x = arr1(point);
        let y = rn.eval(&x).expect("eval should succeed");
        for i in 0..3 {
            assert!(
                y[i] >= output.lower()[[i]] - 1e-5,
                "y[{i}]={} below lower bound {} at point {:?}",
                y[i],
                output.lower()[[i]],
                point
            );
            assert!(
                y[i] <= output.upper()[[i]] + 1e-5,
                "y[{i}]={} above upper bound {} at point {:?}",
                y[i],
                output.upper()[[i]],
                point
            );
        }
    }
}

#[test]
fn test_ibp_contains_eval_custom_gamma() {
    let rn = custom_rn(&[0.5, 2.0, -1.0], 1e-5);
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, -1.0, 0.0]).expect("valid lower shape");
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 1.0, 2.0]).expect("valid upper shape");
    let input = BoundedTensor::new(lower, upper).expect("valid bounded tensor");

    let output = rn.propagate_ibp(&input).expect("IBP should succeed");

    let test_points: Vec<[f32; 3]> = vec![
        [1.0, -1.0, 0.0],
        [3.0, 1.0, 2.0],
        [2.0, 0.0, 1.0],
        [1.5, -0.5, 1.5],
    ];

    for point in &test_points {
        let x = arr1(point);
        let y = rn.eval(&x).expect("eval should succeed");
        for i in 0..3 {
            assert!(
                y[i] >= output.lower()[[i]] - 1e-5,
                "y[{i}]={} below lower bound {} at point {:?}",
                y[i],
                output.lower()[[i]],
                point
            );
            assert!(
                y[i] <= output.upper()[[i]] + 1e-5,
                "y[{i}]={} above upper bound {} at point {:?}",
                y[i],
                output.upper()[[i]],
                point
            );
        }
    }
}

#[test]
fn test_ibp_2d_input() {
    let rn = default_rn(2);
    // [2, 2] input: 2 batch positions, norm_size=2
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0])
        .expect("valid lower shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 3.0, 5.0, 6.0])
        .expect("valid upper shape");
    let input = BoundedTensor::new(lower, upper).expect("valid bounded tensor");

    let output = rn.propagate_ibp(&input).expect("IBP should succeed");
    assert_eq!(output.shape(), &[2, 2]);

    // Verify each batch position
    for batch in 0..2 {
        let xl = arr1(&[input.lower()[[batch, 0]], input.lower()[[batch, 1]]]);
        let xu = arr1(&[input.upper()[[batch, 0]], input.upper()[[batch, 1]]]);
        let xm = (&xl + &xu) / 2.0;
        let y = rn.eval(&xm).expect("eval should succeed");
        for i in 0..2 {
            assert!(
                y[i] >= output.lower()[[batch, i]] - 1e-5,
                "batch={batch} i={i}: y={} < lower={}",
                y[i],
                output.lower()[[batch, i]]
            );
            assert!(
                y[i] <= output.upper()[[batch, i]] + 1e-5,
                "batch={batch} i={i}: y={} > upper={}",
                y[i],
                output.upper()[[batch, i]]
            );
        }
    }
}

// ── Forward-mode IBP tests ──────────────────────────────────────────────────

#[test]
fn test_ibp_forward_mode_contains_eval() {
    let rn = default_rn(3).with_forward_mode(true);
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 1.0, 2.0]).expect("valid lower shape");
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.5, 3.0, 4.0]).expect("valid upper shape");
    let input = BoundedTensor::new(lower, upper).expect("valid bounded tensor");

    let output = rn
        .propagate_ibp(&input)
        .expect("forward IBP should succeed");

    // Forward mode should still contain the center-point evaluation
    let center = arr1(&[1.0, 2.0, 3.0]);
    let y = rn.eval(&center).expect("eval should succeed");
    for i in 0..3 {
        assert!(
            y[i] >= output.lower()[[i]] - 1e-5,
            "forward: y[{i}]={} below lower bound {}",
            y[i],
            output.lower()[[i]]
        );
        assert!(
            y[i] <= output.upper()[[i]] + 1e-5,
            "forward: y[{i}]={} above upper bound {}",
            y[i],
            output.upper()[[i]]
        );
    }
}

// ── Error handling tests ────────────────────────────────────────────────────

#[test]
fn test_ibp_non_finite_input_rejected() {
    // BoundedTensor::new rejects infinite bounds at construction time,
    // which is the correct behavior — non-finite bounds never reach the layer.
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 0.0])
        .expect("valid array shape");
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 1.0]).expect("valid array shape");
    let result = BoundedTensor::new(lower, upper);
    assert!(
        result.is_err(),
        "BoundedTensor should reject infinite bounds"
    );
}

#[test]
fn test_ibp_shape_mismatch_rejected() {
    let rn = default_rn(3);
    // Input has norm_size=2 but ny has size 3
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).expect("valid lower shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).expect("valid upper shape");
    let input = BoundedTensor::new(lower, upper).expect("valid bounded tensor");

    let result = rn.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "IBP should reject input size mismatch (2 vs 3)"
    );
}

#[test]
fn test_ibp_empty_input_rejected() {
    let rn = default_rn(0);
    let result = rn.propagate_ibp(
        &BoundedTensor::new(ArrayD::zeros(IxDyn(&[0])), ArrayD::zeros(IxDyn(&[0])))
            .expect("valid empty tensor"),
    );
    assert!(result.is_err(), "IBP should reject empty input");
}

// ── Constructor tests ───────────────────────────────────────────────────────

#[test]
fn test_invalid_eps_rejected() {
    assert!(RmsNormLayer::new_default(3, f32::NAN).is_err());
    assert!(RmsNormLayer::new_default(3, f32::INFINITY).is_err());
    assert!(RmsNormLayer::new_default(3, -1.0).is_err());
}

#[test]
fn test_eps_floor_applied() {
    let rn = RmsNormLayer::new_default(3, 0.0).expect("valid RmsNorm with eps=0");
    assert!(rn.eps > 0.0, "eps should be clamped above zero");
}

// ── Sampling-based IBP soundness test (brute force) ─────────────────────────

#[test]
fn test_ibp_soundness_random_sampling() {
    // Test with various ny configurations and input ranges
    let configs: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = vec![
        // (ny, lower, upper)
        (
            vec![1.0, 1.0, 1.0],
            vec![0.1, 0.2, 0.3],
            vec![0.5, 0.8, 1.0],
        ),
        (
            vec![2.0, 0.5, -1.0],
            vec![-1.0, 0.0, 1.0],
            vec![1.0, 2.0, 3.0],
        ),
        (vec![1.0, 1.0], vec![-5.0, -5.0], vec![5.0, 5.0]),
        (
            vec![0.1, 10.0, 1.0, 0.5],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
    ];

    for (ny, lower, upper) in &configs {
        let n = ny.len();
        let rn = RmsNormLayer::new(Array1::from_vec(ny.clone()), 1e-5).expect("valid RmsNorm");
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.clone()).expect("valid lower shape"),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.clone()).expect("valid upper shape"),
        )
        .expect("valid bounded tensor");

        let output = rn.propagate_ibp(&input).expect("IBP should succeed");

        // Sample 200 random points and verify containment
        for s in 0..200 {
            let mut point = Vec::with_capacity(n);
            for i in 0..n {
                let t = ((s as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                point.push(lower[i] + (upper[i] - lower[i]) * t);
            }
            let x = Array1::from_vec(point.clone());
            let y = rn.eval(&x).expect("eval should succeed");
            for i in 0..n {
                assert!(
                    y[i] >= output.lower()[[i]] - 1e-4,
                    "ny={ny:?} point={point:?}: y[{i}]={} < lower={}",
                    y[i],
                    output.lower()[[i]]
                );
                assert!(
                    y[i] <= output.upper()[[i]] + 1e-4,
                    "ny={ny:?} point={point:?}: y[{i}]={} > upper={}",
                    y[i],
                    output.upper()[[i]]
                );
            }
        }
    }
}

// ── Forward-mode IBP soundness: Jacobian-based (#3098) ──────────────────────

/// Regression test for #3098: forward-mode IBP must contain all concrete
/// evaluations, not just the center point. The old `max_radius / n` coupling
/// correction underestimated denominator perturbation.
///
/// This test uses the exact counterexample from the issue: dim=8 with large
/// centers and radii where the old formula violated bounds at 13/16 coordinates.
#[test]
fn test_ibp_forward_mode_soundness_issue_3098_counterexample() {
    // Reproduce the issue's counterexample regime: dim=8, center ∈ U(-5,5), radius ∈ U(0.5,5)
    let n = 8;
    let rn = RmsNormLayer::new(Array1::ones(n), 1e-5)
        .unwrap()
        .with_forward_mode(true);

    // Multiple random configurations in the problematic regime
    let configs: Vec<(Vec<f32>, Vec<f32>)> = vec![
        // center, radius (large radii relative to center → high coupling)
        (
            vec![-3.2, 2.1, 4.5, -1.0, 0.3, -2.8, 1.7, 3.9],
            vec![2.5, 1.8, 3.0, 4.0, 1.5, 2.2, 3.5, 0.8],
        ),
        (
            vec![4.0, -4.0, 3.0, -3.0, 2.0, -2.0, 1.0, -1.0],
            vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        ),
        (
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            vec![4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
        ),
    ];

    for (center, rad) in &configs {
        let lower: Vec<f32> = center.iter().zip(rad.iter()).map(|(c, r)| c - r).collect();
        let upper: Vec<f32> = center.iter().zip(rad.iter()).map(|(c, r)| c + r).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.clone()).unwrap(),
        )
        .unwrap();

        let output = rn.propagate_ibp(&input).unwrap();

        // Sample 500 random points and verify containment
        for s in 0..500 {
            let mut point = Vec::with_capacity(n);
            for i in 0..n {
                let t = ((s as u32).wrapping_mul(2654435761_u32) ^ (i as u32).wrapping_mul(7))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                point.push(lower[i] + (upper[i] - lower[i]) * t);
            }
            let x = Array1::from_vec(point.clone());
            let y = rn.eval(&x).unwrap();
            for i in 0..n {
                assert!(
                    y[i] >= output.lower()[[i]] - 1e-3,
                    "center={center:?} rad={rad:?} sample={s} point={point:?}: y[{i}]={} < lower={}",
                    y[i],
                    output.lower()[[i]]
                );
                assert!(
                    y[i] <= output.upper()[[i]] + 1e-3,
                    "center={center:?} rad={rad:?} sample={s} point={point:?}: y[{i}]={} > upper={}",
                    y[i],
                    output.upper()[[i]]
                );
            }
        }
    }
}

/// Forward-mode soundness with non-trivial ny (negative, large, mixed).
#[test]
fn test_ibp_forward_mode_soundness_custom_gamma() {
    let configs: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = vec![
        // (ny, lower, upper)
        (
            vec![2.0, -1.0, 0.5],
            vec![-3.0, -2.0, -1.0],
            vec![3.0, 2.0, 1.0],
        ),
        (
            vec![10.0, 10.0, 10.0],
            vec![0.9, 0.95, 1.0],
            vec![1.1, 1.05, 1.0],
        ),
        (
            vec![-5.0, 3.0, -0.1, 7.0],
            vec![-2.0, -2.0, -2.0, -2.0],
            vec![2.0, 2.0, 2.0, 2.0],
        ),
    ];

    for (ny, lower, upper) in &configs {
        let n = ny.len();
        let rn = RmsNormLayer::new(Array1::from_vec(ny.clone()), 1e-5)
            .unwrap()
            .with_forward_mode(true);
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.clone()).unwrap(),
        )
        .unwrap();

        let output = rn.propagate_ibp(&input).unwrap();

        for s in 0..300 {
            let mut point = Vec::with_capacity(n);
            for i in 0..n {
                let t = ((s as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                point.push(lower[i] + (upper[i] - lower[i]) * t);
            }
            let x = Array1::from_vec(point.clone());
            let y = rn.eval(&x).unwrap();
            for i in 0..n {
                assert!(
                    y[i] >= output.lower()[[i]] - 1e-3,
                    "ny={ny:?} sample={s}: y[{i}]={} < lower={}",
                    y[i],
                    output.lower()[[i]]
                );
                assert!(
                    y[i] <= output.upper()[[i]] + 1e-3,
                    "ny={ny:?} sample={s}: y[{i}]={} > upper={}",
                    y[i],
                    output.upper()[[i]]
                );
            }
        }
    }
}

// CROWN tests moved to tests_crown.rs (#3103) to keep file under 500-line limit.
