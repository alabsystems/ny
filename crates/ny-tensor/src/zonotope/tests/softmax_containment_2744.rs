// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Zonotope softmax output containment tests (#2744).
//!
//! Verifies that zonotope bounds after softmax_affine and
//! softmax_affine_causal contain the true softmax at sampled points.

use super::super::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

/// Simple deterministic PRNG (xorshift32) for sampling without `rand` dep.
struct Xorshift32(u32);

impl Xorshift32 {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// Uniform f32 in [-1, 1].
    fn uniform_neg1_pos1(&mut self) -> f32 {
        (self.next() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn softmax_vec(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_x: Vec<f32> = x.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f32 = exp_x.iter().sum();
    exp_x.iter().map(|e| e / sum).collect()
}

/// Sample a point from a zonotope by choosing ε_k values.
/// Point = center + Σ_k ε_k * a_k.
fn sample_zonotope_point(z: &ZonotopeTensor, epsilons: &[f32]) -> ArrayD<f32> {
    assert_eq!(epsilons.len(), z.n_error_terms());
    let mut point = z.center();
    for (k, &eps) in epsilons.iter().enumerate() {
        let a_k = z.coeffs.index_axis(ndarray::Axis(0), k + 1);
        point += &a_k.mapv(|v| v * eps);
    }
    point
}

/// Assert that every element of `true_output` is within the zonotope bounds.
fn assert_contained(
    bounds: &crate::BoundedTensor,
    true_output: &[f32],
    label: &str,
    sample_idx: usize,
) {
    for (i, &val) in true_output.iter().enumerate() {
        assert!(
            bounds.lower().iter().nth(i).copied().unwrap() <= val,
            "{label} sample {sample_idx}: lower[{i}]={} > true softmax {val:.6e}",
            bounds.lower().iter().nth(i).unwrap()
        );
        assert!(
            bounds.upper().iter().nth(i).copied().unwrap() >= val,
            "{label} sample {sample_idx}: upper[{i}]={} < true softmax {val:.6e}",
            bounds.upper().iter().nth(i).unwrap()
        );
    }
}

// ---------------------------------------------------------------------------
// AC1: 1D softmax containment with sampling
// ---------------------------------------------------------------------------

/// Creates a small 1D zonotope (3 error terms, 3 dimensions), applies
/// softmax_affine, samples N points, computes true softmax, and verifies
/// that the zonotope output bounds contain every true softmax output.
///
/// Part of #2744.
#[test]
fn test_softmax_affine_1d_sampling_containment_2744() {
    // Build zonotope: center=[0.5, -0.3, 1.0], 3 correlated error terms.
    let data: Vec<f32> = vec![
        0.5, -0.3, 1.0, // center
        0.8, 0.2, -0.5, // error term 1
        -0.3, 0.7, 0.4, // error term 2
        0.1, -0.6, 0.9, // error term 3
    ];
    let coeffs = ArrayD::from_shape_vec(IxDyn(&[4, 3]), data).unwrap();
    let z = ZonotopeTensor::new(coeffs).unwrap();
    assert_eq!(z.n_error_terms(), 3);

    let result = z.softmax_affine(-1).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    let n_samples = 200;
    let mut rng = Xorshift32(42);
    for sample in 0..n_samples {
        let epsilons: Vec<f32> = (0..3).map(|_| rng.uniform_neg1_pos1()).collect();
        let point = sample_zonotope_point(&z, &epsilons);
        let input_flat: Vec<f32> = point.iter().cloned().collect();
        let true_sm = softmax_vec(&input_flat);
        assert_contained(&bounds, &true_sm, "1D softmax containment", sample);
    }

    // Also test all 8 corners for completeness.
    for bits in 0..8u32 {
        let epsilons: Vec<f32> = (0..3)
            .map(|k| if bits & (1 << k) != 0 { 1.0 } else { -1.0 })
            .collect();
        let point = sample_zonotope_point(&z, &epsilons);
        let input_flat: Vec<f32> = point.iter().cloned().collect();
        let true_sm = softmax_vec(&input_flat);
        assert_contained(
            &bounds,
            &true_sm,
            "1D softmax corner containment",
            bits as usize,
        );
    }
}

// ---------------------------------------------------------------------------
// AC2: 2D causal softmax containment with sampling
// ---------------------------------------------------------------------------

/// Creates a 2D zonotope (3 error terms, shape [3, 3]), applies causal
/// softmax, samples N points, and verifies containment of the true
/// causal softmax at each sample.
///
/// Part of #2744.
#[test]
fn test_softmax_affine_causal_sampling_containment_2744() {
    // Build zonotope: shape (4, 3, 3) = [center + 3 errors, seq_q=3, seq_k=3]
    let data: Vec<f32> = vec![
        // center: 3x3 attention logits
        0.1, -0.2, 0.3, 0.4, 0.5, -0.1, -0.3, 0.2, 0.6, // error term 1
        0.5, 0.3, -0.2, -0.1, 0.4, 0.3, 0.2, -0.5, 0.1, // error term 2
        -0.3, 0.6, 0.1, 0.4, -0.2, 0.5, 0.1, 0.3, -0.4, // error term 3
        0.2, -0.1, 0.4, -0.3, 0.1, -0.2, 0.5, 0.4, 0.3,
    ];
    let coeffs = ArrayD::from_shape_vec(IxDyn(&[4, 3, 3]), data).unwrap();
    let z = ZonotopeTensor::new(coeffs).unwrap();
    assert_eq!(z.n_error_terms(), 3);

    let result = z.softmax_affine_causal(-1).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    let n_samples = 200;
    let mut rng = Xorshift32(123);
    for sample in 0..n_samples {
        let epsilons: Vec<f32> = (0..3).map(|_| rng.uniform_neg1_pos1()).collect();
        let point = sample_zonotope_point(&z, &epsilons);

        // Compute true causal softmax per query row.
        for qi in 0..3usize {
            let allowed = qi + 1;
            let prefix: Vec<f32> = (0..allowed).map(|j| point[[qi, j]]).collect();
            let sm = softmax_vec(&prefix);

            for (j, &sj) in sm.iter().enumerate() {
                assert!(
                    bounds.lower()[[qi, j]] <= sj,
                    "causal containment sample {sample}: lower[{qi},{j}]={} > true {sj:.6e}",
                    bounds.lower()[[qi, j]]
                );
                assert!(
                    bounds.upper()[[qi, j]] >= sj,
                    "causal containment sample {sample}: upper[{qi},{j}]={} < true {sj:.6e}",
                    bounds.upper()[[qi, j]]
                );
            }

            // Masked positions must be 0.
            for j in allowed..3 {
                assert!(
                    bounds.upper()[[qi, j]].abs() <= 1e-6,
                    "causal mask sample {sample}: position ({qi},{j}) not zero, \
                     got [{}, {}]",
                    bounds.lower()[[qi, j]],
                    bounds.upper()[[qi, j]]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AC3: Zero-radius point evaluation
// ---------------------------------------------------------------------------

/// A zonotope with zero radius (concrete point) should produce exact
/// softmax output with zero-width bounds.
///
/// Part of #2744.
#[test]
fn test_softmax_affine_zero_radius_exact_2744() {
    let values = arr1(&[1.0_f32, 2.0, 3.0]);
    let z = ZonotopeTensor::concrete(values.into_dyn());
    assert_eq!(z.n_error_terms(), 0);

    let result = z.softmax_affine(-1).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    // Compute expected softmax.
    let expected = softmax_vec(&[1.0, 2.0, 3.0]);

    for (i, &exp) in expected.iter().enumerate() {
        let lo = bounds.lower().iter().nth(i).copied().unwrap();
        let hi = bounds.upper().iter().nth(i).copied().unwrap();

        // Center should match true softmax within f32 tolerance.
        assert!(
            (result.center()[i] - exp).abs() < 1e-6,
            "zero-radius center[{i}]={} != expected {exp:.6e}",
            result.center()[i]
        );

        // Bounds width should be zero (no error terms → no approximation error).
        assert!(
            (hi - lo).abs() < 1e-7,
            "zero-radius bounds[{i}] not zero-width: [{lo:.6e}, {hi:.6e}]"
        );

        // Bounds must contain the true value.
        assert!(
            lo <= exp + 1e-6,
            "zero-radius lower[{i}]={lo:.6e} > expected {exp:.6e}"
        );
        assert!(
            hi >= exp - 1e-6,
            "zero-radius upper[{i}]={hi:.6e} < expected {exp:.6e}"
        );
    }
}

/// Zero-radius causal softmax should produce exact output.
///
/// Part of #2744.
#[test]
fn test_softmax_affine_causal_zero_radius_exact_2744() {
    let values = arr2(&[[0.5_f32, 0.3, -0.2], [1.0, -0.5, 0.8], [-0.3, 0.7, 0.1]]);
    let z = ZonotopeTensor::concrete(values.into_dyn());
    assert_eq!(z.n_error_terms(), 0);

    let result = z.softmax_affine_causal(-1).unwrap();
    let center = result.center();
    let bounds = result.to_bounded_tensor().unwrap();

    // Row 0: softmax of [0.5] = [1.0], masked: [_, _]
    assert!(
        (center[[0, 0]] - 1.0).abs() < 1e-6,
        "causal zero-radius row 0 col 0 should be 1.0, got {}",
        center[[0, 0]]
    );
    for j in 1..3 {
        assert!(
            center[[0, j]].abs() < 1e-6,
            "causal zero-radius row 0 col {j} should be 0, got {}",
            center[[0, j]]
        );
    }

    // Row 1: softmax of [1.0, -0.5]
    let sm1 = softmax_vec(&[1.0, -0.5]);
    for (j, &sj) in sm1.iter().enumerate() {
        assert!(
            (center[[1, j]] - sj).abs() < 1e-5,
            "causal zero-radius row 1 col {j}: {} != {sj:.6e}",
            center[[1, j]]
        );
    }
    assert!(
        center[[1, 2]].abs() < 1e-6,
        "causal zero-radius row 1 col 2 should be 0, got {}",
        center[[1, 2]]
    );

    // Row 2: softmax of [-0.3, 0.7, 0.1]
    let sm2 = softmax_vec(&[-0.3, 0.7, 0.1]);
    for (j, &sj) in sm2.iter().enumerate() {
        assert!(
            (center[[2, j]] - sj).abs() < 1e-5,
            "causal zero-radius row 2 col {j}: {} != {sj:.6e}",
            center[[2, j]]
        );
    }

    // All bounds should be zero-width (no error terms → no approximation error).
    for qi in 0..3 {
        for j in 0..3 {
            let lo = bounds.lower()[[qi, j]];
            let hi = bounds.upper()[[qi, j]];
            assert!(
                (hi - lo).abs() < 1e-6,
                "causal zero-radius bounds[{qi},{j}] not zero-width: [{lo:.6e}, {hi:.6e}]"
            );
        }
    }
}
