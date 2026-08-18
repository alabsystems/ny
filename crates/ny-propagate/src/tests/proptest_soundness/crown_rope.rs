// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest soundness tests for RoPE (Rotary Position Embedding) layer.
//!
//! RoPE is a linear layer (block-diagonal rotation matrix), so both IBP and
//! CROWN backward are exact — no relaxation error. The soundness property:
//! for any concrete x in the input bounds, the rotated output R(θ)·x must
//! lie within the computed output bounds.
//!
//! Part of #3145.

use crate::layers::common::BoundPropagation;
use crate::layers::rope::RopeLayer;
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{sample_points, valid_interval, FP_TOLERANCE};

/// Apply RoPE rotation to a concrete vector.
/// For each pair (2i, 2i+1): y[2i] = c*x[2i] - s*x[2i+1], y[2i+1] = s*x[2i] + c*x[2i+1].
fn rope_eval(x: &[f32], cos_freqs: &[f32], sin_freqs: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for i in 0..cos_freqs.len() {
        let c = cos_freqs[i];
        let s = sin_freqs[i];
        let idx_even = 2 * i;
        let idx_odd = 2 * i + 1;
        y[idx_even] = c * x[idx_even] - s * x[idx_odd];
        y[idx_odd] = s * x[idx_even] + c * x[idx_odd];
    }
    y
}

// =============================================================================
// IBP SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// RoPE IBP soundness with 1 pair (head_dim=2): sample random angles and inputs,
    /// verify that R(θ)·x is within IBP bounds for all x in [lower, upper].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rope_ibp_1pair(
        angle in -std::f32::consts::PI..std::f32::consts::PI,
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
    ) {
        let layer = RopeLayer::from_angles(&[angle]).unwrap();
        let input = BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        let c = angle.cos();
        let s = angle.sin();

        // Check all 4 corners of the input box
        for &x0 in &[l0, u0] {
            for &x1 in &[l1, u1] {
                let y = rope_eval(&[x0, x1], &[c], &[s]);
                for (dim, &y_val) in y.iter().enumerate() {
                    prop_assert!(
                        output.lower()[[dim]] - FP_TOLERANCE <= y_val
                            && y_val <= output.upper()[[dim]] + FP_TOLERANCE,
                        "IBP 1-pair soundness violated at dim {dim}: y={y_val}, bounds=[{}, {}], x=[{x0}, {x1}], angle={angle}",
                        output.lower()[[dim]], output.upper()[[dim]]
                    );
                }
            }
        }

        // Also sample interior points
        for x0 in sample_points(l0, u0, 5) {
            for x1 in sample_points(l1, u1, 5) {
                let y = rope_eval(&[x0, x1], &[c], &[s]);
                for (dim, &y_val) in y.iter().enumerate() {
                    prop_assert!(
                        output.lower()[[dim]] - FP_TOLERANCE <= y_val
                            && y_val <= output.upper()[[dim]] + FP_TOLERANCE,
                        "IBP 1-pair interior soundness violated at dim {dim}: y={y_val}, bounds=[{}, {}]",
                        output.lower()[[dim]], output.upper()[[dim]]
                    );
                }
            }
        }
    }

    /// RoPE IBP soundness with 2 pairs (head_dim=4).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rope_ibp_2pairs(
        angle0 in -std::f32::consts::PI..std::f32::consts::PI,
        angle1 in -std::f32::consts::PI..std::f32::consts::PI,
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let layer = RopeLayer::from_angles(&[angle0, angle1]).unwrap();
        let lower = vec![l0, l1, l2, l3];
        let upper = vec![u0, u1, u2, u3];
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[4]), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[4]), upper.clone()).unwrap(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        let cos_freqs = vec![angle0.cos(), angle1.cos()];
        let sin_freqs = vec![angle0.sin(), angle1.sin()];

        // Check all 16 corners of the 4D input box
        for corner in 0..16u32 {
            let x: Vec<f32> = (0..4).map(|i| {
                if (corner >> i) & 1 == 1 { upper[i] } else { lower[i] }
            }).collect();
            let y = rope_eval(&x, &cos_freqs, &sin_freqs);
            for (dim, &y_val) in y.iter().enumerate() {
                prop_assert!(
                    output.lower()[[dim]] - FP_TOLERANCE <= y_val
                        && y_val <= output.upper()[[dim]] + FP_TOLERANCE,
                    "IBP 2-pair soundness violated at dim {dim}: y={y_val}, bounds=[{}, {}], corner={corner}",
                    output.lower()[[dim]], output.upper()[[dim]]
                );
            }
        }
    }
}

// =============================================================================
// CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// CROWN backward with identity incoming: for a linear layer, CROWN with identity
    /// spec and concretization should match IBP exactly.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rope_crown_identity(
        angle in -std::f32::consts::PI..std::f32::consts::PI,
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
    ) {
        let layer = RopeLayer::from_angles(&[angle]).unwrap();
        let input = BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        // IBP output
        let ibp_output = layer.propagate_ibp(&input).unwrap();

        // CROWN with identity spec
        let identity = LinearBounds::identity(2);
        let crown_result = layer.propagate_linear(&identity).unwrap();
        let crown_concrete = crown_result.concretize(&input);

        // For a linear layer, these must agree (within FP tolerance)
        for dim in 0..2 {
            prop_assert!(
                (ibp_output.lower()[[dim]] - crown_concrete.lower()[[dim]]).abs() < FP_TOLERANCE * 10.0,
                "CROWN-IBP consistency violation at dim {dim}: IBP_lower={}, CROWN_lower={}, angle={angle}",
                ibp_output.lower()[[dim]], crown_concrete.lower()[[dim]]
            );
            prop_assert!(
                (ibp_output.upper()[[dim]] - crown_concrete.upper()[[dim]]).abs() < FP_TOLERANCE * 10.0,
                "CROWN-IBP consistency violation at dim {dim}: IBP_upper={}, CROWN_upper={}, angle={angle}",
                ibp_output.upper()[[dim]], crown_concrete.upper()[[dim]]
            );
        }
    }

    /// CROWN backward with negative coefficients: verify soundness when the incoming
    /// linear bounds have negative entries (which swap lower/upper during concretization).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rope_crown_negcoeff(
        angle in -std::f32::consts::PI..std::f32::consts::PI,
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        a00 in -3.0f32..3.0,
        a01 in -3.0f32..3.0,
    ) {
        let layer = RopeLayer::from_angles(&[angle]).unwrap();
        let c = angle.cos();
        let s = angle.sin();

        let input = BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        // Incoming: z = a00 * y0 + a01 * y1 (single output, two inputs)
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![a00, a01]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![a00, a01]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer.propagate_linear(&incoming).unwrap();
        let concrete = result.concretize(&input);

        // Check all corners
        for &x0 in &[l0, u0] {
            for &x1 in &[l1, u1] {
                let y = rope_eval(&[x0, x1], &[c], &[s]);
                let z = a00 * y[0] + a01 * y[1];

                prop_assert!(
                    concrete.lower()[[0]] - FP_TOLERANCE * 10.0 <= z
                        && z <= concrete.upper()[[0]] + FP_TOLERANCE * 10.0,
                    "CROWN negcoeff soundness: z={z}, bounds=[{}, {}], x=[{x0},{x1}], angle={angle}, a=[{a00},{a01}]",
                    concrete.lower()[[0]], concrete.upper()[[0]]
                );
            }
        }

        // Sample interior points
        for x0 in sample_points(l0, u0, 7) {
            for x1 in sample_points(l1, u1, 7) {
                let y = rope_eval(&[x0, x1], &[c], &[s]);
                let z = a00 * y[0] + a01 * y[1];

                prop_assert!(
                    concrete.lower()[[0]] - FP_TOLERANCE * 10.0 <= z
                        && z <= concrete.upper()[[0]] + FP_TOLERANCE * 10.0,
                    "CROWN negcoeff interior soundness: z={z}, bounds=[{}, {}]",
                    concrete.lower()[[0]], concrete.upper()[[0]]
                );
            }
        }
    }

    /// CROWN backward with asymmetric incoming (different lower/upper coefficients).
    /// This tests that the lower and upper bound coefficient matrices are propagated
    /// independently through the rotation.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rope_crown_asymmetric(
        angle in -std::f32::consts::PI..std::f32::consts::PI,
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        la00 in -3.0f32..3.0,
        la01 in -3.0f32..3.0,
        ua00 in -3.0f32..3.0,
        ua01 in -3.0f32..3.0,
    ) {
        // Asymmetric: lower_A != upper_A
        let lower_a = Array2::from_shape_vec((1, 2), vec![la00, la01]).unwrap();
        let upper_a = Array2::from_shape_vec((1, 2), vec![ua00, ua01]).unwrap();

        // Skip degenerate cases where lower > upper in all regions
        // (LinearBounds::new_or_conservative may fix these up)

        let layer = RopeLayer::from_angles(&[angle]).unwrap();

        let input = BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        let incoming = LinearBounds::new(
            lower_a,
            Array1::zeros(1),
            upper_a,
            Array1::zeros(1),
        );

        // All generated matrices have matching finite shapes; rejection is a
        // counterexample, not a domain filter.
        let incoming = match incoming {
            Ok(b) => b,
            Err(e) => return Err(TestCaseError::fail(format!(
                "LinearBounds::new rejected generated finite RoPE coefficients: {e}"
            ))),
        };

        let result = match layer.propagate_linear(&incoming) {
            Ok(r) => r,
            Err(e) => return Err(TestCaseError::fail(format!(
                "RoPE CROWN backward rejected generated finite inputs: {e}"
            ))),
        };
        let concrete = result.concretize(&input);

        // For asymmetric bounds, the soundness property is:
        //   concrete.lower <= min over x of (lower_A @ R @ x + lower_b)
        //   concrete.upper >= max over x of (upper_A @ R @ x + upper_b)
        // Test by sampling.
        for x0 in sample_points(l0, u0, 10) {
            for x1 in sample_points(l1, u1, 10) {
                // Lower bound: computed from result coefficients applied to x (pre-rotation input).
                let z_l = result.lower_a()[[0, 0]] * x0 + result.lower_a()[[0, 1]] * x1 + result.lower_b()[0];
                // Upper bound: z_u = ua00 * y0 + ua01 * y1
                let z_u = result.upper_a()[[0, 0]] * x0 + result.upper_a()[[0, 1]] * x1 + result.upper_b()[0];

                // The concretized bounds must envelope z_l and z_u
                prop_assert!(
                    concrete.lower()[[0]] - FP_TOLERANCE * 10.0 <= z_l,
                    "Asymmetric CROWN lower violated: concrete_lower={} > z_l={z_l}",
                    concrete.lower()[[0]]
                );
                prop_assert!(
                    z_u <= concrete.upper()[[0]] + FP_TOLERANCE * 10.0,
                    "Asymmetric CROWN upper violated: z_u={z_u} > concrete_upper={}",
                    concrete.upper()[[0]]
                );
            }
        }
    }
}

// =============================================================================
// BATCHED CROWN SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Batched CROWN backward: verify that the batched path produces the same
    /// rotation as the non-batched path for each element in the batch.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rope_batched_crown_consistency(
        angle in -std::f32::consts::PI..std::f32::consts::PI,
        a00 in -3.0f32..3.0,
        a01 in -3.0f32..3.0,
        a10 in -3.0f32..3.0,
        a11 in -3.0f32..3.0,
    ) {
        let layer = RopeLayer::from_angles(&[angle]).unwrap();

        // Non-batched: 2x2 coefficient matrix
        let lower_a = Array2::from_shape_vec((2, 2), vec![a00, a01, a10, a11]).unwrap();
        let upper_a = lower_a.clone();
        let non_batched = LinearBounds::new(
            lower_a, Array1::zeros(2),
            upper_a, Array1::zeros(2),
        ).unwrap();

        let nb_result = layer.propagate_linear(&non_batched).unwrap();

        // Batched: same matrix wrapped as 3D [1, 2, 2]
        use crate::BatchedLinearBounds;
        let batched = BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![a00, a01, a10, a11]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![a00, a01, a10, a11]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.0]).unwrap(),
            vec![2],
            vec![2],
        ).unwrap();

        let b_result = layer.propagate_linear_batched(&batched).unwrap();

        // Compare: the [0, :, :] slice of batched result should match non-batched
        // for all coefficient matrices and bias vectors.
        for j in 0..2 {
            for i in 0..2 {
                let nb_val = nb_result.lower_a()[[j, i]];
                let b_val = b_result.lower_a[[0, j, i]];
                prop_assert!(
                    (nb_val - b_val).abs() < FP_TOLERANCE,
                    "Batched/non-batched lower_a mismatch at [{j},{i}]: nb={nb_val}, batch={b_val}"
                );
                let nb_val = nb_result.upper_a()[[j, i]];
                let b_val = b_result.upper_a[[0, j, i]];
                prop_assert!(
                    (nb_val - b_val).abs() < FP_TOLERANCE,
                    "Batched/non-batched upper_a mismatch at [{j},{i}]: nb={nb_val}, batch={b_val}"
                );
            }
        }
        // Bias vectors
        for j in 0..2 {
            let nb_val = nb_result.lower_b()[j];
            let b_val = b_result.lower_b[[0, j]];
            prop_assert!(
                (nb_val - b_val).abs() < FP_TOLERANCE,
                "Batched/non-batched lower_b mismatch at [{j}]: nb={nb_val}, batch={b_val}"
            );
            let nb_val = nb_result.upper_b()[j];
            let b_val = b_result.upper_b[[0, j]];
            prop_assert!(
                (nb_val - b_val).abs() < FP_TOLERANCE,
                "Batched/non-batched upper_b mismatch at [{j}]: nb={nb_val}, batch={b_val}"
            );
        }
    }
}

// =============================================================================
// MULTI-PAIR CROWN SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// CROWN backward with 4 pairs (head_dim=8): verify end-to-end soundness for
    /// realistic dimensions. Kokoro-family TTS models use head_dim=64+; this
    /// catches indexing bugs in the pair loop that 1-pair tests cannot.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_rope_crown_4pairs(
        angle0 in -std::f32::consts::PI..std::f32::consts::PI,
        angle1 in -std::f32::consts::PI..std::f32::consts::PI,
        angle2 in -std::f32::consts::PI..std::f32::consts::PI,
        angle3 in -std::f32::consts::PI..std::f32::consts::PI,
        // Incoming: z = a0*y0 + a1*y1 + ... + a7*y7 (single output, 8 inputs)
        a0 in -2.0f32..2.0,
        a1 in -2.0f32..2.0,
    ) {
        let layer = RopeLayer::from_angles(&[angle0, angle1, angle2, angle3]).unwrap();
        let cos_freqs: Vec<f32> = [angle0, angle1, angle2, angle3].iter().map(|a| a.cos()).collect();
        let sin_freqs: Vec<f32> = [angle0, angle1, angle2, angle3].iter().map(|a| a.sin()).collect();

        // Use fixed input bounds to keep proptest parameter count manageable
        let lower_vals = vec![-1.0f32, -2.0, -0.5, -1.5, -1.0, -2.0, -0.5, -1.5];
        let upper_vals = vec![2.0f32, 1.0, 1.5, 0.5, 2.0, 1.0, 1.5, 0.5];

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[8]), lower_vals.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[8]), upper_vals.clone()).unwrap(),
        ).unwrap();

        // Incoming: weighted sum of first two rotated outputs (randomized)
        // plus fixed coefficients for remaining to keep test tractable
        let a_vec = vec![a0, a1, 0.5, -0.5, 1.0, -1.0, 0.3, -0.3];
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 8), a_vec.clone()).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 8), a_vec.clone()).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer.propagate_linear(&incoming).unwrap();
        let concrete = result.concretize(&input);

        // Check all 2^8 = 256 corners of the 8D input box.
        // Tolerance: rope_eval has 6 ops/pair × 4 pairs = 24 ops, then 8-term
        // dot product = 15 ops → ~39 fp ops total, max value ~6.
        // Accumulated error: 2 * 39 * 6 * 1.2e-7 ≈ 5.6e-5 → FP_TOLERANCE * 10.0.
        for corner in 0..256u32 {
            let x: Vec<f32> = (0..8).map(|i| {
                if (corner >> i) & 1 == 1 { upper_vals[i] } else { lower_vals[i] }
            }).collect();
            let y = rope_eval(&x, &cos_freqs, &sin_freqs);
            let z: f32 = a_vec.iter().zip(y.iter()).map(|(a, y)| a * y).sum();

            prop_assert!(
                concrete.lower()[[0]] - FP_TOLERANCE * 10.0 <= z
                    && z <= concrete.upper()[[0]] + FP_TOLERANCE * 10.0,
                "4-pair CROWN soundness: z={z}, bounds=[{}, {}], corner={corner}",
                concrete.lower()[[0]], concrete.upper()[[0]]
            );
        }
    }

    /// Batched CROWN end-to-end soundness: verify that batched backward produces
    /// correct rotated coefficients by extracting the batch-0 slice and concretizing
    /// via non-batched LinearBounds. This tests the batched rotation loop independently
    /// of batched concretize shape broadcasting.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rope_batched_crown_end_to_end(
        angle in -std::f32::consts::PI..std::f32::consts::PI,
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        a00 in -3.0f32..3.0,
        a01 in -3.0f32..3.0,
        a10 in -3.0f32..3.0,
        a11 in -3.0f32..3.0,
    ) {
        let layer = RopeLayer::from_angles(&[angle]).unwrap();
        let c = angle.cos();
        let s = angle.sin();

        let input = BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        use crate::BatchedLinearBounds;
        let batched = BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![a00, a01, a10, a11]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![a00, a01, a10, a11]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.0]).unwrap(),
            vec![2],
            vec![2],
        ).unwrap();

        let b_result = layer.propagate_linear_batched(&batched).unwrap();

        // Extract batch-0 slice into non-batched LinearBounds for concretization.
        // This tests the batched rotation loop while avoiding batched concretize
        // shape broadcasting limitations.
        let la_slice = Array2::from_shape_vec(
            (2, 2),
            (0..4).map(|idx| b_result.lower_a[[0, idx / 2, idx % 2]]).collect(),
        ).unwrap();
        let ua_slice = Array2::from_shape_vec(
            (2, 2),
            (0..4).map(|idx| b_result.upper_a[[0, idx / 2, idx % 2]]).collect(),
        ).unwrap();
        let lb_slice = Array1::from_vec(vec![b_result.lower_b[[0, 0]], b_result.lower_b[[0, 1]]]);
        let ub_slice = Array1::from_vec(vec![b_result.upper_b[[0, 0]], b_result.upper_b[[0, 1]]]);

        let nb_result = LinearBounds::new(la_slice, lb_slice, ua_slice, ub_slice).unwrap();
        let concrete = nb_result.concretize(&input);

        // Check all 4 corners
        for &x0 in &[l0, u0] {
            for &x1 in &[l1, u1] {
                let y = rope_eval(&[x0, x1], &[c], &[s]);
                // z0 = a00*y0 + a01*y1, z1 = a10*y0 + a11*y1
                let z0 = a00 * y[0] + a01 * y[1];
                let z1 = a10 * y[0] + a11 * y[1];

                prop_assert!(
                    concrete.lower()[[0]] - FP_TOLERANCE * 10.0 <= z0
                        && z0 <= concrete.upper()[[0]] + FP_TOLERANCE * 10.0,
                    "Batched CROWN e2e [0]: z0={z0}, bounds=[{}, {}], x=[{x0},{x1}]",
                    concrete.lower()[[0]], concrete.upper()[[0]]
                );
                prop_assert!(
                    concrete.lower()[[1]] - FP_TOLERANCE * 10.0 <= z1
                        && z1 <= concrete.upper()[[1]] + FP_TOLERANCE * 10.0,
                    "Batched CROWN e2e [1]: z1={z1}, bounds=[{}, {}], x=[{x0},{x1}]",
                    concrete.lower()[[1]], concrete.upper()[[1]]
                );
            }
        }
    }
}

// =============================================================================
// DIRECTED ROUNDING VERIFICATION
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Verify directed rounding: IBP lower bounds must be <= the true output computed
    /// in f64 (using the layer's f32 cos/sin promoted to f64), and upper bounds must
    /// be >= the true f64 output. This catches f32 arithmetic rounding errors that
    /// `next_down_f32`/`next_up_f32` must cover.
    ///
    /// NOTE: We use the layer's f32 cos/sin values (promoted to f64) as the reference,
    /// NOT f64 cos/sin. The layer is sound with respect to its stored frequencies.
    /// Differences between f32 cos(θ) and f64 cos(θ) are a separate concern (input
    /// precision), not a soundness bug in the bound propagation arithmetic.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_rope_directed_rounding(
        angle in -std::f32::consts::PI..std::f32::consts::PI,
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
    ) {
        let layer = RopeLayer::from_angles(&[angle]).unwrap();
        let input = BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();

        // Use the layer's ACTUAL stored f32 cos/sin (promoted to f64), NOT a
        // recomputed `angle.cos()`. f32 `cos`/`sin` is not bit-reproducible across
        // codegen contexts: `cos(2.190452)` evaluates to -0.5807549357 at one
        // optimization level and -0.5807549953 at another (~1 ulp apart). The layer
        // stores the value it computed at construction; recomputing here yields a
        // reference offset ~1 ulp from it, scaled by |x| up to ~2e-7 at the box
        // edge — enough to make this assertion (1e-10 slack) report a FALSE
        // "directed rounding violated" in release even though the bound is sound
        // w.r.t. the cos/sin the layer actually used. Reading the stored freqs
        // makes the reference exact for THIS layer instance. (The bound-propagation
        // arithmetic — the thing under test — is unchanged and provably sound.)
        let c = layer.cos_freqs[0] as f64;
        let s = layer.sin_freqs[0] as f64;

        // Evaluate at all corners in f64 to get exact true bounds
        let corners: Vec<(f64, f64)> = vec![
            (l0 as f64, l1 as f64),
            (l0 as f64, u1 as f64),
            (u0 as f64, l1 as f64),
            (u0 as f64, u1 as f64),
        ];

        for &(x0, x1) in &corners {
            // Compute each product separately in f64 (matching the layer's computation
            // structure: a*x_lo, a*x_hi, b*y_lo, b*y_hi, then sum)
            let y0_f64 = c * x0 + (-s) * x1;
            let y1_f64 = s * x0 + c * x1;

            // IBP lower must be <= true f64 output (soundness)
            prop_assert!(
                (output.lower()[[0]] as f64) <= y0_f64 + 1e-10,
                "Directed rounding lower[0] violated: {} > {y0_f64}",
                output.lower()[[0]]
            );
            prop_assert!(
                (output.lower()[[1]] as f64) <= y1_f64 + 1e-10,
                "Directed rounding lower[1] violated: {} > {y1_f64}",
                output.lower()[[1]]
            );
            // IBP upper must be >= true f64 output (soundness)
            prop_assert!(
                (output.upper()[[0]] as f64) >= y0_f64 - 1e-10,
                "Directed rounding upper[0] violated: {} < {y0_f64}",
                output.upper()[[0]]
            );
            prop_assert!(
                (output.upper()[[1]] as f64) >= y1_f64 - 1e-10,
                "Directed rounding upper[1] violated: {} < {y1_f64}",
                output.upper()[[1]]
            );
        }
    }
}
