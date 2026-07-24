// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward soundness proptests for constant arithmetic layers:
//! MulConstant, DivConstant, SubConstant.
//!
//! All three are linear (affine) operations, so CROWN backward composes exactly
//! (no relaxation). The tolerance accounts only for FP rounding.
//!
//! - MulConstant: y = x * c → A_new = A * diag(c), b_new = b
//! - DivConstant: y = x / c → A_new = A * diag(1/c), b_new = b
//! - SubConstant: y = x - c → A_new = A, b_new = b - A@c  (or negate A for reverse)
//!
//! Part of #40.

use crate::layers::arithmetic::{DivConstantLayer, MulConstantLayer, SubConstantLayer};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::sample_points;

/// Tight tolerance for affine CROWN soundness.
/// These layers are exact affine transforms — only FP rounding introduces error.
const AFFINE_CROWN_TOLERANCE: f32 = 1e-5;

// =============================================================================
// MULCONSTANT CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// MulConstant CROWN backward soundness with identity incoming bounds.
    ///
    /// For y = x * c, CROWN backward scales coefficients by c. With identity
    /// incoming, the concretized bounds should match IBP bounds exactly.
    /// Tests both positive and negative constants.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_mulconstant_crown_identity(
        l0 in -5.0f32..5.0,
        d0 in 0.01f32..3.0,
        l1 in -5.0f32..5.0,
        d1 in 0.01f32..3.0,
        l2 in -5.0f32..5.0,
        d2 in 0.01f32..3.0,
        c in -5.0f32..5.0,
    ) {
        prop_assume!(c.abs() > 0.001);

        let u0 = (l0 + d0).min(5.0);
        let u1 = (l1 + d1).min(5.0);
        let u2 = (l2 + d2).min(5.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = MulConstantLayer::scalar(c);

        // IBP reference
        let ibp_output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        // CROWN backward
        let identity = LinearBounds::identity(3);
        let crown_result = layer.propagate_linear(&identity)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear failed: {e}")
            ))?;
        let concrete = crown_result.concretize(&input);

        // CROWN-IBP equivalence (MulConstant is linear, so should match exactly)
        for i in 0..3 {
            prop_assert!(
                (concrete.lower()[[i]] - ibp_output.lower()[[i]]).abs() <= AFFINE_CROWN_TOLERANCE,
                "MulConstant CROWN-IBP lower mismatch at {i}: crown={}, ibp={}",
                concrete.lower()[[i]], ibp_output.lower()[[i]]
            );
            prop_assert!(
                (concrete.upper()[[i]] - ibp_output.upper()[[i]]).abs() <= AFFINE_CROWN_TOLERANCE,
                "MulConstant CROWN-IBP upper mismatch at {i}: crown={}, ibp={}",
                concrete.upper()[[i]], ibp_output.upper()[[i]]
            );
        }

        // Soundness via sampling
        let spts = sample_points(0.0, 1.0, 7);
        for &t0 in &spts {
            for &t1 in &spts {
                for &t2 in &spts {
                    let x0 = l0 + t0 * (u0 - l0);
                    let x1 = l1 + t1 * (u1 - l1);
                    let x2 = l2 + t2 * (u2 - l2);
                    let y = arr1(&[x0 * c, x1 * c, x2 * c]);

                    for i in 0..3 {
                        prop_assert!(
                            y[i] >= concrete.lower()[[i]] - AFFINE_CROWN_TOLERANCE,
                            "MulConstant lower violation at {i}: y={} < lb={}",
                            y[i], concrete.lower()[[i]]
                        );
                        prop_assert!(
                            y[i] <= concrete.upper()[[i]] + AFFINE_CROWN_TOLERANCE,
                            "MulConstant upper violation at {i}: y={} > ub={}",
                            y[i], concrete.upper()[[i]]
                        );
                    }
                }
            }
        }
    }

    /// MulConstant CROWN with non-identity incoming and negative coefficients.
    ///
    /// Tests that CROWN composition correctly handles the sign interaction between
    /// incoming coefficients and the multiplication constant.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_mulconstant_crown_negative_coeffs(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.01f32..2.0,
        c in -3.0f32..3.0,
        // Incoming: 1 output combining all 3
        k0 in -2.0f32..2.0,
        k1 in -2.0f32..2.0,
        k2 in -2.0f32..2.0,
    ) {
        prop_assume!(c.abs() > 0.01);
        prop_assume!(k0.abs() > 0.01 || k1.abs() > 0.01 || k2.abs() > 0.01);

        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);
        let u2 = (l2 + d2).min(3.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = MulConstantLayer::scalar(c);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 3), vec![k0, k1, k2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![k0, k1, k2]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let crown_result = layer.propagate_linear(&incoming)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear failed: {e}")
            ))?;
        let concrete = crown_result.concretize(&input);
        let crown_lower = concrete.lower()[[0]];
        let crown_upper = concrete.upper()[[0]];

        // Verify: k . (x * c) = (k * c) . x must be in [crown_lower, crown_upper]
        let spts = 5;
        for &x0 in &sample_points(l0, u0, spts) {
            for &x1 in &sample_points(l1, u1, spts) {
                for &x2 in &sample_points(l2, u2, spts) {
                    let combined = k0 * (x0 * c) + k1 * (x1 * c) + k2 * (x2 * c);

                    prop_assert!(
                        combined >= crown_lower - AFFINE_CROWN_TOLERANCE,
                        "MulConstant negative_coeffs lower violation: \
                         k.(x*c)={combined} < lb={crown_lower}, c={c}"
                    );
                    prop_assert!(
                        combined <= crown_upper + AFFINE_CROWN_TOLERANCE,
                        "MulConstant negative_coeffs upper violation: \
                         k.(x*c)={combined} > ub={crown_upper}, c={c}"
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    /// Per-channel MulConstant broadcast soundness for ONNX-style `[C, 1] -> [C, T]`.
    ///
    /// Verifies that dense CROWN backward reconstructs the true broadcasted scale
    /// pattern instead of flat tiling. Part of #3896.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_mulconstant_crown_per_channel_broadcast_3896(
        l00 in -3.0f32..3.0,
        d00 in 0.01f32..2.0,
        l01 in -3.0f32..3.0,
        d01 in 0.01f32..2.0,
        l10 in -3.0f32..3.0,
        d10 in 0.01f32..2.0,
        l11 in -3.0f32..3.0,
        d11 in 0.01f32..2.0,
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
    ) {
        prop_assume!(c0.abs() > 0.01);
        prop_assume!(c1.abs() > 0.01);

        let u00 = (l00 + d00).min(3.0);
        let u01 = (l01 + d01).min(3.0);
        let u10 = (l10 + d10).min(3.0);
        let u11 = (l11 + d11).min(3.0);

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![l00, l01, l10, l11]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![u00, u01, u10, u11]).unwrap(),
        ).unwrap();
        let layer = MulConstantLayer::with_input_shape(
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![c0, c1]).unwrap(),
            vec![2, 2],
        );

        let ibp_output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;
        let identity = LinearBounds::identity(4);
        let crown_result = layer.propagate_linear(&identity)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear failed: {e}")
            ))?;
        let concrete = crown_result.concretize(&input);

        for (idx, (&crown_l, &ibp_l)) in concrete.lower().iter().zip(ibp_output.lower().iter()).enumerate() {
            prop_assert!(
                (crown_l - ibp_l).abs() <= AFFINE_CROWN_TOLERANCE,
                "MulConstant per-channel lower mismatch at {idx}: crown={crown_l}, ibp={ibp_l}"
            );
        }
        for (idx, (&crown_u, &ibp_u)) in concrete.upper().iter().zip(ibp_output.upper().iter()).enumerate() {
            prop_assert!(
                (crown_u - ibp_u).abs() <= AFFINE_CROWN_TOLERANCE,
                "MulConstant per-channel upper mismatch at {idx}: crown={crown_u}, ibp={ibp_u}"
            );
        }

        let spts = sample_points(0.0, 1.0, 5);
        for &t00 in &spts {
            for &t01 in &spts {
                for &t10 in &spts {
                    for &t11 in &spts {
                        let x00 = l00 + t00 * (u00 - l00);
                        let x01 = l01 + t01 * (u01 - l01);
                        let x10 = l10 + t10 * (u10 - l10);
                        let x11 = l11 + t11 * (u11 - l11);
                        let y = [x00 * c0, x01 * c0, x10 * c1, x11 * c1];

                        for (i, expected) in y.iter().enumerate() {
                            prop_assert!(
                                *expected >= concrete.lower()[[i]] - AFFINE_CROWN_TOLERANCE,
                                "MulConstant per-channel lower violation at {i}: y={} < lb={}",
                                expected, concrete.lower()[[i]]
                            );
                            prop_assert!(
                                *expected <= concrete.upper()[[i]] + AFFINE_CROWN_TOLERANCE,
                                "MulConstant per-channel upper violation at {i}: y={} > ub={}",
                                expected, concrete.upper()[[i]]
                            );
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// DIVCONSTANT CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// DivConstant CROWN backward soundness with identity incoming bounds.
    ///
    /// For y = x / c, CROWN backward scales coefficients by 1/c. Tests that
    /// concretized bounds contain x/c for all sampled x.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_divconstant_crown_identity(
        l0 in -5.0f32..5.0,
        d0 in 0.01f32..3.0,
        l1 in -5.0f32..5.0,
        d1 in 0.01f32..3.0,
        l2 in -5.0f32..5.0,
        d2 in 0.01f32..3.0,
        c in -5.0f32..5.0,
    ) {
        // Avoid division by near-zero
        prop_assume!(c.abs() > 0.1);

        let u0 = (l0 + d0).min(5.0);
        let u1 = (l1 + d1).min(5.0);
        let u2 = (l2 + d2).min(5.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = DivConstantLayer::scalar(c);

        // IBP reference
        let ibp_output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        // CROWN backward
        let identity = LinearBounds::identity(3);
        let crown_result = layer.propagate_linear(&identity)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear failed: {e}")
            ))?;
        // Use concretize_sound() which applies directed rounding (next_down/up_f32)
        // at the final f64->f32 cast, covering accumulated ULP error from the
        // DivConstant->MulConstant coefficient scaling path (#1483).
        let concrete = crown_result.concretize_sound(&input);

        // CROWN-IBP equivalence
        for i in 0..3 {
            prop_assert!(
                (concrete.lower()[[i]] - ibp_output.lower()[[i]]).abs() <= AFFINE_CROWN_TOLERANCE,
                "DivConstant CROWN-IBP lower mismatch at {i}: crown={}, ibp={}",
                concrete.lower()[[i]], ibp_output.lower()[[i]]
            );
            prop_assert!(
                (concrete.upper()[[i]] - ibp_output.upper()[[i]]).abs() <= AFFINE_CROWN_TOLERANCE,
                "DivConstant CROWN-IBP upper mismatch at {i}: crown={}, ibp={}",
                concrete.upper()[[i]], ibp_output.upper()[[i]]
            );
        }

        // Soundness via sampling
        let spts = 5;
        for &x0 in &sample_points(l0, u0, spts) {
            for &x1 in &sample_points(l1, u1, spts) {
                for &x2 in &sample_points(l2, u2, spts) {
                    let y = arr1(&[x0 / c, x1 / c, x2 / c]);

                    for i in 0..3 {
                        prop_assert!(
                            y[i] >= concrete.lower()[[i]] - AFFINE_CROWN_TOLERANCE,
                            "DivConstant lower violation at {i}: y={} < lb={}",
                            y[i], concrete.lower()[[i]]
                        );
                        prop_assert!(
                            y[i] <= concrete.upper()[[i]] + AFFINE_CROWN_TOLERANCE,
                            "DivConstant upper violation at {i}: y={} > ub={}",
                            y[i], concrete.upper()[[i]]
                        );
                    }
                }
            }
        }
    }

    /// DivConstant CROWN with non-identity incoming and negative divisor.
    ///
    /// Tests sign interaction: negative c flips the sign of scaled coefficients,
    /// which the CROWN composition must handle correctly.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_divconstant_crown_negative_coeffs(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        c in -3.0f32..3.0,
        k0 in -2.0f32..2.0,
        k1 in -2.0f32..2.0,
    ) {
        prop_assume!(c.abs() > 0.1);
        prop_assume!(k0.abs() > 0.01 || k1.abs() > 0.01);

        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![l0, l1]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![u0, u1]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = DivConstantLayer::scalar(c);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![k0, k1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![k0, k1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let crown_result = layer.propagate_linear(&incoming)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear failed: {e}")
            ))?;
        // Use concretize_sound() for directed rounding at the final cast (#1483).
        let concrete = crown_result.concretize_sound(&input);
        let crown_lower = concrete.lower()[[0]];
        let crown_upper = concrete.upper()[[0]];

        let spts = 7;
        for &x0 in &sample_points(l0, u0, spts) {
            for &x1 in &sample_points(l1, u1, spts) {
                let combined = k0 * (x0 / c) + k1 * (x1 / c);

                prop_assert!(
                    combined >= crown_lower - AFFINE_CROWN_TOLERANCE,
                    "DivConstant negative_coeffs lower violation: {combined} < {crown_lower}"
                );
                prop_assert!(
                    combined <= crown_upper + AFFINE_CROWN_TOLERANCE,
                    "DivConstant negative_coeffs upper violation: {combined} > {crown_upper}"
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    /// Per-channel DivConstant broadcast soundness for ONNX-style `[C, 1] -> [C, T]`.
    ///
    /// Verifies that dense CROWN backward reconstructs the true broadcasted
    /// divisor pattern for the widened `DivConstantLayer` surface in `#3896`.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_divconstant_crown_per_channel_broadcast_3896(
        l00 in -3.0f32..3.0,
        d00 in 0.01f32..2.0,
        l01 in -3.0f32..3.0,
        d01 in 0.01f32..2.0,
        l10 in -3.0f32..3.0,
        d10 in 0.01f32..2.0,
        l11 in -3.0f32..3.0,
        d11 in 0.01f32..2.0,
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
    ) {
        prop_assume!(c0.abs() > 0.1);
        prop_assume!(c1.abs() > 0.1);

        let u00 = (l00 + d00).min(3.0);
        let u01 = (l01 + d01).min(3.0);
        let u10 = (l10 + d10).min(3.0);
        let u11 = (l11 + d11).min(3.0);

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![l00, l01, l10, l11]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![u00, u01, u10, u11]).unwrap(),
        )
        .unwrap();
        let layer = DivConstantLayer::with_input_shape(
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![c0, c1]).unwrap(),
            vec![2, 2],
        );

        let ibp_output = layer.propagate_ibp(&input).map_err(|e| {
            TestCaseError::fail(format!("propagate_ibp failed: {e}"))
        })?;
        let identity = LinearBounds::identity(4);
        let crown_result = layer.propagate_linear(&identity).map_err(|e| {
            TestCaseError::fail(format!("propagate_linear failed: {e}"))
        })?;
        let concrete = crown_result.concretize_sound(&input);

        for (idx, (&crown_l, &ibp_l)) in concrete
            .lower()
            .iter()
            .zip(ibp_output.lower().iter())
            .enumerate()
        {
            prop_assert!(
                (crown_l - ibp_l).abs() <= AFFINE_CROWN_TOLERANCE,
                "DivConstant per-channel lower mismatch at {idx}: crown={crown_l}, ibp={ibp_l}"
            );
        }
        for (idx, (&crown_u, &ibp_u)) in concrete
            .upper()
            .iter()
            .zip(ibp_output.upper().iter())
            .enumerate()
        {
            prop_assert!(
                (crown_u - ibp_u).abs() <= AFFINE_CROWN_TOLERANCE,
                "DivConstant per-channel upper mismatch at {idx}: crown={crown_u}, ibp={ibp_u}"
            );
        }

        let spts = sample_points(0.0, 1.0, 5);
        for &t00 in &spts {
            for &t01 in &spts {
                for &t10 in &spts {
                    for &t11 in &spts {
                        let x00 = l00 + t00 * (u00 - l00);
                        let x01 = l01 + t01 * (u01 - l01);
                        let x10 = l10 + t10 * (u10 - l10);
                        let x11 = l11 + t11 * (u11 - l11);
                        let y = [x00 / c0, x01 / c0, x10 / c1, x11 / c1];

                        for (i, expected) in y.iter().enumerate() {
                            prop_assert!(
                                *expected >= concrete.lower()[[i]] - AFFINE_CROWN_TOLERANCE,
                                "DivConstant per-channel lower violation at {i}: y={} < lb={}",
                                expected,
                                concrete.lower()[[i]]
                            );
                            prop_assert!(
                                *expected <= concrete.upper()[[i]] + AFFINE_CROWN_TOLERANCE,
                                "DivConstant per-channel upper violation at {i}: y={} > ub={}",
                                expected,
                                concrete.upper()[[i]]
                            );
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// SUBCONSTANT CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// SubConstant (y = x - c) CROWN backward soundness with identity incoming.
    ///
    /// For y = x - c, CROWN backward shifts bias: b_new = b - A@c.
    /// Tests CROWN-IBP equivalence and sampling soundness.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_subconstant_crown_identity(
        l0 in -5.0f32..5.0,
        d0 in 0.01f32..3.0,
        l1 in -5.0f32..5.0,
        d1 in 0.01f32..3.0,
        l2 in -5.0f32..5.0,
        d2 in 0.01f32..3.0,
        c in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(5.0);
        let u1 = (l1 + d1).min(5.0);
        let u2 = (l2 + d2).min(5.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = SubConstantLayer::scalar(c);

        // IBP reference
        let ibp_output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        // CROWN backward
        let identity = LinearBounds::identity(3);
        let crown_result = layer.propagate_linear(&identity)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear failed: {e}")
            ))?;
        let concrete = crown_result.concretize(&input);

        // CROWN-IBP equivalence
        for i in 0..3 {
            prop_assert!(
                (concrete.lower()[[i]] - ibp_output.lower()[[i]]).abs() <= AFFINE_CROWN_TOLERANCE,
                "SubConstant CROWN-IBP lower mismatch at {i}: crown={}, ibp={}",
                concrete.lower()[[i]], ibp_output.lower()[[i]]
            );
            prop_assert!(
                (concrete.upper()[[i]] - ibp_output.upper()[[i]]).abs() <= AFFINE_CROWN_TOLERANCE,
                "SubConstant CROWN-IBP upper mismatch at {i}: crown={}, ibp={}",
                concrete.upper()[[i]], ibp_output.upper()[[i]]
            );
        }

        // Soundness via sampling
        let spts = 5;
        for &x0 in &sample_points(l0, u0, spts) {
            for &x1 in &sample_points(l1, u1, spts) {
                for &x2 in &sample_points(l2, u2, spts) {
                    let y = arr1(&[x0 - c, x1 - c, x2 - c]);

                    for i in 0..3 {
                        prop_assert!(
                            y[i] >= concrete.lower()[[i]] - AFFINE_CROWN_TOLERANCE,
                            "SubConstant lower violation at {i}: y={} < lb={}",
                            y[i], concrete.lower()[[i]]
                        );
                        prop_assert!(
                            y[i] <= concrete.upper()[[i]] + AFFINE_CROWN_TOLERANCE,
                            "SubConstant upper violation at {i}: y={} > ub={}",
                            y[i], concrete.upper()[[i]]
                        );
                    }
                }
            }
        }
    }

    /// SubConstant reverse (y = c - x) CROWN backward soundness.
    ///
    /// For y = c - x, CROWN backward negates coefficients and shifts bias.
    /// This tests the reverse subtraction path which flips bounds.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_subconstant_reverse_crown_identity(
        l0 in -5.0f32..5.0,
        d0 in 0.01f32..3.0,
        l1 in -5.0f32..5.0,
        d1 in 0.01f32..3.0,
        l2 in -5.0f32..5.0,
        d2 in 0.01f32..3.0,
        c in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(5.0);
        let u1 = (l1 + d1).min(5.0);
        let u2 = (l2 + d2).min(5.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = SubConstantLayer::new_reverse(ArrayD::from_elem(IxDyn(&[]), c));

        // IBP reference
        let ibp_output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        // CROWN backward
        let identity = LinearBounds::identity(3);
        let crown_result = layer.propagate_linear(&identity)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear failed: {e}")
            ))?;
        let concrete = crown_result.concretize(&input);

        // CROWN-IBP equivalence
        for i in 0..3 {
            prop_assert!(
                (concrete.lower()[[i]] - ibp_output.lower()[[i]]).abs() <= AFFINE_CROWN_TOLERANCE,
                "SubConstant reverse CROWN-IBP lower mismatch at {i}: crown={}, ibp={}",
                concrete.lower()[[i]], ibp_output.lower()[[i]]
            );
            prop_assert!(
                (concrete.upper()[[i]] - ibp_output.upper()[[i]]).abs() <= AFFINE_CROWN_TOLERANCE,
                "SubConstant reverse CROWN-IBP upper mismatch at {i}: crown={}, ibp={}",
                concrete.upper()[[i]], ibp_output.upper()[[i]]
            );
        }

        // Soundness: y = c - x
        let spts = 5;
        for &x0 in &sample_points(l0, u0, spts) {
            for &x1 in &sample_points(l1, u1, spts) {
                for &x2 in &sample_points(l2, u2, spts) {
                    let y = arr1(&[c - x0, c - x1, c - x2]);

                    for i in 0..3 {
                        prop_assert!(
                            y[i] >= concrete.lower()[[i]] - AFFINE_CROWN_TOLERANCE,
                            "SubConstant reverse lower violation at {i}: y={} < lb={}",
                            y[i], concrete.lower()[[i]]
                        );
                        prop_assert!(
                            y[i] <= concrete.upper()[[i]] + AFFINE_CROWN_TOLERANCE,
                            "SubConstant reverse upper violation at {i}: y={} > ub={}",
                            y[i], concrete.upper()[[i]]
                        );
                    }
                }
            }
        }
    }

    /// SubConstant CROWN with non-identity incoming coefficients.
    ///
    /// Tests both forward (y = x - c) and reverse (y = c - x) with non-identity
    /// incoming bounds to exercise the coefficient composition path.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_subconstant_crown_negative_coeffs(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.01f32..2.0,
        c in -3.0f32..3.0,
        k0 in -2.0f32..2.0,
        k1 in -2.0f32..2.0,
        k2 in -2.0f32..2.0,
    ) {
        prop_assume!(k0.abs() > 0.01 || k1.abs() > 0.01 || k2.abs() > 0.01);

        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);
        let u2 = (l2 + d2).min(3.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = SubConstantLayer::scalar(c);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 3), vec![k0, k1, k2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![k0, k1, k2]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let crown_result = layer.propagate_linear(&incoming)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear failed: {e}")
            ))?;
        let concrete = crown_result.concretize(&input);
        let crown_lower = concrete.lower()[[0]];
        let crown_upper = concrete.upper()[[0]];

        let spts = 5;
        for &x0 in &sample_points(l0, u0, spts) {
            for &x1 in &sample_points(l1, u1, spts) {
                for &x2 in &sample_points(l2, u2, spts) {
                    let combined = k0 * (x0 - c) + k1 * (x1 - c) + k2 * (x2 - c);

                    prop_assert!(
                        combined >= crown_lower - AFFINE_CROWN_TOLERANCE,
                        "SubConstant negative_coeffs lower violation: {combined} < {crown_lower}"
                    );
                    prop_assert!(
                        combined <= crown_upper + AFFINE_CROWN_TOLERANCE,
                        "SubConstant negative_coeffs upper violation: {combined} > {crown_upper}"
                    );
                }
            }
        }
    }

    /// SubConstant reverse (y = c - x) CROWN with non-identity incoming.
    ///
    /// The reverse path negates coefficients, which combined with negative incoming
    /// coefficients creates a double-sign-flip that must be handled correctly.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_subconstant_reverse_crown_negative_coeffs(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        c in -3.0f32..3.0,
        k0 in -2.0f32..2.0,
        k1 in -2.0f32..2.0,
    ) {
        prop_assume!(k0.abs() > 0.01 || k1.abs() > 0.01);

        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![l0, l1]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![u0, u1]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = SubConstantLayer::new_reverse(ArrayD::from_elem(IxDyn(&[]), c));

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![k0, k1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![k0, k1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let crown_result = layer.propagate_linear(&incoming)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear failed: {e}")
            ))?;
        let concrete = crown_result.concretize(&input);
        let crown_lower = concrete.lower()[[0]];
        let crown_upper = concrete.upper()[[0]];

        let spts = 7;
        for &x0 in &sample_points(l0, u0, spts) {
            for &x1 in &sample_points(l1, u1, spts) {
                let combined = k0 * (c - x0) + k1 * (c - x1);

                prop_assert!(
                    combined >= crown_lower - AFFINE_CROWN_TOLERANCE,
                    "SubConstant reverse negative_coeffs lower violation: {combined} < {crown_lower}"
                );
                prop_assert!(
                    combined <= crown_upper + AFFINE_CROWN_TOLERANCE,
                    "SubConstant reverse negative_coeffs upper violation: {combined} > {crown_upper}"
                );
            }
        }
    }
}
